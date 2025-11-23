# Dream-7B Delta-Compute & L2C Caching — Master Plan (v2)

_Last updated: 2025‑11‑18_

> **One-line goal.** On top of Dream‑7B’s existing evaluation stack, implement and evaluate **all phases of delta‑compute acceleration** (P0–P4: baseline, teacher traces, rule‑based gate, learned gate, adaptive steps) and **all phases of L2C‑style layer caching** (Phase A–C: heuristic caching, learned router, continuous “time” router), to **reduce latency / FLOPs** while keeping **quality regressions small and measurable**.
>
> **Scope.** This project implements and validates **all** of P0–P4 and Phase A–C. Everything described here is in scope for implementation and GPU evaluation. Status, timelines, and job IDs live in other docs; this file is the **design/spec “bible”**.

Related docs:
- High‑level progress (CN): `docs/PROGRESS_SUMMARY_CN.md`
- Implementation details & integration checklists: `docs/IMPLEMENTATION_ROADMAP.md`
- Experiment notes: `experiments/README.md` and `experiments/*/*.md`

---

## 1. Environment & Repo Layout

**Environment**

- Recommended conda env: `dcllm`
  ```bash
  conda create -n dcllm python=3.10
  conda activate dcllm
  pip install -r requirements.txt
  ```
- Core libraries (aligned with Dream):
  - `torch` as in Dream README (H100 builds tested)
  - `transformers` as in Dream README
  - CUDA: match cluster / local GPU driver
- All experiments assume an activated `dcllm` env.

**Repo layout (this project)**

```text
DLLM-Delta-Compute/
├── src/                      # Core delta-compute + L2C infrastructure
│   ├── tracing.py            # P1 TraceCollector
│   ├── learned_gate.py       # P3 Learned gate
│   ├── adaptive_scheduler.py # P4 Adaptive step scheduler + HybridScheduler
│   ├── learned_router.py     # Phase B fixed-schedule router
│   ├── continuous_router.py  # Phase C continuous router
│   └── __init__.py
├── external/
│   ├── Dream/                # Dream-7B (git submodule)
│   │   └── modeling/         # Dream model + generation utils
│   └── learning-to-cache/    # L2C reference implementation (DiT / U-ViT)
├── scripts/
│   ├── slurm/                # SLURM eval scripts (eval_P*, eval_Phase*)
│   ├── eval/                 # Local eval tools
│   └── training/             # Gate / router training scripts
├── tests/
│   ├── unit/                 # GPU + infra unit tests
│   ├── integration/          # End‑to‑end smoke tests
│   └── visualization/        # Trace plotting utilities
├── experiments/              # Per‑phase experiment artifacts
│   ├── P0_baseline/
│   ├── P1_traces/
│   ├── P2_early_stop/
│   ├── P3_learned_gate/
│   ├── P4_adaptive/
│   ├── PhaseA_caching/
│   ├── PhaseB_router/
│   └── PhaseC_continuous/
└── docs/
    ├── master_plan.md        # THIS FILE – design/spec
    ├── IMPLEMENTATION_ROADMAP.md
    ├── PROGRESS_SUMMARY_CN.md
    └── reports/              # Validation reports
```

**Organization rules**

1. Reusable infrastructure lives in `src/`.
2. All tests live in `tests/` (unit / integration / visualization).
3. Anything that is “run directly” (including SLURM jobs) lives in `scripts/`.
4. All experiment outputs are under `experiments/{phase}/`.
5. This file (`docs/master_plan.md`) is the **source‑of‑truth spec**; more volatile plans go into `IMPLEMENTATION_ROADMAP.md` and per‑experiment READMEs.

---

## 2. Project Overview

We implement **two families of acceleration**, all on top of the same Dream‑7B evaluation stack:

- **Delta‑Compute Track (P0–P4)**  
  Control *how many diffusion steps* and *how many tokens* we actually compute:
  - **P0 – Baseline teacher.** No acceleration; reference for quality & latency.
  - **P1 – Teacher traces.** Collect per‑step, per‑layer statistics.
  - **P2 – Rule‑based gate.** Sequence‑level early stopping based on confidence / entropy.
  - **P3 – Learned gate.** Learned stopping rule using P1 features (sequence‑level in v1; token‑level extensions optional).
  - **P4 – Adaptive steps.** Dynamic diffusion stride based on local truncation error and risk signals.

- **L2C Layer‑Caching Track (Phase A–C)**  
  Control *how often each transformer block is recomputed across diffusion steps*:
  - **Phase A – Heuristic FFN caching.** Fixed hand‑designed schedule: reuse FFN outputs on some steps/layers.
  - **Phase B – Learned router (fixed schedule).** Learn per‑step, per‑layer recompute vs reuse on a fixed schedule.
  - **Phase C – Continuous router (schedule‑transferable).** Learn per‑layer recompute probability as a function of **continuous progress**.

Conceptually:

- P‑track (P0–P4) decides **“how many diffusion updates to do”** and **“when to stop”**.
- Phase‑track (A–C) decides **“for this diffusion update, which layers actually run vs reuse cached outputs”**.

The same Dream generation loop (`external/Dream/modeling/generation_utils.py`) hosts both tracks via:

- `delta_mode`: `"none" | "p2_early_stop" | "p3_learned_gate" | "p4_adaptive" | ...`
- `cache_mode`: `"none" | "l2c_ffn" | "l2c_learned" | "l2c_continuous" | ...`

Each phase in this doc answers three questions:

1. **Goal** – What problem it solves and what success looks like.
2. **Design** – What information it uses, and what the high‑level algorithm is.
3. **Integration** – Where it plugs into Dream (files, config knobs).

Dependencies between phases (for context, not “priority order”):

```text
P0 (baseline teacher)
  └─ P1 (traces)
      ├─ P2 (rule-based gate)
      ├─ P3 (learned gate)
      └─ Phase A (heuristic caching)
            └─ Phase B (learned router, fixed schedule)
                  └─ Phase C (continuous router, schedule transfer)
  └─ P4 (adaptive steps)  # uses same teacher as reference
```

---

## 3. Evaluation Setup & Metrics

**Primary evaluation settings**

- **Model:** `Dream-org/Dream-v0-Instruct-7B`
- **Tasks (Tier A – required):**
  - `gsm8k_cot` via Dream’s `eval_instruct` / lm‑eval harness (EM accuracy).
- **Optional extra tasks (Tier B – nice‑to‑have):**
  - One MMLU subject (e.g. `mmlu_astronomy`).
  - Additional Dream tasks (math/code) as time permits.

**Core metrics per configuration**

- **Latency per sample (ms)** — mean / p50 / p90 over prompts (batch=1).
- **Approximate compute** — number of **denoiser block forward passes** (FFN matmuls dominate).
- **Quality** — EM or accuracy on each task.
- **Speedup vs Teacher** — latency and compute ratio vs P0.

Target envelopes for “acceptable” acceleration modes:

- Latency reduction: **≥ 1.3×** versus P0 on GSM8K.
- EM drop: **≤ 1–2 percentage points** for “safe” modes; up to 3–5pp is reserved for more aggressive experiments.

Advanced metrics (Compute‑AUC, δ_frozen, oracle baselines, etc.) are important for analysis but not required by this spec; they live in `reports/` and per‑experiment docs.

---

## 4. Delta‑Compute Track (P0–P4)

This section defines **what each P‑phase means** from a system perspective. Detailed integration points and CLI examples are in `docs/IMPLEMENTATION_ROADMAP.md` and `scripts/slurm/*.sh`.

### 4.1 P0 – Baseline Teacher

**Goal.** Provide a *stable, reproducible* reference configuration for Dream‑7B Instruct used for all comparisons.

- **Config:**
  - `delta_mode="none"`, `cache_mode="none"`.
  - Fixed diffusion steps (e.g. `steps=256` for GSM8K CoT).
  - Fixed decoding hyper‑parameters: `temperature`, `top_p`, `max_new_tokens`, `top_k`, etc.
- **Code path:**
  - `external/Dream/modeling/modeling_dream.py`
  - `external/Dream/modeling/generation_utils.py::_sample`
  - Evaluation scripts under `external/Dream/eval_instruct`.
- **Deliverables:**
  - Reference EM and latency on GSM8K.
  - Baseline `diffusion_steps`, `max_new_tokens`, and batch size recorded under `experiments/P0_baseline/`.

P0 is the **non‑negotiable reference**: all later phases must be compared against these results.

### 4.2 P1 – Teacher Traces

**Goal.** Collect **lightweight, per‑layer, per‑step statistics** to support both gating (P2/P3/P4) and L2C caching (Phase A–C).

- **What we log (per layer ℓ and diffusion step t):**
  - FFN output norms and cosine similarities between consecutive steps.
  - (Optionally) attention output norms and cosines.
  - Per‑layer timing and per‑step denoiser wall‑clock time.
- **Implementation:**
  - `src/tracing.py::TraceCollector` defines `LayerStepStats` and serialization.
  - `external/Dream/modeling/modeling_dream.py::DreamDecoderLayer.forward` calls `trace_collector.record_layer(...)` with `ffn_output` and `attention_output`.
  - Generation loop (`generation_utils.py::_sample`) creates a `TraceCollector` when `trace_teacher=True` and saves `.pt` files under `experiments/P1_traces/`.
- **Usage:**
  - P2/P3 thresholds and P4 heuristics use trace‑derived statistics.
  - Phase A identifies “stable” layers/steps for conservative caching.
  - Phase B/C training uses traces and/or full layer activations.

### 4.3 P2 – Rule‑Based Gate (Sequence‑Level Early Stopping)

**Goal.** Stop the diffusion process early when the remaining masked tokens look “easy enough”, using a simple rule.

- **Mode:** `delta_mode="p2_early_stop"`.
- **Signals (on masked tokens only):**
  - Max probability per mask position.
  - Entropy of the predicted token distribution.
- **Logic (as implemented in `generation_utils.py`):**
  - At each diffusion step `i < steps - 1`:
    - Compute `masked_max_probs` and `masked_entropy` over positions where `x == mask_token_id`.
    - If either:
      - `masked_max_probs > confidence_threshold` for all masked tokens, or
      - `masked_entropy < entropy_threshold` for all masked tokens,
      then **stop early** and skip remaining steps.
- **Outputs:**
  - Same interface as P0, but with fewer diffusion steps.
  - Logs include how many steps were skipped and why.

P2 is the **simplest gating baseline**; P3 and P4 should strictly improve upon it in terms of flexibility and/or speedup.

### 4.4 P3 – Learned Gate

**Goal.** Replace P2’s hard‑coded thresholds with a **small learned MLP gate** that predicts when it is safe to stop, using richer features.

- **Model & features:**
  - Implemented in `src/learned_gate.py` (`LearnedGate`, `GateFeatures`, `GateTrainer`).
  - `GateFeatures` (9‑dimensional) includes:
    - Layer index, diffusion step index, total layers/steps.
    - FFN norms / cosine sim (from P1 traces).
    - Confidence, entropy, number of masked tokens, total tokens.
- **Labels:**
  - Derived from oracle experiments:
    - If freezing at a given (layer, step) doesn’t harm the final output vs P0, label = 1 (safe).
    - Otherwise label = 0 (unsafe).
- **Mode:** `delta_mode="p3_learned_gate"` (to be wired into `generation_utils.py`).
  - At each step, construct `GateFeatures` from current traces/logits.
  - Use `gate.should_freeze(features, threshold)` to decide whether to stop or freeze.

P3 should **match or beat P2** on GSM8K while giving more nuanced control over where and when we stop.

**Current implementation status (v1).**
- Label生成目前採用 **trace-based heuristic**：`scripts/training/generate_oracle_labels.py` 以 P1 的 FFN cosine 相似度為 proxy（cosine ≥ 阈值視為安全），尚未實作完整的 teacher/student ablation oracle。
- 決策是 **序列層級 early-stop**（使用最後一層的 trace + 當前 logits 的 confidence/entropy），尚未細化到 token-level freeze。
- 評估整合：`delta_mode="p3_learned_gate"` 已接上 Dream 生成迴圈並可載入 checkpoint（在 job 3628597 的 smoke test 中驗證可成功載入 gate）。

### 4.5 P4 – Adaptive Step Scheduling

**Goal.** Dynamically choose **how large a stride** to take along the diffusion schedule, based on local error estimates and risk signals, instead of using a fixed step size.

- **Model:** `src/adaptive_scheduler.py::AdaptiveScheduler`.
- **Signals per step:**
  - **Local truncation error (LTE):** approximate difference between Euler vs Heun‑like updates (implemented via logits change).
  - **Entropy:** uncertainty of token predictions on masked positions.
  - **KL divergence:** change in predicted distributions between consecutive steps.
- **Controller logic (high‑level):**
  - Never adapt before a configurable `min_safe_step` (stride=1).
  - If any signal exceeds its threshold → reduce stride back to 1.
  - If all signals stay below thresholds for `stable_window` steps → gradually increase stride up to `max_stride`.
- **State:** `SchedulerState` tracks current step, stride, history, and summary stats.
- **Hybrid mode:** `HybridScheduler` combines:
  - P4 for stride selection.
  - P3 gate decisions.
  - Phase A caching schedule.

P4 is orthogonal to L2C: it changes **how many diffusion steps we visit**, while Phase A–C change **how much work each visited step performs**.

**Current implementation status (v1).**
- LTE 以 logits 差異作為近似，entropy/kl 採 logits 上的平均值；stride 控制遵循上述安全邏輯。
- 已整合到 `delta_mode="p4_adaptive"`；修正了 entropy 索引的安全性（避免 mask 尺寸不符時的崩潰）。
- 仍需進一步以小規模評估微調閾值、觀察實際 stride 與速度/品質效果。

---

## 5. L2C‑Style Layer Caching Track (Phase A–C)

This section explains how we adapt **L2C layer caching** from image diffusion to Dream’s **token‑wise diffusion generation**.

### 5.1 Core Idea & “Time” Definition

Original L2C (for DiT / U‑ViT) learns to **reuse layer outputs across diffusion timesteps**. The natural “time” variable there is the diffusion timestep `t` (or log‑SNR).

For Dream‑7B’s masked‑token diffusion, we want a notion of “time” that:

1. Matches **generation progress** (how many tokens are already chosen).
2. Is robust when **timestep indices change** (e.g., under P4 adaptive stride or different schedules).
3. Can be shared between **Phase B’s fixed router** and **Phase C’s continuous router**.

To achieve this, we define a **token completion ratio**:

- Let `x ∈ ℤ^{B×N}` be the current token matrix (batch × max length).
- Let `mask_token_id` be the special token indicating *not yet decided*.
- At any diffusion step, define:
  - `M =` average number of masked tokens per sequence (see `num_mask_token` in `generation_utils.py`).
  - `N =` total number of token slots (`max_length`).
  - **Completion ratio:**  
    \[
      p = 1 - \frac{M}{N}  \in [0, 1]
    \]
    where `p=0` at the start (all MASK), and `p≈1` when the sequence is fully filled.

In this project:

- We treat **`p` as the canonical “time” variable** for L2C routers.
- When we say “continuous time” in Phase C, we mean **continuous progress `p`**, not raw diffusion step index.
- Block (layer) position:
  - For simplicity, **Phase C v1 can treat all layers as sharing the same time axis**, i.e., the router mainly conditions on `p`.
  - The implementation (`ContinuousRouter`) still supports per‑layer embeddings so later versions can learn per‑layer differences if needed.

This design is robust when:

- We change the number of diffusion steps (e.g., 128 / 256 / 512).
- P4 changes which steps are actually visited.

The router always sees “how complete the sequence is” instead of “which discrete step index we happen to be on”.

### 5.2 Common Integration Points

All L2C phases are implemented using the same infrastructure:

- **Dream model hooks** – `external/Dream/modeling/modeling_dream.py`:
  - `DreamDecoderLayer` accepts `use_ffn_cache`, `ffn_cache`, `trace_collector`, `diffusion_step`.
  - `DreamModel` propagates `layer_ffn_caches` and `cache_schedule` through all layers and returns updated caches.
- **Generation loop** – `external/Dream/modeling/generation_utils.py::_sample`:
  - `cache_mode` and `cache_schedule` control whether caching is active.
  - `layer_ffn_caches` is carried across steps.
  - `mask_index` and `num_mask_token` give us the token‑level view needed to compute `p`.

Phases A–C **must not** change the Dream backbone weights; they only change:

- When FFN matmuls run vs reuse cached outputs.
- How many steps we visit and how we route through them.

### 5.3 Phase A – Heuristic FFN Caching

**Goal.** Test whether reusing FFN outputs between steps gives **real speedup** with **tolerable quality loss**, using a conservative, fully hand‑designed schedule.

- **Mode:** `cache_mode="l2c_ffn"`.
- **Default behavior (current implementation):**
  - Maintain a per‑layer FFN cache: `layer_ffn_caches`.
  - For a configurable subset of layers (`cache_schedule`), and on selected steps, call
    `DreamDecoderLayer.forward(..., use_ffn_cache=True, ffn_cache=...)`.
  - When `use_ffn_cache` is True and a cache exists:
    - Skip the FFN matmul and reuse stored `mlp_output` from the previous step.
    - Always recompute attention (bidirectional attention remains step‑dependent).
- **Initial schedule (minimal PoC):**
  - Start with **even/odd alternation** or “cache only in later steps”.
  - Cache only **upper layers’ FFNs** where traces suggest high stability.
- **Metrics to inspect:**
  - FFN call count reduction vs P0 (per layer & overall).
  - Latency reduction vs P0 and P2.
  - EM degradation on GSM8K and MMLU‑astronomy.

Phase A defines the **interfaces and safety checks** for FFN caching. If even conservative schedules fail badly, Phase A results become **analysis only** (to guide later learned methods).

### 5.4 Phase B – Learned Router (Fixed Schedule)

**Goal.** Learn, for a **fixed diffusion schedule**, which layers to recompute vs reuse on each step, based on distillation from the teacher.

- **Model:** `src/learned_router.py::FixedScheduleRouter`.
  - Parameters β[m, ℓ] for each step index `m` and layer index `ℓ`.
  - `RouterTrainer` optimizes β with:
    - **Distillation loss:** student (with caching) vs teacher (no caching).
    - **Efficiency regularizer:** encourage lower recompute ratios.
- **Interpretation w.r.t. progress `p`:**
  - On a fixed schedule (e.g., 256 steps), each step index `m` corresponds to a **discrete progress level** `p_m`.
  - Phase B can be seen as learning a discretized version of a function β(ℓ, p).
- **Training data:**
  - Teacher runs without caching; for selected steps, capture per‑layer outputs.
  - Simulate “student” runs where some layers reuse cached activations according to sampled β.
- **Integration:**
  - Mode: `cache_mode="l2c_learned"`.
  - For each step `i`, load router and call `router.get_all_decisions(i, threshold)` to decide per‑layer recompute vs cache, then pass a boolean `cache_schedule` into `DreamModel`.

Phase B should improve upon Phase A by:

- Learning non‑trivial patterns: e.g., “always recompute early, cache mid‑layers later”.
- Achieving better **accuracy vs compute** Pareto front on the same fixed schedule.

**Current implementation status (v1).**
- 訓練目標採用 **cosine-based proxy**（`target ≈ 1 - ffn_cosine_sim`），尚未切換為完整的 teacher/student distillation 損失。
- 目前僅在單一固定 schedule（256 steps）上訓練並產出 `router_final.pt`；評估流程已接上 `cache_mode="l2c_learned"` 並能載入 router（job 3628597 smoke test驗證）。
- 待辦：加入 distillation 損失、分析 β heatmap、彙整 cache ratio vs EM 的 Pareto。

### 5.5 Phase C – Continuous Router (Progress‑Conditioned, Schedule‑Transferable)

**Goal.** Generalize Phase B to a **continuous function of progress**, so the router can transfer across different diffusion schedules and still make sensible decisions.

- **Model:** `src/continuous_router.py::ContinuousRouter`.
  - Inputs:
    - **Progress scalar `p ∈ [0,1]`** (our canonical “time”).
    - Layer index ℓ (via embedding; can be shared or per‑layer).
  - Architecture:
    - `TimeEncoder` (`TimeEncoder.forward(p)`) – sinusoidal or learned embedding of `p`.
    - Optional layer embedding.
    - Small MLP to output β_ℓ(p) ∈ [0,1], interpreted as recompute probability.
- **Key design shift (compared to older drafts):**
  - We **do not rely on raw diffusion step index `t`** in the spec.
  - Instead, we always feed a **progress ratio**:
    - At each visited diffusion step in `generation_utils.py`, compute `p` from the current `x` and `mask_token_id` as described in §5.1.
    - Pass this `p` into the router (in code, this is the scalar `t` fed into `TimeEncoder`).
  - This makes the router:
    - Robust to P4’s adaptive stride (fewer / more steps as long as progress is monotonic).
    - Naturally transferable between schedules with different numbers of steps.
- **Training (multi‑schedule):**
  - Collect training samples across several schedules (e.g., 128 / 256 / 512 steps).
  - For each sample:
    - Record teacher outputs plus the corresponding progress value `p` at that step.
    - Train `ContinuousRouter` to minimize distillation loss + efficiency regularizer, similar to Phase B.
  - `ContinuousRouterTrainer` supports training on multiple schedules and testing transfer via `test_schedule_transfer(...)`.
- **Integration:**
  - Mode: `cache_mode="l2c_continuous"` (name for clarity; exact string configurable).
  - At runtime:
    - Compute `p` at each diffusion step from current masked token ratio.
    - Query router for all layers at once: `router.get_schedule_decisions(p, ...)` or via `get_schedule_decisions(num_steps)` pre‑computed over a progress grid.
    - Convert probabilities into boolean cache flags and pass as `cache_schedule`.

Phase C’s success is measured by:

- Matching Phase B performance on the schedule it trained on.
- Maintaining good accuracy and speedup when the number of diffusion steps changes.
- Producing smooth β_ℓ(p) curves and interpretable heatmaps (see `visualize_schedule`).

**Current implementation status (v1).**
- 訓練目標同樣採 **cosine-based proxy**（`1 - ffn_cosine_sim`），且目前僅在單一 schedule（256 steps）訓練 `continuous_router_final.pt`。
- Eval 整合已可載入 router 並運行 `cache_mode="l2c_continuous"`（job 3628597 smoke test驗證）。
- 待辦：多 schedule 訓練與 transfer、加入 distillation/效率正則、生成 β(p) 熱圖與跨 schedule 表現分析。

---

## 6. Integration Summary (What Needs to Work Together)

This section lists **what must be wired together** for the full system, without prescribing execution order. Concrete TODOs and line‑level hints live in `docs/IMPLEMENTATION_ROADMAP.md`.

- **Dream generation mixin** – `external/Dream/modeling/generation_utils.py`
  - Parse `delta_mode`, `cache_mode`, and all related thresholds / checkpoints.
  - Host:
    - P2 early stopping logic (`delta_mode="p2_early_stop"`).
    - P3 learned gate hooks (feature construction + gate calls).
    - P4 adaptive stride loop (using `AdaptiveScheduler` / `HybridScheduler`).
    - Phase A FFN caching (`cache_mode="l2c_ffn"`).
    - Phase B router decisions (`cache_mode="l2c_learned"`).
    - Phase C continuous router decisions (`cache_mode="l2c_continuous"`) with progress‑based `p`.

- **Dream model** – `external/Dream/modeling/modeling_dream.py`
  - Correctly propagate:
    - `layer_ffn_caches`, `cache_schedule` down to `DreamDecoderLayer`.
    - `trace_collector`, `diffusion_step` for P1 statistics.
  - Ensure the FFN‑caching path produces numerically stable outputs and returns caches.

- **Eval wrappers** – `external/Dream/eval_instruct/lm_eval/models/diffllm.py`
  - Parse new `model_args` such as:
    - `delta_mode`, `cache_mode`.
    - `gate_checkpoint`, `router_checkpoint`.
    - P4 and Phase C hyper‑parameters (stride / thresholds).
  - Forward them into `DreamGenerationConfig`.

- **Training & analysis scripts**
  - `scripts/training/generate_oracle_labels.py` – gate labels for P3.
  - `scripts/training/train_learned_gate.py` – P3 gate training.
  - `scripts/training/train_learned_router.py` – Phase B router training.
  - `scripts/training/train_continuous_router.py` – Phase C router training.
  - `tests/visualization/plot_traces*.py` – P1 trace analysis that guides Phase A/B/C.

When all phases are implemented, the Dream stack should support **any combination** of:

- `delta_mode ∈ {none, p2_early_stop, p3_learned_gate, p4_adaptive}`
- `cache_mode ∈ {none, l2c_ffn, l2c_learned, l2c_continuous}`

with consistent logging, metrics, and reproducibility.

---

## 7. Completion Criteria (Per Phase)

This section summarizes what “done” means for each phase. Detailed testing strategy is in `TESTING_GUIDE.md` and `docs/IMPLEMENTATION_ROADMAP.md`.

- **P0 – Baseline Teacher**
  - Stable runs on GSM8K with fixed config.
  - Reference EM, latency, and compute recorded.

- **P1 – Teacher Traces**
  - `TraceCollector` works on GPU and does not disturb P0 results when disabled.
  - Trace files saved under `experiments/P1_traces/` and readable by plotting scripts.

- **P2 – Rule‑Based Gate**
  - Early stopping triggers on a non‑trivial fraction of GSM8K samples.
  - ≥1.3× speedup vs P0 with ≤2pp EM drop (or clear trade‑off curve documented).

- **P3 – Learned Gate**
  - Gate can be trained to reasonable validation F1 on oracle labels.
  - Integrated `delta_mode="p3_learned_gate"` matches or beats P2 on GSM8K.

- **P4 – Adaptive Steps**
  - Adaptive stride increases in stable regions and falls back to 1 on spikes.
  - Additional speedup over P2 for comparable quality on GSM8K.

- **Phase A – Heuristic Caching**
  - FFN caching path is numerically stable and preserves tensor shapes/dtypes.
  - Clear measurement of compute reduction vs P0.
  - EM degradation within agreed bounds (or documented as too aggressive).

- **Phase B – Learned Router**
  - Router learns meaningful patterns (β heatmaps not uniform).
  - On fixed schedule, Phase B outperforms Phase A in compute/quality trade‑off.

- **Phase C – Continuous Router**
  - Trained router matches Phase B on its training schedule.
  - Demonstrated schedule transfer (e.g., 256→128/512) with <5% relative degradation.
  - Progress‑based time input (`p`) verified to behave sensibly under P4.

Once all above criteria are met and corresponding reports are added under `docs/reports/` and `experiments/*/`, the **Dream‑7B Delta‑Compute & L2C project is considered complete** from a design/spec perspective. Implementation questions, ablations, and further ideas should be documented in `IMPLEMENTATION_ROADMAP.md` or new per‑phase design notes rather than over‑growing this master plan.

---

## 8. Key PoC Decisions (Open Questions Resolved)

- **Oracle labels (P3).** Hybrid plan: start with **heuristic labels** from P1 trace stability (FFN cosine sim, low norm drift) to cover the full (layer, step) grid cheaply, then add **sampled ablations** on the top/bottom 10% most/least stable regions to calibrate. Full exhaustive ablations are optional and only run if heuristic+sampled labels underperform.
- **Teacher traces for routers.** Default = **summary statistics only** (TraceCollector norms/cosines/timing) for all large runs. For Phase B/C training, enable **selective full-activation dumps** on ≤5% of samples/steps to avoid storage blow-up; training scripts must consume stats-first and fall back to activations when present.
- **Adaptive scheduler integration (P4).** PoC uses **stride-control overlay** on the existing diffusion loop (LTE/entropy/KL) without replacing the solver. Full loop refactor (Euler/Heun re-implementation) is deferred unless stride-control underperforms.
- **Multi-schedule training (Phase C).** Train the continuous router on a **mixed set of schedules {128, 256, 512}** using progress `p` as time; warm-start from a 256-step checkpoint is allowed, but final training/eval must include all three schedules with balanced sampling.
- **Evaluation priority/order.** Run in this sequence: 1) P0/P1 verification → 2) P2 early stop → 3) Phase A heuristic caching → 4) P3 learned gate → 5) Phase B learned router → 6) P4 adaptive stride → 7) Phase C continuous router (after Phase B baseline).
- **Checkpoint layout.** Standardize under `experiments/<phase>/` with `checkpoints/` (`{best,final,epoch-*}.pt`), `config.json`/`metadata.json` (hyperparams, data splits, schedule info), `logs/`, and phase artifacts (e.g., `oracle_labels*.json`, `teacher_traces/`, `activations/`). Use `*_best.pt` for primary deployment and keep symlink `latest.pt` to the newest saved checkpoint.
