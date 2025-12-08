# Dream-7B Delta-Compute & L2C Caching — Master Plan (v2)

_Last updated: 2025‑11‑18_

> **One-line goal.** On top of Dream‑7B’s existing evaluation stack, implement and evaluate **all phases of delta‑compute acceleration** (P0–P4: baseline, teacher traces, rule‑based gate, learned gate, adaptive steps) and **all phases of L2C‑style layer caching & skipping** (Phase A–E: heuristic caching, learned router, continuous progress router, continuous layer skipping, FlexiDepth‑style router+adapter), to **reduce latency / FLOPs** while keeping **quality regressions small and measurable**.
>
> **Scope.** This project implements and validates **all** of P0–P4 and Phase A–E. Everything described here is in‑scope for implementation and GPU evaluation; follow‑up ideas live in other docs. Status, timelines, and job IDs live in other docs; this file is the **design/spec “bible”**.

Related docs:
- High‑level progress (CN): `docs/PROGRESS_SUMMARY_CN.md`
- Implementation details & integration checklists: `docs/IMPLEMENTATION_ROADMAP.md`
- Experiment notes: `experiments/README.md` and `experiments/*/*.md`
 - Ongoing development status: `docs/progress_log.md`
 - Experiment registry (jobs, metrics, W&B runs): `docs/experiment_log.md`

---

## 0. Storyline & Research Questions

This project is not only an engineering effort to add knobs to Dream‑7B, but a **research study** of how different forms of delta‑compute behave in a diffusion‑LLM setting.

**RQ1 – Step vs layer methods (and their composition).**  
For Dream‑7B diffusion generation, how much latency / FFN FLOP reduction can we get from **step‑level methods** (primarily P4, with P2/P3 as baselines) vs **layer‑level methods** (Phase C–E) under the same quality budget on GSM8K and a non‑math task (e.g. MMLU‑astronomy)?
- **H1.** Combining adaptive step‑level gating (P4) with progress‑conditioned layer methods (Phase C–E) yields strictly better accuracy–compute trade‑offs than using either axis alone.

**RQ2 – Progress‑based routing and schedule transfer.**  
Does a **progress‑conditioned router** (Phase C–E) actually transfer across diffusion schedules `{128, 256, 512}` and tasks (GSM8K vs MMLU) without schedule‑ or task‑specific re‑tuning?
- **H2.** On GSM8K, a router that uses only progress `p` and layer index as its time signal can maintain ≤5% relative quality drop (per schedule) when evaluated on a new schedule at matched FFN compute, compared to a Phase‑C router trained and evaluated on that schedule. On the chosen MMLU subject, a GSM8K‑trained router maintains ≤5% relative quality drop vs the P0 teacher at matched FFN compute.

**RQ3 – Caching vs skipping vs adapters in masked‑token diffusion.**  
For masked‑token diffusion (Dream‑7B), are **L2C‑style FFN caches** fundamentally weaker than **layer skipping** or **router+adapter**?
- **H3.** On Dream‑7B, FFN caching (Phase A–C) underperforms identity skipping (Phase D) and FlexiDepth‑style adapters (Phase E) at matched FFN FLOP budgets, because cached FFN activations become stale when MASK tokens are replaced by concrete tokens.

All phases (P0–P4, A–E) are designed so that their **completion criteria directly answer some part of RQ1–RQ3**. Subsequent sections spell out the design and integration details.

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
│   ├── PhaseC_continuous/
│   ├── PhaseD_skip/
│   └── PhaseE_flexi/
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
  - **P3 – Learned gate.** Learned stopping rule using P1 features (sequence‑level in this project; token‑level freezing is explicitly out of scope).
  - **P4 – Adaptive steps.** Dynamic diffusion stride based on local truncation error and risk signals.

- **L2C Caching & Layer‑Skipping Track (Phase A–E)**  
  Control *how often each transformer block is recomputed across diffusion steps*, and when entire FFN blocks can be **skipped or approximated** instead of recomputed:
  - **Phase A – Heuristic FFN caching.** Fixed hand‑designed schedule: reuse FFN outputs on some steps/layers.
  - **Phase B – Learned router (fixed schedule).** Learn per‑step, per‑layer recompute vs reuse on a fixed schedule.
  - **Phase C – Continuous router (schedule‑transferable).** Learn per‑layer recompute probability as a function of **continuous progress** (still using FFN caching semantics).
  - **Phase D – Continuous layer skipping (no cache).** Reuse the continuous router, but change the action space to **recompute vs skip FFN** with identity residual (no cached activations).
  - **Phase E – FlexiDepth‑style skipping (router + adapter).** Replace the identity skip branch with a **trainable adapter** as a cheap FFN surrogate, following the core idea of *Adaptive Layer‑skipping in Pre‑trained LLMs* (a.k.a. FlexiDepth).

Conceptually:

-- P‑track (P0–P4) decides **“how many diffusion updates to do”** and **“when to stop”**.
-- Phase‑track (A–E) decides **“for this diffusion update, how much FFN compute we pay per layer (full / cached / skipped / adapter)”**.

The same Dream generation loop (`external/Dream/modeling/generation_utils.py`) hosts both tracks via:

- `delta_mode`: `"none" | "p2_early_stop" | "p3_learned_gate" | "p4_adaptive" | ...`
- `ffn_mode` in this spec (implemented as `cache_mode` in code): `"none" | "l2c_ffn" | "l2c_learned" | "l2c_continuous" | "skip_continuous" | "flexi_ffn"`.
  - Modes `l2c_*` implement **L2C‑style caching** (Phase A–C) and use `layer_ffn_caches`.
  - Modes `skip_continuous` and `flexi_ffn` implement **dynamic depth** (Phase D/E) and **do not use caches**; they interpret the schedule as FFN on/off flags.

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
                  └─ Phase C (continuous router, caching semantics, schedule transfer)
                        ├─ Phase D (continuous layer skipping, identity skip)
                        └─ Phase E (FlexiDepth-style router+adapter)
  └─ P4 (adaptive steps)  # uses same teacher as reference
```

---

## 3. Evaluation Setup & Metrics

**Primary evaluation settings**

- **Model:** `Dream-org/Dream-v0-Base-7B` *(Updated 2025-12-08: Base model achieves ~65% on GSM8K CoT while Instruct model only achieves ~25%)*
- **Official CoT Configuration (aligned with Dream eval):**
  - `max_new_tokens=256` *(must be in model_args, NOT gen_kwargs)*
  - `diffusion_steps=256`
  - `temperature=0.0` (greedy)
  - `top_p=0.95`
  - `batch_size=1`
  - `add_bos_token=true`
  - `num_fewshot=8`
  - Task: `gsm8k_cot`
- **Tasks (Tier A – required):**
  - `gsm8k_cot` via Dream’s `eval_instruct` / lm‑eval harness (EM accuracy).
  - At least one **non‑math subject** (e.g. `mmlu_astronomy`) to probe depth behavior on non‑chain‑of‑thought tasks.

**CoT-Specific Training Notes (added 2025-12-08)**

- **P2 Early Stop Thresholds:** The original `early_stop_confidence_threshold=0.95` and `entropy_threshold=0.1` were tuned for short-answer tasks. For CoT, re-sweep is required (try: `conf=0.6-0.8`, `entropy=0.5-1.5`) to avoid stopping mid-reasoning or never stopping.
- **P3 Gate / PhaseB/C Router Retraining:** Gates and routers trained on short-answer traces may not transfer well to CoT. Collect new P1 traces with CoT prompts and retrain before evaluating.
- **P1 Trace Collection:** Use official CoT config (256 steps, 8-shot, gsm8k_cot) on train split.

**Core metrics per configuration**

- **Latency per sample (ms)** — mean / p50 / p90 over prompts (batch=1).
- **Approximate compute** — number of **denoiser block forward passes** (FFN matmuls dominate).
- **Quality** — EM or accuracy on each task.
- **Speedup vs Teacher** — latency and compute ratio vs P0.

For methods that change FFN usage we will also track:

- **FFN_FLOPs (proxy).** Approximated as  
  \[
    \text{FFN\_FLOPs} = \sum_{\ell} \text{FFN\_params}_\ell \cdot \text{\#(FFN calls at layer }\ell\text{)}.
  \]
  This is the primary budget for RQ1/RQ3 comparisons (“matched FFN compute” = within ±5% FFN_FLOPs).
- **Router/adapter FLOPs (proxy).** Same formula applied to router and adapter modules; we log the ratio
  \[
    \text{Overhead} = \frac{\text{Router+Adapter FLOPs}}{\text{FFN FLOPs saved}}.
  \]
  Dynamic‑depth methods (Phase C–E) should keep this overhead ≤10–15% to be considered practically useful.
  We treat these FLOP counts as **layer‑wise param×call proxies** and do not explicitly correct for sequence‑length variation; since we compare methods under similar `max_new_tokens` and across tasks with broadly comparable answer lengths, we accept this approximation for this project.

Target envelopes for “acceptable” acceleration modes:

- Latency reduction: **≥ 1.3×** versus P0 on GSM8K.
- EM drop: **≤ 1–2 percentage points** for “safe” modes; up to 3–5pp is reserved for more aggressive settings where the trade‑off is explicitly documented.

Advanced metrics (Compute‑AUC, δ_frozen, oracle baselines, etc.) are important for analysis; concrete definitions and plotting utilities live in `reports/` and per‑experiment docs.

**Statistical protocol.**

- Core configurations (those used to answer RQ1–RQ3 and to compare methods in Section 8) should be evaluated with **at least 3 random seeds**.
- Reports under `docs/reports/` should include mean EM/accuracy and 95% confidence intervals.
- Thresholds such as “≤2pp drop” are interpreted in light of these intervals: differences whose confidence intervals overlap the P0 baseline are treated as not statistically significant.

**Data & splits (for training vs evaluation).**

- Routers and gates (P2 thresholds, P3 gate, Phase B/C/D/E routers/adapters) are trained **only** on:
  - Dream’s original training mixture for Dream‑7B (the same “generic instruct” data used for the base model; see `DATA_SOURCES.md` once finalized), and/or
  - A dedicated **train split** of GSM8K / MMLU prompts that is disjoint from the eval splits used in this section.
- P1 traces used for Phase B/C/D/E training are collected **only on train splits**. Evaluation traces on GSM8K / MMLU are used for analysis and visualization, not for training.
- For RQ2 (schedule and task transfer), routers are trained on GSM8K train (plus the generic instruct mixture if used) and evaluated on GSM8K test and the chosen MMLU subject; no MMLU data is used during router training.

**Experiment artifacts & logging.**

- All experiments must produce **reproducible artifacts** with clear naming:
  - `experiments/<phase>/checkpoints/` – model weights (`*_best.pt`, `*_final.pt`, `*_epoch-*.pt`).
  - `experiments/<phase>/configs/` or `metadata.json` – training/eval configuration (delta_mode, ffn_mode, schedule, dataset, seed, FFN_FLOPs target, etc.).
  - `experiments/<phase>/logs/` – per‑run metrics and summaries (e.g., JSON/CSV of EM, FFN_FLOPs, overhead).
  - `logs/slurm/` – SLURM `.out`/`.err` logs for all cluster jobs (training and eval).
- Two global log files under `docs/` are required:
  - `docs/progress_log.md` – human‑readable development journal (what was done, why, status, linked commits and jobs).
  - `docs/experiment_log.md` – registry of all training/eval runs (job IDs, scripts used, metrics, artifacts, W&B run names).
- For remote logging, we use **Weights & Biases (W&B)**:
  - Default project: `dllm_delta_compute` (set via `WANDB_PROJECT`).
  - Entity is configured per user via `WANDB_ENTITY`.
  - Run names should encode phase, schedule, task, and seed (e.g., `phaseC_s256_gsm8k_seed1`, `P4_phaseD_gsm8k_s256_ffn0.7_seed3`).
  - Tags should include at least: phase (`P3`, `PhaseC`, ...), dataset (`gsm8k`, `mmlu_astronomy`), schedule (`s128/s256/s512`), `delta_mode`, `ffn_mode`, and a short description (`baseline`, `matched_ffn0.7`, etc.).

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

- **Threshold selection protocol.**
  - We tune `(confidence_threshold, entropy_threshold)` via a small grid search on a **GSM8K validation subset** disjoint from the final eval set (e.g., 1k prompts).
  - Once selected, these thresholds are **frozen** and reused for all tasks (GSM8K test, MMLU subject) so that P2 is a stable rule‑based baseline.

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
- Current label generation uses a **trace-based heuristic**: `scripts/training/generate_oracle_labels.py` treats high FFN cosine similarity in P1 traces (cosine ≥ threshold) as “safe”, so we have not yet implemented full teacher/student ablation‑based oracles.
- Decisions are **sequence‑level early stops** (using the last layer’s traces plus current logits confidence/entropy); token‑level freezing is out of scope for this project.
- Evaluation integration: `delta_mode="p3_learned_gate"` is wired into the Dream generation loop and can load checkpoints; smoke tests confirm basic functionality.

**Design limitation (sequential nature).**
- In this project we treat gate decisions as **independent binary classifications per step**, with oracle labels derived from full trajectories in an offline manner. We do not model sequential credit assignment (e.g., via RL or sequence‑level loss), so the learned gate may fail to fully capture the long‑term impact of stopping one step earlier. This limitation must be called out explicitly in analysis and any publications.

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
- LTE is currently approximated from logits differences between Euler‑ vs Heun‑like updates, and entropy/KL are computed as averages over masked‑token logits; stride control follows the safety logic above.
- Integration with `delta_mode="p4_adaptive"` is in place; entropy indexing has been hardened to avoid crashes when mask sizes do not match expectations.
- Further small‑scale evaluations are needed to tune thresholds and understand the realized stride distribution vs speed/quality trade‑offs.

---

## 5. L2C‑Style Layer Caching & Skipping Track (Phase A–E)

This section explains how we adapt **L2C layer caching** from image diffusion to Dream’s **token‑wise diffusion generation**, and how we extend the same router infrastructure to **dynamic depth** via layer skipping and adapters.

At a high level, this track has **two related but distinct families** of methods:

- **L2C‑style caching (Phase A–C).** Decide when to **recompute vs reuse cached FFN activations across diffusion steps**, using fixed, learned, and continuous routers.
- **Dynamic depth via skipping / adapters (Phase D–E).** Decide when to **recompute vs skip or approximate FFN blocks within a diffusion step**, without reusing stale activations:
  - Phase D uses an **identity skip branch** (pure residual; no extra parameters).
  - Phase E uses a **small adapter FFN** as a cheap surrogate for the full FFN, following FlexiDepth.

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

This definition intentionally **approximates** progress:

- We use `max_length` (not per‑sample effective length) and average masked tokens `M` across the batch, so `p` can be mis‑estimated when sequences have very different lengths or padding patterns.
- Sequences that never fully unmask (e.g., early EOS) may end with `p < 1`; in practice we still treat their final `p` as “late‑stage” progress.
- For this project we accept these approximations and do not implement per‑sample or per‑token `p`; any artifacts from this choice will be analyzed but not fixed within this master plan.

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

Phases A–E **do not** change the Dream backbone weights; they only change:

- For Phase A–C (caching): when FFN matmuls run vs reuse cached outputs, and how many steps we visit and how we route through them.
- For Phase D–E (dynamic depth): when FFN matmuls run vs are skipped or approximated by adapters within a step.

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
Phase B’s role in the overall research story is two‑fold:
- Provide a fixed‑schedule learned baseline showing the value of data‑driven caching over hand‑crafted schedules (directly comparing against Phase A).
- Serve as a warm‑start and sanity check for Phase C’s continuous router; most cross‑axis comparisons for RQ1/RQ3 will focus on Phase C–E rather than Phase B.

**Current implementation status (v1).**
- Training objective currently uses a **cosine‑based proxy** (`target ≈ 1 - ffn_cosine_sim`) rather than a full teacher/student distillation loss.
- The router has only been trained on a single fixed schedule (256 steps) to produce `router_final.pt`; evaluation code is wired to `cache_mode="l2c_learned"` and can load this router (smoke tests confirm basic functionality).
- TODO: incorporate distillation loss, analyze β heatmaps, and summarize cache ratio vs EM Pareto curves.

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
- **Practical router input design (progress‑based).**
  - Always use **sequence‑level progress** as the primary time signal: `p = 1 - M/N`, where `M` is the number of masked tokens and `N` is the max sequence length. This keeps `p ∈ [0, 1]` and monotonic across any schedule or P4 stride.
  - Treat **block‑level decoded ratio** or other local statistics (per‑layer mask ratio, FFN cosine stability, entropy, etc.) as **optional local features**, not as the main time axis.
  - A practical input to `ContinuousRouter` is:
    ```python
    router_input = concat(
        TimeEncoder(p),             # global progress
        layer_embedding[layer_idx], # layer id
        optional_local_stats,       # e.g., local mask ratio, trace features
        optional_schedule_emb * γ   # small‑weight schedule embedding if needed
    )
    ```
    where `γ` is small so that routing remains primarily driven by `p` instead of overfitting to a particular step grid.
- **Practical training loop (single‑ and multi‑schedule).**
  - **Single‑schedule v1:** train `ContinuousRouter` on a single schedule (e.g., 256 steps) using `p` computed from P1 traces; at evaluation time, reuse the same router on 128/256/512 schedules by recomputing `p` from `x` and `mask_token_id` at each visited step.
  - **Multi‑schedule v2:** build a dataset mixing schedules `{128, 256, 512}`:
    - For each training example, store `(schedule_id, layer_idx, p, teacher target, optional local stats)`.
    - Use a distillation loss such as `||h_student(p; β) - h_teacher||^2` or KL, plus an efficiency regularizer on recompute ratio.
    - Sample batches so that all schedules are reasonably represented; optionally include a weak `schedule_embedding[schedule_id] * γ` in router inputs for fine‑grained adjustments.
- **Interaction with P4.**
  - At runtime, even under `delta_mode="p4_adaptive"`:
    - For every **actually executed** diffusion update, recompute `p = 1 - M/N` from the current `x` and `mask_token_id`.
    - Call the continuous router with this `p` (and layer indices / local stats) to obtain per‑layer recompute/cache decisions.
    - Independently, let P4’s `AdaptiveScheduler` use LTE/entropy/KL to choose the next stride. The router never needs to see raw step indices, only the current progress `p`.
  - In this project we treat P4’s stride controller and Phase C–E routers as **orthogonal modules**: P4 thresholds are tuned on the P0 teacher and then reused when Phase C–E are enabled. We acknowledge that enabling Phase C–E can change the distribution of LTE/entropy/KL and the trajectory of `p`; we do not re‑tune P4 per layer‑method and treat any interaction effects as part of the empirical findings.
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
- Training objective also uses the **cosine‑based proxy** (`1 - ffn_cosine_sim`) and currently trains `continuous_router_final.pt` on a single schedule (256 steps).
- Evaluation integration can load the router and run with `cache_mode="l2c_continuous"` (smoke tests confirm basic functionality).
- TODO (within this project): multi‑schedule training and transfer experiments, add distillation/efficiency regularizers, and generate β(p) heatmaps plus cross‑schedule performance analysis.
- Engineering status: the router already supports progress `p` as the time axis and an optional schedule embedding; the training script can accept multiple trace directories and corresponding schedule IDs (single‑schedule runs keep schedule embedding disabled by default).

### 5.6 Phase D – Continuous Layer Skipping (No Cache)

**Goal.** Reuse the Phase C continuous router, but switch the action space from **recompute vs cache** to **recompute vs skip**. The router directly decides, as a function of progress `p`, which FFN blocks can be skipped entirely without relying on stale cached outputs. This is a closer fit for Dream’s masked‑token diffusion, where the set of masked/unmasked positions changes over time and makes FFN caches less reliable.

- **Mode:** `cache_mode="skip_continuous"` (tentative name; exact string configurable).

- **Model.**
  - Reuse `ContinuousRouter` from Phase C, with the same inputs:
    - Progress scalar `p ∈ [0,1]` computed from the masked‑token ratio (§5.1).
    - Layer index ℓ (embedding).
    - Optional local stats (e.g., FFN norms or cosine stability from P1 traces).
  - Instead of controlling **reuse vs recompute**, the router outputs per‑layer recompute probabilities:
    - `β_ℓ(p) ∈ [0,1]` interpreted as "probability of recomputing this layer’s FFN".
    - At training time, use a soft combination
      \[
        h_{\text{out}} = h_{\text{in}} +
          β_ℓ(p)\cdot \text{FFN}(h_{\text{in}}),
      \]
      which is equivalent to
      \[
        h_{\text{out}} =
          β_ℓ(p)\cdot \left(h_{\text{in}} + \text{FFN}(h_{\text{in}})\right)
          + (1 - β_ℓ(p))\cdot h_{\text{in}},
      \]
      so gradients flow through both paths. At inference time, threshold `β_ℓ(p)` to get a binary "recompute vs skip" decision.

- **Training.**
  - Teacher: Dream‑7B with all FFNs enabled (P0 configuration).
  - Student: Dream‑7B with Phase D skipping active; backbone weights (attention, FFN, LayerNorm) are frozen.
  - Loss:
    - Main loss: standard Dream LM/diffusion loss on GSM8K plus, optionally, the same generic instruct mixture used to train Dream‑7B (see `DATA_SOURCES.md`).
    - Optional distillation loss:
      - Match hidden states or logits between teacher and student on a subset of steps and layers.
    - Skip regularizer:
      - Encourage lower expected depth via a squared penalty on per‑sample compute
        \[
          L_{\text{skip}} = \frac{1}{T}\sum_t \left(\sum_{\ell} β_ℓ(p_t)\right)^2.
        \]
      - Combined objective: `L = L_main + α L_skip`, with α tuned to achieve a target skip ratio (for example 20–30% FFN reduction).

- **Integration.**
  - Uses the same hooks as Phase C:
    - Progress `p` is recomputed at every diffusion step from the current masked‑token ratio.
    - The router is called once per step to produce `β_ℓ(p)` for all layers.
  - `DreamDecoderLayer` receives a per‑layer binary skip flag; if the FFN for that layer is skipped:
    - Attention still runs as in P0 (for simplicity).
    - FFN is replaced by an identity residual: `y = h + 0`, so only attention contributes at that layer.
  - Implementation‑wise, Phase D can share the same `cache_schedule` plumbing as Phase A–C, but the semantics become **"recompute vs skip FFN"** instead of **"recompute vs reuse cached FFN"**.

- **Why this is separate from caching.**
  - Skipping does not rely on reusing old FFN outputs, which may be stale once MASK tokens become concrete tokens.
  - Phase D isolates the question "how much depth can we remove, purely by skipping FFNs" from the orthogonal question "can we safely reuse activations across steps".

- **Success criteria.**
  - On GSM8K, achieve at least ~20–30% FFN FLOP reduction with ≤2pp EM drop compared to P0.
  - Router heatmaps `β_ℓ(p)` show interpretable patterns:
    - Early diffusion steps and heavily masked sequences use more FFN layers.
    - Later steps and near‑complete sequences tend to skip mid or upper FFN layers.

### 5.7 Phase E – FlexiDepth‑Style Layer Skipping (Router + Adapter)

**Goal.** Upgrade Phase D’s identity skip into a **FlexiDepth‑inspired dynamic depth mechanism** where the skip branch is not a pure identity, but a small adapter network. The main Dream‑7B backbone remains frozen; only router and adapters are trained. This Phase adapts the core router+adapter idea from the *Adaptive Layer‑skipping in Pre‑trained LLMs* (FlexiDepth) paper to the Dream‑7B diffusion setting under a simplified per‑sequence gating design.

- **Mode:** `cache_mode="flexi_ffn"` (router+adapter, no caching).

- **Model.**
  - For the top K decoder layers (for example 8 of 32), attach:
    - A bottleneck MLP router (can reuse `ContinuousRouter` or a slightly higher‑capacity variant).
    - A narrow FFN adapter that mimics the structure of the full FFN with much smaller width.
  - Router inputs:
    - Normalized layer input `h_in` (e.g., RMSNorm).
    - Progress scalar `p`.
    - Layer index ℓ embedding.
    - Optional local stats from P1 traces (FFN norms, cosine stability) or logits (entropy, confidence).
  - For each such layer, the forward pass is:
    - Compute gate score `g_ℓ(p, h_in) ∈ (0,1)` via the router.
    - If `g_ℓ > τ` (deep path):
      - Run the full FFN: `ffn_out = FFN(Norm(h_in))`.
    - If `g_ℓ ≤ τ` (shallow path):
      - Run the adapter: `ffn_out = Adapter(Norm(h_in))`.
    - Residual: `h_out = h_in + ffn_out`.
  - During training, a soft combination can be used to provide gradients to both branches; at inference, a hard threshold `τ` selects full FFN vs adapter.

- **Adapter design.**
  - Structure: mini‑FFN with bottleneck dimension `d_r << d_ff`:
    \[
      \text{Adapter}(x) = W_\text{up}\,\phi(W_\text{down} x),
    \]
    where `φ` is GELU or similar, `W_down ∈ ℝ^{d_r×d}`, `W_up ∈ ℝ^{d×d_r}`.
  - Intended role: provide a cheap nonlinear transform so that tokens that frequently skip the full FFN still live in a compatible latent space, instead of pure identity. Directly skipping FFN with an identity shortcut is known (from FlexiDepth ablations) to severely hurt accuracy.

- **Granularity.**
  - In this project we standardize on **per‑sequence gating**:
    - Pool `h_in` across the sequence (e.g., mean over tokens, or mean over masked tokens) before feeding it to the router; produce a single scalar `g_ℓ` per layer and diffusion step.
    - This keeps the control‑flow uniform within a batch and simplifies engineering.
  - Per‑group/per‑token gating is a possible follow‑up direction but is explicitly out of scope for this master plan.

- **Training.**
  - Backbone weights (attention, FFN, LayerNorm) are frozen.
  - Train only:
    - Router parameters.
    - Adapter parameters.
    - Optional progress and layer embeddings.
  - Loss:
    - Main Dream LM/diffusion loss on GSM8K (and, if used, the same generic instruct mixture as above; objective aligned with other phases).
    - Optional distillation loss between teacher (no skipping) and student (router+adapter) on hidden states or logits.
    - Skipping regularizer:
      - Same form as Phase D, but applied to the FlexiDepth gates:
        \[
          L_{\text{skip}} = \frac{1}{T}\sum_t \left(\sum_{\ell} g_{ℓ,t}\right)^2.
        \]
      - Global coefficient α tuned to achieve a target FFN reduction budget (for example 20–40%).

- **Integration.**
  - Reuses the same progress‑based routing infrastructure as Phase C/D.
  - Limited to a subset of layers to keep FLOPs overhead of adapters small.
  - Uses the P1 `TraceCollector` (and any extra logging) to log router scores and visualize depth maps:
    - Which diffusion steps and sequences (and coarse reasoning segments) prefer the full FFN path.
    - Which ones rely mostly on adapters.

- **Research questions.**
  - RQ‑E1: In Dream‑7B diffusion generation, can a FlexiDepth‑style router+adapter reduce FFN FLOPs by ~20–40% while keeping GSM8K EM within ≤2pp of P0?
  - RQ‑E2: Do learned depth maps in the diffusion setting exhibit intuitive patterns at the **sequence / segment level** (for example later diffusion steps and reasoning segments preferring deeper paths), even with per‑sequence gating?
  - RQ‑E3: Under a matched compute budget, how does Phase E (router+adapter) compare against:
    - Phase D (identity skipping).
    - Phase B/C (caching‑based reuse).

- **Success criteria.**
  - Achieve a strictly better accuracy vs compute Pareto front than Phase D on GSM8K.
  - Produce interpretable depth maps that qualitatively resemble the original FlexiDepth findings, adapted to masked‑token diffusion.

---

## 6. Integration Summary (What Needs to Work Together)

This section lists **what must be wired together** for the full system, without prescribing execution order. Concrete TODOs and line‑level hints live in `docs/IMPLEMENTATION_ROADMAP.md`.

- **Dream generation mixin** – `external/Dream/modeling/generation_utils.py`
  - Parse `delta_mode`, `cache_mode` (FFN policy mode in code), and all related thresholds / checkpoints.
  - Host:
    - P2 early stopping logic (`delta_mode="p2_early_stop"`).
    - P3 learned gate hooks (feature construction + gate calls).
    - P4 adaptive stride loop (using `AdaptiveScheduler` / `HybridScheduler`).
    - Phase A FFN caching (`cache_mode="l2c_ffn"`).
    - Phase B router decisions (`cache_mode="l2c_learned"`).
    - Phase C continuous router decisions (`cache_mode="l2c_continuous"`) with progress‑based `p`.
    - Phase D continuous layer skipping decisions (`cache_mode="skip_continuous"`) using the same router but skip semantics.
    - Phase E FlexiDepth‑style router+adapter decisions (`cache_mode="flexi_ffn"`).

- **Dream model** – `external/Dream/modeling/modeling_dream.py`
  - Correctly propagate:
    - `layer_ffn_caches`, `cache_schedule` down to `DreamDecoderLayer`.
    - `trace_collector`, `diffusion_step` for P1 statistics.
  - Ensure the FFN‑caching path produces numerically stable outputs and returns caches.
  - For skip modes (Phase D/E), interpret `cache_schedule` (or an equivalent mask) as **FFN on/off flags** and keep attention behavior identical to P0.
  - In skip modes, `layer_ffn_caches` should remain unset / ignored; any attempt to read or write FFN caches under `ffn_mode ∈ {skip_continuous, flexi_ffn}` should be guarded by assertions to prevent accidental mixing of caching and skipping semantics.

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
  - `scripts/training/train_skip_router.py` – Phase D router fine‑tuning with skip semantics.
  - `scripts/training/train_flexi_adapter.py` – Phase E router+adapter training.
  - `tests/visualization/plot_traces*.py` – P1 trace analysis that guides Phase A/B/C/D/E.
  - All training scripts should optionally integrate with W&B (e.g., via a `--use_wandb` flag and environment variables `WANDB_PROJECT`, `WANDB_ENTITY`), logging metrics such as loss, validation metrics, FFN_FLOPs, and overhead for each run.

When all phases are implemented, the Dream stack should support **any combination** of:

- `delta_mode ∈ {none, p2_early_stop, p3_learned_gate, p4_adaptive}`
- `ffn_mode` (implemented as `cache_mode` in code)  
  `∈ {none, l2c_ffn, l2c_learned, l2c_continuous, skip_continuous, flexi_ffn}`

with consistent logging, metrics, and reproducibility.

---

## 7. Completion Criteria (Per Phase)

This section summarizes what “done” means for each phase. Detailed testing strategy is in `TESTING_GUIDE.md` and `docs/IMPLEMENTATION_ROADMAP.md`.

For each phase, if the numerical targets below (e.g., 2pp EM drop, 1.3× speedup) cannot be reached after reasonable tuning under the protocols in §3 and §8, the phase is still considered **research‑complete** as long as:

- The implementation is correct, numerically stable, and integrated with the Dream stack.
- A clear empirical trade‑off curve (quality vs FFN_FLOPs vs router/adapter overhead) is reported and analyzed under `docs/reports/`.

In such cases, results must be explicitly framed as negative or partially‑negative findings rather than omitted.

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

- **Phase D – Continuous Layer Skipping**
  - Progress‑based router successfully skips ≥20% FFN layers on GSM8K with ≤2pp EM drop.
  - Skip patterns are stable across runs and visualized in depth maps over progress `p`.

- **Phase E – FlexiDepth‑Style Layer Skipping**
  - Router+adapter training converges with backbone frozen.
  - For a matched compute budget, Phase E matches or outperforms Phase D on GSM8K and at least one additional task (e.g., MMLU‑astronomy).

Once all above criteria are met and corresponding reports are added under `docs/reports/` and `experiments/*/`, the **Dream‑7B Delta‑Compute & L2C project is considered complete** from a design/spec perspective. Implementation questions, ablations, and further ideas should be documented in `IMPLEMENTATION_ROADMAP.md` or new per‑phase design notes rather than over‑growing this master plan.

---

## 8. Key Design Decisions (Open Questions Resolved)

- **Oracle labels (P3).** Hybrid plan: start with **heuristic labels** from P1 trace stability (FFN cosine sim, low norm drift) to cover the full (layer, step) grid cheaply, then add **sampled ablations** on the top/bottom 10% most/least stable regions to calibrate. Full exhaustive ablations are out of scope for this project.
- **Teacher traces for routers.** Default = **summary statistics only** (TraceCollector norms/cosines/timing) for all large runs. For Phase B/C/D/E training, enable **selective full‑activation dumps** on ≤5% of samples/steps to avoid storage blow‑up; training scripts must consume stats‑first and fall back to activations when present.
- **Adaptive scheduler integration (P4).** We use a **stride‑control overlay** on the existing diffusion loop (LTE/entropy/KL) without replacing the solver. A full loop refactor (Euler/Heun re‑implementation) is explicitly out of scope for this project.
- **Multi‑schedule training (Phase C).** Train the continuous router on a **mixed set of schedules {128, 256, 512}** using progress `p` as time; warm‑start from a 256‑step checkpoint is allowed, but final training/eval must include all three schedules with balanced sampling.
- **Compute‑matching protocol (RQ3).** When comparing caching (Phase C) vs skipping (Phase D) vs router+adapter (Phase E), we will:
  - Choose several target FFN usage ratios (e.g. 0.8, 0.7, 0.6 of P0’s FFN_FLOPs).
  - For each method and target, sweep its threshold(s) or regularizer `α` on a held‑out GSM8K validation split to find configurations whose FFN_FLOPs lie within ±5% of the target.
  - For each target and method, pick the configuration with the best validation EM under this FFN_FLOPs constraint, then report GSM8K/MMLU test results at these matched points.
  - Conclusions for RQ3 are based on these matched‑compute comparisons, not on arbitrary single hyperparameter choices.
- **Required cross‑axis configurations (RQ1).** To compare step‑ vs layer‑level methods and their composition, we will explicitly evaluate at least the following configurations on GSM8K and the chosen non‑math task:
  - `delta_mode=none,            ffn_mode=none`            (P0 baseline)
  - `delta_mode=p4_adaptive,     ffn_mode=none`            (P4 only)
  - `delta_mode=none,            ffn_mode=l2c_continuous`  (Phase C only)
  - `delta_mode=p4_adaptive,     ffn_mode=l2c_continuous`  (P4 + Phase C)
  - `delta_mode=none,            ffn_mode=skip_continuous` (Phase D only)
  - `delta_mode=p4_adaptive,     ffn_mode=skip_continuous` (P4 + Phase D)
  - `delta_mode=none,            ffn_mode=flexi_ffn`       (Phase E only)
  - `delta_mode=p4_adaptive,     ffn_mode=flexi_ffn`       (P4 + Phase E)
- **Evaluation priority/order.** Run in this sequence: 1) P0/P1 verification → 2) P2 early stop → 3) Phase A heuristic caching → 4) P3 learned gate → 5) Phase B learned router → 6) P4 adaptive stride → 7) Phase C continuous router (after Phase B baseline) → 8) Phase D continuous layer skipping → 9) Phase E FlexiDepth‑style router+adapter.
- **Checkpoint layout.** Standardize under `experiments/<phase>/` with `checkpoints/` (`{best,final,epoch-*}.pt`), `config.json`/`metadata.json` (hyperparams, data splits, schedule info), `logs/`, and phase artifacts (e.g., `oracle_labels*.json`, `teacher_traces/`, `activations/`). Use `*_best.pt` for primary deployment and keep symlink `latest.pt` to the newest saved checkpoint.

**Minimal research slice (if resources are tight).**

- From a research‑question perspective, the minimal set of phases needed to answer RQ1–RQ3 consists of:
  - P0–P2 (baseline + rule‑based gating), P4 (adaptive steps), and Phase C–E (continuous caching, identity skipping, FlexiDepth‑style adapters).
- Phase A/B are still required for infrastructure and fixed‑schedule baselines, but the main cross‑axis comparisons and claims in any paper should be based on P0–P2/P4 and Phase C–E.

---

## 9. Limitations & Assumptions (For Analysis)

This project makes several simplifying assumptions; analyses and any publications must treat these as explicit limitations rather than implicit facts:

- **Progress definition.** Progress `p` uses `max_length` and batch‑average masked token counts; we do not correct for per‑sample length or padding, and sequences that never fully unmask may still be treated as “late‑stage” via their final `p`.
- **Granularity of routing.** All routers in Phase C–E use **per‑sequence gating** (with optional layer embeddings), not per‑token gating; token‑level depth patterns are not explicitly modeled.
- **Sequential credit assignment (P3).** P3 treats each step’s gate decision as an independent binary classification with offline oracle labels; we do not model sequence‑level credit assignment or RL‑style training for gate policies.
- **P4 interaction.** P4 thresholds are tuned on the P0 teacher and reused when Phase C–E are enabled; we do not re‑tune P4 for each layer‑method, even though routers can change LTE/entropy/KL distributions.
- **Training data scope.** Router and gate training is restricted to Dream’s training mixture and GSM8K train splits; cross‑task generalization is evaluated only on the chosen MMLU subject without using MMLU data during training.
