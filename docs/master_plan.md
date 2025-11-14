# Dream-7B Delta-Compute & L2C Caching — Master Plan (v1)

> **One-line goal.** On top of Dream-7B's existing evaluation stack, implement and evaluate **ALL phases of delta-compute acceleration** (P0–P4: baseline, teacher traces, rule-based gate, learned gate, adaptive steps) and **ALL phases of L2C-style layer caching** (Phase A–C: heuristic caching, learned router, continuous-time router), to **reduce latency / FLOPs** while keeping **quality regressions small and measurable**.
>
> **Scope: This project implements and validates ALL P0-P4 and Phase A-C. There is no "future work" — all ideas listed here are in scope for completion and validation.**

---

## 0. Environment & Repos

- **Conda env (recommended name):** `dcllm`  
  - Create locally, e.g.:  
    ```bash
    conda create -n dcllm python=3.10
    conda activate dcllm
    ```
  - Install Dream and this project’s requirements into this env.
- **Core library versions (aligned with Dream README):**
  - `torch==2.5.1`
  - `transformers==4.46.2`
  - CUDA version matched to the GPU cluster / machine.
- **Repo layout (this project):**
  - This repo: `DLLM-Delta-Compute/`
  - Dream submodule / checkout: `external/Dream/`
  - Dream model source (vendored from HF): `external/Dream/modeling/`
  - L2C reference implementation (DiT / U-ViT): `external/learning-to-cache/`
- **Determinism for eval (high-level rule):**
  - All evaluation scripts should fix a random seed (e.g. `seed=42`) and disable cuDNN autotune where necessary to keep POC results reproducible. The exact code-level settings can live in the eval scripts / wrappers.

All experiments in this plan are assumed to run inside an activated **`dcllm`** env (local path is machine-specific).

### Project Structure & Organization Rules

```
DLLM-Delta-Compute/
├── src/                      # Core delta-compute infrastructure
│   ├── tracing.py           # TraceCollector for P1 teacher traces
│   └── __init__.py
├── tests/                    # All test scripts (organized by type)
│   ├── unit/                # Unit tests (GPU validation, infrastructure)
│   ├── integration/         # Full workflow tests (model loading, generation)
│   └── visualization/       # Trace plotting and analysis scripts
├── scripts/                  # All executable scripts
│   ├── slurm/              # SLURM batch job scripts (eval_*.sh)
│   ├── setup/              # Environment setup scripts
│   └── eval/               # Local evaluation scripts
├── experiments/              # Experimental runs (organized by phase)
│   ├── P0_baseline/        # Baseline teacher experiments
│   ├── P1_traces/          # Trace collection runs
│   ├── P2_early_stop/      # Early stopping experiments
│   ├── P3_learned_gate/    # Learned gate experiments
│   ├── P4_adaptive/        # Adaptive step scheduling
│   ├── PhaseA_caching/     # Heuristic FFN caching
│   ├── PhaseB_router/      # Learned router experiments
│   └── PhaseC_continuous/  # Continuous-time router
│   # Each phase directory contains:
│   # ├── logs/        # SLURM outputs (*.out, gitignored)
│   # ├── traces/      # Trace .pt files (gitignored)
│   # ├── results/     # JSON metrics (tracked in git)
│   # └── configs/     # Experiment configs (tracked in git)
├── docs/                     # Documentation
│   ├── master_plan.md       # THIS FILE: Primary specification
│   ├── implementation_summary.md  # Implementation details
│   ├── reports/            # Validation and evaluation reports
│   └── archive/            # Historical documents (reference only)
├── external/                 # External dependencies
│   ├── Dream/              # Dream-7B (git submodule)
│   └── learning-to-cache/  # L2C reference (git submodule)
├── configs/                  # Model and experiment configurations
├── reports/                  # Generated reports and visualizations
├── .gitignore               # Git ignore patterns
├── requirements.txt         # Python dependencies
├── README.md                # Project overview
└── TESTING_GUIDE.md         # Testing instructions
```

**Organization Rules:**

1. **Source code** → `src/` (reusable infrastructure only)
2. **Tests** → `tests/` (unit, integration, visualization)
3. **Scripts** → `scripts/` (slurm, setup, eval subdirectories)
4. **Experiment outputs** → `experiments/{phase}/` (organized by P0-P4, Phase A-C)
5. **Documentation** → `docs/` (master_plan.md is primary, archive/ for old docs)
6. **External repos** → Use git submodules (Dream, learning-to-cache)

**Naming Conventions:**

- SLURM scripts: `eval_{phase}_{task}.sh` (e.g., `eval_P0_baseline.sh`)
- Test scripts: `test_{category}_{name}.py` (e.g., `test_gpu_minimal.py`)
- Results: `{phase}_{task}_{date}_{config}_results.json`
- Traces: `{phase}_{task}_{date}_{config}.pt`

**Git Tracking:**

- ✅ Track: Source code, tests, scripts, docs, result JSONs, configs
- ❌ Ignore: Logs (*.out), traces (*.pt), pycache, large binaries
- ⚙️ Submodules: Dream, learning-to-cache

---

## 1. Big Picture & Phases

We treat **“Delta-Compute for Dream-7B”** and **“L2C-style layer caching”** as **one unified project** with **two tightly coupled tracks**. The goal of this project is to implement and evaluate **all phases** in both tracks end-to-end.

- **Delta-Compute Gating Track (P0–P4)**  
  Implementation-oriented view of gating on top of Dream-7B:
  - **P0 – Baseline Teacher.** Dream-7B inference and evaluation with no acceleration.
  - **P1 – Teacher Traces.** Log per-step and per-layer statistics and costs.
  - **P2 – Rule-Based Gate.** Token and layer freezing using simple rules and a lightweight watchdog.
  - **P3 – Learned Gate.** Lightweight MLP or logistic gate trained on P1 traces.
  - **P4 – Adaptive Step Scheduling.** Sequence-level stride control based on local truncation error (LTE) and risk.

- **L2C Layer-Caching Track (Phase A–C)**  
  L2C-style layer reuse inside the Dream-7B denoiser:
  - **Phase A – Heuristic Layer Caching.** Hand-designed schedules such as alternating full and cache steps.
  - **Phase B – L2C-Style Learned Router (fixed schedule).** Per-layer recompute vs reuse decisions learned on top of a fixed schedule.
  - **Phase C – Continuous-Time Router βₗ(t) (multi-schedule).** Per-layer continuous-time router that can transfer across different diffusion schedules.

Implementation will still be **incremental** for engineering sanity, but **all of P0–P4 and Phase A–C are in scope for this project**. A convenient way to think about the rollout is:

- **Foundation:** bring up **P0 (Teacher)** and **P1 (Teacher traces)** on GSM8K so that all later phases have a stable baseline and diagnostics.
- **First acceleration layer:** implement **P2 (rule-based gate)** and **Phase A (heuristic caching)** as the initial sources of real compute reduction.
- **Learned acceleration layer:** add **P3/P4 (learned gate and adaptive steps)** and **Phase B/C (learned routers)** on top of the same code paths and evaluation stack.

Everything below is written so that **each phase can be implemented and evaluated independently**, while still composing into a single end-to-end system where gating and caching coexist on Dream-7B.

---

## 2. Dream Model Changes & Open-Source References

Delta-compute gating (P1–P4) and L2C-style layer caching cannot be implemented purely by wrapping the existing Dream eval entrypoints. We must **modify Dream’s model implementation itself** to expose block-level hooks and caching.

**Sources of model code:**

- **Dream model implementation (from Hugging Face):**
  - Model hub: `Dream-org/Dream-v0-Instruct-7B`
  - Key files (vendored into this repo under `external/Dream/modeling/`):
    - `configuration_dream.py` — defines `DreamConfig`.
    - `modeling_dream.py` — defines `DreamModel` and transformer blocks.
    - `generation_utils.py` — defines `DreamGenerationMixin`, `DreamGenerationConfig`, and the **diffusion sampling loop** (`diffusion_generate` and `_sample`).
  - In this project, these files are treated as the canonical place to:
    - add **P1 tracing hooks**,
    - add **P2 token/layer gating**,
    - add **L2C-style layer caching (Phase A–C)**.

- **L2C reference implementation (image diffusion):**
  - Repo: `external/learning-to-cache/` (cloned from `https://github.com/horseee/learning-to-cache`).
  - Subdirectories:
    - `DiT/` — L2C for DiT-XL/2.
    - `U-ViT/` — L2C for U-ViT-H/2.
  - We use this code as a **reference for engineering patterns**:
    - how to split transformer blocks into attention / FFN,
    - how to maintain per-layer caches across diffusion steps,
    - how to parameterize and train routers.

**Eval wrappers (unchanged entrypoints, new knobs):**

- Base Dream eval wrapper: `external/Dream/eval/eval.py` (`Dream` LM).
- Instruct eval wrapper: `external/Dream/eval_instruct/lm_eval/models/diffllm.py` (`DiffLLM` LM).
- Both wrappers call `self.model.diffusion_generate(...)` and will be extended to:
  - parse `delta_mode`, `cache_mode`, and a small number of scalar thresholds from `--model_args`,
  - pass these settings into `DreamModel.diffusion_generate` (without creating separate CLI entrypoints).

From a code perspective, **initial implementation stages assumes that editing `external/Dream/modeling/modeling_dream.py` is allowed and expected**. There is no way to realise true token/layer compute skipping or L2C-style caching by wrappers alone.

---

## 3. Design Constraints & Feasibility (Conceptual)

We explicitly acknowledge a few core constraints of Dream-7B’s architecture and how they shape the design of delta-compute and L2C:

- **Bidirectional attention (not causal).**
  - Dream uses **bidirectional attention** (see `DreamAttention.is_causal = False`), so each token’s hidden state depends on the entire current sequence at each step.
  - Any **token-level hidden-state freezing** or caching is therefore an **approximation**: if other tokens change, the “frozen” token’s true hidden state should ideally change as well.
  - **How we respond:**
    - We treat gating/caching as designing a **new approximate inference operator** whose quality is judged on GSM8K accuracy, not on perfectly matching internal hidden states.
    - **Stage 2 implementation does not rely on aggressive token-level hidden-state freezing.** Instead, it starts from coarser controls (sequence-level / layer-level, see P2-v1b and Phase A).
    - Finer-grained token-level gating will be implemented in P2-v2 and P3 stages once we have P1 traces and empirical evidence that such approximations can be controlled.

- **No explicit time embedding in DreamModel.**
  - In `generation_utils._sample`, diffusion “time” appears only via the **mask schedule** (`timesteps`, `p_transfer`, number of tokens updated per step).
  - `DreamModel` itself does **not** receive an explicit time embedding or scalar `t` in its forward pass; each step is simply a forward on the current token sequence `x` and attention mask.
  - **How we respond:**
    - L2C-style FFN caching does **not** suffer from a time-embedding mismatch; errors primarily come from **context drift** (the set of masked/unmasked tokens changes between steps).
    - P1 traces and Phase A experiments are explicitly tasked with **measuring this drift** (e.g., layer-wise deltas across steps) and quantifying its impact on GSM8K accuracy.

- **Context drift across diffusion steps.**
  - Between steps, `x` changes as more positions are filled; reusing FFN outputs or hidden states across steps is inherently inexact.
  - This is the same regime L2C operates in for DiT / U-ViT: caching is done **despite** context changes, and routers / heuristics decide where this is safe.
  - **How we respond:**
    - Phase A starts with **conservative caching** (e.g., only between adjacent steps, only for a subset of layers, only in later steps) to limit drift exposure.
    - P1 traces and targeted ablations (e.g., force caching on/off for specific layers/steps) are used to characterise which layers / time ranges are most robust to reuse.

These constraints are treated as **design drivers**, not blockers: they define what initial implementation stages must measure and de-risk, while P2-v2, P3, P4, Phase B, and Phase C will implement more aggressive token-level gating and learned routers are explored.

---

## 3. Integration with Dream Eval Code (real repo)

We **do not** create a new evaluation entrypoint. Instead, we plug into the **existing Dream eval code**:

- **Base Dream model (LM=“dream”):**
  - File: `external/Dream/eval/eval.py` (`@register_model("dream")`).
  - Shell scripts: `external/Dream/eval/eval_dream_gen.sh`, `eval_dream_mc.sh`.
  - Usage pattern (example):
    ```bash
    # Inside DLLM-Delta-Compute/, in env dcllm
    cd external/Dream/eval
    accelerate launch eval.py --model dream \
      --model_args pretrained=Dream-org/Dream-v0-Base-7B,add_bos_token=true,diffusion_steps=512 \
      --tasks gsm8k_cot \
      --batch_size 1 \
      --output_path evals_results/gsm8k-ns0 \
      --log_samples --confirm_run_unsafe_code
    ```
- **Instruct model with lm-eval-harness (LM=“diffllm”):**
  - Directory: `external/Dream/eval_instruct/`
  - Entry: `eval.sh`
  - Uses: `accelerate launch -m lm_eval --model diffllm --model_args ...`
  - Example (GSM8K CoT, from `eval_instruct/eval.sh`):
    ```bash
    PYTHONPATH=. accelerate launch -m lm_eval \
      --model diffllm \
      --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True, \
                   max_new_tokens=256,diffusion_steps=256,dtype="bfloat16", \
                   temperature=0.1,top_p=0.9,alg="entropy" \
      --tasks gsm8k_cot \
      --device cuda \
      --batch_size 1 \
      --num_fewshot 0 \
      --output_path output_reproduce/gsm8k \
      --log_samples --confirm_run_unsafe_code \
      --apply_chat_template
    ```

**Design rule:** all delta-compute / caching modes must be selectable **through `model_args`**, so we can reuse these commands and scripts.

### 3.1 New knobs in `model_args`

We add **string keys** parsed by the Dream LM wrappers (`Dream` in `eval.py`, and the `diffllm` model used by lm-eval). For initial implementation stages, we keep configuration **as simple scalar arguments only** and avoid external YAML files:

- `delta_mode = "none" | "rule_gate" | "learned_gate" | "adaptive"`
- `cache_mode = "none" | "l2c_heuristic" | "l2c_l2c_fixed" | "l2c_continuous"`
- **initial implementation stages:** gate / cache thresholds are provided directly as scalar `model_args`, e.g.:
  - `delta_entropy_tau=...` (for entropy-based gate),
  - `cache_schedule="alt_full_cache"` (for a simple Phase A schedule).
- **Later stages (v2+):** if we need to sweep many configurations, we can add optional:
  - `delta_config=/path/to/delta_config.yaml`
  - `cache_config=/path/to/cache_config.yaml`
  but these are **explicitly not required** to complete initial implementation stages.

Example (no acceleration, **Teacher**):

```bash
--model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True, \
             max_new_tokens=256,diffusion_steps=256,dtype="bfloat16", \
             temperature=0.1,top_p=0.9,alg="entropy", \
             delta_mode=none,cache_mode=none
```

Example (Phase A L2C heuristic POC on GSM8K):

```bash
--model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True, \
             max_new_tokens=256,diffusion_steps=256,dtype="bfloat16", \
             temperature=0.1,top_p=0.9,alg="entropy", \
             delta_mode=none,cache_mode=l2c_heuristic
```

We then:

- Extend `external/Dream/eval/eval.py::Dream.__init__` / `_generate_batch` to parse these keys from `model_args` and pass them down to the underlying HF `diffllm` model, and
- Add minimal hooks inside Dream’s diffusion sampler (in `external/Dream/src/diffllm/`) to:
  - execute with **no acceleration** when `cache_mode="none"` and `delta_mode="none"`, and
  - call into our delta-compute / caching logic otherwise.

No new top-level CLI is introduced for this POC.

---

## 4. Evaluation Tasks & Metrics (aligned with Dream)

For initial implementation stages we **focus on a single dataset and model** to keep the scope tight and the signal clear:

- **Primary model:** `Dream-org/Dream-v0-Instruct-7B`
- **Primary task (Tier A, required for initial implementation stages):**
  - `gsm8k_cot` (via `external/Dream/eval_instruct/eval.sh` / lm-eval-harness) — EM accuracy.
- **Optional extensions (Tier B, not required for initial implementation stages):**
  - A small **MMLU subject** via lm-eval (e.g. `mmlu_astronomy`),
  - Additional tasks already wired in Dream’s scripts (minerva_math, code, etc.).

**Minimal POC success metrics (per config, on GSM8K):**

- **Latency per sample (ms)**: mean / p50 / p90 over prompts (batch=1).
- **Approximate compute**: number of **block calls** actually executed (denoiser layer forward passes).
- **Task quality**:
  - GSM8K: EM accuracy.
- **Teacher vs accel comparison (when acceleration is enabled):**
  - target ≥ **1.3× latency reduction** on GSM8K,
  - with **≤ 1–2 pp** drop in EM on GSM8K.

More advanced metrics (Compute-AUC, δ_frozen, oracle baselines, etc.) remain **important hypotheses to test**, but they are only required once we move beyond initial implementation stages.

---

## 5. P0 – Baseline Teacher (Dream-7B)

**Goal.** Have a **clean, reproducible Teacher config** for Dream-7B (Base + Instruct) using existing eval scripts.

### 4.1 Steps

1. **Activate env (once per session):**
   ```bash
   source activate dcllm   # or: conda activate dcllm
   ```
2. **Run Instruct teacher on GSM8K CoT** using `external/Dream/eval_instruct/eval.sh` as reference (we can add a dedicated GSM8K-only script if helpful):
   - Freeze:
     - `pretrained=Dream-org/Dream-v0-Instruct-7B`
     - `diffusion_steps` (e.g., 256 for GSM8K CoT POC),
     - decoding hyperparameters (`temperature`, `top_p`, `max_new_tokens`),
     - `delta_mode=none, cache_mode=none`.
3. **(Optional) Run Base teacher** on one simple task via `eval_dream_gen.sh`.
4. **Record baseline:**
   - EM / accuracy for GSM8K / MMLU-astronomy,
   - mean / p50 / p90 latency,
   - `diffusion_steps`, `max_new_tokens`, and `batch_size`.

This **Teacher config** becomes the reference for all P1–P4 gating and L2C caching comparisons on GSM8K (and any optional extra tasks).

---

## 6. P1 – Teacher Traces (Delta-Compute Track)

**Goal.** Collect **lightweight but informative** diagnostics from Dream-7B across diffusion steps, to drive both **rule-based gating (P2)** and **L2C caching** design.

### 5.1 What we log (minimal version)

For a **small subset of prompts** (e.g., 50 GSM8K and 200 MMLU-astronomy questions):

- For each diffusion step `t` and each transformer block `l`:
  - **Layer call count** (sanity check; Teacher always calls every block).
  - **Simple per-layer statistics**, aggregated over tokens:
    - `mean_ffn_norm[l,t]` (L2 norm of FFN output),
    - `mean_cos_ffn_between_steps[l,t]` (cosine similarity vs previous step),
    - optionally the same for attention outputs.
  - **Per-step runtime**:
    - wall-clock time spent in the denoiser for step `t`,
    - optionally a rough per-layer timing if lightweight to collect.

These are sufficient to:

- Confirm that **later diffusion steps change more slowly** (higher cosine similarities),
- Identify **which layers are most expensive / most stable**,
- Decide whether **FFN-only caching** is a good default for Phase A.

### 5.2 Implementation sketch (no heavy code here)

- Add a `trace_teacher` flag passed through `model_args` (e.g., `trace_teacher=true`).
- In Dream’s denoiser implementation (inside `external/Dream/src/diffllm`):
  - Wrap each block with optional tracing hooks that:
    - compute norms / cosines on-the-fly,
    - update a small in-memory accumulator.
- After the run, write a compact JSON/CSV:
  - e.g., `runs/teacher_trace_gsm8k_256steps/trace.json`.

### 5.3 Visualizations (ideas to keep)

For the above traces, we want **just enough plots** to inform later phases:

- **Layer × step heatmaps** of:
  - mean FFN norm,
  - mean cosine between steps.
- **Per-layer runtime bar chart** (for a few representative steps).
- **Simple stability vs cost view**:
  - e.g., scatter of “mean cosine” vs “per-layer time” to spot high-ROI candidates for caching / gating.

No full reporting stack is required at POC v1; a few quick Matplotlib/Seaborn plots saved under `reports/` are enough.

---

## 7. P2 – Rule-Based Delta-Compute Gate (Token / Layer Freezing)

**Goal (concept).** Implement a **rule-based delta-compute gate** that freezes “stable” tokens / layers across diffusion steps to reduce compute, while keeping quality drop small.

To respect the **“one idea per stage”** principle and avoid over-complicating the first implementation, we split P2 into:

- **P2-v1b (minimal gate for Stage 2 implementation, GSM8K-only):**
  - uses a **single scalar signal** (e.g. logit entropy or margin) and **one threshold**,
  - targets only **one coarse type of gating** at a time (e.g. **sequence-level** or **layer-level**),
  - has **no watchdog / rollback** beyond basic sanity checks.
  - **Token-level hidden-state freezing is explicitly not required in P2-v1b** due to the bidirectional-attention concerns discussed in §3.
- **P2-v2 (full multi-signal gate, later):**
  - can combine cosine similarity, ΔL2, logit margin, KL spikes, etc.,
  - may include watchdog / rollback budgets and more complex policies.

The rest of this section is written with this split in mind.

### 7.1 High-level behavior (P2-v1b: minimal gate for Stage 2 implementation)

For the first working gate on GSM8K, we start with the simplest possible setting that can show “some compute can be skipped with limited quality loss”:

- At each diffusion step `t`:
  - Compute a **single scalar stability signal** derived from logits, at either:
    - **sequence-level** (one score per sample), or
    - **layer-level** (one score per layer band / block group).
    - e.g. **logit entropy** `H_t` or **margin** `m_t = p_top1 - p_top2`.
  - Mark sequences / layers as **“stable enough”** if the signal crosses a simple threshold:
    - e.g., `H_t ≤ τ_entropy` (low entropy = confident) **or**
    - `m_t ≥ τ_margin` (large margin = confident).
  - For “stable” sequences / layers:
    - **reuse previous step’s outputs** instead of recomputing matmuls for those layers / positions,
    - manage a small per-step cache inside `DreamGenerationMixin._sample` and the corresponding `DreamModel` blocks.

**Implementation strategy & fallbacks:**

- **First attempt (conservative):**
  - Start with **sequence-level early stopping** or **layer-band gating** (e.g., only for a few top layers and later steps on GSM8K).
  - Do **not** attempt fine-grained token-level hidden freezing in P2-v1b.
- **If conservative gating still causes unacceptable GSM8K regressions:**
  - Fall back to using P2 primarily as an **offline analysis tool**:
    - run oracle experiments where selected layers/steps are forcibly cached/recomputed,
    - use these to identify which layers/steps are too sensitive to be gated and prioritise P4 (adaptive step scheduling) and Phase A instead.

Key constraint (even in v1b):

- **Masks must save real compute**:
  - we modify Dream’s blocks (in `external/Dream/modeling/modeling_dream.py`) to support skipping FFN / attention matmuls for gated positions,
  - we do **not** “compute then overwrite” in P2.

### 7.2 Configuration (P2-v1b, kept minimal)

For Stage 2 implementation on GSM8K:

- Choose **one** gating signal and **one** threshold, e.g.:
  - `delta_mode="rule_gate_entropy"`
  - `delta_entropy_tau=0.5`
- Optionally, include a **very simple safety clamp**:
  - e.g. “never gate before step t_min” or “never gate more than X% of steps”.
- All thresholds are passed via `model_args` as scalars (no YAML config for v1b).

### 7.3 Outlook – P2-v2 (multi-signal gate), P3 (Learned Gate) & P4 (Adaptive Steps)

Once P2-v1b shows a clear speedup/quality tradeoff on GSM8K and P1 traces are in place, we can gradually move toward the full design:

- **P2-v2 – Multi-signal gate (token / layer):**
  - Combine features from P1: cosine similarity, ΔL2, logit margin, entropy, KL spikes, step index, layer band, etc.
  - Introduce a more expressive decision rule (e.g. conjunctions / disjunctions, simple piecewise rules).

- **P3 – Learned Gate (token / layer):**
  - Replace hand-tuned thresholds with a small **MLP / logistic gate** (≲50k params) that predicts “safe to freeze” using P1 features.
  - Train offline on Teacher traces:
    - positives: positions where freezing does not change task outcome and passes basic sanity checks,
    - negatives: positions where freezing causes errors or large logit changes.
  - Calibrate with ROC/PR curves and, if needed, simple conformal quantiles (δ_frozen control).

- **P4 – Adaptive Step Scheduling (sequence-level):**
  - Use **local truncation error (LTE)** estimates (e.g., Euler vs Heun) and risk signals (KL / entropy drift) to adjust **stride** over diffusion steps.
  - Controller logic (high-level):
    - when LTE and risk are small for several consecutive steps, tentatively increase stride (skip steps);
    - if LTE / risk spikes, drop stride back to 1 and reset.

The detailed math and risk-control analysis for P2-v2/P3/P4 will be maintained in separate technical notes (outside this file) once we start implementing those stages.

### 7.4 Metrics & visualizations to capture

For each run with `delta_mode="rule_gate"`:

- **Skip ratios:**
  - fraction of tokens / layers skipped per step and on average.
- **Latency vs Teacher:**
  - mean / p50 / p90, ideally ≥ 1.3× speedup on GSM8K / MMLU-astronomy.
- **Quality vs Teacher:**
  - EM / accuracy deltas.
- **Simple consistency view (optional):**
  - fraction of outputs identical to Teacher (exact match),
  - fraction of final tokens matching Teacher.

We keep logs minimal (e.g., a JSON summary per run), but ensure we can later extend to more advanced metrics (δ_frozen, Compute-AUC, etc.) if useful.

---

## 8. L2C-Style Layer Caching Track (Phase A–C)

This is the **L2C-specific part** of the project, originally described in the previous version of this file. We keep it, but **trim down** to the core ideas needed for a POC.

### 7.1 Core hypothesis

> For Dream-7B, **consecutive diffusion steps** are similar enough that we can **re-use some layers’ outputs across steps** and skip recomputing them, following an L2C-style idea, and still maintain good task quality.

This track focuses on **layer-level caching**, not token-level gating:

- We reuse **transformer block outputs** (FFN, attention, or full residual increments) across **neighboring diffusion steps**.
- We do **not** train or finetune the Dream-7B backbone itself.

### 8.2 Phase A – Heuristic Layer Caching (minimal POC)

**Goal.** Answer the question: “If we reuse some layer outputs between adjacent steps, do we get **measurable speedup** on Dream-7B with tolerable quality loss?”

Basic pattern:

- Fix `diffusion_steps` (e.g., 256 for GSM8K CoT).
- Define two types of steps:
  - **Full steps**: run all layers normally, populate a per-layer cache.
  - **Cache steps**: reuse cached outputs for selected layers instead of recomputing.
- Simple alternating schedule:
  - even `t`: full steps,
  - odd `t`: cache steps.

Default caching strategy (conservative):

- **Recompute attention**, cache **FFN outputs**:
  - At cache steps:
    - run LayerNorm + attention as usual,
    - **skip FFN matmul** and **reuse previous step’s FFN output**.
- This reduces a significant portion of per-layer compute while keeping the time-dependent attention path fresh.

Experiments:

- Start with **GSM8K CoT** and **MMLU-astronomy**:
  - Compare Teacher vs Phase A on:
    - latency (p50/p90),
    - block call counts (FFN calls reduced),
    - EM / accuracy.
- **Phase A is the only L2C piece required for Stage 2 implementation** (if we choose caching rather than gating in that stage).

**Implementation strategy & fallbacks:**

- **First attempt (conservative):**
  - Cache only **FFN outputs** for a **small subset of upper layers** and only between **adjacent steps**, preferably in the **later diffusion steps** where P1 traces show minimal drift.
  - Use a simple fixed schedule (e.g., alternate full/cache steps) and verify with GSM8K that EM drop is within the target range.
- **If even conservative Phase A causes large regressions:**
  - Treat Phase A as an **analysis stage**:
    - use P1 traces and oracle ablations (force reuse of FFN outputs for specific layers/steps) to rank layers/steps by sensitivity to caching,
    - use these findings to guide later architectures or routers, while prioritising P4 (adaptive step scheduling) for real-time acceleration on current Dream checkpoints.

### 8.3 Phase B – L2C-Style Learned Router (fixed schedule, v2+)

Once Phase A and rule-based gating (P2) look promising, we can consider a **learned router**:

- Parameterization:
  - For each cache step index `m` and layer `l`, maintain a parameter β[m,l] ∈ [0,1],
  - Interpret β as “probability to recompute vs reuse”.
- Training idea (high-level):
  - Freeze Dream-7B.
  - For a set of prompts:
    - run Teacher (no caching) and log denoiser outputs at cache steps,
    - simulate routes that mix cached vs recomputed layers,
    - train β to minimize a **distillation loss**:
      - student denoiser output ≈ teacher output,
      - plus a regularizer that encourages caching (fewer recomputes).
- Inference:
  - For each cache step, get β[m,:],
  - threshold to decide per-layer recompute / reuse,
  - apply with the same interfaces built for Phase A.

We deliberately keep this section conceptual; exact training data formats, batch sizes, and loss functions can be refined once we know Phase A is worthwhile.

### 8.4 Phase C – Continuous-Time Router βₗ(t) (multi-schedule, v2+)

If a fixed-schedule router works, we can generalize:

- Replace step index `m` with **continuous time t** or a simple time embedding.
- For each layer ℓ, learn a small function βₗ(t):
  - input: scalar t (or log-SNR),
  - output: recompute probability.
- Train βₗ(t) on one schedule (e.g., 256 steps) and test transfer to others (e.g., 128, 512).

Again, this is **not required for POC v1**; it is a natural continuation once we have solid evidence that layer caching helps.

---

## 9. Debugging & Risks (kept concise)

Key checks before trusting any acceleration results:

1. **Teacher parity.**
   - `delta_mode=none, cache_mode=none` must produce identical results to unmodified Dream evaluation.
2. **Cache correctness (Phase A).**
   - When caching is “disabled” via config (e.g., always recompute), outputs must equal Teacher.
   - When caching is enabled, shapes / dtypes must stay consistent across steps.
3. **Compute vs latency.**
   - Block call counters should match design (e.g., FFN called in only half the steps).
   - Latency should monotonically decrease as we reduce real compute (matmul calls).
4. **Quality sanity.**
   - For a small handful of prompts, inspect generated answers manually:
     - look for systematic failures (e.g., off-by-one errors, truncated answers).
5. **Scope control.**
   - Do not start Phase B/C or full P3/P4 until:
     - Phase A + P2 show clear speedup on Tier A tasks,
     - quality drop is within agreed bounds.

Known risks from the previous drafts (summarized):

- **Time conditioning mismatch**: cached FFN outputs ignore precise time embedding; acceptable for POC but not fundamentally correct.
- **Memory / storage blowup**: storing too many intermediate activations for router training (Phase B/C) can become infeasible; for now, we avoid large-scale logging.
- **No real speedup**: if masks do not skip matmuls, we might reduce "logical calls" but not wall-clock time; micro-benchmarks are needed to validate.

---

## 10. Current Implementation Status

**Last Updated:** November 14, 2025  
**Branch:** PoC-1  
**Overall Progress:** ~80% infrastructure complete, ready for full GPU evaluation

### ✅ Completed (Ready for Testing)

**Infrastructure (P0-P2, Phase A):**
- ✅ `src/tracing.py`: TraceCollector with LayerStepStats, save/load functionality
- ✅ `external/Dream/modeling/modeling_dream.py`: Modified with delta-compute hooks
  - DreamDecoderLayer: Added layer_idx, use_ffn_cache, trace_collector params
  - DreamBaseModel: Added layer_ffn_caches, cache_schedule, trace_collector params
  - DreamModel: Propagates all delta-compute params
- ✅ `external/Dream/modeling/generation_utils.py`: Extended generation config and _sample loop
  - DreamGenerationConfig: Added trace_teacher, delta_mode, cache_mode, thresholds
  - _sample(): P2 early stopping logic, Phase A FFN caching, trace saving
- ✅ `external/Dream/eval_instruct/lm_eval/models/diffllm.py`: model_args parsing
- ✅ Test scripts: unit tests, integration tests, visualization tools
- ✅ SLURM scripts: Phase-specific evaluation scripts (P0, P1, P2, Phase A)
- ✅ Documentation: master_plan, implementation_summary, testing guide

**GPU Validation:**
- ✅ H100 80GB validation passed (Job 3547009)
- ✅ CUDA 12.8, torch 2.9.0+cu128
- ✅ All imports successful, TraceCollector works with GPU tensors

### ⏳ In Progress

**P0 – Baseline Evaluation:**
- ⏳ Requires HuggingFace login (user completed)
- ⏳ Full GSM8K baseline run pending (sbatch scripts/slurm/eval_P0_baseline.sh)
- ⏳ Reference metrics collection (accuracy, latency, memory)

### 🔜 Ready to Start (Infrastructure Complete)

**P1 – Teacher Traces:**
- 🔜 Run trace collection on GSM8K (sbatch scripts/slurm/eval_P1_traces.sh)
- 🔜 Generate visualizations with tests/visualization/plot_traces.py
- 🔜 Analyze stable vs unstable layers/steps
- **Blockers:** None (infrastructure ready, needs P0 baseline first)

**P2 – Early Stopping:**
- 🔜 Evaluate on GSM8K with early stop (sbatch scripts/slurm/eval_P2_early_stop.sh)
- 🔜 Measure skip ratios, latency reduction, accuracy impact
- **Blockers:** None (infrastructure ready, needs P0 baseline for comparison)

**Phase A – FFN Caching:**
- 🔜 Evaluate on GSM8K with caching (sbatch scripts/slurm/eval_PhaseA_caching.sh)
- 🔜 Ablation studies on different cache schedules
- **Blockers:** None (infrastructure ready, needs P1 traces for optimal schedule)

### 📋 Not Yet Implemented

**P3 – Learned Gate:**
- ❌ Gate architecture design (lightweight MLP)
- ❌ Training data preparation from P1 traces
- ❌ Gate training loop
- ❌ Integration into generation_utils
- **Blockers:** Needs P1 traces, P2 baseline results

**P4 – Adaptive Steps:**
- ❌ LTE estimation implementation
- ❌ Risk-aware scheduling logic
- ❌ Dynamic stride control
- **Blockers:** Needs P2/P3 results, advanced feature

**Phase B – Learned Router:**
- ❌ Router architecture (time-conditioned MLP)
- ❌ Training data generation
- ❌ Distillation loss + efficiency regularizer
- ❌ Integration into caching logic
- **Blockers:** Needs Phase A results, P1 traces

**Phase C – Continuous Router:**
- ❌ Continuous-time parameterization
- ❌ Schedule transfer learning
- ❌ Multi-schedule training
- **Blockers:** Needs Phase B results

### Next Steps (Priority Order)

1. **P0 Baseline** (immediate):
   ```bash
   huggingface-cli login  # ✅ Done
   sbatch scripts/slurm/eval_P0_baseline.sh  # ⏳ Next
   ```

2. **P1 Traces** (after P0):
   ```bash
   sbatch scripts/slurm/eval_P1_traces.sh
   python tests/visualization/plot_traces.py experiments/P1_traces/traces/*.pt
   ```

3. **P2 + Phase A** (parallel, after P1):
   ```bash
   sbatch scripts/slurm/eval_P2_early_stop.sh
   sbatch scripts/slurm/eval_PhaseA_caching.sh
   ```

4. **P3, P4, Phase B, Phase C** (implement based on results from 1-3)

---

## 11. Deliverables & Completion Criteria

**This project implements ALL phases P0-P4 and Phase A-C.** The deliverables below are structured incrementally for engineering clarity, but **all stages listed are required for project completion. There is no "future work" or "optional extensions" — everything here is in scope.**

### Stage 1: Infrastructure & Baselines (P0, P1) — REQUIRED

**P0 – Baseline Teacher:**
1. **Environment & Teacher:**
   - `dcllm` conda environment documented and used for all runs.
   - Stable Teacher baselines on GSM8K CoT (Instruct, `diffllm` model).
   - Reference metrics recorded: EM accuracy, mean/p50/p90 latency, memory usage.
   - Config frozen: `diffusion_steps`, `max_new_tokens`, decoding hyperparameters.

**P1 – Teacher Traces:**
2. **Tracing Infrastructure:**
   - Per-step and per-layer statistics collection implemented (`src/tracing.py`).
   - Hooks integrated into `external/Dream/modeling/modeling_dream.py`.
   - Trace features: FFN norms, attention norms, cosine similarities, forward timing.
3. **Trace Analysis & Visualization:**
   - Heatmaps saved under `reports/`: layer × step FFN norms, cosine similarities.
   - Stability analysis: identify which layers/steps have minimal drift between steps.
   - Cost profiling: per-layer runtime, total denoiser time per step.
   - **Deliverable:** `reports/P1_trace_analysis.md` documenting stable vs unstable layers/steps.
4. **Teacher Parity Verification:**
   - With `delta_mode=none, cache_mode=none`: outputs identical to unmodified Dream eval.

### Stage 2: Basic Acceleration (P2, Phase A) — REQUIRED

**P2 – Rule-Based Gate:**
5. **Sequence-Level Early Stopping (P2-v1):**
   - `delta_mode="p2_early_stop"` implemented with confidence + entropy thresholds.
   - Early stop logic verified to actually skip remaining diffusion steps.
   - Measured on GSM8K: skip ratios, step reductions, latency savings, EM accuracy vs Teacher.
   - **Target:** ≥1.3× latency reduction, ≤2pp accuracy drop on GSM8K.
   
6. **Advanced Rule-Based Gating (P2-v2):**
   - Multi-signal gating: entropy + FFN norms + cosine similarity.
   - Layer-specific thresholds informed by P1 trace analysis.
   - Per-layer gating masks that skip FFN/attention matmuls (not "compute then ignore").
   - Token-level gating explored if P1 traces show feasibility despite bidirectional attention.
   - **Deliverable:** JSON summaries of gating decisions, latency vs quality tradeoffs.

**L2C Phase A – Heuristic Caching:**
7. **FFN Caching Implementation:**
   - `cache_mode="l2c_ffn"` with configurable layer schedules (e.g., `cache_schedule="0,1,2,3"`).
   - Heuristic schedules: alternating full/cache steps, later-step-only caching.
   - Integration into `generation_utils._sample()` with proper cache management.
8. **Performance Validation:**
   - Measured on GSM8K: latency reduction, FFN call counts, EM accuracy vs Teacher.
   - Ablation study: which layers/steps benefit most from caching (guided by P1 traces).
   - **Target:** Measurable speedup (≥1.2×), ≤3pp accuracy drop on GSM8K.
   - **Deliverable:** `reports/PhaseA_caching_analysis.md` with recommendations for Phase B.

### Stage 3: Learned Components (P3, Phase B) — REQUIRED

**P3 – Learned Gate:**
9. **Gate Architecture & Training:**
   - Lightweight MLP/logistic gate (<50k params) trained on P1 trace features.
   - Input features: layer_idx, step_idx, FFN norms, cosine similarities, entropy.
   - Output: per-layer or per-token "safe to freeze" probability.
   - Training data: P1 traces from GSM8K, labels from oracle experiments.
10. **Learned Gate Evaluation:**
    - `delta_mode="p3_learned_gate"` implemented and tested on GSM8K.
    - Validation that learned gate meets or exceeds P2 performance.
    - **Target:** ≥P2 accuracy, improved throughput, better generalization across problem types.
    - **Deliverable:** `reports/P3_learned_gate_results.md` with ROC/PR curves, calibration analysis.

**L2C Phase B – Learned Router:**
11. **Router Architecture:**
    - Time-conditioned MLP for per-layer recompute/reuse decisions.
    - Architecture follows L2C reference (DiT/U-ViT patterns in `external/learning-to-cache/`).
    - Parameters: β[step, layer] or continuous βₗ(t) per layer.
12. **Router Training:**
    - Training on GSM8K: distillation loss (student ≈ teacher outputs) + efficiency regularizer.
    - Freeze Dream-7B backbone, train only router parameters.
13. **Router Evaluation:**
    - `cache_mode="l2c_learned"` tested on GSM8K.
    - Validation that learned router outperforms Phase A heuristics.
    - **Target:** Better than Phase A, ≤2pp accuracy drop vs Teacher, measurable speedup.
    - **Deliverable:** `reports/PhaseB_router_results.md` with learned schedules visualization.

### Stage 4: Advanced Features (P4, Phase C) — REQUIRED

**P4 – Adaptive Step Scheduling:**
14. **Dynamic Step Control:**
    - Sequence-level stride adjustments based on local truncation error (LTE) estimates.
    - Risk-aware scheduling: use confidence/entropy to decide when to skip/merge steps.
    - Implements Euler vs Heun LTE estimation or similar.
15. **Adaptive Scheduling Evaluation:**
    - `delta_mode="p4_adaptive"` tested on GSM8K.
    - Measured: average steps used per problem, latency reduction vs fixed-step baselines.
    - **Target:** Further latency cut beyond P2/P3, ≤5pp accuracy drop vs Teacher.
    - Analysis: which problem types benefit most (easy vs hard GSM8K problems).
    - **Deliverable:** `reports/P4_adaptive_steps_analysis.md`.

**L2C Phase C – Continuous-Time Router:**
16. **Continuous Router Implementation:**
    - Per-layer router βₗ(t) as function of continuous time (or log-SNR).
    - Replaces fixed step-indexed parameters from Phase B.
17. **Transfer Learning Validation:**
    - Train on one diffusion schedule (e.g., 256 steps).
    - Test transfer to other schedules (128, 512 steps).
    - **Target:** Match or beat Phase B performance, demonstrate schedule transferability.
18. **Final Router Evaluation:**
    - `cache_mode="l2c_continuous"` tested on GSM8K.
    - **Deliverable:** `reports/PhaseC_continuous_router_results.md` with transfer experiments.

### Final Integration & Documentation — REQUIRED

19. **Comprehensive Evaluation:**
    - All modes (P0, P2-v1, P2-v2, P3, P4, Phase A, Phase B, Phase C) evaluated on GSM8K.
    - Extended evaluation on additional benchmarks (MMLU-astronomy minimum, optionally HumanEval/MATH).
    - Comparison tables: accuracy, latency, memory, estimated FLOPs, skip/cache ratios.
20. **Documentation & Analysis:**
    - `docs/implementation_summary.md`: Complete architecture with diagrams.
    - `reports/final_performance_comparison.md`: Tables, charts, tradeoff analysis.
    - `docs/production_recommendations.md`: Best configurations for different use cases.
21. **Code Quality & Reproducibility:**
    - Clean git history, all changes tracked in PoC-1 branch.
    - Test scripts for all modes (`test_poc_*.py`).
    - Visualization tools (`plot_traces.py`, `plot_performance.py`).
    - Reproducible SLURM job scripts for all evaluation runs.
    - Updated `requirements.txt` with all dependencies.

### Success Criteria

**Project is complete when:**
- ✅ All P0-P4 phases implemented, tested, and documented.
- ✅ All Phase A-C phases implemented, tested, and documented.
- ✅ GSM8K evaluation complete for all modes with documented performance.
- ✅ At least one acceleration mode achieves ≥1.3× latency reduction with ≤3pp accuracy drop.
- ✅ Comprehensive documentation and recommendations delivered.

**No phases are deferred to "future work" or "v2+". Everything above is in scope for this project.**
