# DLLM-Delta-Compute – Experiment Log

## Purpose

This file is the **central registry of all training, validation, and evaluation runs** for the DLLM-Delta-Compute project. It is intended to:

- Record **every non-trivial experiment** (training + eval) with enough metadata to reproduce it.
- Track **SLURM job IDs**, GPU types, scripts used, configs (delta_mode / ffn_mode / schedule), and key metrics.
- Link each run to **git commit hashes** and **Weights & Biases (W&B)** runs when applicable.
- Provide a single place where we can quickly answer:
  - “What was the last good P3 gate checkpoint?”
  - “Which Phase C config gave us 1.3× speedup on GSM8K with <2pp EM drop?”
  - “Which jobs failed and why?”

This log complements per-phase experiment READMEs and W&B dashboards; it is the **authoritative index of experiments run on the cluster**.

## Usage Guidelines

- Append new entries **at the top** (reverse chronological order).
- One **logical experiment** (e.g., a training job + its follow-up evals) can be captured in **one block** with multiple job rows.
- Use **English** for all entries.
- Do not dump full logs or metrics tables; instead:
  - Summarize key metrics (e.g., GSM8K EM, FFN_FLOPs, speedup, overhead).
  - Provide explicit **paths** to logs (e.g., SLURM `.out` files in `logs/slurm/` or similar) and results (e.g., JSON/CSV in `experiments/<phase>/`).

## Naming & Artifact Conventions

- **W&B project:** use a shared project name such as `dllm_delta_compute` (set via `WANDB_PROJECT`), and configure your entity (e.g., `WANDB_ENTITY`) locally.
- **W&B run names:** follow a consistent pattern:
  - `phaseC_s256_gsm8k_seed1`
  - `P4_phaseD_gsm8k_s256_ffn0.7_seed3`
- **SLURM job names:** match the script and config:
  - `dcllm_P3_train_gate_s256`
  - `dcllm_P4_phaseC_eval_gsm8k_s256`
- **Experiment directories:** under `experiments/<phase>/`, store:
  - `checkpoints/` – model weights (`*_best.pt`, `*_final.pt`).
  - `configs/` or `metadata.json` – training/eval config, including schedule, delta_mode, ffn_mode, seed.
  - `logs/` – lightweight logs, summaries, and any per-run metrics files.

Document any deviations from these conventions in this file when they occur.

## Recommended Entry Template

Use the following template for each experiment block (training + evals):

```markdown
### 2025-12-02 23:05 UTC — Phase C router training (256-step GSM8K)
- Phase: Phase C (Continuous Router)
- Goal:
  - Train a single-schedule Phase C router on GSM8K P1 traces (256 steps) and evaluate its ability to reduce FFN_FLOPs with minimal EM drop on GSM8K.
- Codebase:
  - Git commit: `abcdef1` (training script), `2345abc` (router implementation)
- Data:
  - Traces: `experiments/P1_traces/gsm8k_s256/` (train split)
  - Eval tasks: `gsm8k_cot` (test), `mmlu_astronomy` (test)

- Training job:
  - SLURM job id: `1234567`
  - Node / GPU: `H100` (1x), 384GB RAM, 16h limit
  - Script: `scripts/slurm/train_phaseC_gsm8k_s256_h100.sh`
  - Conda env: `dcllm`
  - delta_mode: `none`
  - ffn_mode: `l2c_continuous`
  - schedule: `256` steps
  - W&B:
    - Project: `dllm_delta_compute`
    - Run: `phaseC_s256_gsm8k_seed1`
  - Outputs:
    - Checkpoints: `experiments/PhaseC_continuous/checkpoints/continuous_router_best.pt`
    - Logs: `logs/slurm/train_phaseC_gsm8k_s256_h100_1234567.out`

- Evaluation job(s):
  - SLURM job id: `1234570`
  - Script: `scripts/slurm/eval_PhaseC_continuous_gsm8k_s256_h100.sh`
  - Tasks: `gsm8k_cot`
  - Metrics:
    - GSM8K EM: `0.59`
    - FFN_FLOPs ratio vs P0: `0.72`
    - Router+adapter overhead ratio: `0.08`
  - Logs:
    - `logs/slurm/eval_PhaseC_gsm8k_s256_h100_1234570.out`
    - `experiments/PhaseC_continuous/logs/gsm8k_s256_eval.json`

- Status / Notes:
  - Router converges; EM drop ~1.5pp vs P0 at 28% FFN reduction.
  - This run is a candidate for RQ3 matched-compute comparisons at ~0.7 FFN_FLOPs.
  - TODO:
    - Repeat with seeds `[2, 3]` for confidence intervals.
    - Evaluate on `mmlu_astronomy` using same checkpoint.
```

You can simplify the template for small smoke tests (e.g., 1-step sanity runs) but try to keep:

- **Phase**, **Goal**, **Git commits**, **SLURM job id(s)**, **scripts**, **delta_mode/ffn_mode/schedule**, **key metrics**, and **status**.

## Current Entries

*(Add new experiment blocks directly below this line, newest first.)*

### 2025-12-08 16:56 UTC — Full PoC Evaluation with Base Model (200 Samples)
- Phase: P0-P4, PhaseA-E
- Goal: Complete PoC evaluation with Dream-v0-Base-7B using official CoT config.
- SLURM job id: `3960957` (**COMPLETED** 5hr22min)
- Script: `scripts/slurm/poc_all_phases.sh`
- Model: `Dream-org/Dream-v0-Base-7B`
- Config: max_new_tokens=256, diffusion_steps=256, temp=0.0, 8-shot, batch_size=1
- **RESULTS (200 samples)**:

| Phase | Accuracy | Mean Latency | Tokens/s | Skip Ratio | Notes |
|-------|----------|--------------|----------|------------|-------|
| P0 Baseline | **76%** | 9.3s | 27.7 | - | Reference |
| P2 conf=0.5 | 76% | 9.2s | 28.0 | 0% | No early stop |
| P2 conf=0.4 | 76% | 9.1s | 28.2 | 0% | No early stop |
| P3 Gate | 76% | 10.5s | 24.4 | 0% | **SLOWER** |
| P4 Adaptive | 76% | 10.2s | 25.0 | 0% | **SLOWER** |
| PhaseA Cache | 76% | 9.3s | 27.6 | 0% | No speedup |
| PhaseB Router | 76% | 9.3s | 27.6 | 0% | No speedup |
| PhaseC Router | 76% | 9.3s | 27.6 | 0% | No speedup |
| PhaseD Skip | 76% | 9.2s | 27.7 | 0% | No speedup |
| PhaseE Flexi | 76% | 9.2s | 27.7 | 0% | No speedup |

- **CRITICAL**: All phases show **0% skip ratio** - routers not effective
- **ROOT CAUSE**: SkipRouter.load failed on ContinuousRouter checkpoint (fixed in commit 8fcc5f5)
- Status: **COMPLETED**
- Logs: `logs/poc_all_phases_3960957.out`
- Timing: `reports/timing/poc_all_phases_200_20251208_063832/`


### 2025-12-08 12:00 UTC — Training Jobs Summary
- Phase: P3, PhaseB, PhaseC, PhaseD, PhaseE
- Goal: Train gates/routers using CoT P1 traces from Base model.

| Job ID | Phase | Status | Duration | Notes |
|--------|-------|--------|----------|-------|
| 3960985 | P3 Gate | **COMPLETED** | 27min | Oracle labels generated |
| 3960986 | PhaseB Router | **COMPLETED** | 1hr15min | Using traces |
| 3960987 | PhaseC Router | **TIMEOUT** | 4hr | Need longer time |
| 3960988 | PhaseD Skip | **TIMEOUT** | 4hr | Need longer time |
| 3960989 | PhaseE Flexi | COMPLETED (bad) | 19s | Synthetic data only |
| 3960994 | PhaseE Flexi | **COMPLETED** | 2hr2min | Fixed, uses traces |

- Scripts: `scripts/slurm/train_p3_cot.sh`, `train_phaseB_cot.sh`, etc.
- Traces: `experiments/P1_traces_base_20251208_030520/` (300 traces)
- Checkpoints saved:
  - `experiments/P3_gate_cot_*/checkpoints/learned_gate_final.pt`
  - `experiments/PhaseB_router_cot_*/checkpoints/router_final.pt`
  - `experiments/PhaseE_flexi_cot_20251208_071920/checkpoints/flexi_depth_final.pt`
- **TODO**: Resubmit PhaseC/D with 8hr time limit

### 2025-12-08 08:00 UTC — P1 Trace Collection (Base Model)
- Phase: P1
- Goal: Collect traces from Base model for training gates/routers.
- SLURM job id: `3960754` (COMPLETED)
- Script: `scripts/slurm/p1_base_traces.sh`
- Samples: 300
- Accuracy: 77% (flexible-extract)
- Duration: 55min
- Output: `experiments/P1_traces_base_20251208_030520/`
- Logs: `logs/p1_base_traces_3960754.out`

### 2025-12-08 00:18 UTC — Phase D/E Minimal Validation Test
- Phase: Phase D, Phase E
- Goal: Verify Phase D/E module imports, save/load, and forward pass work correctly.
- SLURM job id: `3960463`
- Script: `scripts/slurm/test_phaseDE.sh`
- Status: **PASSED**
- Results:
  - SkipRouter: 35,201 params, 16/32 layers recompute at p=0.5
  - FlexiDepthManager: 17M params (8 adapted layers × 2M adapter + 305K router)
  - All save/load and forward pass tests passed
- Logs: `logs/test_phaseDE_3960463.out`

### 2025-12-03 00:15 UTC — CoT Baseline Evaluation (P0/P1/P2)
- Phase: P2 Early Stop
- Goal: Evaluate P2 early stopping with CoT settings (256 steps, 8-shot, 1000 samples).
- SLURM job id: `3795597`
- Script: `scripts/slurm/fair_poc_cot_eval.sh`
- delta_mode: `p2_early_stop`
- cache_mode: `none`
- schedule: 256 steps
- Metrics:
  - GSM8K CoT flexible-extract: **58.0%** (±1.56%)
- Logs: `logs/fair_poc_cot_3795597.out`
- Notes: P0 and P1 may have been skipped or failed; only P2 results visible in logs.

### 2025-12-02 23:12 UTC — Sanity Check (PhaseC Router)
- Phase: Phase C Continuous Router
- Goal: Verify PhaseC router is working with 50-sample sanity check.
- SLURM job id: `3795596`
- Script: `scripts/slurm/sanity_check_routers.sh`
- cache_mode: `l2c_continuous`
- schedule: 128 steps
- Metrics:
  - GSM8K flexible-extract: **66.0%** (±6.77%)
- Logs: `logs/sanity_check_3795596.out`
- Notes: Router loaded successfully (14,529 params). Only PhaseC ran; P3/PhaseB may have been skipped.

