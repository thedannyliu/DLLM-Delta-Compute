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

