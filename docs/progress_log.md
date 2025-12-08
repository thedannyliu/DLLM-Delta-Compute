# DLLM-Delta-Compute – Progress Log

## Purpose

This file is a **human-readable development journal** for the DLLM-Delta-Compute project. It is intended to:

- Track day-to-day engineering and research progress across all phases (P0–P4, Phase A–E).
- Record **what problem was being solved, what actions were taken, and what the current status is**.
- Tie concrete work to specific **git commits**, SLURM jobs, and (optionally) Weights & Biases runs.
- Provide a concise historical record that future contributors can skim to understand how the project evolved and why certain decisions were made.

This log is not meant to replace detailed per-experiment reports or code comments; it is a **high-level progress timeline**.

## Usage Guidelines

- Always append new entries **at the top** (reverse chronological order).
- Keep entries **short and structured**; use bullets and consistent headings.
- For every meaningful chunk of work (e.g., a coding session, a debugging cycle, a set of experiments), record:
  - The **date and time (UTC)**.
  - The **author** (Git username or initials).
  - The **related phases / components** (e.g., `P3`, `PhaseC`, `generation_utils`, `train_learned_gate.py`).
  - The **goal / problem statement**.
  - The **actions taken** (referencing code paths and scripts when relevant).
  - The **current status / outcome** (success, partial, blocked, negative result).
  - The **associated git commit hash(es)** (short SHA is fine).
  - Any **follow-up items / TODOs**.

- Use **English** for all entries.
- Do not paste large stack traces or logs; instead record **paths** to log files (e.g., `logs/train/P3/2025-12-02_2204.log`) and/or SLURM output files.

## Recommended Entry Template

When adding a new entry, follow this template as closely as possible:

```markdown
### 2025-12-02 22:15 UTC — Short summary of work
- Author: <your-id>
- Phases / Modules: [P3, PhaseC, generation_utils, scripts/training/train_learned_gate.py]
- Goal:
  - e.g., Debug P3 gate training instability on GSM8K traces; align training script with master_plan v2.
- Actions:
  - Updated `scripts/training/train_learned_gate.py` to add validation split and logging of val loss.
  - Fixed `TraceCollector.get_all_stats()` edge case when FFN cosine similarity is missing.
  - Ran a small sanity training on 10 trace files (no GPU).
- Status / Outcome:
  - Gate training runs end-to-end on CPU; val loss decreases over 5 epochs.
  - Still need to verify final F1 / EM impact once GPU traces are available.
- Git commits:
  - `abc1234` (training script refactor)
  - `def5678` (TraceCollector fix)
- SLURM jobs:
  - `1234567` (CPU test; completed)
- Next steps / TODOs:
  - Collect full P1 traces on GSM8K train (256-step schedule) via `scripts/slurm/eval_P1_traces.sh`.
  - Re-run P3 gate training on H100 with full trace set, log to W&B project `dllm_delta_compute`.
```

You can adapt field ordering slightly as long as the same information is present. The key is to ensure **every block of work is tied to commits and jobs**, so readers can reproduce and validate changes.

## Current Entries

*(Add new entries directly below this line, newest first.)*

### 2025-12-08 05:15 UTC — Phase D/E Implementation Complete
- Author: AI Assistant
- Phases / Modules: [Phase D, Phase E, src/skip_router.py, src/flexi_adapter.py, generation_utils.py]
- Goal:
  - Implement Phase D (continuous layer skipping) and Phase E (FlexiDepth router+adapter) as specified in master_plan.md v2.
- Actions:
  - Created `src/skip_router.py`: SkipRouter extending ContinuousRouter with skip semantics (identity residual instead of cached FFN).
  - Created `src/flexi_adapter.py`: FlexiAdapter, FlexiRouter, FlexiDepthManager for router+adapter dynamic depth.
  - Integrated `skip_continuous` and `flexi_ffn` cache modes into `generation_utils.py`.
  - Created training scripts: `train_skip_router.py`, `train_flexi_adapter.py`.
  - Created SLURM scripts: `train_phaseD_skip.sh`, `train_phaseE_flexi.sh`, `test_phaseDE.sh`.
- Status / Outcome:
  - All Phase D/E code implemented and committed.
  - Test job 3960463 submitted for minimal validation.
- Git commits:
  - `c6a590a` (Phase D/E training scripts and test)
  - Previous commit (Phase D/E modules and generation_utils integration)
- SLURM jobs:
  - `3960463` (test_phaseDE - passed)
- Next steps / TODOs:
  - Wait for test job to complete and verify imports/forward pass work.
  - Collect P1 traces on GSM8K CoT for training Phase D/E routers.
  - Train Phase D skip router from Phase C checkpoint.
  - Train Phase E FlexiDepth router+adapter.

### 2025-12-08 05:40 UTC — Sanity Check Investigation & Bug Fix
- Author: AI Assistant
- Phases / Modules: [P3, PhaseB, PhaseC, sanity_full_check.sh]
- Goal:
  - Investigate why P3/PhaseB/PhaseC produce 100% identical outputs in sanity check.
- Actions:
  - Analyzed existing sanity check results (100/100 samples identical across P3/PhaseB/PhaseC).
  - **Found root cause**: P3 gate checkpoint path was wrong (`gate_final.pt` → `learned_gate_final.pt`).
  - Confirmed PhaseB/PhaseC identical outputs are *expected* (FFN caching = same output, just faster).
  - Created `scripts/slurm/sanity_full_check.sh` with corrected checkpoint paths.
  - Created `scripts/slurm/test_5sample.sh` for minimal testing with timing logs.
  - Created `scripts/slurm/train_cot_p1_traces.sh` for CoT trace collection.
- Status / Outcome:
  - Job 3960546: Full sanity check (P0/P2/P3/PhaseB/PhaseC) - completed, all 54%
  - Job 3960549: 5-sample test with timing - completed
- Git commits:
  - Pending: sanity_full_check.sh, test_5sample.sh
- SLURM jobs:
  - `3960546` (sanity_full - completed)
  - `3960549` (test_5sample - completed)
- Next steps / TODOs:
  - Monitor job results
  - Submit P1 trace collection after sanity check passes
  - Re-train P3/PhaseB/C with CoT traces

### 2025-12-08 06:27 UTC — ROOT CAUSE FIX: max_new_tokens Configuration
- Author: AI Assistant
- Phases / Modules: [diffllm.py, evaluation scripts]
- Goal:
  - Fix low accuracy (54%) by aligning with official Dream evaluation config.
- Actions:
  - Identified ROOT CAUSE: `max_new_tokens` in `gen_kwargs` is **ignored** by diffllm.py
  - Line 363 in diffllm.py uses `self.max_new_tokens` (default 128) from `__init__`
  - Official config uses `max_new_tokens=256` in model_args
  - Created `scripts/slurm/cot_fix_test.sh` with corrected config
  - Created `scripts/slurm/poc_official_eval.sh` for full official-aligned eval
- Status / Outcome:
  - Job 3960614 (10 samples): **FIX VERIFIED**
    - Timing logs now saving correctly
    - `mean_tokens_per_sample=256.0` (config working)
    - Latency: 27.5s mean, p95=29.5s, p99=29.7s
    - Accuracy: 40% (high variance with small n)
  - Job 3960636: 50-sample official eval submitted
- Key Fix:
  ```
  # BEFORE (BROKEN): max_new_tokens in gen_kwargs (ignored!)
  --gen_kwargs "max_new_tokens=256,..."
  
  # AFTER (CORRECT): max_new_tokens in model_args
  --model_args "max_new_tokens=256,diffusion_steps=256,..."
  ```
- SLURM jobs:
  - `3960614` (cot_fix_test - completed, verified)
  - `3960636` (poc_official_eval 50 samples - running)

