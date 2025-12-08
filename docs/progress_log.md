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

