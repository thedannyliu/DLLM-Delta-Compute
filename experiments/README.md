# Experiments Directory

This directory contains all experimental runs, organized by implementation phase.

## Structure

```
experiments/
├── P0_baseline/              # Baseline (teacher) experiments
├── P1_traces/                # Teacher trace collection
├── P2_early_stop/            # Early stopping experiments
├── P3_learned_gate/          # Learned gate experiments
├── P4_adaptive/              # Adaptive step scheduling
├── PhaseA_caching/           # Heuristic FFN caching
├── PhaseB_router/            # Learned router experiments
└── PhaseC_continuous/        # Continuous-time router

Each phase directory contains:
├── logs/          # SLURM job outputs (*.out files)
├── traces/        # Trace .pt files from runs
├── results/       # JSON results and metrics
└── configs/       # Experiment configurations
```

## Naming Conventions

### Log Files
SLURM output files: `slurm-{jobid}.out`

### Trace Files
Format: `{phase}_{task}_{date}_{config}.pt`
- Example: `P1_gsm8k_2025-11-15_256steps.pt`

### Result Files
Format: `{phase}_{task}_{date}_{config}_results.json`
- Example: `P2_gsm8k_2025-11-20_early_stop_results.json`

### Config Files
Format: `{phase}_{experiment_name}.json` or `.yaml`
- Example: `P2_early_stop_confidence095.json`

## Usage

### Running Experiments

All SLURM scripts automatically save outputs to the correct phase directory:
```bash
# Baseline evaluation
sbatch scripts/slurm/eval_P0_baseline.sh  # → experiments/P0_baseline/logs/

# Trace collection
sbatch scripts/slurm/eval_P1_traces.sh    # → experiments/P1_traces/logs/

# Early stopping
sbatch scripts/slurm/eval_P2_early_stop.sh  # → experiments/P2_early_stop/logs/
```

### Accessing Results

Results are saved with timestamps and can be found in respective `results/` directories:
```bash
# View latest P0 baseline results
cat experiments/P0_baseline/results/*_results.json | jq .

# Plot P1 traces
python tests/visualization/plot_traces.py \
  experiments/P1_traces/traces/P1_gsm8k_*.pt \
  --output-dir docs/reports/
```

## Data Retention

- **logs/**: Keep SLURM outputs for debugging (excluded from git via .gitignore)
- **traces/**: Keep trace .pt files for analysis (excluded from git, large files)
- **results/**: Keep JSON results (small, should be committed for reproducibility)
- **configs/**: Always commit configs for reproducibility

## Git Configuration

The following patterns are gitignored:
```
experiments/*/logs/*.out
experiments/*/traces/*.pt
```

Result JSONs and configs are tracked in git for reproducibility.
