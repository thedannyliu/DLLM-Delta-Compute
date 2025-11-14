# Scripts Directory

This directory contains all executable scripts for the DLLM Delta-Compute project.

## Structure

```
scripts/
├── slurm/                    # SLURM batch job scripts
│   ├── slurm_test_poc.sh            # Quick GPU validation test
│   ├── eval_P0_baseline.sh          # P0: Baseline (teacher) evaluation
│   ├── eval_P1_traces.sh            # P1: Teacher trace collection
│   ├── eval_P2_early_stop.sh        # P2: Early stopping evaluation
│   ├── eval_PhaseA_caching.sh       # Phase A: FFN caching evaluation
│   └── slurm_eval_gsm8k.sh          # Generic GSM8K evaluator (legacy)
├── setup/                    # Environment setup scripts
│   └── setup_env.sh                 # Conda environment setup
└── eval/                     # Local evaluation scripts
    └── verify_dream_baseline.sh     # Verify Dream baseline setup
```

## Usage

### SLURM Jobs

All SLURM scripts automatically save logs to the correct `experiments/` subdirectory.

**Quick GPU Test** (8 hours, minimal test):
```bash
sbatch scripts/slurm/slurm_test_poc.sh
```

**P0 Baseline Evaluation** (12 hours, GSM8K):
```bash
sbatch scripts/slurm/eval_P0_baseline.sh
```

**P1 Trace Collection** (12 hours, GSM8K with tracing):
```bash
sbatch scripts/slurm/eval_P1_traces.sh
```

**P2 Early Stopping** (12 hours, GSM8K with early stop):
```bash
sbatch scripts/slurm/eval_P2_early_stop.sh
```

**Phase A FFN Caching** (12 hours, GSM8K with caching):
```bash
sbatch scripts/slurm/eval_PhaseA_caching.sh
```

**Generic Evaluator** (supports multiple modes):
```bash
# Usage: sbatch slurm_eval_gsm8k.sh [model_path] [mode]
sbatch scripts/slurm/slurm_eval_gsm8k.sh "hkust-nlp/Dream-7B" baseline
sbatch scripts/slurm/slurm_eval_gsm8k.sh "hkust-nlp/Dream-7B" trace
sbatch scripts/slurm/slurm_eval_gsm8k.sh "hkust-nlp/Dream-7B" cache
sbatch scripts/slurm/slurm_eval_gsm8k.sh "hkust-nlp/Dream-7B" early_stop
```

### Setup Scripts

**Environment Setup** (run once):
```bash
bash scripts/setup/setup_env.sh
```

This creates the `dcllm` conda environment and installs all dependencies.

### Local Evaluation

**Verify Baseline** (no SLURM, quick check):
```bash
bash scripts/eval/verify_dream_baseline.sh
```

## Output Locations

All scripts save outputs to structured locations:

- **Logs**: `experiments/{phase}/logs/*.out` (SLURM job outputs)
- **Results**: `experiments/{phase}/results/*.json` (evaluation metrics)
- **Traces**: `experiments/{phase}/traces/*.pt` (P1 trace files)

Example for P0 baseline:
```
experiments/P0_baseline/
├── logs/gsm8k_baseline_12345.out        # SLURM output
└── results/gsm8k_20251114_120000/       # Timestamped results
    ├── results.json                      # Metrics
    └── eval.log                          # Full evaluation log
```

## Adding New Scripts

### SLURM Job Template

```bash
#!/bin/bash
#SBATCH -J{job_name}
#SBATCH -N1 --gres=gpu:H100:1
#SBATCH --mem-per-gpu=80G
#SBATCH -t0-12:00:00
#SBATCH -o experiments/{phase}/logs/{name}_%j.out
#SBATCH -qinfinity
#SBATCH -A GT-hl94

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Your commands here
```

### Best Practices

1. **Always** set `-o` to save logs in `experiments/{phase}/logs/`
2. **Always** create timestamped output directories
3. **Always** use `tee` to duplicate output to log files
4. **Validate** environment before running main commands
5. **Report** start/end times and output locations

## Monitoring Jobs

Check job status:
```bash
squeue -u $USER
```

View job output (while running):
```bash
tail -f experiments/P0_baseline/logs/gsm8k_baseline_12345.out
```

Cancel job:
```bash
scancel <job_id>
```
