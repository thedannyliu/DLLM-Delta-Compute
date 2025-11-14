#!/bin/bash
# Submit all completed phases for small-scale testing (100 samples, seed=42)

echo "================================================"
echo "Submitting Small-Scale Evaluation Jobs (100 samples each)"
echo "All jobs use seed=42 for reproducibility"
echo "================================================"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Create necessary directories
mkdir -p experiments/{P0_baseline,P1_traces,P2_early_stop,PhaseA_caching}/logs

# Submit P0 Baseline
echo "1. Submitting P0 Baseline..."
JOB_P0=$(sbatch scripts/slurm/eval_P0_baseline.sh | awk '{print $4}')
echo "   Job ID: $JOB_P0"
echo ""

# Submit P1 Traces (can run in parallel with P0)
echo "2. Submitting P1 Trace Collection..."
JOB_P1=$(sbatch scripts/slurm/eval_P1_traces.sh | awk '{print $4}')
echo "   Job ID: $JOB_P1"
echo ""

# Submit P2 Early Stopping (depends on P0 for comparison)
echo "3. Submitting P2 Early Stopping..."
JOB_P2=$(sbatch --dependency=afterok:$JOB_P0 scripts/slurm/eval_P2_early_stop.sh | awk '{print $4}')
echo "   Job ID: $JOB_P2 (starts after P0 completes)"
echo ""

# Submit Phase A Caching (can run after P1 traces for optimal schedule)
echo "4. Submitting Phase A FFN Caching..."
JOB_PA=$(sbatch --dependency=afterok:$JOB_P1 scripts/slurm/eval_PhaseA_caching.sh | awk '{print $4}')
echo "   Job ID: $JOB_PA (starts after P1 completes)"
echo ""

echo "================================================"
echo "All jobs submitted successfully!"
echo "================================================"
echo ""
echo "Job Dependencies:"
echo "  - P0 Baseline:      $JOB_P0 (runs immediately)"
echo "  - P1 Traces:        $JOB_P1 (runs immediately)"
echo "  - P2 Early Stop:    $JOB_P2 (waits for P0)"
echo "  - Phase A Caching:  $JOB_PA (waits for P1)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  tail -f experiments/P0_baseline/logs/gsm8k_baseline_${JOB_P0}.out"
echo "  tail -f experiments/P1_traces/logs/gsm8k_traces_${JOB_P1}.out"
echo "  tail -f experiments/P2_early_stop/logs/gsm8k_early_stop_${JOB_P2}.out"
echo "  tail -f experiments/PhaseA_caching/logs/gsm8k_caching_${JOB_PA}.out"
echo ""
echo "Expected completion: ~2-4 hours per job"
echo "================================================"
