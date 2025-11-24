#!/bin/bash
# Auto-submit training jobs after P1 completes with sufficient traces

P1_JOB=$1
MIN_TRACES=450  # Accept 450+ traces out of 500 (90%)

if [ -z "$P1_JOB" ]; then
    echo "Usage: $0 <P1_JOB_ID>"
    exit 1
fi

echo "========================================="
echo "Monitoring P1 Job: $P1_JOB"
echo "========================================="

# Wait for P1 to complete
while squeue -j $P1_JOB -u eliu354 2>/dev/null | grep -q $P1_JOB; do
    sleep 30
done

echo "✓ P1 completed!"
echo ""

# Find the latest traces directory
TRACES_DIR=$(find ~/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces -name "traces_500_*" -type d | sort | tail -1)

if [ -z "$TRACES_DIR" ]; then
    echo "ERROR: No traces directory found!"
    exit 1
fi

# Count trace files
TRACE_COUNT=$(ls -1 "$TRACES_DIR"/*.pt 2>/dev/null | wc -l)
echo "Trace directory: $TRACES_DIR"
echo "Trace count: $TRACE_COUNT"

if [ "$TRACE_COUNT" -lt "$MIN_TRACES" ]; then
    echo "ERROR: Insufficient traces! Expected ~500, found $TRACE_COUNT"
    exit 1
fi

echo "✓ Sufficient traces found"
echo ""

# Submit training jobs
echo "========================================="
echo "Submitting Training Jobs"
echo "========================================="

cd ~/scratch/Projects/DLLM-Delta-Compute

P3_JOB=$(sbatch scripts/slurm/train_P3_gate.sh | awk '{print $4}')
PHASEB_JOB=$(sbatch scripts/slurm/train_PhaseB_router.sh | awk '{print $4}')
PHASEC_JOB=$(sbatch scripts/slurm/train_PhaseC_continuous.sh | awk '{print $4}')

echo "✓ Training jobs submitted:"
echo "  P3 Gate:            Job $P3_JOB"
echo "  Phase B Router:     Job $PHASEB_JOB"
echo "  Phase C Continuous: Job $PHASEC_JOB"
echo ""

# Save job IDs for next stage
echo "P3_JOB=$P3_JOB" > /tmp/batch2_jobs.txt
echo "PHASEB_JOB=$PHASEB_JOB" >> /tmp/batch2_jobs.txt
echo "PHASEC_JOB=$PHASEC_JOB" >> /tmp/batch2_jobs.txt

echo "Training job IDs saved to /tmp/batch2_jobs.txt"
echo "========================================="
echo "Monitor training with:"
echo "  squeue -j $P3_JOB,$PHASEB_JOB,$PHASEC_JOB -u eliu354"
echo "========================================="
