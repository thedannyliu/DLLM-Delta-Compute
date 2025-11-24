#!/bin/bash
# Wait for P1 completion, verify traces, and submit training jobs

set -e

P1_JOB=3643107

echo "====================================="
echo "Waiting for P1 (Job $P1_JOB) to complete..."
echo "====================================="

# Wait for P1 to complete
while squeue -j $P1_JOB 2>/dev/null | grep -q $P1_JOB; do
    elapsed=$(sacct -j $P1_JOB --format=Elapsed -n 2>/dev/null | head -1 | tr -d ' ')
    echo "P1 still running... Elapsed: $elapsed"
    sleep 60
done

# Check if P1 succeeded
state=$(sacct -j $P1_JOB --format=State -n 2>/dev/null | head -1 | tr -d ' ')
exitcode=$(sacct -j $P1_JOB --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ')

if [ "$state" != "COMPLETED" ] || [ "$exitcode" != "0:0" ]; then
    echo "ERROR: P1 job failed! State: $state, Exit: $exitcode"
    exit 1
fi

echo "✓ P1 completed successfully!"
echo ""

# Find and verify traces
echo "====================================="
echo "Verifying P1 traces..."
echo "====================================="

TRACE_DIR=$(ls -td /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces/traces_500_* 2>/dev/null | head -1)

if [ -z "$TRACE_DIR" ] || [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: Trace directory not found!"
    echo "Expected: experiments/P1_traces/traces_500_*"
    exit 1
fi

TRACE_COUNT=$(ls -1 "$TRACE_DIR"/*.pt 2>/dev/null | wc -l)

echo "Trace directory: $TRACE_DIR"
echo "Trace count: $TRACE_COUNT"

if [ "$TRACE_COUNT" -lt 400 ]; then
    echo "ERROR: Insufficient traces! Expected ~500, found $TRACE_COUNT"
    exit 1
fi

echo "✓ Traces verified!"
echo ""

# Submit training jobs
echo "====================================="
echo "Submitting Training Jobs..."
echo "====================================="

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "Submitting P3 gate training..."
JOB_P3=$(sbatch scripts/slurm/train_P3_gate.sh | awk '{print $4}')
echo "  Job ID: $JOB_P3"

echo "Submitting Phase B router training..."
JOB_PHASEB=$(sbatch scripts/slurm/train_PhaseB_router.sh | awk '{print $4}')
echo "  Job ID: $JOB_PHASEB"

echo "Submitting Phase C continuous router training..."
JOB_PHASEC=$(sbatch scripts/slurm/train_PhaseC_continuous.sh | awk '{print $4}')
echo "  Job ID: $JOB_PHASEC"

echo ""
echo "====================================="
echo "Training jobs submitted!"
echo "====================================="
echo "P3 Gate: $JOB_P3"
echo "Phase B Router: $JOB_PHASEB"
echo "Phase C Router: $JOB_PHASEC"
echo ""
echo "Monitor with: squeue -j $JOB_P3,$JOB_PHASEB,$JOB_PHASEC"
echo ""
echo "Next: Wait for training to complete, then run submit_batch3_evals.sh"
