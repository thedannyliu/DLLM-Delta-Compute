#!/bin/bash
# Wait for training completion and submit Batch 3 evaluations

set -e

# Training job IDs will be passed as arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <P3_JOB_ID> <PHASEB_JOB_ID> <PHASEC_JOB_ID>"
    echo ""
    echo "Or automatically detect from recent training jobs..."
    
    # Try to find recent training jobs
    P3_JOB=$(sacct -u $USER --format=JobID,JobName,State --starttime=now-1hour -n | grep "train_P3" | head -1 | awk '{print $1}' | cut -d'.' -f1)
    PHASEB_JOB=$(sacct -u $USER --format=JobID,JobName,State --starttime=now-1hour -n | grep "train_PhaseB" | head -1 | awk '{print $1}' | cut -d'.' -f1)
    PHASEC_JOB=$(sacct -u $USER --format=JobID,JobName,State --starttime=now-1hour -n | grep "train_PhaseC" | head -1 | awk '{print $1}' | cut -d'.' -f1)
    
    if [ -z "$P3_JOB" ] || [ -z "$PHASEB_JOB" ] || [ -z "$PHASEC_JOB" ]; then
        echo "ERROR: Could not auto-detect training jobs"
        exit 1
    fi
    
    echo "Auto-detected training jobs:"
    echo "  P3: $P3_JOB"
    echo "  Phase B: $PHASEB_JOB"
    echo "  Phase C: $PHASEC_JOB"
else
    P3_JOB=$1
    PHASEB_JOB=$2
    PHASEC_JOB=$3
fi

echo "====================================="
echo "Waiting for training jobs to complete..."
echo "====================================="
echo "P3 Gate: $P3_JOB"
echo "Phase B Router: $PHASEB_JOB"
echo "Phase C Router: $PHASEC_JOB"
echo ""

# Function to wait for a job
wait_for_job() {
    local job_id=$1
    local job_name=$2
    
    while squeue -j $job_id 2>/dev/null | grep -q $job_id; do
        elapsed=$(sacct -j $job_id --format=Elapsed -n 2>/dev/null | head -1 | tr -d ' ')
        echo "$job_name still running... Elapsed: $elapsed"
        sleep 60
    done
    
    state=$(sacct -j $job_id --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    exitcode=$(sacct -j $job_id --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ')
    
    if [ "$state" != "COMPLETED" ] || [ "$exitcode" != "0:0" ]; then
        echo "ERROR: $job_name failed! State: $state, Exit: $exitcode"
        return 1
    fi
    
    echo "✓ $job_name completed successfully!"
    return 0
}

# Wait for all training jobs
wait_for_job $P3_JOB "P3 training" || exit 1
wait_for_job $PHASEB_JOB "Phase B training" || exit 1
wait_for_job $PHASEC_JOB "Phase C training" || exit 1

echo ""
echo "====================================="
echo "All training completed! Verifying checkpoints..."
echo "====================================="

# Verify checkpoints exist
P3_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
PHASEB_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseB_router/checkpoints/router_final.pt"
PHASEC_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

check_checkpoint() {
    local ckpt=$1
    local name=$2
    
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: $name checkpoint not found at $ckpt"
        return 1
    fi
    
    size=$(ls -lh "$ckpt" | awk '{print $5}')
    echo "✓ $name: $size"
    return 0
}

check_checkpoint "$P3_CKPT" "P3 Gate" || exit 1
check_checkpoint "$PHASEB_CKPT" "Phase B Router" || exit 1
check_checkpoint "$PHASEC_CKPT" "Phase C Router" || exit 1

echo ""
echo "====================================="
echo "Submitting Batch 3 evaluations..."
echo "====================================="

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "Submitting P3 evaluation..."
JOB_P3_EVAL=$(sbatch scripts/slurm/eval_500_p3.sh | awk '{print $4}')
echo "  Job ID: $JOB_P3_EVAL"

echo "Submitting Phase B evaluation..."
JOB_PHASEB_EVAL=$(sbatch scripts/slurm/eval_500_phaseB.sh | awk '{print $4}')
echo "  Job ID: $JOB_PHASEB_EVAL"

echo "Submitting Phase C evaluation..."
JOB_PHASEC_EVAL=$(sbatch scripts/slurm/eval_500_phaseC.sh | awk '{print $4}')
echo "  Job ID: $JOB_PHASEC_EVAL"

echo ""
echo "====================================="
echo "Batch 3 evaluations submitted!"
echo "====================================="
echo "P3 Eval: $JOB_P3_EVAL"
echo "Phase B Eval: $JOB_PHASEB_EVAL"
echo "Phase C Eval: $JOB_PHASEC_EVAL"
echo ""
echo "Monitor with: squeue -j $JOB_P3_EVAL,$JOB_PHASEB_EVAL,$JOB_PHASEC_EVAL"
echo ""
echo "Pipeline will be complete when these jobs finish!"
