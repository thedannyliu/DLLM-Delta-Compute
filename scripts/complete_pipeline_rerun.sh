#!/bin/bash
# Comprehensive re-run of all evaluations with timing metrics
# Respects dependencies between phases

set -e

echo "====================================="
echo "Complete Pipeline Re-run with Timing"
echo "====================================="
echo "Date: $(date)"
echo ""

# Track job IDs
declare -A JOB_IDS
declare -A JOB_STATUS

# Function to submit job and track
submit_job() {
    local script=$1
    local phase=$2
    echo "Submitting $phase..."
    job_id=$(sbatch "$script" | awk '{print $4}')
    JOB_IDS[$phase]=$job_id
    echo "  Job ID: $job_id"
}

# Function to wait for job completion
wait_for_job() {
    local phase=$1
    local job_id=${JOB_IDS[$phase]}
    echo ""
    echo "Waiting for $phase (Job $job_id) to complete..."
    
    while squeue -j "$job_id" 2>/dev/null | grep -q "$job_id"; do
        elapsed=$(sacct -j "$job_id" --format=Elapsed -n 2>/dev/null | head -1 | tr -d ' ')
        echo "  $phase still running... Elapsed: $elapsed"
        sleep 30
    done
    
    state=$(sacct -j "$job_id" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
    exitcode=$(sacct -j "$job_id" --format=ExitCode -n 2>/dev/null | head -1 | tr -d ' ')
    
    if [ "$state" = "COMPLETED" ] && [ "$exitcode" = "0:0" ]; then
        echo "  ✓ $phase COMPLETED successfully"
        JOB_STATUS[$phase]="SUCCESS"
        return 0
    else
        echo "  ✗ $phase FAILED (State: $state, Exit: $exitcode)"
        JOB_STATUS[$phase]="FAILED"
        return 1
    fi
}

# Function to verify traces
verify_traces() {
    local trace_dir=$1
    if [ -d "$trace_dir" ]; then
        local count=$(ls -1 "$trace_dir"/*.pt 2>/dev/null | wc -l)
        if [ "$count" -gt 0 ]; then
            echo "  ✓ Found $count trace files"
            return 0
        fi
    fi
    echo "  ✗ No traces found in $trace_dir"
    return 1
}

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "====================================="
echo "BATCH 1: No Dependencies"
echo "====================================="

# Submit Batch 1 (all can run in parallel)
submit_job "scripts/slurm/eval_500_p0.sh" "P0_Baseline"
submit_job "scripts/slurm/eval_500_p1.sh" "P1_Traces"
submit_job "scripts/slurm/eval_500_p2.sh" "P2_EarlyStop"
submit_job "scripts/slurm/eval_500_p4.sh" "P4_Adaptive"
submit_job "scripts/slurm/eval_500_phaseA.sh" "PhaseA_Heuristic"

echo ""
echo "Batch 1 jobs submitted. Waiting for completion..."

# Wait for all Batch 1 jobs
wait_for_job "P0_Baseline"
wait_for_job "P1_Traces"
wait_for_job "P2_EarlyStop"
wait_for_job "P4_Adaptive"
wait_for_job "PhaseA_Heuristic"

# Verify P1 traces
echo ""
echo "Verifying P1 traces..."
P1_JOB=${JOB_IDS[P1_Traces]}
TRACE_DIR=$(grep "Trace files saved to:" "logs/eval_P1_500_${P1_JOB}.out" | awk '{print $NF}')

if ! verify_traces "$TRACE_DIR"; then
    echo "ERROR: P1 traces not generated. Cannot proceed to training."
    exit 1
fi

echo ""
echo "====================================="
echo "BATCH 2: Training (Requires P1 Traces)"
echo "====================================="

# Submit training jobs
submit_job "scripts/slurm/train_P3_gate.sh" "Train_P3"
submit_job "scripts/slurm/train_PhaseB_router.sh" "Train_PhaseB"
submit_job "scripts/slurm/train_PhaseC_continuous.sh" "Train_PhaseC"

echo ""
echo "Training jobs submitted. Waiting for completion..."

# Wait for all training jobs
wait_for_job "Train_P3"
wait_for_job "Train_PhaseB"
wait_for_job "Train_PhaseC"

echo ""
echo "====================================="
echo "BATCH 3: Evaluations (Requires Training)"
echo "====================================="

# Submit Batch 3 (requires trained checkpoints)
submit_job "scripts/slurm/eval_500_p3.sh" "P3_LearnedGate"
submit_job "scripts/slurm/eval_500_phaseB.sh" "PhaseB_Router"
submit_job "scripts/slurm/eval_500_phaseC.sh" "PhaseC_Continuous"

echo ""
echo "Batch 3 jobs submitted. Waiting for completion..."

# Wait for all Batch 3 jobs
wait_for_job "P3_LearnedGate"
wait_for_job "PhaseB_Router"
wait_for_job "PhaseC_Continuous"

echo ""
echo "====================================="
echo "PIPELINE COMPLETE"
echo "====================================="
echo ""
echo "Job Summary:"
for phase in "${!JOB_IDS[@]}"; do
    echo "  $phase: Job ${JOB_IDS[$phase]} - ${JOB_STATUS[$phase]}"
done

echo ""
echo "All evaluations and training completed!"
echo "Check logs/ directory for detailed results."
echo "Run analysis script to generate comparison tables."
