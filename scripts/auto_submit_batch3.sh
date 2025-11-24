#!/bin/bash
# Auto-submit Batch 3 evaluation jobs after training completes

P3_JOB=$1
PHASEB_JOB=$2
PHASEC_JOB=$3

if [ -z "$P3_JOB" ] || [ -z "$PHASEB_JOB" ] || [ -z "$PHASEC_JOB" ]; then
    echo "Usage: $0 <P3_JOB> <PHASEB_JOB> <PHASEC_JOB>"
    echo ""
    echo "Or source from saved file:"
    echo "  source /tmp/batch2_jobs.txt"
    echo "  $0 \$P3_JOB \$PHASEB_JOB \$PHASEC_JOB"
    exit 1
fi

echo "========================================="
echo "Monitoring Training Jobs"
echo "========================================="
echo "P3 Gate:            $P3_JOB"
echo "Phase B Router:     $PHASEB_JOB"
echo "Phase C Continuous: $PHASEC_JOB"
echo ""

# Wait for all training jobs to complete
for JOB in $P3_JOB $PHASEB_JOB $PHASEC_JOB; do
    echo "Waiting for job $JOB..."
    while squeue -j $JOB -u eliu354 2>/dev/null | grep -q $JOB; do
        sleep 30
    done
    echo "✓ Job $JOB completed"
done

echo ""
echo "✓ All training jobs completed!"
echo ""

# Verify checkpoints exist
echo "========================================="
echo "Verifying Checkpoints"
echo "========================================="

P3_CKPT=~/scratch/Projects/DLLM-Delta-Compute/experiments/P3_learned_gate/checkpoints/gate_final.pt
PHASEB_CKPT=~/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseB_router/checkpoints/router_final.pt
PHASEC_CKPT=~/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt

MISSING=0
for CKPT in "$P3_CKPT" "$PHASEB_CKPT" "$PHASEC_CKPT"; do
    if [ -f "$CKPT" ]; then
        echo "✓ Found: $CKPT"
    else
        echo "✗ Missing: $CKPT"
        MISSING=$((MISSING + 1))
    fi
done

if [ $MISSING -gt 0 ]; then
    echo ""
    echo "ERROR: $MISSING checkpoint(s) missing! Check training logs."
    exit 1
fi

echo ""
echo "✓ All checkpoints verified"
echo ""

# Submit Batch 3 evaluation jobs
echo "========================================="
echo "Submitting Batch 3 Evaluations"
echo "========================================="

cd ~/scratch/Projects/DLLM-Delta-Compute

P3_EVAL=$(sbatch scripts/slurm/eval_500_p3.sh | awk '{print $4}')
PHASEB_EVAL=$(sbatch scripts/slurm/eval_500_phaseB.sh | awk '{print $4}')
PHASEC_EVAL=$(sbatch scripts/slurm/eval_500_phaseC.sh | awk '{print $4}')

echo "✓ Batch 3 evaluations submitted:"
echo "  P3 (Learned Gate):       Job $P3_EVAL"
echo "  Phase B (Router):        Job $PHASEB_EVAL"
echo "  Phase C (Continuous):    Job $PHASEC_EVAL"
echo ""

# Save job IDs
echo "P3_EVAL=$P3_EVAL" > /tmp/batch3_jobs.txt
echo "PHASEB_EVAL=$PHASEB_EVAL" >> /tmp/batch3_jobs.txt
echo "PHASEC_EVAL=$PHASEC_EVAL" >> /tmp/batch3_jobs.txt

echo "Batch 3 job IDs saved to /tmp/batch3_jobs.txt"
echo "========================================="
echo "Monitor evaluations with:"
echo "  squeue -j $P3_EVAL,$PHASEB_EVAL,$PHASEC_EVAL -u eliu354"
echo "========================================="
