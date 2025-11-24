#!/bin/bash

# Monitor multiple P1 jobs (different GPU types)
P1_JOBS="3644946 3644875 3644887 3644874"  # L40S A100 H100 H200

echo "========================================="
echo "Monitoring Multiple P1 Jobs"
echo "L40S: 3644872, A100: 3644875"
echo "H100: 3644887, H200: 3644874"
echo "========================================="

TRACES_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/traces/p1_100"
TRAINING_SCRIPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/scripts/slurm/eval_100_p3_l40s.sh"
SUBMITTED=0

while true; do
    # Check if any P1 job completed successfully
    COMPLETED=0
    for JOB_ID in $P1_JOBS; do
        STATUS=$(sacct -j $JOB_ID -o State -n | head -1 | tr -d ' ')
        if [[ "$STATUS" == "COMPLETED" ]]; then
            echo "[$(date)] Job $JOB_ID completed!"
            COMPLETED=1
            break
        elif [[ "$STATUS" == "RUNNING" ]]; then
            echo "[$(date)] Job $JOB_ID is running..."
        fi
    done
    
    # Check trace count
    if [ -d "$TRACES_DIR" ]; then
        TRACE_COUNT=$(find "$TRACES_DIR" -name "*.json" 2>/dev/null | wc -l)
        echo "[$(date)] Found $TRACE_COUNT traces"
        
        if [ $TRACE_COUNT -ge 90 ] && [ $SUBMITTED -eq 0 ]; then
            echo "[$(date)] ✓ Sufficient traces found! Submitting training..."
            sbatch "$TRAINING_SCRIPT"
            SUBMITTED=1
            echo "[$(date)] Training submitted. Exiting monitor."
            break
        fi
    fi
    
    sleep 60
done
