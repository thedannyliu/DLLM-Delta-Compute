#!/bin/bash

# Monitor L40S and H100 P1 jobs
P1_L40S=3645078
P1_H100=3645083

echo "========================================="
echo "Monitoring P1 Jobs (L40S & H100)"
echo "L40S: $P1_L40S, H100: $P1_H100"
echo "========================================="

TRACES_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/traces/p1_100"
TRAINING_SCRIPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/scripts/slurm/eval_100_p3_l40s.sh"
SUBMITTED=0

while true; do
    # Check trace count
    if [ -d "$TRACES_DIR" ]; then
        TRACE_COUNT=$(find "$TRACES_DIR" -name "*.json" 2>/dev/null | wc -l)
        echo "[$(date)] Found $TRACE_COUNT traces"
        
        if [ $TRACE_COUNT -ge 90 ] && [ $SUBMITTED -eq 0 ]; then
            echo "[$(date)] ✓ Sufficient traces! Submitting training..."
            sbatch "$TRAINING_SCRIPT"
            SUBMITTED=1
            echo "[$(date)] Training submitted. Exiting."
            break
        fi
    fi
    
    # Check job status
    L40S_STATUS=$(squeue -j $P1_L40S -h -o "%T" 2>/dev/null)
    H100_STATUS=$(squeue -j $P1_H100 -h -o "%T" 2>/dev/null)
    echo "[$(date)] L40S: ${L40S_STATUS:-DONE}, H100: ${H100_STATUS:-DONE}"
    
    sleep 60
done
