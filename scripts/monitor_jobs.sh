#!/bin/bash
# Job monitoring script for 500-sample evaluations

JOBS="3641704 3641705 3641706 3641707 3641708"
PHASE_NAMES=("Phase_A" "P0" "P1" "P2" "P4")

echo "======================================"
echo "Monitoring 500-Sample Evaluation Jobs"
echo "======================================"
echo ""
echo "Jobs: ${JOBS}"
echo "Started: $(date)"
echo ""

while true; do
    clear
    echo "======================================"
    echo "Job Status - $(date)"
    echo "======================================"
    echo ""
    
    squeue -u eliu354 -o "%.10i %.12P %.30j %.8u %.2t %.10M %.6D %R"
    
    echo ""
    echo "======================================"
    echo "Completed Jobs"
    echo "======================================"
    
    for job_id in $JOBS; do
        status=$(sacct -j $job_id --format=State --noheader | head -1 | tr -d ' ')
        if [ "$status" = "COMPLETED" ]; then
            echo "✓ Job $job_id - COMPLETED"
        elif [ "$status" = "FAILED" ]; then
            echo "✗ Job $job_id - FAILED"
        elif [ "$status" = "RUNNING" ]; then
            echo "▶ Job $job_id - RUNNING"
        elif [ "$status" = "PENDING" ] || [ -z "$status" ]; then
            echo "⏳ Job $job_id - PENDING"
        fi
    done
    
    echo ""
    echo "Press Ctrl+C to stop monitoring"
    echo "Refreshing in 30 seconds..."
    
    sleep 30
done
