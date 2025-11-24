#!/bin/bash
# Monitor all running jobs and report status

BATCH1_JOBS="3643107 3643149 3643150 3643151 3643152"

echo "========================================="
echo "Job Status Monitor"
echo "========================================="
echo "Time: $(date)"
echo ""

echo "Batch 1 Jobs (No Dependencies):"
echo "  P1 Traces: 3643107"
echo "  P0 Baseline: 3643149"
echo "  P2 Early Stop: 3643150"
echo "  P4 Adaptive: 3643151"
echo "  Phase A: 3643152"
echo ""

# Check each job
for job_id in $BATCH1_JOBS; do
    status=$(sacct -j $job_id --format=JobID,JobName,State,Elapsed -n 2>/dev/null | head -1)
    if echo "$status" | grep -q "RUNNING"; then
        elapsed=$(echo "$status" | awk '{print $4}')
        name=$(echo "$status" | awk '{print $2}')
        echo "🔄 Job $job_id ($name): RUNNING - Elapsed: $elapsed"
    elif echo "$status" | grep -q "COMPLETED"; then
        name=$(echo "$status" | awk '{print $2}')
        elapsed=$(echo "$status" | awk '{print $4}')
        echo "✅ Job $job_id ($name): COMPLETED - Time: $elapsed"
    elif echo "$status" | grep -q "FAILED"; then
        name=$(echo "$status" | awk '{print $2}')
        echo "❌ Job $job_id ($name): FAILED"
    elif echo "$status" | grep -q "PENDING"; then
        name=$(echo "$status" | awk '{print $2}')
        echo "⏳ Job $job_id ($name): PENDING"
    else
        echo "? Job $job_id: UNKNOWN STATUS"
    fi
done

echo ""
echo "To continuously monitor: watch -n 30 '$0'"
