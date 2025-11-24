#!/bin/bash
# Monitor all pipeline stages

echo "========================================"
echo "Pipeline Status Monitor"
echo "Date: $(date)"
echo "========================================"
echo ""

# Batch 1: Evaluations
echo "--- Batch 1: Baseline Evaluations ---"
if [ -f /tmp/batch1_jobs.txt ]; then
    source /tmp/batch1_jobs.txt
    for job in $P1_JOB $P0_JOB $P2_JOB $P4_JOB $PHASEA_JOB; do
        sacct -j $job --format="JobID%10,JobName%20,State%12,Elapsed%12" --noheader | grep "^$job " || echo "$job     (not found)"
    done
else
    echo "Job info not found. Recent jobs:"
    sacct -u eliu354 --starttime=2025-11-24T17:00:00 --format="JobID%10,JobName%20,State%12,Elapsed%12" --noheader | grep -E "eval_.*_500" | head -5
fi
echo ""

# Batch 2: Training
echo "--- Batch 2: Training ---"
if [ -f /tmp/batch2_jobs.txt ]; then
    source /tmp/batch2_jobs.txt
    for job in $P3_JOB $PHASEB_JOB $PHASEC_JOB; do
        sacct -j $job --format="JobID%10,JobName%20,State%12,Elapsed%12" --noheader | grep "^$job " || echo "$job     (pending submission)"
    done
else
    echo "(Not yet submitted - waiting for P1)"
fi
echo ""

# Batch 3: Learned Model Evaluations
echo "--- Batch 3: Learned Model Evaluations ---"
if [ -f /tmp/batch3_jobs.txt ]; then
    source /tmp/batch3_jobs.txt
    for job in $P3_EVAL $PHASEB_EVAL $PHASEC_EVAL; do
        sacct -j $job --format="JobID%10,JobName%20,State%12,Elapsed%12" --noheader | grep "^$job " || echo "$job     (pending submission)"
    done
else
    echo "(Not yet submitted - waiting for training)"
fi
echo ""

# Automation status
echo "--- Automation Status ---"
if ps aux | grep -v grep | grep -q "auto_submit_training.sh 3644323"; then
    echo "✓ Training auto-submission: ACTIVE (monitoring P1 job 3644323)"
else
    echo "✗ Training auto-submission: NOT RUNNING"
fi

if ps aux | grep -v grep | grep -q "auto_submit_batch3.sh"; then
    echo "✓ Batch 3 auto-submission: ACTIVE"
else
    echo "○ Batch 3 auto-submission: NOT STARTED (will run after training)"
fi
echo ""

echo "========================================"
