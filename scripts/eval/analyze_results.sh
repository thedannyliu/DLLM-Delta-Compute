#!/bin/bash
# Auto-generated analysis script
# Run this after all jobs complete

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "============================================================"
echo "Result Analysis"
echo "============================================================"
echo ""

# Find latest logs
P0_LOG=$(ls -t $PROJECT_ROOT/experiments/P0_baseline/logs/*.out | head -1)
P1_LOG=$(ls -t $PROJECT_ROOT/experiments/P1_traces/logs/*.out | head -1)
P2_LOG=$(ls -t $PROJECT_ROOT/experiments/P2_early_stop/logs/*.out | head -1)

echo "Analyzing logs..."
echo ""

# Extract metrics
echo "=== P0 Baseline ==="
python "$PROJECT_ROOT/scripts/eval/count_gsm8k_responses.py" "$P0_LOG"
echo ""

echo "=== P1 Traces ==="
python "$PROJECT_ROOT/scripts/eval/count_gsm8k_responses.py" "$P1_LOG"
echo ""

echo "=== P2 Early Stopping ==="
python "$PROJECT_ROOT/scripts/eval/count_gsm8k_responses.py" "$P2_LOG"
echo ""

# Check early stopping
echo "=== Early Stopping Verification ==="
if grep -q "⚡ EARLY STOP" "$P2_LOG"; then
    echo "✓ Early stopping WAS triggered!"
    grep "⚡ EARLY STOP" "$P2_LOG" | head -5
    echo "..."
    echo "Total early stops: $(grep -c '⚡ EARLY STOP' "$P2_LOG")"
else
    echo "❌ Early stopping was NOT triggered!"
    echo "This indicates the thresholds may be too strict."
fi
echo ""

# Check traces
echo "=== Trace Collection Verification ==="
TRACE_COUNT=$(ls -1 $PROJECT_ROOT/experiments/P1_traces/traces/*.json 2>/dev/null | wc -l)
if [ $TRACE_COUNT -gt 0 ]; then
    echo "✓ Found $TRACE_COUNT trace files"
    ls -lh $PROJECT_ROOT/experiments/P1_traces/traces/ | head -5
else
    echo "❌ No trace files found!"
fi
echo ""

# Compare timing
echo "=== Timing Comparison ==="
echo "P0: $(grep 'walltime=' "$P0_LOG" | awk -F'walltime=' '{print $2}' | awk -F',' '{print $1}')"
echo "P2: $(grep 'walltime=' "$P2_LOG" | awk -F'walltime=' '{print $2}' | awk -F',' '{print $1}')"
echo ""

echo "============================================================"
echo "Analysis complete!"
echo "============================================================"
