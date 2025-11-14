#!/bin/bash
#
# Comprehensive re-evaluation with proper controls
# Ensures fair comparison by clearing caches and running on different nodes
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================================"
echo "DLLM Delta-Compute: Comprehensive Re-evaluation"
echo "Date: $(date)"
echo "============================================================"
echo ""

# Configuration
CONDA_ENV="dcllm"
LIMIT=100
SEED=42
NUM_FEWSHOT=5

# Phase definitions
declare -A PHASES
PHASES[P0]="Baseline (no acceleration)"
PHASES[P1]="Teacher Trace Collection"
PHASES[P2]="Early Stopping"
PHASES[PhaseA]="FFN Caching"

echo "Configuration:"
echo "  - Dataset: GSM8K"
echo "  - Samples: $LIMIT"
echo "  - Seed: $SEED"
echo "  - Few-shot: $NUM_FEWSHOT"
echo ""

# Step 1: Clear HuggingFace cache to ensure fair timing
echo "Step 1: Cache Management"
echo "------------------------------"
read -p "Clear HuggingFace cache for fair comparison? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "⚠️  Clearing HF cache..."
    rm -rf ~/.cache/huggingface/hub/models--Dream-org--Dream-v0-Instruct-7B
    echo "✓ Cache cleared"
else
    echo "⚠️  WARNING: Using cached model - timing may not be fair!"
fi
echo ""

# Step 2: Verify patches are applied
echo "Step 2: Verification"
echo "------------------------------"
echo "Checking Dream patches..."

if grep -q "⚡ EARLY STOP" "$PROJECT_ROOT/external/Dream/modeling/generation_utils.py"; then
    echo "✓ Early stopping patch applied"
else
    echo "❌ Early stopping patch NOT applied!"
    echo "   Run: Apply patches from patches/early_stop_logging.patch.py"
    exit 1
fi

if grep -q "from .tracing import TraceCollector" "$PROJECT_ROOT/external/Dream/modeling/generation_utils.py"; then
    echo "✓ Tracing import fixed"
else
    echo "❌ Tracing import NOT fixed!"
    exit 1
fi

if [ -f "$PROJECT_ROOT/external/Dream/modeling/tracing.py" ]; then
    echo "✓ tracing.py exists"
else
    echo "❌ tracing.py NOT found!"
    exit 1
fi
echo ""

# Step 3: Submit jobs with explicit node exclusion
echo "Step 3: Job Submission"
echo "------------------------------"
echo "Submitting jobs with node exclusion to ensure fair comparison..."
echo ""

# Submit P0 on node group 1
echo "[1/4] Submitting P0 Baseline..."
JOB_P0=$(sbatch --exclude=atl1-1-03-013-8-0 "$PROJECT_ROOT/scripts/slurm/eval_P0_baseline.sh" | awk '{print $4}')
echo "  Job ID: $JOB_P0"

# Wait a bit to ensure different nodes
sleep 5

# Submit P1 on node group 2  
echo "[2/4] Submitting P1 Trace Collection..."
JOB_P1=$(sbatch --exclude=atl1-1-03-013-8-0 "$PROJECT_ROOT/scripts/slurm/eval_P1_traces.sh" | awk '{print $4}')
echo "  Job ID: $JOB_P1"

# Submit P2 depending on P0, but on different node
echo "[3/4] Submitting P2 Early Stopping (depends on P0)..."
JOB_P2=$(sbatch --dependency=afterok:$JOB_P0 --exclude=atl1-1-03-013-8-0 "$PROJECT_ROOT/scripts/slurm/eval_P2_early_stop.sh" | awk '{print $4}')
echo "  Job ID: $JOB_P2 (waits for $JOB_P0)"

# Submit Phase A depending on P1
echo "[4/4] Submitting Phase A FFN Caching (depends on P1)..."
JOB_PHASEA=$(sbatch --dependency=afterok:$JOB_P1 "$PROJECT_ROOT/scripts/slurm/eval_PhaseA_caching.sh" | awk '{print $4}')
echo "  Job ID: $JOB_PHASEA (waits for $JOB_P1)"

echo ""
echo "============================================================"
echo "All jobs submitted!"
echo "============================================================"
echo ""
echo "Job Summary:"
echo "  P0 Baseline:      $JOB_P0"
echo "  P1 Traces:        $JOB_P1"
echo "  P2 Early Stop:    $JOB_P2 (after P0)"
echo "  Phase A Caching:  $JOB_PHASEA (after P1)"
echo ""

# Step 4: Monitoring commands
echo "Monitoring Commands:"
echo "------------------------------"
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "Watch live logs:"
echo "  tail -f experiments/P0_baseline/logs/gsm8k_baseline_${JOB_P0}.out"
echo "  tail -f experiments/P1_traces/logs/gsm8k_traces_${JOB_P1}.out"
echo "  tail -f experiments/P2_early_stop/logs/gsm8k_early_stop_${JOB_P2}.out"
echo "  tail -f experiments/PhaseA_caching/logs/gsm8k_caching_${JOB_PHASEA}.out"
echo ""
echo "Check for early stopping logs:"
echo "  grep '⚡ EARLY STOP' experiments/P2_early_stop/logs/gsm8k_early_stop_${JOB_P2}.out"
echo ""
echo "Check for trace generation:"
echo "  ls -lh experiments/P1_traces/traces/"
echo ""

# Step 5: Create analysis script for after completion
cat > "$PROJECT_ROOT/scripts/eval/analyze_results.sh" << 'EOF'
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
EOF

chmod +x "$PROJECT_ROOT/scripts/eval/analyze_results.sh"

echo "Step 6: Post-completion Analysis"
echo "------------------------------"
echo "After jobs complete, run:"
echo "  bash $PROJECT_ROOT/scripts/eval/analyze_results.sh"
echo ""

echo "============================================================"
echo "Setup complete! Jobs are running..."
echo "Expected completion: ~4-6 hours"
echo "============================================================"
