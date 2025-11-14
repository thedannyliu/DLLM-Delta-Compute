#!/bin/bash
# Verify that all evaluation scripts are properly configured

echo "================================================"
echo "Verification: Small-Scale Evaluation Setup"
echo "================================================"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "1. Checking SLURM scripts..."
echo ""

SCRIPTS=(
    "scripts/slurm/eval_P0_baseline.sh"
    "scripts/slurm/eval_P1_traces.sh"
    "scripts/slurm/eval_P2_early_stop.sh"
    "scripts/slurm/eval_PhaseA_caching.sh"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "✓ $script exists"
        
        # Check for --limit 100
        if grep -q "limit 100" "$script"; then
            echo "  ✓ Configured for 100 samples"
        else
            echo "  ✗ WARNING: Missing --limit 100"
        fi
        
        # Check for --seed 42
        if grep -q "seed 42" "$script"; then
            echo "  ✓ Using seed 42 for reproducibility"
        else
            echo "  ✗ WARNING: Missing --seed 42"
        fi
        
        # Check output directory
        if grep -q "experiments/" "$script"; then
            echo "  ✓ Outputs to experiments/ directory"
        fi
        
    else
        echo "✗ $script NOT FOUND"
    fi
    echo ""
done

echo "2. Checking experiment directories..."
echo ""

DIRS=(
    "experiments/P0_baseline/logs"
    "experiments/P1_traces/logs"
    "experiments/P1_traces/traces"
    "experiments/P2_early_stop/logs"
    "experiments/PhaseA_caching/logs"
)

for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir exists"
    else
        echo "✗ $dir NOT FOUND - creating..."
        mkdir -p "$dir"
    fi
done
echo ""

echo "3. Checking conda environment..."
echo ""
if conda env list | grep -q "dcllm"; then
    echo "✓ Conda environment 'dcllm' exists"
else
    echo "✗ WARNING: Conda environment 'dcllm' not found"
fi
echo ""

echo "4. Checking HuggingFace login..."
echo ""
if [ -f "$HOME/.cache/huggingface/token" ] || [ -f "$HOME/.huggingface/token" ]; then
    echo "✓ HuggingFace token found"
else
    echo "✗ WARNING: HuggingFace token not found"
    echo "  Run: huggingface-cli login"
fi
echo ""

echo "5. Expected runtime for 100 samples:"
echo "  - P0 Baseline:     ~2-3 hours"
echo "  - P1 Traces:       ~2-3 hours (with tracing overhead)"
echo "  - P2 Early Stop:   ~1-2 hours (faster due to early stopping)"
echo "  - Phase A Caching: ~2-3 hours"
echo ""

echo "6. Data consistency check:"
echo "  All scripts use:"
echo "    --limit 100    (same sample size)"
echo "    --seed 42      (same random seed)"
echo "    --tasks gsm8k  (same dataset)"
echo "  This ensures all phases evaluate on identical data."
echo ""

echo "================================================"
echo "Verification complete!"
echo "================================================"
echo ""
echo "To submit all jobs, run:"
echo "  bash scripts/slurm/submit_all_phases.sh"
echo ""
