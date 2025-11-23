#!/bin/bash
#SBATCH -Jsmoke_test_all        # Job name
#SBATCH -N1 --gres=gpu:L40S:1   # 1 node, 1 L40S GPU
#SBATCH --mem-per-gpu=40G       # Memory per GPU
#SBATCH -t0-02:00:00            # 2 hours time limit
#SBATCH -o logs/comprehensive_smoke_test_%j.out
#SBATCH -p ice-gpu              # Queue name

# Comprehensive smoke test for all P0-P4 and Phase A-C
# Tests with limit=1 to verify each phase can execute

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "============================================"
echo "Comprehensive Smoke Test - All Phases"
echo "============================================"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA available:"
python -c "import torch; print(f'  PyTorch: {torch.__version__}'); print(f'  CUDA: {torch.cuda.is_available()}')"
echo "Started at: $(date)"
echo "============================================"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Install missing dependencies if needed
pip install -q sacrebleu evaluate scikit-learn sqlitedict word2number pytablewriter 2>&1 | grep -v "Requirement already satisfied" || true

MODEL_PATH="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Function to run a test
run_test() {
    local test_name="$1"
    local model_args="$2"
    local extra_args="$3"
    
    echo "=================================================="
    echo "TEST: $test_name"
    echo "=================================================="
    echo "Model args: $model_args"
    echo "Extra args: $extra_args"
    echo "Starting at: $(date)"
    
    cd external/Dream/eval_instruct
    
    python -m lm_eval --model diffllm \
        --model_args "$model_args" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit 1 \
        --seed 42 \
        $extra_args \
        2>&1 | tail -50
    
    local exit_code=$?
    cd ../../..
    
    if [ $exit_code -eq 0 ]; then
        echo "✓ $test_name PASSED"
    else
        echo "✗ $test_name FAILED (exit code: $exit_code)"
    fi
    echo ""
    
    return $exit_code
}

# Track results
declare -a RESULTS
declare -a NAMES

# ============================================
# P0 - Baseline
# ============================================
test_name="P0_Baseline"
NAMES+=("$test_name")
if run_test "$test_name" \
    "pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=False"; then
    RESULTS+=("PASS")
else
    RESULTS+=("FAIL")
fi

# ============================================
# P1 - Traces (collection only, no viz in smoke test)
# ============================================
test_name="P1_Traces"
NAMES+=("$test_name")
TRACE_OUTPUT="$(pwd)/experiments/P1_traces/traces_smoke_$TIMESTAMP"
mkdir -p "$TRACE_OUTPUT"
if run_test "$test_name" \
    "pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=True,trace_output_dir=$TRACE_OUTPUT"; then
    RESULTS+=("PASS")
    # Check if trace was created
    if [ -n "$(ls -A $TRACE_OUTPUT/*.pt 2>/dev/null)" ]; then
        echo "  ✓ Trace files created"
    else
        echo "  ⚠ Warning: No trace files found"
        RESULTS[-1]="WARN"
    fi
else
    RESULTS+=("FAIL")
fi

# ============================================
# P2 - Early Stop
# ============================================
test_name="P2_EarlyStop"
NAMES+=("$test_name")
if run_test "$test_name" \
    "pretrained=$MODEL_PATH,delta_mode=p2_early_stop,cache_mode=none,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1"; then
    RESULTS+=("PASS")
else
    RESULTS+=("FAIL")
fi

# ============================================
# Phase A - Heuristic Caching
# ============================================
test_name="PhaseA_Caching"
NAMES+=("$test_name")
if run_test "$test_name" \
    "pretrained=$MODEL_PATH,delta_mode=none,cache_mode=l2c_ffn,cache_schedule=0,1,2,3"; then
    RESULTS+=("PASS")
else
    RESULTS+=("FAIL")
fi

# ============================================
# P3 - Learned Gate
# ============================================
test_name="P3_LearnedGate"
NAMES+=("$test_name")
GATE_CHECKPOINT="$(pwd)/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
if [ -f "$GATE_CHECKPOINT" ]; then
    if run_test "$test_name" \
        "pretrained=$MODEL_PATH,delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=$GATE_CHECKPOINT,gate_threshold=0.7"; then
        RESULTS+=("PASS")
    else
        RESULTS+=("FAIL")
    fi
else
    echo "✗ $test_name SKIPPED - checkpoint not found: $GATE_CHECKPOINT"
    RESULTS+=("SKIP")
fi

# ============================================
# Phase B - Learned Router (Fixed Schedule)
# ============================================
test_name="PhaseB_Router"
NAMES+=("$test_name")
ROUTER_CHECKPOINT="$(pwd)/experiments/PhaseB_router/checkpoints/router_final.pt"
if [ -f "$ROUTER_CHECKPOINT" ]; then
    if run_test "$test_name" \
        "pretrained=$MODEL_PATH,delta_mode=none,cache_mode=l2c_learned,router_checkpoint=$ROUTER_CHECKPOINT,router_threshold=0.5"; then
        RESULTS+=("PASS")
    else
        RESULTS+=("FAIL")
    fi
else
    echo "✗ $test_name SKIPPED - checkpoint not found: $ROUTER_CHECKPOINT"
    RESULTS+=("SKIP")
fi

# ============================================
# Phase C - Continuous Router
# ============================================
test_name="PhaseC_Continuous"
NAMES+=("$test_name")
ROUTER_CHECKPOINT="$(pwd)/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"
if [ -f "$ROUTER_CHECKPOINT" ]; then
    if run_test "$test_name" \
        "pretrained=$MODEL_PATH,delta_mode=none,cache_mode=l2c_continuous,router_checkpoint=$ROUTER_CHECKPOINT,router_threshold=0.5"; then
        RESULTS+=("PASS")
    else
        RESULTS+=("FAIL")
    fi
else
    echo "✗ $test_name SKIPPED - checkpoint not found: $ROUTER_CHECKPOINT"
    RESULTS+=("SKIP")
fi

# ============================================
# P4 - Adaptive Steps
# ============================================
test_name="P4_Adaptive"
NAMES+=("$test_name")
if run_test "$test_name" \
    "pretrained=$MODEL_PATH,delta_mode=p4_adaptive,cache_mode=none,adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0"; then
    RESULTS+=("PASS")
else
    RESULTS+=("FAIL")
fi

# ============================================
# Summary Report
# ============================================
echo ""
echo "=================================================="
echo "COMPREHENSIVE SMOKE TEST SUMMARY"
echo "=================================================="
echo "Completed at: $(date)"
echo ""

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
WARN_COUNT=0

for i in "${!NAMES[@]}"; do
    name="${NAMES[$i]}"
    result="${RESULTS[$i]}"
    
    printf "%-20s : %s\n" "$name" "$result"
    
    case "$result" in
        PASS) ((PASS_COUNT++)) ;;
        FAIL) ((FAIL_COUNT++)) ;;
        SKIP) ((SKIP_COUNT++)) ;;
        WARN) ((WARN_COUNT++)) ;;
    esac
done

echo ""
echo "Summary: $PASS_COUNT passed, $FAIL_COUNT failed, $SKIP_COUNT skipped, $WARN_COUNT warnings"
echo "=================================================="

# Exit with failure if any tests failed
if [ $FAIL_COUNT -gt 0 ]; then
    exit 1
fi

exit 0
