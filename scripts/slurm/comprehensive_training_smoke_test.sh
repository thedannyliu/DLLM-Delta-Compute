#!/bin/bash
#SBATCH --job-name=train_smoke_test
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=2:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Comprehensive Training Smoke Test
# Tests: P3 Gate Training, Phase B Router Training, Phase C Router Training
# ==================================================

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "====================================="
echo "Environment Setup"
echo "====================================="
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Track results
RESULTS=()
declare -a FAILED_TESTS=()
declare -a PASSED_TESTS=()
declare -a SKIPPED_TESTS=()
declare -a WARNINGS=()

# Helper function to report test status
report_test() {
    local test_name=$1
    local status=$2  # PASS, FAIL, SKIP, WARN
    local message=$3
    
    if [ "$status" = "PASS" ]; then
        PASSED_TESTS+=("$test_name")
        echo "✓ $test_name PASSED"
    elif [ "$status" = "FAIL" ]; then
        FAILED_TESTS+=("$test_name")
        echo "✗ $test_name FAILED: $message"
    elif [ "$status" = "SKIP" ]; then
        SKIPPED_TESTS+=("$test_name")
        echo "○ $test_name SKIPPED: $message"
    elif [ "$status" = "WARN" ]; then
        WARNINGS+=("$test_name: $message")
        echo "⚠ $test_name WARNING: $message"
    fi
    echo ""
}

# ==================================================
# Step 0: Generate minimal traces for training
# ==================================================
echo "====================================="
echo "Step 0: Generating Minimal Traces"
echo "====================================="

TRACE_DIR="$(pwd)/experiments/training_smoke_test/traces_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TRACE_DIR"

echo "Collecting traces with limit=3 for training data..."

cd external/Dream/eval_instruct

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p1_traces,trace_output_dir=$TRACE_DIR \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 3 \
    --output_path "$(pwd)/results/training_smoke_traces"

cd ../../..

# Check if traces were generated
TRACE_COUNT=$(ls -1 $TRACE_DIR/*.pt 2>/dev/null | wc -l)
if [ "$TRACE_COUNT" -ge 1 ]; then
    report_test "TraceGeneration" "PASS"
else
    report_test "TraceGeneration" "FAIL" "No trace files generated"
    # Exit early since training needs traces
    echo "Cannot continue without traces. Exiting."
    exit 1
fi

echo "Generated $TRACE_COUNT trace files in $TRACE_DIR"
echo ""

# ==================================================
# Test 1: P3 Learned Gate Training
# ==================================================
echo "====================================="
echo "Test 1: P3 Learned Gate Training"
echo "====================================="

GATE_OUTPUT="experiments/training_smoke_test/P3_gate"
mkdir -p "$GATE_OUTPUT"

echo "Step 1a: Generating oracle labels..."
python scripts/training/generate_oracle_labels.py \
    --trace_dir "$TRACE_DIR" \
    --output "$GATE_OUTPUT/oracle_labels.json" \
    --cosine_threshold 0.99

if [ $? -ne 0 ]; then
    report_test "P3_OracleLabels" "FAIL" "Oracle label generation failed"
else
    report_test "P3_OracleLabels" "PASS"
fi

echo "Step 1b: Training learned gate (3 epochs)..."
python scripts/training/train_learned_gate.py \
    --trace_dir "$TRACE_DIR" \
    --oracle_labels "$GATE_OUTPUT/oracle_labels.json" \
    --output_dir "$GATE_OUTPUT/checkpoints" \
    --num_epochs 3 \
    --batch_size 4

if [ $? -ne 0 ]; then
    report_test "P3_GateTraining" "FAIL" "Gate training failed"
else
    # Check if checkpoint was created
    if [ -f "$GATE_OUTPUT/checkpoints/learned_gate_final.pt" ]; then
        report_test "P3_GateTraining" "PASS"
    else
        report_test "P3_GateTraining" "WARN" "Training completed but no final checkpoint found"
    fi
fi

# ==================================================
# Test 2: Phase B Router Training
# ==================================================
echo "====================================="
echo "Test 2: Phase B Router Training"
echo "====================================="

PHASEB_OUTPUT="experiments/training_smoke_test/PhaseB_router"
mkdir -p "$PHASEB_OUTPUT"

echo "Training Phase B router (3 epochs)..."
python scripts/training/train_learned_router.py \
    --trace_dir "$TRACE_DIR" \
    --output_dir "$PHASEB_OUTPUT/checkpoints" \
    --num_epochs 3 \
    --batch_size 4

if [ $? -ne 0 ]; then
    report_test "PhaseB_Training" "FAIL" "Router training failed"
else
    # Check if checkpoint was created
    if [ -f "$PHASEB_OUTPUT/checkpoints/router_final.pt" ]; then
        report_test "PhaseB_Training" "PASS"
    else
        report_test "PhaseB_Training" "WARN" "Training completed but no final checkpoint found"
    fi
fi

# ==================================================
# Test 3: Phase C Continuous Router Training
# ==================================================
echo "====================================="
echo "Test 3: Phase C Continuous Router"
echo "====================================="

PHASEC_OUTPUT="experiments/training_smoke_test/PhaseC_continuous"
mkdir -p "$PHASEC_OUTPUT"

echo "Training Phase C router (3 epochs)..."
python scripts/training/train_continuous_router.py \
    --trace_dirs "$TRACE_DIR" \
    --schedule_ids 0 \
    --output_dir "$PHASEC_OUTPUT/checkpoints" \
    --num_epochs 3 \
    --batch_size 4 \
    --total_steps 256

if [ $? -ne 0 ]; then
    report_test "PhaseC_Training" "FAIL" "Continuous router training failed"
else
    # Check if checkpoint was created
    if [ -f "$PHASEC_OUTPUT/checkpoints/continuous_router_final.pt" ]; then
        report_test "PhaseC_Training" "PASS"
    else
        report_test "PhaseC_Training" "WARN" "Training completed but no final checkpoint found"
    fi
fi

# ==================================================
# Summary
# ==================================================
echo ""
echo "=================================================="
echo "TRAINING SMOKE TEST SUMMARY"
echo "=================================================="
echo "Completed at: $(date)"
echo ""

for test in "TraceGeneration" "P3_OracleLabels" "P3_GateTraining" "PhaseB_Training" "PhaseC_Training"; do
    if [[ " ${PASSED_TESTS[@]} " =~ " ${test} " ]]; then
        printf "%-25s: PASS\n" "$test"
    elif [[ " ${FAILED_TESTS[@]} " =~ " ${test} " ]]; then
        printf "%-25s: FAIL\n" "$test"
    elif [[ " ${SKIPPED_TESTS[@]} " =~ " ${test} " ]]; then
        printf "%-25s: SKIP\n" "$test"
    fi
done

# Print warnings
if [ ${#WARNINGS[@]} -gt 0 ]; then
    echo ""
    echo "Warnings:"
    for warning in "${WARNINGS[@]}"; do
        echo "  ⚠ $warning"
    done
fi

echo ""
echo "Summary: ${#PASSED_TESTS[@]} passed, ${#FAILED_TESTS[@]} failed, ${#SKIPPED_TESTS[@]} skipped, ${#WARNINGS[@]} warnings"
echo "=================================================="

# Exit with error if any tests failed
if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    exit 1
fi
