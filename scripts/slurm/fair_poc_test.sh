#!/bin/bash
#SBATCH --job-name=fair_poc_test
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/fair_poc_test_%j.out
#SBATCH --error=logs/fair_poc_test_%j.err

# ==============================================================================
# Minimal Test for Fair PoC Evaluation (1 sample per phase)
# ==============================================================================
# Verifies all phases work correctly before running full evaluation
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration - minimal test
SAMPLES=1
BATCH_SIZE=1
DIFFUSION_STEPS=128
NUM_FEWSHOT=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
RESULTS_BASE="external/Dream/eval_instruct/results/fair_test_${TIMESTAMP}"

mkdir -p ${RESULTS_BASE}
mkdir -p logs

echo "=============================================="
echo "Fair PoC Minimal Test (1 sample per phase)"
echo "=============================================="
echo "Testing all phases with consistent settings:"
echo "- num_fewshot: ${NUM_FEWSHOT}"
echo "- diffusion_steps: ${DIFFUSION_STEPS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

PASS_COUNT=0
FAIL_COUNT=0
declare -A RESULTS

# Function to run minimal test
test_phase() {
    local PHASE=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    
    echo ""
    echo "====== Testing ${PHASE} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    
    local OUTPUT=$(python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE}${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},do_sample=False,temperature=0.0,alg=entropy,alg_temp=0.1" \
        --tasks gsm8k \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size ${BATCH_SIZE} \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_BASE}/${PHASE} 2>&1)
    
    local EXIT_CODE=$?
    
    if [ ${EXIT_CODE} -eq 0 ]; then
        echo "✓ ${PHASE}: PASS"
        RESULTS[${PHASE}]="PASS"
        ((PASS_COUNT++))
    else
        echo "✗ ${PHASE}: FAIL (exit code: ${EXIT_CODE})"
        echo "Error output: ${OUTPUT}" | tail -20
        RESULTS[${PHASE}]="FAIL"
        ((FAIL_COUNT++))
    fi
    
    return ${EXIT_CODE}
}

# Test all phases
echo ""
echo "Starting minimal tests..."

# P0: Baseline
test_phase "P0_baseline" "none" "none" "" || true

# P1: Teacher Traces
test_phase "P1_traces" "p1_traces" "none" ",trace_output_dir=${CKPT_DIR}/P1_traces/fair_test_${TIMESTAMP}" || true

# P2: Early Stop (with corrected delta_mode)
test_phase "P2_early_stop" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1" || true

# P3: Learned Gate
if [ -f "${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt" ]; then
    test_phase "P3_learned_gate" "p3_learned_gate" "none" ",gate_checkpoint=${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt" || true
else
    echo "⚠ P3: SKIP (checkpoint not found)"
    RESULTS["P3_learned_gate"]="SKIP"
fi

# P4: Adaptive
test_phase "P4_adaptive" "p4_adaptive" "none" ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10" || true

# Phase A: Heuristic Caching
test_phase "PhaseA_heuristic" "none" "l2c_ffn" ",cache_schedule=1;3;5;7;9;11;13;15;17;19;21;23;25;27;29;31;33;35;37;39;41;43;45;47;49;51;53;55;57;59;61;63;65;67;69;71;73;75;77;79;81;83;85;87;89;91;93;95;97;99;101;103;105;107;109;111;113;115;117;119;121;123;125;127" || true

# Phase B: Learned Router
if [ -f "${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt" ]; then
    test_phase "PhaseB_router" "none" "l2c_learned" ",router_checkpoint=${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt" || true
else
    echo "⚠ PhaseB: SKIP (checkpoint not found)"
    RESULTS["PhaseB_router"]="SKIP"
fi

# Phase C: Continuous Router
if [ -f "${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt" ]; then
    test_phase "PhaseC_continuous" "none" "l2c_continuous" ",router_checkpoint=${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt" || true
else
    echo "⚠ PhaseC: SKIP (checkpoint not found)"
    RESULTS["PhaseC_continuous"]="SKIP"
fi

# Summary
echo ""
echo "=============================================="
echo "Minimal Test Summary"
echo "=============================================="
echo "Passed: ${PASS_COUNT}"
echo "Failed: ${FAIL_COUNT}"
echo ""
echo "Phase Results:"
for phase in P0_baseline P1_traces P2_early_stop P3_learned_gate P4_adaptive PhaseA_heuristic PhaseB_router PhaseC_continuous; do
    echo "  ${phase}: ${RESULTS[$phase]:-N/A}"
done
echo "=============================================="

if [ ${FAIL_COUNT} -gt 0 ]; then
    echo "WARNING: Some phases failed. Check logs before running full evaluation."
    exit 1
else
    echo "All phases passed! Ready for full evaluation."
    exit 0
fi
