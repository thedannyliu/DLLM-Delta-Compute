#!/bin/bash
#SBATCH --job-name=fair_poc_eval
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/fair_poc_%j.out
#SBATCH --error=logs/fair_poc_%j.err

# ==============================================================================
# Fair PoC Evaluation Script
# ==============================================================================
# This script runs ALL phases with CONSISTENT settings:
# - num_fewshot: 8 (same for all phases)
# - samples: 500 (statistically meaningful)
# - diffusion_steps: 128 (consistent across all)
# - Reports BOTH strict-match and flexible-extract metrics
#
# Fixes from previous evaluation:
# 1. P2 delta_mode typo: "p2_earlystop" -> "p2_early_stop"
# 2. Consistent num_fewshot across all phases
# 3. Consistent diffusion_steps
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=500
BATCH_SIZE=1
DIFFUSION_STEPS=128
NUM_FEWSHOT=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
RESULTS_BASE="external/Dream/eval_instruct/results/fair_eval_${TIMESTAMP}"

mkdir -p ${RESULTS_BASE}
mkdir -p reports/timing/fair_eval_${TIMESTAMP}

echo "=============================================="
echo "Fair PoC Evaluation"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Samples: ${SAMPLES}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Diffusion Steps: ${DIFFUSION_STEPS}"
echo "Num Fewshot: ${NUM_FEWSHOT}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation
run_eval() {
    local PHASE=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local TIMING_LOG="${CKPT_DIR}/../reports/timing/fair_eval_${TIMESTAMP}/${PHASE}.json"
    local RESULTS_DIR="${RESULTS_BASE}/${PHASE}"
    
    echo ""
    echo "====== ${PHASE} ======"
    echo "delta_mode: ${DELTA_MODE}"
    echo "cache_mode: ${CACHE_MODE}"
    echo "extra_args: ${EXTRA_ARGS}"
    echo "Start: $(date)"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_LOG}${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},do_sample=False,temperature=0.0,alg=entropy,alg_temp=0.1" \
        --tasks gsm8k \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size ${BATCH_SIZE} \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}
    
    local EXIT_CODE=$?
    echo "Completed: $(date)"
    echo "Exit code: ${EXIT_CODE}"
    return ${EXIT_CODE}
}

# ==============================================================================
# P0: Baseline (No acceleration)
# ==============================================================================
run_eval "P0_baseline" "none" "none" ""

# ==============================================================================
# P1: Teacher Traces (same as P0, just collecting traces)
# Note: For fair comparison, P1 should have same performance as P0
# ==============================================================================
run_eval "P1_traces" "p1_traces" "none" ",trace_output_dir=${CKPT_DIR}/P1_traces/fair_eval_${TIMESTAMP}"

# ==============================================================================
# P2: Rule-Based Early Stop
# FIXED: Using correct delta_mode "p2_early_stop" (with underscore)
# ==============================================================================
run_eval "P2_early_stop" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1"

# ==============================================================================
# P3: Learned Gate
# ==============================================================================
if [ -f "${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt" ]; then
    run_eval "P3_learned_gate" "p3_learned_gate" "none" ",gate_checkpoint=${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt"
else
    echo "WARNING: P3 gate checkpoint not found, skipping P3"
fi

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
run_eval "P4_adaptive" "p4_adaptive" "none" ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10"

# ==============================================================================
# Phase A: Heuristic FFN Caching
# Uses every-other-step caching schedule
# ==============================================================================
run_eval "PhaseA_heuristic" "none" "l2c_ffn" ",cache_schedule=1;3;5;7;9;11;13;15;17;19;21;23;25;27;29;31;33;35;37;39;41;43;45;47;49;51;53;55;57;59;61;63;65;67;69;71;73;75;77;79;81;83;85;87;89;91;93;95;97;99;101;103;105;107;109;111;113;115;117;119;121;123;125;127"

# ==============================================================================
# Phase B: Learned Router (Fixed Schedule)
# ==============================================================================
if [ -f "${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt" ]; then
    run_eval "PhaseB_router" "none" "l2c_learned" ",router_checkpoint=${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt"
else
    echo "WARNING: Phase B router checkpoint not found, skipping Phase B"
fi

# ==============================================================================
# Phase C: Continuous Router
# ==============================================================================
if [ -f "${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt" ]; then
    run_eval "PhaseC_continuous" "none" "l2c_continuous" ",router_checkpoint=${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt"
else
    echo "WARNING: Phase C router checkpoint not found, skipping Phase C"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "Fair PoC Evaluation Complete"
echo "=============================================="
echo "Results saved to: ${RESULTS_BASE}"
echo "Timing logs saved to: reports/timing/fair_eval_${TIMESTAMP}/"
echo "Completed: $(date)"
