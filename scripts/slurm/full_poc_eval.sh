#!/bin/bash
#SBATCH --job-name=full_poc_eval
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/full_poc_eval_%j.out
#SBATCH --error=logs/full_poc_eval_%j.err

# ==============================================================================
# Full PoC Evaluation - All Phases (P0-P4, Phase A-E)
# ==============================================================================
# Runs comprehensive evaluation of all PoC phases with CoT settings:
# - Model: Dream-org/Dream-v0-Instruct-7B
# - Task: gsm8k_cot
# - Settings: 8-shot, 256 diffusion steps, greedy decoding
# - Samples: 1000
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=1000
BATCH_SIZE=1
DIFFUSION_STEPS=256
NUM_FEWSHOT=8
MAX_NEW_TOKENS=256
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
RESULTS_BASE="experiments/full_eval_${TIMESTAMP}"

mkdir -p ${RESULTS_BASE}
mkdir -p reports/timing/full_eval_${TIMESTAMP}

echo "=============================================="
echo "Full PoC Evaluation - All Phases"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Samples: ${SAMPLES}"
echo "Diffusion Steps: ${DIFFUSION_STEPS}"
echo "Max New Tokens: ${MAX_NEW_TOKENS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation
run_eval() {
    local PHASE=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local TIMING_LOG="${CKPT_DIR}/../reports/timing/full_eval_${TIMESTAMP}/${PHASE}.json"
    local RESULTS_DIR="${RESULTS_BASE}/${PHASE}"
    
    echo ""
    echo "====== ${PHASE} ======"
    echo "delta_mode: ${DELTA_MODE}"
    echo "cache_mode: ${CACHE_MODE}"
    echo "Start: $(date)"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_LOG},add_bos_token=True${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size ${BATCH_SIZE} \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}
    
    echo "Completed: $(date)"
}

# ==============================================================================
# P0: Baseline
# ==============================================================================
run_eval "P0_baseline" "none" "none" ""

# ==============================================================================
# P1: Teacher Traces (for training data collection)
# ==============================================================================
run_eval "P1_traces" "p1_traces" "none" ",trace_output_dir=${CKPT_DIR}/P1_traces/full_eval_${TIMESTAMP}"

# ==============================================================================
# P2: Rule-Based Early Stop
# ==============================================================================
run_eval "P2_early_stop" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1"

# ==============================================================================
# P3: Learned Gate
# ==============================================================================
if [ -f "${CKPT_DIR}/P3_learned_gate/checkpoints/gate_final.pt" ]; then
    run_eval "P3_learned_gate" "p3_learned_gate" "none" ",gate_checkpoint=${CKPT_DIR}/P3_learned_gate/checkpoints/gate_final.pt"
else
    echo "WARNING: P3 gate checkpoint not found, skipping P3"
fi

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
run_eval "P4_adaptive" "p4_adaptive" "none" ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10"

# ==============================================================================
# Phase A: Heuristic FFN Caching
# ==============================================================================
# Cache every other layer on every other step
CACHE_LAYERS=$(seq -s ';' 1 2 63)
run_eval "PhaseA_heuristic" "none" "l2c_ffn" ",cache_schedule=${CACHE_LAYERS}"

# ==============================================================================
# Phase B: Learned Router
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
# Phase D: Continuous Layer Skipping
# ==============================================================================
if [ -f "${CKPT_DIR}/PhaseD_skip/checkpoints/skip_router_final.pt" ]; then
    run_eval "PhaseD_skip" "none" "skip_continuous" ",router_checkpoint=${CKPT_DIR}/PhaseD_skip/checkpoints/skip_router_final.pt"
else
    echo "WARNING: Phase D skip router checkpoint not found, skipping Phase D"
fi

# ==============================================================================
# Phase E: FlexiDepth Router+Adapter
# ==============================================================================
if [ -f "${CKPT_DIR}/PhaseE_flexi/checkpoints/flexi_depth_final.pt" ]; then
    run_eval "PhaseE_flexi" "none" "flexi_ffn" ",router_checkpoint=${CKPT_DIR}/PhaseE_flexi/checkpoints/flexi_depth_final.pt"
else
    echo "WARNING: Phase E FlexiDepth checkpoint not found, skipping Phase E"
fi

# ==============================================================================
# Combined: P4 + Phase C
# ==============================================================================
if [ -f "${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt" ]; then
    run_eval "P4_PhaseC_combined" "p4_adaptive" "l2c_continuous" ",router_checkpoint=${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt,adaptive_max_stride=4"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "Full PoC Evaluation Complete"
echo "=============================================="
echo "Results saved to: ${RESULTS_BASE}"
echo "Timing logs saved to: reports/timing/full_eval_${TIMESTAMP}/"
echo "Completed: $(date)"
