#!/bin/bash
#SBATCH --job-name=poc_official
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/poc_official_%j.out
#SBATCH --error=logs/poc_official_%j.err

# ==============================================================================
# Official Configuration PoC Evaluation
# ==============================================================================
# ALIGNED WITH OFFICIAL DREAM EVAL:
# - max_new_tokens=256 in MODEL_ARGS
# - diffusion_steps=256 in MODEL_ARGS  
# - temperature=0.0, top_p=0.95, add_bos_token=true
# - gsm8k_cot, 8-shot, batch_size=1
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=${1:-1000}  # Default 1000, can override with arg
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="experiments/poc_official_${TIMESTAMP}"
TIMING_BASE="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/poc_official_${TIMESTAMP}"
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"

# Create directories
mkdir -p ${OUTPUT_BASE}
mkdir -p ${TIMING_BASE}
mkdir -p logs

echo "=============================================="
echo "Official Configuration PoC Evaluation"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Timestamp: ${TIMESTAMP}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Official Config:"
echo "  max_new_tokens=256 (in model_args)"
echo "  diffusion_steps=256 (in model_args)"
echo "  temperature=0.0, top_p=0.95, add_bos_token=true"
echo "  gsm8k_cot, 8-shot, batch_size=1"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation with OFFICIAL config
run_official_eval() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local TIMING_PATH="${TIMING_BASE}/${NAME}.json"
    
    echo ""
    echo "====== ${NAME} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    echo "Start: $(date)"
    
    # CRITICAL: max_new_tokens and diffusion_steps in model_args!
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_PATH}${EXTRA_ARGS}" \
        --gen_kwargs "do_sample=False,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot 8 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --log_samples \
        --output_path "${OUTPUT_DIR}" || echo "WARNING: ${NAME} failed"
    
    echo "Completed: $(date)"
}

# ==============================================================================
# P0: Baseline (no delta-compute)
# ==============================================================================
run_official_eval "P0_baseline" "none" "none" ""

# ==============================================================================
# P2: Early Stop
# ==============================================================================
# Using slightly relaxed thresholds for CoT
run_official_eval "P2_early_stop" "p2_early_stop" "none" \
    ",early_stop_confidence_threshold=0.8,early_stop_entropy_threshold=0.5"

# ==============================================================================
# P3: Learned Gate (if checkpoint exists)
# ==============================================================================
GATE_CKPT="${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt"
if [ -f "${GATE_CKPT}" ]; then
    run_official_eval "P3_learned_gate" "p3_learned_gate" "none" \
        ",gate_checkpoint=${GATE_CKPT},gate_threshold=0.5"
else
    echo "WARNING: P3 gate checkpoint not found: ${GATE_CKPT}"
fi

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
run_official_eval "P4_adaptive" "p4_adaptive" "none" \
    ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10"

# ==============================================================================
# Phase A: Heuristic FFN Caching
# ==============================================================================
CACHE_LAYERS=$(seq -s ';' 1 2 63)
run_official_eval "PhaseA_caching" "none" "l2c_ffn" ",cache_schedule=${CACHE_LAYERS}"

# ==============================================================================
# Phase B: Learned Router (if checkpoint exists)
# ==============================================================================
ROUTER_B="${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt"
if [ -f "${ROUTER_B}" ]; then
    run_official_eval "PhaseB_router" "none" "l2c_learned" \
        ",router_checkpoint=${ROUTER_B}"
else
    echo "WARNING: PhaseB router checkpoint not found: ${ROUTER_B}"
fi

# ==============================================================================
# Phase C: Continuous Router (if checkpoint exists)
# ==============================================================================
ROUTER_C="${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt"
if [ -f "${ROUTER_C}" ]; then
    run_official_eval "PhaseC_continuous" "none" "l2c_continuous" \
        ",router_checkpoint=${ROUTER_C}"
else
    echo "WARNING: PhaseC router checkpoint not found: ${ROUTER_C}"
fi

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Official PoC Evaluation Complete"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}"
echo "Timing: ${TIMING_BASE}"
echo "Completed: $(date)"
