#!/bin/bash
#SBATCH --job-name=poc_base_200
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/poc_base_200_%j.out
#SBATCH --error=logs/poc_base_200_%j.err

# ==============================================================================
# Full PoC Evaluation with Base Model - 200 Samples
# ==============================================================================
# Official Configuration (aligned with Dream eval):
# - Model: Dream-org/Dream-v0-Base-7B
# - max_new_tokens=256, diffusion_steps=256
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
SAMPLES=200
MODEL="Dream-org/Dream-v0-Base-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="experiments/poc_base_${SAMPLES}_${TIMESTAMP}"
TIMING_BASE="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/poc_base_${SAMPLES}_${TIMESTAMP}"
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
WANDB_PROJECT="dllm_poc_base_200"

# Create directories
mkdir -p ${OUTPUT_BASE}
mkdir -p ${TIMING_BASE}
mkdir -p logs

# Export for W&B
export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_RUN_GROUP="poc_base_${TIMESTAMP}"

echo "=============================================="
echo "Full PoC Evaluation - Base Model - ${SAMPLES} Samples"
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
run_base_eval() {
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
    
    # Print timing summary
    if [ -f "${TIMING_PATH}" ]; then
        echo "Timing: $(cat ${TIMING_PATH} | python -c 'import sys,json; d=json.load(sys.stdin); print(f"mean={d.get(\"mean_time_per_sample\",0):.2f}s, tok/s={d.get(\"tokens_per_second\",0):.1f}")')"
    fi
}

# ==============================================================================
# P0: Baseline (no delta-compute) - REFERENCE
# ==============================================================================
run_base_eval "P0_baseline" "none" "none" ""

# ==============================================================================
# P2: Early Stop - Try different thresholds for CoT
# ==============================================================================
# Test relaxed thresholds since CoT has different entropy patterns
run_base_eval "P2_early_stop_conf80" "p2_early_stop" "none" \
    ",early_stop_confidence_threshold=0.8,early_stop_entropy_threshold=0.5"

run_base_eval "P2_early_stop_conf70" "p2_early_stop" "none" \
    ",early_stop_confidence_threshold=0.7,early_stop_entropy_threshold=0.8"

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
run_base_eval "P4_adaptive" "p4_adaptive" "none" \
    ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10"

# ==============================================================================
# Phase A: Heuristic FFN Caching (odd layers)
# ==============================================================================
CACHE_LAYERS=$(seq -s ';' 1 2 63)
run_base_eval "PhaseA_caching" "none" "l2c_ffn" ",cache_schedule=${CACHE_LAYERS}"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "PoC Evaluation Complete (Basic Phases)"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}"
echo "Timing: ${TIMING_BASE}"
echo "Completed: $(date)"
echo ""
echo "Next: Check results and if P0 ~65%, proceed with:"
echo "  1. P1 trace collection"
echo "  2. P3/PhaseB/C training"
echo "  3. Full 1500-sample eval"
