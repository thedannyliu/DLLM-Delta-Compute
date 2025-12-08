#!/bin/bash
#SBATCH --job-name=poc_reeval_fixed
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/poc_reeval_fixed_%j.out
#SBATCH --error=logs/poc_reeval_fixed_%j.err

# ==============================================================================
# Re-Evaluation with All Fixes Applied:
# 1. P2 early stop: 90% percentage-based instead of all()
# 2. SkipRouter.load: auto-convert ContinuousRouter checkpoints
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"
export WANDB_PROJECT="dllm_poc_fixed_200"

SAMPLES=200
MODEL="Dream-org/Dream-v0-Base-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/poc_reeval_fixed_${TIMESTAMP}"
TIMING_DIR="reports/timing/poc_reeval_fixed_${TIMESTAMP}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TIMING_DIR}
mkdir -p logs

echo "=============================================="
echo "Re-Evaluation with All Fixes Applied"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "Timing: ${TIMING_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "WandB: ${WANDB_PROJECT}"
echo "=============================================="
echo "Fixes applied:"
echo "  1. P2: 90% percentage-based early stop (not all())"
echo "  2. SkipRouter: auto-convert ContinuousRouter checkpoints"
echo "=============================================="

cd external/Dream/eval_instruct

# Common config
BASE_CONFIG="pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16"
COT_CONFIG="max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True"

# Function to run a phase
run_phase() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    
    echo ""
    echo "====== ${NAME} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    echo "Start: $(date)"
    
    export WANDB_RUN_GROUP="poc_reeval_fixed_${TIMESTAMP}"
    export WANDB_RUN_NAME="${NAME}"
    
    START_TIME=$(date +%s)
    
    python -m lm_eval \
        --model diffllm \
        --model_args "${BASE_CONFIG},${COT_CONFIG},delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_DIR}/${NAME}.json${EXTRA_ARGS}" \
        --gen_kwargs "do_sample=False,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot 8 \
        --batch_size 1 \
        --limit ${SAMPLES} || echo "WARNING: ${NAME} failed"
    
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    
    echo "Completed: $(date)"
    
    # Print timing summary
    if [ -f "${TIMING_DIR}/${NAME}.json" ]; then
        python3 -c "
import json
with open('${TIMING_DIR}/${NAME}.json') as f:
    d = json.load(f)
mean = d.get('mean_time_per_sample', 0)
tps = d.get('tokens_per_second', 0)
skip = d.get('skip_ratio', 0)
print(f'Timing: mean={mean:.1f}s, tok/s={tps:.1f}, skip_ratio={skip*100:.2f}%')
" 2>/dev/null || echo "Timing parse failed"
    fi
}

# ==== P0 Baseline ====
run_phase "P0_baseline" "none" "none" ""

# ==== P2 Early Stop (with fix: 90% threshold) ====
run_phase "P2_conf50_ratio90" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.5,early_stop_entropy_threshold=2.0,early_stop_ratio=0.9"
run_phase "P2_conf40_ratio90" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.4,early_stop_entropy_threshold=3.0,early_stop_ratio=0.9"
run_phase "P2_conf30_ratio80" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.3,early_stop_entropy_threshold=4.0,early_stop_ratio=0.8"

# ==== PhaseC Router (with trained checkpoint if available) ====
PHASEC_CKPT=$(ls -t experiments/PhaseC_router_cot_*/checkpoints/continuous_router_*.pt 2>/dev/null | head -1)
if [ -n "${PHASEC_CKPT}" ]; then
    echo "Using PhaseC checkpoint: ${PHASEC_CKPT}"
    run_phase "PhaseC_router" "none" "l2c_continuous" ",router_checkpoint_path=${PHASEC_CKPT}"
fi

# ==== PhaseD Skip (with SkipRouter fix) ====
if [ -n "${PHASEC_CKPT}" ]; then
    echo "Using PhaseC checkpoint for PhaseD: ${PHASEC_CKPT}"
    run_phase "PhaseD_skip" "none" "skip_continuous" ",router_checkpoint_path=${PHASEC_CKPT}"
fi

# ==== PhaseE FlexiDepth ====
PHASEE_CKPT=$(ls -t experiments/PhaseE_flexi_cot_*/checkpoints/flexi_depth_*.pt 2>/dev/null | head -1)
if [ -n "${PHASEE_CKPT}" ]; then
    echo "Using PhaseE checkpoint: ${PHASEE_CKPT}"
    run_phase "PhaseE_flexi" "none" "flexi_ffn" ",adapter_checkpoint_path=${PHASEE_CKPT}"
fi

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Re-Evaluation COMPLETE"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}"
echo "Timing: ${TIMING_DIR}"
echo "Completed: $(date)"
