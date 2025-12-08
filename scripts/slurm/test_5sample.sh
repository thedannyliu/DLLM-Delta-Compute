#!/bin/bash
#SBATCH --job-name=test_5sample
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=2:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/test_5sample_%j.out
#SBATCH --error=logs/test_5sample_%j.err

# ==============================================================================
# Minimal 5-Sample Test with WandB Logging
# ==============================================================================
# Runs P0, P2, P3, PhaseC with 5 samples to verify:
# 1. Metrics (latency, skip_ratio) are logged correctly
# 2. WandB integration works
# 3. Early stopping is functioning
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=5
DIFFUSION_STEPS=256
MAX_NEW_TOKENS=256
NUM_FEWSHOT=8
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/test_5sample_${TIMESTAMP}"

# Correct checkpoint paths
GATE_CKPT="experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
ROUTER_C_CKPT="experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

# WandB settings
export WANDB_PROJECT="dllm_delta_compute_eval"
export WANDB_MODE="online"

mkdir -p ${OUTPUT_DIR}
mkdir -p reports/timing/test_5sample_${TIMESTAMP}
mkdir -p logs

echo "=============================================="
echo "Minimal 5-Sample Test"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "WandB Project: ${WANDB_PROJECT}"
echo "=============================================="

cd external/Dream/eval_instruct

# P0: Baseline
echo ""
echo "====== P0 Baseline ======"
TIMING_LOG="${OUTPUT_DIR}/../../../reports/timing/test_5sample_${TIMESTAMP}/P0_baseline.json"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=none,cache_mode=none,add_bos_token=True,timing_log_path=${TIMING_LOG}" \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path "${OUTPUT_DIR}/P0_baseline"

echo "P0 timing log:"
cat ${TIMING_LOG} 2>/dev/null || echo "No timing log found"

# P2: Early Stop
echo ""
echo "====== P2 Early Stop ======"
TIMING_LOG="${OUTPUT_DIR}/../../../reports/timing/test_5sample_${TIMESTAMP}/P2_early_stop.json"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=p2_early_stop,cache_mode=none,add_bos_token=True,timing_log_path=${TIMING_LOG},early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1" \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path "${OUTPUT_DIR}/P2_early_stop"

echo "P2 timing log:"
cat ${TIMING_LOG} 2>/dev/null || echo "No timing log found"

# P3: Learned Gate
echo ""
echo "====== P3 Learned Gate ======"
TIMING_LOG="${OUTPUT_DIR}/../../../reports/timing/test_5sample_${TIMESTAMP}/P3_learned_gate.json"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=p3_learned_gate,cache_mode=none,add_bos_token=True,timing_log_path=${TIMING_LOG},gate_checkpoint=${GATE_CKPT},gate_threshold=0.5" \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path "${OUTPUT_DIR}/P3_learned_gate" || echo "P3 failed"

echo "P3 timing log:"
cat ${TIMING_LOG} 2>/dev/null || echo "No timing log found"

# PhaseC: Continuous Router
echo ""
echo "====== PhaseC Continuous ======"
TIMING_LOG="${OUTPUT_DIR}/../../../reports/timing/test_5sample_${TIMESTAMP}/PhaseC_continuous.json"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=none,cache_mode=l2c_continuous,add_bos_token=True,timing_log_path=${TIMING_LOG},router_checkpoint=${ROUTER_C_CKPT}" \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path "${OUTPUT_DIR}/PhaseC_continuous" || echo "PhaseC failed"

echo "PhaseC timing log:"
cat ${TIMING_LOG} 2>/dev/null || echo "No timing log found"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "5-Sample Test Complete"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}"
echo "Timing logs: reports/timing/test_5sample_${TIMESTAMP}/"
