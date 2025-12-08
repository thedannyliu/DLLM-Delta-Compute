#!/bin/bash
#SBATCH --job-name=dream_official_test
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=2:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/dream_official_%j.out
#SBATCH --error=logs/dream_official_%j.err

# ==============================================================================
# Official Dream Wrapper Test (from eval/eval.py)
# ==============================================================================
# Uses the official `dream` model wrapper to verify baseline accuracy.
# This is the exact configuration from the official Dream repo.
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

SAMPLES=${1:-50}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/dream_official_${TIMESTAMP}"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=============================================="
echo "Official Dream Wrapper Test"
echo "=============================================="
echo "Using: external/Dream/eval/eval.py (--model dream)"
echo "Samples: ${SAMPLES}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Use the official eval directory with the 'dream' model
cd external/Dream/eval

# GSM8K CoT with official settings
# - Model: Dream-v0-Instruct-7B (officially base, but we use instruct for higher accuracy)
# - temperature=0 (greedy for gsm8k_cot)
# - diffusion_steps=256, max_new_tokens=256
# - 8-shot

echo ""
echo "====== Test 1: Instruct Model (our target) ======"
python eval.py --model dream \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=true" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --log_samples \
    --output_path "../../../${OUTPUT_DIR}/instruct"

echo ""
echo "====== Test 2: Base Model (official config) ======"
python eval.py --model dream \
    --model_args "pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=true" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --log_samples \
    --output_path "../../../${OUTPUT_DIR}/base"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Official Eval Complete"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}"
