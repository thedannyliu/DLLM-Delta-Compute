#!/bin/bash
#SBATCH --job-name=p2_fix_test
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=00:30:00
#SBATCH --output=logs/p2_fix_test_%j.out
#SBATCH --error=logs/p2_fix_test_%j.err

###############################################################################
# Minimal test for P2 early_stop fix
# Tests with 1 sample to verify the IndexError is fixed
###############################################################################

# Source bashrc first (before set -u to avoid unbound variable issues)
source ~/.bashrc
conda activate dcllm

set -eo pipefail

PROJECT_ROOT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute"
DREAM_ROOT="${PROJECT_ROOT}/external/Dream"
cd "${PROJECT_ROOT}"

export PYTHONPATH="${PROJECT_ROOT}:${DREAM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

mkdir -p logs

cd "${DREAM_ROOT}"

echo "=== Testing P2 early_stop fix (1 sample) ==="
python -m eval_instruct.lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,dtype=bfloat16,max_length=2048,diffusion_steps=128,delta_mode=p2_early_stop,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1 \
    --tasks gsm8k \
    --batch_size 1 \
    --num_fewshot 8 \
    --limit 1 \
    --output_path "${PROJECT_ROOT}/reports/timing/p2_fix_test"

echo ""
echo "=== P2 FIX TEST PASSED ==="
echo "The IndexError has been fixed!"
