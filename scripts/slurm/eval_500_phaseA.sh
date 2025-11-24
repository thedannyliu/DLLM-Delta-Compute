#!/bin/bash
#SBATCH --job-name=eval_PhaseA_500
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Phase A: Heuristic Caching - 500 Sample Evaluation
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "Phase A: Heuristic Caching Evaluation"
echo "====================================="
echo "Samples: 500"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

# Define cache schedule - comma-separated layer indices to cache
# For 32 layers, cache every other layer: 1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31
CACHE_SCHEDULE="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31"

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_ffn,cache_schedule=${CACHE_SCHEDULE} \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 500 \
    --output_path results/PhaseA_500_$(date +%Y%m%d_%H%M%S)

echo ""
echo "Completed: $(date)"
