#!/bin/bash
#SBATCH --job-name=eval_phase_a100C_100
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:A100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Phase C: Continuous Router - 500 Sample Evaluation
# Requires: Pre-trained continuous router checkpoint
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "Phase C: Continuous Router Evaluation"
echo "====================================="
echo "Samples: 100"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
TIMING_LOG="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/PhaseC_100_${RUN_ID}.json"

# Check for continuous router checkpoint
ROUTER_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

if [ ! -f "$ROUTER_CKPT" ]; then
    echo "ERROR: Continuous router checkpoint not found at $ROUTER_CKPT"
    echo "Phase C requires trained continuous router. Skipping evaluation."
    exit 1
fi

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_continuous,router_checkpoint=${ROUTER_CKPT},timing_log_path=${TIMING_LOG}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 100 \
    --output_path results/PhaseC_100_${RUN_ID}

echo ""
echo "Completed: $(date)"
