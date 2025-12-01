#!/bin/bash
#SBATCH --job-name=eval_phase_a100_h200_h100B_100
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=2:00:00
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Phase B: Learned Router - 500 Sample Evaluation
# Requires: Pre-trained router checkpoint
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "Phase B: Learned Router Evaluation"
echo "====================================="
echo "Samples: 100"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
TIMING_LOG="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/PhaseB_100_${RUN_ID}.json"

# Check for router checkpoint
ROUTER_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/PhaseB_router/checkpoints/router_final.pt"

if [ ! -f "$ROUTER_CKPT" ]; then
    echo "ERROR: Router checkpoint not found at $ROUTER_CKPT"
    echo "Phase B requires trained router. Skipping evaluation."
    exit 1
fi

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_learned,router_checkpoint=${ROUTER_CKPT},timing_log_path=${TIMING_LOG}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 100 \
    --output_path results/PhaseB_100_${RUN_ID}

echo ""
echo "Completed: $(date)"
