#!/bin/bash
#SBATCH --job-name=train_phaseD
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/train_phaseD_%j.out
#SBATCH --error=logs/train_phaseD_%j.err

# ==============================================================================
# Phase D Skip Router Training
# ==============================================================================
# Trains a skip router (Phase D) by fine-tuning from Phase C continuous router.
# The router learns to skip FFN layers entirely vs recompute.
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
PHASE_C_CKPT="experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"
OUTPUT_DIR="experiments/PhaseD_skip"
TRACES_DIR="experiments/P1_traces"
WANDB_PROJECT="dllm_delta_compute"

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs

echo "=============================================="
echo "Phase D Skip Router Training"
echo "=============================================="
echo "Phase C checkpoint: ${PHASE_C_CKPT}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Train skip router
python scripts/training/train_skip_router.py \
    --phase_c_checkpoint ${PHASE_C_CKPT} \
    --traces_dir ${TRACES_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --skip_weight 0.1 \
    --target_skip_ratio 0.25 \
    --use_wandb \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "phaseD_skip_s256_gsm8k"

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}/checkpoints/"
