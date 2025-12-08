#!/bin/bash
#SBATCH --job-name=train_phaseE
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/train_phaseE_%j.out
#SBATCH --error=logs/train_phaseE_%j.err

# ==============================================================================
# Phase E FlexiDepth Router+Adapter Training
# ==============================================================================
# Trains FlexiDepth-style router + adapter (Phase E).
# Backbone is frozen; only router and adapter parameters are trained.
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
MODEL="Dream-org/Dream-v0-Instruct-7B"
OUTPUT_DIR="experiments/PhaseE_flexi"
TRACES_DIR="experiments/P1_traces"
WANDB_PROJECT="dllm_delta_compute"

# Adapt top 8 layers only (layers 24-31 for 32-layer model)
ADAPTED_LAYERS="24,25,26,27,28,29,30,31"

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p ${OUTPUT_DIR}/logs

echo "=============================================="
echo "Phase E FlexiDepth Training"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Adapted layers: ${ADAPTED_LAYERS}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Train FlexiDepth
python scripts/training/train_flexi_adapter.py \
    --model ${MODEL} \
    --traces_dir ${TRACES_DIR} \
    --output_dir ${OUTPUT_DIR} \
    --adapted_layers ${ADAPTED_LAYERS} \
    --adapter_dim 256 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --skip_weight 0.1 \
    --target_skip_ratio 0.30 \
    --use_wandb \
    --wandb_project ${WANDB_PROJECT} \
    --wandb_run_name "phaseE_flexi_s256_gsm8k"

echo "Training complete. Checkpoint saved to ${OUTPUT_DIR}/checkpoints/"
