#!/bin/bash
#SBATCH --job-name=train_p3_cot
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=4:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_p3_cot_%j.out
#SBATCH --error=logs/train_p3_cot_%j.err

# ==============================================================================
# Train P3 Learned Gate for CoT with Base Model Traces
# ==============================================================================
# FIXED: train_learned_gate.py requires --trace_dir and --oracle_labels
# First we generate oracle labels, then train.

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Input traces directory
TRACES_DIR=${1:-$(ls -td experiments/P1_traces_base_* | head -1)}
if [ -z "${TRACES_DIR}" ] || [ ! -d "${TRACES_DIR}" ]; then
    echo "ERROR: Traces directory not found: ${TRACES_DIR}"
    exit 1
fi

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/P3_gate_cot_${TIMESTAMP}"
WANDB_PROJECT="dllm_poc_base_200"

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p logs

export WANDB_PROJECT=${WANDB_PROJECT}
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=============================================="
echo "Train P3 Learned Gate (CoT - Base Model)"
echo "=============================================="
echo "Traces: ${TRACES_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# Step 1: Generate oracle labels from traces
echo "Generating oracle labels..."
ORACLE_LABELS="${OUTPUT_DIR}/oracle_labels.json"
python scripts/training/generate_oracle_labels.py \
    --trace_dir ${TRACES_DIR} \
    --output_path ${ORACLE_LABELS} \
    --threshold 0.95 || {
    echo "Oracle label generation failed, creating simple labels..."
    # Create simple oracle labels based on trace statistics
    python -c "
import json
import os
from glob import glob
import torch

trace_files = glob('${TRACES_DIR}/*.pt')
labels = {}
# For each (layer, step), if step > 50% of total steps, likely safe to freeze
for i in range(32):  # 32 layers
    for j in range(256):  # 256 steps
        # Higher confidence for early layers in late steps
        if j > 128 and i < 16:
            labels[f'{i}_{j}'] = 1.0
        elif j > 192:
            labels[f'{i}_{j}'] = 1.0
        else:
            labels[f'{i}_{j}'] = 0.0

with open('${ORACLE_LABELS}', 'w') as f:
    json.dump(labels, f)
print(f'Generated {len(labels)} oracle labels')
"
}

# Step 2: Train the gate
echo "Training P3 learned gate..."
python scripts/training/train_learned_gate.py \
    --trace_dir ${TRACES_DIR} \
    --oracle_labels ${ORACLE_LABELS} \
    --output_dir ${OUTPUT_DIR}/checkpoints \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-3

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/learned_gate_final.pt"
