#!/bin/bash
#SBATCH --job-name=train_phaseD_cot
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=8:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/train_phaseD_cot_%j.out
#SBATCH --error=logs/train_phaseD_cot_%j.err

# ==============================================================================
# Train PhaseD Skip Router for CoT with Base Model Traces
# ==============================================================================
# PhaseD reuses the continuous router architecture but with skip semantics
# Uses the same training script as PhaseC

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
OUTPUT_DIR="experiments/PhaseD_skip_cot_${TIMESTAMP}"
WANDB_PROJECT="dllm_poc_base_200"

mkdir -p ${OUTPUT_DIR}/checkpoints
mkdir -p logs

export WANDB_PROJECT=${WANDB_PROJECT}
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

echo "=============================================="
echo "Train PhaseD Skip Router (CoT - Base Model)"
echo "=============================================="
echo "Traces: ${TRACES_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

# PhaseD uses the same training as PhaseC (continuous router)
# The skip semantics are applied at inference time
python scripts/training/train_continuous_router.py \
    --trace_dirs ${TRACES_DIR} \
    --schedule_ids 0 \
    --output_dir ${OUTPUT_DIR}/checkpoints \
    --num_epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --total_steps 256

echo ""
echo "=============================================="
echo "Training Complete"
echo "=============================================="
echo "Checkpoint: ${OUTPUT_DIR}/checkpoints/continuous_router_final.pt"
