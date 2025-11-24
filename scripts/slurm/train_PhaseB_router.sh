#!/bin/bash
#SBATCH --job-name=train_PhaseB
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Find the latest traces directory with 500 samples
TRACE_DIR=$(ls -td experiments/P1_traces/traces_500_* 2>/dev/null | head -1)

if [ -z "$TRACE_DIR" ] || [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: No P1 traces_500 directory found!"
    exit 1
fi

echo "Using traces from: $TRACE_DIR"
TRACE_COUNT=$(ls -1 "$TRACE_DIR"/*.pt 2>/dev/null | wc -l)
echo "Found $TRACE_COUNT trace files"

echo "Training Phase B Router..."
python scripts/training/train_learned_router.py \
    --trace_dir "$TRACE_DIR" \
    --output_dir experiments/PhaseB_router/checkpoints \
    --num_epochs 5
