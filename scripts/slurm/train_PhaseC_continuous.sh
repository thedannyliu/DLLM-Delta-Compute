#!/bin/bash
#SBATCH --job-name=train_PhaseC
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

echo "Training Phase C Continuous Router..."
python scripts/training/train_continuous_router.py \
    --trace_dir experiments/P1_traces/traces \
    --output_dir experiments/PhaseC_continuous/checkpoints \
    --num_epochs 5
