#!/bin/bash
#SBATCH --job-name=train_P3
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

echo "Generating Oracle Labels..."
python scripts/training/generate_oracle_labels.py \
    --trace_dir experiments/P1_traces/traces \
    --output experiments/P3_learned_gate/oracle_labels.json \
    --cosine_threshold 0.99

echo "Training Learned Gate..."
python scripts/training/train_learned_gate.py \
    --trace_dir experiments/P1_traces/traces \
    --oracle_labels experiments/P3_learned_gate/oracle_labels.json \
    --output_dir experiments/P3_learned_gate/checkpoints \
    --num_epochs 5
