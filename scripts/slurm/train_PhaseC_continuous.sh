#!/bin/bash
#SBATCH --job-name=train_PhaseC
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
source ~/.bashrc
conda activate dcllm

echo "Training Phase C Continuous Router..."
python scripts/training/train_continuous_router.py \
    --trace_dir experiments/P1_traces/traces \
    --output_dir experiments/PhaseC_continuous/checkpoints \
    --num_epochs 50
