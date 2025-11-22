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

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
source ~/.bashrc
conda activate dcllm

echo "Training Phase B Router..."
python scripts/training/train_learned_router.py \
    --trace_dir experiments/P1_traces/traces \
    --output_dir experiments/PhaseB_router/checkpoints \
    --num_epochs 5
