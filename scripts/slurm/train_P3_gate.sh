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

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
source ~/.bashrc
conda activate dcllm

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
