#!/bin/bash
#SBATCH -Jdream_test           # Job name
#SBATCH -N1 --gres=gpu:H100:1  # 1 node, 1 H100 GPU
#SBATCH --mem-per-gpu=40G      # Memory per GPU
#SBATCH -t0-02:00:00           # 2 hours time limit
#SBATCH -otest_poc_%j.out      # Standard output log
#SBATCH -qinfinity             # Queue name
#SBATCH -A GT-hl94             # Account

# Load modules
module load cuda/12.1

# Activate conda environment
source ~/.bashrc
conda activate dcllm

# Run test
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
python test_poc_v1a.py \
    --model_path hkust-nlp/Dream-7B \
    --output_dir ./test_traces \
    --device cuda

echo "Test completed at $(date)"
