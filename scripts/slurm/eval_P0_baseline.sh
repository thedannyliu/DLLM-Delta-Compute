#!/bin/bash
#SBATCH -Jdream_P0              # Job name
#SBATCH -N1 --gres=gpu:h100:1   # 1 node, 1 H100 GPU
#SBATCH --mem-per-gpu=40G       # Memory per GPU
#SBATCH -t0-08:00:00            # 8 hours time limit
#SBATCH -o experiments/P0_baseline/logs/gsm8k_baseline_%j.out
#SBATCH -p ice-gpu              # Queue name

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Install missing dependencies if needed
pip install -q sacrebleu evaluate scikit-learn sqlitedict word2number 2>&1 | grep -v "Requirement already satisfied" || true

# Setup
MODEL_PATH="Dream-org/Dream-v0-Instruct-7B"
OUTPUT_DIR="experiments/P0_baseline/results/gsm8k_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== P0 Baseline Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Started at: $(date)"

# Run baseline evaluation (no acceleration)
# Small-scale test: 100 samples, fixed seed for reproducibility
cd external/Dream/eval_instruct

# Create output directory before tee
mkdir -p "../../../../$OUTPUT_DIR"

python -m lm_eval --model diffllm \
  --model_args pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=False \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 100 \
    --seed 42 \
    --output_path ../../../../$OUTPUT_DIR/results.json \
    --log_samples \
    2>&1 | tee ../../../../$OUTPUT_DIR/eval.log

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
