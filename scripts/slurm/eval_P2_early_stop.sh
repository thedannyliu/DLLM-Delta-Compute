#!/bin/bash
#SBATCH -Jdream_P2              # Job name
#SBATCH -N1 --gres=gpu:a100:1   # 1 node, 1 H100 GPU
#SBATCH --mem-per-gpu=80G       # Memory per GPU
#SBATCH -t0-12:00:00            # 12 hours time limit
#SBATCH -o experiments/P2_early_stop/logs/gsm8k_early_stop_%j.out
#SBATCH -p coc-gpu              # Queue name
#SBATCH -A coc              # Account

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

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Setup
MODEL_PATH="hkust-nlp/Dream-7B"
OUTPUT_DIR="experiments/P2_early_stop/results/gsm8k_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== P2 Early Stopping Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Started at: $(date)"

# Run with early stopping enabled
# Small-scale test: 100 samples, fixed seed for reproducibility
cd external/Dream/eval_instruct

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_PATH,delta_mode=p2_early_stop,cache_mode=none,trace_teacher=False,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1 \
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
