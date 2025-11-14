#!/bin/bash
#SBATCH -Jdream_PhaseA          # Job name
#SBATCH -N1 --gres=gpu:H100:1   # 1 node, 1 H100 GPU
#SBATCH --mem-per-gpu=80G       # Memory per GPU
#SBATCH -t0-12:00:00            # 12 hours time limit
#SBATCH -o experiments/PhaseA_caching/logs/gsm8k_caching_%j.out
#SBATCH -qinfinity              # Queue name
#SBATCH -A GT-hl94              # Account

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Setup
MODEL_PATH="hkust-nlp/Dream-7B"
OUTPUT_DIR="experiments/PhaseA_caching/results/gsm8k_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo "=== Phase A FFN Caching Evaluation ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Started at: $(date)"

# Run with FFN caching enabled (layers 0-3)
cd external/Dream/eval_instruct

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_PATH,delta_mode=none,cache_mode=l2c_ffn,trace_teacher=False,cache_schedule=0,1,2,3 \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path $OUTPUT_DIR/results.json \
    --log_samples \
    2>&1 | tee $OUTPUT_DIR/eval.log

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
