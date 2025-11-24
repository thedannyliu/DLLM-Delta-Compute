#!/bin/bash
#SBATCH --job-name=eval_p1_100
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# P1: Teacher Traces - 500 Sample Evaluation
# No dependencies, generates traces for P3 training
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "P1: Teacher Traces Evaluation"
echo "====================================="
echo "Samples: 100"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
TRACE_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces/traces_100_${RUN_ID}"
TIMING_LOG="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/P1_100_${RUN_ID}.json"

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p1_traces,trace_output_dir=${TRACE_DIR},cache_mode=none,timing_log_path=${TIMING_LOG}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 100 \
    --output_path results/P1_100_${RUN_ID}

echo ""
echo "Trace files saved to: $TRACE_DIR"
echo "Completed: $(date)"
