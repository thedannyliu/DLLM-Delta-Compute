#!/bin/bash
#SBATCH --job-name=eval_P3_500
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# P3: Learned Gate - 500 Sample Evaluation
# Requires: Pre-trained gate checkpoint (from P1 traces)
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "P3: Learned Gate Evaluation"
echo "====================================="
echo "Samples: 500"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
TIMING_LOG="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/P3_500_${RUN_ID}.json"

# Check for gate checkpoint
GATE_CKPT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"

if [ ! -f "$GATE_CKPT" ]; then
    echo "ERROR: Gate checkpoint not found at $GATE_CKPT"
    echo "P3 requires trained gate. Skipping evaluation."
    exit 1
fi

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=${GATE_CKPT},timing_log_path=${TIMING_LOG}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 500 \
    --output_path results/P3_500_${RUN_ID}

echo ""
echo "Completed: $(date)"
