#!/bin/bash
#SBATCH --job-name=eval_PhaseC_continuous
#SBATCH --output=experiments/PhaseC_continuous/logs/gsm8k_continuous_%j.out
#SBATCH --error=experiments/PhaseC_continuous/logs/gsm8k_continuous_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00

# Phase C - Continuous-Time Router Evaluation

set -e

echo "============================================"
echo "Phase C - Continuous-Time Router Evaluation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo ""

# Activate environment
source /home/hice1/eliu354/.bashrc
conda activate dcllm

# Project directory
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Create output directories
mkdir -p experiments/PhaseC_continuous/logs
mkdir -p experiments/PhaseC_continuous/results

# Configuration
MODEL="Dream-org/Dream-v0-Instruct-7B"
ROUTER_CHECKPOINT="experiments/PhaseC_continuous/checkpoints/continuous_router_best.pt"
DIFFUSION_STEPS=256  # Can test different schedules (128, 256, 512) for transfer
MAX_NEW_TOKENS=256
BATCH_SIZE=1
NUM_SAMPLES=100
SEED=42
ROUTER_THRESHOLD=0.5

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Router checkpoint: $ROUTER_CHECKPOINT"
echo "  Diffusion steps: $DIFFUSION_STEPS"
echo "  Router threshold: $ROUTER_THRESHOLD"
echo ""

# Check if router checkpoint exists
if [ ! -f "$ROUTER_CHECKPOINT" ]; then
    echo "ERROR: Router checkpoint not found at $ROUTER_CHECKPOINT"
    echo "Please train the continuous router first"
    exit 1
fi

# Run evaluation
echo "Running Phase C evaluation with continuous router..."
echo "============================================"

cd external/Dream/eval_instruct

PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=$DIFFUSION_STEPS,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",cache_mode="l2c_continuous",router_checkpoint="$ROUTER_CHECKPOINT",router_threshold=$ROUTER_THRESHOLD \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --limit $NUM_SAMPLES \
    --seed $SEED \
    --output_path ../../../experiments/PhaseC_continuous/results/gsm8k_${SLURM_JOB_ID} \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template

echo ""
echo "============================================"
echo "Phase C evaluation complete!"
echo "End time: $(date)"
echo ""

# Test schedule transfer (run with different step counts)
echo "To test schedule transfer, run with different diffusion_steps:"
echo "  128 steps: sbatch --export=DIFFUSION_STEPS=128 scripts/slurm/eval_PhaseC_continuous.sh"
echo "  512 steps: sbatch --export=DIFFUSION_STEPS=512 scripts/slurm/eval_PhaseC_continuous.sh"
