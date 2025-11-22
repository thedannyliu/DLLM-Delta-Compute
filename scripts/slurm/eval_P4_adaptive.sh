#!/bin/bash
#SBATCH --job-name=eval_P4_adaptive
#SBATCH --output=experiments/P4_adaptive/logs/gsm8k_adaptive_%j.out
#SBATCH --error=experiments/P4_adaptive/logs/gsm8k_adaptive_%j.err
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00

# P4 - Adaptive Step Scheduling Evaluation on GSM8K

set -e

echo "============================================"
echo "P4 - Adaptive Step Scheduling Evaluation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
source /home/hice1/eliu354/.bashrc
conda activate dcllm

# Check environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Project directory
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Create output directories
mkdir -p experiments/P4_adaptive/logs
mkdir -p experiments/P4_adaptive/results

# Configuration
MODEL="Dream-org/Dream-v0-Instruct-7B"
DIFFUSION_STEPS=256
MAX_NEW_TOKENS=256
BATCH_SIZE=1
NUM_SAMPLES=100
SEED=42

# Adaptive scheduler parameters
MAX_STRIDE=4
LTE_THRESHOLD=0.01
ENTROPY_THRESHOLD=2.0
KL_THRESHOLD=0.1
STABLE_WINDOW=5
MIN_SAFE_STEP=10

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Diffusion steps: $DIFFUSION_STEPS"
echo "  Max stride: $MAX_STRIDE"
echo "  LTE threshold: $LTE_THRESHOLD"
echo "  Entropy threshold: $ENTROPY_THRESHOLD"
echo "  KL threshold: $KL_THRESHOLD"
echo "  Stable window: $STABLE_WINDOW"
echo "  Min safe step: $MIN_SAFE_STEP"
echo ""

# Run evaluation
echo "Running P4 evaluation with adaptive scheduling..."
echo "============================================"

cd external/Dream/eval_instruct

PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=$DIFFUSION_STEPS,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",delta_mode="p4_adaptive",adaptive_max_stride=$MAX_STRIDE,adaptive_lte_threshold=$LTE_THRESHOLD,adaptive_entropy_threshold=$ENTROPY_THRESHOLD,adaptive_kl_threshold=$KL_THRESHOLD,adaptive_stable_window=$STABLE_WINDOW,adaptive_min_safe_step=$MIN_SAFE_STEP \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --limit $NUM_SAMPLES \
    --seed $SEED \
    --output_path ../../../experiments/P4_adaptive/results/gsm8k_${SLURM_JOB_ID} \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template

echo ""
echo "============================================"
echo "P4 evaluation complete!"
echo "End time: $(date)"
echo ""

# Summary
echo "Results saved to: experiments/P4_adaptive/results/gsm8k_${SLURM_JOB_ID}"
echo "Log saved to: experiments/P4_adaptive/logs/gsm8k_adaptive_${SLURM_JOB_ID}.out"
