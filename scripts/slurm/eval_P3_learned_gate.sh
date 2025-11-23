#!/bin/bash
#SBATCH --job-name=eval_P3_learned_gate
#SBATCH --output=experiments/P3_learned_gate/logs/gsm8k_learned_gate_%j.out
#SBATCH --error=experiments/P3_learned_gate/logs/gsm8k_learned_gate_%j.err
#SBATCH --partition=ice-gpu
#SBATCH --gres=gpu:H100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=6:00:00

# P3 - Learned Gate Evaluation on GSM8K

set -e

echo "============================================"
echo "P3 - Learned Gate Evaluation"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
echo ""

# Activate environment
module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Check environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Project directory
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Create output directories
mkdir -p experiments/P3_learned_gate/logs
mkdir -p experiments/P3_learned_gate/results

# Configuration
MODEL="Dream-org/Dream-v0-Instruct-7B"
GATE_CHECKPOINT="experiments/P3_learned_gate/checkpoints/learned_gate_best.pt"
DIFFUSION_STEPS=256
MAX_NEW_TOKENS=256
BATCH_SIZE=1
NUM_SAMPLES=100
SEED=42

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Gate checkpoint: $GATE_CHECKPOINT"
echo "  Diffusion steps: $DIFFUSION_STEPS"
echo "  Max new tokens: $MAX_NEW_TOKENS"
echo "  Batch size: $BATCH_SIZE"
echo "  Num samples: $NUM_SAMPLES"
echo "  Seed: $SEED"
echo ""

# Check if gate checkpoint exists
if [ ! -f "$GATE_CHECKPOINT" ]; then
    echo "ERROR: Gate checkpoint not found at $GATE_CHECKPOINT"
    echo "Please train the gate first using scripts/training/train_learned_gate.py"
    exit 1
fi

# Run evaluation
echo "Running P3 evaluation with learned gate..."
echo "============================================"

cd external/Dream/eval_instruct

PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL,trust_remote_code=True,max_new_tokens=$MAX_NEW_TOKENS,diffusion_steps=$DIFFUSION_STEPS,dtype="bfloat16",temperature=0.1,top_p=0.9,alg="entropy",delta_mode="p3_learned_gate",gate_checkpoint="$GATE_CHECKPOINT",gate_threshold=0.5 \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size $BATCH_SIZE \
    --limit $NUM_SAMPLES \
    --seed $SEED \
    --output_path ../../../experiments/P3_learned_gate/results/gsm8k_${SLURM_JOB_ID} \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template

echo ""
echo "============================================"
echo "P3 evaluation complete!"
echo "End time: $(date)"
echo ""

# Summary
echo "Results saved to: experiments/P3_learned_gate/results/gsm8k_${SLURM_JOB_ID}"
echo "Log saved to: experiments/P3_learned_gate/logs/gsm8k_learned_gate_${SLURM_JOB_ID}.out"
echo ""
echo "To check results:"
echo "  cat experiments/P3_learned_gate/logs/gsm8k_learned_gate_${SLURM_JOB_ID}.out"
