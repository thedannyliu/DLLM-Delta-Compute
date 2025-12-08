#!/bin/bash
#SBATCH --job-name=sanity_check
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/sanity_check_%j.out
#SBATCH --error=logs/sanity_check_%j.err

# Load environment
module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Configuration
MODEL="Dream-org/Dream-v0-Base-7B"
DIFFUSION_STEPS=128
LIMIT=50
OUTPUT_ROOT="experiments/sanity_check"

echo "Starting Sanity Check (Limit: ${LIMIT})"

# Function to run eval
run_eval() {
    local PHASE_NAME=$1
    local EXTRA_ARGS=$2
    
    echo "Running ${PHASE_NAME}..."
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},${EXTRA_ARGS}" \
        --tasks gsm8k \
        --num_fewshot 8 \
        --batch_size 1 \
        --limit ${LIMIT} \
        --output_path "${OUTPUT_ROOT}/${PHASE_NAME}" \
        --log_samples
}

# P3: Learned Gate
# Note: Using existing checkpoint from fair eval
GATE_CKPT="experiments/P3_learned_gate/checkpoints/gate_final.pt"
run_eval "P3_learned_gate" "delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=${GATE_CKPT},gate_threshold=0.5"

# Phase B: Learned Router
ROUTER_B_CKPT="experiments/PhaseB_router/checkpoints/router_final.pt"
run_eval "PhaseB_l2c_learned" "delta_mode=none,cache_mode=l2c_learned,router_checkpoint=${ROUTER_B_CKPT},router_threshold=0.5"

# Phase C: Continuous Router
ROUTER_C_CKPT="experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"
run_eval "PhaseC_l2c_continuous" "delta_mode=none,cache_mode=l2c_continuous,router_checkpoint=${ROUTER_C_CKPT}"

echo "Sanity Check Complete. Results in ${OUTPUT_ROOT}"
