#!/bin/bash
#SBATCH --job-name=test_all_coc
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=1:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Test All Phases with Minimal Samples (1 sample each)
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "Testing All Phases (1 sample each)"
echo "====================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute"

# Test P0 - Baseline
echo ""
echo "========== Testing P0 (Baseline) =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=none" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --output_path results/test_P0_${RUN_ID}
P0_STATUS=$?
echo "P0 Exit Status: $P0_STATUS"

# Test Phase A - Heuristic Caching (the fixed one)
echo ""
echo "========== Testing Phase A (Heuristic Caching) =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_ffn,cache_schedule=1;3;5;7;9;11;13;15;17;19;21;23;25;27;29;31" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --output_path results/test_PhaseA_${RUN_ID}
PHASE_A_STATUS=$?
echo "Phase A Exit Status: $PHASE_A_STATUS"

# Test Phase B - Learned Router
echo ""
echo "========== Testing Phase B (Learned Router) =========="
ROUTER_CKPT="${PROJECT_ROOT}/experiments/PhaseB_router/checkpoints/router_final.pt"
if [ -f "$ROUTER_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_learned,router_checkpoint=${ROUTER_CKPT},router_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit 1 \
        --output_path results/test_PhaseB_${RUN_ID}
    PHASE_B_STATUS=$?
else
    echo "WARNING: Router checkpoint not found at $ROUTER_CKPT"
    PHASE_B_STATUS=1
fi
echo "Phase B Exit Status: $PHASE_B_STATUS"

# Summary
echo ""
echo "====================================="
echo "Test Summary"
echo "====================================="
echo "P0 (Baseline):       $([ $P0_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase A (Heuristic): $([ $PHASE_A_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase B (Router):    $([ $PHASE_B_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo ""
echo "Completed: $(date)"
