#!/bin/bash
#SBATCH --job-name=test_all_phases
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
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

# Test P1 - Traces
echo ""
echo "========== Testing P1 (Traces) =========="
TRACE_DIR="${PROJECT_ROOT}/experiments/P1_traces/test_${RUN_ID}"
mkdir -p "$TRACE_DIR"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p1_traces,trace_output_dir=${TRACE_DIR},cache_mode=none" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --output_path results/test_P1_${RUN_ID}
P1_STATUS=$?
echo "P1 Exit Status: $P1_STATUS"
echo "Trace files: $(ls -1 ${TRACE_DIR}/*.pt 2>/dev/null | wc -l)"

# Test P2 - Early Stop
echo ""
echo "========== Testing P2 (Early Stop) =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p2_early_stop,cache_mode=none,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --output_path results/test_P2_${RUN_ID}
P2_STATUS=$?
echo "P2 Exit Status: $P2_STATUS"

# Test P3 - Learned Gate
echo ""
echo "========== Testing P3 (Learned Gate) =========="
GATE_CKPT="${PROJECT_ROOT}/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
if [ -f "$GATE_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=${GATE_CKPT},gate_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit 1 \
        --output_path results/test_P3_${RUN_ID}
    P3_STATUS=$?
else
    echo "WARNING: Gate checkpoint not found at $GATE_CKPT"
    P3_STATUS=1
fi
echo "P3 Exit Status: $P3_STATUS"

# Test P4 - Adaptive Steps
echo ""
echo "========== Testing P4 (Adaptive Steps) =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p4_adaptive,cache_mode=none,adaptive_max_stride=4,adaptive_min_safe_step=10" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 1 \
    --output_path results/test_P4_${RUN_ID}
P4_STATUS=$?
echo "P4 Exit Status: $P4_STATUS"

# Test Phase A - Heuristic Caching
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

# Test Phase C - Continuous Router
echo ""
echo "========== Testing Phase C (Continuous Router) =========="
CONT_ROUTER_CKPT="${PROJECT_ROOT}/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"
if [ -f "$CONT_ROUTER_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_continuous,router_checkpoint=${CONT_ROUTER_CKPT},router_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit 1 \
        --output_path results/test_PhaseC_${RUN_ID}
    PHASE_C_STATUS=$?
else
    echo "WARNING: Continuous router checkpoint not found at $CONT_ROUTER_CKPT"
    PHASE_C_STATUS=1
fi
echo "Phase C Exit Status: $PHASE_C_STATUS"

# Summary
echo ""
echo "====================================="
echo "Test Summary"
echo "====================================="
echo "P0 (Baseline):       $([ $P0_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P1 (Traces):         $([ $P1_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P2 (Early Stop):     $([ $P2_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P3 (Learned Gate):   $([ $P3_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P4 (Adaptive Steps): $([ $P4_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase A (Heuristic): $([ $PHASE_A_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase B (Router):    $([ $PHASE_B_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase C (Continuous):$([ $PHASE_C_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo ""
echo "Completed: $(date)"
