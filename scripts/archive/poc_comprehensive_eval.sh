#!/bin/bash
#SBATCH --job-name=poc_all_eval
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:H100:1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Comprehensive PoC Evaluation (All Phases)
# ==================================================
# Per master_plan.md evaluation order:
# 1) P0/P1 verification
# 2) P2 early stop
# 3) Phase A heuristic caching
# 4) P3 learned gate
# 5) Phase B learned router
# 6) P4 adaptive stride
# 7) Phase C continuous router

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "PoC Comprehensive Evaluation"
echo "====================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

RUN_ID=$(date +%Y%m%d_%H%M%S)
PROJECT_ROOT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute"
SAMPLES=100

# Create results directory
RESULTS_DIR="${PROJECT_ROOT}/experiments/poc_eval_${RUN_ID}"
mkdir -p "$RESULTS_DIR"

echo "Results will be saved to: $RESULTS_DIR"

# ==================================================
# P0 - Baseline (no acceleration)
# ==================================================
echo ""
echo "========== P0: Baseline =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=none,cache_mode=none" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}/P0_baseline

P0_STATUS=$?
echo "P0 Status: $P0_STATUS"

# ==================================================
# P1 - Teacher Traces (collect traces)
# ==================================================
echo ""
echo "========== P1: Teacher Traces =========="
TRACE_DIR="${PROJECT_ROOT}/experiments/P1_traces/eval_${RUN_ID}"
mkdir -p "$TRACE_DIR"

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p1_traces,cache_mode=none,trace_output_dir=${TRACE_DIR}" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}/P1_traces

P1_STATUS=$?
echo "P1 Status: $P1_STATUS"
echo "Trace files: $(ls -1 ${TRACE_DIR}/*.pt 2>/dev/null | wc -l)"

# ==================================================
# P2 - Early Stop (rule-based)
# ==================================================
echo ""
echo "========== P2: Rule-Based Early Stop =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p2_early_stop,cache_mode=none,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}/P2_early_stop

P2_STATUS=$?
echo "P2 Status: $P2_STATUS"

# ==================================================
# Phase A - Heuristic FFN Caching (odd layers)
# ==================================================
echo ""
echo "========== Phase A: Heuristic Caching =========="
# Cache odd-indexed layers (1,3,5,7,...,31) - 16 layers total
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=none,cache_mode=l2c_ffn,cache_schedule=1;3;5;7;9;11;13;15;17;19;21;23;25;27;29;31" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}/PhaseA_heuristic

PHASE_A_STATUS=$?
echo "Phase A Status: $PHASE_A_STATUS"

# ==================================================
# P3 - Learned Gate
# ==================================================
echo ""
echo "========== P3: Learned Gate =========="
GATE_CKPT="${PROJECT_ROOT}/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"

if [ -f "$GATE_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=${GATE_CKPT},gate_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}/P3_learned_gate
    P3_STATUS=$?
else
    echo "WARNING: Gate checkpoint not found at $GATE_CKPT"
    P3_STATUS=1
fi
echo "P3 Status: $P3_STATUS"

# ==================================================
# Phase B - Learned Router
# ==================================================
echo ""
echo "========== Phase B: Learned Router =========="
ROUTER_CKPT="${PROJECT_ROOT}/experiments/PhaseB_router/checkpoints/router_final.pt"

if [ -f "$ROUTER_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=none,cache_mode=l2c_learned,router_checkpoint=${ROUTER_CKPT},router_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}/PhaseB_router
    PHASE_B_STATUS=$?
else
    echo "WARNING: Router checkpoint not found at $ROUTER_CKPT"
    PHASE_B_STATUS=1
fi
echo "Phase B Status: $PHASE_B_STATUS"

# ==================================================
# P4 - Adaptive Stride
# ==================================================
echo ""
echo "========== P4: Adaptive Stride =========="
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p4_adaptive,cache_mode=none,adaptive_max_stride=4,adaptive_min_safe_step=10" \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}/P4_adaptive

P4_STATUS=$?
echo "P4 Status: $P4_STATUS"

# ==================================================
# Phase C - Continuous Router
# ==================================================
echo ""
echo "========== Phase C: Continuous Router =========="
CONT_ROUTER_CKPT="${PROJECT_ROOT}/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

if [ -f "$CONT_ROUTER_CKPT" ]; then
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=none,cache_mode=l2c_continuous,router_checkpoint=${CONT_ROUTER_CKPT},router_threshold=0.5" \
        --tasks gsm8k \
        --num_fewshot 5 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}/PhaseC_continuous
    PHASE_C_STATUS=$?
else
    echo "WARNING: Continuous router checkpoint not found at $CONT_ROUTER_CKPT"
    PHASE_C_STATUS=1
fi
echo "Phase C Status: $PHASE_C_STATUS"

# ==================================================
# Summary
# ==================================================
echo ""
echo "====================================="
echo "PoC Evaluation Summary"
echo "====================================="
echo "P0 (Baseline):          $([ $P0_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P1 (Teacher Traces):    $([ $P1_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P2 (Early Stop):        $([ $P2_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase A (Heuristic):    $([ $PHASE_A_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P3 (Learned Gate):      $([ $P3_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase B (Learned Rtr):  $([ $PHASE_B_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "P4 (Adaptive Stride):   $([ $P4_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo "Phase C (Continuous):   $([ $PHASE_C_STATUS -eq 0 ] && echo 'PASS' || echo 'FAIL')"
echo ""
echo "Results saved to: $RESULTS_DIR"
echo "Completed: $(date)"
