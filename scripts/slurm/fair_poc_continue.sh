#!/bin/bash
#SBATCH --job-name=fair_poc_continue
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=384GB
#SBATCH --time=16:00:00
#SBATCH --output=logs/fair_poc_continue_%j.out
#SBATCH --error=logs/fair_poc_continue_%j.err

###############################################################################
# Fair PoC Evaluation - Continue from P2 (P0/P1 already completed in job 3695979)
#
# Previous job completed:
#   - P0_baseline: strict=45.6%, flexible=51.8% (500 samples, 8-shot)
#   - P1_traces:   strict=45.6%, flexible=51.8% (500 samples, 8-shot)
#
# This continues with: P2 → P3 → P4 → Phase A → Phase B → Phase C
#
# FIXES APPLIED:
#   1. P2 delta_mode typo: "p2_earlystop" -> "p2_early_stop" (fixed in generation_utils.py)
#   2. Phase A/B/C use cache_mode, not delta_mode
#   3. Use local lm_eval package via editable install
###############################################################################

set -e

# Load modules and activate conda
module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Paths
PROJECT_ROOT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute"
DREAM_ROOT="${PROJECT_ROOT}/external/Dream"
cd "${PROJECT_ROOT}"

# Verify environment
echo "Python: $(which python)"
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create directories
mkdir -p logs
mkdir -p "${PROJECT_ROOT}/reports/timing"

# Export paths
export PYTHONPATH="${PROJECT_ROOT}:${DREAM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

# Ensure local lm_eval is installed
echo "Ensuring local lm_eval package is installed..."
pip install -e "${DREAM_ROOT}/eval_instruct" --quiet 2>/dev/null || true

# === Evaluation Parameters (CONSISTENT FOR ALL PHASES) ===
MODEL="Dream-org/Dream-v0-Instruct-7B"
TASK="gsm8k"
NUM_FEWSHOT=8           # Consistent 8-shot for all
BATCH_SIZE=1
DIFFUSION_STEPS=128
NUM_SAMPLES=500         # 500 samples for statistical significance
TIMESTAMP="fair_eval_$(date +%Y%m%d_%H%M%S)"

echo "=============================================="
echo "Fair PoC Evaluation - Continuation"
echo "Model: ${MODEL}"
echo "Task: ${TASK}"
echo "Num fewshot: ${NUM_FEWSHOT}"
echo "Diffusion steps: ${DIFFUSION_STEPS}"
echo "Num samples: ${NUM_SAMPLES}"
echo "Timestamp: ${TIMESTAMP}"
echo "=============================================="

# Results directory
RESULTS_DIR="${PROJECT_ROOT}/reports/timing/${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"

# Checkpoint paths
P3_CHECKPOINT="${PROJECT_ROOT}/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
PHASEB_CHECKPOINT="${PROJECT_ROOT}/experiments/PhaseB_router/checkpoints/router_final.pt"
PHASEC_CHECKPOINT="${PROJECT_ROOT}/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

cd "${DREAM_ROOT}/eval_instruct"

# ==============================================================================
# P2: Confidence-based Early Stop
# ==============================================================================
echo ""
echo "=== P2: Confidence-based Early Stop ==="
echo "Testing delta_mode=p2_early_stop"
python -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p2_early_stop,cache_mode=none,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1 \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/P2_early_stop"

# ==============================================================================
# P3: Learned Gate
# ==============================================================================
echo ""
echo "=== P3: Learned Gate ==="
if [ -f "${P3_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${P3_CHECKPOINT}"
    python -m lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p3_learned_gate,cache_mode=none,gate_checkpoint=${P3_CHECKPOINT},gate_threshold=0.5 \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/P3_learned_gate"
else
    echo "SKIP: P3 checkpoint not found at ${P3_CHECKPOINT}"
fi

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
echo ""
echo "=== P4: Adaptive Stride ==="
python -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p4_adaptive,cache_mode=none,adaptive_window=4,adaptive_threshold=0.01 \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/P4_adaptive"

# ==============================================================================
# Phase A: Heuristic FFN Caching (cache_mode=l2c_ffn)
# ==============================================================================
echo ""
echo "=== Phase A: Heuristic FFN Caching ==="
python -m lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=none,cache_mode=l2c_ffn \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/PhaseA_l2c_ffn"

# ==============================================================================
# Phase B: Learned Router (cache_mode=l2c_learned)
# ==============================================================================
echo ""
echo "=== Phase B: Learned Router ==="
if [ -f "${PHASEB_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${PHASEB_CHECKPOINT}"
    python -m lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=none,cache_mode=l2c_learned,router_checkpoint=${PHASEB_CHECKPOINT} \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/PhaseB_l2c_learned"
else
    echo "SKIP: Phase B checkpoint not found at ${PHASEB_CHECKPOINT}"
fi

# ==============================================================================
# Phase C: Continuous Router (cache_mode=l2c_continuous)
# ==============================================================================
echo ""
echo "=== Phase C: Continuous Router ==="
if [ -f "${PHASEC_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${PHASEC_CHECKPOINT}"
    python -m lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=none,cache_mode=l2c_continuous,router_checkpoint=${PHASEC_CHECKPOINT} \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/PhaseC_l2c_continuous"
else
    echo "SKIP: Phase C checkpoint not found at ${PHASEC_CHECKPOINT}"
fi

# ==============================================================================
# Summary
# ==============================================================================
echo ""
echo "=============================================="
echo "CONTINUATION COMPLETE"
echo "=============================================="
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Summary of results (strict-match | flexible-extract):"
echo "------------------------------------------------------"
for phase_dir in "${RESULTS_DIR}"/*/; do
    if [ -d "$phase_dir" ]; then
        phase_name=$(basename "$phase_dir")
        # Find the results JSON file
        results_file=$(find "$phase_dir" -name "results*.json" 2>/dev/null | head -1)
        if [ -n "$results_file" ] && [ -f "$results_file" ]; then
            strict=$(python3 -c "import json; d=json.load(open('$results_file')); r=d.get('results',{}).get('gsm8k',{}); print(f\"{r.get('exact_match,strict-match','N/A'):.3f}\")" 2>/dev/null || echo "N/A")
            flexible=$(python3 -c "import json; d=json.load(open('$results_file')); r=d.get('results',{}).get('gsm8k',{}); print(f\"{r.get('exact_match,flexible-extract','N/A'):.3f}\")" 2>/dev/null || echo "N/A")
            echo "  ${phase_name}: strict=${strict} | flexible=${flexible}"
        else
            echo "  ${phase_name}: No results file found"
        fi
    fi
done

echo ""
echo "Previous results from job 3695979:"
echo "  P0_baseline: strict=0.456 | flexible=0.518"
echo "  P1_traces:   strict=0.456 | flexible=0.518"
