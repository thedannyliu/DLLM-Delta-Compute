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
###############################################################################

# Source bashrc first (before set -u to avoid unbound variable issues)
source ~/.bashrc
conda activate dcllm

set -eo pipefail

# Paths
PROJECT_ROOT="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute"
DREAM_ROOT="${PROJECT_ROOT}/external/Dream"
cd "${PROJECT_ROOT}"

# Verify environment
which python
python --version
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Create directories
mkdir -p logs
mkdir -p "${PROJECT_ROOT}/reports/timing"

# Export paths
export PYTHONPATH="${PROJECT_ROOT}:${DREAM_ROOT}:${PYTHONPATH:-}"
export HF_HOME="${PROJECT_ROOT}/.cache/huggingface"

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

# Trace and checkpoint paths
TRACE_DIR="${PROJECT_ROOT}/experiments/P1_traces/${TIMESTAMP}"
P3_CHECKPOINT="${PROJECT_ROOT}/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
PHASEB_CHECKPOINT="${PROJECT_ROOT}/experiments/PhaseB_router/checkpoints/router_final.pt"
PHASEC_CHECKPOINT="${PROJECT_ROOT}/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

cd "${DREAM_ROOT}"

echo ""
echo "=== P2: Confidence-based Early Stop ==="
echo "Testing delta_mode=p2_early_stop"
python -m eval_instruct.lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p2_early_stop,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1 \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/P2_early_stop"

cp "${RESULTS_DIR}/P2_early_stop/${MODEL//\//__}/results.json" "${RESULTS_DIR}/P2_early_stop.json" 2>/dev/null || \
cp "${RESULTS_DIR}/P2_early_stop/results.json" "${RESULTS_DIR}/P2_early_stop.json" 2>/dev/null || true

echo ""
echo "=== P3: Learned Gate (requires checkpoint) ==="
if [ -f "${P3_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${P3_CHECKPOINT}"
    python -m eval_instruct.lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p3_learned_gate,gate_checkpoint=${P3_CHECKPOINT},gate_threshold=0.5 \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/P3_learned_gate"
    
    cp "${RESULTS_DIR}/P3_learned_gate/${MODEL//\//__}/results.json" "${RESULTS_DIR}/P3_learned_gate.json" 2>/dev/null || \
    cp "${RESULTS_DIR}/P3_learned_gate/results.json" "${RESULTS_DIR}/P3_learned_gate.json" 2>/dev/null || true
else
    echo "SKIP: P3 checkpoint not found at ${P3_CHECKPOINT}"
fi

echo ""
echo "=== P4: Adaptive Stride ==="
python -m eval_instruct.lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=p4_adaptive,adaptive_window=4,adaptive_threshold=0.01 \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/P4_adaptive"

cp "${RESULTS_DIR}/P4_adaptive/${MODEL//\//__}/results.json" "${RESULTS_DIR}/P4_adaptive.json" 2>/dev/null || \
cp "${RESULTS_DIR}/P4_adaptive/results.json" "${RESULTS_DIR}/P4_adaptive.json" 2>/dev/null || true

echo ""
echo "=== Phase A: Binary Skip Router ==="
python -m eval_instruct.lm_eval \
    --model diffllm \
    --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=phaseA_binary_skip \
    --tasks ${TASK} \
    --batch_size ${BATCH_SIZE} \
    --num_fewshot ${NUM_FEWSHOT} \
    --limit ${NUM_SAMPLES} \
    --output_path "${RESULTS_DIR}/PhaseA_binary_skip"

cp "${RESULTS_DIR}/PhaseA_binary_skip/${MODEL//\//__}/results.json" "${RESULTS_DIR}/PhaseA_binary_skip.json" 2>/dev/null || \
cp "${RESULTS_DIR}/PhaseA_binary_skip/results.json" "${RESULTS_DIR}/PhaseA_binary_skip.json" 2>/dev/null || true

echo ""
echo "=== Phase B: Feature-based Router ==="
if [ -f "${PHASEB_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${PHASEB_CHECKPOINT}"
    python -m eval_instruct.lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=phaseB_router,router_checkpoint=${PHASEB_CHECKPOINT} \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/PhaseB_router"
    
    cp "${RESULTS_DIR}/PhaseB_router/${MODEL//\//__}/results.json" "${RESULTS_DIR}/PhaseB_router.json" 2>/dev/null || \
    cp "${RESULTS_DIR}/PhaseB_router/results.json" "${RESULTS_DIR}/PhaseB_router.json" 2>/dev/null || true
else
    echo "SKIP: Phase B checkpoint not found at ${PHASEB_CHECKPOINT}"
fi

echo ""
echo "=== Phase C: Continuous Router ==="
if [ -f "${PHASEC_CHECKPOINT}" ]; then
    echo "Using checkpoint: ${PHASEC_CHECKPOINT}"
    python -m eval_instruct.lm_eval \
        --model diffllm \
        --model_args pretrained=${MODEL},dtype=bfloat16,max_length=2048,diffusion_steps=${DIFFUSION_STEPS},delta_mode=phaseC_continuous,router_checkpoint=${PHASEC_CHECKPOINT} \
        --tasks ${TASK} \
        --batch_size ${BATCH_SIZE} \
        --num_fewshot ${NUM_FEWSHOT} \
        --limit ${NUM_SAMPLES} \
        --output_path "${RESULTS_DIR}/PhaseC_continuous"
    
    cp "${RESULTS_DIR}/PhaseC_continuous/${MODEL//\//__}/results.json" "${RESULTS_DIR}/PhaseC_continuous.json" 2>/dev/null || \
    cp "${RESULTS_DIR}/PhaseC_continuous/results.json" "${RESULTS_DIR}/PhaseC_continuous.json" 2>/dev/null || true
else
    echo "SKIP: Phase C checkpoint not found at ${PHASEC_CHECKPOINT}"
fi

echo ""
echo "=============================================="
echo "CONTINUATION COMPLETE"
echo "=============================================="
echo "Results in: ${RESULTS_DIR}/"
ls -la "${RESULTS_DIR}/"
echo ""
echo "Summary of strict-match accuracy:"
for f in "${RESULTS_DIR}"/*.json; do
    if [ -f "$f" ]; then
        name=$(basename "$f" .json)
        acc=$(python3 -c "import json; d=json.load(open('$f')); print(d.get('results',{}).get('gsm8k',{}).get('exact_match,strict-match',d.get('results',{}).get('gsm8k',{}).get('acc_norm,none','N/A')))" 2>/dev/null || echo "N/A")
        echo "  ${name}: ${acc}"
    fi
done
