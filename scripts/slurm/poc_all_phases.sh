#!/bin/bash
#SBATCH --job-name=poc_all_phases
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/poc_all_phases_%j.out
#SBATCH --error=logs/poc_all_phases_%j.err

# ==============================================================================
# Complete PoC Evaluation - All Phases with Base Model
# ==============================================================================
# P0, P2 (aggressive thresholds), P4, PhaseA, PhaseB, PhaseC, PhaseD, PhaseE
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=${1:-200}
MODEL="Dream-org/Dream-v0-Base-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="experiments/poc_all_phases_${SAMPLES}_${TIMESTAMP}"
TIMING_BASE="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/poc_all_phases_${SAMPLES}_${TIMESTAMP}"
WANDB_PROJECT="dllm_poc_base_${SAMPLES}"

# Checkpoint paths (use latest trained models)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
P3_GATE_PATH=$(ls -td ${CKPT_DIR}/P3_gate_cot_*/checkpoints/learned_gate_final.pt 2>/dev/null | head -1)
PHASEB_ROUTER_PATH=$(ls -td ${CKPT_DIR}/PhaseB_router_cot_*/checkpoints/router_final.pt 2>/dev/null | head -1)
PHASEC_ROUTER_PATH=$(ls -td ${CKPT_DIR}/PhaseC_router_cot_*/checkpoints/continuous_router_final.pt 2>/dev/null | head -1)
# Fallback to existing models if new ones not available
[ -z "$P3_GATE_PATH" ] && P3_GATE_PATH="${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt"
[ -z "$PHASEB_ROUTER_PATH" ] && PHASEB_ROUTER_PATH="${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt"
[ -z "$PHASEC_ROUTER_PATH" ] && PHASEC_ROUTER_PATH="${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt"

mkdir -p ${OUTPUT_BASE}
mkdir -p ${TIMING_BASE}
mkdir -p logs

export WANDB_PROJECT=${WANDB_PROJECT}
export WANDB_RUN_GROUP="poc_all_${TIMESTAMP}"

echo "=============================================="
echo "Complete PoC Evaluation - All Phases"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Timestamp: ${TIMESTAMP}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Checkpoints:"
echo "  P3 Gate: ${P3_GATE_PATH}"
echo "  PhaseB Router: ${PHASEB_ROUTER_PATH}"
echo "  PhaseC Router: ${PHASEC_ROUTER_PATH}"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation
run_eval() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    local TIMING_PATH="${TIMING_BASE}/${NAME}.json"
    
    echo ""
    echo "====== ${NAME} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    echo "Start: $(date)"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_PATH}${EXTRA_ARGS}" \
        --gen_kwargs "do_sample=False,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot 8 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --log_samples \
        --output_path "${OUTPUT_DIR}" || echo "WARNING: ${NAME} failed"
    
    echo "Completed: $(date)"
    if [ -f "${TIMING_PATH}" ]; then
        python3 -c "import json; d=json.load(open('${TIMING_PATH}')); print(f'Timing: mean={d.get(\"mean_time_per_sample\",0):.1f}s, tok/s={d.get(\"tokens_per_second\",0):.1f}, skip_ratio={d.get(\"skip_ratio_steps\",0):.2%}')" 2>/dev/null || echo "Timing: (error reading)"
    fi
}

# ==============================================================================
# P0: Baseline Reference
# ==============================================================================
run_eval "P0_baseline" "none" "none" ""

# ==============================================================================
# P2: Early Stop with AGGRESSIVE thresholds for CoT
# Try multiple thresholds to find one that works
# ==============================================================================
run_eval "P2_conf50_ent20" "p2_early_stop" "none" \
    ",early_stop_confidence_threshold=0.5,early_stop_entropy_threshold=2.0"

run_eval "P2_conf40_ent30" "p2_early_stop" "none" \
    ",early_stop_confidence_threshold=0.4,early_stop_entropy_threshold=3.0"

# ==============================================================================
# P3: Learned Gate (if checkpoint available)
# ==============================================================================
if [ -f "${P3_GATE_PATH}" ]; then
    run_eval "P3_learned_gate" "p3_learned_gate" "none" \
        ",gate_checkpoint_path=${P3_GATE_PATH}"
else
    echo "WARNING: P3 gate checkpoint not found, skipping P3"
fi

# ==============================================================================
# P4: Adaptive Stride
# ==============================================================================
run_eval "P4_adaptive" "p4_adaptive" "none" \
    ",adaptive_max_stride=4,adaptive_lte_threshold=0.01,adaptive_entropy_threshold=2.0,adaptive_min_safe_step=10"

# ==============================================================================
# Phase A: Heuristic FFN Caching (odd layers)
# ==============================================================================
CACHE_LAYERS=$(seq -s ';' 1 2 63)
run_eval "PhaseA_caching" "none" "l2c_ffn" ",cache_schedule=${CACHE_LAYERS}"

# ==============================================================================
# Phase B: Learned Router (if checkpoint available)
# ==============================================================================
if [ -f "${PHASEB_ROUTER_PATH}" ]; then
    run_eval "PhaseB_router" "none" "l2c_learned" \
        ",router_checkpoint_path=${PHASEB_ROUTER_PATH}"
else
    echo "WARNING: PhaseB router checkpoint not found, skipping PhaseB"
fi

# ==============================================================================
# Phase C: Continuous Router (if checkpoint available)
# ==============================================================================
if [ -f "${PHASEC_ROUTER_PATH}" ]; then
    run_eval "PhaseC_continuous" "none" "l2c_continuous" \
        ",router_checkpoint_path=${PHASEC_ROUTER_PATH}"
else
    echo "WARNING: PhaseC router checkpoint not found, skipping PhaseC"
fi

# ==============================================================================
# Phase D: Continuous Layer Skipping (reuses PhaseC router with skip semantics)
# ==============================================================================
if [ -f "${PHASEC_ROUTER_PATH}" ]; then
    run_eval "PhaseD_skip" "none" "skip_continuous" \
        ",router_checkpoint_path=${PHASEC_ROUTER_PATH}"
else
    echo "WARNING: PhaseD requires PhaseC router, skipping PhaseD"
fi

# ==============================================================================
# Phase E: FlexiDepth (if adapter available)
# ==============================================================================
PHASEE_ADAPTER_PATH=$(ls -td ${CKPT_DIR}/PhaseE_flexi_*/checkpoints/*_final.pt 2>/dev/null | head -1)
if [ -f "${PHASEE_ADAPTER_PATH}" ]; then
    run_eval "PhaseE_flexi" "none" "flexi_ffn" \
        ",adapter_checkpoint_path=${PHASEE_ADAPTER_PATH}"
else
    echo "WARNING: PhaseE adapter checkpoint not found, skipping PhaseE"
fi

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Complete PoC Evaluation DONE"
echo "=============================================="
echo "Results: ${OUTPUT_BASE}"
echo "Timing: ${TIMING_BASE}"
echo "Completed: $(date)"
