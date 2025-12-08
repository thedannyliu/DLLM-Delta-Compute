#!/bin/bash
#SBATCH --job-name=sanity_full
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=4:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/sanity_full_%j.out
#SBATCH --error=logs/sanity_full_%j.err

# ==============================================================================
# Full Sanity Check: P0, P2, P3, PhaseB, PhaseC with Diff Comparison
# ==============================================================================
# Runs 50-sample evaluation of all phases and diffs predictions
# Uses correct checkpoint paths and Instruct model with CoT settings
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=50
DIFFUSION_STEPS=256
MAX_NEW_TOKENS=256
NUM_FEWSHOT=8
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/sanity_full_${TIMESTAMP}"

# Correct checkpoint paths
GATE_CKPT="experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
ROUTER_B_CKPT="experiments/PhaseB_router/checkpoints/router_final.pt"
ROUTER_C_CKPT="experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=============================================="
echo "Full Sanity Check - All Phases"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Diffusion Steps: ${DIFFUSION_STEPS}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Checkpoints:"
echo "  P3 Gate: ${GATE_CKPT} (exists: $([ -f ${GATE_CKPT} ] && echo 'yes' || echo 'NO!'))"
echo "  PhaseB Router: ${ROUTER_B_CKPT} (exists: $([ -f ${ROUTER_B_CKPT} ] && echo 'yes' || echo 'NO!'))"
echo "  PhaseC Router: ${ROUTER_C_CKPT} (exists: $([ -f ${ROUTER_C_CKPT} ] && echo 'yes' || echo 'NO!'))"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation with sample logging
run_eval() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local RESULTS_DIR="${OUTPUT_DIR}/${NAME}"
    
    echo ""
    echo "====== ${NAME} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    echo "Start: $(date)"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},add_bos_token=True${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --log_samples \
        --output_path "${RESULTS_DIR}" || echo "WARNING: ${NAME} failed"
    
    echo "Completed: $(date)"
}

# P0: Baseline (no delta-compute)
run_eval "P0_baseline" "none" "none" ""

# P2: Early Stop (sequence-level)
run_eval "P2_early_stop" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1"

# P3: Learned Gate (FIXED checkpoint path)
run_eval "P3_learned_gate" "p3_learned_gate" "none" ",gate_checkpoint=${GATE_CKPT},gate_threshold=0.5"

# PhaseB: Learned Router
run_eval "PhaseB_router" "none" "l2c_learned" ",router_checkpoint=${ROUTER_B_CKPT}"

# PhaseC: Continuous Router
run_eval "PhaseC_continuous" "none" "l2c_continuous" ",router_checkpoint=${ROUTER_C_CKPT}"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Comparing Predictions"
echo "=============================================="

python -c "
import json
import glob
from pathlib import Path

output_dir = '${OUTPUT_DIR}'
phases = ['P0_baseline', 'P2_early_stop', 'P3_learned_gate', 'PhaseB_router', 'PhaseC_continuous']

predictions = {}
for phase in phases:
    pattern = Path(output_dir) / phase / '**' / 'samples_gsm8k_cot*.jsonl'
    files = glob.glob(str(pattern), recursive=True)
    if not files:
        print(f'WARNING: No samples file for {phase}')
        continue
    
    samples = []
    with open(files[0], 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    predictions[phase] = samples
    print(f'{phase}: {len(samples)} samples')

# Compare P0 vs others
print()
print('=' * 60)
p0 = predictions.get('P0_baseline')
if p0:
    for phase in phases[1:]:
        if phase not in predictions:
            continue
        
        same = 0
        for i, (s0, sp) in enumerate(zip(p0, predictions[phase])):
            r0 = s0.get('resps', [[]])[0][0] if s0.get('resps') else ''
            rp = sp.get('resps', [[]])[0][0] if sp.get('resps') else ''
            if r0 == rp:
                same += 1
        
        total = len(p0)
        diff = total - same
        print(f'P0 vs {phase}: {same}/{total} same ({100*same/total:.1f}%), {diff} different')
        
        # Critical check: if P2/P3 is 100% same as P0, early stop is not working!
        if phase in ['P2_early_stop', 'P3_learned_gate'] and diff == 0:
            print(f'  ⚠️  WARNING: {phase} produced identical outputs - early stopping may not be triggering!')

# Compare all pair-wise to show PhaseB/PhaseC are expected to be same
print()
print('PhaseB vs PhaseC (expected same if caching = no change):')
if 'PhaseB_router' in predictions and 'PhaseC_continuous' in predictions:
    same = sum(1 for s1, s2 in zip(predictions['PhaseB_router'], predictions['PhaseC_continuous'])
               if s1.get('resps', [[]])[0][0] == s2.get('resps', [[]])[0][0])
    print(f'  {same}/{len(predictions[\"PhaseB_router\"])} same')
"

echo ""
echo "=============================================="
echo "Sanity Check Complete"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}"
