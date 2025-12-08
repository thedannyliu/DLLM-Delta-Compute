#!/bin/bash
#SBATCH --job-name=sanity_diff
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=4:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/sanity_diff_%j.out
#SBATCH --error=logs/sanity_diff_%j.err

# ==============================================================================
# Sanity Check: Compare Predictions Across Phases
# ==============================================================================
# Uses --log_samples to save predictions and diff them to verify phases
# are producing different outputs (not all identical)
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

SAMPLES=50
MODEL="Dream-org/Dream-v0-Base-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_BASE="experiments/sanity_diff_${TIMESTAMP}"

mkdir -p ${OUTPUT_BASE}
mkdir -p logs

echo "=============================================="
echo "Sanity Check: Prediction Comparison"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Output: ${OUTPUT_BASE}"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run eval with log_samples
run_with_samples() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local OUTPUT_DIR="${OUTPUT_BASE}/${NAME}"
    
    echo ""
    echo "====== ${NAME} ======"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE}${EXTRA_ARGS}" \
        --gen_kwargs "do_sample=False,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot 8 \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --log_samples \
        --output_path "${OUTPUT_DIR}" || echo "WARNING: ${NAME} failed"
}

# Run P0 (baseline)
run_with_samples "P0_baseline" "none" "none" ""

# Run PhaseA (caching)
CACHE_LAYERS=$(seq -s ';' 1 2 63)
run_with_samples "PhaseA_caching" "none" "l2c_ffn" ",cache_schedule=${CACHE_LAYERS}"

# Run PhaseC (continuous router)
PHASEC_ROUTER=$(ls -t experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt 2>/dev/null | head -1)
if [ -f "${PHASEC_ROUTER}" ]; then
    run_with_samples "PhaseC_continuous" "none" "l2c_continuous" ",router_checkpoint_path=${PHASEC_ROUTER}"
fi

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Comparing Predictions"
echo "=============================================="

# Extract predictions and compare
python3 << 'EOF'
import json
import os
from pathlib import Path

output_base = os.environ.get('OUTPUT_BASE', 'experiments/sanity_diff_latest')

phases = ['P0_baseline', 'PhaseA_caching', 'PhaseC_continuous']
predictions = {}

for phase in phases:
    phase_dir = Path(output_base) / phase
    if not phase_dir.exists():
        print(f"Skipping {phase}: directory not found")
        continue
    
    # Find the samples file
    for f in phase_dir.glob("**/*samples*.jsonl"):
        preds = []
        with open(f) as fp:
            for line in fp:
                try:
                    d = json.loads(line)
                    preds.append(d.get('resps', [['']])[0][0] if isinstance(d.get('resps'), list) else '')
                except:
                    preds.append('')
        predictions[phase] = preds
        print(f"Loaded {len(preds)} predictions from {phase}")
        break

# Compare predictions
if len(predictions) >= 2:
    phases_list = list(predictions.keys())
    for i, p1 in enumerate(phases_list):
        for p2 in phases_list[i+1:]:
            preds1 = predictions[p1]
            preds2 = predictions[p2]
            if len(preds1) != len(preds2):
                print(f"\n{p1} vs {p2}: Different lengths ({len(preds1)} vs {len(preds2)})")
                continue
            
            same = sum(1 for a, b in zip(preds1, preds2) if a == b)
            diff = len(preds1) - same
            print(f"\n{p1} vs {p2}: {same} same, {diff} different ({diff/len(preds1)*100:.1f}% differ)")
else:
    print("Not enough phases to compare")
EOF

echo ""
echo "=============================================="
echo "Sanity Check Complete"
echo "=============================================="
echo "Results in: ${OUTPUT_BASE}"
