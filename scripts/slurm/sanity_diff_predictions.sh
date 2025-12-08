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
# Sanity Check: Diff P3 vs PhaseB vs PhaseC Predictions
# ==============================================================================
# Uses --log_samples to capture per-sample predictions, then diffs results
# to verify that routers are actually changing predictions vs degenerating to P3/P0
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=50
DIFFUSION_STEPS=128
NUM_FEWSHOT=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/sanity_diff_${TIMESTAMP}"
CKPT_DIR="experiments"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs

echo "=============================================="
echo "Sanity Check: P3 vs PhaseB vs PhaseC Predictions"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Samples: ${SAMPLES}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation with sample logging
run_eval_with_samples() {
    local NAME=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local RESULTS_DIR="${OUTPUT_DIR}/${NAME}"
    
    echo ""
    echo "====== ${NAME} ======"
    echo "delta_mode: ${DELTA_MODE}, cache_mode: ${CACHE_MODE}"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},add_bos_token=True${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=256,do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size 1 \
        --limit ${SAMPLES} \
        --log_samples \
        --output_path "${RESULTS_DIR}"
    
    echo "Completed: $(date)"
}

# P3: Learned Gate
run_eval_with_samples "P3_learned_gate" "p3_learned_gate" "none" \
    ",gate_checkpoint=${CKPT_DIR}/P3_learned_gate/checkpoints/learned_gate_final.pt"

# PhaseB: Learned Router
run_eval_with_samples "PhaseB_router" "none" "l2c_learned" \
    ",router_checkpoint=${CKPT_DIR}/PhaseB_router/checkpoints/router_final.pt"

# PhaseC: Continuous Router
run_eval_with_samples "PhaseC_continuous" "none" "l2c_continuous" \
    ",router_checkpoint=${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Comparing Predictions"
echo "=============================================="

# Python script to compare predictions
python -c "
import json
import os
from pathlib import Path

output_dir = '${OUTPUT_DIR}'
phases = ['P3_learned_gate', 'PhaseB_router', 'PhaseC_continuous']

# Load predictions for each phase
predictions = {}
for phase in phases:
    samples_file = Path(output_dir) / phase / 'samples_gsm8k_cot_*.jsonl'
    import glob
    files = glob.glob(str(samples_file))
    if not files:
        print(f'WARNING: No samples file found for {phase}')
        continue
    
    samples = []
    with open(files[0], 'r') as f:
        for line in f:
            samples.append(json.loads(line))
    predictions[phase] = samples
    print(f'{phase}: {len(samples)} samples loaded')

# Compare predictions pairwise
if len(predictions) >= 2:
    phase_names = list(predictions.keys())
    print(f'\\nComparing predictions:')
    print('=' * 60)
    
    for i in range(len(phase_names)):
        for j in range(i+1, len(phase_names)):
            p1, p2 = phase_names[i], phase_names[j]
            
            if p1 not in predictions or p2 not in predictions:
                continue
            
            same = 0
            diff = 0
            
            for idx, (s1, s2) in enumerate(zip(predictions[p1], predictions[p2])):
                # Compare the 'resps' field which contains model outputs
                resp1 = s1.get('resps', [[]])[0][0] if s1.get('resps') else ''
                resp2 = s2.get('resps', [[]])[0][0] if s2.get('resps') else ''
                
                if resp1 == resp2:
                    same += 1
                else:
                    diff += 1
                    if diff <= 5:  # Print first 5 differences
                        print(f'\\nSample {idx} differs:')
                        print(f'  {p1}: {resp1[:100]}...' if len(resp1) > 100 else f'  {p1}: {resp1}')
                        print(f'  {p2}: {resp2[:100]}...' if len(resp2) > 100 else f'  {p2}: {resp2}')
            
            total = same + diff
            print(f'\\n{p1} vs {p2}:')
            print(f'  Same: {same}/{total} ({100*same/total:.1f}%)')
            print(f'  Different: {diff}/{total} ({100*diff/total:.1f}%)')
else:
    print('Not enough predictions to compare')
"

echo ""
echo "=============================================="
echo "Sanity Check Complete"
echo "=============================================="
echo "Results saved to: ${OUTPUT_DIR}"
