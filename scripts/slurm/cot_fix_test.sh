#!/bin/bash
#SBATCH --job-name=cot_fix_test
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=2:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/cot_fix_test_%j.out
#SBATCH --error=logs/cot_fix_test_%j.err

# ==============================================================================
# CoT Configuration Fix Test
# ==============================================================================
# Tests the FIXED configuration with:
# - max_new_tokens=256 in MODEL_ARGS (not gen_kwargs!)
# - diffusion_steps=256 in MODEL_ARGS
# - Official CoT settings: 8-shot, greedy, gsm8k_cot
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration - ALIGNED WITH OFFICIAL
SAMPLES=10
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="experiments/cot_fix_test_${TIMESTAMP}"
TIMING_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/reports/timing/cot_fix_test_${TIMESTAMP}"

mkdir -p ${OUTPUT_DIR}
mkdir -p ${TIMING_DIR}
mkdir -p logs

echo "=============================================="
echo "CoT Configuration Fix Test"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "Timing: ${TIMING_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "KEY FIX: max_new_tokens=256 in model_args (not gen_kwargs!)"
echo "=============================================="

cd external/Dream/eval_instruct

# P0 Baseline with FIXED config
echo ""
echo "====== P0 Baseline (FIXED CONFIG) ======"
echo "model_args: max_new_tokens=256,diffusion_steps=256"
echo ""

python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=none,cache_mode=none,timing_log_path=${TIMING_DIR}/P0_baseline.json" \
    --gen_kwargs "do_sample=False,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --log_samples \
    --output_path "${OUTPUT_DIR}/P0_baseline"

echo ""
echo "P0 Result:"
cat ${OUTPUT_DIR}/P0_baseline/*/results_*.json 2>/dev/null | python -c "import sys,json; d=json.load(sys.stdin); print('  Flex-Extract:', d.get('results',{}).get('gsm8k_cot',{}).get('exact_match,flexible-extract','N/A')); print('  Strict-Match:', d.get('results',{}).get('gsm8k_cot',{}).get('exact_match,strict-match','N/A'))" || echo "  Results parsing failed"

echo ""
echo "P0 Timing:"
cat ${TIMING_DIR}/P0_baseline.json 2>/dev/null || echo "  No timing log found"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "Test Complete"
echo "=============================================="
echo "Results: ${OUTPUT_DIR}"
echo "Timing: ${TIMING_DIR}"
