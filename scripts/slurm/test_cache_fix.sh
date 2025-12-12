#!/bin/bash
#SBATCH --job-name=test_cache_fix
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=01:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/test_cache_fix_%j.out
#SBATCH --error=logs/test_cache_fix_%j.err

# ==============================================================================
# Minimal Test: Verify FFN Caching Fix Works
# Tests 10 samples with PhaseA caching to verify skip_ratio > 0%
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

export HF_ALLOW_CODE_EVAL=1
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)"

SAMPLES=10
MODEL="Dream-org/Dream-v0-Base-7B"

echo "=============================================="
echo "Minimal Test: Verify FFN Caching Fix"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Test 1: Baseline P0 (should have skip_ratio=0%)
echo ""
echo "====== TEST 1: P0 Baseline (expect skip_ratio=0%) ======"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=none,cache_mode=none" \
    --gen_kwargs "do_sample=False,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES}

# Test 2: PhaseA Caching (should have skip_ratio > 0% NOW)
echo ""
echo "====== TEST 2: PhaseA Caching (expect skip_ratio > 0% with fix) ======"
# Cache every other layer
CACHE_SCHEDULE="1;3;5;7;9;11;13;15;17;19;21;23;25;27;29;31"
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=none,cache_mode=l2c_ffn,cache_schedule=${CACHE_SCHEDULE}" \
    --gen_kwargs "do_sample=False,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES}

echo ""
echo "=============================================="
echo "Minimal Test Complete"
echo "=============================================="
echo "If PhaseA shows skip_ratio > 0%, the fix works!"
echo "Completed: $(date)"
