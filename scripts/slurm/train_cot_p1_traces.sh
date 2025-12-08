#!/bin/bash
#SBATCH --job-name=train_cot_p1
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/train_cot_p1_%j.out
#SBATCH --error=logs/train_cot_p1_%j.err

# ==============================================================================
# P1 Trace Collection for CoT Training
# ==============================================================================
# Collects P1 traces with CoT settings to train P3 gate and PhaseB/C routers.
# Uses GSM8K train split (first 500 samples to avoid compute explosion).
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=500
DIFFUSION_STEPS=256
MAX_NEW_TOKENS=256
NUM_FEWSHOT=8
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACES_DIR="experiments/P1_traces_cot_${TIMESTAMP}"

mkdir -p ${TRACES_DIR}
mkdir -p logs

echo "=============================================="
echo "P1 Trace Collection for CoT"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Diffusion Steps: ${DIFFUSION_STEPS}"
echo "Traces Output: ${TRACES_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Run P1 trace collection
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,delta_mode=p1_traces,cache_mode=none,add_bos_token=True,trace_output_dir=${TRACES_DIR}" \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=${MAX_NEW_TOKENS},do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot ${NUM_FEWSHOT} \
    --batch_size 1 \
    --limit ${SAMPLES} \
    --output_path "${TRACES_DIR}/eval_results"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "P1 Trace Collection Complete"
echo "=============================================="
echo "Traces saved to: ${TRACES_DIR}"
echo "Number of traces: $(ls ${TRACES_DIR}/*.pt 2>/dev/null | wc -l || echo 0)"
