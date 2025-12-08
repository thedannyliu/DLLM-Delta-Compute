#!/bin/bash
#SBATCH --job-name=p1_base_traces
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/p1_base_traces_%j.out
#SBATCH --error=logs/p1_base_traces_%j.err

# ==============================================================================
# P1 Trace Collection with Base Model for CoT Training
# ==============================================================================
# Uses Base model with official CoT config to collect traces for:
# - P3 learned gate training
# - PhaseB/C/D/E router training
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=${1:-500}
MODEL="Dream-org/Dream-v0-Base-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACES_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces_base_${TIMESTAMP}"

mkdir -p ${TRACES_DIR}
mkdir -p logs

echo "=============================================="
echo "P1 Trace Collection - Base Model"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Traces Output: ${TRACES_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "=============================================="

cd external/Dream/eval_instruct

# Run P1 trace collection with official Base model config
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=p1_traces,cache_mode=none,trace_output_dir=${TRACES_DIR}" \
    --gen_kwargs "do_sample=False,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES}

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

TRACE_COUNT=$(ls ${TRACES_DIR}/*.pt 2>/dev/null | wc -l || echo 0)

echo ""
echo "=============================================="
echo "P1 Trace Collection Complete"
echo "=============================================="
echo "Traces saved to: ${TRACES_DIR}"
echo "Number of traces: ${TRACE_COUNT}"
echo ""
echo "Next steps:"
echo "1. Train P3 gate: sbatch scripts/slurm/train_p3_gate.sh ${TRACES_DIR}"
echo "2. Train PhaseB router: sbatch scripts/slurm/train_phaseB_router.sh ${TRACES_DIR}"
echo "3. Train PhaseC router: sbatch scripts/slurm/train_phaseC_router.sh ${TRACES_DIR}"
