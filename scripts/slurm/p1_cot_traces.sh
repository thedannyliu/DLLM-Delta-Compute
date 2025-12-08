#!/bin/bash
#SBATCH --job-name=p1_cot_traces
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/p1_cot_traces_%j.out
#SBATCH --error=logs/p1_cot_traces_%j.err

# ==============================================================================
# P1 Trace Collection for CoT Training (Official Config)
# ==============================================================================
# Collects P1 traces with OFFICIAL CoT settings for retraining:
# - P3 learned gate
# - PhaseB learned router
# - PhaseC continuous router
# - PhaseD skip router
# - PhaseE flexi adapter
# ==============================================================================

set -e

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration - ALIGNED WITH OFFICIAL
SAMPLES=${1:-500}  # Default 500, can override
MODEL="Dream-org/Dream-v0-Instruct-7B"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
TRACES_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces_cot_${TIMESTAMP}"

mkdir -p ${TRACES_DIR}
mkdir -p logs

echo "=============================================="
echo "P1 Trace Collection for CoT - Official Config"
echo "=============================================="
echo "Model: ${MODEL}"
echo "Samples: ${SAMPLES}"
echo "Traces Output: ${TRACES_DIR}"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "Official Config:"
echo "  max_new_tokens=256"
echo "  diffusion_steps=256"
echo "  temperature=0.0, top_p=0.95"
echo "  add_bos_token=true"
echo "  gsm8k_cot, 8-shot"
echo "=============================================="

cd external/Dream/eval_instruct

# Run P1 trace collection with OFFICIAL config (max_new_tokens in model_args!)
python -m lm_eval \
    --model diffllm \
    --model_args "pretrained=${MODEL},trust_remote_code=True,dtype=bfloat16,max_new_tokens=256,diffusion_steps=256,temperature=0.0,top_p=0.95,add_bos_token=True,delta_mode=p1_traces,cache_mode=none,trace_output_dir=${TRACES_DIR}" \
    --gen_kwargs "do_sample=False,alg=entropy" \
    --tasks gsm8k_cot \
    --num_fewshot 8 \
    --batch_size 1 \
    --limit ${SAMPLES}

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo ""
echo "=============================================="
echo "P1 Trace Collection Complete"
echo "=============================================="
echo "Traces saved to: ${TRACES_DIR}"
echo "Number of traces: $(ls ${TRACES_DIR}/*.pt 2>/dev/null | wc -l || echo 0)"
echo ""
echo "Next steps:"
echo "1. Train P3 gate: scripts/slurm/train_p3_gate.sh ${TRACES_DIR}"
echo "2. Train PhaseB router: scripts/slurm/train_phaseB_router.sh ${TRACES_DIR}"
echo "3. Train PhaseC router: scripts/slurm/train_phaseC_router.sh ${TRACES_DIR}"
