#!/bin/bash
#SBATCH --job-name=eval_phaseC
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=0:30:00
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Phase C Continuous Router Evaluation

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

SAMPLES=100
BATCH_SIZE=1
DIFFUSION_STEPS=32
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"

echo "====== Phase C: Continuous Router ======"
echo "Start time: $(date)"
nvidia-smi

RESULTS_DIR="external/Dream/eval_instruct/results/PhaseC_${SAMPLES}_${TIMESTAMP}"

cd external/Dream/eval_instruct
python -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16 \
    --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},do_sample=False,temperature=0.001,alg=entropy,alg_temp=0.1,delta_mode=p0_baseline,cache_mode=l2c_continuous,router_checkpoint=${CKPT_DIR}/PhaseC_continuous/checkpoints/continuous_router_final.pt" \
    --tasks gsm8k \
    --num_fewshot 0 \
    --batch_size ${BATCH_SIZE} \
    --limit ${SAMPLES} \
    --output_path ${RESULTS_DIR}

echo "Phase C completed at $(date)"
