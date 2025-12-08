#!/bin/bash
#SBATCH --job-name=test_phaseA_fix
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=0:15:00
#SBATCH --mem=80G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "Testing Phase A fix with 3 samples..."
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

CACHE_SCHEDULE="1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31"

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p0_baseline,cache_mode=l2c_ffn,cache_schedule=${CACHE_SCHEDULE} \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 3 \
    --output_path results/test_phaseA_fix_$(date +%Y%m%d_%H%M%S)

echo "Test completed: $(date)"
