#!/bin/bash
#SBATCH -Jdream_test           # Job name
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 08:00:00
#SBATCH -o logs/dream_test_poc_%j.out      # Standard output log

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run test
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
python test_poc_v1a.py \
    --model_path hkust-nlp/Dream-7B \
    --output_dir ./test_traces \
    --device cuda

echo "Test completed at $(date)"
