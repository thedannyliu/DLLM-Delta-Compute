#!/bin/bash
#SBATCH --job-name=train_P3
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Find the latest traces directory with 500 samples
TRACE_DIR=$(ls -td experiments/P1_traces/traces_500_* 2>/dev/null | head -1)

if [ -z "$TRACE_DIR" ] || [ ! -d "$TRACE_DIR" ]; then
    echo "ERROR: No P1 traces_500 directory found!"
    echo "Expected: experiments/P1_traces/traces_500_*"
    exit 1
fi

echo "Using traces from: $TRACE_DIR"
TRACE_COUNT=$(ls -1 "$TRACE_DIR"/*.pt 2>/dev/null | wc -l)
echo "Found $TRACE_COUNT trace files"

if [ "$TRACE_COUNT" -lt 100 ]; then
    echo "WARNING: Only $TRACE_COUNT traces found, expected ~500"
fi

echo "Generating Oracle Labels..."
python scripts/training/generate_oracle_labels.py \
    --trace_dir "$TRACE_DIR" \
    --output experiments/P3_learned_gate/oracle_labels.json \
    --cosine_threshold 0.99

echo "Training Learned Gate..."
python scripts/training/train_learned_gate.py \
    --trace_dir "$TRACE_DIR" \
    --oracle_labels experiments/P3_learned_gate/oracle_labels.json \
    --output_dir experiments/P3_learned_gate/checkpoints \
    --num_epochs 5
