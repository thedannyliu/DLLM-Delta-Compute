#!/bin/bash
#SBATCH -Jdream_P1              # Job name
#SBATCH -N1 --gres=gpu:L40S:1   # 1 node, 1 L40S GPU
#SBATCH --mem-per-gpu=40G       # Memory per GPU
#SBATCH -t0-00:30:00            # 30 mins time limit
#SBATCH -o experiments/P1_traces/logs/gsm8k_traces_%j.out
#SBATCH -p ice-gpu              # Queue name
#SBATCH --account=coc
              # Account

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Install missing dependencies if needed
pip install -q sacrebleu evaluate scikit-learn sqlitedict word2number pytablewriter 2>&1 | grep -v "Requirement already satisfied" || true

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Setup
MODEL_PATH="Dream-org/Dream-v0-Instruct-7B"
OUTPUT_DIR="experiments/P1_traces/results/gsm8k_$(date +%Y%m%d_%H%M%S)"
TRACE_DIR="experiments/P1_traces/traces"
mkdir -p $TRACE_DIR
mkdir -p $OUTPUT_DIR $TRACE_DIR

echo "=== P1 Teacher Trace Collection ==="
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Traces: $TRACE_DIR"
echo "Started at: $(date)"

# Run with trace collection enabled
# Small-scale test: 100 samples, fixed seed for reproducibility
cd external/Dream/eval_instruct
# Create output directory before tee
mkdir -p "../../../$OUTPUT_DIR"

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=True,trace_output_dir=../../../$TRACE_DIR \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 5 \
    --seed 42 \
    --output_path ../../../$OUTPUT_DIR/results.json \
    --log_samples \
    2>&1 | tee ../../../$OUTPUT_DIR/eval.log

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
echo "Traces saved to: $TRACE_DIR"

# Generate visualizations
echo "Generating trace visualizations..."
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute
python tests/visualization/plot_traces.py $TRACE_DIR/*.pt --output_dir docs/reports/P1_traces/
