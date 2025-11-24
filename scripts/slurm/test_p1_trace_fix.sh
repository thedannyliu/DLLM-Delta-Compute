#!/bin/bash
#SBATCH --job-name=test_p1_fix
#SBATCH --account=coc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:L40S:1
#SBATCH --time=0:30:00
#SBATCH --mem=40G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ==================================================
# Test P1 Trace Generation Fix
# ==================================================

module load cuda/12.1
module load anaconda3/2023.03

source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

echo "====================================="
echo "Testing P1 Trace Generation Fix"
echo "====================================="
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "Started: $(date)"
echo ""

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/external/Dream/eval_instruct

TRACE_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces/test_fix_$(date +%Y%m%d_%H%M%S)"

echo "Trace output directory: $TRACE_DIR"
echo "Testing with 3 samples..."
echo ""

python -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,delta_mode=p1_traces,trace_output_dir=${TRACE_DIR},cache_mode=none \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --limit 3 \
    --output_path results/test_p1_fix_$(date +%Y%m%d_%H%M%S)

echo ""
echo "====================================="
echo "Checking trace files..."
echo "====================================="

if [ -d "$TRACE_DIR" ]; then
    TRACE_COUNT=$(ls -1 "$TRACE_DIR"/*.pt 2>/dev/null | wc -l)
    echo "✓ Trace directory exists: $TRACE_DIR"
    echo "✓ Trace files generated: $TRACE_COUNT"
    
    if [ "$TRACE_COUNT" -ge 3 ]; then
        echo "✓✓✓ SUCCESS! P1 trace generation is working!"
        ls -lh "$TRACE_DIR"/*.pt
        exit 0
    else
        echo "✗ FAILED: Expected 3 trace files, got $TRACE_COUNT"
        exit 1
    fi
else
    echo "✗ FAILED: Trace directory not created"
    exit 1
fi
