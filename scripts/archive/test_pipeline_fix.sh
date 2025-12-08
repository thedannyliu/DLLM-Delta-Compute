#!/bin/bash
#SBATCH -Jtest_pipeline         # Job name
#SBATCH -N1 --gres=gpu:L40S:1   # 1 node, 1 L40S GPU
#SBATCH --mem-per-gpu=40G       # Memory per GPU
#SBATCH -t0-00:30:00            # 30 mins time limit
#SBATCH -o logs/test_pipeline_%j.out
#SBATCH -p ice-gpu              # Queue name
#SBATCH --account=coc

# Load modules
module load cuda/12.1
module load anaconda3/2023.03

# Activate conda environment
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm

# Verify environment
echo "Python: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

echo "=== 1. Testing P1 Visualization ==="
TRACE_DIR="experiments/P1_traces/traces"
PLOT_DIR="docs/reports/P1_traces_test"
mkdir -p $PLOT_DIR
# Find one trace file
TRACE_FILE=$(ls $TRACE_DIR/*.pt | head -n 1)
if [ -z "$TRACE_FILE" ]; then
    echo "Error: No trace files found in $TRACE_DIR"
else
    echo "Plotting $TRACE_FILE..."
    python tests/visualization/plot_traces.py "$TRACE_FILE" --output_dir "$PLOT_DIR"
    if [ $? -eq 0 ]; then
        echo "✓ P1 Visualization successful"
    else
        echo "✗ P1 Visualization failed"
    fi
fi

echo "=== 2. Testing P3 Eval (Learned Gate) ==="
GATE_CHECKPOINT="$(pwd)/experiments/P3_learned_gate/checkpoints/learned_gate_final.pt"
if [ ! -f "$GATE_CHECKPOINT" ]; then
    echo "Error: P3 checkpoint not found at $GATE_CHECKPOINT"
else
    cd external/Dream/eval_instruct
    PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
        --model diffllm \
        --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,delta_mode="p3_learned_gate",gate_checkpoint="$GATE_CHECKPOINT",gate_threshold=0.5 \
        --tasks gsm8k_cot \
        --device cuda \
        --batch_size 1 \
        --limit 1 \
        --output_path ../../../experiments/P3_learned_gate/results/test_run \
        --confirm_run_unsafe_code \
        --apply_chat_template
    
    if [ $? -eq 0 ]; then
        echo "✓ P3 Eval successful"
    else
        echo "✗ P3 Eval failed"
    fi
    cd ../../..
fi

echo "=== 3. Testing PhaseB Eval (Learned Router) ==="
ROUTER_CHECKPOINT="$(pwd)/experiments/PhaseB_router/checkpoints/router_final.pt"
if [ ! -f "$ROUTER_CHECKPOINT" ]; then
    echo "Error: PhaseB checkpoint not found at $ROUTER_CHECKPOINT"
else
    cd external/Dream/eval_instruct
    PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
        --model diffllm \
        --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,cache_mode="l2c_learned",router_checkpoint="$ROUTER_CHECKPOINT",router_threshold=0.5 \
        --tasks gsm8k_cot \
        --device cuda \
        --batch_size 1 \
        --limit 1 \
        --output_path ../../../experiments/PhaseB_router/results/test_run \
        --confirm_run_unsafe_code \
        --apply_chat_template
    
    if [ $? -eq 0 ]; then
        echo "✓ PhaseB Eval successful"
    else
        echo "✗ PhaseB Eval failed"
    fi
    cd ../../..
fi

echo "=== 4. Testing PhaseC Eval (Continuous Router) ==="
ROUTER_CHECKPOINT="$(pwd)/experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt"
if [ ! -f "$ROUTER_CHECKPOINT" ]; then
    echo "Error: PhaseC checkpoint not found at $ROUTER_CHECKPOINT"
else
    cd external/Dream/eval_instruct
    PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
        --model diffllm \
        --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,cache_mode="l2c_continuous",router_checkpoint="$ROUTER_CHECKPOINT",router_threshold=0.5 \
        --tasks gsm8k_cot \
        --device cuda \
        --batch_size 1 \
        --limit 1 \
        --output_path ../../../experiments/PhaseC_continuous/results/test_run \
        --confirm_run_unsafe_code \
        --apply_chat_template
    
    if [ $? -eq 0 ]; then
        echo "✓ PhaseC Eval successful"
    else
        echo "✗ PhaseC Eval failed"
    fi
    cd ../../..
fi

echo "=== 5. Testing P4 Eval (Adaptive) ==="
cd external/Dream/eval_instruct
PYTHONPATH=../../..:$PYTHONPATH accelerate launch -m lm_eval \
    --model diffllm \
    --model_args pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,delta_mode="p4_adaptive" \
    --tasks gsm8k_cot \
    --device cuda \
    --batch_size 1 \
    --limit 1 \
    --output_path ../../../experiments/P4_adaptive/results/test_run \
    --confirm_run_unsafe_code \
    --apply_chat_template

if [ $? -eq 0 ]; then
    echo "✓ P4 Eval successful"
else
    echo "✗ P4 Eval failed"
fi
cd ../../..

echo "All tests completed."
