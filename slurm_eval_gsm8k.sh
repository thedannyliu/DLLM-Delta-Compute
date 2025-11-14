#!/bin/bash
#SBATCH -Jdream_gsm8k          # Job name
#SBATCH -N1 --gres=gpu:H100:1  # 1 node, 1 H100 GPU
#SBATCH --mem-per-gpu=80G      # Memory per GPU
#SBATCH -t0-12:00:00           # 12 hours time limit
#SBATCH -ogsm8k_eval_%j.out    # Standard output log
#SBATCH -qinfinity             # Queue name
#SBATCH -A GT-hl94             # Account

# Load modules
module load cuda/12.1

# Activate conda environment
source ~/.bashrc
conda activate dcllm

cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Arguments
MODEL_PATH=${1:-"hkust-nlp/Dream-7B"}
MODE=${2:-"baseline"}  # baseline, trace, cache, early_stop
OUTPUT_DIR="./eval_results/${MODE}_$(date +%Y%m%d_%H%M%S)"

echo "Running GSM8K evaluation: MODE=$MODE"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"

mkdir -p $OUTPUT_DIR

# Build model_args based on mode
case $MODE in
    baseline)
        MODEL_ARGS="pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=False"
        ;;
    trace)
        MODEL_ARGS="pretrained=$MODEL_PATH,delta_mode=none,cache_mode=none,trace_teacher=True,trace_output_dir=$OUTPUT_DIR/traces"
        mkdir -p $OUTPUT_DIR/traces
        ;;
    cache)
        MODEL_ARGS="pretrained=$MODEL_PATH,delta_mode=none,cache_mode=l2c_ffn,cache_schedule=0,1,2,3,trace_teacher=False"
        ;;
    early_stop)
        MODEL_ARGS="pretrained=$MODEL_PATH,delta_mode=p2_early_stop,cache_mode=none,trace_teacher=False"
        ;;
    *)
        echo "Unknown mode: $MODE"
        exit 1
        ;;
esac

# Run evaluation
cd external/Dream/eval_instruct

python -m lm_eval \
    --model diffllm \
    --model_args $MODEL_ARGS \
    --tasks gsm8k \
    --num_fewshot 5 \
    --batch_size 1 \
    --output_path $OUTPUT_DIR/results.json \
    --log_samples \
    2>&1 | tee $OUTPUT_DIR/eval.log

echo "Evaluation completed at $(date)"
echo "Results saved to: $OUTPUT_DIR"
