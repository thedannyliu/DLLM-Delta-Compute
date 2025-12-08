#!/bin/bash
#SBATCH --job-name=fair_poc_cot
#SBATCH --account=coc
#SBATCH --partition=ice-gpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=nvidia-gpu
#SBATCH --time=16:00:00
#SBATCH --mem=384G
#SBATCH --output=logs/fair_poc_cot_%j.out
#SBATCH --error=logs/fair_poc_cot_%j.err

# Load environment
module load cuda/12.1
module load anaconda3/2023.03
source /usr/local/pace-apps/manual/packages/anaconda3/2023.03/etc/profile.d/conda.sh
conda activate dcllm
cd /home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute

# Configuration
SAMPLES=1000
BATCH_SIZE=1
DIFFUSION_STEPS=256
NUM_FEWSHOT=8
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
CKPT_DIR="/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/experiments"
RESULTS_BASE="external/Dream/eval_instruct/results/fair_eval_cot_${TIMESTAMP}"

mkdir -p ${RESULTS_BASE}
mkdir -p reports/timing/fair_eval_cot_${TIMESTAMP}

echo "=============================================="
echo "Fair PoC CoT Evaluation (Official Settings)"
echo "=============================================="
echo "Timestamp: ${TIMESTAMP}"
echo "Samples: ${SAMPLES}"
echo "Diffusion Steps: ${DIFFUSION_STEPS}"
echo "Model: Dream-v0-Instruct-7B"
echo "=============================================="

cd external/Dream/eval_instruct

# Function to run evaluation
run_eval() {
    local PHASE=$1
    local DELTA_MODE=$2
    local CACHE_MODE=$3
    local EXTRA_ARGS=$4
    local TIMING_LOG="${CKPT_DIR}/../reports/timing/fair_eval_cot_${TIMESTAMP}/${PHASE}.json"
    local RESULTS_DIR="${RESULTS_BASE}/${PHASE}"
    
    echo ""
    echo "====== ${PHASE} ======"
    echo "delta_mode: ${DELTA_MODE}"
    echo "cache_mode: ${CACHE_MODE}"
    
    python -m lm_eval \
        --model diffllm \
        --model_args "pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,dtype=bfloat16,delta_mode=${DELTA_MODE},cache_mode=${CACHE_MODE},timing_log_path=${TIMING_LOG},add_bos_token=True${EXTRA_ARGS}" \
        --gen_kwargs "diffusion_steps=${DIFFUSION_STEPS},max_new_tokens=256,do_sample=False,temperature=0.0,top_p=0.95,alg=entropy" \
        --tasks gsm8k_cot \
        --num_fewshot ${NUM_FEWSHOT} \
        --batch_size ${BATCH_SIZE} \
        --limit ${SAMPLES} \
        --output_path ${RESULTS_DIR}
}

# P0: Baseline
run_eval "P0_baseline" "none" "none" ""

# P1: Traces (Collect traces for training)
run_eval "P1_traces" "p1_traces" "none" ",trace_output_dir=${CKPT_DIR}/P1_traces/cot_eval_${TIMESTAMP}"

# P2: Early Stop (Need to tune thresholds, using conservative defaults for now)
# Note: CoT might need different thresholds.
run_eval "P2_early_stop" "p2_early_stop" "none" ",early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1"

# P3/PhaseB/C: Placeholder until trained
echo "Skipping P3/PhaseB/C until trained on CoT traces"
