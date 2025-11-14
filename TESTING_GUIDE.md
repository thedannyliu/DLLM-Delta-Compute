# Delta-Compute POC v1a Testing Guide

## Quick Start

### 1. Local CPU Test (Development)
```bash
# Quick syntax check
python tests/integration/test_poc_v1a.py --device cpu --model_path hkust-nlp/Dream-7B
```

### 2. GPU Quick Test (2 hours)
```bash
# Submit SLURM job for quick validation
sbatch scripts/slurm/slurm_test_poc.sh

# Monitor progress
tail -f test_poc_*.out

# Check results
ls -lh experiments/P1_traces/traces/
```

### 3. Full GSM8K Evaluation (12 hours)
```bash
# Run baseline
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B baseline

# Run with tracing
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B trace

# Run with FFN caching
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B cache

# Run with early stopping
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B early_stop
```

### 4. Visualize Traces
```bash
# After tracing run completes, visualize results
python tests/visualization/plot_traces.py experiments/P1_traces/traces/trace_sample_*.pt --output_dir ./plots

# View generated plots
ls plots/
# -> ffn_norm_heatmap.png
# -> cosine_sim_heatmap.png
# -> runtime_per_layer.png
# -> stability_vs_cost.png
# -> step_progression.png
```

## Configuration Options

### Model Args (for eval scripts)
```bash
# Baseline teacher (no acceleration)
MODEL_ARGS="pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=none,trace_teacher=False"

# Teacher with P1 tracing
MODEL_ARGS="pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=none,trace_teacher=True,trace_output_dir=./traces"

# L2C Phase A FFN caching (cache first 4 layers on even steps)
MODEL_ARGS="pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=l2c_ffn,cache_schedule=0,1,2,3"

# P2 early stopping
MODEL_ARGS="pretrained=hkust-nlp/Dream-7B,delta_mode=p2_early_stop,cache_mode=none,early_stop_confidence_threshold=0.95"
```

### Generation Config Parameters
- `trace_teacher`: Enable P1 teacher trace collection
- `delta_mode`: `none` | `p2_early_stop`
- `cache_mode`: `none` | `l2c_ffn`
- `trace_output_dir`: Directory to save trace .pt files
- `cache_schedule`: Comma-separated layer indices (e.g., "0,1,2,3")
- `early_stop_confidence_threshold`: Confidence threshold for P2 (default: 0.95)
- `early_stop_entropy_threshold`: Entropy threshold for P2 (default: 0.1)

## Expected Outputs

### test_poc_v1a.py
```
=== Test 1: Baseline Teacher ===
Generated: [model output]

=== Test 2: Teacher with Tracing ===
✓ Trace saved to experiments/P1_traces/traces/trace_sample_123456.pt
  Trace stats: 32 layers, 10 steps

=== Test 3: FFN Caching ===
Generated: [model output with caching]

=== Verifying Output Parity ===
✓ Outputs are IDENTICAL (perfect parity)
```

### Trace File Structure
```python
{
    'layer_stats': List[List[Dict]],  # [num_layers][num_steps]
    # Each dict contains:
    # - 'ffn_output_norm': float
    # - 'cosine_sim': float (skip vs residual)
    # - 'runtime_ms': float
}
```

## Troubleshooting

### Import Error: TraceCollector not found
Make sure `src/` is in your Python path:
```bash
export PYTHONPATH=/home/hice1/eliu354/scratch/Projects/DLLM-Delta-Compute/src:$PYTHONPATH
```

### CUDA Out of Memory
Reduce batch size or max_new_tokens:
```bash
python tests/integration/test_poc_v1a.py --device cuda --batch_size 1
```

### SLURM Job Not Starting
Check queue status:
```bash
squeue -u $USER
sacct -j <job_id> --format=JobID,JobName,State,ExitCode
```

## Next Steps

1. **Baseline Metrics**: Run baseline eval to establish reference accuracy/speed
2. **Trace Analysis**: Collect and visualize P1 traces on 50+ samples
3. **Cache Schedule Optimization**: Try different layer combinations (e.g., "0,1,2,3" vs "28,29,30,31")
4. **Early Stopping Tuning**: Adjust confidence/entropy thresholds for P2
5. **Full Evaluation**: Run all modes on full GSM8K test set
6. **Speedup Measurement**: Compare wall-clock time and FLOPs across modes

## File Descriptions

- `test_poc_v1a.py`: Quick validation script (3 modes)
- `slurm_test_poc.sh`: SLURM job for quick GPU test
- `slurm_eval_gsm8k.sh`: SLURM job for full GSM8K evaluation
- `plot_traces.py`: Visualization script for trace analysis
- `src/tracing.py`: TraceCollector implementation
- `external/Dream/modeling/generation_utils.py`: Extended generation config & _sample loop
- `external/Dream/modeling/modeling_dream.py`: Modified model with delta-compute hooks
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`: Eval wrapper with model_args parsing
