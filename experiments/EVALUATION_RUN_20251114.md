# Small-Scale Evaluation Summary

**Date**: November 14, 2025  
**Branch**: PoC-1  
**Configuration**: 100 samples, seed=42, A100 GPU

## Submitted Jobs

All jobs submitted successfully with proper dependencies:

| Phase | Job ID | Status | GPU | Dependencies | Expected Runtime |
|-------|--------|--------|-----|--------------|------------------|
| **P0 Baseline** | 3549896 | Running | A100 | None | 2-3 hours |
| **P1 Traces** | 3549897 | Running | A100 | None | 2-3 hours |
| **P2 Early Stop** | 3549898 | Pending | A100 | Wait for P0 | 1-2 hours |
| **Phase A Caching** | 3549899 | Pending | A100 | Wait for P1 | 2-3 hours |

## Configuration Details

### Common Settings
- **Dataset**: GSM8K
- **Sample Size**: 100 (--limit 100)
- **Random Seed**: 42 (--seed 42)
- **Few-shot**: 5
- **Batch Size**: 1
- **Model**: hkust-nlp/Dream-7B

This ensures all phases evaluate on **identical data** for fair comparison.

### Phase-Specific Settings

**P0 - Baseline**:
```bash
--model_args pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=none,trace_teacher=False
```

**P1 - Teacher Traces**:
```bash
--model_args pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=none,trace_teacher=True,trace_output_dir=experiments/P1_traces/traces
```

**P2 - Early Stopping**:
```bash
--model_args pretrained=hkust-nlp/Dream-7B,delta_mode=p2_early_stop,cache_mode=none,trace_teacher=False,early_stop_confidence_threshold=0.95,early_stop_entropy_threshold=0.1
```

**Phase A - FFN Caching**:
```bash
--model_args pretrained=hkust-nlp/Dream-7B,delta_mode=none,cache_mode=l2c_ffn,trace_teacher=False,cache_schedule=0,1,2,3
```

## Output Locations

### Logs (SLURM outputs)
```
experiments/P0_baseline/logs/gsm8k_baseline_3549896.out
experiments/P1_traces/logs/gsm8k_traces_3549897.out
experiments/P2_early_stop/logs/gsm8k_early_stop_3549898.out
experiments/PhaseA_caching/logs/gsm8k_caching_3549899.out
```

### Results (JSON metrics)
```
experiments/P0_baseline/results/gsm8k_20251114_*/results.json
experiments/P1_traces/results/gsm8k_20251114_*/results.json
experiments/P2_early_stop/results/gsm8k_20251114_*/results.json
experiments/PhaseA_caching/results/gsm8k_20251114_*/results.json
```

### Traces (P1 only)
```
experiments/P1_traces/traces/*.pt
```

## Monitoring Commands

### Check job status
```bash
squeue -u $USER
```

### View live logs
```bash
# P0 Baseline
tail -f experiments/P0_baseline/logs/gsm8k_baseline_3549896.out

# P1 Traces
tail -f experiments/P1_traces/logs/gsm8k_traces_3549897.out

# P2 Early Stop
tail -f experiments/P2_early_stop/logs/gsm8k_early_stop_3549898.out

# Phase A Caching
tail -f experiments/PhaseA_caching/logs/gsm8k_caching_3549899.out
```

### Check results
```bash
# View results summary
cat experiments/P0_baseline/results/*/results.json | jq '.results.gsm8k'
cat experiments/P1_traces/results/*/results.json | jq '.results.gsm8k'
cat experiments/P2_early_stop/results/*/results.json | jq '.results.gsm8k'
cat experiments/PhaseA_caching/results/*/results.json | jq '.results.gsm8k'
```

## Expected Metrics

### P0 - Baseline (Reference)
- **Accuracy**: Target baseline accuracy on 100 samples
- **Latency**: Mean generation time per sample
- **Steps**: Full diffusion steps (no early stopping)

### P1 - Traces
- **Accuracy**: Should match P0 (no acceleration)
- **Traces**: Layer × Step statistics saved to .pt files
- **Overhead**: ~10-20% slower due to tracing

### P2 - Early Stopping
- **Accuracy**: Target ≤2% drop vs P0
- **Latency**: Target 30-50% reduction
- **Steps**: Average steps before early stop

### Phase A - FFN Caching
- **Accuracy**: Target ≤3% drop vs P0
- **Latency**: Target 20-30% reduction
- **Cache Hits**: Fraction of steps using cached FFN outputs

## Analysis Plan

After all jobs complete:

1. **Baseline Metrics** (from P0):
   - Accuracy, latency, memory usage
   - Establish reference for comparisons

2. **Trace Analysis** (from P1):
   ```bash
   python tests/visualization/plot_traces.py \
     experiments/P1_traces/traces/*.pt \
     --output-dir docs/reports/P1_traces/
   ```
   - Generate heatmaps (FFN norms, cosine similarities)
   - Identify stable vs unstable layers/steps
   - Optimize cache schedules for Phase A

3. **Performance Comparison**:
   - Create comparison table: P0 vs P2 vs Phase A
   - Plot accuracy vs speedup tradeoffs
   - Identify best configuration

4. **Next Steps Decision**:
   - If P2/Phase A show promise: proceed to P3/Phase B
   - If quality drop too large: refine thresholds, try different schedules
   - If speedup insufficient: analyze bottlenecks, optimize implementation

## Files Modified

### Configuration
- `scripts/slurm/eval_P0_baseline.sh` - Added --limit 100, --seed 42
- `scripts/slurm/eval_P1_traces.sh` - Added --limit 100, --seed 42
- `scripts/slurm/eval_P2_early_stop.sh` - Added --limit 100, --seed 42
- `scripts/slurm/eval_PhaseA_caching.sh` - Added --limit 100, --seed 42

### Infrastructure
- `scripts/slurm/submit_all_phases.sh` - Batch submission with dependencies
- `scripts/slurm/verify_setup.sh` - Pre-flight checks

### Git Commits
- `ea8ff39`: Configure small-scale evaluation (100 samples, seed=42)
- `9243338`: Update SLURM configuration for coc-gpu partition
- `d54408c`: Correct output paths in evaluation scripts

## Notes

- All jobs use identical data (--limit 100, --seed 42) for fair comparison
- P2 and Phase A wait for P0/P1 to complete (dependency tracking)
- Logs and results are organized by phase in `experiments/` directory
- Traces from P1 will guide optimization of Phase A cache schedules
- A100 GPUs used (40GB HBM2, coc-gpu partition)

## Status

✅ All jobs submitted successfully  
⏳ Estimated completion: 4-6 hours total (parallel + dependencies)  
📊 Results will be available in `experiments/*/results/` directories
