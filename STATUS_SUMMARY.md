# DLLM Delta-Compute PoC Results Summary

**Date:** November 30, 2025  
**Cluster:** Georgia Tech PACE ICE  
**Evaluation:** GSM8K (100 samples, batch_size=1, diffusion_steps=32)

## Overview

All 8 PoC phases have been implemented and evaluated. This document summarizes the results.

## Evaluation Results

| Phase | Description | GSM8K Accuracy | Mean Latency | Status |
|-------|-------------|----------------|--------------|--------|
| **P0** | Baseline (full diffusion) | 51.0% | ~1.2s | ✅ Complete |
| **P1** | Teacher Traces | 46.0% | ~1.2s | ✅ Complete |
| **P2** | Rule-Based Early Stop | 46.0% | ~0.9s | ✅ Complete |
| **P3** | Learned Gate | **56.0%** | 0.546s | ✅ Complete |
| **P4** | Adaptive Stride | 38.0% | ~0.8s | ✅ Complete |
| **Phase A** | Heuristic FFN Caching | 45.0% | 1.169s | ✅ Complete |
| **Phase B** | Learned Router | **56.0%** | 0.548s | ✅ Complete |
| **Phase C** | Continuous Router | 53.0% | 0.540s | ✅ Complete |

## Key Findings

### Best Performers (Accuracy)
1. **P3 Learned Gate** and **Phase B Learned Router**: 56.0% (5% improvement over baseline)
2. **Phase C Continuous Router**: 53.0% (2% improvement over baseline)

### Best Performers (Speed)
1. **Phase C Continuous Router**: 0.540s mean latency (2.2x speedup)
2. **P3 Learned Gate**: 0.546s mean latency (2.2x speedup)
3. **Phase B Learned Router**: 0.548s mean latency (2.2x speedup)

### Observations
- **P3, Phase B, Phase C** all achieve significant speedups (~2x) while maintaining or improving accuracy
- **P4 Adaptive Stride** shows the most aggressive acceleration but with notable accuracy loss (-13%)
- **Phase A Heuristic Caching** shows minimal benefit over baseline in this evaluation
- **P1 and P2** show slight accuracy degradation compared to P0 baseline

## Trained Models

| Model | Checkpoint Location | Size |
|-------|---------------------|------|
| P3 LearnedGate | `experiments/P3_learned_gate/checkpoints/learned_gate_final.pt` | 9.2 KB |
| Phase B Router | `experiments/PhaseB_router/checkpoints/router_final.pt` | 34.6 KB |
| Phase C Router | `experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt` | 62.1 KB |

## Dependencies Verified

Per master_plan.md requirements:
- ✅ P0 baseline established
- ✅ P1 traces collected (100 training traces)
- ✅ P2 and P3 depend on P1 traces
- ✅ Phase A, B, C use Learning-to-Cache approach
- ✅ All phases use same evaluation protocol

## Bug Fixes Applied

1. **cache_schedule delimiter**: Fixed from comma to semicolon separator in `generation_utils.py`
2. **SLURM GPU constraint**: Added `nvidia-gpu` constraint to avoid AMD MI210 GPUs

## Recommendations

1. **Production Use**: P3 (Learned Gate) or Phase B (Learned Router) offer the best balance of speed and accuracy
2. **Maximum Speed**: Phase C offers slightly faster inference with minimal accuracy trade-off
3. **Conservative**: P0 baseline for maximum accuracy when speed is not critical

## Reproduction

```bash
# Submit evaluation jobs
sbatch scripts/slurm/eval_p3.sh
sbatch scripts/slurm/eval_phaseA.sh
sbatch scripts/slurm/eval_phaseB.sh
sbatch scripts/slurm/eval_phaseC.sh
```

## Git Commits

- `0086386` - Add working eval scripts with nvidia-gpu constraint
- Previous commits documented in git history
