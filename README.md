# DLLM Delta-Compute & L2C Caching for Dream-7B

Accelerating diffusion language models through delta-compute and learning-to-cache techniques.

## Overview

This project implements inference acceleration for **Dream-7B** (a diffusion-based LLM) using:
- **P1 Teacher Traces**: Collect per-layer/per-step statistics to identify redundant computation
- **P2 Early Stopping**: Terminate diffusion early when output stabilizes
- **L2C FFN Caching**: Cache FFN outputs in stable layers to reduce compute

**Goal**: Reduce inference latency by 30-50% with <2% accuracy drop on GSM8K.

## Quick Start

### 1. Clone with Submodules
```bash
git clone --recursive git@github.com:thedannyliu/DLLM-Delta-Compute.git
cd DLLM-Delta-Compute
```

### 2. Setup Environment
```bash
# On PACE ICE cluster
./setup_env.sh

# Or manually:
conda create -n dcllm python=3.10
conda activate dcllm
pip install torch==2.5.1 transformers==4.46.2 accelerate matplotlib seaborn
```

### 3. Run Quick Test
```bash
# CPU test (development)
python test_poc_v1a.py --device cpu

# GPU test (submit SLURM job)
sbatch slurm_test_poc.sh
```

### 4. Full Evaluation
See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive instructions.

## Project Structure

```
DLLM-Delta-Compute/
├── src/
│   ├── tracing.py               # TraceCollector for P1 statistics
│   └── __init__.py
├── external/
│   ├── Dream/                   # Dream-7B submodule (modified)
│   │   ├── modeling/
│   │   │   ├── modeling_dream.py      # Modified with tracing/caching hooks
│   │   │   └── generation_utils.py    # Extended config & _sample loop
│   │   └── eval_instruct/
│   │       └── lm_eval/models/diffllm.py  # Eval wrapper with model_args
│   └── learning-to-cache/       # Reference L2C implementation
├── docs/
│   ├── master_plan.md           # Project specification
│   ├── development_log.md       # Implementation progress
│   └── implementation_issues.md # Known issues & blockers
├── test_poc_v1a.py              # Quick 3-mode validation
├── plot_traces.py               # Trace visualization
├── slurm_test_poc.sh            # Quick GPU test (2hr)
├── slurm_eval_gsm8k.sh          # Full GSM8K eval (12hr)
├── TESTING_GUIDE.md             # Comprehensive usage guide
└── README.md                    # This file
```

## Features

### Implemented (POC v1a)

- ✅ **P1 Teacher Traces**: Collect FFN norms, cosine similarities, runtimes
- ✅ **P2 Early Stopping**: Confidence + entropy thresholds
- ✅ **L2C FFN Caching**: Alternating schedule, configurable layers
- ✅ **Eval Integration**: Full lm-eval harness support
- ✅ **Visualization**: Heatmaps, stability analysis, runtime plots
- ✅ **SLURM Jobs**: Ready for PACE ICE cluster

### Pending

- ⏳ GPU validation and baseline metrics
- ⏳ Trace analysis on 50+ GSM8K samples
- ⏳ Cache schedule optimization (grid search)
- ⏳ Full GSM8K evaluation in all modes
- ⏳ Speedup measurements (wall-clock + FLOPs)

## Configuration Modes

| Mode | Description | Key Parameters |
|------|-------------|----------------|
| **baseline** | Unmodified Dream-7B | `delta_mode=none, cache_mode=none` |
| **trace** | Teacher with P1 tracing | `trace_teacher=True, trace_output_dir=./traces` |
| **cache** | L2C FFN caching | `cache_mode=l2c_ffn, cache_schedule=0,1,2,3` |
| **early_stop** | P2 early stopping | `delta_mode=p2_early_stop, confidence_threshold=0.95` |

## Git Workflow

This project uses a **submodule workflow**:

1. Make changes in `external/Dream/`
2. Commit changes **inside the submodule**:
   ```bash
   cd external/Dream
   git add <files>
   git commit -m "[Feature] Description"
   ```
3. Update main repo to reference new submodule commit:
   ```bash
   cd ../..
   git add external/Dream
   git commit -m "Update Dream submodule to <commit>"
   ```

### Branches

- `main`: Stable releases
- `PoC-1`: Current development branch (POC v1a + v1b)

## Testing

### Quick Validation (2 hours)
```bash
sbatch slurm_test_poc.sh
# Check: test_traces/ for saved traces
# Check: test_poc_*.out for logs
```

### Full GSM8K Eval (12 hours)
```bash
# Baseline
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B baseline

# With caching
sbatch slurm_eval_gsm8k.sh hkust-nlp/Dream-7B cache
```

### Visualization
```bash
python plot_traces.py test_traces/trace_sample_*.pt --output_dir ./plots
```

## Documentation

- [TESTING_GUIDE.md](TESTING_GUIDE.md): How to run tests and evaluations
- [docs/master_plan.md](docs/master_plan.md): Project specification and phases
- [docs/development_log.md](docs/development_log.md): Implementation progress tracking
- [docs/implementation_issues.md](docs/implementation_issues.md): Known blockers

## Requirements

- **Hardware**: NVIDIA GPU with ≥40GB VRAM (H100 recommended)
- **Software**:
  - Python 3.10
  - PyTorch 2.5.1 (CUDA 12.1)
  - Transformers 4.46.2
  - Accelerate 0.34.2
- **Cluster**: PACE ICE with SLURM scheduler

## Citation

If you use this code, please cite:

```bibtex
@misc{dllm-delta-compute,
  title={Delta-Compute Acceleration for Diffusion Language Models},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/thedannyliu/DLLM-Delta-Compute}}
}
```

## References

- **Dream-7B**: [arXiv:2406.11989](https://arxiv.org/abs/2406.11989)
- **Learning-to-Cache (L2C)**: DiT/U-ViT layer caching techniques
- **Delta Denoising Score**: Efficient diffusion sampling

## License

See [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
