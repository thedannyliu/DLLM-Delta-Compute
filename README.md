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
bash scripts/setup/setup_env.sh

# Or manually:
conda create -n dcllm python=3.10
conda activate dcllm
pip install -r requirements.txt
```

### 3. Run Quick Test
```bash
# GPU test (submit SLURM job)
sbatch scripts/slurm/slurm_test_poc.sh

# Or run unit tests locally
python tests/unit/test_infrastructure.py
```

### 4. Full Evaluation
See [TESTING_GUIDE.md](TESTING_GUIDE.md) for comprehensive instructions.

## Project Structure

```
DLLM-Delta-Compute/
├── src/                         # Core infrastructure
│   ├── tracing.py              # TraceCollector for P1 statistics
│   └── __init__.py
├── tests/                       # All test scripts
│   ├── unit/                   # Unit tests (GPU validation, infrastructure)
│   ├── integration/            # Integration tests (full workflows)
│   └── visualization/          # Trace plotting and analysis
├── scripts/                     # Executable scripts
│   ├── slurm/                  # SLURM batch jobs (eval_P0_baseline.sh, etc.)
│   ├── setup/                  # Environment setup
│   └── eval/                   # Local evaluation scripts
├── experiments/                 # Experiment outputs (organized by phase)
│   ├── P0_baseline/            # Baseline experiments
│   ├── P1_traces/              # Trace collection
│   ├── P2_early_stop/          # Early stopping
│   └── PhaseA_caching/         # FFN caching
│   # Each contains: logs/, traces/, results/, configs/
├── docs/                        # Documentation
│   ├── master_plan.md          # **PRIMARY**: Project specification
│   ├── implementation_summary.md  # Implementation details
│   ├── reports/                # Validation reports
│   └── archive/                # Historical docs
├── external/                    # External dependencies (git submodules)
│   ├── Dream/                  # Dream-7B (modified for delta-compute)
│   └── learning-to-cache/      # L2C reference implementation
├── requirements.txt             # Python dependencies
├── TESTING_GUIDE.md            # Testing instructions
└── README.md                   # This file
```

For detailed structure and organization rules, see [docs/master_plan.md](docs/master_plan.md).

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

### Quick Validation (8 hours)
```bash
sbatch scripts/slurm/slurm_test_poc.sh
# Check: experiments/P0_baseline/logs/ for SLURM output
```

### Phase-Specific Evaluation (12 hours each)
```bash
# P0: Baseline
sbatch scripts/slurm/eval_P0_baseline.sh

# P1: Trace Collection
sbatch scripts/slurm/eval_P1_traces.sh

# P2: Early Stopping
sbatch scripts/slurm/eval_P2_early_stop.sh

# Phase A: FFN Caching
sbatch scripts/slurm/eval_PhaseA_caching.sh
```

### Visualization
```bash
python tests/visualization/plot_traces.py \
  experiments/P1_traces/traces/*.pt \
  --output-dir docs/reports/P1_traces/
```

## Documentation

- **[docs/master_plan.md](docs/master_plan.md)**: ⭐ Primary specification (P0-P4, Phase A-C, progress tracking)
- **[docs/implementation_summary.md](docs/implementation_summary.md)**: Implementation details and architecture
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)**: How to run tests and evaluations
- **[docs/reports/](docs/reports/)**: Validation and evaluation reports
- **[docs/README.md](docs/README.md)**: Documentation organization guide

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
