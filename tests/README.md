# Tests Directory

This directory contains all test scripts for the DLLM Delta-Compute project.

## Structure

```
tests/
├── unit/                  # Unit tests for individual components
│   ├── test_gpu_minimal.py          # Minimal GPU validation (no model download)
│   └── test_infrastructure.py       # Infrastructure component tests
├── integration/           # Integration tests for full workflows
│   └── test_poc_v1a.py              # Full 3-mode validation (baseline, trace, cache)
└── visualization/         # Visualization and analysis scripts
    └── plot_traces.py               # Trace visualization (5 plot types)
```

## Running Tests

### Unit Tests

**Minimal GPU Test** (no model download required):
```bash
python tests/unit/test_gpu_minimal.py
```

**Infrastructure Test** (local, no GPU):
```bash
python tests/unit/test_infrastructure.py
```

### Integration Tests

**Full POC Test** (requires HuggingFace login and model access):
```bash
python tests/integration/test_poc_v1a.py
```

### Visualization

**Plot Traces** (requires trace .pt files):
```bash
python tests/visualization/plot_traces.py <trace_file.pt> --output-dir reports/
```

Available plot types:
- `ffn_norm_heatmap`: Layer × Step FFN output norms
- `cosine_sim_heatmap`: Layer × Step cosine similarities
- `runtime_per_layer`: Per-layer forward time
- `stability_vs_cost`: Stability (cosine sim) vs runtime scatter
- `step_progression`: Mean metrics evolution across steps

## Test Coverage

- ✅ GPU availability and CUDA setup
- ✅ Delta-compute infrastructure imports
- ✅ TraceCollector functionality
- ✅ DreamGenerationConfig extensions
- ✅ Model loading and generation
- ✅ Baseline mode (no acceleration)
- ✅ Trace collection mode
- ✅ FFN caching mode
- ⏳ Full GSM8K evaluation (requires SLURM job)
