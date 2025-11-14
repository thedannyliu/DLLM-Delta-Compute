#!/usr/bin/env python3
"""
Minimal GPU test - only test imports and CUDA availability
No model download required
"""

import sys
import torch
from pathlib import Path

print("=== Minimal GPU Test ===\n")

# Test 1: Check CUDA
print("Test 1: CUDA Availability")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  Device count: {torch.cuda.device_count()}")
    print(f"  Device name: {torch.cuda.get_device_name(0)}")
    print(f"  Device memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
print()

# Test 2: Imports
print("Test 2: Delta-Compute Imports")
sys.path.insert(0, str(Path(__file__).parent / "external" / "Dream"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from modeling.modeling_dream import DreamModel, DreamDecoderLayer
    from modeling.generation_utils import DreamGenerationConfig
    from tracing import TraceCollector
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 3: Config
print("\nTest 3: DreamGenerationConfig")
config = DreamGenerationConfig(
    trace_teacher=True,
    delta_mode='p2_early_stop',
    cache_mode='l2c_ffn',
    cache_schedule='0,1,2,3',
)
print(f"  ✓ trace_teacher: {config.trace_teacher}")
print(f"  ✓ delta_mode: {config.delta_mode}")
print(f"  ✓ cache_mode: {config.cache_mode}")

# Test 4: TraceCollector on GPU
if torch.cuda.is_available():
    print("\nTest 4: TraceCollector on GPU")
    collector = TraceCollector(enabled=True, num_layers=4)
    dummy_tensor = torch.randn(1, 10, 768).cuda()
    collector.start_step(0)
    collector.record_layer(0, 0, dummy_tensor)
    collector.end_step()
    print(f"  ✓ TraceCollector works on GPU")
    print(f"  ✓ Recorded norm: {collector.stats[0][0].ffn_output_norm:.2f}")

print("\n=== All Tests Passed! ===")
print("\nEnvironment is ready for full evaluation.")
print("To run full test with model:")
print("  1. Login to HuggingFace: huggingface-cli login")
print("  2. Or use local model path: --model_path /path/to/model")
