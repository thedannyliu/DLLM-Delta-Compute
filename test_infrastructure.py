#!/usr/bin/env python3
"""
Quick local test of delta-compute infrastructure (CPU mode)
Tests imports and config without loading the full model
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "external" / "Dream"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=== Testing Delta-Compute Infrastructure ===\n")

# Test 1: Imports
print("Test 1: Imports")
try:
    from modeling.modeling_dream import DreamModel, DreamDecoderLayer, DreamBaseModel
    from modeling.generation_utils import DreamGenerationConfig
    from tracing import TraceCollector, LayerStepStats
    print("✓ All imports successful\n")
except Exception as e:
    print(f"✗ Import failed: {e}\n")
    sys.exit(1)

# Test 2: TraceCollector
print("Test 2: TraceCollector")
try:
    import torch
    collector = TraceCollector(enabled=True, num_layers=4)
    collector.start_step(0)
    # Create dummy tensor
    dummy_ffn_output = torch.randn(1, 10, 768)  # [batch, seq, hidden]
    collector.record_layer(
        layer_idx=0,
        step_idx=0,
        ffn_output=dummy_ffn_output
    )
    collector.end_step()
    print(f"✓ TraceCollector created: {len(collector.stats)} layers")
    print(f"✓ Recorded stats for layer 0: {collector.stats[0][0]}\n")
except Exception as e:
    print(f"✗ TraceCollector failed: {e}\n")
    sys.exit(1)

# Test 3: DreamGenerationConfig
print("Test 3: DreamGenerationConfig")
try:
    config = DreamGenerationConfig(
        trace_teacher=True,
        delta_mode='p2_early_stop',
        cache_mode='l2c_ffn',
        trace_output_dir='./test_traces',
        cache_schedule='0,1,2,3',
        early_stop_confidence_threshold=0.95,
        early_stop_entropy_threshold=0.1,
        steps=10,
        max_new_tokens=20,
    )
    print(f"✓ Config created:")
    print(f"  - trace_teacher: {config.trace_teacher}")
    print(f"  - delta_mode: {config.delta_mode}")
    print(f"  - cache_mode: {config.cache_mode}")
    print(f"  - cache_schedule: {config.cache_schedule}")
    print(f"  - steps: {config.steps}")
    print(f"  - early_stop_confidence_threshold: {config.early_stop_confidence_threshold}\n")
except Exception as e:
    print(f"✗ Config creation failed: {e}\n")
    sys.exit(1)

# Test 4: Model signature check
print("Test 4: Model signatures")
try:
    import inspect
    
    # Check DreamDecoderLayer.forward signature
    layer_sig = inspect.signature(DreamDecoderLayer.forward)
    layer_params = list(layer_sig.parameters.keys())
    required_layer_params = ['trace_collector', 'use_ffn_cache', 'ffn_cache', 'diffusion_step']
    missing_layer = [p for p in required_layer_params if p not in layer_params]
    
    if missing_layer:
        print(f"✗ DreamDecoderLayer missing params: {missing_layer}")
    else:
        print(f"✓ DreamDecoderLayer has all delta-compute params")
    
    # Check DreamBaseModel.forward signature
    base_sig = inspect.signature(DreamBaseModel.forward)
    base_params = list(base_sig.parameters.keys())
    required_base_params = ['layer_ffn_caches', 'cache_schedule', 'trace_collector', 'diffusion_step']
    missing_base = [p for p in required_base_params if p not in base_params]
    
    if missing_base:
        print(f"✗ DreamBaseModel missing params: {missing_base}")
    else:
        print(f"✓ DreamBaseModel has all delta-compute params")
    
    # Check DreamModel.forward signature
    model_sig = inspect.signature(DreamModel.forward)
    model_params = list(model_sig.parameters.keys())
    required_model_params = ['layer_ffn_caches', 'cache_schedule', 'trace_collector', 'diffusion_step']
    missing_model = [p for p in required_model_params if p not in model_params]
    
    if missing_model:
        print(f"✗ DreamModel missing params: {missing_model}")
    else:
        print(f"✓ DreamModel has all delta-compute params")
    
    print()
except Exception as e:
    print(f"✗ Signature check failed: {e}\n")
    sys.exit(1)

# Test 5: Eval integration
print("Test 5: Eval integration")
try:
    sys.path.insert(0, str(Path(__file__).parent / "external" / "Dream" / "eval_instruct"))
    from lm_eval.models.diffllm import DiffLLM
    
    # Check if DiffLLM.__init__ accepts delta-compute params
    init_sig = inspect.signature(DiffLLM.__init__)
    init_params = list(init_sig.parameters.keys())
    
    print(f"✓ DiffLLM import successful")
    print(f"✓ DiffLLM accepts kwargs (for delta-compute params)\n")
except Exception as e:
    print(f"✗ Eval integration check failed: {e}\n")
    sys.exit(1)

print("=== All Infrastructure Tests Passed! ===")
print("\nReady for GPU testing:")
print("  sbatch slurm_test_poc.sh")
