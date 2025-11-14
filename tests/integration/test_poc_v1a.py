#!/usr/bin/env python3
"""
Test script for POC v1a: P0 + P1 Teacher Traces
Tests:
1. Baseline teacher (delta_mode=none, cache_mode=none)
2. Teacher with tracing (trace_teacher=True)
3. Verify parity between runs
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add external/Dream to path
sys.path.insert(0, str(Path(__file__).parent / "external" / "Dream"))
sys.path.insert(0, str(Path(__file__).parent / "src"))

from modeling.modeling_dream import DreamModel
from modeling.generation_utils import DreamGenerationConfig
from transformers import AutoTokenizer

def test_baseline(model, tokenizer, device):
    """Test baseline teacher without any acceleration"""
    print("\n=== Test 1: Baseline Teacher ===")
    
    prompt = "Question: If there are 3 apples and you take 2, how many do you have?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    gen_config = DreamGenerationConfig(
        max_new_tokens=50,
        steps=10,  # Reduced for fast testing
        temperature=0.0,
        alg='origin',
        trace_teacher=False,
        delta_mode='none',
        cache_mode='none',
    )
    
    print(f"Input: {prompt}")
    print(f"Config: steps={gen_config.steps}, temp={gen_config.temperature}")
    
    with torch.no_grad():
        outputs = model.diffusion_generate(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=gen_config,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    return outputs

def test_with_tracing(model, tokenizer, device, output_dir):
    """Test teacher with P1 tracing enabled"""
    print("\n=== Test 2: Teacher with Tracing ===")
    
    prompt = "Question: If there are 3 apples and you take 2, how many do you have?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    gen_config = DreamGenerationConfig(
        max_new_tokens=50,
        steps=10,
        temperature=0.0,
        alg='origin',
        trace_teacher=True,
        delta_mode='none',
        cache_mode='none',
        trace_output_dir=output_dir,
    )
    
    print(f"Input: {prompt}")
    print(f"Config: steps={gen_config.steps}, tracing={gen_config.trace_teacher}")
    
    with torch.no_grad():
        outputs = model.diffusion_generate(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=gen_config,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    # Check if trace was saved
    trace_files = list(Path(output_dir).glob("trace_sample_*.pt"))
    if trace_files:
        print(f"✓ Trace saved to {trace_files[0]}")
        trace = torch.load(trace_files[0])
        print(f"  Trace stats: {len(trace['layer_stats'])} layers, {len(trace['layer_stats'][0])} steps")
    else:
        print("✗ No trace file found!")
    
    return outputs

def test_ffn_caching(model, tokenizer, device):
    """Test L2C Phase A FFN caching"""
    print("\n=== Test 3: FFN Caching ===")
    
    prompt = "Question: If there are 3 apples and you take 2, how many do you have?\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    # Cache first 4 layers on even steps
    gen_config = DreamGenerationConfig(
        max_new_tokens=50,
        steps=10,
        temperature=0.0,
        alg='origin',
        trace_teacher=False,
        delta_mode='none',
        cache_mode='l2c_ffn',
        cache_schedule='0,1,2,3',
    )
    
    print(f"Input: {prompt}")
    print(f"Config: cache_mode={gen_config.cache_mode}, schedule={gen_config.cache_schedule}")
    
    with torch.no_grad():
        outputs = model.diffusion_generate(
            inputs=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            generation_config=gen_config,
        )
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Generated: {generated_text}")
    
    return outputs

def verify_parity(outputs1, outputs2):
    """Verify two outputs are identical"""
    print("\n=== Verifying Output Parity ===")
    
    if torch.equal(outputs1, outputs2):
        print("✓ Outputs are IDENTICAL (perfect parity)")
        return True
    else:
        diff = (outputs1 != outputs2).sum().item()
        print(f"✗ Outputs differ in {diff} positions")
        print(f"  Output 1: {outputs1[:20]}")
        print(f"  Output 2: {outputs2[:20]}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='hkust-nlp/Dream-7B',
                        help='Path or HuggingFace ID of Dream model')
    parser.add_argument('--output_dir', type=str, default='./test_traces',
                        help='Directory to save trace outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    print(f"Using device: {args.device}")
    print(f"Loading model: {args.model_path}")
    
    # Load model and tokenizer
    model = DreamModel.from_pretrained(args.model_path).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model.eval()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run tests
    baseline_outputs = test_baseline(model, tokenizer, args.device)
    traced_outputs = test_with_tracing(model, tokenizer, args.device, args.output_dir)
    cached_outputs = test_ffn_caching(model, tokenizer, args.device)
    
    # Verify parity
    verify_parity(baseline_outputs, traced_outputs)
    
    print("\n=== Test Summary ===")
    print("✓ All tests completed!")
    print(f"Traces saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
