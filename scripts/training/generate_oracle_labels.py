#!/usr/bin/env python3
# coding=utf-8
"""
Generate oracle labels for P3 Learned Gate training.

This script analyzes P1 traces to determine which (layer, step) pairs
are safe to freeze without hurting quality.

Usage:
    python scripts/training/generate_oracle_labels.py \
        --trace_dir experiments/P1_traces/traces \
        --output experiments/P3_learned_gate/oracle_labels.json \
        --cosine_threshold 0.99 \
        --norm_threshold 0.1
"""

import argparse
import json
import sys
from pathlib import Path
from glob import glob
import torch
import numpy as np
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from tracing import TraceCollector, LayerStepStats

def main():
    parser = argparse.ArgumentParser(description="Generate oracle labels for P3 gate training")
    parser.add_argument('--trace_dir', type=str, required=True,
                       help='Directory containing P1 trace .pt files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output JSON file for oracle labels')
    parser.add_argument('--cosine_threshold', type=float, default=0.99,
                       help='Minimum cosine similarity to consider safe')
    parser.add_argument('--norm_threshold', type=float, default=0.1,
                       help='Maximum norm change ratio to consider safe')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Oracle Label Generation for P3 Learned Gate")
    print("=" * 80)
    print(f"Trace directory: {args.trace_dir}")
    print(f"Output file: {args.output}")
    print(f"Cosine threshold: {args.cosine_threshold}")
    print(f"Norm threshold: {args.norm_threshold}")
    print()
    
    # Find trace files
    trace_files = glob(f"{args.trace_dir}/*.pt")
    print(f"Found {len(trace_files)} trace files")
    
    if len(trace_files) == 0:
        print("ERROR: No trace files found!")
        return 1
    
    # Aggregate stats
    print("Aggregating statistics...")
    
    # Store lists of metrics for each (layer, step)
    # Key: (layer_idx, step_idx)
    cosine_sims = defaultdict(list)
    
    max_layer = 0
    max_step = 0
    
    for i, trace_file in enumerate(trace_files):
        if i % 10 == 0:
            print(f"Processing file {i+1}/{len(trace_files)}...", end='\r')
            
        try:
            collector = TraceCollector.load(trace_file)
            stats = collector.get_all_stats()
            
            for layer_idx, step_dict in stats.items():
                max_layer = max(max_layer, layer_idx)
                for step_idx, layer_stats in step_dict.items():
                    max_step = max(max_step, step_idx)
                    
                    # Cosine similarity
                    if layer_stats.ffn_cosine_sim is not None:
                        cosine_sims[(layer_idx, step_idx)].append(layer_stats.ffn_cosine_sim)
                    
        except Exception as e:
            print(f"\nError reading {trace_file}: {e}")
            continue
            
    print(f"\nProcessed {len(trace_files)} files.")
    
    # Generate labels
    print("Generating labels...")
    oracle_labels = {}
    
    safe_count = 0
    total_count = 0
    
    # Iterate over all possible (layer, step) pairs
    for layer_idx in range(max_layer + 1):
        for step_idx in range(max_step + 1):
            key = (layer_idx, step_idx)
            
            sims = cosine_sims.get(key, [])
            
            if not sims:
                # No data, assume unsafe
                label = 0.0
            else:
                avg_sim = np.mean(sims)
                
                # Heuristic: High cosine similarity -> Safe to freeze
                if avg_sim >= args.cosine_threshold:
                    label = 1.0
                    safe_count += 1
                else:
                    label = 0.0
            
            # Store as string key "layer_step"
            oracle_labels[f"{layer_idx}_{step_idx}"] = label
            total_count += 1
            
    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(oracle_labels, f, indent=2)
    
    print(f"✓ Generated {len(oracle_labels)} oracle labels")
    print(f"  Positive (safe to freeze): {safe_count} ({100*safe_count/total_count:.1f}%)")
    print(f"  Negative (must recompute): {total_count - safe_count} ({100*(total_count - safe_count)/total_count:.1f}%)")
    print(f"\nSaved to: {args.output}")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
