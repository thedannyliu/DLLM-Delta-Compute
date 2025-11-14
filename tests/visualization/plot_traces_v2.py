#!/usr/bin/env python3
"""
Visualization script for P1 teacher traces.
Creates heatmaps and plots from saved TraceCollector outputs (.pt format).
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

def load_trace(trace_path):
    """Load trace data from .pt file."""
    data = torch.load(trace_path, map_location='cpu')
    
    # Parse metadata
    metadata = data['metadata']
    num_layers = metadata['num_layers']
    
    # Parse stats into numpy arrays
    stats_dict = data['stats']
    
    # Find dimensions
    max_step = 0
    for layer_data in stats_dict.values():
        for step_idx in layer_data.keys():
            max_step = max(max_step, int(step_idx))
    num_steps = max_step + 1
    
    # Extract arrays
    ffn_norms = np.full((num_layers, num_steps), np.nan)
    cosine_sims = np.full((num_layers, num_steps), np.nan)
    runtimes = np.full((num_layers, num_steps), np.nan)
    
    for layer_idx_str, step_dict in stats_dict.items():
        layer_idx = int(layer_idx_str)
        for step_idx_str, stat in step_dict.items():
            step_idx = int(step_idx_str)
            ffn_norms[layer_idx, step_idx] = stat['ffn_output_norm']
            if stat['ffn_cosine_sim'] is not None:
                cosine_sims[layer_idx, step_idx] = stat['ffn_cosine_sim']
            if stat['forward_time_ms'] is not None:
                runtimes[layer_idx, step_idx] = stat['forward_time_ms']
    
    return {
        'num_layers': num_layers,
        'num_steps': num_steps,
        'ffn_norms': ffn_norms,
        'cosine_sims': cosine_sims,
        'runtimes': runtimes,
        'metadata': metadata
    }

def plot_ffn_norm_heatmap(data, output_path):
    """Plot layer x step heatmap of FFN output norms."""
    plt.figure(figsize=(14, 8))
    sns.heatmap(data['ffn_norms'], cmap='viridis', cbar_kws={'label': 'L2 Norm'})
    plt.xlabel('Diffusion Step', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.title('FFN Output Norms (Layer × Step)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / 'ffn_norm_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_cosine_sim_heatmap(data, output_path):
    """Plot layer x step heatmap of cosine similarity."""
    plt.figure(figsize=(14, 8))
    sns.heatmap(data['cosine_sims'], cmap='RdYlGn', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Cosine Similarity'})
    plt.xlabel('Diffusion Step', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.title('FFN Output Cosine Similarity (vs Previous Step)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / 'cosine_sim_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_runtime_heatmap(data, output_path):
    """Plot layer x step heatmap of forward pass runtime."""
    if np.all(np.isnan(data['runtimes'])):
        print("  ⚠ Skipping runtime heatmap (no timing data)")
        return
        
    plt.figure(figsize=(14, 8))
    sns.heatmap(data['runtimes'], cmap='YlOrRd', cbar_kws={'label': 'Time (ms)'})
    plt.xlabel('Diffusion Step', fontsize=12)
    plt.ylabel('Layer Index', fontsize=12)
    plt.title('Forward Pass Runtime (Layer × Step)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = output_path / 'runtime_heatmap.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_layer_averages(data, output_path):
    """Plot average metrics per layer."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Average FFN norm per layer
    layer_ffn_avg = np.nanmean(data['ffn_norms'], axis=1)
    axes[0].bar(range(data['num_layers']), layer_ffn_avg, color='steelblue', alpha=0.7)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Average FFN Norm')
    axes[0].set_title('Average FFN Norm per Layer')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Average cosine sim per layer
    layer_cos_avg = np.nanmean(data['cosine_sims'], axis=1)
    axes[1].bar(range(data['num_layers']), layer_cos_avg, color='coral', alpha=0.7)
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Average Cosine Similarity')
    axes[1].set_title('Average Cosine Similarity per Layer')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    
    # Average runtime per layer
    if not np.all(np.isnan(data['runtimes'])):
        layer_runtime_avg = np.nanmean(data['runtimes'], axis=1)
        axes[2].bar(range(data['num_layers']), layer_runtime_avg, color='green', alpha=0.7)
        axes[2].set_xlabel('Layer Index')
        axes[2].set_ylabel('Average Runtime (ms)')
        axes[2].set_title('Average Runtime per Layer')
        axes[2].grid(axis='y', alpha=0.3)
    else:
        axes[2].text(0.5, 0.5, 'No timing data', ha='center', va='center',
                    transform=axes[2].transAxes, fontsize=14)
        axes[2].set_title('Average Runtime per Layer')
    
    plt.tight_layout()
    save_path = output_path / 'layer_averages.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def plot_step_progression(data, output_path):
    """Plot how metrics evolve across steps."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # FFN norm progression
    step_ffn_avg = np.nanmean(data['ffn_norms'], axis=0)
    axes[0].plot(range(data['num_steps']), step_ffn_avg, marker='o', 
                color='steelblue', linewidth=2, markersize=4)
    axes[0].set_xlabel('Diffusion Step')
    axes[0].set_ylabel('Average FFN Norm')
    axes[0].set_title('FFN Norm Evolution Across Diffusion Steps')
    axes[0].grid(alpha=0.3)
    
    # Cosine sim progression
    step_cos_avg = np.nanmean(data['cosine_sims'], axis=0)
    axes[1].plot(range(data['num_steps']), step_cos_avg, marker='o',
                color='coral', linewidth=2, markersize=4)
    axes[1].set_xlabel('Diffusion Step')
    axes[1].set_ylabel('Average Cosine Similarity')
    axes[1].set_title('Cosine Similarity Evolution Across Diffusion Steps')
    axes[1].axhline(y=0, color='gray', linestyle='--', linewidth=1)
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    save_path = output_path / 'step_progression.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {save_path}")

def print_summary(data):
    """Print summary statistics."""
    print("\n" + "="*60)
    print("TRACE SUMMARY STATISTICS")
    print("="*60)
    
    print(f"\nDimensions:")
    print(f"  Layers: {data['num_layers']}")
    print(f"  Steps:  {data['num_steps']}")
    print(f"  Total:  {data['num_layers'] * data['num_steps']} measurements")
    
    print(f"\nFFN Output Norms:")
    ffn = data['ffn_norms'][~np.isnan(data['ffn_norms'])]
    print(f"  Mean:   {np.mean(ffn):.4f}")
    print(f"  Std:    {np.std(ffn):.4f}")
    print(f"  Min:    {np.min(ffn):.4f}")
    print(f"  Max:    {np.max(ffn):.4f}")
    print(f"  Median: {np.median(ffn):.4f}")
    
    print(f"\nCosine Similarities:")
    cos = data['cosine_sims'][~np.isnan(data['cosine_sims'])]
    if len(cos) > 0:
        print(f"  Mean:   {np.mean(cos):.4f}")
        print(f"  Std:    {np.std(cos):.4f}")
        print(f"  Min:    {np.min(cos):.4f}")
        print(f"  Max:    {np.max(cos):.4f}")
        print(f"  Median: {np.median(cos):.4f}")
    else:
        print("  No data available")
    
    if not np.all(np.isnan(data['runtimes'])):
        print(f"\nForward Pass Runtimes (ms):")
        rt = data['runtimes'][~np.isnan(data['runtimes'])]
        print(f"  Mean:   {np.mean(rt):.4f}")
        print(f"  Std:    {np.std(rt):.4f}")
        print(f"  Min:    {np.min(rt):.4f}")
        print(f"  Max:    {np.max(rt):.4f}")
        print(f"  Total:  {np.sum(rt):.2f} ms")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize Dream model traces from P1 evaluation'
    )
    parser.add_argument('trace_file', type=str,
                       help='Path to trace .pt file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: trace_file_plots/)')
    args = parser.parse_args()
    
    # Validate input
    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(f"❌ Error: Trace file not found: {trace_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = trace_path.parent / f"{trace_path.stem}_plots"
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"TRACE VISUALIZATION")
    print(f"{'='*60}")
    print(f"Input:  {trace_path}")
    print(f"Output: {output_path}")
    print(f"{'='*60}\n")
    
    # Load trace
    print("Loading trace data...")
    try:
        data = load_trace(trace_path)
        print(f"  ✓ Loaded trace with {data['num_layers']} layers, {data['num_steps']} steps")
    except Exception as e:
        print(f"❌ Error loading trace: {e}")
        sys.exit(1)
    
    # Print summary
    print_summary(data)
    
    # Generate plots
    print("Generating visualizations...")
    plot_ffn_norm_heatmap(data, output_path)
    plot_cosine_sim_heatmap(data, output_path)
    plot_runtime_heatmap(data, output_path)
    plot_layer_averages(data, output_path)
    plot_step_progression(data, output_path)
    
    print(f"\n{'='*60}")
    print(f"✓ All visualizations generated successfully!")
    print(f"{'='*60}\n")

if __name__ == '__main__':
    main()
