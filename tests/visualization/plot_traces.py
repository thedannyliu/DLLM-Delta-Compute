#!/usr/bin/env python3
"""
Visualization script for P1 teacher traces
Creates heatmaps and plots from saved TraceCollector outputs
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def get_stats_matrix(trace_data, key):
    """Helper to extract a matrix of values [layers, steps] for a given key"""
    stats_dict = trace_data['stats']
    num_layers = trace_data['metadata']['num_layers']
    
    # Find max step index
    max_step = 0
    for layer_data in stats_dict.values():
        for step_idx in layer_data.keys():
            max_step = max(max_step, int(step_idx))
    num_steps = max_step + 1
    
    matrix = np.zeros((num_layers, num_steps))
    # Initialize with NaNs to distinguish missing data
    matrix[:] = np.nan
    
    for layer_idx_str, step_dict in stats_dict.items():
        layer_idx = int(layer_idx_str)
        for step_idx_str, stat in step_dict.items():
            step_idx = int(step_idx_str)
            if key in stat and stat[key] is not None:
                matrix[layer_idx, step_idx] = stat[key]
                
    return matrix, num_layers, num_steps

def plot_ffn_norm_heatmap(trace_data, output_path):
    """Plot layer x step heatmap of FFN output norms"""
    matrix, num_layers, num_steps = get_stats_matrix(trace_data, 'ffn_output_norm')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap='viridis', cbar_kws={'label': 'FFN Output Norm'})
    plt.xlabel('Diffusion Step')
    plt.ylabel('Layer Index')
    plt.title('FFN Output Norms (Layer × Step)')
    plt.tight_layout()
    plt.savefig(output_path / 'ffn_norm_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved FFN norm heatmap to {output_path / 'ffn_norm_heatmap.png'}")

def plot_cosine_sim_heatmap(trace_data, output_path):
    """Plot layer x step heatmap of cosine similarity (skip vs residual)"""
    # Note: key is 'ffn_cosine_sim' in TraceCollector
    matrix, num_layers, num_steps = get_stats_matrix(trace_data, 'ffn_cosine_sim')
    
    # Plot
    plt.figure(figsize=(12, 8))
    sns.heatmap(matrix, cmap='RdYlGn', vmin=-1, vmax=1, 
                cbar_kws={'label': 'Cosine Similarity'})
    plt.xlabel('Diffusion Step')
    plt.ylabel('Layer Index')
    plt.title('Cosine Similarity: Skip vs Residual (Layer × Step)')
    plt.tight_layout()
    plt.savefig(output_path / 'cosine_sim_heatmap.png', dpi=150)
    plt.close()
    print(f"Saved cosine similarity heatmap to {output_path / 'cosine_sim_heatmap.png'}")

def plot_runtime_per_layer(trace_data, output_path):
    """Plot average runtime per layer across all steps"""
    matrix, num_layers, num_steps = get_stats_matrix(trace_data, 'forward_time_ms')
    
    # Compute average runtime per layer (ignoring NaNs)
    avg_runtimes = np.nanmean(matrix, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.bar(range(num_layers), avg_runtimes, color='steelblue')
    plt.xlabel('Layer Index')
    plt.ylabel('Average Runtime (ms)')
    plt.title('Average FFN Runtime per Layer')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'runtime_per_layer.png', dpi=150)
    plt.close()
    print(f"Saved runtime plot to {output_path / 'runtime_per_layer.png'}")

def plot_stability_analysis(trace_data, output_path):
    """
    Plot stability vs cost trade-off:
    - X-axis: Average cosine similarity (stability)
    - Y-axis: Average runtime (cost)
    - Each point is a layer
    """
    cos_matrix, num_layers, _ = get_stats_matrix(trace_data, 'ffn_cosine_sim')
    time_matrix, _, _ = get_stats_matrix(trace_data, 'forward_time_ms')
    
    # Compute per-layer averages
    layer_cos_sims = np.nanmean(cos_matrix, axis=1)
    layer_runtimes = np.nanmean(time_matrix, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(layer_cos_sims, layer_runtimes, 
                         c=range(num_layers), cmap='viridis', 
                         s=100, alpha=0.7, edgecolors='black')
    
    # Annotate a few key layers
    for i in [0, num_layers//4, num_layers//2, 3*num_layers//4, num_layers-1]:
        if not np.isnan(layer_cos_sims[i]) and not np.isnan(layer_runtimes[i]):
            plt.annotate(f'L{i}', (layer_cos_sims[i], layer_runtimes[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    plt.colorbar(scatter, label='Layer Index')
    plt.xlabel('Average Cosine Similarity (Stability)')
    plt.ylabel('Average Runtime (ms) (Cost)')
    plt.title('Stability vs Cost Trade-off per Layer')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'stability_vs_cost.png', dpi=150)
    plt.close()
    print(f"Saved stability vs cost plot to {output_path / 'stability_vs_cost.png'}")

def plot_step_progression(trace_data, output_path):
    """Plot metrics evolution across diffusion steps (averaged over layers)"""
    norm_matrix, _, num_steps = get_stats_matrix(trace_data, 'ffn_output_norm')
    cos_matrix, _, _ = get_stats_matrix(trace_data, 'ffn_cosine_sim')
    
    # Compute step-wise averages
    step_ffn_norms = np.nanmean(norm_matrix, axis=0)
    step_cos_sims = np.nanmean(cos_matrix, axis=0)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # FFN norms
    ax1.plot(range(num_steps), step_ffn_norms, marker='o', color='steelblue')
    ax1.set_xlabel('Diffusion Step')
    ax1.set_ylabel('Avg FFN Norm')
    ax1.set_title('Average FFN Norm across Diffusion Steps')
    ax1.grid(alpha=0.3)
    
    # Cosine similarity
    ax2.plot(range(num_steps), step_cos_sims, marker='o', color='coral')
    ax2.set_xlabel('Diffusion Step')
    ax2.set_ylabel('Avg Cosine Similarity')
    ax2.set_title('Average Cosine Similarity across Diffusion Steps')
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / 'step_progression.png', dpi=150)
    plt.close()
    print(f"Saved step progression plot to {output_path / 'step_progression.png'}")

def print_summary_stats(trace_data):
    """Print summary statistics of the trace"""
    stats_dict = trace_data['stats']
    num_layers = trace_data['metadata']['num_layers']
    
    # Find max step
    max_step = 0
    for layer_data in stats_dict.values():
        for step_idx in layer_data.keys():
            max_step = max(max_step, int(step_idx))
    num_steps = max_step + 1
    
    print("\n=== Trace Summary Statistics ===")
    print(f"Number of layers: {num_layers}")
    print(f"Number of steps: {num_steps}")
    
    # Collect all values
    all_ffn_norms = []
    all_cos_sims = []
    all_runtimes = []
    
    for layer_dict in stats_dict.values():
        for stat in layer_dict.values():
            if 'ffn_output_norm' in stat and stat['ffn_output_norm'] is not None:
                all_ffn_norms.append(stat['ffn_output_norm'])
            if 'ffn_cosine_sim' in stat and stat['ffn_cosine_sim'] is not None:
                all_cos_sims.append(stat['ffn_cosine_sim'])
            if 'forward_time_ms' in stat and stat['forward_time_ms'] is not None:
                all_runtimes.append(stat['forward_time_ms'])
    
    if all_ffn_norms:
        print(f"\nFFN Output Norms:")
        print(f"  Mean: {np.mean(all_ffn_norms):.4f}")
        print(f"  Std:  {np.std(all_ffn_norms):.4f}")
        print(f"  Min:  {np.min(all_ffn_norms):.4f}")
        print(f"  Max:  {np.max(all_ffn_norms):.4f}")
    
    if all_cos_sims:
        print(f"\nCosine Similarities:")
        print(f"  Mean: {np.mean(all_cos_sims):.4f}")
        print(f"  Std:  {np.std(all_cos_sims):.4f}")
        print(f"  Min:  {np.min(all_cos_sims):.4f}")
        print(f"  Max:  {np.max(all_cos_sims):.4f}")
    
    if all_runtimes:
        print(f"\nRuntimes (ms):")
        print(f"  Mean: {np.mean(all_runtimes):.4f}")
        print(f"  Std:  {np.std(all_runtimes):.4f}")
        print(f"  Min:  {np.min(all_runtimes):.4f}")
        print(f"  Max:  {np.max(all_runtimes):.4f}")
        print(f"  Total: {np.sum(all_runtimes):.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('trace_files', type=str, nargs='+',
                       help='Path to saved trace .pt file(s)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save plots (default: same as trace file)')
    args = parser.parse_args()
    
    for trace_file in args.trace_files:
        # Load trace
        trace_path = Path(trace_file)
        if not trace_path.exists():
            print(f"Warning: Trace file not found: {trace_path}")
            continue
        
        print(f"\nProcessing trace: {trace_path}")
        try:
            trace_data = torch.load(trace_path)
        except Exception as e:
            print(f"Error loading {trace_path}: {e}")
            continue
        
        # Determine output directory
        if args.output_dir:
            output_path = Path(args.output_dir) / trace_path.stem
        else:
            output_path = trace_path.parent / 'plots' / trace_path.stem
        
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving plots to: {output_path}")
        
        # Print summary
        try:
            print_summary_stats(trace_data)
            
            # Generate plots
            plot_ffn_norm_heatmap(trace_data, output_path)
            plot_cosine_sim_heatmap(trace_data, output_path)
            plot_runtime_per_layer(trace_data, output_path)
            plot_stability_analysis(trace_data, output_path)
            plot_step_progression(trace_data, output_path)
            
            print(f"✓ Plots generated for {trace_path.name}")
        except Exception as e:
            print(f"Error processing {trace_path.name}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
