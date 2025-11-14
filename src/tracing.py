# coding=utf-8
"""
Tracing utilities for Dream model to collect per-step/per-layer statistics.
Used for P1 Teacher Traces analysis.
"""

import time
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
import json
from pathlib import Path


@dataclass
class LayerStepStats:
    """Statistics for a single layer at a single diffusion step."""
    layer_idx: int
    step_idx: int
    ffn_output_norm: float
    attention_output_norm: Optional[float] = None
    ffn_cosine_sim: Optional[float] = None  # vs previous step
    attention_cosine_sim: Optional[float] = None
    forward_time_ms: Optional[float] = None


class TraceCollector:
    """
    Collects and manages tracing statistics across diffusion steps.
    
    Usage:
        collector = TraceCollector(enabled=True, num_layers=32)
        
        # In diffusion loop:
        for step in range(steps):
            collector.start_step(step)
            for layer_idx, layer in enumerate(layers):
                output = layer(...)
                collector.record_layer(layer_idx, step, output)
            collector.end_step()
        
        # Save results
        collector.save('trace.json')
    """
    
    def __init__(
        self, 
        enabled: bool = False,
        num_layers: int = 32,
        track_attention: bool = False,
        track_timing: bool = True
    ):
        self.enabled = enabled
        self.num_layers = num_layers
        self.track_attention = track_attention
        self.track_timing = track_timing
        
        # Storage: {layer_idx: {step_idx: LayerStepStats}}
        self.stats: Dict[int, Dict[int, LayerStepStats]] = {
            i: {} for i in range(num_layers)
        }
        
        # Cache previous step outputs for cosine similarity
        self.prev_ffn_outputs: Dict[int, torch.Tensor] = {}
        self.prev_attn_outputs: Dict[int, torch.Tensor] = {}
        
        # Timing
        self.current_step: Optional[int] = None
        self.step_start_time: Optional[float] = None
        self.layer_start_times: Dict[int, float] = {}
        
    def start_step(self, step_idx: int):
        """Mark the start of a diffusion step."""
        if not self.enabled:
            return
        self.current_step = step_idx
        if self.track_timing:
            self.step_start_time = time.time()
    
    def end_step(self):
        """Mark the end of a diffusion step."""
        if not self.enabled:
            return
        self.current_step = None
        self.step_start_time = None
        
    def start_layer(self, layer_idx: int):
        """Mark the start of a layer forward pass."""
        if not self.enabled or not self.track_timing:
            return
        self.layer_start_times[layer_idx] = time.time()
    
    def record_layer(
        self,
        layer_idx: int,
        step_idx: int,
        ffn_output: torch.Tensor,
        attention_output: Optional[torch.Tensor] = None
    ):
        """
        Record statistics for a layer at a given step.
        
        Args:
            layer_idx: Index of the transformer layer (0-31)
            step_idx: Current diffusion step
            ffn_output: Output tensor from FFN [batch, seq_len, hidden_dim]
            attention_output: Optional attention output tensor
        """
        if not self.enabled:
            return
            
        # Compute FFN output norm
        with torch.no_grad():
            ffn_norm = torch.norm(ffn_output, p=2).item()
            
            # Compute cosine similarity with previous step
            ffn_cos_sim = None
            if layer_idx in self.prev_ffn_outputs:
                prev_ffn = self.prev_ffn_outputs[layer_idx]
                if prev_ffn.shape == ffn_output.shape:
                    ffn_cos_sim = F.cosine_similarity(
                        ffn_output.flatten(),
                        prev_ffn.flatten(),
                        dim=0
                    ).item()
            
            # Cache for next step
            self.prev_ffn_outputs[layer_idx] = ffn_output.detach().clone()
            
            # Attention statistics (optional)
            attn_norm = None
            attn_cos_sim = None
            if self.track_attention and attention_output is not None:
                attn_norm = torch.norm(attention_output, p=2).item()
                if layer_idx in self.prev_attn_outputs:
                    prev_attn = self.prev_attn_outputs[layer_idx]
                    if prev_attn.shape == attention_output.shape:
                        attn_cos_sim = F.cosine_similarity(
                            attention_output.flatten(),
                            prev_attn.flatten(),
                            dim=0
                        ).item()
                self.prev_attn_outputs[layer_idx] = attention_output.detach().clone()
            
            # Timing
            forward_time = None
            if self.track_timing and layer_idx in self.layer_start_times:
                forward_time = (time.time() - self.layer_start_times[layer_idx]) * 1000  # ms
        
        # Store statistics
        stats = LayerStepStats(
            layer_idx=layer_idx,
            step_idx=step_idx,
            ffn_output_norm=ffn_norm,
            attention_output_norm=attn_norm,
            ffn_cosine_sim=ffn_cos_sim,
            attention_cosine_sim=attn_cos_sim,
            forward_time_ms=forward_time
        )
        
        self.stats[layer_idx][step_idx] = stats
    
    def get_stats(self, layer_idx: int, step_idx: int) -> Optional[LayerStepStats]:
        """Retrieve statistics for a specific layer and step."""
        return self.stats.get(layer_idx, {}).get(step_idx)
    
    def get_all_stats(self) -> Dict[int, Dict[int, LayerStepStats]]:
        """Get all collected statistics."""
        return self.stats
    
    def save(self, filepath: str):
        """Save statistics to JSON file."""
        if not self.enabled:
            return
            
        # Convert to serializable format
        data = {
            'metadata': {
                'num_layers': self.num_layers,
                'track_attention': self.track_attention,
                'track_timing': self.track_timing
            },
            'stats': {}
        }
        
        for layer_idx, step_dict in self.stats.items():
            data['stats'][layer_idx] = {}
            for step_idx, stats in step_dict.items():
                data['stats'][layer_idx][step_idx] = asdict(stats)
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"✓ Saved trace statistics to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'TraceCollector':
        """Load statistics from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        collector = TraceCollector(
            enabled=False,  # Don't collect new data
            num_layers=data['metadata']['num_layers'],
            track_attention=data['metadata']['track_attention'],
            track_timing=data['metadata']['track_timing']
        )
        
        # Reconstruct stats
        for layer_idx_str, step_dict in data['stats'].items():
            layer_idx = int(layer_idx_str)
            for step_idx_str, stats_dict in step_dict.items():
                step_idx = int(step_idx_str)
                stats = LayerStepStats(**stats_dict)
                collector.stats[layer_idx][step_idx] = stats
        
        return collector
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Compute summary statistics across all layers and steps."""
        if not self.enabled or not self.stats:
            return {}
        
        all_ffn_norms = []
        all_cos_sims = []
        all_times = []
        
        for layer_stats in self.stats.values():
            for stats in layer_stats.values():
                all_ffn_norms.append(stats.ffn_output_norm)
                if stats.ffn_cosine_sim is not None:
                    all_cos_sims.append(stats.ffn_cosine_sim)
                if stats.forward_time_ms is not None:
                    all_times.append(stats.forward_time_ms)
        
        summary = {
            'ffn_norm': {
                'mean': sum(all_ffn_norms) / len(all_ffn_norms) if all_ffn_norms else 0,
                'min': min(all_ffn_norms) if all_ffn_norms else 0,
                'max': max(all_ffn_norms) if all_ffn_norms else 0,
            },
            'cosine_similarity': {
                'mean': sum(all_cos_sims) / len(all_cos_sims) if all_cos_sims else 0,
                'min': min(all_cos_sims) if all_cos_sims else 0,
                'max': max(all_cos_sims) if all_cos_sims else 0,
            },
            'forward_time_ms': {
                'mean': sum(all_times) / len(all_times) if all_times else 0,
                'min': min(all_times) if all_times else 0,
                'max': max(all_times) if all_times else 0,
            }
        }
        
        return summary
    
    def clear(self):
        """Clear all collected statistics (for next sample)."""
        if not self.enabled:
            return
        for layer_idx in range(self.num_layers):
            self.stats[layer_idx] = {}
        self.prev_ffn_outputs = {}
        self.prev_attn_outputs = {}
