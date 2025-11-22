#!/usr/bin/env python3
# coding=utf-8
"""
Training script for Phase B Learned Router (Fixed Schedule).

Usage:
    python scripts/training/train_learned_router.py \
        --trace_dir experiments/P1_traces/traces \
        --output_dir experiments/PhaseB_router/checkpoints \
        --num_epochs 50
"""

import argparse
import sys
from pathlib import Path
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from learned_router import FixedScheduleRouter, RouterConfig
from tracing import TraceCollector

class TraceDataset(Dataset):
    def __init__(self, trace_files):
        self.trace_files = trace_files
        self.data = []
        
        print(f"Loading {len(trace_files)} trace files...")
        for i, f in enumerate(trace_files):
            if i % 10 == 0:
                print(f"Loading {i}/{len(trace_files)}...", end='\r')
            try:
                collector = TraceCollector.load(f)
                stats = collector.get_all_stats()
                self.data.append(stats)
            except Exception as e:
                print(f"\nError loading {f}: {e}")
        print(f"\nLoaded {len(self.data)} samples.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def main():
    parser = argparse.ArgumentParser(description="Train Phase B Learned Router")
    parser.add_argument('--trace_dir', type=str, required=True,
                       help='Directory containing P1 trace .pt files')
    parser.add_argument('--output_dir', type=str, default='experiments/PhaseB_router/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find traces
    trace_files = glob(f"{args.trace_dir}/*.pt")
    if not trace_files:
        print("No trace files found!")
        return 1
        
    # Initialize Router
    print("Initializing FixedScheduleRouter...")
    config = RouterConfig(
        num_layers=args.num_layers,
        num_steps=args.num_steps
    )
    router = FixedScheduleRouter(config).to(args.device)
    
    optimizer = optim.Adam(router.parameters(), lr=args.learning_rate)
    
    # Load data
    dataset = TraceDataset(trace_files)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda x: x)
    
    print("Starting training...")
    
    for epoch in range(args.num_epochs):
        router.train()
        total_loss = 0
        
        for batch_stats in dataloader:
            optimizer.zero_grad()
            
            # Compute loss based on cosine similarity proxy
            # We want router probability (recompute) to be high when cosine sim is low
            # Target prob = 1 - cosine_sim
            
            batch_loss = 0
            count = 0
            
            for stats in batch_stats:
                for layer_idx, step_dict in stats.items():
                    for step_idx, layer_stats in step_dict.items():
                        if layer_stats.ffn_cosine_sim is not None:
                            # Target: 1.0 if dissimilar (must recompute), 0.0 if similar (can cache)
                            # We use a soft target: 1 - cos_sim
                            # Clamp to [0, 1] just in case
                            target = max(0.0, min(1.0, 1.0 - layer_stats.ffn_cosine_sim))
                            
                            # Get router probability for this (layer, step)
                            if step_idx < args.num_steps and layer_idx < args.num_layers:
                                prob = router.forward(step_idx, layer_idx)
                                
                                # MSE Loss
                                batch_loss += (prob - target) ** 2
                                count += 1
            
            if count > 0:
                batch_loss = batch_loss / count
                batch_loss.backward()
                optimizer.step()
                total_loss += batch_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            router.save(f"{args.output_dir}/router_epoch_{epoch+1}.pt")
            
    # Save final
    router.save(f"{args.output_dir}/router_final.pt")
    print(f"Saved final model to {args.output_dir}/router_final.pt")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
