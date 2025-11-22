#!/usr/bin/env python3
# coding=utf-8
"""
Training script for Phase C Continuous Router.

Usage:
    python scripts/training/train_continuous_router.py \
        --trace_dir experiments/P1_traces/traces \
        --output_dir experiments/PhaseC_continuous/checkpoints \
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

from continuous_router import ContinuousRouter, ContinuousRouterConfig
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
    parser = argparse.ArgumentParser(description="Train Phase C Continuous Router")
    parser.add_argument('--trace_dir', type=str, required=True,
                       help='Directory containing P1 trace .pt files')
    parser.add_argument('--output_dir', type=str, default='experiments/PhaseC_continuous/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=256, 
                       help="Assumed total steps for calculating progress from step_idx")
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Find traces
    trace_files = glob(f"{args.trace_dir}/*.pt")
    if not trace_files:
        print("No trace files found!")
        return 1
        
    # Initialize Router
    print("Initializing ContinuousRouter...")
    config = ContinuousRouterConfig(
        num_layers=args.num_layers,
        hidden_dim=64
    )
    router = ContinuousRouter(config).to(args.device)
    
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
            
            batch_loss = 0
            count = 0
            
            for stats in batch_stats:
                for layer_idx, step_dict in stats.items():
                    for step_idx, layer_stats in step_dict.items():
                        if layer_stats.ffn_cosine_sim is not None:
                            # Target: 1.0 if dissimilar (must recompute), 0.0 if similar (can cache)
                            target = max(0.0, min(1.0, 1.0 - layer_stats.ffn_cosine_sim))
                            
                            # Calculate progress (time)
                            # In real generation, this is 1 - mask_ratio
                            # Here we approximate with step_idx / total_steps
                            progress = step_idx / max(1, args.total_steps - 1)
                            
                            # Forward pass
                            # router.forward takes (time, layer_indices)
                            # We can batch this efficiently, but for now let's do per-sample for simplicity
                            # or construct a batch of inputs
                            
                            # Let's do it per-sample for clarity in this PoC script
                            # (Optimization: batch all (time, layer) pairs)
                            
                            time_tensor = torch.tensor(progress, dtype=torch.float32, device=args.device)
                            layer_tensor = torch.tensor([layer_idx], dtype=torch.long, device=args.device)
                            
                            prob = router(time_tensor, layer_tensor) # [1]
                            
                            # MSE Loss
                            batch_loss += (prob[0] - target) ** 2
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
            router.save(f"{args.output_dir}/continuous_router_epoch_{epoch+1}.pt")
            
    # Save final
    router.save(f"{args.output_dir}/continuous_router_final.pt")
    print(f"Saved final model to {args.output_dir}/continuous_router_final.pt")
    
    return 0

if __name__ == '__main__':
    sys.exit(main())
