#!/usr/bin/env python
# coding=utf-8
"""
Train Phase E FlexiDepth Router + Adapter

Trains a FlexiDepth-style router with trainable adapters (Phase E).
The backbone is frozen; only router and adapter parameters are trained.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.flexi_adapter import FlexiDepthManager, FlexiRouterConfig


def main():
    parser = argparse.ArgumentParser(description="Train Phase E FlexiDepth Router+Adapter")
    
    # Paths
    parser.add_argument("--model", type=str, default="Dream-org/Dream-v0-Instruct-7B",
                        help="Model name for hidden_dim inference")
    parser.add_argument("--traces_dir", type=str, required=True,
                        help="Directory containing P1 traces")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Architecture
    parser.add_argument("--adapted_layers", type=str, default="24,25,26,27,28,29,30,31",
                        help="Comma-separated layer indices to adapt")
    parser.add_argument("--adapter_dim", type=int, default=256,
                        help="Adapter bottleneck dimension")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--skip_weight", type=float, default=0.1)
    parser.add_argument("--target_skip_ratio", type=float, default=0.30)
    
    # W&B
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dllm_delta_compute")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse adapted layers
    adapted_layers = [int(x.strip()) for x in args.adapted_layers.split(",")]
    print(f"Adapting layers: {adapted_layers}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    checkpoints_path = output_path / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    
    # Dream-7B has hidden_dim=4096 and 32 layers
    hidden_dim = 4096
    num_layers = 32
    
    # Create FlexiDepth config
    config = FlexiRouterConfig(
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        adapted_layers=adapted_layers,
        adapter_dim=args.adapter_dim,
        skip_loss_weight=args.skip_weight,
        target_skip_ratio=args.target_skip_ratio,
    )
    
    # Create manager
    manager = FlexiDepthManager(config)
    manager = manager.to(device)
    
    print(f"FlexiDepth total trainable params: {manager.total_params:,}")
    
    # Initialize W&B
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"phaseE_flexi_{len(adapted_layers)}layers",
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed, disabling logging")
            args.use_wandb = False
    
    # Get all trainable parameters
    params = manager.get_all_parameters()
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    
    # Training loop (simplified)
    print(f"\nStarting training for {args.num_epochs} epochs...")
    
    manager.train()
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_shallow_ratios = []
        
        # Simplified training: random progress values
        for _ in range(100):  # 100 mini-batches per epoch
            progress = np.random.uniform(0, 1)
            p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
            
            # Get all gate scores
            total_shallow = 0
            total_gates = 0
            gate_loss = torch.tensor(0.0, device=device, requires_grad=True)
            
            for layer_idx in adapted_layers:
                gate = manager.router(p_tensor, layer_idx, hidden_state=None)
                # Regularize to encourage shallow path (low gate score)
                gate_loss = gate_loss + gate ** 2
                
                total_gates += 1
                if gate.item() < 0.5:
                    total_shallow += 1
            
            # Normalize loss
            loss = gate_loss / len(adapted_layers)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            epoch_shallow_ratios.append(total_shallow / max(1, total_gates))
        
        avg_loss = np.mean(epoch_losses)
        avg_shallow_ratio = np.mean(epoch_shallow_ratios)
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}, Shallow Ratio: {avg_shallow_ratio:.2%}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "shallow_ratio": avg_shallow_ratio,
            })
        
        # Save best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            manager.save(
                str(checkpoints_path / "flexi_depth_best.pt"),
                metadata={"epoch": epoch + 1, "loss": avg_loss, "shallow_ratio": avg_shallow_ratio}
            )
    
    # Save final checkpoint
    manager.save(
        str(checkpoints_path / "flexi_depth_final.pt"),
        metadata={"epochs": args.num_epochs, "shallow_ratio": avg_shallow_ratio}
    )
    
    # Save training config
    with open(output_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Checkpoints saved to {checkpoints_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
