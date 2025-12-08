#!/usr/bin/env python
# coding=utf-8
"""
Train Phase D Skip Router

Fine-tunes a skip router from a Phase C continuous router checkpoint.
The router learns to skip FFN layers entirely vs recompute.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from src.skip_router import SkipRouter, SkipRouterConfig, SkipRouterTrainer


def load_traces(traces_dir: str, max_samples: int = 1000) -> List[Dict]:
    """Load P1 traces for training."""
    traces = []
    traces_path = Path(traces_dir)
    
    if not traces_path.exists():
        print(f"Warning: Traces directory {traces_dir} not found")
        return traces
    
    trace_files = list(traces_path.glob("*.pt"))[:max_samples]
    print(f"Loading {len(trace_files)} trace files from {traces_dir}")
    
    for trace_file in trace_files:
        try:
            trace = torch.load(trace_file, map_location='cpu')
            traces.append(trace)
        except Exception as e:
            print(f"Warning: Failed to load {trace_file}: {e}")
    
    return traces


def main():
    parser = argparse.ArgumentParser(description="Train Phase D Skip Router")
    
    # Paths
    parser.add_argument("--phase_c_checkpoint", type=str, required=True,
                        help="Path to Phase C continuous router checkpoint")
    parser.add_argument("--traces_dir", type=str, required=True,
                        help="Directory containing P1 traces")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for checkpoints")
    
    # Training
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--skip_weight", type=float, default=0.1)
    parser.add_argument("--target_skip_ratio", type=float, default=0.25)
    parser.add_argument("--max_samples", type=int, default=1000)
    
    # W&B
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dllm_delta_compute")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create output directory
    output_path = Path(args.output_dir)
    checkpoints_path = output_path / "checkpoints"
    checkpoints_path.mkdir(parents=True, exist_ok=True)
    
    # Load Phase C router as initialization
    print(f"Loading Phase C router from {args.phase_c_checkpoint}")
    if os.path.exists(args.phase_c_checkpoint):
        skip_router = SkipRouter.from_continuous_router(args.phase_c_checkpoint)
    else:
        # Create new router if no Phase C checkpoint
        print("Phase C checkpoint not found, creating new skip router")
        config = SkipRouterConfig(
            num_layers=32,  # Dream-7B has 32 layers
            skip_loss_weight=args.skip_weight,
            target_skip_ratio=args.target_skip_ratio,
        )
        skip_router = SkipRouter(config)
    
    skip_router = skip_router.to(device)
    
    # Load traces
    traces = load_traces(args.traces_dir, args.max_samples)
    print(f"Loaded {len(traces)} traces")
    
    # Initialize W&B
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or f"phaseD_skip_{len(traces)}traces",
                config=vars(args),
            )
        except ImportError:
            print("Warning: wandb not installed, disabling logging")
            args.use_wandb = False
    
    # Create trainer
    trainer = SkipRouterTrainer(
        router=skip_router,
        learning_rate=args.learning_rate,
        skip_weight=args.skip_weight,
        target_skip_ratio=args.target_skip_ratio,
        device=device,
    )
    
    # Training loop (simplified - full implementation would use trace data)
    print(f"\nStarting training for {args.num_epochs} epochs...")
    
    best_skip_ratio = 0.0
    for epoch in range(args.num_epochs):
        epoch_losses = []
        epoch_skip_ratios = []
        
        for trace_idx, trace in enumerate(traces):
            # Extract progress values from trace
            if isinstance(trace, dict) and 'step_stats' in trace:
                # Use trace statistics
                for step_idx, step_stats in trace.get('step_stats', {}).items():
                    progress = step_idx / max(1, len(trace.get('step_stats', {})) - 1)
                    
                    # Compute skip loss (simplified)
                    p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
                    layer_indices = torch.arange(skip_router.config.num_layers, device=device)
                    betas = skip_router(p_tensor, layer_indices)
                    
                    # Skip loss: encourage low β (high skipping)
                    skip_loss = trainer.compute_skip_loss(p_tensor.unsqueeze(0))
                    
                    trainer.optimizer.zero_grad()
                    skip_loss.backward()
                    trainer.optimizer.step()
                    
                    with torch.no_grad():
                        skip_ratio = 1.0 - betas.mean().item()
                        epoch_losses.append(skip_loss.item())
                        epoch_skip_ratios.append(skip_ratio)
            else:
                # Fallback: use random progress values for training
                for _ in range(10):
                    progress = np.random.uniform(0, 1)
                    p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
                    
                    skip_loss = trainer.compute_skip_loss(p_tensor.unsqueeze(0))
                    
                    trainer.optimizer.zero_grad()
                    skip_loss.backward()
                    trainer.optimizer.step()
                    
                    with torch.no_grad():
                        layer_indices = torch.arange(skip_router.config.num_layers, device=device)
                        betas = skip_router(p_tensor, layer_indices)
                        skip_ratio = 1.0 - betas.mean().item()
                        epoch_losses.append(skip_loss.item())
                        epoch_skip_ratios.append(skip_ratio)
        
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        avg_skip_ratio = np.mean(epoch_skip_ratios) if epoch_skip_ratios else 0.0
        
        print(f"Epoch {epoch+1}/{args.num_epochs} - Loss: {avg_loss:.4f}, Skip Ratio: {avg_skip_ratio:.2%}")
        
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "loss": avg_loss,
                "skip_ratio": avg_skip_ratio,
            })
        
        # Save best checkpoint
        if avg_skip_ratio > best_skip_ratio:
            best_skip_ratio = avg_skip_ratio
            skip_router.save(
                str(checkpoints_path / "skip_router_best.pt"),
                metadata={"epoch": epoch + 1, "skip_ratio": avg_skip_ratio}
            )
    
    # Save final checkpoint
    skip_router.save(
        str(checkpoints_path / "skip_router_final.pt"),
        metadata={"epochs": args.num_epochs, "skip_ratio": avg_skip_ratio}
    )
    
    # Save training config
    with open(output_path / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best skip ratio: {best_skip_ratio:.2%}")
    print(f"Checkpoints saved to {checkpoints_path}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
