#!/usr/bin/env python3
# coding=utf-8
"""
Training script for P3 Learned Gate.

Usage:
    python scripts/training/train_learned_gate.py \
        --trace_dir experiments/P1_traces/traces \
        --oracle_labels experiments/P3_learned_gate/oracle_labels.json \
        --output_dir experiments/P3_learned_gate/checkpoints \
        --num_epochs 50 \
        --batch_size 32
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from glob import glob

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from learned_gate import LearnedGate, GateTrainer


def load_oracle_labels(filepath: str) -> dict:
    """
    Load oracle labels from JSON file.
    
    Expected format:
    {
        "0_10": 1.0,  # (layer_0, step_10) -> safe to freeze
        "0_11": 1.0,
        "15_5": 0.0,  # (layer_15, step_5) -> must recompute
        ...
    }
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert string keys to tuples
    oracle_labels = {}
    for key, value in data.items():
        layer_idx, step_idx = map(int, key.split('_'))
        oracle_labels[(layer_idx, step_idx)] = float(value)
    
    return oracle_labels


def main():
    parser = argparse.ArgumentParser(description="Train P3 Learned Gate")
    parser.add_argument('--trace_dir', type=str, required=True,
                       help='Directory containing P1 trace .pt files')
    parser.add_argument('--oracle_labels', type=str, required=True,
                       help='Path to oracle labels JSON file')
    parser.add_argument('--output_dir', type=str, default='experiments/P3_learned_gate/checkpoints',
                       help='Output directory for checkpoints')
    parser.add_argument('--num_epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--val_split', type=float, default=0.2,
                       help='Validation split ratio')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("P3 Learned Gate Training")
    print("=" * 80)
    print(f"Trace directory: {args.trace_dir}")
    print(f"Oracle labels: {args.oracle_labels}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print()
    
    # Load oracle labels
    print("Loading oracle labels...")
    oracle_labels = load_oracle_labels(args.oracle_labels)
    print(f"Loaded {len(oracle_labels)} oracle labels")
    print(f"  Positive (safe to freeze): {sum(v == 1.0 for v in oracle_labels.values())}")
    print(f"  Negative (must recompute): {sum(v == 0.0 for v in oracle_labels.values())}")
    print()
    
    # Find trace files
    print("Finding trace files...")
    trace_files = glob(f"{args.trace_dir}/*.pt")
    print(f"Found {len(trace_files)} trace files")
    
    if len(trace_files) == 0:
        print("ERROR: No trace files found!")
        return 1
    
    # Initialize gate
    print("\nInitializing LearnedGate...")
    gate = LearnedGate(
        input_dim=9,
        hidden_dim=32,
        num_hidden_layers=2,
        dropout=0.1
    )
    print()
    
    # Initialize trainer
    print("Initializing GateTrainer...")
    trainer = GateTrainer(
        gate=gate,
        learning_rate=args.learning_rate,
        device=args.device
    )
    print()
    
    # Prepare training data
    print("Preparing training data...")
    all_features, all_labels = trainer.prepare_training_data(trace_files, oracle_labels)
    print(f"Total samples: {len(all_features)}")
    
    # Split train/val
    num_val = int(len(all_features) * args.val_split)
    num_train = len(all_features) - num_val
    
    indices = torch.randperm(len(all_features))
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_features = all_features[train_indices]
    train_labels = all_labels[train_indices]
    val_features = all_features[val_indices]
    val_labels = all_labels[val_indices]
    
    print(f"Training samples: {len(train_features)}")
    print(f"Validation samples: {len(val_features)}")
    print()
    
    # Train
    print("Starting training...")
    print("=" * 80)
    trainer.train(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        early_stopping_patience=10
    )
    print("=" * 80)
    
    # Save final model
    final_path = Path(args.output_dir) / 'learned_gate_final.pt'
    gate.save(str(final_path), metadata={
        'num_samples': len(all_features),
        'num_epochs': args.num_epochs,
        'final_val_loss': trainer.val_losses[-1] if trainer.val_losses else None
    })
    
    print(f"\n✓ Training complete!")
    print(f"Best model saved to: {args.output_dir}/learned_gate_best.pt")
    print(f"Final model saved to: {final_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
