# coding=utf-8
"""
P3 - Learned Gate for Delta-Compute acceleration.
Lightweight MLP that predicts "safe to freeze" based on P1 trace features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class GateFeatures:
    """Features extracted from current state for gate decision."""
    layer_idx: int
    step_idx: int
    total_steps: int
    total_layers: int
    
    # From P1 traces
    ffn_output_norm: float
    ffn_cosine_sim: Optional[float] = None
    attention_output_norm: Optional[float] = None
    attention_cosine_sim: Optional[float] = None
    
    # From current generation state
    max_confidence: float = 0.0
    entropy: float = 0.0
    num_masked_tokens: int = 0
    total_tokens: int = 0
    
    def to_tensor(self, device: torch.device = None) -> torch.Tensor:
        """Convert features to normalized tensor for MLP input."""
        # Normalize features
        features = [
            self.layer_idx / max(1, self.total_layers),  # 0-1
            self.step_idx / max(1, self.total_steps),    # 0-1
            min(self.ffn_output_norm / 100.0, 1.0),      # rough normalization
            self.ffn_cosine_sim if self.ffn_cosine_sim is not None else 0.5,  # 0-1
            min(self.attention_output_norm / 100.0, 1.0) if self.attention_output_norm is not None else 0.5,
            self.attention_cosine_sim if self.attention_cosine_sim is not None else 0.5,
            self.max_confidence,  # 0-1
            min(self.entropy / 10.0, 1.0),  # rough normalization
            self.num_masked_tokens / max(1, self.total_tokens),  # 0-1
        ]
        
        tensor = torch.tensor(features, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor


class LearnedGate(nn.Module):
    """
    Lightweight MLP gate (<50k params) that predicts whether it's safe to freeze
    a layer at a given step.
    
    Architecture:
        Input: 9 features (layer_idx, step_idx, norms, cosine_sims, confidence, entropy, mask_ratio)
        Hidden: 2 layers with 32 hidden units each
        Output: 1 sigmoid probability (0 = recompute, 1 = safe to freeze)
    
    Total params: 9*32 + 32 + 32*32 + 32 + 32*1 + 1 = 1,377 params (well under 50k)
    """
    
    def __init__(
        self,
        input_dim: int = 9,
        hidden_dim: int = 32,
        num_hidden_layers: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i in range(num_hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Count parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"LearnedGate initialized with {self.num_params:,} parameters")
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, input_dim] or [input_dim] tensor
        
        Returns:
            probabilities: [batch_size, 1] or [1] - probability of freezing being safe
        """
        return self.network(features)
    
    def should_freeze(
        self,
        features: GateFeatures,
        threshold: float = 0.5,
        device: torch.device = None
    ) -> bool:
        """
        Decide whether to freeze based on gate prediction.
        
        Args:
            features: GateFeatures object
            threshold: Probability threshold for freezing decision (default 0.5)
            device: torch device
        
        Returns:
            True if safe to freeze, False if should recompute
        """
        self.eval()
        with torch.no_grad():
            feature_tensor = features.to_tensor(device).unsqueeze(0)  # [1, input_dim]
            prob = self.forward(feature_tensor).item()
            return prob >= threshold
    
    def save(self, filepath: str, metadata: Dict = None):
        """Save gate weights and metadata."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'num_params': self.num_params,
            'config': {
                'input_dim': 9,
                'hidden_dim': 32,
                'num_hidden_layers': 2,
                'dropout': 0.1
            }
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved LearnedGate to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'LearnedGate':
        """Load gate from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config = checkpoint.get('config', {})
        gate = LearnedGate(
            input_dim=config.get('input_dim', 9),
            hidden_dim=config.get('hidden_dim', 32),
            num_hidden_layers=config.get('num_hidden_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
        
        gate.load_state_dict(checkpoint['state_dict'])
        print(f"✓ Loaded LearnedGate from {filepath} ({gate.num_params:,} params)")
        
        return gate


class GateTrainer:
    """
    Trainer for LearnedGate using P1 traces and oracle labels.
    
    Training strategy:
        1. Load P1 traces from multiple samples
        2. For each (layer, step) pair, extract features
        3. Generate labels using oracle experiments:
           - positive (1): freezing this layer/step doesn't harm final output
           - negative (0): freezing causes quality degradation
        4. Train gate with BCE loss + optional regularization
    """
    
    def __init__(
        self,
        gate: LearnedGate,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        device: str = 'cuda'
    ):
        self.gate = gate.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            gate.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        self.criterion = nn.BCELoss()
        
        self.train_losses = []
        self.val_losses = []
    
    def prepare_training_data(
        self,
        trace_files: List[str],
        oracle_labels: Dict[Tuple[int, int], float]  # (layer_idx, step_idx) -> label
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from P1 traces and oracle labels.
        
        Args:
            trace_files: List of .pt trace files from P1
            oracle_labels: Dictionary mapping (layer, step) to binary label
                          1.0 = safe to freeze, 0.0 = must recompute
        
        Returns:
            features: [N, input_dim] tensor
            labels: [N, 1] tensor
        """
        from tracing import TraceCollector
        
        all_features = []
        all_labels = []
        
        for trace_file in trace_files:
            collector = TraceCollector.load(trace_file)
            stats_dict = collector.get_all_stats()
            
            for layer_idx, step_dict in stats_dict.items():
                for step_idx, stats in step_dict.items():
                    # Check if we have oracle label for this (layer, step)
                    if (layer_idx, step_idx) not in oracle_labels:
                        continue
                    
                    # Extract features
                    features = GateFeatures(
                        layer_idx=layer_idx,
                        step_idx=step_idx,
                        total_steps=max(s for s in step_dict.keys()) + 1,
                        total_layers=len(stats_dict),
                        ffn_output_norm=stats.ffn_output_norm,
                        ffn_cosine_sim=stats.ffn_cosine_sim,
                        attention_output_norm=stats.attention_output_norm,
                        attention_cosine_sim=stats.attention_cosine_sim,
                        # Note: confidence/entropy would come from generation logs
                        # For now, use defaults or load from separate file
                    )
                    
                    all_features.append(features.to_tensor())
                    all_labels.append(oracle_labels[(layer_idx, step_idx)])
        
        if not all_features:
            raise ValueError("No training data found! Check trace files and oracle labels.")
        
        features_tensor = torch.stack(all_features)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32).unsqueeze(1)
        
        return features_tensor, labels_tensor
    
    def train_epoch(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        batch_size: int = 32
    ) -> float:
        """Train for one epoch."""
        self.gate.train()
        
        # Shuffle data
        perm = torch.randperm(len(train_features))
        train_features = train_features[perm]
        train_labels = train_labels[perm]
        
        total_loss = 0.0
        num_batches = 0
        
        for i in range(0, len(train_features), batch_size):
            batch_features = train_features[i:i+batch_size].to(self.device)
            batch_labels = train_labels[i:i+batch_size].to(self.device)
            
            self.optimizer.zero_grad()
            
            predictions = self.gate(batch_features)
            loss = self.criterion(predictions, batch_labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def evaluate(
        self,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        batch_size: int = 128
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.gate.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for i in range(0, len(val_features), batch_size):
                batch_features = val_features[i:i+batch_size].to(self.device)
                batch_labels = val_labels[i:i+batch_size].to(self.device)
                
                predictions = self.gate(batch_features)
                loss = self.criterion(predictions, batch_labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Compute metrics
        import numpy as np
        preds = np.array(all_preds).flatten()
        labels = np.array(all_labels).flatten()
        
        # Binary predictions at 0.5 threshold
        binary_preds = (preds >= 0.5).astype(float)
        accuracy = (binary_preds == labels).mean()
        
        # Precision/recall for positive class (safe to freeze)
        tp = ((binary_preds == 1) & (labels == 1)).sum()
        fp = ((binary_preds == 1) & (labels == 0)).sum()
        fn = ((binary_preds == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'loss': total_loss / num_batches,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self,
        train_features: torch.Tensor,
        train_labels: torch.Tensor,
        val_features: torch.Tensor,
        val_labels: torch.Tensor,
        num_epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 10
    ):
        """Full training loop with early stopping."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training LearnedGate for {num_epochs} epochs...")
        print(f"Training samples: {len(train_features)}, Validation samples: {len(val_features)}")
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_features, train_labels, batch_size)
            val_metrics = self.evaluate(val_features, val_labels)
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_metrics['loss'])
            
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Val Loss: {val_metrics['loss']:.4f}, "
                f"Acc: {val_metrics['accuracy']:.3f}, F1: {val_metrics['f1']:.3f}"
            )
            
            # Early stopping
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                # Save best model
                self.gate.save('checkpoints/learned_gate_best.pt', metadata={'epoch': epoch, 'val_loss': best_val_loss})
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
