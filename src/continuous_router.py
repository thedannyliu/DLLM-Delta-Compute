# coding=utf-8
"""
Phase C - Continuous-Time Router for L2C-style layer caching.
Learns per-layer router βₗ(t) as a function of continuous time,
enabling transfer across different diffusion schedules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class ContinuousRouterConfig:
    """Configuration for continuous-time router."""
    num_layers: int
    time_embedding_dim: int = 64
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1
    time_encoding: str = 'sinusoidal'  # 'sinusoidal' or 'learned'


class TimeEncoder(nn.Module):
    """
    Encode continuous time t into a fixed-dimensional embedding.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        encoding_type: str = 'sinusoidal',
        max_period: float = 10000.0
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.encoding_type = encoding_type
        self.max_period = max_period
        
        if encoding_type == 'learned':
            # Learned embedding
            self.time_mlp = nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.SiLU(),
                nn.Linear(embedding_dim, embedding_dim)
            )
        elif encoding_type == 'sinusoidal':
            # Sinusoidal encoding (similar to Transformer positional encoding)
            # No parameters needed
            pass
        else:
            raise ValueError(f"Unknown encoding_type: {encoding_type}")
    
    def sinusoidal_encoding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time encoding.
        
        Args:
            t: [batch_size] or [batch_size, 1] tensor of time values in [0, 1]
        
        Returns:
            encoding: [batch_size, embedding_dim] tensor
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [batch_size, 1]
        
        # Compute frequencies
        half_dim = self.embedding_dim // 2
        emb = math.log(self.max_period) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device, dtype=t.dtype) * -emb)
        
        # Compute sin/cos embeddings
        emb = t * emb.unsqueeze(0)  # [batch_size, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # [batch_size, embedding_dim]
        
        return emb
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time.
        
        Args:
            t: Time values in [0, 1]. Can be:
               - scalar: single time value
               - [batch_size]: batch of time values
        
        Returns:
            [batch_size, embedding_dim] or [embedding_dim] encoding
        """
        # Ensure tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        
        # Handle scalar
        is_scalar = t.dim() == 0
        if is_scalar:
            t = t.unsqueeze(0)
        
        if self.encoding_type == 'sinusoidal':
            encoding = self.sinusoidal_encoding(t)
        else:  # learned
            if t.dim() == 1:
                t = t.unsqueeze(-1)  # [batch_size, 1]
            encoding = self.time_mlp(t)
        
        if is_scalar:
            encoding = encoding.squeeze(0)
        
        return encoding


class ContinuousRouter(nn.Module):
    """
    Continuous-time router βₗ(t) for each layer l.
    
    Takes as input:
        - t: continuous time (normalized to [0,1])
        - l: layer index
    
    Outputs:
        - β ∈ [0,1]: probability to recompute (1) vs cache (0)
    
    Architecture:
        1. Time encoder: t -> time_emb
        2. Layer embedding: l -> layer_emb
        3. MLP: [time_emb || layer_emb] -> β
    
    This allows the router to:
        - Generalize across different diffusion schedules (via continuous t)
        - Learn layer-specific patterns
        - Transfer from one schedule to another (e.g., train on 256 steps, test on 512)
    """
    
    def __init__(self, config: ContinuousRouterConfig):
        super().__init__()
        self.config = config
        
        # Time encoder
        self.time_encoder = TimeEncoder(
            embedding_dim=config.time_embedding_dim,
            encoding_type=config.time_encoding
        )
        
        # Layer embedding
        self.layer_embedding = nn.Embedding(
            num_embeddings=config.num_layers,
            embedding_dim=config.time_embedding_dim
        )
        
        # MLP: [time_emb || layer_emb] -> β
        mlp_input_dim = config.time_embedding_dim * 2
        layers = []
        prev_dim = mlp_input_dim
        
        for i in range(config.num_hidden_layers):
            layers.extend([
                nn.Linear(prev_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.mlp = nn.Sequential(*layers)
        
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"ContinuousRouter initialized with {self.num_params:,} parameters")
    
    def forward(
        self,
        t: torch.Tensor,
        layer_idx: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute recompute probability.
        
        Args:
            t: Time values in [0,1]. Shape: [] or [batch_size]
            layer_idx: Layer indices. Shape: [] or [batch_size] or [num_layers]
                       If None, compute for all layers
        
        Returns:
            β probabilities. Shape depends on inputs:
                - t scalar, layer_idx scalar: [] scalar
                - t [B], layer_idx [B]: [B]
                - t scalar, layer_idx [L]: [L] (all layers at time t)
                - t [B], layer_idx None: [B, L] (all layers at batch of times)
        """
        # Encode time
        time_emb = self.time_encoder(t)  # [B, D] or [D]
        
        # Handle different layer_idx cases
        if layer_idx is None:
            # All layers
            layer_idx = torch.arange(self.config.num_layers, device=time_emb.device)
        
        # Embed layers
        layer_emb = self.layer_embedding(layer_idx)  # [L, D] or [B, D] or [D]
        
        # Broadcast and concatenate
        if time_emb.dim() == 1 and layer_emb.dim() == 2:
            # time: [D], layer: [L, D] -> expand time to [L, D]
            time_emb = time_emb.unsqueeze(0).expand(layer_emb.shape[0], -1)
        elif time_emb.dim() == 2 and layer_emb.dim() == 1:
            # time: [B, D], layer: [D] -> expand layer to [B, D]
            layer_emb = layer_emb.unsqueeze(0).expand(time_emb.shape[0], -1)
        elif time_emb.dim() == 2 and layer_emb.dim() == 2 and time_emb.shape[0] != layer_emb.shape[0]:
            # time: [B, D], layer: [L, D] -> expand to [B, L, D]
            time_emb = time_emb.unsqueeze(1).expand(-1, layer_emb.shape[0], -1)  # [B, L, D]
            layer_emb = layer_emb.unsqueeze(0).expand(time_emb.shape[0], -1, -1)  # [B, L, D]
        
        # Concatenate
        combined = torch.cat([time_emb, layer_emb], dim=-1)
        
        # MLP
        beta = self.mlp(combined).squeeze(-1)
        
        return beta
    
    def get_decision(
        self,
        t: float,
        layer_idx: int,
        threshold: float = 0.5,
        device: str = 'cuda'
    ) -> bool:
        """
        Make binary decision for a single (t, layer).
        
        Returns:
            True = recompute, False = cache
        """
        self.eval()
        with torch.no_grad():
            t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
            l_tensor = torch.tensor(layer_idx, dtype=torch.long, device=device)
            
            prob = self.forward(t_tensor, l_tensor).item()
            return prob >= threshold
    
    def get_schedule_decisions(
        self,
        num_steps: int,
        threshold: float = 0.5,
        device: str = 'cuda'
    ) -> Dict[int, Dict[int, bool]]:
        """
        Get decisions for an entire diffusion schedule.
        
        Args:
            num_steps: Number of diffusion steps
            threshold: Decision threshold
            device: torch device
        
        Returns:
            {step_idx: {layer_idx: bool}} nested dict of decisions
        """
        self.eval()
        schedule = {}
        
        with torch.no_grad():
            for step_idx in range(num_steps):
                t = step_idx / max(1, num_steps - 1)  # Normalize to [0, 1]
                t_tensor = torch.tensor(t, dtype=torch.float32, device=device)
                
                # Get all layers at once
                layer_indices = torch.arange(self.config.num_layers, device=device)
                probs = self.forward(t_tensor, layer_indices)  # [num_layers]
                
                schedule[step_idx] = {
                    layer_idx: probs[layer_idx].item() >= threshold
                    for layer_idx in range(self.config.num_layers)
                }
        
        return schedule
    
    def visualize_schedule(
        self,
        num_steps: int,
        save_path: str = None,
        device: str = 'cuda'
    ):
        """
        Visualize learned router schedule as heatmap.
        
        Args:
            num_steps: Number of steps to visualize
            save_path: Path to save plot (if None, just display)
            device: torch device
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        self.eval()
        
        # Compute probabilities for all (step, layer) pairs
        steps = torch.linspace(0, 1, num_steps, device=device)
        layers = torch.arange(self.config.num_layers, device=device)
        
        with torch.no_grad():
            # Get all probabilities
            probs = torch.zeros(num_steps, self.config.num_layers)
            for i, t in enumerate(steps):
                probs[i] = self.forward(t, layers).cpu()
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            probs.T.numpy(),  # [layers, steps]
            cmap='RdYlGn',
            vmin=0, vmax=1,
            cbar_kws={'label': 'Recompute Probability'},
            xticklabels=num_steps // 10,
            yticklabels=self.config.num_layers // 4
        )
        plt.xlabel('Diffusion Step')
        plt.ylabel('Layer Index')
        plt.title('Continuous Router Schedule (Green=Cache, Red=Recompute)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"✓ Saved schedule visualization to {save_path}")
        else:
            plt.show()
    
    def save(self, filepath: str, metadata: Dict = None):
        """Save router weights and metadata."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'num_layers': self.config.num_layers,
                'time_embedding_dim': self.config.time_embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_hidden_layers': self.config.num_hidden_layers,
                'dropout': self.config.dropout,
                'time_encoding': self.config.time_encoding
            },
            'num_params': self.num_params
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved ContinuousRouter to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'ContinuousRouter':
        """Load router from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config_dict = checkpoint['config']
        config = ContinuousRouterConfig(
            num_layers=config_dict['num_layers'],
            time_embedding_dim=config_dict.get('time_embedding_dim', 64),
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_hidden_layers=config_dict.get('num_hidden_layers', 2),
            dropout=config_dict.get('dropout', 0.1),
            time_encoding=config_dict.get('time_encoding', 'sinusoidal')
        )
        
        router = ContinuousRouter(config)
        router.load_state_dict(checkpoint['state_dict'])
        
        print(f"✓ Loaded ContinuousRouter from {filepath} ({router.num_params:,} params)")
        return router


class ContinuousRouterTrainer:
    """
    Trainer for continuous-time router.
    Similar to Phase B but trains on multiple schedules simultaneously.
    """
    
    def __init__(
        self,
        router: ContinuousRouter,
        learning_rate: float = 1e-4,
        efficiency_weight: float = 0.1,
        device: str = 'cuda'
    ):
        self.router = router.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            router.parameters(),
            lr=learning_rate
        )
        
        self.efficiency_weight = efficiency_weight
        self.train_losses = []
        self.val_losses = []
    
    def train_on_multiple_schedules(
        self,
        train_data: List[Dict],  # Contains samples from different schedules
        val_data: List[Dict],
        num_epochs: int = 100,
        early_stopping_patience: int = 15
    ):
        """
        Train router on multiple diffusion schedules.
        
        train_data format:
            Each sample should have:
            {
                'schedule_steps': int (e.g., 128, 256, 512),
                'step_idx': int (absolute step in that schedule),
                'teacher_outputs': Dict[int, torch.Tensor],
                'student_forward_fn': callable,
                'mask': torch.Tensor (optional)
            }
        
        This allows the router to learn patterns that generalize across schedules.
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training ContinuousRouter on multiple schedules for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.router.train()
            epoch_loss = 0.0
            
            np.random.shuffle(train_data)
            
            for sample in train_data:
                loss = self._train_step(sample)
                epoch_loss += loss
            
            avg_loss = epoch_loss / len(train_data)
            self.train_losses.append(avg_loss)
            
            # Validation
            val_metrics = self._evaluate(val_data)
            self.val_losses.append(val_metrics['avg_loss'])
            
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_loss:.4f}, "
                f"Val Loss: {val_metrics['avg_loss']:.4f}, "
                f"Cache Ratio: {val_metrics['cache_ratio']:.2%}"
            )
            
            # Early stopping
            if val_metrics['avg_loss'] < best_val_loss:
                best_val_loss = val_metrics['avg_loss']
                patience_counter = 0
                self.router.save(
                    'checkpoints/continuous_router_best.pt',
                    metadata={'epoch': epoch, 'val_loss': best_val_loss}
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")
    
    def _train_step(self, sample: Dict) -> float:
        """Single training step."""
        # Normalize time
        t = sample['step_idx'] / max(1, sample['schedule_steps'] - 1)
        t_tensor = torch.tensor(t, dtype=torch.float32, device=self.device)
        
        # Get router decisions for all layers
        layer_indices = torch.arange(self.router.config.num_layers, device=self.device)
        router_probs = self.router(t_tensor, layer_indices)  # [num_layers]
        
        # Use Gumbel-Softmax for differentiable sampling
        # (Similar to Phase B implementation)
        
        # Compute loss (distillation + efficiency)
        # This would integrate with student_forward_fn from sample
        
        # Placeholder - actual implementation needs integration
        loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def _evaluate(self, val_data: List[Dict]) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.router.eval()
        
        # Placeholder
        return {
            'avg_loss': 0.0,
            'cache_ratio': 0.5
        }


def test_schedule_transfer(
    router: ContinuousRouter,
    train_schedule_steps: int,
    test_schedule_steps: int,
    test_data,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Test schedule transfer: train on one schedule, test on another.
    
    Args:
        router: Trained continuous router
        train_schedule_steps: Number of steps used during training (e.g., 256)
        test_schedule_steps: Number of steps to test on (e.g., 128 or 512)
        test_data: Test samples using test_schedule_steps
        device: torch device
    
    Returns:
        Metrics showing transfer performance
    """
    print(f"Testing transfer: trained on {train_schedule_steps} steps, testing on {test_schedule_steps} steps")
    
    router.eval()
    router.to(device)
    
    # Get decisions for test schedule
    decisions = router.get_schedule_decisions(test_schedule_steps, device=device)
    
    # Evaluate on test data
    # (Would integrate with actual Dream evaluation)
    
    return {
        'test_schedule_steps': test_schedule_steps,
        'transfer_accuracy': 0.0,  # Placeholder
        'transfer_speedup': 0.0,   # Placeholder
        'cache_ratio': sum(
            sum(1 for recompute in layer_decisions.values() if not recompute)
            for layer_decisions in decisions.values()
        ) / (len(decisions) * router.config.num_layers)
    }
