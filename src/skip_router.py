# coding=utf-8
"""
Phase D - Continuous Layer Skipping (No Cache)

Reuses the ContinuousRouter from Phase C but changes the action space from
"recompute vs cache" to "recompute vs skip". The router outputs β_ℓ(p) ∈ [0,1]
interpreted as probability of recomputing this layer's FFN.

At training time, use soft combination:
    h_out = h_in + β_ℓ(p) * FFN(h_in)

At inference time, threshold β to get binary "recompute vs skip" decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import math

from .continuous_router import ContinuousRouter, ContinuousRouterConfig


@dataclass
class SkipRouterConfig:
    """Configuration for skip router (Phase D)."""
    num_layers: int
    time_embedding_dim: int = 64
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    dropout: float = 0.1
    time_encoding: str = 'sinusoidal'
    # Training regularization
    skip_loss_weight: float = 0.1  # Weight for skip regularizer
    target_skip_ratio: float = 0.25  # Target ~25% FFN skip


class SkipRouter(ContinuousRouter):
    """
    Phase D Skip Router - extends ContinuousRouter with skip semantics.
    
    The key difference from Phase C caching:
    - Phase C: β=1 means recompute, β=0 means use cached FFN output
    - Phase D: β=1 means recompute, β=0 means skip FFN entirely (identity)
    
    The router architecture is identical, only the interpretation changes.
    """
    
    def __init__(self, config: SkipRouterConfig):
        # Create ContinuousRouterConfig from SkipRouterConfig
        continuous_config = ContinuousRouterConfig(
            num_layers=config.num_layers,
            time_embedding_dim=config.time_embedding_dim,
            hidden_dim=config.hidden_dim,
            num_hidden_layers=config.num_hidden_layers,
            dropout=config.dropout,
            time_encoding=config.time_encoding
        )
        super().__init__(continuous_config)
        self.skip_config = config
        print(f"SkipRouter initialized for Phase D (skip semantics)")
    
    def get_skip_decisions(
        self,
        progress: float,
        threshold: float = 0.5,
        device: str = 'cuda'
    ) -> List[bool]:
        """
        Get binary skip decisions for all layers at given progress.
        
        Args:
            progress: Completion ratio p ∈ [0, 1]
            threshold: Decision threshold
            device: torch device
        
        Returns:
            List[bool]: True = recompute FFN, False = skip FFN (identity)
        """
        self.eval()
        with torch.no_grad():
            p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
            layer_indices = torch.arange(self.config.num_layers, device=device)
            probs = self.forward(p_tensor, layer_indices)
            return [p.item() >= threshold for p in probs]
    
    def soft_skip_forward(
        self,
        h_in: torch.Tensor,
        ffn_output: torch.Tensor,
        layer_idx: int,
        progress: float,
        device: str = 'cuda'
    ) -> torch.Tensor:
        """
        Soft skip combination for training (differentiable).
        
        h_out = h_in + β_ℓ(p) * FFN(h_in)
        
        Args:
            h_in: Input hidden state [batch, seq_len, hidden_dim]
            ffn_output: Output from FFN(h_in) [batch, seq_len, hidden_dim]
            layer_idx: Current layer index
            progress: Completion ratio p ∈ [0, 1]
            device: torch device
        
        Returns:
            h_out: Combined output
        """
        p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
        l_tensor = torch.tensor(layer_idx, dtype=torch.long, device=device)
        beta = self.forward(p_tensor, l_tensor)  # scalar in [0, 1]
        
        # Soft combination: h_out = h_in + β * FFN(h_in)
        # When β=1: h_out = h_in + FFN(h_in) (full forward)
        # When β=0: h_out = h_in (identity skip)
        return h_in + beta * ffn_output
    
    def save(self, filepath: str, metadata: Dict = None):
        """Save skip router weights and metadata."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'type': 'skip_router',
            'state_dict': self.state_dict(),
            'config': {
                'num_layers': self.config.num_layers,
                'time_embedding_dim': self.config.time_embedding_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_hidden_layers': self.config.num_hidden_layers,
                'dropout': self.config.dropout,
                'time_encoding': self.config.time_encoding,
            },
            'skip_config': {
                'skip_loss_weight': self.skip_config.skip_loss_weight,
                'target_skip_ratio': self.skip_config.target_skip_ratio,
            },
            'num_params': self.num_params
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved SkipRouter to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'SkipRouter':
        """Load skip router from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config_dict = checkpoint['config']
        skip_config_dict = checkpoint.get('skip_config', {})
        
        config = SkipRouterConfig(
            num_layers=config_dict['num_layers'],
            time_embedding_dim=config_dict.get('time_embedding_dim', 64),
            hidden_dim=config_dict.get('hidden_dim', 128),
            num_hidden_layers=config_dict.get('num_hidden_layers', 2),
            dropout=config_dict.get('dropout', 0.1),
            time_encoding=config_dict.get('time_encoding', 'sinusoidal'),
            skip_loss_weight=skip_config_dict.get('skip_loss_weight', 0.1),
            target_skip_ratio=skip_config_dict.get('target_skip_ratio', 0.25),
        )
        
        router = SkipRouter(config)
        router.load_state_dict(checkpoint['state_dict'])
        
        print(f"✓ Loaded SkipRouter from {filepath} ({router.num_params:,} params)")
        return router
    
    @staticmethod
    def from_continuous_router(continuous_router_path: str) -> 'SkipRouter':
        """
        Initialize SkipRouter from a trained Phase C ContinuousRouter.
        
        This enables warm-starting Phase D training from Phase C.
        """
        continuous = ContinuousRouter.load(continuous_router_path)
        
        config = SkipRouterConfig(
            num_layers=continuous.config.num_layers,
            time_embedding_dim=continuous.config.time_embedding_dim,
            hidden_dim=continuous.config.hidden_dim,
            num_hidden_layers=continuous.config.num_hidden_layers,
            dropout=continuous.config.dropout,
            time_encoding=continuous.config.time_encoding,
        )
        
        skip_router = SkipRouter(config)
        skip_router.load_state_dict(continuous.state_dict())
        
        print(f"✓ Initialized SkipRouter from ContinuousRouter ({skip_router.num_params:,} params)")
        return skip_router


class SkipRouterTrainer:
    """
    Trainer for Phase D SkipRouter.
    
    Training objective:
    - Main loss: LM/diffusion loss on GSM8K (backbone frozen, router trainable)
    - Skip regularizer: Encourage skipping (lower depth)
      L_skip = (1/T) * sum_t [ (sum_l β_l(p_t))^2 ]
    """
    
    def __init__(
        self,
        router: SkipRouter,
        learning_rate: float = 1e-4,
        skip_weight: float = 0.1,
        target_skip_ratio: float = 0.25,
        device: str = 'cuda'
    ):
        self.router = router.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(router.parameters(), lr=learning_rate)
        self.skip_weight = skip_weight
        self.target_skip_ratio = target_skip_ratio
        
        self.train_losses = []
        self.val_losses = []
        self.skip_ratios = []
    
    def compute_skip_loss(
        self,
        progress_values: torch.Tensor,  # [batch]
    ) -> torch.Tensor:
        """
        Compute skip regularization loss.
        
        L_skip = mean_t [ (sum_l β_l(p_t))^2 / num_layers^2 ]
        
        This encourages lower average depth while being scale-invariant.
        """
        batch_size = progress_values.shape[0]
        num_layers = self.router.config.num_layers
        
        # Get all β values: [batch, num_layers]
        layer_indices = torch.arange(num_layers, device=self.device)
        betas = []
        for p in progress_values:
            beta_layer = self.router(p, layer_indices)  # [num_layers]
            betas.append(beta_layer)
        betas = torch.stack(betas)  # [batch, num_layers]
        
        # Compute regularizer: squared sum of β, normalized
        depth_per_sample = betas.sum(dim=1)  # [batch] - effective depth
        normalized_depth = depth_per_sample / num_layers  # [batch] - in [0, 1]
        
        # Penalize depth above target (encourage skipping)
        target_depth = 1.0 - self.target_skip_ratio
        excess_depth = F.relu(normalized_depth - target_depth)
        skip_loss = (excess_depth ** 2).mean()
        
        return skip_loss
    
    def train_step(
        self,
        progress: float,
        ffn_outputs: List[torch.Tensor],  # Per-layer FFN outputs
        hidden_states: List[torch.Tensor],  # Per-layer input hidden states
        target_outputs: torch.Tensor,  # Teacher final output
        main_loss_fn: callable,  # LM loss function
    ) -> Dict[str, float]:
        """
        Single training step with soft skip combination.
        
        Args:
            progress: Completion ratio p ∈ [0, 1]
            ffn_outputs: List of FFN outputs per layer
            hidden_states: List of hidden states per layer
            target_outputs: Teacher's final output for distillation
            main_loss_fn: Function to compute LM loss
        
        Returns:
            Dict with loss components
        """
        self.router.train()
        
        p_tensor = torch.tensor(progress, dtype=torch.float32, device=self.device)
        
        # Get β values for all layers
        layer_indices = torch.arange(self.router.config.num_layers, device=self.device)
        betas = self.router(p_tensor, layer_indices)  # [num_layers]
        
        # Compute student output with soft skip
        # This would require integration with the model forward pass
        # For now, compute regularizer only
        skip_loss = self.compute_skip_loss(p_tensor.unsqueeze(0))
        
        # Total loss (main_loss would be added in full implementation)
        total_loss = self.skip_weight * skip_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Track skip ratio
        with torch.no_grad():
            skip_ratio = 1.0 - betas.mean().item()
            self.skip_ratios.append(skip_ratio)
        
        return {
            'total_loss': total_loss.item(),
            'skip_loss': skip_loss.item(),
            'skip_ratio': skip_ratio,
        }
