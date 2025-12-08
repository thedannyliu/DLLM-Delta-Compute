# coding=utf-8
"""
Phase E - FlexiDepth-Style Layer Skipping (Router + Adapter)

Extends Phase D by replacing the identity skip branch with a trainable
adapter network. The adapter is a small FFN that provides a cheap
surrogate for the full FFN when skipping.

Architecture per adapted layer:
- Router: g_ℓ(p, h_in) ∈ (0, 1) - decides deep vs shallow path
- Adapter: narrow FFN (d_r << d_ff) as cheap FFN surrogate

Forward:
    if g_ℓ > τ (deep path):
        ffn_out = FFN(Norm(h_in))
    else (shallow path):
        ffn_out = Adapter(Norm(h_in))
    h_out = h_in + ffn_out

Training: Backbone frozen, only router + adapter trained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import math

from .continuous_router import ContinuousRouter, ContinuousRouterConfig, TimeEncoder


@dataclass
class FlexiAdapterConfig:
    """Configuration for FlexiDepth adapter."""
    hidden_dim: int  # Model hidden dimension (d)
    adapter_dim: int = 256  # Bottleneck dimension (d_r << d_ff)
    activation: str = 'gelu'  # Activation function
    

@dataclass 
class FlexiRouterConfig:
    """Configuration for FlexiDepth router + adapter (Phase E)."""
    num_layers: int
    hidden_dim: int  # Model hidden dimension
    adapted_layers: List[int] = field(default_factory=list)  # Which layers have adapters
    # Router config
    time_embedding_dim: int = 64
    router_hidden_dim: int = 128
    router_num_layers: int = 2
    dropout: float = 0.1
    time_encoding: str = 'sinusoidal'
    # Adapter config
    adapter_dim: int = 256
    adapter_activation: str = 'gelu'
    # Training
    skip_loss_weight: float = 0.1
    target_skip_ratio: float = 0.3  # Target ~30% using shallow path


class FlexiAdapter(nn.Module):
    """
    Lightweight adapter FFN as cheap surrogate for full FFN.
    
    Structure: mini-FFN with bottleneck
        Adapter(x) = W_up * φ(W_down * x)
    
    where φ is activation, W_down ∈ R^{d_r × d}, W_up ∈ R^{d × d_r}
    """
    
    def __init__(self, config: FlexiAdapterConfig):
        super().__init__()
        self.config = config
        
        # Bottleneck structure
        self.down_proj = nn.Linear(config.hidden_dim, config.adapter_dim, bias=False)
        self.up_proj = nn.Linear(config.adapter_dim, config.hidden_dim, bias=False)
        
        if config.activation == 'gelu':
            self.activation = nn.GELU()
        elif config.activation == 'silu':
            self.activation = nn.SiLU()
        elif config.activation == 'relu':
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown activation: {config.activation}")
        
        # Initialize with small weights
        nn.init.normal_(self.down_proj.weight, std=0.02)
        nn.init.zeros_(self.up_proj.weight)
        
        self.num_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adapter.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
        
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        return self.up_proj(self.activation(self.down_proj(x)))


class FlexiRouter(nn.Module):
    """
    FlexiDepth-style router for Phase E.
    
    For each adapted layer, outputs gate score g_ℓ(p, h_in) ∈ (0, 1).
    Uses progress p and optionally pooled hidden state for routing.
    """
    
    def __init__(self, config: FlexiRouterConfig):
        super().__init__()
        self.config = config
        
        # Time encoder (progress)
        self.time_encoder = TimeEncoder(
            embedding_dim=config.time_embedding_dim,
            encoding_type=config.time_encoding
        )
        
        # Layer embeddings only for adapted layers
        self.adapted_layers = config.adapted_layers if config.adapted_layers else list(range(config.num_layers))
        self.num_adapted = len(self.adapted_layers)
        
        self.layer_embedding = nn.Embedding(
            num_embeddings=config.num_layers,
            embedding_dim=config.time_embedding_dim
        )
        
        # Optional hidden state projection (for input-dependent routing)
        self.hidden_proj = nn.Linear(config.hidden_dim, config.time_embedding_dim)
        
        # MLP: [time_emb || layer_emb || hidden_emb] -> g
        mlp_input_dim = config.time_embedding_dim * 3
        layers = []
        prev_dim = mlp_input_dim
        
        for i in range(config.router_num_layers):
            layers.extend([
                nn.Linear(prev_dim, config.router_hidden_dim),
                nn.SiLU(),
                nn.Dropout(config.dropout)
            ])
            prev_dim = config.router_hidden_dim
        
        layers.extend([
            nn.Linear(prev_dim, 1),
            nn.Sigmoid()
        ])
        
        self.mlp = nn.Sequential(*layers)
        
        self.num_params = sum(p.numel() for p in self.parameters())
        print(f"FlexiRouter initialized with {self.num_params:,} parameters")
    
    def forward(
        self,
        progress: torch.Tensor,  # [batch] or scalar
        layer_idx: int,
        hidden_state: torch.Tensor = None,  # [batch, seq, hidden_dim]
    ) -> torch.Tensor:
        """
        Compute gate score for a specific layer.
        
        Args:
            progress: Completion ratio p ∈ [0, 1]
            layer_idx: Layer index
            hidden_state: Optional hidden state for input-dependent routing
        
        Returns:
            Gate score g ∈ (0, 1). Shape: [batch] or scalar
        """
        # Encode progress
        time_emb = self.time_encoder(progress)  # [batch, D] or [D]
        
        # Layer embedding
        l_tensor = torch.tensor(layer_idx, dtype=torch.long, device=time_emb.device)
        layer_emb = self.layer_embedding(l_tensor)  # [D]
        
        # Hidden state embedding (pool over sequence)
        if hidden_state is not None:
            # Mean pooling over sequence
            h_pooled = hidden_state.mean(dim=1)  # [batch, hidden_dim]
            h_emb = self.hidden_proj(h_pooled)  # [batch, D]
        else:
            # Use zeros if no hidden state provided
            if time_emb.dim() == 1:
                h_emb = torch.zeros(self.config.time_embedding_dim, device=time_emb.device)
            else:
                h_emb = torch.zeros(time_emb.shape[0], self.config.time_embedding_dim, device=time_emb.device)
        
        # Broadcast and concatenate
        if time_emb.dim() == 1:
            combined = torch.cat([time_emb, layer_emb, h_emb], dim=-1)
        else:
            layer_emb = layer_emb.unsqueeze(0).expand(time_emb.shape[0], -1)
            combined = torch.cat([time_emb, layer_emb, h_emb], dim=-1)
        
        # MLP
        gate = self.mlp(combined).squeeze(-1)
        
        return gate
    
    def get_all_gates(
        self,
        progress: float,
        hidden_states: Dict[int, torch.Tensor] = None,
        device: str = 'cuda'
    ) -> Dict[int, float]:
        """
        Get gate scores for all adapted layers.
        
        Args:
            progress: Completion ratio p
            hidden_states: Optional dict {layer_idx: hidden_state}
            device: torch device
        
        Returns:
            Dict {layer_idx: gate_score}
        """
        self.eval()
        gates = {}
        
        with torch.no_grad():
            p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
            
            for layer_idx in self.adapted_layers:
                h = hidden_states.get(layer_idx) if hidden_states else None
                gate = self.forward(p_tensor, layer_idx, h)
                gates[layer_idx] = gate.item() if gate.dim() == 0 else gate.mean().item()
        
        return gates


class FlexiDepthModule(nn.Module):
    """
    Complete FlexiDepth module for a single layer.
    
    Combines router and adapter for dynamic depth control.
    """
    
    def __init__(
        self,
        layer_idx: int,
        router: FlexiRouter,
        hidden_dim: int,
        adapter_dim: int = 256,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.router = router
        
        # Create adapter for this layer
        adapter_config = FlexiAdapterConfig(
            hidden_dim=hidden_dim,
            adapter_dim=adapter_dim,
            activation=activation
        )
        self.adapter = FlexiAdapter(adapter_config)
        
        self.num_params = self.adapter.num_params
        print(f"FlexiDepthModule[layer {layer_idx}]: adapter has {self.num_params:,} parameters")
    
    def forward(
        self,
        h_in: torch.Tensor,
        ffn_fn: callable,
        progress: float,
        threshold: float = 0.5,
        training: bool = False,
        device: str = 'cuda'
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass with dynamic depth selection.
        
        Args:
            h_in: Input hidden state [batch, seq, hidden_dim]
            ffn_fn: Function to compute full FFN output
            progress: Completion ratio p
            threshold: Gate threshold (inference only)
            training: Whether in training mode
            device: torch device
        
        Returns:
            h_out: Output hidden state
            stats: Dict with routing statistics
        """
        p_tensor = torch.tensor(progress, dtype=torch.float32, device=device)
        
        # Get gate score
        gate = self.router(p_tensor, self.layer_idx, h_in)  # [batch] or scalar
        
        if training:
            # Soft combination for gradient flow
            ffn_out = ffn_fn(h_in)  # Full FFN output
            adapter_out = self.adapter(h_in)  # Adapter output
            
            # Weighted combination: g * FFN + (1-g) * Adapter
            if gate.dim() == 0:
                combined = gate * ffn_out + (1 - gate) * adapter_out
            else:
                gate = gate.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
                combined = gate * ffn_out + (1 - gate) * adapter_out
            
            h_out = h_in + combined
            used_deep = True  # Both paths used in training
        else:
            # Hard decision at inference
            use_deep = gate.mean().item() >= threshold if gate.dim() > 0 else gate.item() >= threshold
            
            if use_deep:
                ffn_out = ffn_fn(h_in)
            else:
                ffn_out = self.adapter(h_in)
            
            h_out = h_in + ffn_out
            used_deep = use_deep
        
        stats = {
            'layer_idx': self.layer_idx,
            'gate_score': gate.mean().item() if gate.dim() > 0 else gate.item(),
            'used_deep_path': used_deep,
        }
        
        return h_out, stats


class FlexiDepthManager:
    """
    Manager for all FlexiDepth modules across layers.
    
    Handles initialization, saving/loading, and training coordination.
    """
    
    def __init__(self, config: FlexiRouterConfig):
        self.config = config
        
        # Create router
        self.router = FlexiRouter(config)
        
        # Create adapters for specified layers
        self.modules: Dict[int, FlexiDepthModule] = {}
        for layer_idx in config.adapted_layers:
            module = FlexiDepthModule(
                layer_idx=layer_idx,
                router=self.router,
                hidden_dim=config.hidden_dim,
                adapter_dim=config.adapter_dim,
                activation=config.adapter_activation
            )
            self.modules[layer_idx] = module
        
        self.total_adapter_params = sum(m.num_params for m in self.modules.values())
        self.total_params = self.router.num_params + self.total_adapter_params
        
        print(f"FlexiDepthManager: {len(self.modules)} adapted layers, "
              f"{self.total_params:,} total trainable params")
    
    def get_all_parameters(self) -> List[nn.Parameter]:
        """Get all trainable parameters (router + all adapters)."""
        params = list(self.router.parameters())
        for module in self.modules.values():
            params.extend(module.adapter.parameters())
        return params
    
    def to(self, device: str) -> 'FlexiDepthManager':
        """Move all modules to device."""
        self.router = self.router.to(device)
        for layer_idx in self.modules:
            self.modules[layer_idx] = self.modules[layer_idx].to(device)
        return self
    
    def train(self):
        """Set to training mode."""
        self.router.train()
        for module in self.modules.values():
            module.train()
    
    def eval(self):
        """Set to evaluation mode."""
        self.router.eval()
        for module in self.modules.values():
            module.eval()
    
    def save(self, filepath: str, metadata: Dict = None):
        """Save router and all adapters."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'type': 'flexi_depth',
            'router_state_dict': self.router.state_dict(),
            'adapter_state_dicts': {
                layer_idx: module.adapter.state_dict()
                for layer_idx, module in self.modules.items()
            },
            'config': {
                'num_layers': self.config.num_layers,
                'hidden_dim': self.config.hidden_dim,
                'adapted_layers': self.config.adapted_layers,
                'time_embedding_dim': self.config.time_embedding_dim,
                'router_hidden_dim': self.config.router_hidden_dim,
                'router_num_layers': self.config.router_num_layers,
                'dropout': self.config.dropout,
                'time_encoding': self.config.time_encoding,
                'adapter_dim': self.config.adapter_dim,
                'adapter_activation': self.config.adapter_activation,
                'skip_loss_weight': self.config.skip_loss_weight,
                'target_skip_ratio': self.config.target_skip_ratio,
            },
            'total_params': self.total_params,
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved FlexiDepthManager to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FlexiDepthManager':
        """Load from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config_dict = checkpoint['config']
        config = FlexiRouterConfig(
            num_layers=config_dict['num_layers'],
            hidden_dim=config_dict['hidden_dim'],
            adapted_layers=config_dict['adapted_layers'],
            time_embedding_dim=config_dict.get('time_embedding_dim', 64),
            router_hidden_dim=config_dict.get('router_hidden_dim', 128),
            router_num_layers=config_dict.get('router_num_layers', 2),
            dropout=config_dict.get('dropout', 0.1),
            time_encoding=config_dict.get('time_encoding', 'sinusoidal'),
            adapter_dim=config_dict.get('adapter_dim', 256),
            adapter_activation=config_dict.get('adapter_activation', 'gelu'),
            skip_loss_weight=config_dict.get('skip_loss_weight', 0.1),
            target_skip_ratio=config_dict.get('target_skip_ratio', 0.3),
        )
        
        manager = FlexiDepthManager(config)
        manager.router.load_state_dict(checkpoint['router_state_dict'])
        
        for layer_idx, state_dict in checkpoint['adapter_state_dicts'].items():
            if layer_idx in manager.modules:
                manager.modules[layer_idx].adapter.load_state_dict(state_dict)
        
        print(f"✓ Loaded FlexiDepthManager from {filepath} ({manager.total_params:,} params)")
        return manager
