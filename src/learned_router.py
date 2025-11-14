# coding=utf-8
"""
Phase B - Learned Router for L2C-style layer caching.
Learns per-layer recompute/reuse decisions on a fixed diffusion schedule.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass


@dataclass
class RouterConfig:
    """Configuration for learned router."""
    num_layers: int
    num_steps: int  # For fixed schedule
    hidden_dim: int = 64
    dropout: float = 0.1
    temperature: float = 1.0  # For Gumbel-Softmax during training


class FixedScheduleRouter(nn.Module):
    """
    Learned router for fixed diffusion schedule (Phase B).
    
    For each cache step m and layer l, maintains a parameter β[m,l] ∈ [0,1]
    that represents the probability of recomputing vs reusing cached output.
    
    Architecture:
        - Simple per-step, per-layer parameters (no input features)
        - During training: optimize β to minimize distillation loss + efficiency regularizer
        - During inference: threshold β to make binary decisions
    
    Note: This is the simplest form. Can be extended to condition on features.
    """
    
    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config
        
        # β[step, layer] parameters
        # Initialize to 0.5 (balanced between recompute and cache)
        self.beta = nn.Parameter(
            torch.ones(config.num_steps, config.num_layers) * 0.5
        )
        
        self.num_params = self.beta.numel()
        print(f"FixedScheduleRouter initialized with {self.num_params:,} parameters")
    
    def forward(self, step_idx: int, layer_idx: int = None) -> torch.Tensor:
        """
        Get recompute probability for given step and layer.
        
        Args:
            step_idx: Diffusion step index
            layer_idx: Layer index (if None, return all layers)
        
        Returns:
            probabilities: Sigmoid(β) values in [0,1]
                          1 = recompute, 0 = cache
        """
        beta_values = self.beta[step_idx]
        if layer_idx is not None:
            beta_values = beta_values[layer_idx]
        
        return torch.sigmoid(beta_values)
    
    def get_decision(
        self,
        step_idx: int,
        layer_idx: int,
        threshold: float = 0.5,
        deterministic: bool = True
    ) -> bool:
        """
        Make binary decision: recompute or cache?
        
        Args:
            step_idx: Diffusion step index
            layer_idx: Layer index
            threshold: Probability threshold (default 0.5)
            deterministic: If True, use threshold; if False, sample
        
        Returns:
            True = recompute, False = cache
        """
        with torch.no_grad():
            prob = self.forward(step_idx, layer_idx).item()
            
            if deterministic:
                return prob >= threshold
            else:
                return torch.rand(1).item() < prob
    
    def get_all_decisions(
        self,
        step_idx: int,
        threshold: float = 0.5
    ) -> Dict[int, bool]:
        """Get decisions for all layers at given step."""
        decisions = {}
        with torch.no_grad():
            probs = self.forward(step_idx)
            for layer_idx in range(self.config.num_layers):
                decisions[layer_idx] = probs[layer_idx].item() >= threshold
        return decisions
    
    def gumbel_softmax_sample(
        self,
        step_idx: int,
        layer_idx: int = None,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Sample using Gumbel-Softmax for differentiable discrete decisions during training.
        
        Returns:
            soft decision in [0,1] that approximates hard binary decision
        """
        logits = self.beta[step_idx]
        if layer_idx is not None:
            logits = logits[layer_idx].unsqueeze(0)
        
        # Binary Gumbel-Softmax: treat as [cache, recompute] logits
        binary_logits = torch.stack([
            -logits,  # cache score (inverse)
            logits    # recompute score
        ], dim=-1)
        
        # Gumbel-Softmax
        gumbel_sample = F.gumbel_softmax(binary_logits, tau=temperature, hard=False)
        
        # Return recompute probability (index 1)
        return gumbel_sample[..., 1]
    
    def save(self, filepath: str, metadata: Dict = None):
        """Save router weights and metadata."""
        from pathlib import Path
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.state_dict(),
            'config': {
                'num_layers': self.config.num_layers,
                'num_steps': self.config.num_steps,
                'hidden_dim': self.config.hidden_dim,
                'dropout': self.config.dropout,
                'temperature': self.config.temperature
            },
            'num_params': self.num_params
        }
        
        if metadata is not None:
            checkpoint['metadata'] = metadata
        
        torch.save(checkpoint, filepath)
        print(f"✓ Saved FixedScheduleRouter to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'FixedScheduleRouter':
        """Load router from checkpoint."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        config_dict = checkpoint['config']
        config = RouterConfig(
            num_layers=config_dict['num_layers'],
            num_steps=config_dict['num_steps'],
            hidden_dim=config_dict.get('hidden_dim', 64),
            dropout=config_dict.get('dropout', 0.1),
            temperature=config_dict.get('temperature', 1.0)
        )
        
        router = FixedScheduleRouter(config)
        router.load_state_dict(checkpoint['state_dict'])
        
        print(f"✓ Loaded FixedScheduleRouter from {filepath} ({router.num_params:,} params)")
        return router


class RouterTrainer:
    """
    Trainer for learned router using distillation from teacher outputs.
    
    Training strategy:
        1. Run teacher (no caching) on training prompts, collect denoiser outputs at each step
        2. For each prompt, simulate different router decisions (which layers to cache)
        3. Optimize β to minimize:
           - Distillation loss: ||student_output - teacher_output||
           - Efficiency regularizer: encourage caching (minimize recompute ratio)
    """
    
    def __init__(
        self,
        router: FixedScheduleRouter,
        learning_rate: float = 1e-3,
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
    
    def distillation_loss(
        self,
        student_outputs: Dict[int, torch.Tensor],  # {layer_idx: output}
        teacher_outputs: Dict[int, torch.Tensor],
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute distillation loss between student (with caching) and teacher (no caching).
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_idx in teacher_outputs.keys():
            if layer_idx not in student_outputs:
                continue
            
            teacher_out = teacher_outputs[layer_idx]
            student_out = student_outputs[layer_idx]
            
            # MSE loss on outputs
            if mask is not None:
                # Only compute on masked positions
                loss = F.mse_loss(student_out[mask], teacher_out[mask])
            else:
                loss = F.mse_loss(student_out, teacher_out)
            
            total_loss += loss
            num_layers += 1
        
        return total_loss / max(1, num_layers)
    
    def efficiency_regularizer(
        self,
        router_decisions: Dict[int, torch.Tensor]  # {layer_idx: recompute_prob}
    ) -> torch.Tensor:
        """
        Regularizer that encourages caching (lower recompute probability).
        L_eff = mean(β) → minimize this to prefer caching
        """
        all_probs = torch.stack(list(router_decisions.values()))
        return all_probs.mean()
    
    def train_step(
        self,
        step_idx: int,
        teacher_outputs: Dict[int, torch.Tensor],
        student_forward_fn,  # Function that takes router decisions and returns student outputs
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Train router for one step.
        
        Args:
            step_idx: Current diffusion step
            teacher_outputs: Ground truth outputs from teacher (no caching)
            student_forward_fn: Function(decisions) -> student_outputs
                               Takes dict of layer decisions, runs student model
            mask: Optional mask for computing loss only on relevant positions
        
        Returns:
            loss: Total loss
            info: Dictionary with loss components
        """
        self.router.train()
        self.optimizer.zero_grad()
        
        # Get router decisions (differentiable via Gumbel-Softmax)
        router_decisions = {}
        for layer_idx in range(self.router.config.num_layers):
            router_decisions[layer_idx] = self.router.gumbel_softmax_sample(
                step_idx, layer_idx, temperature=self.router.config.temperature
            )
        
        # Run student model with these decisions
        student_outputs = student_forward_fn(router_decisions)
        
        # Compute losses
        distill_loss = self.distillation_loss(student_outputs, teacher_outputs, mask)
        eff_loss = self.efficiency_regularizer(router_decisions)
        
        total_loss = distill_loss + self.efficiency_weight * eff_loss
        
        # Backward
        total_loss.backward()
        self.optimizer.step()
        
        info = {
            'total_loss': total_loss.item(),
            'distill_loss': distill_loss.item(),
            'efficiency_loss': eff_loss.item(),
            'avg_recompute_prob': eff_loss.item()
        }
        
        return total_loss, info
    
    def evaluate(
        self,
        val_data: List[Dict],  # List of {step_idx, teacher_outputs, student_forward_fn}
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Evaluate router on validation set."""
        self.router.eval()
        
        total_distill_loss = 0.0
        total_recompute_ratio = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for sample in val_data:
                step_idx = sample['step_idx']
                teacher_outputs = sample['teacher_outputs']
                student_forward_fn = sample['student_forward_fn']
                mask = sample.get('mask', None)
                
                # Get hard decisions
                decisions = self.router.get_all_decisions(step_idx, threshold)
                
                # Convert to tensor for student_forward_fn
                decision_tensor = {
                    layer_idx: torch.tensor(1.0 if recompute else 0.0, device=self.device)
                    for layer_idx, recompute in decisions.items()
                }
                
                # Run student
                student_outputs = student_forward_fn(decision_tensor)
                
                # Compute loss
                distill_loss = self.distillation_loss(student_outputs, teacher_outputs, mask)
                recompute_ratio = sum(decisions.values()) / len(decisions)
                
                total_distill_loss += distill_loss.item()
                total_recompute_ratio += recompute_ratio
                num_samples += 1
        
        return {
            'distill_loss': total_distill_loss / num_samples,
            'recompute_ratio': total_recompute_ratio / num_samples,
            'cache_ratio': 1.0 - (total_recompute_ratio / num_samples)
        }
    
    def train(
        self,
        train_data: List[Dict],
        val_data: List[Dict],
        num_epochs: int = 50,
        early_stopping_patience: int = 10
    ):
        """Full training loop."""
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Training RouterTrainer for {num_epochs} epochs...")
        print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_distill = 0.0
            epoch_eff = 0.0
            
            # Shuffle training data
            np.random.shuffle(train_data)
            
            for sample in train_data:
                loss, info = self.train_step(
                    sample['step_idx'],
                    sample['teacher_outputs'],
                    sample['student_forward_fn'],
                    sample.get('mask', None)
                )
                
                epoch_loss += info['total_loss']
                epoch_distill += info['distill_loss']
                epoch_eff += info['efficiency_loss']
            
            avg_loss = epoch_loss / len(train_data)
            avg_distill = epoch_distill / len(train_data)
            avg_eff = epoch_eff / len(train_data)
            
            self.train_losses.append(avg_loss)
            
            # Validation
            val_metrics = self.evaluate(val_data)
            self.val_losses.append(val_metrics['distill_loss'])
            
            print(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {avg_loss:.4f} (Distill: {avg_distill:.4f}, Eff: {avg_eff:.4f}), "
                f"Val Loss: {val_metrics['distill_loss']:.4f}, "
                f"Cache Ratio: {val_metrics['cache_ratio']:.2%}"
            )
            
            # Early stopping
            if val_metrics['distill_loss'] < best_val_loss:
                best_val_loss = val_metrics['distill_loss']
                patience_counter = 0
                self.router.save(
                    'checkpoints/learned_router_best.pt',
                    metadata={'epoch': epoch, 'val_loss': best_val_loss}
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Training complete. Best val loss: {best_val_loss:.4f}")


# Utility functions for data preparation

def collect_teacher_traces(
    model,
    prompts: List[str],
    diffusion_steps: int,
    device: str = 'cuda'
) -> List[Dict]:
    """
    Run teacher model (no caching) and collect intermediate layer outputs
    for router training.
    
    Returns:
        List of training samples, each containing:
        {
            'prompt_id': int,
            'step_idx': int,
            'teacher_outputs': {layer_idx: tensor},
            'mask': tensor (optional)
        }
    """
    # This is a placeholder - actual implementation would integrate with
    # Dream's generation loop and capture intermediate states
    raise NotImplementedError("collect_teacher_traces needs integration with Dream model")


def create_student_forward_fn(
    model,
    x: torch.Tensor,
    step_idx: int,
    cache: Dict[int, torch.Tensor]
):
    """
    Create a forward function for student model that respects router decisions.
    
    Args:
        model: Dream model
        x: Current input tokens
        step_idx: Current diffusion step
        cache: Cached layer outputs from previous step
    
    Returns:
        Function that takes router decisions and returns layer outputs
    """
    def forward_fn(decisions: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        Args:
            decisions: {layer_idx: recompute_prob} (0=cache, 1=recompute)
        
        Returns:
            {layer_idx: output_tensor}
        """
        # This would integrate with Dream's forward pass
        # For each layer:
        #   if decision[layer] > 0.5 or cache is None: recompute
        #   else: use cached output
        raise NotImplementedError("Student forward_fn needs Dream integration")
    
    return forward_fn
