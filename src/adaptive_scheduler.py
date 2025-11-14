# coding=utf-8
"""
P4 - Adaptive Step Scheduling for Dream diffusion generation.
Dynamically adjusts diffusion step stride based on local truncation error (LTE)
and risk signals (entropy, KL divergence).
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass
import numpy as np


@dataclass
class SchedulerState:
    """State maintained by adaptive scheduler across steps."""
    current_step: int
    current_stride: int
    total_steps: int
    
    # History for adaptive decisions
    recent_lte: List[float]  # Local truncation errors
    recent_entropy: List[float]  # Entropy values
    recent_kl: List[float]  # KL divergences
    
    # Statistics
    total_steps_taken: int = 0
    total_steps_skipped: int = 0
    stride_changes: int = 0


class AdaptiveScheduler:
    """
    Adaptive step scheduler that adjusts diffusion step stride based on:
    1. Local Truncation Error (LTE): Euler vs Heun step comparison
    2. Entropy: High entropy = uncertain, need smaller stride
    3. KL divergence: Large changes = need smaller stride
    
    Controller logic:
        - Start with stride=1 (safe)
        - If LTE, entropy, KL all small for N consecutive steps -> increase stride
        - If any spike detected -> reduce stride back to 1
        - Never skip early steps (first K steps always stride=1)
    """
    
    def __init__(
        self,
        min_stride: int = 1,
        max_stride: int = 4,
        lte_threshold: float = 0.01,
        entropy_threshold: float = 2.0,
        kl_threshold: float = 0.1,
        stable_window: int = 5,
        min_safe_step: int = 10,
        device: str = 'cuda'
    ):
        """
        Args:
            min_stride: Minimum stride (always 1 for safety)
            max_stride: Maximum stride (how many steps to skip at once)
            lte_threshold: Threshold for local truncation error
            entropy_threshold: Threshold for entropy (lower = more confident)
            kl_threshold: Threshold for KL divergence between steps
            stable_window: Number of consecutive stable steps before increasing stride
            min_safe_step: Never adapt stride before this step
            device: torch device
        """
        self.min_stride = min_stride
        self.max_stride = max_stride
        self.lte_threshold = lte_threshold
        self.entropy_threshold = entropy_threshold
        self.kl_threshold = kl_threshold
        self.stable_window = stable_window
        self.min_safe_step = min_safe_step
        self.device = device
        
        # History
        self.prev_logits: Optional[torch.Tensor] = None
        self.prev_hidden: Optional[torch.Tensor] = None
    
    def estimate_lte_euler_heun(
        self,
        model,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        step_idx: int,
        total_steps: int
    ) -> float:
        """
        Estimate local truncation error by comparing Euler and Heun steps.
        
        Euler: x_{n+1} = x_n + h * f(x_n)
        Heun: x_{n+1} = x_n + h/2 * (f(x_n) + f(x_n + h*f(x_n)))
        
        LTE ≈ ||Heun - Euler||
        
        For diffusion, f() is the denoising update based on model prediction.
        """
        with torch.no_grad():
            # Get current prediction (Euler step)
            probs_euler = F.softmax(mask_logits, dim=-1)
            pred_tokens_euler, _ = probs_euler.max(dim=-1)
            
            # Estimate Heun: would need one more model forward
            # For efficiency, use a simplified approximation:
            # Compare confidence change rate
            if self.prev_logits is not None and self.prev_logits.shape == mask_logits.shape:
                # Approximate LTE as change in logits
                logit_change = torch.norm(mask_logits - self.prev_logits, p=2)
                lte = logit_change.item() / (mask_logits.numel() ** 0.5)  # Normalized
            else:
                lte = 0.0  # First step, no history
            
            self.prev_logits = mask_logits.clone()
        
        return lte
    
    def compute_entropy(self, logits: torch.Tensor, mask: torch.Tensor = None) -> float:
        """
        Compute entropy of predicted distribution.
        Higher entropy = more uncertainty = should use smaller stride.
        """
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            epsilon = 1e-10
            log_probs = torch.log(probs + epsilon)
            entropy = -torch.sum(probs * log_probs, dim=-1)
            
            if mask is not None:
                # Only compute on masked positions
                entropy = entropy[mask]
            
            return entropy.mean().item()
    
    def compute_kl_divergence(
        self,
        current_logits: torch.Tensor,
        prev_logits: torch.Tensor,
        mask: torch.Tensor = None
    ) -> float:
        """
        Compute KL divergence between current and previous step predictions.
        Large KL = distribution changed significantly = should use smaller stride.
        """
        if prev_logits is None:
            return 0.0
        
        with torch.no_grad():
            current_probs = F.softmax(current_logits, dim=-1)
            prev_probs = F.softmax(prev_logits, dim=-1)
            
            epsilon = 1e-10
            kl = torch.sum(
                current_probs * (torch.log(current_probs + epsilon) - torch.log(prev_probs + epsilon)),
                dim=-1
            )
            
            if mask is not None:
                kl = kl[mask]
            
            return kl.mean().item()
    
    def decide_stride(
        self,
        state: SchedulerState,
        lte: float,
        entropy: float,
        kl: float
    ) -> int:
        """
        Decide stride for next step based on current signals.
        
        Logic:
            1. If before min_safe_step: always stride=1
            2. If any signal above threshold: reduce stride to 1
            3. If all signals below threshold for stable_window steps: increase stride
            4. Otherwise: keep current stride
        """
        # Never adapt early steps
        if state.current_step < self.min_safe_step:
            return self.min_stride
        
        # Update history
        state.recent_lte.append(lte)
        state.recent_entropy.append(entropy)
        state.recent_kl.append(kl)
        
        # Keep only recent window
        if len(state.recent_lte) > self.stable_window:
            state.recent_lte.pop(0)
            state.recent_entropy.pop(0)
            state.recent_kl.pop(0)
        
        # Check if any signal is above threshold (unstable)
        unstable = (
            lte > self.lte_threshold or
            entropy > self.entropy_threshold or
            kl > self.kl_threshold
        )
        
        if unstable:
            # Reduce stride immediately
            new_stride = self.min_stride
            if new_stride != state.current_stride:
                state.stride_changes += 1
            return new_stride
        
        # Check if stable for window
        if len(state.recent_lte) >= self.stable_window:
            all_stable = (
                all(e < self.lte_threshold for e in state.recent_lte) and
                all(h < self.entropy_threshold for h in state.recent_entropy) and
                all(k < self.kl_threshold for k in state.recent_kl)
            )
            
            if all_stable and state.current_stride < self.max_stride:
                # Gradually increase stride
                new_stride = min(state.current_stride + 1, self.max_stride)
                state.stride_changes += 1
                return new_stride
        
        # Keep current stride
        return state.current_stride
    
    def get_next_step_index(
        self,
        model,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_token_id: int,
        state: SchedulerState
    ) -> Tuple[int, Dict]:
        """
        Compute next step index with adaptive stride.
        
        Returns:
            next_step: The index of the next diffusion step to execute
            info: Dictionary with diagnostic information
        """
        # Compute signals
        mask = (x == mask_token_id)
        
        lte = self.estimate_lte_euler_heun(model, x, mask_logits, state.current_step, state.total_steps)
        entropy = self.compute_entropy(mask_logits, mask)
        kl = self.compute_kl_divergence(mask_logits, self.prev_logits, mask)
        
        # Decide stride
        new_stride = self.decide_stride(state, lte, entropy, kl)
        
        # Update state
        old_stride = state.current_stride
        state.current_stride = new_stride
        
        # Compute next step
        next_step = min(state.current_step + new_stride, state.total_steps)
        steps_skipped = next_step - state.current_step - 1
        
        state.total_steps_taken += 1
        state.total_steps_skipped += steps_skipped
        
        # Diagnostic info
        info = {
            'lte': lte,
            'entropy': entropy,
            'kl': kl,
            'old_stride': old_stride,
            'new_stride': new_stride,
            'steps_skipped': steps_skipped,
            'total_skipped': state.total_steps_skipped,
            'stride_changes': state.stride_changes
        }
        
        return next_step, info
    
    def create_adaptive_schedule(
        self,
        total_steps: int,
        initial_stride: int = 1
    ) -> SchedulerState:
        """Create initial scheduler state."""
        return SchedulerState(
            current_step=0,
            current_stride=initial_stride,
            total_steps=total_steps,
            recent_lte=[],
            recent_entropy=[],
            recent_kl=[],
            total_steps_taken=0,
            total_steps_skipped=0,
            stride_changes=0
        )
    
    def get_summary_stats(self, state: SchedulerState) -> Dict:
        """Get summary statistics for logging."""
        return {
            'total_steps_taken': state.total_steps_taken,
            'total_steps_skipped': state.total_steps_skipped,
            'theoretical_steps': state.total_steps,
            'speedup_ratio': state.total_steps / max(1, state.total_steps_taken),
            'skip_percentage': 100.0 * state.total_steps_skipped / max(1, state.total_steps),
            'stride_changes': state.stride_changes,
            'avg_recent_lte': np.mean(state.recent_lte) if state.recent_lte else 0.0,
            'avg_recent_entropy': np.mean(state.recent_entropy) if state.recent_entropy else 0.0,
            'avg_recent_kl': np.mean(state.recent_kl) if state.recent_kl else 0.0,
        }


class HybridScheduler:
    """
    Combines adaptive scheduling (P4) with learned gating (P3) and caching (Phase A).
    This is the ultimate configuration that uses all acceleration techniques together.
    """
    
    def __init__(
        self,
        adaptive_scheduler: AdaptiveScheduler,
        learned_gate: Optional['LearnedGate'] = None,
        cache_schedule: Optional[List[int]] = None,
        device: str = 'cuda'
    ):
        self.adaptive = adaptive_scheduler
        self.gate = learned_gate
        self.cache_schedule = cache_schedule
        self.device = device
    
    def should_cache_layer(self, layer_idx: int, step_idx: int) -> bool:
        """Decide if a layer should use cached output."""
        if self.cache_schedule is None:
            return False
        return layer_idx in self.cache_schedule
    
    def should_freeze_layer(
        self,
        layer_idx: int,
        step_idx: int,
        features: 'GateFeatures'
    ) -> bool:
        """Decide if a layer should be frozen (using learned gate)."""
        if self.gate is None:
            return False
        
        return self.gate.should_freeze(features, threshold=0.5, device=self.device)
    
    def compute_hybrid_schedule(
        self,
        model,
        x: torch.Tensor,
        mask_logits: torch.Tensor,
        mask_token_id: int,
        state: SchedulerState,
        layer_features: Dict[int, 'GateFeatures'] = None
    ) -> Tuple[int, Dict, Dict[int, str]]:
        """
        Compute next step with hybrid acceleration:
            1. Adaptive scheduling decides stride
            2. Per-layer decisions (cache/freeze/recompute)
        
        Returns:
            next_step: Next diffusion step index
            info: Diagnostic information
            layer_decisions: {layer_idx: 'cache'|'freeze'|'compute'}
        """
        # Get adaptive step
        next_step, info = self.adaptive.get_next_step_index(
            model, x, mask_logits, mask_token_id, state
        )
        
        # Per-layer decisions (if not skipping to next step)
        layer_decisions = {}
        if layer_features is not None:
            for layer_idx, features in layer_features.items():
                if self.should_cache_layer(layer_idx, state.current_step):
                    layer_decisions[layer_idx] = 'cache'
                elif self.should_freeze_layer(layer_idx, state.current_step, features):
                    layer_decisions[layer_idx] = 'freeze'
                else:
                    layer_decisions[layer_idx] = 'compute'
        
        info['layer_decisions'] = layer_decisions
        
        return next_step, info, layer_decisions
