#!/usr/bin/env python3
"""
Enhanced evaluation script with detailed logging for debugging.
Adds step-level tracking and statistics.
"""

import sys
import os
import json
import torch
import logging
from pathlib import Path

# Setup detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EarlyStopTracker:
    """Track early stopping statistics during generation."""
    
    def __init__(self):
        self.samples = []
        self.total_steps_saved = 0
        self.total_steps_possible = 0
        
    def record_sample(self, sample_id, actual_steps, max_steps, stopped_early, 
                     final_confidence=None, final_entropy=None):
        """Record statistics for one sample."""
        self.samples.append({
            'sample_id': sample_id,
            'actual_steps': actual_steps,
            'max_steps': max_steps,
            'stopped_early': stopped_early,
            'steps_saved': max_steps - actual_steps if stopped_early else 0,
            'final_confidence': final_confidence,
            'final_entropy': final_entropy
        })
        self.total_steps_saved += (max_steps - actual_steps) if stopped_early else 0
        self.total_steps_possible += max_steps
        
    def get_summary(self):
        """Get summary statistics."""
        if not self.samples:
            return {}
            
        early_stopped = sum(1 for s in self.samples if s['stopped_early'])
        total_samples = len(self.samples)
        
        return {
            'total_samples': total_samples,
            'early_stopped_count': early_stopped,
            'early_stop_rate': early_stopped / total_samples if total_samples > 0 else 0,
            'total_steps_saved': self.total_steps_saved,
            'total_steps_possible': self.total_steps_possible,
            'step_reduction': self.total_steps_saved / self.total_steps_possible if self.total_steps_possible > 0 else 0,
            'avg_actual_steps': sum(s['actual_steps'] for s in self.samples) / total_samples if total_samples > 0 else 0,
            'avg_max_steps': sum(s['max_steps'] for s in self.samples) / total_samples if total_samples > 0 else 0,
        }
    
    def save_detailed_log(self, output_path):
        """Save detailed per-sample logs."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'samples': self.samples
            }, f, indent=2)
        
        logger.info(f"Saved early stopping details to {output_path}")

def main():
    """Wrapper to add tracking to Dream evaluation."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_args', type=str, required=True)
    parser.add_argument('--tasks', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--tracking_output', type=str, required=True)
    args = parser.parse_args()
    
    # Initialize tracker
    tracker = EarlyStopTracker()
    
    # TODO: Hook into Dream's generation loop
    # This requires modifying the Dream codebase or using monkey patching
    
    logger.info("Enhanced evaluation with early stopping tracking")
    logger.info(f"Model args: {args.model_args}")
    logger.info(f"Tasks: {args.tasks}")
    
    # Save results
    summary = tracker.get_summary()
    logger.info(f"Early stopping summary: {summary}")
    tracker.save_detailed_log(args.tracking_output)

if __name__ == '__main__':
    main()
