"""
Patch for Dream generation_utils.py to add proper early stopping tracking and trace collection.
Apply this patch to external/Dream/modeling/generation_utils.py
"""

# Add after line 523 (in the early stopping check):

# ORIGINAL CODE:
"""
            # P2 early stopping check (simple sequence-level)
            if generation_config.delta_mode == "p2_early_stop" and i < steps - 1:
                # Check if all tokens meet confidence threshold
                probs = torch.softmax(mask_logits, dim=-1)
                max_probs, _ = probs.max(dim=-1)
                
                # Also compute entropy-based stopping
                epsilon = 1e-10
                log_probs = torch.log(probs + epsilon)
                entropy = -torch.sum(probs * log_probs, dim=-1)
                
                if (max_probs > generation_config.early_stop_confidence_threshold).all() or \
                   (entropy < generation_config.early_stop_entropy_threshold).all():
                    logger.info(f"Early stopping at step {i}/{steps} (confidence: {max_probs.mean():.3f}, entropy: {entropy.mean():.3f})")
                    break
"""

# PATCHED CODE:
"""
            # P2 early stopping check (sequence-level with detailed logging)
            if generation_config.delta_mode == "p2_early_stop" and i < steps - 1:
                # Check ONLY on masked tokens (not already decided tokens)
                mask = (x == mask_token_id)
                
                if mask.any():
                    # Get probabilities for masked positions only
                    probs = torch.softmax(mask_logits, dim=-1)
                    max_probs, _ = probs.max(dim=-1)
                    
                    # Compute entropy for masked positions
                    epsilon = 1e-10
                    log_probs = torch.log(probs + epsilon)
                    entropy = -torch.sum(probs * log_probs, dim=-1)
                    
                    # Get statistics for masked tokens only
                    masked_max_probs = max_probs[mask]
                    masked_entropy = entropy[mask]
                    
                    # Check stopping criteria on masked tokens
                    confidence_check = (masked_max_probs > generation_config.early_stop_confidence_threshold).all()
                    entropy_check = (masked_entropy < generation_config.early_stop_entropy_threshold).all()
                    
                    # Log every 10 steps or on stop
                    if i % 10 == 0 or confidence_check or entropy_check:
                        logger.info(
                            f"Step {i}/{steps}: masked_tokens={mask.sum().item()}, "
                            f"avg_confidence={masked_max_probs.mean():.3f}, "
                            f"avg_entropy={masked_entropy.mean():.3f}, "
                            f"conf_thresh={generation_config.early_stop_confidence_threshold}, "
                            f"entr_thresh={generation_config.early_stop_entropy_threshold}"
                        )
                    
                    if confidence_check or entropy_check:
                        logger.info(
                            f"⚡ EARLY STOP at step {i+1}/{steps}! "
                            f"Saved {steps - (i+1)} steps ({100*(steps-(i+1))/steps:.1f}% reduction). "
                            f"Reason: {'confidence' if confidence_check else 'entropy'}"
                        )
                        break
"""

print("""
To apply this patch:
1. Open external/Dream/modeling/generation_utils.py
2. Find the P2 early stopping section (around line 523)
3. Replace with the patched code above
4. This will add:
   - Proper masking (only check undecided tokens)
   - Detailed logging every 10 steps
   - Clear early stopping messages
   - Statistics on step reduction
""")
