# Implementation Roadmap for Remaining PoC Phases

**Last Updated:** November 14, 2025  
**Status:** Infrastructure complete for P3, P4, Phase B, Phase C  
**Next Steps:** Integration with Dream model and training/evaluation

---

## Overview

This document outlines the implementation roadmap for completing the remaining phases (P3, P4, Phase B, Phase C) of the DLLM Delta-Compute project as specified in `master_plan.md`.

### Current Status Summary

**✅ Completed (Ready for evaluation):**
- P0: Baseline teacher
- P1: Teacher traces infrastructure  
- P2: Early stopping (sequence-level)
- Phase A: Heuristic FFN caching

**🔧 Infrastructure Complete (Needs integration):**
- P3: Learned gate architecture
- P4: Adaptive step scheduling
- Phase B: Learned router (fixed schedule)
- Phase C: Continuous-time router

**⚠️ Critical Dependencies:**
All P3-P4 and Phase B-C implementations require integration points with Dream's `generation_utils.py` and `modeling_dream.py` that are currently placeholders.

---

## Phase-by-Phase Implementation Plan

### P3 - Learned Gate

**Goal:** Replace rule-based thresholds (P2) with a learned MLP gate.

**Infrastructure Created:**
- ✅ `src/learned_gate.py`: LearnedGate model (~1.4k params), GateTrainer, GateFeatures
- ✅ `scripts/training/train_learned_gate.py`: Training script
- ✅ `scripts/training/generate_oracle_labels.py`: Oracle label generation (placeholder)
- ✅ `scripts/slurm/eval_P3_learned_gate.sh`: Evaluation script

**Integration Points Needed:**

1. **In `external/Dream/modeling/generation_utils.py`:**
   ```python
   # After line ~500 (where P2 early stopping is)
   if generation_config.delta_mode == "p3_learned_gate":
       from ....src.learned_gate import LearnedGate, GateFeatures
       
       # Load gate
       gate = LearnedGate.load(generation_config.gate_checkpoint)
       gate = gate.to(self.device)
       gate.eval()
       
       # For each step, compute features and decide
       features = GateFeatures(
           layer_idx=layer_idx,
           step_idx=i,
           total_steps=steps,
           total_layers=len(self.model.layers),
           ffn_output_norm=compute_from_trace_collector(),
           ffn_cosine_sim=compute_from_trace_collector(),
           max_confidence=masked_max_probs.mean().item(),
           entropy=masked_entropy.mean().item(),
           num_masked_tokens=mask.sum().item(),
           total_tokens=x.numel()
       )
       
       if gate.should_freeze(features, threshold=generation_config.gate_threshold):
           # Skip remaining steps or freeze specific layers
           break
   ```

2. **Extend `DreamGenerationConfig`:**
   ```python
   gate_checkpoint: Optional[str] = None
   gate_threshold: float = 0.5
   ```

3. **Oracle label generation:**
   - Replace placeholder in `generate_oracle_labels.py` with actual ablation experiments
   - For each (layer, step) pair: run with that pair frozen, compare to teacher
   - Label as 1 (safe) if output quality maintained, 0 (unsafe) otherwise

**Training Workflow:**
```bash
# Step 1: Generate oracle labels (requires P0 baseline + ablation runs)
python scripts/training/generate_oracle_labels.py \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --task gsm8k_cot \
    --num_samples 100 \
    --output experiments/P3_learned_gate/oracle_labels.json

# Step 2: Train gate on P1 traces + oracle labels
python scripts/training/train_learned_gate.py \
    --trace_dir experiments/P1_traces/traces \
    --oracle_labels experiments/P3_learned_gate/oracle_labels.json \
    --output_dir experiments/P3_learned_gate/checkpoints \
    --num_epochs 50

# Step 3: Evaluate
sbatch scripts/slurm/eval_P3_learned_gate.sh
```

**Success Criteria:**
- Gate training converges (val F1 > 0.7)
- P3 evaluation matches or exceeds P2 performance on GSM8K
- Latency reduction ≥ 1.3×, accuracy drop ≤ 2pp

---

### P4 - Adaptive Step Scheduling

**Goal:** Dynamically adjust diffusion step stride based on LTE and risk signals.

**Infrastructure Created:**
- ✅ `src/adaptive_scheduler.py`: AdaptiveScheduler, SchedulerState, HybridScheduler
- ✅ `scripts/slurm/eval_P4_adaptive.sh`: Evaluation script

**Integration Points Needed:**

1. **In `external/Dream/modeling/generation_utils.py` main loop:**
   ```python
   if generation_config.delta_mode == "p4_adaptive":
       from ....src.adaptive_scheduler import AdaptiveScheduler, SchedulerState
       
       # Initialize scheduler
       scheduler = AdaptiveScheduler(
           max_stride=generation_config.adaptive_max_stride,
           lte_threshold=generation_config.adaptive_lte_threshold,
           entropy_threshold=generation_config.adaptive_entropy_threshold,
           kl_threshold=generation_config.adaptive_kl_threshold,
           stable_window=generation_config.adaptive_stable_window,
           min_safe_step=generation_config.adaptive_min_safe_step,
           device=self.device
       )
       
       state = scheduler.create_adaptive_schedule(steps)
       
       # In diffusion loop
       for i in range(steps):
           # Normal step execution
           ...
           
           # Adaptive decision
           next_step, info = scheduler.get_next_step_index(
               model=self.model,
               x=x,
               mask_logits=mask_logits,
               mask_token_id=mask_token_id,
               state=state
           )
           
           # Log info
           if i % 10 == 0:
               logger.info(f"Step {i}: stride={info['new_stride']}, "
                          f"LTE={info['lte']:.4f}, entropy={info['entropy']:.2f}")
           
           # Jump to next_step (skip intermediate steps)
           if next_step > i + 1:
               logger.info(f"Skipping steps {i+1} to {next_step-1}")
               i = next_step - 1
   ```

2. **Extend `DreamGenerationConfig`:**
   ```python
   adaptive_max_stride: int = 4
   adaptive_lte_threshold: float = 0.01
   adaptive_entropy_threshold: float = 2.0
   adaptive_kl_threshold: float = 0.1
   adaptive_stable_window: int = 5
   adaptive_min_safe_step: int = 10
   ```

**Evaluation Workflow:**
```bash
# Evaluate with adaptive scheduling
sbatch scripts/slurm/eval_P4_adaptive.sh

# Compare to P0 baseline and P2 early stopping
python scripts/eval/compare_phases.py \
    --baseline experiments/P0_baseline/results \
    --p2 experiments/P2_early_stop/results \
    --p4 experiments/P4_adaptive/results
```

**Success Criteria:**
- Adaptive stride increases in stable regions (later steps)
- Further latency reduction beyond P2 (target ≥1.5×)
- Accuracy maintained within 3-5pp of teacher

---

### Phase B - Learned Router (Fixed Schedule)

**Goal:** Learn per-layer recompute/cache decisions for a fixed diffusion schedule.

**Infrastructure Created:**
- ✅ `src/learned_router.py`: FixedScheduleRouter, RouterTrainer
- ✅ `scripts/slurm/eval_PhaseB_router.sh`: Evaluation script

**Integration Points Needed:**

1. **Teacher trace collection** (extend P1):
   ```python
   # In generation_utils._sample(), collect intermediate layer outputs
   if generation_config.cache_mode == "collect_for_router_training":
       layer_outputs = {}  # {step_idx: {layer_idx: tensor}}
       
       # During forward pass, save each layer's output
       for layer_idx, layer in enumerate(self.model.layers):
           output = layer(...)
           layer_outputs.setdefault(i, {})[layer_idx] = output.detach()
       
       # Save after generation
       torch.save(layer_outputs, f"teacher_outputs_{sample_id}.pt")
   ```

2. **Student forward function** (for training):
   ```python
   def create_student_forward_fn(model, x, step_idx, cache):
       def forward_fn(decisions):
           # decisions: {layer_idx: recompute_prob}
           outputs = {}
           for layer_idx, layer in enumerate(model.layers):
               if decisions[layer_idx] < 0.5 and layer_idx in cache:
                   # Use cached output
                   outputs[layer_idx] = cache[layer_idx]
               else:
                   # Recompute
                   outputs[layer_idx] = layer(x)
           return outputs
       return forward_fn
   ```

3. **Router integration in generation:**
   ```python
   if generation_config.cache_mode == "l2c_learned":
       from ....src.learned_router import FixedScheduleRouter
       
       router = FixedScheduleRouter.load(generation_config.router_checkpoint)
       router = router.to(self.device)
       router.eval()
       
       # For each step, get layer decisions
       decisions = router.get_all_decisions(i, threshold=generation_config.router_threshold)
       
       # Apply decisions (integrate with Phase A caching logic)
       for layer_idx, should_recompute in decisions.items():
           if not should_recompute and layer_idx in layer_ffn_caches:
               # Use cache
               use_cached = True
   ```

**Training Workflow:**
```bash
# Step 1: Collect teacher traces with layer outputs
# (Extend P1 to save per-layer intermediate activations)

# Step 2: Train router (requires custom training script)
python scripts/training/train_learned_router.py \
    --teacher_traces experiments/P1_traces/teacher_outputs \
    --num_layers 32 \
    --num_steps 256 \
    --output_dir experiments/PhaseB_router/checkpoints

# Step 3: Evaluate
sbatch scripts/slurm/eval_PhaseB_router.sh
```

**Success Criteria:**
- Router learns meaningful patterns (visualize β matrix)
- Phase B outperforms Phase A heuristic caching
- Cache ratio 40-60%, accuracy drop ≤ 2pp

---

### Phase C - Continuous-Time Router

**Goal:** Generalize router to continuous time, enable schedule transfer.

**Infrastructure Created:**
- ✅ `src/continuous_router.py`: ContinuousRouter, TimeEncoder, ContinuousRouterTrainer
- ✅ `scripts/slurm/eval_PhaseC_continuous.sh`: Evaluation script

**Integration Points Needed:**

Similar to Phase B, but:

1. **Time normalization:**
   ```python
   # In generation loop
   t_normalized = i / max(1, steps - 1)  # Normalize to [0, 1]
   
   # Get decisions from continuous router
   decisions = router.get_schedule_decisions(steps)  # Precompute for entire schedule
   ```

2. **Multi-schedule training:**
   ```python
   # Collect training data from multiple schedules
   train_data = []
   for schedule_steps in [128, 256, 512]:
       # Collect teacher outputs for this schedule
       outputs = run_teacher_with_schedule(schedule_steps)
       train_data.extend(outputs)
   
   # Train router
   trainer.train_on_multiple_schedules(train_data, val_data)
   ```

**Evaluation Workflow:**
```bash
# Train on 256-step schedule
python scripts/training/train_continuous_router.py \
    --train_schedules 256 \
    --output_dir experiments/PhaseC_continuous/checkpoints

# Test transfer: evaluate on 128, 256, 512 steps
sbatch --export=DIFFUSION_STEPS=128 scripts/slurm/eval_PhaseC_continuous.sh
sbatch --export=DIFFUSION_STEPS=256 scripts/slurm/eval_PhaseC_continuous.sh
sbatch --export=DIFFUSION_STEPS=512 scripts/slurm/eval_PhaseC_continuous.sh

# Compare transfer performance
python scripts/eval/analyze_schedule_transfer.py \
    --router_checkpoint experiments/PhaseC_continuous/checkpoints/continuous_router_best.pt \
    --results_dir experiments/PhaseC_continuous/results
```

**Success Criteria:**
- Router transfers across schedules with <5% performance degradation
- Phase C matches Phase B on 256 steps
- Visualization shows smooth β(t) curves

---

## Integration Summary

### Critical Files to Modify

1. **`external/Dream/modeling/generation_utils.py`:**
   - Add P3 gate integration (around line 550)
   - Add P4 adaptive scheduler (replace main loop structure)
   - Add Phase B/C router integration (in caching logic)

2. **`external/Dream/modeling/modeling_dream.py`:**
   - Ensure per-layer forward hooks work with all modes
   - Cache management for routers

3. **`external/Dream/eval_instruct/lm_eval/models/diffllm.py`:**
   - Parse new model_args: `gate_checkpoint`, `router_checkpoint`, `adaptive_*`

### Key Dependencies

```
P0 (baseline) 
  → P1 (traces) 
    → P2 (early stop) 
      → P3 (learned gate) [needs P1 traces + oracle labels]
      → P4 (adaptive) [needs P2 baseline]
    → Phase A (caching)
      → Phase B (learned router) [needs P1 layer outputs]
        → Phase C (continuous router) [needs Phase B baseline]
```

---

## Testing Strategy

### Unit Tests
- Test gate forward pass, training step
- Test scheduler stride decisions
- Test router parameter updates
- Test time encoding

### Integration Tests  
- Test gate integration in generation loop
- Test scheduler with mock diffusion steps
- Test router decisions affect cache usage

### End-to-End Validation
- Small GSM8K subset (10 samples) for each phase
- Compare outputs to teacher (should be close)
- Verify latency metrics make sense

---

## Questions for Clarification

### 1. Oracle Label Generation (P3)
**Question:** How should we generate oracle labels for the learned gate?

**Options:**
- A) **Ablation experiments**: For each (layer, step), freeze it and run full evaluation, compare accuracy
  - Pro: Accurate labels based on actual impact
  - Con: Very expensive (32 layers × 256 steps × 100 samples = 819k forward passes)
  
- B) **Heuristic labels**: Use P1 traces to label stable (high cosine sim, low drift) as safe to freeze
  - Pro: Fast, no additional evaluation needed
  - Con: May not reflect actual impact on final accuracy
  
- C) **Sampled ablation**: Test only a subset of (layer, step) pairs strategically
  - Pro: Balanced cost/accuracy
  - Con: Need to decide sampling strategy

**Recommendation:** Start with B (heuristic) for initial training, validate with C (sampled ablation) on key regions.

---

### 2. Teacher Trace Collection (Phase B)
**Question:** Should we collect full intermediate layer activations for router training?

**Concerns:**
- Storage: 32 layers × 256 steps × [B, L, H] = very large
- Memory: May OOM during collection
- I/O: Slow to save/load for training

**Options:**
- A) Collect everything, subsample during training
- B) Collect only summary statistics (norms, similarities)
- C) On-the-fly: Generate during router training (slow but memory-efficient)

**Recommendation:** B for PoC (reuse P1 infrastructure), upgrade to A if router performance insufficient.

---

### 3. Adaptive Scheduler Integration (P4)
**Question:** Should adaptive scheduler replace the entire diffusion loop or just control stride?

**Options:**
- A) **Stride control only**: Keep existing loop, skip steps based on scheduler
  - Pro: Minimal code changes
  - Con: May not capture all LTE estimation benefits
  
- B) **Full loop replacement**: Implement Euler/Heun with LTE estimation
  - Pro: More accurate LTE, potentially better control
  - Con: Major refactoring, high risk

**Recommendation:** Start with A (stride control) for PoC, consider B if results justify complexity.

---

### 4. Multi-Schedule Training (Phase C)
**Question:** Which schedules should we train on, and how to balance them?

**Options:**
- A) Train on 256 only, test transfer to 128 and 512
- B) Train on mix of [128, 256, 512] with equal weighting
- C) Train on mix with schedule-adaptive weighting (e.g., favor 256)

**Recommendation:** A for initial PoC (matches current evaluations), upgrade to B if transfer is poor.

---

### 5. Evaluation Priorities
**Question:** Given limited GPU time, which phases should be prioritized?

**Proposed Order:**
1. **P2 verification** (currently running) - baseline for all gating
2. **Phase A validation** (currently running) - baseline for all caching  
3. **P1 analysis** - needed for P3 oracle labels and Phase B router training
4. **P3 training & eval** - learned acceleration (depends on P1)
5. **Phase B training & eval** - learned caching (depends on P1)
6. **P4 eval** - advanced scheduling (depends on P2)
7. **Phase C training & eval** - schedule transfer (depends on Phase B)

**Alternative (if P2/Phase A fail):**
- Focus on making P2 and Phase A work before advancing to learned versions
- Use failed attempts as negative examples for what not to learn

**Recommendation:** Wait for current runs (P0, P1, P2, Phase A) to complete, analyze results, then decide next priority.

---

### 6. Checkpoint Management
**Question:** How should we organize trained checkpoints across phases?

**Proposed Structure:**
```
experiments/
  P3_learned_gate/
    checkpoints/
      learned_gate_best.pt       # Best validation
      learned_gate_final.pt      # End of training
      learned_gate_epoch_*.pt    # Intermediate (optional)
    oracle_labels.json
    training_log.txt
  
  PhaseB_router/
    checkpoints/
      learned_router_best.pt
      learned_router_256steps.pt  # Specific schedule
    teacher_traces/               # Intermediate activations
  
  PhaseC_continuous/
    checkpoints/
      continuous_router_best.pt
      continuous_router_trained_on_256.pt
```

**Recommendation:** Use `_best.pt` for main checkpoints, `_final.pt` for last epoch, tag with training details in metadata.

---

## Next Actions (Immediate)

1. **Wait for current jobs to complete** (P0: 3552400, P1: 3552442, P2: 3552443, Phase A: 3552444)

2. **Analyze P1 traces:**
   ```bash
   python tests/visualization/plot_traces_v2.py experiments/P1_traces/traces/*.pt
   ```
   - Identify stable vs unstable layers/steps
   - Guide Phase A schedule selection
   - Inform P3 oracle label generation

3. **Verify P2 early stopping:**
   ```bash
   grep '⚡ EARLY STOP' experiments/P2_early_stop/logs/*.out
   ```
   - Confirm it actually triggers
   - Measure actual speedup vs P0
   - Decide if P3 is needed or if P2 is sufficient

4. **Validate Phase A caching:**
   ```bash
   python scripts/eval/compare_phases.py --p0 --phase_a
   ```
   - Check accuracy degradation
   - Measure speedup
   - Decide if Phase B worth pursuing

5. **Generate P3 oracle labels** (if P2 shows promise):
   ```bash
   # Start with heuristic labels
   python scripts/training/generate_oracle_labels.py \
       --trace_dir experiments/P1_traces/traces \
       --output experiments/P3_learned_gate/oracle_labels_heuristic.json \
       --strategy heuristic
   ```

6. **Begin P3 training** (if labels ready):
   ```bash
   python scripts/training/train_learned_gate.py \
       --trace_dir experiments/P1_traces/traces \
       --oracle_labels experiments/P3_learned_gate/oracle_labels_heuristic.json \
       --num_epochs 50
   ```

---

## Estimated Timeline

**Assuming current jobs complete successfully:**

- **Week 1 (Current):**
  - ✅ Infrastructure for P3, P4, Phase B, Phase C
  - ⏳ P0, P1, P2, Phase A evaluation running
  
- **Week 2:**
  - Analyze P1 traces, verify P2/Phase A
  - Generate P3 oracle labels (heuristic)
  - Train P3 gate (1-2 days)
  - Evaluate P3 (6 hours)
  
- **Week 3:**
  - Integrate P4 adaptive scheduler
  - Evaluate P4 (6 hours)
  - Extend P1 to collect layer outputs for Phase B
  - Train Phase B router (2-3 days)
  
- **Week 4:**
  - Evaluate Phase B (6 hours)
  - Train Phase C continuous router (2-3 days)
  - Evaluate Phase C with schedule transfer (1 day)
  
- **Week 5:**
  - Comprehensive evaluation across all phases
  - Generate final reports and visualizations
  - Documentation and analysis

**Total: ~5 weeks to complete all phases, assuming no major blockers.**

---

## Risk Mitigation

### High-Risk Items

1. **P2/Phase A might not work:**
   - Mitigation: Detailed analysis of why they fail guides learned methods
   - Fallback: Focus on analysis rather than acceleration for PoC

2. **Oracle label generation too expensive:**
   - Mitigation: Use heuristic labels, validate on subset
   - Fallback: Train gate in unsupervised manner (predict trace features)

3. **Router training requires too much memory:**
   - Mitigation: Collect summary statistics instead of full activations
   - Fallback: Simplify router architecture, reduce training data

4. **Schedule transfer doesn't work (Phase C):**
   - Mitigation: Train on multiple schedules simultaneously
   - Fallback: Phase B per-schedule routers are still useful

### Validation Checkpoints

After each phase:
- ✅ Code compiles and runs without errors
- ✅ Outputs match expected shape/format
- ✅ Quality metrics within acceptable range vs teacher
- ✅ Latency/compute metrics show improvement
- ✅ Integration doesn't break existing functionality

---

## Conclusion

All infrastructure for P3, P4, Phase B, and Phase C is now in place. The main remaining work is:

1. **Integration:** Connect new modules to Dream's generation loop
2. **Training:** Generate labels, collect traces, train gates/routers
3. **Evaluation:** Run comprehensive experiments on GSM8K
4. **Analysis:** Compare across all phases, generate final report

The questions above require clarification to proceed efficiently. Please prioritize answering questions 1, 2, and 5 as they block immediate next steps.
