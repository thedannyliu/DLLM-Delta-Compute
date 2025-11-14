# Development Log - Dream Delta-Compute & L2C Caching

**Project:** DLLM-Delta-Compute  
**Branch:** PoC-1  
**Start Date:** November 14, 2025

---

## Overview

This document tracks the implementation of delta-compute acceleration and L2C-style caching for Dream-7B, following the master_plan.md (v1).

**Implementation Phases:**
- **POC v1a:** P0 (Baseline) + P1 (Teacher Traces)
- **POC v1b:** P2-v1b (Sequence-level Early Stopping) OR L2C Phase A (FFN Caching)

---

## Development Roadmap

### Phase 0: Setup ✅ / ⏳ / ❌

- [ ] Create conda environment `dcllm`
- [ ] Install dependencies (torch 2.5.1, transformers 4.46.2)
- [ ] Verify Dream model loads correctly
- [ ] Setup git tracking for changes

### Phase 1: P0 - Baseline Teacher

**Goal:** Establish reproducible baseline on GSM8K CoT

**Tasks:**
- [ ] Run `external/Dream/eval_instruct/eval.sh` on GSM8K
- [ ] Record: EM accuracy, latency (mean/p50/p90), diffusion_steps=256
- [ ] Verify `delta_mode=none, cache_mode=none` works

**Expected Output:**
- Baseline metrics JSON
- Verification that unmodified Dream eval works

### Phase 2: P1 - Teacher Traces

**Goal:** Collect per-step/per-layer statistics to inform gating and caching

**Implementation Plan:**

1. **Add tracing infrastructure:**
   - [ ] Modify `DreamDecoderLayer` to support optional tracing
   - [ ] Add hooks to compute:
     - FFN output L2 norm
     - Cosine similarity between consecutive steps
     - Per-layer timing
   - [ ] Create `TraceCollector` class to aggregate stats

2. **Integrate with eval:**
   - [ ] Add `trace_teacher=true` flag to model_args
   - [ ] Pass trace config through eval.py → diffllm.py → DreamModel
   - [ ] Save traces to `runs/teacher_trace_gsm8k_256steps/trace.json`

3. **Visualizations:**
   - [ ] Create plotting script: `scripts/plot_teacher_traces.py`
   - [ ] Generate: layer×step heatmaps, per-layer runtime, stability plots
   - [ ] Save to `reports/p1_traces/`

**Files to Modify:**
- `external/Dream/modeling/modeling_dream.py`
- `external/Dream/modeling/generation_utils.py`
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`

**Expected Output:**
- Trace JSON with shape: `{layer_id: {step_id: {mean_ffn_norm, cos_sim, timing}}}`
- Visualizations showing stability trends

### Phase 3: P2-v1b - Sequence-level Early Stopping

**Goal:** Implement simple early stopping to skip late diffusion steps

**Implementation Plan:**

1. **Add stability metrics computation:**
   - [ ] In `_sample` loop, compute entropy of masked positions
   - [ ] Compute margin (top1 - top2) confidence
   - [ ] Create `should_early_stop(logits, x, threshold)` function

2. **Modify sampling loop:**
   - [ ] Add early stopping condition in `generation_utils._sample`
   - [ ] Track actual steps executed vs. planned steps
   - [ ] Add `delta_mode="early_stop"` parsing

3. **Configuration:**
   - [ ] Add `delta_entropy_tau` to model_args
   - [ ] Add `delta_margin_tau` to model_args
   - [ ] Add `delta_min_steps` (safety: never stop before this)

**Files to Modify:**
- `external/Dream/modeling/generation_utils.py`
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`

**Expected Output:**
- JSON metrics: `{samples: [{planned_steps, actual_steps, speedup, em_match}]}`
- Target: 1.2-1.5× speedup with ≤1pp EM drop

### Phase 4: L2C Phase A - FFN Caching

**Goal:** Reuse FFN outputs between adjacent diffusion steps

**Implementation Plan:**

1. **Modify DreamDecoderLayer:**
   - [ ] Add `forward(..., use_ffn_cache=False, ffn_cache=None)`
   - [ ] Implement conditional FFN computation vs. reuse
   - [ ] Return FFN output for caching

2. **Modify _sample loop:**
   - [ ] Maintain `layer_cache` dict across steps
   - [ ] Implement alternating schedule: even=full, odd=cached
   - [ ] Add `cache_mode="l2c_heuristic"` parsing

3. **Configuration:**
   - [ ] Add `cache_schedule` to model_args
   - [ ] Add `cache_layers` (which layers to cache, e.g., "16-31")
   - [ ] Add `cache_start_step` (only cache after this step)

**Files to Modify:**
- `external/Dream/modeling/modeling_dream.py` (DreamDecoderLayer)
- `external/Dream/modeling/generation_utils.py` (_sample method)
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`

**Expected Output:**
- Measured FFN call reduction (target: 50% on cached steps)
- Speedup measurement (target: 1.2-1.4×)
- Quality impact (target: ≤1pp EM drop)

### Phase 5: Evaluation Integration

**Goal:** Make all modes accessible via model_args

**Implementation Plan:**

1. **Extend model_args parsing:**
   - [ ] Modify `diffllm.py` to parse new flags
   - [ ] Pass config dict to `DreamModel.diffusion_generate()`
   - [ ] Add validation for incompatible flag combinations

2. **Create convenience scripts:**
   - [ ] `scripts/eval_teacher.sh` - baseline
   - [ ] `scripts/eval_early_stop.sh` - P2-v1b
   - [ ] `scripts/eval_ffn_cache.sh` - Phase A

**Files to Create/Modify:**
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`
- `scripts/eval_*.sh`

### Phase 6: Testing & Validation

**Goal:** Verify correctness before full eval

**Tasks:**
- [ ] Create `tests/test_teacher_parity.py` - verify no change when modes disabled
- [ ] Create `tests/test_early_stop.py` - test on 5 samples
- [ ] Create `tests/test_ffn_cache.py` - test on 5 samples
- [ ] Verify shapes, dtypes, device consistency
- [ ] Manual inspection of 3-5 generated answers

### Phase 7: Full Evaluation & Analysis

**Goal:** Complete POC v1b on full GSM8K

**Tasks:**
- [ ] Run Teacher baseline (already done in P0)
- [ ] Run P2-v1b with multiple thresholds
- [ ] Run Phase A with multiple cache schedules
- [ ] Collect full metrics: EM, latency, block call counts
- [ ] Generate comparison tables and plots

**Expected Deliverables:**
- JSON results for each config
- Comparison table: Teacher vs. P2-v1b vs. Phase A
- Plots: speedup vs. quality tradeoff

---

## Issues & Blockers

### Active Issues

*(Issues will be added here as they arise)*

### Resolved Issues

*(Resolved issues will be moved here)*

---

## Git Commit Strategy

Each major milestone will be committed with descriptive messages:

```
# Example commit structure
git add <files>
git commit -m "[P1] Add teacher tracing infrastructure to modeling_dream.py"
git commit -m "[P2-v1b] Implement sequence-level early stopping in _sample"
git commit -m "[Phase A] Add FFN caching to DreamDecoderLayer"
git commit -m "[Eval] Integrate model_args parsing for delta_mode and cache_mode"
git commit -m "[Results] Complete POC v1b evaluation on GSM8K"
```

Branches:
- `PoC-1` (current) - main development branch
- Create feature branches if needed: `feature/p1-tracing`, `feature/p2-early-stop`

---

## Implementation Notes

### Code Style Guidelines
- Follow existing Dream codebase style
- Add docstrings for new functions
- Use type hints where applicable
- Keep modifications minimal and focused

### Testing Strategy
- Test on 5 samples before full eval
- Verify teacher parity first (modes disabled = identical output)
- Manual inspection of generated text
- Check GPU memory usage doesn't explode

### Performance Considerations
- Tracing should be low-overhead (disabled by default)
- Cache memory should be managed (clear between samples)
- Use in-place operations where possible

---

## Next Steps

**Immediate (Today):**
1. ✅ Create this development log
2. ⏳ Setup conda environment
3. ⏳ Verify Dream baseline runs
4. ⏳ Start P1 implementation

**Short-term (This Week):**
- Complete P1: Teacher Traces
- Choose between P2-v1b or Phase A for POC v1b
- Implement chosen approach
- Test on small subset

**Medium-term (Next Week):**
- Full GSM8K evaluation
- Generate results and plots
- Write up findings in docs/
- Decide on POC v2 direction

---

## Questions & Decisions Needed

*(Questions that need answering will be tracked here)*

1. **Environment:** Should we create dcllm env or use existing one?
2. **Eval subset:** How many GSM8K samples for P1 traces? (suggested: 50)
3. **POC v1b choice:** Start with P2-v1b (early stop) or Phase A (FFN cache)?
   - **Recommendation:** P2-v1b first (simpler, lower risk)

---

## Progress Tracking

**Week 1 (Nov 14-20):**
- [ ] Setup + P0
- [ ] P1 implementation
- [ ] P1 visualization

**Week 2 (Nov 21-27):**
- [ ] POC v1b implementation (P2-v1b or Phase A)
- [ ] Testing on subset
- [ ] Full evaluation

**Week 3 (Nov 28+):**
- [ ] Analysis and documentation
- [ ] POC v2 planning

---

## References

- Master Plan: `docs/master_plan.md`
- Feasibility Analysis: `docs/feasibility_re_evaluation.md`
- Dream README: `external/Dream/README.md`
- L2C Reference: `external/learning-to-cache/`

---

*Last Updated: November 14, 2025*
