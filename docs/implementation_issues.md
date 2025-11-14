# Implementation Issues & Questions

## Date: November 14, 2025

---

## Issue #1: DreamModel.forward signature extension

**Status:** ⏳ In Progress

**Description:**
Need to extend `DreamModel.forward` to pass through the new tracing and caching parameters to `DreamBaseModel.forward`.

**Current State:**
- ✅ Modified `DreamDecoderLayer` to support tracing and FFN caching
- ✅ Modified `DreamBaseModel.forward` to handle new parameters
- ⏳ Need to modify `DreamModel.forward` wrapper

**Files Affected:**
- `external/Dream/modeling/modeling_dream.py` (DreamModel class)

**Solution:**
Add optional parameters to DreamModel.forward and pass them through to self.model() call.

---

## Issue #2: Integration with generation_utils.py

**Status:** ⏳ Pending

**Description:**
Need to modify `generation_utils._sample` to:
1. Create and manage `TraceCollector` instance
2. Pass diffusion_step to model forward
3. Implement caching schedule logic
4. Implement early stopping logic

**Files Affected:**
- `external/Dream/modeling/generation_utils.py` (_sample method)

**Dependencies:**
- Requires Issue #1 to be completed first

---

## Issue #3: Evaluation wrapper integration

**Status:** ⏳ Pending

**Description:**
Need to extend eval wrappers to parse new model_args flags:
- `trace_teacher=true/false`
- `delta_mode=none|early_stop|...`
- `cache_mode=none|l2c_heuristic|...`
- Associated threshold parameters

**Files Affected:**
- `external/Dream/eval_instruct/lm_eval/models/diffllm.py`
- `external/Dream/eval/eval.py`

---

## Issue #4: Testing strategy

**Status:** ⏳ Pending

**Description:**
Need to create test scripts to verify:
1. Teacher parity (modes disabled = identical output)
2. Tracing works without errors
3. Caching doesn't break shapes/dtypes
4. Early stopping actually saves compute

**Files to Create:**
- `tests/test_teacher_parity.py`
- `tests/test_tracing.py`
- `tests/test_caching.py`

---

## Question #1: Should we support gradient checkpointing with new features?

**Context:**
Current implementation adds parameters to layer forward. Gradient checkpointing path currently ignored for simplicity.

**Options:**
A. Skip gradient checkpointing for POC (training not our focus)
B. Add support for gradient checkpointing path as well

**Decision:** TBD
**Recommendation:** Option A for POC v1

---

## Question #2: Cache memory management

**Context:**
FFN caches are stored as tensors. For long sequences/many layers, this could use significant memory.

**Considerations:**
- Clear caches between samples?
- Limit cache to certain layers only?
- Use CPU offloading for caches?

**Decision:** TBD
**Recommendation:** Clear caches between samples, cache all layers by default for POC

---

## Question #3: Tracing performance impact

**Context:**
Computing norms and cosine similarities adds overhead.

**Measurement needed:**
- Benchmark with/without tracing on 10 samples
- Measure overhead percentage

**Decision:** TBD
**Recommendation:** Only enable tracing when explicitly requested via model_args

---

*File will be updated as new issues arise during implementation*
