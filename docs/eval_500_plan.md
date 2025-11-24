# 500-Sample Evaluation Plan and Status

**Date:** November 24, 2025  
**Objective:** Comprehensive PoC evaluation of all phases with 500 samples each

## Evaluation Order (By Dependencies)

### Batch 1: No Dependencies (SUBMITTED)
- ✅ **Phase A** (Heuristic Caching) - Job 3641699
- ✅ **P0** (Baseline) - Job 3641700
- ✅ **P1** (Teacher Traces) - Job 3641701
- ✅ **P2** (Early Stop) - Job 3641702
- ✅ **P4** (Adaptive) - Job 3641703

**Status:** All submitted targeting H100, pending resources

### Batch 2: Requires Training (BLOCKED)
- ⏸️ **Phase B** (Learned Router) - Requires: `experiments/PhaseB_router/checkpoints/router_final.pt`
- ⏸️ **Phase C** (Continuous Router) - Requires: `experiments/PhaseC_continuous/checkpoints/continuous_router_final.pt`
- ⏸️ **P3** (Learned Gate) - Requires: `experiments/P3_learned_gate/checkpoints/learned_gate_final.pt`

**Status:** Blocked on training pipeline completion

---

## Training Pipeline Status

### Issue: Trace Generation Failure
**Job 3641620** (training smoke test) failed because P1 traces were not saved correctly.

**Root Cause:** The `trace_output_dir` parameter in lm_eval is relative to the execution context. When running from `external/Dream/eval_instruct`, the absolute path is still computed incorrectly, resulting in trace files not being written.

### Workaround Options
1. **Use existing traces:** We have 1 trace file from previous smoke test at:
   - `experiments/P1_traces/traces_smoke_20251123_161950/trace_sample_6527316552987314061.pt`
   - Not sufficient for proper training (need 100+ samples)

2. **Wait for P1 Job 3641701:** This job will generate 500 traces
   - Once complete, can use these traces for training
   - Location: `experiments/P1_traces/traces_500_<timestamp>/`

3. **Fix trace generation in training pipeline:** Debug the path issue in comprehensive_training_smoke_test.sh

### Recommended Approach
**Wait for P1 (Job 3641701) to complete**, then:
1. Use the 500 traces to train P3 gate
2. Use the 500 traces to train Phase B router
3. Use the 500 traces to train Phase C continuous router
4. Submit Batch 2 evaluations after training

---

## Metrics to Track

For each phase evaluation, we need to extract:

1. **Accuracy** 
   - Exact match on gsm8k
   - Reported by lm_eval

2. **Throughput**
   - Tokens per second (tok/s)
   - Need to add timing instrumentation

3. **Latency**
   - P95, P97, P99 percentiles
   - Per-sample generation time
   - Need to add timing instrumentation

### Current Gap: Timing Metrics
The lm_eval framework doesn't natively report detailed timing metrics. We need to:
- [ ] Add timing logging to `external/Dream/eval_instruct/lm_eval/models/diffllm.py`
- [ ] Log per-sample generation time
- [ ] Compute percentiles after evaluation
- [ ] Add tokens/second calculation

---

## Job Monitoring

### Active Jobs (Batch 1)
```bash
# Check status
squeue -u eliu354

# Check detailed job info
scontrol show job <JOBID>

# Monitor log files
tail -f logs/eval_PhaseA_500_<jobid>.out
tail -f logs/eval_P0_500_<jobid>.out
tail -f logs/eval_P1_500_<jobid>.out
tail -f logs/eval_P2_500_<jobid>.out
tail -f logs/eval_P4_500_<jobid>.out
```

### GPU Availability Strategy
1. **Primary:** H100 (requested for all jobs)
2. **Backup:** H200 (if H100 queue too long)
3. **Fallback:** L40S, A100, RTX6000 (if high-end GPUs unavailable)

---

## Next Steps

### Immediate (Current Session)
1. ⏳ Monitor Batch 1 jobs (5 jobs pending H100)
2. ⏳ Wait for at least P0, P1, Phase A to complete
3. 📝 Add timing instrumentation for metrics
4. ⏳ Once P1 completes: Launch training for P3, Phase B, Phase C
5. ⏳ Once training completes: Submit Batch 2 evaluations

### Medium-term
1. Analyze results from all 8 phases
2. Compare accuracy, throughput, latency across methods
3. Generate comparison tables and visualizations
4. Document findings in comprehensive report

### Long-term
1. Address training pipeline trace generation bug
2. Scale to larger evaluation sets (1000+ samples)
3. Multi-GPU evaluation for throughput testing
4. Production deployment considerations

---

## Git Commits

- **4beeaec** - Initial phase readiness documentation
- **211ec8b** - 500-sample evaluation scripts for all phases (current)

Next commits will include:
- Timing instrumentation additions
- Training scripts execution results
- Batch 2 evaluation submissions
- Results analysis and comparison

---

## Resources

**Compute:**
- Georgia Tech PACE ICE Cluster
- Account: coc
- QOS: coc-ice
- Conda env: dcllm

**Data:**
- Dataset: gsm8k (grade school math)
- Samples: 500 per phase
- Few-shot: 5-shot prompting
- Batch size: 1 (for fair timing comparison)

**Model:**
- Dream-org/Dream-v0-Instruct-7B
- Masked diffusion LLM
- 7B parameters
