# Complete Re-run with Timing Metrics

## Batch 1: No Dependencies (Submitted)

| Phase | Job ID | Status | Notes |
|-------|--------|--------|-------|
| P1 (Traces) | 3643107 | RUNNING | Started earlier, ~14 min elapsed |
| P0 (Baseline) | 3643149 | SUBMITTED | Just submitted |
| P2 (Early Stop) | 3643150 | SUBMITTED | Just submitted |
| P4 (Adaptive) | 3643151 | SUBMITTED | Just submitted |
| Phase A (Heuristic) | 3643152 | SUBMITTED | Just submitted |

## Batch 2: Training (Pending P1 completion)

Will submit after P1 completes and traces verified:
- Train P3 Gate
- Train Phase B Router  
- Train Phase C Continuous Router

## Batch 3: Evaluations (Pending training completion)

Will submit after training completes:
- P3 (Learned Gate)
- Phase B (Router)
- Phase C (Continuous)

## Timeline

- Submitted: $(date)
- Expected Batch 1 completion: ~30 min
- Expected training completion: +1.5 hours
- Expected Batch 3 completion: +30 min
- Total: ~2.5 hours
Mon Nov 24 12:41:34 EST 2025
