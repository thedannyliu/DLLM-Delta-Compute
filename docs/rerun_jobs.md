# Complete Re-run with Timing Metrics

**Status: AUTOMATED PIPELINE ACTIVE**

## Batch 1: No Dependencies (RUNNING)

| Phase | Job ID | Status | Started | Notes |
|-------|--------|--------|---------|-------|
| P1 (Traces) | 3643107 | RUNNING | 12:26 | Will auto-trigger training |
| P0 (Baseline) | 3643149 | RUNNING | 12:41 | |
| P2 (Early Stop) | 3643150 | PENDING | - | Waiting for GPU |
| P4 (Adaptive) | 3643151 | PENDING | - | Waiting for GPU |
| Phase A (Heuristic) | 3643152 | PENDING | - | Waiting for GPU |

## Batch 2: Training (AUTO-TRIGGERED when P1 completes)

**Automation Active:** Background script (PID 4036511) monitoring P1

Will auto-submit after P1 completes and traces verified:
- Train P3 Gate (~/30 min)
- Train Phase B Router (~/30 min)
- Train Phase C Continuous Router (~/30 min)

Script: `scripts/wait_and_submit_training.sh`
Log: `logs/auto_training_submit.log`

## Batch 3: Evaluations (AUTO-TRIGGERED after training)

Will auto-submit after all training completes:
- P3 (Learned Gate) eval
- Phase B (Router) eval
- Phase C (Continuous) eval

Script: `scripts/wait_and_submit_batch3.sh`

## Timeline

- Submitted: $(date)
- Expected Batch 1 completion: ~30 min
- Expected training completion: +1.5 hours
- Expected Batch 3 completion: +30 min
- Total: ~2.5 hours
Mon Nov 24 12:41:34 EST 2025
