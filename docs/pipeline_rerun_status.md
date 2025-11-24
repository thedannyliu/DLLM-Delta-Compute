# Pipeline Re-run Status Report

**Date:** November 24, 2025, 17:05 EST  
**Status:** RUNNING - Fully Automated

---

## 清理完成 ✓

### 空間釋放
- **Before:** 300GB/300GB (100% - FULL!)
- **After:** 127.3GB/300GB (42.4%)
- **Freed:** 172.7GB

### 清理內容
1. 刪除舊的 `traces/` 目錄：14GB
2. 刪除大型 activation 檔案 (>100MB)：~159GB
3. 保留小型統計檔案供訓練使用

---

## 程式碼修改 ✓

### generation_utils.py (line 413-430)
**問題：** 5% 的樣本會存完整 activations，每個檔案 5-7GB！

**修復：**
```python
# OLD: save_activations = (random.random() < 0.05) if ...
# NEW: save_activations = False  # Always disable
```

**影響：** 未來不會再產生巨大檔案

---

## Pipeline 架構

```
Batch 1: Baseline Evaluations (500 samples each)
├── P0 (Baseline)         → Job 3644324 [PENDING]
├── P1 (Traces)           → Job 3644323 [PENDING] ← 關鍵！
├── P2 (Early Stop)       → Job 3644325 [PENDING]
├── P4 (Adaptive)         → Job 3644326 [PENDING]
└── Phase A (L2C FFN)     → Job 3644327 [PENDING]
    ↓ (P1 完成後自動觸發)
    
Batch 2: Training (~30 min each)
├── P3 Gate              → 等待 P1
├── Phase B Router       → 等待 P1
└── Phase C Continuous   → 等待 P1
    ↓ (訓練完成後自動觸發)
    
Batch 3: Learned Model Evaluations (500 samples each)
├── P3 (Learned Gate)    → 等待訓練
├── Phase B (Router)     → 等待訓練
└── Phase C (Continuous) → 等待訓練
```

---

## 自動化狀態

### ✓ Active Processes
- **PID 1604214:** `auto_submit_training.sh` 
  - Monitoring: P1 job 3644323
  - Will auto-submit training when P1 completes with ≥450 traces

### ○ Ready to Trigger
- **Batch 3 script:** `auto_submit_batch3.sh`
  - Will be triggered after training completes
  - Verifies all checkpoints before submission

---

## 監控指令

### 查看 Pipeline 狀態
```bash
scripts/check_pipeline_status.sh
```

### 查看 Batch 1 Jobs
```bash
squeue -u eliu354 --format="%.10i %.12P %.30j %.10T %.12M"
```

### 查看自動化日誌
```bash
tail -f logs/auto_training_*.log
```

### 手動檢查 P1 traces
```bash
ls -1 ~/scratch/Projects/DLLM-Delta-Compute/experiments/P1_traces/traces_500_*/|*.pt | wc -l
```

---

## 預期時間表

| Stage | Duration | Status |
|-------|----------|--------|
| Batch 1 Queue Wait | ~? min | In progress |
| Batch 1 Execution | ~25-30 min each | Pending |
| Training Queue Wait | ~? min | Not started |
| Training Execution | ~30 min each (3 parallel) | Not started |
| Batch 3 Queue Wait | ~? min | Not started |
| Batch 3 Execution | ~25-30 min each | Not started |
| **Total Estimated** | **2-4 hours** | - |

---

## Git Commits

1. **3ad8a7b** - Disable save_activations (清理 + 程式碼修改)
2. **97727b6** - Add pipeline automation scripts
3. **7951d4a** - Fix Phase A/B/C eval scripts (earlier)

---

## Next Steps

### Automatic (無需人工介入)
1. ✓ Batch 1 jobs queued
2. ○ Wait for P1 to complete
3. ○ Auto-submit training (via PID 1604214)
4. ○ Wait for training to complete
5. ○ Manually trigger Batch 3:
   ```bash
   source /tmp/batch2_jobs.txt
   nohup scripts/auto_submit_batch3.sh $P3_JOB $PHASEB_JOB $PHASEC_JOB > logs/auto_batch3_$(date +%Y%m%d_%H%M%S).log 2>&1 &
   ```

### Verification Points
- [ ] P1 generates ≥450 traces (lightweight, no activations)
- [ ] Training produces 3 checkpoint files
- [ ] All 8 phases complete successfully
- [ ] Extract timing metrics (tok/s, P95/P97/P99)
- [ ] Compare accuracy across all phases

---

## 重要修正

### Phase A 腳本錯誤 (已修正)
- **錯誤:** `cache_mode=phaseA_heuristic` (不存在的模式)
- **修正:** `cache_mode=l2c_ffn` + 逗號分隔的 layer indices
- **Schedule:** `"1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31"` (every other layer)

### P1 Trace 不完整問題
- **上次:** 375/500 traces (job 結束前中斷)
- **本次:** 預期完整 500 traces (無大檔案，不會空間不足)

---

## Contact Points

**Monitor script:** `scripts/check_pipeline_status.sh`  
**Job IDs saved:** `/tmp/batch1_jobs.txt`, `/tmp/batch2_jobs.txt`, `/tmp/batch3_jobs.txt`  
**Logs:** `logs/auto_training_*.log`, `logs/auto_batch3_*.log`
