# Dream Delta-Compute 可行性重新評估

## 基於實際 generation_utils.py 源碼的分析

在仔細閱讀了 `external/Dream/modeling/generation_utils.py` 後，我需要對之前的分析進行**重大修正**。

---

## 1. Critical Issue #1: generation_utils.py 是否找得到？

### ✅ **完全解決，之前的擔憂不成立**

**事實：**
- `generation_utils.py` 確實存在於 `external/Dream/modeling/` 中
- 完整的 `diffusion_generate` 和 `_sample` 方法都可見
- 核心 diffusion sampling loop 的邏輯完全透明

**關鍵代碼結構：**

```python
def _sample(self, input_ids, attention_mask, generation_config, ...):
    # 初始化：pad input_ids 到 max_length，用 mask_token_id 填充
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
    
    # Timestep schedule
    timesteps = torch.linspace(1, eps, steps + 1, device=x.device)
    
    # 主 diffusion loop
    for i in range(steps):
        mask_index = (x == mask_token_id)
        
        # 【關鍵】調用模型 forward
        logits = self(x, attention_mask, tok_idx).logits
        logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)
        
        # Hook for custom logits manipulation
        logits = generation_logits_hook_func(i, x, logits)
        
        mask_logits = logits[mask_index]
        t = timesteps[i]
        s = timesteps[i + 1]
        
        # 根據 alg 決定 unmask 哪些 tokens
        if alg == 'origin':
            # 隨機選擇
            p_transfer = 1 - s / t if i < steps - 1 else 1
            ...
        else:  # 'maskgit_plus', 'topk_margin', 'entropy'
            # 基於 confidence 選擇
            confidence, x0 = sample_tokens(mask_logits, ...)
            number_transfer_tokens = int(num_mask_token * (1 - s / t))
            _, transfer_index = torch.topk(full_confidence, number_transfer_tokens)
            x[row_indices, transfer_index] = x_[row_indices, transfer_index]
        
        # Hook for custom token manipulation
        x = generation_tokens_hook_func(i, x, logits)
```

**結論：**
- ✅ 源碼完全可見，可以隨意修改
- ✅ 提供了 hooks (`generation_logits_hook_func`, `generation_tokens_hook_func`) 供 POC 使用
- ✅ 之前的擔憂「核心組件缺失」**完全錯誤**

---

## 2. Critical Issue #2: Time Conditioning 機制

### ✅ **對方說得對：Dream 沒有顯式 time embedding 傳入模型**

**關鍵發現：**

從 `_sample` 方法的分析：

```python
# timesteps 只用於計算 unmask 比例
timesteps = torch.linspace(1, eps, steps + 1, device=x.device)

for i in range(steps):
    t = timesteps[i]
    s = timesteps[i + 1]
    
    # 【關鍵】模型 forward 沒有 time 參數！
    logits = self(x, attention_mask, tok_idx).logits
    
    # timestep 只用於這裡：
    p_transfer = 1 - s / t  # 或 number_transfer_tokens 計算
```

**與典型 continuous diffusion 的對比：**

```python
# Typical continuous diffusion (如 DiT):
def forward(self, x, t):
    t_emb = self.time_mlp(t)  # time embedding
    for layer in self.layers:
        x = layer(x, t_emb)  # 每層都用到 time
    return x

# Dream:
def forward(self, x, attention_mask, tok_idx):
    # 沒有 time 參數！
    for layer in self.layers:
        x = layer(x, ...)  # 沒有 time conditioning
    return x
```

**這意味著：**

1. **Dream 的 "time" 完全體現在離散狀態上：**
   - 早期 steps：大部分位置是 `<|mask|>`
   - 後期 steps：大部分位置已確定，只有少數 `<|mask|>`
   - 模型看到的是**當前的 token sequence**，而非 "第幾步"

2. **模型本身是 time-agnostic 的：**
   - `DreamModel.forward(x)` 對於相同的 `x`，無論在哪個 diffusion step，輸出都相同
   - Time information 只在 `_sample` 的 scheduling logic 中

3. **對 FFN caching 的影響：**
   - ✅ **沒有 "time embedding mismatch" 的問題**
   - ⚠️ 但仍有 "上下文變動" 的問題：
     - 如果 step t 時某個 token 已經 unmask，但 step t-1 時還是 mask
     - 兩個 steps 的 `x` 不同 → hidden states 不同 → cache 會有誤差

**結論：**
- ✅ 對方正確：**沒有顯式 time embedding**
- ✅ 我之前的擔憂「time embedding mismatch」**不適用於 Dream**
- ⚠️ 但「上下文變動導致 cache 誤差」仍然存在（這是另一個問題）

---

## 3. Critical Issue #3: Token-level Gating 是否可行？

### ⚠️ **部分同意對方，但仍有重要限制**

對方的論點：
> "Dream 的 token‑level gating 是可以做的近似，但比 AR LLM 敏感許多"

**我的重新評估：**

#### 3.1 技術上確實可以實現

從 `_sample` 的結構來看：

```python
# 每個 step 都是：
for i in range(steps):
    # 1. 識別哪些位置是 mask
    mask_index = (x == mask_token_id)
    
    # 2. 完整 forward（所有位置）
    logits = self(x, attention_mask, tok_idx).logits
    
    # 3. 只用 mask 位置的 logits
    mask_logits = logits[mask_index]
    
    # 4. 更新部分 mask 位置
    x[selected_positions] = sampled_tokens
```

**如果要做 token-level gating / caching：**

```python
# 修改 DreamModel.forward 支持 position-wise caching
def forward(self, x, attention_mask, tok_idx, token_cache_mask=None, prev_hidden_states=None):
    hidden_states = self.embed_tokens(x)
    
    for layer in self.layers:
        if token_cache_mask is not None and prev_hidden_states is not None:
            # 對於 cache 的 positions，復用 prev hidden states
            # 對於非 cache 的 positions，正常計算
            hidden_states = selective_layer_forward(
                layer, hidden_states, prev_hidden_states, token_cache_mask
            )
        else:
            hidden_states = layer(hidden_states, ...)
    
    return self.lm_head(hidden_states)
```

**技術上可行** ✅

#### 3.2 但有根本性的語義問題

**問題 1: "穩定 token" 的定義在 diffusion 中不清楚**

在 AR LLM：
- `x_{<t}` 已經生成，不會再改變
- 可以明確 cache `x_{<t}` 的 hidden states

在 Dream diffusion：
- 所有 positions 同時演化
- 某個 position 在 step t 被 unmask ≠ 它在 step t+1 就不會改變
  - 因為 bidirectional attention 會讓它受到其他位置變化的影響
  - 它的 hidden states 在每個 step 都會變

**舉例：**
```
Step 5:  "The [MASK] is [MASK]"
Step 6:  "The cat is [MASK]"      # unmask 了 "cat"
Step 7:  "The cat is sleeping"     # unmask 了 "sleeping"
```

在 Step 7：
- "cat" 的 token value 沒變
- **但** "cat" 的 hidden states 會變，因為：
  - 它的 attention 現在看到 "sleeping" 而非 `[MASK]`
  - 雙向 attention 讓它受到右邊上下文影響

**所以：**
- 你可以 cache "cat" 這個 **token value**（它確實不會再變）
- 但你**不能 cache "cat" 的 hidden states**（它會因為上下文變化而變）

#### 3.3 可能可行的方案：Partial Computation Gating

**不是 "完全凍結某些 positions"，而是 "對穩定 positions 做粗略計算"：**

```python
# 方案 A: 分級計算精度
for position in sequence:
    if is_stable(position):
        # 用更少的 layers 或更簡單的 approximation
        hidden[position] = lightweight_forward(hidden[position])
    else:
        # 完整計算
        hidden[position] = full_forward(hidden[position])
```

```python
# 方案 B: 早停某些 positions 的迭代
for position in sequence:
    if is_stable(position) and step > early_stop_threshold:
        # 直接用 step N 的結果，不再更新
        x[position] = x_frozen[position]
```

**這些方案的共同點：**
- 不依賴 "cache hidden states"（因為會 drift）
- 而是 "減少對穩定 positions 的計算量"

#### 3.4 最保守但可行的方案：Sequence-level Early Stopping

與其做 position-wise gating，不如做 **sequence-level early stopping**：

```python
for i in range(steps):
    logits = self(x, ...)
    
    # 計算整個 sequence 的 stability signal
    if sequence_is_stable(logits, x, threshold):
        # 提前終止整個 diffusion process
        break
    
    x = update_x(x, logits, ...)
```

**優點：**
- 語義清晰
- 不需要修改 DreamModel 內部
- 只需在 `_sample` loop 中加 early stopping logic

**缺點：**
- 粒度粗（all-or-nothing）
- 但對 POC 足夠

---

## 4. 修訂後的可行性評估

### ✅ 完全可行的方案

| 方案 | 可行性 | 複雜度 | 預期效果 |
|------|--------|--------|----------|
| **Sequence-level Early Stopping** | ✅✅✅ | 低 | 1.2-1.5× 提速 |
| **Step-level Skipping** (跳過部分 diffusion steps) | ✅✅✅ | 低 | 1.3-2× 提速 |
| **Layer-level FFN Caching** (L2C Phase A) | ✅✅ | 中 | 1.2-1.4× 提速 |
| **Learned Router** (step/layer level) | ✅✅ | 中-高 | 1.5-2× 提速 |

### ⚠️ 有風險但值得嘗試的方案

| 方案 | 可行性 | 主要風險 | 建議 |
|------|--------|----------|------|
| **Position-wise Adaptive Precision** | ⚠️ | 實現複雜，質量不確定 | v2 再考慮 |
| **Token-level FFN Caching** | ⚠️ | Bidirectional attention drift | 需要大量實驗 |

### ❌ 不建議的方案

| 方案 | 為何不可行 |
|------|-----------|
| **Token-level Hidden State Freezing** | 與 bidirectional attention 根本衝突 |
| **Per-token Time-aware Gating** | Dream 沒有 per-token time 概念 |

---

## 5. 對 Master Plan 的具體建議

### 修訂 P2: Rule-Based Gate

**原計畫（不可行）：**
> Token-level freezing with masks that skip compute for "stable tokens"

**修訂建議：**

**P2a: Sequence-level Early Stopping** ✅ POC v1b
```python
# 在 _sample loop 中加入：
for i in range(steps):
    logits = self(x, ...)
    
    # 計算 stability metrics
    entropy = compute_entropy(logits[mask_index])
    margin = compute_margin(logits[mask_index])
    
    if entropy < tau_entropy and margin > tau_margin:
        # 提前結束
        break
```

**P2b: Step-level Skipping** ✅ POC v1b
```python
# 修改 timestep schedule
timesteps_full = torch.linspace(1, eps, steps + 1)
timesteps_skip = timesteps_full[::skip_stride]  # 例如 [0, 2, 4, ...]

for i, t in enumerate(timesteps_skip):
    logits = self(x, ...)
    # 用更大的 stride 更新更多 tokens
```

### 保持 L2C Phase A：FFN Caching

**修訂理由：**
- ✅ 沒有 time embedding mismatch 問題（已驗證）
- ⚠️ 但有上下文變動問題（需要實驗評估）

**實現策略（保守）：**

```python
# 只在相鄰 steps 間 cache，不跨太多 steps
class CachedDreamDecoderLayer(DreamDecoderLayer):
    def forward(self, hidden_states, use_ffn_cache=False, ffn_cache=None, ...):
        # Attention 永遠重新計算（因為上下文會變）
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, ... = self.self_attn(hidden_states, ...)
        hidden_states = residual + hidden_states
        
        # FFN 可以選擇性 cache
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        
        if use_ffn_cache and ffn_cache is not None:
            # 使用 cache（來自上一個 step）
            mlp_out = ffn_cache
        else:
            mlp_out = self.mlp(hidden_states)
        
        hidden_states = residual + mlp_out
        return hidden_states, mlp_out  # 返回供下一步 cache
```

**關鍵約束：**
- 只在 **alternating steps** cache（even full, odd cached）
- 不跨多個 steps cache（因為 context drift 會累積）

### 新增 P2c: Adaptive Step Stride

**靈感來自 DPM-Solver 等 adaptive ODE solvers**

```python
def adaptive_sample(self, input_ids, ...):
    x = F.pad(input_ids, ...)
    
    current_time = 1.0
    step_count = 0
    
    while current_time > eps and step_count < max_steps:
        logits = self(x, ...)
        
        # 估計 "local truncation error" 或 confidence
        stability = estimate_stability(logits, x)
        
        # 根據 stability 調整 stride
        if stability > high_threshold:
            stride = large_stride  # 可以大步走
        else:
            stride = small_stride  # 需要小心
        
        next_time = max(current_time - stride, eps)
        x = update_x(x, logits, current_time, next_time)
        
        current_time = next_time
        step_count += 1
```

---

## 6. 最終結論

### 對方說得對的部分 ✅

1. **generation_utils.py 完全可得**
   - 我之前的擔憂是錯的
   - 源碼完整，甚至有 hooks 支持

2. **Dream 沒有顯式 time embedding**
   - 確實如此
   - Time 只在 scheduling logic，不在模型 forward 中

3. **Token-level gating 技術上可以實現**
   - 可以在 forward 中加 position-wise logic
   - 但...

### 對方沒說清楚的部分 ⚠️

1. **Token-level gating 的語義問題**
   - 技術上可行 ≠ 概念上合理
   - Bidirectional attention 讓 "凍結 hidden states" 很難 justify
   - 需要重新定義成 "adaptive precision" 或 "early stopping"

2. **FFN caching 的 drift 問題**
   - 雖然沒有 time embedding mismatch
   - 但 context drift 仍會導致 cached FFN output 不準確
   - 需要實驗驗證誤差大小

3. **實現複雜度被低估**
   - "在 DreamModel 的 block 裡做 token‑wise masking / caching" 不是 trivial
   - 需要處理：
     - Shape broadcasting
     - Gradient checkpointing compatibility
     - Distributed training (if needed)
     - Cache memory management

### 我的最終建議 📋

**POC v1a (Week 1-2): Validation & Simple Baseline**
1. ✅ 跑 Teacher baseline（GSM8K）
2. ✅ 收集簡單的 per-step statistics（不需要跨 layers）
3. ✅ 實現 **Sequence-level Early Stopping**（最簡單）

**POC v1b (Week 3-4): 選一個方向深入**

**Option A: Step-level Acceleration** (推薦)
- Step-level skipping with simple schedule
- Adaptive step stride based on confidence
- 風險低，效果可預期（1.3-2× speedup）

**Option B: L2C FFN Caching** (如果對 latency 要求高)
- Alternating full/cached steps
- 只 cache FFN，attention 重算
- 需要先做小規模驗證（5 samples）

**不建議在 POC v1 做：**
- ❌ Token-level hidden state caching/freezing
- ❌ 複雜的 learned gate (P3)
- ❌ Multi-signal gating (P2-v2)

---

## 7. 立即 Action Items

### Priority 0: 確認理解正確

創建測試來驗證我的分析：

```python
# tests/test_dream_properties.py

def test_time_independence():
    """驗證 Dream forward 不依賴 diffusion step"""
    model = load_dream_model()
    x = create_test_sequence()
    
    # 相同 input，多次 forward
    with torch.no_grad():
        out1 = model(x, attention_mask, tok_idx)
        out2 = model(x, attention_mask, tok_idx)
    
    assert torch.allclose(out1.logits, out2.logits)
    print("✅ Confirmed: Dream forward is deterministic and time-independent")

def test_bidirectional_attention():
    """驗證 attention 確實是雙向的"""
    model = load_dream_model()
    
    # 兩個序列：只有最後一個 token 不同
    x1 = torch.tensor([[1, 2, 3, 4, 5]])
    x2 = torch.tensor([[1, 2, 3, 4, 999]])
    
    with torch.no_grad():
        out1 = model(x1, ...)
        out2 = model(x2, ...)
    
    # 如果是 causal，前 4 個 positions 應該相同
    # 如果是 bidirectional，前 4 個 positions 也會不同
    if torch.allclose(out1.logits[:, :4], out2.logits[:, :4]):
        print("❌ Attention is causal (unexpected)")
    else:
        print("✅ Confirmed: Attention is bidirectional")
```

### Priority 1: 最小 POC (Sequence Early Stopping)

```python
# experiments/01_early_stopping_poc.py

def early_stopping_generate(model, input_ids, steps=256, tau=0.5):
    """Modified _sample with early stopping"""
    x = F.pad(input_ids, ..., value=mask_token_id)
    timesteps = torch.linspace(1, eps, steps + 1)
    
    for i in range(steps):
        mask_index = (x == mask_token_id)
        
        if mask_index.sum() == 0:
            print(f"Early stop at step {i}: all positions unmasked")
            break
        
        logits = model(x, ...)
        mask_logits = logits[mask_index]
        
        # 計算 entropy
        probs = F.softmax(mask_logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(-1).mean()
        
        if entropy < tau:
            print(f"Early stop at step {i}: entropy {entropy:.3f} < {tau}")
            break
        
        # 正常 update
        x = update_x(x, logits, ...)
    
    return x

# 在 5 個 GSM8K samples 上測試
results = []
for sample in test_samples:
    out_full = model.diffusion_generate(sample, steps=256)
    out_early = early_stopping_generate(model, sample, steps=256, tau=0.5)
    
    results.append({
        'full_steps': 256,
        'early_steps': actual_steps,
        'speedup': 256 / actual_steps,
        'output_match': (out_full == out_early).all()
    })
```

### Priority 2: FFN Cache 驗證實驗

```python
# experiments/02_ffn_cache_error.py

def measure_cache_error():
    """測量 FFN caching 的誤差"""
    model = load_dream_model()
    
    # 模擬兩個相鄰 steps
    x_t = create_test_sequence()  # step t
    x_t1 = modify_few_tokens(x_t)  # step t+1（少數 tokens 變了）
    
    with torch.no_grad():
        # Full computation
        out_t = model(x_t, ...)
        out_t1 = model(x_t1, ...)
        
        # Cached computation（復用 step t 的某些 layer outputs）
        out_t1_cached = model_with_cache(x_t1, cache_from=out_t)
    
    # 測量誤差
    mse = F.mse_loss(out_t1.logits, out_t1_cached.logits)
    cosine = F.cosine_similarity(
        out_t1.logits.flatten(),
        out_t1_cached.logits.flatten(),
        dim=0
    )
    
    print(f"MSE: {mse:.6f}, Cosine: {cosine:.6f}")
    
    # 決策
    if cosine > 0.99:
        print("✅ FFN caching looks safe")
    else:
        print("⚠️ FFN caching has significant error")
```

---

## 8. 更新後的 Risk Assessment

| Risk | 之前評估 | 現在評估 | 變化原因 |
|------|---------|---------|---------|
| generation_utils.py 缺失 | 🔴 Critical | ✅ Resolved | 文件存在 |
| Time conditioning 未知 | 🟡 Medium | ✅ Resolved | 確認無顯式 time emb |
| Token-level gating 不可行 | 🔴 High | 🟡 Medium | 技術上可行，但需重新設計 |
| FFN caching 誤差大 | 🟡 Medium | 🟡 Medium | 無 time mismatch，但有 context drift |
| 實現複雜度高 | 🟡 Medium | 🟡 Medium | 維持不變 |

**整體風險：** 🟡 Medium → 🟢 Low-Medium（顯著改善）

---

## 9. 與對方觀點的最終對比

| 論點 | 對方 | 我的評估 | 共識？ |
|------|------|----------|--------|
| generation_utils.py 可得 | ✅ 是 | ✅ 是 | ✅ 完全同意 |
| Dream 無 time embedding | ✅ 是 | ✅ 是 | ✅ 完全同意 |
| Token-level gating 可行 | ⚠️ 技術上可行，風險較大 | ⚠️ 可實現但需重新定義 | ⚠️ 部分同意 |
| FFN caching 安全 | ✅ 樂觀 | ⚠️ 謹慎樂觀 | ⚠️ 需要實驗驗證 |
| 應該先做粗粒度 | ✅ 同意 | ✅ 強烈同意 | ✅ 完全同意 |

**最大分歧：**
- 對方認為 token-level gating 主要風險是 "敏感度高"
- 我認為主要問題是 "語義不清" + "bidirectional attention drift"
- **但我們都同意：POC 應該從 step-level 開始**

---

## 結論

經過重新評估，我承認：

1. ✅ **我之前的分析過於悲觀**，特別是對 generation_utils.py 和 time conditioning 的擔憂
2. ✅ **對方的核心觀點正確**：技術上可行，但需要從粗粒度開始
3. ⚠️ **但我仍然認為**：token-level gating 需要重新設計成 "adaptive precision" 或 "early stopping"，而非單純的 "hidden state freezing"

**最佳路徑：**
- POC v1a: Teacher + Simple Stats + **Sequence-level Early Stopping**
- POC v1b: **Step-level Skipping** 或 **L2C FFN Caching**（二選一）
- POC v2: Learned Router (step/layer level)
- 未來: Token-level Adaptive Precision (if needed)

**關鍵是：先證明粗粒度的方法有效，再考慮細粒度。**
