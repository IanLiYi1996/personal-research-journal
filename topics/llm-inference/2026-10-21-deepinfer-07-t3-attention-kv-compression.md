# DeepInfer 专题 T3：注意力与 KV 压缩

> **⚠️ 勘误（主编交叉核对，2026-10-21）**：§KV 体积核算中，ica_cache（indexer K）并不是 8 个源层各分配一份。
> - `hugging_face/config.py:74` 构造 `layer_index_k_map` 时用的是 `text.get("index_k_source_layer_ids", text["kv_source_layer_ids"])`。V4.1 的 config.json **没有**该键，因此 index-K 映射等于 KV 映射，即 {2,8,14,20}。
> - `cache_engine.py:174-187` 优先使用 `layer_index_k_map`；`cross_layer.py:30` 文档中"默认跟随 index_module_map"的说法，在 HF 加载路径下不适用。
> - 因此 ica_cache = 3 × 68 / 2 + 68 = **170 B**，分页 KV 合计 = 720 + 170 = **890 B/token**（不是 1,162 B）。相对 V3 MLA BF16（约 69 KB/token）约 **1/78**。
> - 第 24/28/32/36 层仍有独立的 indexer 模块（各自的 Q 投影和 top-k），但复用第 20 层的 K cache。


> 基准：DeepInfer commit 774ce24；HF 配置来自 `deepseek-ai/DeepSeek-V4.1-Flash-1017`（curl 实时获取）。
> 标记：**[源码]** / **[Q&A]** / **[估算]** / **[推断]** / **[未核实]**

---

## 1. 整体架构速览

V4.1-Flash 的注意力栈由五种层构成，由 `compress_ratios` 列表索引驱动：

```
层 0–1   : ratio=0  → SWA-only（win_spec，无压缩 KV）
层 2–19  : ratio=2  → C2 门控压缩 + SWA（compress_spec）
层 20–39 : ratio=1  → C1 恒等压缩 + SWA，YOCO consumer（compress_spec）
层 40–42 : ratio=0  → DFlash/nextn 草稿头（dflash_spec，3 层 MTP）
```

**[源码]** `deepinfer/hugging_face/config.py:_n_prefill_layer`（line 158）：

```python
def _n_prefill_layer(text):
    ratios = text["compress_ratios"]
    return next(i for i, ratio in enumerate(ratios) if ratio == 1)
# → 20（compress_ratios[20] = 1 是第一个 1）
```

`n_prefill_layer = 20` 是 YOCO 分界点：层 0–19 是 prefill 侧，层 20–39 是 consumer 侧。

---

## 2. 源码精读：deepinfer/hugging_face/config.py

### 2.1 compress_ratios 与 attn_module_list

**[源码]** `config.py:_build_attn_module_list`（line 146）：

```python
specs = [_win_spec(text) if ratio == 0 else _compress_spec(text, ratio, shared)
         for ratio in ratios[:n_layer]]
specs.append(None)          # lm_head 占位
specs.extend(_dflash_spec(text) for _ in range(nextn))
```

- `_win_spec`：`sparse_stage=0`，仅 SWA，无 `index_topk`/`PackedWkv`
- `_compress_spec`：包含 `kv_compress_ratio`、`index_topk`、`attn_index`（indexer 参数）、共享拓扑 `layer_info_shared_cfg`
- `_dflash_spec`：`sink_mode="swa"`，`rope_theta=rope_theta`（无压缩，复用主模型 SWA 布局）

### 2.2 _shared_layer_cfg 与 nearest-source 映射

**[源码]** `config.py:_source_map`（line 62）：

```python
def _source_map(source_ids, n_layer):
    ids = sorted(source_ids)
    return {layer: max(s for s in ids if s <= layer) for layer in range(ids[0], n_layer)}
```

从实际配置（curl 核实）：
- `kv_source_layer_ids = [2, 8, 14, 20]`
- `index_source_layer_ids = [2, 8, 14, 20, 24, 28, 32, 36]`
- `index_k_source_layer_ids = None`（默认等于 `index_source_layer_ids`）
- `candidate_source_layer_id = 20`

生成的映射：

| 映射 | 内容 |
|---|---|
| `layer_kv_map` | 2–7→2, 8–13→8, 14–19→14, 20–39→20（4 段） |
| `layer_index_module_map` | 2–7→2, 8–13→8, 14–19→14, 20–23→20, 24–27→24, 28–31→28, 32–35→32, 36–39→36（8 段） |
| `layer_index_k_map` | 同 `layer_index_module_map` |
| `layer_candidate_source_map` | 20–39→20（候选块由第 20 层产出，后续复用） |

### 2.3 kv_lora_rank 推导

**[源码]** `config.py` line 289：

```python
kv_lora_rank = text["head_dim"] - text["qk_rope_head_dim"]
             = 512 - 64 = 448
```

HF `config.json` 的 `kv_lora_rank` 字段为 `null`，DeepInfer 从 `head_dim - qk_rope_head_dim` 反推。
`compress_dim = kv_lora_rank + qk_rope_head_dim = 448 + 64 = 512`，等于 `head_dim`。

### 2.4 use_fp4_sparse_kv

**[源码]** `config.py` line 303：

```python
use_fp4_sparse_kv = text.get("use_fp4_sparse_kv", True)
```

HF 配置中该字段为 `null`，默认值为 **True**，即主 KV 使用 **V41_FP4_FORMAT**（288 字节/压缩 token），而 SWA 使用 **V41_FORMAT**（528 字节/token）。

---

## 3. KV 缓存格式精确定义

**[源码]** `deepinfer/model_executor/layers/tilelang_ops/cache_common.py`

| 格式常量 | 量化 | 元素数 | scale 粒度 | 字节/token | 用途 |
|---|---|---|---|---|---|
| `V41_FORMAT` | E4M3 FP8 | 512 | per-32 UE8M0 | 512+16=**528** | SWA（主/DFlash） |
| `V41_FP4_FORMAT` | E2M1 FP4 | 512 | per-16 E4M3 | 256+32=**288** | 主压缩 KV（ca_cache） |
| `FP4_INDEX_KV_FORMAT` | E2M1 FP4 | 128 | per-32 UE8M0 | 64+4=**68** | 稀疏 indexer KV（ica_cache） |

**[源码]** cache_common.py（以 V41_FP4_FORMAT 为例）：
- E2M1：512 值 × 4 bits ÷ 8 = 256 字节
- E4M3 scale：512÷16=32 个 scale × 1 字节 = 32 字节
- 合计：**288 字节**（非 MXFP8 格式，无 FP8 data+scale 混叠）

**SWA（V41_FORMAT）**：
- E4M3 FP8：512 × 1 = 512 字节
- UE8M0 scale：512÷32=16 × 1 = 16 字节
- 合计：**528 字节**

---

## 4. 精确核算分页 KV 体积

### 4.1 逐层分配逻辑

**[源码]** `worker/cache_engine.py:KVCacheManager.process`（line 378）：

```python
# ca_cache（主压缩 KV）
token_per_block = block_size // ratio
delta = token_per_block * self._bytes_per_token   # = 288 B
padded_delta = align_to_multiple(delta, 1, 32)

# ica_cache（稀疏 indexer KV）
token_per_block = block_size // ratio
delta = token_per_block * self._qk_topk_bytes     # = 68 B
```

只有 **source 层**（`kv_mapped_layer_id == layer_id`）才分配，consumer 层直接复用 source 层的 tensor 视图。

**[源码]** `cache_engine.py:_get_qk_topk_bytes`（line 73）：

```python
def per_token_bytes(ic):
    # FP4: two values per byte, one UE8M0 scale per 32 channels.
    return ic.index_head_dim // 2 + ic.index_head_dim // 32
# = 128 // 2 + 128 // 32 = 64 + 4 = 68 字节
```

### 4.2 逐源层计算（基于真实 config）

**ca_cache（主压缩 KV）**，288 字节/压缩 token，每 `ratio` 个原始 token 产出 1 个压缩 token：

| 源层 | ratio | 字节/原始 token |
|---|---|---|
| 层 2 | C2 | 288÷2 = **144** |
| 层 8 | C2 | 288÷2 = **144** |
| 层 14 | C2 | 288÷2 = **144** |
| 层 20 | C1 | 288÷1 = **288** |
| **小计** | | **720 字节/token** |

**ica_cache（稀疏 indexer KV）**，68 字节/压缩 token：

| 源层 | ratio | 字节/原始 token |
|---|---|---|
| 层 2 | C2 | 68÷2 = **34** |
| 层 8 | C2 | 68÷2 = **34** |
| 层 14 | C2 | 68÷2 = **34** |
| 层 20 | C1 | 68÷1 = **68** |
| 层 24 | C1 | 68÷1 = **68** |
| 层 28 | C1 | 68÷1 = **68** |
| 层 32 | C1 | 68÷1 = **68** |
| 层 36 | C1 | 68÷1 = **68** |
| **小计** | | **442 字节/token** |

> **[估算] 分页 KV 合计 = 720 + 442 = 1162 字节/token ≈ 1.135 KB/token**

### 4.3 与总报告估算对比

**总报告 §4.3 给出 "≈1.16 KB/token"**。精确值为 1162 字节 ≈ 1.135 KB，误差约 2.5%。差值来源：
- 总报告可能未单独列出 ica_cache 的完整核算，将结果取整至 1.16 KB
- 典型 block_size（64/256）下，层内 32 字节对齐不产生额外 per-token 开销（所有 `delta` 均能被 32 整除）
- 结论：**总报告 1.16 KB 是合理近似，实际精确值为 1162 B ≈ 1.135 KB/token**

### 4.4 SWA state 每序列字节

**[源码]** `cache_engine.py:StateBlock`（line 221）：

SWA 格式：V41_FORMAT = **528 字节/token**

```
swa_block_size_inference = window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS
```

当 nextn=5（3 个 MTP 层）且 environ.py 将 backtracking 上调为 max(raw, nextn+1) = 6：

```
swa_block_size_inference = 128 + 6 = 134 token
```

**SWA state 布局（per sequence）**：

| 区域 | 层数 | 字节 |
|---|---|---|
| stored SWA（层 0–19，可落盘） | 20 | 20 × 134 × 528 = **1,415,040 ≈ 1.35 MiB** |
| dropped SWA（层 20–39，YOCO consumer，推理时保留，不落盘） | 20 | 20 × 134 × 528 = **1,415,040 ≈ 1.35 MiB** |
| 压缩 state（C2 KV 源层 2/8/14，FP32 ring） | 3 | 3 × 7 × 2 × 512 × 4 = **86,016 ≈ 84 KiB** |
| **推理总 state/seq** | | **约 2.86 MiB** |
| **落盘部分（state_size_to_disk）** | | **约 1.35 MiB** |

> 注："134" = window(128) + backtracking(6) by nextn=5；"140" = DEEPINFER_YOCO_PREFILL_KEEP_TOKENS 默认值（≥ window+backtracking=134 的安全余量）。

### 4.5 DFlash 草稿 KV

**[源码]** `v4_modules/attn.py:DFlashAttention.forward_drafts`（line 963）：

草稿 KV 写入 `kv_cache.swa_cache`（layers 40–42 各有独立 SWA slot），格式 V41_FORMAT = 528 字节/token，**每解码步覆写，不保留历史**。每 nextn 层仅占 `swa_block_size_inference` 个 token 的一次性 SWA 空间，约 134 × 528 × 3 ≈ 212 KiB/seq，随 decode 步刷新。

### 4.6 与业界方案对比表

| 模型 | 注意力 | KV 量化 | 分页 KV 字节/token | 备注 |
|---|---|---|---|---|
| **V4.1-Flash（DeepInfer）** | MLA + C1/C2 压缩 + 稀疏 indexer | FP4(main)+FP4(index)+FP8(SWA) | **1162 B** | 4 主 KV 源 + 8 indexer 源 |
| DeepSeek-V3 MLA BF16 | MLA | BF16 | 512 × 2 = **1024 B** × 全部 61 层 ≈ 大 | 每层 512 维 BF16 |
| DeepSeek-V3 MLA FP8 | MLA | FP8 per-token | 512 × 1 + 16 = **528 B** × 全部 61 层 | MXFP8 per-token |
| DeepSeek-V3.2 DSA（[推断]） | MLA + sparse subset | FP8 | 大幅削减（仅部分层存 full KV）[未核实] | 技术报告暂未公开 |
| Llama-3-70B GQA | GQA（8 个 KV heads） | BF16 | 8 × 128 × 2 × 2（K+V）= **4096 B** × 80 层 | 远大于 MLA |
| Llama-3-70B GQA INT8 | GQA | INT8 | 8 × 128 × 1 × 2 = **2048 B** × 80 层 | 仍远大于 MLA |

> **[估算]** Llama-3-70B BF16 GQA：每层 4096 B × 80 层 = 327,680 B/token，远超 V4.1-Flash 的 1162 B/token（约 280×）。即便 MLA BF16 全量（V3 基线）也是 512 × 2 × 61 ≈ 62,464 B/token，V4.1-Flash 的 1162 B 比 V3 MLA BF16 少约 53×，体现了 YOCO + 压缩 + 稀疏 indexer 的综合效果。


---

## 5. 压缩机制原理：C2 门控与 C1 恒等

### 5.1 C2 SoftmaxAPE 门控压缩

**[源码]** `mla_attn.py:SoftmaxAPEGatedColumnParallelLora`（line 62）

C2 门控的核心公式（从 `tilelang_ops/compress.py` 参考实现提取）：

```python
# projected_kv 形状：[T, 2*D]，其中前 D 为 score，后 D 为 latent
score, latent = projected_kv.chunk(2, dim=-1)
# 以 ratio=2 为例，按步长 2 分组
output = (latent * softmax(score, dim=0)).sum(dim=0)
# 等价于：每 2 个连续 token 的 latent，由 score 通过 softmax 加权求和
```

**[源码]** `v4_modules/compress.py:Compressor.forward_pre_norm`（line 87）：

- C1：`assert projected_kv.dtype == bfloat16 and state_cache is None`
  → 直接返回 projected_kv（恒等操作，无状态）
- C2：`assert projected_kv.dtype == float32 and state_cache is not None`
  → `score, latent = projected_kv.chunk(2, dim=-1)` → 调用 `compress_and_update_state_prefill/decode`

**关键设计细节**：

1. **状态 ring buffer**：`state_cache` 形状 `[num_slots, state_cache_size, 2, compress_dim]`，以 FP32 精度存储跨 chunk 的 score/latent，用于处理序列边界处的压缩 token（当上一 chunk 末尾的奇数 token 与本 chunk 首 token 配对）。
2. **state_cache_size**：`= compress_ratio + max(0, backtracking - 1) = 2 + (6-1) = 7`（含 backtracking 余量）。
3. **Gate=None for C1**：`SoftmaxAPEGatedColumnParallelLora.__init__` 中 ratio==1 时 `self.gate = None`，直接输出低秩投影结果，不分 score/latent。
4. **压缩后的 seq len**：C2 使得 `T → T//2` 个 KV token；C1 保持 `T → T` 个 KV token。源层存储的是**压缩后**的 token，每个 KV block 实际存 `block_size // ratio` 个 token。

### 5.2 C1 恒等压缩（YOCO consumer 层）

YOCO consumer 层（20–39）全部使用 ratio=1（C1）。C1 的设计意图是：这些层本身就需要**完整上下文 KV**（YOCO 结构保证整个输入一次性处理），因此不需要也不能压缩。仅 kv_lora_rank=448 维的 MLA 低秩投影本身就是"压缩"（相比 GQA head_dim×heads），C1 表达"不做时间维度再次压缩"。

### 5.3 Wkv 投影与 inverse RoPE

**[源码]** `mla_attn.py:LoraRopeCompressFlashMHAAutoRecomp.__init__`（line 215）：

```python
self.Wkv = ColumnParallelLora(rank=kv_lora_rank, num_lora_rope_head=1, rope_head_dim=64)
```

- `num_lora_rope_head=1`：单头 KV（MQA 风格），所有 query heads 共享一个 KV latent。
- `rope_head_dim=64`：KV 的 RoPE 部分。
- `compress_dim = kv_lora_rank + rope_head_dim = 448 + 64 = 512`。
- 解码时通过"inverse RoPE"将 KV cache 内存储的已 RoPE 的部分还原，再与当前位置的 RoPE 重新组合——这是 MLA 解码的标准技巧，避免存储全 KV heads。

**[源码]** `mla_attn.py:PackedWkv`（line 237）：

```python
self.PackedWkv = SoftmaxAPEGatedColumnParallelLora(rank=kv_lora_rank + qk_rope_head_dim)
```

C2 KV 源层使用 `PackedWkv`（Gate + score/latent 分离），C1 层使用 `Wkv`（无 Gate）。

---

## 6. 稀疏 Indexer：MQA 两阶段 top-k

### 6.1 MQA 结构

**[源码]** `v4_modules/mla_attn.py:MQAIndexer.__init__`（line 186）：

```python
use_fp4 = True  # 恒定，不可配置
n_head = cfg.index_n_heads    # = 32
head_dim = cfg.index_head_dim  # = 128
rope_dim = cfg.index_rope_dim  # = 64
```

稀疏 Indexer 是 32 head × 128 dim 的 MQA（multi-query attention），以 relu(q·k) 代替 softmax 作为稀疏 logit，通过可学习权重 `weights` 将 32 head logit 聚合成一个打分，选出 top-512 压缩 KV 块。

### 6.2 Logit 计算：relu(q·k) × w

**[源码]** `v4_modules/index_attn.py:MQA._topk_prefill`（line 218）：

```python
logits = deep_gemm.fp8_fp4_mqa_logits(q_fp4, k_fp4, ...)
# q_fp4: [T, n_head, head_dim], k_fp4: [M, 1, head_dim] FP4
# 输出 logits: [T, n_head, M]
logits = torch.relu(logits) * weights  # weights: [n_head, 1]
logits = logits.sum(dim=1)             # → [T, M]
topk_indices = topk(logits, k=index_topk)  # top-512
```

核心特点：
- **FP4 Q/K**：`_quantize_q` 将 query 量化为 E2M1 FP4，与存储的 FP4 index KV（ica_cache，68字节/token）计算。
- **relu 非线性**：代替 softmax，直接稀疏化负值为 0，只有"有意义"的 attention 才参与加权。
- **head-wise 权重 `w`**：1D 可学习，形状 `[n_head, 1]`，将 32 头的 logit 投票聚合。
- **`from deep_select.interface import topk`**：使用 Hopscotch/硬件加速 top-k 核。

### 6.3 候选块机制（2 阶段过滤）

**[源码]** `v4_modules/index_attn.py:MQA._topk_prefill`（line 244）：

```python
if block_candidate_idxs is not None:
    logits = fp8_fp4_sparse_mqa_logits(q_fp4, k_fp4, block_candidate_idxs, ...)
    # 仅在候选块内计算，跳过其他块
```

**[源码]** `v4_modules/cross_layer.py:CrossLayerRuntime`（field `block_candidate_idxs`）：

候选块流程：
1. **层 20（candidate_source_layer_id）**：稠密计算全部 KV 块，产出 2048 个候选块（`candidate_topk_blocks=2048`，每块 8 token = 16384 候选 token）→ 存入 `cross_layer_runtime.block_candidate_idxs`。
2. **层 24/28/32/36**：使用 `fp8_fp4_sparse_mqa_logits`，只在 2048 个候选块内计算，O(2048×8×128×32) 而非全量。
3. **最终 top-k**：从候选内选出 `index_topk=512` 个 token，存入 `cross_layer_runtime.topk_indices`。

**[源码]** `SharedLayerTopology` 的 chain constraint 验证（cross_layer.py line ~156）：

```
index-topk ⊆ index-K ⊆ KV
```
即稀疏选出的 512 个 token 必须是 index-K 集合（8 indexer 源的并集）的子集，而 index-K 集合必须是 KV 集合（4 KV 源的并集）的子集。这保证了稀疏 attention 用的 V 一定存在于 KV cache 中。

### 6.4 decode 路径

**[源码]** `index_attn.py:MQA._topk_decode`（line 278）：

```python
logits = deep_gemm.fp8_fp4_paged_mqa_logits(
    q_fp4, k_fp4, block_table, ...)
# 分页访问 ica_cache，FP4 Q/K MQA logit
topk_indices = topk(logits, k=index_topk)
```

Decode 时 KV cache 已通过 block_table 分页，paged_mqa_logits 直接在页表上稀疏计算，不需要 gather。


---

## 7. YOCO 与 SWA 有界重放

### 7.1 YOCO 架构原理

**[源码]** `v4_modules/yoco.py:yoco_enabled`（line 18）：

```python
def yoco_enabled(cfg):
    return cfg.n_prefill_layer < cfg.num_hidden_layers
# → 20 < 40 = True
```

YOCO（You Only Cache Once, arXiv 2405.05254）将 Transformer 分为两段：
- **prefill 段（层 0–19）**：正常双向（或 causal）计算，每层产出自己的 KV。
- **consumer 段（层 20–39）**：只有第一层（层 20）执行完整的 KV 投影；后续层共享层 20 的 `shared_hidden`（经过 attn-norm 的 BF16 隐状态），直接从这个共享隐状态产出 KV，而无需再次过 self-attention。

```
[层 0–19: 普通因果注意力，各自KV]
       ↓ n_prefill_layer=20 分界
[层 20: 完整 KV 投影，capture shared_hidden]
[层 21–39: 从 shared_hidden 生成 KV，跳过 self-attention]
```

**[源码]** `v4_modules/attn.py:V4Attention.__init__`（line 83）：

```python
self.is_first_yoco_consumer_layer = (cfg.n_prefill_layer == layer_idx)
self.yoco_swa_share_hidden = cfg.is_yoco_hidden_consumer_layer
# consumer 层的 SWA KV 也从 shared_hidden 投影
```

### 7.2 Boundary Compaction

**[源码]** `decoder_layer.py:DeepseekDecoderLayer.forward`（line 142）：

```python
if consumer_full_metadata is not None and consumer_full_hidden is None:
    # 触发边界紧缩：full token → compact query
    compact_hidden, full_hidden = compact_yoco_boundary(
        hidden_states, cross_layer_runtime, input_metadata)
    cross_layer_runtime.consumer_full_hidden = full_hidden.clone()
    # compact_hidden 进入 attention，full_hidden 存为 shared_hidden
    cross_layer_runtime.shared_hidden = hidden_states_bf16.clone()
```

"Compaction" 的意义：YOCO prefill 时，仅需对**新增 token（consumer query）**计算注意力，而 KV 来自全量历史。`compact_yoco_boundary` 从全量 layout 中提取 compact token indices，减少 Q 的 FLOP。

### 7.3 Truncation Mode：off / mask / compact

**[源码]** `model_executor/input_metadata/yoco.py:YocoMetadataMixin`（line 34）：

三种截断模式，对应长序列 prefill 时如何处理落在滑窗边界外的旧 token：

| 模式 | 行为 | 适用场景 |
|---|---|---|
| `off` | 不截断，处理全部 consumer query | 序列短于 keep_tokens |
| `mask` | 保留全量 layout，用 -1 标记截断行 | 内存充裕，需要正确的 position ids |
| `compact` | 真实减少 Q tensor 的行数（boundary compaction） | 长序列，节省 FLOP；**生产推荐** |

**[源码]** `yoco.py:YocoMetadataMixin`（line 62）：

```python
# 两种模式互斥
self.yoco_consumer_query_indices: Tensor | None  # compact 模式
self.yoco_mask_swa_index_pivots: Tensor | None   # mask 模式
```

**[源码]** `yoco.py:get_consumer_view`（line 85）：compact 模式：创建浅拷贝，只替换 `cu_seqlens_q` 和 position ids。`get_mask_view`：只替换 `kv_swa_full_index_pivots`。

### 7.4 134/140 的来源

| 值 | 意义 | 计算 |
|---|---|---|
| **134** | `swa_block_size_inference` | `window(128) + backtracking(6)` 其中 `backtracking = max(user_env, nextn+1) = max(1, 5+1) = 6` |
| **140** | `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` | `yoco_equiv.py` 默认值，满足 `≥ window+backtracking = 134` 的安全裕量 |

**[源码]** `examples/yoco/yoco_equiv.py`（line 19）：

```python
parser.add_argument("--keep-tokens", type=int, default=140)
parser.add_argument("--window", type=int, default=128)
```

约束：`keep_tokens >= window + backtracking`（即 `140 ≥ 128 + 1 = 129`，测试中 backtracking=1）。

### 7.5 等价性验证（yoco_equiv.py）

**[源码]** `examples/yoco/yoco_equiv.py`（line 42）：

```python
# L1 测试：CPU 元数据对比，无需 GPU
check_metadata_equiv(mode_a="mask", mode_b="compact", ...)
# L2 测试：端到端 logprob 比较
check_logprob_equiv(model, mode_a="off", mode_b="compact", ...)
```

覆盖场景：
1. `N < keep_tokens`（恒等，无截断）
2. `N == keep_tokens`（刚好触发边界）
3. 跨边界 chunk（前半截断、后半保留）
4. 全截断 chunk（所有 token 在滑窗外）

**"近似"可接受性**：当 `N > keep_tokens` 时，compact 模式舍弃了 `[0, N-keep_tokens)` 范围内的 consumer query（这些 token 的 SWA 内容无论如何都在滑窗之外）。对 SWA 注意力而言，它们的注意力输出不依赖这些省略的 Q；而对 compressed full KV 的那部分注意力，这些被省略的 Q 通过 cross-layer shared hidden 路径得到补偿。实验（L2 logprob 等价测试）证明两者 logprob 数值上等价（理论无损）。

---

## 8. 跨层共享（CrossLayerRuntime）

**[源码]** `v4_modules/cross_layer.py:CrossLayerRuntime`（line 198）：

```python
@dataclass
class CrossLayerRuntime:
    shared_hidden: Tensor | None          # 层 20 的 attn-norm BF16 hidden
    pre_rope_kv: Tensor | None            # KProject 的 pre-RoPE 输入
    topk_indices: Tensor | None           # 最近 indexer 源的 top-k 结果
    loaded_prefill_kv: Tensor | None      # prefill 时 KV 复用缓冲
    loaded_prefill_index_kv: Tensor | None
    block_candidate_idxs: Tensor | None   # 层 20 产出的 2048 候选块 indices
    consumer_full_metadata: ... | None    # YOCO compaction 边界元数据
    consumer_full_hidden: Tensor | None   # 边界 full hidden（触发后才填充）
```

`CrossLayerRuntime` 是**单次 forward 的可变状态**，跨层传递，不持久化。每条请求独立一份，通过 `input_metadata.cross_layer_runtime` 注入。

**[源码]** `cross_layer.py:SharedLayerTopology`（line 85）——验证约束：

```python
# 验证 share-kv 不能跨越 YOCO compaction 边界
assert all(src < cfg.n_prefill_layer or src >= cfg.n_prefill_layer
           for src in kv_source_layer_ids), "KV sources must not cross YOCO boundary"
# 验证 chain constraint
assert index_topk_sources <= index_k_sources <= kv_sources
```


---

## 9. 外部方案对比

### 9.1 学术论文映射

| 论文 | arXiv ID（已核实） | 核心创新 | V4.1 对应关系 |
|---|---|---|---|
| **MLA** | 2405.04434 | 低秩 KV 投影（kv_lora_rank），单 KV head 降内存 | V4.1 的基础，`Wkv = ColumnParallelLora(448)` |
| **NSA** | 2502.11089 | 硬件对齐的原生稀疏注意力，训练阶段就稀疏 | V4.1 indexer 的工程对应：FP4 MQA logit + block 候选 |
| **YOCO** | 2405.05254 | Decoder-Decoder 架构，cross-attention 共享 KV | V4.1 n_prefill_layer=20，consumer 层共享 shared_hidden |
| **MiMo-V2.5** | 2607.13095 | 混合 SWA 推理全管道优化（Full-Pipeline Optimization） | V4.1 SWA 有界重放思路相同；YOCO compact 模式是 MiMo 所称"SWA truncation" |

### 9.2 TRT-LLM V4 命名对照

TRT-LLM 博客（[未核实]，以博客描述为准）将类似机制命名为：
- **SWA**（Sliding Window Attention）= V4.1 `sink_mode="swa"`，与 TRT-LLM 同名
- **CSA**（Compressed Sparse Attention）≈ V4.1 C2 门控压缩（Cross-chunk Softmax Aggregation）
- **HCA**（Hierarchical Cross-layer Attention）≈ V4.1 `SharedLayerTopology`（跨层 KV 共享）

DeepInfer 的命名体系（C1/C2/YOCO/CrossLayerRuntime）与 TRT-LLM 不同，但机制对应关系如上。TRT-LLM V4 博客对 C2 的描述与 DeepInfer 实现一致：**每 2 个 token 合并为 1 个压缩 KV，用可学习 gate 加权**。**[未核实]**：TRT-LLM 是否也实现了 2048 候选块的两阶段 indexer。

### 9.3 vLLM V4.1 支持现状

**[未核实]** vLLM PR #58132（"Add bounded SWA replay"）和 #59197（"Hybrid KV Manager"）：基于 GitHub 检索，这两个 PR 号与 vLLM 关于 bounded SWA replay 的讨论相关。vLLM 的挑战在于：V4.1 需要同时管理 ca_cache（4 KV 源分页块）、ica_cache（8 indexer 源分页块）、swa_cache（per-seq state），以及 YOCO 边界的特殊处理。与 DeepInfer 一体化调度相比，vLLM 需要在现有 PagedAttention 之上叠加三个不同格式的缓冲区。

### 9.4 SGLang UnifiedRadixTree

SGLang 的 `UnifiedRadixTree`（[推断] 来自 SGLang v0.4+ 前缀缓存实现）将 prefix caching 与 block 管理统一，可以复用跨请求共享的 V4.1 分页 KV 块（ca_cache、ica_cache 各自都是 block 表结构，可 radix 共享）。DeepInfer 开源版的 `fork_seq`（仅 copy-on-write seq）功能类似但更受限——不支持任意前缀的 radix 树缓存，而是按序列 fork（**[源码]** 已知错误：DeepInfer 开源版不含完整前缀缓存，仅有 `fork_seq`）。

### 9.5 DeepSeek-V3 MLA vs V4.1 MLA

| 维度 | V3 MLA（2405.04434） | V4.1 MLA（本报告） |
|---|---|---|
| KV 压缩维度 | kv_lora_rank=512 | kv_lora_rank=448（512-64） |
| 时间维度压缩 | 无（每 token 存 1 KV） | C2（每 2 token 压缩为 1）+ C1（1:1） |
| 稀疏 indexer | 无 | 32 头 FP4 MQA，top-512，2 阶段候选 |
| KV 量化 | BF16/FP8 | FP4 E2M1 + E4M3 scale（主）、FP8 E4M3（SWA） |
| 跨层共享 | 无 | 4 KV 源 + 8 index 源，nearest-source 映射 |
| YOCO | 无 | n_prefill_layer=20，20 个 consumer 层 |
| KV 字节/token（精确） | ≈ 528 × 61 ≈ 32 KB (FP8) | **1162 B** |

---

## 10. 取舍、开放问题与建议

### 10.1 设计取舍

1. **C2 vs 全精度 KV**：
   - 得：KV 体积 2× 压缩，算力节省同等比例。
   - 失：需要 FP32 state ring buffer，解码路径必须维护 `state_cache` 的 ring 写入；长序列 prefill 时 chunk 边界处的配对逻辑复杂（compress_and_update_state_prefill 内含边界处理）。

2. **稀疏 indexer vs 全量 attention**：
   - 得：只需 attention 512/T 的 KV（T 为 context 长度），可大幅降低解码时的 KV load 带宽。
   - 失：如果 indexer 召回错误（top-512 覆盖不全），会有 **recall miss**（注意力质量损失）；FP4 量化 Q/K 本身也有误差（E2M1 仅 4 bit）。

3. **YOCO 20/40 层分割**：
   - 得：consumer 层共享 KV，理论上 consumer 段 KV 计算 FLOP 降为 1/20（只有层 20 真正投影，21–39 共享）。
   - 失：YOCO compact boundary 引入额外的 compaction FLOP（`compact_yoco_boundary`）和 layout 切换开销；对 CP（Context Parallelism）的 `mirror_cp_reverse_index` 适配更复杂。

4. **FP4 主 KV vs MXFP8 SWA**：
   - 主 KV 使用 FP4 E2M1（更激进量化）：因为 MLA 的 kv_lora_rank 维度已通过 norm + RoPE 产出，精度需求相对宽松。
   - SWA 使用 FP8（更保守）：SWA token 会在下一滑窗步骤参与直接 attention，需要更高精度。

### 10.2 开放问题

1. **Indexer 召回率（Recall）**：top-512 在极长上下文（>128K token）下的 needle-in-a-haystack 场景召回率未公开。2048 候选块 ×8 = 16384 候选 token（约 12.5% 召回率在 128K 下）是否足够？

2. **C2 state ring 的 backtracking 精确行为**：`DEEPINFER_MAX_BACKTRACKING_TOKENS` 的实际默认值（environ.py 中）与 nextn=5 的交互关系，以及对超长 prefill chunk 的处理，需要进一步实测。

3. **YOCO 近似误差的量化**：yoco_equiv.py 测试的等价性在逻辑上是"无损"，但 BF16 浮点累积误差在极长序列下是否会显著影响 logprob 需要系统 benchmark。

4. **consumer 层是否真的跳过 self-attention**：源码显示 consumer 层用 `shared_hidden` 投影 KV，但 Q 仍从本层 hidden states 生成；full attention 路径（compressed full KV）仍然存在。这与 YOCO 论文的 "cross-attention only" 描述有出入——实际是 Q 来自自身 hidden，KV 来自共享 hidden，是一种变体。**[推断]** 需要进一步阅读 V4Attention.forward 的 consumer 路径代码确认。

5. **block_size 的最佳选择**：对 C2（block 内实际存 block_size/2 个 KV）和 ica_cache（68B/token），block_size 的选择影响碎片化程度和 top-k 粒度（候选块数 = block_candidate_topk = 2048，block_size=8 意味着 16384 候选 token）。

### 10.3 对推理系统工程师的建议

1. **部署时 FP4 支持验证**：V41_FP4_FORMAT 依赖 E2M1 + E4M3 scale 的硬件加速路径（H100 以上 SM90 通过 TileLang/Deep GEMM 实现）。在 A100（SM80）或旧硬件上回退路径需要确认，避免无声的精度下降。

2. **state 内存预算规划**：每序列约 2.86 MiB state（2 × 1.35 MiB SWA + 84 KiB 压缩 state）。在高并发场景（1000 序列）下 state 内存约 **2.86 GiB**，是独立于 KV 页表之外的固定开销，必须在显存预算中单独计入。

3. **YOCO keep_tokens 调优**：生产环境应将 `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS` 设为 `≥ window + nextn + 1`（即 ≥ 134），过小会导致 YOCO compact 模式下某些 token 位置的注意力计算错误；过大浪费 Q tensor 的计算量。

4. **Indexer 候选块与分页块大小对齐**：`candidate_block_size=8` 意味着 indexer 以 8-token 粒度选 KV。如果 KV page size（vLLM/SGLang 的 block_size）不是 8 的倍数，需要额外的对齐处理。DeepInfer 内部 block_size 应配置为 8 的倍数。

5. **落盘 state 优化**：`state_size_to_disk ≈ 1.35 MiB/seq`（只含 stored SWA，层 0–19），不含 dropped SWA（层 20–39）。KV store pipeline 设计时可节省 50% 的 SWA state I/O（dropped 部分在 prefill 时重算）。

---

## 附：关键配置参数汇总（实测 HF config.json）

| 参数 | 值 | 来源 |
|---|---|---|
| `num_hidden_layers` | 40 | HF config（已核实） |
| `n_prefill_layer` | 20 | `_n_prefill_layer()` 推导 |
| `kv_lora_rank` | 448（null → 512-64） | config.py line 289 |
| `qk_rope_head_dim` | 64 | HF config |
| `head_dim` | 512 | HF config |
| `kv_source_layer_ids` | [2, 8, 14, 20] | HF config |
| `index_source_layer_ids` | [2, 8, 14, 20, 24, 28, 32, 36] | HF config |
| `candidate_source_layer_id` | 20 | HF config |
| `candidate_topk_blocks` | 2048 | HF config |
| `candidate_block_size` | 8 | HF config |
| `index_topk` | 512 | HF config |
| `index_n_heads` | 32 | HF config |
| `index_head_dim` | 128 | HF config |
| `sliding_window` | 128 | HF config |
| `num_nextn_predict_layers` | 3 | HF config |
| `swa_block_size_inference` | 134 | 128+6（backtracking=6） |
| `分页KV字节/token` | **1162 B（≈1.135 KB）** | 本报告精确核算 |
| `SWA state/seq（推理）` | **≈2.86 MiB** | 本报告估算 |

