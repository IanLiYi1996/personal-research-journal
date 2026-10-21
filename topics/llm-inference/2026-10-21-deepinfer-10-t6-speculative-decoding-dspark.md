# DeepInfer 专题 T6：投机解码 —— DSpark / DFlash / 置信度调度的变长验证

> 撰写时间：2026-10-21 | 代码基准：DeepInfer commit 774ce24
> 面向资深推理系统工程师；区分标记：**[源码]**、**[Q&A]**、**[估算]**、**[推断]**、**[未核实]**

---

## 目录
1. DSpark 算法溯源与论文核实
2. 算法原理：半自回归块起草与 Confidence Head
3. DeepInfer 源码精读：DFlash 模型结构
4. 变长验证调度：MTPVarlen 算法详解与伪代码
5. GPU 侧置信度分配：TileLang 内核
6. Q&A 数据表深度解读：step_time 与边际成本反推
7. 外部方案对比表
8. 与异步调度 / CUDA Graph / 结构化输出 / 确定性的交互与冲突
9. 开放问题与建议

---

## 1. DSpark 算法溯源与论文核实

### 1.1 原始论文状态

**[未核实]** DSpark 尚无可检索到的独立 arXiv 论文。以下是已核实的发现：

- LMSYS 博客《DSpark in SGLang: Speculative Decoding with Confidence-Driven, Variable-Length Verification》（2026-07-06，SGLang Team）**[URL: https://www.lmsys.org/blog/2026-07-06-dspark-sglang]** 明确写道：
  > "The algorithm and its gains are from the DSpark paper."
  > "Thanks to the DSpark authors and to DeepSeek for the algorithm and the models."

  但未给出 arXiv 链接。

- arXiv 全文搜索 `DSpark` 返回结果中：
  - DLite (2610.23787, 2026-10-16) 写道 "outperforms the state-of-the-art speculative decoding method, DSpark"，将其视为 SOTA 基线；
  - ReaDy (2610.20503, 2026-10-14) 写道 "Building on DSpark's parallel backbone and sequential prediction head"。

- DeepInfer 代码中所有配置字段均以 `dspark_` 前缀命名（`dspark_block_size`, `dspark_markov_rank`, `dspark_noise_token_id` 等），说明 DSpark 算法来自 DeepSeek 官方，而非 SGLang 首创。这与 _brief.md 的已知纠错一致。

- **推断**：DSpark 极可能以 DeepSeek V4.1 Flash 内部技术报告形式存在，或计划随 DeepInfer 开源同期/稍后发布，暂未以独立论文形式出现在 arXiv。本文所有算法细节均直接从 DeepInfer 源码提取。

### 1.2 MTP 背景（DeepSeek-V3）

Multi-Token Prediction (MTP) 最早在 DeepSeek-V3 技术报告中提出：训练时为每个主 Transformer 块附加若干 "next-n prediction head"，使模型在自回归训练过程中额外预测未来 n 步 token，以提升表示质量并天然支持推测解码起草。V3 用 1 层 MTP；V4.1 Flash（DFlash）扩展至多层，并加入 Markov 修正头与 Confidence Head，形成 DSpark。

---

## 2. 算法原理：半自回归块起草与 Confidence Head

### 2.1 整体架构

DSpark 在标准 MTP 基础上做了三个关键扩展：

```
 主模型最后 N 层 hidden states (dspark_target_layer_ids)
        ↓ concat → Linear(N*H→H) + RMSNorm (projection)
        ↓
 DFlash Layer 0,1,...,nextn-1  (独立 Transformer 块，不使用 YOCO)
        ↑ 每个位置输入：[golden_token; noise_token × (block_size-1)]
        ↑ 嵌入后展平输入 drafts forward
        ↓
 shared_head: MarkovGPTHead → 各位置 logits
        + Markov 修正: W2(W1(prev_token))  加到 logits
        ↓ 逐步采样 → draft_token_ids[0..nextn-1]
 ConfidenceHead: sigmoid(cat(hidden, markov_emb) @ W / T) → confidence_rates[0..nextn-1]
```

**[源码]** `dflash.py:139`：`self.noise_token_id = cfg.dflash_cfg.dspark_noise_token_id`

**[源码]** `dflash.py:230-239`（`_embed_draft_init_tokens`）：
```python
draft_init_tokens[:, 0] = golden_tokens        # 第 0 位：上轮接受的最后 token
draft_init_tokens[:, 1:] = noise_token_id      # 第 1~nextn-1 位：noise token
return self.embed_tokens(draft_init_tokens.flatten())
```

这是"半自回归"的关键：block 内第 1 步用真实 token，后续用 noise token（不是正确 token，也不是 padding），强迫模型在单步前向中并行预测所有 nextn 位置，而不退化为自回归，以换取极低的起草延迟。

### 2.2 Markov 修正头（MarkovGPTHead / AcceptRatePredictor）

**[源码]** `heads.py:26-43`：
```python
class MarkovGPTHead(GPTHead):
    """Final DFlash head with a first-order Markov correction to logits."""
    def __init__(self, cfg, layer_id=None):
        super().__init__(cfg, layer_id)
        self.markov_rank = cfg.dflash_cfg.dspark_markov_rank  # 低秩嵌入维度
        self.markov_w1 = nn.Embedding(cfg.vocab_size, self.markov_rank)  # token → 低秩嵌入
        self.markov_w2 = ReplicatedLinear(self.markov_rank, cfg.vocab_size, ...)  # 低秩 → vocab logits

class AcceptRatePredictor(nn.Module):
    """DFlash acceptance-rate prediction head."""
    def __init__(self, input_dim):
        self.proj = ReplicatedLinear(input_dim, 1, bias=False, ...)
        # input_dim = hidden_size + markov_rank
```

Markov 修正的作用：当前 draft token 的 logits 基于上一个 draft token（已采样）做一阶条件修正，注入 token-to-token 转移先验，弥补 noise token 输入造成的上下文缺失。

### 2.3 Confidence Head 的训练与推断

**[源码]** `dflash.py:46-79`（`ConfidenceHead`）、`tilelang_ops/dflash.py`（`confidence_head_fused`）：

```python
# 推断公式：
confidence_i = sigmoid( cat(hidden_i, markov_emb_i) @ W / temperature[i] )
# hidden_i: [req_cnt, hidden_size], bf16
# markov_emb_i = W1(draft_token_{i-1}): [req_cnt, markov_rank], bf16
# W: [hidden_size + markov_rank, 1], bf16（ReplicatedLinear，全精度复制到所有 TP rank）
# temperature[i]: 标量，可为 per-step 不同值（由 DFLASH_CONFIDENCE_HEAD_TEMPERATURE 控制）
```

TileLang kernel（`confidence_head_fused_tl`）将 cat + gemv + sigmoid + temperature 缩放融合为单个内核，直接输出 fp32 结果，避免中间 tensor。

**[推断]** Confidence Head 在训练时对应接受/拒绝标签（二分类），使用 on-policy drafting 数据：草稿分布 q 下的实际接受率作为监督信号。PIPO（2605.27255）的论文也采用类似思路——confidence head + OPD 训练。

**两步回传（two-step back）问题**：**[源码]** `mtp_varlen.hpp:162-168` 注释明确：当前步调度时（第 N 步），上步（N-1）的 confidence 还在 GPU 上尚未回传，因此：
- `best_bsz` 计算使用 N-2 轮的 `prev_accept_rates`
- 每请求 `seqlen_q` 分配使用 N-1 轮的 `accept_rates`（需等 ZOS D2H 完成）

这就是"两步回传"的本质：每轮调度依赖的置信度信号有 1 轮的固有延迟。

---

## 3. DeepInfer 源码精读：DFlash 模型结构

### 3.1 关键配置参数

| 参数 | 来源 | 含义 |
|------|------|------|
| `dspark_block_size` | `dflash_cfg` | 半自回归块大小（=nextn） |
| `dspark_noise_token_id` | `dflash_cfg` | noise padding token id |
| `dspark_markov_rank` | `dflash_cfg` | Markov 低秩修正维度 |
| `dspark_target_layer_ids` | `dflash_cfg` | 主模型特征层列表 |
| `dspark_target_attn_input` | `dflash_cfg` | 是否用 attn 输入作特征 |
| `dspark_n_routed_experts` | `dflash_cfg` | draft MoE 专家数 |
| `DEEPINFER_USE_NEXTN_DECODE` | env | 最大 draft 数，默认 5 |
| `DEEPINFER_ENABLE_MTP_VARLEN` | env | 是否启用变长验证（bool） |
| `DEEPINFER_MAX_BACKTRACKING_TOKENS` | env | max(input, nextn+1)，SWA ring 大小附加项 |
| `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` | env | 标量或 nextn 元 tuple，控制置信度分布锐度 |

**[源码]** `environ.py:129-141`：

```python
DEEPINFER_USE_NEXTN_DECODE = EnvVar.int(default=5)
DEEPINFER_ENABLE_MTP_VARLEN = EnvVar.bool()
DEEPINFER_MAX_BACKTRACKING_TOKENS = EnvVar(lambda v: max(_int(v), envs.DEEPINFER_USE_NEXTN_DECODE + 1))
# DFLASH_CONFIDENCE_HEAD_TEMPERATURE: 标量 → 所有 nextn 步用同一温度
#   逗号分隔 tuple → 每步独立温度（必须与 nextn 等长）
```

`DEEPINFER_MAX_BACKTRACKING_TOKENS` 至少为 nextn+1，是因为投机解码最多回退 nextn 步再重采样 1 token，需要足够的 SWA ring 缓冲区。**[源码]** `cache_engine.py:93,133`：SWA ring size = `window_size + DEEPINFER_MAX_BACKTRACKING_TOKENS`。

### 3.2 main cache 路径

**[源码]** `dflash.py:198-228`（`update_main_cache`）：
在主模型 forward 过程中，从 `dspark_target_layer_ids` 指定的层提取 hidden states，经过 `projection + projection_norm` 后，通过 `main_kv_proj` 同时为所有 MTP 层计算 key，写入 SWA KV cache 的对应位置。这是"预写 main cache"步骤，在 CUDA graph 中与主模型一起 replay。

关键：被拒绝的 draft 位置的 main key 会在随后的 `forward_drafts` 中被 draft key 覆盖，无需按接受结果屏蔽（因为覆盖在读取之前发生）。

### 3.3 DFlashCudaGraphRunner 的设计

**[源码]** `dflash_cuda_graph_runner.py`：

- DFlash 拥有**独立于主模型**的 CUDA graph，捕获 `run_drafts`（包含 cp_broadcast + build_dflash_input + forward_drafts + sample_drafts + pool write）。
- batch_sizes 从 `DEEPINFER_DFLASH_CUDAGRAPH_BATCH_SIZE_INTERVAL` 生成，与主模型 graph 的 batch sizes 可不同。
- replay 时：`copy_dflash_decode_input` 用真实输入覆写 graph 的 pinned buffer，然后 `graph.replay()`，最后 `narrow(req_cnt)` 剪掉 padding。
- **设计亮点**：draft model graph 和主模型 graph 解耦，可独立调整 batch size tier。

---

## 4. 变长验证调度：MTPVarlen 算法详解与伪代码

### 4.1 整体架构

MTPVarlen 在调度器（`scheduler.hpp:652,664`）中被调用：
- `_mtp_varlen.schedule(scheduled, seqlens_q)` — 输出每请求 seqlen_q
- `_mtp_varlen.update_runtime(num_seqs, graph_size, forward_ms, sampler_ms)` — 更新耗时估计

### 4.2 运行时估计模块

**SamplerRuntimeEstimator**（`mtp_varlen.hpp:20-42`）：
- 滑动窗口大小 9，至少 5 个样本后开始给出估计
- 取中位数（对称堆实现），对同一 `num_seqs` 聚合，有回退到全局样本

**ForwardRuntimeEstimator**（`mtp_varlen.hpp:44-131`）：
- 窗口 5000 步，每个 graph size 至少 10 个样本
- 两层中位数：先按 graph_size 分组各取中位数（`_samples_by_graph`），再对各组中位数取中位数（`_graph_medians`）
- 输入是 `(forward_ms - baseline_ms) / graph_size`（归一化残差），得到"每 token 的额外边际耗时"
- 斜率取 max(0, slope)，防止负斜率外推产生非正耗时

### 4.3 best_bsz 选择算法（伪代码）

```
输入:
  scheduled: 当前批次序列列表，每个序列有 nd 个 draft token 和 accept_rates
  _decode_perf_curve: [(graph_size_0, baseline_ms_0), ..., (graph_size_K, baseline_ms_K)]
  slope_ms: ForwardRuntimeEstimator.get()  (边际每 token 耗时)
  sampler_ms: SamplerRuntimeEstimator.get() (采样器基础耗时)

步骤:
1. 构建 acc_queue_for_bsz:
   对每个 seq i，将其 nd 个"累乘接受率"加入队列（prev_accept_rates，即 N-2 轮）
   若无 rates 且有 draft → pad 1.0；若无 rates 且无 draft → pad 0.0
2. 按接受率从高到低排序 acc_queue_for_bsz
3. 初始化: cur_expected = num_seqs, cur_bsz = num_seqs
4. 对每个候选 total_q（即 cur_bsz），找到满足 cur_bsz ≤ graph_size 的最小 graph tier：
   time_ms = baseline_ms + slope_ms * graph_size + sampler_ms
   throughput = cur_expected / time_ms
   记录吞吐量最大时的 best_bsz
5. 遍历 acc_queue_for_bsz（每步 cur_expected += rate, cur_bsz += 1），重复步骤 4
6. 输出 best_bsz

算法复杂度:
  - 构建队列: O(num_seqs × nextn)
  - 排序: O(num_seqs × nextn × log(num_seqs × nextn))
  - 遍历候选: O(num_seqs × nextn × K)
  - 总体: O(N·D·log(N·D) + N·D·K)，其中 N=num_seqs, D=nextn≤5, K≤几十个 graph tier
```

**[源码]** `mtp_varlen.hpp:150-267`（`MTPVarlen::schedule`）关键片段：

```cpp
// 构建两个队列
acc_queue_for_bsz     // 用 N-2 轮 accept_rates，计算最优 best_bsz
acc_queue_for_seqs_q  // 用 N-1 轮 accept_rates，按 best_bsz 分配各请求 seqlen_q

// 遍历枚举：golden token 全部参加（+num_seqs）；之后逐步加入接受率最高的 draft 位置
float cur_expected = num_seqs;
int   cur_bsz      = num_seqs;
for (auto &acc : acc_queue_for_bsz) {
    cur_expected += acc.first;   // += 该位置的接受率
    cur_bsz      += 1;
    // 找到 cur_bsz 对应的 graph tier，计算吞吐量
    double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
    if (cur_expected / time_ms > best_throughput) {
        best_throughput = ...; best_bsz = cur_bsz;
    }
}

// 分配每请求 seqlen_q：top-(best_bsz - num_seqs) 的位置各加 1
seqlens_q.assign(num_seqs, 1);
for (int i = 0; i < best_bsz - num_seqs; ++i)
    seqlens_q[acc_queue_for_seqs_q[i].second] += 1;
```

### 4.4 双时序延迟处理

MTPVarlen 通过 `schedule_cnt` / `process_cnt` 两个计数器判断每个请求当前可用的接受率"属于哪一轮"：

| 状态 | 含义 | bsz 用 | seqlen_q 用 |
|------|------|--------|------------|
| `process_cnt == schedule_cnt` | forward+postprocess 配对，accept_rates=N-1轮 | `prev_accept_rates`（N-2轮） | `accept_rates`（N-1轮） |
| `schedule_cnt == process_cnt + 1` | ZOS 异步，accept_rates=N-2轮 | `accept_rates`（N-2轮） | `accept_rates`（N-2轮） |

**[源码]** `mtp_varlen.hpp:195-208`


## 5. GPU 侧置信度分配：TileLang 内核

### 5.1 `compute_seqlens_q_by_confidence_rates`

**[源码]** `tilelang_ops/schedule_mtp_varlen.py:11-98`

在 ZOS 路径下，CPU 侧 `best_bsz` 已知（由上步调度结果决定），GPU 上需要根据最新置信度（上一步 DFlash graph replay 产生）重新分配各请求 seqlen_q。

```python
def compute_seqlens_q_by_confidence_rates(
    total_q: int,       # CPU 侧确定的总 token 预算
    remap: Tensor,      # [num_seqs], int32：prev_batch_idx 或 -1（新请求）
    prev_confidence_rates: Tensor,  # [n_prev_seq, nextn], fp32
    seqlens_q: Tensor,  # [num_seqs], int32（输出）
):
    # Phase 1: 收集置信度 + 累乘
    # remap[i] == -1 → 新请求，置信度全 0（不分配 draft 位置）
    # remap[i] >= 0 → 从 prev_batch 对应行提取置信度，cumprod(dim=1)
    cum_rates = cumprod(prev_confidence_rates[remap], dim=1)  # [num_seqs, nextn]

    # Phase 2: 全局排名，分配 top-(total_q - num_seqs) 个 (seq, pos) 位置
    flat_rates = cum_rates.reshape(-1)          # [num_seqs * nextn]
    sorted_indices = argsort(flat_rates, descending=True, stable=True)
    # 排名 < extra_budget 的 (seq, pos) 对：该 seq 的 seqlen_q += 1
    seqlens_q = 1 + wins_per_seq               # wins = top-k 命中数
```

**算法特点**：
- 累乘而非独立置信度：`cum_rates[i, j] = prod(conf[i, 0..j])`，反映"前 j+1 个 draft token 全部被接受的概率"，与链式规则一致。
- 稳定排序保持确定性：同分时按原始 flat index 小的优先（等价于同 seq 优先低位置，跨 seq 优先序号小的）。
- **[源码]** 注释（`prepare_tensor_input_zos.py:93`）指出 Ascend (NPU) 版本有额外约束：`num_seqs * nextn * 4 ≤ 32KB`（UB 内存限制）。

### 5.2 数据流与并发关系

```
Step N-1:  DFlash graph replay → MTPInfoPool 写入 confidence_rates（GPU）
Step N:    _correct_seqlens_qk:
             1. remap + prev_confidence_rates → compute_seqlens_q_by_confidence_rates (GPU)
             2. seqlens_q D2H (non_blocking, event 同步) 
             3. CPU seqlens_q 更新给 metadata_zos
           后续: main model forward 使用修正后的 seqlens_q
```

**[源码]** `worker.py:218-240`：D2H 同步通过 `correction.d2h_evt.synchronize()` 完成，在 `execute_sample` 之前生效。


## 6. Q&A 数据表深度解读：step_time 与边际成本反推

Q&A 中的吞吐量数据（EP64/DP64 benchmark，offline）：

| conc (seqs/rank) | fixed-5 avg_veri | dyn avg_veri | fixed-5 accept_len | dyn accept_len | fixed-5 tput | dyn tput | gain |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 5 | 4.44 | 3.32 | 3.24 | 4891 | 4816 | −1.5% |
| 32 | 5 | 3.65 | 3.25 | 3.17 | 8964 | 9111 | +1.6% |
| 48 | 5 | 3.94 | 3.30 | 3.26 | 12032 | 12467 | +3.6% |
| 60 | 5 | 3.58 | 3.30 | 3.21 | 13730 | 14466 | +5.4% |
| 72 | 5 | 3.24 | 3.29 | 3.12 | 14962 | 15961 | +6.7% |
| 96 | 5 | 3.24 | 3.30 | 3.15 | 16817 | 18660 | +11.0% |

### 6.1 step_time 反推

**[估算]** 利用 `throughput(tok/s/rank) = conc × (1 + accept_len) / step_time_ms × 1000`，反推各场景 step_time（ms）：

```
step_time = conc × (1 + accept_len) / throughput × 1000

conc=16:  fixed-5: 16 × 4.32 / 4891 × 1000 = 14.13 ms
          dynamic:  16 × 4.24 / 4816 × 1000 = 14.09 ms
          step_time 节省: 0.04 ms（几乎相同）

conc=96:  fixed-5: 96 × 4.30 / 16817 × 1000 = 24.55 ms
          dynamic:  96 × 4.15 / 18660 × 1000 = 21.35 ms
          step_time 节省: 3.20 ms（节省 13.0%）
```

### 6.2 验证 token 边际成本估算

**[估算]** 每减少 1 个验证 token 节省的 step_time：

```
conc=96:  Δstep_time = 3.20 ms; Δavg_veri = 5.00 - 3.24 = 1.76 tokens
          边际成本 = 3.20 / 1.76 ≈ 1.82 ms/token

conc=16:  Δstep_time ≈ 0.04 ms; Δavg_veri = 5.00 - 4.44 = 0.56 tokens
          边际成本 = 0.04 / 0.56 ≈ 0.07 ms/token
```

### 6.3 低并发（conc=16）为何为负

**[估算+推断]** 四重因素：

1. **step_time 对验证数量不敏感**：conc=16 时 batch 小，attention 是 memory-bound 的低效区，减少 0.56 个验证 token 节省 0.07 ms/token 仅 0.04 ms，几乎被噪声淹没。
2. **接受率下降**：dyn accepted_len=3.24 < fixed 3.32，每步少产出 0.08 × 16 = 1.28 tokens，折算 -1.28/14.1ms = -90.8 tok/s。
3. **置信度调度开销**：GPU kernel + D2H 同步在极短 step_time（14 ms）下占比相对显著。
4. **两步延迟精度损失**：新进入批次的请求（remap=-1）置信度设 0，获得 seqlen_q=1，可能低估实际接受率。

### 6.4 高并发（conc=96）为何 +11%

**[估算+推断]**：

1. **step_time 高度敏感**：conc=96 时 batch 大，attention 进入 bandwidth-bound 区域，减少 1.76 个验证 token 节省 3.20 ms step_time（1.82 ms/token 边际成本），远超接受率小幅下降的代价。
2. **批次稳定，置信度精度高**：大并发下 remap 命中率更高（请求更长寿），两步延迟的冷启动影响占比小。
3. **验证计算量与 KV 读取正比于 batch × seqlen_q**：减少 35% 的验证 token 直接减少约 35% 的 attention KV 带宽消耗。

### 6.5 盈亏平衡点估算

**[估算]**：从上表可见，约在 conc ≈ 25-30 seqs/rank 时进入正收益区（对应 EP64 全系统约 1600-1920 seqs）。


## 7. 外部方案对比表

| 方案 | 论文/来源 | 起草结构 | 调度策略 | Confidence Head | CUDA Graph | 变长验证 |
|------|-----------|----------|----------|-----------------|------------|----------|
| **DSpark (DeepInfer)** | DeepSeek [未核实独立论文] | 半自回归块 + noise token + Markov head | 成本表驱动，置信度排名选 top-k | ✓ 独立 Confidence Head，TileLang fused | ✓ 独立 DFlash CUDA Graph | ✓ CPU+GPU 联合，ZOS 修正 |
| **MTP (V3)** | DeepSeek-V3 tech report | 多层 next-n head，无 noise token | 固定 nextn | ✗ | ✓（与主模型共用） | ✗（固定长度） |
| **EAGLE-3** | arXiv 2503.01840 (2025-03) | 多层特征融合，直接 token 预测（训练期测试） | 固定长度为主 | ✗ | ✓（SGLang） | 部分（SpecForge） |
| **SpecForge** | arXiv 2603.18567 (2026-03) | 基于 EAGLE-3 的开源训练框架 | — | ✗ | ✓（生产集成） | ✗ |
| **MiMo-V2-Flash** | arXiv 2601.02780 (2026-01) | MTP 3 层，SWA 混合注意力 | 固定长度 | ✗ | — | ✗（最高 3.6 accept_len, 2.6x） |
| **PIPO** | arXiv 2605.27255 (2026-05) | latent compressor + MTP head 镜像 | 置信度 confidence head | ✓（OPD 训练） | — | ✓（去除 verifier pass） |
| **DLite** | arXiv 2610.23787 (2026-10) | 无历史上下文，重用 target hidden state | 无 | ✗ | — | — |
| **SGLang DSpark** | LMSYS 博客 2026-07-06 | 同 DSpark | cap-accept / cap-total / 全动态三种模式 | ✓（采用 DSpark 论文方案） | ✓ ragged verify | ✓ ZOS 集成 |
| **vLLM** | [未核实 PR 状态] | EAGLE/MTP | adaptive block verification（进行中）| [未核实] | ✓ | dynamic SD（WIP）|
| **TRT-LLM MTP** | [未核实] | MTP/EAGLE | 固定 | [未核实] | ✓ | ✗（已知 WIP DSpark 支持）|

### 7.1 LMSYS 博客中 SGLang 的 Verify Modes

**[URL]** 博客提到三种模式（`verify modes`）：
- **cap-accept**：给定 accept 长度上限，在约束内最大化吞吐
- **cap-total**：给定 total_q 上限，按置信度分配给各请求
- **全动态（fully dynamic）**：同 DeepInfer 的 MTPVarlen，用成本表选 best_bsz

SGLang 的 **ragged verify under CUDA graph**：将所有请求的验证 token 前压打包（而非固定宽度 N×W 的填充），使 total_q 对齐到最近的捕获 tier。与 DeepInfer 的 DFlash CUDA Graph 设计思路一致。

SGLang 的 **ZOS（zero-overhead scheduling）**：与 DeepInfer 的异步调度（overlapping forward + post-process）原理相同，DSpark 仅需加入两步回传的置信度中继（几乎无额外特殊处理）。

**SPS cost table**（SGLang）：等同于 DeepInfer 的 `ForwardRuntimeEstimator + SamplerRuntimeEstimator`，都是离线 profile + 在线拟合斜率。SGLang 用仿射拟合（截距 + 斜率），DeepInfer 用两层中位数更稳健。


## 8. 与异步调度 / CUDA Graph / 结构化输出 / 确定性的交互与冲突

### 8.1 与异步调度（ZOS）的交互

**[源码]** `worker.py:218-240` + `zos_status.py`：

MTPVarlen 与 ZOS 的集成是最精细的部分。整个流水线：

```
Step N-1 GPU:  forward → sample → (write confidence_rates 到 pinned GPU buffer)
Step N   CPU:  schedule → best_bsz 已知
Step N   GPU:  compute_seqlens_q_by_confidence_rates  ← 使用 step N-1 的 confidence_rates
Step N   D2H:  seqlens_q → CPU (event 同步)
Step N   CPU:  scheduler_outputs.update_metadata_for_sampling_zos(seqlens_q)
Step N   GPU:  main_model_forward(使用修正后的 seqlens_q)
```

**冲突点**：若 step N-1 的 post-process 还没完成（ZOS 状态 `process_cnt < schedule_cnt`），则本步的 `_correct_seqlens_qk` 里使用的仍是 N-2 轮置信度，接受率估计滞后一步。代码注释指出这是已知限制，通过两队列机制（bsz 用 N-2，seqlen_q 也退化为 N-2）处理。

**[源码]** `worker.py:132`：ZOS + ENABLE_MTP_VARLEN 同时启用时，有额外的 pinned buffer 申请（`seqlens_q_cpu`）用于 D2H。

### 8.2 与 CUDA Graph 的交互

两个 CUDA Graph 分离：
- **主模型 graph**：捕获 `update_main_cache + forward`，seqlen_q 通过 copy-into-pinned 注入
- **DFlash graph**：捕获 `run_drafts`，独立输入缓冲，独立 batch_sizes

**挑战**：ragged verify 打破了固定形状假设。DeepInfer 的解法是 DFlash graph 本身仍使用固定 batch_sizes（按 num_seqs 而非 total_q 选取），total_q 的变化体现为每请求 seqlen_q 不同，通过 `build_dflash_input` 根据 `seqlen_q` 动态构造元数据（位置、KV 映射）。

**[推断]** 这意味着 CUDA graph 的静态形状假设只约束 num_seqs 维度，而 total_q 维度在 graph 内部动态处理，可能通过预分配最大 total_q buffer 实现。

### 8.3 与结构化输出（SO）的交互与冲突

**[源码]** `sampler.py:301-408`（`_get_so_mask`）：

SO 与 MTP 的集成有"前向展开"逻辑：
1. 对每个请求，matcher 先接受 `last_accepted_token + last_output_token`（本步实际接受的 token）
2. 然后逐步 `accept_token(draft_token_0), accept_token(draft_token_1), ...`
3. 每接受一步，若受约束则生成该 draft position 的 vocab bitmask
4. 最后 `matcher.rollback(num_accepted_steps)` 撤销所有预测接受（因为 draft 尚未经主模型验证）

**冲突/风险**：
- **Rollback 实现依赖**：SO matcher 必须支持正确的 rollback（最多 nextn 步）。`structure_output/matcher.py:20` 中 `MAX_ROLLBACK_TOKENS = nextn + 2`。
- **SO + 全拒绝**：若 draft 全被拒绝且重采样 token 不满足 SO 约束，`so_status` 设为 -1（abort），请求被杀死——这是投机采样与 SO 无法 100% 兼容的固有张力。
- **SO bitmask 的 step 数量**：预分配 `max_MTP_value + 1` 个 bitmask（step 0 = 主模型 fresh token，step 1..nextn = draft positions），Bitmask 在另一个 CUDA stream 异步 H2D。

**[源码]** `sampler.py:326-327`：`bitmasks_cpu` 初始化为 `max_MTP_value + 1` 个 vocab_mask。

### 8.4 与确定性的约束

**[源码]** `tilelang_ops/dflash.py:test_confidence_head` 中断言 10 次调用结果一致（`torch.equal`）。TileLang 内核设计上保证确定性。

`compute_seqlens_q_by_confidence_rates_ref` 使用 `stable=True` argsort，保证同分时按 flat index 排序，输出确定。

**潜在问题**：当 `DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 的温度过低时，confidence 输出趋向饱和（0/1），argsort 中同分情况增多，依赖 stable sort 的确定性。当温度过高时，置信度趋向均匀分布，变长调度退化为固定长度（所有位置机会均等）。


## 9. 开放问题与建议

### 9.1 已知开放问题

**1. 最优温度的确定**
`DEEPINFER_DFLASH_CONFIDENCE_HEAD_TEMPERATURE` 当前需人工设置（标量或 per-step tuple）。如何根据模型任务分布自动校准温度，使置信度分布更接近真实接受概率？

**2. Noise token 的最优化**
当前 noise token 是固定的单一 token ID。是否可以用随机 token、均匀随机或 draft distribution 采样代替？ReaDy（2610.20503）的实验表明，更丰富的上下文（HLA recurrent state）可以弥补 noise token 带来的上下文缺失，但 noise token 策略本身尚无系统研究。

**3. 两步延迟下的 best_bsz 精度**
当前 best_bsz 使用 N-2 轮置信度（因 ZOS 延迟），对快速变化的批次（短请求，高完成率）意味着置信度信号滞后两步，可能选出次优 bsz。是否可以在 CPU 侧用轻量模型预测置信度衰减？

**4. 多温度 per-step 的训练对齐**
per-step temperature（tuple 形式）意味着 confidence head 对不同起草步数的接受率进行独立校准。是否有对应的训练方案保证每步温度与真实接受率分布对齐？

**5. 非 CUDA 图路径的性能**
`use_cuda_graph=False` 时，`profile_decode_perf()` 返回空列表，`MTPVarlen::schedule` 退化为固定 nextn。对于动态 batch 大小或 Ascend 平台（CUDA graph 支持受限），变长调度失效。

**6. vLLM / TRT-LLM DSpark 进展**
vLLM 的 adaptive block verification / dynamic SD（引自博客 sgl-project/sglang#30344 提到的 roadmap）和 TRT-LLM DSpark WIP 的具体进展**[未核实]**，需通过 gh PR 追踪。

### 9.2 对从业者的建议

1. **阈值配置**：在生产上线时，先以 `DEEPINFER_ENABLE_MTP_VARLEN=0` 建立 fixed-5 baseline，再逐步在中高并发下启用变长调度，监控 `batch_confidence_error`（sampler output 中会回传该指标）以判断置信度校准质量。

2. **成本表 profile**：`profile_decode_perf()` 在 warm_up_model 阶段自动运行（每 batch_size 5 次测量、trim 各 1 个异常值），但首次运行会增加启动时间。建议通过 `DecodePerfCache` 机制持久化结果（代码中有 `load_decode_perf_cache()` 路径），加速重启。

3. **SO + MTP 配合**：若使用结构化输出，确保 SO matcher 正确实现 `rollback(nextn + 2)` 步以上，否则若 draft token 全被拒绝、重采样后 SO matcher 状态不一致会导致请求 abort（so_status=-1）。

4. **MAX_BACKTRACKING_TOKENS**：该值不仅影响 SWA ring 大小（内存开销），也决定 KV cache 命中策略中 "最大回退长度"（Q&A 中提到的 6 tokens 余量）。若增大 nextn，必须同时调整 `DEEPINFER_MAX_BACKTRACKING_TOKENS`，否则会触发 assert。

5. **per-step temperature 调参**：通常第 1 个 draft token 接受率最高，后续递减。可以给靠后位置更高的温度（更软的置信度），使其更倾向"放弃"，减少低概率位置的浪费验证。

### 9.3 仍未核实的点

- DSpark 独立论文（arXiv ID 或发表时间），尤其是训练细节（loss 设计、OPD 比例、数据集规模）
- vLLM adaptive block verification PR 的合并状态及对应 DSpark 支持程度
- TRT-LLM MTP DSpark WIP 的实现细节
- DeepSeek 内部版本 draft model 相较开源版本的具体差异（Q&A 提到"内部版本稍新但差异小"）

---

## 附录 A：关键路径数据流图

```
[主模型 forward: CUDA Graph 1]
  target_layer_ids hidden → projection → main_kv_proj → 写 SWA cache (所有 MTP 层)

[DFlash forward: CUDA Graph 2]
  golden_token → embed → [noise, noise, ...] → embed → cat
       ↓  forward_drafts (nextn 个 DFlash Transformer 层)
  hidden_states_before_norm
       ↓  post_norm
  hidden_states  →  lm_head  →  logits_0..nextn-1
  Markov: logits += W2(W1(draft_token_{i-1}))
  draft_token_i ~ sample(logits_i)
  confidence_i = sigmoid(cat(hidden_i, W1(draft_token_{i-1})) @ W_conf / T_i)
  → MTPInfoPool.store(token_ids, confidence_rates, logprobs)

[Sampler: 与主模型共用 stream]
  prepare_verification_metadata(ftp, seqlen_q, total_q, nextn)
    → row_to_req, local_step, is_verify
  process_sampling(logits, …, last_nextn_logprobs, draft_token_ids)
    → accept_result_verify
  get_accept_result(…) → sampled_tokens, last_valid_offset
  finalize_verification(…) → verified_tokens, batch_confidence_error

[scheduler.hpp: CPU]
  process_MTPInfo → seq.mtp.draft_tokens, accept_rates (cumprod), prev_accept_rates
  MTPVarlen::schedule → seqlens_q (CPU side)
  [若 ZOS] compute_seqlens_q_by_confidence_rates (GPU) → seqlens_q D2H
```

## 附录 B：核实的 arXiv 论文列表

| arXiv ID | 标题 | 时间 | 关联 |
|----------|------|------|------|
| 2503.01840 | EAGLE-3: Scaling up Inference Acceleration... | 2025-03 | 特征级起草对比基线 |
| 2603.18567 | SpecForge: A Flexible and Efficient Open-Source... | 2026-03 | EAGLE-3 开源训练框架 |
| 2601.02780 | MiMo-V2-Flash Technical Report | 2026-01 | MTP 3 层，MoE，2.6x 加速 |
| 2605.27255 | PIPO: Pair-In, Pair-Out: Latent Multi-Token Prediction | 2026-05 | 类 DSpark confidence head + OPD |
| 2610.23787 | DLite: History-free Speculative Decoding | 2026-10 | DSpark 改进，agentic 场景 |
| 2610.20503 | ReaDy: Recurrent States as Dynamic Context | 2026-10 | 基于 DSpark backbone 改进 |

