# DeepInfer 专题 T8：确定性与批次不变推理、采样器、结构化输出

> 面向资深推理系统工程师。技术深度优先，所有结论给出处。  
> 标记：**[源码]** 读代码确认 · **[Q&A]** 官方文档 · **[估算]** 推算（含算式）· **[推断]** 合理推测 · **[未核实]**

---

## 1. 根因：为什么推理结果随 batch 变化

### 1.1 浮点非结合性 × kernel 分工变化

浮点加法不满足结合律：`(a + b) + c ≠ a + (b + c)`（舍入误差逐轮累积）。这本身不足以造成不确定性——同一份数据、同一种切分，每次都给出相同的位精确结果。**真正的不确定来源是：batch 组成不同时，kernel 的任务切分方式随之改变，规约顺序随之改变。**

Thinking Machines *Defeating Nondeterminism in LLM Inference*（2025-09，[https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)）系统梳理了各来源：

| 来源 | 机制 |
|---|---|
| **Split-K / Split-KV** | GEMM 沿 K 维拆分；K 维总长非 tile 整数倍时 tile 数随 batch 变化，partial sum 的规约顺序改变 |
| **原子加（Atomic Add）** | Sparse attention、softmax 跨 CTA 规约使用原子加，多 CTA 完成顺序不固定 |
| **规约顺序** | RMSNorm、softmax 等 reduction kernel 的 warp/block 分工随 batch token 数 / 序列长度变化 |
| **CUDA graph 尺寸选择** | 不同 graph 对应不同 grid 形状 → 不同分工 |
| **EP 多次规约** | Expert Parallel 下，dispatch/combine 的规约在不同 rank 间 `all-reduce`；EPLB 动态调整专家分布时，单次 step 可能触发多轮 reduce，每轮结果作为下一轮输入，误差叠加 |

该文关键论断：**"concurrency + floating-point 假说不完整"** — 同一组数据跑同一份 GEMM 每次结果完全一致；不确定性来自 *who computes what together*，而非并发本身。

### 1.2 Vosti 的形式化视角

arXiv 2609.38981 *Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference*：

> 核心定义（原文 abstract 摘录）：*"under a fixed model and deployment configuration, requests with the same prompt and initial sampler state produce bitwise-identical logits at corresponding output positions across executions"*

Vosti 的测试发现：**vLLM 的 batch-invariant mode 和 SGLang 的 deterministic mode 在某些 execution variation 下仍产生不同 logits**，说明两者都未达到该形式化规范。Vosti 采用：
- Verus 归纳证明：调度器 + paged prefix-sharing KV-cache 保持 logit 不变性
- Triton 分析器：跨 batch/query-length/分页 KV 的逐位相等核证明

---

## 2. DeepInfer 的确定性实现

### 2.1 全局入口：runtime_config.py

```python
# deepinfer/worker/runtime_config.py:62-69
if PLATFORM.is_npu:
    # TODO(NPU): 待完整支持 batch invariant 后移除
    assert not envs.DEEPINFER_BATCH_INVARIANT
if PLATFORM.is_gpu:
    deep_gemm.set_pdl(True)
    tile_kernels.set_pdl(True)
if envs.DEEPINFER_BATCH_INVARIANT or PLATFORM.is_npu:  # TODO(NPU): 防止 CI 数值不稳定
    deep_gemm.use_deterministic_algorithms(True)
    tile_kernels.use_deterministic_algorithms(True)
```

**[源码]** 两个关键点：

1. **NPU 不支持 batch invariant**（line 63 硬 assert）。注释 "待完整支持后移除" 表明这是已知 limitation，而非永久设计。
2. NPU 即使不开 `DEEPINFER_BATCH_INVARIANT` 也会走 `use_deterministic_algorithms(True)`——目的是"防止 CI 数值不稳定"，属于 NPU 数值稳定性的权宜之计，与 GPU 批次不变的语义不同。

入口：`DEEPINFER_BATCH_INVARIANT`（`deepinfer/environ.py:197`），默认 `False`；`examples/offline_inference.py:43` 为离线推理 default 开启它。

### 2.2 注意力后端：attention_backend.py + attn.py

**FlashMLA decode 分支**（`attention_backend.py:141`）**[源码]**：

```python
output, lse = flash_mla_with_kvcache(
    ...
    enable_batch_invariant=batch_invariant,   # line 141
)
```

`batch_invariant` 的值来自 `attn.py:666, 1002`：

```python
batch_invariant=envs.DEEPINFER_BATCH_INVARIANT
```

该参数透传给 flash_mla（外部库），由其内部切换到确定性规约路径（split-KV 策略不随 token 数变化）。

**prefill 路径（attn.py:89）**：`use_fused_norm_rope_attn_rope_cast` 在 `DEEPINFER_BATCH_INVARIANT=True` 且 `engine_type == PROMPT` 且非 sparse 非 unpacked KV 的情况下被禁用。该 fused kernel 内部有依赖 batch 维度的优化路径，禁用后回落到分步执行以确保确定性。

### 2.3 EP 规约：userbuffer.py

```python
# deepinfer/model_executor/parallel_utils/userbuffer.py:79
allow_multiple_reduction=not (
    envs.DEEPINFER_BATCH_INVARIANT
    and envs.DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL > 0
),
```

**[源码]** 逻辑：当 `BATCH_INVARIANT=1` **且** EPLB（Expert Load Balancing）的专家重排间隔 > 0 时，禁用 EPBuffer 的多次规约优化。原因：多次规约让每步可以做增量 reduce（前一步结果作为后一步输入），而这会因批次组成变化导致不同步序，破坏确定性。关掉后每步的 all-reduce 是独立的，但丧失了 EPLB 带来的带宽节省。

### 2.4 为什么 ENABLE_MTP_VARLEN=0 才能完全确定性

变长 MTP（`DEEPINFER_ENABLE_MTP_VARLEN`）的工作方式（`worker.py:218, 230`）：

```python
if envs.DEEPINFER_ENABLE_MTP_VARLEN:
    compute_seqlens_q_by_confidence_rates(
        total_q, remap[:n_seq], 
        prev.sampler_output.confidence_rates_gpu,
        seqlens_q[:n_seq]
    )
```

每个 decode step 的 `seqlens_q`（投机验证步数）由**上一步采样结果的置信度**决定，而置信度又来自 MTP 的 logit。因此：

```
step N logits → confidence_rates_N → seqlens_q_{N+1} → CUDA graph tile → 规约顺序_{N+1}
```

这就是"本步的计算图依赖上一步输出"的因果链。如果 step N 由于批次组成不同而得到略微不同的 confidence_rates，step N+1 的 seqlens_q 就不同，进而 CUDA graph 的分发路径不同，规约顺序不同。所以 `ENABLE_MTP_VARLEN=1` 与逐位确定性根本不兼容。

**[推断]** 这意味着开 MTP+VARLEN 的生产部署（对吞吐率有利）与 RL on-policy 校验（需要确定性）天然冲突，用户必须在两者间取舍。


---

## 3. 确定性采样器：random_sampling + fused_sample

### 3.1 RNG 种子体系

每条序列有一个 `seq_seed: uint64`，由调度器在序列创建时分配（跨 batch 唯一）。采样时不使用全局随机状态，而是：

```
combined = seq_seed + f(sample_position * 3 + sample_case)
```

其中：
- `sample_position`：该序列当前生成的 token 序号（int32）
- `sample_case`：`0` = 主模型，`1` = nextn draft（`is_nextn_predict=True`）
- `f(x)`：混合函数，参见 `random_sampling.py:79`

**参考实现（pure-PyTorch，`random_sampling.py:70-84`）[源码]**：

```python
def random_sampling_ref(...):
    sample_case = 1 if is_nextn_predict else 0
    seq = (sample_positions.to(torch.int64) * 3 + sample_case) * i64(0x9E3779B97F4A7C15)
    combined = seq_seeds.to(torch.int64) + (seq ^ (seq * (1 << 13)))
    u = _hash_to_uniform(combined).unsqueeze(-1)   # SplitMix64 → [0,1)
    cdf = probs.double().cumsum(dim=-1)
    target = u * cdf[:, -1:]
    return (cdf > target).long().argmax(dim=-1).int()
```

### 3.2 SplitMix64 哈希

`_hash_to_uniform`（`random_sampling.py:58-67`）实现 SplitMix64：

```python
z = z + 0x9E3779B97F4A7C15
z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
z = (z ^ (z >> 27)) * 0x94D049BB133111EB
z = z ^ (z >> 31)
return (z >> 11) * (1.0 / 2**53)   # → f64 in [0, 1)
```

魔数 `0x9E3779B97F4A7C15 = 2^64 / φ`（黄金比例乘法哈希，bit mixing）。这是标准的 SplitMix64 算法（Guy Steele Jr. 等，2014），保证任意位置 `(seq_seed, sample_position)` 产生高质量均匀分布，与其他 row 的状态完全解耦。

### 3.3 CUDA kernel：两层定点采样（cuda/random_sampling.py）

**[源码]** 参数：`vocab_size ≥ 8192` 时 `threads=256, BD=2048`；小词表则 `BD = next_power_of_2(vocab_size)`。

**两层结构：**

```
Level-1（粗定位）：
  - 每个 thread 持有 BD 维的 acc_frag（float32→uint32/uint64 定点）
  - 对每个 chunk（vocab_size / BD 块）做定点累加
  - block 内 inclusive cumsum 得 lane CDF
  - RNG 产生 r ∈ [0, total)，二分搜索找到 lane k0

Level-2（精选列）：
  - 收集 probs[row, k0 + m*BD] for m in 0..n_slots
  - 再次定点化 + cumsum，用余量 r2 找到精确列
```

**定点精度**（`float_to_fixed`）：`uint32` 模式 `scale_bits=31`，分辨率约 `5×10^-10`；`uint64` 模式从 IEEE754 浮点位精确转换，完全保留 f32 精度。

**拒绝采样循环**（`sample_offset` 内，thread 0 执行）：

```python
limit = (~cast(0, SCALE_DTYPE)) // total * total   # 最大整数倍
rnd = limit   # 触发进入循环
while rnd >= limit:
    rnd = rng_rand()
r = rnd % total
```

这是无偏离散采样（von Neumann rejection），避免模数偏差。由于 `T.rng_init(seed, seq)` 确定性初始化，整个过程对同一 `(seq_seed, sample_position)` 完全可重现，与其他 row 无关。

**批次不变性保证**：每 row 独立计算，没有跨 row 的共享可变状态。同一 row 在任意批次位置、任意批次大小下得到相同结果，`test_random_sampling_batch_invariant`（`random_sampling.py:310-332`）以排列不变性和单条 vs 批量一致性两种方式验证这一点。

### 3.4 MTP draft 的 RNG 隔离：fused_sample.py

```python
# deepinfer/model_executor/layers/tilelang_ops/fused_sample.py:15,93
_MTP_SEED_OFFSET = 1000000007   # 质数，避免与主模型种子对齐

seed_offset = 0 if tag == "default" else _MTP_SEED_OFFSET
```

**[源码]** `tag != "default"` 时（MTP draft 路径），`seq_seed` 加上 `1_000_000_007`（大质数）后再做 SplitMix64，保证 draft 的采样流和主模型的采样流在同一 `(position)` 上不相关。

### 3.5 拒绝采样中的残差重采样

当 draft token 被拒绝时，从**残差分布**重采样（`fused_sample.py:232-246`，参考实现）：

```python
residual = (sample_probs - (draft_logprobs - DRAFT_PROB_PENALTY).exp()).relu_()
accept_probs = (1.0 - residual.sum(dim=-1)).clamp_min_(0.0)
```

`DRAFT_PROB_PENALTY` 是编译期常量，防止 draft logprob 数值溢出导致 accept 率估计偏高。接受判断用 SplitMix64 产生的均匀 U，再做 `exp(target_lp - draft_lp) > U` 检验（标准投机解码 accept 规则）。

### 3.6 PPL 黄金基线机制

**[源码]** `tests/models/offline_decode_ppl.json`：存储 `nvidia` / `ascend` × `short` / `long` × `dp_rank` 的 decode PPL 列表（精确 float64）。

`test_offline_inference.py:154`：
```python
assert actual == expected   # 精确浮点相等，不允许任何误差
```

这是最严格的回归测试——若任何 kernel 改动破坏批次不变性，这个测试立即失败。更新流程：`examples/refresh_offline_decode_ppl.py` 在多节点部署下通过跨节点文件锁（`fcntl.flock`）按 DP rank 合并写入，保证并发安全。


---

## 4. 结构化输出（Structured Output, SO）

### 4.1 系统架构

**[Q&A]**（`doc/inference_deployment_qa.md:284-288`）：

> *"Compilation and inference are decoupled: SO compilation runs asynchronously, and a request retrieves the compiled result before it enters the engine. Compilation stage (on a dedicated SO machine): the format constraint is compiled into a grammar automaton."*

关键设计：**编译在独立的 SO 专用机器上进行**（开源版不含该部分），引擎进程只运行 matcher 和 bitmask 生成。这样 xgrammar 编译的 CPU 密集型工作不占用推理机器的 CPU 资源，也与 CUDA graph 不干扰。

### 4.2 编译器：compiler.py

**[源码]** `deepinfer/structure_output/compiler.py:14-71`：

```python
class XGrammarCompiler:
    # xgrammar <= 0.1.x had production incidents with cache enabled
    # DoS vector: see CVE-2025-32381
    if self.cache_limit_bytes <= 0:
        logger.warning(
            "xgrammar cache is enabled WITHOUT a size limit "
            "(DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES=%d). "
            "... may cause OOM (DoS vector, see CVE-2025-32381)."
        )
```

**CVE-2025-32381**（[源码] 编译器注释直接提及）：xgrammar 语法缓存无上限增长可被用于 OOM DoS，攻击面是恶意客户端提交大量不同 JSON schema。DeepInfer 的缓解：

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `DEEPINFER_XGRAMMAR_DISABLE_CACHE` | `False` | 完全禁用缓存（kill switch）|
| `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES` | `10 GB` | 限制缓存上限 |

代码注释明确指出 `0` 时有 DoS 风险，生产环境需设正值（默认 10 GB 足以容纳大量不同 schema）。

支持的 schema 类型：`json`（`strict_mode=True`），`regex`，`ebnf`（grammar）。

### 4.3 Matcher：matcher.py

#### trigger token 与 KMP 匹配

**[源码]** `matcher.py:24-33`：实现了标准 KMP 失败函数，`accept_token` 的触发阶段（trigger pending 时）用 KMP 滚动匹配：

```python
while progress > 0 and token != self.trigger_tokens[progress]:
    progress = self._trigger_failure[progress - 1]
if token == self.trigger_tokens[progress]:
    progress += 1
```

用途：模型在 `<think>...</think>` 之后才进入结构化输出，`trigger_tokens` 存储该终止符的 token ID 序列。

#### bound_token_id

触发后的"绑定 token"：`bound_token_id` 标识结构化内容的起始分界符（如 `{` 的 token ID）。触发序列完成后，matcher 等待 bound token 出现再开始约束（`tokens_after_bound >= 0`）。

#### post-EOS mask

`fill_vocab_mask`（`matcher.py:181-203`）：

```python
if self.finished():           # tokens_after_eos >= 0
    vocab_mask[idx].copy_(self.mask_after_eos)  # 全 1 mask
    return True
```

EOS 之后不再约束（全允许），避免 SO 约束干扰 EOS 后的 padding。

#### MTP rollback

`rollback`（`matcher.py:206-232`）：投机解码验证时，draft token 可能被部分接受，matcher 需要回滚到对应历史状态：

```python
MAX_ROLLBACK_TOKENS = envs.DEEPINFER_USE_NEXTN_DECODE + 2
```

回滚历史保存在 `_progress_history`（trigger progress 的环形缓冲区）以及 `inner_matcher.rollback(n)`（xgrammar 内置回滚）。`+2` 是额外余量。

**MTP 时 sampler 侧的双重模拟（`sampler.py:366-376`）[源码]**：

```python
for k, draft_token in enumerate(seq_metadata.last_next_n_token):
    if matcher.accept_token(draft_token):
        if matcher.constrained():
            matcher.fill_vocab_mask(self.bitmasks_cpu[k + 1], idx)
        num_accepted_steps += 1
    else:
        break
matcher.rollback(num_accepted_steps)   # 回滚，不改变 matcher 真实状态
```

sampler 用 matcher **模拟** draft token 的接受，计算每个投机步的 bitmask，最后回滚恢复 matcher 状态，只有真正被验证接受的 token 才会在下一 step 通过 `accept_token` 永久推进。

#### 异常隔离

`sampler.py:377-386`：**[源码]** matcher 异常只杀死该请求（`so_status[idx] = -1`），不传染其他请求，也不触发 NCCL barrier 超时：

```python
except Exception:
    logger.exception("[SO] matcher failed, killing request %s", ...)
    matcher.set_aborted("SO matcher runtime failure")
    so_status[idx] = -1
```

### 4.4 bitmask 在 sampler 中的应用

**[源码]** `sampler.py:162-164`（prefill 时）：

```python
if so_indices is not None and len(so_indices[0]):
    XGrammarUtils.apply_vocab_mask_by_step_inplace(
        logits, so_bitmasks[0], so_indices[0]
    )
```

bitmask 在 so_copy_stream（独立 CUDA stream）上异步 H2D 拷贝，通过 `torch.cuda.Event` 与主流同步：

```python
if so_mask_ready is not None:
    torch.cuda.current_stream().wait_event(so_mask_ready)
```

MTP decode 时为每个投机步备有独立的 bitmask（`so_bitmasks[k]`, k=0..max_MTP_value）。

### 4.5 SOMatcherBuilder 的异步编译

**[源码]** `so_matcher_builder.py`：单 `ThreadPoolExecutor`，可绑定 CPU 核（`bind_cpus`）。`submit()` 异步提交编译任务，`pump()` 轮询已完成的 future。主线程只在 `pump()` 中消费结果，避免阻塞调度循环。超时由 `DEEPINFER_XGRAMMAR_TIMEOUT`（默认 20 秒）控制，超时直接返回 `aborted_matcher`。

### 4.6 与业界方案对比

| 方案 | 语法表达 | 实现 | MTP 支持 | 缓存 DoS 防护 |
|---|---|---|---|---|
| **DeepInfer xgrammar** | JSON/regex/EBNF | xgrammar + KMP trigger + rollback | ✓（per-step bitmask + rollback）| CVE-2025-32381 已缓解，10GB 上限 |
| **vLLM** xgrammar | JSON/regex/EBNF/grammar | xgrammar 原生 | ✓（0.8+）| [未核实] |
| **SGLang** xgrammar | 同上 | xgrammar + radix cache prefix skip | ✓（0.4+）| [未核实] |
| **llguidance** | JSON/regex/LARK EBNF | Rust NFA 编译 | [未核实] | 无 OOM DoS 公开报告 |
| **outlines** | JSON/regex/CFG | 基于 Pydantic 的 JSON schema → NFA | 不官方支持 | 无 cache 机制 |

DeepInfer 相对 outlines 的主要优势：xgrammar 原生位图 mask 可直接传 GPU；rollback API 契合 MTP 验证流程。相对 llguidance：xgrammar 有 Python binding 且与 transformers tokenizer 直接对接，但 llguidance 在多步推进性能上有优势 **[未核实]**。


---

## 5. 业界方案对比

### 5.1 vLLM Batch Invariant（issue #27433）

GitHub issue `vllm-project/vllm#27433` *"Batch Invariant Feature and Performance Optimization"*（open，基于 Thinking Machines 博客）已完成的里程碑：

- 基础框架（PR #25603）、FlashInfer、DeepSeek-V3 支持、DeepGEMM on Blackwell、TP8 on Blackwell、CUDA graph、文档
- 另有 RL 示例 PR

**开关**：`VLLM_BATCH_INVARIANT=1`（beta）。

**Vosti 的发现**：即使开启后，某些调度变化（如 prefix-sharing 的命中/未命中路径切换）仍会产生不同 logits，说明 vLLM 未达到完整形式化规范。

### 5.2 SGLang 确定性模式（issue #10278, #45481）

**实现**（issue `#10278` *"Support deterministic inference with Batch Invariant Ops"*，closed）：

- `--enable-deterministic-inference` 开关
- 覆盖 CUDA graph、temperature>0、FlashInfer/Triton/FA3 注意力后端
- Radix cache 的 FA3/Triton 路径已支持，FlashInfer 尚未完成
- Blackwell TP4 下仍有确定性问题（`#11513`，open）

**已知限制**（issue `#45481`，open）：PP/NEXTN 下并发到达的请求仍可得到不同 greedy 输出，问题在 TP1/PP2 + NEXTN 路径。

### 5.3 Thinking Machines 博客的定量发现

博客中报告了在同一 hardware、同一 prompt、不同 batch size 下，greedy 输出的 token 开始分叉的位置分布。性能代价：确定性 matmul（去掉 split-K 优化）在典型 decode batch 下带来约 **10-20%** 的吞吐损失 **[未核实，原文有图但未公开数字]**。

### 5.4 RL 训推一致性需求

**verl / slime** 等主流 RLHF 框架需要推理侧采样与训练侧 log prob 计算精确一致（on-policy 校验）：

- **verl**（HybridEngine）：`actor_rollout_ref` 与 `reference_policy` 共享权重，对同一 prompt 两次 forward 必须得到相同 logit，否则 KL 估计有偏。
- **slime**：同上，on-policy 采样后用 reference model 计算 baseline；若推理和训练的 logit 因批次组成不同而不一致，PPO 的 old_log_prob 会系统性偏移。

DeepInfer 提供 `DEEPINFER_BATCH_INVARIANT=1` 作为满足此要求的机制。但需注意：这仅保证推理侧批次不变，**训练侧（PyTorch DDP/FSDP）的规约也可能引入类似不确定性**，需要 `torch.use_deterministic_algorithms(True)` 配合。

---

## 6. Sampler 的其他特性

### 6.1 force_topk

`DEEPINFER_SAMPLER_FORCE_TOPK = EnvVar.int()`（`environ.py:150`，默认空/0）。

**[源码]** `sampler.py:99,461`：只对 top-K 个 token 做 softmax 和采样，跳过 vocab_size 完整的 softmax。需要维护 `token_index` 映射表（top-K 的 token ID → 原始 vocab 索引）。

**[估算]** 对 vocab_size=129280 的 V4.1，完整 softmax 每个 token 读 129280×4B≈520KB；force_topk=1024 时读 1024×4B≈4KB，**减少 99% 的带宽**。但需要先做一次 argsort/topk（写回）。在 decode batch 较小时（对 memory-bandwidth limited kernel），节省显著。

### 6.2 decode_token_blacklist

`DEEPINFER_DECODE_TOKEN_BLACKLIST = EnvVar.str()`（`environ.py:151`）。

**[源码]** `sampler.py:168-169`：
```python
if self.decode_token_blacklist is not None:
    logits.index_fill_(-1, self.decode_token_blacklist, -torch.inf)
```

在 logits 上直接 `-inf` 填充，发生在 temperature/top-p/top-k 之前，保证黑名单 token 在任何采样配置下都不会被输出。典型用途：屏蔽特定的控制 token、避免 safety bypass token。

### 6.3 Sampler 正确性验证：test_sampler.py

**[源码]** 使用**随机化 PIT + KS 检验**（Probability Integral Transform + Kolmogorov-Smirnov）：

1. 生成 rollout，提取每 token 的 `sum_smaller_probs`（累积概率 < 被采样 token 的概率质量）和 `token_logprobs`
2. `U = ssp - p * Uniform(0,1)` → 应服从 U(0,1)
3. KS 统计量应在 0.001 置信水平通过

这比简单的频率检验更强：直接验证**采样分布的精确性**，而非期望值匹配。测试设置 `DEEPINFER_BATCH_INVARIANT=1`（`test_sampler.py:121`），确保在确定性模式下也能通过统计检验。

### 6.4 Vision Encoder 批次不变性：test_ve_batch_invariant.py

**[源码]** `test_ve_batch_invariant.py`：独立测试视觉编码器（ViT）的批次不变性：

- 对每种图片分辨率生成固定随机图片（seed = `IMAGE_RNG_SEED + shape_index`）
- 单图与 N 图批次：要求 `torch.equal`（逐位相等）
- 混合分辨率批次：同样要求逐位相等

说明 ViT 的 GPU 路径（无 LLM 权重参与）也通过了批次不变性设计。

---

## 7. 性能代价与取舍

### 7.1 已知代价来源

| 代价来源 | 位置 | 性质 |
|---|---|---|
| `use_deterministic_algorithms(True)` | DeepGEMM + TileKernels | 禁用 split-K 等利用 batch 形状的优化；代价视矩阵形状而定 |
| `enable_batch_invariant=True` 的 FlashMLA | attention_backend.py | 固定 split-KV 策略，可能导致 SM 利用率下降 |
| `allow_multiple_reduction=False` | userbuffer.py（EP buffer）| 只有开 EPLB 时才有影响；关掉优化后每步 EP reduce 的 latency 上升 |
| 禁用 `fused_norm_rope_attn_rope_cast` | attn.py prefill 路径 | 分步执行代替 fused kernel，约 1-3 个 kernel launch overhead **[估算]** |
| PPL 黄金基线测试 | test_offline_inference.py | CI 代价（精确浮点比较），不影响推理性能 |

**[推断]** DeepInfer 默认（`BATCH_INVARIANT=0`）在生产中走性能优化路径；`BATCH_INVARIANT=1` 是"评估/RL 模式"的显式降速选项。在 `offline_inference.py:43` 默认开启（离线推理），在 eval 脚本 `run_eval.py:74` 作为可选参数。

### 7.2 取舍矩阵

| 场景 | 推荐配置 | 理由 |
|---|---|---|
| 最高吞吐（服务部署） | `BATCH_INVARIANT=0, ENABLE_MTP_VARLEN=1` | 变长 MTP 按置信度动态调整 speculate 步数，提高接受率 |
| RL on-policy 校验 | `BATCH_INVARIANT=1, ENABLE_MTP_VARLEN=0` | 确保同一 prompt 在任意批次下 logit 相同 |
| 评测可复现 | `BATCH_INVARIANT=1, ENABLE_MTP_VARLEN=0` | 同上；PPL 黄金基线要求精确匹配 |
| SO 高并发 | SO 独立编译机 + 缓存上限 | 防 CVE-2025-32381 OOM，减少推理机 CPU 竞争 |


---

## 8. 开放问题与建议

### 8.1 开放问题

1. **NPU 批次不变性的实现路径**：`runtime_config.py:63` 的 assert 表明这是进行中的工作，但源码中未见具体实现。Ascend TileLang kernel 是否支持与 CUDA 等价的确定性规约，是核心技术挑战 **[未核实]**。

2. **Vosti 指出的根本缺口**：vLLM 和 SGLang 的实现均未通过 Vosti 的形式化规范。DeepInfer 同样没有进行形式化验证。PPL 黄金基线测试是经验性而非数学保证——如果将来某次 kernel 改动恰好在测试 prompt 上给出相同 PPL 但其他 prompt 不同，测试不能捕获。

3. **CUDA graph 选择的确定性**：CUDA graph 按 `(is_prompt, total_q)` 选择不同 graph 尺寸，不同尺寸意味着不同 grid 形状。现有机制是对每个 graph 尺寸独立测试确定性，但**跨 graph 的逐位相同**（同一 prompt 进入不同 graph）并未保证 **[推断]**。

4. **SO + MTP 的完整覆盖**：`MAX_ROLLBACK_TOKENS = DEEPINFER_USE_NEXTN_DECODE + 2`——如果用户设置非常大的 nextn，rollback history 的内存开销线性增长；且如果投机步数超过 max_rollback，会抛出 RuntimeError。生产中 nextn=5 时为 7，合理。

5. **xgrammar 版本锁定**：DeepInfer 没有在代码中固定 xgrammar 版本号（只有注释"xgrammar <= 0.1.x had production incidents"），未来 xgrammar 升级引入的行为变化（如语法解析差异）可能导致 SO 输出静默改变。

### 8.2 对从业者的建议

1. **RL 训推一致性**：先开 `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` 验证 logit 位精确相同，再逐步评估能否接受 `ENABLE_MTP_VARLEN=1` 带来的随机性（若 RL 框架容忍小幅偏差，可换取 ~10% 吞吐提升 **[估算]**）。

2. **评测基线锁定**：参照 `offline_decode_ppl.json` 机制，为自己的模型和硬件建立精确 PPL 黄金基线，而不是仅用近似 tolerance 比较。

3. **CVE-2025-32381 缓解**：生产部署 SO 时务必设置 `DEEPINFER_XGRAMMAR_CACHE_LIMIT_BYTES`（建议 2-10 GB，视 schema 多样性和内存预算），不要使用 0（无限缓存）。

4. **SO 编译隔离**：若并发 SO 请求多，参照 Q&A 的独立编译机设计，将 xgrammar 编译移到专用进程/机器，避免与 GPU 计算竞争 CPU。

5. **批次不变性的测试层次**：基础 sampler 层用 `test_random_sampling_batch_invariant` 单元测试覆盖；模型层用 `test_batch_invariant.py`（需要完整模型）端到端验证；PPL 黄金基线作为回归守护。三层缺一不可。

---

## 9. 架构流程图

```
DEEPINFER_BATCH_INVARIANT=1 时的数据流
───────────────────────────────────────────────────
请求 → 调度器（按 total_q 选 CUDA graph）
        │
        ▼
  forward pass
   ├─ DeepGEMM (deterministic, split-K 关闭)
   ├─ TileKernels (deterministic)
   ├─ FlashMLA decode (enable_batch_invariant=True)
   ├─ EPBuffer (allow_multiple_reduction=False)
   └─ fused_norm_rope_attn_rope_cast (prefill 非 sparse 非 packed KV → 禁用)
        │
        ▼
  logits → Sampler
   ├─ blacklist 填 -inf
   ├─ SO bitmask (copy_stream 异步 H2D)
   ├─ fused_sample (temperature/top-p/top-k)
   └─ random_sampling kernel
        ├─ RNG: SplitMix64(seq_seed + offset(sample_pos*3+case))
        ├─ Level-1: 定点 CDF 粗搜索 (per-row 独立)
        ├─ Level-2: 精确列选择 (无偏拒绝采样)
        └─ MTP draft: seed_offset = 1_000_000_007

MTP 验证路径:
   SO matcher.simulate_draft() → fill_vocab_mask[k] → rollback()
   → accept/reject → residual 重采样
   → 真正 accept_token() 推进 matcher 状态

PPL 黄金基线 (CI):
   实测 decode_ppl → fcntl.flock 跨节点合并
   → tests/models/offline_decode_ppl.json
   → assert actual == expected  (精确浮点)
```

---

## 参考来源

| 引用 | 来源类型 |
|---|---|
| `deepinfer/worker/runtime_config.py:62-69` | **[源码]** |
| `deepinfer/model_executor/layers/attention_backend.py:85,141` | **[源码]** |
| `deepinfer/model_executor/models/v4_modules/attn.py:89,666,1002` | **[源码]** |
| `deepinfer/model_executor/parallel_utils/userbuffer.py:79` | **[源码]** |
| `deepinfer/model_executor/layers/tilelang_ops/random_sampling.py:58-84,310-332` | **[源码]** |
| `deepinfer/model_executor/layers/tilelang_ops/cuda/random_sampling.py:1-290` | **[源码]** |
| `deepinfer/model_executor/layers/tilelang_ops/fused_sample.py:15,93` | **[源码]** |
| `deepinfer/model_executor/layers/sampler.py:91-112,163-169,323-408` | **[源码]** |
| `deepinfer/structure_output/compiler.py:14-71` | **[源码]** |
| `deepinfer/structure_output/matcher.py:24-232` | **[源码]** |
| `deepinfer/structure_output/so_matcher_builder.py:19-138` | **[源码]** |
| `tests/models/test_batch_invariant.py` | **[源码]** |
| `tests/models/test_ve_batch_invariant.py` | **[源码]** |
| `tests/models/test_sampler.py` | **[源码]** |
| `tests/models/offline_decode_ppl.json` + `examples/refresh_offline_decode_ppl.py` | **[源码]** |
| `deepinfer/worker/worker.py:132,218,230` | **[源码]** |
| `doc/inference_deployment_qa.md:278-288` | **[Q&A]** |
| Thinking Machines, *Defeating Nondeterminism in LLM Inference*, Sep 2025 — https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/ | **[核实]** |
| arXiv 2609.38981, *Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference* | **[核实]** |
| GitHub `vllm-project/vllm#27433` *Batch Invariant Feature and Performance Optimization* | **[核实]** |
| GitHub `sgl-project/sglang#10278` *Support deterministic inference with Batch Invariant Ops* | **[核实]** |
| GitHub `sgl-project/sglang#45481` *Qwen3.5-35B-A3B TP1/PP2+NEXTN nondeterminism under deterministic mode* | **[核实]** |
| CVE-2025-32381 (xgrammar cache OOM DoS) — 源码注释中引用，未查阅 CVE 原始记录 | **[未核实 CVE 原文]** |

