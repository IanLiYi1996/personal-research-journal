# DeepInfer 专题 T2：MoE 并行 —— 宽 EP、DeepEP、Mega MoE、EPLB、Engram 表并行

> **⚠️ 勘误（主编交叉核对，2026-10-21）**：§2.4 的单专家体积误用了 DeepSeek-V3 的维度（7168 / 2048）。V4.1-Flash 的实际值是 hidden=5120、moe_intermediate=2304。
> - 单专家参数量：3 × 5120 × 2304 = 35.4M。
> - MXFP4 体积：17.7 MB，加上 UE8M0 scale（每 32 个元素 1 B）1.1 MB，合计约 **18.8 MB/专家**。
> - EP8 每卡 49 个专家 ≈ **0.92 GB**/步，EP64 每卡 7 个专家 ≈ **0.13 GB**/步，降幅约 **86%**（"约 85%"的结论不变）。
> - 冗余专家显存应按 V4.1 的 **40** 个 MoE 层计算，不是 56 层：每卡 1 个冗余 ≈ 18.8 MB × 40 ≈ **0.75 GB**。


> **基线**：DeepInfer commit 774ce24；官方 Q&A doc/inference_deployment_qa.md；今日 2026-10-21。
> 标记：**[源码]** 代码确认 · **[Q&A]** 官方文档 · **[估算]** 推算+算式 · **[推断]** 合理推测 · **[未核实]** 需进一步确认

---

## 1. 源码精读

### 1.1 门控机制：sqrtsoftplus + e_score_correction_bias + noaux_tc

**[源码]** `deepinfer/model_executor/models/v4_modules/moe.py:46-67`

`TopKGate` 持有三个参数：`Wgate`（隐状态→专家 logit，bf16，不参与 fp8 量化）、`e_score_correction_bias`（文字 token 路由偏置，fp32）、`e_score_correction_bias_vl`（图像 token 路由偏置，fp32）。

```
scoring_func = "sqrtsoftplus"    # score = sign(x) * sqrt(softplus(|x|))，抑制长尾大值
topk_method = "noaux_tc"         # 训练时无辅助损失 + 动态 token-count 纠偏
norm_topk_prob = True            # top-k 权重归一化后乘 routed_scaling_factor
```

**[源码]** `mlp.py:364-401`：`gate_forward` 在推理时分两路调用门控内核：
- GPU 上走 `deep_gemm.bf16_mega_gate`（`use_mega_gate=True`，内含 fused GEMM + 路由逻辑）；
- Ascend 上走 `tile_kernels.moe.moe_topk_gate`（传统两步）。

关键参数传入：
```python
gate_kwargs = dict(
    bias=self.e_score_correction_bias,           # 每步偏置修正，抵消 noaux_tc 引入的不均衡
    image_bias=self.e_score_correction_bias_vl,  # 视觉 token 专用偏置
    to_physical_map=eplb_layer.logical_to_physical_map,   # 逻辑→物理槽位映射
    logical_count=eplb_layer.logical_count,      # 每个逻辑专家的副本数（用于负载分流）
    scoring_func="sqrtsoftplus",
    routed_scaling_factor=self.routed_scaling_factor,
)
```

**物理/逻辑专家数语义**（`mlp.py:161-205`）：
- `num_logical_experts`：模型定义的路由专家数（V4.1 = 384）；
- `num_extra_experts`：冗余副本数（`n_redundant_experts`，由 model_loader 写入，取整保证整除 EP 大小）；
- `num_physical_experts = 384 + num_extra_experts`；
- `num_local_physical_experts = num_physical_experts / ep_size`。

### 1.2 MMA 路径与 Mega MoE 条件

**[源码]** `mlp.py:184-188`：

```python
self.use_mega_moe = (
    (PLATFORM.is_gpu or self.mma_type == "fp8xfp4")
    and self.mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16")
    and PG.is_within_nvl_domain(PG.expert_parallel_group())   # EP 组在同一 NVLink 域内
)
```

Mega MoE 自动启用条件：GPU 平台 + EP 组在 NVLink 域内。Ascend 跨节点场景退化为普通 EP（dispatch → grouped-GEMM → combine）。

Mega MoE 使用 `deep_gemm.SymmBuffer`（对称内存）存放 `x/x_sf/topk_idx/topk_weights` 等输入，调用 `deep_gemm.fp8_fp4_mega_moe` / `deep_gemm.bf16_mega_moe` 实现 dispatch+GEMM+SwiGLU+GEMM+combine 的单内核融合 **[源码]** `mlp.py:548-559`。

权重格式转换：`_convert_mega_moe_weight` 调用 `deep_gemm.transform_weights_for_mega_moe`，将 gate_up / down_proj 的 FP4/FP8 权重重排成 Mega MoE 专用 layout **[源码]** `mlp.py:93-106`。

### 1.3 userbuffer.py：通信缓冲分配与 SM 预算

**[源码]** `parallel_utils/userbuffer.py`

`_UserBufferManager` 是进程级单例，统一管理 4 类缓冲：

| 缓冲类型 | 用途 | 构造参数 |
|---|---|---|
| `EPBuffer` (DeepEP) | 非 Mega MoE 的跨卡 dispatch/combine | `allow_multiple_reduction=not(batch_invariant and rearrange_interval>0)` |
| `SymmBuffer` (DeepGEMM) | Mega MoE 对称内存，逐层共享同一实例 | `group/num_experts/num_max_tokens/num_topk/hidden/intermediate_hidden/...` |
| `EngramBuffer` (DeepEP) | Engram n-gram 表分片存储与 RDMA 拉取 | `allow_hybrid_mode=True`，EP 流量走 NVLink sl3 |
| `CPManager` | 上下文并行 CP 的桶式 all-gather | `BucketBuffer`，按 session 管理生命周期 |

SM 预算自动计算 **[源码]** `userbuffer.py:86-90`：
```python
if envs.DEEPINFER_USERBUFFER_COMM_SMS == 0:
    num_sms = ep_buffer.get_theoretical_num_sms(num_experts, num_topk)
    compute_num_sms = min(32, device_num_sms // 4)
    envs.DEEPINFER_USERBUFFER_COMM_SMS = min(num_sms, device_num_sms - compute_num_sms)
```
保留至少 `device_num_sms // 4`（最少 32）个 SM 给计算；通信 SM 数从拓扑和通信量分析推导，不再需要手工调参（DeepEP V2 的改进之一）。

`limit_num_sms` 上下文管理器 **[源码]** `userbuffer.py:40-51`：在 combine 阶段同时跑 shared expert forward 时，把 DeepGEMM 和 TileKernels 可用 SM 上限减去通信占用数，避免 SM 争抢。

### 1.4 parallel_state.py：进程组 einops 构造

**[源码]** `parallel_state.py:70-127`

DeepInfer 的 5 类进程组，全部通过 `einops.rearrange` 从 `ranks = torch.arange(max(enp_size, dp_size * cp_size))` 派生：

```python
# DP 组：按 (replica, dp, cp) 重排，同一 dp 列为一组
dp_ranks = rearrange(ranks, "(replica dp cp) -> (replica cp) dp", dp=dp_size, cp=cp_size)

# CP 组：按 (dp, cp) 重排，同一 dp 行为一组
cp_ranks = rearrange(ranks, "(dp cp) -> dp cp", cp=cp_size)

# EP 组：连续 ep_size 个 rank 为一组
ep_ranks = rearrange(ranks, "(edp ep) -> edp ep", ep=ep_size)

# ENP 组（Engram 并行）：连续 enp_size 个 rank 为一组
enp_ranks = rearrange(ranks, "(replica enp) -> replica enp", enp=enp_size)

# EP-in-DP 组：连续 ep_size 个 rank 为一组（与 EP 等同，单独建以便 DP 内 EP shuffle）
ep_in_dp_ranks = rearrange(ranks, "(edp ep_in_dp) -> edp ep_in_dp", ep_in_dp=ep_size)
```

关键约束：`#DP = #EP`（all-to-all 和 attention DP 步长对齐）；ENP 可独立于 EP（Engram 表可以用更大范围的分片）。
生产不启用 TP 和 CP（Q&A 明确）；`enp_size` 与 `ep_size` 可不同 **[推断]**（代码中 `max(enp_size, dp_size*cp_size)` 确认两者可独立配置）。

### 1.5 eplb/__init__.py：ExpertLoadBalancer 逐函数解析

**[源码]** `model_executor/eplb/__init__.py`

#### resolve_num_extra_experts（第22-28行）

```python
def resolve_num_extra_experts(num_routed_experts, min_num_extra_experts, ep_size):
    padding = -(num_routed_experts + min_num_extra_experts) % ep_size
    return min_num_extra_experts + padding
```

取不小于下限、且使 `num_physical_experts` 整除 `ep_size` 的最小冗余数。
- Prefill（min=4）：`(384+4) % ep_size`；若 ep_size=4，`388 % 4 = 0`，冗余=4；
- Decode（min=1）：`(384+1) % 64 = 385 % 64 = 1`，需 padding=63，即冗余=64，每卡 `(384+64)/64 = 7` **[Q&A 确认]**。

#### 环形滑动窗口（第102-123行）

```python
self._expert_count_window = torch.zeros(
    expert_count_window_size, num_moe_layers, num_local_physical_experts, dtype=torch.int)
self._expert_count_window_index = 0
# 首次重排提前到 1/4 周期，纠正冷启动不均衡
initial_rearrangement_delay = max(1, self._rearrangement_interval // 4)
self._rearrangement_step = max(0, self._rearrangement_interval - initial_rearrangement_delay)
```

- 窗口大小 `W = DEEPINFER_EXPERT_COUNT_WINDOW_SIZE`，默认 100（prefill）/ 1000（decode）；
- 重排周期 `T = DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL`，默认 1000（prefill）/ 10000（decode）；
- **首次重排** 提前到第 `T/4` 步（而非第 T 步），尽早纠正冷启动的不均衡——这是实现中的一个工程细节，`_brief.md` 点名要求解析 **[源码]**；
- 每步把 `expert_count`（本地各物理槽位 token 数）写入环形窗口的当前位置。

#### _rearrange_experts（第222-278行）

流程：
1. `_initialize_shuffle()`：首次调用时检查是否可用 `symm_mem`（单节点 GPU），决定走 IPC 镜像路还是 P2P 路；
2. `scatter_add_ + all_reduce`：把本地物理 slot 统计映射回逻辑 expert，全局聚合 W 步数据；
3. `rebalance_experts_global()`：两阶段贪心算法（见 1.6）；
4. 调用对应的 shuffle 函数搬运权重；
5. 原子更新 `physical_to_logical_map`、`logical_to_physical_map`、`logical_count`。


### 1.6 eplb/policy.py：rebalance_experts_global 两阶段贪心

**[源码]** `model_executor/eplb/policy.py`

#### 阶段一：冗余副本分配（第110-123行）

目标函数：对 W 步历史数据的每副本负载求和，最小化"分配后所有专家的最大每副本负载的总和"。

伪代码：
```python
for i in range(num_redundant_experts):
    # 每个专家的当前每副本负载（加微小扰动避免并列）
    score = (tokens_per_expert + j * 1e-4) / logical_count  # [W, L, n_logical]
    # 如果给专家 j 再加一个副本，它的每副本负载变为
    score_added = tokens_per_expert / (logical_count + 1)
    # 找出最大负载专家，替换为 score_added，重算各 step 最大值后按 step 求和取 argmin
    redundancy_indices = (new_max_per_step).sum(0).argmin(-1)  # [num_moe_layers]
    # 将本轮冗余副本分配给 redundancy_indices 所指的逻辑专家
```

**复杂度**：`O(num_redundant * W * num_moe_layers * num_logical_experts)`，在 decode 配置（冗余=64, W=1000, L=56层, 384专家）下约 `64 × 1000 × 56 × 384 ≈ 1.38×10⁹` 次浮点运算 **[估算]**，全部在 CPU 完成（tensor device=logical_expert_token_counts.device，该 tensor 经过 all_reduce 后仍在 GPU，故实际在 GPU 执行）。

#### 阶段二：physical expert 均衡分配到各 rank（_batch_greedy_indices）

目标函数（第43-45行注释）：
```
minimize Σ_{step i}  max_{rank j}  Σ_{expert e assigned to rank j} load(e, step i)
```
即在多步历史批次上，最小化"峰值 rank 负载之和"，逐 expert 贪心，每次选当前得分最低的 rank 放置，满载的 rank 排除。

**复杂度**：`O(num_physical * num_ranks * num_steps * num_layers)`，
Decode 下：`448 × 64 × 1000 × 56 ≈ 1.6×10⁹`，同样在 GPU 执行 **[估算]**。

整个 `_rearrange_experts` 会做一次 `torch.cuda.synchronize()`（仅 rank 0 记录时间），持续时间实测在 **数百毫秒**（log 记录 `time=0.xxx s`），期间 GPU 闲置——这是下文取舍分析中"重排停顿"的主要来源 **[推断]**。

### 1.7 eplb/shuffle.py：symm_mem 层内镜像 vs P2P 分桶

**[源码]** `model_executor/eplb/shuffle.py`

两条路径的选择在 `_initialize_shuffle`（eplb/__init__.py:196-220）：

```python
self._intranode_shuffle = PG.is_within_nvl_domain(ep_group) and PLATFORM.is_gpu
```

**路径 A（单节点 IPC 镜像）**：`expert_model_parallel_shuffle_inplace_intranode`
- 为每个权重 tensor 分配 `symm_mem.empty`，构成大小为 `1层 × n_weights × [num_local_experts, hidden]` 的对称内存镜像；
- shuffle 时：barrier → 所有 rank 将自己本层权重拷入镜像 → barrier → 每卡直接读对端镜像，无 P2P 发送；
- 镜像只有一层大小，逐层刷新复用，节省 `num_moe_layers` 倍显存 **[源码]** `__init__.py:205-208`。

**路径 B（跨节点 batched P2P）**：`expert_model_parallel_shuffle_inplace`
- 按 `(对端 rank 差 mod DEEPINFER_REARRANGE_P2P_GROUPS)` 分桶（默认 1 桶）；
- 逐桶调用 `dist.batch_isend_irecv`；
- `_paired_sender` 函数确保收发双方一致配对，避免死锁。

关键观察：P2P 桶分配的对端差模运算（`(ep_rank - dst) % len(buckets)`）是一种隐式的**环形通信调度**，在单桶时退化为顺序发送，多桶时并行度更高但对带宽需求更大。

### 1.8 engram.py：Engram 表并行与 EngramBuffer

Engram 是 V4.1-Flash 独有的 n-gram 词元嵌入系统，用于给注意力层的隐状态注入 n-gram 上下文。

**架构** **[源码]** `v4_modules/engram.py`：
- `CompressedTokenizer`：将原始词表压缩（NFKC/NFD 归一化+小写），减小 n-gram 哈希表大小；
- `NgramEmbedding`：各层的嵌入分片，weight shape = `[num_local_embeddings, n_embed_per_head]`；
- 分片规则：按 n-gram 哈希值模 `enp_size` 决定哪个 rank 持有；
- `EngramRuntime.allocate`：调用 `userbuffer.allocate_engram`，构造 `EngramBuffer`；  
  - EP 通信走 NVLink，Engram 流量用 sl3（sl2 留给 KV cache 流量隔离）**[源码]** `userbuffer.py:155`；
  - 支持 FP8 嵌入（`fp8_embed=True`），`gather_sf` 将 ue8m0 scale 打包成 int32/int16；
- `EngramRuntime.fetch(hashes)`：返回逐层 hook（callable），每层 Engram forward 时调用；
- 生命周期：`initialize()` 将权重写入 `EngramBuffer`，之后 `weight` 和 `sf` 释放（减少 GPU 常驻内存）。

ENP 进程组（`PG.enp`）独立于 EP 组，允许 Engram 分片跨越更大范围——使哈希表分布更均匀，减少热点 **[推断]**。

---

## 2. Q&A 参数对照

### 2.1 冗余专家的确切数字

**[Q&A]** + **[源码 resolve_num_extra_experts]**：

| 阶段 | min_redundant | EP 大小 | 计算 | 实际冗余 | 每卡专家数 |
|---|---|---|---|---|---|
| Prefill | 4 | 4 | `(384+4)%4=0`，padding=0 | 4 | `388/4=97` |
| Prefill | 4 | 8 | `(384+4)%8=4`，padding=4 | 8 | `392/8=49` |
| Decode | 1 | 64 | `(384+1)%64=1`，padding=63 | 64 | **`(384+64)/64=7`** |
| Decode | 1 | 128 | `(384+1)%128=1`，padding=127 | 128 | `(384+128)/128=4` |

EP64 每卡 **7 个路由专家** + 1 个 shared expert（共享，不参与 EP 分片，在每卡本地计算），**[Q&A 明确确认]**。

### 2.2 窗口、周期与 moe_lb 实测值

| 阶段 | 统计窗口 W | 重排周期 T | 首次重排步 | moe_lb（slowest/mean） |
|---|---|---|---|---|
| Prefill | 100 步 | 1,000 步 | `max(1, 1000//4)=250`，第 750 步触发 | 1.074（EP4DP4）/ 1.045（EP2DP2） |
| Decode | 1,000 步 | 10,000 步 | 第 7,500 步触发 | **1.090** |

**[源码]** `eplb/__init__.py:121-123`：首次重排延迟 = `max(1, T//4)`，因此 decode 首次触发在第 `T - T//4 = 7500` 步。

`moe_lb` 定义（**[源码]** `__init__.py:298-311`）：`mean_peak_rank_load / mean_rank_load`，即窗口内最峰值 rank 的负载 / 全局均值，1.09 意味着最忙的 rank 比均值多承担约 9%。

### 2.3 EP8 vs EP64 图解读

**[源码]** 图像 `doc/images/wide_ep_decode_pareto.png`，条件：context=131,072，nextn=3 固定，每 token 接受率 75%（τ=3.25）。

| 并发 (seqs/rank) | EP64 吞吐 (tok/s/GPU) | EP8 吞吐 (tok/s/GPU) | EP64 增益 | 交互速度 EP64 | 交互速度 EP8 |
|---|---|---|---|---|---|
| 256 | ~26,100 | ~25,000 | +4.4% | ~100 | ~100 |
| 128 | ~22,900 | ~19,300 | +18.7% | ~175 | ~150 |
| 64 | ~15,800 | ~12,800 | +23.4% | ~245 | ~240 |
| 32 | ~9,600 | ~7,000 | +37.1% | ~295 | ~225 |
| 8 | ~2,600 | ~1,600 | +62.5% | ~330 | ~240 |

关键观察：
1. EP64 在**低并发**（32 以下）时增益最大（37–62%），因为此时每卡专家 GEMM 的 token 数少，更宽 EP 能聚合更大批；
2. EP64 的交互速度（per-user tokens/s）在中低并发时也更高，说明宽 EP 在保高吞吐的同时不牺牲延迟；
3. EP64 配置是 `384+64 experts`（7/卡），EP8 是 `384+32 experts`（52/卡）——EP8 冗余更多（相对），但分散到 8 卡效果有限；
4. 在 256 并发时 EP8 与 EP64 差距收窄，因为此时每卡专家 GEMM 已经是算力受限，宽 EP 带来的访存降低效益减少。

### 2.4 宽 EP 降低每步专家访存的定量模型 [估算]

**模型假设**：V4.1-Flash，384 路由专家 + top-6，MXFP4 权重（per-32 ue8m0），`hidden=7168`，`intermediate=2048`（每 expert 的 intermediate_size 为 moe_intermediate_size），EP 大小为 P，每卡持有 `E_local = (384 + redundant) / P` 个专家，batch size = B seqs/rank。

**单个专家权重大小**（MXFP4，gate_up+down_proj）：
```
gate_up: hidden(7168) × 2×intermediate(2048) × 0.5 bytes/element = 14.68 MB
down:    intermediate(2048) × hidden(7168)  × 0.5 bytes/element =  7.34 MB
单专家总权重 ≈ 14.68 + 7.34 = 22.02 MB
```

scale factor overhead（ue8m0，per-32）：`22.02MB × (1/32) ≈ 0.69 MB/expert`，合计 **≈22.7 MB/expert** **[估算]**。

**每步激活的唯一专家数**（假设 token 路由均匀分布）：
- 每 rank 有 `B × 6` 次选择，从 `E_local` 个专家中选；
- 激活唯一专家期望 ≈ `E_local × (1 - (1 - 1/E_local)^{B×6})`；
- 当 B 较小时，`B×6 << E_local`，期望 ≈ `B × 6`（几乎每次都新的）；
- 当 B 较大时，趋向 `E_local`。

| EP 大小 | E_local（约） | B=16 时唯一专家数 | 专家访存 | B=64 时唯一专家数 | 专家访存 |
|---|---|---|---|---|---|
| EP8  | 49 | ≈49 | 49×22.7 = 1.11 GB | ≈49 | 1.11 GB |
| EP64 | 7  | ≈7  | 7×22.7 = 0.16 GB  | ≈7  | 0.16 GB |

EP64 相比 EP8 每步专家权重读取约 **减少 85%**（从 1.11 GB 降至 0.16 GB）**[估算]**。
但 EP64 引入了 all-to-all 通信：每 rank 发送 `B × 6 × 7168 × 2 bytes ≈ B × 0.86 MB`（BF16），接收等量；相比 EP8 增加了 8× 的通信量（远端 rank 增多），不过在 NVLink 300+ GB/s 的带宽下远小于 HBM 访存节省。

**结论** **[估算]**：EP64 对 EP8 的吞吐优势主要来自 **HBM 带宽节省**，在 batch 较小（16–64/rank）时效果最显著（减少 80%+ 访存）；在 256 并发时专家 GEMM 变为计算瓶颈，访存节省转化效率降低。


---

## 3. 外部对比

### 3.1 DeepEP V2.5（deepseek-ai/DeepEP）

**[源码]** gh README（2026-10-21 获取）

DeepEP V2 是 DeepInfer 的底层 EP 通信库，V2.5 在 V2 基础上新增：

- **`EPBuffer`, `EngramBuffer`, `PPBuffer`, `BucketBuffer`** 拆分（原 V1 的 `ElasticBuffer`）；
- **`BufferAllocator`**：在缓冲构造前预规划对称 tensor 分配；
- **`EPBuffer.lb_prefetch_weights` / `lb_reduce_grads`**：支持 MoonEP/UltraEP 风格的动态冗余 expert：在 expert 计算前通过 NVLink 预取权重，计算后积累 FP32 梯度；
- **`EngramBuffer`**：多层 n-gram 表的分布式存储和 RDMA 拉取，支持 CPU 存储（CPU RDMA storage）；
- **NCCL GIN 后端**（取代 NVSHMEM）：要求 NCCL ≥ 2.32.3；不再依赖 NVSHMEM；
- **JIT 编译**（DeepJIT）：kernels 运行时针对当前设备编译，安装时无需 GPU。

**与 DeepInfer 的对应**：`userbuffer.py:85` 使用 `EPBuffer(allow_hybrid_mode=True)`（同时支持高吞吐和低延迟两种模式，对应 V2 的统一接口）；`allow_multiple_reduction` 在启用 batch_invariant 且有 rearrange_interval 时关闭（防止同一 handle 多次 combine）。

DeepEP 已提供 Ascend 版本（`DeepEP-Ascend`，API 完全一致）。

### 3.2 pplx-kernels（perplexityai/pplx-kernels）→ pplx-garden

**[未核实最新状态]** pplx-kernels 已标注 DEPRECATED，新版迁移至 `pplx-garden`。旧版特点：
- 支持 NVLink（EP8 内节点）/ IBGDA（EP64–128 跨节点）/ IBRC / EFA 四种传输；
- EP128 + IBGDA dispatch 延迟约 126 µs（1 token/GPU），EP64 约 121 µs；
- **不支持 FP8 dispatch**（DeepEP V2 的 FP8 dispatch 是关键差异）。

### 3.3 UCCL-EP / GPU 发起通信（arXiv 2610.01380）

标题：*GPU-Initiated Communication: Dissecting Down to the Bone*（2026-10-01）**[arXiv 核实]**

核心发现：
- GPU 发起通信的硬件下限：发起 0.7 µs，完成 4.0 µs；
- 库开销额外增加最多 4.6 µs（NCCL GIN、DeepEP、UCCL-EP 等）；
- 共享队列时延迟升高 1–3 个数量级（bulk 流量与细粒度流量不能共享 QP）；
- 达到 260M msg/s 的 IB 上限需要 doorbell batching + 队列并行，但有资源成本；
- 对 DeepEP 的启示：EP all-to-all 约 3000 个活跃连接时 NIC msg rate 降低 59%，是超大规模（EP256+）的隐患。

### 3.4 DeepGEMM Mega MoE（deepseek-ai/DeepGEMM）

**[源码]** gh README（2026-10-21 获取）

Mega MoE 于 2026-04-16 发布（PR #304），最新优化：
- 2026-09-10 新增 Mega Gate（fused gate GEMM + routing）、locality domain（SM 局部性感知调度）；
- 2026-09-30 增加 DeepGEMM-Ascend 版本；
- 支持 `fp8xfp4`（FP8 激活 × MXFP4 权重，DeepInfer 主路径）和 `fp8xfp8` 两种 MMA 类型；
- `deep_gemm.SymmBuffer` 基于 PyTorch ≥ 2.9 的 `symm_mem`，只在 NVLink 域内有效；
- 对于 shared expert fusion（`fuse_shared=True`），`mega_moe` 同时接收 shared 权重并在 kernel 内融合。

Q&A 说法：**"Mega MoE 显著快于非融合 MoE，无论 EP 大小和 batch 大小"** **[Q&A]**——意味着即使在 EP8 预填充下，融合 kernel 的通信-计算重叠也优于显式 dispatch/combine 的流水线。

### 3.5 SGLang TBO + LPLB + Waterfill（LMSYS 博客 2026-06-26）

**[未核实完整博客]** 部分内容通过 curl 获取：

**Waterfill**（SGLang PR #25391 等）：针对 shared expert 的调度时均衡算法。
- shared expert 在每个 rank 上对所有 token 都执行，若不均衡则 overloaded rank 还要额外承担；
- Waterfill 将 shared expert 计算"派发"到路由负载较轻的 rank，"填平"负载低谷；
- 与 EPLB 的物理副本放置正交，可叠加。

**LPLB**（SGLang PR #24515）：线性规划负载均衡器，针对冗余 expert 副本的调度时路由。
- 每层独立求解 LP，决定哪个 token 路由到哪个冗余副本；
- 在 2 个 Hopper 节点上，LPLB 相比纯 EPLB 静态放置额外提升 +0.84%–+7.34%（MMLU/GPQA/GSM8K）。

**DeepInfer 对应**：调度时路由在门控 `logical_to_physical_map` 中静态确定（由 e_score_correction_bias 和 noaux_tc 间接影响），没有显式的调度时 LP/Waterfill **[推断]**。DeepInfer 依靠更高频的 EPLB 重排（每 10,000 步 decode）+ bias 修正来平衡，牺牲灵活性换取零调度开销。

### 3.6 vLLM DBO + Elastic EP + EPLB

**[源码]** gh `vllm-project/vllm`，EPLB 相关路径：
- `tests/distributed/eplb_utils.py`
- `examples/ray_serving/elastic_ep/`（`.buildkite/scripts/scheduled_integration_test/` 多个 eplb 测试）

vLLM 的 EPLB 实现与 DeepInfer 思路类似：静态冗余副本分配 + 周期性重排。Elastic EP（`examples/ray_serving/elastic_ep/scale.py`）允许运行时动态改变 EP 大小，对应 Q&A 中的 EP 故障处理（健康 GPU 重新确定 EP 大小）。

关键差异：vLLM 仍支持 TP（小规模部署），而 DeepInfer 生产不用 TP；vLLM 的 EPLB 可能不含 DeepInfer 的"首次 1/4 周期提前重排"优化 **[未核实]**。

### 3.7 METRO 的批评（arXiv 2512.09277）

标题：*Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens*（2025-12-10）**[arXiv 核实]**

METRO 的核心观点：
- 在 **decode 的访存受限区间**，决定每步耗时的是"每卡要读多少个不同专家的权重"，而非"每卡要处理多少 token"；
- 按 token 数均衡（EPLB 的默认目标）可能增加每卡激活的唯一专家数，反而加重访存压力；
- METRO 改为均衡"每卡激活专家数"（allGather 方案），报告：decode 延迟降低 11–22%，固定 SLO 下吞吐最高提升 4.11×。

**DeepInfer 的反驳角度**（**[推断]**）：
1. EP64 时每卡只有 7 个专家，即使最极端情况也只读 7 个，专家访存已经很低（≈0.16 GB，见 2.4 估算），进一步"均衡激活专家数"的空间有限；
2. METRO 的 allGather scheme 引入全局拓扑感知路由，需要额外 all-gather 通信，对 NVLink 延迟敏感；
3. METRO 的实验在 8–16 B200 GPU 上，EP 规模小（EP8/EP16），与 DeepInfer EP64/EP128 场景不同；
4. DeepInfer 的 e_score_correction_bias（noaux_tc 修正偏置）本身有一定的激活专家均匀化效果。

**适用场景建议**：METRO 对 **小 EP、大 batch、dense 冗余不足** 的场景更有价值；宽 EP（EP64+）下 EPLB 已足够，METRO 增益递减。

### 3.8 MegaScale-Infer（arXiv 2504.02263，字节）

标题：*MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism*（2025）**[arXiv 核实]**

核心思路："注意力-FFN 异构解耦"（AFD）：
- 将每层的注意力模块和 FFN/expert 模块部署在不同 GPU 集群，独立扩缩；
- 两类模块形成 ping-pong 流水线（micro-batch 交替在注意力 GPU 和 expert GPU 间穿梭）；
- expert GPU 可以用更大批次（异步积累 micro-batch），提升 expert GEMM 利用率。

**与 DeepInfer 的区别**：DeepInfer 将注意力（DP）和 expert（EP）放在同一套 GPU（`#DP = #EP`），不跨节点异构分离；Mega MoE 通过 SymmBuffer 在节点内把通信-计算重叠掉，无需跨节点 micro-batch 流水线。MegaScale-Infer 在"注意力和 expert 规模不匹配"时有优势，DeepInfer 在"同质超节点"（NVL72/GB200x64）上更简单高效 **[推断]**。


---

## 4. 取舍分析

### 4.1 token 均衡 vs 激活专家均衡

| 维度 | EPLB（token 均衡）| METRO（激活专家均衡）|
|---|---|---|
| 优化目标 | 最大 rank 的 token 数最小 | 最大 rank 的唯一激活专家数最小 |
| 适合场景 | 大 EP（EP64+），每卡专家少，访存主导 | 小 EP（EP8–16），每卡专家多，激活去重收益大 |
| 通信开销 | all-reduce（本地统计汇总）| all-gather（全局 top-k 知识）|
| 实现复杂度 | 低（贪心 + 历史窗口）| 中（LP/allGather + 动态路由）|
| DeepInfer 选择 | EPLB（token 均衡），EP64/128 | 不使用 METRO |

**[推断]** DeepInfer 用 EP64 把每卡专家降到 7 个，使"激活专家均衡"的天花板极低（最多 7），token 均衡已是充分条件。METRO 的增益空间被宽 EP 本身压缩。

### 4.2 融合（Mega MoE）vs 微批重叠（普通 EP）

| 维度 | Mega MoE（SymmBuffer 融合）| 普通 EP（dispatch→GEMM→combine）|
|---|---|---|
| 适用域 | 仅 NVLink 同节点（`is_within_nvl_domain`）| NVLink + RDMA 均可 |
| 延迟 | 更低（通信完全隐藏在 tensor core 内）| 可用微批流水线降低 |
| 内存 | 需要对称内存（symm_mem），对 PyTorch ≥ 2.9 有依赖 | 仅 EPBuffer，无额外约束 |
| Batch 适应性 | 所有 batch 大小均有效（Q&A：显著快于非融合）| 小 batch 时通信延迟相对较大 |
| Ascend 支持 | 受限（跨节点时不用 Mega MoE）| 完整支持 |

**DeepInfer 策略**：在 NVLink 域内优先 Mega MoE；跨 RDMA（Ascend 超节点跨机）退到普通 EP + `EPBuffer`。

### 4.3 冗余专家显存成本

| 配置 | 额外副本数 | 单专家权重（MXFP4）| 额外显存/卡（56层） |
|---|---|---|---|
| Prefill EP4，冗余 4 | 每卡 1 个冗余 | ≈22.7 MB | **1 × 22.7MB × 56 = 1.27 GB** |
| Decode EP64，冗余 64 | 每卡 1 个冗余 | ≈22.7 MB | **1 × 22.7MB × 56 = 1.27 GB** |

**[估算]** 每卡 1 个冗余 expert 的代价约 1.27 GB，相比 B200 HBM（192 GB）或 H100（80 GB）是可接受的开销。
若 min_redundant 从 1 增到 4（prefill 的配置移到 decode），代价约 5 GB/卡，仍然可接受但可能影响 KV cache 容量。

### 4.4 重排停顿开销

- 每次 `_rearrange_experts` 触发一次 `all_reduce`（逻辑 expert token 统计汇总），然后调用 `rebalance_experts_global`（CPU+GPU 贪心），再调用 shuffle（P2P 或 symm_mem），最后同步。
- 单节点 IPC 路径：2 次 barrier + n_layers 次拷贝（逐层刷新镜像），时延数百毫秒 **[推断]**（`_log_rearrangement_metrics` 记录 elapsed_time）。
- 跨节点 P2P 路径：需要 `batch_isend_irecv`，带宽受限于 RDMA，更慢。
- 重排期间 forward 可继续（函数加了 `@allow_cuda_sync`），但 map 更新前后的步不能保证完全一致路由——这是"批不变性"（`DEEPINFER_BATCH_INVARIANT`）的约束来源：**[源码]** `userbuffer.py:79` 中 `allow_multiple_reduction=not(batch_invariant and rearrange_interval>0)`，启用批不变性 + 重排时限制 combine 的多次调用。

---

## 5. 开放问题与建议

### 5.1 Mega MoE 的跨节点扩展

当前 Mega MoE 依赖 `symm_mem`（NVLink），超出单节点（EP > NVL 域大小）后退化为普通 EP。随着 GB200 NVL72（72 卡一域）和 GB300 NVL144 的普及，Mega MoE 覆盖域将扩大，但 256 卡以上仍需普通 EP + RDMA。**开放问题**：RDMA 下有无类似 Mega MoE 的融合方案？（参考 DeepEP 的 inter-node 低延迟路径）

### 5.2 EPLB 重排停顿的消除

当前 `_rearrange_experts` 与 forward 不完全重叠（权重更新需要同步），在 decode 10,000 步重排时可能引入数百毫秒停顿，在高负载下放大尾延迟。**建议**：采用"影子权重 + 双缓冲"方案，在后台异步完成 shuffle，仅在下一重排周期开始时原子切换指针；或参考 DeepEP V2.5 的 `lb_prefetch_weights` 预取模式。

### 5.3 调度时均衡与静态放置的组合

DeepInfer 目前仅做静态放置（EPLB），不做 SGLang Waterfill / LPLB 式的调度时均衡。**开放问题**：在 EP64 下，每步批次仍有短暂不均衡（moe_lb=1.09，即最峰值 rank 比均值慢 9%），是否可以在 gate_forward 内以极低开销（数十 µs）做 per-batch 的 token 分流？DeepEP V2.5 的 `lb_prefetch_weights` 提供了 NVLink 权重预取基础。

### 5.4 Engram 与 EPLB 的交互

Engram 使用独立的 ENP 进程组，n-gram 哈希表分片不受 EP 分片影响。当 EP 重排时 expert 位置变化，Engram 表不变——两者语义独立 **[源码]**。但 Engram `EngramBuffer` 的 sl3（与 KV cache sl2 隔离）在高带宽需求下是否会与 EP all-to-all 争抢 NVLink 带宽？**[未核实]**，建议实测不同 sl 配置下的吞吐变化。

### 5.5 冗余专家数量的自适应

当前冗余数量在启动时固定（取整后不变），由 `resolve_num_extra_experts` 确定。**建议**：基于 moe_lb 实测值动态调整冗余数（moe_lb 过高则增加冗余，显存充裕时增加 min_redundant；moe_lb 接近 1 时降低冗余以释放显存）。这需要在线重启部分 EP 实例或热扩展 expert 数组。

### 5.6 对从业者的建议

1. **小规模部署（EP < 64）**：考虑在 EPLB 基础上叠加 METRO 的激活专家均衡思路——每卡专家多时，均衡激活专家数比均衡 token 数更直接降低 HBM 带宽压力；
2. **单节点部署（≤8 卡）**：EP8 + Mega MoE（NVLink 域内）优先于 EP8 + 普通 dispatch；symm_mem 需要 PyTorch ≥ 2.9；
3. **大规模跨节点（EP64+）**：宽 EP 是主要杠杆；EPLB 的首次 1/4 周期提前重排（DeepInfer 的工程细节）值得在其他框架中采用；
4. **EPLB 复现**：`rebalance_experts_global` 的贪心算法在 GPU 上执行，但逻辑全在 Python 层，可直接复制到 vLLM/SGLang；`_batch_greedy_indices` 是核心区分点；
5. **Engram 表复用**：Engram 的 ENP 分组独立于 EP，允许在 EP 重排时保持哈希表不变——这是"通信域解耦"的好范例，适用于其他附属分布式表结构。

---

## 附：关键符号与环境变量速查

| 符号/变量 | 默认值 | 含义 |
|---|---|---|
| `DEEPINFER_EXPERT_COUNT_WINDOW_SIZE` | 100 | 历史统计窗口步数 W |
| `DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL` | （无默认，外部设置）| 重排周期 T；decode=10000，prefill=1000 |
| `DEEPINFER_REARRANGE_P2P_GROUPS` | 1 | P2P shuffle 的分桶数 |
| `DEEPINFER_MOE_LB_STAT_INTERVAL` | 128 | LB 统计 flush 间隔（all_gather 周期）|
| `DEEPINFER_USERBUFFER_COMM_SMS` | 0（自动）| 通信用 SM 数；0 = 自动推导 |
| `DEEPINFER_BATCH_INVARIANT` | False | 批不变性模式（限制 combine 多次调用）|
| `DEEPINFER_REARRANGE_AFTER_PREFILL_IN_NORMAL_ENGINE` | False | prefill 后立即触发 rearrange |
| `DEEPINFER_FORCE_IMBALANCE_RATIO` | 0（关闭）| 压测用：强制首卡路由倾斜比例 |
| `DEEPINFER_ENGRAM_FORCE_GPU_STORAGE` | False（Ascend=True）| Engram 表强制放 GPU 显存 |
| `use_mega_moe` | True（NVLink 域内 GPU）| 是否启用 Mega MoE 融合内核 |
| `moe_lb` | 1.090（decode）| 实测最峰 rank / 均值，来自 Q&A |

