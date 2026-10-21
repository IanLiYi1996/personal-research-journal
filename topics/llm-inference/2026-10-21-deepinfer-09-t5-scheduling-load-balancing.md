# DeepInfer 专题 T5：调度——引擎内调度器、零开销异步调度 ZOS、DP 同步、负载均衡与弹性

> 面向资深推理系统工程师；标记约定：**[源码]** 读代码确认 | **[Q&A]** 官方文档 | **[估算]** 推算含算式 | **[推断]** 合理推测 | **[未核实]** 仍待验证

---

## 1. 引擎内调度器（C++ Scheduler）

### 1.1 架构概述

DeepInfer 的调度器完全实现在 `csrc/engine/scheduler.hpp` 中，以 C++ 单线程同步对象形式暴露给 Python。调度主入口为 `_schedule()`，输出一个 `SchedulerOutputs`。整个调度与 Block 分配在同一把互斥锁（`BlockManagerMutexAcquire`）内完成，保证 KV block 分配的原子性。**[源码]** `scheduler.hpp:275–345`

三条队列：
- `waiting`（双端队列）：等待 prefill 的请求，按到达顺序排列；
- `prompted`（双端队列）：prefill 完成、待提升为 running 的请求（包含 fork 标记）；
- `running`（向量）：当前 decode 中的请求。

每步结束由 `process_outputs` 负责更新这三条队列的状态及 KV block 归属。**[源码]** `scheduler.hpp:872–1033`

### 1.2 步型投票（DP 同步）

NORMAL 引擎的核心挑战是：DP 组内所有 rank 必须在同一步做相同类型（prefill / decode）的 forward，否则 EP 通信无法配对。投票逻辑如下（**[源码]** `scheduler.hpp:295–345`）：

```
bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);
bool is_all_point = (decode_step % prefill_attempt_interval == 0);
```

**ALL 点**（默认间隔 `DEEPINFER_PREFILL_ATTEMPT_INTERVAL=70`）：
```cpp
do_prefill = (_dp_allreduce(want ? 1 : 0, c10d::ReduceOp::MIN) == 1); // 全员有活才做
```

**ANY 点**（默认间隔 `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL=100`）：
```cpp
do_prefill = (_dp_allreduce(want ? 0 : 1, c10d::ReduceOp::MIN) == 0); // 任一 rank 有活即做
```

两类点重合时按 ANY 算（防止 ANY 被 ALL 覆盖消失）。`want` 的判断条件是同时满足：
1. GPU KV cache 使用率 ≤ `GPU_CACHE_PREFILL_WATERMARK = 0.9f`（**[源码]** `scheduler.hpp:80`）；
2. `_has_prefill_waiting()`：waiting 队头存在且 `schedule.schedulable=true`。

**二次确认投票**：第一轮通过 ALL/ANY 决策后，进入 `_schedule_prefill` 仍可能返回空（slot 满 / KV 块不足）。若全组都空手，继续做空的 prompt step 会清零 `decode_step`，导致下一步又落在 ALL 点、又全空手——陷入 running 满 + waiting 非空的死循环。因此有第二轮投票：
```cpp
bool any_scheduled = (_dp_allreduce(outputs ? 0 : 1, c10d::ReduceOp::MIN) == 0);
if (!any_scheduled) return _schedule_decode(acquire);
```
没有任何 rank 排出 prefill 时回退到 decode。**[源码]** `scheduler.hpp:329–340`

**陪跑 rank**：`want=false` 但 `any_scheduled=true` 时，本 rank 发出一个空的 prompt step（`num_batched_tokens=0`），以保持步型一致。**[源码]** `scheduler.hpp:339–340`

`decode_step` 在每次 prompt step 后清零（**[源码]** `scheduler.hpp:677`），确保 decode_step 各 rank 在相同时刻以相同值参与投票，allreduce 的次数与语义全组对齐。

### 1.3 prefill 调度（`_schedule_prefill`）

**[源码]** `scheduler.hpp:351–554`

关键参数与约束（按执行顺序）：

| 约束项 | 实现 |
|--------|------|
| `prefill_max_sum_seqlen_k` | 累计 K 长预算（软上限），**[源码]** line 444–449 |
| `max_num_prompts` = `max_num_seqs` | 并发 seq 上限 |
| `cp_align = cp_size * 2`（mirror CP） | seqlen_q 按 cp_align 对齐，**[源码]** line 470–471 |
| `chunk_checkpoints` | 强制在指定位置切块（例如 multimodal 边界） |
| `_adjust_chunk_size` | 禁止在图像 PAD 区内部切块，**[源码]** line 217–260 |

**分块逻辑**：当加入当前 seq 后 `num_batched_tokens` 超出预算时，截断到：
```
new_workload = (max_num_batched_tokens - num_batched_tokens) / cp_align * cp_align
```
截断后若仍为 chunked（`seqlen_q < seq_workload`），且未启用 `chunked_prefill` 则 break（整个 seq 推迟）；启用后继续调度但本次 step 只排一个 seq 的 chunk（NORMAL 引擎下 `is_chunked` 时直接 break，防止多 chunked seq 混入同一步）。**[源码]** `scheduler.hpp:543–545`

**分叉 seq 的处理（fork_seq 分支）**：`branch_id` 以 `"_"` 开头表示刚 fork 尚未正式入队的 seq。调度器在 `_schedule_prefill` 中识别此标志，分配增量 blocks、clone state、clone blocks（写入 `clone_blocks` / `clone_states` 向量供 GPU 侧异步拷贝），然后去掉前导 `"_"` 正式化。**[源码]** `scheduler.hpp:398–437`

### 1.4 decode 调度（`_schedule_decode`）

**[源码]** `scheduler.hpp:558–660`

**LCFS 排序与抢占**：
```cpp
std::sort(running, [](a,b){ return a->get_output_len() > b->get_output_len(); }); // 输出越长 → 排得越靠前 → 越容易成为抢占受害者
```
然后从尾部（输出最少的 seq）开始分配 `_append_slot`，若没有空闲 block 则对头部（输出最多的 seq）调用 `_preempt`。这是一种近似 LCFS 的块驱逐策略：优先抢占已生成 token 最多的请求，因为它们"物超所值"——任何时刻继续它的边际收益最低。**[源码]** `scheduler.hpp:622–643`

**`_preempt` 行为**：
- `stop_preempted_request=true` 或 SO 请求：将 seq 置 `FINISHED_PREEMPTED`，直接释放 KV blocks，进 `ignored_seqs` 返回错误原因。
- 否则：调用 `seq.reset_for_recompute()`（清除 `cache.num_tokens`、`total_inflight_tokens`、MTP 状态），插回 `waiting` 队头，标记 WAITING，释放 KV blocks 供重算。
- **[源码]** `scheduler.hpp:766–790`

注意：`stop_preempted_request` 在 COMPLETION 引擎下被强制设为 `true`（构造函数检查），因为 COMPLETION 引擎没有 prefill 通道无法重算。**[源码]** `scheduler.hpp:102–104`

**decode 步的清除已结束 waiting seq**：decode 分支开头扫全 `waiting` 队列，弹出所有 `is_finished()` 的 seq（abort/VE 失败等），防止这类 seq 的 `want` 永远为 false 而无法被 ALL 投票清掉。**[源码]** `scheduler.hpp:567–578`

### 1.5 `_append_slot`

每个 decode step，seq 的 KV 长度可能越界当前 block table 边界，`_append_slot` 负责按需分配新 block 或释放多余 block（例如乐观推进后 draft 未被接受时 block_table 可能多出一个）：

```cpp
void _append_slot(Sequence &seq) {
    int num_blocks = seq.get_num_blocks(); // 含 total_inflight_tokens + draft_len
    if (block_table.size() < num_blocks)   allocate_single_nolock();
    else if (block_table.size() > num_blocks) free_blocks_nolock(redundant);
}
```
**[源码]** `scheduler.hpp:817–828`

### 1.6 `start_inflight_step` / `finish_inflight_step`（乐观推进）

ZOS 的核心辅助状态机在 `sequence.hpp` 的 `SequenceScheduleState` 中：

```cpp
void start_inflight_step(int token_count) {
    total_inflight_tokens += token_count;   // 累加本步正在途的 seqlen_q
    last_inflight_tokens = token_count;
}
void finish_inflight_step() {               // ZOS 后处理中调用
    if (total == last) { total = last = 0; }
    else               { total = last; }     // 仍有上一步在途
}
```
**[源码]** `sequence.hpp:244–256`

这允许调度器在 N+1 步 schedule 时把 N 步的在途 token 也计入 `get_num_blocks()`，避免 block 不足。同时 `KvCacheState::advance_optimistically` / `commit_processed_tokens` 配对使用：先乐观地把 `cache.num_tokens` 推进 seqlen_q，采样后再按实际接受的 draft token 数回填。**[源码]** `sequence.hpp:216–221`

### 1.7 `process_outputs`

`process_outputs` 是 CPU 侧与 GPU 侧输出的对接点，主要逻辑：
1. 调用 `finish_inflight_step()` 更新在途计数；
2. `advance_optimistically(seqlen_q)` 乐观推进 `cache.num_tokens`；
3. 处理 MTP 接受的 draft tokens；
4. `commit_processed_tokens(token_cnt)` 回填实际值；
5. chunked prefill 未完成的 seq 插回 `waiting` 队头（**[源码]** `scheduler.hpp:1018–1019`）；
6. ZOS 异步被抢占的 seq 已在 `_preempt` 中复位，这里跳过记账，防止 `cache.num_tokens` 错误推进（**[源码]** `scheduler.hpp:919–924`）。

**[源码]** `scheduler.hpp:872–1033`

### 1.8 `fork_seq`

`fork_seq` 是系统树形 beam / 多路 speculative 的基础接口：

- 复制父 seq 的 block table 并 `pin_blocks`（引用计数 +1），避免被 GC 回收；
- state block 同步 pin；
- prompt 未完成（`prompt_incomplete=true`）的 fork 直接进 `waiting` 队头，等 prefill；
- prompt 已完成的 fork 进 `prompted` 队尾，带 `forked=true` 标记，decode 时 clone 最后一个写一半的 block。

**[源码]** `scheduler.hpp:1061–1122`

---

## 2. FlopsEstimator 与 MFU 统计

### 2.1 设计定位

`FlopsEstimator`（`csrc/engine/flops_estimator.hpp`）是**纯 MFU 统计工具**，不用于运行时调度决策。每步 schedule 时按 `(q, k, is_prompt, full_len)` 计算一次 FP8-equivalent FLOP 估计，累加到 `SchedulerOutputs::total_tflops`，供 `StatsLogger` 计算 MFU。**[源码]** `flops_estimator.hpp:38–45`

### 2.2 代价分项

FlopsEstimator 拆分为以下项，各项均折算为 FP8 等效（FP4=1，FP8=2，BF16=4，TF32/FP32=8）：

| 分项 | 来源 |
|------|------|
| MLP/MoE（含共享专家） | `mlp_flops_per_row`：3 × hidden × width × (top_k + n_shared) × n_layer |
| Router GEMM | `router_flops_per_row`：hidden × n_routed_experts × n_layer |
| HC（Hierarchical Compression） | `hc_flops_per_row`：pre+sublayer × n_hc_layer，FP32/TF32 |
| Engram（compressed tokenizer projection） | `engram_flops_per_row`：FP8 kv proj + FP32 dot |
| 注意力 QK/AV pairs | `calc_pair_flops`：按 KV compression ratio 的 prefix sum |
| SWA 注意力对数 | `full_swa_coef × swa_pairs`（capped prefix） |
| Q/WV 投影 | `full_attn_per_row + consumer_attn_per_row` |
| KProject（index attention） | `calc_kproj_flops` |
| LM Head | `calc_logits_flops`：prefill 只算最后 1 行，decode 算全 q 行 |

**[源码]** `flops_estimator.hpp:64–235`

YoCo compact 模式下 consumer 层的 query 行数从 `start` 移至 `cutoff = full_len - keep_tokens`，对应的 FLOP 也相应缩减。**[源码]** `flops_estimator.hpp:268–278`

### 2.3 FlopsEstimator 无法用于 cost(q,c) 求解 [推断]

Q&A 描述的"拟合二次代价模型 `cost(q,c)` 求最大 q"（见 §3.2）在开源代码中**没有实现**。FlopsEstimator 是事后 MFU 统计（调度后 per-seq 调用），其设计不支持"给定时间预算求最优 q"。真实的 cost 模型依赖运行时采样的 `forward_ms`，而 FlopsEstimator 仅依赖静态模型结构参数。

开源版中唯一有 step 耗时反馈的是 `MTPVarlen`（`mtp_varlen.hpp`），它维护了 `decode_perf_curve`（`(graph_size, baseline_ms)` 向量）+ `ForwardRuntimeEstimator`（斜率 slope_ms/token）+ `SamplerRuntimeEstimator`（sampler 中位数），并用这套参数求 best_bsz（即 best total_q）：

```cpp
// 近似等效于 maximize E[accepted] / estimated_time：
double cur_throughput = cur_expected / (baseline_ms + slope_ms * graph_size + sampler_ms);
```
**[源码]** `mtp_varlen.hpp:242–249`

这是一个单维（total_q）优化，**没有 per-request 的 prefix length c 作为自变量**，与 Q&A 描述的二维二次模型不同。**[推断]** Q&A 中 `cost(q,c)` 二次模型属于未开源的 Prefill Load Balancer 组件，参见 §3.2。

---

## 3. 零开销异步调度（ZOS）

### 3.1 问题与目标

同步调度路径是：`schedule(N) → forward(N) → sample(N) → [GPU→CPU sync] → post_process(N) → schedule(N+1)`。其中 `post_process` 和 `schedule(N+1)` 都是纯 CPU 逻辑，GPU 在这段时间空闲。ZOS 的目标是将这段 CPU 时间与 GPU 的 `forward(N+1)` 完全重叠。**[Q&A]** "Asynchronous scheduling"章节

### 3.2 实现时序（CPU/GPU 两条泳道）

```
时间轴 →

GPU: |--- forward(N) ---|--- sample(N) ---|--- forward(N+1) ---|--- sample(N+1) ---|
CPU: |                  |                 |post_process(N-1)   |post_process(N)    |
     |                  |schedule(N+1)    |launch_forward(N+1) |                   |
     |                  |launch_forward   |                    |                   |
     |--- ZOS: CPU busy overlapping -----|                                         |
```

具体步骤（**[源码]** `worker.py:execute_model_step:88–189`）：
1. **forward(N)**：`execute_main_model`，GPU 异步执行；
2. **post_process(N-1)**：`_post_process_step(prev_step_state)`，同步等待 `sample_end_evt(N-1)` 后处理，CPU 侧更新 scheduler；
3. **sample(N)**：`execute_sample`，GPU 异步执行；
4. **pre_sampler_sync**：`DeferredRelease.release()`，释放上上步资源；
5. **post_process(N)**：decode step 时不在此执行，存入 `prev_step_state`，留到下一步的步骤 2 执行。

关键：步骤 2 与 GPU forward(N) 在时间上重叠。步骤 4（pre_sampler_sync）在 sample 发射后、下一步 forward 发射前执行，与 GPU sample 计算重叠。**[源码]** `worker.py:162–188`

`ZOS` 的前提是调度器无需等待 sample 结果即可完成 schedule(N+1)：调度器用的是 `cache.num_tokens`（乐观推进值）而非 GPU 实际接受数。GPU 上的 seqlens_q/k 修正由 `compute_seqlens_k_zos` 内核在 GPU 上异步完成。**[源码]** `worker.py:209–244`

PROMPT 引擎步不走 ZOS（总是当场 post_process），因为 prompt step 的信息必须立即更新 scheduler（新 running seq 的状态）。**[源码]** `worker.py:183`

### 3.3 remap 机制（`prepare_zos_inputs`）

当 N+1 步的 batch 与 N 步有交集时，不重新传 token_ids，而是用 `remap_indices[i] = j`（N+1 步第 i 个请求在 N 步第 j 个位置的 logit/embed 输出中可以找到上下文）。  
新请求（N 步没有）用 `remap=-1` 标记为 fresh，需要从 host 重传 input_ids。**[源码]** `zos.cpp:147–195`

这减少了 H2D 传输量：两步 batch 完全相同且顺序不变时，只需传 `seqlens_q/k` 更新，无需重传 input_ids。`prepare_token_inputs` 在有 `remap_ptr` 时只填充 `fresh_input_ids`（每 seq 一行，`max_seqlen_q` 列），skip 掉已经在 GPU 上的 token。**[源码]** `zos.cpp:83–138`

### 3.4 ZOS 状态下抢占的处理

抢占发生在 decode 调度（`_schedule_decode`）中，此时被抢占的 seq 可能仍在 N 步的途中（`total_inflight_tokens > 0`）。`process_outputs` 检查到这类 seq：

```cpp
if (!scheduler_outputs.prompt_run && seq.schedule.status == SequenceStatus::WAITING) {
    // 本步输出作废，跳过记账
    continue;
}
```
`_preempt` 已调用 `reset_for_recompute()` 复位计数并将其插回 waiting 队头。**[源码]** `scheduler.hpp:919–924`

### 3.5 `forbid_cuda_sync` 机制

调度器和 worker 中的 CPU 代码段（forward/sample 发射期间）不能触发 GPU-CPU 同步，否则会破坏 ZOS 的 overlap。`forbid_cuda_sync()` 通过 `torch.cuda.set_sync_debug_mode("error")` 在线程级别强制检查，若违规则抛出 RuntimeError。

`_SyncDebugArbiter` 实现了跨线程的 allow/forbid 计数：全局共享一个 `forbid_count` 计数器，当 `forbid_count > 0 && allow_count == 0` 时将 CUDA 同步模式设为 `"error"`。这样 `allow_cuda_sync()` 能在 forbid 范围内开豁口（例如 tokenizer 线程合法调用 GPU），不误杀合法操作。**[源码]** `tests/test_cuda_sync_guards.py:1–213`

环境变量 `DEEPINFER_DISABLE_SYNC_GUARD=1` 可关闭此守卫（debug 用）。**[源码]** `environ.py:DEEPINFER_DISABLE_SYNC_GUARD`


---

## 4. 未开源的调度机制（来自官方 Q&A）

### 4.1 Prefill Load Balancer：四档分桶与配额再分配

**[Q&A]** "Data-parallel load balancing" → "How is the Prefill Load Balancer implemented?"

请求按总上下文长度分为四档：
- 档 1：< 16K token
- 档 2：16K–64K token
- 档 3：64K–256K token
- 档 4：≥ 256K token

每档分配固定数量的 PE（Prefill Engine）实例；各档的 PE 数量**每轮（round）动态再分配**，按该档在上一轮消耗的"计算配额占比"为权重。每档还有独立的**单次批次计算配额上限**（per-batch compute-quota cap）。**[Q&A]**

**跨 DP 组的拉取（pull）模式**：PE 不被推送任务，而是主动从队列中按"剩余配额"拉取请求。任何 PE 可以命中任何 prefix，因此无前缀亲和路由——全局 KV 存储是无亲和分配成立的前提。**[Q&A]**

**DP 组内的分配**：一批请求到达后，每个新请求路由到 DP 组内"当前配额负载最低"的 rank。**[Q&A]**

### 4.2 单卡时间预算与二次代价模型 cost(q,c) [Q&A + 推断]

**[Q&A]** 每个 prefill step 给定时间预算 `compute_quota_limit`（纳秒），拟合二次代价模型 `cost(q, c)` 预测 step 耗时（q = 当步 token 数，c = 已计算的 prefix 长度）。调度器求解：

```
q* = argmax q  s.t.  cost(q, c) ≤ budget,  q ≤ 32768
```

即"给定时间预算，找最大合法 q"。  
时间预算本身还通过 DP 组内的 **all-reduce MIN** 对齐：
```
budget = all_reduce(local_budget, MIN)   // DP 组内最低配额
```
确保最慢的 rank 决定全组的批次大小，避免一个 rank 超出预算导致步型超时。**[Q&A]**

**[推断]** `cost(q,c)` 二次模型的形式估计为：
```
cost(q, c) = α·q·c + β·q² + γ·q + δ
```
其中：
- `α·q·c`：注意力部分（Q×K pairs 与 q 和 c 各线性），对应 FlopsEstimator 的 `pair_flops`；
- `β·q²`：prefill 中 q 自身的自注意力；
- `γ·q + δ`：FFN 和固定启动开销（线性于 q + 常数项）。

代入时间预算 `T` 求解 q，变为一元二次不等式，求正根取整：
```
q* = floor((-（α·c + γ）+ sqrt((α·c+γ)² + 4β(T-δ))) / 2β)
q* = min(q*, 32768)
```
然后对 DP 组内 `budget_i = T - cost(current_q_i, c_i)` 做 all_reduce(MIN) 求最紧约束再迭代一轮。

开源代码中 `FlopsEstimator::calc_flops` 可提供各项 FLOP 数，若做时间换算只需乘以 1/GPU_FLOPS_RATE，可作为 cost 模型的系数估计起点。**[推断]**

### 4.3 Decode Load Balancer：双维平衡启发式

**[Q&A]** "短请求 → 发到并发低但 KV cache 总长度大的 GPU；长请求 → 发到 KV cache 总长度小的 GPU"。参见 DualPath 的 intra-group scheduling 算法。

DualPath（arXiv 2602.21548）的 intra-group 调度的核心思想是：simultaneously balance request count (affecting batch size and compute per step) and total KV length (affecting attention cost). 同时平衡这两个维度可以避免一个维度极端导致的性能瓶颈。**[推断]** DeepInfer decode LB 采用类似的双维启发式：

```
score(rank) = w1 * normalize(queue_depth) + w2 * normalize(total_kv_len)
assign_to = argmin score
```
长请求（大 KV）发到 total_kv_len 小的 rank（该 rank 注意力开销有余量）；  
短请求发到 queue_depth 低但 total_kv_len 大的 rank（该 rank 空有 KV 开销但批次不饱和）。

### 4.4 Vision Encoder Load Balancer

**[Q&A]** 图像以**机器粒度**路由：prefill 机器随机选择一台 VE 机，然后选该 VE 机上"当前正在处理图像 patch 数最少的 GPU"。patch 数 = `(width/patch_size) × (height/patch_size)`。

### 4.5 整体效率

**[Q&A]** 实际部署的整体吞吐约为**离线理想值的 80%**，其余 20% 来自跨 DP/EP 的不均衡、LB 不完美、空跑步等开销。具体分解：EPLB 引起的最慢/平均 = 1.074（EP4DP4）~1.090（EP64）；DP 调度（投票 overhead、陪跑步）约额外损耗 5–10%。**[Q&A + 估算]**

---

## 5. MTP Varlen 调度（开源版的动态验证长度）

**[源码]** `csrc/engine/mtp_varlen.hpp:133–280`

这是开源版中**唯一**有步时反馈的调度优化：

**算法**：
1. 从 `decode_perf_curve`（profiled `(graph_size, baseline_ms)` 向量）选 `best_bsz = total_q`，最大化 `E[accepted_tokens] / estimated_step_time`；
2. 用各 seq 的 `accept_rates`（来自 Confidence Head，累乘后的 draft token 接受率）按降序排 draft token，分配 seqlen_q；
3. ZOS 模式下第 3 步延迟到 GPU 上：`compute_seqlens_q_by_confidence_rates` 内核用 N-1 步的 accept rates 在 GPU 上重新分配 seqlen_q，结果 D2H 后修正 CPU 侧的 `scheduler_outputs.seqlens_q`。**[源码]** `worker.py:218–234`

**时间反馈**：`update_mtp_runtime(num_seqs, graph_size, forward_ms, sampler_ms)` 每步从 driver worker 回调（**[源码]** `llm_engine.py:997–1002`），用滑动窗口中位数维护斜率 `slope_ms/token` 和按 `num_seqs` 分组的 sampler 时间估计。

---

## 6. 外部方案对比

### 6.1 Sarathi-Serve（arXiv 2403.02310）

- **核心**：chunked prefill + stall-free schedule：将 prefill 请求切成近似等大 chunk，插入 decode batch 而不暂停 decode（论文标题 "Taming Throughput-Latency Tradeoff"）。
- **与 DeepInfer 的异同**：DeepInfer 也做 chunked prefill，但 **NORMAL 引擎下每步只 chunk 一个请求**（`is_chunked` 后直接 break），且不混合 P/D 批次——见 §7.1 取舍分析。Sarathi 混合 P/D 节省延迟但增加步型复杂度；DeepInfer 彻底分离 P/D 以适应 DP+EP 的 allreduce 对称性约束。
- **来源核实**：arxiv 2403.02310 = "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve"，作者 Agrawal 等，Microsoft Research India，2024。**[核实]**

### 6.2 NanoFlow（arXiv 2408.12757）

- **核心**：设备内并行（intra-device parallelism）：将 batch 切成 nano-batch，各自在独立 SM/内存 partition 上运行，重叠 compute、memory、networking。
- **与 DeepInfer ZOS 的关系**：Q&A 明确提及"following existing implementations (SGLang, NanoFlow)"——DeepInfer ZOS 借鉴了 NanoFlow 的 CPU/GPU overlap 思路，但不切分 nano-batch（DeepInfer 单步 batch 已足够大，切分反而增加调度复杂度）。
- **来源核实**：arxiv 2408.12757 = "NanoFlow: Towards Optimal Large Language Model Serving Throughput"，作者 Zhu 等，2024。**[核实]**

### 6.3 SGLang overlap scheduler

- **核心**：`event_loop_overlap`（**[核实]** `sgl-project/sglang scheduler.py:2011`）实现与 DeepInfer ZOS 等价的 CPU/GPU 流水线重叠；`enable_overlap = not disable_overlap_schedule and not use_mlx()`。
- **与 DeepInfer 的差异**：SGLang 使用 Python 级 `asyncio` 事件循环驱动调度，DeepInfer 将调度逻辑下沉到 C++，进一步减少 Python GIL 开销和 Python 层延迟。两者都用 remap/skip 机制避免重传上步已在 GPU 的输入。
- **来源核实**：SGLang `scheduler.py` line 2011 `def event_loop_overlap` 存在。**[核实]**

### 6.4 vLLM V1 异步调度

- vLLM V1 的 async scheduling 通过 `AsyncLLMEngine` + `asyncio` 实现，scheduler 在 CPU 线程运行，与 GPU 流水线化。其 `SchedulerOutputs` 设计和 ZOS 思路相近。
- 与 DeepInfer 的不同：vLLM V1 调度单元是 `SchedulerGroup`（支持 DP），但 DP 同步机制不同——vLLM V1 目前无 DP allreduce 投票点，各 DP rank 独立调度（适用于 TP+DP 而非 EP+DP 场景）。**[推断]**

### 6.5 TRT-LLM Overlap Scheduler

- TRT-LLM 的 `OverlapScheduler` 将 forward 与 post-process 流水线化，思路与 DeepInfer ZOS 一致；TRT-LLM 用 CUDA Graph 加速 forward，在 overlap 模式下 CUDA Graph 的 seqlen 参数需要提前确定，与 ZOS 的 seqlen_q/k 动态修正有相似挑战。**[推断/未核实]**

### 6.6 Dynamo KV Router + Planner

- Dynamo（NVIDIA）中 KV Router 基于 prefix hash 做前缀亲和路由，Planner 负责全局 disaggregated P/D 调度。
- 与 DeepInfer 的核心差异：DeepInfer **主动放弃前缀亲和**（任意 PE 均可命中任意 prefix，依赖全局 3FS 共享存储），Dynamo 偏向 prefix locality。DeepInfer 的选择基于 KV 存储已全局化，亲和路由带来的网络开销小于调度不平衡带来的损失。**[Q&A]**

### 6.7 llm-d EPP/GIE

- llm-d (IBM) 的 EPP（Expert Parallel Placement）和 GIE（Global Inference Engine）目标与 DeepInfer EPLB + Prefill LB 类似：全局专家负载均衡 + 跨节点调度。
- 两者差异在于 llm-d 更侧重 Kubernetes 原生的弹性，而 DeepInfer 的 EPLB 是引擎内实时计算（100–1000 步统计窗口，symm_mem NVLink shuffle）。**[Q&A + 推断]**

### 6.8 sgl-router

- sgl-router 是 SGLang 的外部请求路由器，支持 prefix-aware routing（基于 radix cache 命中率）和 round-robin。
- 与 DeepInfer Prefill LB 的区别：sgl-router 做的是单机 TP 副本间的 routing，DeepInfer Prefill LB 做的是多机 EP+DP 组间的 routing，量级和信息维度完全不同。**[推断]**

### 6.9 SARA（arXiv 2609.26763）

- 标题："SARA: SLO-Aware Resource Allocation for Disaggregated Agentic LLM Services"（Liu 等，2026.09）
- 建立了 prefill M/G/k 队列 + decode 广义 birth-death 过程的解析模型，直接从 SLO 约束反推所需 PE/CE 数量。
- 与 DeepInfer 的关系：DeepInfer 的自动扩缩（autoscale）基于简单的利用率启发式（prefill 看 non-cached tokens，decode 看 running requests，利用率 > 0.8 禁止缩容）。SARA 的解析模型可为 DeepInfer 的扩缩策略提供理论基础。**[核实]** arxiv 2609.26763 存在。

### 6.10 Crossflow（arXiv 2609.27085）

- 标题："Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving"（Xu 等，2026.09）
- 核心：每个 decode 节点发布短生命期的可撤销 lease，允许 decode 节点临时承担 prefill 工作，弹性调整 P/D 边界而无需修改节点角色。测量：相比静态 P/D，几何均值吞吐提升 16–17%，高负载时最高 +43%。
- 与 DeepInfer 的关系：DeepInfer 当前使用静态 P/D 分离；Crossflow 指出流量比在 agent 场景下日内变化 24.5 倍，静态分配浪费 17% 容量。这是 DeepInfer 架构的一个已知缺口，也是 Q&A 中 autoscale 部分（"rate-limited, bounded above and below"调整）仅能部分缓解的场景。**[核实]** arxiv 2609.27085 存在。

### 6.11 DualPath（arXiv 2602.21548）

- 标题："DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference"（Wu 等，2026.02）
- 核心：引入"存储 → decode → 经计算网络 RDMA → prefill"第二 KV 加载路径，平衡 prefill 和 decode NIC 带宽。全局调度器动态选择路径。
- DeepInfer 官方 Q&A 直接引用此文作为 Decode Load Balancer 的算法参考，表明其 intra-group 双维平衡启发式与 DualPath 的 intra-group 调度算法相同或高度相似。**[Q&A]** "See DualPath's intra-group scheduling algorithm for details."

### 外部方案对比表

| 系统 | overlap 调度 | P/D 分离 | 前缀亲和 | DP 步型同步 | 二次代价模型 |
|------|------------|---------|---------|------------|------------|
| DeepInfer | ZOS (C++) | 静态，引擎类型标记 | 无（全局存储） | allreduce MIN×2 | prefill LB（未开源） |
| SGLang | event_loop_overlap (Python) | 可选 | radix cache 亲和 | 无（独立 TP 副本） | 无 |
| NanoFlow | nano-batch intra-device overlap | 无 | 无 | 无 | 无 |
| Sarathi-Serve | 无 | 无，P/D 混合批 | 无 | 无 | 无 |
| vLLM V1 | asyncio 异步 | 无（可外部分离） | 可选 | 无 | 无 |
| Crossflow | 无 | 弹性（lease）| 部分 | 无 | 无 |
| Dynamo | 无 | 有 | 强（KV Router） | 无 | 无 |


---

## 7. 取舍分析

### 7.1 为何不混合 P/D 批（DP+EP 步型同步的代价）

DeepInfer 严格分离 prefill 和 decode，同一步内不混合两类请求。原因是 DP+EP 架构下每步的 allreduce 通信形状取决于 batch 的 token 数，P 步（大 token block）和 D 步（小 token × 多 seq）的通信量级相差 10-100 倍，混合后无法使用统一的 CUDA Graph（graph 捕获的通信拓扑固定）。此外，DP allreduce 要求所有 rank 的 tensor shape 一致，混合 P/D 使 shape 难以统一。

**代价**：prefill 完成后 decode 队头等待不超过 `prefill_attempt_interval`（默认 70）个 decode step 才能开始，产生 TTFT 的"批次等待"抖动（最坏 70 × decode_step_time ≈ 70 × 30ms ≈ 2.1s at EP64）。但 ALL 投票点确保只在真正有工作时才集体做 prompt，ANY 保底点（100 步）确保孤立 rank 最多等 3s。**[估算]**

### 7.2 LCFS 抢占与重算 vs swap

开源版 Scheduler 只实现重算（recompute）抢占：被抢占的 seq 丢弃所有 KV block，插回 waiting 队头重新 prefill。swap（KV offload 到 CPU）没有实现（`SequenceStatus::SWAPPED` 枚举存在但未使用，**[源码]** `sequence.hpp:33`）。

**取舍**：
- 重算适合 KV 价值高、但计算成本相对低的情况（DeepInfer KV 高度压缩，每 token 仅 ~1KB，重算 FLOP 却没有减少）；
- swap 需要 PCIe 带宽，DeepInfer 在超高并发场景下 PCIe 带宽已被 KV 的 H2D 加载占用；
- DeepInfer 通过 `stop_preempted_request` 的可配置性，允许部署方选择直接结束而非重算，减少不可预测的延迟。

### 7.3 前缀亲和 vs 全局池

DeepInfer 选择无前缀亲和的全局 KV 存储（3FS）。前缀亲和路由（如 SGLang、Dynamo）在命中率高时减少 KV fetch 延迟，但与 DP 负载均衡冲突：前缀亲和意味着某些 rank 会积压热前缀而另一些 rank 闲置。  
全局存储使任何 PE 都能以等概率命中任何 prefix，LB 可以完全按计算配额决策，命中率损失由 3FS 的低延迟（内存级 SSD）弥补。这一选择在 KV 高度压缩（~1KB/token）且 3FS 提供 40GB/s 读带宽的条件下成立。**[Q&A]**

### 7.4 时间预算 vs token 预算

token 预算（如 `max_num_batched_tokens`）在 prefill 中仍然存在（开源版 `chunk_size` 基于 token 数），但 Q&A 描述的生产系统叠加了**时间预算**（compute_quota_limit）。时间预算的优势：直接控制 TTFT，避免因 token 数相同但 KV 长度不同（attention quadratic）导致的时间差异；可以通过 all-reduce MIN 在 DP 组内对齐最慢者，防止步型超时。缺点是需要一个准确的代价模型，拟合和维护成本更高。**[Q&A + 推断]**

---

## 8. 开放问题与建议

### 8.1 开放问题

1. **ALL 投票点间隔的最优值**：默认 70 是经验值。对于 KV 命中率很高（大量 prompt 短于 1 块）的场景，interval=70 导致 decode 侧长期饥饿 prefill；对于 TTFT 敏感服务，interval 可能要降到 10–20。需要根据 SLO 和流量分布建立自适应调节模型。

2. **DP allreduce 开销**：每隔 70 步做一次（有时两次）allreduce。对 DP128，GPU 间 allreduce 延迟约 10–50 μs（NCCL intra-node NVLink），相比 30ms 的 decode step 占比 < 0.2%，可忽略。但在 DEEPINFER_ZERO_OVERHEAD_SCHEDULING=1 时调度器切换为 CPU gloo group（**[源码]** `llm_engine.py:313`），gloo allreduce 在高 DP 下延迟可达 1ms 级，需评估。

3. **ZOS 与抢占的正确性**：当 N+1 步 schedule 发生抢占（将 N 步某个在途 seq 打入 waiting）时，N 步的 sample 输出已无用。`process_outputs` 用状态检查跳过这些 seq 的记账，但 MTP 的 `process_cnt`/`schedule_cnt` 在这种情况下的一致性需要更严格的形式化分析，目前依赖代码注释而非不变式验证。

4. **MTPVarlen 与 ZOS 的两步延迟**：Q&A 提到"CPU 调度用 N-2 步的 accept_rate 计算 best_bsz"，ZOS 下 GPU 用 N-1 步修正 per-request seqlen_q。两个时间窗口不同步，在 accept_rate 快速变化（例如长回答序列中间出现的低接受率段）时可能低估 best_bsz，实际效果略差于同步版。

5. **prefill_max_sum_seqlen_k 的未开源语义**：`DEEPINFER_PREFILL_MAX_SUM_SEQLEN_K` 默认为 `2^30`（近乎不限制），但生产中 Q&A 描述的 per-batch compute-quota cap 实际由时间预算而非 token 数决定。开源用户若不理解这一点，可能误以为 token 数预算已经够用，而忽视了 attention 的 quadratic 开销在长 KV 时的实际代价。

6. **Crossflow 式弹性的可行性**：DeepInfer 当前 autoscale 是粗粒度（分钟级）的节点级弹性，Crossflow 的 lease 机制允许秒级的 P/D 边界调整。在 DeepInfer 的 EP+DP 架构下，一个 decode supernode 临时分出几个 rank 做 prefill 需要打破当前的通信组划分，是否可行需要深入评估。

### 8.2 对从业者的建议

1. **优先调整 `DEEPINFER_PREFILL_ATTEMPT_INTERVAL`**：对 TTFT 要求严格（< 2s）的服务，将其降低到 20–30，并相应降低 `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL` 到 50。代价是 decode 批次更频繁被打断，TBT 的方差增大。

2. **`stop_preempted_request` 的选择**：COMPLETION 引擎强制为 true；NORMAL 引擎在长上下文（> 32K）部署时建议也设为 true，否则重算代价与首次 prefill 相当，而用户体验损失相同。

3. **ZOS 与 CUDA Graph 的兼容性**：`DEEPINFER_ZERO_OVERHEAD_SCHEDULING` 依赖 `DEEPINFER_ENABLE_MTP_VARLEN` 才能在 GPU 上修正 varlen 分配（否则 CPU 仍需等 D2H），实际部署需同时开启两个环境变量。

4. **80% 效率基线**：在评估 DeepInfer 性能时，应以 Q&A 的"offline benchmark 的 80%"为基准，不要用单机脱离 LB 的 benchmark 直接外推集群效率。DP 投票 overhead + EPLB 不均衡 + 抢占重算共同贡献了这 20% 损失。

5. **FlopsEstimator 仅供 MFU 监控**：不应将其误用于调度优化。若需在开源版中模拟 cost(q,c) 时间预算，应采集实际 forward_ms 数据拟合二次模型，而非依赖 FlopsEstimator 的 FP8-equivalent FLOP 计数（后者不含启动开销和 dispatch overhead）。

---

## 附：关键 path:line 索引

| 机制 | 文件:行号 |
|------|---------|
| GPU_CACHE_PREFILL_WATERMARK = 0.9 | `csrc/engine/scheduler.hpp:80` |
| ALL/ANY 投票点 & allreduce | `csrc/engine/scheduler.hpp:308–319` |
| 二次确认投票 | `csrc/engine/scheduler.hpp:329–340` |
| prefill_max_sum_seqlen_k 检查 | `csrc/engine/scheduler.hpp:444–449` |
| cp_align 对齐 | `csrc/engine/scheduler.hpp:90, 470–471` |
| chunk_checkpoints | `csrc/engine/scheduler.hpp:478–485` |
| _adjust_chunk_size (multimodal) | `csrc/engine/scheduler.hpp:235–260` |
| LCFS decode 排序 | `csrc/engine/scheduler.hpp:624–643` |
| _preempt (重算/停止) | `csrc/engine/scheduler.hpp:766–790` |
| _append_slot | `csrc/engine/scheduler.hpp:817–828` |
| start_inflight_step / finish | `csrc/engine/sequence.hpp:244–256` |
| advance_optimistically / commit | `csrc/engine/sequence.hpp:216–221` |
| fork_seq | `csrc/engine/scheduler.hpp:1061–1122` |
| FlopsEstimator::calc_flops | `csrc/engine/flops_estimator.hpp:38–45` |
| MTPVarlen::schedule (best_bsz) | `csrc/engine/mtp_varlen.hpp:150–267` |
| ZOS execute_model_step 流水 | `deepinfer/worker/worker.py:88–189` |
| _correct_seqlens_qk (remap) | `deepinfer/worker/worker.py:191–245` |
| prepare_zos_inputs (remap) | `csrc/engine/zos.cpp:142–196` |
| DeferredRelease | `deepinfer/engine/llm_engine.py:67–120` |
| PREFILL_ATTEMPT_INTERVAL default=70 | `deepinfer/environ.py:174` |
| PREFILL_ANY_ATTEMPT_INTERVAL default=100 | `deepinfer/environ.py:177` |
| DEEPINFER_ZERO_OVERHEAD_SCHEDULING | `deepinfer/environ.py:179` |
| forbid_cuda_sync 机制 | `tests/test_cuda_sync_guards.py:1–213` |

