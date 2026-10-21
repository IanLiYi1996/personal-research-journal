# DeepInfer 专题 T1：系统形态 —— E/P/D 分离、PD 传输、AFD、容错与弹性

> **⚠️ 勘误（主编交叉核对，2026-10-21）**：
> ① 分页 KV 实测为 **890 B/token**，不是文中的 1.16 KB。依据是 `hugging_face/config.py:74`：indexer K 跟随主 KV 的源层（2/8/14/20），而 V4.1 的 config.json 中没有 `index_k_source_layer_ids`。
> ② §PD 带宽估算中，"8 卡 prefill 每秒 20 万 token"的取值偏低。官方 prefill 约 87k tok/s/GPU，8 卡约 70 万 tok/s。重新计算：分页 KV 为 70 万 × 890 B ≈ 623 MB/s ≈ **5 Gbps**；送往 CE 的完整推理态 SWA 为 40 层 × 134 × 528 B ≈ 2.8 MB/请求，按平均非缓存长度 4k 算约 175 请求/s，≈ **4 Gbps**。合计约 **10 Gbps 量级**，仍远低于 400 Gbps 的预算，结论不变。


> **截至 2026-10-21**。DeepInfer 源码路径基于 commit `774ce24`；所有 `path:line` 均在撰写时二次确认存在。arXiv 标题已通过 `curl` 核实。标记说明：**[源码]** 读代码确认 / **[Q&A]** 官方文档 / **[估算]** 推算（附算式）/ **[推断]** 合理推测 / **[未核实]** 暂无一手来源。

---

## 0. 导读

DeepSeek V4.1 Flash 推理系统的架构核心是 **E（Vision Encoder）/ P（Prefill）/ D（Decode）三段分离**，配以独立的 3FS 存储节点。本专题从四个维度展开：

1. DeepInfer 源码中为 PD 分离预留的接口结构（已开源的"框架层"）
2. 官方 Q&A 中的部署事实与 PD 传输带宽估算
3. 学术与工业谱系：从 Splitwise 到 AFD 的演进
4. 开源 PD 传输库的现状与从业者组件选型建议

---

## 1. DeepInfer 源码中为 PD 分离预留的结构

### 1.1 EngineType 枚举

**[源码]** `csrc/engine/scheduler.hpp:39-43`：

```cpp
enum class EngineType : int {
    NORMAL     = 0,  // process prompts & completions
    PROMPT     = 1,  // only process prompts
    COMPLETION = 2,  // only process completions
};
```

Python 侧镜像定义位于 `deepinfer/engine/engine_type.py:4-13`，与 C++ 保持一致，注释要求两处同步。`NORMAL` 是兼容模式（单实例自带 prefill + decode）；`PROMPT` 是纯 PE 角色；`COMPLETION` 是纯 CE/Decode 角色。

`engine_type` 通过 `EngineArgs.engine_type`（`deepinfer/engine/arg_utils.py:41`）传入，默认值为 `NORMAL`。

### 1.2 调度器对 EngineType 的感知

**[源码]** `csrc/engine/scheduler.hpp:288-343`，`_schedule()` 函数：

```cpp
switch (engine_type) {
case EngineType::PROMPT:
    return _schedule_prefill(acquire);   // PE 只做 prefill
case EngineType::COMPLETION:
    return _schedule_decode(acquire);    // CE 只做 decode
case EngineType::NORMAL: {
    // 投票决定本组做 prompt 还是 decode ...
}
```

`PROMPT` 模式：每一步无条件调用 `_schedule_prefill()`，即使没有排上请求也返回空的 prefill 结果，保持步型一致。`COMPLETION` 模式：无条件调用 `_schedule_decode()`。

### 1.3 stop_preempted_request — CE 强制使用终止而非重算

**[源码]** `csrc/engine/scheduler.hpp:102-104`：

```cpp
TORCH_CHECK(engine_type != EngineType::COMPLETION || stop_preempted_request,
    "COMPLETION engine cannot recompute preempted requests (no prefill channel); "
    "it must be deployed with stop_preempted_request=true (--stop-preempted-request)");
```

CE 没有 prefill 通道，被抢占的请求无法重算，因此必须以 `stop_preempted_request=True` 启动（`arg_utils.py:40` 默认已为 `True`）。抢占时 `_preempt()` 函数（`scheduler.hpp:768-780`）将序列状态设为 `FINISHED_PREEMPTED`，调用 `free_seq_nolock()` 释放 KV 块，然后把 seq 推入 `ignored_seqs`，触发上层的"终止并重路由"逻辑（具体重路由在未开源的调度器里）。

### 1.4 llm_engine.py 中的 PD 分离空钩子

**[源码]** `deepinfer/engine/llm_engine.py:801-861`：

```python
def init_pd_split(self) -> None:
    "虚函数，初始化 PD 分离需要的数据结构"

def should_copy_state(self, seq: Sequence) -> bool:
    "虚函数，判断是否要保存 seq 当前状态"
    return False

def do_save_kvcache(self, scheduled, finished) -> None:
    "虚函数，保存 kvcache, pre_sampler_sync 时调用"

def pe_prepare_save_kvcache(self, scheduled, finished) -> None:
    "虚函数，pe 准备保存 kvcache"

def pe_do_save_kvcache(self, scheduled, finished) -> None:
    "虚函数，pe 保存 kvcache"

def send_processed_prompts(self) -> None:
    "虚函数，把已经处理完的 prompt sequence 发送给 CE 并从 scheduler 中删除"
```

调用点：`llm_engine.py:837` `do_save_kvcache` 在 decode step 的 `pre_sampler_sync` 之前触发；`llm_engine.py:839` `pe_prepare_save_kvcache` 在 prompt step 时触发；`llm_engine.py:989` `pe_do_save_kvcache` 在 prompt step 后半段触发；`llm_engine.py:1011` `send_processed_prompts` 在 PE step 末尾触发。

**[源码]** 这些方法在开源版中全部是空体（`pass`）。开源代码**没有**继承自 `LLMEngine` 的 `PromptEngine` 或 `CompletionEngine` 子类，意味着 PE/CE 的具体实现（KV 传输、序列迁移等）属于未开源的内部代码。

`init_pd_split()` 在 `EngineType != NORMAL` 时被调用（`llm_engine.py:454-455`），内部有 `copy_state` 机制的启动代码——为 state 开辟 CPU 镜像：`llm_engine.py:803-809`（读取 `cache_engine.gpu_states`，分配 pinned host buffer）。`should_copy_state()` 的调用在 `llm_engine.py:933`，检测需要拷贝 state 的 seq 列表，用于"解码结束后把 SWA state 传出去"。

### 1.5 StateBlock 布局与落盘切片

**[源码]** `deepinfer/worker/cache_engine.py:221-231`：

```
具体布局（从低地址到高地址）：
    [stored region: SWA]
    [dropped region: YOCO consumer SWA | compression state]
```

- `stored region`：需要持久化到 3FS 的前 20 层 SWA KV（每条序列约 20 × 134 × 528 B ≈ 1.35 MiB）。
- `dropped region`：YOCO consumer SWA（靠 prefill 重算恢复，无需存盘）+ C2 压缩边界累积量 `ca_state`/`ica_state`（按 128-token 粒度对齐，无跨边界依赖）。

`select_state_slice_to_store()`（`cache_engine.py:763`）：参数 `TO_CE=False` 时只读 `state_size_to_disk` 字节（stored region）落盘；`TO_CE=True` 时读 `total_bytes_inference` 字节（stored + dropped），用于 PE→CE 的完整 state 迁移。Dropped region 里的 consumer SWA 还要做 ring 重排（kvstore ring → inference window），compression state 直接复制。

`load_state_from_store_to_prefill_cache()`（`cache_engine.py:821`）：从落盘的 state tensor（布局为 `[stored SWA]`）写回 prefill GPU 的 state cache，供 SWA 有界重放时使用。

**[源码]** `state_size_to_disk`（`cache_engine.py:507`）= V4.1-Flash 中 `20 × 134 × 528 = 1,415,040 B ≈ 1.35 MiB`（每条序列定长，与上下文长度无关）。Q&A 原文与代码完全对应。

### 1.6 VE 独立进程池

**[源码]** `deepinfer/ve/multimodal_processor.py`：`MultimodalProcessor` 通过 `torch.multiprocessing.fork` 启动若干 worker 子进程（`_start_workers()`），每个 worker 独立运行图像编解码。GPU-rank ≤ 0 时，若 `DEEPINFER_USE_MULTIMODAL_POOL=1`，则在 rank 0 上创建 `SharedTorchMemoryPool`（pinned host memory），大小由 `DEEPINFER_MULTIMODAL_POOL_SIZE_GB` 控制，多个 GPU rank 共享。

`deepinfer/engine/arg_utils.py:58` 显式在 `COMPLETION` 模式下关闭多模态池（`DEEPINFER_USE_MULTIMODAL_POOL=0`），因为 CE 不做图像处理。**[Q&A]** 视觉编码器部署形态：单 GPU 一个实例，推荐廉价卡（如 A10 等），并有独立的 Vision Encoder Load Balancer 做机器级随机路由，图像级以 patch 数为代价函数在机器内最优 GPU 上处理。

### 1.7 补齐一个 PE/CE 子类需要实现什么

基于以上分析，完整实现 PE/CE 子类需要至少以下接口：

| 接口 | 触发时机 | 实现要点 |
|------|----------|----------|
| `init_pd_split()` | `__init__` 末尾 | 初始化 KV 传输客户端/服务端；为 copy_state buffer 申请 host memory |
| `should_copy_state(seq)` | 每步 pre_sampler_sync | PE 在 prompt step 结束时返回 True；CE 不需要 |
| `pe_prepare_save_kvcache(sched, fin)` | prompt step | 触发 GDR/RDMA 异步传输；把 block KV + state KV 的句柄/地址推入发送队列 |
| `pe_do_save_kvcache(sched, fin)` | prompt step 后半段 | 等待发送完成；通知 CE 可以开始 decode |
| `do_save_kvcache(sched, fin)` | decode step | CE 侧在解码结束后把 SWA state 回写存储 |
| `send_processed_prompts()` | PE step 末尾 | 把完成 prefill 的 seq（含 block_table、state_id）序列化后发给 CE |
| `finalize_seq(seq)` | CE 序列结束 | 通知存储层把 SWA state 落盘（`select_state_slice_to_store(TO_CE=False)`）；释放 KV 块 |

**数据流图：**

```
Client Request
     │
     ▼
[Entrypoint / Load Balancer]
     │              │
     ▼              ▼
[Vision Encoder]  [Prefill Engine (PE)]   ←── 3FS (block KV fetch, GDR swap-in)
     │              │
     │   image      │  完成 prefill 后：
     │  embeddings  │  ① block KV (via RDMA/NIXL) ──────────────────────┐
     └──────────────►  ② SWA state (TO_CE=True, select_state_slice)  ──►│
                    │  ③ seq metadata (block_table, position) ──────────►│
                    │                                                     │
                    ▼                                                     ▼
               send_processed_prompts()                    [Decode Engine (CE)]
                                                                  │
                    3FS (block KV write-back after decode) ◄──────┘
                    SWA state in-memory dir (per-machine)  ◄──────┘
```

---

## 2. Q&A 部署事实与 PD 传输带宽估算

### 2.1 部署参数汇总

**[Q&A]** 来自 `doc/inference_deployment_qa.md`：

| 维度 | 参数 |
|------|------|
| Prefill (PE) | EP2–8 / DP2–8；使用 HBM 能装下的最小 EP/DP |
| Decode (CE) | EP64–128 / DP64–128（Wide-EP）；使用能拿到的最大 EP/DP |
| Vision Encoder (VE) | 单 GPU 一个实例；推荐廉价卡 |
| P:D GPU 比 | 约 1:1 |
| 最小高效部署单元 | 64 卡 prefill + 64 卡 decode = 128 卡 |
| 跨机房互联预算 | 每 8 GPU 约 400 Gbps |
| CE 下线宽限期 | 1 分钟（Grace Period）；在途请求及其 KV 一次性转发给其他 CE |
| EP 故障重确定 | 按健康 GPU 数重新确定 EP size；整实例单卡故障即不可用 |
| 配额保护阈值 | utilization ≥ 0.8 时禁止缩容 |
| KV TTL | Agent 流量 3 天；Chatbot 流量 7 天 |
| 存储比 | 约 14.4 TB logical KV / GPU |
| KV 存储总量 | 约 100 PiB SSD（3FS，10+2 纠删码） |

### 2.2 PD 传输带宽需求估算 **[估算]**

**分页 KV 大小**（来自 §4.3，`cache_engine.py`）：约 **1.16 KB/token**（C2 层约 432+102=534 B/token × 3 层 + C1 层约 288+68=356 B × 6 层，合计不含对齐约 1.16 KB）。

**SWA state 传输**（TO_CE）：`state_size_to_disk` = 1.35 MiB/seq（仅 stored region），但 PE→CE 需要完整 inference state（包含 dropped region），实际略大。保守估计 ~1.5 MiB/seq。

**估算场景**：一个 64-GPU decode 超节点（EP64）配 64-GPU prefill（8 × EP8 节点）：

```
设 prefill 有效吞吐 T_p tokens/s / node（EP8，8 GPU），
   请求平均输入长 L_in tokens，请求速率 R = T_p / L_in req/s / node

① 分页 KV 传输速率（per node）：
   T_p × 1.16 KB = T_p × 1.16 × 10^3 B/s
   
   若 T_p = 200,000 tok/s（8 × B200 GPU at EP8，prefill 较大 batch）：
   = 200,000 × 1.16 KB = 232 MB/s ≈ 1.86 Gbps

② SWA state 传输速率（per node）：
   R × 1.5 MiB = (T_p / L_in) × 1.5 MiB
   
   若 L_in = 1024 tokens：
   = (200,000 / 1024) × 1.5 MiB ≈ 195 × 1.5 MiB = 293 MB/s ≈ 2.34 Gbps

③ 合计 / node ≈ 4.2 Gbps（远低于 400 Gbps 预算）

④ 整群 8 nodes × 4.2 Gbps = 33.6 Gbps 实际消耗
```

**结论**：在典型批处理场景下，PD 传输对网络的压力极小（< 10% of 400 Gbps）。**400 Gbps 是针对最差情况（长上下文、高并发、突发 burst）预留的安全余量，而非常态消耗**。V4.1-Flash 的 KV 极度压缩（约 DeepSeek-V3 的 1/60）是实现跨机房 PD 的根本原因。若用 DeepSeek-V3 的 MLA（约 69 KB/token），同样场景下需要约 200× 更多带宽，跨机房将完全不可行。

**[Q&A]** 实际使用的是不同但相邻机房，纤维中断（施工挖断）是现实中的故障模式，强调容错靠配额冗余而非网络鲁棒。


---

## 3. 学术与工业谱系深读

### 3.1 PD 分离的演进路线

| 工作 | arXiv | 核心创新 | 与 DeepInfer 的关系 |
|------|-------|----------|---------------------|
| **Splitwise** | 2311.18677 | prefill/decode 放不同硬件；论证了两阶段特性差异的根本性 | 最早学术验证，DeepInfer 的 P:D 1:1 比例与其一致 |
| **DistServe** | 2401.09670 (OSDI'24) | 以 goodput（同时满足 TTFT+TPOT 的请求率）为优化目标 | DeepInfer 的双维负载均衡（req 数 + KV 长度）与 DualPath 一致，延伸了 DistServe 的思路 |
| **Mooncake** | 2407.00079 (FAST'25 最佳论文) | KV 为中心的分离架构；全局 CPU/SSD KV 池；过载时提前拒绝 | DeepInfer 的专用 3FS 存储节点是更激进的形式；Mooncake Transfer Engine 被 vLLM/SGLang 采纳为传输层 |
| **semi-PD** | 2504.19867 | SM 级计算分离 + 统一存储；避免权重复制和 KV 迁移 | 解决了 PD 分离在中小规模的存储浪费问题，但不适用于 EP64–128 这样的大规模场景 |
| **MegaScale-Infer** | 2504.02263（字节跳动）| 注意力-FFN 分离（AFD）+ ping-pong 微批次流水 + M2N 通信库 | DeepSeek 未采用 AFD；MegaScale 的 M2N 库与 NIXL 定位相近 |
| **Step-3 AFD** | 2507.19427（阶跃星辰）| MFA（Multi-Matrix Factorization Attention）+ AFD；321B VLM | 在 4K context FP8 下实现 4039 tok/s/GPU（对比 DeepSeek-V3 的 2324），50ms TPOT SLA |
| **AFD 设计空间** | 2605.28302 | 系统性评估 AFD 在不同工作负载/SLO 下的适用边界；严格 TTFT/TPOT SLO 下 non-AFD 不可行时 AFD 维持约 4K tok/s | 为"何时上 AFD"给出了量化判据 |
| **DualPath** | 2602.21548（含 DeepSeek 相关作者）| 存储-to-decode 双路 KV 加载 + RDMA 传给 prefill；打破 NIC 带宽不均衡；agentic 推理 1.87×↑ | 直接描述了 DeepSeek 内部的 decode 负载均衡算法（Q&A 中"参见 DualPath 的组内调度算法"） |
| **CloudMatrix-Infer** | 2506.12708（华为）| Ascend 910 × 384，EP320，peer-to-peer P/D/Cache 三路独立扩缩；UB 全互联 | DeepInfer 支持 Ascend 950；CloudMatrix 是更激进的 UB 硬件架构，EP320 超过 DeepSeek 的 EP128 |

### 3.2 DualPath 的特殊性

**注意**：DualPath（2602.21548）的作者包含 DeepSeek 关联研究人员，Q&A 中明确引用了该论文（"参见 DualPath 的组内调度算法"），因此 DualPath 既是外部论文也是 DeepSeek 内部实践的公开描述。其核心贡献：

1. **双路 KV 加载**：在 agentic 多轮对话场景，KV 从存储不止走 storage→prefill 一条路，同时走 storage→decode→prefill（RDMA）的另一条路，缓解 prefill 侧 NIC 饱和。
2. **组内调度算法**：同时平衡请求数（决定 step batch size）和 KV 长度（决定 attention 计算代价），短请求发给"低并发但 KV 长度大"的 GPU，长请求发给"KV 长度小"的 GPU。这正是 Q&A 描述的 Decode Load Balancer 原理。

### 3.3 AFD 设计空间要点

2605.28302 给出了在 DeepSeek-V3.2 上的评估，关键结论：

- **严格 SLO 场景**（TTFT ≤ 200ms、TPOT ≤ 50ms）+ 长上下文（≥ 4K 输入 / ≥ 1K 输出）时，AFD 能维持系统吞吐约 4K tok/s，而 non-AFD 完全不可行。
- **短上下文或宽松 SLO** 时，P/D 分离（不 AFD）仍然是更简单有效的选择。
- **GPU 分区比例**：注意力 GPU 数 / FFN GPU 数 = 模型注意力算力需求 / FFN 算力需求，随工作负载 ISL/OSL 比变化。
- **互联拓扑**：AFD 在跨节点时增加了注意力与 FFN 之间额外一次 all-to-all，对 interconnect 要求更高。

---

## 4. PD 传输库对比

### 4.1 各主流传输库

| 库 | 仓库 / arXiv | 定位 | KV 传输支持 | 当前状态（2026-10） |
|----|-------------|------|-------------|---------------------|
| **NIXL**（NVIDIA Inference Xfer Library）| github.com/ai-dynamo/nixl | GPU 间低延迟 RDMA 传输；支持 UCX/NVLink | 是；vLLM NixlConnector、llm-d 采用 NIXL RDMA | 1322 stars；vLLM 和 llm-d 主推 |
| **Mooncake Transfer Engine** | github.com/kvcache-ai/Mooncake（6850 stars）| KV 缓存中心化分离架构的传输层；RDMA + NVMe-oF | 是；SGLang MooncakeConnector、vLLM MooncakeConnector | FAST'25 最佳论文；被 SGLang/vLLM 集成 |
| **DeepEP** | github.com/deepseek-ai/DeepEP | EP 专家并行 all-to-all；低延迟/高吞吐双模式 | **否**；DeepEP 传输的是 token 激活，不是 KV cache | V2.5 切换到 NCCL GIN 后端 |
| **vLLM KV Connector** | vllm/distributed/kv_transfer/kv_connector/ | 插件化接口；底层可换 NIXL/Mooncake/LMCache 等 | 是；多后端 | v1/ 目录：NixlConnector、LMCacheConnector 等 |
| **SGLang disaggregation** | sgl-project/sglang（disaggregation/）| 内置 prefill/decode 分离；支持 Mooncake、NIXL、MoriIO | 是；prefill_complete 分配策略 | 目录含 nixl/、mooncake/、decode.py / prefill.py |
| **TRT-LLM disaggregated** | NVIDIA/TensorRT-LLM | 高度优化的分离推理；KV 通过 NIXL 传输 | 是 | 与 Dynamo 配合使用 |
| **Dynamo** | github.com/ai-dynamo/dynamo（8345 stars）| 数据中心级分布式推理框架；调度器 + KV 路由 | 是；KV 路由感知调度 | 覆盖 vLLM/TRT-LLM 后端 |
| **llm-d** | github.com/llm-d/llm-d | K8s 原生推理；EPP 调度 + D pod sidecar 代理 | 是；通过 NIXL RDMA | 架构：decode pod 内含 routing sidecar；EPP 选 P+D 对 |
| **LMCache** | vllm/kv_transfer 集成 | CPU/GPU/SSD 多级 KV 缓存；跨引擎 KV 复用 | 是；SGLang LMCacheConnector | 聚焦缓存复用，非实时 PD 传输 |

**重要澄清**：DeepEP 是**专家并行 all-to-all 传输库**（dispatch token → expert → combine），与 KV cache 传输是完全不同的数据路径。KV 从 P 传给 D 使用的是独立通道（NIXL/Mooncake 等），不经 DeepEP。

### 4.2 vLLM KV Connector 架构

**[源码-vLLM]** `vllm/distributed/kv_transfer/kv_connector/v1/` 目录包含插件化接口（`base.py`）。SGLang 的类似结构在 `disaggregation/` 目录下，`prefill.py` 中含 `disaggregation_decode_allocation_policy == "prefill_complete"` 逻辑：在 GPU forward 期间提前向 decode 发送 STAGING_REQ，使 decode 侧能在 prefill 完成时立即分配 staging buffer，降低 KV 传输开始的等待时间。

### 4.3 llm-d 的 P/D sidecar 模式

**[GitHub]** `llm-d/llm-d/docs/architecture/disaggregation/pd-disaggregation.md`：

```
Client → Proxy → EPP （选 P+D 对）
                  → D Routing Sidecar
                         → P Worker（max_tokens=1, do_remote_decode=True）
                         ← KVTransferParams
                  → D Worker（do_remote_prefill=True，通过 NIXL RDMA 拉取 KV）
```

Decode pod 内置 routing sidecar（Envoy 风格代理），EPP 决策后 sidecar 协调 P→D KV 传输。P Worker 的 max_tokens=1 使其在完成 prefill 后立即返回 KVTransferParams，D Worker 主动向 P Worker 拉取 KV（NIXL RDMA pull），避免 push 路径上的 P NIC 饱和。这与 DualPath 的 storage-to-decode pull 思路异曲同工。

---

## 5. AFD vs 宽 EP 取舍分析

### 5.1 DeepSeek 的选择：宽 EP 而非 AFD

DeepSeek V4.1-Flash 的 decode 选择了 EP64–128 的宽专家并行，而不是将注意力和 FFN 进一步分离（AFD）。**[推断]** 可能的原因：

1. **scale-up 域足够大**：GB200 NVL72 和 Ascend 超节点把 64–128 GPU 全部收纳在高带宽 NVLink/UB 域内，EP all-to-all 的延迟极低（< 1 µs 量级），不需要通过 AFD 来减少 EP 通信。
2. **KV 极度压缩消除了注意力 KV 带宽瓶颈**：V4.1-Flash 注意力只需读 ≈ 1.16 KB/token 的 KV，远比 MHA/MQA 低，注意力 GPU 的访存压力不构成瓶颈。而 AFD 的主要收益是把 KV-heavy 的注意力 GPU 和 weight-heavy 的 FFN GPU 分开独立扩缩。
3. **故障域管理**：AFD 需要注意力与 FFN 节点之间额外一次 all-to-all 通信（MegaScale-Infer 的 ping-pong 流水），使故障域和流水线编排更复杂；DeepSeek 倾向于在固定的 scale-up 超节点内部解决问题。
4. **MoE 负载均衡已在 EP 内解决**：EPLB + 冗余专家机制（EP64 时每 GPU 约 7 个专家）把 `moe_lb`（最慢/均值）控制在约 1.09，专家访存不均衡问题基本解决，不需要 FFN 独立扩缩。

### 5.2 什么场景下 AFD 有优势

根据 2605.28302 的分析：

| 维度 | AFD 胜出 | 宽 EP / PD 胜出 |
|------|---------|----------------|
| SLO 严格度 | TTFT ≤ 200ms + TPOT ≤ 50ms | 宽松 SLO |
| 上下文长度 | 长上下文（≥ 4K 输入）| 短上下文 |
| scale-up 域 | 跨节点需要异构扩缩（注意力 GPU ≠ FFN GPU）| 同质大超节点（NVL72 / CM384）|
| 注意力 KV 大小 | MHA/GQA（KV 大，访存受限）| MLA / C2 压缩（KV 小）|
| 模型结构 | 注意力 FLOPs / FFN FLOPs 比例悬殊 | 两者比例均衡 |

Step-3（321B VLM）用 MFA + AFD 在 4K context 下达到 4039 tok/s/GPU vs DeepSeek-V3 的 2324——这是异构硬件+长上下文场景下 AFD 优势的最新实证。

### 5.3 EP 故障域与弹性 EP

**[Q&A]** EP 故障模型：
- 单实例内所有 GPU 共享一个通信域（NCCL communicator），一张卡故障导致整实例不可用。
- 重启后按健康 GPU 数**重新确定 EP size**，实例实际 EP 可能与原始 EP 不同（如 EP8 → EP7 → EP6…）。
- 容错策略：依赖"冗余实例配额 + 请求级重路由"，而非缩短单实例重启时间（重启约需数分钟）。

**vLLM 弹性 EP（Elastic EP）**：**[GitHub]** `vllm-project/vllm` 有活跃的 Elastic EP PR 系列（至少 5 个开放 PR，含 `[Feature][Elastic EP] Correlate external scaling operations`、`[Bugfix][Elastic EP] Clear scaling flag on failure` 等），提供 `/scale_elastic_ep` API 端点，允许在不重启引擎的情况下动态调整 EP 大小。这比 DeepSeek 的"重启 + 重定 EP size"方案更具弹性，但复杂度更高（需要处理取消安全性、API 鉴权等）。

---

## 6. 对从业者：用开源组件拼出接近 DeepSeek 的 E/P/D 系统

### 6.1 架构目标

复现一个"中等规模"的 E/P/D 系统（如 8P + 64D，EP8 prefill + EP64 decode）：

```
[API Gateway]
     │
     ▼
[Load Balancer] ←── prefix hash 路由（可选 RadixAttention / APC）
     ├── [Vision Encoder (VE)]：单 GPU，A10/L40S
     ├── [Prefill Pool (PE)]：N × 8-GPU 节点，EP8
     └── [Decode Pool (CE)]：M × 8-GPU 节点，EP64 on NVL16 / NVL72
                                        │
                              [KV Cache Storage]：3FS or NVMe-oF
```

### 6.2 组件选型建议

| 功能层 | 推荐方案 | 备注 | 风险 |
|--------|---------|------|------|
| **推理引擎** | vLLM V1 + EP | 社区最活跃；Elastic EP 快速演进 | EP 故障路径仍在完善；vLLM EP 性能不及 DeepInfer |
| **推理引擎（备选）** | SGLang + disaggregation | disagg 架构最完整；Mooncake/NIXL 均已集成 | SGLang 对 Ascend 950 支持有 PR 但仍实验性 |
| **KV 传输** | NIXL（RDMA）| llm-d 和 vLLM 主推；pull 模式避免 P NIC 饱和 | 需要 RDMA 网络；TCP fallback 性能差 |
| **KV 传输（备选）** | Mooncake Transfer Engine | FAST'25；适合 CPU-pooled KV 架构 | 与 SGLang 深度耦合 |
| **KV 存储** | LMCache（CPU/GPU/SSD）| 开源、成熟；跨实例 KV 复用 | 不含 3FS 级别的分布式纠删码高可用 |
| **EP 通信** | DeepEP V2（NCCL GIN）| DeepSeek 官方；低延迟和高吞吐双模 | SM100/SM103 最优，A100/H100 性能有差距 |
| **负载均衡** | llm-d EPP + disagg-profile-handler | K8s 原生；P+D 对选择；prefix 感知 | 相对新，生产验证不足 |
| **负载均衡（备选）** | Dynamo KV router | Dynamo 框架内；KV 路由感知 | 依赖 Dynamo 生态 |
| **EPLB** | vLLM 内置 EPLB / SGLang EPLB | 均参照 DeepSeek EPLB 论文实现 | 冗余专家数量和 re-assign 周期需调优 |

### 6.3 关键差距与风险

1. **KV 存储高可用性**：开源方案（LMCache、Mooncake Store）没有 3FS 的 10+2 纠删码和 100 PiB 量级。自建需要接入 Ceph/JuiceFS/Weka 等，或用 S3 but 延迟不达标。
2. **SWA state 的独立管理**：DeepInfer 的 `StateBlock` 布局（stored/dropped region）和有界重放（SWA bounded replay）是其独特设计，开源框架没有对等机制。如果模型不使用 SWA（如标准 GQA），此问题不存在。
3. **GPU 故障恢复**：DeepSeek 依赖"冗余实例 + 请求重路由"容忍分钟级重启；开源方案中 vLLM Elastic EP 提供了更细粒度的扩缩但仍处于快速迭代中，稳定性需评估。
4. **跨机房 PD**：只有 KV 极度压缩（如 V4.1-Flash 的 MLA+C2+FP4）才能让跨机房 PD 实用。使用 DeepSeek-V3（标准 MLA）或 LLaMA（GQA）的部署，跨机房 PD 带宽压力会大很多。
5. **MoE 专家 GEMM**：Mega MoE（融合 all-to-all + GEMM）在开源版本中尚无完整等价物；pplx-kernels 和 DeepEP 部分覆盖，但端到端 MFU 仍有差距。

### 6.4 最小可用的 PD 分离搭建步骤

```bash
# 1. 选择 SGLang (最完整的 disagg 支持)
#    启动 prefill server
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --disagg-prefill \
    --ep-size 8 \
    --port 30001

# 2. 启动 decode server
python -m sglang.launch_server \
    --model deepseek-ai/DeepSeek-V3 \
    --disagg-decode \
    --ep-size 64 \
    --port 30002

# 3. KV 传输后端：选择 NIXL（RDMA 环境）或 Mooncake（CPU KV pool）
#    通过环境变量 SGLang_DISAGG_KV_BACKEND=nixl 或 mooncake 配置

# 4. 配置 llm-d EPP 做 P+D 对路由
#    或使用 Dynamo 的 KV-aware 调度器
```

---

## 7. 开放问题

1. **PE/CE 之间的 KV 传输协议细节**：序列元数据（block_table 布局、state_id 映射）如何序列化并传给 CE？开源版本 `send_processed_prompts()` 是空体，实现不可见。
2. **`do_save_kvcache` 在 CE 侧的触发时机**：decode 结束后何时把 SWA state 写盘？是每个 decode step 写一次，还是 seq 结束时统一写？代码中是 decode step 的 `pre_sampler_sync` 触发，但 state 落盘粒度（128-token 边界）在 CE 侧如何维护仍不明确。
3. **EP 动态重配的 communicator 重建代价**：EP 故障后按健康 GPU 数重建 NCCL communicator，NCCL init 本身的时延是否被隐藏？Q&A 只说"重启约数分钟"，具体流程未披露。
4. **VE 的 embedding 是否进入前缀缓存**：图像 embedding 理论上可以缓存（节省反复 VE 计算），但 Q&A 和代码中对 VE 结果的缓存机制没有明确说明。**[未核实]**
5. **semi-PD 与 wide-EP 的组合可行性**：semi-PD 的 SM 级分离可以在同一节点内实现 prefill/decode 异步，是否可以叠加宽 EP？2504.19867 没有评估此组合。

---

## 附录：关键 arXiv 标题核实

通过 `curl https://arxiv.org/abs/<ID>` 核实：

| arXiv ID | 标题（已核实） |
|----------|---------------|
| 2311.18677 | Splitwise: Efficient generative LLM inference using phase splitting |
| 2401.09670 | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving |
| 2407.00079 | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving |
| 2504.19867 | semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage |
| 2504.02263 | MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism |
| 2507.19427 | Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding |
| 2605.28302 | How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving |
| 2602.21548 | DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference |
| 2506.12708 | Serving Large Language Models on Huawei CloudMatrix384 |

