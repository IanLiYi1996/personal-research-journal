# GPU Infra Benchmark 知识总结：Blackwell / Hopper 推理与训练实践要点

- **Date:** 2026-05-19
- **Tags:** #GPU基础设施 #推理优化 #分布式训练 #SGLang #Megatron #NCCL #PD分离 #MoE #DeepEP

## Context

整理一份 GPU Infra benchmark 实践合集的**知识性结论**——剔除日志、客户场景、内部链接等业务细节，只留可复用的工程经验：

- 每代 GPU 实例的硬件画像与互联拓扑（NVLink、跨节点 RDMA）
- NCCL 测试的指标定义、busbw 公式、关键环境变量
- **PD 分离的技术原理与工程权衡**（本次重点扩充）
- 推理侧（SGLang / vLLM）部署范式选择：KV transfer engine、MTP、并发控制
- **PD 分离 2025-2026 最新进展**（cross-DC、intra-GPU、KV 压缩、SLO 调度、硬件分离、EPD 多模态扩展）
- 训练侧（Megatron-LM）：MoE dispatcher、EP 并行、网卡分离、NUMA 亲和

公开来源已验证：[NCCL Tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)、[DeepEP](https://github.com/deepseek-ai/DeepEP)、[DistServe (OSDI'24)](https://arxiv.org/abs/2401.09670)、[Splitwise (ISCA'24)](https://arxiv.org/abs/2311.18677)、[Mooncake](https://arxiv.org/abs/2407.00079)，以及 14 篇 2025H2~2026H1 的 PD 分离最新工作（详见三补章 + References）。

---

## 一、Blackwell / Hopper 实例画像

| 项目 | Blackwell 8GPU 实例 | Blackwell-Ultra 8GPU 实例 | Hopper H200 8GPU 实例 |
|---|---|---|---|
| GPU | 8 × Blackwell（B200） | 8 × Blackwell Ultra（B300） | 8 × H200 |
| GPU 显存 | ~180 GB/卡（HBM3e） | ~268 GB/卡（HBM3e） | 141 GB/卡（HBM3e） |
| 跨节点 RDMA 带宽 | **3.2 Tbps** | **6.4 Tbps** | 3.2 Tbps |
| NVLink | Gen5，最高 14.4 TB/s 双向（1.8 TB/s/GPU） | Gen5（同代） | Gen4 |
| FP4 / FP8 算力 | 高 | **vs B200：1.5× FP4 TFLOPS（无稀疏）** | 不支持原生 FP4 |

**几个关键观察：**

1. **B300 真正的杀手锏不是 FLOPS，而是网络翻倍 + 显存升 50%**：跨节点 6.4 Tbps RDMA 直接改写 PD 分离场景下 KV 传输的瓶颈位置。
2. **NVLink Gen5 节点内带宽极其充裕**（14.4 TB/s 全双工），节点内 TP 几乎不会成为瓶颈，瓶颈一律外移到跨节点。
3. **H200 → B300 的显存提升**（141 → 268 GB/卡）让单节点可装下的模型规模翻倍——很多原本必须跨节点的部署变回单节点。

---

## 二、NCCL Test 指标怎么读

### 2.1 公式（NCCL 官方）

| Collective | algbw | busbw 修正因子 |
|---|---|---|
| **AllReduce** | S/t | `2(n-1)/n` |
| ReduceScatter / AllGather | S/t | `(n-1)/n` |
| **AlltoAll** | S/t | `(n-1)/n` |
| Broadcast / Reduce | S/t | 1（root 即瓶颈） |

> busbw 是 **算法独立的**——只要走 send/recv，无论 ring/tree/double binary 都得到同一个 busbw，它逼近"硬件链路真实吞吐"（NVLink/PCIe/RDMA）。

### 2.2 为什么 busbw 会"超过物理带宽"？

`2(n-1)/n` 在 n→∞ 时趋向 2。所以对于跨节点 AllReduce，busbw 在大规模时**数值上可以超过单链路物理带宽**——这是设计上的归一化效果，不代表违反物理。

### 2.3 经验阈值

- 2 节点 AllReduce 的 peak busbw 应当**接近物理带宽的上限**（如 3.2 Tbps ≈ 400 GB/s 时跑出 ≈ 402 GB/s 是合理的）
- 如果 busbw < 物理带宽的 70%，应当排查：拓扑、NIC 绑定、CPU affinity、ALGO/PROTO 选择

### 2.4 关键环境变量

| 变量 | 默认 | 何时调整 |
|---|---|---|
| `NCCL_CROSS_NIC` | 0（适合 rail-optimized） | **非 rail-optimized 网络（如多数云厂商 RDMA）试 1 或 2** |
| `NCCL_IGNORE_CPU_AFFINITY` | 0 | 试 1 看是否更好（覆盖 NCCL 自动绑核策略） |
| `NCCL_ALGO` | auto | 通常保持 auto；只在小消息或特定 collective 退化时手动 |
| `NCCL_PROTO` | auto | 同上 |

**通用准则**：先 default 跑一遍拿基线，再针对**消息大小段位**单独调优。AlltoAll 性能尤其敏感，建议针对实际 workload 单独立 case 调。

---

## 三、PD 分离（Prefill-Decode Disaggregation）技术原理

> 这一章是本笔记的重点扩充：把 PD 分离从"工程选择"还原成"为什么能 work"。

### 3.1 LLM 推理的"双相位"本质

LLM 推理可以拆成两个截然不同的相位：

```
请求到达
   │
   ▼
┌────────────────┐         ┌──────────────────┐
│  Prefill 阶段   │  KV →   │  Decode 阶段       │
│  处理 N 个输入  │ ───→   │  逐 token 自回归   │
│  token（并行）  │         │  生成 M 个输出 tok │
└────────────────┘         └──────────────────┘
  Compute-bound             Memory-bandwidth-bound
  低 batch size 即可吃满 GPU  必须大 batch 才能利用算力
  耗时正比于 N²（attention） 耗时正比于 M（每步常数）
  目标 SLO：TTFT             目标 SLO：TPOT
```

**这是 PD 分离的物理基础**（DistServe / Splitwise 的核心论证）：

| 维度 | Prefill | Decode |
|---|---|---|
| 计算特征 | **Compute-bound**（FLOPS 是瓶颈） | **Memory-bandwidth-bound**（HBM 带宽是瓶颈） |
| 单次工作量 | O(N²) attention + O(N) FFN | O(1) 每步 |
| 适合的并行 | TP 多卡 + 大模型权重切分 | DP 多副本 + 大 batch |
| 硬件偏好 | 高 FLOPS / 中等显存 | 高 HBM 带宽 / 大显存装 KV cache |
| SLO | **TTFT**（时间到首 token） | **TPOT**（每输出 token 时间） |

### 3.2 共置（Non-PD）的两个根本痛点

把 prefill 和 decode 放在同一组 GPU 上（标准 vLLM/TGI 默认部署）会带来：

**痛点 1：相互干扰（Prefill-Decode Interference）**

- 一个长 prefill 请求（如 16K 输入）会**阻塞所有正在 decode 的请求**——因为 GPU 一次只能跑一个 forward pass，prefill 占用计算资源时 decode 必须等待
- 表现：**TPOT 抖动剧烈**，p99 远高于 p50
- DistServe 论文用的术语是 "prefill-decoding interferences"

**痛点 2：资源耦合（Resource Coupling）**

- 单一 deployment plan 必须同时满足 TTFT 和 TPOT 两个 SLO
- 实际系统通常**只能优先一个，过度配置另一个**——浪费资源

### 3.3 PD 分离的解法

```
┌──────────────────────┐    KV cache 传输    ┌──────────────────────┐
│  Prefill Cluster     │ ──────────────────> │  Decode Cluster      │
│  (高 FLOPS GPU)      │      (RDMA/NVLink)  │  (高 HBM 带宽 GPU)    │
│  TP 切分模型权重      │                     │  DP + 大 batch        │
│  专注 TTFT           │                     │  专注 TPOT           │
└──────────────────────┘                     └──────────────────────┘
```

**核心收益：**

1. **零干扰**：Prefill 节点不参与 decode，反之亦然
2. **资源解耦**：可独立扩容 prefill 集群（响应 input 长度变化）和 decode 集群（响应 batch 大小变化）
3. **硬件异构**：Splitwise 提出 prefill 用强 GPU、decode 用便宜 GPU，可降低 20% 成本
4. **并行策略独立**：prefill 用 TP=8 切分模型；decode 可用 DP=8 跑大 batch

**核心代价：**

- **KV cache 跨节点传输**：每个请求 prefill 完成后，必须把它产生的 KV tensor 从 prefill 节点传到 decode 节点
- KV cache 大小 = `2 × n_layers × n_heads × head_dim × seq_len × dtype_size × n_requests`
- 一个 70B 模型 16K input：单请求 KV ≈ **几 GB 量级**——传输时间变成关键

### 3.4 KV 传输工程：三种主流方案

| 方案 | 机制 | 优势 | 劣势 |
|---|---|---|---|
| **NIXL**（NVIDIA） | 高性能 KV transfer abstraction，支持多种 backend（NVLink/RDMA/网络） | 吞吐高、稳定性好、并发场景表现优秀 | 较新，生态对接成本 |
| **Mooncake TE**（Moonshot） | RDMA-based，KVCache-centric 调度，跨 CPU/DRAM/SSD pooling | 长上下文场景吞吐报告最高 +525%（论文数字） | 在某些 RDMA 网络上稳定性挑战、需打 patch |
| **NCCL P2P send/recv** | 把 KV 当 tensor 通过 NCCL 传 | 实现最简单 | 不是为 KV 模式优化，吞吐和延迟都次优 |

**实践经验**：

- 高并发场景下 **NIXL 在成功率上有数量级优势**——某长输入 benchmark 中 NIXL 100% 成功率 vs Mooncake 23%
- **TPOT 上 Mooncake 略优**（0.x ms 量级）但 **TTFT 和稳定性 NIXL 全面领先**
- KV 传输引擎的真正考验不是 happy path 吞吐，而是 **高 QPS + 长 input 下的失败率**

### 3.5 SLO 分配：怎么决定 N:M 比例

DistServe 给出的方法论：

1. 对 prefill 集群：根据 input 长度分布 + TTFT SLO，反推需要多少 prefill 副本
2. 对 decode 集群：根据 output 长度分布 + TPOT SLO + KV 容量，反推需要多少 decode 副本
3. 整体优化：在满足两个 SLO 的前提下，最小化总 GPU 数

**经验法则**（基于公开 benchmark）：

- **短输入（如 4K）**：1 Prefill : 1 Decode (1P1D) 通常够用
- **中等输入（16K~32K）**：可能要 1P2D 或 2P2D
- **超长输入（120K+）**：**反而不一定是 PD 分离最优**——KV 传输成本占比剧增，多节点 Non-PD 可能反超

> **反直觉但重要的发现**：120K input 场景下，**4 节点 Non-PD 部署可以比 2P2D PD 分离更优**——因为 prefill 阶段单请求耗时已经很长（数秒），跨节点传几 GB KV 的代价不再被忽略，反而把 PD 分离的零干扰优势抵消掉了。

### 3.6 何时用、何时不用 PD 分离

**强烈建议用：**
- 高并发短输入（QPS 高、TTFT SLO 紧）
- 资源池足够大（>4 节点）
- KV 传输基础设施成熟（RDMA + 调好的 NIXL/Mooncake）

**慎用或不用：**
- 单节点资源够装下模型（不要为了"现代架构"而过度设计）
- 超长输入（KV 传输成本压垮收益）
- KV 传输引擎不稳定的环境（高并发下崩盘比共置 latency 抖动严重得多）

### 3.7 衍生概念：MTP（Multi-Token Prediction）

PD 分离解决"两相位干扰"，**MTP 解决"decode 阶段慢"本身**——相当于**推测解码（speculative decoding）的对偶**：模型一次 forward 输出多个候选 token，验证后批量提交。

**实测收益规律：**

| 配置 | 吞吐提升 | TPOT 降低 |
|---|---|---|
| 短输入 + 高并发 | **+57% ~ +73%** | **−34% ~ −40%** |
| 长输入 + 高并发 | +32% ~ +34% | −12% ~ −13% |

**经验法则**：MTP 在**短输入 + 高并发**收益最大；输入越长、batch 越大，边际收益递减（接受长度饱和在 ~3.85 token/step 左右）。

### 3.8 max-concurrency 的隐性作用

很多人只知道调 batch size，不知道**并发数和 batch size 是两个旋钮**：

- 不设 `max-concurrency` → 吞吐**几乎不涨**（有时反而略降），但 **TTFT 大幅膨胀**
- 设 `max-concurrency=100` 是个性价比高的默认值

原因：并发越高，prefill 队列越长，每个新请求的 TTFT 等待时间被拖累。吞吐瓶颈通常已经在 GPU 算力，单纯堆并发并不能突破。

---

## 三补：PD 分离最新进展（2025H2 ~ 2026H1）

DistServe / Splitwise / Mooncake 之后的一年里，PD 分离从"是否分"转到"怎么分得更精细"。下面按主题归纳最新论文。

### A. 跨数据中心 PD 分离

**PrfaaS（Prefill-as-a-Service, Moonshot AI, 2026-04, [arXiv:2604.15039](https://arxiv.org/abs/2604.15039)）**

- 核心观察：Hybrid-attention 架构（Mamba-Transformer 等）让 KVCache 体积大幅缩小，**首次让"跨 DC 传 KV"在物理上可行**
- 设计：长上下文 prefill 卸载到独立 compute-dense 集群，KV 通过普通以太网（非 RDMA）回传到 PD 本地集群解码
- 关键 trick：模型侧 KV 效率优化 + 系统侧选择性 offloading + 带宽感知调度 + cache-aware 请求放置
- 效果：1T 参数 hybrid 模型上 **54% 吞吐 / 64% P90 TTFT 改善**，等成本下 ~15% 吞吐增益

> **Implication**：PD 分离的部署边界从"同一 RDMA 网络域内"扩展到"跨 DC"。这意味着推理能力可以**像 CDN 一样分布式调度**。

### B. Intra-GPU PD 分离（不分节点，分 SM）

**Nexus（[arXiv:2507.06608](https://arxiv.org/abs/2507.06608), 2025-07）**

- 核心观察：传统 inter-node PD 分离的最大代价是 KV 传输；如果在**单 GPU 内**把 SM 切两半呢？
- 方法：**proactive intra-GPU disaggregation**——主动（而非反应式）按 workload 划分 GPU 资源给 prefill/decode 两个 stream
- 解决"chunked prefill 在 vLLM/SGLang 里都做了，但 SM 利用率仍随 workload 抖动"的问题

**RAPID-Serve（[arXiv:2601.11822](https://arxiv.org/abs/2601.11822), 2026-01, AMD MI300X）**

- 同样思路：单 GPU 内并发跑 prefill + decode，通过 **CU masking**（Compute Unit 掩码）控制资源分配
- 在 AMD MI300X 上实现，证明该思路非 NVIDIA 专属

> **Implication**：PD 分离不再是"分节点"的代名词。**Intra-GPU PD** 在小集群、低并发场景可能比 inter-node PD 更省资源。

### C. KV 传输瓶颈攻坚

**DualPath（[arXiv:2602.21548](https://arxiv.org/abs/2602.21548), 2026-02）**

- 观察：multi-turn agentic 推理下，KV cache 是从**外部存储**（不只是另一个节点）加载的，prefill 节点的存储 NIC 饱和、decode 节点的 NIC 闲置——**带宽不对称**
- 解法：双路径 KV 加载——传统 storage-to-prefill + 新增 storage-to-decode（再 RDMA 转给 prefill）
- 配合全局调度器动态平衡负载
- 效果：**离线吞吐 1.87×，在线吞吐 1.96×**

**SplitZip（[arXiv:2605.01708](https://arxiv.org/abs/2605.01708), 2026-05）**

- 第一个**GPU 友好的无损 KV 压缩器**，专为 PD 分离传输优化
- 利用浮点数指数位的冗余：top-16 频繁指数走定长编码，稀有指数走稀疏 escape 流
- BF16 上：压缩 613 GB/s，解压 2181 GB/s——基本不拖慢传输路径
- 端到端：**KV 传输 1.32× / TTFT 1.30× / 请求吞吐 1.23×**

> **Implication**：KV 传输从"传裸 tensor"进化到"传压缩流"，在跨节点 / 跨 DC 场景的边际收益越发明显。

### D. SLO 感知的精细调度

**Kairos（[arXiv:2605.02329](https://arxiv.org/abs/2605.02329), 2026-05）**

- 解决 long-tail request length 在 PD 分离下的两个症状：
  - Prefill 端：长请求阻塞短请求（head-of-line blocking）
  - Decode 端：stragglers 导致 batch 拉胯
- 双机制：**urgency-based priority scheduling**（prefill）+ **slack-guided adaptive batching**（decode）
- 提升 TTFT/TPOT/E2E SLO attainment 各 **23.9% / 27.1% / 33.8%**

**HexAGenT（[arXiv:2605.16637](https://arxiv.org/abs/2605.16637), 2026-05）**

- 处理 **agentic workflow**（多步 LLM 调用 DAG）+ **异构 GPU PD 集群（A100/H100/H200 混部）** 的联合调度
- 把每个请求建模为 online-revealed DAG，按 workflow 完成 horizon 排优先级
- 联合决定 prefill 放置 / decode 放置 / 队列优先级
- SLO scale 减少 **20.1%（95% attainment）和 33.0%（99% attainment）**

**AMPD（[arXiv:2602.14516](https://arxiv.org/abs/2602.14516), 2026-02）**

- 针对 **multi-round 推理**（agent 反复调用、迭代检索）下 prefill-decode workload 是**交错**的
- 自适应决定增量 prefill 在哪做（prefill 节点还是 decode 节点）+ 怎么调度
- 配合 planning 算法做 phase 级资源分配

> **Implication**：通用 FCFS + continuous batching 已不够用。**workload pattern（agentic / multi-round / long-tail）成为调度第一性原理**。

### E. 资源比例理论化

**SLO-Aware Compute Resource Allocation（[arXiv:2603.04716](https://arxiv.org/abs/2603.04716), 2026-03）**

- 第一个把 PD 资源分配做成**理论模型 + 经验 benchmark 的混合**
- Prefill 阶段用 **M/M/1 queuing theory** 建模，从 max prefill throughput + TTFT SLO 反推实际可用 prefill 吞吐
- Decode 阶段：找满足 TPOT 的 batch size，再实测得到 decode 吞吐
- 输出最优 P:D 比例，可在生产场景准确预测

> **Implication**：之前"短输入 1P1D / 中输入 2P2D"是经验，现在有了**可推导的公式**。

### F. 能耗与硬件维度

**DualScale（[arXiv:2602.18755](https://arxiv.org/abs/2602.18755), 2026-02）**

- 把 **DVFS（动态频率调节）+ phase-aware placement** 联合优化
- Prefill 用 MPC（model predictive control）控制频率，因 queue 演化复杂
- Decode 用轻量 slack-aware adaptation，因 memory-bound 动态平滑
- 16× H100 跑 Llama 3.3 70B：vs DistServe **prefill 节能 39% / decode 节能 48%**，SLO 不破

**DUET（[arXiv:2603.15530](https://arxiv.org/abs/2603.15530), 2026-03）**

- **硬件层 PD 分离**：把 prefill 和 decode 分给不同的芯片封装
- Prefill 包：systolic array chiplets + off-package memory（适合大矩阵乘 + 长 SSM）
- Decode 包：vector-unit arrays + 高带宽 in-package memory（适合 token-by-token）
- 在 Nemotron-H-56B / Zamba2-7B / Llama3-8B 上 vs B200 GPU：**4× TTFT, 1.4× 吞吐, 1.5× lower TBT**

> **Implication**：Splitwise 提"用不同 GPU"，DUET 直接做到"用不同芯片"——PD 异构走到了硬件协同设计层。

### G. 基础设施层（开源框架）

**LMCache（[arXiv:2510.09665](https://arxiv.org/abs/2510.09665), 2025-10）**

- 企业级 KV cache layer，把 KV 存到 **GPU 显存外**（DRAM/SSD）
- 跨查询 / 跨 inference engine（vLLM、SGLang）共享 KV cache
- 提供 batched data movement / compute-IO pipelining / modular KV connector
- 对比直接重算：吞吐显著提升

**xLLM 技术报告（[arXiv:2510.14686](https://arxiv.org/abs/2510.14686), 2025-10）**

- 京东开源的多加速器 LLM 推理框架
- 解耦 service-engine 架构 + 智能调度 + **PD 分离 / EPD 分离**（多模态加 Encode 阶段）双策略
- 全局 KV Cache 管理 + 多层执行流水线优化 + adaptive graph mode + xTensor 内存管理 + 推测解码 + 动态 EPLB

### H. 多模态扩展：EPD 分离

**EPD Disaggregation（[arXiv:2501.05460](https://arxiv.org/abs/2501.05460), 2024-12）**

- 多模态模型推理多了一个 **Encode 阶段**（图像/视频特征提取），不能简单套 PD
- 把 Encode / Prefill / Decode **三阶段独立扩缩**
- 分别优化 TTFT 和 TPOT，资源利用率显著提升

> **Implication**：随着 VLM 普及，**EPD 分离会成为多模态推理的标配**——和纯文本 PD 是平行框架，不是替代。

### 综合：2025-2026 趋势矩阵

| 维度 | 2024 主流（DistServe / Splitwise） | 2025H2 ~ 2026H1 进展 |
|---|---|---|
| 分离粒度 | Inter-node PD | + **Intra-GPU PD**（Nexus, RAPID-Serve） |
| KV 传输 | 同一 RDMA 网络 | + **跨 DC**（PrfaaS）+ **GPU 压缩**（SplitZip）+ **存储侧双路径**（DualPath） |
| 调度 | FCFS + continuous batching | **SLO-aware**（Kairos）+ **workflow-aware**（HexAGenT）+ **multi-round-aware**（AMPD） |
| 资源分配 | 经验拍脑袋 | **M/M/1 理论模型**（SLO-Aware） |
| 能耗 | 不考虑 | **DVFS + 调度联合**（DualScale） |
| 硬件 | 同质 GPU | **硬件级分离 chiplet**（DUET）+ **异构集群**（HexAGenT） |
| 模态 | 文本 PD | + **多模态 EPD** |

### 给工程实践的提示

1. **生产部署起步**：先把 [LMCache](https://github.com/LMCache/LMCache) 或 [xLLM](https://arxiv.org/abs/2510.14686) 跑通，比从零搭省半年
2. **agentic / multi-turn workload**：去看 DualPath + AMPD + HexAGenT，传统 PD 分离会被拉胯
3. **长尾 request 严重的场景**：Kairos 的 urgency scheduling + slack adaptation 是当前 SOTA
4. **multi-modal 推理**：直接上 EPD 分离，不要在 PD 上硬套
5. **硬件采购规划**：DUET 的硬件分离思路 + Splitwise 的异构 GPU 思路联合考虑——下一代 inference 集群可能不是同构的
6. **跨 DC 部署**：PrfaaS 是已知唯一可行方案，但要求是 hybrid-attention 模型（KV 小到能跨 DC 传）

---

## 四、推理框架选择

### 4.1 SGLang vs vLLM 当前分工

| 量化 | SGLang | vLLM |
|---|---|---|
| **FP4 MoE** | 当前只支持 TP | **TP / DP / EP 全支持** ← FP4 优选 |
| **FP8 MoE** | DP attention / PD 分离 / MTP 等新特性更新快 ← 优选 | 也支持但跟进略慢 |
| Dense FP16/BF16 | 都成熟 | 都成熟 |

### 4.2 SGLang 部署关键 flag

- `--enable-pd-disaggregation`：开启 PD 分离
- `--kv-transfer-engine nixl|mooncake`：KV 传输后端
- `--enable-dp-attention`：DP attention（DeepSeek 系模型必备）
- `--speculative-num-steps`：MTP 步数
- `--max-running-requests`：单步 batch 上限
- `--max-concurrency`：并发数上限（**默认不设，建议设 100**）

---

## 五、训练侧实战经验（Megatron-LM + MoE）

### 5.1 端到端流水线（Kubernetes 编排）

```
Step 0: 存储 + 权限准备
Step 1: HF safetensors → Megatron checkpoint
        - megatron-bridge AutoBridge.import_ckpt() (TP=1, PP=1, EP=1)
        - 上传分布式存储
        - Python 3.13 兼容性陷阱：删 site-packages/typing.py（vendored 冲突）
Step 2: Tokenizer + preprocess_data.py 分词上传
Step 3: Kubeflow PyTorchJob CRD 编排多节点训练
        - ckpt 下载到本地 NVMe → torchrun 启动
```

### 5.2 Megatron-LM 并行策略约束

- `world_size = TP × PP × CP × DP`
- `world_size / PP ≥ EP`
- **EP 必须整除 MoE 专家数**

例：Qwen3-235B-A22B 实测组合

| Plan | TP | PP | CP | EP | DP |
|---|---|---|---|---|---|
| PlanD | 4 | 1 | 8 | 32 | 1 |
| PlanE | 4 | 2 | 2 | 16 | 2 |

> 现实约束：满足上述约束 + 不 OOM，剩下的合法组合通常**寥寥无几**——并行度搜索很快收敛。

### 5.3 MoE Token Dispatcher：NCCL vs DeepEP/UCCL-EP

| Dispatcher | 特点 |
|---|---|
| **NCCL alltoall** | 标准、稳定，但通用 collective 没有针对 MoE 的优化 |
| **DeepEP** | DeepSeek 开源的 MoE 专用通信库：FP8 dispatch / BF16 combine、节点内 NVLink kernel（~726 GB/s on SM100, EP=8）、跨节点 RDMA、低 SM 占用 |
| **UCCL-EP** | DeepEP 在异构硬件 + 非 IB 网络上的社区移植版（适配 EFA / Broadcom / CX7 / AMD GPU） |

**Megatron 配置：**

```bash
# NCCL 标准方案
--moe-token-dispatcher-type alltoall \
--moe-grouped-gemm \
--moe-pad-expert-input-to-capacity \
--moe-router-fusion

# DeepEP / UCCL-EP（不兼容上面两个 flag）
--moe-token-dispatcher-type flex \
--moe-enable-deepep
```

### 5.4 三连优化（实测在 Qwen3-235B 训练上叠加 +10~15%）

1. **NCCL alltoall → DeepEP / UCCL-EP**：MoE dispatch 真正瓶颈消除
2. **网卡分离**（关键 trick）：
   - 8 张 RDMA 网卡按拓扑亲和**分两组**
   - **奇数 NIC → MoE all-to-all 流量**
   - **偶数 NIC → 其他集合通信（gradient AllReduce 等）**
   - 避免两类流量在同张卡上抢带宽 → 实测对训练效率影响显著
3. **NUMA 亲和绑定**：每个 GPU 进程绑到拓扑相邻的 CPU+内存，避免跨 NUMA 内存访问

---

## 六、基础设施层 benchmark 工具

[Microsoft SuperBenchmark](https://github.com/microsoft/superbenchmark) 是 GPU infra 层 benchmark 的标准工具，覆盖：

- **Kernel Launch Latency**（CPU→GPU 调度成本）
- **PyTorch MatMul Performance**（compute peak）
- **Memory Bandwidth**：D2H / H2D / Device-to-Device

**调试方法论**：
1. 先 NCCL Test 拿 **busbw 基线**（应达物理带宽 70~90%）
2. 再用 superbenchmark 测 **Kernel Latency / 显存带宽**（隔离软件栈和硬件问题）
3. 最后才上模型 benchmark（避免软件栈问题归因到硬件）

---

## 七、跨场景的工程经验抽象

### 推理侧"五件套"

1. **PD 分离 vs Non-PD 不是二选一**：根据输入长度决定
   - 短输入（<16K）→ PD 分离 TPOT 优
   - 超长输入（>100K）→ 多节点 Non-PD 反而更好（KV 传输成本反超）
2. **KV transfer 引擎只看稳定性**：高并发下成功率比 happy-path 吞吐重要 10×
3. **MTP 是免费午餐**：能开就开，特别是短输入高并发场景
4. **max-concurrency 一定要设**：默认不设是性能陷阱，100 是合理起点
5. **FP4 用 vLLM，FP8 用 SGLang**：当前生态分工，未来可能改变

### 训练侧"三件套"

1. **MoE 用 DeepEP / UCCL-EP**，不用 NCCL alltoall（在 EP ≥ 16 的场景下）
2. **多 RDMA 网卡必须分流**：all-to-all 与其他集合通信走不同物理网卡
3. **NUMA 亲和**永远是免费收益

---

## Open Questions

- **6.4 Tbps RDMA 在 NCCL 实测能跑多高 busbw？** 按 3.2 Tbps 接近 400 GB/s 的比例，6.4 Tbps 理论上能到 ~800 GB/s——这对超长上下文 LLM 推理（KV 传输瓶颈）是质变。
- **DP attention 开/关在长输入场景的一般规律是什么？** 有实测显示 4 节点 Non-PD + DP enable 反而最优，这和单节点经验相反。是否因 inter-node 通信成本占比的临界点？
- **PD 分离的"长输入临界点"是多少？** 16K 还是 ≥120K？这个临界点应该是 KV 传输带宽和 prefill 单请求耗时的函数，理论上可以推导。
- **NIXL 在多 PD 节点（>2P2D）时的 scale-out 表现？** 公开测试还停留在 1P1D / 2P2D，4P4D 及以上是否有新瓶颈？
- **异构 GPU PD 部署是否真的可行？** Splitwise 论文提出强 GPU 跑 prefill / 弱 GPU 跑 decode，但实际生产部署罕见——是工程复杂度还是运维不愿意？

## References

### PD 分离原理（奠基）
- [DistServe (OSDI'24, arXiv:2401.09670)](https://arxiv.org/abs/2401.09670) — Prefill-Decode 分离的奠基论文，正式提出 prefill-decoding interferences 概念
- [Splitwise (ISCA'24, arXiv:2311.18677)](https://arxiv.org/abs/2311.18677) — Microsoft 提出的异构硬件 PD 分离，1.4× 吞吐 / 20% 成本下降
- [Mooncake (arXiv:2407.00079)](https://arxiv.org/abs/2407.00079) — Kimi 生产系统的 KVCache-centric 架构，模拟场景 +525% 吞吐

### PD 分离最新进展（2025H2 ~ 2026H1）
- [PrfaaS (arXiv:2604.15039, 2026-04)](https://arxiv.org/abs/2604.15039) — 跨数据中心 PD 分离
- [Nexus (arXiv:2507.06608, 2025-07)](https://arxiv.org/abs/2507.06608) — Intra-GPU 主动 PD 分离
- [RAPID-Serve (arXiv:2601.11822, 2026-01)](https://arxiv.org/abs/2601.11822) — AMD MI300X 上的 CU masking 实现
- [DualPath (arXiv:2602.21548, 2026-02)](https://arxiv.org/abs/2602.21548) — KV 双路径加载，存储 NIC 不对称问题
- [SplitZip (arXiv:2605.01708, 2026-05)](https://arxiv.org/abs/2605.01708) — GPU 友好的无损 KV 压缩
- [Kairos (arXiv:2605.02329, 2026-05)](https://arxiv.org/abs/2605.02329) — SLO-aware 双端调度
- [HexAGenT (arXiv:2605.16637, 2026-05)](https://arxiv.org/abs/2605.16637) — Agentic workflow + 异构 PD 集群调度
- [AMPD (arXiv:2602.14516, 2026-02)](https://arxiv.org/abs/2602.14516) — Multi-round 推理的交错 PD 调度
- [SLO-Aware Compute Resource Allocation (arXiv:2603.04716, 2026-03)](https://arxiv.org/abs/2603.04716) — M/M/1 队列建模 P:D 比例
- [DualScale (arXiv:2602.18755, 2026-02)](https://arxiv.org/abs/2602.18755) — DVFS + phase-aware placement 联合优化
- [DUET (arXiv:2603.15530, 2026-03)](https://arxiv.org/abs/2603.15530) — 硬件层 PD 分离（chiplet 级别）
- [LMCache (arXiv:2510.09665, 2025-10)](https://arxiv.org/abs/2510.09665) — 企业级 KV cache layer
- [xLLM (arXiv:2510.14686, 2025-10)](https://arxiv.org/abs/2510.14686) — 京东开源 PD/EPD 双策略推理框架
- [EPD Disaggregation (arXiv:2501.05460, 2024-12)](https://arxiv.org/abs/2501.05460) — 多模态三阶段分离

### 工具与库
- [NCCL Tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) — busbw 公式权威定义
- [DeepEP](https://github.com/deepseek-ai/DeepEP) — DeepSeek 开源的 MoE 通信库
- [UCCL](https://github.com/uccl-project/uccl) — DeepEP 的异构硬件适配（EFA / AMD / Broadcom）
- [Microsoft SuperBenchmark](https://github.com/microsoft/superbenchmark) — GPU infra 层标准化测试
- [SGLang](https://github.com/sgl-project/sglang) — 推理框架，PD 分离 / MTP / NIXL 支持
- [vLLM](https://github.com/vllm-project/vllm) — 主流推理框架，FP4 MoE 多并行支持完善
- [NIXL](https://github.com/ai-dynamo/nixl) — NVIDIA 高性能 KV transfer abstraction
- [Mooncake TE](https://github.com/kvcache-ai/Mooncake) — KV cache transfer engine
