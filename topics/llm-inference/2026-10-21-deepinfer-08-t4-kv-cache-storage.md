# DeepInfer 专题 T4：KV 缓存分层与存储

> **⚠️ 勘误（主编交叉核对，2026-10-21）**：§KV 体积估算中"约 3.4 KB/token，若不共享则 8–10 KB"的说法**不成立**。
> - 源码和 HF config 已确认 YOCO consumer 层（20–39）共享第 20 层的主 KV。kv_source = {2,8,14,20}，indexer K 同样只存于这 4 层。
> - 分页 KV = 主 KV 720 B + indexer K 170 B = **890 B/token**（推导见 T3 勘误和总报告 §4.3）。
> - 本文后续凡是基于 3.4 KB/token 的容量和带宽推算，数值都应**缩小约 3.9 倍**。100 PiB 存储能容纳的 token 数相应增大，"TTL 达到天级"的结论因此更加宽裕。


> 面向资深推理系统工程师。标记说明：**[源码]** 读代码确认；**[Q&A]** 官方文档；**[估算]** 含算式；**[推断]** 合理推测；**[未核实]** 未能通过可用渠道核实。
>
> 本文 commit 基线：`774ce24`；外部资源访问日期：2026-10-21。

---

## 0. 摘要

DeepSeek V4.1-Flash 推理系统的 KV 缓存架构围绕三个约束构建：**上百万亿 token/天的吞吐量、长达 7 天的 TTL、以及 B200/Ascend 950 上极低的每 token KV 尺寸**。核心设计是把 Block KV 放到专用 3FS 存储集群（10+2 EC，~100 PiB SSD，GDR swap-in），把 SWA 状态放在 Entrypoint 内存目录（400G VLAN 跨机拉取），并以"有界重放"把 SWA 缺失时的重计算代价压到常数。开源代码包含了 SWA/StateBlock 的完整物理布局与落盘逻辑，但 KV Store 读写、Entrypoint 路由、命中策略本体均未开源。

---

## 1. 存储介质分层全表

**[Q&A]** `doc/inference_deployment_qa.md` § "KV cache storage"

| 内容 | 介质 | 冗余方式 | 管理方 |
|---|---|---|---|
| 飞行中 Block KV + State KV | 推理 GPU HBM | 无（volatile） | Prefill / Decode 实例本地 |
| SWA KV 缓存 | 宿主机内存 | 单副本；不上 3FS | Entrypoint 内存目录 + 400G VLAN 跨机拉取 |
| Paged Block KV 缓存 | 专用 3FS 存储集群（NVMe SSD） | 依托 3FS 自身冗余（10+2 EC） | kvcache-store（未开源） |

**关键设计取舍：** Block KV 放专用集群（而非本机 SSD）的核心收益是"每台 Prefill 机都能命中任意 prefix，无需前缀亲和路由"，从而前缀命中逻辑与负载均衡路由策略解耦；存储节点和计算节点故障率相差超一个数量级，不相互影响。

---

## 2. 3FS 分布式文件系统

**[Q&A + 外部]** GitHub: `deepseek-ai/3FS`，README 2026-10-21 读取。

### 2.1 架构

3FS（Fire-Flyer File System）使用 **CRAQ（Chain Replication with Apportioned Queries）** 实现强一致性，元数据服务使用 **FoundationDB** 做事务 KV 后端（无状态元数据 + POSIX 文件接口）。存储节点通过 RDMA（InfiniBand / RoCE）互联，计算节点通过 **USRBIO API** 绕过内核直接操作 I/O（类 DPDK 思路）。

### 2.2 性能指标（源自 README，均已核实）

| 指标 | 数值 | 条件 |
|---|---|---|
| 聚合读带宽（压测） | **6.6 TiB/s** | 180 节点，每节点 2×200 Gbps IB + 16×14 TiB NVMe SSD；500+ 客户端，每客户端 1×200 Gbps |
| KVCache 读峰值 | **40 GiB/s**（客户端侧每节点） | 每节点 1×400 Gbps NIC |
| KVCache GC IOPS | （见文档图表） | GC Worker 负责实时回收 |

### 2.3 在 DeepSeek 推理中的配置

**[Q&A]**：

- **推荐存储服务器配置**：20 × 30.73 TB NVMe SSD，提供 ≥40 GB/s 读 + ≥5 GB/s 写，保证 24 小时数据保留量。
- **10+2 EC（纠删码）**：每 EC 组 12 节点（10 数据 + 2 校验），可同时容忍 2 节点故障。读操作向组内**全部 11 个（或 12 个）节点并发发出请求**，只需任意 10 个响应即可解码返回（"快速响应者赢"策略，天然吸收长尾延迟）。

> **注意**：3FS 开源 README 和代码中**未显式说明 10+2 EC 实现**。EC 功能在 Q&A 中由 DeepSeek 描述，但 GitHub 源码中未见 `erasure` 或 `reed-solomon` 相关代码（**[未核实]**）。README 仅描述 CRAQ 强一致性和 RDMA，未提及 EC。

- **GC Worker**：每存储节点 1 个实例，周期性按 **LRU 策略**淘汰过期 KV 缓存，将磁盘使用率维持在水位线以下。TTL 到期对于 kvcache-store 本身不设逻辑，由 3FS GC 统一管理。
- **总容量**：约 **100 PiB** SSD 存储（物理容量）。
- **TTL**：Agent 流量 **3 天**，Chatbot 流量 **7 天**。实测 TTL 从 7 天降到约 3 天会导致 Chatbot 命中率损失 0.5%–1%（Chatbot 流量有浓重的长尾重复命中特征）。

---

## 3. 容量与带宽建模 [估算]

### 3.1 Block KV 每 token 字节数（V4.1-Flash）

**[源码]** `deepinfer/model_executor/layers/tilelang_ops/cache_common.py`

V41_FP4_FORMAT（主 KV）：
```
quantized_elements = 512, dtype = e2m1 (FP4 = 0.5 B/element)
elements_per_sf = 16, sf_format = e4m3 (1 B/scale)
quantized_bytes = 512 × 0.5 = 256 B
num_scales = 512 / 16 = 32
sf_bytes = 32 × 1 = 32 B
bytes_per_token = 256 + 32 = 288 B/layer/token
```

V41_FORMAT（SWA MXFP8）：
```
quantized_elements = 512, dtype = e4m3 (1 B/element)
elements_per_sf = 32, sf_format = ue8m0 (1 B/scale)
quantized_bytes = 512 B; sf_bytes = 512/32 × 1 = 16 B
bytes_per_token = 512 + 16 = 528 B/layer/token
```

其中 `kv_lora_rank + qk_rope_head_dim = 512`（V41_FORMAT.num_elements，由代码断言约束）。

MXFP4_INDEX KV（indexer K）：
```
quantized_elements = 128, dtype = e2m1 → 64 B
elements_per_sf = 32, sf_format = ue8m0 → 4 B
bytes_per_token = 68 B/layer/token
```

**Block KV 跨层累计（V4.1-Flash，40 层，YOCO 划分）：**

| 层段 | 层数 | 压缩率 | 主 KV/原始 token | Indexer K/原始 token |
|---|---|---|---|---|
| Layer 0–1（SWA only） | 2 | — | 0 B | 0 B |
| Layer 2–19（C2 + SWA） | 18 | 2:1 | 144 B/层 | ~34 B/层（**[推断]**） |
| Layer 20–39（YOCO consumer） | 20 | YOCO 共享 | 同 producer 最后一层 KV（单副本）**[推断]** | — |

**YOCO 关键优化**：Consumer 层（20–39）的 Block KV 全部由 producer 最后一层的 hidden state 投影得到，**不需要单独存储 20 层独立 KV**。**[推断]** 实际存储中 consumer 层复用同一份 block KV（代码中 share-kv map 机制：`kv_mapped_layer_id != layer_id` 时不占用新空间 `cache_engine.py:448`）。

估算 Block KV（仅主 KV，含 YOCO 共享）：
```
C2 层：18 × 144 = 2,592 B
Consumer 共享副本：288 B（1 层）
Indexer（C2 层）：18 × 34 ≈ 612 B
总计：约 3,492 B ≈ 3.4 KB/token
```

**与总报告对比**：总报告估算"约 1 KB/token [估算]"，本文估算 ~3.4 KB/token。差异来源于总报告可能仅计算 SWA state 或单层 MLA 维度；本文包括全部 Block KV 层（含 indexer），但需注意 YOCO 共享假设带有 **[推断]** 标注。**[未核实]** 若 consumer 层不使用 share-kv，总量约 8–10 KB/token。

### 3.2 100 PiB 的自洽性检验

**存储服务器数量 [估算]**：
```
每台服务器原始容量：20 × 30.73 TB = 614.6 TB
物理总容量：100 PiB = 100 × 1024 TB = 102,400 TB
服务器数量：102,400 / 614.6 ≈ 166 台
```

与 3FS README 中 180 节点 6.6 TiB/s 集群高度吻合（**不超过 20% 误差**）。

**10+2 EC 后有效逻辑容量 [估算]**：
```
逻辑容量 = 102,400 × (10/12) ≈ 85,333 TB ≈ 83 PiB
```

**读带宽上限 [估算]**：
```
166 台 × 40 GB/s/台 = 6,640 GB/s ≈ 6.1 TiB/s（client 侧 read）
```

这与 3FS 180 节点集群 6.6 TiB/s 的实测值基本吻合。

**在线推理 GPU 数量 [估算]**：
Q&A："14.4 TB per GPU measured as logical size"：
```
GPU 数量 = 83 PiB 逻辑 / 14.4 TB = 85,333 / 14.4 ≈ 5,926 GPU
（若取 100 PiB 物理：约 7,111 GPU）
```

约 6,000–7,000 推理 GPU（prefill + decode 合计）。

**每 token Block KV 从容量反推 [估算]**：

假设日均流量 155T token，平均 TTL 5 天（agent 3 天 + chatbot 7 天各占半），缓存命中率 r=80%（新写入占 20%），则每天新存 KV token 数：
```
新存 KV token/天 = 155T × (1−0.8) = 31T token/天
7 天窗口内存量 = 31T × 5 天 = 155T 个独立缓存 token
等效 KV/token = 83 PiB / 155T = 85,333 TB / 155e12 ≈ 550 bytes/token
```

550 bytes/token 比 3.4 KB/token 的估算小约 6 倍。考虑到：
1. 实际缓存命中率可能更高（>80%），新写入更少；
2. Block KV 按 128-token 粒度（KVSTORE_GRAIN）写入，不是每 token；
3. YOCO consumer KV 若确实共享则可能远低于 3.4 KB；
4. 100 PiB 是 3FS 集群总容量，其中可能包含训练检查点等非 KV 流量（**[未核实]**）。

结论：**100 PiB + 14.4 TB/GPU + 155T tok/day + 5 天 TTL 在数量级上自洽**；精确的 Block KV 字节数因 YOCO 共享 / 命中率细节无法从公开信息精确反推。

### 3.3 单台存储服务器能支撑多少 prefill 卡的 swap-in 带宽？

每台存储服务器：40 GB/s 读取带宽。
每卡 PCIe 4.0 x16 ≈ 64 GB/s（B200 接近 128 GB/s with Gen5），H2D（GDR）受 PCIe 限制约 32–64 GB/s。

若每块 prefill GPU 需要 swap-in 带宽 B GB/s（来自 3FS，通过存储 NIC）：
```
存储服务器能支撑的 prefill 卡数 ≈ 40 / B
```
- 若 B = 2 GB/s（平均 prefill swap-in）：≈ 20 卡/服务器
- 若 B = 5 GB/s（高缓存命中率场景）：≈ 8 卡/服务器

166 台服务器 × 20 卡 = 3,320 prefill 卡；166 × 8 = 1,328 prefill 卡。与约 3,000–4,000 prefill GPU 的量级相符。

**V3 vs V4.1 对比 [估算]**：V3 使用 MXFP8 主 KV（512 B/token/层）和 BF16/FP8 量化，无 YOCO，有 61 层。每 token Block KV ≈ 61 × 512 ≈ 31 KB（无 EC 开销），远高于 V4.1 的 ~3.4 KB；V4.1 存储需求降低约一个数量级，是 TTL 从 V3 时代数小时拉长到 7 天的物质基础。

---

## 4. 源码解析：StateBlock、落盘接口与 KVSTORE_GRAIN

**[源码]** `deepinfer/worker/cache_engine.py`（commit 774ce24，全文可读，共 979 行）

### 4.1 KVSTORE_GRAIN

```python
# cache_engine.py:32
KVSTORE_GRAIN = 128
"""kvstore 落盘粒度：只在 token 数为 128 的倍数处落盘"""
```

SWA KV 只在位置为 128 的整倍数时写盘，由此引出 `select_state_slice_to_store` 里的断言：
```python
# cache_engine.py:787
assert last_hit_position % KVSTORE_GRAIN == 0
```

### 4.2 StateBlock 物理布局

**[源码]** `cache_engine.py:221–306`

```
StateBlock 内存布局（低地址到高地址）：
  [stored region: SWA]          ← 需要落盘；PE→CE 搬运；20 个 SWA 层
  [dropped region: YOCO consumer SWA | compression state]  ← 不落盘
```

关键字段（由模型 config 计算，不依赖 block_size）：

| 字段 | 含义 | V4.1-Flash 数值 |
|---|---|---|
| `swa_store_num_layer` | 需落盘的 SWA 层数 | **20**（layer 0–19，producer） |
| `swa_drop_num_layer` | YOCO consumer SWA 层数（不落盘） | 20（layer 20–39） |
| `swa_block_size_inference` | 推理窗口 token 数 | **134**（=128 + MAX_BACKTRACKING_TOKENS=6） |
| `swa_bytes_per_layer_inference` | 每层推理 SWA 字节数 | 134 × 528 = 70,752 B |
| `state_size_to_disk` | 落盘总字节数 | **20 × 70,752 = 1,415,040 B ≈ 1.35 MB/seq** |

`state_size_to_disk = self.swa_store_total_bytes_inference_pad`（`cache_engine.py:306`）

Q&A 中 "20×134×528" 的来源：`swa_store_num_layer=20`，`swa_block_size_inference=128+6=134`，`swa_size_per_token=528`（MXFP8）。**已在源码中完全核实。**

### 4.3 get_kvstore_swa_block_size

**[源码]** `cache_engine.py:101–110`

```python
def get_kvstore_swa_block_size(window_size: int) -> int:
    return get_inference_swa_block_size(window_size) + (
        (envs.DEEPINFER_PREFILL_STORE_EXTRA_SIZE + get_yoco_store_extra_tokens())
        if envs.DEEPINFER_PREFILL_STORE_EXTRA_STATE else 0
    )
```

`DEEPINFER_PREFILL_STORE_EXTRA_STATE`（布尔型，默认关）开启时，kvstore 视角的 ring 额外存 `PREFILL_STORE_EXTRA_SIZE`（默认 128 token）+ YOCO 截断余量，以保证多轮命中。推理视角（inference block size）保持不变。

### 4.4 select_state_slice_to_store 与 load_state_from_store_to_prefill_cache

**[源码]** `cache_engine.py:763–848`

两个核心接口（全开源，可调用）：

- `select_state_slice_to_store(state_id, last_hit_position, state_cache, state_tensor, TO_CE)`:
  从 GPU state cache 中读出需要落盘的那段（`TO_CE=False`）或完整 PE→CE 传输 state（`TO_CE=True`）。extra-state 关闭时直接截取前 `state_size_to_disk` 字节；开启时需对 SWA ring buffer 做重排（`select_state_slice` 核函数）。
  
- `load_state_from_store_to_prefill_cache(state_id, last_hit_position, state_tensor, state_cache)`:
  从存储 state tensor 写入 prefill state cache 的指定 slot；extra-state 开启时调 `load_state_to_cache` 做逆向重排。

### 4.5 命中策略 L−134

**[Q&A]** § "KV cache hits and recomputation"：

```
hit_length = min(L − 134, L1)
```

- `L` = 输入总 token 数；
- `134 = 128 (SWA window) + 6 (backtracking budget)`：YOCO 截断要求最后 134 token 不可复用；
- `L1` = Block KV 查询返回的命中长度（128-token block 的倍数）；
- 若 `[L1−128, L1]` 区间内有 SWA 缓存 → hit = L1；否则 hit = L1−128（退一块，重计算 128 token）。

此逻辑（`L − 134` 上限）在开源代码中的 `select_state_slice_to_store` 断言 `last_hit_position % KVSTORE_GRAIN == 0` 和 `get_inference_swa_block_size` 中有间接体现，但命中判断本体（查询 Block KV + 查询 SWA 目录 + 退一块决策）属于**未开源**的 kvcache-store / Entrypoint 逻辑。

### 4.6 PREFILL_STORE_EXTRA_* 环境变量

**[源码]** `deepinfer/environ.py:163–164`

```python
DEEPINFER_PREFILL_STORE_EXTRA_STATE = EnvVar.bool()  # 默认关
DEEPINFER_PREFILL_STORE_EXTRA_SIZE = EnvVar.int(default=128)  # 追加 128 token ring
```

开启后 kvstore ring 比推理窗口多存 128 token，保证在 KVSTORE_GRAIN 对齐处总有完整的 inference 窗口可用。生产是否开启 **[未核实]**（代码路径两分支均有覆盖，测试 `test_state_cache_ops.py` 两种情况都测了）。

### 4.7 fork_seq（前缀缓存的开源等价物）

**[源码]** `examples/benchmark.py:337–349`

开源版没有 KV Store / Entrypoint，但提供了 `scheduler.fork_seq()` 接口：将一个已完成 prefill 的序列（source_seq）的 cached token 直接"分叉"给若干新序列，复用其 state slot 和 block 指针，无需重新 prefill。这是开源版中**唯一可直接利用前缀缓存的入口**。

### 4.8 scheduler 中的 hit_tokens / swa_valid_start / zero_states hook

**[源码]** 全库搜索结果：

- `hit_tokens` / `swa_valid_start`：仅在 `tests/engine/test_flops_estimator.py` 中出现，用于 FLOPs 计算中的 SWA 有效起始位置参数。**主调度路径（scheduler 本体）中不存在**，属于未开源的 kvcache-store 命中返回结构。

- `zero_states`：在 `deepinfer/engine/llm_engine.py:890` 和 `deepinfer/worker/base.py:160` 中出现，对应于新序列分配 state slot 时清零操作（`clone_and_zero_states_and_blocks`，C++ 扩展实现），是开源代码的一部分。

**小结：开源 vs 未开源**

| 功能 | 开源状态 |
|---|---|
| StateBlock 物理布局、字节数计算 | ✅ 开源 |
| select_state_slice_to_store / load_state_from_store_to_prefill_cache | ✅ 开源 |
| KVSTORE_GRAIN、get_kvstore_swa_block_size | ✅ 开源 |
| fork_seq（前缀共享） | ✅ 开源 |
| zero_states / clone_and_zero hooks | ✅ 开源 |
| kvcache-store（Block KV 读写、索引） | ❌ 未开源 |
| Entrypoint（命中查询、哈希链匹配） | ❌ 未开源 |
| hit_tokens / swa_valid_start 调度字段 | ❌ 未开源（只在测试中有形参） |
| 三阶段（match / fetch / swap-in）流水 | ❌ 未开源 |
| GDR swap-in 实现 | ❌ 未开源 |

---

## 5. GDR Swap-in 与三阶段流水

**[Q&A]** § "How is the cache-related overhead overlapped away?"

三阶段：

```
1. Matching（匹配）  ──── Entrypoint 本地 kvcache-store CHECK role
                          hash-chain prefix match over metadata only
                          [必须在路由前完成，无法隐藏]
2. Fetching（拉取）  ──── 独立进程异步执行
                          3FS → host shared memory
                          [与 prefill 计算重叠]
3. Swap-in（装载）   ──── engine 进程内线程
                          host shared memory → HBM
                          [每层 H2D，与 forward 计算重叠]
```

**GDR（GPU-Direct RDMA）**：swap-in 使用 RDMA NIC 直接写 HBM，绕过 CPU 和 CUDA 拷贝引擎。理由：CUDA copy engine 对 KV cache block 细粒度拷贝表现差，GDR 用 RDMA NIC 饱和 PCIe 带宽而不消耗 SM 资源。

开源代码中未见 GDR 相关实现；`userbuffer.py` 中有 `use_cpu_rdma_storage`（DeepEP V2 Engram 相关 RDMA 存储），但这属于 EP 通信，非 KV swap-in。

---

## 6. DualPath（arXiv 2602.21548）深读

**标题**：DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference
**作者**：Wu Yongtong, Chen Shaoyuan, Zhong Yinmin 等（北京大学 / DeepSeek 联合，含 Jin Xin, Huang Panpan）
**时间**：2026-02-25（arXiv）

### 6.1 问题背景

在 PD 分离架构下，传统设计中 KV Cache 始终从存储节点直接发往 **prefill 引擎**。Agentic 工作负载（长上下文、多轮、高命中率）导致 prefill 侧存储 NIC 持续饱和，而 decode 侧存储 NIC 大量空闲。从 Ampere 到 Blackwell 的 GPU 迭代中，I/O-compute 比下降了 14.4×，使问题更严重。

### 6.2 DualPath 机制

引入 **双路径 KV Cache 加载**：

```
路径 A（传统）：3FS → Prefill NIC → Prefill HBM → [KV transfer P→D]
路径 B（新增）：3FS → Decode NIC → Decode HBM → [RDMA via compute network → Prefill HBM]
```

路径 B 通过 decode 侧空闲存储 NIC 拉取 KV，然后经 compute NIC（東西向，带宽远大于存储 NIC）以 RDMA 推送给 prefill engine，天然与 model execution 通信隔离（compute network 和 storage network 物理隔离）。

### 6.3 调度算法

**全局调度器**动态决策每个请求走哪条路径，目标是同时平衡：
1. Prefill / Decode GPU 间的计算负载；
2. Prefill / Decode 侧存储 NIC 的带宽利用率。

具体算法：
- 维护 prefill 和 decode 引擎的当前存储 NIC 利用率；
- 当 prefill NIC 接近饱和且 decode NIC 有余量时，新到达的命中请求优先走路径 B；
- 路径 B 引入的额外延迟（decode→prefill RDMA）需纳入 TTFT 预算。

**与 DeepSeek Q&A 的关系**：Q&A § "Decode Load Balancer" 中写道："See DualPath's intra-group scheduling algorithm for details." DualPath 论文中"intra-group scheduling"指在一个 Decode 实例内同时平衡 **请求数（决定 batch size）** 与 **总 KV 长度（决定 attention 开销）**——短请求送到并发低但总 KV 长的 GPU，长请求送到总 KV 短的 GPU。**[推断]** Q&A 中引用的是 DualPath 论文的这一局部算法，而非整个 DualPath 双路径机制（DeepSeek 生产系统可能已集成类似双路径，但未在 Q&A 中明确说明）。

### 6.4 性能数字

- 离线推理吞吐：最高 +87%（1.87×）
- 在线 serving 吞吐：平均 +96%（1.96×），不违反 SLO
- 评测在三个模型上进行，使用真实 agentic workload

---

## 7. 业界方案对比

### 7.1 主要方案速览

| 方案 | 存储介质 | 路由策略 | 粒度 | 论文/项目 |
|---|---|---|---|---|
| **RadixAttention (SGLang)** | GPU HBM（Radix tree） | 最长前缀匹配，前缀亲和路由 | token 级 | SGLang 2312.07104 |
| **UnifiedRadixTree (vLLM)** | GPU HBM，可扩展 | 单机/多机前缀共享 | block 级 | vLLM v0.5+ |
| **vLLM APC + KV connectors** | HBM/CPU/SSD 分层 | 前缀亲和 + 连接器插件 | block 级 | vLLM v0.6+ |
| **vLLM HiSparse** | GPU HBM | — | block 级 | vLLM PR |
| **SGLang HiCache** | CPU/NVMe 多层 | 分层 RadixTree | block 级 | SGLang docs |
| **LMCache** (2510.09665) | GPU/CPU/SSD + 跨引擎 | 前缀匹配 + 模块化 connector | token 级 | Liu et al. 2025 |
| **CacheBlend** (2405.16444) | 可离线存储 | 前缀无关，部分重计算融合 | token 级 | Yao et al. 2024/2025 |
| **Mooncake Store** (2407.00079) | DRAM + SSD 池化 | KVCache-centric 亲和路由 | chunk 级 | Qin et al. FAST'25 |
| **NVIDIA KVBM** | HBM + PCIe SSD | — | block 级 | NVIDIA Blackwell MIG |
| **DeepSeek 3FS cluster** | NVMe SSD（专用集群） | 无前缀亲和（全局池） | 128-token grain | 本文 Q&A |

### 7.2 全局池 vs 亲和路由

**亲和路由（RadixAttention / Mooncake）**：把有相同前缀的请求路由到同一台机器，最大化本机 HBM/DRAM 命中率。缺点：前缀分布不均时产生热点；scale-in/out 需迁移缓存；与负载均衡目标有内在冲突。

**全局池（DeepSeek 3FS）**：任何 prefill 机器都可访问任意缓存，前缀路由与负载均衡完全解耦。代价：命中判断（matching）必须在路由前完成（无法与计算重叠）；每次命中需通过网络拉取 KV（latency 比本机 HBM 高）。DeepSeek 用 GDR 和异步 fetch 流水隐藏拉取延迟，用 SWA bounded replay（常数重计算代价）容忍 SWA 缺失。

**LMCache** (2510.09665, Liu et al.)：开源 KV 缓存层，插件式对接 vLLM/SGLang，支持 GPU→CPU→SSD 分层和跨引擎共享（PD 传输）。企业级用户报告：远程存储拉取 KV 对 TTFT 有显著收益；context truncation（截断上下文）可将前缀命中率降低约 50%（与 DeepSeek TTL 3→7 天命中率损失结论一致）。

**CacheBlend** (2405.16444, Yao et al.)：针对 RAG 场景，预计算多个文档片段的 KV 后**选择性重计算**一小部分 token（以修正跨文档的 attention 偏差），使 non-prefix KV 也能被复用。TTFT 降低 2.2–3.3×，吞吐提升 2.8–5×（对比全量重计算）。思路与 DeepSeek SWA bounded replay 有异曲同工之处（都是用少量重计算换存储/命中灵活性）。

**DeepSeek API 硬盘缓存定价**：**[未核实]** api-docs.deepseek.com 在 2026-10-21 的请求无法正常解析 pricing 页面内容（JS 渲染，curl 取不到数据）。从已知信息：DeepSeek API 提供 prefix cache（disk cache），命中时输入 token 价格打折（约 0.1× 或 0.5×），具体折扣以官网为准。

---

## 8. 取舍分析与开放问题

### 8.1 全局 KV 池的根本约束

DeepSeek 选全局 3FS 池的前提是 **V4.1-Flash 的 KV 够小**（FP4 + C2 + YOCO = ~3 KB/token vs V3 的 ~30 KB）。如果 KV 没有这一量级的压缩，存储带宽会成为硬约束，亲和路由会更有吸引力。这是"模型-系统协同设计"的典型案例。

### 8.2 Matching 延迟无法隐藏

**所有**前缀缓存系统（含 DeepSeek）都面临命中判断必须在调度前完成的问题。DeepSeek 用 Entrypoint 的哈希链 prefix match（纯元数据，本地内存）最小化 matching 时间。SGLang/vLLM 的 RadixTree match 更简单（纯本地 CPU 操作），但没有跨机命中能力。

### 8.3 DualPath 的适用场景

DualPath 对 **decode 侧存储 NIC 大量空闲**的场景才有显著收益。DeepSeek 生产系统 decode 用宽 EP（EP64–128），每个 EP 组的 decode 实例存储 NIC 利用率本身较低，DualPath 机制有很好的应用场景。Q&A 提到的 decode LB 引用了 DualPath intra-group 调度算法，说明 DeepSeek 已借鉴或采用了其中部分思路。

---

## 9. 建议：自建 DeepSeek 式 KV 存储的组件清单

### 9.1 最小可行方案

```
1. 分布式 SSD 文件系统
   - 推荐：3FS 开源版（MIT 协议）
   - 关键配置：CRAQ 强一致；10+2 EC；GC Worker（LRU 水位控制）
   - 硬件：每节点 ≥16 × NVMe SSD（≥1 TB/块）+ 2×200 Gbps RDMA NIC

2. KV 元数据索引（hash-chain prefix match）
   - 自研或基于 LMCache 的 connector 框架
   - 需持久化到 3FS 或 FoundationDB

3. 异步 Fetch + GDR Swap-in
   - 独立进程从 3FS 拉到 host shared memory（POSIX shm 或 RDMA buffer）
   - RDMA NIC 直写 GPU HBM（GPUDirect Storage 或 GDR）
   - 需要 host-side RDMA NIC（如 ConnectX-7 / BlueField-3）

4. SWA 内存目录
   - 可以是 Redis / 分布式 hashmap，记录 sequence_hash → machine_ip
   - 跨机拉取通过 400G VLAN（RDMA 或 TCP）
```

### 9.2 容量规划清单

| 参数 | 计算方式 | 示例（1000 GPU 集群） |
|---|---|---|
| 每 GPU 逻辑存储 | 14.4 TB（参考比） | 14,400 TB = 14.1 PiB |
| 物理存储（10+2 EC） | 逻辑 × 12/10 = × 1.2 | ~17 PiB 物理 |
| 存储服务器数 | 物理 / 614 TB | ~28 台 |
| 存储读带宽需求 | GPU 数 × prefill swap-in 带宽 | 1000 × 2 GB/s = 2 TB/s（需 28×40=1.12 TB/s 不足，需扩至 50 台） |
| SWA 内存目录 | max_concurrent_seqs × 1.35 MB | 1M seqs × 1.35 MB = 1.35 TB（需大内存机器或 NVMe-backed） |
| TTL 与 GC | 按流量比率设定 | Agent 3 天；Chatbot 7 天 |

### 9.3 关键注意事项

1. **Block KV 粒度 = 128 token**（KVSTORE_GRAIN），元数据索引按此粒度建 hash。
2. **SWA state 大小 = 20 × 134 × 528 B ≈ 1.35 MB/seq**（V4.1-Flash），不要用总 state size（含 YOCO consumer 的 dropped region）。
3. **有界重放**（bounded replay）是容忍 SWA miss 的关键工程手段，避免了从 0 重算整个上下文。
4. **GDR 必须有 host RDMA NIC + 驱动支持（nv_peer_mem 或 GDS）**；若暂不具备，可先用 cudaMemcpy，但会消耗 PCIe 带宽并抢占 SM。
5. 3FS 开源版**未包含 10+2 EC**（至少未在 README/代码中公开），自建时需额外评估或选用其他具备 EC 能力的分布式存储（如 Ceph RADOS）。

---

## 10. 参考

| 来源 | 标识 |
|---|---|
| DeepInfer Q&A | `doc/inference_deployment_qa.md` |
| DeepInfer 源码 | `deepinfer/worker/cache_engine.py`（L32, L101, L221, L306, L544, L763, L821）|
| 3FS README | github.com/deepseek-ai/3FS（2026-10-21 读取）|
| DualPath 论文 | arXiv 2602.21548（Wu et al., 2026-02-25）|
| LMCache 论文 | arXiv 2510.09665（Liu et al., 2025-10-08）|
| CacheBlend 论文 | arXiv 2405.16444（Yao et al., 2024-05-26）|
| Mooncake 论文 | arXiv 2407.00079（Qin et al., 2024-06-24，FAST'25 Best Paper）|
