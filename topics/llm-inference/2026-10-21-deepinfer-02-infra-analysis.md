# DeepInfer 调研 02：大规模 LLM 推理 Infra 技术调研与解析
## 以 DeepSeek DeepInfer / V4.1-Flash 生产系统为主线

> 截至 **2026-10-21**。DeepInfer 的内容来自源码（commit `774ce24`）和 `doc/inference_deployment_qa.md`。业界方案来自论文和项目文档，所引 arXiv 编号均已逐一核对存在。
> 标 **[估算]** 的是本文根据源码推算的数字，标 **[未核实]** 的是没有找到一手来源的内容。

---

## 0. 摘要

DeepInfer 和 V4.1-Flash 的组合，是目前公开资料最完整的一套"**模型-系统协同设计**"的大规模 MoE 推理系统。本文把它拆成 9 个技术层，逐层对照业界最新方案：

| # | 技术层 | DeepInfer 的选择 | 业界主要替代方案 | 一句话判断 |
|---|---|---|---|---|
| 1 | 系统形态 | Vision Encoder / Prefill / Decode 三段分离（E/P/D）+ 独立存储节点 | DistServe、Splitwise、Mooncake、semi-PD、AFD（MegaScale-Infer、Step-3） | PD 分离已是共识；下一个分歧点是"注意力-FFN 分离"要不要做 |
| 2 | 并行与 MoE | 注意力数据并行 + 宽 EP64–128，加冗余专家 EPLB、Mega MoE 融合 kernel | DeepEP / pplx / UCCL-EP；TBO / DBO；LPLB / METRO | 宽 EP 是 decode 降本的主杠杆；负载均衡的目标函数仍有争议 |
| 3 | 注意力与 KV 压缩 | SWA(128) + C2/C1 压缩 KV + FP4 稀疏 indexer + YOCO + 跨层共享 | MLA、DSA / NSA、混合 SWA、KV 量化 | KV 降到约 0.9 KB/token（源码核实），从根本上改写了存储与网络的设计约束 |
| 4 | KV 缓存与存储 | 专用 3FS 集群（10+2 纠删码，约 100 PiB，TTL 3–7 天）+ GDR swap-in + SWA 有界重放 | RadixAttention、APC、LMCache、Mooncake Store、HiCache、KVBM | "分离式全局 KV 池"在大规模场景胜出，代价是命中计算无法隐藏 |
| 5 | 调度 | ZOS 异步调度；prefill 用代价模型求预算；decode 双维负载均衡 | Sarathi chunked prefill、NanoFlow、Dynamo / llm-d 路由 | CPU 开销已可压到零，负载均衡（约为理想值的 80%）是新瓶颈 |
| 6 | 投机解码 | DSpark：块式起草 + 置信度头 + 按边际收益变长验证 | EAGLE-3、MTP、动态草稿长度 | 从"猜多少个"转向"验证多少个" |
| 7 | 低精度 | MXFP4 专家权重、FP4 主 KV、MXFP4 indexer、FP8 激活 / SWA | NVFP4、FP8 | FP4 已从权重扩展到 KV |
| 8 | 确定性 | 批次不变 kernel + 按 (seed, position) 播种的采样 | Thinking Machines、vLLM / SGLang 确定性模式、Vosti | 正在成为 RL 和评测的刚需，严格的逐位确定仍很难 |
| 9 | 硬件 | SM100 + Ascend 950，经 TileLang 双后端 | GB200 / GB300 NVL72、CloudMatrix384 | "scale-up 域大小"决定 EP 上限，跨厂商靠 DSL 抽象 |

---

## 1. 第一性原理：为什么大规模 MoE 推理长成这样

**两个阶段的瓶颈完全不同：**
- **Prefill**：每个 token 都参与矩阵乘，算术强度高，瓶颈在**算力**。
- **Decode**：每步每条序列只出 1 个（或 1+k 个投机）token，要读全部激活专家的权重和全部 KV，瓶颈在**显存带宽**。

**MoE 放大了 decode 的矛盾。** V4.1-Flash 有 384 个路由专家，每个 token 只激活 top-6。单卡 batch 小时，每个专家只分到个位数 token，GEMM 完全是访存受限。要提高专家 GEMM 的利用率，只能让**每个专家看到更多 token**，有两种办法：
- 把专家摊到更多卡上（EP↑），每卡只放少数专家，在全局把 token 汇聚过来；
- 增大全局 batch，即 DP↑。

这就是 **注意力数据并行 + 宽 EP** 的根本动机：注意力按请求切分（DP），MoE 按专家切分（EP），`#DP = #EP`。

**KV 大小决定了几乎所有系统设计的边界：**
- 单卡并发上限：decode 每卡能放多少条序列；
- PD 之间的传输带宽；
- 前缀缓存能保留多久、需要多少存储；
- 能否跨机房部署。

V4.1 在模型侧把 KV 压到极小（§3），这直接让 DeepSeek 可以选择"**专用存储集群 + 长 TTL + 跨相邻机房 PD**"这套系统设计。这是模型-系统协同设计最典型的例子。

---

## 2. 系统形态：E/P/D 分离与演进路线

### 2.1 学术与工业脉络

| 工作 | 年份 | 核心观点 | 关键数字 |
|---|---|---|---|
| **Splitwise**（arXiv 2311.18677） | 2023 | prefill 与 decode 的特性不同，可以放在不同硬件上 | 同等成本下吞吐提升 1.4 倍，或同功耗下提升 2.35 倍 |
| **DistServe**（arXiv 2401.09670，OSDI'24） | 2024 | 以 goodput（同时满足 TTFT 和 TPOT 的请求率）为目标做分离 | 请求率提升 7.4 倍，或 SLO 收紧 12.6 倍 |
| **Mooncake**（arXiv 2407.00079，FAST'25 最佳论文） | 2024 | 以 KV 缓存为中心：池化 CPU / SSD，做成全局 KV 层；过载时提前拒绝 | 长上下文场景多服务 75% 请求 |
| **DeepSeek-V3 推理系统** | 2025-02 | 节点内 EP 加大规模 PD 分离 | H800 每节点输入 73.7k / 输出 14.8k tok/s，成本利润率 545% |
| **MegaScale-Infer**（arXiv 2504.02263，字节） | 2025 | 注意力与专家分离（"Disaggregated Expert Parallelism"），两类模块独立扩缩 | — |
| **semi-PD**（arXiv 2504.19867） | 2025 | 计算按 SM 切分做分离，存储统一，避免权重重复和 KV 迁移 | DeepSeek 系列端到端延迟降低 1.27–2.58 倍 |
| **Step-3 AFD**（arXiv 2507.19427） | 2025 | 注意力-FFN 分离 + MFA 注意力 | 50 ms TPOT 下 4,039 tok/s/GPU，对比 DeepSeek-V3 的 2,324 |
| **AFD 设计空间研究**（arXiv 2605.28302） | 2026 | 系统性评估 AFD 的适用边界 | — |
| **DeepSeek V4.1 系统**（本文） | 2026 | E/P/D 三段分离 + 专用 3FS 存储；decode 用宽 EP64–128 | 单集群每天 155 万亿 token |

### 2.2 DeepInfer 的选择与解读

DeepInfer 的整体链路是：DeepSeek Recipe（API 与 token 互转）→ Entrypoint（缓存匹配 + 路由）→ 三类负载均衡器 → E / P / D 三种服务 → 3FS。

几个关键选择：

1. **Prefill 用"能放下的最小 EP"（EP2–8），decode 用"能拿到的最大 EP"（EP64–128）。**
   - Prefill 本身算力受限，batch 已经足够大。EP 越小，all-to-all 越少，故障域也越小。
   - Decode 需要更大 EP 来摊薄专家访存。Q&A 中给出了 EP8 与 EP64 的对比图：宽 EP 显著降低每步的专家访存量。
2. **Vision Encoder 单卡部署，推荐用便宜的卡。** 视觉编码是无状态的纯计算，独立部署后可以用低端卡，这是 EPD 比 PD 多出来的成本收益。
3. **最小高效部署单元是 64 卡 prefill + 64 卡 decode。** 这说明这套设计面向**超大规模**服务，对中小规模部署并不友好。这也是 vLLM / SGLang 仍然需要支持 TP、小 EP 等多种形态的原因。
4. **EP 故障处理：** 一张卡坏了，整个实例不可用。重启时按剩下的健康卡**重新确定 EP 大小**。容错靠的是"冗余实例配额 + 请求级重路由"，而不是缩短单实例重启时间（重启要分钟级）。
5. **生产上不用 TP、也不用 CP。** 代码里实现了 CP，但生产不启用，说明 DP + EP 在这个模型规模下已经够用。TP 在宽 EP 场景下反而增加通信。

### 2.3 开放问题：要不要做注意力-FFN 分离

MegaScale-Infer、Step-3 和 2605.28302 都在推动 AFD：注意力（访存受限，KV 带状态）和 FFN（专家 GEMM，无状态）放到**不同硬件**上，各自独立扩缩。

DeepSeek 的做法是在**同一个 scale-up 域内**用宽 EP 加 Mega MoE 融合 kernel 来解决专家利用率问题，没有引入 AFD。可能的原因有两点 **[推断]**：
- NVL72 和 Ascend 超节点让 64–128 卡全部处在同一高带宽域内，EP 的通信代价已经很低；
- AFD 需要额外的注意力-FFN 往返通信和流水编排，故障域也更复杂。

---

## 3. 并行与 MoE：宽 EP、EPLB、通信库

### 3.1 通信库

| 库 | 来源 | 特点 |
|---|---|---|
| **DeepEP（V2）** | DeepSeek | 统一的 `EPBuffer` 接口：`do_cpu_sync=True` 走高吞吐（prefill），`False` 走低延迟（decode）；FP8 dispatch、BF16 combine；V2.5 起后端从 NVSHMEM 切换到 NCCL GIN（要求 NCCL ≥ 2.32.3） |
| **DeepEP-Ascend** | DeepSeek / 华为 | DeepInfer 的 Ascend 后端，包名同为 `deep_ep` |
| **pplx-kernels** | Perplexity | GPU 发起的 all-to-all，用于其生产环境 |
| **UCCL-EP** | 学界 | GPU 发起 RDMA；arXiv 2610.01380 测得在约 0.7 µs 的硬件下限之上额外增加约 4.6 µs 的发起延迟 |
| **NCCL / HCCL** | 厂商 | DeepInfer 把 NCCL 的 SM 占用限制为 32（`DEEPINFER_NCCL_NUM_SMS`） |

### 3.2 Mega MoE：用融合代替重叠

业界主流的"通信-计算重叠"做法是 **TBO / DBO**（SGLang TBO、vLLM DBO）：把一个 batch 切成两个微批，A 在算的时候 B 在通信。NanoFlow（arXiv 2408.12757）更进一步，在设备内并行跑多个 nano-batch，吞吐提升 1.91 倍。

DeepInfer / DeepGEMM 走的是另一条路，即 **Mega MoE**：把 dispatch、linear1、SwiGLU、linear2、combine 融合进**同一个 kernel**，用 SymmBuffer 加 NVLink 在 kernel 内部让通信与 tensor core 重叠。代码中，只要 EP 组位于 NVLink 域内就会自动启用（`mlp.py:184-188`）。

Q&A 的说法是："在 scale-up 域内，无论 EP 大小和 batch 大小，Mega MoE 都显著快于非融合 MoE。"

| 方案 | 优点 | 缺点 |
|---|---|---|
| TBO / DBO（微批重叠） | 通用，跨节点 RDMA 也能用 | 需要切微批，小 batch 时效果差；调度复杂 |
| Mega MoE（kernel 融合） | 没有微批开销，小 batch 也有效 | 只能用在 scale-up 域内；kernel 复杂度极高，和硬件强绑定 |

### 3.3 EPLB：冗余专家与目标函数之争

**DeepInfer 的实现**（`eplb/__init__.py`、`policy.py`）：

**冗余专家的数量**会向上取整，使 `(路由专家数 + 冗余数)` 能被 `ep_size` 整除。生产参数如下：

| 阶段 | 每卡冗余专家 | 统计窗口 | 重排周期 | 实测 moe_lb（最慢 rank / 平均） |
|---|---|---|---|---|
| Prefill | 4 | 100 步 | 每 1,000 步 | 1.074（EP4）/ 1.045（EP2） |
| Decode | 1（EP64 时取整后每卡 7 个专家） | 1,000 步 | 每 10,000 步 | 1.090 |

**算法分两阶段贪心：**
1. 把冗余槽位分给"会让每步每副本最大负载最小"的逻辑专家；
2. 把副本按负载降序放到"使**历史各步最大负载之和**最小"的 rank 上。这是对整个窗口内的步序列联合优化，而不只看平均负载。

**权重搬运**：在 NVLink 域内用 `torch symmetric_memory` 逐层镜像拷贝，其余情况用 P2P。

**业界替代方案：**
- **SGLang LPLB**：用线性规划逐层求解副本分配。**Waterfill** 则在 dispatch 时均衡共享专家的 token。
- **METRO**（arXiv 2512.09277）提出了根本性的批评：在 decode 这种访存受限的区间，EPLB 按 token 数均衡反而会**增加每卡激活的专家数**，而决定耗时的其实是"要读多少个专家的权重"。METRO 改为均衡"每卡激活专家数"，报告 decode 延迟降低 11–22%，固定 SLO 下吞吐最高提升 4.11 倍。

**解读**：DeepInfer 在 decode 阶段每卡只放 1 个冗余专家、每 10,000 步才重排一次，说明在 EP64 下"专家权重访存"已经很均匀，主要矛盾不在 token 均衡上。METRO 的思路对小 EP、大 batch 的部署可能更有价值 **[推断]**。

### 3.4 Engram 与"表并行"

V4.1 在第 1 层和第 14 层注入 n-gram 哈希 embedding：
- 表规模：每个 n-gram 1600 万行，n 为 2–4，8 个头 × 256 维，FP8 存储；
- 表按 rank 分片，查表走 DeepEP `EngramBuffer`，表也可以放在 CPU 内存。

这引入了第三种并行维度（表并行，代码中称 "enp" group）。vLLM v0.32 也已跟进，做了 Engram 的 FP8 跨 DP 打包交换和 CPU 卸载查表加速。

---

## 4. 注意力与 KV 压缩：从 MLA 到 V4.1 的"四重压缩"

### 4.1 演进脉络

| 代际 | 技术 | 效果 |
|---|---|---|
| DeepSeek-V2/V3 | **MLA**（arXiv 2405.04434）：K/V 压缩到 512 维的低秩隐向量，另加 64 维 RoPE | KV 比 MHA 小 93.3%，最大生成吞吐提升 5.76 倍 |
| 2025 | **NSA**（arXiv 2502.11089）：可训练的分层稀疏（粗粒度压缩 + 细粒度选择） | 64k 以上长度时质量持平或更好 |
| DeepSeek-V3.2 | **DSA**：用 lightning indexer 选出 top-k token 做稀疏 MLA | 长上下文计算量从 O(L²) 降到 O(L·k) |
| 微软 | **YOCO**（arXiv 2405.05254）：先由 self-decoder 生成一份全局 KV，cross-decoder 各层复用 | 长上下文显存降低数个数量级 |
| 小米 MiMo 等 | **混合 SWA**（arXiv 2607.13095）：滑窗层与全注意力层交错，KV 只需 O(W) | — |
| **DeepSeek-V4.1-Flash** | 把以上全部组合（见 4.2） | 约 0.9 KB/token（890 B） |

### 4.2 V4.1 的注意力结构（源码解析）

**层的组成（共 40 层，另有 3 层 MTP）：**
- 第 0–1 层：纯 SWA，窗口 128；
- 第 2–19 层：SWA + **C2 压缩**，即每 2 个 token 门控 softmax 合成 1 个；
- 第 20–39 层：SWA + **C1**（不压缩）。

**YOCO 划分：** 第 0–19 层是 producer（"编码器"），第 20–39 层是 consumer（"解码器"）。consumer 层的 SWA K/V 和压缩 KV 都由**第 20 层的 hidden state 投影**得到，只有 Q 用本层自己的 hidden。

**跨层共享：**
- 主 KV 的源层只有第 2、8、14、20 层，其余层复用最近的前序源层；
- indexer 的源层是第 2、8、14、20、24、28、32、36 层；
- 第 20 层先选出 2,048 个"候选块"（每块 8 个 token），后续各层的 indexer 只在这些候选块里计算；
- 共享不跨越 YOCO 边界。

**稀疏 indexer（类 DSA）：**
- logits = `relu(q·k)·w`，32 个头 × 128 维，Q 和 K 都量化到 FP4；
- 用 DeepGEMM 的 `fp8_fp4_*mqa_logits` 计算 logits，用 DeepSelect 选 top-512。

**核心注意力：**
- 单 KV 头，512 维（448 维隐向量 + 64 维 RoPE），每头一个 FP32 attention sink；
- 由 FlashMLA 稀疏 kernel 完成；decode 时**用同一个 kernel 同时 attend SWA cache 和压缩 cache**（`extra_k_cache`）。

**mHC：** 残差流扩为 4 路，用 Sinkhorn 归一化做混合。这是为了训练稳定性而设计的结构，在推理端带来额外的融合 kernel 需求（TileKernels、DeepGEMM `mega_mhc`）。

### 4.3 KV 体积核算 **[源码 + HF config]**

> **修订（专题深挖后）**：初版把 indexer K 算成 8 个源层（约 1.16 KB/token），这是错的。
> - `hugging_face/config.py:74` 中：`layer_index_k_map = _source_map(text.get("index_k_source_layer_ids", text["kv_source_layer_ids"]))`。
> - V4.1 的 `config.json` 里**没有** `index_k_source_layer_ids` 这个键，所以 indexer K 跟随主 KV 的源层，即 2、8、14、20。
> - indexer 模块（各自的 Q 和 top-k）仍然在 8 个源层上分别计算，但 K cache 只存 4 份。

分页 KV 只为"源层"分配（`cache_engine.py` 中的 `process()`）：

| 组成 | 源层 | 每个条目的字节数 | 每个原始 token 的字节数 |
|---|---|---|---|
| 主 KV，C2 | 2、8、14 | 288 B（FP4 e2m1 + 每 16 个元素一个 e4m3 scale，`V41_FP4_FORMAT`） | 3 × 288 / 2 = 432 |
| 主 KV，C1 | 20（YOCO 第 20–39 层共享） | 288 B | 288 |
| indexer K，C2 | 2、8、14 | 68 B（128/2 + 128/32，`_get_qk_topk_bytes`） | 3 × 68 / 2 = 102 |
| indexer K，C1 | 20 | 68 B | 68 |
| **分页 KV 合计** | | | **890 B/token**（未计每个块 32 字节对齐的填充） |
| SWA 状态（每条序列定长，与上下文长度无关） | 落盘的是第 0–19 层 | 528 B（MXFP8） | 每条序列 20 × 134 × 528 = 1,415,040 B ≈ 1.35 MiB |

**对比**：DeepSeek-V3 的 MLA 每层每 token 576 维、共 61 层，在 BF16 下约 **69 KB/token**。V4.1-Flash 的分页 KV 约为它的 **1/78**。这里的口径是"每个上下文 token"；V4.1 另外还有每条序列约 1.35 MiB 的定长 SWA 状态。

**系统层面的连锁反应：**
1. **跨机房 PD 可行。** Q&A 给出的预算是每 8 卡约 400 Gbps，DeepSeek 实际的推理节点分布在不同但相邻的机房。
2. **KV 能保留 3–7 天。** 100 PiB 存储能装下的上下文 token 数，比 V3 时代多两个数量级。
3. **Decode 不受 KV 容量约束。** Q&A 明确说 decode 不开 KV offload，也不用 Hi-Sparse，只受带宽和算力约束。
4. **SWA 状态变成新的"难存"对象。** 它是每条序列定长的状态，而且只在 128 token 边界上才有意义，因此需要单独的存储、目录和重放机制（§5.3）。

### 4.4 SWA 有界重放：用"近似"换"常数代价"

**问题：** 前缀命中时，分页 KV 可以从 3FS 读回来，但 SWA 状态可能缺失：它只存在产生它的那台机器的内存里，而且只在 128 的倍数位置落盘。按理说，要精确重建第 i 层的 SWA，需要把整个前缀再算一遍。

**做法：** 只重算最近 128 个 token，并把注意力窗口截断在重算段之内。即对位置 i 只看 `[max(s, i−127), i]`，得到一个**近似的** SWA KV。

**两处应用：**
- **Encoder 侧**：SWA 未命中时，回退 1 个块（128 token）重算，分页 KV 原样复用、不覆盖。
- **Decoder 侧**：consumer 的 20 层在 prefill 时只算最后 128 个 token（另加 6 个投机回退位，共 134；代码默认保留 140）。这部分结果只给 decode 使用，不进入前缀缓存。效果是**prefill 总计算量几乎减半**。

**命中策略：**
- 输入长度为 L 时，最大命中长度为 `L − 134`；
- 先查分页 KV，得到 L1（128 的倍数）；
- 再查 `[L1−128, L1]` 区间里是否有 SWA 状态：有，则命中长度为 L1；没有，则为 L1 − 128。

**业界对比：**
- vLLM v0.32 已跟进 decoder 侧有界重放（#58132），以及 P/D 场景下已加载 KV 跳过重放（#59197）。
- MiMo-V2.5 的混合 SWA 方案（arXiv 2607.13095）用 SWA 感知的前缀树加 GCache 处理同一问题。
- 本质区别在于：DeepSeek 接受**数值近似**，换来与上下文长度、层数都无关的**常数重算代价**。这类近似能成立，前提是模型训练时就适配了它（代码里有 `examples/yoco/` 做 mask 与 compact 的等价验证）。**系统侧的近似需要模型侧配合**，这是协同设计的又一例证。

---

## 5. KV 缓存分层与存储

### 5.1 方案谱系

| 方案 | 层级 | 粒度 / 索引 | 特点 |
|---|---|---|---|
| **RadixAttention**（SGLang，arXiv 2312.07104） | GPU | 基数树，按 token 精确前缀匹配 | 在 agent / RAG 负载上吞吐最高提升 6.4 倍；UnifiedRadixTree 已扩展到 SWA、Mamba、DSA |
| **vLLM APC + 混合分配器** | GPU | 块哈希 | 简单高效；支持滑窗、Mamba、chunked local 等混合注意力 |
| **SGLang HiCache / HiSparse** | GPU → CPU → NIXL（SSD / 对象存储） | 基数树分层 | 在 v0.5.22 支持了稀疏注意力加 MTP 的长上下文 CPU 驻留 |
| **LMCache**（arXiv 2510.09665） | GPU → CPU → SSD → 远端（Redis、Mooncake、S3、GDS） | 独立守护进程 | 不和引擎同生共死；支持 CacheBlend 做非前缀复用（arXiv 2405.16444：TTFT 降低 2.2–3.3 倍） |
| **Mooncake Store** | 集群级 CPU / SSD 池 | 全局 | 以 KV 缓存为中心的架构；Transfer Engine 已被几乎所有主流引擎集成 |
| **NVIDIA KVBM**（Dynamo） | GPU → CPU → SSD → S3 / Blob | 块 | 支持 TRT-LLM 和 vLLM 后端 |
| **DeepSeek 3FS + kvcache-store** | 专用存储集群 | 哈希链前缀 | 见 5.2 |

### 5.2 DeepSeek 的 KV 存储设计（来自 Q&A）

| 内容 | 存放位置 | 冗余方式 | 管理者 |
|---|---|---|---|
| 正在使用的分页 KV + 状态 KV | 推理卡显存 | — | P / D 实例 |
| SWA KV | **本机内存**（3FS 上不存） | 单副本；跨机命中时经 400G VLAN 拉取 | P / D 实例 + Entrypoint 内存目录 |
| 分页 KV（主 KV + indexer K） | **专用 3FS 集群** | 3FS 的 10+2 纠删码，应用层不做副本 | kvcache-store |

**关键参数：**
- 3FS 总容量约 100 PiB SSD；TTL 为 agent 流量 3 天、chatbot 流量 7 天。chatbot 长尾命中更重，TTL 缩到 3 天会损失 0.5–1% 的命中率。
- 存储与在线 GPU 的配比：每 GPU 约 14.4 TB（逻辑容量）。
- 单台存储服务器：20 块 30.73 TB SSD；读 ≥ 40 GB/s，写 ≥ 5 GB/s。
- 读一个纠删码组时，同时向 11 个节点发请求，任意 10 个返回即可解码，用来对抗长尾延迟。
- 过期回收：每个存储节点跑一个 GC Worker，按 LRU 把磁盘占用控制在水位线以下；kvcache-store 自身不做过期判断。

**三段流水与开销隐藏：**

| 阶段 | 做法 | 能否被隐藏 |
|---|---|---|
| 匹配 | Entrypoint 查本地 kvcache-store（CHECK 角色），只查元数据，做哈希链前缀匹配 | **不能**：路由之前就必须算出命中长度 |
| 拉取 | 独立进程把 KV 从 3FS 异步读到主机共享内存 | 能 |
| Swap-in | 引擎进程内的线程把 KV 从共享内存搬到显存，逐层与前向计算重叠 | 能。不用 CUDA copy engine（细粒度块拷贝性能差），改用 **GDR**：RDMA 网卡直接把 PCIe 带宽打满，且不占 SM |

**为什么不用"分层存储 + 前缀亲和路由"：**
- 全局共享池让**任何一台 prefill 机器都能命中任何前缀**，从而消除了"前缀亲和"与"负载均衡"之间的冲突。prefill 负载均衡器因此**不做**前缀亲和路由，这和 Dynamo、llm-d 的 KV 感知路由形成鲜明对比。
- 存储与计算解耦，可以各自扩缩容。
- 计算节点的故障率比存储节点高一个数量级以上，KV 放在存储集群上不受计算节点故障影响。

### 5.3 DualPath：存储带宽也能负载均衡

Q&A 中 decode 负载均衡"参见 DualPath 的组内调度算法"。对应论文是 **DualPath**（arXiv 2602.21548，*Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference*）。

- **问题：** 在多轮 agent 负载下，瓶颈从计算转到了 KV 存储 I/O。prefill 机器的存储网卡被打满，decode 机器的存储网卡却闲着。
- **做法：** 在传统的"存储 → prefill"路径之外，新增"**存储 → decode → 经计算网络 RDMA 转给 prefill**"这条路径，再配一个全局调度器，动态平衡 P 和 D 的负载。
- **效果：** 离线吞吐最高提升 1.87 倍，在线吞吐平均提升 1.96 倍，且不违反 SLO。

**解读：** 当 KV 命中率很高（agent 场景）、而计算量因 KV 压缩又被压得很低时，**存储带宽本身变成了需要调度的资源**。这是 KV 缓存从"加速手段"变成"一等公民"之后自然出现的新问题。

### 5.4 业界的经验与取舍

| 维度 | 本地分层（HiCache / LMCache 本地模式） | 全局池（Mooncake Store / 3FS / KVBM 远端） |
|---|---|---|
| 命中延迟 | 低 | 较高，需要 RDMA / GDR 才能隐藏 |
| 命中率 | 依赖亲和路由，与负载均衡冲突 | 任意机器都能命中 |
| 规模经济 | 中小规模就划算 | 需要几十 PB 以上才能体现 TTL 优势 |
| 运维 | 简单 | 需要专门的存储团队，比如运维 3FS |

**建议：** 中小规模部署可以先用 SGLang HiCache 或 vLLM + LMCache；集群达到百卡以上、且 agent 流量占比高时，再考虑 Mooncake Store 或 3FS 这类全局池。全局池的前提是 KV 足够小，像 V4.1 这样的模型最适合。

---

## 6. 调度：从"零开销"到"负载均衡"

### 6.1 引擎内调度

**Chunked prefill（Sarathi-Serve，arXiv 2403.02310）：**
- 做法：长 prompt 切块，与 decode 混合调度，减少 decode 卡顿。Mistral-7B 上服务容量提升 2.6 倍。
- **DeepInfer 不做 prefill / decode 混合批**，每一步要么全是 prefill、要么全是 decode。原因是 DP + EP 要求所有 DP rank 步型一致，MoE 的 all-to-all 才能对齐。因此：
  - 各 DP rank 通过 all-reduce **投票**决定这一步是否做 prefill（"ALL 点"要求全员同意，"ANY 点"一人同意即可）；
  - 没有 prefill 任务的 rank 跑一个空的占位 prefill。
- 在 PD 分离部署下，P 和 D 本来就是不同实例，混合批的需求天然消失。

**零开销异步调度（ZOS）：**
- 参考了 SGLang 的 overlap 调度器和 NanoFlow。
- GPU 在执行 `forward(N) → sample(N)` 时，CPU 并行执行 `schedule(N+1) → launch_forward(N+1) → post-process(N−1) → launch_sample(N+1)`。
- 实现要点（`worker.py`、`zos.cpp`、`scheduler.hpp`）：
  - 用乐观计数器 `total_inflight_tokens` 预先占用 KV 块；
  - 上一步的 token 直接在 GPU 上通过 remap 复用，CPU 只填充新加入的序列；
  - k 长度在 GPU 上按实际接受数纠正：`k += q − (nextn − accepted)`；
  - 被抢占序列的过期输出直接丢弃；
  - 所有 DP、CP 之间的同步都走 gloo CPU group，保证不触发 GPU 同步。代码中还有 `forbid_cuda_sync` 守卫，并配有测试 `test_cuda_sync_guards.py`。
- vLLM V1、SGLang、TRT-LLM 都已实现等价能力。**CPU 调度开销问题在业界已基本解决。**

### 6.2 Prefill 负载均衡：用时间预算代替 token 预算

（这部分来自 Q&A，开源代码中**没有**实现。）

- **分档：** 按总上下文长度分成 <16K、16–64K、64–256K、≥256K 四档。各档分到的引擎数，按上一轮各档在算力配额需求中的占比动态重新分配。
- **DP 组之间：** 引擎按剩余配额**主动拉取**任务；由于存储是全局池，不做前缀亲和。
- **DP 组内：** 每个请求交给当前配额负载最低的 rank。
- **单卡内：** 每个 prefill 步给一个时间预算 `compute_quota_limit`（单位纳秒）。用拟合好的二次代价模型 `cost(q, c)` 预测耗时，其中 q 是新 token 数、c 是该序列已算过的前缀长度。在不超过 32,768 token 上限的前提下，求出恰好用完剩余时间预算的最大 q；时间预算本身再在 DP 组内做一次 all-reduce MIN。

**解读：** 稀疏注意力加上前缀命中之后，同样的 token 数在不同前缀长度下的代价相差很大，固定 token 预算会造成各 DP rank 步长不一致，于是 MoE all-to-all 要等最慢的 rank。用"时间预算 + 代价模型"把各 rank 的步长对齐，本质上是**为 DP + EP 的同步语义服务的**。开源的 `FlopsEstimator` 已经对注意力、MLP、路由、HC、Engram 分别建模，可以作为复现起点。

### 6.3 Decode 负载均衡

- **目标：** 同时平衡两个维度：请求数（决定 batch 大小，影响 MoE 和其余部分）和 KV 总长度（决定注意力代价）。
- **启发式：** 短请求发给"并发低但 KV 总长度大"的卡，长请求发给"KV 总长度小"的卡。
- **效果：** 整体约为理想均衡离线基准的 **80%**。这 20% 的差距是目前最大的单项损失，也是业界下一个优化重点。

**业界对比：**
- **Dynamo KV-aware router**：按"负载 + KV 重合度"给 worker 打分，在 Qwen3-Coder 480B 上 TTFT 快 2 倍（Baseten 数据）。
- **llm-d GIE**：K8s Gateway API 扩展，按前缀缓存、负载和延迟预测做路由。
- **sgl-router**：Rust 实现，基于引擎上报的 KV 事件流在本地构建基数树做路由，支持长度分档。
- **排队论方向**：SARA（arXiv 2609.26763）把 prefill 建模为 M/G/k、decode 建模为广义生灭过程，goodput 提升 26.6%。

### 6.4 自动扩缩与容错（Q&A）

- **扩缩依据：** prefill 按非缓存 token 量，decode 按运行中的请求数；调整有限速，并设上下限。
- **冗余：** 配额利用率高于 0.8 时禁止缩容；资源池里保留空闲的备用机器。
- **优雅下线：** decode 引擎收到下线通知后有 1 分钟宽限期，未完成的请求连同其 KV 一次性转发给其他 decode 引擎继续解码，以缓解宽 EP 下长尾请求造成的浪费。
- **业界对比：** Dynamo Planner 按 SLA 自动扩缩（阿里云报告 SLA 违约减少 80%，TCO 降低 5%）；llm-d 有 workload-variant-autoscaler 并集成 KEDA。

---

## 7. 投机解码：从"猜得准"到"验证得值"

### 7.1 方案谱系

| 方案 | 草稿来源 | 关键数字 |
|---|---|---|
| **MTP**（DeepSeek-V3 起） | 模型自带的 MTP 层 | MiMo-V2-Flash 用 3 层 MTP 时接受长度 3.6，解码加速 2.6 倍（arXiv 2601.02780） |
| **EAGLE-3**（arXiv 2503.01840） | 融合多层特征的独立草稿头，直接预测 token | 最高 6.5 倍；在 SGLang、batch 64 下吞吐提升 1.38 倍 |
| **SpecForge**（arXiv 2603.18567） | EAGLE-3 训练框架 | 训练提速 9.9 倍；端到端加速最高 4.48 倍 |
| **动态 / 自适应验证** | vLLM 的 adaptive block verification，PIPO 置信度头（arXiv 2605.27255） | — |
| **DSpark**（DeepSeek） | 块式半自回归起草 + 置信度头 + 变长验证 | 见 7.2 |

### 7.2 DSpark 解析（源码 + Q&A）

**草稿模型（DFlash）：**
- 取主模型第 37、38、39 层的注意力输入特征，拼接后投影；共 3 层，只用 SWA，MoE 为 128 个专家、top-3。
- **一次前向并行起草整个块**：输入是 `[golden token, noise ×4]`，块长 5。
- 输出头 `MarkovGPTHead` 在 logits 上叠加一个秩 256 的一阶马尔可夫项。

**置信度头：** `AcceptRatePredictor` 是一个线性层，输入为 hidden 拼接马尔可夫 embedding（5120 + 256 维），输出 1 维，每步有各自的温度。它预测每个草稿 token 被接受的概率。

**变长验证：**
1. 用实测得到的硬件吞吐曲线（"时间 vs. CUDA graph 大小"）作为代价。
2. 把所有请求的草稿位置按累计接受概率排序，贪心加入，选出使"期望产出 token 数 ÷ 预计耗时"最大的总验证长度。
3. 与异步调度的配合：
   - CPU 只决定**总预算**，因为它决定用哪个 CUDA graph；
   - 每个请求分到多少长度，由 GPU 上的 TileLang kernel 根据上一步的置信度**重新分配**（累乘、全局排序取 top-k、融合 remap scatter）。
   - 由于异步调度落后一步，CPU 用的是 N−2 步的接受率，这就是所谓的"两步回传"。

**效果（EP64/DP64）：**

| 每 rank 并发 | 固定验证 5 个 → 动态验证的平均长度 | 平均接受长度（固定 → 动态） | 吞吐变化 |
|---:|---|---|---:|
| 16 | 5 → 4.44 | 3.32 → 3.24 | −1.5% |
| 60（生产工作点） | 5 → 3.58 | 3.30 → 3.21 | **+5.4%** |
| 96 | 5 → 3.24 | 3.30 → 3.15 | **+11.0%** |

**解读：**
- **并发越高，变长验证越有价值。** 在高并发下 GPU 接近算力饱和，每多验证一个 token 都有实际代价；在低并发（访存受限）时，多验证几乎是免费的，所以动态策略反而略亏。
- 这体现了范式的转变：投机解码的优化目标从"提高接受率"转向"**在算力预算下最大化有效吞吐**"，与第 6 节的调度器形成了闭环。
- 开源生态的跟进：SGLang 自 2026-07 起实现了 DSpark（LMSYS 博客 *Adopting DSpark in SGLang*，明确致谢 DeepSeek 的算法）；vLLM v0.32 有 DSpark 的 EPLB 和图内元数据；TRT-LLM 的 DSpark 还在开发中。

---

## 8. 低精度：FP4 从权重走到 KV

### 8.1 格式

| 格式 | 分组 | scale | 硬件 |
|---|---|---|---|
| FP8 e4m3 | 逐块（如 128 或 32） | FP32（SM90）或 UE8M0（SM100，4 个打包成一个 int32） | Hopper 及以后 |
| **MXFP4** | 32 | UE8M0（2 的幂次） | Blackwell 原生 |
| **NVFP4** | 16 | e4m3，另有一个全局 FP32 scale | Blackwell 原生 |

### 8.2 V4.1 中的用法（源码）

| 对象 | 格式 |
|---|---|
| 路由专家权重 | MXFP4（e2m1 + 每 32 个元素一个 UE8M0 scale）；GEMM 为 `m_grouped_fp8_fp4_gemm`（FP8 激活 × FP4 权重） |
| 非专家线性层 | FP8，权重块 32×32，UE8M0 scale |
| **主 KV** | **FP4 e2m1，每 16 个元素一个 e4m3 scale**，每条 288 B。相当于去掉全局 scale 的 NVFP4 |
| indexer Q/K | MXFP4（每 32 个元素一个 UE8M0） |
| SWA KV | MXFP8（512 维 FP8 + 16 B 的 UE8M0 scale = 528 B） |
| Engram 表 | FP8 |

### 8.3 风险提示

- arXiv 2609.04105（硬件感知的 FP4 FlashAttention-4）发现：在训练中把注意力概率和 V 用 MXFP4 表示**会发散**。推理端的 FP4 KV 能用，依赖于模型在训练时已经适配（V4.1 的 checkpoint 本身就要求 `expert_dtype=fp4`）。
- 结论：**FP4 KV 不是可以随意打开的推理开关，而是需要模型配合的协同设计。** 对第三方模型直接套用 FP4 KV，需要先做精度评估。

---

## 9. 确定性与批次不变

**问题根源：** 浮点加法不满足结合律。batch 的组成变化会改变 kernel 的切分方式（split-K、注意力的并行规约顺序、MoE 的规约顺序），于是同一个 prompt 在不同批次里得到不同输出。这会破坏 RL 中 on-policy 的一致性、评测的可复现性，以及回归测试。

| 方案 | 做法 | 局限 |
|---|---|---|
| **Thinking Machines**，*Defeating Nondeterminism in LLM Inference*（2025-09） | 系统梳理了不确定性的来源，提出批次不变的 RMSNorm、matmul、attention kernel | 有一定性能代价 |
| **vLLM**（`VLLM_BATCH_INVARIANT=1`，beta） | 批次不变 kernel，支持 NVIDIA、AMD、Intel | Vosti 指出某些调度变化下仍会发散 |
| **SGLang 确定性模式** | 对齐 chunk 长度，固定 CUDA graph 分派 | DP+PP+MTP 下仍有问题（#45481） |
| **Vosti**（arXiv 2609.38981） | 与状态无关的 kernel 选择；用 Verus 形式化验证调度器和 KV 缓存 | 学术原型 |
| **DeepInfer** | `DEEPINFER_BATCH_INVARIANT=1`：DeepGEMM、TileKernels 走确定性算法；注意力后端开启批次不变；开 EPLB 时关闭 DeepEP 的多次规约 | 仅支持 NVIDIA |

**DeepInfer 的确定性采样：**
- 每条序列有一个 uint64 种子 `seq_seed`；RNG 按 `(seq_seed, sample_position)` 初始化，再做定点 CDF 和拒绝采样。
- 因此采样结果与 batch 大小和行序都无关，有 `test_random_sampling_batch_invariant` 测试覆盖。
- 草稿模型和主模型的采样使用不同的 RNG 流，种子偏移量为 1000000007。

**限制：** 完整的确定性要求同时设置 `BATCH_INVARIANT=1` 和 `ENABLE_MTP_VARLEN=0`。变长验证让每步的计算图依赖于上一步的置信度，和逐位确定性存在冲突。

---

## 10. 硬件：scale-up 域决定 EP 上限

| 平台 | scale-up 域 | DeepSeek 相关的公开数据 |
|---|---|---|
| NVIDIA H800 / H100 | 8 卡 NVLink | V3/R1：每节点输入 73.7k / 输出 14.8k tok/s |
| NVIDIA GB200 / GB300 NVL72 | 72 卡 NVLink，单卡 1.8 TB/s | TRT-LLM 在 GB300 上跑 V4-Pro 峰值 1,618 tok/s/GPU；SGLang 在 GB300 上跑 V4 同交互性吞吐提升 5 倍；Dynamo 宣称 R1 提升 7 倍 **[未核实]** |
| NVIDIA SM100（DeepInfer 实测） | 64 卡位于同一 scale-up 域 | V4.1-Flash decode 每卡 15.7k–26.3k tok/s（按 75% 接受率折算） |
| 华为 CloudMatrix384（Ascend 910C） | 384 NPU，UB 全互联 | CloudMatrix-Infer（arXiv 2506.12708）跑 R1：prefill 每 NPU 6,688 tok/s；decode 每 NPU 1,943 tok/s（TPOT < 50 ms）；TPOT 15 ms 时每 NPU 538 tok/s；EP320，INT8 |
| 华为 Ascend 950 | 超节点 | DeepInfer 官方支持；没有公开性能数字 |

**跨厂商的工程方法（DeepInfer）：**
1. **TileLang 双后端。** `tilelang_ops/ops.py` 按平台从 `cuda/` 或 `asc/` 子包导出同名 kernel，缺少的 kernel 抛出 `KernelUnavailableError`。
2. **同名替身库。** DeepGEMM-Ascend、DeepEP-Ascend 安装后的包名分别仍是 `deep_gemm`、`deep_ep`，上层代码基本不用改。
3. **import hook。** `npu_ops/patch_imports.py` 在导入时替换个别 NVIDIA 专用接口，例如 DeepSelect 的 topk 和 flash-attn-4。
4. **`transfer_to_npu`。** 把 `torch.cuda` 映射到 `torch.npu`。

**解读：** 这是目前开源界规模最大的"**同一份模型代码在 NVIDIA 和国产 NPU 上都达到生产级**"的实践。它的方法论是：在 kernel 层用 DSL（TileLang）做抽象，在库层保持接口兼容。这对国产芯片生态的意义，大于 DeepInfer 本身对 vLLM / SGLang 的竞争意义。

---

## 11. 趋势判断

1. **模型-系统协同设计成为主战场。** YOCO 有界重放、FP4 KV、跨层共享、置信度头，这些都是"训练时就为推理系统设计好"的结构。通用引擎只能事后适配，而模型厂商自研引擎的优势会持续存在。vLLM 和 SGLang 的应对办法是**与模型厂商提前协作**：这次两者都在 DeepInfer 发布前完成了 V4.1 适配。
2. **KV 从"缓存"变成"存储"。** KV 压到约 0.9 KB/token 后，它的经济性接近普通对象存储，于是 TTL 能做到"天"级，跨机房 PD 也变得可行。下一阶段的瓶颈是**存储带宽调度**（DualPath）和**命中计算无法隐藏**的问题。
3. **负载均衡是最大的剩余损失。** 引擎内 CPU 开销已经压到零，但跨 DP / EP 的不均衡仍造成约 20% 的损失（DeepSeek 报告整体约为理想值的 80%）。时间预算加代价模型的调度、EPLB 的目标函数（token 数还是激活专家数）、DualPath 式的存储均衡，都会是 2027 年的热点。
4. **投机解码与调度融合。** 验证长度成为调度器的决策变量。下一步可能会把 SLO（每请求的 TPOT 目标）也纳入进来，做到按请求区分投机强度；SGLang 的 DSpark 博客已经在探索这一点。
5. **确定性成为基础能力。** RL 训推一致和评测复现都在推动它，但与变长投机、动态调度之间存在根本张力，需要接受"可配置的确定性等级"。
6. **scale-up 域越大，系统越简单。** NVL72、CloudMatrix384 和 Ascend 超节点让宽 EP 加融合 kernel 能替代复杂的跨节点重叠和 AFD 方案。硬件演进会持续重塑软件架构的取舍。

---

## 12. 给推理团队的落地建议

| 你的情况 | 建议 |
|---|---|
| 要上线 V4.1-Flash | 用 SGLang 或 vLLM；照抄 DeepInfer Q&A 的参数（EPLB 冗余数、窗口、周期；prefill 长度分档；decode EP64+；YOCO 保留 140 token；nextn=5 + 变长验证）作为起点 |
| 想复现 DeepSeek 级别的 KV 存储 | 3FS（已开源）+ Mooncake Transfer Engine / NIXL（含 HF3FS 后端）+ 自研哈希链元数据服务；参考 Q&A 的容量配比（每 GPU 14.4 TB）和 TTL |
| 做性能对标 | 用 DeepInfer 的 `examples/sweep/run_sweep.sh` 在 SM100 上跑出官方 Pareto 曲线，再用同样的配置跑 vLLM / SGLang，量化差距 |
| 做调度研究 | 精读 `csrc/engine/scheduler.hpp`（DP 投票、抢占）、`mtp_varlen.hpp`（变长预算）、`zos.cpp`（异步 remap），再对照 SGLang 的 overlap 调度器 |
| 做国产卡适配 | 参考 DeepInfer 的"TileLang 双后端 + 同名替身库 + import hook"三件套 |
| 做 RL 训推一致 | 参考 DeepInfer 按 `(seed, position)` 播种的采样器和批次不变开关；评估 Vosti 的形式化思路 |

---

## 参考文献与来源

**DeepSeek：**
- DeepInfer：https://github.com/deepseek-ai/DeepInfer （`doc/inference_deployment_qa.md`）
- Open Infra Index：https://github.com/deepseek-ai/open-infra-index
- 3FS / DeepEP / DeepGEMM / FlashMLA / EPLB：https://github.com/deepseek-ai
- DualPath：https://arxiv.org/abs/2602.21548
- MLA（V2）：https://arxiv.org/abs/2405.04434
- NSA：https://arxiv.org/abs/2502.11089

**系统形态：**
- Splitwise：https://arxiv.org/abs/2311.18677
- DistServe：https://arxiv.org/abs/2401.09670
- Mooncake：https://arxiv.org/abs/2407.00079
- MegaScale-Infer：https://arxiv.org/abs/2504.02263
- semi-PD：https://arxiv.org/abs/2504.19867
- Step-3：https://arxiv.org/abs/2507.19427
- AFD 设计空间：https://arxiv.org/abs/2605.28302

**MoE：**
- METRO：https://arxiv.org/abs/2512.09277
- GPU 发起通信分析：https://arxiv.org/abs/2610.01380
- NanoFlow：https://arxiv.org/abs/2408.12757

**注意力与 KV：**
- YOCO：https://arxiv.org/abs/2405.05254
- MiMo-V2.5 混合 SWA：https://arxiv.org/abs/2607.13095
- FlashAttention-3：https://arxiv.org/abs/2407.08608
- FlashInfer：https://arxiv.org/abs/2501.01005
- FP4 FlashAttention-4：https://arxiv.org/abs/2609.04105

**缓存：**
- SGLang / RadixAttention：https://arxiv.org/abs/2312.07104
- CacheBlend：https://arxiv.org/abs/2405.16444
- LMCache：https://arxiv.org/abs/2510.09665
- NIXL：https://github.com/ai-dynamo/nixl

**调度：**
- Sarathi-Serve：https://arxiv.org/abs/2403.02310
- SARA：https://arxiv.org/abs/2609.26763
- Dynamo：https://github.com/ai-dynamo/dynamo
- llm-d：https://github.com/llm-d/llm-d

**投机解码：**
- EAGLE-3：https://arxiv.org/abs/2503.01840
- SpecForge：https://arxiv.org/abs/2603.18567
- MiMo-V2-Flash：https://arxiv.org/abs/2601.02780
- PIPO：https://arxiv.org/abs/2605.27255
- LMSYS DSpark：https://www.lmsys.org/blog/2026-07-06-dspark-sglang

**确定性：**
- Thinking Machines：https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- Vosti：https://arxiv.org/abs/2609.38981

**硬件：**
- CloudMatrix384：https://arxiv.org/abs/2506.12708
- TRT-LLM V4 技术博客：https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog
