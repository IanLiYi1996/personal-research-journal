# DeepInfer 调研 01：竞品产品对比

> 数据截至 **2026-10-21**。GitHub 数据通过 `gh api` 实时拉取；功能信息来自各项目 README、release notes 和官方博客。
> DeepInfer 部分来自源码阅读（commit `774ce24`）。标 **[未核实]** 的条目没有找到一手来源。
> 本文没有在同一套硬件上实测任何引擎，各家公布的性能数字口径不同，**不可直接横向比较**（见 §5）。

---

## 0. 结论先行

1. **DeepInfer 不是通用推理引擎的竞争者，而是一份"官方参考实现"。** 它只支持 1 个模型（DeepSeek-V4.1-Flash）和 2 种硬件（NVIDIA SM100、Ascend 950），且只开源了离线引擎。线上服务必需的 KV 存储、PD 传输、负载均衡和 HTTP 服务都不在开源范围内。
2. **V4.1 的"首发支持"已被 vLLM 和 SGLang 抢先或同步完成。**
   - vLLM v0.32.0 于 10-18 发布，SGLang v0.5.22 于 10-19 发布，release notes 里都列出了 V4.1 专项优化（bounded replay、DSpark、PD）。
   - 两者都早于 DeepInfer 开源（10-20），说明 DeepSeek 采取了"先把能力给上游、再开源自研引擎"的策略。这与 2026-04 发布的 *Path to Open-Sourcing the DeepSeek Inference Engine* 的思路一致。
3. **DeepInfer 的真正价值是"生产级设计的可读参考"。** 具体包括：
   - 部署 Q&A 中披露的一组生产参数，例如 EPLB 参数、负载均衡分档、3FS 容量和 TTL；
   - 零开销调度与 DSpark 变长验证的 C++ 实现；
   - YOCO/SWA 有界重放、FP4 KV 等模型侧设计。
   想做 V4.1 极致调优的团队可以对照它，校准自己的 vLLM/SGLang 部署。
4. **选型建议（一句话）：**
   - 要上线服务：选 **SGLang** 或 **vLLM**，规模化部署再配 **llm-d / Dynamo**，KV 层配 **Mooncake / LMCache**。
   - NVIDIA 全栈且追求极限 Blackwell 性能：选 **TensorRT-LLM + Dynamo**。
   - 国产卡多芯片：选 **xLLM / vllm-ascend / SGLang-NPU**。
   - 单机或消费级：选 **KTransformers / llama.cpp**。
   - DeepInfer 适合用于研究、基准对标、批量离线推理和评测。

---

## 1. 竞品版图

| 层次 | 代表项目 | 和 DeepInfer 的关系 |
|---|---|---|
| **通用推理引擎**（引擎 + 服务） | vLLM、SGLang、TensorRT-LLM、LMDeploy | 直接竞品；都已支持 DeepSeek-V4 系列 |
| **分布式编排层** | llm-d（CNCF Sandbox）、NVIDIA Dynamo、sgl-router | 补齐 DeepInfer 未开源的负载均衡、路由、PD 和自动扩缩 |
| **KV / 传输基础设施** | Mooncake、LMCache、NIXL、3FS | 补齐 DeepInfer 未开源的 KV 存储和 PD 传输 |
| **国产芯片引擎** | xLLM（京东→开放原子）、MindIE-LLM（华为）、vllm-ascend | 与 DeepInfer 的 Ascend 路线竞争 |
| **边缘 / 异构** | KTransformers、llama.cpp、Ollama | 不同市场，基本不重叠 |
| **DeepSeek 自家开源组件** | FlashMLA、DeepGEMM、DeepEP、3FS、EPLB、TileKernels、DeepSelect、deepseek-recipe | DeepInfer 的底座，同时也被所有竞品复用 |

---

## 2. 基本面对比（GitHub 实时数据，2026-10-21）

| 项目 | Stars | 协议 | 最新版本 / 日期 | 主导方 | 定位 |
|---|---:|---|---|---|---|
| **DeepInfer** | 14,009 | Apache-2.0 | 无 release（10-20 初始提交） | DeepSeek | V4.1-Flash 离线引擎 |
| vLLM | 94,495 | Apache-2.0 | v0.32.0 / 10-18 | 社区（伯克利起源，多厂商贡献） | 通用引擎事实标准 |
| SGLang | 37,530 | Apache-2.0 | v0.5.22 / 10-19 | LMSYS 社区 + RadixArk 等 | 通用引擎，强 DeepSeek 与 RL 生态 |
| TensorRT-LLM | 14,890 | Apache-2.0*（GitHub 标为 NOASSERTION） | 稳定版 v1.2.1 / 04-20；RC 版 v1.4.0rc0 / 10-14 | NVIDIA | NVIDIA 专属高性能库 |
| NVIDIA Dynamo | 8,345 | Apache-2.0*（同上） | v1.5.1 / 10-07 | NVIDIA | 分布式编排（后端可选 vLLM/SGLang/TRT-LLM） |
| llm-d | 4,892 | Apache-2.0 | **v1.0.0 / 10-20**（与 DeepInfer 同日） | Red Hat、Google、IBM、CoreWeave、NVIDIA；CNCF Sandbox | K8s 原生分布式推理 |
| LMDeploy | 8,111 | Apache-2.0 | v0.18.0 / 09-28 | 上海 AI Lab（InternLM） | TurboMind + PyTorch 双引擎 |
| Mooncake | 6,850 | Apache-2.0 | v0.3.13.post1 / 08-31 | 月之暗面（Kimi）+ 清华 | KV 传输与存储基础设施 |
| LMCache | 12,133 | Apache-2.0 | v0.5.6 / 10-14 | TensorMesh / UChicago | KV 缓存层 |
| xLLM | 1,603 | Apache-2.0 | v0.11.0 / 09-16 | 京东 → 开放原子基金会 | 国产多芯片引擎 |
| vllm-ascend | 2,997 | Apache-2.0 | v0.23.0.post1 / 09-22 | 华为主导，vLLM 官方组织 | vLLM 的 Ascend 插件 |
| KTransformers | 19,618 | Apache-2.0 | v0.7.2 / 10-15 | 趋境 + 清华 | CPU/GPU 异构，本地部署 |
| llama.cpp | 131,866 | MIT | v0.7.0 / 10-15 | ggml 社区 | 边缘 / 消费级 |

**观察：**
- DeepInfer 上线约 1 天就达到 1.4 万 star，与 TensorRT-LLM 相当，说明关注度极高。但仓库关闭了 Issues，144 个 PR 都还没有合并，社区协作模式尚未建立。
- vLLM 的单个版本就有 363 位贡献者（121 位是首次贡献），SGLang 单版本有 245 位贡献者、950 个 PR，社区规模比 DeepInfer 高出 1–2 个数量级。

---

## 3. 功能矩阵

图例：✅ 已开源可用｜🟡 部分 / 实验 / 依赖外部组件｜❌ 不支持或未开源｜❓ 未核实

### 3.1 DeepSeek-V4.1 专项能力

| 能力 | DeepInfer | vLLM v0.32 | SGLang v0.5.22 | TRT-LLM | LMDeploy | xLLM |
|---|---|---|---|---|---|---|
| V4.1-Flash 支持 | ✅ 唯一模型 | ✅（10-18） | ✅（10-19） | 🟡 已支持 V4-Flash/Pro，V4.1 ❓ | 🟡 V4 仅 TurboMind 路径 | 🟡 V4 首日支持，V4.1 ❓ |
| SWA 有界重放（YOCO decoder replay） | ✅ | ✅ #58132 | ❓ | ❓ | ❓ | ❓ |
| DSpark（置信度变长验证） | ✅ C++ `MTPVarlen`，GPU 侧重分配 | ✅ 含 DSpark EPLB | ✅ 07 月起支持，有 LMSYS 博客 | 🟡 DSpark 开发中 | ❓ | ❓ |
| Engram（n-gram 嵌入） | ✅ | ✅ FP8 跨 DP 交换 | ❓ | ❓ | ❓ | ❓ |
| Mega MoE（DeepGEMM 融合 dispatch-GEMM-combine） | ✅ | ✅ FlashInfer/AITER 版 | ✅ | ✅ DeepGEMM MegaMoE | ❓ | ❓ |
| FP4 稀疏 KV + FP4 indexer | ✅ | ✅ | ✅ | ✅ | ❓ | ❓ |

### 3.2 引擎通用能力

| 能力 | DeepInfer | vLLM | SGLang | TRT-LLM |
|---|---|---|---|---|
| 在线 OpenAI 兼容服务 | ❌ 仅 `LLM.generate()` | ✅ | ✅ | ✅（trtllm-serve / Triton） |
| 前缀缓存 | ❌ 只有 `fork_seq` | ✅ APC + 混合分配器 | ✅ RadixAttention / UnifiedRadixTree | ✅ |
| 分层 KV（CPU/SSD/远端） | ❌ 生产版用 3FS，未开源 | ✅ 9 种 KV Connector | ✅ HiCache（GPU/CPU/NIXL） | ✅ 经 Dynamo KVBM |
| PD / EPD 分离 | ❌ 只留了接口（空虚函数） | ✅ NIXL / Mooncake / LMCache | ✅ Mooncake / NIXL，EPD | ✅ CTX/GEN 池 + Mooncake |
| Attention DP + 宽 EP | ✅ EP64–128 生产验证 | ✅ DeepEP 三后端 + Elastic EP | ✅ DeepEP + DP attention | ✅ DEP\<N\> |
| EPLB | ✅ 冗余专家 + 两阶段贪心 + symm_mem 搬运 | ✅ 含迁移批处理、弹性 EP | ✅ EPLB + LPLB（线性规划）+ Waterfill | ✅ |
| 计算与通信重叠 | 🟡 主要依赖 Mega MoE 融合 | ✅ DBO | ✅ TBO | ✅ |
| 上下文并行 CP | ✅ 镜像均衡切分（生产上不用） | ✅ DCP | ✅ prefill CP | ✅ |
| 零开销异步调度 | ✅ ZOS | ✅ V1 异步调度 | ✅ overlap 调度器（DeepInfer 文档中也注明参考了 SGLang） | ✅ |
| 批次不变 / 确定性 | ✅ 仅 NVIDIA | ✅ beta，`VLLM_BATCH_INVARIANT` | ✅（DP+PP+MTP 下仍有已知问题） | ❓ |
| 结构化输出 | ✅ xgrammar，支持 MTP 回滚 | ✅ xgrammar / llguidance | ✅ xgrammar | ✅ |
| 多模态（视觉） | ✅ 内置 ViT | ✅ | ✅ | ✅ |
| RL 集成（权重热更新、休眠模式） | ❌ | ✅ Sleep L1/L2、权重传输 | ✅ verl / slime / Miles / AReaL 主力 rollout 引擎 | 🟡 |
| 模型覆盖 | 1 个 | 数百个 | 50+ 个家族 | 主流模型 |

### 3.3 硬件覆盖

| 硬件 | DeepInfer | vLLM | SGLang | TRT-LLM | xLLM | KTransformers |
|---|---|---|---|---|---|---|
| NVIDIA Blackwell 数据中心卡（SM100/SM103） | ✅ 仅 SM100 | ✅ | ✅ | ✅ 主战场 | ❌ | 🟡 |
| NVIDIA Hopper（SM90） | ❌（社区 PR #148 待合） | ✅ | ✅ | ✅ | ❌ | ✅ |
| 更早的 NVIDIA 卡 / 消费卡 | ❌ | ✅ | ✅ | 🟡 | ❌ | ✅ |
| AMD MI300/MI350 | ❌ | ✅ AITER | ✅ | ❌ | ❌ | ✅ |
| Ascend A2/A3 | ❌ | ✅ vllm-ascend | ✅ | ❌ | ✅ | ✅ |
| **Ascend 950** | ✅ 官方调优 | ❓ vllm-ascend 未明确标注 | ✅ README 列出 950PR/DT | ❌ | ❓ | ❓ |
| TPU / Intel / 其他国产卡 | ❌ | ✅ TPU、XPU | ✅ TPU、XPU、摩尔线程等 | ❌ | ✅ 寒武纪、海光、沐曦、天数、摩尔线程 | 🟡 |

> **更正说明**：部分二手资料称"只有 DeepInfer 支持 Ascend 950"，这个说法**不准确**。SGLang 的 README 已明确列出 Ascend 950PR/DT，并且 v0.5.14 起支持 DeepSeek-V4 NPU。DeepInfer 的差异在于"DeepSeek 官方在 950 上调优过、而且有生产验证"。

---

## 4. 逐个竞品画像

### 4.1 vLLM（+ llm-d / vllm-ascend）

**定位**：推理引擎的事实标准，模型和硬件覆盖面最广。DeepInfer 的 Python 骨架本身就派生自 vLLM（21 个文件带有 "Derived from vLLM" 标注）。

**V4.1 支持**：v0.32.0 专设了 "DeepSeek-V4 and V4.1" 章节，内容包括：
- decoder 侧 SWA 有界重放，重放层可进入 CUDA graph（#58132）；
- 末层 KV 源裁剪，带来 +3.6%；
- Engram 的 FP8 跨 DP 交换；
- V4 与 MTP 的融合起草；
- V4.1 encoder 的 CUDA graph 打包，平均 TTFT 降低 11%；
- DSpark 草稿模型的 EPLB；
- P/D 场景下已加载的 KV 跳过重放（#59197）。

AMD 侧有 AITER MegaMoEV2 和 MI355X 上的 V4.1-Flash mono decode。

**分布式部署**：
- llm-d v1.0.0（CNCF Sandbox）提供 K8s Gateway API 推理扩展、KV 感知路由、延迟预测、P/D sidecar、自动扩缩和 Batch API。
- 宽 EP 公布结果：256 张 B200（16×16 P/D）上 decode 约 3.1k tok/s/GPU，集群约 50k tok/s。

**优势**：
- 栈完全开源；
- 9 种 KV Connector；
- Elastic EP；
- 支持 RL 的 Sleep 模式；
- 约每 2 周发一个版本。

**劣势**：
- 通用性带来抽象层开销；
- V4.1 的极限性能能否追平 DeepSeek 自己的实现，目前没有公开的同口径数据 [未核实]；
- 宽 EP 部署需要 NVSHMEM、IBGDA 等底层配置，运维门槛高。

### 4.2 SGLang（+ sgl-router / Mooncake / SpecForge / Miles）

**定位**：DeepSeek 大规模部署在开源界的"标杆实现者"。LMSYS 在 2025 年发布过 96 张 H100 上的 PD 分离 + 大规模 EP 复现，之后又有 GB200/GB300 系列博客。SGLang 也是 RL 训练框架 rollout 引擎的首选。

**V4 系列支持**：
- v0.5.14（2026-06）首日支持 V4，PyTorch 博客称在 GB300 上同交互性下吞吐提升 5 倍；同一版本加入了 Ascend NPU 支持。
- v0.5.16（2026-07）支持 DSpark。LMSYS 博客 *Adopting DSpark in SGLang* 明确说明算法来自 DeepSeek 的论文。在 8 张 B300、TP8、bs=1 下，V4-Pro 达到 383.7 tok/s。
- v0.5.22（10-19）：V4.1 在 H200 上每卡吞吐最高提升 2.3 倍；在 MI350X/MI355X 上借助 DSpark 提升 1.9 倍；decode 支持 DP attention；PD 传输加入 NIXL 后端。

**特色能力**：
- RadixAttention / UnifiedRadixTree 已能覆盖 SWA、Mamba、DSA 这类注意力；
- HiCache 分层 KV，以及面向稀疏注意力的 HiSparse；
- sgl-router 用 Rust 实现，支持 KV 感知、P/D 分发和长度分档；
- 负载均衡方面有 LPLB（线性规划）和 Waterfill。

**优势**：
- 对 DeepSeek 新特性跟进最快；
- PD 分离和宽 EP 经过最多公开复现；
- 硬件覆盖最广：Ascend 950 在列，并有大量国产卡适配中。

**劣势**：
- 确定性模式在 DP+PP+MTP 组合下仍有未解决的问题（#45481）；
- 迭代极快，导致版本稳定性有波动。

> **更正说明**：有二手说法称"DSpark 是 SGLang 首创、DeepInfer 是后来跟进"，这是**错误**的。DSpark 是 DeepSeek 的算法，SGLang 是较早的开源**实现方**；DeepInfer 则是 DeepSeek 内部生产实现的开源版。

### 4.3 TensorRT-LLM + NVIDIA Dynamo

**TRT-LLM** 是 PyTorch 原生的高性能库。Tech Blog 26 详细介绍了它在 Blackwell 上对 V4 的支持，覆盖：
- SWA / CSA / HCA 三种注意力；
- 在线 Compressor 与分页持久状态；
- mHC；
- hash 路由与学习门控两种 MoE；
- MTP-1 到 MTP-3；
- EPLB；
- DeepGEMM MegaMoE。

DSpark 仍在开发中。在 GB300 上做 V4-Pro 分离式部署时，峰值吞吐从 984 提升到 1,618 tok/s/GPU，时间是 2026-04 到 05 月。

**Dynamo** 是编排层，组件包括：
- KV 感知路由；
- NIXL；
- KVBM（GPU→CPU→SSD→远端）；
- Planner，按 SLA 自动扩缩（阿里云实测 SLA 违约减少 80%）；
- Grove，NVL72 拓扑感知的组调度。

**和 DeepInfer 的关系**：
- 在 NVIDIA 上，这是最完整的"引擎 + 编排"闭环。
- Dynamo 正好补齐 DeepInfer 缺的层，二者在理论上可以组合使用。
- 劣势是只支持 NVIDIA；稳定版和 RC 版之间相差约半年。

### 4.4 LMDeploy

- TurboMind（C++）+ PyTorch 双引擎。2025 年较早集成了 FlashMLA、DeepGEMM、DeepEP、EPLB 和 micro-batch 重叠，PD 分离通过 DLSlime / Mooncake 实现。
- **V4 只走 TurboMind 路径**，量化版和 Ascend 版均不支持 V4。对 V4.1 的跟进明显落后于 vLLM 和 SGLang。

### 4.5 国产芯片路线：xLLM / MindIE-LLM / vllm-ascend

**xLLM**：
- 京东零售在生产中使用，2026-07 捐赠给开放原子基金会。
- 覆盖 Ascend A2/A3、寒武纪、摩尔线程、海光、沐曦、天数，KV 层用 Mooncake。
- V4 首日支持，但不支持 NVIDIA Blackwell。

**MindIE-LLM**：
- 华为官方 Ascend 推理套件，2025-12 开源。
- GitHub 上只是镜像（约 28 star），主仓库在 gitcode，社区可见度低。V4 / 950 的支持情况 [未核实]。

**vllm-ascend**：
- Ascend 上社区最活跃的路径，支持 Atlas A2/A3。
- 是否针对 950 做过专项调优 [未核实]。

**和 DeepInfer 的对比**：DeepInfer 在 Ascend 950 上的 kernel 是 DeepSeek 与华为共同打磨的生产实现（DeepGEMM-Ascend、DeepEP-Ascend、TileLang `asc/` 后端），这是它在国产卡方向上最有分量的差异点。

### 4.6 边缘与异构：KTransformers / llama.cpp / Ollama

- **KTransformers**：热专家留在 GPU，冷专家放到 CPU / 磁盘。2026-05 支持 V4-Flash，08 月支持 Ascend + CPU 卸载，10 月支持 V4-Flash 的 LoRA 微调。面向没有集群的用户。
- **llama.cpp / Ollama**：GGUF 量化，面向开发者和终端用户，和 DeepInfer 不在同一市场。

### 4.7 基础设施类：Mooncake / LMCache / NIXL / 3FS

这一类不是引擎，而是 DeepInfer 未开源部分的开源替代：
- **Mooncake**：Transfer Engine + Store，已被 vLLM、SGLang、TRT-LLM、LMDeploy、xLLM、vllm-ascend 全部集成。
- **LMCache**：KV 缓存层，12k star。
- **NIXL**：NVIDIA 的传输库。
- **3FS**：DeepSeek 自家的文件系统，已开源，约 10k star。但 DeepInfer 的 "kvcache-store" 服务本身没有开源。

详见技术调研文档。

---

## 5. 性能数据：各家公布的数字，以及为什么不能直接比

| 来源 | 模型 | 硬件 / 并行 | 数字 | 口径说明 |
|---|---|---|---|---|
| DeepInfer README | V4.1-Flash | 64×SM100，EP64DP64，128k 上下文 | decode：每 rank 64 请求时约 **15.7k tok/s/GPU**（约 246 tok/s/用户）；每 rank 256 请求时约 **26.3k tok/s/GPU**（约 103 tok/s/用户） | 离线基准；nextn=3 固定，按 75% 逐 token 接受率折算（τ=3.25）；不含负载均衡损耗 |
| DeepInfer README | V4.1-Flash | 4×SM100，EP4DP4 | prefill：chunk 32k 时约 **87k tok/s/GPU**（单步约 380 ms） | 只计非缓存 token；开启 decoder 有界重放（140 token） |
| DeepInfer Q&A | V4.1-Flash | 线上集群 | 单集群每天 **155 万亿 token**；每 GPU 小时收入 **38 美元**（低谷价） | 线上负载均衡效率约为离线理想值的 **80%** |
| DeepSeek 2025-02 | V3/R1 | H800 节点 | 每节点输入 73.7k / 输出 14.8k tok/s；成本利润率 545% | 历史基线，可用来对照两代系统 |
| llm-d v0.5 | DeepSeek（R1 级别） | 256×B200，P/D + 宽 EP | decode 约 3.1k tok/s/GPU | 不同模型（激活参数约为 V4.1-Flash 的 3 倍）、不同负载 |
| TRT-LLM blog 26 | V4-Pro | GB300，分离式部署 | 峰值 1,618 tok/s/GPU（8K 输入 / 1K 输出） | 1.6T 参数模型，和 Flash 不可比 |
| SGLang | V4-Pro | 8×B300，TP8，bs=1 | 383.7 tok/s/用户（DSpark） | 低延迟单流指标 |
| SGLang v0.5.22 | V4.1 | 8×H200 | 相对上一版本 **2.3 倍** | 版本间相对提升，不是绝对值 |

**结论**：目前**没有**任何同模型、同硬件、同负载、同接受率口径的 DeepInfer 与 vLLM / SGLang / TRT-LLM 对比。值得做的事情是：用 DeepInfer 的 `examples/sweep/run_sweep.sh` 与 vLLM / SGLang 的 bench 工具，在同一套 SM100 集群上跑同一组 Pareto 曲线（decode：每 rank 请求数 8–256；prefill：chunk 4k–64k）。这也是 DeepInfer 开源后最直接的用途之一。

---

## 6. SWOT：DeepInfer 视角

| | 内容 |
|---|---|
| **优势 S** | ① 官方生产实现，V4.1 的每个设计（YOCO、DSpark、Engram、FP4 KV）都有一手代码；② 由 C++ 调度器 + ZOS + MTPVarlen 组成的调度闭环，工程质量高；③ 同一代码库同时支持 SM100 与 Ascend 950，TileLang `cuda/` 与 `asc/` 两套后端一一对应；④ 部署 Q&A 公开了大量生产参数，行业里罕见；⑤ 批次不变与确定性采样有测试覆盖 |
| **劣势 W** | ① 只支持 1 个模型、2 种硬件，Hopper 不支持；② 没有服务层、前缀缓存、KV 存储、PD 传输和负载均衡；③ 关闭 Issues，PR 合并流程不明，部分注释为中文；④ 依赖版本锁得很死（torch 2.13、CUDA 13、NCCL 2.32.3、GCC 13 及以上） |
| **机会 O** | ① 成为 V4.x 性能的"真值基线"，推动上游追齐；② 借助社区 PR 获得 SM90、OpenAI 服务、HiCache 等扩展；③ 为国产卡（Ascend 950）提供可复制的大规模 MoE 推理范式；④ 后续 DeepSeek 新模型的推理代码会持续在这里开源 |
| **威胁 T** | ① vLLM / SGLang 已经在 DeepInfer 发布前完成 V4.1 适配，用户没有迁移动力；② NVIDIA（TRT-LLM + Dynamo）和 llm-d 占据编排层；③ 如果长期不开放服务层，社区可能在 fork 上自建，造成分裂 |

---

## 7. 选型建议

| 场景 | 推荐 | 理由 |
|---|---|---|
| 生产在线服务 V4.1，NVIDIA | **SGLang** 或 **vLLM**；规模化部署加 **llm-d / Dynamo**，KV 层用 **Mooncake / LMCache** | 栈完整、社区大、PD 分离与宽 EP 有公开复现 |
| NVIDIA Blackwell 极限性能，接受厂商锁定 | **TRT-LLM + Dynamo** | Blackwell 专项调优最深，有 NVL72 拓扑调度 |
| Ascend 950 / 国产卡 | 先用 **DeepInfer**（950 官方调优）做基线，再在 **SGLang-NPU / vllm-ascend / xLLM** 中选能上线的 | DeepInfer 缺服务层，需要其他引擎补齐 |
| 大批量离线推理、评测、数据合成（SM100） | **DeepInfer** 可直接使用 | 离线 API 足够，性能是官方基线，还有 `eval/` 现成脚本 |
| RL rollout | **SGLang**（verl / slime / Miles）或 **vLLM**（Sleep 模式） | DeepInfer 没有权重热更新接口 |
| 推理系统研究与教学 | **DeepInfer**（读代码）+ SGLang / vLLM（做实验） | DeepInfer 代码量约 5.4 万行，集中且只服务一个模型，比通用框架易读 |
| 单机、消费级、本地 | **KTransformers / llama.cpp / Ollama** | 资源门槛最低 |

---

## 8. 需要持续跟踪的信号

1. DeepInfer 是否开放 Issues、合并社区 PR，尤其是 SM90（#148）和 OpenAI 服务（#147）。注意：FlashMLA 自 2026-09-30 起已**移除 Hopper 支持**（README breaking change），SM90 移植需要自建或回退注意力 kernel，难度远大于 PR 表面上的改动规模（详见专题 T9）。
2. DeepSeek 是否开源 kvcache-store、负载均衡器和 PD 传输。Q&A 写的是"暂未完全开源"。
3. vLLM / SGLang 和 DeepInfer 的同口径基准，看社区或 SemiAnalysis InferenceX 是否会出。
4. vllm-ascend / MindIE 对 Ascend 950 的官方支持说明。
5. TRT-LLM 对 DSpark 的支持进度。

---

## 参考来源

- DeepInfer：https://github.com/deepseek-ai/DeepInfer （README、`doc/inference_deployment_qa.md`、源码）
- vLLM v0.32.0 release notes：https://github.com/vllm-project/vllm/releases/tag/v0.32.0
- llm-d：https://github.com/llm-d/llm-d ；https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale
- SGLang v0.5.22：https://github.com/sgl-project/sglang/releases/tag/v0.5.22 ；DSpark 博客：https://www.lmsys.org/blog/2026-07-06-dspark-sglang ；GB300 V4 博客：https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/
- TRT-LLM V4 技术博客：https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md
- Dynamo：https://github.com/ai-dynamo/dynamo
- LMDeploy：https://github.com/InternLM/lmdeploy ；xLLM：https://github.com/xLLM-AI/xllm ；KTransformers：https://github.com/kvcache-ai/ktransformers ；Mooncake：https://github.com/kvcache-ai/Mooncake
- DeepSeek Open Infra Index：https://github.com/deepseek-ai/open-infra-index
