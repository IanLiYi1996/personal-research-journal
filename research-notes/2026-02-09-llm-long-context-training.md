# LLM 长上下文训练技术总结

- **Date:** 2026-02-09
- **Tags:** LLM, long-context, RoPE, attention, training, position-encoding, Llama3, GLM, QwenLong

## Context

本文综合整理了 Notion 笔记中关于 LLM 长上下文处理的核心技术，涵盖长度外推、上下文窗口扩展、训练基础设施、上下文压缩、业界训练方案以及长文本推理的强化学习等多个维度。

来源页面包括：文本长度扩充、文本大模型篇、上下文压缩与效率提升、Llama3.1 tech report 解读、Demystery LLM Agent (QwenLong-L1)、分布式训练篇、研究思路阶段性总结等。

---

## 一、问题本质

Transformer 的注意力机制计算复杂度为 O(n²·d)，内存复杂度同样是 O(n²)，这直接导致了预定义上下文长度限制（通常 512~4K tokens）。突破这一瓶颈是长上下文研究的核心目标。

**具体挑战：**

- 注意力机制的 O(n²) 复杂度导致长文本推理时间和成本过高
- 示例：处理 10K token 的文档，GPT-4o 需要约 $0.25，耗时 10+ 秒
- 盲目压缩可能遗漏关键信息，影响下游任务性能
- 压缩后的文本需保持逻辑通顺，避免语义断裂和指代不清

**KV Cache 显存估算：**

存储一个 token 的 KV Cache 所需内存 ≈ `2 × 2 × 层数 × KV头数 × 每头维度`（第一个 2 表示 K 和 V，第二个 2 表示 fp16 的 2 字节）。

以 7B Llama-2 模型处理 10K tokens 为例：`2 × 2 × 32 × 32 × 128 × 10000 ≈ 5GB`，几乎是半精度模型参数所需内存的三分之一。

---

## 二、扩充方法分类

### 2.1 免训练长度外推（Train Short, Test Long）

核心思路：不用长序列数据额外训练，仅在推理阶段修改位置编码实现外推。

**为什么主流 LLM 选择 RoPE？**

1. 不带显式远程衰减，对于旨在 Long Context 的模型至关重要
2. 通过不同频率的三角函数有效区分长程和短程，达到类似层次位置编码的效果
3. 直接作用于 Q/K，不改变 Attention 形式，与 Flash Attention 更契合，更容易 Scale Up

相比之下，ALIBI、KERPLE 等虽然也称为位置编码，但实际上只是一种 Attention Bias，没有太多位置信息，且不适用于 Encoder。它们无法在单个头内有效区分长程和短程，需要在不同头设置不同的 Decay 因子来实现。

**主要方法：**

| 方法 | 核心思想 | 优缺点 |
| --- | --- | --- |
| **Sliding Window + Attention Sink** | 保留开头几个 token（注意力"锚点/回收站"）+ 滑动窗口 | 简单但牺牲远程依赖 |
| **位置插值 (PI)** | 将位置编码乘以 L_train/L_test 缩放到训练范围内 | 压缩局部分辨率，需少量微调(~1000步) |
| **NTK-Aware Scaled RoPE** | 考虑高低频不同缩放策略 | 比 PI 更精细 |
| **Leaky ReRoPE / ReRoPE** | "保近压远"：窗口内保持原始位置，窗口外用内插 | 效果好但实现较复杂 |
| **YaRN** | 结合 NTK-Aware 和注意力温度缩放 | 综合性方案 |
| **Self-Extend** | 在 Leaky ReRoPE 基础上加 Round 运算 | 进一步减轻 OOD |

#### Attention Sink 详解

开头 Token 重要性的两种理解角度：

1. **绝对位置的"锚点"**：相对位置编码原则上只能识别相对位置，但有些任务可能比较依赖绝对位置。通过开头几个绝对位置约等于 0 的 Token 作为"标的"，每个 Token 就能测出自己的绝对位置。
2. **注意力的"回收站"**：由于 attention 求和为 1，模型会将一部分注意力放到没什么信息量的前几个 token 上，起到"不注意"的作用。去掉后注意力就全乱了。

#### 位置插值 (PI) 详解

由 kaiokendev 首先提出（博客），随后 Meta 在论文中发布同样方法并命名为 "Positional Interpolation (PI)"。

PI 不算真正的免训练长度外推方案，因为位置内插之后同样会有 PPL 爆炸的问题。原因在于：尽管避免了远处位置越界，但同时压缩了邻近 Token 的距离，严重扰乱了模型的局部分辨率。语言模型本身是非常依赖于局部关系的任务。

PI 的价值在于：有资源做长文本微调的场景下，PI 能提供更好的初始化模型。仅需约 1000 步长文本训练就能得到有效的长文本模型，比不做修改直接微调效率高很多。

#### Leaky ReRoPE / ReRoPE 详解

实现免训练长度外推的要领是"**保近压远**"，即"保证局部不失真"和"压缩远处不越界"。

- 设定窗口大小 w，在窗口内不改变相对位置（"局部不失真"）
- 在窗口外使用位置内插（"远处不越界"）
- 如果将内插因子 k 取到无穷大 → 极简 ReRoPE：窗口外位置编码都变为 w，理论上具备无限外推潜力

从 Loss 来看，Leaky ReRoPE 和 ReRoPE 几乎不损失训练长度内的效果，并且 Context 越长 Loss 越低，说明远程依赖得以保留。

**实现挑战**：RoPE 只能通过绝对位置方式实现相对位置编码，意味着只能实现线性增长的相对位置。Leaky ReRoPE 的分段线性需要算两次 Attention 矩阵然后拼接。好消息是结合 Flash Attention 的分块计算，只有窗口边界附近的块需要重复计算，增加的成本几乎可以忽略。

#### 外推税问题

免训练外推方法都无法保证训练长度内效果不变。设原模型为 f(x)，修改后为 f⁺(x)，当 x 长度不超过 L_train 时，无法保证 f(x) ≡ f⁺(x)。

**解决方案：**

- **Dynamic Scaling**：根据当前位置动态调整 Scale 因子 s(pos) = max(L_train, pos+1)/L_train。折中方案是"局部静态"——prefill 阶段计算 L_test，整轮对话使用同一个 θᵢ，下轮再更新。
- **CLEX**：将 θᵢ(pos) 假设为关于 pos 的连续函数，用神经 ODE 建模，通过微调拟合参数。实验显示不断 Dynamic Scaling 下去可以得到近乎无限的长度外推能力。

#### 架构层面改进

- **HWFA (Hybrid Window-Full Attention)**：前 L-1 层用小窗口 RoPE + Window Attention，最后一层 NoPE + Full Attention。改进版 HWFA2 = HWFA + ReRoPE，使用更大 Window Size 并恢复 Full Attention 的 RoPE，可追平标准 Attention 训练效果。Zebra 方法类似。
- **Key Norm**：对 Key 做 L2 归一化。标准 Attention 中模型倾向于增大 ‖kₙ‖（维度灾难使其比增大 cos(qₘ,kₙ) 更容易），导致 cos(qₘ,kₙ) 训练不充分——这是 Attention 无法长度外推的更本质原因。Key Norm 归一化所有 ‖kₙ‖ 为 1，迫使模型一心调整 cos(qₘ,kₙ)。注意：Key Norm 只在与 RoPE 配合时才体现外推能力，RoPE 的旋转作用丰富了夹角多样性（类似数据增广）。
- **CoCA**：修改注意力使每组 qₘ⁽ⁱ⁾,kₘ⁽ⁱ⁾ 都有 cos=1，使较大的 cos 值尽量被训练过，但有降低每个注意力头能力上限的风险。

#### RoPE Base 与上下文长度的关系

论文 "Base of RoPE Bounds Context Length" 指出：

- 更大的训练长度应该选择更大的 base，这与训练策略无关
- 提出了"长期衰减"（long-term decay）属性，推导出 RoPE base 的下限
- 如果 base 小于下限，模型可能只能保持低 PPL，但失去从长上下文中检索信息的能力
- Llama 3 将 RoPE base frequency 超参数增加到 500,000

#### 其他外推方法

- **位置扰动法**：训练阶段对位置编码进行扰动（类似数据增广），包括 PoSE 等。实验中不算太稳定。
- **调小 Base**：微调阶段调小 Base 配合短文本微调后能体现外推能力，但可能损失模型本身能力。高频学习局部相对距离，低频学习远程绝对距离，两者都很重要。用进制类比：低频=高位，去掉高位结果相当于求模。
- **不同 base 平均**：论文发现同一模型改不同 base 然后输出取平均，能增强整体性能，表明不同大小的 base 各有所长。

#### 深层洞察

> 长度外推技术虽然有了长足的进展，但依然还是一件很神秘的事情。推理阶段将 RoPE 换成 ReRoPE 就能体现一定的长度外推效果，但预训练阶段就换成 ReRoPE，训练出来的模型没有一丁点长度外推能力。这大体也跟 Key Norm 那里的分析有关，训练阶段就换 ReRoPE 降低了 qₙ,kₘ 夹角的多样性，反而让训练没那么充分。很多长度外推技术可能跟架构绑定——ALIBI、KERPLE、XPOS 等在 Multi-Head Attention + Pre Norm 上有效，在 Single Head 的 GAU + Post Norm 上从未测出长度外推能力。

### 2.2 上下文窗口扩展

直接扩大 LLM 的序列长度处理能力：

- **Flash Attention**：通过分块计算和 SRAM 优化，减少 HBM 访问，IO 感知的精确注意力
- **Flash Attention v2**：进一步优化分块计算和 kernel 融合
- **HyperAttention**：近似注意力方法
- **KV Cache + 量化**：通过将 KV Cache 量化为较低精度，可以在不牺牲太多质量的情况下显著减少内存占用，支持更长的生成
- **PagedAttention**：允许在非连续内存空间中存储连续的 K/V，将每个序列的 KV Cache 划分为固定大小的块，内存共享可降低 55% 内存使用，提升 2.2x 吞吐量

### 2.3 模型架构演进

| 架构 | 特点 | 长上下文优势 |
| --- | --- | --- |
| **SSM (S4/Mamba)** | 线性复杂度的递归结构，A/B/C/D 四个可学习矩阵 | 天然支持长序列，推理效率高 |
| **TTT** | 将隐藏层看作内层模型，自监督任务训练 | 解决复杂度和记忆的矛盾 |
| **SAMBA** | Mamba + SwiGLU + 滑动窗口注意力的混合架构 | 4K 训练外推到 1M，线性解码复杂度 |

#### SSM 详细介绍

SSM 特指深度学习中的状态空间模型，开篇之作为 2021 年的 S4，最火的变体为 Mamba。泛指一切线性 RNN 模型（RWKV、RetNet、LRU 等也可归入此类）。

基于三个时间相关变量：x(t) 状态变量、u(t) 状态输入、y(t) 输出，以及四个可学习矩阵 A(状态矩阵)、B(控制矩阵)、C(输出矩阵)、D(命令矩阵)。需要从连续形式离散化后才能在数字系统中使用。

#### TTT 详细介绍

TTT 旨在解决复杂度和长文本内容记忆之间的矛盾。主要创新：

1. **TTT 层**：将隐藏层看作内层模型，设计自监督任务训练隐藏模型。输入是序列内容 xₜ，模型参数为 Wₜ，每步对参数更新。
2. **TTT-Linear**：大量设计实验使效果优于 Transformer 和 Mamba
3. **Mini-batch + Dual form**：提升 TTT 计算效率

训练任务为自监督：构造序列 x̃ₜ，预测真实序列 xₜ（类似 denoising autoencoders）。

### 2.4 长文本利用优化

- **CoPE (Contextual Position Encoding)**：位置编码依赖上下文内容而非 token 顺序。通过门控机制测量距离，位置值可以是分数提供更细致的位置信息。每个查询向量根据上下文独立测量距离，可同时使用 token 位置和句子位置。
- **Retrieval Head**：模型中负责长文本检索的特殊注意力头（<5%），具有普遍性、稀疏性、内在性（短文本预训练即已存在）、动态激活等属性。完全剪除导致模型无法检索并产生幻觉。屏蔽检索头严重影响抽取式 QA 和 CoT 推理。
- **Infini-attention**：在单个 Transformer 块中构建掩码局部注意力和长期线性注意力。计算流程：(1) 拆分为预定长度的 segment (2) 计算局部 attention (3) 用 Q 与之前保存的信息 M 计算全局 attention (4) 通过学习门控标量 β 整合局部和全局 (5) 用当前 KV 更新 M。在 1M 序列上实现 114x 压缩比。
- **LongQLoRA**：结合 Position Interpolation + QLoRA + Shift Short Attention，推理时用全局注意力与 Flash Attention/vLLM 无缝兼容。

---

## 三、长文本训练基础设施

### 3.1 序列并行方案对比

| 维度 | DeepSpeed Ulysses | Ring Attention |
| --- | --- | --- |
| **切分方式** | 沿序列维度切 Q/K/V，All2All 转置 | FlashAttention 的分布式版本，P2P 通信 |
| **通信量** | O(N×d)，与 GPU 数无关 | O(N²×d/(P×c))，随序列长度平方增长 |
| **通信方式** | All2All（对网络拓扑要求高） | P2P（对网络需求低） |
| **模型泛化** | 受 head 数限制（GQA/MQA 时 P 不能太大） | 对架构参数不敏感 |
| **变长输入** | 友好 | 处理困难 |
| **总评** | 实现简单，可能成为主流 | 实现复杂但泛化性更好 |

### 3.2 4D 并行 (Llama 3)

Llama 3 使用 4D 并行来分片模型，并行化顺序为 **TP → CP → PP → DP(FSDP)**：

- **Tensor Parallelism (TP)**：减少内存占用，intra-node 通信
- **Context Parallelism (CP)**：将输入上下文分为多个段，减少长序列的内存瓶颈
- **Pipeline Parallelism (PP)**：通过 PipeDream-Flush / 1F1B 调度减少 bubble
- **Data Parallelism (DP)**：FSDP 分片优化器状态和梯度

### 3.3 训练时的显存挑战

**显存消耗分析（以 1B 参数为例）：**

- 模型本体（fp16）：2Ψ bytes
- 梯度（fp16）：2Ψ bytes
- AdamW 优化器：fp32 模型副本 4Ψ + momentum 4Ψ + variance 4Ψ = 12Ψ bytes
- **总计 = 16Ψ**。GPT-2 (1.5B) 至少需要 24GB

**Activation 挑战：** GPT-2 (1.5B)，序列长度 1K，batch size 32 → Activation 约 60GB。Activation Checkpointing 以 33% 重计算为代价，降至约 8GB。但 100B 模型 batch size 32 即使用 checkpointing 仍需 60GB。

**其他优化技术：**

- **混合精度 (FP16/BF16)**：降低显存并提速 2-4x
- **FP8 训练**：MS-AMP 框架比 BF16 内存减少 27%-42%，比 Megatron-LM 快 64%
- **梯度累积**：累积多 Batch 梯度后统一更新，达到大 Batch 效果
- **MQA/GQA**：Key-Value 共享减少数据读取
- **Offload 技术**：ZeRO-Offload / ZeRO-Infinity，参数/激活在 CPU 和 GPU 间来回

**训练性能公式：** 单 batch 总时间 = 数据加载 + 模型前反向 + 优化器 + 后处理 + 通信 + 调度

---

## 四、业界训练方案

### 4.1 Llama 3.1 长上下文训练

**模型规格**：8B/70B/405B，Context Length = 128K

**三阶段训练流程：**

#### 阶段一：初始预训练

- 使用 8K token 上下文窗口在 15.6T token 上预训练
- **关键设计**：使用注意力掩码防止同一序列中不同文档之间的自注意力——在标准预训练中影响有限，但对长序列持续预训练很重要
- RoPE base frequency 设为 500,000（对 ≤32K 有效）
- 128K token 词汇表（tiktoken 100K + 28K 非英语），压缩率从 3.17 提升到 3.94 字符/token
- Batch size 渐进式增大：4M tokens (seq_len=4096) → 8M (seq_len=8192) → 16M
- 预训练期间动态调整数据配比：增加非英语数据、上采样数学数据、后期加入更新的网络数据

#### 阶段二：长上下文预训练

> 在预训练的最后阶段在长序列上训练。不提前训练长序列，因为自注意力层计算量随序列长度呈二次方增长。

- **分六个阶段逐步增加上下文长度**：从 8K → 128K
- 使用约 **800B training token** 完成长上下文预训练
- 通过两个标准评估适应成功：(1) 短上下文评估性能完全恢复 (2) 完美解决对应长度的大海捞针任务

#### 阶段三：数据退火

- 对最终 40M token 线性退火学习率到 0，保持 128K 上下文长度
- 调整数据混合，上采样高质量数据源
- 计算退火期间检查点的平均值生成最终预训练模型

#### 后训练：长上下文 SFT

> 必须仔细调整数据配比以平衡短文本和长文本能力。

人工标注长文本 SFT 数据不切实际，因此主要依赖合成数据：

- **QA 数据**：从预训练混合中筛选长文档，分割成 8K 块，用 Llama 3 生成 QA 对，训练时提供完整文档
- **摘要数据**：分层摘要——先对 8K 块摘要，再对摘要再次摘要。基于摘要生成需要全局理解的 QA
- **长文本代码推理**：解析 Python import 依赖，移除关键文件，要求模型识别依赖并生成缺失代码

按序列长度（16K/32K/64K/128K）分类合成数据。**消融实验：混合 0.1% 合成长文本数据即可在长短文本基准上优化性能。**

**DPO 阶段**：只要 SFT 模型适合长文本任务，DPO 中仅使用短文本数据就不会影响长文本性能。

#### 长上下文安全性

- 长文本模型容易受多轮越狱攻击
- 在 SFT 数据中加入安全行为示例（上下文中存在不安全行为示范时）
- 缓解策略有效中和了 256 轮攻击，对错误拒绝率几乎无影响
- 特别关注：逐步升级违规 (escalation)——良性对话逐步引导到违规内容，长上下文模型更易受此攻击

### 4.2 GLM Long (1M Context)

**整体流程**：Continue Pre-Training → SFT → RLHF(DPO)

每个阶段混合长短文本数据，保持通用能力。

#### Continue Pre-Training

- **两阶段策略**：8K → 128K → 1M
- **位置编码**：调整 RoPE Base 扩展位置编码分辨率（经过充分继续预训练后，各种位置编码方法差异较小）
- **Packing + Attention 分隔**：用 Flash Attention Varlen 实现，避免无效长距离依赖（这对激活长文本能力至关重要）
- **数据构建**：
  - 第一阶段(128K)：原始预训练数据 4B + 上采样 >8K 数据 3B + 上采样 >32K 数据 3B，确保各长度区间 token 总量一致
  - 第二阶段(1M)：引入**人造长文本数据**（两阶段聚类拼接策略：先分类再用 Embedding 模型聚类），总计原始预训练 3.5B + 上采样 >8K/1.5B + >32K/1.5B + >128K/1.5B + 人造长文本 2B token

#### SFT 阶段

- **数据构建**（基于 SCM 短窗口模型生成更长 SFT 数据）：
  - 单片段自我指导 (SCI)：随机选取与 SCM 窗口匹配的片段，生成 QA
  - 多片段自我指导 (MCI)：随机选取多个片段，需综合多片段信息的复杂问题
  - 多级摘要：分割成短片段逐个摘要，再汇总生成答案
- 实验验证：用新生成的 128K SFT 数据训练的模型与标注数据模型性能相近
- 最终用 GLM4-128K 作为 SCM 生成 128K~1M 的 SFT 数据
- **Sorted Packing 训练**：融合 Packing 和 Sorted Batching 优点，根据计算量构建同批次 Pack 减少 GPU 气泡时间，配合梯度累积避免排序偏差
- **Loss Reweighting**：不同 Pack 包含数据量不同导致 loss 不均衡，对 loss 重新平衡

#### RLHF(DPO)

- 沿用短文本 Reward 模型（仅保留问题和答案，舍弃长文档）
- 尝试用长文本语言模型作 Reward 模型，结果波动较大，不如短文本 Reward 模型
- **关键未解决问题**：长文本 Reward 模型的数据标注极具挑战性

#### 训练 Infra

**变长序列并行的三种策略：**

1. **循环变长**：逐个序列应用 Ring Attention（子序列少时高效，多时效率下降）
2. **原生变长**：修改 Ring Attention 原生支持变长（高效但超长上下文时显存暴增）
3. **分治变长**：结合前两者，将 Pack 分为若干子 Pack 循环计算（GLM4-9B-Chat-1M 采用）

---

## 五、长文本推理的强化学习：QwenLong-L1

**来源**：Demystery LLM Agent 笔记

大型推理模型（OpenAI-o1、DeepSeek-R1 等）在短文本推理中表现接近人类专家，但处理长文本（如 120K tokens）时面临：

1. **训练效率低**：长文本推理奖励收敛慢，输出熵显著降低限制了策略优化中的探索
2. **优化不稳定**：长文本输出长度变化大，导致 KL 散度波动

**QwenLong-L1 框架**：渐进式语境扩展（Progressive Context Scaling）

三个核心组件：

1. **预热 SFT**：用高质量演示数据建立鲁棒初始策略
2. **课程引导分阶段 RL**：分阶段增加输入长度，稳定 short → long 优化过程
3. **难度感知回顾采样**：基于难度分数（基模型平均奖励的倒数）进行重要性采样

**混合奖励机制**：

- 规则验证：严格答案匹配和格式验证（精确度）
- LLM-as-a-Judge：语义等价性评估（召回率）

---

## 六、上下文压缩与效率提升

### 6.1 基于 SDP 的语义压缩

#### 句内压缩

提取核心语义成分：

- 核心谓词（必须保留）
- 核心论元（Agent, Patient, Theme, Experiencer）
- 按重要性排序修饰成分（依存距离、语义角色类型、词性权重）

示例：原句"在一个阳光明媚的春天早晨，小明高高兴兴地背着书包走在去学校的路上" → 压缩为"小明走在去学校的路上"（省略时间/天气/状态/方式修饰）

#### 句间压缩

构建篇章级语义依存图：

- 识别跨句关系：指代、因果、时序、扩展/细化
- 提取核心事件链
- 句子重要性评分：位置权重 + 事件链中心性 + 连接度 + 语义复杂度 + 新信息量

#### 长上下文记忆管理

**语义依存驱动的记忆存储**：

- 提取核心语义（谓词、实体、核心论元、语义框架）
- 基于语义相关性检索（谓词重叠 0.4 + 实体重叠 0.3 + 框架相似度 0.3）
- 记忆清理策略：重要性 0.4 + 新旧度 0.3 + 访问频率 0.3

**滑动窗口 + 语义锚点**：

- 窗口溢出时将最老内容创建为语义锚点（核心事件 + 关键实体 + 简洁摘要）
- 完整上下文 = 锚点摘要 + 当前窗口完整内容
- 示例：100 轮对话压缩为 500(锚点) + 2000(窗口) = 2500 tokens（原始 10000+）

### 6.2 压缩方法对比

| 方法 | 优势 | 劣势 |
| --- | --- | --- |
| 规则+启发式 | 即插即用、可解释 | 不够精准 |
| 监督学习 (Seq2Seq + Pointer) | 精确控制保留/删除 | 需标注数据 |
| 强化学习 | 直接优化下游任务 | 训练不稳定，奖励稀疏 |
| 自监督预训练+微调 | 学习语义表示 | 两阶段复杂度 |
| 混合方法 | 结合规则可控性和模型灵活性 | 复杂度较高 |

### 6.3 SDP 在 LLM 时代的研究方向

（来源：研究思路阶段性总结）

- SDP 帮助压缩输入长度：基于依存分析的记忆存储，有选择性存储历史，提升效率和上下文长度
- 从句级别 SDP 扩展到篇章级别：判断当前句子与对话历史的关系和重要性
- 模型训练 vs 免训练方法的对比研究
- SDP 辅助减轻幻觉：纠正长文本输入理解偏差，在 RAG 中对召回内容精简去重纠偏
- SDP 带来可解释性：研究每个位置关注的内容是否与依存图重合

---

## 七、评估方法

| 评测框架 | 特点 |
| --- | --- |
| **大海捞针 (Needle-in-a-Haystack)** | 在长文本中随机插入无关句子，测试检索能力。Llama 3 实现 100% 检索 |
| **Multi-Needle** | 插入 4 个针头检索 2 个，Llama 3 接近完美 |
| **LongBench-Chat** | 128K 长度，人工标注，偏实际场景 |
| **InfiniteBench** | 100K-200K 长度，12 类任务（En.QA, En.MC 等） |
| **ZeroSCROLLS** | 零样本长文本自然语言理解基准 |
| **Ruler** | 评估模型真实上下文长度 |

**远程依赖评测方案**：固定评测最后 L_train tokens 的指标，逐步增加输入 Context 长度。若远程依赖有效保留，Context 越长指标应越好。

---

## Open Questions

- 训练阶段用 ReRoPE 反而没有外推能力，推理阶段才有效——这背后的机制是 qₙ,kₘ 夹角多样性的降低导致训练不充分？能否用数据增广缓解？
- 长文本 Reward 模型如何标注和训练？GLM 和 Llama 3 团队均指出这是关键未解决问题
- 不同架构（MHA vs GQA vs MQA）对长度外推能力的影响缺乏系统研究。Llama 3 用 8 个 KV head 的 GQA
- SSM/Mamba 类模型能否完全替代 Attention 处理长上下文？SAMBA 等混合架构表现出色
- 超过 1M context 后，模型真正的信息利用率有多少？
- QwenLong-L1 的渐进式语境扩展能否与 Llama 3 的分阶段长上下文预训练结合？
- SDP 句内/句间压缩在实际长文本 LLM 推理中的效率提升量化数据
- 长上下文安全性：逐步升级违规攻击的根本防御方案

## References

- [Notion: llm-intro 文本长度扩充](https://www.notion.so/36433019108d4beebdd897b8207a1af4)
- [Notion: 上下文压缩与效率提升](https://www.notion.so/2884f7b30a57806f90ded07424e6a2e7)
- [Notion: llm-intro 文本大模型篇](https://www.notion.so/32d8e2ddc6fd4203a6cb3a42f851ef55)
- [Notion: Llama3.1 tech report 解读(1)](https://www.notion.so/e89061d6cad34411b67437b583af784b)
- [Notion: Demystery LLM Agent](https://www.notion.so/1b04f7b30a5780f28903c3c706781211)
- [Notion: llm-train 分布式训练篇](https://www.notion.so/2e69e8b81a9846c99069aef780eddfbe)
- [Notion: 研究思路/方法-阶段性总结](https://www.notion.so/1c306907b3fd4f9b95c1e8615c9da2f5)
- [苏剑林 - Transformer升级之路系列](https://spaces.ac.cn/archives/8265)
- Llama 3.1: The Llama 3 Herd of Models (Meta, 2024)
- LM-Infinite (arXiv:2308.16137)
- Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)
- Positional Interpolation (arXiv:2306.15595)
- YaRN (arXiv:2309.00071)
- CLEX (arXiv:2310.16450)
- Self-Extend (arXiv:2401.01325)
- SAMBA (arXiv:2406.07522)
- CoPE (arXiv:2405.18719)
- Infini-attention (arXiv:2404.07143)
- Retrieval Head (arXiv:2404.15574)
- Base of RoPE Bounds Context Length (arXiv:2405.14591)
- QwenLong-L1: Towards Long-Context Large Reasoning Models with RL
- LongQLoRA / LongLoRA
- DeepSpeed Ulysses / Ring Attention
- GLM Long Context Training (智谱AI技术博客)
- CoCA (arXiv:2309.08646)
- PoSE (arXiv:2309.10400)
- Scaling Laws of RoPE-based Extrapolation (arXiv:2310.05209)
