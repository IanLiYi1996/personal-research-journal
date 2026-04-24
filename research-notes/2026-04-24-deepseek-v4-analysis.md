# DeepSeek-V4 深度解读：百万 Token 上下文的高效智能

- **Date:** 2026-04-24
- **Tags:** #deepseek #MoE #long-context #sparse-attention #muon-optimizer #post-training #technical-report

## Context

2026 年 4 月 22 日，DeepSeek 发布了 DeepSeek-V4 系列的预览版本，包含两个模型：

| 模型 | 总参数 | 激活参数 | 上下文长度 | 精度 |
|------|--------|---------|----------|------|
| **DeepSeek-V4-Pro** | 1.6T | 49B | 1M | FP4+FP8 混合 |
| **DeepSeek-V4-Flash** | 284B | 13B | 1M | FP4+FP8 混合 |

这是 DeepSeek 继 V3（2024.12）和 V3.2（2025.12）之后的又一代旗舰模型。技术报告长达 55 页，内容密度极高。V4 的核心叙事是：**打破超长上下文处理的效率瓶颈**。

![DeepSeek-V4 性能总览](2026-04-24-deepseek-v4-analysis/fig1-performance.png)

---

## 一、核心创新概览

DeepSeek-V4 在三个维度实现了突破：

1. **混合注意力架构 (CSA + HCA)** — 将 1M 上下文的推理 FLOPs 降至 V3.2 的 27%，KV cache 降至 10%
2. **流形约束超连接 (mHC)** — 用双随机矩阵约束残差连接，提升深层网络训练稳定性
3. **Muon 优化器** — 替代 AdamW 用于大部分参数，实现更快收敛

---

## 二、架构深度剖析

### 2.1 继承自 V3 的设计

V4 保留了 DeepSeek-V3 的几个核心组件：

- **DeepSeekMoE**：细粒度路由专家 + 共享专家的 MoE 架构
- **Multi-Token Prediction (MTP)**：多 token 预测模块和目标函数
- **MLA (Multi-head Latent Attention)**：V4 通过 CSA/HCA 进一步压缩了 MLA 的 KV cache

**V4 的 MoE 变更：**
- 激活函数从 `Sigmoid(·)` 改为 `Sqrt(Softplus(·))`
- 移除了对路由目标节点数的约束
- 前 3 层 MoE 使用 **Hash 路由**（按 token ID 哈希确定专家），替代可学习路由

### 2.2 混合注意力架构：CSA + HCA ⭐⭐

这是 V4 最重要的架构创新。核心思路：**在不同层使用不同压缩率的注意力机制**。

#### 2.2.1 Compressed Sparse Attention (CSA)

CSA 整合了**压缩**和**稀疏**两种注意力加速策略：

**第一步：KV 压缩**
- 计算两组 KV entries ($C^a, C^b$) 和对应的压缩权重 ($Z^a, Z^b$)
- 每 $m$ 个 token 的 KV 被压缩为 1 个 entry，序列长度压缩 $\frac{1}{m}$ 倍
- 压缩使用 **learnable softmax 加权求和**，非简单池化

**第二步：Lightning Indexer 稀疏选择**
- 对压缩后的 KV entries 再次压缩得到 indexer keys
- 通过低秩投影生成 indexer queries
- 计算 index score 并通过 **top-k 选择器**保留最相关的压缩 KV entries
- 最终只对选中的稀疏 KV entries 做核心注意力

**第三步：滑动窗口补充**
- 额外保留最近 $n_{\text{win}}$ 个未压缩 KV entries（滑动窗口）
- 与压缩后的稀疏 KV entries 拼接后做注意力

#### 2.2.2 Heavily Compressed Attention (HCA)

HCA 采用更激进的压缩率 $m'$（$m' \gg m$），但**不使用稀疏选择**：

- 每 $m'$ 个 token 压缩为 1 个 entry
- 直接对所有压缩 KV entries 做全注意力
- 同样保留滑动窗口分支

#### 2.2.3 CSA 与 HCA 的交错部署

- V4-Flash（43 层）：前 2 层用纯滑动窗口注意力，后续层 CSA 和 HCA **交替使用**
- V4-Pro（61 层）：前 2 层用 HCA，后续层 CSA 和 HCA 交替使用
- CSA 压缩率 $m = 4$，HCA 压缩率 $m' = 128$

#### 2.2.4 关键设计细节

- **Partial RoPE**：仅对 query/KV 的**最后 64 维**应用 RoPE，既保留绝对位置信息，又通过 $-i$ 偏移实现相对位置编码
- **Attention Sink**：使用可学习的 sink logits，允许注意力分数总和不为 1
- **Grouped Output Projection**：将 $n_h$ 个输出分成 $g$ 组，先投影到低维再还原，降低输出投影成本
- **Query & KV Entry Normalization**：在核心注意力前对 query 和 KV entry 做 RMSNorm，防止 attention logit 爆炸

#### 2.2.5 效率分析

以 BF16 GQA8（head dim=128）为基准，在 **1M 上下文**设置下：

| 指标 | V3.2 → V4-Pro | V3.2 → V4-Flash |
|------|---------------|-----------------|
| 单 token 推理 FLOPs | 降至 **27%** | 降至 **10%** |
| KV cache 大小 | 降至 **10%** | 降至 **7%** |
| 相对 BF16 GQA8 基线 | KV cache 降至约 **2%** | — |

这意味着 V4 可以在现有硬件上**日常**支持 1M 上下文推理。

### 2.3 流形约束超连接 (mHC) ⭐

**问题**：标准 Hyper-Connections (HC) 通过扩展残差流宽度来提升模型性能，但在深层堆叠时频繁出现数值不稳定。

**mHC 的核心创新**：将残差映射矩阵 $B_l$ 约束到**双随机矩阵流形**（Birkhoff 多面体）：

$$B_l \in \mathcal{M} := \{M \in \mathbb{R}^{n \times n} \mid M\mathbf{1}_n = \mathbf{1}_n, \mathbf{1}_n^T M = \mathbf{1}_n^T, M \geqslant 0\}$$

**为什么选择双随机矩阵？**
- 谱范数 $\|B_l\|_2$ 恒等于 1 → 残差变换非扩张 → 前向/反向传播数值稳定
- 双随机矩阵集合在乘法下封闭 → 深层堆叠依然稳定
- 输入/输出变换 $A_l, C_l$ 通过 Sigmoid 约束为非负有界

**实现方法**：
1. 通过 RMSNorm 归一化输入
2. 生成动态（输入依赖）和静态（输入无关）两个分量
3. 输入/输出映射用 Sigmoid 约束
4. 残差映射用 **Sinkhorn-Knopp 算法**（20 次迭代）投影到双随机矩阵流形

**Wall-time 开销**：通过融合 kernel 和重计算策略，mHC 仅增加 **6.7%** 的训练时间。

### 2.4 Muon 优化器

V4 是首个在**万亿参数**规模采用 Muon 优化器的模型。

**Muon vs AdamW 的分工**：
- **Muon**：大部分模块（注意力、MoE 等）
- **AdamW**：embedding、prediction head、mHC 的静态偏置和 gating、RMSNorm 权重

**V4 版 Muon 的改进**：
- **Hybrid Newton-Schulz 迭代**：10 次迭代分两阶段——前 8 步用 $(a,b,c) = (3.4445, -4.7750, 2.0315)$ 快速逼近，后 2 步用 $(2, -1.5, 0.5)$ 精确稳定
- **RMS 重缩放**：对更新矩阵重缩放以复用 AdamW 的超参数
- **去除 QK-Clip**：V4 的注意力架构允许直接对 query 和 KV entries 做 RMSNorm，无需 QK-Clip

**Muon + ZeRO 的兼容性问题**：
Muon 需要完整梯度矩阵做正交化，而 ZeRO 将参数分片到不同 rank。V4 的解决方案：
- Dense 参数：限制 ZeRO 分片上限 + 背包算法分配
- MoE 参数：将每个专家的下投影/上投影/门矩阵展平为一个向量后均匀分布

---

## 三、基础设施创新

### 3.1 细粒度通信-计算重叠的 Expert Parallelism

**核心洞察**：在 MoE 层中，通信延迟可以被计算完全隐藏。

V4 将专家分成多个 **wave**，每个 wave 包含少量专家。当一个 wave 的通信完成后，计算立即开始，无需等待其他专家。理论加速 **1.92×**（vs 朴素方案），实测 1.50-1.73×。

**对硬件设计者的建议**：
- 每 GBps 互联带宽足以隐藏 6.1 TFLOP/s 的通信，带宽超过此阈值后收益递减
- 建议未来芯片提供更低延迟的跨 GPU 信号机制（pull-based > push-based）
- 建议用低成本元素级激活函数替代 SwiGLU，去除指数/除法运算

### 3.2 TileLang DSL

V4 使用 TileLang（一种 Domain-Specific Language）开发融合 kernel，替代手写 CUDA。亮点：
- **Host Codegen**：将 host 侧逻辑移入生成代码，CPU 验证开销从数十微秒降至 <1 微秒
- **Z3 SMT Solver**：集成 Z3 用于张量索引的形式化分析，在编译时验证属性并优化

### 3.3 批量不变和确定性 Kernel

V4 **全栈**实现了 bitwise 确定性（训练/推理 bitwise 一致），这在调试 loss spike 和保证后训练一致性方面极为重要。

### 3.4 FP4 量化感知训练 (QAT)

- 在后训练阶段引入 FP4 (MXFP4) QAT
- MoE 专家权重和 CSA indexer 的 QK 路径使用 FP4
- 关键发现：**FP4 → FP8 反量化是无损的**（FP8 E4M3 比 FP4 E2M1 多 2 个指数位）

---

## 四、预训练

### 4.1 数据

- V4-Flash 训练 **32T** tokens，V4-Pro 训练 **33T** tokens
- 相比 V3 重点加强：长文档（科学论文、技术报告）、多语言长尾知识、Agentic 数据（mid-training 引入）
- 使用 **sample-level attention masking**（不同于 V3 的 document packing 方式）

### 4.2 训练配置

| 配置 | V4-Flash | V4-Pro |
|------|----------|--------|
| 层数 | 43 | 61 |
| 隐藏维度 $d$ | 4096 | 7168 |
| 总专家数 | 256 routed + 1 shared | 384 routed + 1 shared |
| 激活专家数 | 6 | 6 |
| 专家中间维度 | 2048 | 3072 |
| CSA 压缩率 $m$ | 4 | 4 |
| HCA 压缩率 $m'$ | 128 | 128 |
| CSA attention top-k | 512 | 1024 |
| 查询头数 $n_h$ | 64 | 128 |
| mHC 扩展因子 $n_{\text{hc}}$ | 4 | 4 |
| MTP 深度 | 1 | 1 |
| 优化器 | Muon + AdamW | Muon + AdamW |
| 序列长度 | 4K → 16K → 64K → 1M | 4K → 16K → 64K → 1M |

### 4.3 训练稳定性

训练万亿参数 MoE 模型面临严重的 loss spike 问题。V4 发现了两个有效但理论机制尚不明确的技巧：

**1. Anticipatory Routing（预判式路由）**
- 解耦骨干网络和路由网络的同步更新
- 用历史参数 $\theta_{t-\Delta t}$ 计算路由索引，用当前参数 $\theta_t$ 做特征计算
- 路由索引提前一步「预判式」计算并缓存
- 仅在检测到 loss spike 时自动激活，平时关闭
- 额外 wall-time 开销约 20%

**2. SwiGLU Clamping**
- 将 SwiGLU 的线性分量 clamp 到 $[-10, 10]$，门控分量上限 cap 到 10
- 有效消除 outlier，稳定训练

### 4.4 Base 模型评估

V4-Flash-Base（13B 激活 / 284B 总）**全面超越** V3.2-Base（37B 激活 / 671B 总），在更少参数下实现更强性能。V4-Pro-Base 进一步在几乎所有维度建立新高，特别是：
- SimpleQA-Verified: 55.2（vs V3.2 的 28.3）— 近翻倍
- FACTS Parametric: 62.6（vs V3.2 的 27.1）— 超两倍
- LongBench-V2: 51.5（vs V3.2 的 40.2）

---

## 五、后训练：专家培育 + On-Policy 蒸馏 ⭐

### 5.1 两阶段范式

V4 的后训练管线与 V3.2 类似，但做了一个关键替换：**混合 RL 阶段被 On-Policy Distillation (OPD) 完全取代**。

**Stage 1: Specialist Training（领域专家培育）**

对每个目标领域（数学、代码、Agent、指令遵循等）独立训练一个专家模型：
1. SFT（高质量领域数据）
2. RL with GRPO（领域特定奖励）

**三种推理模式**：Non-Think / Think High / Think Max，通过不同的长度惩罚和上下文窗口区分。

**Generative Reward Model (GRM)**：
- 用于 hard-to-verify 任务（替代传统标量 reward model）
- **关键设计**：让 actor 网络本身兼任 GRM
- 对 GRM 也施加 RL 优化 → 联合提升评判能力和生成能力
- 仅需少量人类标注数据即可泛化

**Stage 2: On-Policy Distillation（统一模型整合）**

多个领域专家 → 一个统一模型：

$$\mathcal{L}_{\text{OPD}}(\theta) = \sum_{i=1}^N w_i \cdot D_{\text{KL}}(\pi_\theta \| \pi_{E_i})$$

关键点：
- 使用 **reverse KL**（学生采样 + 教师评分），确保 on-policy
- 采用 **full-vocabulary logit distillation**（非简单 token-level KL 近似），梯度更稳定
- 同时使用 **10+ 个教师模型**
- 教师权重按需从分布式存储加载，仅缓存最后一层 hidden states

### 5.2 Interleaved Thinking

V4 改进了 V3.2 的 thinking 管理策略：
- **Tool-calling 场景**：所有 reasoning content **全程保留**（V3.2 会在新 user message 时丢弃）
- **普通对话**：仍然丢弃前轮 reasoning（保持上下文简洁）

### 5.3 Quick Instruction

在输入序列末尾附加特殊 token，让模型在正式生成前**并行完成辅助任务**（搜索判断、标题生成、查询生成、领域分类等），共享 KV cache，显著降低 TTFT。

### 5.4 DSec — 弹性沙箱平台

V4 构建了 **DeepSeek Elastic Compute (DSec)**，一个生产级弹性沙箱平台：
- 四种执行基底：Function Call / Container / microVM / fullVM
- 基于 3FS 分布式文件系统 + EROFS 分层加载
- 支持**抢占式调度 + 断点续运**（WAL 日志粒度到 token 级）
- 单集群管理数十万并发沙箱实例

---

## 六、评估结果解读

### 6.1 DeepSeek-V4-Pro-Max vs 前沿模型

| 能力维度 | 表现 |
|---------|------|
| **知识** | 开源 SOTA。SimpleQA 领先开源模型 20+ 分，但仍落后 Gemini-3.1-Pro |
| **推理** | 超越 GPT-5.2 和 Gemini-3.0-Pro。HMMT 95.2%，IMO 89.8%。Codeforces 3206 分（首次开源匹配闭源）|
| **代码** | LiveCodeBench 93.5（最佳），SWE-Verified 80.6（与 Opus 4.6/Gemini 3.1 持平）|
| **长上下文** | MRCR 1M: 83.5（落后 Opus 4.6 的 92.9），CorpusQA 1M: 62.0 |
| **Agent** | MCPAtlas 73.6，Toolathlon 51.8。接近 Opus 4.6，落后 GPT-5.4 |
| **中文写作** | 功能性写作赢率 62.7% vs Gemini-3.1-Pro；白领任务 non-loss 率 63% vs Opus 4.6 |
| **内部 R&D 编码** | Pass rate 67%，超越 Sonnet 4.5（47%），接近 Opus 4.5（70%）|

### 6.2 V4-Flash-Max vs V4-Pro-Max

- Flash-Max 在**推理**任务上接近 Pro-Max（代码和数学差距很小）
- Flash-Max 在**知识**任务上明显落后（更少参数 = 更少记忆容量）
- Flash-Max 在**Agent**任务上也稍逊（复杂工具使用场景需要更大模型）

### 6.3 推理 effort 模式对比

- Non-Think → Think High：几乎所有任务**大幅提升**（如 HLE 从 ~8% 跳到 ~30%+）
- Think High → Think Max：**数学和代码**继续提升，知识和 Agent 提升较小
- 这证实了测试时计算扩展（test-time scaling）的有效性

---

## 七、与同期模型的对比分析

### 7.1 DeepSeek-V4 vs Nemotron 3 Super

| 维度 | DeepSeek-V4-Pro | Nemotron 3 Super |
|------|-----------------|------------------|
| 总参数 | 1.6T | 120.6B |
| 激活参数 | 49B | 12.7B |
| 长上下文方案 | CSA+HCA 混合注意力 | Hybrid Mamba-Attention |
| 效率提升路径 | 压缩 KV cache + 稀疏注意力 | 线性时间 Mamba + LatentMoE 潜空间 |
| 优化器 | Muon | AdamW |
| 训练精度 | FP4 QAT（后训练） | NVFP4（全程预训练） |
| 开源 | 权重 | 权重+数据+训练配方 |

两者代表了长上下文效率优化的**不同哲学**：V4 选择在注意力层面做压缩和稀疏化，保留 Transformer 主体；Nemotron 选择在序列建模层面用线性时间 Mamba 替代大部分注意力层。

### 7.2 V4 在 DeepSeek 系列中的演进

| 特性 | V2 | V3 | V3.2 | **V4** |
|------|-----|-----|------|--------|
| 总参数 | 236B | 671B | 671B | **1.6T** |
| 激活参数 | 21B | 37B | 37B | **49B** |
| 上下文 | 128K | 128K | 128K | **1M** |
| 注意力 | MLA | MLA | MLA+DSA | **CSA+HCA** |
| 优化器 | AdamW | AdamW | AdamW | **Muon** |
| MoE | DeepSeekMoE | DeepSeekMoE | DeepSeekMoE | DeepSeekMoE+Hash |
| 后训练 | SFT+RL | SFT+RL | SFT+RL | **Specialist+OPD** |

---

## 八、个人评价与启示

### 8.1 最打动我的创新

1. **CSA 的 Lightning Indexer**：将稀疏注意力的「选哪些 KV」这个问题转化为一个轻量级检索问题（低秩 query → compressed indexer keys → top-k），这比简单的固定窗口或随机采样要优雅得多。

2. **mHC 的双随机矩阵约束**：从流形约束的角度解决深层网络稳定性问题，比简单的梯度裁剪或 LayerNorm 更有理论保证。Sinkhorn-Knopp 20 次迭代的开销也很合理。

3. **Anticipatory Routing**：「路由用历史参数、计算用当前参数」这个解耦思路非常巧妙。虽然理论解释尚不充分，但实用价值极高。

4. **OPD 替代混合 RL**：用多教师 on-policy 蒸馏统一多个领域专家，避免了多目标 RL 的 reward hacking 和能力冲突问题。full-vocabulary logit distillation 保证了知识的忠实传递。

### 8.2 局限性与未解问题

1. **架构复杂度**：报告自己也承认「保留了很多初步验证的组件和技巧，架构相对复杂」。CSA + HCA + mHC + Muon + Hash Routing + Anticipatory Routing + SwiGLU Clamping... 哪些是真正必要的？未来需要 ablation 精简。

2. **长上下文退化**：MRCR 任务显示，128K 后性能开始下降。1M 上下文的实用性还有提升空间。

3. **闭源差距仍存在**：在知识密集型任务（SimpleQA、FACTS）上，V4 仍显著落后 Gemini-3.1-Pro。在 Agent 任务上也未达到 GPT-5.4 水平。报告自评「落后前沿约 3-6 个月」。

4. **Anticipatory Routing 的理论空白**：为什么解耦路由和骨干更新能消除 loss spike？MoE 层的 outlier 是如何产生的？这些基础问题尚无解答。

### 8.3 对行业的影响

- **1M 上下文的民主化**：V4 证明了通过架构创新可以让万亿参数模型日常运行在百万 token 上下文中，这将深刻影响 Agent、RAG、长文档分析等下游应用。
- **Muon 优化器的工业化**：V4 是首个在此规模使用 Muon 的模型，验证了其在万亿参数下的可行性。预计更多团队会跟进。
- **后训练范式转变**：Specialist + OPD 的两阶段范式可能取代混合 RL 成为主流，因为它更模块化、可扩展。

---

## Open Questions

1. **CSA vs HCA 的最优层间配比**是什么？不同任务类型是否需要不同的注意力模式分配？
2. **Muon 在超大规模下的收敛特性**如何？与 AdamW 的最终性能差距是收敛速度优势还是最终解质量优势？
3. **Anticipatory Routing 的理论机制**是什么？是否与 MoE 的 rich-get-richer 动态有关？
4. **FP4 QAT 是否能从后训练前移到预训练阶段**？（如 Nemotron 3 Super 的 NVFP4 全程训练）
5. **1M 上下文的实际利用率**有多高？大部分推理任务是否真正需要如此长的上下文？

---

## References

1. DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. DeepSeek-AI, 2026. [Technical Report](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)
2. DeepSeek-V3 Technical Report. arXiv:2412.19437, 2024.
3. DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models. arXiv:2512.02556, 2025.
4. DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. Nature 645, 2025.
5. Muon: An Optimizer for Hidden Layers in Neural Networks. Jordan et al., 2024.
6. mHC: Manifold-Constrained Hyper-Connections. Xie et al., arXiv:2512.24880, 2026.
7. TileLang: Bridge Programmability and Performance in Modern Neural Kernels. Wang et al., ICLR 2026.
8. On-Policy Distillation. Lu and Lab, Thinking Machines Lab, 2025.
