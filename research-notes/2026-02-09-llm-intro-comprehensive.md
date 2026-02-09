# LLM 大模型技术全景总结（索引）

- **Date:** 2026-02-09
- **Tags:** LLM, Transformer, pre-training, SFT, RLHF, DPO, alignment, reasoning, multimodal, Mamba, KAN, distributed-training

## Context

基于 Notion「llm-intro 大模型技术总结」体系下的 10+ 个子页面整理而成。本文件为**索引和概要**，每个主题的完整详细内容见对应的专题笔记：

| 专题笔记 | 内容 | 来源 Notion 页面 |
| --- | --- | --- |
| [架构与训练详解](2026-02-09-llm-intro-architecture.md) | Attention/归一化/位置编码/激活函数/Scaling Law/LoRA/量化/部署 | 文本大模型篇 |
| [从零构建 LLM](2026-02-09-llm-intro-build-from-scratch.md) | Tokenizer/Embedding/Attention/MLP/GPT 完整代码/训练循环 | 语言模型解释与训练 |
| [Post-training 101](2026-02-09-llm-intro-post-training.md) | SFT/RL(RLHF/RLAIF/RLVR)/PPO/GRPO/DPO/评估方法 | Post-training 101 |
| [多模态大模型](2026-02-09-llm-intro-multimodal.md) | CLIP/Flamingo/架构/训练/MoE-LLaVA | 多模态大模型篇 |
| [推理能力](2026-02-09-llm-intro-reasoning.md) | 推理类型/公理化训练/因果推理 | 大模型推理reasoning |
| [幻觉详解](2026-02-09-llm-intro-hallucination.md) | 成因/检测(FActScore/SAFE)/缓解(RAG/CoVe/ITI) | 幻觉相关 |
| [受控生成与角色定制](2026-02-09-llm-intro-controlled-gen.md) | CTG/知识编辑/ICL/角色扮演 | 受控生成+角色定制 |

---

## 一、技术体系总览

```text
一、基座模型
  ├── Transformer架构与机制（Attention/位置编码/归一化）
  ├── Mamba/Mamba2/Jamba（SSM系列）
  └── KANs（Kolmogorov-Arnold Networks）

二、指令微调技术
  ├── LoRA / DoRA / LoftQ / GaLore
  └── Mixture of LoRAs

三、对齐技术
  ├── DPO / KTO / IPO / SimPO
  └── PPO / GRPO / REINFORCE

四、模型量化
  ├── GPTQ / SmoothQuant / AWQ / GGUF
  └── 4-bit NormalFloat / Double Quantization

五、多模态技术
  ├── MoE-LLaVA / Mini-Gemini / VideoLLaMA2
  └── 视觉编码器 + 连接器 + LLM 架构

六、图与大模型
  ├── GraphGPT / 知识图谱+LLM
  └── 推荐系统+大模型

七、具身智能
  └── LLM + 感知器 + RL + 系统设计
```

---

## 二、语言模型原理

### 2.1 从神经元到语言模型

- **神经元模型**：输入 × 权重 + 偏置 → 激活函数 → 输出
- **万能近似定理**：只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数
- **BP 反向传播**：用链式法则计算损失函数对每层权重的梯度，通过梯度下降更新权重

### 2.2 数据科学

**三类训练数据**：

| 阶段 | 数据特点 | 数据形式 |
| --- | --- | --- |
| 预训练 | 覆盖面广、量大、尽可能优质 | 纯文本 |
| 指令微调 | 数据要精，量不要求多 | `<instruct, answer>` |
| 强化学习 | 数据要精，量多于 SFT | `<instruct, answer_accept, answer_reject>` |

**数据处理**：文本类（清洗/去重/质量过滤）、代码类（语法检查/执行验证）、多模态（格式对齐）

**数据质量评估指标**：

- **困惑度(Perplexity)**：PPL 越低，模型赋予文本的概率越高
- **ErrorL2-Norm (EL2N)**：EL2N 分数低 = 模型早期学习到（简单），高 = 需要更多迭代（困难）
- **记忆分数(Memorization)**：贪婪生成 N 个 token 中与原始数据精确匹配的比例

---

## 三、从零构建 LLM

### 3.1 Tokenizer

| 方法 | 特点 | 代表模型 |
| --- | --- | --- |
| BPE | 频数最高的相邻子词合并 | GPT-2 (Byte-level BPE) |
| WordPiece | 提升语言模型概率最大的相邻子词合并 | BERT |
| Unigram | 减量法，先大词表再不断丢弃 | - |
| SentencePiece | 输入视为原始流（含空格），使用 BPE/Unigram | ALBERT, XLNet, T5 |

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize("Using a Transformer network is simple")
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']
ids = tokenizer.convert_tokens_to_ids(tokens)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]
```

### 3.2 核心模块实现

**Embedding**：

数学本质是以 one-hot 为输入的单层全连接。token embedding + position embedding 相加。

```python
wte = nn.Embedding(config.vocab_size, config.n_embd)
wpe = nn.Embedding(config.block_size, config.n_embd)
```

**Causal Self-Attention**：

通过计算 token 间相似性捕捉语义关系。

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)
```

**MLP (FFN)**：两个线性层拟合数据分布，存储大部分知识。

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))
```

**Transformer Block**：Pre-Norm + Attention + Residual + MLP

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

**完整 GPT 模型**：Embedding → N × Block → LayerNorm → LM Head，权重共享（wte.weight = lm_head.weight）

### 3.3 训练流程

**预训练 (CLM)**：因果语言建模，预测下一个 token。通过 Mask Attention 保证因果性，通过 shifted labels 计算 Cross-Entropy Loss。

```python
shift_logits = lm_logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()
loss = CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
```

**SFT 多轮对话标签构造**（最优方法 — 第三种）：

```text
inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
labels = <-100>  <assistant1> <-100>  <assistant2> <-100>  <assistant3>
```

因果注意力（Masked Attention）保证第一轮输出不受后续轮次影响。

**RLHF/PPO 训练**：三阶段——Rollout & Evaluation → Make Experience → Optimization

奖励函数：r = r_θ − λ·KL(π‖π₀)，防止模型输出乱码来欺骗奖励模型。

**解码策略**（优先级：temperature > topK > topP > typicalP）：

- Greedy Decoding：简单但单调
- Random Sampling：多样但可能不连贯
- Beam Search：平衡质量和多样性
- Temperature：低温→更确定，高温→更随机
- Top-K / Top-P (Nucleus)：限制采样范围
- 猜测解码：草稿模型预测 + 原始 LLM 并行验证

---

## 四、模型架构核心知识

### 4.1 Attention 机制

**Multi-Head Attention**：`MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O`

- 时间复杂度：O(n²·d)
- 参数量：4·n·d²

**Attention 优化**：

| 方法 | 原理 | 效果 |
| --- | --- | --- |
| **KV-Cache** | 缓存历史 K/V 避免重复计算 | 计算 O(n²)→O(n)，但长序列存储瓶颈 |
| **MQA** | 所有 Q 头共享一组 K/V | 提高 30-40% 吞吐 |
| **GQA** | 分组 Q 头共享 K/V | MHA 和 MQA 的折中 |
| **Flash Attention** | 分块计算+SRAM优化，减少 HBM 访问 | IO 优化，无精度损失 |
| **Paged Attention** | 非连续内存存储 K/V，按块划分 | 内存降低 55%，吞吐提升 2.2x |

### 4.2 归一化

- **Layer Norm**：μ = (1/H)Σxᵢ，σ = √((1/H)Σ(xᵢ-μ)²+ε)
- **RMS Norm**：去掉减均值部分，只保留方差归一化。LLaMA 采用。
- **Pre-Norm vs Post-Norm**：同深度下 Post-Norm 更优，但 LLaMA 用 Pre-Norm（模型足够深 + 恒等分支有利于梯度传播）

### 4.3 位置编码

| 类型 | 方法 | 特点 |
| --- | --- | --- |
| 绝对位置 | Sinusoidal, Learnable, **RoPE** | RoPE 既保留绝对位置又保留相对性 |
| 相对位置 | XLNet式, T5式, DeBERTa式, **ALiBi** | 通过微调注意力运算分辨相对位置 |
| 融合式 | 绝对+相对融合 | - |

**RoPE 优化**：NTK-RoPE, NTK-logn, 窗口注意力, Sparse Attention

### 4.4 激活函数

| 函数 | 公式 | 特点 |
| --- | --- | --- |
| Sigmoid | 1/(1+e⁻ˣ) | 输出(0,1)，梯度消失 |
| ReLU | max(0,x) | 收敛快，可能神经元坏死 |
| SiLU (Swish) | x·σ(βx) | 处处可导，非单调 |
| **SwiGLU** | Swish(xW+b)⊗(xV+c) | LLaMA 使用，收敛快，性能好 |

### 4.5 模型架构

- **Encoder**：BERT 系列
- **Decoder-only**：GPT / LLaMA / Qwen（主流）
- **Prefix-decoder**：GLM
- **Encoder-Decoder**：T5

### 4.6 Loss 函数

- 回归：L1, MSE, Huber
- 判别：BCE, Cross-Entropy, Focal Loss（样本不均衡）
- 排序：MarginRanking, info-nce loss

---

## 五、Post-training 101

### 5.1 从预训练到指令调优的旅程

Base model 通过 next-token prediction 编码知识，但不够有用。Post-training 将其变为 helpful, honest, harmless 的助手。

**E2E 生命周期**：Pre-training → SFT → RL (RLHF/RLAIF/RLVR)

### 5.2 SFT 详解

**数据集**：`(x, y)` 指令-响应对，规模通常 O(10K)~O(100K)

**损失函数**：NLL（等价 Cross-Entropy）

$$L_{SFT}(θ) = -E_{(x,y)~D} Σ_t log p_θ(y_t | x, y_{<t})$$

**数值稳定实现**：log-sum-exp trick

**数据质量维度**：Correctness / Consistency / Completeness / Clarity / Coverage / Verifiability / Balance / Alignment

**关键注意事项**：

- SFT 损失函数与预训练完全相同，区别在于数据是结构化的指令对
- 需要 EOS token 教模型停止生成
- 容易过拟合（数据量小）和灾难性遗忘
- 课程设计：从简单到复杂，从短到长

### 5.3 RL 训练技术

**统一目标**：max_π E[r(x,y)] − β·KL(π‖π₀)

**奖励类型对比**：

| 类型 | 奖励来源 | 适用场景 | 优缺点 |
| --- | --- | --- | --- |
| **RLHF** | Reward Model (人类偏好) | 通用对话/安全对齐 | 鲁棒但昂贵 |
| **RLAIF** | LLM Judge + 宪法 | 规模化对齐 | 便宜但有 Judge 偏差 |
| **RLVR** | 程序化验证(精确匹配/单元测试) | 数学/代码 | 精确但稀疏 |
| **Process RM** | 步级别打分 | 长推理/工具调用 | 更细粒度但标注昂贵 |
| **Rubric-guided** | 基于评分标准的聚合得分 | 多维质量评估 | 灵活但可能被 gaming |

**RL 算法对比**：

| 算法 | 需要 Critic? | 需要 RM? | On-policy? | 最佳场景 |
| --- | --- | --- | --- | --- |
| **PPO** | Yes (value fn) | Yes | Yes | RLHF/RLAIF |
| **GRPO** | No (critic-free) | Yes | Yes | RLVR (数学/代码) |
| **REINFORCE** | No (baseline) | Yes | Yes | 简单高吞吐 |
| **DPO** | No | No (直接偏好) | No | 便宜稳定的偏好调优 |

**Reward Model 训练**：基于 Bradley-Terry 模型

$$L_{pair}(θ) = -E[log σ(r_θ(x,y_w) - r_θ(x,y_l))]$$

**GRPO 核心**：对每个 prompt 采样 K 个回答，用组均值作 baseline，A_i = r_i - r̄，无需训练 value network

### 5.4 评估方法

**自动评估**：

| 领域 | 基准 | 指标 |
| --- | --- | --- |
| 数学 | GSM8K | Exact match accuracy |
| 代码 | HumanEval | Pass@k |
| MCQ 推理 | MMLU | Accuracy |
| 事实性 | TruthfulQA | Truthfulness score |

**LLM Judge 评估**：Pairwise comparison / Pointwise scoring / Reference-aware grading / Safety red-teaming

**人类评估**：Pointwise (Likert-scale) / Pairwise preference → 聚合指标：Net Win Rate / ELO Score

**ELO 更新**：E_A = 1/(1+10^((R_B-R_A)/400))，R_A' = R_A + K·(S_A - E_A)

---

## 六、指令微调技术

### 6.1 LoRA 系列

**LoRA**：W = W₀ + Δ = W₀ + BA（低秩分解）

| 变体 | 核心创新 |
| --- | --- |
| **AdaLoRA** | 参数化 SVD，自适应分配秩 |
| **QLoRA** | 4-bit NormalFloat + Double Quantization + Paged Optimizers |
| **LongLoRA** | 分组+偏移模拟全局注意力 + Embedding/Norm 层微调 |
| **DoRA** | 权重分解的低秩适应 |
| **LoftQ** | 量化+LoRA 联合优化 |
| **GaLore** | 利用 Weight Gradient 的 low rank 特性 |
| **Mixture of LoRAs** | Routing + MoA 架构 |

### 6.2 对齐技术

| 方法 | 核心思想 |
| --- | --- |
| **DPO** | 直接偏好优化，无需 RM |
| **KTO** | 基于 HALOs，使用 KL 估计 |
| **IPO** | 序列似然校准 |
| **SimPO** | 简化 DPO，无需 reference model |

### 6.3 量化技术

| 方法 | 核心思想 | 场景 |
| --- | --- | --- |
| **GPTQ** | OBQ 改进，逐行并行+分批更新+分组量化 | GPU 部署 |
| **SmoothQuant** | 平滑 activation 分布再量化 | GPU 推理 |
| **AWQ** | 选取最有价值 1% 权重保护 | GPU 部署 |
| **GGUF** | CPU 友好的量化格式 | CPU 推理 |

---

## 七、推理能力 (Reasoning)

### 7.1 推理类型

- **演绎推理**：前提为真→结论必然为真
- **归纳推理**：观察→可能正确的结论
- **溯因推理**：基于最佳解释的结论
- **类比/因果/概率推理**

### 7.2 公理化训练 (Axiomatic Training)

**核心思想**：从被动符号演示中直接学习公理规则。

**方法**：

- 将因果公理表示为 `<前提, 假设, 结果(是/否)>`
- 训练数据通过枚举传递性公理生成
- 使用 GPT-2 架构 (67M 参数) 从零训练
- 数据扰动：节点名称随机化 + 因果图拓扑多样化(顺序+随机翻转) + 链长度变化(3-6节点)
- 损失函数基于标签而非 next-token prediction

**关键发现**：

- NoPE(无位置编码) 模型在长序列泛化上表现最好
- 67M 参数 TS2(NoPE) 在因果推理上媲美 GPT-4
- 过度扰动反而阻碍泛化
- 公理化训练可从简单序列泛化到复杂因果结构

---

## 八、分布式训练

### 8.1 并行策略

| 策略 | 切分方式 | 适用场景 |
| --- | --- | --- |
| **DP/DDP** | 数据切分，参数同步 | 单机多卡/多机 |
| **FSDP/ZeRO** | 分片优化器状态/梯度/参数 | 大模型显存优化 |
| **TP** | 张量切分（水平） | 高带宽 intra-node |
| **PP** | 层级切分（垂直），1F1B 调度 | 跨机器 |
| **SP/CP** | 序列维度切分 | 长序列训练 |
| **3D/4D** | TP+PP+DP(+CP) | 超大规模训练 |

### 8.2 训练性能指标

- **吞吐率**：tokens/s 或 samples/s
- **MFU (Model FLOPs Utilization)**：模型计算 / 机器峰值算力
- **HFU (Hardware FLOPs Utilization)**：考虑重计算后的 MFU
- **线性度**：多机多卡吞吐 / 单卡吞吐 × 卡数（目标接近 1）

**MFU 计算**：每 step 前反向 FLOPs ≈ 96·B·s·l·h²（忽略 Attention 和 LM head 项）

### 8.3 显存消耗

模型(fp16) 2Ψ + 梯度(fp16) 2Ψ + AdamW 12Ψ = **16Ψ**

GPT-2(1.5B)：至少 24GB。Activation 更是大头：1.5B + seq_len=1K + bs=32 → 60GB

### 8.4 优化技术

混合精度(FP16/BF16/FP8) / Activation Checkpointing / 梯度累积 / MQA-GQA / Offload / FlashAttention

---

## 九、Scaling Law

**核心结论**：

- 模型表现与规模(N/D/C)强相关，与 shape 弱相关
- 幂方法则：L(N)=N_c/N^α_N, L(D)=D_c/D^α_D
- 模型参数增大 8x → 数据需增大 5x
- 收敛是低效的：计算量增大时，训练大模型比小模型更高效（即使不收敛）
- 最佳 batch size 与 loss 成幂方关系

---

## 十、模型评估

### 评估方式分类

**考虑标准答案**：

- 严格匹配：BLEU, ROUGE
- 语义匹配：BERTScore

**不考虑标准答案**：困惑度(PPL)

**评估阶段**：

| 阶段 | 方法 |
| --- | --- |
| 粗糙对比 | ChatGPT 基准 / 众包评价 / AlpacaEval(GPT-4 + 众包 + ELO) |
| 全面评估-综合 | C-Eval(52任务) / FlagEval |
| 全面评估-任务导向 | SocKET / TrustGPT / ChatGraph |
| 全面评估-领域导向 | MMCU(医/法/心/教) / FinanceIQ(金融) / PromptCBLUE(医疗NLP) |

---

## 十一、模型幻觉

### 原因

- **数据层面**：训练数据含虚假信息 / 重复信息导致知识 bias
- **模型层面**：解码算法不确定性 / 暴露偏差 / 参数知识错误

### 检测方法

| 场景 | 方法 |
| --- | --- |
| 有标准答案 | ROUGE/BLEU 对比 / Knowledge F1 |
| 无标准答案 | IE 抽取验证 / QA 生成对比 / NLI 蕴含检测 / 人工评估 |

### 改善方法

- **数据阶段**：使用置信度更高的数据
- **训练阶段**：PPO 对齐 / Fine-tuning / 增量学习(模型更新/神经元修正)
- **生成阶段**：降低随机性 / CoT+校验 / CoVe(Chain of Verification) / 生成+排序
- **后处理**：迭代评估和调整
- **外部反馈**：知识检索增强(RAG)

---

## 十二、训练经验与实践

### OOM 解决方案

- **训练层面**：LoRA / 减小 batch size/maxlen / 参数冻结 / 量化 / 混合精度 / 蒸馏 / 分布式
- **工具层面**：toma 算法

### 增加处理文本长度

- **硬件**：增加显存 / 量化模型
- **模型**：高压缩率 tokenizer / 位置编码改进 / 注意力优化(KV Cache/局部/窗口/逐步softmax) / Recurrent Transformer
- **外部技巧**：知识增强 / 文本分块 / Prompt 压缩(LinguaR)

### 数据配比经验

- 预训练：领域数据/通用数据 = **1:5** 最优
- SFT 微调：领域数据/通用数据 = **1:10** 最优

### 模型部署

- vLLM / TGI / TRT-LLM + Triton

---

## 十三、多模态大模型

### 13.1 核心架构

```text
编码器(视觉/音频) → 连接器(对齐到LLM空间) → LLM → [可选]生成器
```

**编码器**：NFNet-F6, ViT, CLIP ViT, EVA-CLIP ViT（视觉）；C-Former, HuBERT, Whisper（音频）

**连接器三种类型**：

| 类型 | 方法 | 特点 |
| --- | --- | --- |
| 基于投影 | MLP/多层MLP | 简单直接 |
| 基于查询 | Cross-Attention（可训练query + 编码特征作key） | 压缩到固定长度 |
| 基于融合 | 在LLM内部特征级融合 | 深度集成 |

### 13.2 CLIP：多模态基石

将文本和图像映射到共享向量空间。训练数据：400M 图文配对。对比学习 N=32768，效率比传统方法高 12 倍。

应用：图像分类（零样本）/ 文本-图像检索 / 图像生成（DALL-E 用 CLIP 筛选）/ 文本生成（多模态 LLM 的视觉编码器基石）

### 13.3 Flamingo = CLIP + 语言模型

- **视觉编码器**：对比学习的 CLIP 式模型
- **语言模型**：Chinchilla + Perceiver Resampler（统一为 64 个标准输出）+ GATED XATTN-DENSE（缺少则性能降 4.2%）
- **训练数据**：M3W(4300万网页) + ALIGN(18亿图文对) + LTIP(3.12亿对) + VTP(2700万短视频)

### 13.4 训练过程

1. **预训练**：训练连接器 Projector，冻结编码器和 LLM
2. **多模态微调**：SFT + RLHF，使模型符合人类意图

### 13.5 代表模型

- **MoE-LLaVA**：视觉大模型 + MoE，Hard/Soft Routers，三阶段训练
- **Mini-Gemini**：Dual Vision Encoders + Patch Info Mining
- **VideoLLaMA2**：视频理解 + 多任务微调

### 13.6 趋势

- 融合更丰富数据类型（ULIP：语言+图片+3D点云；ImageBind；Pathways）
- 更智能指令响应（MultiInstruct, LLaVA, InstructBLIP）
- 高效适配器（BLIP-2 以 1/54 参数超越 Flamingo-80B 8.7%）
- 输出多模态化（文本形式或多模态 Token）

---

## 十四、受控生成

### 14.1 可控文本生成 (CTG)

控制条件三种分类：

- **语义控制**：情感/主题/毒性检测的逆过程
- **结构控制**：生成文本的结构层次
- **词法控制**：包含指定关键词

### 14.2 技术流派

**In-context Learning**：从类比学习，无需参数更新

**SFT + RLHF**：

- InstructGPT 流程：人工 SFT → 偏好数据训练 RM(6B) → PPO 优化
- PEFT 三类：Prefix/Prompt-Tuning / Adapter-Tuning / LoRA

**知识编辑**：

| 类型 | 方法 | 特点 |
| --- | --- | --- |
| 保留参数-Memory | SERAC | 编辑示例存储在 Memory，检索器提取 |
| 保留参数-额外参数 | T-Patcher, CaliNET, GRACE | 引入额外可训练参数 |
| 修改参数-Meta-learning | KE, MEND | 超网络学习权重变化量 |
| 修改参数-Locate-and-Edit | Knowledge Neuron, ROME, MEMIT, PMET | 定位知识参数后直接更新 |

**后处理**：解码阶段控制（temperature/top-k/top-p/beam search/contrastive decoding）+ 正则/有限状态自动机规范输出格式

---

## 十五、幻觉详解

### 15.1 幻觉类型

- **情境幻觉**：输出与源内容不一致
- **外部幻觉**：输出与预训练知识/外部世界不一致

### 15.2 成因深入

- 预训练数据含过时/缺失/不正确信息
- **微调引入新知识的偏差**（Gekhman et al. 2024）：LLM 学习新知识慢于已知知识；一旦学习新知识会增加幻觉倾向；最佳配比是大多数 Known + 少数 Unknown

### 15.3 检测方法

| 方法 | 核心思想 | 代表工作 |
| --- | --- | --- |
| **检索增强评估** | 分解为原子事实逐一验证 | FActScore, SAFE(比人工好且便宜20x), FacTool |
| **采样一致性** | 多次采样检查一致性 | SelfCheckGPT(黑盒，无需外部KB) |
| **校准未知知识** | 评估模型对自身知识的感知 | TruthfulQA(最好LLM 58%, 人类94%), SelfAware |
| **间接查询** | 生成参考文献辅助细节验证 | Agrawal et al.(间接>直接) |

### 15.4 缓解方法

**RAG + 编辑归因**：

- **RARR**：研究阶段(寻找证据) → 修订阶段(编辑不支持内容)
- **Self-RAG**：端到端学习反思，四种反射标记（Retrieve/IsRel/IsSup/IsUse）

**行动链**：

- **CoVe(验证链)**：基线响应 → 计划验证问题 → 独立执行验证 → 最终输出。关键发现：指令调整和 CoT 不减少幻觉；分解两步 CoVe 有效
- **RECITE(背诵增强生成)**：先背诵相关信息再生成

**采样方法**：

- 事实核抽样：动态调整 p = max(ω, p·λ^(t-1))
- **ITI(推理时干预)**：线性探针识别与事实性相关的注意力头，推理时转移激活到"真实"方向

**微调**：

- **FLAME**：SFT 用比模型更具事实性的数据 + RLHF 用 FActScore 作奖励
- **DPO 事实性调整**：生成对比对 → 真实性注释 → DPO 微调
- **WebGPT / GopherCite**：Web 搜索 + SFT→RM→RL，学习引用来源

---

## 十六、角色定制与指令遵循

### 16.1 角色特征三要素

- **一致性**：稳定的属性和行为
- **拟人化**：自然的人际交互
- **吸引力**：引起用户兴趣和参与

### 16.2 数据来源

| 类型 | 方法 |
| --- | --- |
| 人类角色扮演 | 众包工作者两两配对对话 |
| LLM 合成 | GPT-4 生成角色描述+对话，人工口语化改写 |
| 文学作品提取 | 剧本/小说提取对话及角色描述 |
| 人机交互 | 深度用户与 CharacterGLM 多轮交互 |

### 16.3 训练流程

角色 prompt 设计（Claude-2 合成多样 prompt）→ SFT 微调（6B~66B ChatGLM）→ 自我完善（人机交互数据迭代优化）

### 16.4 评估维度

一致性 / 拟人化 / 吸引力 / 质量(流畅度+连贯性) / 安全性 / 正确性(幻觉检测) / 整体

---

## Open Questions

- 公理化训练能否扩展到更复杂的推理任务？与 CoT/RL 如何结合？
- Mamba/SSM 系列能否真正替代 Transformer？混合架构(Jamba)是否是最优解？
- KAN 在 LLM 中的实际应用前景如何？
- Post-training 中 SFT 和 RL 的最优配比和顺序是什么？DeepSeek R1 的 RL-first 范式会成为主流吗？
- 长文本 Reward Model 如何有效训练？
- 如何在不牺牲通用能力的前提下进行领域特化？

## References

- [Notion: llm-intro 大模型技术总结](https://www.notion.so/d856ce9195544049921129ad439ad071)
- [Notion: llm-intro 文本大模型篇](https://www.notion.so/32d8e2ddc6fd4203a6cb3a42f851ef55)
- [Notion: llm-intro 语言模型解释与训练](https://www.notion.so/b6610c8baed14ce090f98fa4104aac22)
- [Notion: Post-training 101](https://www.notion.so/26f4f7b30a5781f8a51dfeaef4c41260)
- [Notion: llm-intro 大模型推理reasoning](https://www.notion.so/901af12928c9471ea808e032f38c8a7a)
- [Notion: llm-train 分布式训练篇](https://www.notion.so/2e69e8b81a9846c99069aef780eddfbe)
- [Notion: llm-intro 多模态大模型篇](https://www.notion.so/9f29b09895204024aa13690f0f0a7021)
- [Notion: llm-intro 幻觉相关](https://www.notion.so/7e61659b4ed7485b90d48e46784a1f23)
- [Notion: llm-intro 大模型受控生成](https://www.notion.so/cdcfc464ed62456a893141014c002ba0)
- [Notion: llm-intro 角色定制与指令遵循](https://www.notion.so/ba915aed2fbe454f99d0ed540105acea)
- Attention Is All You Need (Vaswani et al., 2017)
- Training language models to follow instructions with human feedback (Ouyang et al., NeurIPS 2022)
- DeepSeek-R1 (Guo et al., 2025)
- DeepSeek-V3 Technical Report (2024)
- LIMA: Less Is More for Alignment (Zhou et al., 2023)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- DPO (Rafailov et al., 2023)
- PPO (Schulman et al., 2017)
- LoRA (Hu et al., 2021)
- Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2023)
- Post-training 101 (Fang & Sankararaman, 2025)
- RLHF Book (Nathan Lambert, 2025)
