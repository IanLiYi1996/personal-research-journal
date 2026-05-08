# CoT 机制 / 因果解释性 / RLVR 代价 — Mini-Survey (2023–2025)

- **Date**: 2026-05-08
- **Tags**: #cot #rlvr #mechanistic-interpretability #causal-analysis #survey #math-reasoning

## Context

整理一篇围绕 [KisMATH (Saha et al. 2025)](../papers/2025-kismath-causal-cot-graph.md) 的同期/上下游工作小综述。KisMATH 用因果图工具论证 LLM 在数学推理中"真的在做某种推理"，但这个结论必须放进当前学界的整体辩论里看 — 否则容易被同期的反向证据动摇。

本 survey 覆盖 **3 个主题、12 篇论文**，按主题自顶向下组织。每个主题先给"故事线"，再做单篇评析。

---

## 主题 1：CoT 机制论辩 — 真推理 vs 装饰品

### 故事线

CoT 提示自 Wei et al. 2022 起被广泛认为是"激发推理能力"的关键。但 2023 年 Lanham 等开始用扰动实验质疑 CoT 的 faithfulness — 发现模型常常**即使 CoT 被破坏也能给出正确答案**。这一线索逐渐发展成"近似检索派"的核心证据：

- **2023 年**: Lanham 把 CoT 当黑盒做行为扰动 → 发现忠实度因任务/规模而异
- **2025 上半年**: Li (Sky-T1)、Stechly 用更激进的扰动 — 即使把推理步骤改错或随机化，性能下降很小
- **2025 中**: Wang (Beyond 80/20) 转向更细粒度 — 只有 ~20% 的高熵 token 才真正承载推理选择
- **2025 下**: KisMATH 用因果图 + 表达式级干预反击 — **结构信息是真因果中介**

辩论的焦点已经从"CoT 有没有用"转移到了"CoT 的哪部分有用"。

### 论文评析

#### 1. [Wei et al. 2022 — Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903)

- **结论**: 在 prompt 里加几个"step-by-step"示例，模型在复杂数学/常识任务上准确率显著提升（PaLM 540B 在 GSM8K 从 17.9% → 56.9%）
- **方法学**: 行为级，对比 standard prompting 和 CoT prompting
- **局限**: 只证明了"用就有效"，没说为什么有效。后续辩论的起点

#### 2. [Lanham et al. 2023 — Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702) (Anthropic)

- **关键发现 (3 条)**:
  1. **任务依赖性**：模型在不同任务上对 CoT 的依赖程度差异巨大 — 有时强依赖，有时几乎不看
  2. **CoT 提升不来自额外计算**：把 CoT 替换成等长占位符，性能不复原 → 不是单纯 test-time compute
  3. **越大越不忠实**：随着模型规模增大，CoT 越来越像"事后合理化"
- **方法**: 三种扰动 — 截断、添加错误、释义重写 — 看答案稳定性
- **意义**: 第一篇严肃质疑 CoT faithfulness 的论文，给后续怀疑派定调

#### 3. [Li et al. 2025 — Structure not Content (Sky-T1)](https://arxiv.org/abs/2502.07374) (UC Berkeley + NovaSky)

- **核心实验**: 用 17k 长 CoT 样本对 Qwen2.5-32B 做 SFT → AIME 2024 从 16.7% 涨到 56.7% (+40%)
- **关键消融**:
  - 训练样本中**最终答案改错**：性能只下降 3.2 个百分点
  - **删除推理关键词**（"so", "therefore", "first" 等）：影响很小
  - **打乱推理步骤顺序**：性能显著下降
  - **删除某些步骤**：性能显著下降
- **结论**: 长 CoT 的**结构（步骤间逻辑顺序）**关键，**内容（具体表述、答案对错）**次要
- **与 KisMATH 的关系**: 一致 — KisMATH 抽出的 CCG 正是这种结构。Li 在行为层面观察到结构重要性，KisMATH 在因果层面给出机制

#### 4. [Stechly et al. 2025 — Beyond Semantics: Reasonless Intermediate Tokens](https://arxiv.org/abs/2505.13775)

- **设置**: 不用预训练 LLM，**从头训练** transformer 在形式可验证的推理任务上
- **激进发现**:
  - 训练在**完全无关的 corrupted traces**上的模型，性能与训练在正确 traces 上**几乎相同**
  - corrupted-trace 模型在 **OOD 任务上反而更好**
  - GRPO 后训练能涨 solution accuracy，但 **trace 的有效性没改善**
  - Trace 长度与问题复杂度**几乎无关**（不是 inference-time scaling 的 free lunch）
- **结论**: 警告不要把 CoT 拟人化或当作算法步骤的真实反映
- **与 KisMATH 的张力**: 这是对"真推理派"最激进的反击。但要注意：Stechly 用 toy 任务 + 从头训练，KisMATH 用真实预训练 LLM + 数学竞赛题。可能的统一解释 — pretrained LLM 在 corrupted traces 中也能"投影"出有效结构（与 Shao 的"唤醒 pretraining 表示"假说一致）

#### 5. [KisMATH (Saha et al. 2025)](https://arxiv.org/abs/2507.11408)

- 详见 [独立笔记](../papers/2025-kismath-causal-cot-graph.md)
- **关键回应**: 在数学领域用表达式级 attention suppression（p<10⁻³⁰⁰）证明：(1) CoT 中间 token 是答案的真因果中介；(2) 模型偏好 CCG 抽出的路径
- **整体定位**: 把"CoT 有用"的辩论从行为层面（performance 变化）提到机制层面（信息流）

### 主题 1 综合判断

- 在**数学推理领域**，CoT 的因果作用是真实的（KisMATH 强证据）
- 但**结构 >> 内容**（Li 强证据，KisMATH 间接支持）
- 在**形式化任务上**，pretrained LLM 似乎能从"结构噪声"中恢复有效推理（Stechly），暗示 LLM 内部存在一个比 CoT 更深层的推理机制 — CoT 可能只是它的"投影"

---

## 主题 2：因果与解释性方法学

### 故事线

如何**正确**地分析 CoT 是个开放问题。早期工作要么**手工标注**（Tan 2023, 27 题），要么**整段 CoT 当原子**（Paul 2024）— 都不能给出细粒度结论。2025 年涌现出三条方法路线：

1. **句子级**（Bogdan, Lee）：可读性好，但可能错过表达式级因果
2. **表达式级**（KisMATH）：最细粒度，需要符号化解析
3. **Token 熵级**（Wang）：完全去结构化，关注高熵 token 的角色

每条路线各有取舍，但都共享同一个核心干预手段 — **attention suppression**。

### 论文评析

#### 6. [Tan 2023 — Causal Abstraction for Chain-of-Thought](https://aclanthology.org/2023.blackboxnlp-1.12/) (BlackboxNLP)

- **方法**: 手工为 27 个 GSM8K 例子标注因果图，做节点级干预
- **关键观察**: 节点级干预常引发 LLM "self-correcting" — 单点扰动会被周围 context 修正
- **局限**: 规模太小，无法做统计推断
- **意义**: 第一次提出"图对齐干预比随机扰动更有信息量"，被 KisMATH 直接引用为方法学先驱

#### 7. [Paul et al. 2024 — Making Reasoning Matter](https://aclanthology.org/2024.findings-emnlp.882/) (EMNLP)

- **理论框架**: 把 CoT 推理建模为概率图模型，区分**直接效应 (DE)** 和**间接效应 (IE)**
- **方法**: 训练时用 mediator loss 优化 IE → 提升 faithfulness
- **局限**: 把整段 CoT 当原子中介，看不到内部结构
- **KisMATH 的扩展**: 把整段 CoT 拆成 CCG，DE/IE 分析细化到表达式级

#### 8. [Bogdan et al. 2025 — Thought Anchors](https://arxiv.org/abs/2506.19143) (Anthropic)

- **三种互补方法**:
  1. **Black-box rollout sampling**: 100 次 rollouts 测每个句子的 counterfactual importance（计算贵）
  2. **White-box attention aggregation**: 看哪些句子被后续句子大量关注（"broadcasting"）
  3. **Causal attribution via attention suppression**: 屏蔽某句子的 attention，看后续 token 受影响
- **关键发现**: 存在 **"thought anchors"** — 通常是 planning 或 backtracking 句子，对后续推理有不成比例的影响
- **方法学反思**: 作者明确指出 attention aggregation **不是可靠的因果代理**；rollout sampling 太贵
- **KisMATH 借鉴**: 选用最严谨的 attention suppression 路线，下沉到表达式级
- **工具**: www.thought-anchors.com (开源可视化)

#### 9. [Lee et al. 2025 — ReasoningFlow](https://arxiv.org/abs/2506.02532)

- **方法**: 句子级标注，带语义化边类型（computation / planning / backtracking / reflection）
- **规模**: 30 个手工标注的推理轨迹
- **意义**: 边类型更丰富，但规模仍小；可作为 KisMATH 之后的语义增强方向

#### 10. [Wang et al. 2025 — Beyond 80/20 Rule: High-Entropy Forking Tokens](https://arxiv.org/abs/2506.01939)

- **核心观察**: CoT 中只有 ~20% 的 token 是高熵的，这些 token 是"分叉点" — 决定推理走向
- **关键实验**: 只在这 20% 高熵 token 上做 PG 更新（其余冻结）：
  - Qwen3-8B：性能与全梯度持平
  - **Qwen3-32B**: AIME'25 +11.04，AIME'24 +7.71（**反超**全梯度）
  - Qwen3-14B: AIME'25 +4.79，AIME'24 +5.21
- **反向实验**: 只在 80% 低熵 token 上训 → 性能崩盘
- **与 KisMATH 的关系**: 完美互补
  - Wang 找到高熵 token 是 RL 的真实作用对象
  - KisMATH 在 R-path rank 分布上观察到这些 fork — "钟形"模型保留 fork，"指数型"压平了 fork
  - 两篇加起来给出 RLVR 机制的完整图景

### 主题 2 综合判断

- **方法学最佳实践**：attention suppression > rollout sampling（成本）> attention aggregation（不可靠）
- **粒度选择**：取决于研究目标 — 句子级看 planning，表达式级看计算依赖，token 级看决策分叉
- **规模门槛**：≥1000 题才能做严肃统计推断；KisMATH 是当前最大的因果图数据集

---

## 主题 3：RLVR 后训练的代价

### 故事线

RLVR (Reinforcement Learning with Verifiable Rewards) 是 DeepSeek-R1、OpenAI o1 的核心后训练范式。官方叙事是"RL 让模型学会反思、回溯、探索"。但 2025 年中出现了三篇互相加强的论文，质疑这一叙事：

1. **Yue 提出现象**：RL 训练后 pass@k 在大 k 时反被 base model 反超
2. **Shao 提供反例**：虚假奖励也能让 Qwen 涨点 → RL 不依赖真信号
3. **Wang 找到关键变量**：高熵 token 是 RL 真正在调整的对象
4. **KisMATH 给出几何化诊断**：钟形 vs 指数型分布

四篇加起来构成一个完整的论断：**RLVR 不是在"教模型推理"，而是在重新分布 base model 已有的概率质量；这种重新分布有时能提升 pass@1，但代价是缩小推理能力的探索边界。**

### 论文评析

#### 11. [Yue et al. 2025 — Does RL Really Incentivize Reasoning Beyond Base Model?](https://arxiv.org/abs/2504.13837) (THU)

- **核心实验**: 在多个模型族 + 多个基准上扫 pass@k (k 从 1 到 256)
- **关键发现**:
  - 小 k (k=1)：RL 模型 > Base 模型
  - **大 k (k=256)**: Base 模型 ≥ RL 模型（**反超**）
  - RL 模型生成的所有推理路径，都已包含在 base model 的采样分布里
- **机制**: RL 把概率质量集中到 reward 高的路径上 → 单次采样更高效，但**采样多样性下降**
- **额外发现**: **蒸馏**能引入新知识（与 RL 不同）
- **意义**: 颠覆"RL 训练 = 学到新推理能力"的官方叙事
- **Project**: limit-of-RLVR.github.io

#### 12. [Shao et al. 2025 — Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)

- **震撼实验**: Qwen2.5-Math-7B 在 MATH-500 上的提升幅度（vs 真 ground truth 的 +29.1%）：
  - 随机奖励：**+21.4%**
  - 格式奖励（只看是否合规）：+13.8%
  - **错误标签**奖励（主动反向）：**+24.1%**
  - 1-shot RL：+26.0%
  - Majority voting：+27.1%
- **但**：在 Llama3 / OLMo2 上，这些虚假奖励都不起作用
- **细节**: Qwen2.5-Math 有"code reasoning"行为（写代码不执行）— RLVR 后从 65% → 90%，**即使奖励虚假**
- **结论**: RLVR 主要在**唤醒 pretraining 期间学到的有用推理表示**，不是教新东西
- **方法学警告**: 未来 RLVR 研究应该在多个 model family 上验证，不要只用 Qwen
- **与 Yue 的关系**: 互补 — Yue 说"RL 不引入新东西"，Shao 说"那 RL 在做什么？答：在唤醒已有的"

#### 13. [Wang et al. 2025 — Beyond 80/20 Rule](https://arxiv.org/abs/2506.01939)

- 见主题 2 第 10 篇
- **额外贡献**: 通过 token 熵的视角统一 Yue 和 Shao 的发现 — RL 的所有"魔法"都发生在那 20% 的高熵 token 上

#### 14. [KisMATH (Saha et al. 2025)](https://arxiv.org/abs/2507.11408)

- 详见 [独立笔记](../papers/2025-kismath-causal-cot-graph.md)
- **末尾发现的几何化诊断**:
  - **DeepSeek R1 32B**（蒸馏自 R1 671B）: 钟形分布，log P(ℛ) 方差 0.92 → 保留 fork tokens → pass@10 达 **90%**
  - **Qwen3 32B** (RLVR 训练): 指数型分布，方差仅 0.0002 → fork 被压平 → pass@10 仅 **87%**
- **几何化结论**: 分布形态 = 探索能力 = pass@k 上限
- **与前三篇的整合**: 给 Yue 的"RL 缩小边界"、Shao 的"RL 唤醒已有"、Wang 的"高熵 token 是关键"提供了一个统一的可视化指标

### 主题 3 综合判断

- **RLVR 不教新推理**，只重分配概率（Yue, Shao, KisMATH 一致）
- **代价是探索能力**: 单次采样略涨，多次采样上限下降
- **机制是压平高熵分叉点**: 这是 Wang 的发现，KisMATH 用分布形态可视化
- **蒸馏是更好的选择？**: Yue 明确说"distillation does introduce new knowledge"；KisMATH 中 DeepSeek R1 32B（蒸馏）确实优于 Qwen3 32B（RLVR）
- **实践建议**:
  - 不要只在 Qwen 系上验证 RLVR 方法（Shao）
  - 监控训练过程中 R-path rank 分布的形态变化（KisMATH 启发）
  - 优先在高熵 token 上做 PG 更新（Wang 的工程化方案）

---

## 整体小结

把三个主题串起来：

1. **CoT 在数学领域确实承载因果信息**（KisMATH），但**结构 >> 内容**（Li），且**LLM 能从噪声轨迹中恢复结构**（Stechly）— 暗示真正的"推理引擎"在 pretraining 中已经形成
2. **方法学正在收敛到 attention suppression + 表达式级因果图**（KisMATH）这条最严谨的线，配合 token 熵分析（Wang）形成完整工具箱
3. **RLVR 的官方叙事正在被系统性挑战** — Yue → Shao → Wang → KisMATH 形成了一条紧凑的反驳链，结论是"RL 主要在唤醒和重分配，不在教导"

### 对未来工作的指引

- **监控训练**: 用 KisMATH 的 R-path rank 分布形态作为 RLVR 训练的可微/可观测指标，警惕过早压平
- **混合训练**: 蒸馏（引入新知识）+ 高熵 token 选择性 PG（Wang）+ 图对齐 SFT（保留结构） 可能是更好的组合
- **跨领域验证**: 上述结论几乎全在数学/代码领域 — 在科学推理、agent 规划、多模态推理上是否成立？
- **基础理论**: Stechly 暗示 LLM 内部有一个比 CoT 更深的"推理引擎"。如何从机制可解释性角度找到它？

## Open Questions

- 钟形 vs 指数型分布是否能作为后训练的早停信号？训练过程中观察什么时候开始压平 fork tokens？
- KisMATH 的 IE 强证据 vs Stechly 的 corrupted traces 也能 work — 是否因为 LLM pretrained 时就已学到结构？能否设计实验区分？
- "蒸馏 > RLVR" 在 Yue 和 KisMATH 中都被支持，但蒸馏要求强教师；如果没有强教师，RL 是否仍是次优解？
- 高熵 fork tokens 是否对应人类思维中的"决策点"？是否可以用 RL 显式增强它们的多样性而非压平？
- 上述结论能否外推到 agentic RL（多轮交互）？KisMATH 框架能否扩展到 agent trajectory 而非单次 CoT？

## References

### 主题 1 — CoT 机制论辩
- Wei et al. 2022 [Chain of thought prompting elicits reasoning in LLMs](https://arxiv.org/abs/2201.11903)
- Lanham et al. 2023 [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)
- Li et al. 2025 [LLMs Can Easily Learn to Reason from Demonstrations: Structure not content](https://arxiv.org/abs/2502.07374)
- Stechly et al. 2025 [Beyond Semantics: The Unreasonable Effectiveness of Reasonless Intermediate Tokens](https://arxiv.org/abs/2505.13775)
- Saha et al. 2025 [KisMATH](https://arxiv.org/abs/2507.11408)

### 主题 2 — 因果与解释性方法学
- Tan 2023 [Causal abstraction for chain-of-thought reasoning](https://aclanthology.org/2023.blackboxnlp-1.12/)
- Paul et al. 2024 [Making reasoning matter](https://aclanthology.org/2024.findings-emnlp.882/)
- Bogdan et al. 2025 [Thought Anchors](https://arxiv.org/abs/2506.19143)
- Lee et al. 2025 [ReasoningFlow](https://arxiv.org/abs/2506.02532)
- Wang et al. 2025 [Beyond 80/20 Rule](https://arxiv.org/abs/2506.01939)

### 主题 3 — RLVR 后训练的代价
- Yue et al. 2025 [Does RL Really Incentivize Reasoning Beyond Base Model?](https://arxiv.org/abs/2504.13837)
- Shao et al. 2025 [Spurious Rewards](https://arxiv.org/abs/2506.10947)
- Wang et al. 2025 [Beyond 80/20 Rule](https://arxiv.org/abs/2506.01939)
- Saha et al. 2025 [KisMATH](https://arxiv.org/abs/2507.11408)

### 本笔记库内交叉引用
- [Paper: 2025 KisMATH Causal CoT Graph](../papers/2025-kismath-causal-cot-graph.md)
- [Paper: 2025 Agentic RL Survey](../papers/2025-agentic-rl-survey.md)
- [Paper: 2026 Model Spec Midtraining](../papers/2026-model-spec-midtraining.md)
- [Paper: 2026 Ouro Looped LM](../papers/2026-ouro-looped-lm.md)
