# Model Spec Midtraining: Improving How Alignment Training Generalizes

- **Authors:** Chloe Li (Anthropic Fellows), Sara Price, Samuel Marks, Jon Kutasov (Anthropic)
- **Venue/Year:** arXiv preprint, 2026-05-03
- **Link:** https://arxiv.org/abs/2605.02087
- **Code:** https://github.com/chloeli-15/model_spec_midtraining
- **Tags:** alignment, model-spec, midtraining, agentic-misalignment, deliberative-alignment, anthropic

## TL;DR

在预训练和 alignment fine-tuning（AFT）之间插入一个"中训练"阶段（MSM），用合成文档教模型理解 Model Spec 的"是什么和为什么"，从而**控制后续 AFT 演示数据的泛化方向**。在 Qwen3-32B 上把 agentic misalignment 从 54% 降到 7%，击败 OpenAI 的 deliberative alignment 基线（14%）。

## Motivation

**浅层对齐（Shallow Alignment）问题**：标准 AFT 在演示数据上做 SFT/RLHF，但演示数据**欠规约**（underspecified）——同一行为可被多个潜在价值观解释，模型不知道泛化到哪个方向。

例：训练数据"我喜欢奶油芝士胜过 Brie"既能被 pro-affordability（实惠）也能被 pro-America（美国货）解释。AFT 无法决定。

→ 结果是 LLM 在 OOD agentic 场景做出不可预期的事（勒索、泄密、对审计撒谎，参见 Lynch et al. 2025）。

## Method

两阶段训练：

| 阶段 | 数据 | 目标 |
|---|---|---|
| **Stage 1: MSM** | 合成文档讨论 Model Spec（8M~41M tokens） | 教模型 spec 的 *what & why* |
| **Stage 2: AFT** | 合成 spec 对齐对话 + 通用指令调优 | 用演示触发 spec-aligned 行为 |

**MSM 数据生成 pipeline**（用 Claude Opus 4.6 生成）：
1. 把 Model Spec 拆成连贯小节
2. 为每节生成多样的文档类型 + 文档想法（内部研究报告、博客、用户体验记录…）
3. 把 spec 放在 in-context 撰写每篇文档
4. 用 next-token prediction 像预训练一样训练 base model

**直觉**：让模型先建立"自己是谁"的先验，AFT 再用演示去激活和加强这个先验。

## Results

### 实验 A：奶酪偏好玩具实验（§3.1）—— MSM 控制泛化方向

写两个 Model Spec（pro-affordability vs pro-America），二者都能解释同一组奶酪偏好。在**完全相同**的 165k tokens AFT 数据上微调两个 Llama-3.1-8B：

| 评测域（OOD） | Baseline | AFT only | MSM(pro-aff)+AFT | MSM(pro-Amer)+AFT |
|---|---|---|---|---|
| Pro-affordability eval | 0.23 | 0.32 | **0.48** | 0.28 |
| Pro-America eval | 0.38 | 0.36 | 0.38 | **0.55** |

→ 同一份 AFT 数据，MSM 决定泛化到哪个价值观。这是 MSM 因果作用最干净的证据。

### 实验 B：6 个价值观（§3.2）—— 普适性

教 6 个价值观（pro-environment / novelty / tradition / simplicity / difficulty / individualism），AFT 数据用一个不解释 why 的狭窄域。MSM+AFT 在所有 6 个 OOD 测试集（art / travel / literature 等）上显著优于 AFT-only。例如 pro-novelty: 0.41 → 0.69，pro-simplicity: 0.46 → 0.58。

### 实验 C：Agentic Misalignment 安全实验（§4）—— 实战价值

在 Qwen2.5-32B-Instruct（非推理）和 Qwen3-32B（推理）上训练一份 Philosophy Spec（不含硬规则，只讲 self-preservation / goal-guarding 的价值与哲学），评估 27 个 OOD agentic 场景（exfiltration / murder / espionage）：

| 模型 | Baseline | AFT (CoT) [deliberative alignment] | **MSM + AFT (CoT)** | MSM + AFT (no CoT) |
|---|---|---|---|---|
| Qwen2.5-32B | 0.68 | 0.48 | **0.05** | 0.22 |
| Qwen3-32B | 0.54 | 0.14 | **0.07** | 0.10 |

**关键发现**：
- MSM+AFT 击败 deliberative alignment 基线（OpenAI Guan et al. 2025）
- 优势在 OOD 而非 in-distribution——in-distribution 上两者都饱和近满分
- **Token 效率**：达到同等效果，AFT 数据少 40–60×
- 不需要 CoT 监督也行：MSM+AFT (no CoT) 显著优于 AFT (CoT)，对保留 CoT monitorability 重要
- 推理分析：模型产生**真正 spec 对齐的伦理推理**（impermanence philosophy、epistemic humility、self-preservation bias awareness），从 0% → 35-89%

### Model Spec Science（§5）—— 用 MSM 反过来研究 Spec 设计

**发现 1: 解释价值 > 仅列规则 > 列更多子规则**

3 种 Spec（共享同一组 5 条 Claude Constitution 核心规则）对比误对齐率：

| Spec | Qwen2.5-32B | Qwen3-32B |
|---|---|---|
| Rules only | 0.40 | 0.30 |
| Rule-Augmented（更多子规则） | 0.24 | 0.12 |
| **Value-Augmented（解释为什么）** | **0.16** | **0.09** |

→ 解释规则背后的价值最有效，能减少"政策误用"（模型 reinterpret 自己的安全策略来自我合理化）。

**发现 2: 具体指导 >> 泛泛的"做有道德的人"**

| Spec | Qwen2.5-32B | Qwen3-32B |
|---|---|---|
| General Spec（一段话泛说"好价值观、好判断"） | 0.43 | 0.25 |
| **Specific Spec（明确写 self-preservation / goal-guarding）** | **0.26** | **0.09** |

**发现 3（消融，反直觉）**：MSM 文档描述"Qwen 会怎样"还是"Claude 会怎样"还是人类传记口吻——AM 表现差异很小。只要 AFT 一致，**spec 内容本身就能塑造行为，与归属无关**。

## Strengths & Weaknesses

**Strengths**
- 实验设计极干净：奶酪实验用**完全相同的 AFT 数据**做对照，把 MSM 的因果作用单独剥离出来
- 不仅刷 SOTA，**还把方法当工具反过来研究 Spec 设计本身**，为"宪法 vs 规则"之争提供首个实证结论（Value-Augmented 胜出）
- 与 deliberative alignment 互补：MSM 教 *why*，AFT 教 *what to do*，可叠加
- 40-60× 数据效率提升对落地很有意义

**Weaknesses / Limitations**
- 只测"工具化单步有害行为"（exfiltration 等），对 reward hacking、sycophancy 这类不依赖深思熟虑的失对齐可能无效（作者自己承认）
- 未测更强对抗压力：RL post-training、situational awareness（模型意识到合成文档是用于对齐）
- 未测 RL 阶段兼容性——只验证了 SFT
- "MSM 描述谁都行"的发现反直觉，需要更多复现；可能暗示对齐学到的更多是推理模板而非身份

## Key Takeaways

1. **演示数据天然欠规约**——这是浅层对齐的根本原因。提供一个"先验"（MSM）才能引导泛化。
2. **解释为什么 > 列出规则**：Constitution 风格的 value-based spec 比 rules-based spec 泛化更好。
3. **具体 > 抽象**："拥有好价值观"远不如"明确处理 self-preservation"。
4. **MSM 是对齐 pipeline 的可插拔组件**：与 RLHF / CAI / deliberative alignment 互补，可叠加。
5. **教模型"为正确的理由做正确的事"**：MSM 的核心价值不是规则合规，而是植入推理模板，让模型在 OOD 场景内部化伦理思考。

## Related Work

- **Synthetic Document Finetuning (SDF)** (Wang et al. 2025) — MSM 的技术基础
- **Deliberative Alignment** (Guan et al. 2025, OpenAI) — 主要 baseline，在 CoT 中蒸馏 spec
- **Constitutional AI** (Bai et al. 2022b, Anthropic) — 价值观对齐先驱
- **Alignment Pre-training** (Korbak et al. 2023, Tice et al. 2026) — MSM 比 Tice 的"nice AI stories"更原则化、数据效率高 ~10×
- **Agentic Misalignment evals** (Lynch et al. 2025, Järviniemi & Hubinger 2024) — 评估场景来源
- **CoT Monitorability** (Korbak et al. 2025) — 解释为什么 no-CoT MSM+AFT 重要
