# HF Daily Papers 中文摘要：2026-08-01 ~ 08-03（含 07/30–31 迟到补录）

- **Date:** 2026-08-03
- **Tags:** #hf-daily-papers #digest #memory #rsi #gui-agent #world-model #ai-for-science

## Context

- **覆盖范围**：08/01–08/03 抓取窗口（上一份 digest 覆盖至 07/31）。**08/01–02 为周末投稿空档（各 0 篇）**，08/03 仅 8 篇；同时**复查 07/31 桶发现涨到 38 篇**（迟到条目），故本份实际是"周末空档 + 前两日迟到补录"。
- **数据获取**：逐日调用 HF `daily_papers` API。
- **去重**：延续上期修正后的做法——对照**最近 9 份 HF digest 累计 266 个 arXiv id** 去重（HF 日期桶会回溯含旧论文）。
- **体量**：窗口唯一 46 篇，去重后 **新增 46 篇**（全部为此前未覆盖）。因总量小，本份**收录全部 46 篇**而非取 Top 25。
- **主线信号**：**记忆（memory）从外部模块走向原生能力**是本期最强信号——Metis（258▲）明确提出 "memory foundation model"，另有 Memory Decoder、MemHarness、Σ-Mem、Filesystem-Based Memory 四篇同期。次线是 **RSI/AI4AI**（Frontis-MA1 168▲）与 **GUI agent**（Qwen-UI-Agent 286▲）。

> **联动**：本期两条主线都与本项目既有笔记直接相接——记忆线接 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 的 Pillar I「上下文与记忆」，RSI 线接 [W31 cross-digest](/weekly/2026-W31.md) 里记录的"五家同时表态 RSI"。

## 论文总览表（全部 46 篇，按 upvotes）

| # | arXiv | 标题 | ▲ |
|---|---|---|---|
| 1 | [2607.28618](https://huggingface.co/papers/2607.28618) | AskChem：以"论断"为中心的化学文献综合基础设施 | 292 |
| 2 | [2607.28227](https://huggingface.co/papers/2607.28227) | **Qwen-UI-Agent 技术报告**：面向真实世界的下一代 GUI agent 基座 | 286 |
| 3 | [2607.26760](https://huggingface.co/papers/2607.26760) | **Metis：记忆基础模型（Memory Foundation Model）** | 258 |
| 4 | [2607.28568](https://huggingface.co/papers/2607.28568) | **Frontis-MA1**：面向 ML 工程递归自我改进的 AI4AI 模型 | 168 |
| 5 | [2607.28624](https://huggingface.co/papers/2607.28624) | PhiZero：围绕"物理语言"构建的世界模型 | 160 |
| 6 | [2607.27380](https://huggingface.co/papers/2607.27380) | VideoCoCo：Code-as-CoT 做物理一致的视频生成 | 65 |
| 7 | [2607.27919](https://huggingface.co/papers/2607.27919) | Memory Decoder at Scale：预训练的参数化长期记忆 | 49 |
| 8 | [2607.28595](https://huggingface.co/papers/2607.28595) | Beacon：知道**何时**以及**如何**做 agentic 视觉推理 | 48 |
| 9 | [2607.26497](https://huggingface.co/papers/2607.26497) | **BM25 Wins at Scale**：RAG 范式的规模化研究 | 46 |
| 10 | [2607.28022](https://huggingface.co/papers/2607.28022) | Flux-OPD：带演化上下文的 on-policy 蒸馏 | 41 |
| 11 | [2607.27616](https://huggingface.co/papers/2607.27616) | MPIE-Bench：解剖学合理的多人交互编辑基准 | 37 |
| 12 | [2607.28625](https://huggingface.co/papers/2607.28625) | ACE-Data-0：以人为中心的环境采集作为具身数据引擎 | 36 |
| 13 | [2607.27816](https://huggingface.co/papers/2607.27816) | Beyond Borrowed Histories：面向交互角色扮演评测的人物对齐用户模拟 | 31 |
| 14 | [2607.28509](https://huggingface.co/papers/2607.28509) | RefCaptioner：多参考图接地的视频描述 | 25 |
| 15 | [2607.26769](https://huggingface.co/papers/2607.26769) | **See2Think**：多模态模型真的用了中间视觉状态吗？ | 22 |
| 16 | [2607.27703](https://huggingface.co/papers/2607.27703) | SpatialCLI：先用空间工具学推理，再脱离工具 | 21 |
| 17 | [2607.28582](https://huggingface.co/papers/2607.28582) | β-OPSD：用策略优化推导、用自蒸馏训练 | 20 |
| 18 | [2607.28611](https://huggingface.co/papers/2607.28611) | Chimera：混合视觉扩散 Transformer 的设计与 Chinchilla 式扩展 | 17 |
| 19 | [2607.28362](https://huggingface.co/papers/2607.28362) | ShadowDancer：从"视频及其影子"学统一动力学表示 | 17 |
| 20 | [2607.28617](https://huggingface.co/papers/2607.28617) | AISPA：以用户为中心的系统提示审计 | 15 |
| 21 | [2607.28410](https://huggingface.co/papers/2607.28410) | 大语言模型能执行母单（parent orders）吗？（金融） | 15 |
| 22 | [2607.28272](https://huggingface.co/papers/2607.28272) | **MemHarness：记忆是被重构的，而非重放的** | 14 |
| 23 | [2607.27372](https://huggingface.co/papers/2607.27372) | Explorative Modeling：解锁预训练的第三条轴 | 13 |
| 24 | [2607.26056](https://huggingface.co/papers/2607.26056) | INTACT：免搜索世界模型的同构意图-动作学习 | 13 |
| 25 | [2607.27958](https://huggingface.co/papers/2607.27958) | Σ-Mem：面向多 agent 系统的在线可靠性记忆 | 12 |
| 26 | [2607.28374](https://huggingface.co/papers/2607.28374) | LEDGERMIND：溯源约束的多模态 agentic 推理 | 10 |
| 27 | [2607.28415](https://huggingface.co/papers/2607.28415) | QQWorld：用分位数-分位数匹配做世界模型正则化 | 9 |
| 28 | [2607.28074](https://huggingface.co/papers/2607.28074) | **Echoverse**：训练 computer-use agent 的深度演化环境 | 9 |
| 29 | [2607.27230](https://huggingface.co/papers/2607.27230) | **Multi-Head Attention Residuals** | 8 |
| 30 | [2607.26637](https://huggingface.co/papers/2607.26637) | 面向 LLM agent 的文件系统式记忆：组织与演化 | 7 |
| 31 | [2607.23782](https://huggingface.co/papers/2607.23782) | N₀-VTLA：视觉-触觉-语言-动作模型的规模化 | 5 |
| 32 | [2607.28627](https://huggingface.co/papers/2607.28627) | ReToken：一个 token 改善视觉语言模型 | 5 |
| 33 | [2607.23802](https://huggingface.co/papers/2607.23802) | **从 RLVR 到 RLSVR**：任务变换诱导自可验证性 | 4 |
| 34 | [2607.26627](https://huggingface.co/papers/2607.26627) | 重思投机解码中的有损验证 | 4 |
| 35 | [2607.27652](https://huggingface.co/papers/2607.27652) | **Harness-G**：面向搜索 agent 的图结构 harness | 3 |
| 36 | [2607.23193](https://huggingface.co/papers/2607.23193) | OmniScope：模态解耦的 token 压缩 | 3 |
| 37 | [2607.18806](https://huggingface.co/papers/2607.18806) | AI Tour Meeting：LLM agent 做团队旅行规划 | 3 |
| 38 | [2607.29679](https://huggingface.co/papers/2607.29679) | 视觉生成中文本条件的 scaling 性质 | 2 |
| 39 | [2607.28308](https://huggingface.co/papers/2607.28308) | 超越几何互补性：稀疏…中的相干重叠 | 2 |
| 40 | [2607.28319](https://huggingface.co/papers/2607.28319) | Fairness Pruning：定位 GLU-MLP 层中的人群偏见 | 2 |
| 41 | [2607.20891](https://huggingface.co/papers/2607.20891) | **Deep Research 可靠吗？** 误导性知识诱发错误结论 | 2 |
| 42 | [2607.29677](https://huggingface.co/papers/2607.29677) | ExtractBench：schema 引导的企业文档抽取基准 | 1 |
| 43 | [2607.28675](https://huggingface.co/papers/2607.28675) | Meshy T2：用 flow matching 做快速原生网格生成 | 1 |
| 44 | [2607.29025](https://huggingface.co/papers/2607.29025) | 面向多参考一致性的评估-验证奖励 | 0 |
| 45 | [2607.25289](https://huggingface.co/papers/2607.25289) | AMRD：轻量化的自适应多教师关系蒸馏 | 0 |
| 46 | [2607.16922](https://huggingface.co/papers/2607.16922) | Pedestrian Archetypes Extension | 0 |

## 分主题详解

### 主题一：记忆从"外部模块"走向"原生能力"（本期最强，5 篇）

这是本期我认为**最有结构性意义**的一簇——它们共同指向同一个转向：**agent 记忆一直是外挂模块，现在开始被做进模型本体**。
- **Metis：Memory Foundation Model**（#3, ▲258，见 Deep Dive 1）：明确提出 "memory foundation model" 概念，把记忆做成 backbone 内**持续演化的记忆状态** + **原生记忆过程**。
- **Memory Decoder at Scale**（#7, ▲49）：**预训练的参数化长期记忆**——记忆不再是检索出来的，而是被训进参数。
- **MemHarness：记忆是被重构的，而非重放的**（#22, ▲14）：标题即论点，直指当前"存-取"式记忆的认知错位。
- **Σ-Mem**（#25, ▲12）：面向多 agent 系统的**在线可靠性记忆**。
- **面向 LLM agent 的文件系统式记忆**（#30, ▲7）：另一条路——把记忆组织成文件系统并让它演化。

> 五篇同期出现，且从"做进权重"（Memory Decoder）到"当文件系统用"（Filesystem-Based）覆盖了完整光谱。这正好填补我 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 里 Pillar I「上下文与记忆」的最新进展。

### 主题二：RSI / AI4AI 与自蒸馏（6 篇）

- **Frontis-MA1**（#4, ▲168，见 Deep Dive 2）：把递归自我改进落到**可执行的 ML 工程**测试床，35B 模型在 MLE-Bench Lite 上从 39.39% → **71.21%**，逼近 2.8T 的 Kimi K3。
- **Flux-OPD**（#10, ▲41）：带**演化上下文**的 on-policy 蒸馏（延续本项目记录的 OPD 主线）。
- **β-OPSD**（#17, ▲20）：用策略优化推导、用自蒸馏训练。
- **从 RLVR 到 RLSVR**（#33, ▲4）：**任务变换诱导"自可验证性"**——不需要外部 verifier，把任务改写成自身可验证的形式。这个思路很妙，且直接呼应 Cherny 说的"自我验证是新瓶颈"（见 [harness 保质期深读](/research-notes/2026-07-31-blog-harness-shelf-life.md)）。
- **Explorative Modeling**（#23, ▲13）：声称解锁**预训练的第三条轴**。
- **AMRD**（#45）：自适应多教师关系蒸馏。

### 主题三：Agent 与 harness 工程（9 篇，篇数最多）

- **Qwen-UI-Agent 技术报告**（#2, ▲286）：面向真实世界的**下一代 GUI agent 基座模型**——本期第 2 名，说明 GUI agent 已进入"做基座"阶段。
- **Echoverse**（#28, ▲9）：训练 computer-use agent 的**深度演化环境**——注意这篇上周已在 [Tech Blogs W31h](/tech-blogs/2026-W31h.md) 以 MSR 博客形式出现，**现在论文也上了**（博客→论文的时间差）。
- **Harness-G**（#35, ▲3）：面向搜索 agent 的**图结构 harness**——harness 开始有专门的结构设计论文。
- **Beacon**（#8, ▲48）：知道**何时**以及**如何**做 agentic 视觉推理（"何时"这个维度常被忽略）。
- **SpatialCLI**（#16, ▲21）：先用空间工具学推理，**再脱离工具**——工具作为训练脚手架而非永久依赖，与"能力内化"主线同构。
- **AISPA**（#20, ▲15）：以用户为中心的**系统提示审计**——呼应本周 Reddit 的 agent 破坏事故。
- **LEDGERMIND**（#26, ▲10）：**溯源约束**的多模态 agentic 推理。
- **Deep Research 可靠吗？**（#41, ▲2）：**误导性知识诱发错误结论**——直击 deep research 类产品的可靠性。
- AI Tour Meeting（#37）：LLM agent 做团队旅行规划。

### 主题四：世界模型 / 具身 / 视频（8 篇）

- **PhiZero**（#5, ▲160）：围绕**"物理语言"**构建世界模型——本期第 5 名。
- **VideoCoCo**（#6, ▲65）：**Code-as-CoT** 做物理一致的视频生成（用代码当思维链，很有意思的一招）。
- **ShadowDancer**（#19, ▲17）：从"视频及其影子"学**统一动力学表示**，教世界模型任意动作。
- **QQWorld**（#27, ▲9）：分位数-分位数匹配做世界模型正则化。
- **INTACT**（#24, ▲13）：**免搜索**世界模型的同构意图-动作学习。
- **ACE-Data-0**（#12, ▲36）：以人为中心的环境采集作为**具身数据引擎**——与本周 Tech Blogs 记的"物理 AI 从采数据到造世界"（World Labs 收购 SceniX）形成对照，两条路线并行。
- **N₀-VTLA**（#31, ▲5）：视觉-**触觉**-语言-动作模型规模化（触觉模态少见）。
- Meshy T2（#43）：flow matching 做原生网格生成。

### 主题五：检索与评测的"反直觉"结果

- **BM25 Wins at Scale**（#9, ▲46）：RAG 范式的规模化研究得出——**在规模上 BM25 胜出**。这是个值得注意的反直觉结论：稠密检索并非在所有规模下都占优。
- **See2Think**（#15, ▲22）：**多模态模型真的用了中间视觉状态吗？** 又一篇"戳穿"式评测（延续本项目记录的"测失效模式"趋势）。
- **MPIE-Bench**（#11, ▲37）：解剖学**合理性**的多人交互编辑基准。
- **Beyond Borrowed Histories**（#13, ▲31）：人物对齐的用户模拟做角色扮演评测。
- ExtractBench（#42）：企业文档抽取基准。

### 主题六：科学与垂域

- **AskChem**（#1, ▲292，本期榜首）：**以"论断（claim）"为中心**的化学文献综合基础设施。这个设计取向值得注意——不是做检索或摘要，而是把**可追溯的论断**当一等公民。恰好与本周 Tech Blogs 的 [OpenAI 十项数学进展深读](/research-notes/2026-08-03-blog-openai-ten-math-advances.md)（可验证性）同频。
- **LLM 能执行母单吗？**（#21, ▲15）：金融执行类任务。
- **Fairness Pruning**（#40）：定位 GLU-MLP 层中的人群偏见。

### 主题七：架构

- **Multi-Head Attention Residuals**（#29, ▲8）：**注意 —— 这正是 [Kimi K3 技术报告](/research-notes/2026-07-27-kimi-k3-report.md) 里 Attention Residuals（AttnRes）的多头推广**。K3 报告引用的 AttnRes 是 [57]，这篇是其后续，值得单独追。
- **Chimera**（#18, ▲17）：混合视觉扩散 Transformer 的 Chinchilla 式扩展研究。
- **重思投机解码中的有损验证**（#34, ▲4）、**OmniScope**（#36）token 压缩、**ReToken**（#32）。

## Deep Dive 1：Metis —— 把记忆做进模型本体（▲258）

[arXiv:2607.26760](https://huggingface.co/papers/2607.26760)（MemTensor 上海 / 人大 / NUS / 上交 / 同济）

**它挑战的现状**：近年 agent 的原生能力不断被"内化"进基座模型——多模态基座、大推理模型都是这么来的。**但 agent 记忆至今主要靠外部模块实现，原生记忆能力基本无人探索。**

**核心提法**：**memory foundation model（记忆基础模型）**。论文把"原生记忆"形式化为两件事：
1. backbone 内部一个**持久且动态演化的记忆状态**（persistent and dynamically evolving memory state）；
2. **原生记忆过程**（native memory procedures）——记忆的读写由模型自身完成，而非外部编排。

![Metis 的三重对比：**架构**上外部记忆是"解耦"(推理与记忆分离)，原生记忆是"耦合"(把记忆并入 Transformer block)；**优化**上外部方案的梯度在检索处被阻断(Blocked)，原生方案支持端到端梯度(引入 Hyper Memory Parameter)；**效率**上外部方案要串行付出准备与适应的额外时间，原生方案可与原始注意力**并行**（arXiv:2607.26760 Fig.）](2026-08-03-hf-daily-papers-aug01-03/metis-x1.png)

这张图把"为什么外挂记忆不够"讲得很清楚，三个层面各有硬伤：
- **架构**：外部记忆与推理解耦 → 记忆无法参与表示学习；
- **优化**：检索是不可微操作 → **梯度被阻断**，记忆模块无法端到端训练；
- **效率**：准备（embedding/匹配/重排/拼接）与适应（抽取/摘要）都是**串行额外开销**。

Metis 的对策分别是：把记忆并入 backbone、引入 **Hyper Memory Parameter** 使其可微、让记忆注意力与原始注意力**并行**。

**我的看法**：这篇的价值在于**把一个工程惯例上升成研究问题**。我在 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 里记过一条判断：**"harness 里显式实现的能力，后续会被内化进模型策略"**——记忆是这条判断最典型的待验证项，而 Metis 正是在做这件事。结合本周 Cherny 说的"harness 保质期半年"（[深读](/research-notes/2026-07-31-blog-harness-shelf-life.md)），**如果原生记忆成立，那今天所有外挂记忆框架（Mem0/MemGPT 类）都在保质期倒计时里**。

一个待观察点：论文附录有 "Roadmap for Memory Foundation Models" 与 "Transfer to a Llama Backbone" 两节——**可迁移性**是这条路线能否普及的关键（若只在自家 backbone 成立，影响就有限）。

## Deep Dive 2：Frontis-MA1 —— 把 RSI 做成可执行的测试床（▲168）

[arXiv:2607.28568](https://huggingface.co/papers/2607.28568)

**问题设定很清晰**：递归自我改进（RSI）要求 AI 系统能改进"造 AI 的过程"（即 **AI4AI**）；而**机器学习工程（MLE）提供了一个具体、可执行的测试床**——因为 MLE 任务有明确的执行反馈。

**他们做了什么**：开源全栈系统 **OpenMLE**，三层：
- **OpenMLE-Gym**：带执行反馈的可验证任务环境；
- **OpenMLE-RL**：算子学习；
- **OpenMLE-Evo**：长程搜索。

在此之上后训练 **Frontis-MA1（35B）** 作为 MLE 的**元进化 agent**，把后训练与推理统一到**四个原子程序演化算子**上：**Draft / Improve / Debug / Crossover**。同一批算子先用**执行接地的 SFT + RL** 训练（数据对所有评测基准做过去重），再**组合成长程搜索**——**学习与进化耦合在同一个循环里**。

![MLE-Bench Lite 结果：左=所有模型×所有 harness 的 Medal avg@3，**Frontis-MA1-35B（橙色）达 71.2**，位列第三、仅次于 Kimi K3(72.7) 与 GPT-5.6 Sol(72.7)，超过 GPT-5.5(68.2)、GLM-5.2、Kimi K2.6、MiniMax M3 等；右=模型规模×分数的 Pareto 图（横轴对数），**35B 的 Frontis 位于左上角 Pareto 前沿**，与 1T–10T 级模型同档（arXiv:2607.28568 Fig.1，从 PDF 渲染）](2026-08-03-hf-daily-papers-aug01-03/frontis-fig1.png)

**关键数据**（MLE-Bench Lite，**单张 RTX 4090 限 12GB 显存、每任务 12 小时预算**）：

| 配置 | Medal Average |
|---|---|
| 基座模型 | 39.39% |
| + OpenMLE-Evo | **60.61%** |
| + OpenMLE-Evo-Max（基准无关的经验先验 + 异步搜索） | **71.21%** |

对比：**超过 GPT-5.5 + Codex，逼近 GPT-5.6 Sol 与 2.8T 的 Kimi K3**。

**迁移性验证**（held-out NatureBench Lite）——这是我认为最有说服力的部分，他们做了**双向消融**：
- 框架固定、**换上训练后的模型**：Match-SOTA 50% → **70%**；
- 模型固定、**换上 OpenMLE-Evo**：20% → **50%**。

**两个组件各自独立有效**，而不是只有整体好看。**模型权重与 OpenMLE 全栈已开源。**

**我的看法**：
- **这是我见过把 RSI 做得最"可落地"的一篇**。RSI 常年停留在概念或玩具设定，而这篇把它锚定在 MLE——一个**有执行反馈、可自动评分**的真实领域。这与 [W31 cross-digest](/weekly/2026-W31.md) 里记的"五家实验室同时表态 RSI"形成鲜明对比:**别人在表态,这篇在给可复现的栈**。
- **"35B 在单张 4090 上逼近 2.8T"** 这个结果与本周另一条主线同源:上期 digest 的 TurboVLA（0.2B 打平 8.3B）也是"**把复杂度从参数量转移到别处**"——TurboVLA 转到架构精简,Frontis 转到**搜索与算子设计**。
- **双向消融是方法论亮点**。很多"框架+模型"的工作只报整体提升,无法判断功劳归属;这篇明确拆开了。

**待验证点**：MLE-Bench 的任务本质是 Kaggle 式竞赛,**它与"真正的 AI 研究"之间仍有距离**（调参/特征工程 ≠ 提出新方法）。论文用 NatureBench 做了 held-out,但 RSI 的终极主张（AI 改进 AI 的**科研**过程）还需要更强的测试床。

## 趋势分析

1. **记忆正在经历"从 harness 到模型"的内化**。5 篇同期（Metis / Memory Decoder / MemHarness / Σ-Mem / Filesystem-Based）覆盖了从"训进权重"到"当文件系统用"的完整光谱。**Metis 明确提出 memory foundation model 这个范畴，是本期最有结构性意义的一步**。若这条路线成立，今天的外挂记忆框架都在保质期倒计时里——这与本周 Tech Blogs 的"harness 保质期半年"互为印证。

2. **RSI 从表态走向可复现的栈**。上周记录的是"五家实验室同时表态 RSI"（W31 cross-digest）；本周 Frontis-MA1 直接给出开源全栈 + 权重 + **双向消融**。**35B 在单张 4090 上逼近 2.8T** 的结果，与上期 TurboVLA（0.2B 打平 8.3B）同构：**把复杂度从参数量转移到别处**（一个转架构、一个转搜索与算子）。

3. **"自我验证"成为独立研究对象**。RLVR→**RLSVR**（任务变换诱导自可验证性）、Beacon（知道何时该做视觉推理）、β-OPSD（自蒸馏）——三篇从不同角度攻击同一问题。这正是 Cherny 说的"**自我验证是新瓶颈、也是大家做得最差的一环**"，本期学界开始正面回应。

4. **"可验证 / 可溯源"成为科学类系统的设计前提**。本期榜首 AskChem 以**论断（claim）为中心**、LEDGERMIND 做**溯源约束**推理、Deep Research 可靠性研究戳穿误导性知识——与本周 Tech Blogs 的 OpenAI Lean 4 形式化、Google Science One 的 Chain-of-Evidence 是**同一趋势的研究侧**：**AI 做科研，验证机制必须内建而非事后补**。

## Open Questions

1. **原生记忆能否迁移到别家 backbone？** Metis 附录有 "Transfer to a Llama Backbone" 一节，但这是这条路线能否普及的关键——若只在自家 backbone 成立，影响力有限。
2. **MLE-Bench 与"真正的 AI 研究"的距离有多远？** Frontis 证明了 AI 能做好 Kaggle 式 MLE，但"提出新方法"是另一回事。RSI 需要更强的测试床。
3. **BM25 在规模上胜出**（#9）该如何解释？如果稠密检索的优势会随规模消失，那大量 RAG 系统的技术选型需要重估。这条反直觉结论值得独立复核。
4. **See2Think 的"多模态模型真的用了中间视觉状态吗"** 若答案是否，那一大批"视觉 CoT"工作的机制解释需要修正。
5. **Multi-Head Attention Residuals**（#29）与 Kimi K3 的 AttnRes 是什么关系？K3 引用的是 [57]，这篇看起来是多头推广——**深度方向的注意力是否会成为下一代架构标配**，值得追。

## References

> 本期新增 46 篇，已全部入 `references.bib`（库 1870 条；过程中遇 arXiv 429，靠退避重试 + OpenAlex fallback 分三批完成）。以下列前 20 的 HF link（完整 46 篇见总览表）。

- AskChem — https://huggingface.co/papers/2607.28618
- Qwen-UI-Agent — https://huggingface.co/papers/2607.28227
- Metis: Memory Foundation Model — https://huggingface.co/papers/2607.26760
- Frontis-MA1 — https://huggingface.co/papers/2607.28568
- PhiZero — https://huggingface.co/papers/2607.28624
- VideoCoCo — https://huggingface.co/papers/2607.27380
- Memory Decoder at Scale — https://huggingface.co/papers/2607.27919
- Beacon — https://huggingface.co/papers/2607.28595
- BM25 Wins at Scale — https://huggingface.co/papers/2607.26497
- Flux-OPD — https://huggingface.co/papers/2607.28022
- MPIE-Bench — https://huggingface.co/papers/2607.27616
- ACE-Data-0 — https://huggingface.co/papers/2607.28625
- Beyond Borrowed Histories — https://huggingface.co/papers/2607.27816
- RefCaptioner — https://huggingface.co/papers/2607.28509
- See2Think — https://huggingface.co/papers/2607.26769
- SpatialCLI — https://huggingface.co/papers/2607.27703
- β-OPSD — https://huggingface.co/papers/2607.28582
- Chimera — https://huggingface.co/papers/2607.28611
- ShadowDancer — https://huggingface.co/papers/2607.28362
- AISPA — https://huggingface.co/papers/2607.28617

> 说明：Metis 配图取自其 arXiv HTML 版；**Frontis-MA1 无 HTML 版（图片式 PDF），配图为从 PDF 首页渲染并裁剪**（已在图注标明）。08/01–02 为周末空档，本份实际覆盖 08/03 新增 + 07/30–31 迟到补录。
