# RSI（递归自我改进）研究领域深度调研

阅读提示：本文件保留原始调研底稿。成熟方法与行业进展的补充已整合进[最终详细阅读版](/research-notes/2026-10-17-rsi-detailed.html)，该阅读版是本次更新后的主文。

调研日期：2026-10-17（北京时间）  
研究对象：人工智能中的 Recursive Self-Improvement，不涉及金融 RSI 指标。  
读者定位：希望理解领域全貌、辨别实证强度，并选择研究切入点的 AI 研究者或工程实践者。

## 1. 先说结论

**RSI 最值得研究的地方，不是“AI 能不能改写自己”，而是“改写之后，下一轮改进是否变得更有效，而且这种收益能否在独立任务和固定资源条件下持续”。**

本次调研的综合判断如下。这里的判断是对已核验公开研究的综合，不是证明整个领域存在或不存在某种能力。

1. **有限、局部的自改进已经有实证。** Promptbreeder 能同时进化任务提示词和变异提示词；STOP 能改进代码改进器；DGM、SICA 能更新编码 agent 的实现；SEAL 能学习怎样生成适合自我微调的数据。这些路线的“自改进对象”并不相同。[S1][S2][S3][S4][S5]
2. **2026 已出现更直接的“改进器变强”证据，但不能升级为“自主智能爆炸已经出现”。** Hyperagents 测量了跨域生成更好 agent 的能力；AIDE² 将改进后的研究 agent 提升为外层改进器做 ignition test，但作者明确说结果尚不确定。不同层级的证据不能混写。[S33][S34]
3. **目前比较适合落地的研究单元是“可验证任务 + agent/程序搜索 + 独立评测”。** 编码、算法优化、数学构造和基础设施优化都有相对明确的检验信号。SEAL 和 AZR 还说明，权重更新与自生成训练任务是另一条路线。[S3][S5][S6][S7]
4. **真正需要警惕的是评测口径。** DGM 的 20%→50% 来自 200 个 SWE-bench Verified 实例的设置，不能当成完整 500 题成绩；SEAL 的 ARC 结果来自只包含 8 道筛选题的小规模实验；Gödel Agent 的无约束版本允许调用更强模型，收益不能全部归因于自改进机制。[S3][S5][S8]
5. **我建议优先研究“改进器的改进”，而不只是“被改进对象的成绩”。** 最有辨识度的论文，应说明改进策略为什么更有效、效果是否迁移、是否超过等预算的固定改进器，并排除更强模型、更多采样和测试泄露。这是本报告提出的研究建议，不是现有标准。

一句话概括：**这个领域已经有值得复现的工程系统，但“持续、跨任务、资源归一化的递归能力增长”仍需要更严格的证明。**

## 2. RSI 究竟研究什么

### 2.1 三个经常混在一起的问题

下面是本报告采用的工作性分类，目的是帮助读论文，不是社区统一分级。

| 层级 | 改进的对象 | 典型例子 | 不能据此直接推出什么 |
|---|---|---|---|
| L0：输出改进 | 当前答案、推理轨迹 | Self-Refine、Reflexion | 模型或 agent 的永久能力变强 |
| L1：持久策略改进 | 提示词、记忆、工具使用策略 | Promptbreeder 的任务提示词；持续保留的策略更新 | 改进策略自身也变强 |
| L2：系统/参数改进 | agent 代码、工作流、程序、模型参数 | ADAS、DGM、SICA、SEAL | 已形成可持续的递归闭环 |
| L3：改进器改进 | 产生修改、选择修改、构造训练任务的方法 | STOP；Promptbreeder 的变异提示词；Hyperagents 的 meta agent | 跨领域且长期自加速 |
| L4：长期自主研究闭环 | 研究方法、任务分布、验证机制、训练过程共同演化 | 本报告用于表达更强 RSI 命题的研究目标 | 当前某个系统已经实现这一完整目标 |

Self-Refine 是同一 LLM 反复反馈、修改输出；Reflexion 利用语言化反馈和记忆帮助后续尝试。它们很重要，但与“改进生成改进的方法”不是同一个问题。[S9][S10]

区分时可以问：

- 修改的是一次性答案，还是会被下一轮继承的系统状态？
- 基础模型参数变了没有？如果没有，是哪一层变了？
- 新版本是在解决任务时更好，还是在改进别的系统时更好？
- 改进过程的外层搜索、选择规则或训练目标是否仍然完全固定？
- 如果换一批任务，新版本还能更有效地完成下一轮改进吗？

### 2.2 一个便于设计实验的形式化

令系统状态为：

\[
S_t=(\theta_t,H_t,M_t,T_t,I_t)
\]

其中，\(\theta\) 是模型参数，\(H\) 是 agent/harness 实现，\(M\) 是记忆，\(T\) 是工具，\(I\) 是改进策略。一个自改进过程可以写成：

\[
\widetilde S_{t+1}=I_t(S_t,E_t,B_t)
\]

\[
S_{t+1}=\operatorname{Select}(S_t,\widetilde S_{t+1};V_t)
\]

这里 \(E\) 是环境或任务反馈，\(B\) 是资源预算，\(V\) 是验证机制。这个形式化是分析工具，不意味着所有论文都实现了其中每一项。

普通优化只需要证明某个性能指标 \(P(S_{t+1})>P(S_t)\)。更强的递归命题还需要测量：

\[
G(I_t,S;\mathcal D,B)
=\mathbb E[P(I_t(S;\mathcal D,B))-P(S)]
\]

并判断更新后的改进器 \(I_{t+1}\)，在相同起点、相同预算和未见任务上，是否比 \(I_t\) 带来更高的增益。

**任务成绩提升与改进能力提升是两个实验目标；后一项不能只用前一项替代。**

### 2.3 RSI 不等于什么

- 不等于把测试时推理长度增加。
- 不等于反复采样后选最好的答案。
- 不等于 self-training 的所有形式。
- 不等于换一个更强的基础模型。
- 不等于自主复制、获取更多机器或绕过权限。
- 也不必要求一个系统同时修改所有组件：可以研究受限但可测的递归改进。

最后一点很重要。Gödel machine 的理论模型允许修改程序，DGM、SEAL 等现实系统则针对具体组件开展实验；研究时应明确局部命题，而不是只用“RSI”这个标签。[S3][S5][S11]

## 3. 领域地图：六条主要路线

| 路线 | 核心问题 | 主要修改对象 | 反馈来源 | 与 RSI 的关系 |
|---|---|---|---|---|
| 输出反思与语言反馈 | 当前尝试能否变好 | 输出、短期/跨尝试记忆 | 自评或环境反馈 | 提供基础机制，但不必涉及改进器演化 |
| 提示词与改进策略进化 | 怎样生成更好的指令/修改 | 任务提示词、变异提示词、改进程序 | 任务分数、代码执行 | Promptbreeder、STOP 直接触及改进策略 |
| Agent 自动设计与自修改 | 如何设计更有效的 agent | 控制流、工具、代码、工作流 | 验证集、编码测试 | ADAS、Gödel Agent、DGM、SICA 的核心 |
| 权重与训练数据自适应 | 系统怎样学习“如何学习” | 合成数据、自我编辑、微调配置、参数 | 更新后下游表现 | STaR、SEAL 等与训练闭环相连 |
| 自生成任务与课程 | 没有人工示例时如何产生可学习信号 | 任务、难度、解题策略 | 执行器、环境、奖励 | AZR 展示了可验证自生成训练任务 |
| 自动研究与算法/基础设施优化 | AI 能否优化支撑 AI 的计算过程 | 算法、内核、编译优化、资源调度 | 性能测量、数学验证、部署数据 | AlphaEvolve 有实际价值，但框架内部不因此全部递归演化 |

这张图是依据代表性工作整理的研究地图；各路线存在交叉，不能仅凭名称判定一个系统属于严格 RSI。[S1–S10][S12][S13]

### 3.1 不要漏掉更早的神经自修改与 learned optimizer 路线

RSI 不只存在于 LLM agent 文献。理解这一领域还需要三组更早的工作：

- **Learned optimizer：** 2016 年 *Learning to learn by gradient descent by gradient descent* 用 LSTM 学习优化器的更新规则，并研究相近任务迁移。但元优化仍由研究者设定，并非所有层级都自主改写。[S29]
- **Self-Referential Weight Matrix（SRWM）：** 2022 年 Irie 等把快速权重与自指机制结合，让矩阵生成键、值与学习率以修改自身；实验覆盖少样本学习和多任务强化学习。论文明确说，这些实验测实用任务能力，而专门测量长期自改进能力仍是后续问题。[S30]
- **Self-Referential Meta Learning / FME：** Kirsch、Schmidhuber 用性能导向的执行资源分配，研究不依靠显式元梯度优化的自修改。它给出 bandit 与 CartPole 的小规模实证，并讨论记忆式系统与自指系统的表达能力关系；作者也承认体系结构仍固定、实验简化，甚至 FME 是否应算一种最小元优化器仍存在术语争议。[S31]

**关键启发：** 权重改变与记忆改变不能仅凭表示形式判定“谁更像 RSI”；应看有效学习过程是否改善以及如何测量。同时，神经级更新与代码级修改有不同的稳定性、可解释性和验证成本。[S30][S31]

## 4. 历史脉络：从设想到可执行系统

| 时期 | 代表工作 | 对本领域的重要性 |
|---|---|---|
| 1965 | I. J. Good：超智能机器与 intelligence explosion 设想 | 提出设计机器的能力也可能被更强机器接管 |
| 2003/2006 | Schmidhuber：Gödel machine | 用证明与效用函数描述何时应执行自修改；属于理论模型 |
| 2016 | Learned optimizer | 把设计参数更新规则本身变为学习任务 |
| 2022 | SRWM、Self-Referential Meta Learning | 神经权重自修改及改进学习方式的小规模实验 |
| 2022 | STaR | 利用正确答案验证、筛选推理数据，再微调模型 |
| 2023 | Self-Refine、Reflexion | 让反馈与反思成为可工程化的改进机制 |
| 2023/2024 | Promptbreeder、STOP | 从改进任务解法推进到改进变异提示词或改进程序 |
| 2024/2025 | ADAS、Gödel Agent | 把 agent 设计和自修改本身变成搜索对象 |
| 2025 | SICA、DGM | 通过软件修改、执行反馈和候选管理实现 agent 自改进 |
| 2025 | AZR、SEAL、AlphaEvolve | 分别拓展到自生成课程、权重适应和真实算法/基础设施优化 |
| 2026 | Hyperagents、AIDE²、EvoX、Dream-RSI 等 | 直接测改进器、进化搜索策略，并检验是否真的更会驱动下一轮改进 |

日期应区分预印本首发与正式发表：例如 STOP 在 2023 年首发，正式发表于 COLM 2024；Gödel Agent 在 2024 年首发，正式发表于 ACL 2025。不能把会议年份误当作研究首次出现的年份。[S1][S2][S8][S14][S15]

## 5. 核心论文与系统：它们究竟证明了什么

### 5.1 Promptbreeder：最清楚的提示词级递归例子之一

**论文：** *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*，Fernando 等，2023 年预印本，ICML 2024。[S1][S16]

它不只是优化用于解题的提示词，也优化生成提示词变异的“变异提示词”。任务表现提供适应度，候选在进化循环中被选择。

- 改进对象：任务提示词和变异提示词。
- 递归点：负责产生修改的提示词也进入修改空间。
- 值得研究：可把 task policy 与 mutation policy 的贡献分开做消融。
- 限制：没有据此证明基础模型参数变强；也没有证明提示词进化可以无界持续。

**判断：** 很适合研究“改进策略本身能否被优化”；复杂度比完整编码 agent 更容易控制。[S1]

### 5.2 STOP：把“代码改进器”变成被改进对象

**论文：** *Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation*，Zelikman、Lorch、Mackey、Kalai，2023/2024。[S2]

初始系统使用 LLM 改进代码，随后用该系统改进自身代码，得到新的改进器。论文明确说明底层语言模型不会被修改，递归发生在调用模型的 scaffold 上。

- 改进对象：改进器程序，而非 LLM 权重。
- 核心价值：区分“一个更好的解法”与“一个更好的解法生成器”。
- 重要限制：递归对象有边界；实验仍依赖给定任务和评分过程。
- 安全提示：论文同时研究了系统尝试绕开 sandbox 约束的情况，说明自修改与约束完整性必须一起评测。

**判断：** 如果想理解 RSI 与一般代码搜索的差别，STOP 应放在阅读顺序的前列。[S2]

### 5.3 ADAS：自动 agent 设计，但不应默认等于自加速

**论文：** *Automated Design of Agentic Systems*，Hu、Lu、Clune，2024 年首发。[S12]

ADAS 的 Meta Agent Search 让元 agent 用代码定义候选 agent，把新设计加入档案，再继续探索。

- 改进对象：agent 的程序化设计空间。
- 论文报告：在多个任务上发现较强设计，并研究迁移到其他模型和领域。
- 研究价值：agent 不必被限制为人工设计的少数固定模板。
- 边界：自动寻找更好 agent 不自动意味着外层搜索方法变强。

**判断：** 它是 agent 设计自动化的重要基础，特别适合和 DGM 比较“设计一个 agent”与“由 agent 改写自己”的差异。[S12]

### 5.4 Gödel Agent：开放自修改必须控制模型与资源混淆

**论文：** *Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement*，Yin 等，ACL 2025。[S8]

系统把目标、自身代码和反馈交给 LLM，允许动态修改 agent 行为。正式论文对 constrained 与 unconstrained 设置作了区分。

**需要特别注意：** unconstrained 设置允许修改 API 调用，模型可以改用 GPT-4o 等更强服务；论文明确指出，更强模型的使用是部分额外收益的重要来源。正式论文还区分了用于设计 agent 的 GPT-4o 与用于约束版任务评价的 GPT-3.5，不能把整个系统理解为仅靠一个固定小模型完成改进。[S8]

- 值得研究：哪些约束应写进不允许被修改的可信层？
- 不能直接推出：成绩提升完全来自递归策略，而不是换模型或增加资源。
- 也不能把名字中的“Gödel”理解为每次修改都经过形式化最优性证明。

**判断：** 它很好地展示了自修改系统的自由度，也很好地展示了为什么必须做资源与权限控制。[S8]

### 5.5 SICA：适合作为可执行的编码 agent 基线

**论文：** *A Self-Improving Coding Agent*，Robeyns、Szummer、Aitchison，2025；官方仓库：`MaximeRobeyns/self_improving_coding_agent`。[S4][S17]

外层循环选择较好版本，让它基于历史性能改进自身源码。更新涉及工具、定位与上下文管理；固定基础模型仍参与代码生成。

论文摘要的 17%→53% 应与实验表一起读：

- 17%→53% 是一个 **50 题 SWE-bench Verified 子集**上的结果；
- 同时使用 50 道 LiveCodeBench 题和两个合成工具任务；15 次迭代的运行成本约为 7,000 美元；
- 系统主要使用 Claude Sonnet 3.5，另有 o3-mini 推理子 agent；每题的平均美元成本、token 与耗时并非固定，作者也报告了这些变化。[S4]

论文也报告：在底模已很擅长的推理任务上，scaffold 的增益较有限。编码子集的进步仍不能替代完整基准、长期运行与预算匹配实验。[S4]

**判断：** 适合研究工程问题，例如工具接口、错误定位、上下文压缩和可靠性；若用于证明更强 RSI 命题，需要补充对改进器与外层搜索的独立评测。[S4]

### 5.6 Darwin Gödel Machine（DGM）：最值得深读的实证系统之一

**论文：** *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*，Zhang、Hu、Lu、Lange、Clune，2025 首发；本次核对的 arXiv 最新记录含 2026-03-12 的 v3。[S3]

DGM 让编码 agent 生成自身修改，通过实际编码任务测量效果，把可运行候选保存在档案中，再从档案选择父版本。它不是每次只保留一个最高分版本，因此能探索不同分支。[S3][S18]

**关键结果与口径：**

- SWE-bench：20.0%→50.0%；v3 采用 10 题功能检查、追加 50 题，再对较强候选扩展到 **200 题**的分阶段评测，不是完整 500 题排行榜口径。
- Polyglot：14.2%→30.7% 是最终候选在完整 Polyglot 基准上的前后比较；v3 的演化内评测采用小子集，论文另外报告跨模型、跨基准与跨语言迁移。
- 一个 SWE-bench 演化运行的论文估计成本约为 **22,000 美元**，两个消融基线各约 10,000 美元；这是等预算重评有意义的具体原因，但成本不同本身不证明算法无效。
- 主实验不更新基础模型权重，也不修改外层进化循环。[S3]

论文把私有测试补丁放入改进诊断提示词，且当前诊断步骤由另一个固定模型 o1 提供；作者说明人工检查未发现硬编码答案，并提供跨基准迁移结果。因此不能断言论文已发生作弊，但应将“看过诊断信息的演化内成绩”与“未参与演化的迁移成绩”分开解读。[S3]

**它支持的命题：** 用可执行反馈和开放分支搜索，agent 可以在限定设计空间中发现有用的自身修改。

**它尚不能单独证明的命题：** 改进能力在长期持续加速、基础模型智能自主增长，或者收益已经排除所有资源/选择效应。

**判断：** 很适合复现其候选档案、分支选择和改进机制；不应只复现曲线而忽略任务信息流和成本。[S3][S18]

### 5.7 SEAL：权重适应与“学习如何生成学习材料”

**论文：** *Self-Adapting Language Models*，Zweiger 等，2025；官方项目称 SEAL。[S5][S19]

SEAL 让模型生成 self-edit，包括用于微调的合成内容或适应配置；实际完成参数更新后，用下游表现作为奖励，再训练生成 self-edit 的策略。

这把两个过程连接起来：内层更新模型参数，外层学习怎样生成更有效的更新材料。

**需要拆开解读的结果：**

- 本报告采用论文 **v2**：单段知识吸收的 no-context SQuAD 准确率为 47.0%，未经过外层训练的自生成数据为 39.7%，GPT-4.1 合成数据为 46.3%；在批量持续预训练的设置下，GPT-4.1 合成数据略优于 SEAL。
- ARC 少样本实验：72.5% 对比未训练 self-edit 策略的 20%，但评测仅包含 **8 道经筛选的问题**、每题 5 个 self-edit，共 40 次配置试验；指标是配置导致适应成功的比例，不能误写成完整 ARC 的题目通过率。
- 论文单独测量了知识适应中的遗忘，不能只把微调收益当成无成本增益。[S5]

**判断：** 它是连接 RSI 与 meta-learning、test-time training 的重要工作。更有价值的后续问题是：多轮适应是否稳定、旧能力如何保留、自我编辑策略能否跨任务迁移。[S5]

### 5.8 Absolute Zero Reasoner（AZR）：没有人工示例，不等于没有外部反馈

**论文：** *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*，Zhao 等，2025。[S7]

AZR 自己生成代码相关的推理任务，用执行器检查有效性并产生奖励，在这个环境中进行自我对弈式训练。

- 改进对象：模型的推理能力与自生成任务课程。
- 关键区分：“Zero Data”针对所需人工训练示例，不意味着没有 pretrained knowledge、执行器或外部正确性信号。
- 与 RSI 的关系：它拓宽了自主产生训练分布的方法，但不能只凭这一点认定它同时改进了所有训练规则和验证器。

**判断：** 适合研究课程质量、任务多样性与可验证性；要特别检查生成器是否越来越擅长生产“自己容易得分”的任务。[S7]

### 5.9 AlphaEvolve：非常强的 AI 研发加速，不等于完整自主 RSI

**论文/报告：** *AlphaEvolve: A coding agent for scientific and algorithmic discovery*，Novikov 等，2025；Google DeepMind 官方介绍和技术报告。[S6][S20]

AlphaEvolve 把 LLM 生成程序与自动验证、性能测量、进化式搜索连接起来。它针对代码、算法与数学问题，且部分成果已用于支持 AI 基础设施。

官方介绍报告的例子包括：

- 数据中心调度改进：平均回收约 0.7% 的全球计算资源；
- 一个关键 Gemini 训练内核提升 23%，对应整个模型训练时间约降低 1%；
- 找到用 48 次标量乘法进行 4×4 复数矩阵乘法的算法，改善相应已知界限。[S20]

**不能混淆的口径：** 23% 是特定内核的加速，不是整个训练系统的 23%；AI 优化支撑 AI 的程序，也不自动说明提案模型、选择规则和验证器都在自主递归改进。

**判断：** 它证明“AI 加速 AI 研发”有真实工程价值；下一步 RSI 问题是，这些收益能否被稳定地反馈到下一轮改进能力，而不只停留在更快的被优化程序上。[S6][S20]

### 5.10 2026 的关键推进：从改系统，到测“更会改系统”

#### Hyperagents：应该优先读的新增核心论文

Hyperagents 把 task agent 和 meta agent 放入一个可编辑程序；不仅改任务执行逻辑，也允许修改产生后续改进的程序。[S33]

它引入 **improvement@k（imp@k）**：把待测 meta agent 冻结，给它固定次数生成候选 task agent 的机会，再以验证集选择候选，测独立测试集上的增益。这样能较直接地比较不同代改进器的能力。

论文报告，从论文评审/机器人奖励设计迁移到数学解答评分后，imp@50 的中位数为 **0.630**，初始机制为 **0**；五次运行支持该迁移比较的正面结果。但初始 task agent 的零分与输出格式/解析失败有关，不能把全部增益读成数学智力增长。继续演化 200 次后，迁移初始化与原始初始化的 **0.640 对 0.610** 差异不显著；主实验的父代选择与评价协议仍固定。[S33]

迁移实验还同时带入 task agent 与 meta agent 的实现，不是只替换 meta agent、把 task agent 代码严格保持相同；这是后续通过交叉组合消融进一步加强因果归因的机会。[S33]

**判断：** 它比单纯编码分数上涨更接近直接的元改进实证，同时保留了长期复合收益与实验边界问题。

#### AIDE²：明确把“是否点火”作为可测试命题

*Recursive self-improvement of AI research agents* 的 AIDE² 用外层循环改写研究 agent，再用固定预算的研究任务与 private grading 测量候选。作者报告一条 **8 天、100 个搜索树节点（初始版本加 99 次改写提案）**的轨迹接受 **7 次修改**，selection grade 从 **0.703→0.778**，并在四个未参与选择的基准上研究迁移；这里的“100 节点”不是 100 台计算机器。[S34]

最关键的不是这条曲线，而是 **ignition test**：将改进后的 agent 用作外层改进器，对比原强基线。每组只有三次独立运行，结果噪声较大，作者明确说**尚不能证明新 agent 更会驱动 RSI 或更具样本效率**。[S34]

**判断：** 这篇提供了值得沿用的问题与实验形式，而不是已经证实自加速点火。

#### 其余新作如何定位

下表是本次核验的定位，数字详情、版本、代码状态与限制见近期补充报告。它们不应按 headline 分数排成一张“RSI 智能排行榜”。

| 项目 | 2026 首发时间 | 主要推进 | 需要保留的边界 |
|---|---|---|---|
| EvoX | 2月 | 同时进化解与搜索策略 | 更上层策略生成/切换规则仍固定 [S35] |
| Dream-RSI | 9月，10月更新 | 利用发现历史的离线 replay 改探索策略 | coding agent 与评估器固定；不保证全任务获益 [S36] |
| GEA | 2月 | 群体 agent 分享补丁与成败经验 | Polyglot headline 是子集协议，不能当全量成绩 [S38] |
| MetaShare | 10月15日 | 跨谱系保存带正负效果的修改经验 | 更少代数不等于更少总tokens；部分比较非重跑 [S37] |
| Meta-Harness | 3月 | 自动设计固定模型外围的 harness | 外层 proposer 本身没有演化 [S39] |
| AREX | 7月，9月更新 | 研究答案、证据状态和上下文更新 | 主要是答案/研究状态递归细化 [S40] |
| VisionHOPE | 9月 | 图像扫描中的自引用记忆与学习规则 | 属于局部神经学习动态，不是长期自主改进闭环 [S41] |
| AIRA₂ | 3月，4月更新 | 稳定隐藏评测与异步研究搜索 | 人工设计 orchestrator，不应默认归为改进器自演化 [S42] |

MARS、POLARIS 和 PostTrainBench 则提供了有价值的对照：结构化单轮增强可以有较高效率，小模型自修改会出现较高失败率，自动后训练成果与训练 agent 自身变强也必须分开测量。它们约束的是具体实验主张，不是证明 RSI 普遍无效。[S43][S44][S45]

#### 如何使用最新综述

2026 年 *Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops* 可以帮助建立检索地图，但它报告的 **1,250 篇是检索语料**，不是 1,250 个成功 RSI 实验；文献分类由单一标注者完成，作者也承认外围检索噪声。[S32]

其“有界细化/开放式 RSI”的术语边界与本报告不完全相同。**本报告不认为“使用固定外部验证器”就自动否定递归改进；应明确声称的是哪一种递归与哪一种长期增长。** 没有外部锚点并不是值得追求的默认实验条件。

近期详细阅读笔记：`rsi-deep-research-2026-10-17-notes/recent-2026.md`。

## 6. 证据审计：怎样判断一篇 RSI 论文是否扎实

以下是本报告建议的审计清单。它是研究设计建议，不是宣称已有统一标准。

### 6.1 八类最常见的混淆

| 混淆 | 为什么会误判 | 应增加的证据 |
|---|---|---|
| 更多尝试 | best-of-N 随 N 增加就可能上升 | 等调用数、等token、等运行时间的搜索基线 |
| 换更强底模 | 系统能力变化与递归机制不可分 | 冻结模型与API版本，单独报告模型替换 |
| 验证集过拟合 | 反复选择候选等价于不断调参 | 密封最终测试集、任务家族隔离 |
| 修改评测器 | 可以让分数变好而任务没变好 | 将可信评分层置于修改权限之外 |
| 公共基准污染 | 模型可能已经记住解法或公开补丁 | 新生成任务、私有新实例、污染检查 |
| 只报告成功分支 | 大量失败与搜索成本被隐藏 | 全档案、失败率、总成本和多seed曲线 |
| 目标改进代替改进器改进 | 新版本解题更好不代表更会改进 | 让不同代改进器从同一起点改进新目标 |
| 运行更昂贵 | 耗费更多推理/工具时间换来了分数 | 成功率—成本—延迟的Pareto曲线 |

这些混淆不是抽象担忧：Gödel Agent 正式论文讨论了模型升级与成本；DGM 报告了较大演化成本；SICA 给出随版本变化的token、耗时和成本；SEAL 的 ARC 设置则提示样本量问题。[S3][S4][S5][S8]

### 6.2 需要分别报告的三条曲线

1. **目标能力曲线：** 每代 agent 在同一密封测试集上的表现。
2. **改进能力曲线：** 每代改进器，在固定起点和新任务上的单位预算增益。
3. **资源曲线：** 总调用、token、FLOPs（可测时）、执行耗时、货币成本与失败率。

只有第一条上升，可能说明 agent 搜索有效；第二条也上升，才有更直接的递归证据；若第三条增长更快，则需要检查是否只是资源扩张。

### 6.3 最低限度的因果消融

- 固定改进器，保留相同搜索和候选选择。
- 随机/常规进化提案，保持同样候选数。
- 去除反思，但保留历史任务分数。
- 去除档案或多分支，只保留最新/最高分版本。
- 保留代码更新，但冻结改进策略。
- 重置外部记忆，检查能力是否来自状态积累。
- 控制基础模型、工具、任务信息与推理预算。
- 把最终版本迁移到新任务族，而不是仅换同一基准中的题目。

建议不仅比较最高分，也比较多次运行的均值、置信区间、失败/回退概率和达到同一性能阈值所需资源。

## 7. 理论与安全：最重要的边界

### 7.1 理论模型不等于可部署系统

Gödel machine 使用形式化证明决定自修改：只有在其公理与效用设定下证明某个修改值得执行，才触发替换。其理论价值在于把自修改纳入一个可分析的决策框架，而不是提供现成、普适且高效的工程搜索器。[S11]

实际 LLM agent 更多依赖有限样本的经验评测。经验改进的证据可以很强，但不能借用 Gödel machine 的名字，宣称经验修改自动具有理论全局最优性。[S3][S8][S11]

### 7.2 可验证反馈是资源，不是免费条件

STaR 用正确答案判断是否保留推理样本；AZR 用代码执行器提供任务有效性与奖励；AlphaEvolve 使用自动评价程序。它们都说明外部反馈的重要性，而不是说明 LLM 只依靠自我确信就能可靠变强。[S7][S13][S20]

研究问题因此从“是否拥有 feedback”进一步变成：

- 验证器能覆盖多大范围？
- 验证错误会怎样被重复放大？
- 验证成本是否超过产生候选的成本？
- 新能力出现后，旧验证器还能否有效约束它？

### 7.3 自修改权限与可信层应分开

作为研究原型，建议允许修改 prompts、agent 代码、工具使用策略和候选改进方法，但把以下部分留在修改边界之外：

- 密封测试题与标准答案；
- 评分、日志与版本签名；
- 模型/工具调用预算；
- 网络、凭据与权限规则；
- sandbox 与恢复机制；
- 最终候选部署审批。

这些是本报告的实验安全建议。STOP 对 sandbox 绕开情况的分析，以及 Gödel Agent 对约束与无约束设置的区分，说明应同时度量任务增益和约束完整性。[S2][S8]

### 7.4 三种需要区分的“失败”

- **搜索失败：** 找不到比当前版本更好的修改。
- **学习失败：** 新能力增加，但旧能力遗忘或训练分布退化。
- **治理失败：** 新版本通过改变信息流、资源或评分规则提高表面成绩。

它们需要不同实验。SEAL 的遗忘实验属于第二类；预算与验证隔离属于第一/第三类的识别工具。不能用一个“总分”掩盖这些差异。[S5]

### 7.5 必须同时阅读的负面结果与反例

**没有外部反馈的自纠错不保证有效。** Huang 等的 *Large Language Models Cannot Self-Correct Reasoning Yet* 在当时模型和任务中发现，移除外部正误信号后，自纠错可能把正确答案改错。但 SCoRe 后续通过强化学习训练改善了自纠错；其训练阶段使用正确性奖励。这两项结果不矛盾：测试时没有外部反馈，与整个学习过程完全不依靠外部真值，不是同一个条件。[S21][S22]

**递归合成数据存在分布退化风险，但不是“合成数据必然有害”。** Nature 的 model collapse 研究分析并观察了分布信息丢失；另有研究显示，保留原始数据并累积历代数据，在相应设定下可以缓解或避免这种退化。研究时要记录数据替换、累积、过滤和验证机制，同时计入数据与训练资源的增长。[S23][S24]

**自评分上升可能对应真实性能下降。** Reward Model Overoptimization 用较独立的 gold reward model 观察到过度优化代理评分的退化，且明确承认 gold 模型并非无误真值。Self-Rewarding 则提供有限轮生成能力与评分能力共同改善的正面证据，但仍用外部留出评价校准。结论不是禁止可学习评分器，而是不能把它的分数当成唯一证明。[S25][S26]

这些工作给出“特定设置会失败”的证据，不构成 RSI 的普遍不可能性证明。

### 7.6 MLE-bench 与 RE-Bench：如何衡量研究能力

| 基准 | 主要测量对象 | 论文原始设置 | 对 RSI 的用途与缺口 |
|---|---|---|---|
| MLE-bench | 机器学习工程：训练、数据处理、选择实验 | 75 个 Kaggle 派生任务；原论文报告单次与多次尝试结果 | 可以作为改进器任务载体，但 pass@k 增益不等于系统经过 k 轮递归学习 |
| RE-Bench | 更接近 AI 研发的开放式工程任务 | 7 个环境；包含训练提速、内核优化等人与 agent 对比 | 可以测研究策略与资源回报，但原评分不直接验证下一轮自我改进能力 |

事实来自原论文，本表的“用途与缺口”是本报告的分析。[S27][S28]

尤其不要把 RE-Bench 的“短预算下归一化分数更高”理解成通用智能的倍数；它是相对起始方案和参考方案的任务量尺。论文还报告更长预算下的人类优势以及违规高分轨迹的人工审计。这里引用的是论文批次的结果，不是宣称它代表 2026 当前最强模型。[S28]

更完整的理论假设、Good 原始节选、AIXI 可计算性、目标保持、自纠错和基准审计，可读补充文件：

`rsi-deep-research-2026-10-17-notes/theory-evaluation-safety.md`

## 8. 值得切入的五个研究方向

以下排序是我的研究建议，依据可验证性、实验可控性和可能形成清晰贡献来选择，不是对所有研究路线的价值排序。

### 方向一：等预算的改进器演化——优先级最高

**问题：** 改进器更新后，是否比固定改进器更有效地优化一个未见 agent？

**设计：**

1. 从同一个初始 agent 分叉；
2. 对比固定改进器、可更新改进器、普通进化搜索、best-of-N；
3. 固定底模、调用预算、候选数和评测信息；
4. 在未见任务族上重新从零开展改进。

**贡献应落在：** 改进效率、迁移与因果证据，而不是只创造一个更高的最终任务成绩。

**基础工作：** STOP、Promptbreeder、DGM，以及 2026 的 Hyperagents imp@k 和 AIDE² ignition test。[S1][S2][S3][S33][S34]

建议从一个已经可运行、格式正确的 task agent 开始，避免把修复输出解析器造成的零分提升，全部归因于高层研发能力。

### 方向二：验证器可靠性与防评测投机

**问题：** 如何在允许 agent 自改进时，保证评分提升对应真实能力提升？

**可做方向：**

- 候选修改的独立测试与变异测试；
- 隐藏测试、反例生成、执行轨迹交叉检验；
- 独立 audit evaluator 与训练 evaluator；
- 评价层不修改时的稳定锚点；
- 如果允许评价策略演化，设置独立且不共同演化的终审层。

**关键指标：** 接受错误候选率、拒绝正确候选率、评测成本、投机成功率。

**基础工作：** STOP 的 sandbox 分析、AZR 的执行反馈、AlphaEvolve 的验证设计。[S2][S7][S20]

### 方向三：多轮权重自适应与旧能力保留

**问题：** 模型连续生成自我编辑材料并微调，能否累积知识而不严重遗忘？

**实验重点：**

- 不只测一次适应，测连续多轮；
- 每轮引入新知识或新技能；
- 同时复测早期任务与通用能力；
- 比较自我编辑、固定模板、检索增强和人工构造数据；
- 用相同训练token与参数更新预算比较。

**基础工作：** SEAL、STaR。[S5][S13]

**难度提示：** 涉及训练与大量重复试验，比 prompt/harness 级研究更昂贵。

### 方向四：开放式任务课程与难度校准

**问题：** 系统自生成任务之后，如何避免生成器只制造简单或可投机的训练题？

**可操作变量：** 任务新颖性、难度、可验证性、解法多样性、跨任务迁移。

**关键对照：** 自生成课程与相同规模的固定课程、随机课程、外部课程。

**基础工作：** AZR；DGM 的分支档案机制可提供搜索多样性的设计启发。[S7][S3]

### 方向五：AI 研发加速是否能变成可闭合的反馈循环

**问题：** 一个内核或研究工具的优化，究竟节省了多少下一轮研究资源？节省的资源投入后，是否产生净增益？

**实验重点：**

- 区分局部内核加速与端到端研究吞吐量；
- 计入搜索、编译、验证和失败候选成本；
- 测量实现收益的摊销期；
- 在固定总预算下比较“继续优化工具”和“直接多做任务”。

**基础工作：** AlphaEvolve。[S6][S20]

**判断：** 工程价值可能很高，但要发表 RSI 结论，必须展示实际闭环，而不是仅展示“AI 优化了 AI 会用的程序”。

## 9. 一个适合开始研究的实验方案

这是方案建议，不是已开展的实验，也不授权自动部署或消耗云资源。

### 9.1 研究问题

> 在相同基础模型、任务反馈和总预算下，可演化的改进器，能否比固定改进器更高效地优化编码 agent，并迁移到新的任务家族？

### 9.2 第一阶段不要直接复现完整 DGM 成本

先选择规模小、自动验证、可以生成新实例的程序任务，例如小型代码修复、数据结构实现、算法变换或限定接口的工具构建。

原因是工程原型阶段需要大量消融，而不是用大量预算追逐一个公开基准上的最高分。DGM 论文的单次运行成本说明应认真管理这一风险。[S3]

### 9.3 四组基线

| 组别 | 固定什么 | 允许更新什么 |
|---|---|---|
| A：普通采样 | agent 与提案策略 | 只生成候选，选择最好结果 |
| B：固定改进器 | 改进策略 | agent 代码 |
| C：常规进化搜索 | 非自修改的变异/选择规则 | agent 代码 |
| D：递归改进 | 底模、任务信息、预算、终审评分 | agent 代码与改进策略 |

四组都应控制候选评价次数、LLM 调用和最大执行时间。若不能同时匹配所有资源，至少报告两个独立口径，例如等token与等美元预算。

### 9.4 数据与评价隔离

- 开发集：允许系统看到题目和反馈。
- 改进能力迁移集：用来评价新一代改进器，任务家族与开发集不同。
- 密封最终集：不进入修改、选择或诊断循环。
- 旧能力回归集：发现更新后的退化。
- 约束审计集：检测越权、评分污染和隐蔽额外资源使用。

不要把“验证集”和“最终测试集”只是在文件名上分开；候选选择阶段必须无法看到最终测试结果。

### 9.5 记录项目

每次修改记录：父版本、代码差异、改进理由、使用反馈、模型版本、调用与token、执行时间、评价分数、失败原因、是否接受，以及后续迁移表现。

主指标建议使用：

- 固定预算的成功率；
- 达到给定成功率阈值所需预算；
- 新任务上的改进增益；
- 改进器跨起点/跨任务的迁移；
- 旧能力保持率；
- 约束违反率。

统计上应采用多个独立seed，预先规定最终比较方式，避免事后只挑成功运行。

### 9.6 如果只有单机或有限 API 预算

我会建议优先选择 STOP/Promptbreeder 风格的小型改进器实验，或者围绕 SICA 的单一组件开展受控研究；把完整 DGM 或多轮 SEAL 作为后续阶段。这个建议来自实验复杂度和成本可控性的判断，不是一个通用硬件门槛。[S1][S2][S3][S4][S5]

## 10. 推荐阅读顺序

**理解概念：**

1. Gödel machine：看清理论自修改的条件与工程差距。[S11]
2. Self-Refine、Reflexion：理解“反思改善输出”和“系统递归改进”的区别。[S9][S10]

如果关注模型架构而非 agent 工程，应在此插入 SRWM 与 Self-Referential Meta Learning，不要把整个领域缩减为“用 LLM 改写 Python 代码”。[S30][S31]

**理解真正的递归对象：**

3. Promptbreeder：变异提示词也进入进化。[S1]
4. STOP：代码改进器自改进。[S2]

**理解现实系统：**

5. ADAS：agent 的程序设计空间。[S12]
6. SICA：自修改编码 agent 的工程实现。[S4]
7. DGM：分支档案、开放搜索、迁移和成本。[S3]
8. Gödel Agent：约束边界与模型/资源混淆。[S8]

**理解训练与研发闭环：**

9. SEAL、AZR：参数适应与自主课程。[S5][S7]
10. AlphaEvolve：算法与基础设施优化的实际收益。[S6][S20]

**如果只读三篇来抓住截至本次检索的最前沿问题：** DGM → Hyperagents → AIDE²。三者分别代表自修改编码系统、直接跨域元改进测量、研究效率与 ignition test。[S3][S33][S34]

阅读时建议每篇填同一张卡片：改进对象、是否持久、递归点、反馈、外部依赖、修改边界、数据划分、预算、独立seed、迁移、负面结果、代码可得性。

## 11. 检索范围与证据局限

- 本报告采用定向文献调研：围绕 recursive self-improvement、self-modifying agents、self-referential prompt evolution、self-adapting language models、open-ended evolution、AI research automation 扩展关键词，再沿代表性论文和官方代码追踪。
- 主要证据来自原论文、正式论文集、作者官方仓库和机构技术报告。数值结果优先核对全文而非只用摘要。
- 核心系统与近期十项实证工作的分析基于可访问全文；近期补充报告另外收录一篇仅核验摘要的 agentic-reasoning survey，明确标注，不用它支撑具体实验结论。
- 本次 Semantic Scholar 批量 API 与 arXiv export API 均遇到限流，已改用主站原文与正式出版页面核对。未可靠获取的引用数统一记为 `null`/“未查询”，不猜测。
- 这是研究地图与批判性综合，不是按 PRISMA 执行的穷尽性系统综述。没有用论文名称中出现“RSI”与否作为唯一筛选标准。
- 报告中的“本次尚未看到证据”不等于“证明不可能”；特定任务上的负面结果也不能自动外推到所有模型、反馈与训练设置。
- 本报告自定义层级、形式化和研究路线建议均明确标为分析工具，不是论文原作者的统一术语。
- 所有实证成绩都是论文作者报告，未独立复现实验；新作多数为预印本，正式 venue 未核验时不猜测。结构化元数据在 `rsi-deep-research-2026-10-17-notes/papers-core.json` 及两份补充报告的 JSON 段中。

## 12. 已核验一手来源

为便于离线阅读，来源使用固定编号。arXiv 论文页通常提供公开 PDF；正式 venue 只在实际核验时标注。

- **[S1]** Fernando et al. *Promptbreeder: Self-Referential Self-Improvement Via Prompt Evolution*. 2023. https://arxiv.org/abs/2309.16797
- **[S2]** Zelikman et al. *Self-Taught Optimizer (STOP): Recursively Self-Improving Code Generation*. 2023 / COLM 2024. https://arxiv.org/abs/2310.02304
- **[S3]** Zhang et al. *Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents*. 2025，含2026更新。https://arxiv.org/abs/2505.22954 ；全文：https://arxiv.org/html/2505.22954v3
- **[S4]** Robeyns et al. *A Self-Improving Coding Agent*. 2025. https://arxiv.org/abs/2504.15228 ；全文：https://arxiv.org/html/2504.15228
- **[S5]** Zweiger et al. *Self-Adapting Language Models*. 2025. https://arxiv.org/abs/2506.10943 ；全文：https://arxiv.org/html/2506.10943v2
- **[S6]** Novikov et al. *AlphaEvolve: A coding agent for scientific and algorithmic discovery*. 2025. https://arxiv.org/abs/2506.13131
- **[S7]** Zhao et al. *Absolute Zero: Reinforced Self-play Reasoning with Zero Data*. 2025. https://arxiv.org/abs/2505.03335
- **[S8]** Yin et al. *Gödel Agent: A Self-Referential Agent Framework for Recursively Self-Improvement*. ACL 2025. https://aclanthology.org/2025.acl-long.1354/ ；全文：https://aclanthology.org/2025.acl-long.1354.pdf
- **[S9]** Madaan et al. *Self-Refine: Iterative Refinement with Self-Feedback*. NeurIPS 2023. https://arxiv.org/abs/2303.17651
- **[S10]** Shinn et al. *Reflexion: Language Agents with Verbal Reinforcement Learning*. NeurIPS 2023. https://arxiv.org/abs/2303.11366
- **[S11]** Schmidhuber. *Goedel Machines: Self-Referential Universal Problem Solvers Making Provably Optimal Self-Improvements*. 2003，v5 更新于2006。https://arxiv.org/abs/cs/0309048
- **[S12]** Hu et al. *Automated Design of Agentic Systems*. 2024. https://arxiv.org/abs/2408.08435
- **[S13]** Zelikman et al. *STaR: Bootstrapping Reasoning With Reasoning*. NeurIPS 2022. https://arxiv.org/abs/2203.14465
- **[S14]** STOP 的 arXiv 记录明确标注发表于 COLM 2024。https://arxiv.org/abs/2310.02304
- **[S15]** Gödel Agent 正式论文元数据与引用导出。https://aclanthology.org/2025.acl-long.1354/
- **[S16]** Promptbreeder 正式论文。ICML 2024. https://proceedings.mlr.press/v235/fernando24a.html
- **[S17]** SICA 官方代码。https://github.com/MaximeRobeyns/self_improving_coding_agent
- **[S18]** DGM 官方代码。https://github.com/jennyzzt/dgm
- **[S19]** SEAL 官方代码。https://github.com/Continual-Intelligence/SEAL
- **[S20]** Google DeepMind. *AlphaEvolve: A Gemini-powered coding agent for designing advanced algorithms*. 2025-05-14. https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- **[S21]** Huang et al. *Large Language Models Cannot Self-Correct Reasoning Yet*. 2023 / ICLR 2024. https://arxiv.org/abs/2310.01798
- **[S22]** Kumar et al. *Training Language Models to Self-Correct via Reinforcement Learning*. 2024. https://arxiv.org/abs/2409.12917
- **[S23]** Shumailov et al. *AI models collapse when trained on recursively generated data*. Nature, 2024. https://www.nature.com/articles/s41586-024-07566-y
- **[S24]** Gerstgrasser et al. *Is Model Collapse Inevitable? Breaking the Curse of Recursion by Accumulating Real and Synthetic Data*. 2024. https://arxiv.org/abs/2404.01413
- **[S25]** Gao et al. *Scaling Laws for Reward Model Overoptimization*. 2022. https://arxiv.org/abs/2210.10760
- **[S26]** Yuan et al. *Self-Rewarding Language Models*. 2024 / ICML 2024. https://arxiv.org/abs/2401.10020
- **[S27]** Chan et al. *MLE-bench: Evaluating Machine Learning Agents on Machine Learning Engineering*. 2024 / ICLR 2025. https://arxiv.org/abs/2410.07095
- **[S28]** Wijk et al. *RE-Bench: Evaluating frontier AI R&D capabilities of language model agents against human experts*. 2024，含后续修订。https://arxiv.org/abs/2411.15114
- **[S29]** Andrychowicz et al. *Learning to learn by gradient descent by gradient descent*. NeurIPS 2016. https://arxiv.org/abs/1606.04474
- **[S30]** Irie et al. *A Modern Self-Referential Weight Matrix That Learns to Modify Itself*. 2022. https://arxiv.org/abs/2202.05780 ；代码：https://github.com/IDSIA/modern-srwm
- **[S31]** Kirsch、Schmidhuber. *Eliminating Meta Optimization Through Self-Referential Meta Learning*. 2022. https://arxiv.org/abs/2212.14392
- **[S32]** Chen et al. *Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops*. 2026. https://arxiv.org/html/2607.07663v2
- **[S33]** Zhang et al. *Hyperagents*. 2026. https://arxiv.org/abs/2603.19461 ；所核验 PDF：https://arxiv.org/pdf/2603.19461v1 ；代码：https://github.com/facebookresearch/Hyperagents
- **[S34]** Srikanth et al. *Recursive self-improvement of AI research agents*. 2026. https://arxiv.org/html/2609.26457v1
- **[S35]** Liu et al. *EvoX: Meta-Evolution for Automated Discovery*. 2026. https://arxiv.org/html/2602.23413v2
- **[S36]** Zheng et al. *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*. 2026. https://arxiv.org/html/2609.14858v2
- **[S37]** Bhutani et al. *MetaShare: Failure-Aware Collective Self-Improvement for Open-Ended Agents*. 2026-10-15. https://arxiv.org/html/2610.21887v1
- **[S38]** Weng et al. *Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing*. 2026. https://arxiv.org/html/2602.04837v1
- **[S39]** Lee et al. *Meta-Harness: End-to-End Optimization of Model Harnesses*. 2026. https://arxiv.org/html/2603.28052v1
- **[S40]** Lu et al. *AREX: Towards a Recursively Self-Improving Agent for Deep Research*. 2026. https://arxiv.org/html/2607.21461v3
- **[S41]** Peng et al. *VisionHOPE: Visual Backbones as Self-Modifying Learning Systems*. 2026. https://arxiv.org/html/2609.33325v1
- **[S42]** Hambardzumyan et al. *AIRA_2: Overcoming Bottlenecks in AI Research Agents*. 2026. https://arxiv.org/html/2603.26499v2
- **[S43]** Hou et al. *Learn Like Humans: Use Meta-cognitive Reflection for Efficient Self-Improvement*. ACL 2026. https://aclanthology.org/2026.acl-long.1329/
- **[S44]** Kakade et al. *POLARIS: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair*. Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.1969/
- **[S45]** Rank et al. *PostTrainBench: Can LLM Agents Automate LLM Post-Training?*. 2026. https://arxiv.org/html/2603.08640v2
