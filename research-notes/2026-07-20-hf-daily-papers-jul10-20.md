# HF Daily Papers 中文摘要：2026-07-10 ~ 07-20

- **Date:** 2026-07-20
- **Tags:** #hf-daily-papers #digest #agent #multimodal #video #rl #distillation #long-horizon

## Context

- **覆盖范围**：2026-07-10 ~ 07-20（上一份 digest 覆盖至 07-09）。
- **数据获取**：逐日调用 HF `daily_papers` API（`limit=100&sort=publishedAt`）。07/11–12、07/18–19 为周末投稿空档（0 篇）。
- **本周体量**：合计 **141 篇唯一论文**，与上一份 digest **无重叠**。按 upvotes 降序取 **Top 25 精选**。
- **主线信号**：本周最强信号是 **Agent / Harness（42 篇）** 与 **视频/多模态（38 篇）** 双主线；**RL/蒸馏/自进化（16 篇）**为第三主线。榜首两篇——Harness Handbook（201）与 LongStraw（180）——都指向"**长程 agent 的系统工程**"这条大主线。
- **约束**：所有引用为真实 HF/arXiv link，全部 141 篇已入 `references.bib`（`add_paper.py` 核验，个别未上 arXiv 索引的除外）。

> **联动**：本周 #2 的 **LongStraw** 我已单独写过深读 [`2026-07-20-longstraw-longcontext-rl.md`](/research-notes/2026-07-20-longstraw-longcontext-rl.md)，本文不重复深挖、只在趋势里点名；#1 Harness Handbook 与本项目 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 的 Pillar I（harness）直接呼应。

## 论文总览表（Top 25 by upvotes）

| # | arXiv | 标题 | ▲ |
|---|---|---|---|
| 1 | [2607.13285](https://huggingface.co/papers/2607.13285) | Harness Handbook：让演化中的 agent harness 可读/可导航/可编辑 | 201 |
| 2 | [2607.14952](https://huggingface.co/papers/2607.14952) | LongStraw：固定 GPU 预算下突破 2M token 的长上下文 RL | 180 |
| 3 | [2607.14935](https://huggingface.co/papers/2607.14935) | VideoChat3：全开源、高效通用的视频 MLLM | 151 |
| 4 | [2607.03118](https://huggingface.co/papers/2607.03118) | Vidu S1：实时交互式视频生成模型 | 138 |
| 5 | [2607.05394](https://huggingface.co/papers/2607.05394) | Weak-to-Strong Generalization via Direct On-Policy Distillation | 131 |
| 6 | [2607.13125](https://huggingface.co/papers/2607.13125) | Boogu-Image-0.1：增强开源统一多模态理解与生成 | 127 |
| 7 | [2607.12463](https://huggingface.co/papers/2607.12463) | Function-Aware FIM：面向编码 agent 基座的中训练 | 105 |
| 8 | [2607.10383](https://huggingface.co/papers/2607.10383) | ABot-N1：通用视觉语言导航基座模型 | 101 |
| 9 | [2607.12395](https://huggingface.co/papers/2607.12395) | Ring-Zero：把 Zero-RL 扩到万亿参数以涌现推理 | 93 |
| 10 | [2607.14777](https://huggingface.co/papers/2607.14777) | SEED：面向 agentic RL 的自进化 on-policy 蒸馏 | 93 |
| 11 | [2607.05382](https://huggingface.co/papers/2607.05382) | Search Beyond What Can Be Taught：演化 agentic 视觉生成的知识边界 | 86 |
| 12 | [2607.10350](https://huggingface.co/papers/2607.10350) | ABot-AgentOS：带终身多模态记忆的通用机器人 agent OS | 84 |
| 13 | [2607.11886](https://huggingface.co/papers/2607.11886) | Read It Back：预训练 MLLM 是文生图的零样本奖励模型 | 82 |
| 14 | [2607.09024](https://huggingface.co/papers/2607.09024) | 视频生成模型是通用视觉学习器 | 81 |
| 15 | [2607.08964](https://huggingface.co/papers/2607.08964) | Long-Horizon-Terminal-Bench：稠密奖励评测长程终端任务 | 74 |
| 16 | [2607.10400](https://huggingface.co/papers/2607.10400) | SynthDocBench：长上下文视觉文档理解的可控基准 | 70 |
| 17 | [2603.29616](https://huggingface.co/papers/2603.29616) | Video-Oasis：重思视频理解的评测 | 64 |
| 18 | [2607.15257](https://huggingface.co/papers/2607.15257) | SearchOS-V1：鲁棒的开放域信息检索 agent 协作 | 62 |
| 19 | [2607.12625](https://huggingface.co/papers/2607.12625) | KnowAct-GUIClaw：自进化记忆与技能的个人 GUI 助手 | 56 |
| 20 | [2607.09657](https://huggingface.co/papers/2607.09657) | Scalable Visual Pretraining for Language Intelligence | 55 |
| 21 | [2601.16211](https://huggingface.co/papers/2601.16211) | Why Can't I Open My Drawer：缓解零样本组合动作识别的对象捷径 | 54 |
| 22 | [2607.09125](https://huggingface.co/papers/2607.09125) | 低重叠拍摄下的 4D 人-场景重建 | 53 |
| 23 | [2607.13639](https://huggingface.co/papers/2607.13639) | OvisOCR2 技术报告 | 53 |
| 24 | [2607.15207](https://huggingface.co/papers/2607.15207) | BadWAM：世界-动作模型"梦对了却做错了" | 49 |
| 25 | [2607.11487](https://huggingface.co/papers/2607.11487) | LightMem-Ego：面向日常生活的 AI 记忆 | 47 |

## 分主题详解

### 主题一：Agent Harness 与长程系统工程（本周最强）

榜首扎堆于此，延续了近两月"harness 是下一个战场"的主线。
- **Harness Handbook**（#1, ▲201，见 Deep Dive 1）：把 agent harness 代码库自动结构化成"行为中心"的手册，解决 harness 演化中的**行为定位**瓶颈。
- **LongStraw**（#2, ▲180）：固定 GPU 预算下把 GRPO 长上下文 RL 推到 2M+ token——已单独深读。
- **SearchOS-V1**（#18, ▲62）：开放域信息检索的多 agent 协作系统。
- **KnowAct-GUIClaw**（#19, ▲56）：带**自进化记忆与技能**的个人 GUI 助手——把"越用越会"落到桌面操作。
- **Long-Horizon-Terminal-Bench**（#15, ▲74）：用稠密奖励评测 agent 在**长程终端任务**上的极限——直击"长程"的可测性。
- **AgentCompass**（▲39）：统一的 agent 能力评测基础设施。

### 主题二：视频理解与生成（多模态第一大类）

- **VideoChat3**（#3, ▲151）：全开源、高效的通用视频 MLLM，主打"开放 + 通用"。
- **Vidu S1**（#4, ▲138）：**实时交互式**视频生成——把视频生成从离线推向交互。
- **Video Generation Models are General-Purpose Vision Learners**（#14, ▲81）：论证视频生成模型可作通用视觉表征学习器，呼应"生成即理解"。
- **Video-Oasis**（#17, ▲64）/ **SynthDocBench**（#16, ▲70）：分别重思视频理解评测、长上下文视觉文档理解基准。
- **4D 人-场景重建**（#22, ▲53）/ **MetaView**（▲35，单目新视角合成）：4D/3D 视觉持续升温。

### 主题三：统一多模态理解与生成

- **Boogu-Image-0.1**（#6, ▲127）：增强开源的**统一**多模态理解+生成。
- **Read It Back**（#13, ▲82）：发现预训练 MLLM 可直接当文生图的**零样本奖励模型**——省掉专门训练 reward model。
- **Concurrent Image Understanding and Generation**（▲30）：自纠式的理解-生成并发。
- **OvisOCR2**（#23, ▲53）：OCR 专项技术报告。

### 主题四：RL / 蒸馏 / 自进化

- **Weak-to-Strong via Direct-OPD**（#5, ▲131，见 Deep Dive 2）：在小模型上做 RL、再把"RL 引起的策略偏移"蒸给大模型，绕开大模型昂贵的 rollout。
- **Ring-Zero**（#9, ▲93）：把 Zero-RL 扩到**万亿参数**以涌现推理。
- **SEED**（#10, ▲93）：面向 agentic RL 的**自进化 on-policy 蒸馏**。
- **Search Beyond What Can Be Taught**（#11, ▲86）：演化 agentic 视觉生成的知识边界（自进化思想进入视觉生成）。
- **Function-Aware FIM**（#7, ▲105）：以函数感知的 fill-in-the-middle 做编码 agent 基座的**中训练**。

### 主题五：具身 / 机器人 / 世界模型

- **ABot-N1**（#8, ▲101）：通用**视觉语言导航**基座模型。
- **ABot-AgentOS**（#12, ▲84）：带**终身多模态记忆**的机器人 agent OS——H3（跨任务经验积累）在具身域的落地。
- **Xiaomi-Robotics-1 / U0**（▲36/43）：VLA 模型规模化 + 统一具身合成。
- **BadWAM**（#24, ▲49）：揭示世界-动作模型"梦对了却做错"——生成的想象正确但动作执行错误，指出 WAM 的一个失效模式。

### 主题六：记忆 / 长上下文 / 效率

- **LightMem-Ego**（#25, ▲47）：面向日常生活的轻量 AI 记忆。
- **RAGU**（▲31）：多步 GraphRAG 引擎 + 紧凑领域适配。
- **xHC（Expanded Hyper-Connections）**（▲36）/ **KronQ**（▲32，Kronecker 分解 Hessian 的 LLM 量化）：架构与量化侧的效率工作。

## Deep Dive 1：Harness Handbook（▲201，本周榜首）

[arXiv:2607.13285](https://huggingface.co/papers/2607.13285)（Tencent HY LLM Frontier / Indiana / UMD / UGA / NUS）

**问题**：现代 agent 的能力不只取决于基座模型，还取决于 **harness**——构造 prompt、管理状态、调用工具、协调执行的那层代码。而 harness 需要随模型/API/环境/需求**不断修改**。修改前，开发者（或 coding agent）必须先定位"实现某个行为的所有代码位置"。难点在于：生产级 harness 往往**庞大、耦合紧、行为分散**在多个文件/函数/执行阶段/状态转移里，而修改需求描述的是"系统应该做什么"，代码库却是按文件/函数/模块组织的。**行为定位（behavior localization）成了 harness 演化的核心瓶颈**。

**方法**：
1. **Harness Handbook**——一种**以行为为中心**的表示，通过**静态程序分析 + LLM 辅助的行为结构化**，自动从 harness 代码库合成；它把实现知识围绕"系统行为"组织，并把每个行为**链接到对应源码**。
2. **BGPD（Behavior-Guided Progressive Disclosure，行为引导的渐进式披露）**——引导 coding agent 从高层行为描述**逐层下钻**到相关实现细节（L1 系统概览 → L2 组件概览 → L3 单元深入），并对候选位置**对照当前源码验证**。

![Harness Handbook 结构（左：系统概览/执行阶段/状态与数据/组件的分层目录；中：主循环行为条目——目的/触发条件/输入/处理步骤/输出/异常等）+ 渐进式披露 L1→L2→L3（右：从系统概览到组件概览再到单元深入，信息由浅入深，附各层包含的信息清单）（arXiv:2607.13285 Fig.）](2026-07-20-hf-daily-papers-jul10-20/harness-handbook-x1.png)

**效果**：在两个开源 agent harness 的多样化修改请求上，Handbook 辅助的规划**显著提升行为定位准确率与编辑计划质量**。

**我的看法**：这篇拿榜首不意外——它精准命中了本项目 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 里 Pillar I 的痛点：**当能力越来越多地沉淀在 harness，harness 本身的可维护性就成了瓶颈**。有意思的是它把"读懂代码库"从传统的"按文件检索"翻转成"按行为检索"——这与软件工程里的 feature location 是同一个老问题，但在 agent harness 这种"行为高度分散"的代码里尤为尖锐。它也间接印证了我在长程 agent 笔记里的判断：harness 工程正在成为独立的工程学科。

## Deep Dive 2：Weak-to-Strong via Direct On-Policy Distillation（▲131）

[arXiv:2607.05394](https://huggingface.co/papers/2607.05394)

**问题**：RLVR（可验证奖励 RL）是提升推理的强配方，但**对每个新的强模型都重跑一遍代价极高**——训练时目标模型要生成海量 rollout。模型越大，后训练本身越成为瓶颈。

**核心思路（weak-to-strong）**：在**小模型**上跑 RL（rollout 便宜），再把它学到的东西迁移去提升**大模型**。但直接蒸馏"RL 之后的弱 teacher"不够——teacher 的最终策略把"有用的 RL 收益"和"小模型自身的局限"混在了一起。

**方法 Direct-OPD**：只迁移 teacher 的 **RL 引起的策略偏移（policy shift）**，而非其绝对策略。做法是把 **post-RL teacher 与它自己 pre-RL 的 reference 对比**，用二者的 **log-ratio 作为给 student 的稠密隐式奖励**。

![(a) Direct-OPD vs 普通 OPD：训练中 Direct-OPD(蓝)的 AIME24 avg@32 持续爬升到 ~0.63，而普通 OPD(红)反而跌破 teacher 甚至掉到 0.50——直接蒸馏 post-RL teacher 会把小模型的局限一起学过来；(b) weak-to-strong 泛化：以小 teacher(ref 28.5)指导，Qwen3-1.7B/4B、R1-Distill-7B 分别 +14.1/+5.1/+6.4，全部超过 teacher（arXiv:2607.05394 Fig.）](2026-07-20-hf-daily-papers-jul10-20/w2s-opd-x1.png)

**关键结果**（AIME24 avg@32）：

| Student | Initial | Direct-OPD | 提升 |
|---|---|---|---|
| Qwen3-1.7B | 48.x | 62.x | **+14.1** |
| Qwen3-4B | 72.6 | 77.7 | **+5.1** |
| R1-Distill-7B | 56.x | 63.x | **+6.4** |

三个 student 用**同一个弱 teacher**（teacher 自身水平 ref 28.5）指导后**全部反超 teacher**，实现真正的 weak-to-strong 泛化。

**我的看法**：这篇的巧思在于**"蒸偏移而非蒸策略"**——把 log(π_post / π_pre) 当稠密奖励，等于只提取了"RL 到底改变了什么"，剥离了弱模型的能力天花板。这和本项目 [推理努力度综述](/research-notes/2026-07-20-blog-reasoning-effort.md) 里 DeepSeek V4 的 on-policy 蒸馏、以及本周 SEED（#10）的自进化蒸馏是同一股潮流：**当大模型 RL 越来越贵，"在小模型上做实验、把增量迁移到大模型"会成为标准范式**。一个待验证点是——这套"策略偏移"在数学之外（如 agent 长程任务）是否同样可迁移。

## 其他值得关注（精选剩余，一句话）

- **Scalable Visual Pretraining for Language Intelligence**（#20, ▲55）：用视觉预训练反哺语言智能，探索"看图学语言"。
- **Why Can't I Open My Drawer?**（#21, ▲54）：缓解零样本组合动作识别里的"对象捷径"偏差。
- **AdvancedMathBench**（▲32）：高等数学推理基准套件。
- **From Pixels to States**（▲32）：重思交互式世界模型的状态表示。
- **KeyFrame-Compass / MultiRef-Compass / Blind-Spots-Bench**（▲37/32/31）：多模态评测"Compass/Bench"系列继续扩张——本周评测类论文密集。
- **Cura 1T**（▲32）：面向 agentic 医疗的专用万亿模型。
- **Loop the Loopies!**（▲31）：循环/迭代结构的探索（标题俏皮，内容待考）。

## 趋势分析

1. **长程 agent 的"系统工程转向"坐实**。榜首两篇（Harness Handbook + LongStraw）+ SEED/SearchOS/KnowAct/AgentCompass/Long-Horizon-Terminal-Bench，共同指向：**agent 的下一步竞争在 harness 的可维护性、长程 RL 的可训练性、以及长程能力的可评测性**——而非单纯堆模型。这与本项目 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 的两支柱框架完全同频。
2. **"在小模型上做 RL、把增量迁移到大模型"成为显学**。Direct-OPD（#5）、SEED（#10）、Ring-Zero（#9）从不同角度攻击"大模型后训练太贵"——蒸偏移、自进化蒸馏、Zero-RL 扩规模。后训练的**成本结构**正在被重新设计。
3. **视频从"生成"走向"实时交互 + 通用表征"**。Vidu S1（实时交互）、VideoChat3（开源通用理解）、"视频生成模型是通用视觉学习器"三篇合起来，暗示视频模型正从单一生成任务，走向**交互式 + 作为通用视觉基座**。
4. **自进化记忆下沉到具身与个人助手**。ABot-AgentOS（终身记忆）、KnowAct-GUIClaw（自进化技能）、LightMem-Ego（日常记忆）——H3（跨任务经验积累）从论文概念开始进入具体产品形态。

## Open Questions

1. Harness Handbook 的"行为中心表示"能否**自动跟随 harness 演化保持同步**？行为定位的准确率在 harness 快速迭代时会不会衰减？
2. Direct-OPD 的"蒸策略偏移"在**非可验证领域**（agent、创意、多轮对话）是否还成立？log-ratio 作为稠密奖励的前提是 teacher 的偏移"纯正"，这在没有 verifiable reward 时能否保证？
3. 本周评测类论文（Compass/Bench 系列）井喷——**评测的碎片化**是否会稀释信号？需要一个像 RULER 之于长上下文那样的"戳穿虚标"的统一 agent 评测。
4. 视频生成模型作"通用视觉学习器"，与专门的视觉编码器相比，在**下游理解任务**上的性价比究竟如何？

## References

> 本周覆盖 141 篇，全部经 `add_paper.py` 核验入 `references.bib`（个别未上 arXiv 索引者除外）。以下列 Top 25 的 HF link（完整列表见总览表）。

- Harness Handbook — https://huggingface.co/papers/2607.13285
- LongStraw — https://huggingface.co/papers/2607.14952 （另见本项目深读）
- VideoChat3 — https://huggingface.co/papers/2607.14935
- Vidu S1 — https://huggingface.co/papers/2607.03118
- Weak-to-Strong via Direct-OPD — https://huggingface.co/papers/2607.05394
- Boogu-Image-0.1 — https://huggingface.co/papers/2607.13125
- Function-Aware FIM — https://huggingface.co/papers/2607.12463
- ABot-N1 — https://huggingface.co/papers/2607.10383
- Ring-Zero — https://huggingface.co/papers/2607.12395
- SEED — https://huggingface.co/papers/2607.14777
- Search Beyond What Can Be Taught — https://huggingface.co/papers/2607.05382
- ABot-AgentOS — https://huggingface.co/papers/2607.10350
- Read It Back — https://huggingface.co/papers/2607.11886
- Video Generation as Vision Learners — https://huggingface.co/papers/2607.09024
- Long-Horizon-Terminal-Bench — https://huggingface.co/papers/2607.08964
- SynthDocBench — https://huggingface.co/papers/2607.10400
- Video-Oasis — https://huggingface.co/papers/2603.29616
- SearchOS-V1 — https://huggingface.co/papers/2607.15257
- KnowAct-GUIClaw — https://huggingface.co/papers/2607.12625
- Scalable Visual Pretraining — https://huggingface.co/papers/2607.09657
- Why Can't I Open My Drawer — https://huggingface.co/papers/2601.16211
- 4D Human-Scene Reconstruction — https://huggingface.co/papers/2607.09125
- OvisOCR2 — https://huggingface.co/papers/2607.13639
- BadWAM — https://huggingface.co/papers/2607.15207
- LightMem-Ego — https://huggingface.co/papers/2607.11487

> 说明：deep-dive 配图取自各自 arXiv HTML（已注明）；LongStraw 未在本文重复深挖，详见 [`2026-07-20-longstraw-longcontext-rl.md`](/research-notes/2026-07-20-longstraw-longcontext-rl.md)。
