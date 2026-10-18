# OpenMOSS：Agent 评测、长上下文与可靠性／可解释性支线的一手发现

截至日期：2026-10-17。检索及核查日期：2026-10-17。用途：供 OpenMOSS 总报告整合；不覆盖其基础模型、语音、视觉及机器人主线。

## 0. 证据规则与阅读结论

本次采用 research 与 academic-search 工作流：仅从官方 GitHub、作者项目页、arXiv 全文及正式 proceedings 追踪机制、表格与发布状态；本分支未进一步分发子任务。所有数值均为作者报告，本次未重跑训练、Docker 评测或付费 API，不把“有代码”写成“独立复现成功”。`citation_count` 未查询，统一为 `null`；项目没有确认的专属论文时不虚构论文记录。

优先整合的结论：

1. **Agent 支线在把评价对象从文本正确率推向可执行工程与持续工作。** ABC-Bench 要求服务启动并通过外部 API 测试；SWE-bench Science 检查科学软件的私有测试与科学契约；ContextWeave 检查跨会话工作区质量和个人偏好。它们是三种不同的评测，不是三个 OpenMOSS agent 模型，也不能相加为统一能力分数。[ABC 全文](https://arxiv.org/pdf/2601.11077)、[Science 全文](https://arxiv.org/pdf/2608.19799v2)、[ContextWeave 全文](https://arxiv.org/pdf/2608.04830)
2. **截至日期必须区分论文结果与更新榜单。** SWE-bench Science 论文中最高 Pass@1 为 Claude-Opus-5／Claude Code 的 47.90%；官方榜单标注更新于 2026-10-09，列出 GPT-6 Astra／Codex 50.42%。因此“当前最强仍低于 50%”已不是截至日期成立的说法；这些模型名称仅按该 benchmark 的标注转述，不据此验证供应商发布情况。[论文表 2](https://arxiv.org/pdf/2608.19799v2)、[官方榜单](https://swescience.github.io/)
3. **增加记忆不等于可靠性单调提升。** ContextWeave 最强配置把 Workspace Score 从 68.08 提至 78.20、Preference Score 从 41.50 提至 70.60，同时该配置的记忆诱发问题任务比例达到 7.39%；更轻量的两个记忆配置该比例为 0.35%。论文明确承认记忆组件同时存在检索策略与上下文长度等混杂因素，不能把观察直接解释成“完整历史因果优于摘要”。[表 1、§5.2.3](https://arxiv.org/pdf/2608.04830)
4. **长上下文至少分为容量、使用质量、安全三个问题。** ReAttention 是无需训练的 KV 检索与有限注意力作用域方法；RoPE++ 是需要训练的注意力架构变化；LongSafety 是长上下文安全数据、评测及 SFT 研究。它们分别提供不同证据，不构成“上下文越长就越安全／越会推理”的证明。[ReAttention](https://arxiv.org/pdf/2407.15176v3)、[RoPE++](https://arxiv.org/pdf/2512.07525)、[LongSafety](https://arxiv.org/pdf/2411.06899v2)
5. **可解释性已不只是小模型演示，但也不是安全证明。** Llama Scope 提供 256 个 SAE／transcoder；Lorsa 用数万稀疏注意力单元近似注意力计算；Llamascopium 是支持这些方法的框架。重建误差、共享 QK、自动解释评价及归因时冻结注意力等限制仍然存在。[Llama Scope](https://arxiv.org/pdf/2410.20526)、[Lorsa 正式版](https://proceedings.iclr.cc/paper_files/paper/2026/file/7e57d4c3bb2bd2396a551e7aeea19aed-Paper-Conference.pdf)、[CRM 作者报告](https://interp.open-moss.com/posts/complete-replacement)
6. **开源成熟度并不齐整。** LongSafety README 仍把训练启动配置列为 TODO、微调权重写为未来发布；Hybrid-Mechanics 虽已有 2026-10-07 的 arXiv 论文，其仓库可见文件仍主要是 README 和 PDF。不能用组织总体“开放可复现”的愿景替代单项目发布审计。[LongSafety README](https://github.com/OpenMOSS/LongSafety/blob/master/README.md)、[Hybrid 仓库](https://github.com/OpenMOSS/Hybrid-Mechanics)、[Hybrid arXiv](https://arxiv.org/abs/2610.10114)

## 1. 项目身份与精确论文映射

下表的“身份”是根据交付物作出的分类，不是热度或科研价值排名。未确认正式发表的平台写作预印本；正式 venue 见第 7 节元数据中的一手来源。

| 项目与官方仓库 | 精确论文 ID | 交付物／身份 |
| --- | --- | --- |
| [ABC-Bench](https://github.com/OpenMOSS/ABC-Bench) | [2601.11077](https://arxiv.org/abs/2601.11077) | 后端工程 benchmark＋执行协议；另有 Qwen3-8B／32B-ABC SFT 权重，不能与 benchmark 混称 |
| [SWE-bench-Science](https://github.com/OpenMOSS/SWE-bench-Science) | [2608.19799](https://arxiv.org/abs/2608.19799) | 科学软件 repository-level benchmark，不是科学基础模型 |
| [AgentHPOBench](https://github.com/OpenMOSS/AgentHPOBench) | [2607.29626](https://arxiv.org/abs/2607.29626) | 顺序超参优化评测与 harness，不是自主科学家模型 |
| [ContextWeave](https://github.com/OpenMOSS/ContextWeave) | [2608.04830](https://arxiv.org/abs/2608.04830) | 工作流／agent 记忆 benchmark，不是新记忆模型 |
| [UnifiedToolHub](https://github.com/OpenMOSS/UnifiedToolHub) | 未确认专属论文 | 工具数据标准化、标注、训练数据准备及评测工程框架 |
| [HalluQA](https://github.com/OpenMOSS/HalluQA) | [2310.03368](https://arxiv.org/abs/2310.03368) | 中文幻觉 benchmark＋评价脚本 |
| [Say-I-Dont-Know](https://github.com/OpenMOSS/Say-I-Dont-Know) | [2401.13275](https://arxiv.org/abs/2401.13275) | 模型特定 Idk 数据、对齐训练与评价研究；不是名为“Say-I-Dont-Know”的通用基础模型 |
| [LongSafety](https://github.com/OpenMOSS/LongSafety) | [2411.06899](https://arxiv.org/abs/2411.06899) | 安全训练数据＋LongSafetyBench＋SFT 研究；公开训练／权重材料不完整 |
| [ReAttention](https://github.com/OpenMOSS/ReAttention) | [2407.15176](https://arxiv.org/abs/2407.15176) | 无训练推理方法与 kernel／评测实现，不是独立基础模型 |
| [rope_pp](https://github.com/OpenMOSS/rope_pp) | [2512.07525](https://arxiv.org/abs/2512.07525) | RoPE++ 架构、预训练与评价代码、实验模型 |
| [Thus-Spake-Long-Context-LLM](https://github.com/OpenMOSS/Thus-Spake-Long-Context-LLM) | [2502.17129](https://arxiv.org/abs/2502.17129) | 综述＋资源清单，不是训练框架或模型 |
| [Llamascopium](https://github.com/OpenMOSS/Llamascopium)／[Lorsa](https://github.com/OpenMOSS/Lorsa) | 核心：[2410.20526](https://arxiv.org/abs/2410.20526)、[2504.20938](https://arxiv.org/abs/2504.20938)；延伸：[2508.16929](https://arxiv.org/abs/2508.16929)、[2509.17196](https://arxiv.org/abs/2509.17196) | 分布式稀疏解释框架／稀疏注意力分解；解释模块的权重不等同于新的通用 LLM |
| [BandPO](https://github.com/OpenMOSS/BandPO) | [2603.04918](https://arxiv.org/abs/2603.04918) | LLM RL 优化算法与训练实现，不是 benchmark |
| [Hybrid-Mechanics](https://github.com/OpenMOSS/Hybrid-Mechanics) | [2610.10114](https://arxiv.org/abs/2610.10114) | 长上下文混合架构机制研究，仓库主要公开论文材料 |
| [GSF-chi](https://github.com/OpenMOSS/GSF-chi) | [2609.12532](https://arxiv.org/abs/2609.12532) | 手性分子图 Transformer；跨域研究边线，不是 agent／长上下文 LLM |

## 2. 优先 agent 评测：从“代码像答案”到“工作真正完成”

### 2.1 ABC-Bench：部署是编码能力的一部分

**机制与对象。** 224 个任务来自 127 个 MIT 许可仓库，覆盖 8 种语言、19 个框架；其中 92 个任务还要求自行配置环境。ABC-Pipeline 使用具有命令行能力的 GPT-5 探索仓库、生成 API 验证与环境，再构造任务；评价要求服务真实启动，接受外部 HTTP/API 测试，而非仅比较补丁或局部单测。[仓库](https://github.com/OpenMOSS/ABC-Bench)、[全文 §4、附录仓库表](https://arxiv.org/pdf/2601.11077)

**论文实证。** 默认 OpenHands，同一任务执行三次独立尝试，表中报告平均 Pass@1，而不是“三次任一成功”的 Pass@3。[表 2、§5.1](https://arxiv.org/pdf/2601.11077)

| 模型／设置 | 全部 224 任务平均 Pass@1 | 官方证据 |
| --- | ---: | --- |
| Claude Sonnet 4.5／OpenHands | 63.2 ± 1.9% | [表 2](https://arxiv.org/pdf/2601.11077) |
| DeepSeek-V3.2／OpenHands | 50.1 ± 1.9% | [表 2](https://arxiv.org/pdf/2601.11077) |
| GPT-5／OpenHands | 49.4 ± 1.9% | [表 2](https://arxiv.org/pdf/2601.11077) |
| Qwen3-8B：SFT 前 → 后 | 8.3% → 13.9% | [图 6、§6.2](https://arxiv.org/pdf/2601.11077) |
| Qwen3-32B：SFT 前 → 后 | 8.9% → 33.8% | [图 6、§6.2](https://arxiv.org/pdf/2601.11077) |

**可靠性发现。** 在 92 个环境任务中，GPT-5 与 DeepSeek-V3.2 的环境构建成功率低于 50%，但构建成功后功能测试成功率高于 80%；这说明环境阶段能遮蔽已有编码能力。交互轮次与成功率相关系数为 0.87，不过它是跨模型观察，不证明“强行多运行几轮”能因果提高成功率。[§5.2、图 4–5](https://arxiv.org/pdf/2601.11077)

**边界与发布。** SFT 使用公开 Nex-N1 的 agentic coding 子集；官方单独链接两个 ABC 权重。benchmark、SFT 权重和任务生成流水线是不同成果。论文中的生成／测试 oracle 由模型参与构造，实际可靠性仍受 API 测试覆盖率、仓库分布、harness 与预算影响；“通过此测试”不能升级为完整后端安全、负载或生产可维护性保证。后一句是基于评价协议的分析，而非论文测得的生产指标。[§4、§6.1–6.2](https://arxiv.org/pdf/2601.11077)、[权重与执行入口](https://github.com/OpenMOSS/ABC-Bench)

### 2.2 SWE-bench Science：把科学契约纳入软件修复

**机制与对象。** 119 个任务、98 个仓库、20 个科学领域，划分 Issue-driven、Expert-exploratory、Engineering-integration 三类。任务不只要求代码运行，还要求保持单位、坐标、数值不变量、格式语义和科学假设；私有验证覆盖替代路径、边界输入与回归行为。[全文 §3–4](https://arxiv.org/pdf/2608.19799v2)、[官方仓库](https://github.com/OpenMOSS/SWE-bench-Science)

**指标必须分开。** PublicScore／PrivateScore 是适用测试的平均分；Fail2Pass 看原先失败的私有测试被修复的比例；Pass2Pass 看原先通过的测试是否保持；Pass@1 只有全部适用私有测试通过才为 1。因此公共测试接近满分并不等于任务完全成功。[§5](https://arxiv.org/pdf/2608.19799v2)

| 证据层 | 关键结果 | 解释 |
| --- | --- | --- |
| 论文 v2、表 2 | Claude-Opus-5／Claude Code：Public 96.64%、Private 75.11%、Pass@1 47.90% | 私有平均分与精确任务成功率不是同一指标。[论文](https://arxiv.org/pdf/2608.19799v2) |
| 论文 v2、表 2 | DeepSeek-V4-Pro：Public 100%、Pass@1 42.02% | 公共测试全过仍留下科学泛化／完整修复问题。[论文](https://arxiv.org/pdf/2608.19799v2) |
| 官方榜单，标注更新 2026-10-09 | GPT-6 Astra／Codex：Private 83.18%、Pass@1 50.42% | 截至日期的在线发布结果；不是原论文表 2。[榜单](https://swescience.github.io/) |
| 同一在线榜单 | DeepSeek-V4-Pro（max）：Pass@1 39.50% | 与论文 42.02% 不同；当前页面未充分解释差值，不合并为同一次运行。[榜单](https://swescience.github.io/)、[论文](https://arxiv.org/pdf/2608.19799v2) |

**比 leaderboard 更有用的发现。** 作者把失败归为科学知识／抽象不足、探索失误／表层修复、覆盖／系统集成不完整、科学知识泛化失败，并把运行／评价路径失败另记。这是外部行为和任务轨迹的失败分析，不能称为神经网络内部机制解释。[§6.1、表 3](https://arxiv.org/pdf/2608.19799v2)

**科学信息并非单调有益。** 91 个可分离信息任务的配对比较保留相同仓库与执行环境，只移除额外科学指导。GPT-5.6-sol 的 PrivateScore 从 73.23% 提至 74.06%，但 Pass@1 从 36.26% 降至 31.87%；DeepSeek-V4-flash 的 Pass@1 则从 16.48% 提至 23.08%，同时消耗更多 token。论文明确称这些是描述性差异，没有建立统计显著性或因果效果，不能写成“科学知识降低／提高 agent 能力”的普适定律。[§4.2、§6.2、图 6–7](https://arxiv.org/pdf/2608.19799v2)

**复现边界。** 发布包含 119 个任务，但默认只选择 96 个不受 restricted-license gate 限制的任务；其余 23 个需显式 opt-in。环境／验证镜像按 digest 固定，记录任务选择 hash 与运行配置；公开源材料不含参考修复补丁、私有验证测试源码或 agent 轨迹。不能拿默认 96 任务直接对比论文全部 119 任务；也不能因私有测试源码不公开就断言评测无法执行——执行镜像与源码级重建是两种不同开放程度。[发布 README](https://github.com/OpenMOSS/SWE-bench-Science)、[数据契约](https://github.com/OpenMOSS/SWE-bench-Science/blob/main/docs/dataset-contract.md)、[架构文档](https://github.com/OpenMOSS/SWE-bench-Science/blob/main/docs/architecture.md)

### 2.3 ContextWeave：记忆必须对下游工作负责

**机制与对象。** 将 14 位参与者数月工作流匿名化并重构为 1,005 个可执行任务；568 个主工作流任务进入核心评测，437 个背景／临时任务保留在历史中。每个子任务从参考 Docker 状态开始，并对齐参考轨迹，控制前序执行漂移；它测的是记忆对后续任务的帮助，不是让同一个 agent 在完全自由的数月生命周期中持续积累所有错误。[§4、§5.1](https://arxiv.org/pdf/2608.04830)、[运行说明](https://github.com/OpenMOSS/ContextWeave)

**评价。** 主要指标是工作区完成质量 Workspace Score 与用户偏好符合度 Preference Score；辅助检查历史相关性、减少重复探索、历史能否解决执行问题，以及记忆是否引入新问题。主实验固定 Codex harness 与执行模型，比较六个记忆组件；GPT-5.5 用于评分／诊断，workspace 评分还会检查候选与参考环境及执行证据。[§4.5、§5.2.1](https://arxiv.org/pdf/2608.04830)

| 配置 | Workspace | Preference | 记忆诱发问题任务比例 | 官方证据 |
| --- | ---: | ---: | ---: | --- |
| No Recall | 68.08 | 41.50 | 不定义 | [表 1](https://arxiv.org/pdf/2608.04830) |
| mem0 | 72.48 | 49.73 | 0.35% | [表 1](https://arxiv.org/pdf/2608.04830) |
| memos | 70.01 | 46.38 | 0.35% | [表 1](https://arxiv.org/pdf/2608.04830) |
| langmem | 75.29 | 57.37 | 5.11% | [表 1](https://arxiv.org/pdf/2608.04830) |
| a-mem | 78.20 | 70.60 | 7.39% | [表 1](https://arxiv.org/pdf/2608.04830) |

**含义与限制。** a-mem 的具体经验更有可操作性，少重复探索，但更容易带来记忆诱发问题；7.39% 包括后来解决了的问题，不是“最终失败率”。组件之间还同时变化了检索策略和上下文量，作者只作经验关联解释。固定 mem0 后，五种执行模型均获得下游收益，但仍有模型间差异；记忆系统质量与使用者模型能力必须一起考察。[§5.2.3、§5.3](https://arxiv.org/pdf/2608.04830)

**复现与外推。** 仓库公开任务压缩包、参考日志／rubric、GHCR 镜像下载、插件与评分入口，但 reference evidence 的私有构造流水线没有完整公开。论文局限承认 rubric 的人工校准仍在进行、参与者／领域有限、harness 覆盖不足，并估计一次完整配置评价约需 200 美元；这是作者当时设置的成本，不是本次预算或价格承诺。真实工作流经过匿名化、重构和状态控制，不能直接等同于线上长期用户实验。[README](https://github.com/OpenMOSS/ContextWeave)、[论文 Limitations](https://arxiv.org/pdf/2608.04830)

### 2.4 AgentHPOBench：计分协议会改变“agent 胜过 HPO”的结论

30 个可执行研究仓库任务、7 类研究领域，每项先执行共享基线，再进行 5 次有约束的超参干预。模型看配置、指标与日志，不能自由改数据划分、目标指标或评价代码；limited budget 约为原实验训练预算的 10%。这是顺序实验决策 benchmark，不是开放式论文复现或创新能力测试。[§3–4](https://arxiv.org/pdf/2607.29626)

主要分数为有界归一化提升 MBNS：将相对 baseline→paper/repo anchor 的提升截到 [-1,1]；BWR 是胜过相应基线的任务比例，MAA 是锚点达成度。最终步协议中 Claude Sonnet 4.6 为 MBNS 0.407、BWR 76.7%、MAA 79.5%；Qwen3-32B MBNS 为 0.148；移除中间反馈降至 0.052。不能把 MBNS 0.407 读成 40.7% 正确率。[表 2、表 5、§3.4](https://arxiv.org/pdf/2607.29626)

关键反证是附录的 best-so-far：Random Search／TPE／BOHB 的 MBNS 分别为 0.325／0.298／0.291，均超过所有测试的 open-weight agents；Qwen3-32B 为 0.195。最终步同时评价搜索与“不要丢掉已有收益”，而传统 HPO 通常返回 incumbent，不能只用主表宣称 LLM 全面优于 HPO。API 只作一次审计运行，局部开源模型／传统方法为 0、1、42 三种子；一些上游目标还是 test metric，因此这不是独立 held-out 泛化证明。[附录 §8、§11.3、表 12–14](https://arxiv.org/pdf/2607.29626)

公开仓库给出上游 pinned commit、task adapter、配置、种子 provenance 与结果审计，但第三方仓库、数据、模型、checkpoint 和 runtime logs 不随 Git 分发；“有统一协议”不等于低成本一键完全重现作者全套运行。[README](https://github.com/OpenMOSS/AgentHPOBench)、[任务 provenance](https://github.com/OpenMOSS/AgentHPOBench/blob/main/docs/task-repositories.md)

### 2.5 UnifiedToolHub：有用的工程底座，不冒充新的论文或模型

整合 API-Bank、BFCL、MTU-Bench、Seal-Tools、TaskBench、ToolAlpaca 等不同工具调用数据，统一格式、标签、模型模板与训练数据输出。示例指标是 ExactMatch、ToolAccuracy、ParameterAccuracy；`single_first` 只评价首个调用块，`single_last` 的此前步骤用 golden 值，不能把这类离线分步指标当作真实 agent 长程执行成功率。[README](https://github.com/OpenMOSS/UnifiedToolHub/blob/main/README.md)

README 明确提醒：它使用模型官方模板，而部分原数据集论文使用自定义对话格式，结果不应与原论文直接比较；`train` 工作流示例生成训练用数据，后续运行自己的训练代码，不代表项目已发布一个新的 tool-use 模型。官方 README 未给专属论文，精确名称检索未确认对应 arXiv；这里记录为“未确认专属论文”，而不是“证明不存在任何论文”。工程功能与科研创新／实证指标应分别评价。[官方说明](https://github.com/OpenMOSS/UnifiedToolHub/blob/main/README.md)、[英文 README](https://github.com/OpenMOSS/UnifiedToolHub/blob/main/README.en.md)

## 3. 可靠性／安全：三个问题，不是一个“幻觉率”

### 3.1 HalluQA：中文对抗性问答与评分 oracle

450 个问题覆盖 30 个领域，包含 Misleading 175、Misleading-hard 69、Knowledge 206；用 ChatGPT3.5／Puyu／GLM-130B 参与对抗筛选，并人工写正确／错误答案与证据。衡量 non-hallucination rate：要求回答流畅、直接、受参考正确答案支持；仅当参考答案支持“不可回答”时，“不知道”才按无幻觉处理，不是随意拒答就得分。[§2、§3.2](https://arxiv.org/pdf/2310.03368v4)

GPT-4-0613 是评审，论文用 5 次判断投票；对 600 个抽样回答的人类一致率，单次／五次为 93.33%／93.50%。论文评价 24 个模型，18 个低于 50%；官方历史结果中 GPT-4-0613 为 53.11%、ERNIE-Bot 为 69.33%，不是 2026 年模型排名。2024 年添加的多选任务与原生成任务指标也不能混用。[表 4 与附录结果](https://arxiv.org/pdf/2310.03368v4)、[仓库更新及历史结果](https://github.com/OpenMOSS/HalluQA)

分析边界：对特定模型构造的 450 个中文对抗样本不代表自然流量；闭源评审版本、投票与提示影响复现。误导性错误和知识缺失是不同失效类型，中文知识 QA 结果不能推出代码、工具状态或长期记忆可靠性。[构造及评价协议](https://arxiv.org/pdf/2310.03368v4)

### 3.2 Say-I-Dont-Know：知识边界表达，而非通用自知证明

为每个模型采样每问 10 个回答，按回答正确比例决定是否“知道”；默认 Ik threshold=1.0，即 10 个全对才标已知。已知问题用正确回答，未知问题用拒答模板；比较 prompting、SFT、DPO、PPO、Best-of-N、HIR。Truthful 是正确作答 Ik-Ik 与正确拒答 Ik-Idk 之和，不是所有知识问题的答对率，也不是意识／内省能力的证明。[arXiv §3–4](https://arxiv.org/pdf/2401.13275v2)、[ICML 正式论文](https://raw.githubusercontent.com/mlresearch/v235/main/assets/cheng24i/cheng24i.pdf)

Llama-2-7b-chat 在 TriviaQA 的 Truthful：Prompting 66.93%、SFT 74.75%、DPO 77.89%、BoN(N=10) 78.96%。SFT 的正确作答为 28.57%，DPO 为 39.30%，体现“拒错”和“不过度拒答”的权衡；Natural Questions 上 HIR 72.30% 高于 BoN 67.95%，不能将 TriviaQA 最优方法推广到全部分布。[正式版表 1](https://raw.githubusercontent.com/mlresearch/v235/main/assets/cheng24i/cheng24i.pdf)

**重要勘误：** README 的 SFT 示例评价输出列出 38.37／40.59／78.96，实际与论文 BoN 行一致；SFT 数值应按正式表格取 28.57／46.19／74.75。仓库有数据、训练脚本和输出示例，本次未核验全部 checkpoint 下载。知识标签依赖模型、采样和词面匹配，随模型更换／工具检索变化需重新校准。[README 示例](https://github.com/OpenMOSS/Say-I-Dont-Know)、[正式表 1、数据迁移消融](https://raw.githubusercontent.com/mlresearch/v235/main/assets/cheng24i/cheng24i.pdf)

### 3.3 LongSafety：长上下文能力与安全对齐分开训练

将输入场景分为 Query Harmful、Partially Harmful、Fully Harmful；LongSafety 训练集为 17k 样本、8 项任务、平均 40.9k tokens；LongSafetyBench 为 1k 样本、10 项任务、平均 41.9k。论文摘要把“10 tasks”用于训练集的表述与正文／README 的 8 项不一致，采用正文 §3.2 的 8 项及评测 §3.3 的 10 项。[全文](https://arxiv.org/pdf/2411.06899v2)、[官方 README](https://github.com/OpenMOSS/LongSafety)

多选评测区分 HarmAwareness（知道有害）与 SafeResponse（选择安全回应）；主评测长度 32k，超长样本从中部截断。Qwen2.5-7B 的 SR 从 51.44% 提至 66.52%，HA 从 70.50% 提至 76.30%；Llama3.1-8B 的 SR 从 43.19% 提至 50.57%。Qwen2.5-1.5B 的 SR 提升，但 HA 从 68.50% 降至 66.20%，所以不是所有指标都单调提升。[表 4、§4](https://arxiv.org/pdf/2411.06899v2)

训练使用 InternEvo；≥7B 模型实验用 16×A100 80G，32k 长度、1,000 步。主要消融比较长上下文安全数据、短安全数据及一般长上下文对齐，支持作者的有限结论：仅一般长上下文对齐或短安全训练不能替代长安全数据。局限包括长输入／短输出为主、长输出场景不足；多选安全选择不等同于真实开放生成、agent 工具行为或任意攻击下的保障。[§4–5、Limitations](https://arxiv.org/pdf/2411.06899v2)

发布审计：数据与评测可获取；README 仍写训练启动脚本／配置 TODO、权重以后发布。因此不能宣称作者整套 SFT 已具备完整的一键复现路径，更不能把论文中“+LongSafety”的模型行理解为已全部公开下载的模型产品。正式发表为 LLMSEC 2025 workshop，不是 ACL 主会论文。[README](https://github.com/OpenMOSS/LongSafety/blob/master/README.md)、[ACL Anthology 正式记录](https://aclanthology.org/2025.llmsec-1.4/)

## 4. 长上下文：检索式推理扩展、位置架构与知识地图

### 4.1 ReAttention：有限作用域不意味着有限历史存储

先用不带位置的 QK 相关性对全历史 KV 做 top-k 选择，并保留邻近片段、全局／局部片段；再把选中片段按相对次序重排，在有限长度中重新应用位置编码及常规注意力。默认示例 32 全局＋4,096 局部＋127×32 选中 tokens，合计 8,192。无需改基础模型参数，但仍依赖足够历史 KV 存储和检索计算；“无限”是资源条件下避免固定位置／注意力窗口瓶颈的设计，不是实测无限容量或恒定总内存。[§2–3](https://arxiv.org/pdf/2407.15176v3)

LongBench 表 1 中 Llama3.1-8B 全注意力平均 38.79，ReAttention 38.63；Llama3.2-3B 分别 40.76、39.43。它的重要证据是扩展时保持部分下游质量，而非所有普通任务显著提升。NIAH 多针测试到 1M，§3.3／图 3 具体报告 Llama3.2-3B 至少 2M，以及 Qwen2-1B-32k 到 4M（128×）。百万 token 查针不能替代同长度科学推理、工作流或安全评价。[表 1、§3.3、图 3](https://arxiv.org/pdf/2407.15176v3)

**模型名称／结果冲突必须保留：** arXiv v3 摘要和 ICLR 摘要将 4M／128×归于 Llama3.2-3B，正文 §3.3／图 3 则归于 Qwen2-1B；README 又写 Qwen2-1.5B。这里按正文图表记录 Qwen2-1B，不无声统一这些表述。README 已发布近似 Triton kernel，但精确 top-k kernel 和效率分析代码仍为 TODO；“无额外 overhead”只在作者指定优化和测量设置成立，不保证任意部署端到端无成本。[全文](https://arxiv.org/pdf/2407.15176v3)、[ICLR 官方摘要](https://proceedings.iclr.cc/paper_files/paper/2025/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html)、[README TODO](https://github.com/OpenMOSS/ReAttention)

### 4.2 RoPE++／rope_pp：实部与虚部共享投影的架构实验

对 RoPE 复数内积原本未用于注意力分数的虚部增加并行注意力头，严格使用负虚部。EH 保持头数、减半 QKV 参数和 KV cache；EC 保持 cache 大小、增加一倍头数。它要从头训练，不能把普通 LLM 装一个推理补丁就获得论文效果。[§3、Limitations](https://arxiv.org/pdf/2512.07525)、[代码／模型入口](https://github.com/OpenMOSS/rope_pp)

776M、32k 继续预训练后，在跨长度平均指标中：RoPE／EH／EC 的 RULER 为 27.4／28.6／29.4，BABILong 为 22.8／19.4／24.1；可见半缓存方案并非所有任务都保持性能。附录还有 1.5B 实验：RULER 为 35.1／31.0／37.5，BABILong 为 29.5／32.9／22.9，EC 同样不是全胜。不能只引用小规模主表均值就写成“全规模普遍改善”。[表 2、表 6](https://arxiv.org/pdf/2512.07525)、[ICLR 正式全文](https://proceedings.iclr.cc/paper_files/paper/2026/file/cea5c4a422fd4e8b77fe3246559b34b1-Paper-Conference.pdf)

论文指出仍无法即插即用长度外推，超过支持长度后 perplexity 依然上升；资源限制使模型规模未到 7B。虚部头的全局注意力／加噪实验支持机制假说，但不等于证明其在实际大规模 agent 轨迹或安全场景有效。适合定位为有机制分析、训练代码及实验权重的架构研究，不是已经验证通用生产 LLM 的方案。[附录 C–D、Limitations](https://arxiv.org/pdf/2512.07525)、[正式 venue](https://proceedings.iclr.cc/paper_files/paper/2026/hash/cea5c4a422fd4e8b77fe3246559b34b1-Abstract-Conference.html)

### 4.3 Thus Spake Long-Context LLM：不能把综述覆盖当自研成果

`2502.17129` 的 v2 于 2025-11-11 更新；从 architecture、infrastructure、training、evaluation 四面整理生命周期，进一步包含长度外推、KV、记忆、架构、训练／推理基础设施、长预训练／后训练、长多模态及评价，并提出 10 个未答问题。它为以上支线提供共同词汇，不提供一个新的统一模型或新的胜率。[v2 全文](https://arxiv.org/pdf/2502.17129v2)、[资源仓库](https://github.com/OpenMOSS/Thus-Spake-Long-Context-LLM)

元数据按 arXiv v2：作者包含 Ziwei He，旧 README citation 作者列表不同，不能机械复制。综述涉及的外部工作也不能算成 OpenMOSS 自研；上下文窗口、有效使用、记忆和资源代价应分列，不能用“支持 N tokens”作为所有长上下文问题的总代表。[arXiv 作者／版本记录](https://arxiv.org/abs/2502.17129)、[v2 分类与开放问题](https://arxiv.org/pdf/2502.17129v2)

## 5. 可解释性与优化支线：已有实物，仍有忠实度缺口

### 5.1 Llamascopium／Lorsa：框架、稀疏字典与替换模型要拆开

**框架身份。** Llamascopium 原名 Language-Model-SAEs／`lm-saes`，支持分布式 SAE、Lorsa、CLT、Crosscoder 等训练、分析及可视化；当前默认分支页面为 `dev`，不应仅抓 `main` 的旧 README。Lorsa 独立仓库明确是迁移入口，完整实现转入该框架；软件 citation `OpenMoss Language Model Sparse Autoencoders` 本身不是一篇具有新 arXiv ID 的论文。[Llamascopium](https://github.com/OpenMOSS/Llamascopium)、[dev README](https://github.com/OpenMOSS/Llamascopium/blob/dev/README.md)、[Lorsa 迁移说明](https://github.com/OpenMOSS/Lorsa)

**Llama Scope，2410.20526。** 在 Llama3.1-8B-Base 的 32 层上，对残差、attention 输出、MLP 输出及 transcoder 四种位置，各训练 32K／128K 宽度，合计 256 个稀疏模块。改良 TopK 的 decoder 范数处理、K 退火及 JumpReLU 后处理；在 held-out 50M tokens 评价 L0、explained variance、ΔLM loss，部分残差设置把 L0 从约 150 降到 50而保持或改善重建。交付的是字典／模块权重与解释底座，不是提升后的 Llama 基础模型；这些重建／稀疏指标也不是事实正确率。[全文 §2–4](https://arxiv.org/pdf/2410.20526)、[官方权重](https://huggingface.co/fnlp/Llama-Scope)

**Lorsa，2504.20938／ICLR 2026。** 用过完备的 attention heads、rank-1 OV 与每 token TopK 激活近似原 attention；QK 在一组 heads 内共享，不是所有 QK 都独立低秩。Llama3.1-8B 的示例每层 32K 稀疏 heads、128 个活跃、512M 模块参数；这更像为解释购买冗余表示，不是推理压缩保证。论文恢复 induction／name mover 等机制，并发现算术单元；24 个 Pythia 自动解释比较为 6 胜、3 负、15 平，支持“与 SAE 相当”，不支持“所有模型解释性大幅超越 SAE”。[正式全文表 1、自动解释与局限](https://proceedings.iclr.cc/paper_files/paper/2026/file/7e57d4c3bb2bd2396a551e7aeea19aed-Paper-Conference.pdf)、[arXiv](https://arxiv.org/abs/2504.20938)

**两项直接后续。** `2508.16929` 当前题名为 *Dimensional Collapse in Transformer Attention Outputs: A Challenge for Sparse Dictionary Learning*，旧 `main` README 仍用旧题名；ICML 2026 正式记录确认注意力输出有效维度约全空间 60%，活跃子空间初始化把 1M-feature attention SAE 的死特征从 87% 降至低于 1%。这是字典训练效率／几何问题的解决，不是消除全部解释误差。`2509.17196`／ICLR 2026 用 cross-snapshot crosscoder 跟踪预训练概念演化、结合特征归因和两阶段学习解释；其模型样本／机制分析不能直接当作所有 foundation models 的统一发展定律。[ASI 正式记录](https://proceedings.mlr.press/v306/wang26bh.html)、[ASI 全文](https://arxiv.org/pdf/2508.16929v4)、[Crosscoder 全文](https://arxiv.org/pdf/2509.17196v2)、[正式 venue](https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html)

**CRM 的最新定位。** 2026-02 的作者报告 *Bridging the Attention Gap: Complete Replacement Models for Complete Circuit Tracing* 将 Lorsa 与 transcoder 组合，在 Qwen3-1.7B 上作完整替换与 circuit tracing；归因时冻结 attention pattern 与 layernorm 分母。报告展示字符串索引、induction 等案例及干预忠实度检查，但承认重建误差、抑制／未激活特征和变量绑定等解释缺口。“complete”主要指架构路径覆盖，不是所有行为已被解释。作者报告没有给出确认的专属 arXiv ID，本次也未确认独立论文；不把它标成新的正式会议论文。[作者全文报告](https://interp.open-moss.com/posts/complete-replacement)、[框架 News](https://github.com/OpenMOSS/Llamascopium/blob/dev/README.md)

**整体判断。** SAE suite 和可训练框架已超过单个玩具 PoC；Lorsa／CRM 的具体解释仍是近似替换和案例研究，并有自动解释评审与重建忠实度限制。没有发现上述工作直接证明 OpenMOSS 的某个语音、具身或 agent 模型因采用它们而更安全；可解释性资源与安全保证必须分写。这是对已有证据范围的判断，不是对未调查项目的否定。[Scope](https://arxiv.org/pdf/2410.20526)、[Lorsa 局限](https://proceedings.iclr.cc/paper_files/paper/2026/file/7e57d4c3bb2bd2396a551e7aeea19aed-Paper-Conference.pdf)、[CRM 局限](https://interp.open-moss.com/posts/complete-replacement)

### 5.2 BandPO：稳定探索的训练算法，不自动带来诚实／安全

`2603.04918` 分析 PPO／GRPO 固定 clipping：正优势但旧概率低的 token，其可增加概率约被 `epsilon × old_probability` 限制，容易过早裁剪。Band 将一般 f-divergence trust region 投影为依赖旧概率的上下界；KL 用数值求解，TV／Pearson χ² 有特定闭式形式。理论的全局最优指该边界投影的凸优化问题，不是保证整个 LLM RL 训练全局最优。[§4–5](https://arxiv.org/pdf/2603.04918)

在 Qwen2.5-3B 的三个数学测试集上，平均 mean@32／pass@32：GRPO 17.57／32.60，Clip-Higher 20.47／40.60，BandKL(δ=0.05) 22.00／42.01；DeepSeek-R1-Distill-Qwen-7B 的 mean@32 为 49.04→51.44。32 样本平均成功与至少一次成功不等同于单次确定性正确率。实验为 8×H200、三次重复；主要证据是数学 RLVR、熵与探索，不包括 agent 工具行为、幻觉校准或安全。单项也并非处处最优：7B AIME2025 的 pass@32，BandKL 0.05 为 57.21，0.03 为 61.46。[表 1、§6](https://arxiv.org/pdf/2603.04918)

仓库提供基于 verl 的 operator 与实验脚本，但模型／数据／checkpoint 大文件不直接随 Git 分发。可作为 OpenMOSS 后训练算法支线，不能据此宣称已成为所有旗舰模型的训练标准；未见该集成的共同实验或发布说明。[官方仓库](https://github.com/OpenMOSS/BandPO)、[全文](https://arxiv.org/pdf/2603.04918)

## 6. 两条边线及与总体路线的联系

### 6.1 只做身份识别，不延展边线实验

- **Hybrid-Mechanics 是长上下文主支的新增机制研究边线，不是与 LLM 无关的项目。** 论文 `2610.10114` 于 2026-10-07 提交、标注 under review，研究全注意力与 SWA／gated linear attention 混合及位置偏置协作。官方仓库当前仅可见 `.gitignore`、README、论文 PDF，未见可运行训练／评价代码；因此有论文，公开实现不足，不能称“无 paper”或“可复现模型发布”。[arXiv](https://arxiv.org/abs/2610.10114)、[全文](https://arxiv.org/pdf/2610.10114)、[仓库文件](https://github.com/OpenMOSS/Hybrid-Mechanics)
- **GSF-chi 是分子 AI／手性图学习跨域边线。** `2609.12532` 于 2026-09-11 提交，模型让手性场调制 query-key 旋转，读出区分镜像偶／奇分量，用于构型、光学旋转和 ECD。仓库有模型、配置和性质测试，数据与权重另行获取；它虽采用 Chiral-RoPE 术语，也不是文本长度外推、agent 安全或记忆评测，不能因同含 RoPE 就归成同一长上下文方案。[论文](https://arxiv.org/abs/2609.12532)、[全文](https://arxiv.org/pdf/2609.12532)、[仓库](https://github.com/OpenMOSS/GSF-chi)

### 6.2 有证据的“总体联系”与不能统一的地方

官方 OpenMOSS 总览将团队定位为覆盖架构、训练、评测、解释与应用的全栈研究，并分别列出工具／agent、解释性、长上下文、对齐／安全和多模态方向；MOSS 被列为 founding tool-augmented conversational model。该总览支持“组织级研究组合”的叙述，不支持“全部 repo 属于同一模型、同一团队成员或同一训练流水线”。[官方总览](https://github.com/sii-research/OpenMOSS)

| 可用于总报告的联系（分析） | 支持它的研究对象 | 不应升级成的说法 |
| --- | --- | --- |
| 从 tool-use 助手走向执行型工作评价 | 工具格式与调用评价→服务部署→科学软件→长期个人工作区。[UnifiedToolHub](https://github.com/OpenMOSS/UnifiedToolHub)、[ABC](https://arxiv.org/pdf/2601.11077)、[Science](https://arxiv.org/pdf/2608.19799v2)、[ContextWeave](https://arxiv.org/pdf/2608.04830) | 已有一个统一 OpenMOSS agent 在所有这些任务上领先 |
| 持续工作需要容量、历史使用及防错 | 有限 attention scope、跨会话记忆、安全长输入与知识拒答提供互补问题分解。[ReAttention](https://arxiv.org/pdf/2407.15176v3)、[ContextWeave](https://arxiv.org/pdf/2608.04830)、[LongSafety](https://arxiv.org/pdf/2411.06899v2)、[Idk](https://arxiv.org/pdf/2401.13275v2) | 长窗口自动替代记忆；记忆增益等于无幻觉；安全拒答等于工程完成 |
| 从外部失败分析到内部表示解释 | benchmark 诊断行为，SAE／Lorsa 重建内部表示与 attention 计算。[Science](https://arxiv.org/pdf/2608.19799v2)、[Scope](https://arxiv.org/pdf/2410.20526)、[Lorsa](https://arxiv.org/pdf/2504.20938) | 一套 circuit tracing 已解释并修复了 benchmark 的全部失败 |
| 架构、训练和评价互为研究支撑 | RoPE++ 改架构，BandPO 改更新约束，长上下文综述提供分类地图。[RoPE++](https://arxiv.org/pdf/2512.07525)、[BandPO](https://arxiv.org/pdf/2603.04918)、[综述](https://arxiv.org/pdf/2502.17129v2) | 三者已在同一旗舰模型上做了联合消融并证明协同增益 |

尤其不能统一三种“稀疏／top-k”：ReAttention 选择历史 KV 片段；Lorsa 选择稀疏解释单元；RoPE++ 利用实／虚分量组织 attention heads。相同词汇不表示相同机制或相同优化目标。[ReAttention §2](https://arxiv.org/pdf/2407.15176v3)、[Lorsa §3](https://arxiv.org/pdf/2504.20938)、[RoPE++ §3](https://arxiv.org/pdf/2512.07525)

建议总报告最终表述：**OpenMOSS 在本支线形成了“可执行评价—长程信息利用—行为可靠性—机制分析—训练优化”的互补研究组合；组织级关联明确，统一模型／联合训练／跨支线安全收益的证据仍不足。** 这是基于上述项目分别发表与分别评价的综合判断，不是作者公布的一条统一产品架构。[团队方向](https://github.com/sii-research/OpenMOSS)、[ABC](https://arxiv.org/pdf/2601.11077)、[ContextWeave](https://arxiv.org/pdf/2608.04830)、[LongSafety](https://arxiv.org/pdf/2411.06899v2)、[Lorsa](https://arxiv.org/pdf/2504.20938)、[BandPO](https://arxiv.org/pdf/2603.04918)

## 7. 轻量论文元数据（academic-search schema）

以下 JSON 使用 schema 的核心字段，另加 `source_urls` 明示一手核对入口。`year` 有正式发表时采用正式年份，否则采用首发预印本年份；`publication_date` 记录首发 arXiv 日期，因此可能早于正式 `year`。作者按来源列出的姓名／顺序：ReAttention、LongSafety 用正式发表记录，其余用 arXiv 或对应 proceedings。`open_pdf` 只表示合法公开 PDF 已找到／访问，不表示实验已独立复现。预印本的 venue 不表示拒稿或没有未来发表。

UnifiedToolHub 没有确认的专属论文，不强行构造缺标题／作者的“论文”；Llamascopium 的软件 citation 与 CRM 作者网页报告也不冒充 arXiv 文献。下列 17 条包括 Llamascopium 的四项直接核心／后续论文，以及仅作身份判定的 Hybrid 和 GSF。

```json
[
  {
    "title": "ABC-Bench: Benchmarking Agentic Backend Coding in Real-World Development",
    "authors": ["Jie Yang", "Honglin Guo", "Li Ji", "Jiazheng Zhou", "Rui Zheng", "Zhikai Lei", "Shuo Zhang", "Zhiheng Xi", "Shichun Liu", "Yuxin Wang", "Bo Wang", "Yining Zheng", "Tao Gui", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-01-16", "venue": "arXiv preprint", "arxiv_id": "2601.11077",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2601.11077",
    "code_url": "https://github.com/OpenMOSS/ABC-Bench", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2601.11077", "https://github.com/OpenMOSS/ABC-Bench"]
  },
  {
    "title": "SWE-bench Science: Can Coding Agents Resolve Engineering Tasks in Science?",
    "authors": ["Zhipeng Xu", "Jiahao Lu", "Yining Zheng", "Yuxin Wang", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-08-20", "venue": "arXiv preprint", "arxiv_id": "2608.19799",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2608.19799v2",
    "code_url": "https://github.com/OpenMOSS/SWE-bench-Science", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2608.19799", "https://swescience.github.io/"]
  },
  {
    "title": "AgentHPOBench: A Benchmark For Evaluating LLM Agents as Sequential Hyperparameter Optimizers",
    "authors": ["Tianyu Huai", "Tingshuo Fan", "Xinchi Chen", "Yining Zheng", "Yuxin Wang", "Shuang Chen", "Jie Zhou", "Xuanjing Huang"],
    "year": 2026, "publication_date": "2026-07-31", "venue": "arXiv preprint", "arxiv_id": "2607.29626",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2607.29626",
    "code_url": "https://github.com/OpenMOSS/AgentHPOBench", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2607.29626", "https://github.com/OpenMOSS/AgentHPOBench"]
  },
  {
    "title": "ContextWeave: A Real-World Workflow Benchmark",
    "authors": ["Bo Wang", "Yuqian Yao", "Enxi Wang", "Luozhijie Jin", "Yang Liu", "Yiran Suo", "Yuxuan Cai", "Enyu Zhou", "Yufei Gao", "Honglin Guo", "Tianyu Huai", "Li Ji", "Zhikai Lei", "Bufan Li", "Lizhi Lin", "Jinxiu Liu", "Jie Yang", "Jiazheng Zhou", "Maosen Zhou", "Pengfang Qian", "Shichun Liu", "Guanshan Liu", "Hao Zheng", "Yunhao Yu", "Hang Yan", "Jihua Kang", "Xinchi Chen", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-08-05", "venue": "arXiv preprint", "arxiv_id": "2608.04830",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2608.04830",
    "code_url": "https://github.com/OpenMOSS/ContextWeave", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2608.04830", "https://github.com/OpenMOSS/ContextWeave"]
  },
  {
    "title": "Evaluating Hallucinations in Chinese Large Language Models",
    "authors": ["Qinyuan Cheng", "Tianxiang Sun", "Wenwei Zhang", "Siyin Wang", "Xiangyang Liu", "Mozhi Zhang", "Junliang He", "Mianqiu Huang", "Zhangyue Yin", "Kai Chen", "Xipeng Qiu"],
    "year": 2023, "publication_date": "2023-10-05", "venue": "arXiv preprint", "arxiv_id": "2310.03368",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2310.03368v4",
    "code_url": "https://github.com/OpenMOSS/HalluQA", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2310.03368", "https://github.com/OpenMOSS/HalluQA"]
  },
  {
    "title": "Can AI Assistants Know What They Don't Know?",
    "authors": ["Qinyuan Cheng", "Tianxiang Sun", "Xiangyang Liu", "Wenwei Zhang", "Zhangyue Yin", "Shimin Li", "Linyang Li", "Zhengfu He", "Kai Chen", "Xipeng Qiu"],
    "year": 2024, "publication_date": "2024-01-24", "venue": "ICML 2024, PMLR 235:8184-8202", "arxiv_id": "2401.13275",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2401.13275v2",
    "code_url": "https://github.com/OpenMOSS/Say-I-Dont-Know", "source_platforms": ["arxiv", "pmlr", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2401.13275", "https://proceedings.mlr.press/v235/cheng24i.html"]
  },
  {
    "title": "LongSafety: Enhance Safety for Long-Context LLMs",
    "authors": ["Mianqiu Huang", "Xiaoran Liu", "Shaojun Zhou", "Mozhi Zhang", "Qipeng Guo", "Linyang Li", "Pengyu Wang", "Yang Gao", "Chenkun Tan", "Linlin Li", "Qun Liu", "Yaqian Zhou", "Xipeng Qiu", "Xuanjing Huang"],
    "year": 2025, "publication_date": "2024-11-11", "venue": "The First Workshop on LLM Security (LLMSEC 2025), pp. 26-47", "arxiv_id": "2411.06899",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2411.06899v2",
    "code_url": "https://github.com/OpenMOSS/LongSafety", "source_platforms": ["arxiv", "acl_anthology", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2411.06899", "https://aclanthology.org/2025.llmsec-1.4/"]
  },
  {
    "title": "ReAttention: Training-Free Infinite Context with Finite Attention Scope",
    "authors": ["Xiaoran Liu", "Ruixiao Li", "Zhigeng Liu", "Qipeng Guo", "Yuerong Song", "Kai Lv", "Hang Yan", "Linlin Li", "Qun Liu", "Xipeng Qiu"],
    "year": 2025, "publication_date": "2024-07-21", "venue": "ICLR 2025", "arxiv_id": "2407.15176",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2407.15176v3",
    "code_url": "https://github.com/OpenMOSS/ReAttention", "source_platforms": ["arxiv", "iclr_proceedings", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2407.15176", "https://proceedings.iclr.cc/paper_files/paper/2025/hash/ee1f0da706829d7f198eac0edaacc338-Abstract-Conference.html"]
  },
  {
    "title": "Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs",
    "authors": ["Xiaoran Liu", "Yuerong Song", "Zhigeng Liu", "Zengfeng Huang", "Qipeng Guo", "Zhaoxiang Liu", "Shiguo Lian", "Ziwei He", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2025-12-08", "venue": "ICLR 2026", "arxiv_id": "2512.07525",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2512.07525",
    "code_url": "https://github.com/OpenMOSS/rope_pp", "source_platforms": ["arxiv", "iclr_proceedings", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2512.07525", "https://proceedings.iclr.cc/paper_files/paper/2026/hash/cea5c4a422fd4e8b77fe3246559b34b1-Abstract-Conference.html"]
  },
  {
    "title": "Thus Spake Long-Context Large Language Model",
    "authors": ["Xiaoran Liu", "Ruixiao Li", "Mianqiu Huang", "Zhigeng Liu", "Yuerong Song", "Qipeng Guo", "Siyang He", "Qiqi Wang", "Linlin Li", "Qun Liu", "Ziwei He", "Yaqian Zhou", "Xuanjing Huang", "Xipeng Qiu"],
    "year": 2025, "publication_date": "2025-02-24", "venue": "arXiv survey", "arxiv_id": "2502.17129",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2502.17129v2",
    "code_url": "https://github.com/OpenMOSS/Thus-Spake-Long-Context-LLM", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2502.17129", "https://github.com/OpenMOSS/Thus-Spake-Long-Context-LLM"]
  },
  {
    "title": "Llama Scope: Extracting Millions of Features from Llama-3.1-8B with Sparse Autoencoders",
    "authors": ["Zhengfu He", "Wentao Shu", "Xuyang Ge", "Lingjie Chen", "Junxuan Wang", "Yunhua Zhou", "Frances Liu", "Qipeng Guo", "Xuanjing Huang", "Zuxuan Wu", "Yu-Gang Jiang", "Xipeng Qiu"],
    "year": 2024, "publication_date": "2024-10-27", "venue": "arXiv preprint", "arxiv_id": "2410.20526",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2410.20526",
    "code_url": "https://github.com/OpenMOSS/Llamascopium", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2410.20526", "https://github.com/OpenMOSS/Llamascopium"]
  },
  {
    "title": "Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition",
    "authors": ["Zhengfu He", "Junxuan Wang", "Rui Lin", "Xuyang Ge", "Wentao Shu", "Qiong Tang", "Junping Zhang", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2025-04-29", "venue": "ICLR 2026", "arxiv_id": "2504.20938",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2504.20938",
    "code_url": "https://github.com/OpenMOSS/Llamascopium", "source_platforms": ["arxiv", "iclr_proceedings", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2504.20938", "https://github.com/OpenMOSS/Lorsa", "https://proceedings.iclr.cc/paper_files/paper/2026/hash/7e57d4c3bb2bd2396a551e7aeea19aed-Abstract-Conference.html"]
  },
  {
    "title": "Dimensional Collapse in Transformer Attention Outputs: A Challenge for Sparse Dictionary Learning",
    "authors": ["Junxuan Wang", "Xuyang Ge", "Wentao Shu", "Zhengfu He", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2025-08-23", "venue": "ICML 2026, PMLR 306:126304-126331", "arxiv_id": "2508.16929",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2508.16929v4",
    "code_url": "https://github.com/OpenMOSS/Llamascopium", "source_platforms": ["arxiv", "pmlr", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2508.16929", "https://proceedings.mlr.press/v306/wang26bh.html"]
  },
  {
    "title": "Evolution of Concepts in Language Model Pre-Training",
    "authors": ["Xuyang Ge", "Wentao Shu", "Jiaxing Wu", "Yunhua Zhou", "Zhengfu He", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2025-09-21", "venue": "ICLR 2026", "arxiv_id": "2509.17196",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2509.17196v2",
    "code_url": "https://github.com/OpenMOSS/Llamascopium", "source_platforms": ["arxiv", "iclr_proceedings", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2509.17196", "https://proceedings.iclr.cc/paper_files/paper/2026/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html"]
  },
  {
    "title": "BandPO: Bridging Trust Regions and Ratio Clipping via Probability-Aware Bounds for LLM Reinforcement Learning",
    "authors": ["Yuan Li", "Bo Wang", "Yufei Gao", "Yuqian Yao", "Xinyuan Wang", "Zhangyue Yin", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-03-05", "venue": "arXiv preprint", "arxiv_id": "2603.04918",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2603.04918",
    "code_url": "https://github.com/OpenMOSS/BandPO", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2603.04918", "https://github.com/OpenMOSS/BandPO"]
  },
  {
    "title": "Mechanics of Long-Context Hybrid Models Part 1.1: From Hybrid Attention to Hybrid Position",
    "authors": ["Xiaoran Liu", "Ziwei He", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-10-07", "venue": "arXiv preprint; under review", "arxiv_id": "2610.10114",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2610.10114",
    "code_url": "https://github.com/OpenMOSS/Hybrid-Mechanics", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2610.10114", "https://github.com/OpenMOSS/Hybrid-Mechanics"]
  },
  {
    "title": "GSF-χ: Global Stereochemical Fields for Chiral Graph Transformers",
    "authors": ["Jiaqing Xie", "Yuxin Wang", "Xipeng Qiu"],
    "year": 2026, "publication_date": "2026-09-11", "venue": "arXiv preprint", "arxiv_id": "2609.12532",
    "citation_count": null, "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2609.12532",
    "code_url": "https://github.com/OpenMOSS/GSF-chi", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17",
    "source_urls": ["https://arxiv.org/abs/2609.12532", "https://github.com/OpenMOSS/GSF-chi"]
  }
]
```

## 8. 整合时不要丢失的审计标记

- 数字出处优先级：论文具体表格／正文 > 摘要或 README 概括；当前官方榜单另列日期，不回填原论文。Science 当前榜单与论文差异、Idk README 的 BoN／SFT 错配、ReAttention 的 4M 模型名称冲突、LongSafety 8／10 项任务冲突、综述与 ASI 旧题名／作者均已在上文保留出处。
- “未确认专属论文”只用于 UnifiedToolHub 和 CRM 网页报告的本次识别范围，不表示全网不存在；Hybrid-Mechanics 有 `2610.10114`，不得判为无论文。[UnifiedToolHub](https://github.com/OpenMOSS/UnifiedToolHub)、[CRM](https://interp.open-moss.com/posts/complete-replacement)、[Hybrid](https://arxiv.org/abs/2610.10114)
- “复现不完整”与“无科研价值”不是同义词：LongSafety 训练配置／权重缺口、Hybrid 无可见训练代码、ReAttention 部分 kernel／效率发布 TODO、Science 隐藏测试源码，具体缺口不同，按项目分别陈述。[LongSafety](https://github.com/OpenMOSS/LongSafety)、[Hybrid](https://github.com/OpenMOSS/Hybrid-Mechanics)、[ReAttention](https://github.com/OpenMOSS/ReAttention)、[Science](https://github.com/OpenMOSS/SWE-bench-Science)
- 不使用 GitHub star、README 热度、“first”宣传词或模型名称新旧为科研价值证据；本报告的价值判断只依赖问题设定、实证、机制证据、发布完整度与明确的外推边界。
