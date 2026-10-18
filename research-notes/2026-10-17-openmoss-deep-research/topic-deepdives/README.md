# OpenMOSS 研究主线与技术专题

按研究问题划分方向，再在每个方向下深入讲代表工作。各篇从背景和出发点讲起，接着分析表示、训练或推理方法，最后解释实验和源码能支持哪些结论。可以直接选择感兴趣的方向。

| 研究方向 | 背景与出发点 | 深入阅读 |
|---|---|---|
| 基础语言模型与统一多模态 | 预训练模型怎样学会对话、用工具，并接收和生成不同模态？ | [MOSS 与 AnyGPT：对话监督、插件协议、离散接口和两阶段对齐](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/09-foundation-and-omni-models.md) |
| 语音生成与交互 | 语义、声线和说话人需要不同信息；高保真与低延迟怎样兼顾？ | [SpeechGPT／MOSS-Speech、CAT、TTS 与 TTSD：表示和生成顺序](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/01-audio-representation-and-generation.md) |
| 音频理解与结构化转写 | 听懂内容、辨认声音、解释音乐与标记说话人，为什么不能只靠转写？ | [MOSS-Audio、MTD 与 MOSS-Music：连续表示、长音频和联合标注](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/10-audio-understanding-and-transcription.md) |
| 视觉与视频理解 | 离线感知怎样扩展到持续输入，模型何时应该发言？ | [MOSS-Video-Preview 与 MOSS-VL：感知训练、时间、缓存和互动](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/02-streaming-video-time-and-cache.md) |
| 多模态生成 | 语义匹配不保证音画同步；文字要求也不保证像素中的字和布局正确。 | [OmniVAE／MOVA：表示对齐与联合去噪](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/03-audiovisual-latents-and-joint-denoising.md)；[MOSS-Image：文字、布局监督与规格执行](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/04-image-layout-supervision-and-specification.md) |
| 具身智能 | 理解意图、生成可执行动作、预测后果和验证结果，分别需要什么？ | [RoboOmni／FRoM-W1：机器人策略与全身控制](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/11-robotics-policy-and-control.md)；[D2PO／WAP／EasyWAM／ETA：世界建模、规划和执行证据](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/05-world-action-and-embodied-agent.md) |
| 工具使用与数字 Agent | 会写调用、会用历史经验，与完成实际工作之间还差什么？ | [工具训练、ABC、ContextWeave、SWE-bench Science 与 AgentHPOBench：执行和实验决策](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/06-agent-memory-and-executable-evaluation.md) |
| 模型架构与生成范式 | 能容纳更长输入不等于能用好远处证据；并行生成也不自动省计算。 | [ReAttention／RoPE++／LongLLaDA：长度、位置和检索](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/07-long-context-and-diffusion-optimization.md)；[Sparse-dLLM：遮盖生成与缓存近似](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/13-diffusion-language-models.md) |
| 后训练与强化学习 | 任务反馈怎样更新模型，训练概率是否对应真实生成过程？ | [DiRL／DiPO、BandPO 与 Planner-R1：轨迹、更新区间和奖励](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/14-post-training-and-rl.md) |
| 诚实性与安全对齐 | 事实错误、不知道答案和不应回答，为什么需要分别研究？ | [HalluQA、Idk 与 LongSafety：拒答、覆盖率和长上下文安全](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/12-honesty-and-long-context-safety.md) |
| 机制可解释性 | 能解释某个特征，是否已经解释了模型怎样计算？ | [Llama Scope、Lorsa 与 CRM：稀疏表示、计算替换和因果检验](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/08-mechanistic-interpretability.md) |

这个划分依据[组织项目目录](https://github.com/OpenMOSS)和[团队公开研究总览](https://github.com/sii-research/OpenMOSS/blob/14d3cf31265a5d3de9dea89a1d0857ea3d976579/README.md)，是便于阅读的研究分类，不是官方部门划分。D2PO 和 World-Aware-Planning 虽在组织外的仓库，但由官方总览列入，因此纳入分析；不按共同作者无限扩张范围。

训练框架、数据管线、评测和部署适配贯穿各方向。CoLLiE 放在基础模型篇，VLABench 放在机器人篇，GAOKAO-MM／FutureOmni 放在感知篇；它们支持研究与验证，不另算一种模型路线。MOSS-Image 也作为多模态生成的深入案例，而非与整个具身或语音方向并列。

从这些代表工作可见，团队持续研究的是怎样保留模态信息、表达时间和条件，并把目标行为变成可训练、可检验的任务。具体方案包括离散 token、连续编码器、跨注意力和潜空间扩散，不能归结为“所有模态都塞进同一个 LLM”，也没有证据证明各项目已经整合为一个通用模型。[AnyGPT](https://aclanthology.org/2024.acl-long.521/)、[MOSS-Audio](https://arxiv.org/abs/2606.01802)、[MOSS-VL](https://arxiv.org/abs/2608.15045)、[MOVA](https://arxiv.org/abs/2602.08794)

材料核查于 2026-10-17。覆盖公开主线及其代表工作，不逐篇穷尽团队全部论文。实验分数来自作者，计算例和反例为分析，不是独立模型实验；源码版本与论文实验不一致的地方，在各篇说明。
