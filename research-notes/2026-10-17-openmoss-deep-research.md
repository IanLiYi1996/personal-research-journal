# OpenMOSS 深度调研：项目、论文与技术路线

调研截止：2026-10-17（北京时间）。本报告是对公开材料的技术分析，不是 OpenMOSS 官方路线图，也不是独立性能复现。

具体方法与源码分析见[研究主线与技术专题目录](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/README.md)。目录按研究问题分层，补入基础模型、音频理解、机器人控制和安全对齐，并将长上下文、扩散推理与后训练分开。本文保留项目版图综述，方法细节以专题中的论文版本和固定源码为依据。

配套材料：[语音／音频专题](/research-notes/2026-10-17-openmoss-deep-research/audio-primary-findings.md) · [具身／世界模型专题](/research-notes/2026-10-17-openmoss-deep-research/embodied-primary-findings.md) · [Agent／可靠性专题](/research-notes/2026-10-17-openmoss-deep-research/agents-reliability-primary-findings.md) · [核心论文 JSON](/research-notes/2026-10-17-openmoss-deep-research/core-papers-metadata.json)。结合组织的公开项目清单与约 45 项去重论文记录；论文阅读全文、元数据核验和项目源码核验的证据深度分别标注，未宣称每篇都开展了完整复现。

## 1. 核心判断

**OpenMOSS 更适合被理解为“以多模态表示为基础、面向实时交互和具身行动的研究／开源技术群”，而不是一个持续升级同一文本基座的模型品牌。** 2023 年的 MOSS、2024 年的 AnyGPT，以及 2026 年的 MOSS-VL、MOVA、OmniVAE 和 MOSS-Image，并不共享同一个骨干或同一种生成范式。[组织主页](https://github.com/OpenMOSS)、[AnyGPT](https://github.com/OpenMOSS/AnyGPT)、[MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)、[MOVA](https://github.com/OpenMOSS/MOVA)、[OmniVAE](https://github.com/OpenMOSS/OmniVAE)、[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)。

五个值得优先把握的判断：

1. **路线从“接口统一”深化到“表示可学习性与时序对齐”。** AnyGPT 用离散序列统一不同模态；MOSS-VL 通过跨注意力解耦视觉与语言；OmniVAE 把语义和音画时间对应关系前移到压缩表示的训练中。共同问题不是“是否都用 Transformer”，而是“如何保留信息，让下游模型更容易学会跨模态关系”。[AnyGPT 论文](https://arxiv.org/abs/2402.12226)、[MOSS-VL 论文](https://arxiv.org/abs/2608.15045)、[OmniVAE 论文](https://arxiv.org/abs/2607.23855)。
2. **统一的目标，不等于统一的架构。** 文本／语音可以采用自回归或离散去噪；图像／视频采用连续潜空间扩散；实时视频理解采用独立视觉表示和文本查询。把这种组合归纳为“全部模态 token 化后塞进一个 LLM”已不足以解释当前项目。[AnyGPT](https://github.com/OpenMOSS/AnyGPT)、[DiRL](https://github.com/OpenMOSS/DiRL)、[MOVA](https://github.com/OpenMOSS/MOVA)、[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)。
3. **实时性是系统约束，不只是推理加速指标。** 未来帧不可见、视频到达时答案可能正在生成、模型需要决定何时发言，这些都改变训练数据、注意力掩码、缓存与服务调度。[MOSS-VL 报告](https://arxiv.org/html/2608.15045v1)、[实时推理说明](https://github.com/OpenMOSS/MOSS-VL/tree/main/inference/realtime)、[生产服务说明](https://github.com/OpenMOSS/MOSS-VL/tree/main/third_party)。
4. **“模型＋数据／训练配方＋评测／部署”的组合，比单一榜单成绩更能说明其研究风格。** 但各项目的开放程度不同，必须逐项核查，不能用组织的“开源”标签替代模型、数据和第三方依赖的许可证。[MOSS 的三种许可证](https://github.com/OpenMOSS/MOSS)、[MOVA 训练与评测](https://github.com/OpenMOSS/MOVA)、[OmniVAE 三模块代码](https://github.com/OpenMOSS/OmniVAE)。
5. **“实时多模态 → 世界模型 → 行动 → 持续学习”可以作为组合路线的分析框架，不能当成已经完成的统一闭环。** 音画生成的同步性并不自动证明反事实动力学能力；机器人实验也不自动证明无监督自主成长。具身分支见后续章节，所有战略连接均标为分析推断。

## 2. 调研边界与证据规则

公开 GitHub API 在本次查询返回 **69 个公开仓库**，包括模型、数据集、评测、演示、课程和第三方适配仓库。这个数字是查询时点的库存，不是“69 个独立原创模型”。API 的 `fork` 字段也不完整表达衍生关系：例如 `sglang`、`llama.cpp` 的介绍明确是 OpenMOSS 适配版，但其字段不一定标记为 fork。[仓库清单 API](https://api.github.com/orgs/OpenMOSS/repos?per_page=100&type=public)、[组织项目分类](https://github.com/OpenMOSS)。

官方组织主页将 OpenMOSS 描述为由邱锡鹏教授在上海创智学院／Shanghai Innovation Institute（SII）领导，与复旦大学及 MOSI.AI 协作的研究团队。论文的具体署名机构因项目不同而变化；GitHub 组织、论文协作网络与商业主体的法律／资产关系不能仅凭署名推定。[官方组织介绍](https://github.com/OpenMOSS)。

本报告采用以下优先级：

- 论文正文和明确的实验协议：判断方法、数据与能力边界；
- 当前官方仓库、源码、配置和模型卡：判断真正发布了什么；
- 官方项目页：补充发布历史、演示和愿景；
- 二手报道不作为核心技术结论的依据。

特别核查了当前仓库和论文之间的差异。例如 MOSS-VL 技术报告已于 2026 年 8 月发布，当前原始 README 也已链接报告；不能根据旧版本或搜索缓存的未完成 TODO 判断其仍无报告。[当前 README](https://raw.githubusercontent.com/OpenMOSS/MOSS-VL/main/README.md)、[论文](https://arxiv.org/abs/2608.15045)。

未开展 GPU／机器人实测，未复现任何排行榜。文中成绩均为作者报告；不依据 stars 衡量科研水平；未查询论文引用数。没有找到正式论文、没有公开数据或未见独立复现时，均保留这一证据边界。

## 3. 项目版图与路线演进

### 3.1 先分层，再看项目

| 研究方向 | 代表工作 | 应当回答的问题 |
|---|---|---|
| 基础语言模型与统一多模态 | MOSS、AnyGPT | 对话、工具协议与跨模态接口如何学得？ |
| 语音生成与交互 | MOSS-Speech、CAT、MOSS-TTS、MOSS-TTSD | 内容、声线、多人一致性和延迟如何取舍？ |
| 音频理解与转写 | MOSS-Audio、MOSS-Music、MTD | 声音语义、音乐结构与说话人时间标记如何表示？ |
| 视觉与视频理解 | MOSS-Video-Preview、MOSS-VL | 离线感知怎样扩展为持续观察与及时发言？ |
| 多模态生成 | OmniVAE、MOVA、MOSS-Image | 音画同步、精确文字和空间布局如何进入监督？ |
| 具身智能 | RoboOmni、FRoM-W1、D2PO、WAP、EasyWAM、OpenETA | 意图、控制、后果预测和执行验证怎样衔接？ |
| 工具使用与数字 Agent | UnifiedToolHub、ABC、ContextWeave、SWE-bench Science、AgentHPOBench | 怎样训练调用与工作轨迹，确认任务完成并利用实验反馈？ |
| 模型架构与生成范式 | ReAttention、RoPE++、LongLLaDA、Sparse-dLLM | 长度、检索、生成顺序和计算复用分别改善什么？ |
| 后训练与强化学习 | DiRL／DiPO、BandPO、Embodied-Planner-R1 | 轨迹、奖励和概率更新如何对应实际过程？ |
| 诚实性与安全对齐 | HalluQA、Say-I-Dont-Know、LongSafety | 事实错误、知识拒答和有害请求怎样分别评价？ |
| 机制可解释性 | Llamascopium、Llama Scope、Lorsa、CRM | 表示解释和计算机制解释有什么区别？ |

表格与[技术专题目录](/research-notes/2026-10-17-openmoss-deep-research/topic-deepdives/README.md)采用同一分类。CoLLiE、评测、数据管线和部署适配作为跨方向支撑，不与模型路线混为一层；MOSS-Image 是生成方向的案例。划分不代表正式部门，也不证明项目已经共享接口。D2PO 和 WAP 由官方总览列入，虽不在 OpenMOSS 组织内，仍纳入研究范围。[组织目录](https://github.com/OpenMOSS)、[团队总览](https://github.com/sii-research/OpenMOSS/blob/14d3cf31265a5d3de9dea89a1d0857ea3d976579/README.md)。

### 3.2 时间线的技术含义

**2023：先把“对话＋工具”做成可用的开放系统。** MOSS-003 是 16B 中英双语模型，仓库给出基座、SFT、plugin、量化版本及多轮对话数据。插件版把搜索、图片生成、计算器和方程求解纳入模型训练和执行路径。其约 110 万条 SFT 和约 30 万条工具增强对话均是作者披露，不应把这里的“工具调用”直接等同于今天复杂的长程自治 Agent。[MOSS 仓库](https://github.com/OpenMOSS/MOSS)。

相关正式论文是 Tianxiang Sun 等的 *MOSS: An Open Conversational Large Language Model*，发表于 *Machine Intelligence Research*，Crossref 记录在线发行日期 2024-05-20、DOI `10.1007/s11633-024-1502-8`。模型发布年和论文发表年应分开。早期工程基础还包括 CoLLiE（EMNLP 2023 System Demonstrations），以 DeepSpeed／PyTorch 为基础组合 DP／PP／TP／ZeRO、LoRA／LOMO 与指标监测；它是训练工具，不是一个新基座模型，也不能据此推定今天所有模型仍在用 CoLLiE。[MOSS 论文元数据](https://api.crossref.org/works/10.1007/s11633-024-1502-8)、[出版商入口](https://link.springer.com/article/10.1007/s11633-024-1502-8)、[CoLLiE 论文](https://arxiv.org/abs/2312.00407)、[CoLLiE 实现](https://github.com/OpenMOSS/CoLLiE)。

**2024：把统一对象从文本扩展到多模态。** AnyGPT 的重要性在于不改造基本语言模型，就通过专用 tokenizer 把语音、图像、音乐编码为离散序列，建立任意模态输入输出的接口。HalluQA、知识边界研究和可解释性工具则说明团队并非只做生成能力。[AnyGPT 论文](https://arxiv.org/abs/2402.12226)、[HalluQA](https://github.com/OpenMOSS/HalluQA)、[Say-I-Dont-Know](https://github.com/OpenMOSS/Say-I-Dont-Know)、[Llamascopium](https://github.com/OpenMOSS/Llamascopium)。

**2025：沿“长对话／长上下文／行动／新采样范式”多线展开。** MOSS-TTSD、MOSS-Speech、VLABench、RoboOmni、LongLLaDA、DiRL 不是单一版本迭代，而是在不同任务上探索是否需要不同表示、时序建模和训练目标。[组织目录](https://github.com/OpenMOSS)、[LongLLaDA](https://arxiv.org/abs/2506.14429)、[DiRL](https://arxiv.org/abs/2512.22234)。

**2026：模块组合与工程化显著增强。** Audio Tokenizer、MOSS-TTS 家族和 Nano 提供从高质量到端侧的不同层级；MOSS-VL 支持连续视觉流；MOVA 和 OmniVAE 分别在生成器层与表示层处理音画同步；MOSS-Image 进一步展示“外部 Agent 做规格规划，自有生成器执行”的系统分工。这里的“系统化”是公开产物的分析，不是对组织商业战略的断言。[MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)、[MOVA](https://github.com/OpenMOSS/MOVA)、[OmniVAE](https://github.com/OpenMOSS/OmniVAE)、[MOSS-Image 技术报告](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

## 4. 多模态表示：从 AnyGPT 到 OmniVAE

### 4.1 AnyGPT：统一离散接口，而非一个模型包办所有信号处理

AnyGPT 的主张是 **“模态专属编解码器＋离散序列语言模型”**。语音用 SpeechTokenizer，声学／副语言细节的补全依赖 SoundStorm；图像使用 SEED tokenizer 和图像解码组件；音乐用 EnCodec。语言模型主要学习这些离散单元与文本之间的关联与组合。[论文](https://arxiv.org/abs/2402.12226)、[仓库的下载与依赖说明](https://github.com/OpenMOSS/AnyGPT)。

训练分为多模态对齐和对话指令学习；AnyInstruct 为合成的多模态对话／指令数据。其研究价值是把多种任务转成统一接口，并探索多模态对话数据怎样构造，不是证明离散自回归在图像保真度、音频码率和流式响应上均最优。[论文全文](https://arxiv.org/html/2402.12226v1)、[官方数据集](https://huggingface.co/datasets/fnlp/AnyInstruct)。

**分析：** 此路线有三个内在代价：模态专属 tokenizer 的信息瓶颈、序列长度与自回归成本、token 级统一却不保证时间／空间对应关系自然形成。这解释了后续项目为何分别引入更强的音频 tokenizer、跨注意力视频模型和联合训练 VAE；但没有证据表明 AnyGPT 的模型权重被直接迁入所有后续系统。

### 4.2 OmniVAE：让“压缩得好”与“下游容易学”同时成立

很多 VAE 优化的是重建：压缩后能否还原原始信号。OmniVAE 问的是更进一步的问题：**重建精确的潜空间，是否也是生成器容易建模、两个模态容易同步的潜空间？** 论文认为，仅有重建目标不能保证这两点。[OmniVAE 论文](https://arxiv.org/abs/2607.23855)。

其设计保留独立音频和视频编解码分支：视频基于 Wan2.2 VAE，音频为 DAC 风格的连续 VAE。训练时额外加入两类目标：

- 对齐时间片段的双向 InfoNCE：匹配同一视频内对应音视频片段，并使用时间内／跨样本负例；
- 模态内语义蒸馏：通过冻结的 Qwen3-Omni 视觉／音频编码器，向对应潜变量注入语义结构。

推理时不用这些对比学习头和蒸馏投影头，两个 VAE 可以独立执行。这不是“所有模态共用一个编码器”，而是**独立、可解码的潜空间通过联合训练变得语义与时间可对齐**。[论文第 3 节](https://arxiv.org/html/2607.23855v2)、[源码与三模块结构](https://github.com/OpenMOSS/OmniVAE)。

可以将其目标概括为：

`L = L_reconstruction + λ_semantic·L_distillation + λ_alignment·L_segment_InfoNCE`

这是说明思路的简式；实际训练还包含各分支的 KL、对抗和重建项及训练阶段安排，不能据此复刻全部训练。

**分析：** 这是 OpenMOSS 路线中很值得研究的变化：把跨模态同步的部分负担从大生成器前移到 tokenizer／VAE。它有机会降低下游学习难度，但“无额外推理头”不代表训练免费，也不代表任何新模态或任何生成器都会受益。

## 5. MOSS-VL：实时视频的关键在注意力与时间语义

### 5.1 不是把更多帧塞进上下文

MOSS-VL 系列包括 Base、离线 Instruct 和 Realtime；官方列出 11B 参数与 256K 上下文。报告采用 Qwen3-8B 语言骨干与 Qwen3-VL 视觉编码器，插入跨注意力层连接视觉表示与文本状态，而不是在所有语言自注意力层把视觉 token 与文本 token 简单拼接。[论文](https://arxiv.org/abs/2608.15045)、[当前模型发布表](https://github.com/OpenMOSS/MOSS-VL)、[0708 模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-VL-Instruct-0708)。

报告的语言解码栈由 36 个文本自注意力层和 12 个跨注意力层组成。文本执行自己的因果推理，视觉特征存放在独立 KV 表示中供查询。这有利于新增视频帧和已生成文本缓存之间的解耦。[论文架构部分](https://arxiv.org/html/2608.15045v1)。

**分析：** 在忽略编码器内部开销、层数与实现常数的简化比较中，拼接注意力的大块成本近似 `O((T+V)^2)`，解耦后主要是文本 `O(T^2)` 和跨模态 `O(TV)`。但视觉编码、视觉缓存增长和跨注意力仍有成本，不能把这种设计说成“无限视频、固定内存、全系统线性复杂度”。

### 5.2 从离线回答到连续状态更新

Realtime 训练必须让文本只能访问对应时间点之前已经到达的视觉内容。模型不仅回答问题，还要决定是否发出主动提醒、什么时候沉默、什么时候根据新帧修正认知。论文用具有时间对应关系的数据和注意力掩码处理这些问题，不能仅靠离线 QA 模型加一个摄像头解决。[技术报告](https://arxiv.org/html/2608.15045v1)、[实时官方说明](https://github.com/OpenMOSS/MOSS-VL/tree/main/inference/realtime)。

实现这一行为的训练设计值得单独看：每个到达帧对应一个决策槽，用 `<|silence|>` 和 `<|response|>` 两个状态 token 与普通文字交替；发言／等待仍是 next-token prediction，而不是新增一个决策头。沉默槽占多数，统一交叉熵会促使模型一直沉默，因此报告使用 focal factor 与全局批次的逆类别频率权重；另外不监督助手回合末尾的 end token，以免用户插话使模型误学为“会话结束”。Realtime-SFT 的训练 token 不到总量 3%，但不是仅靠修改 system prompt 获得能力，也没有在本报告中使用 RL。[报告第 4 节](https://arxiv.org/html/2608.15045v1)。

为避免长序列中稠密 `T×V` 可见性 mask 的内存成本，团队修改 FlashAttention-3：`cross_kv_boundary` 用每个 query 一个 32-bit 边界表达可以看到的视觉 KV 前缀，超界 tiles 直接跳过。这里“每行的可见前缀”正是时序语义与算子优化相结合的例子。[报告第 5 节](https://arxiv.org/html/2608.15045v1)、[发布的 FlashAttention 衍生代码](https://github.com/OpenMOSS/MOSS-VL/tree/main/third_party/flash-attention-src)。

源码和工程发布提供了更强证据：`inference/offline` 与 `inference/realtime` 分离；实时接口包括 `create_realtime_session(...)`、`online_generate(...)`；参考 Transformers 进程默认一次服务一个会话，SGLang-Omni 版本则提供动态多会话调度和副本扩展；完整 demo 另外集成记忆、ASR 与 TTS。[推理总览](https://github.com/OpenMOSS/MOSS-VL/blob/main/inference/README.md)、[后端目录](https://github.com/OpenMOSS/MOSS-VL/tree/main/third_party)。

**判断：** 这一方向的可迁移技术价值比“在某个视频榜单超过谁”更高：感知流和生成流的解耦、时间可见性掩码、事件驱动的发言策略、连续状态和缓存管理。但当前的交互和主动发言成绩，不能直接等同于具身行动的成功率、长期记忆正确率或安全预警可靠性。

报告自身还保留了一个关键边界：它称“生成时继续感知”为 L5，但公开量化评测覆盖 L2–L4，L5 主要由演示和源码定性证明，专门基准尚缺。离线能力也不是全面领先：例如其比较表中 MMMU 为 51.1，而 Qwen3-VL-8B 为 69.6；报告说明 baseline 的采帧等设置可能不同。这说明路线倾向于实时和时间推理，不能扩写为“最强通用 VLM”。[报告表 1、表 5 与讨论](https://arxiv.org/html/2608.15045v1)。

## 6. MOVA：把音画同步放到生成过程，而不是后期拼接

MOVA 在双分支模型里联合生成视频和音频。报告描述 **32B 总参数、18B 激活参数**：视频塔采用 Wan2.2 I2V A14B 的高／低噪声专家，音频塔为自行预训练的 1.3B text-to-audio DiT，另有 2.6B 双向 Bridge。A14B 的“14B”是视频专家的激活规模，不能误解为 18B 视频＋14B 音频。模型通过 Bridge 跨注意力联合去噪，配合时间一致的数据和分阶段训练建立同步关系。[MOVA 论文](https://arxiv.org/abs/2602.08794)、[论文正文](https://arxiv.org/html/2602.08794v2)、[仓库](https://github.com/OpenMOSS/MOVA)。

其关键不只是“两个网络都很大”，而是让视觉运动、嘴形、音频内容和发生时间在生成过程中相互约束。Aligned RoPE 将不同频率的音视频 latent 映射到共同时间尺度；训练独立采样两个模态的噪声时间，并可选择不同 sigma shift；双 CFG 分别调节**文本条件和跨模态 Bridge 条件**的引导，而非简单“各调一条音量”。对同步性的判断要看音画对齐、唇形、转写与说话人切换等指标，而非仅看一张生成视频截图。[报告](https://arxiv.org/html/2602.08794v2)、[公开评测实现](https://github.com/OpenMOSS/MOVA/tree/main/evaluation)。

报告的消融也显示真正的取舍：增强跨模态 guidance 能改善同步／唇形，但 cpCER 变差，说明“嘴动得更对”可能牺牲“谁说了什么”的正确性。多目标质量不是单个同步指标越高越好。[报告表 4、表 5](https://arxiv.org/html/2602.08794v2)。

当前仓库不止提供推理：有 LoRA 训练、多 GPU、低资源配置和评测。低显存可运行不能混同于低成本训练：作者在 8 秒 360p LoRA 设置下给出 RTX 4090 约 18GB VRAM、约 80GB 主机内存、每步约 600 秒，并明确不建议这种设置；其 8×H100 设置报告约 22.2 秒／步。它们是特定工作负载的作者测量，不应外推为任何分辨率或任何硬件的速度。[资源表与限制](https://github.com/OpenMOSS/MOVA#lora-fine-tuning)。

**与 OmniVAE 的联系：** MOVA 在生成器层解决交互，OmniVAE 在表示层改善对应关系，二者互补。但当前证据并不表明已发布的 MOVA 检查点直接换用了 OmniVAE；不要把“同一个团队的相关工作”写成“已验证的一条产品依赖链”。

**与世界模型的边界：** 联合音画生成能学到时间相关性，不等于能回答“机器人采取某动作后，会发生什么”的受控反事实。动作条件、状态转换的物理一致性、闭环规划收益和长时滚动误差，必须另外验证。

## 7. MOSS-Image：结构化规格规划与渲染能力的分工

这是 2026-10-16 新创建的仓库，调研时已公开 Base／SFT、Diffusers／Gradio／ComfyUI 路径与独立 PDF 技术报告。其新近程度意味着材料和实现仍可能快速变化。[仓库](https://github.com/OpenMOSS/MOSS-Image)、[报告 PDF](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)、[模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base)。

### 7.1 5B 指的是 DiT，不是整个系统的所有参数

报告使用从头训练的 5B single-stream DiT，但冻结文本编码器是 Qwen3.5-9B，图像潜表示来自 FLUX.2 VAE。其全系统内存与依赖不能只按“5B”估算。发布配置可核查 42 层、3072 隐藏维度、24 个头、4096 文本特征维度与 128 latent channels；ComfyUI 源码能看到文本投影、两层 text-only Transformer、joint attention、共享调制和 4D RoPE。[报告架构节](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)、[发布配置](https://huggingface.co/OpenMOSS-Team/MOSS-Image-Base/blob/main/transformer/config.json)、[ComfyUI 模型源码](https://github.com/OpenMOSS/MOSS-Image/blob/main/comfyui/moss-image/model.py)。

### 7.2 技术重点主要在数据、监督和接口

报告强调：去重和剔除 AI 生成图像后，保留广泛真实数据，不再额外按审美／质量统一筛掉样本；残留水印、模糊和压缩瑕疵通过 caption 标注。这个政策有前提：原始来源可能已经做过 NSFW 等过滤，不能简化为“完全不筛数据”。[报告第 3 节](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

监督包含短／中／长 caption、多行文本的精确转写和换行、对象／文字的 bbox 与结构化布局；训练逐渐扩展分辨率和文本容量。布局 JSON 与自然语言经同一文本条件接口输入，而不是另加一个必须单独安装的外部布局控制网络。Muon–AdamW 的消融也属于配方，而非模型新范式。[技术报告](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

**分析：** “能写长文字”并不只是模型参数更大；训练时知道每行写什么、哪里写、标注如何与图像几何一致，是更直接的原因。但报告并没有完全隔离所有数据策略的因果贡献，不应把某项配方单独说成已证明的唯一原因。

### 7.3 Agent 的能力不能计入自有 DiT 参数

布局增强器使用外部 GPT-6 和模块化任务指导，可检索知识／视觉参考，生成文字、层级、bbox 和版式 JSON；本地 skill 和 `compile_prompt.py` 负责坐标校验、转换与序列化。视觉参考用于 Agent 的规划，并非直接作为 DiT 图像条件。[报告第 4 节和限制节](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)、[发布的 prompt-enhancer skill](https://github.com/OpenMOSS/MOSS-Image/tree/main/skills/moss-image-prompt-enhancer)。

因此，系统是“意图／事实／版式规格规划 → 生成器执行”。它不是自有模型独立完成所有推理，也不是现已支持图生图编辑。报告明确限制为文生图，当前不直接接受参考图，也不提供原图条件下的编辑；所谓 zero-shot modification 是修改布局后重新生成。[技术报告限制节](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

### 7.4 读成绩时应如何保留边界

作者新建 LLTB：240 个中英文长文本任务，平均目标约 1339.8 字符；其 Mean-6 覆盖率为 96.3。指标复用 OCR 与词／字符 multiset recall，衡量覆盖，不等于逐字逐序无错、语义事实正确或排版审美满分。RLA 有 61 个海报／信息图／UI 任务，通过盲评 pairwise preference 拟合 Elo。[报告实验节与附录 G](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

端到端系统和共享增强 prompt 下的 backbone 排名是两个不同实验；三个图中 Elo 分别拟合，不能跨组比较。其作者报告的端到端第二名不代表所有通用文生图任务第二；亦不能把外部增强器贡献归给渲染器。报告提供置信区间，接近的排名应看区间和样本量，而非只看点估计。

## 8. 扩散语言模型：独立的生成范式研究支线

### 8.1 LongLLaDA：先研究能力边界，再做无训练长度外推

LongLLaDA 的发现是：在特定 LLaDA／Dream 与自回归模型比较中，扩散语言模型直接外推长度时 perplexity 相对稳定，但仍可能只检索靠近生成位置的局部上下文。**能输出稳定文本不代表远处信息被有效利用。** 方法把 NTK-based RoPE 外推迁移到 LLaDA，在论文设置下把 4K 扩至 24K；较长设置仍会出现 lost-in-the-middle，不能无限外推。[论文](https://arxiv.org/abs/2506.14429)、[全文](https://arxiv.org/html/2506.14429v1)、[代码](https://github.com/OpenMOSS/LongLLaDA)。

其解释基于双向注意力与 RoPE 训练相对位置范围：扩散模型看过正负相对距离，自回归仅看一侧；这可以解释部分分布外稳定性，但不能取消预训练中有效距离范围的约束。论文还观察到检索、聚合和合成 QA 的优劣不同，不支持“扩散模型全面优于自回归”的结论。[论文机制与任务实验](https://arxiv.org/html/2506.14429v1)。

### 8.2 Sparse-dLLM：双向迭代生成的缓存与注意力瓶颈

Sparse-dLLM 结合动态缓存淘汰和稀疏注意力，提供无需重新训练的扩散语言模型推理加速。扩散迭代会反复更新中间状态，不能简单照搬自回归“历史 token 永不改变”的 KV-cache 假设。[论文](https://arxiv.org/abs/2508.02558)、[代码](https://github.com/OpenMOSS/Sparse-dLLM)。

作者速度结果是特定任务、稀疏度、长度和硬件下的效果；不是保证所有扩散模型均能获得固定倍数加速，也不是跨模态生成已采用该实现的证据。

### 8.3 DiRL：强化学习的随机变量需要与生成步骤一致

DiRL 基于外部 SDAR-8B-Chat 做 SFT 与 RL，并非 OpenMOSS 从头预训练的文本基座。SFT 使用随机 mask；DiPO 的关键在 generation-step 层面构造优化目标，避免把离散扩散生成硬套成自回归 token policy，结合 DAPO 的动态采样过滤零优势方差组。[论文](https://arxiv.org/abs/2512.22234)、[官方实现说明](https://github.com/OpenMOSS/DiRL)。

工程上使用 Flex-Attention、Accelerate、LMDeploy rollout 和在线更新，目的是使训练目标、采样过程和服务中的策略一致。数学基准成绩是论文当时所列 8B／32B baseline 下的作者比较，不是 2026 年 10 月所有数学模型的最新排名。[代码与评测说明](https://github.com/OpenMOSS/DiRL)。

**总体判断：** 这一支线研究的是上下文、计算复用和后训练目标的边界。它与视听扩散生成有范式层的共性，但没有证据支持“下一代 OpenMOSS 将把所有项目迁到扩散语言模型”。

## 9. 音频／语音：三条任务路线，而非所有项目共用一个 codec

完整语音机制、版本、十篇论文元数据与部署审计见 `/research-notes/2026-10-17-openmoss-deep-research/audio-primary-findings.md`。理解这一分支，首先要分清任务。

| 路线 | 代表项目 | 前端／目标 | 不能混称为 |
|---|---|---|---|
| 直接语音交互 | MOSS-Speech | ASR 导向的单码本语义 token，直接生成语音回答 | 只读给定文本的 TTS |
| 高保真音频生成 | MOSS-TTS、TTSD、Realtime、Nano | CAT 多层 RVQ，文字／参考声音到波形 | 自主理解意图、规划对话和调用工具 |
| 音频理解／转写 | MOSS-Audio、Music、Transcribe-Diarize | 连续音频特征连接文本 decoder | 音乐生成或共享 CAT token 的同一模型 |

三条路线由不同论文与配置支撑；即使前端都写 12.5Hz，也不代表表示、模型参数或权重相同。[MOSS-Speech](https://arxiv.org/abs/2510.00499)、[CAT](https://arxiv.org/abs/2602.10934)、[MOSS-Audio](https://arxiv.org/abs/2606.01802)、[MTD](https://arxiv.org/abs/2601.01554)。

### 9.1 从 SpeechGPT 的文本中介，到 MOSS-Speech 的直接回答

SpeechGPT（2305.11000，EMNLP 2023 Findings）的 chain-of-modality 显式经过语音指令、文本指令、文本答案、语音答案。SpeechTokenizer（2308.16692，ICLR 2024）则分层表示语义与声学：首层 RVQ 对齐 HuBERT 语义，其余补细节。SpeechGPT-Gen（2401.13527）让语言模型生成语义，再用 conditional flow matching 补声学感知 latent。这是历史关联链，相关仓库不都位于 OpenMOSS 组织下，不能因共同作者把全部外部工作算成组织自身仓库。[SpeechGPT](https://arxiv.org/abs/2305.11000)、[SpeechTokenizer](https://arxiv.org/abs/2308.16692)、[SpeechGPT-Gen](https://arxiv.org/abs/2401.13527)。

MOSS-Speech（2510.00499，ICLR 2026）的重要变化是推理回答不先产出文本答案：Qwen3-8B 前 32 层共享，末端分成文本／语音分支；语音用严格因果、单码本 12.5Hz 的语义表示和 flow-matching 解码。这里“无文本指导”指回答路径，不代表训练没有文字、ASR 或成熟文本模型。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)、[源码／模型](https://github.com/OpenMOSS/MOSS-Speech)。

它也暴露取舍：冻结／渐解冻和混入文字训练减轻遗忘，却未无损保留文本基座能力；正式表格的 MMLU／CMMLU 仍明显低于原 Qwen3-8B。严格因果 tokenizer 的 ASR probe 也较原 tokenizer 退化。因而，减少文本中介可能改善语音原生性和路径长度，但不会自动保证回答知识更准确。[正式论文表 2、5、6](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)。

### 9.2 CAT：低帧率、可变码率、语义与重建联合优化

原始 MOSS-Audio-Tokenizer 为约 1.6B 的因果 Transformer codec，通过波形 patchify 降至 12.5Hz、32 层×1024 项 RVQ；码率可选前 K 层。名义 token 码率为 `12.5×K×log2(1024)`，原版本从 K=1 的 125bps 到 K=32 的 4kbps，不含封装开销，也不是无损编码。语义通过训练期 0.5B 文本 decoder 的 ASR／多说话人 ASR／caption 辅助目标注入，同时优化声学重建。[论文](https://arxiv.org/pdf/2602.10934v2)、[配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/blob/main/config.json)。

**分析：** 它把“容易由 LLM 消费”和“能还原语音／环境声／音乐细节”一起优化，但层数越多，生成模型的深度预测成本越高。12.5Hz 是时间帧率，不是所有码本索引相加后的 token 速率；48kHz stereo 的 v2 和约 22M Nano codec 也不能直接继承原 24kHz mono 版本的全部实验结论。[版本与流式约束](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)、[Nano 配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano/blob/main/config.json)。

### 9.3 TTS／TTSD：把时间轴与码本深度分开

MOSS-TTS（2603.18090）的两种设计很好地说明结构取舍：Delay pattern 将不同 RVQ 层按帧错位，大骨干每步并行预测不同时间／层的 token；Global-local 则让大骨干沿时间走，小 Transformer 在每帧内逐层生成。前者方便吞吐，首个完整帧要等 delay 填满；后者无需跨帧 delay，但有局部深度自回归成本。对比还使用了不同大小的骨干，不能当作完全控制变量的架构消融。[论文第 4 节](https://arxiv.org/pdf/2603.18090v2)、[Delay](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_delay)、[Local](https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_local)。

MOSS-TTSD（2603.19739）是在这条生成线之上，把说话人标签、参考音色和完整剧本映射成多人声音。数据演进从分散 ASR／diarization 标签协调到内部 MTD 统一标注，是比“多一个模型名字”更重要的关系；当前 v1.0 使用 CAT 前 16 层和 Delay。它是 script-to-conversation，不是开放式对话智能。[TTSD 论文](https://arxiv.org/pdf/2603.19739v1)。

评测也有明显边界：公开 TTSD-eval 默认只在 S1／S2 两个参考之间判断音色，而模型声称可生成 1–5 人；测试最长 720 秒，不能独立证明一小时无漂移。TTS 论文自己的最长英文输入桶亦存在明显词错率和声纹退化。[TTSD-eval 源码](https://github.com/OpenMOSS/TTSD-eval/blob/master/tools/run_similarity.py)、[TTS 长文实验](https://arxiv.org/pdf/2603.18090v2)。

Realtime 是接收外部 LLM 增量文字并输出音频的双流 TTS，其首音／LLM-text→TTS 延迟不等于完整用户语音→理解→回答的延迟，更不等于 full-duplex。Nano 是独立约 100M global-local 生成模型配小 codec 和 ONNX CPU 部署，不是把 Qwen 大模型简单量化到 100M；公开路径存在，但缺统一硬件和完整任务的 CPU 延迟对照。[Realtime 模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)、[Nano](https://github.com/OpenMOSS/MOSS-TTS-Nano)。

### 9.4 理解模型：连续表示和全局任务，而非生成 codec 的简单反向

MTD（2601.01554）联合输出文字、说话人和时间戳，0.9B 公开版采用 Whisper-Medium 风格 encoder 和 Qwen3-0.6B 风格 decoder。长段单次全局转写不等于 encoder 不分块，公开前端仍按 30 秒抽特征；低离线 RTF 不等于实时逐词转写。论文以 CER／cpCER 为主，细粒度时间戳与 streaming SATS 仍列为未来方向。[论文](https://arxiv.org/pdf/2601.01554v7)、[实现](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)。

MOSS-Audio（2606.01802）用连续特征、DeepStack 多层连接和时间 marker 接 Qwen 文本 decoder，覆盖语音、音乐和环境声的理解，训练还有 reasoning cold start 与 DAPO。MOSS-Music 在该骨干上继续音乐训练，做 caption／歌词／结构／和弦／QA，不是音乐波形生成；本次未确认 Music 独立论文，不把 Audio 论文中所有训练细节自动继承给它。[Audio 论文](https://arxiv.org/pdf/2606.01802v3)、[Music 当前说明](https://github.com/OpenMOSS/MOSS-Music)。

**整体判断：** 音频是任务分化和部署梯度都较完整的一条线。最有价值的是语义／声学表示的任务化选择、帧率与码本深度解耦、长说话人记忆及数据标注闭环；最大的阅读风险是把“实时”“端到端”“长音频”“低 RTF”和“原生对话”当作同一个能力。

## 10. 具身智能：VLA、WAM 与任务 Agent 是三个不同层

详见 `/research-notes/2026-10-17-openmoss-deep-research/embodied-primary-findings.md`，包含十个项目、九篇论文及固定 SHA 代码证据。

### 10.1 三条路线的对象不同

| 路线 | 核心映射 | OpenMOSS 代表 | 已证实的范围 |
|---|---|---|---|
| VLA／全模态操作 | `p(action∣observation, instruction/audio)` | RoboOmni | 输入语境、意图确认和动作生成 |
| WAM／世界—动作模型 | `p(future observation, action∣current observation, instruction)`，或相关分解 | EasyWAM | 联合视频／动作监督、不同专家与推理路径 |
| 任务级 Agent | 观察→选工具→执行→新观察→验证 | OpenETA | 宿主执行约束、反馈、经验候选管理 |

这不是三者的公平胜负榜：训练数据、可用视角／工具、任务预算和指标都不同。[RoboOmni](https://arxiv.org/abs/2510.23763)、[EasyWAM](https://github.com/OpenMOSS/EasyWAM)、[ETA](https://arxiv.org/abs/2608.03924)、[WAM 综述](https://arxiv.org/abs/2605.12090)。

### 10.2 EasyWAM：世界监督与在线想象可以分开

EasyWAM 没有本次可确认的独立论文，主要是开放训练／评测框架；不能把 Awesome-WAM 综述当作其方法论文。它支持 Unified、MoT、MoT-Joint、MoT-IDM、Hidden 五种结构，源码确有 `loss_video` 与 `loss_action`，复用 Wan／Cosmos／FLUX 等基础模型。[固定仓库](https://github.com/OpenMOSS/EasyWAM/tree/dd5b06153abd1c04eff99a3933dc09baf0906fa2)、[模型目录](https://github.com/OpenMOSS/EasyWAM/tree/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model)。

最值得研究的是 MoT：**训练有未来视频监督，默认动作推理可以只用当前视觉特征而不生成未来视频。** Unified／Joint 等仍联合去噪；Hidden 则使用视频网络中间表示。由此不能说“不输出 RGB 就没有世界建模”，也不能反向说“训练预测视频就证明推理时需要视觉想象”。[MoT 代码](https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_mot.py)、[架构分析](https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/blogs/blog01_arch_zh.md)。

作者在 Wan2.2 全参数设置报告 Unified 的 LIBERO 平均 98.6%、LIBERO-Plus 79.0%，MoT 对应 96.7%／71.7%；Cosmos 下架构排序不同。它们说明训练架构与分布外测试有研究价值，不足以因果证明“测试时出视频一定更好”。本项目证据主要为模拟器策略评测，尚不验证真实接触安全或部署中持续学习。[同一固定版本及协议](https://github.com/OpenMOSS/EasyWAM/tree/dd5b06153abd1c04eff99a3933dc09baf0906fa2)、[LIBERO 评测定义](https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/instructions/benchmark/libero.md)。

### 10.3 OpenETA：执行与验证闭环已实现，学习收益未被提前兑现

ETA 论文（2608.03924）把 Planner、Interface、外部 World 分开。此处 World 首先是执行环境，不是神经动力学模型。接口要求一次改变世界的动作之后获取新观测，并区分工具完成、状态改变和任务成功；经验／skills 不是绕过权限的隐藏执行程序。[论文](https://arxiv.org/html/2608.03924v1)、[架构与动作管线](https://github.com/OpenMOSS/OpenETA/tree/b02260228e8c90fbc954e4eeb4ce4175249da9ba/docs)。

这是真实的任务治理设计，但没有证据表明它已经把 EasyWAM、RoboOmni 和 FRoM-W1 连成同一个已评测链路。论文把 VLA 短时控制与 WAM 预测／方案比较列作后续能力；仓库某些 policy 适配项目也仍未完成。[论文后续方向](https://arxiv.org/html/2608.03924v1)、[功能矩阵](https://github.com/OpenMOSS/OpenETA/tree/b02260228e8c90fbc954e4eeb4ce4175249da9ba)。

尤其应保留持续学习的负结果：代码能提出、审阅和验证经验候选，但论文的探索研究中没有候选通过全部晋级门槛；默认也不自动应用经验。它证明了一部分无效／回退经验可以被拒绝，不证明“持续学习已稳定提升”。10 月 RoboDojo RSI 是跨版本、人类辅助、尚未填满预算的累计覆盖，不是单版本、固定协议下的全自主因果增益。[论文 §6](https://arxiv.org/html/2608.03924v1)、[经验更新实现](https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/agent/runtime/self_improvement.py)、[RSI 协议](https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/docs/robodojo-rsi-results.md)。

实机方面，UR5e 的真实控制路径和开发录像存在；部分旧 README 的 stub 描述已落后于源码。但录像案例不是冻结重复试验，没有可审计总体成功率，也不能从 API 有碰撞检查参数推定该路径真正执行了完整碰撞验证。[论文实机章节](https://arxiv.org/html/2608.03924v1)、[真实执行代码](https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/real/mcp/observation_core.py)。

### 10.4 RoboOmni 与 FRoM-W1：两种实在的子闭环

RoboOmni（2510.23763，ICLR 2026）基于 Qwen2.5-Omni，将多模态语境、对话确认和 FAST+ 动作序列连接起来。OmniAction 合成了隐式语境、语音与轨迹。其目标是对话／动作最大似然，不含动作条件化未来世界预测，因此属于全模态 VLA 而非 WAM；“LIBERO-Real”指真人语音输入的仿真测试，不是实机 LIBERO。WidowX 实机展示是独立证据，公开全模态复现链仍有缺口。[论文](https://arxiv.org/html/2510.23763v3)、[训练实现](https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/training/train_omni.py)。

FRoM-W1（2601.12799）用 H-GPT 将语言→CoT→离散人体运动，H-ACT 将 SMPL-X 重定向到 H1／G1，经 PPO teacher 和 DAgger student 实现跟踪。实机展示说明语言运动生成到全身控制的管线可行；但参考人体运动不是对环境物体后果的预测，推理阶段的 RFT 是仿真内针对目标运动适配，不是实机终身学习。论文也展示失稳、追踪失败、自碰撞和 sim-to-real 的限制。[论文](https://arxiv.org/html/2601.12799v1)、[项目](https://github.com/OpenMOSS/FRoM-W1)。

### 10.5 数据与评测资产：价值大，但不应计入模型自身能力

VLABench（2412.18194，ICCV 2025）提供 100 类任务、技能组合、语义／物理规则及泛化测量。正式版、早期预印本与当前代码的 track 定义和数据预算不同，复现必须锁版本；其有限 sim＋real 小实验也不是全部 100 类任务的实机验证。[ICCV 正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.pdf)、[仓库](https://github.com/OpenMOSS/VLABench)。

其余工作要按对象拆开：Embodied-Planner-R1（2506.23127）是在 ALFWorld／ScienceWorld 纯文本环境做交互 RL；VehicleWorld（2509.06736）是车舱设备状态／API benchmark，不是驾驶世界模型；VehicleArena（2609.35916）是由 SUMO 执行物理世界的持续交通 Agent 环境，不是神经动力学模型；FutureOmni（2601.13836，ICML 2026）通过视听前缀预测未来事件，却不含 Agent 动作干预。Awesome-WAM（2605.12090）是综述和索引，不能把其中收集的外部模型实验计入 OpenMOSS 自研结果。[Planner-R1](https://arxiv.org/abs/2506.23127)、[VehicleWorld](https://arxiv.org/abs/2509.06736)、[VehicleArena](https://arxiv.org/abs/2609.35916)、[FutureOmni](https://arxiv.org/abs/2601.13836)、[综述](https://arxiv.org/abs/2605.12090)。

**整体判断：** 感知→动作、语言→运动→控制、世界表示↔动作监督、执行→观察→验证四类子闭环已有证据；它们统一运行且持续学习稳定有效，仍是重要的未证实部分。

## 11. Agent、长程信息利用与可靠性：评价对象正在从回答转向工作

详细的一手发现、17 篇论文元数据和发布审计见同目录 `/research-notes/2026-10-17-openmoss-deep-research/agents-reliability-primary-findings.md`。以下只保留对整体路线判断最重要的结论。

### 11.1 三种 benchmark，分别补上三种真实任务约束

| 项目／论文 | 评价对象 | 相比“生成正确代码”的新增约束 |
|---|---|---|
| [ABC-Bench，2601.11077](https://arxiv.org/abs/2601.11077) | 224 个后端任务，8 种语言、19 个框架 | 探索仓库、配置环境、服务真实启动、通过外部 API 测试 |
| [SWE-bench Science，2608.19799](https://arxiv.org/abs/2608.19799) | 119 个任务、98 个科学软件仓库、20 个领域 | 修复软件时还要维持单位、坐标、数值不变量和科学语义 |
| [ContextWeave，2608.04830](https://arxiv.org/abs/2608.04830) | 14 人多月工作流重构，1,005 个任务，其中 568 个核心评价任务 | 历史经验能否帮助下游执行、工作区质量与个人偏好 |

**分析：** 它们改变的不只是题目难度，而是成功的定义：后端服务真的可用，科学结论的计算契约未被破坏，长期任务不需要每次重新摸索。它们首先是评测／harness 成果，不是证明一个统一 OpenMOSS Agent 已经在这些任务上领先。

ABC-Bench 还发布独立的 Qwen3-8B／32B-ABC SFT 权重，不能把该权重与 benchmark 混称。论文观察到环境建设阶段可以遮蔽已有的功能实现能力；通过外部 API 测试也不代表通过生产安全、负载和可维护性审计。[ABC 论文](https://arxiv.org/pdf/2601.11077)、[发布入口](https://github.com/OpenMOSS/ABC-Bench)。

Science 的论文最高 Pass@1 为 47.90%，当前官网标注 2026-10-09 更新，最高为 50.42%；不能截至 10 月仍复述摘要的“当前最好低于 50%”。默认公开运行选 96 个不受 restricted-license gate 限制的任务，另 23 个需显式 opt-in；默认子集不能与全部 119 个任务的论文结果直接比较。私有验证镜像可执行与私有测试源码可重建也是不同层级。[论文](https://arxiv.org/pdf/2608.19799v2)、[当前榜单](https://swescience.github.io/)、[运行／数据契约](https://github.com/OpenMOSS/SWE-bench-Science)。

### 11.2 ContextWeave：记忆有收益，也会带来执行错误

最强配置在固定模型／harness 下，把 Workspace 从 68.08 提升到 78.20，Preference 从 41.50 到 70.60；同时该配置中记忆诱发问题的任务比例为 7.39%。这包括后来恢复的问题，不是最终失败率。不同记忆组件还同时改变检索策略与上下文量，不能把结果直接当成“全文历史因果优于摘要”的实验。[论文表 1 与局限](https://arxiv.org/pdf/2608.04830)。

它的重要启发是：**记忆系统不能只优化检索相关性，还要对执行中正确使用历史负责。** 实验从参考 Docker 状态重新开始子任务以控制前序漂移，因此不是一个 Agent 在完全自由的数月生命周期里持续累积全部错误。真实长期部署仍需要处理记忆版本、失效、冲突、误导和恢复。

### 11.3 AgentHPOBench：正确的比较协议比主表排名重要

30 个可执行机器学习任务、共享 baseline 后 5 次有约束的超参干预，用归一化提升而非“答对率”评分。主表采用最终步，附录 best-so-far 中 Random Search／TPE／BOHB 超过所有测试的 open-weight agents。传统 HPO 通常返回当前最佳 incumbent，因此不能仅按最终步主表宣称 LLM 全面胜过 HPO。[论文](https://arxiv.org/abs/2607.29626)、[正文与附录](https://arxiv.org/pdf/2607.29626)、[代码](https://github.com/OpenMOSS/AgentHPOBench)。

UnifiedToolHub 则是把多个工具调用数据集标准化的工程底座。它的分步 offline ExactMatch／ToolAccuracy／ParameterAccuracy 与真实自治任务成功率不是同一回事，本次未确认其专属论文。[官方说明](https://github.com/OpenMOSS/UnifiedToolHub)。

### 11.4 幻觉、知识边界和长上下文安全，不应合成一个“可信度分数”

- **HalluQA（2310.03368）：** 中文对抗性事实问答与参考答案核查，覆盖误导和知识缺失；历史模型成绩不是当前模型排名。[论文](https://arxiv.org/abs/2310.03368)。
- **Can AI Assistants Know What They Don't Know?（2401.13275，ICML 2024）：** 按某个模型的多次采样正确率构造已知／未知标签，再训练正确回答或拒答；知识边界随模型和检索能力改变，需要重新校准。它不是意识／内省的证明，且正确拒答与过度拒答之间存在取舍。[论文](https://proceedings.mlr.press/v235/cheng24i.html)。
- **LongSafety（2411.06899，LLMSEC 2025 workshop）：** 把长输入的有害查询／部分有害／全部有害分开，训练约 17K 样本；长窗口、普通长上下文对齐和短安全训练不能互相替代。主要评价包括多选 HarmAwareness 和 SafeResponse，不等于任意工具执行或开放生成的安全保障。README 仍有训练启动配置 TODO 与权重发布缺口。[论文](https://arxiv.org/abs/2411.06899)、[当前发布状态](https://github.com/OpenMOSS/LongSafety)。

### 11.5 长上下文与机制解释：重要，但不是直接的系统安全证书

**ReAttention（2407.15176，ICLR 2025）** 先在全历史 KV 中检索有用片段，再按相对次序重排并在有限注意力作用域中重新编码位置。其“无限”不代表历史 KV 存储或检索成本恒定；百万 token 查针也不等于同长度的科学推理和持续工作已解决。[论文](https://arxiv.org/abs/2407.15176)、[源码与 TODO](https://github.com/OpenMOSS/ReAttention)。

**RoPE++（2512.07525，ICLR 2026）** 把复数旋转内积通常未使用的负虚部引入额外注意力头，研究保持头数时减少 QKV／KV cache，或保持 cache 时增加头数的取舍。它需要训练，不是普通 LLM 的即插即用外推补丁；776M／1.5B 实验中不同任务有不同得失，尚不证明 7B 以上系统或 Agent 同样受益。[论文](https://arxiv.org/abs/2512.07525)、[仓库](https://github.com/OpenMOSS/rope_pp)。

**Llamascopium／Llama Scope／Lorsa** 建立分布式 SAE、transcoder、稀疏注意力分解和可视化工具。Llama Scope 的 256 个稀疏模块是解释资源，不是增强后的 Llama；Lorsa 用稀疏、过完备的 attention heads 和低秩 OV 近似原注意力，解释冗余表示可能更大，而非保证压缩。最新 CRM 作者报告组合 Lorsa 与 transcoder 做完整替换路径和 circuit tracing，但“complete”是路径覆盖，仍有重建和忠实度缺口。[Scope 论文](https://arxiv.org/abs/2410.20526)、[Lorsa 论文](https://arxiv.org/abs/2504.20938)、[当前 dev 框架](https://github.com/OpenMOSS/Llamascopium/tree/dev)、[CRM 作者报告](https://interp.open-moss.com/posts/complete-replacement)。

**BandPO（2603.04918）** 用依赖旧概率的更新上下界，缓解固定 clipping 对低概率、正优势 token 的过早限制；实验主要是数学 RLVR，不直接证明工具行为更可靠、幻觉更少或更安全。它的 trust-region 投影问题有理论性质，不等于整个 LLM RL 优化保证全局最优。[论文](https://arxiv.org/abs/2603.04918)、[代码](https://github.com/OpenMOSS/BandPO)。

**新增机制与跨域边线：** Hybrid-Mechanics 已有 2026-10-07 的 `2610.10114`，研究混合注意力与位置偏置的配合，但仓库目前主要是论文材料；GSF-χ（2609.12532）是手性分子图 Transformer，不能因为使用 Chiral-RoPE 就归入文本长上下文方案。[Hybrid](https://github.com/OpenMOSS/Hybrid-Mechanics)、[论文](https://arxiv.org/abs/2610.10114)、[GSF-χ](https://arxiv.org/abs/2609.12532)。

**整体判断：** 这条支线提供“可执行评价—长程信息利用—行为可靠性—内部机制分析—训练优化”的互补问题分解。但当前未见这些方法在同一个旗舰模型上做统一训练与联合消融，不能把组织级关联写成已验证的集成系统。

## 12. 综合分析：真正共通的不是模型结构，而是五个技术问题

### 12.1 统一究竟发生在哪一层？

| 统一层次 | 代表项目 | 具体含义 | 不能推导出的结论 |
|---|---|---|---|
| 接口／序列 | AnyGPT | 不同模态转成离散序列，复用语言建模接口 | 所有模态的保真度、延迟都最优 |
| 时间与访问语义 | MOSS-VL、MOVA | 帧与文字的可见关系、音视频的同一物理时间 | 模型已经理解全部物理因果 |
| 可学习的潜表示 | Audio Tokenizer、OmniVAE | 保留重建细节，又注入语义和跨模态对应 | 一个编码器适用于所有模态 |
| 规格与执行接口 | MOSS-Image | Agent 输出结构化内容／布局，DiT 执行 | 自有 DiT 独立完成事实研究和设计推理 |
| 任务与环境闭环 | OpenETA、EasyWAM、Agent 评测 | 感知、执行、验证、反馈被纳入同一任务流程 | 已经实现开放世界无限持续学习 |

这是本报告的分析分类，不是官方架构图。前四行分别由对应论文与代码支撑，最后一行需要结合具身与 Agent 的任务级证据，而不能由前四行直接推论。[AnyGPT](https://arxiv.org/abs/2402.12226)、[MOSS-VL](https://arxiv.org/abs/2608.15045)、[MOVA](https://arxiv.org/abs/2602.08794)、[OmniVAE](https://arxiv.org/abs/2607.23855)、[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)、[OpenETA](https://github.com/OpenMOSS/OpenETA)、[EasyWAM](https://github.com/OpenMOSS/EasyWAM)。

### 12.2 数据工程往往比“用了哪种 Transformer”更能解释结果

从早期 MOSS 的工具对话、AnyGPT 的合成任意模态指令，到 MOSS-VL 的时间锚定／沉默与修正轨迹，再到 MOSS-Image 的精确文字、换行和布局标注，可见反复出现的策略是：**先把目标行为写成可监督的数据与接口，再优化模型学会它。** 这不是否定架构创新，而是说明能力通常来自任务定义、表示和数据管线的共同作用。[MOSS](https://github.com/OpenMOSS/MOSS)、[AnyGPT](https://arxiv.org/abs/2402.12226)、[MOSS-VL](https://arxiv.org/html/2608.15045v1)、[MOSS-Image 报告](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)。

**分析：** 因而，最难复刻的部分可能不是下载检查点，而是持续构建正确的交互数据、统一标注语义和可靠评测。仅凭“参数小、榜单高”很容易忽略这些投入。反过来，依赖外部模型生成标注／规划，也会引入上游能力、成本、偏差和许可依赖。

### 12.3 在哪里复用，在哪里自己训练？

| 项目 | 可核查的外部基础 | 自己主要增加的能力层 |
|---|---|---|
| AnyGPT | Llama 类语言骨干，SpeechTokenizer、SEED、EnCodec 等 | 跨模态离散对齐和 AnyInstruct |
| MOSS-VL | Qwen3-8B、Qwen3-VL 视觉编码器 | gated cross-attention、XRoPE、课程训练和实时状态学习 |
| MOVA | Wan2.2 视频骨干和既有 VAE | 音频骨干、双向 Bridge、对齐数据／联合训练 |
| OmniVAE | Wan2.2 VAE、冻结 Qwen3-Omni 教师 | 音视频潜空间联合对齐和语义蒸馏 |
| MOSS-Image | Qwen3.5-9B 文本编码器、FLUX.2 VAE、外部 GPT-6 规划 | 从头预训练的 5B DiT、文本／布局监督与执行接口 |
| DiRL | SDAR-8B-Chat、LMDeploy、LLaMA-Factory 等 | 扩散模型 SFT／DiPO 目标与高效在线后训练 |

具体依赖由论文和发布代码确定，表格不表示各项依赖的许可证等同。[AnyGPT](https://github.com/OpenMOSS/AnyGPT)、[MOSS-VL](https://arxiv.org/html/2608.15045v1)、[MOVA](https://arxiv.org/html/2602.08794v2)、[OmniVAE](https://arxiv.org/abs/2607.23855)、[MOSS-Image 报告](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)、[DiRL](https://github.com/OpenMOSS/DiRL)。

**我的判断：** 这是一条务实的能力层创新路线：复用强基座，把投入集中在表示、跨模态连接、任务数据和系统闭环。它不应被误读为“没有自己的模型”，也不应被包装为“所有关键组件从头自主训练”。

### 12.4 研究能力与工程能力怎样分别评价？

研究价值优先看：是否提出可检验的问题、有没有机制解释、是否设置合理消融、结果是否超出自建基准。工程价值优先看：当前检查点与论文是否一致、训练和评测入口是否完整、实时性是否覆盖并发／中断／长期缓存、端侧运行是否有完整成本统计。

一个值得肯定的可观察信号，是 MOSS-VL 发布特殊注意力算子和服务适配，MOVA 发布 LoRA／评测，OmniVAE 发布 VAE／DiT／同步 probe 三类实验入口。相比之下，论文写明训练配方，并不自动表示仓库包含重训整个预训练规模所需的全部数据、脚本和资产。[MOSS-VL](https://github.com/OpenMOSS/MOSS-VL)、[MOVA](https://github.com/OpenMOSS/MOVA)、[OmniVAE](https://github.com/OpenMOSS/OmniVAE)、[MOSS-Image](https://github.com/OpenMOSS/MOSS-Image)。

### 12.5 未来路线：有根据的推断与尚未证实的部分

| 判断 | 本报告置信度 | 根据／限制 |
|---|---|---|
| 实时多模态交互仍是持续投入的方向 | 高 | 音频、视频的独立项目、训练目标和服务实现形成持续证据 |
| 从工具使用扩展到物理世界任务和反馈学习 | 中高 | 具身项目、WAM 框架及更真实的任务评测；整合程度因项目而异 |
| 表示层会继续承担更多跨模态对齐工作 | 中高 | Audio Tokenizer 与 OmniVAE 的方法论一致，但未来产品安排未知 |
| 各模型将组合成模块化 Agent，而不一定全部合为一个模型 | 中 | MOSS-VL demo、MOSS-Image 的模块组合支持此解释；不能当成官方承诺 |
| 已形成从视听生成到机器人控制的统一预训练基座 | 低／证据不足 | 未见跨全部模块的共享检查点、统一动作接口和联合闭环验证 |
| 已解决开放世界的长期自主学习和安全问题 | 不成立于现有证据 | 需要持续、可审计的任务轨迹、失误恢复与安全约束验证 |

这些是路线分析，不是内部计划、融资／商业战略或未来发布时间预测。

## 13. 如何辨别实力与宣传：七个核查点

1. **拆开自建与外部基准。** LLTB／RLA、VLABench 等解决具体评测缺口，但创建者也训练模型，应检查 held-out、防污染和独立复现；现有外部基准上的成绩也要核查采样配置。
2. **拆开 backbone 与完整 system。** 外部 Agent、ASR／TTS、记忆、检索和 prompt enhancement 的贡献，不能全部计入一个检查点。
3. **拆开总参数、激活参数和实际依赖。** MOVA 的 MoE、MOSS-Image 的文本编码器就是典型；运行成本还包含 KV-cache、VAE、服务、主机内存。
4. **拆开理论能力、已发布代码与量化评测。** MOSS-VL 的 L5 演示不等同于 L5 有系统基准；仓库声明某种功能，也要检查实际入口。
5. **拆开感知、预测和行动。** 看懂事件、生成像真的视频、预测动作后果、让机器人成功执行，是四种不同证据。
6. **拆开公开权重、训练代码和数据。** 有 checkpoint 和报告不等于任何人都能重做大规模预训练；“配方披露”仍很有价值，但层级不同。
7. **拆开代码、权重、数据和第三方许可。** MOSS 老仓库代码 Apache-2.0、模型 GNU AGPL-3.0、数据 CC BY-NC-4.0；不能用仓库语言／许可证卡片推断全部资产均可自由商用。此处为官方标识记录，不构成法律意见。[MOSS](https://github.com/OpenMOSS/MOSS)。

上述核查点由本报告所读论文与代码中的实际差异归纳。[MOSS-VL](https://arxiv.org/html/2608.15045v1)、[MOVA](https://arxiv.org/html/2602.08794v2)、[MOSS-Image](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)、[组织目录](https://github.com/OpenMOSS)。

若要进入第二阶段实测，本报告建议以下验收问题。它们是研究设计建议，不是本次已经做过的实验：

- **实时视频：** 中断和新帧到达时，回答是否根据真实已见证据更新？多并发、长时间运行的延迟／显存是否稳定？遗漏和误报分别多少？
- **音频系统：** 在相同硬件／码率／采样率下比较首包延迟、整体实时因子、长对话词错率、说话人一致性和音质，不把不同工作负载混在一起。
- **视听生成：** 分开测试单人／多人、说话／非语音、短／长时段，对同步、文字内容、说话人、物理一致性分别报告。
- **WAM／具身：** 与无世界模型的同预算 VLA 比较闭环任务成功率、视觉滚动误差、规划开销与真实机器人安全失败。
- **Agent：** 除 pass rate，记录成本、超时、失败恢复、工具执行日志、记忆污染和验证假阳性。
- **图像布局：** 除 OCR coverage，检查逐序编辑距离、文本事实、布局可读性、同一外部 planner 下的 backbone 对照。

## 14. 阅读路径与材料

### 14.1 先读机制，再读榜单

推荐阅读顺序是一个理解依赖顺序，不是论文重要性排行榜：

1. **AnyGPT（2024）**：建立统一模态接口的历史起点。
2. **MOSS-Audio-Tokenizer 和 OmniVAE（2026）**：理解表示的重建、语义与可学习性。
3. **MOSS-VL Technical Report（2026）**：重点读跨注意力、时间掩码、Realtime-SFT 与系统限制。
4. **MOVA（2026）**：重点读 Bridge、Aligned RoPE、独立噪声时间和双 CFG 的取舍。
5. **MOSS-Image Technical Report（2026）**：重点读多粒度标注、文本／布局统一条件接口，以及外部规划器的边界。
6. **VLABench → RoboOmni／FRoM-W1 → EasyWAM／OpenETA**：按评测、动作能力、框架和任务闭环的次序理解具身线。
7. **ABC-Bench／SWE-bench-Science／ContextWeave**：转向任务完成、执行环境和长期记忆。
8. **LongLLaDA → Sparse-dLLM → DiRL**：若关注模型机制，再读长度外推、迭代推理优化和扩散强化学习。

每篇的准确入口及代码在正文对应小节；核心论文结构化元数据见同目录 JSON。其 `citation_count=null` 表示未查询；arXiv 论文记录的是 DataCite 发行的 arXiv 标识 DOI，不等于已核实正式会议／期刊 DOI；MOSS-Image 的 `doi=null` 表示本次未核实，不表示一定没有 DOI。该集合只覆盖主报告中主代理直接核查的十项，不声称是整个组织的全部论文库；其中 MOSS 老论文以元数据／仓库为主要核查依据，未把它算作本次已阅读全文的论文。

### 14.2 核心论文轻量目录

| 论文 | 作者标识 | 首发年 | 发表信息边界 | 标识／公开全文 | 代码 |
|---|---|---|---|---|---|
| MOSS: An Open Conversational Large Language Model | Tianxiang Sun 等 | 2024 | Machine Intelligence Research；模型发布早于论文 | [DOI](https://doi.org/10.1007/s11633-024-1502-8)，本次未核实全文 | [MOSS](https://github.com/OpenMOSS/MOSS) |
| CoLLiE: Collaborative Training of Large Language Models in an Efficient Way | Kai Lv 等 | 2023 | EMNLP 2023 System Demonstrations（仓库与 arXiv 声明） | [2312.00407](https://arxiv.org/abs/2312.00407)，open_pdf | [CoLLiE](https://github.com/OpenMOSS/CoLLiE) |
| AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling | Jun Zhan 等 | 2024 | [ACL 2024，Long Papers](https://aclanthology.org/2024.acl-long.521/) | [2402.12226](https://arxiv.org/abs/2402.12226)，open_pdf | [AnyGPT](https://github.com/OpenMOSS/AnyGPT) |
| MOSS-VL Technical Report | Pengyu Wang 等 | 2026 | arXiv 技术报告 | [2608.15045](https://arxiv.org/abs/2608.15045)，open_pdf | [MOSS-VL](https://github.com/OpenMOSS/MOSS-VL) |
| MOVA: Towards Scalable and Synchronized Video-Audio Generation | OpenMOSS Team 等 | 2026 | arXiv 技术报告 | [2602.08794](https://arxiv.org/abs/2602.08794)，open_pdf | [MOVA](https://github.com/OpenMOSS/MOVA) |
| OmniVAE: An Audio-Video VAE with Cross-Modal Alignment for Joint Generation | Jun Zhan、Chen Yang、Yitian Gong 等 | 2026 | arXiv | [2607.23855](https://arxiv.org/abs/2607.23855)，open_pdf | [OmniVAE](https://github.com/OpenMOSS/OmniVAE) |
| MOSS-Image Technical Report | MOSI.AI | 2026 | 官方 PDF；未核实 arXiv ID | [公开 PDF](https://raw.githubusercontent.com/OpenMOSS/MOSS-Image/main/paper/MOSS_Image_Technical_Report.pdf)，open_pdf | [MOSS-Image](https://github.com/OpenMOSS/MOSS-Image) |
| LongLLaDA: Unlocking Long Context Capabilities in Diffusion LLMs | Xiaoran Liu 等 | 2025 | arXiv comments 声明 AAAI 2026 接收 | [2506.14429](https://arxiv.org/abs/2506.14429)，open_pdf | [LongLLaDA](https://github.com/OpenMOSS/LongLLaDA) |
| Sparse-dLLM: Accelerating Diffusion LLMs with Dynamic Cache Eviction | Yuerong Song 等 | 2025 | 本次未核实正式 venue | [2508.02558](https://arxiv.org/abs/2508.02558)，open_pdf | [Sparse-dLLM](https://github.com/OpenMOSS/Sparse-dLLM) |
| DiRL: An Efficient Post-Training Framework for Diffusion Language Models | Ying Zhu 等 | 2025 | arXiv | [2512.22234](https://arxiv.org/abs/2512.22234)，open_pdf | [DiRL](https://github.com/OpenMOSS/DiRL) |

作者、日期与标题通过 arXiv 页面结构化 `citation_*` 字段和官方 PDF 核对；预印本首发年与会议年份分开列示。一次 arXiv bulk API 查询返回 429 后切换到官方单篇页面，没有把 API 限流解释成论文不存在。

## 15. 最终结论

最准确的概括不是“OpenMOSS 在做某个万能模型”，而是：

**他们在探索一套可以把不同模态的表示、实时交互、生成与行动连接起来的开放研究栈；从单模型能力，逐步走向能够执行任务并接受反馈的系统。**

技术上最值得带走的四点是：把跨模态对齐前移到表示层；让时间可见性与调度成为训练设计的一部分；用结构化规格把规划与渲染／执行分开；用真实任务验证模型，而不是只验证输出像不像正确答案。

但证据也明确限制这句话：现有项目是多条相关研究线与模块组合，不是已经被统一训练、统一验证的全模态自治系统。最值得关注的下一步，不是新的命名或更大的参数量，而是跨项目共享接口、同预算闭环收益、失败恢复和独立复现实证。
