# OpenMOSS 具身智能与世界模型：一手证据报告

截止：2026-10-17（北京时间）。范围仅限 VLABench、Embodied-Planner-R1、RoboOmni、FRoM-W1、OpenETA、EasyWAM、Awesome-WAM、VehicleWorld、VehicleArena、FutureOmni。下列提交时间另按 GitHub 的 UTC 时间记录。

## 1. 核心判断

**现有证据支持“多个子闭环与开放基础设施正在形成”，不支持“perception、world model、action、continuous learning 已经汇入一个统一、经实机验证的系统”。** RoboOmni 实际连接了全模态输入与动作生成；FRoM-W1 连接了语言、人体运动表示与人形机器人控制；EasyWAM 实际训练视频—动作模型；OpenETA 实际连接了感知工具、执行、反馈与可审计经验管理。但 OpenETA 论文将 VLA/WAM 调用和进一步训练闭环列为后续方向，其自我改进实验也未得到可重复的增益。这是对下列一手材料的综合判断，而非组织官方战略声明。[RoboOmni 正文][P3]、[FRoM-W1 正文][P4]、[EasyWAM 实现][EW-code]、[ETA 正文 §6–9][P5]

优先结论：

- **OpenETA 是系统层集成，不是已经训练出的统一世界模型。** 有真实执行代码，也有开发录像；论文正式证据与后续仓库版本必须分开。自我改进的安全晋级机制已实现，但论文候选全部未通过，10 月 RSI 则是人类辅助、跨版本、未完成的累计覆盖。[论文][P5]、[真实 MCP 实现][ETA-real-core]、[RSI 协议][ETA-RSI]
- **EasyWAM 的世界—动作连接不是口号。** 代码有未来视频和动作损失，也有不同的推理路径；尤其 MoT 训练有未来视频监督，默认动作推理却不生成未来视频，不能简单把“不出视频”等同于“没有世界建模”。目前公开结果主要是模拟器策略评测，没有本项目的定量真实机器人或持续学习证据。[模型实现][EW-code]、[架构分析][EW-blog]、[官方 README][C6]
- **RoboOmni 是全模态 VLA 路线，不是 WAM。** 其预测目标是对话和动作，未展示动作条件化的未来世界预测；实机证据来自 WidowX 展示，公开仓库的全模态实机复现链仍有缺口。[论文 §4–5][P3]、[LIBERO 入口][RO-eval]、[实机入口][RO-real]
- **FRoM-W1 有 H1/G1 实机全身运动证据，但不是场景交互通用机器人。** 它生成并跟踪人体运动，不等同于理解物体状态、预测交互后果、完成接触丰富任务。推理阶段 RFT 是针对参考运动的仿真适配，不是持续实机经验学习。[论文 §2–3][P4]
- **VLABench 是有价值的评测/数据资产，而非统一模型。** 正式版含有限实机数据混合实验；但其正式论文、早期 arXiv、当前代码的评测定义存在差异，不能混用。[ICCV 正文][VLA-final]、[补充材料 §10.2][VLA-supp]、[仓库][C1]

## 2. 调研口径与证据边界

采用 research 与 academic-search 工作流，本分支独立完成，未进一步分发子任务。仅使用官方 GitHub、作者官网、作者 Hugging Face、arXiv 正文和正式会议出版页面。检索发现入口不作为技术结论证据。

正文读取以截止日前版本为准：VLABench 同时核验 ICCV 正式版及补充材料；RoboOmni 核验 arXiv v3，其主文与 v2 文本一致；FutureOmni、VehicleArena 使用 v2，其余关联 arXiv 论文使用 v1。九篇 arXiv PDF 均验证返回 PDF 内容，`full_text_status=open_pdf`。RoboOmni 的 ICLR 身份已由 OpenReview 官方元数据核验，但其会议 PDF 路由本次未成功读取，技术分析不冒充来自会议终稿。[arXiv 版本页][A3]、[RoboOmni OpenReview][RO-venue]、[FutureOmni 版本页][A9]、[VehicleArena 版本页][A8]

下文实验成绩、数据规模、硬件资源与作者性能比较均为**作者自报，未经本次复跑或独立审计**；实现结论来自静态读码，不等于运行成功。代码存在、模型权重存在、开发演示存在、冻结协议评测存在、部署安全成立，是不同证据级别。未找到证据只表示“本次审阅范围内未证实”，不是证明其永远不存在。

### 固定仓库快照

以下为截止日前默认分支最近提交；链接锁定 SHA，避免后续 README 漂移。该表是官方仓库记录，不是性能数字。

| 项目 | 固定 SHA | 提交日期 UTC |
|---|---|---|
| VLABench | [cf588fe60c0c7282174fe979f5913170cfe69017][C1] | 2025-11-11 |
| Embodied-Planner-R1 | [50cb93cfc02725a1864edc39c0a2170ab233ba61][C2] | 2025-08-01 |
| RoboOmni | [935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895][C3] | 2026-03-28 |
| FRoM-W1 | [457d84d44e638c1070473dd48cedde19752cad09][C4] | 2026-06-05 |
| OpenETA | [b02260228e8c90fbc954e4eeb4ce4175249da9ba][C5] | 2026-10-16 |
| EasyWAM | [dd5b06153abd1c04eff99a3933dc09baf0906fa2][C6] | 2026-10-09 |
| Awesome-WAM | [aa6cd0508a089e7556dd2fa0347e847c91ca986b][C7] | 2026-09-06 |
| VehicleWorld | [6536be9b88a92619357d6c02325af00a8f3c25c1][C8] | 2025-09-16 |
| VehicleArena | [421d5fdc4e1802d03844e85e4ac3e612fa6dfc88][C9] | 2026-10-06 |
| FutureOmni | [5b292a0438e5b829d8777bb6426afaa58cf12469][C10] | 2026-01-22 |

## 3. 五个重点项目

### 3.1 OpenETA：最接近系统集成，但世界模型和持续学习效果不能提前兑现

**定位与实现。** 关联论文是 *ETA: A New Agentic Paradigm for Embodied Tasks*，arXiv `2608.03924`。ETA 将任务级 Planner、宿主 Interface、外部 World 分离：Planner 提议工具调用，Interface 检查授权和前置证据，模拟器/机器人执行后返回结果及新观测。World 在这里首先是外部执行环境，不是训练出来的潜在动力学模型。读码和文档支持模块化感知、抓取、运动、记忆、轨迹记录与实验入口已存在。[论文 §3–4][P5]、[架构文档][ETA-arch]、[固定仓库][C5]

其关键约束是一次只执行一个改变世界的动作，之后满足新观测义务，再做依赖当前状态的决策；工具完成、世界状态改变、任务成功三个断言分离。Skills 是可读的经验指导，不是绕过工具权限的隐藏执行程序。此设计提供的是反馈和治理边界，不能由此推出模型已学会物理因果。[论文 §3–4][P5]、[动作管线][ETA-pipeline]

**VLA/WAM 接口的实际边界。** README 的功能矩阵明确区分已适配和未适配能力；列出的 OpenVLA、OpenVLA-OFT、openpi、GR00T 策略适配项没有完成标记。论文主实验没有以 VLA 或任务专用策略作为工具，并将未来 VLA 短时控制、WAM 预测/方案比较列为后续能力。未发现把 EasyWAM、RoboOmni、FRoM-W1 接入同一个 ETA 执行与学习闭环的已评测链路。因此不能把“可组合”写成“已经组合”。[README 功能矩阵][C5]、[论文 §5、8][P5]

**两套 LIBERO 实验不能混成一个成绩。** 以下均为作者自报：

| 设置 | 协议 | 结果与限制 |
|---|---|---|
| Full OpenETA | 四套各十任务，四十任务 × 十个预注册 seed；GPT-5.6 Luna/medium；无额外任务策略训练、无协助、评测中不载入新经验；仅官方正 reward 算成功 | `56/400=14.0%` episode success；Long 为 `1/100`。三格模拟器 handle 过期仍计入原分母；排除后的 `56/397` 只是诊断 |
| OpenETA for Codex | 一百三十任务，含 LIBERO-90；五个有序 seed；同一视觉/工具契约比较 Luna、Terra、Sol | Sol 第一 seed 解出 `92/130=70.8%`；前五次任一次成功覆盖 `117/130=90.0%` |

来源：[论文 §5 与附录 B][P5]。后者是任务级有序尝试覆盖，不是“一次部署有九成成功率”；也不是可直接与 EasyWAM 的四十任务、多 rollout 成功率相减的指标。Codex 配置允许正交视角、点选到三维位置及抓取预览，应与普通机器人相机观测设置分开比较。[论文 §5.1.2][P5]

README 的历史消息提到六个 typed tools，论文称三个 embodied tools，并另外列出 evaluator 生命周期调用。其本质是统计范围不同：感知/操作能力是 `observe`、`mark_point`、`move_to`，生命周期工具不能被写成额外机器人智能。[README][C5]、[论文 §5.1.2][P5]

**连续学习：机制存在，效果未成立。** `self_improvement.py` 有 episode 后审阅、候选提案、独立评阅与可选会话级应用；默认 `auto_apply_reviewed=False`。`promoted_memory.py` 保存带来源和审阅信息的经验。这里首先是非参数化经验、Skills/playbooks 更新，不是已经部署的神经网络在线训练。[自我改进代码][ETA-improve]、[经验晋级代码][ETA-memory]

论文采用隔离审阅、原任务复现、配对 held-out 验证和安全不退化门槛；五组探索性研究是自适应设计的序列，不是同一冻结协议下的五次独立确认实验。作者自报没有候选通过全部晋级门槛；例如 exact-task playbook 从 baseline `4/30` 降到 candidate `1/30`。这支持“能阻止无效或回退经验进入公共库”，不支持“持续学习已带来稳定成功率增益”。[论文 §6、附录 E][P5]

截止日最近的 RoboDojo RSI 更新也必须单列：作者自报四十二任务累计覆盖 `23/42→27/42→28/42`，即 `54.8%→64.3%→66.7%`。它跨 v108/v109/v111/v112，含人类诊断、批准、任务措辞补充和历史经验；每次 seed 为零、模型新会话、最多二百 tool calls，没有整回合 wall-clock/native-action 上限；存在未填满尝试预算的任务。额外 completion checker 还提供特权语义反馈。因此它不是 v112 单版本成绩、不是固定版本 pass@k、不是全自主 RSI，更不是某项修改的因果增益估计。[2026-10-16 官方协议与逐任务账本入口][ETA-RSI]

**真实机器人：代码、开发录像、正式评测三层分开。** 论文记录 UR5e–Robotiq 的开发录像：一段海绵到托盘包含完整操作链；另一段甜椒展示抓取和运输，但最终入篮关系没有归档视角支撑。选帧不是独立试验，不能估计成功率；作者还记录深度不完整及 protective stop 问题，但未建立原因诊断。[论文 §7、附录 G][P5]

截至固定 SHA，真实 MCP 的 `step_env`、`move_to` 和夹爪调用已有实际执行路径，`RealRobotEnv._apply_action` 接入关节/末端/夹爪命令，UR5e 驱动调用 RTDE 的 `moveJ/moveL`。**同一快照 `real/README.md` 仍把 MCP 控制写成 stub，属于文档—代码冲突；不能以该陈旧描述否认实现，也不能以实现存在反推硬件已通过测试。** Franka 驱动仍被文档列为 stub。[真实 MCP][ETA-real-core]、[真实环境][ETA-real-env]、[UR5e 驱动][ETA-ur5]、[真实部署 README][ETA-real-doc]

运动实现有单次位移/旋转拒绝门槛，但 `move_to` 的 `enable_collision_check` 参数未在该函数中进入碰撞验证路径，返回的 real reward 也不是经审计的任务完成判定。本报告只据此确认执行路径和验证缺口，不把它称为完整物理安全保障或部署成功。[真实 MCP 固定实现][ETA-real-core]

**结论：** perception—action—verification 的系统闭环已落到代码；内部预测式 world model、统一 VLA/WAM 调用和可靠持续学习收益仍是待证方向。[P5]、[C5]

### 3.2 EasyWAM：实际世界—动作训练基础设施，不是独立论文或持续学习系统

**论文关联。** 截止日未发现官方单独 EasyWAM 论文/arXiv ID。官方引用是软件 `@misc`，作者为 EasyWAM-Team；不能将 Awesome-WAM 的综述 `2605.12090` 冒充其方法论文。它明确参考 DreamZero、FastWAM、DiT4DiT、ImageWAM，但“基于/参考”不意味着一比一复现原论文全部数据与能力。[官方引用与致谢][C6]、[综述][P6]

**架构和连接点。** 固定代码存在五种模型，并有 `loss_video` 与 `loss_action` 的训练目标；这是可核验的世界—动作连接，不只是使用了 Video DiT 名称。训练和测试必须分别描述：

| EasyWAM 变体 | 已核验的结构/训练 | 动作推理时的边界 |
|---|---|---|
| Unified | 一个 Video DiT 接收视频、动作及状态 token；联合预测未来视频与动作 | 联合潜变量去噪；可以不解码 RGB，但不等于没有预测未来 |
| MoT | Video DiT 与 Action DiT 通过混合 self-attention 交互；训练含视频和动作损失 | 默认 first-frame-causal 路径只用当前视觉上下文预测动作，不生成未来视频；其他配置有不同路径 |
| MoT-Joint | 双专家共享混合注意力，联合去噪视频和动作 | 保留联合预测 |
| MoT-IDM | teacher-forced 条件视频参与动作训练，同时有视频目标 | 动作接口调用 joint inference，不能把训练使用真实未来视频误写成部署拥有真实未来 |
| Hidden | 视频分支的中间 hidden states 条件化独立 Action DiT | 利用未来视频预测相关特征；不必输出可见视频文件 |

来源：[Unified][EW-unified]、[MoT][EW-mot]、[MoT-Joint][EW-joint]、[MoT-IDM][EW-idm]、[Hidden][EW-hidden]、[官方架构分析][EW-blog]。MoT 配置中的视频与动作损失权重均为非零；所以本项目尤其能说明“训练期预测监督”和“推理期显式想象”不是同一个问题。[MoT 配置][EW-mot-config]

官方支持 Wan2.2-TI2V-5B、Cosmos-Predict2.5-2B 和 FLUX.2 Klein-4B，以及全参数和 LoRA 路径；公开 HF 包至少含 LIBERO 的 Unified/Wan 检查点、LoRA 权重与统计文件。它是开放训练基础设施，而非从头训练这些基础视频模型或开放其全部原始预训练数据。[README][C6]、[已核验模型文件][EW-HF]

**数据与评测。** 当前 LIBERO 配方使用 LeRobot v3 数据，拼接 agent/wrist 双相机；作者/配置记录三十三个 action/state 时间步、只解码九个视频时间戳。训练示例是离线数据上的梯度优化和 checkpoint 工作流，不是 robot rollout→新数据→在线再训练→部署的连续学习闭环。[LIBERO 数据指南][EW-data]

官方 LIBERO 默认四套任务、每任务五十 trials；LIBERO-Plus 在同一 LIBERO 检查点上做七类扰动，完整集合为一万零三十个扰动任务、每个一次；LIBERO-Pro 默认四套 × 五维扰动、二百任务 × 五十 trials，并明确只覆盖单扰动 leaderboard，不覆盖后来七案例扩展或复合扰动。还存在 RoboTwin、RoboDojo、RoboCasa365 的训练/评测入口，但支持入口不能替代每种设置的完成评测记录。[LIBERO 协议][EW-eval]、[Plus 协议][EW-plus]、[Pro 协议][EW-pro]、[README][C6]

**少量关键成绩，全部作者自报。**

| Wan2.2 全参数变体 | LIBERO 平均成功率 | LIBERO-Plus 平均 |
|---|---:|---:|
| Unified | 98.6% | 79.0% |
| Hidden | 98.8% | 77.6% |
| MoT | 96.7% | 71.7% |

来源：[固定 README][C6]、[架构分析][EW-blog]。Cosmos 下架构排序不同且 Plus 总平均更接近；官方分析也承认这些差异不足以因果证明“测试时预测视频必然提升泛化”。本报告因此不把小幅总分差异或 Wan 上单组比较升级为 WAM 架构定律。[EW-blog]

README 的训练吞吐比较是在相同 MoT 架构和指定硬件下对框架实现的作者测量；不能将工程加速当作新模型的算法增益。HF 卡片可能保留较早成绩，本报告数字以固定 GitHub 快照及对应协议为准。[C6]、[EW-HF]

**边界：** 这是五个重点项目中最直接的学习式 world—action 耦合实现；但其公开策略成功率并不验证预测的物理正确性、动作—想象一致性、反事实可靠性或真实接触安全。没有证据表明它已和 OpenETA 的经验晋级/实机轨迹组成持续学习系统。[模型代码][EW-code]、[评测文档][EW-eval]、[ETA 后续方向][P5]

### 3.3 RoboOmni：把声音、视觉、意图确认接到动作，但未接预测式世界模型

**论文与消歧。** OpenMOSS 对应 *RoboOmni: Proactive Robot Manipulation in Omni-modal Context*，arXiv `2510.23763`，ICLR 2026 Poster 已由官方 OpenReview 记录核验。另有 Dong Wang 等作者的同名 RoboOmni，标题为 *Actions Are Just Another Modality for Vision-Language Models*；本报告不引入后者 CALVIN/MTAP 成绩。身份需按标题、作者与 ID 而非短名匹配。[OpenMOSS 论文][A3]、[官方 OpenReview 检索记录][RO-venue-api]

**结构与目标。** 论文采用 Perceiver–Thinker–Talker–Executor；Qwen2.5-Omni 的多模态编码/Thinker 融合图像、语音和环境音，FAST+ 扩展离散动作词表并解码为末端控制。论文以 Qwen2.5-Omni-3B 为 backbone，动作词表二千零四十八、七维控制、六步 action chunk（作者描述）。目标是对话回复和专家动作序列的自回归最大似然，未出现未来世界状态预测项。因此它是增强输入与交互能力的全模态 VLA，不是已验证 WAM。[正文 §4、5.1、5.5][P3]

**数据来源。** OmniAction 的作者自报规模为 141,162 episodes。它以 Open-X Embodiment 任务/轨迹为基础，GPT-4o 改写隐式语境与后续对话，MOSS-TTSD/CosyVoice/Gemini-TTS 合成语音，再混入多人重叠、声音事件和背景声，配合抽样人工验证。不能将这个总量写成同量真实家庭对话或从环境在线探索所得数据。[正文 §3][P3]、[作者 HF 数据页][RO-data]

**协议与实机。** OmniAction-LIBERO-TTS 从四套共四十个操作任务构造六种上下文类型，即二百四十个任务变体；作者自报总体成功率 85.6%。OmniAction-LIBERO-Real 用真人录音驱动同一仿真任务，作者自报 76.6%；这个 “Real” 指语音来源，**不表示 LIBERO 在真实机器人上运行**。独立 WidowX 250S 实机实验则经过本地示范微调，展示意图推断、确认和抓取放置案例，未在所读正文中提供足以审计整体实机成功率的冻结任务—重复试验账本。[正文 §3.3、5.2–5.5、附录 B][P3]

比较还受输入与训练配置影响：文本/ASR 基线本来无法保留全部非语言音频信息；这些比较支持音频渠道和联合训练在该任务设置的价值，不证明所有端到端方案都优于所有模块化系统。[正文 §5.1、5.6][P3]

**仓库实现小于完整叙事。**

- 发布了 Thinker 训练、LIBERO 工具类、FAST 解码、HF 权重与 RLDS 数据，因而存在实际模型/数据资产。[训练实现][RO-train]、[LIBERO 工具类][RO-utils]、[权重文件][RO-HF]
- 训练入口使用 `Qwen2_5OmniThinkerForConditionalGeneration`，并冻结 audio tower；本次没有核验到同等完整的 Talker 联训与可直接运行的全模态实机交互栈。论文的端到端总体架构，不应转述为仓库全部组件均已完成联训发布。[RO-train]、[RO-utils]
- 固定 LIBERO 入口导入 `roboomni_utils_chatmode`，但已检查公开目录树只有 `roboomni_utils.py` 等文件；入口默认十次 rollout、seed 七。此为静态依赖缺口和仓库默认，不是对论文完整 rollout 预算的确认。[LIBERO 入口][RO-eval]、[公开目录][RO-tree]
- `experiments/bridge/run_widowx.py` 使用 Nora，相关工具类加载 Qwen2.5-VL；不能把这条 text-VLA 实机入口当作已复现论文全部音频确认能力。[实机入口][RO-real]、[Nora 工具类][RO-nora]

**结论：** perception—intent—confirmation—action 有论文与代码证据；未来状态生成、物理后果校验驱动的模型更新、部署中的持续学习均未证实。[P3]、[C3]

### 3.4 FRoM-W1：语言到全身运动和实机跟踪，不能外推为一般交互世界模型

**结构。** 论文 *FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions*，arXiv `2601.12799`。H-GPT 以 VQ-VAE 离散化全身人体运动，用 Llama-3.1-8B 的 LoRA 学习指令→CoT→运动 token；H-ACT 将运动重建为 SMPL-X，再重定向到 H1/G1，使用 PPO 训练 privileged teacher，并通过 DAgger 蒸馏可部署 student，最终交给 RoboJuDo。[正文 §2][P4]、[H-GPT 文档][FR-HGPT]、[H-ACT 文档][FR-HACT]

此处“生成未来运动”不等于“预测物体/环境受到动作干预后的状态”：H-GPT 输出主要是人体运动序列，H-ACT 跟踪参考并维持平衡；并未验证环境视觉感知→物体因果预测→交互动作的一体模型。手部目标可直接重定向，论文默认全身 RL 设置不把高自由度手指全部纳入训练，不能把手部运动展示解读为接触丰富灵巧操作已解决。[正文 §2.3、3.2][P4]

**数据与实验。** H-GPT 使用 HumanML3D-X、Motion-X 及 GPT-4o 生成的 CoT；Motion-X 的作者自报规模为 81,084 标注片段。H-ACT 预训练使用重定向 AMASS；原数据和人体模型有许可/授权限制，不是所有原始资源都可无条件重新分发。[正文 §2.2、3.1][P4]、[数据准备与限制][C4]、[H-GPT 文档][FR-HGPT]

HumanML3D-X 测人体运动分布、文本检索与多样性；δHumanML3D-X 用改写/噪声指令测试语言稳健性。作者自报标准集 FID：T2M-GPT 0.677，H-GPT 无 CoT 0.229、有 CoT 0.255；CoT 反而在分布内略差，而改写和噪声测试改善。FID 与文本匹配也不总同步，不能把低 FID 直接翻译成指令遵循或机器人任务成功。[正文 §3.1、表1–2][P4]

**RFT 与连续学习区分。** 推理阶段 RFT 将 teacher/student 针对目标参考运动在仿真中继续训练，再部署追踪；属于特定运动的适配。不是机器人经历物理任务后自动更新语言生成器/世界模型，也未展示跨任务保留能力、遗忘控制与长期经验累积。[正文 §2.3.2、3.2][P4]

**真实 H1/G1 证据与限制。** 论文明确展示两类实机的拟象、拉琴式动作、行走后模拟接电话，以及不同跟踪策略的部署；同时展示极端下蹲失稳、快速目标运动追踪失败，并讨论动作—指令不匹配、生成不稳定、自碰撞与 sim-to-real gap。这支持语言运动生成到真实控制的管线可行性，不支持一般家庭交互成功率或长期自主运行。[正文 §3.3、图12–14][P4]、[作者项目页][FR-site]

**开放实现范围。** H-GPT 模型、数据、loss/metric 和推理/评测代码存在；H-ACT 的 retarget、human2humanoid、RoboJuDo 通过子模块接入，初始化与外部模型准备是复现条件。HF 包包含 LoRA/tokenizer/baseline 及 H1/G1 student policy 资源，但 README 仍有 teacher TBD、AMASS-G1 下载链接待发布；H-GPT 新文档明确训练需自行构建 Lightning Trainer launcher，示例 `scripts.demo.py` 本身是推理/验证入口。故“完全开源框架”不能等价于“一条命令完整复现所有训练和实机实验”。[子模块定义][FR-submodules]、[H-GPT 文档][FR-HGPT]、[资源状态][C4]、[HF 文件][FR-HF]

**结论：** language—motion representation—proprioceptive control 的跨层连接较实在；场景预测式世界模型和连续实机学习没有被证明。[P4]

### 3.5 VLABench：评测和数据的战略价值高，但协议需锁版本

**资产与构建。** 对应 arXiv `2412.18194`，2024 年预印本、ICCV 2025 正式发表。作者自报一百类任务（六十 primitive、四十 composite）、163 类/2,164 个资产。MuJoCo/dm_control 环境覆盖常识、语义、空间、物理规则及长程技能组合；自动示范来自带先验/标注信息的技能库、运动规划和筛选，语言由 GPT-4o 增广。它是仿真基准与数据流水线，不是本身训练出的 world model 或机器人 foundation model。[正式正文 §3][VLA-final]、[仓库][C1]

**正式版协议。** ICCV 主实验 tracks 1–5 使用五个 primitive 任务、每任务二百训练示范，评测每任务五十 episodes；track 6 使用五个 composite 任务、其余设置一致（均作者自报）。指标包括 IS（接近正确操作对象）、PS（完成子任务比例）和严格 SR；VLM 的技能/参数/DAG 匹配则是另一种评测，不能当机器人执行成功率。作者自报 π0-Base Track 1 SR 13.2%、Track 6 SR 2.0%，但只支持这一数据预算/任务集/动作表示下的结论。[正式正文 §3.3–4.1、表2][VLA-final]、[补充 §9][VLA-supp]

**重要版本差异。**

| 材料 | 不应混用之处 |
|---|---|
| 早期 arXiv v1 | 采用较早的加权 Progress Score 与数据配方 |
| ICCV 正式论文 | PS 为完成子任务比例，并单列 IS、SR；Track 6 是长程 composite learning |
| 当前仓库 | Track 6 名为 unseen texture；Track 5 开放自选跨任务划分；标准 primitive 微调数据为十任务、每任务五百样本 |

来源：[arXiv 正文][P1]、[ICCV 正文][VLA-final]、[仓库 Evaluate 与数据说明][C1]。这些不是同一评测设置；当前较新 checkpoint 成绩也不能并入旧论文结果表证明连续进步来自单一因素。

**实机证据不能忽略，也不能夸大。** 补充材料 §10.2 有真实 Select Mahjong 小实验：作者自报五十条真实遥操作样本，π0-Base 仅真实数据为 `3/20` 成功，加入二百条仿真样本为 `5/20`。这是有限的 sim-data utility 证据，而不是一百任务基准的实机验证；缺乏更大重复和统计控制时，不能照抄作者措辞升级为“仿真质量已充分保证真实部署”。[ICCV 补充材料][VLA-supp]

仓库已有轨迹生成、RLDS/LeRobot 转换、固定 evaluation tracks、VLA/VLM 评测与 checkpoint 链接；完整训练基础设施、更多 composite/pretraining 数据和部分 leaderboard/模型接入仍在 preview/todo。双臂/人形接口测试也不等于已发布同覆盖范围的跨 embodiment 实机结果。[固定 README][C1]、[HF 数据资源][VLA-HF]

**结论：** 最值得利用的是检验语义、长程和分布外能力的任务设计，以及固定数据/初始状态，而不是把旧模型在高难低数据预算下的失败当作 VLA/WAM 整体路线失败。[VLA-final]、[C1]

## 4. 其余项目：定位与必要限制

### Embodied-Planner-R1

对应 *Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning*，arXiv `2506.23127`。Qwen2.5-7B-Instruct 在 ReAct 多轮文本交互中进行 grouped rollouts，以任务完成二值 reward 和 IPO 优化策略；训练数据主要由 ALFWorld/ScienceWorld 任务及在线训练探索形成，不依赖专家示范作为该方法的核心训练信号。仓库有 verl 环境服务器、训练脚本和两套权重。[正文 §3–4][P2]、[仓库][C2]、[ALFWorld 权重][R1-HF]

作者自报 ALFWorld 97.78%、ScienceWorld 79.92%；每任务一次、最多三十步，ScienceWorld 主实验启用全部五项简化功能。论文明确限制为纯文本环境、视觉扩展留待未来；因此这里的交互 RL 不是实机感知—运动学习或显式世界预测。摘要泛化 drop 与正文表2口径不一致，本报告不以该单值做战略推断。代码入口仍使用 `grpo` 名称，且公开 ScienceWorld easy 脚本 batch size 为六十四，与正文通用训练描述一百二十八任务并不完全相同，复现需核对实际配置，不能仅按论文名寻找 IPO 开关。[正文 §4、Limitations][P2]、[easy 训练配置][R1-script]

### Awesome-WAM

对应综述 arXiv `2605.12090`，正式 arXiv 标题为 *World Action Models: The Next Frontier in Embodied AI*；README BibTeX 中出现另一标题，引用以 arXiv 首页/PDF 为准。仓库是综述、论文索引、读论文博客与 leaderboard，不是 OpenMOSS 自己训练的单一 WAM，也没有自己的机器人实验可被归到组织业绩中。[arXiv][A6]、[正文][P6]、[仓库][C7]

综述区分 cascaded/joint WAM、显式/潜在预测、视频/动作耦合及 world models for VLA training/evaluation，并强调视觉保真、物理合理性、动作可执行性与两者因果一致性不是同一评测。其 §7 明确把控制变量不足、推理延迟、非视觉物理状态、长期规划、联合评测和物理安全列为开放问题。综述为路线提供分类与问题清单，不能为 OpenMOSS 的部署能力背书。[正文 §2、4、6–7][P6]

### VehicleWorld

对应 arXiv `2509.06736`，正式 venue 为 Findings of EMNLP 2025。它是人工构造的可执行车舱设备状态环境，作者自报三十模块、二百五十 API、六百八十属性、1,291 任务；SFC 读取状态并预测目标状态/生成修改代码，FC+SFC 先做状态感知式设备选择再使用 API。实验主要是 prompting，不是训练驾驶 world model。[正文 §3–6][P7]、[ACL 正式记录][VW-venue]、[仓库][C8]

指标按应改/不应改属性和改变趋势评估，多轮平均；作者自报 FC 平均 accuracy 51.1%、SFC 61.5%。SFC 的直接状态操作与真实车物理执行权限不同，不能用该比较证明神经世界模型优于 VLA。论文明确这是受控模拟、生产车辆部署需另外验证安全和可靠性；未见真实车辆实验或持续学习闭环。[正文 §5.3、6、Limitations/Ethics][P7]

### VehicleArena

对应 arXiv `2609.35916`。与 VehicleWorld 的车舱状态修改不同，它让 SUMO 负责物理执行，DA 只能发速度、制动、变道和路口动作，不能直接改车辆位置；Personal Agent 提乘客请求，Judge 基于冻结要求和执行历史验收，多车目标独立而共享道路。[正文 §3][P8]、[仓库架构][VA-arch]

作者自报一百开发任务、八十 Basic 加三十二 MultiLLM 测试任务，每模型每任务三次；MultiLLM 固定三辆 peer 的模型/角色，只替换被测车。世界按 0.1 秒仿真步演进于 agent 唤醒之间，**不表示物理时间与 LLM 墙钟推理延迟等同**。截止版本固定 SUMO 1.25.0、地图包和逐场景 deadline，匹配参考只替换 ego controller。[正文 §3–5][P8]、[冻结复现说明][C9]、[HF 地图][VA-HF]

到达、驾驶、请求满足、车舱及外部交通影响分别报告；无法验证的请求为 NA，不是成功或零分。作者主表最高到达率为 Basic 65.0%、MultiLLM 65.6%，并展示其他指标高不代表到达；分析图另外有 completion-weighted index，所以“主评测无单一总分”不等于全文绝不组合指标。其环境反馈闭环真实存在，但交通世界由模拟器计算，不是神经世界模型学习；也没有实车或终身学习证据。[正文 §4–5、附录 A][P8]

### FutureOmni

对应 arXiv `2601.13836`，ICML 2026 已由 PMLR 正式记录核验；终稿作者含 Min Zhang，仓库 BibTeX 漏列，元数据以正式记录为准。作者自报九百一十九视频、1,034 多选 QA、八领域；从视听前缀预测未来事件，并用延迟/逆因果和单模态冲突选项控制部分捷径，配有七千样本 OFF 指令微调。[PMLR][FO-venue]、[正文 §3–5][P9]、[HF 数据][FO-HF]

OFF 冻结视听 encoder、LoRA 更新文本 backbone，属于离线预测训练。作者自报最佳 QA accuracy 64.8%；Qwen2.5-Omni 总体从 47.48% 到 48.51%，但 Sound/Music 分项下降，不能写成所有能力都提高。新增采集视频也不自动证明所有闭源基础模型预训练绝无污染。该基准没有 agent 的动作干预、机器人执行或 world—action coupling；未来事件选择能力只能作为候选感知/推理诊断。[正文 §4–5、表4–6][P9]、[仓库][C10]

公开代码主要覆盖推理、特征提取及 LLaMA-Factory 适配，而不是全部实验模型训练环境。固定 `infer_vllm.py` 中有 `requires=True` 参数拼写和 `tranformers` 导入等静态缺口，说明“代码开放”不能等价于未经整理即可运行；本次未修改或执行它们。[推理入口][FO-code]

## 5. VLA、WAM、ETA：比较应围绕目标和证据，不应围绕命名

下表是分析性归纳。VLA/WAM 的概率形式来自综述；ETA 的能力与边界来自论文/实现，不能理解为三条路线已有公平胜负比较。[WAM 定义 §2][P6]、[ETA §2–4][P5]

| 维度 | VLA / RoboOmni 代表路径 | WAM / EasyWAM 代表路径 | ETA / OpenETA 代表路径 |
|---|---|---|---|
| 学习/决策对象 | 观测、指令/声音→动作；RoboOmni 加对话 | 未来观测/潜变量与动作共同监督或条件耦合 | 任务级下一工具选择、证据与反馈管理 |
| 典型目标 | `p(a∣o,l)`，扩展到全模态 | `p(o',a∣o,l)`；cascaded 可分解 | 不要求一个端到端动作生成网络 |
| 已有实际连接 | 输入语义与 motor token | 视频动力学表征与动作损失/去噪 | 感知、几何、控制、验证、经验候选 |
| 风险 | 输入理解与动作精度、数据覆盖、长程状态 | 预测错误、生成延迟、虚假物理一致性 | 调用/观测成本、接口误差、任务预算和工具能力瓶颈 |
| 持续学习 | SFT 不等于部署中学习 | 离线联合训练不等于 online adaptation | 有经验演化代码；论文未验证稳定增益 |
| 本范围最强实机证据 | RoboOmni WidowX 定性案例 | EasyWAM 本项目未证实 | UR5e 开发录像与执行代码，无审计成功率 |

FRoM-W1 属于语言—运动生成—追踪控制的分层路线，不宜强塞进一般操作 VLA/WAM 分类；VLABench 和其余环境/QA 项目则是数据、执行世界或评测资产。它们可辅助这些路线，但“benchmark 支持某能力”与“项目模型已经拥有某能力”不同。[FRoM 任务定义][P4]、[VLABench 设计][VLA-final]、[VehicleArena][P8]、[FutureOmni][P9]

### 对“是否真正整合”的最终判定

| 连接 | 截止日判断 | 一手证据 |
|---|---|---|
| 全模态 perception→action | RoboOmni 已有模型、训练与实验；公开复现链不完整 | [P3]、[RO-train]、[RO-eval] |
| 语言→人体运动→真实人形控制 | FRoM-W1 已实现并展示；一般环境交互未成立 | [P4]、[FR-HACT] |
| 学习式 world representation↔action | EasyWAM 已有联合监督与不同 inference 路径 | [EW-unified]、[EW-mot]、[EW-idm] |
| perception→act→observe→verify | OpenETA 已实现任务级闭环 | [P5]、[ETA-real-core] |
| experience→候选→审阅/验证 | OpenETA 有代码与实验；效果不稳定，论文无晋级候选 | [ETA-improve]、[P5] |
| 四者统一并稳定 continuous learning | 本范围未证实；论文未来方向不能视为已完成 | [P5]、[C5]、[C6] |

“闭环”至少分三层：控制器的本体反馈、任务级观察重规划、跨任务学习更新。前两者存在不自动证明第三者；使用 RL、生成 CoT、保存 memory、恢复 checkpoint 也分别不足以证明终身学习。[FRoM 控制训练][P4]、[R1 训练][P2]、[ETA 学习实验][P5]、[EasyWAM 工作流][C6]

## 6. 谨慎的战略推断

以下均为研究者推断，不是 OpenMOSS 官方承诺。

1. **更像“模型能力 + agent harness + 数据/评测”的组合布局，而不是一条单一大模型路线。** RoboOmni补全交互语境，FRoM-W1补全跨形态运动控制，EasyWAM降低 predictive policy 实验成本，OpenETA补全可审计任务闭环，VLABench提供较难的泛化测量。这一互补结构有证据；但共享作者、组织和网页链接不能证明它们已共享 backbone、统一训练数据或部署接口。[C1]、[C3]、[C4]、[C5]、[C6]
2. **近期可验证的技术抓手是把已有模型作为受约束能力接入任务闭环，而不是先宣称四模块统一。** ETA 明确容纳未来 VLA/WAM 工具；但是否有益需要同一 Planner、观测、seed、动作预算和数据下的配对消融。Full ETA 成绩和 EasyWAM 高分不能跨协议直接比较，更不能由此认定模块化或端到端谁普遍更优。[P5]、[EW-eval]、[VLA-final]
3. **WAM 投资宜关注最小有用预测，不宜把输出逼真视频当最终目标。** MoT 训练有预测目标而动作推理可省视频，说明世界监督与在线视觉想象可分离；当前跨 backbone 的排序不稳定，不支持“必须生成视频”或“预测必可删”的普适结论。应测预测是否改变错误动作、能否拒绝错误未来，以及闭环控制的收益/成本。[EW-mot]、[EW-blog]、[P6]
4. **continuous learning 是当前最关键的证据缺口之一，不是已跨越的里程碑。** R1 是文本环境参数学习，FRoM 是目标运动仿真适配，OFF 是离线语料微调，ETA 是非参数经验演化；这些尚未汇成经 held-out 实机验证的闭环。ETA 的负结果和 RSI 限定说明应优先消除混杂，再声称长期进步。[P2]、[P4]、[P9]、[P5]、[ETA-RSI]

建议主报告使用的表述：

> 截至 2026-10-17，OpenMOSS 已在全模态操作、人形运动控制、世界—动作模型训练与任务级 agent harness 上形成可核验的开放组件，并出现若干实际子闭环；但跨组件统一世界模型、持续经验学习收益及通用真实机器人可靠性仍未被现有公开实验共同证实。

上述表述由本报告五个重点项目的一手证据归纳，不意味着对组织其他未纳入范围项目的全面结论。[P3]、[P4]、[P5]、[C6]、[VLA-final]

若后续要证伪/增强该判断，最有价值的是三类新证据：同一系统下的 VLA/WAM 工具配对消融；想象未来与执行后观测的耦合/反事实评测；冻结候选、模型、任务措辞、checker、预算与 seed 后的跨任务学习及真实机器人长期非退化实验。此处是建议而非已实现事实。

## 7. 核心论文元数据

采用 academic-search 的结构化字段；`year` 在已确认会议论文时取正式出版年，另以 `first_public_year` 标明预印本年。`citation_count=null` 表示本次未查询引用数，绝不是零；不以引用数决定前沿项目的重要性。无正式 venue 证据的条目保留 arXiv preprint，不推断录用。元数据来源逐条列在 `metadata_url`。

VehicleWorld 的正式 DOI 另经 OpenAlex 与 Unpaywall API 确认 gold OA，合法全文为 ACL PDF；其余条目不把 arXiv 的标识 DOI 冒充正式出版 DOI。[ACL DOI/OA 页面][VW-venue]、[OpenAlex API][VW-OA]、[Unpaywall API][VW-UP]

```json
[
  {
    "project": "VLABench",
    "title": "VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics Manipulation with Long-Horizon Reasoning Tasks",
    "authors": ["Shiduo Zhang", "Zhe Xu", "Peiju Liu", "Xiaopeng Yu", "Yuan Li", "Qinghui Gao", "Zhaoye Fei", "Zhangyue Yin", "Zuxuan Wu", "Yu-Gang Jiang", "Xipeng Qiu"],
    "year": 2025,
    "first_public_year": 2024,
    "publication_date": "2024-12-24",
    "publication_type": "conference",
    "venue": "ICCV 2025",
    "arxiv_id": "2412.18194",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2412.18194v1",
    "code_url": "https://github.com/OpenMOSS/VLABench",
    "citation_count": null,
    "metadata_url": "https://openaccess.thecvf.com/content/ICCV2025/html/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.html",
    "source_platforms": ["arxiv", "CVF", "GitHub"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "Embodied-Planner-R1",
    "title": "Unleashing Embodied Task Planning Ability in LLMs via Reinforcement Learning",
    "authors": ["Zhaoye Fei", "Li Ji", "Siyin Wang", "Junhao Shi", "Jingjing Gong", "Xipeng Qiu"],
    "year": 2025,
    "publication_date": "2025-06-29",
    "publication_type": "preprint",
    "venue": "arXiv preprint; no formal venue verified",
    "arxiv_id": "2506.23127",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2506.23127v1",
    "code_url": "https://github.com/OpenMOSS/Embodied-Planner-R1",
    "citation_count": null,
    "metadata_url": "https://arxiv.org/abs/2506.23127",
    "source_platforms": ["arxiv", "GitHub"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "RoboOmni",
    "title": "RoboOmni: Proactive Robot Manipulation in Omni-modal Context",
    "authors": ["Siyin Wang", "Jinlan Fu", "Feihong Liu", "Xinzhe He", "Huangxuan Wu", "Junhao Shi", "Kexin Huang", "Zhaoye Fei", "Jingjing Gong", "Zuxuan Wu", "Yu-Gang Jiang", "See-Kiong Ng", "Tat-Seng Chua", "Xipeng Qiu"],
    "year": 2026,
    "first_public_year": 2025,
    "publication_date": "2025-10-27",
    "publication_type": "conference",
    "venue": "ICLR 2026 Poster",
    "arxiv_id": "2510.23763",
    "read_arxiv_version": "v3",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2510.23763v3",
    "code_url": "https://github.com/OpenMOSS/RoboOmni",
    "citation_count": null,
    "metadata_url": "https://openreview.net/forum?id=OJh7oBCYhL",
    "source_platforms": ["arxiv", "OpenReview", "GitHub", "HuggingFace"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "FRoM-W1",
    "title": "FRoM-W1: Towards General Humanoid Whole-Body Control with Language Instructions",
    "authors": ["Peng Li", "Zihan Zhuang", "Yangfan Gao", "Yi Dong", "Sixian Li", "Changhao Jiang", "Shihan Dou", "Zhiheng Xi", "Enyu Zhou", "Jixuan Huang", "Hui Li", "Jingjing Gong", "Xingjun Ma", "Tao Gui", "Zuxuan Wu", "Qi Zhang", "Xuanjing Huang", "Yu-Gang Jiang", "Xipeng Qiu"],
    "year": 2026,
    "publication_date": "2026-01-19",
    "publication_type": "preprint",
    "venue": "arXiv technical report; no formal venue verified",
    "arxiv_id": "2601.12799",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2601.12799v1",
    "code_url": "https://github.com/OpenMOSS/FRoM-W1",
    "citation_count": null,
    "metadata_url": "https://arxiv.org/abs/2601.12799",
    "source_platforms": ["arxiv", "GitHub", "HuggingFace"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "OpenETA",
    "title": "ETA: A New Agentic Paradigm for Embodied Tasks",
    "authors": ["Yitong Chen", "Zezheng Huai", "Sixian Li", "Yubang Wang", "Haozhe Zhang", "Yifei Zhang", "Hechang Chen", "Jingjing Gong", "Yu-Gang Jiang", "Xipeng Qiu"],
    "year": 2026,
    "publication_date": "2026-08-04",
    "publication_type": "preprint",
    "venue": "arXiv preprint; no formal venue verified",
    "arxiv_id": "2608.03924",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2608.03924v1",
    "code_url": "https://github.com/OpenMOSS/OpenETA",
    "citation_count": null,
    "metadata_url": "https://arxiv.org/abs/2608.03924",
    "source_platforms": ["arxiv", "GitHub"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "EasyWAM",
    "title": "EasyWAM: A Unified and Efficient Framework for Training and Evaluating World Action Models",
    "authors": ["EasyWAM-Team"],
    "year": 2026,
    "publication_type": "software",
    "venue": null,
    "arxiv_id": null,
    "paper_found": false,
    "full_text_status": "unknown",
    "full_text_note": "No standalone paper identified; this is a software citation, not an inaccessible paper.",
    "pdf_url": null,
    "code_url": "https://github.com/OpenMOSS/EasyWAM",
    "citation_count": null,
    "metadata_url": "https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/README.md",
    "source_platforms": ["GitHub", "HuggingFace"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "Awesome-WAM",
    "title": "World Action Models: The Next Frontier in Embodied AI",
    "authors": ["Siyin Wang", "Junhao Shi", "Zhaoyang Fu", "Xinzhe He", "Feihong Liu", "Chenchen Yang", "Yikang Zhou", "Zhaoye Fei", "Jingjing Gong", "Jinlan Fu", "Mike Zheng Shou", "Xuanjing Huang", "Xipeng Qiu", "Yu-Gang Jiang"],
    "year": 2026,
    "publication_date": "2026-05-12",
    "publication_type": "review",
    "venue": "arXiv survey; no formal venue verified",
    "arxiv_id": "2605.12090",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2605.12090v1",
    "code_url": "https://github.com/OpenMOSS/Awesome-WAM",
    "code_note": "Survey resources and leaderboard, not a trained model implementation.",
    "citation_count": null,
    "metadata_url": "https://arxiv.org/abs/2605.12090",
    "source_platforms": ["arxiv", "GitHub"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "VehicleWorld",
    "title": "VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction",
    "authors": ["Jie Yang", "Jiajun Chen", "Zhangyue Yin", "Shuo Chen", "Yuxin Wang", "Yiran Guo", "Yuan Li", "Yining Zheng", "Xuanjing Huang", "Xipeng Qiu"],
    "year": 2025,
    "publication_date": "2025-09-08",
    "publication_type": "conference",
    "venue": "Findings of EMNLP 2025",
    "doi": "10.18653/v1/2025.findings-emnlp.23",
    "arxiv_id": "2509.06736",
    "read_arxiv_version": "v1",
    "full_text_status": "open_pdf",
    "open_access_status": "gold",
    "pdf_url": "https://arxiv.org/pdf/2509.06736v1",
    "published_pdf_url": "https://aclanthology.org/2025.findings-emnlp.23.pdf",
    "code_url": "https://github.com/OpenMOSS/VehicleWorld",
    "citation_count": null,
    "metadata_url": "https://aclanthology.org/2025.findings-emnlp.23/",
    "source_platforms": ["arxiv", "ACL Anthology", "OpenAlex", "Unpaywall", "GitHub"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "VehicleArena",
    "title": "VehicleArena: A Realistic Urban Environment for Multi-Agent Driving",
    "authors": ["Jie Yang", "Jiajun Chen", "Jiazheng Zhou", "Mianqiu Huang", "Yining Zheng", "Yuxin Wang", "Xipeng Qiu"],
    "year": 2026,
    "publication_date": "2026-09-28",
    "publication_type": "preprint",
    "venue": "arXiv preprint; no formal venue verified",
    "arxiv_id": "2609.35916",
    "read_arxiv_version": "v2",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2609.35916v2",
    "code_url": "https://github.com/OpenMOSS/VehicleArena",
    "citation_count": null,
    "metadata_url": "https://arxiv.org/abs/2609.35916",
    "source_platforms": ["arxiv", "GitHub", "HuggingFace"],
    "fetched_at": "2026-10-17"
  },
  {
    "project": "FutureOmni",
    "title": "FutureOmni: Evaluating Future Forecasting from Omni-Modal Context for Multimodal LLMs",
    "authors": ["Qian Chen", "Jinlan Fu", "Changsong Li", "Min Zhang", "See-Kiong Ng", "Xipeng Qiu"],
    "year": 2026,
    "publication_date": "2026-01-20",
    "publication_type": "conference",
    "venue": "ICML 2026; PMLR 306",
    "arxiv_id": "2601.13836",
    "read_arxiv_version": "v2",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2601.13836v2",
    "code_url": "https://github.com/OpenMOSS/FutureOmni",
    "citation_count": null,
    "metadata_url": "https://proceedings.mlr.press/v306/chen26ad.html",
    "source_platforms": ["arxiv", "PMLR", "GitHub", "HuggingFace"],
    "fetched_at": "2026-10-17"
  }
]
```

`publication_date` 上表统一记录首次预印本公开日期；正式会议日期与年份由 `venue/year` 和出版页面给出。不能将该日期误读为所有论文的会议发表日期。

## 8. 官方来源索引

正文每个事实段附可点击官方 URL；下面保留其完整地址用于审计。固定 SHA 的代码链接优先于浮动 `main`。HF 资源确认文件存在，不代表重现作者实验或获取全部上游数据。

### 论文正文与正式出版

[P1]: https://arxiv.org/html/2412.18194v1
[P2]: https://arxiv.org/html/2506.23127v1
[P3]: https://arxiv.org/html/2510.23763v3
[P4]: https://arxiv.org/html/2601.12799v1
[P5]: https://arxiv.org/html/2608.03924v1
[P6]: https://arxiv.org/pdf/2605.12090v1
[P7]: https://arxiv.org/html/2509.06736v1
[P8]: https://arxiv.org/html/2609.35916v2
[P9]: https://arxiv.org/html/2601.13836v2
[A3]: https://arxiv.org/abs/2510.23763
[A6]: https://arxiv.org/abs/2605.12090
[A8]: https://arxiv.org/abs/2609.35916
[A9]: https://arxiv.org/abs/2601.13836
[VLA-final]: https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.pdf
[VLA-supp]: https://openaccess.thecvf.com/content/ICCV2025/supplemental/Zhang_VLABench_A_Large-Scale_ICCV_2025_supplemental.pdf
[RO-venue]: https://openreview.net/forum?id=OJh7oBCYhL
[RO-venue-api]: https://api2.openreview.net/notes/search?term=RoboOmni&limit=10
[VW-venue]: https://aclanthology.org/2025.findings-emnlp.23/
[FO-venue]: https://proceedings.mlr.press/v306/chen26ad.html
[VW-OA]: https://api.openalex.org/works/https://doi.org/10.18653/v1/2025.findings-emnlp.23
[VW-UP]: https://api.unpaywall.org/v2/10.18653/v1/2025.findings-emnlp.23?email=academic-search@example.org

### 固定官方仓库

[C1]: https://github.com/OpenMOSS/VLABench/tree/cf588fe60c0c7282174fe979f5913170cfe69017
[C2]: https://github.com/OpenMOSS/Embodied-Planner-R1/tree/50cb93cfc02725a1864edc39c0a2170ab233ba61
[C3]: https://github.com/OpenMOSS/RoboOmni/tree/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895
[C4]: https://github.com/OpenMOSS/FRoM-W1/tree/457d84d44e638c1070473dd48cedde19752cad09
[C5]: https://github.com/OpenMOSS/OpenETA/tree/b02260228e8c90fbc954e4eeb4ce4175249da9ba
[C6]: https://github.com/OpenMOSS/EasyWAM/tree/dd5b06153abd1c04eff99a3933dc09baf0906fa2
[C7]: https://github.com/OpenMOSS/Awesome-WAM/tree/aa6cd0508a089e7556dd2fa0347e847c91ca986b
[C8]: https://github.com/OpenMOSS/VehicleWorld/tree/6536be9b88a92619357d6c02325af00a8f3c25c1
[C9]: https://github.com/OpenMOSS/VehicleArena/tree/421d5fdc4e1802d03844e85e4ac3e612fa6dfc88
[C10]: https://github.com/OpenMOSS/FutureOmni/tree/5b292a0438e5b829d8777bb6426afaa58cf12469

### OpenETA

[ETA-arch]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/docs/architecture.md
[ETA-pipeline]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/docs/agent-action-pipeline.md
[ETA-RSI]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/docs/robodojo-rsi-results.md
[ETA-improve]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/agent/runtime/self_improvement.py
[ETA-memory]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/agent/runtime/promoted_memory.py
[ETA-real-doc]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/real/README.md
[ETA-real-core]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/real/mcp/observation_core.py
[ETA-real-env]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/real/env.py
[ETA-ur5]: https://github.com/OpenMOSS/OpenETA/blob/b02260228e8c90fbc954e4eeb4ce4175249da9ba/real/robots/ur5e.py

### EasyWAM

[EW-code]: https://github.com/OpenMOSS/EasyWAM/tree/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model
[EW-unified]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_unified.py
[EW-mot]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_mot.py
[EW-joint]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_mot_joint.py
[EW-idm]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_mot_idm.py
[EW-hidden]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/src/model/easywam_hidden.py
[EW-mot-config]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/configs/model/easywam_mot_wan22.yaml
[EW-blog]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/blogs/blog01_arch_zh.md
[EW-data]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/instructions/data/libero.md
[EW-eval]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/instructions/benchmark/libero.md
[EW-plus]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/instructions/benchmark/libero_plus.md
[EW-pro]: https://github.com/OpenMOSS/EasyWAM/blob/dd5b06153abd1c04eff99a3933dc09baf0906fa2/docs/instructions/benchmark/libero_pro.md
[EW-HF]: https://huggingface.co/OpenMOSS-Team/EasyWAM-Unified-Wan22/tree/794bd9bf249090e2336a0a1534eb43ac8eb69593

### RoboOmni / FRoM-W1 / 其他资产

[RO-train]: https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/training/train_omni.py
[RO-utils]: https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/experiments/libero/roboomni_utils.py
[RO-eval]: https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/experiments/libero/run_libero_eval.py
[RO-tree]: https://github.com/OpenMOSS/RoboOmni/tree/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/experiments/libero
[RO-real]: https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/experiments/bridge/run_widowx.py
[RO-nora]: https://github.com/OpenMOSS/RoboOmni/blob/935b3d7fa5d1fcd55d6ab32ba8d4fa1285961895/experiments/bridge/nora_utils.py
[RO-HF]: https://huggingface.co/OpenMOSS-Team/RoboOmni/tree/3a84313809c3b16fad1295a2e3814754bf179cfe
[RO-data]: https://huggingface.co/datasets/OpenMOSS-Team/OmniAction
[FR-HGPT]: https://github.com/OpenMOSS/FRoM-W1/blob/457d84d44e638c1070473dd48cedde19752cad09/H-GPT/README.md
[FR-HACT]: https://github.com/OpenMOSS/FRoM-W1/blob/457d84d44e638c1070473dd48cedde19752cad09/H-ACT/README.md
[FR-submodules]: https://github.com/OpenMOSS/FRoM-W1/blob/457d84d44e638c1070473dd48cedde19752cad09/.gitmodules
[FR-HF]: https://huggingface.co/OpenMOSS-Team/FRoM-W1/tree/c3c162263f52db7466841b6194695d7bd1c50478
[FR-site]: https://openmoss.github.io/FRoM-W1/
[VLA-HF]: https://huggingface.co/VLABench
[R1-HF]: https://huggingface.co/OpenMOSS-Team/Embodied_Planner-R1-Alfworld/tree/28b5b82aba51505f31db9aeb3d0f40d889195471
[R1-script]: https://github.com/OpenMOSS/Embodied-Planner-R1/blob/50cb93cfc02725a1864edc39c0a2170ab233ba61/examples/grpo_trainer/sci_easy.sh
[VA-arch]: https://github.com/OpenMOSS/VehicleArena/blob/421d5fdc4e1802d03844e85e4ac3e612fa6dfc88/docs/02-architecture.md
[VA-HF]: https://huggingface.co/datasets/OpenMOSS-Team/VehicleArena/tree/1a1667d5215db7367fd039ca486711cf5abe933e
[FO-HF]: https://huggingface.co/datasets/OpenMOSS-Team/FutureOmni/tree/9229dbddf9c2e846eefb4a180a4e220a3f98983c
[FO-code]: https://github.com/OpenMOSS/FutureOmni/blob/5b292a0438e5b829d8777bb6426afaa58cf12469/eval/infer_vllm.py
