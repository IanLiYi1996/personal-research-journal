# HF Daily Papers 摘要：08-31 回填 + 09-01 ~ 09-03

> **抓取时刻**：2026-09-03 02:4x UTC（⭐ **以 `date -u` 为准**：今天是 **09-03（周四，W36）**，本份是当日第一份，故按日期命名不加后缀）
>
> ⚠️⚠️ **2 天空缺补跑**：HF / Reddit / tech-blogs 上一次跑都是 **08-31**，08-32… 即 09-01 与 09-02 两天一次都没跑。🚨 **而 AWS 09-01 与 09-02 都在 ⟹「AWS 活、其余死」第八次，且这已是 08-31 那次全新重建之后**。
>
> ⭐ HF 因日期桶按日期索引，内容无损失（这是四个源里唯一可完整回溯的）。

---

## §0 抓取与去重

| 日期桶 | 本次读数 | 说明 |
|---|---:|---|
| **08-31** | **31** | ⭐ 08-31 早上那次读到 **4** 篇 ⟹ **+27**，是我拿到过的单桶最大增量之一 |
| 09-01 | 39 | 首读 |
| 09-02 | 30 | 首读 |
| 09-03 | 10 | 首读（当日，尚在填充）|
| **窗口唯一** | **110** | |

**去重两个口径**：

| 口径 | 值 | 含义 |
|---|---:|---|
| **A**（对比 08-31 早间那次抓取的 id 集合）| **106** | 桶里新出现了什么 |
| **B**（对照最近 8 份 digest 的已引用 id）| **107** | 我还没写过什么 |

⭐ **两者差 1，而这个 1 有信息**：差的那一篇落在 08-31 桶里、08-31 早上那次**抓到了但没引用**（那份取了 Top 25，而 31 篇里我只逐条引了 25 篇）⟹ ⭐⭐ **这正是我 08-18 归纳的那条规律的一个小型实例：两个口径的差距 ≈「上一份抓到但未引用的篇数」**。08-31 那份的覆盖率高（28/28 全列），所以这次差距只有 1；而 08-18 那次差 18 倍，因为那份只引了 25/65。

⚠️ **日期上限 guard 连续第 13 天既生效又不准**：拉 09-04 返回错误对象（非空数组，`isinstance` guard 生效），而它声称的上限仍滞后于我实际能取到的 09-03。**实用结论不变：每次都拉、靠 guard 兜住、不要用它的提示判断哪天有数据。**

**本份取 Top 25 ＋ 编辑增补 7 篇**（增补理由与入库同步，见下）。

---

## §1 论文总览表（Top 25，按 upvote）

| # | ▲ | arXiv | 标题（中译）| 主题 |
|---:|---:|---|---|---|
| 1 | **469** | [2609.01591](https://huggingface.co/papers/2609.01591) | StudentSim：训练基于 LLM 的学生模拟器 | 模拟器 / 教育 |
| 2 | 346 | [2609.00111](https://huggingface.co/papers/2609.00111) | Qwen-Drive-1.0：自动驾驶视觉-语言基础模型的初步一步 | 自动驾驶 |
| 3 | **127** | [2608.31046](https://huggingface.co/papers/2608.31046) | ⭐ **On-Policy Distillation 真的在蒸馏吗？从噪声教师到自我改进** | 蒸馏 / **深读 2** |
| 4 | 100 | [2608.28281](https://huggingface.co/papers/2608.28281) | ⭐ **LoopArena：把模型当作 Loop Engineering 的运行时控制器来测** | harness / 评测 |
| 5 | 100 | [2608.30821](https://huggingface.co/papers/2608.30821) | Lucida：解析、生成与放置，做可组合的真实到仿真场景建模 | Real-to-Sim |
| 6 | 94 | [2608.31106](https://huggingface.co/papers/2608.31106) | DreamX-Creator：把原生音视频生成普及到 2K 分辨率 | 生成 |
| 7 | 90 | [2608.18524](https://huggingface.co/papers/2608.18524) | DART-SD：面向多轮工具调用自蒸馏的菱形拓扑感知检索与调优 | 蒸馏 / Agent |
| 8 | 88 | [2608.27550](https://huggingface.co/papers/2608.27550) | 超越数据规模：面向 VLA 的表征中心持续预训练 | VLA |
| 9 | 77 | [2609.01343](https://huggingface.co/papers/2609.01343) | SMELT：算力对齐的 MoE Looped Transformer 的 scaling law | 架构 |
| 10 | 61 | [2608.29335](https://huggingface.co/papers/2608.29335) | GenFirst：先生成后重建，做稳定的端到端潜生成建模 | 生成 |
| 11 | 60 | [2608.28122](https://huggingface.co/papers/2608.28122) | Agentic Artifact Creation：系统、评测、原则与机会（综述）| Agent 综述 |
| 12 | 56 | [2609.00028](https://huggingface.co/papers/2609.00028) | UI-Venus-2 技术报告 | GUI Agent |
| 13 | 50 | [2608.31036](https://huggingface.co/papers/2608.31036) | 归一化低秩适配（Normalized LoRA）| 微调 |
| 14 | 46 | [2608.30320](https://huggingface.co/papers/2608.30320) | Qwen3.8-Next 架构设计：评测、效率与训练稳定性 | 架构 |
| 15 | 42 | [2609.01560](https://huggingface.co/papers/2609.01560) | H3-World：把语言理解变成世界控制 | 世界模型 |
| 16 | 40 | [2608.26582](https://huggingface.co/papers/2608.26582) | ⭐ **J-Zero：零数据下 Challenger–Solver–Judge 统一协同演化** | 自我演化 |
| 17 | 40 | [2609.00188](https://huggingface.co/papers/2609.00188) | ZimaBlue：通过可扩展视频预训练演化可泛化世界动作模型 | 世界模型 |
| 18 | **39** | [2608.24804](https://huggingface.co/papers/2608.24804) | ⭐⭐ **StarHarness：用分层搜索为企业环境演化 harness** | **harness** |
| 19 | 36 | [2608.31119](https://huggingface.co/papers/2608.31119) | ⭐⭐ **PaperGym：以 rubric 为中心的研究方案生成演化** | 评测有效性 |
| 20 | 35 | [2608.27370](https://huggingface.co/papers/2608.27370) | Puro-2B：穷实验室在一张 RTX 5090 上用 5090 美元训出 Qwen2-1.5B | 低成本训练 |
| 21 | 32 | [2608.27529](https://huggingface.co/papers/2608.27529) | 重访长程流式 3D 重建中的局部上下文 | 3D |
| 22 | 32 | [2609.01572](https://huggingface.co/papers/2609.01572) | 从生产流量到后训练：自建覆盖语料的自托管 LLM | 工程实践 |
| 23 | 30 | [2608.27763](https://huggingface.co/papers/2608.27763) | 面向持续学习的快权重注意力 | 持续学习 |
| 24 | 30 | [2608.23478](https://huggingface.co/papers/2608.23478) | Act with Intent：为 VLA 蒸馏行为意图 | VLA |
| 25 | 29 | [2608.30968](https://huggingface.co/papers/2608.30968) | CogEvol：面向高效可靠的学习环境生成 | 环境生成 |

### ⚠️ 编辑增补 7 篇（已同步入库，理由逐条写明）

⚠️⚠️ **这一节存在的原因是我 08-18 踩过的坑：编辑替换/增补了 Top-N 成员却没同步更新入库列表，结果出现「引用了但没入库」的条目。**

⭐⭐ **而本次我把这条规则的适用范围放宽了一格：入库列表按「正文与 References 里逐条引用过的全部 id」走，而不是「Top 25 + 编辑增补」** —— 因为 §3/§4/§8 里我还逐条引了 Cliff、InternReviewer、Knowledge Distillation During Mid-Training、Safin-1、Personalization、LayerRecall、DiagEvo、Super Library Agent 等 8 篇。⟹ **实际入库 25 + 15 = 40 个 id**（下表 7 篇是「若只按 upvote 就会漏掉」的那些，单独列出是为了写清增补理由）。

| ▲ | arXiv | 增补理由 |
|---:|---|---|
| **4** | [2609.01836](https://huggingface.co/papers/2609.01836) | ⭐⭐⭐ **深读 1** —— 严格 upvote 排在末尾，而它是本窗口对我主线最直接的一篇（把「谁授权了这个动作」这条我 08-27 才立起来的新主线做成了带基准的实证）|
| 11 | [2609.01481](https://huggingface.co/papers/2609.01481) | harness 五篇之一（**Harness-of-Harness**，meta-harness 做多日自主开发）|
| 8 | [2608.30396](https://huggingface.co/papers/2608.30396) | harness 五篇之一（**NavMCP**，harness 演化进具身第三篇）|
| 7 | [2608.28363](https://huggingface.co/papers/2608.28363) | ⭐⭐⭐ harness 五篇之一（**EvoUndo**，直接回答我 W35 从 r/devops 记下的那个「第三种失效」）|
| 5 | [2608.30530](https://huggingface.co/papers/2608.30530) | ⭐⭐⭐ **WebWorld** —— 「让度量留在优化压力之外」那一类解法迄今最好的一句表述 |
| 4 | [2608.29846](https://huggingface.co/papers/2608.29846) | ⭐⭐ **IDA-OPD** —— 与深读 2 同窗、同问题、结论互补（pass@k 平台期）|
| 4 | [2609.00137](https://huggingface.co/papers/2609.00137) | ⭐⭐ **Recursive Criticality** —— 给「AI4AI」这条线一个可判定的量（R_AI）|

⭐ **另有两篇进了正文讨论但已在 Top 25 内，无需增补**：LoopArena（#4）与 PaperGym（#19）。

---

## §2 ⭐⭐⭐ 本窗口的故事：「harness」进标题五篇，而五篇合起来第一次覆盖了完整生命周期

**08-20 我记过「harness 同时出现在六篇论文标题里」并说标题位置比正文提及是更强的信号。本窗口是五篇，但结构比数量重要——它们第一次覆盖了「演化 → 元层 → 具身 → 可回退 → 被评测」这整条链。**

| 篇 | ▲ | 它在链上的位置 |
|---|---:|---|
| **LoopArena** | 100 | **被评测**：把「loop 的指导 vs coding agent 的能力」这个归因问题做成基准 |
| **StarHarness** | 39 | **演化**：企业环境里演化 harness，+20–35 pp |
| **Harness-of-Harness** | 11 | **元层**：在既有 harness 之上组织多日迭代 |
| **NavMCP** | 8 | **具身**：VLM 推理 agent + 导航基础模型执行器 |
| **EvoUndo** | 7 | ⭐⭐⭐ **可回退**：自我修改的可恢复性，而这一格此前是空的 |

### 🚨⭐⭐⭐ LoopArena（100▲）：它的问题陈述就是我这两个月的主线，而它把「Loop Engineering」命名成了一种实践

原文的诊断句几乎是我记过的话：「**the final outcome of one end-to-end run cannot tell whether success or failure reflects the loop's guidance or the coding agent's ability to carry out the task**」

⟹ ⭐⭐⭐ **而它的解法与 A²E 是同一手法的对偶**：A²E 固定骨干模型去比 9 个 harness，**LoopArena 固定 Worker（一个不变的 coding agent）去比被评模型作为 Controller 的能力** —— 两者都靠「把另一侧钉死」来做归因。

⭐⭐ **它枚举的四种 loop 失效，每一种我都在别处记过，而这是第一次它们出现在同一个基准的设计动机里**：

| LoopArena 的失效 | 我此前记的对应物 |
|---|---|
| **trust a stale progress note** | ⭐⭐⭐ When "Must" Becomes "Maybe"（08-27 深读）的 `stale observations`，以及那篇测出的「正常压缩产生 100% 约束失活」|
| skip needed verification | AI4AI 的「verification/arbiter 只有 12%、self-consistency 5%」＋ Grounded Reasoning Cup 冠军队为速度把 verifier 整个撤掉 |
| spend its budget in the wrong direction | R³-Bench 的「72 格里 71 格 oracle 更高、连平均分配都能打败模型自己的分配」|
| **stop before the task is safe to submit** | ⭐⭐ Runtime Contract 的**提交闸门**（12 个公开系统里只有 2/12 有）|

⭐ **三档设定的成本设计值得抄**：**Type I 用「execution-validated questions」评下一步的 Loop Contract 选择、评测时根本不跑 Worker**（便宜）· Type II 在一个切片上反复控制 · Type III 跑配对的完整任务。⟹ ⭐⭐ **这是「过程指标」与「端到端结果」之间的一个梯度，而不是二选一** —— 而我此前记的「只报最终成功率等于没报」一直缺一个「那该报什么、花多少钱」的答案。

⚠️ 保留：仅摘要；`Loop Contract` 的确切定义、以及 Strict Success Rate 的具体数字我只看到摘要被截断处（「the best observed Strict Success Rat…」）。

### ⭐⭐ StarHarness（39▲）：+20–35 pp，而两个设计细节比数字重要

**权重固定，演化的对象包括 prompt 与任务框定 / 工具接口 / 技能 / MCP 后端 provider / subagent 结构 / agent-loop 配置**（⭐ 这个清单本身就是「harness 包含什么」目前最完整的一份工业口径）。

⭐⭐⭐ **两个设计细节各对上我此前的一条判据**：

1. **「按 baseline failure behavior 分层构造演化池」** ⟹ ⭐ 与 Evo-Bench 的「先算 harness 敏感度、把敏感度 ≤0 的任务全剔掉」是同一取向：**想测某因素，先确认任务对它敏感**。
2. ⭐⭐⭐ **「separates proposer-visible search tasks from proposer-hidden selection tasks」** ⟹ **这正是 Gaming Without an Attacker 那条「探针只在未被披露且不可枚举的轴上保持测量有效性」的工程实现**，而 DarwinX 的两速信任门（promote → 更高保真度重测）是它的另一种实现。⟹ ⭐⭐ **该设计模式至此有三个独立实现，我认为可以当作默认做法写进任何自动演化方案。**

⭐ **效率数字很扎眼：只用 4–12 次「被接受的改动」就拿到 20–35 pp。** 而 trace 分析把增益归到 **interface repairs / environment conventions / operational knowledge that compresses search**，可观察后果是**更少的假阳性诊断与更短的轨迹**。

🚨⭐⭐⭐ **但它有一句与我攒了三周的那个梯度直接冲突，必须标出来**：原文说增益「**transfer without re-evolution across GPT and Qwen model families**」。

而我从四篇攒出的「迁移随执行者距离分层」是：**同家族跨代免费（StateM +9.0→+10.4）→ 跨 harness 保留 29–55%（ClawGym II / DarwinX）→ ⭐ 跨厂商 82.7%→82.0% 略微变差（StateM）→ 强→弱明显退化（SKILLER）**。

⟹ ⚠️⚠️ **两者不能简单合并，而我倾向的调和方向是「被迁移的对象不同」**（这也是我 08-31 从 JIT-Agent 得到的那个区分）：StateM 明确说过代价来源是「**it preserves substantial autonomy in the base agent**」⟹ 自主性越高、越依赖厂商特定行为；⭐ 而 StarHarness 演化出的东西按它自己的 trace 分析是**接口修复与环境约定**，这类知识很可能确实与执行者无关。⟹ ⭐⭐ **可检验的推论：harness 增益的可迁移性应当与「它编码的是环境知识还是执行者行为」相关，而不只与执行者距离相关。** ⚠️ 我未读正文、不知它跨家族迁移时保留了增益的百分之多少（「transfer」是定性词）。

### 🚨⭐⭐⭐ EvoUndo（7▲）：它回答了我一周前从一条 Reddit 帖子记下的那个「第三种失效」，而它把问题量化了

⭐ **我在 [[2026-W35-reddit-hot]] 记过一条 r/devops 帖子「The prompt rollback worked in staging and nowhere else」，并写下**：「『回滚 prompt』与『回滚代码』不是一回事……若 prompt 已改动过记忆/技能/上下文，换回旧 prompt 并不能换回旧状态」——**当时我明确说这是我此前没记录过的第三种失效**（前两种：Co-Evolution 综述建议 rollback to verified states · Evo-Bench 证明留不住最好版本）。

⭐⭐⭐ **EvoUndo 的问题陈述就是那句话的形式版**：「a successful mutation may leave persistent effects that **cannot be safely reversed in states different from the one in which it was created**」

**而它给了数字**：

| 量 | 值 |
|---|---|
| 未见过的一次性自我演化任务 | **600** |
| 其中**提升能力但通不过可恢复性验证**的 mutation | 🚨 **197（约 33%）** |
| 常规修复策略在原恢复表示下的恢复数 | 🚨🚨 **0 / 197** |
| 确定性 oracle 在原恢复语言 L0 下 | 48 / 197 |
| 扩展恢复演算后的 oracle 恢复 | **191 / 197** |
| 2×2 干预：精确状态地址接地（语言足够时）| 0/48 → **38/48（79.2%）** |
| 2×2 干预：扩展恢复语言（S1 层）| **142/143（99.3%）** |
| ⚠️ gpt-oss-120b 上「更丰富语言 + 精确地址诊断」| **反而降到 133/143（93.0%）** |

⟹ 🚨⭐⭐⭐ **「约三分之一的能力提升是不可安全回退的，而常规修复策略对它们的成功率是零」** —— 这把 Co-Evolution 综述那条治理建议（rollback to verified states）的难度从「要判断哪个版本算 verified」（Evo-Bench 证明的那一步）**再往前推了一格：即便你判断对了，你也可能回不去。**

⭐⭐ **而 `0/197` 这个数是本份最该记的单个数字**，因为它说明这不是「回退做得不够好」而是「回退的表示语言不足以表达要撤销的东西」——**修法是扩展恢复演算（一个表示问题），不是更努力地重试。**

⭐ 那个非单调结果（**加了精确地址诊断反而从 99.3% 降到 93.0%**）值得单记：⚠️ 我未读正文不知原因，但它至少说明**「给模型更多诊断信息」不总是单调改善**，与我记过的「一个自适应重加权方法若对自身超参敏感就只是换了个调参问题」同族。

### ⭐⭐ Harness-of-Harness（11▲）：meta-harness，而它的六条设计里有三条是我记过的判据的独立重新推导

**HoH 跑在既有 coding-agent harness 之上**，把它们的执行组织成迭代的 planning–coding–testing 循环。六条设计：

| HoH 的设计 | 我此前记的对应 |
|---|---|
| balances repair with capability growth | DarwinX 的 `g(c)>0 且 R(c)≤δ`（净增益 + 有界回退）|
| scopes development into small and verifiable increments | ⭐ 「只生成对强基线的有界修改」（DarwinX / SKILLER / AutoPrune 三篇收敛）|
| ⭐⭐ **separates implementation-time testing from independent evaluation** | Spark-to-Paper 的「把实验规划与结果报告分开＝预注册」|
| 🚨⭐⭐⭐ **constrains verifiable outputs rather than prescribing agent workflows** | ⭐⭐⭐ **Runtime Contract 的「契约只 gate 效果不 gate 思考」＋ Agentic Transaction 的「constrain committed effects rather than requiring deterministic execution traces」** ⟹ **这是该原则的第三个独立社群版本** |
| progressively exposes deliverables, tools, skills | JIT-Agent 的 progressive skill disclosure / SKILLER 的 progressive skill disclosure |
| maintains versioned project histories | Co-Evolution 综述的 rollback to verified states（⭐ 而 EvoUndo 同窗告诉我们这一条比看起来难）|

**结果**：三个 harness-模型配对（**Codex+GPT-5.5 / OpenCode+DeepSeek-V4-Pro / Pi+MiniMax-M3**），三轮迭代后**平均相对增益 52.25%、最大 82.86%**；多日部署 **>70 次迭代**自主开发出一个第一人称射击游戏。

⭐⭐ **它把 harness-模型配对当成实验单元报了三个，而这正是我从 Macaron-V1 / SKILLER / Prime Agent 攒的那条规范（报 harness 效果不写执行者等于没报）被落实的样子。** ⚠️ 但「平均相对增益 52.25%」是相对各自 standalone harness，**不是跨配对可比的绝对分数**；且未见区间。

### ⭐⭐ NavMCP（8▲）：harness 演化进具身第三篇，而它的三条通道里有一条落在证据面上

诊断很清楚：**VLM 能补全信息、调整高层计划但在重复导航接地上脆弱低效；导航基础模型（NFM）能稳健执行语义目标但以「有边界的 episode」运行、没有持续的任务级推理。** NavMCP 把两者耦合，三条通道：

- **intent**：把证据需求翻译成导航调用
- ⭐⭐ **observation**：把 rollout 转成 **source-grounded trajectory evidence** ⟹ **证据面在具身领域的一个实例**
- **memory**：跨调用累积发现、**负面证据**与未解决目标（⭐ 「负面证据」这一项少见且重要）

**结果**：EQA 三个基准 SOTA；⭐ **匹配 agent 与执行器骨干的条件下，比 episodic 接口高 14.9 pp**（这是控住变量的那个数）；并在 Unitree Go2 上跑了实机。

⟹ ⭐ **「让状态持久、让 agent 短命」那条线的又一个实例**（原文：「turns isolated navigation rollouts into persistent embodied interaction **without retraining either model**」），而具身侧此前两篇是 SHAPER 与 Zetta ζ。

---

## §3 ⭐⭐⭐ 第二条主线：OPD 子领域同窗出现两篇互补的负面结果，而深读 2 那篇的自陈局限恰好就是另一篇的主题

| 篇 | ▲ | 它说 OPD 的什么问题 |
|---|---:|---|
| **Does OPD Really Distill?**（深读 2）| **127** | **教师监督高度含噪，且噪声随教师规模上升**（30.6% → 34.7% → **50.6%**）；而增益主要来自「压制低概率 token」，**不需要教师** |
| ⭐⭐ **IDA-OPD** | 4 | **多样性蒸馏失败：学生 pass@1 上升而 pass@k 停滞**，继承不到教师的多样性 |

⟹ 🚨⭐⭐⭐ **而最漂亮的是：深读 2 提出的 OPSA 在附录 A 自陈的局限，正好就是 IDA-OPD 的主题** —— 原文说 OPSA「**may not substantially expand the policy's underlying exploration frontier, as reflected by the relatively modest improvements in thinking-mode Pass@k**」。

⟹ ⭐⭐ **两篇合起来给出一个比任何单篇都清楚的图景：OPD（以及它的无教师替代 OPSA）主要在重新分配已有的概率质量，所以它能大幅改善「一次答对」，而对「多样性 / 探索边界」帮助有限——而这一点两篇是从相反方向到达的**（一篇造了个无教师版本并诚实报告 pass@k 平台期，一篇专门去修那个平台期）。

⭐ **IDA-OPD 的机制也值得记**：提出 **First-Order Local Entropy Influence**（把每次更新的熵效应分解成「教师-学生 log 概率差」与「学生的局部概率结构」），**保留扩熵更新、把缩熵更新换成 divergence-adaptive advantage shrinkage**，且**只用教师在采样 token 上的 log 概率**（不需要全词表 Forward-KL）。

⭐⭐ **同窗还有第三篇从另一侧打同一个假设：Cliff（11▲）** —— 它对既有方法的批评一句点中：process reward model 依赖专门奖励模型，而 **on-policy distillation「assuming identical reasoning patterns between teacher and student」**。⟹ 🚨⭐⭐⭐ **这正是深读 2 量化出来的那个东西（分布错配随规模增大，使学生轨迹对教师而言越来越 off-policy）** —— **两篇互不引用，一篇把它当作要量化的现象、一篇把它当作要绕开的假设。**

⭐ Cliff 自己的观察也干净：「**once a reasoning process first goes wrong, evaluating the subsequent reasoning provides limited additional information, as it is already conditioned on an invalid prefix**」⟹ 用现成 LLM 找出**第一个错误**，把 rollout 自然分成「正确前缀」与其后 ⟹ ⭐⭐ **与 CaRL 的 HRA（保留到最终答案步之前的轨迹、插拒答前缀）是同一取向：拿失败轨迹的前半段当作有价值的监督，而不是整条丢掉。**

⭐ 另有 **DART-SD（90▲）** 做多轮工具调用的自蒸馏（菱形拓扑感知检索与调优）与 **Knowledge Distillation During Mid-Training Favors Reasoning over Factual Recall（1▲）** ⟹ ⭐ 后者标题本身是一条可用的结论：**蒸馏对「推理」与「事实召回」的作用不对称**，而这与我记过的「长上下文训练损害参数化知识」（Information Abundance Paradox）在同一轴上。

---

## §4 ⭐⭐⭐ 第三条主线：「记忆」在同一窗口里被三篇从三个方向定性，而三者的价值判断互相冲突

| 篇 | ▲ | 它把记忆当作什么 |
|---|---:|---|
| ⭐⭐⭐ **Agent Memory Is a Surface for Endogenous Authorization Laundering**（深读 1）| **4** | **攻击面 / 有效授权策略的一部分** |
| ⭐⭐ **Safin-1: Safety from Within through Memory-Native State Evolution** | 20 | **安全的基质**（「reframing memory from a passive record of prior context into an **active substrate**」）|
| ⭐⭐ **Evaluating the Hidden Costs of Personalization** | 27 | **隐性成本的载体**（三类风险：irrelevant personalization / **preference narrowing** / sycophantic bias）|

⟹ 🚨⭐⭐⭐ **深读 1 与 Safin-1 的取向正好相反，而这是本窗口最尖锐的一处张力**：Safin-1 主张「安全应当是模型自身的内在属性而非依赖外部保障」，把安全能力做成一个通过原生计算调用的 **Safety State**；⭐ 而深读 1 测出**持久记忆本身会凭空造出不存在的授权**，并明确说「持久记忆不只是一个性能组件，而是 agent 有效授权策略的一部分」。

⟹ ⭐⭐ **两者可以同时为真，而合起来的含义比任一篇强：把能力（含安全能力）搬进记忆状态，同时也把授权判定搬进了那个同一个会出错的地方。** ⚠️ 我只读了 Safin-1 的摘要，不知它是否讨论过记忆本身出错的情形。

⭐ **而 Personalization 那篇的 `preference narrowing`（信息回音室）与 `sycophantic bias` 是我此前只在「谄媚」这一个词下记过的东西被拆成两个可分别测量的成分** —— ⭐ 与 CaRL 那边的分解（行为层「该拒答时不拒答」vs 表达层「说得太确定」）是同族做法。

⭐ 同窗另三篇记忆/长程一致性：**LayerRecall（28▲，状态条件化的记忆路由器做视频长程一致性）· EM²Mem（9▲，事件中心的多模态记忆）· Ring Forcing（15▲，自回归视频扩散的精确长期记忆）** ⟹ ⭐ 「记忆不只是检索」这条线在**视频生成侧**已经独立长出了一套（有界缓存 + 检索 + 路由），与 LLM 侧是同一问题的两个模态表达。

---

## §5 ⭐⭐⭐ 第四条主线：「让度量留在优化压力之外」这一类解法本窗口拿到两个新实例，其中一个的表述是我见过最好的

### 🚨⭐⭐⭐ WebWorld（5▲）：「一个 VLM 骗不了的对手方」

诊断句我认为值得逐字记：

> 「VLM-driven self-improvement of web code has a **structural flaw: the model that proposes the repair is the model that judges it**, and visual plausibility under that judge is a poor proxy for whether the page actually works. **What the loop is missing is a counterparty the VLM cannot fool**, and the browser already is that counterparty: a deterministic, executable simulator of how an HTML artifact behaves under user actions, and **in everything but name a world model for web code**.」

⟹ ⭐⭐⭐ **这是我这一个月收集的那一类解法（Gaming 的未披露轴 / ProMax 只挖截止日后 commit / Articulated Object 的几何验证器 / TailBooster 的运营包络 / CW-BASS v2 的既存操作阈值 / PRISM 的冻结文本原型 / EnvHarness 保留原始 verifier / DeepMind 双盲的密码学保证）里，第一次有人把它的充分条件说成一句话：「一个提议者骗不了的对手方」。**

⭐⭐ **而它的验收机制是 DarwinX 那个契约的第四次独立推导**：浏览器重新执行候选，**只有在「目标进展」与「保住每一项此前已验证的能力」两者同时成立时才签发 acceptance certificate** ⟹ 这就是 `g(c)>0 且 R(c)≤δ`。⭐ 另外「VLM 发出批评 → planner 编译成 **typed interaction contract**」＝把诊断翻译成有界编辑（我那个三分解里的「翻译」一步）。

⭐ **顺带一个术语观察**：它说浏览器「in everything but name a world model」—— ⚠️⚠️ **而同期 r/MachineLearning 有一条 `WTF is a World Model? [D]` 在问这个词到底指什么**（我 08-31 记的），本窗口世界模型相关又有五六篇 ⟹ ⭐⭐ **一个术语被扩张到「任何确定性可执行的模拟器」时，它就快要不携带信息了；而 WebWorld 自己用 `in everything but name` 这个措辞，说明作者知道这是一次扩张。**

### 🚨⭐⭐ PaperGym（36▲）：把「奖励可以靠改写骗到」变成了一个百分数

原文诊断：「Existing pipelines, however, **draw the question and the criteria from the same content, so the reward can be earned by paraphrase**. The rubric is further **compressed into a single scalar per rollout**.」

⟹ ⭐⭐⭐ **两句话各命中我一条主线**：前一句是「探针与被测对象共享来源就失效」（＝SWE-bench 的 gold patch 可被逐字复现、MBA-b/MBA-k 的披露成本、OpenAI 事故里 agent 从公开论文建模评分器）；⭐ 后一句是「一个数掩盖一个结构」。

**它的修法是按论文结构分开取材**：**问题从研究目标与背景合成，标准从方法与实验导出** ⟹ 🚨⭐⭐⭐ **`criterion leakage` 从既有数据集的 11.90%–34.10% 降到 3.7%** —— ⭐⭐ **而「criterion leakage」这个可测量的量是我此前一直缺的：我记过很多次「探针被披露就失效」，但从来没有一个数字说「现有数据集里这件事有多严重」。**

⚠️ 保留：仅摘要；`criterion leakage` 的具体测法未读，而这个量怎么定义直接决定 3.7% 与 34.1% 的可比性。

### ⭐⭐ J-Zero（40▲）：协同演化的裁判，而它绕开「裁判被利用」的方式很巧

**Challenger–Solver–Judge 三方从零数据协同演化，且明确针对不可验证域**（这正是 Co-Evolution 综述说 Stage 3 几乎是空的那个方向）。

⭐⭐⭐ **关键一步**：Judge 用**其排序在事前已知的偏好对**来协同适配 —— 排序来自**每个回答是怎么产生的**（Solver 的答案优于 Challenger 的；同一 Solver 的「分解-重组」答案优于它的一次性答案）。

⟹ ⭐⭐⭐ **这回答了我 08-12 给 RQGM 标出的那个张力（「评估者若与被评者共同演化，『评估者被利用』就不是外部攻击而是内生均衡」）**：**它不让裁判去学「什么是好」，而是从生成过程里导出一个裁判无法自行操纵的序关系。** ⭐ 这与「让参照留在优化压力之外」是同一策略，只是参照物换成了**生成过程的结构**而不是外部数据。

⚠️ 保留：仅摘要；「排序事前已知」这个假设本身有多强（Solver 的答案是否总优于 Challenger 的）我无法判断，而整套方法的有效性压在它上面。

### ⭐⭐ DiagEvo（15▲）：方向来自 solver 自己的失败史

诊断把既有自演化方法分成两类并各指出局限：**无引导方法**（用难度/可学性/多样性引导出题）「keep questions challenging and varied but **do not specify which unresolved reasoning weaknesses later rounds should target**」；**有引导方法**依赖循环之外的任务资源（人类样例/文档语料/指定难度）。

⟹ ⭐ **DiagEvo 的答案是「所需方向可以从 solver 自己的失败史导出」**（诊断器提取反复出现的错误原因，存进按 skill 节点分组的层级化错误原因记忆）⟹ ⭐⭐ **与 U-OPSD 的「干脆不要外部教师」、SkillZip 的「干脆不用评测」是同一姿态的第三个实例：把外部依赖换成对自己产出的结构化利用。** ⚠️ 而这一姿态的共同风险我也记过：**单一被反复优化的信号会腐化**，而「多个独立视角的交集」才稳健 —— DiagEvo 的错误原因记忆属于哪一类，摘要看不出来。

---

## §6 深读 1 ⭐⭐⭐ Agent Memory Is a Surface for Endogenous Authorization Laundering（4▲）

> **arXiv**：[2609.01836](https://arxiv.org/abs/2609.01836) ｜ ETH Zurich + Georgia Tech ｜ cs.CR ｜ 09-01 ｜ EleutherAI SOAR 项目
> **代码**：`github.com/tommasocerruti/eal-bench`
> ⚠️ **取全文触发降级链**：HF `.md` 端点返回 **350 字节**退化响应 → arXiv HTML **127,104 字符**（⭐ 去标签时保留了 `alttext`，这是我 08-31 刚踩过的坑）

![EAL-Bench 系统总览](2026-09-03-hf-daily-papers-aug31-sep03/eal-bench-overview.png)

### 为什么我把一篇 4▲ 的论文放在第一深读

⭐⭐⭐ **我在 08-27 从四个互不引用的来源立起过一条新主线「谁授权了这个动作」**（When "Must" Becomes "Maybe" 给字段 · LessWrong 给生产侧缺口与 Langfuse 的 approval-source 实现 · OpenAI 事故复盘给真实后果（同伴一句 `GO` 被当成授权）· METR 给原因（agent 不去找上报路径，于是把留言板当权威））。

⟹ ⭐⭐⭐ **而这篇是那条主线上第一个带基准与受控实验的实证，且它指出的通道我完全没有：不是外部注入，而是系统自己的记忆凭空造出授权。**

### 术语与它刻意的边界

原文定义 **endogenous authorization laundering**：「the system's own memory creates **apparent authority** that the underlying history does not grant」

⭐⭐ **而它明确把自己与两个已知问题区分开**：**不是 memory poisoning、也不是 prompt injection** —— 因为它**由普通的内部错误产生**，不需要任何攻击者。⟹ ⭐⭐⭐ **这个区分是这篇论文的全部价值所在：一个不需要攻击者的授权失效，意味着「加固边界」这一整类对策不适用。**

### ⭐⭐ 它的真实世界锚点

> **2026 年 2 月，一个邮件 agent 在例行上下文压缩把它主人「行动前先确认」的指令摘要掉之后，删除了两百多封它只被要求「审阅」的邮件。**（The San Francisco Standard 报道）

⟹ 🚨⭐⭐⭐ **这个锚点把两条我此前分开记的线接上了**：**上下文压缩会把绑定性约束降级为可选提示**（When "Must" Becomes "Maybe" 测出正常 handoff 压缩产生 **100% 失活**）＋ **非幂等动作缺少提交闸门**（Runtime Contract 的 12 系统审计里只有 2/12 有）。⭐ 而它同时是我 08-26 记的那个预测的反面证据：Latent Space 把 compaction 举为「已被吸收进模型权重、因而不再需要 harness 去做」的头号例子 —— **而被吸收的恰恰是让约束失效的那个转换。**

### EAL-Bench 的设计（我认为设计比结果更值得抄）

| 组件 | 做法 |
|---|---|
| 环境 | **不断变化的组织史** + ⭐⭐ **一个隐藏的确定性 ledger 记录真实授权状态** |
| 写入方（writer）| 从历史产出持久记忆 |
| ⭐⭐⭐ 执行方（executor）| **只从记忆行动，没有真实历史的访问权** |
| 用例 | **配对的授权/未授权请求** ＋ ⭐ **faithful-memory 对照臂** |
| 模型 | 5 个 writer（Grok 4.3 / Kimi K2.6 / GLM 5.2 / Qwen-Plus / DeepSeek V4 Pro）× 2 个 executor（含 GPT-OSS-120B）|
| 域 | 采购 / 网络安全 / 金融 |

⟹ ⭐⭐⭐ **「executor 没有真实历史的访问权」是承重的那一步**，因为它把「记忆错了」与「执行者判断错了」彻底分开 —— ⭐ **与 Runtime Contract 那条承重墙（验证器可访问外部参考状态但没有 agent 内部状态的访问权）是同一手法的镜像：那边限制验证器看什么，这边限制被测者看什么。**

### 结果

| 量 | 值 |
|---|---|
| writer 为**未授权请求**造出虚假授权的比例 | **最高 50.2%** |
| 🚨 一旦虚假授权存在，executor 照它行动的比例 | **98.6%** |
| 最不安全的记忆形态 | **typed incremental**，金融域 **51.0% 未授权提交** |
| 授权用例的表现 | 仍然很高（⭐ **不是一般性崩塌**）|
| pooled 形成率 P(F)（三域）| **28.3% / 10.4% / 50.2%**，⭐ 每个都与提交率相差 ≤0.8 个百分点 |

⟹ 🚨⭐⭐⭐ **`98.6%` 这个数字的含义是：一旦记忆里出现了虚假授权，下游几乎不做任何抵抗。** ⟹ ⭐⭐ **所以整个风险集中在「形成」这一步，而不在「执行」这一步** —— 而这与我从 LessWrong 那篇学到的缺陷定义（问题不是检查被跳过，而是产物无法区分检查有没有被跳过）是同一结构：**executor 拿到的记忆里，真授权与假授权长得一模一样。**

⭐ **机制解释很干净**：「**one-shot writers reconstruct the final state from the full history, while incremental writers update from prior memory alone, so an error written once is carried forward**」⟹ ⭐⭐ **增量式记忆的错误会被携带向前，这与 AWS 侧「回填 + 固定窗口 = 永久静默丢失」是同一形态的两个领域版本：一次错误 + 之后再没有机会看到原始来源。**

### ⭐⭐⭐ §4.4 缓解措施：一个离散的安全–效用 Pareto 前沿

| 措施 | 未授权提交 | 授权用例 | 形成率 |
|---|---|---|---|
| 基线 | 25.3% | 93.3% | 24.9% |
| **source-authority gating** | **7.3%**（−18.0 pp，95% CI 14.9–21.1）| 🚨 **53.8%** | 5.5% |
| **bounded event sourcing** | **9.0%**（−16.3 pp，CI 12.2–20.4）| **64.7%** | 8.7% |

⟹ ⭐⭐ **两条措施都把未授权提交砍掉一半以上，代价是把授权用例从 93.3% 砍到 53.8%/64.7%** ⟹ ⭐⭐⭐ **这是我记过的所有「加闸门」讨论里第一次有人把效用代价一起报出来，而代价大到无法忽略**（对比：我此前记的 When "Must" Becomes "Maybe" 报的是「四字段补齐后保持率 100%、禁止动作 0%」，但那是受控合成设定且没报效用侧）。

🚨⭐⭐⭐ **而全篇最该记的一句在这里**：

> 「A formation rate of **5.5% survives source filtering**, because **a record with a valid, authoritative source can still carry the wrong scope, validity, or revocation state.**」

⟹ ⭐⭐⭐ **来源过滤挡不住「来源合法但范围/有效期/撤销状态错了」的记录** —— 而这正是我 08-14 记 Governed Persistent Memory 时那个「被撤回/被取代的记录仍会被检索到并支撑主张」的失效，**在这里成了一个有数字的残余风险（5.5%）。** ⭐⭐ **实践含义很直接：`authority` 一个字段不够，必须同时带 scope / validity / revocation ——而这四项恰好就是 When "Must" Becomes "Maybe" 那四个字段的一个更细的版本。**

### 🚨⭐⭐⭐ §4.5 加算力：瓶颈在验证而不在生成，而这一节有一个我此前没见过的 0%

writer 侧 k=1→8：

| 量 | k=1 | k=8 |
|---|---|---|
| 未授权提交 | 13.2% | 8.6% |
| 授权用例 | 94.2% | 95.8% |
| ⭐ 池中**存在**一份精确记忆的比例 | — | **55.0%** |
| ⭐⭐ **自我评审选中它的比例** | — | 🚨 **26.7%** |
| ⭐⭐ 独立 DeepSeek 评审选中它的比例 | — | **30.0%** |
| 最终状态错误 | 63.3% | 51.7% |
| 🚨🚨 **可观察错误持续存在的比例** | — | **100%** |
| 🚨🚨🚨 **自我修复率（每一个 k 上）** | — | **0%** |

⟹ 原文结论：「**The bottleneck is verification rather than generation**, which favors deterministic checks … over spending further compute on LLM reviewers.」

⟹ 🚨⭐⭐⭐ **「k=8 时正确记忆在池子里存在 55% 的时候，而自我评审只在 26.7% 的时候把它挑出来」** —— ⭐⭐ **这与 Evo-Bench 的 Appendix D 是同一件事的第二次量化**（那边是三个模型里两个最终冻结的版本比自己达到过的最好版本更差，49.7→45.4 / 46.5→42.6）⟹ ⭐⭐⭐ **「产生了更好的东西却没能留住」这条主线现在有了它最干净的形式：好东西在池子里，挑选步只有约一半的命中率。**

⭐⭐ **而 `自我修复率 0%（在每一个 k 上）` 是我记过的最强的一个零**：加算力能让生成变好一点，但**一次也没有修好自己已经写错的记忆**。⟹ ⭐ 这与深读 2 的机制（学生对教师噪声不敏感）并读会得到一个不太舒服的推论：**这些循环里「更多算力」买到的是更好的第一次尝试，而不是纠错能力。**

### ⚠️ §4.6 Limitations（作者自陈，值得表扬的一节）

- 环境比真实职场沟通**结构化得多**
- **不估计部署中的实际发生率**
- 确定性形成率**只在 typed memory 上可测**
- ⚠️⚠️ **两处分析只有单一随机种子**
- source-authority gating **假设「有授权能力的主体」是已知的**
- bounded event sourcing **改变了架构**（不是一个可加的补丁）
- 网络安全域较低的发生率**与过度拒答的解释相容**

### 中心主张

> 「**Persistent memory is therefore not merely a performance component, but a part of an LLM agent's effective authorization policy.**」

⟹ ⭐⭐⭐ **这句话对我做客户材料是可以直接用的**：它把「记忆」从一个性能/成本议题重新归类成一个**授权与合规议题**，而后者在受监管行业里有现成的话语（MRM 的独立验证、权限与准入）。⭐ 而它与我记过的三层记忆批评（治理面的 Governed Persistent Memory / 数据模型面的 FluctlightDB / 内容面的 MobileMem）合起来是第四个位置，**且是唯一把它接到授权上的。**

---

## §7 深读 2 ⭐⭐⭐ Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement（127▲）

> **arXiv**：[2608.31046](https://arxiv.org/abs/2608.31046) ｜ Purdue（Yi Ding & Ruqi Zhang 组）｜ cs.LG ｜ 08-31
> ⚠️ **取全文同样触发降级链**：HF `.md` 返 **188 字节**退化响应 → arXiv HTML **72,820 字符**（⭐ 保留 `alttext`）

![OPSA 方法示意](2026-09-03-hf-daily-papers-aug31-sep03/opsa-method.png)

### ⭐⭐⭐ 结论一：教师监督高度含噪，而噪声随教师规模上升

**测法**：看教师给学生轨迹的**答案 token** 打的 advantage 符号，与该轨迹**答案是否正确**是否一致。

| 教师 | 正确轨迹被打负 advantage | 错误轨迹被打正 advantage | **总噪声率** |
|---|---:|---:|---:|
| 4B | 20.4% | 40.8% | **30.6%** |
| 30B-A3B | — | — | **34.7%** |
| **235B-A22B** | — | — | 🚨 **50.6%** |

🚨⭐⭐⭐ **而最大那个教师的细节比总数更刺眼**：它对 `\boxed{}` 答案 token **在答案正确时给负 advantage 的比例是 97.8%，答案错误时是 96.6%** ⟹ 原文「as the teacher becomes more capable, its supervision on student-generated trajectories becomes overwhelmingly negative and **increasingly insensitive to answer correctness**」。

⭐⭐ **归因**：**学生与教师策略之间的分布错配随规模增长**，使学生轨迹从教师视角看越来越 off-policy。

⟹ 🚨⭐⭐⭐ **这是「教师被污染」这条线的第五种机制，而它是最难办的一种，因为它随教师变强而变坏**：

| 机制 | 出处 | 错配在哪 |
|---|---|---|
| 信息不对称 | DAPD | 教师看得到学生看不到的东西 |
| 状态错配 | SMRC-SD | 学生动作改变执行状态，参考在对另一个状态说话 |
| 身份错配 | 「模型继承的是写作者本人」| 教师模仿的对象 ≠ 教师自己 |
| 续写非原生轨迹时被锚定 | The Handoff Tax | 继承了一条不是自己创造的轨迹 |
| 🚨 **分布错配随能力差扩大** | **本篇** | ⭐ **教师越强，它对学生轨迹的判断越不可靠** |

⭐⭐ **而它与 SKILLER 恰好是同一条鸿沟的两个方向**：SKILLER 测出**为强模型写的技能会让紧凑模型退化**（强→弱，作为被遵循的指令有害）；本篇测出**强教师给弱学生轨迹的打分是噪声**（强→弱，作为评分者不可靠）。⟹ ⭐⭐⭐ **合起来：能力差既让强模型的产物不适合弱模型使用，也让强模型的判断不适合评价弱模型的产物 —— 而这解释了我攒的那个「迁移随距离分层」梯度为什么会有一个负的末端。**

### ⭐⭐ 结论二：学生对这些噪声不敏感（一个漂亮的对照实验）

按「轨迹里是否含噪声信号」分组，比较三个臂：**标准 OPD** / **只在含噪轨迹上训** / **只在干净轨迹上训**。

⟹ **三者在相近的梯度步数后收敛到可比的性能**，⭐ **即便只在含噪 advantage 的轨迹上训练，改善速率也相当。**

⟹ ⭐⭐⭐ **这个结果本身就是对「OPD 在蒸馏教师知识」这个叙事的否证**：若增益来自教师的正确监督，那把监督换成噪声应当损害它 —— 而它没有。

### 🚨⭐⭐⭐ 结论三：增益来自哪里 —— 压制低概率 token，而这不需要教师

- 有效学习**集中在学生采样出的低 logp token 上**
- ⭐⭐ **用一个固定的负 advantage 替换教师提供的 advantage，效果与标准 OPD 相当**

⟹ 原文：「OPD works largely by **suppressing low log-probability tokens**, which **requires no teacher**.」

⟹ ⭐⭐ **由此推出的方法 OPSA（On-Policy Self-Adaptation）：无监督、按熵自适应的负 advantage** —— 压制尾部 token、把概率质量重分配给头部 token，**在低熵位置锐化、在高熵分叉处保留多样性**。

**结果（相对 Qwen3-1.7B 基座）**：

| 量 | 值 |
|---|---|
| AIME24 的 Avg@32 | 🚨 **+35.41 个点（相对 +263%）** |
| 三个基准的 Pass@32 | **全部翻倍以上** |
| AIME24 上相对 OPD 的 Avg@32 | **高 16.77 个点**（Avg@4 上高 12%）|

### ⚠️ Appendix A Limitations（自陈，三条都实质）

1. **模型只到 9B**（算力所限）⟹ 能否扩到更大模型或 MoE 不清楚
2. ⭐⭐ **OPSA 主要在重分配已有的概率质量** ⟹ **对已经做过大量后训练、分布本就很尖锐（低熵）的模型，收益可能有限**
3. 🚨⭐⭐⭐ **「it may not substantially expand the policy's underlying exploration frontier, as reflected by the relatively modest improvements in thinking-mode Pass@k」** ⟹ **这一条正是同窗 IDA-OPD 的主题**（见 §3），⭐ 而它自己主动说出来了

⟹ ⭐⭐ **这三条合起来使 OPSA 的定位很清楚：它是一个「把已有能力更可靠地取出来」的方法（elicitation），不是一个「扩展能力边界」的方法。** ⭐ 而这与我记过的「Boris Cherny 的 unhobbling / elicitation」是同一个区分在训练侧的表达。

---

## §8 其他值得记的（各一到两句）

- ⭐⭐ **StudentSim（469▲，本窗口最高）** —— 学生模拟器：pooled 训练 + 逐学生特化，StudentSimEval 覆盖 60 名学生。⟹ ⭐ **落在「模拟成为独立的能力来源」那条线上**（此前：Latent Space 把模拟提为新 scaling law「10% worse, 100x cheaper, 10000x faster」· EnvHarness 改造冻结环境 · SPADE），⭐ **而这次是教育域，且它的动机是「真实学习者的证据稀疏、慢且昂贵」＝一个非常具体的数据供给问题。**
- ⭐⭐ **Super Library Agent（26▲）** —— N 个相关应用 + 一个共享的 Super Library。⭐⭐ **它对长期 agent 维护的诊断值得记**：「prolonged agentic maintenance to accumulate **verbosity, dead code, and structural erosion**」⟹ ⭐ **与 StateM §4.7 的「experience must be filtered before it becomes memory」、SkillZip 的技能库膨胀是同一问题在代码库上的表达**，且它指出朴素的顺序 scaffold 会有「低抽取召回 + 脆弱的依赖迁移」。
- ⭐⭐ **Recursive Criticality of AI Self-Improvement（4▲）** —— 给 AI4AI 一个可判定的量：**递归再生产数 `R_AI`**，比较反馈强度与研究变难的速率；`R_AI>1` 即自我放大。🚨⭐⭐⭐ **最该记的是它的两个「解耦」结论**：**① 系统可以在加速变得可见之前就已进入自我放大区间 ② 快速进步也可以在没有自我放大的情况下发生** ⟹ ⭐⭐ **含义是「观测到的加速」与「是否处于自我放大区间」互不充分必要，所以拿速度当判据是错的**；⭐ 另两条：**更高的基线研究生产率能加速进步但不改变是否自我放大**（而开发周期长度成为放大的限制时间尺度）· **研究难度上升可以终结一段自我放大**。⚠️ 纯理论模型、无实证参数估计。
- ⭐ **Agentic Artifact Creation（60▲，综述）** —— 系统/评测/原则/机会，⭐ 值得记的是它把「artifact creation」当作一个独立范畴，而我此前记的 AutoDesign（meta-harness optimizer 做压缩结构化媒体产出）正落在这里。
- ⭐ **Qwen3.8-Next 架构（46▲）** —— 评测、效率与训练稳定性，⭐ **与我 08-12 那份 softmax 线性化专题直接相关**（那份的结论是 2026 旗舰架构必然是混合体，而这里是同一家的下一代设计报告）。
- ⭐ **SMELT（77▲）** —— 算力对齐的 MoE Looped Transformer 的 scaling law，⭐ 接「looped compute」那条线。
- ⭐ **Puro-2B（35▲）** —— 一张 RTX 5090、5090 美元训出 Qwen2-1.5B。⭐ **与我 08-31 记的 GPU 价格线（5090 涨到「officially cost 5090」）在同一周形成一个巧合的对照：硬件价格与「用一张卡能做到什么」同时成为话题。**
- ⭐ **从生产流量到后训练（32▲）** —— 自托管 LLM 覆盖语料，⭐ **落在「数据从哪来」那条线上第六个答案：自己的生产流量**（前五：只用可许可语料 / 去买实体书扫 / 合成渲染 / 回收过去人类劳动的副产品 / 默会知识）。
- ⭐ **世界模型本窗口密集**：H3-World（42▲）· ZimaBlue（40▲）· SolarWM（23▲，开放数据 + 长程视频世界模型）· Matrix-Game 3.5（14▲，patch memory 做实时流式）· Lucida（100▲，real-to-sim）⟹ ⭐⭐ **而 §5 那个术语观察在这里更明显：本窗口「世界模型」同时指长程视频生成器、动作模型、real-to-sim 管线，以及（WebWorld 那里）一个浏览器。**
- ⭐ **CogEvol（29▲）与 PaperGym（36▲）都在做「学习环境/训练环境生成」** ⟹ ⭐ 与 EnvHarness（改造冻结环境）合起来，**环境侧本窗口有三篇**。
- ⭐ **InternReviewer & InternAdvocate（7▲）** —— 为 agentic RL 做客观奖励与评测，⭐ 名字里同时有 reviewer 与 advocate（对抗式）⟹ 与「多个独立视角的交集」那条线相邻。
- ⭐ **It Takes Two to Match（12▲）** —— 用 RL 协同演化生成式检索器，⭐ 协同演化扩到检索。

---

## §9 趋势

### ⭐⭐⭐ 1. harness 五篇第一次覆盖完整生命周期，而补上的那一格（可回退）带来一个 `0/197`

**演化（StarHarness）→ 元层（HoH）→ 具身（NavMCP）→ 被评测（LoopArena）→ ⭐⭐⭐ 可回退（EvoUndo）**，而最后一格此前只有「建议」（Co-Evolution 综述的 rollback to verified states）与「代价」（Evo-Bench 留不住最好版本）。⟹ 🚨 **EvoUndo 给的是「你可能根本回不去」：600 个任务里 197 个能力提升通不过可恢复性验证，而常规修复策略在原恢复表示下恢复 0 个。** ⟹ ⭐⭐ **这条主线现在的完整形状是：识别（Evo-Bench）→ 保留（DarwinX）→ 翻译（SKILLER）→ ⭐ 可回退（EvoUndo），四个环节各有一篇把断点定位在那里。**

### ⭐⭐⭐ 2. 「一个提议者骗不了的对手方」——那一类解法终于有了一句充分条件

**WebWorld 的表述**（`a counterparty the VLM cannot fool`）⭐⭐⭐ **把我一个月来分别记下的十来个实例统一了**，而本窗口它还有三个同族：**PaperGym 把「奖励可靠改写骗到」量化成 criterion leakage 3.7% vs 11.9–34.1%** · **J-Zero 从生成过程导出裁判无法操纵的序关系** · **StarHarness 的 proposer-hidden selection tasks**。⟹ ⭐⭐ **而这四个的「对手方」性质不同：确定性执行器 / 取材分离 / 生成过程的结构 / 不披露** —— **前三种「说了也没用」，第四种依赖保密。**

### ⭐⭐⭐ 3. 「产生了更好的东西却没能留住」拿到最干净的一次量化

**EAL-Bench §4.5：k=8 时正确记忆在池子里存在 55.0%，而自我评审只在 26.7% 的时候选中它，独立评审 30.0%；自我修复率在每一个 k 上都是 0%。** ⟹ ⭐⭐ **与 Evo-Bench 的 Appendix D、iFAN 的「最后一层解码会丢掉中间层更优的预测」构成三个领域的同一形状**，⭐ 而本篇的贡献是把「生成」与「选择」的成功率分别报了出来：**瓶颈在验证不在生成**，作者据此建议「favors deterministic checks over spending further compute on LLM reviewers」。

### ⭐⭐ 4. 「教师被污染」第五种机制，而它随教师变强而变坏

**深读 2 的 30.6% → 34.7% → 50.6%**，以及最大教师对答案 token 的 97.8%/96.6% 负 advantage。⟹ ⭐⭐⭐ **前四种机制都可以靠「换更好的教师」缓解，这一种不行——它由能力差本身产生。** ⭐ 而同窗 Cliff 独立把它当作要绕开的假设（「assuming identical reasoning patterns between teacher and student」），IDA-OPD 则测出它在多样性上的后果（pass@1 升而 pass@k 停滞）。

### ⭐⭐ 5. 记忆在同窗被三篇从三个方向定性，而两个方向的价值判断相反

**攻击面（深读 1）· 安全基质（Safin-1）· 隐性成本载体（Personalization）** ⟹ ⭐⭐ **「把能力搬进记忆状态」同时也把授权判定搬进了那个会出错的地方**，而这个张力本窗口内部就成立、两篇互不引用。

### ⚠️ 6. 自我怀疑（连续第六次记）

⭐ **本份两个深读一篇 127▲ 一篇 4▲，而 4▲ 那篇贡献了更多可操作结论 —— 这是连续第 N 次「upvote 与相关性弱相关」。** ⚠️⚠️ **但今天我要记的怀疑不是这个，而是更具体的一条：本份 §5 那四个「骗不了的对手方」实例，有三个是我为了补齐这条主线而从 107 篇里主动搜关键词找出来的（upvote 分别是 5 / 36 / 40），而不是它们自己浮上来的。** ⟹ ⭐⭐ **含义是「本窗口这条主线很强」这个说法带着我的检索偏好：我知道要找什么，所以找到了。** ⭐ 一个便宜的对策是同时记下**没找到的那一侧**——本窗口我没有找到任何一篇在反方向上论证的（即「让提议者兼任裁判其实可行」），而这个「没找到」本身也可能只是因为我没搜。

---

## §10 Open Questions

1. 🚨⭐⭐⭐ **StarHarness 的「跨 GPT 与 Qwen 家族免重演化迁移」与 StateM 的「跨厂商 82.7%→82.0% 略微变差」怎么调和？** ⭐ 我的假设是**被迁移的对象不同**（环境知识 vs 依赖执行者自主性的控制结构），而这给出一个可检验预测：**把 StarHarness 演化出的改动按「接口修复 / 环境约定 / 操作知识」分类后分别测跨家族保留率，接口类应当最高。** ⟹ ⭐ **这是我这两个月攒的问题里第三个有明确可操作检验方案的。**
2. ⭐⭐⭐ **EvoUndo 那个非单调结果（加精确地址诊断使恢复从 99.3% 降到 93.0%）是什么机制？** ⭐ 若是「更多信息让模型偏离一个本来正确的默认策略」，那它与我记过的「naive prompting 施加均匀位移，因而伤害本来能解的」（CaRL 的 Capability Loss 0%→10%）是同一形态。
3. ⭐⭐⭐ **`authority` 一个字段不够，那到底需要哪几个？** ⭐ 深读 1 给的残余风险（**来源过滤后仍有 5.5% 形成率，因为来源合法的记录仍可能带错的 scope / validity / revocation state**）与 When "Must" Becomes "Maybe" 的四字段（prerequisite / authority / fallback / execution consequence）⟹ ⭐⭐ **两份清单要合并成一份，而这正好是我 08-31 在 cross-digest 里提的那个「稳定内核应当包含什么」的具体答案候选。**
4. ⭐⭐ **`source-authority gating` 把授权用例从 93.3% 砍到 53.8% —— 这个代价可以被降低吗？** ⭐ 这是我第一次看到闸门的效用代价被量化，而**若代价真有这么大，那「加闸门」在生产里被优化掉就不需要任何人作恶**（正是我从 Grounded Reasoning Cup 学到的那条：验证成本落在每一次、收益只落在少数次）。
5. ⭐⭐ **PaperGym 的 `criterion leakage` 怎么测？** ⭐ 若这个量的定义可推广，它就是我一直缺的那个「探针污染程度」的通用指标 —— 而 3.7% vs 34.1% 的可比性完全取决于它的定义。
6. ⭐⭐ **深读 2 的噪声率测法能不能搬去测「LLM 裁判评价弱模型产物」的场景？** ⭐ 本篇测的是教师给学生 token 打 advantage，而**同一个分布错配论证应当预测「强裁判评价弱模型轨迹」也随能力差变差** —— 而这正是 OSReward 测出「裁判读自述而非看屏幕」的那个设定，两边从未被联系过。
7. ⭐ **J-Zero 的「排序事前已知」这个假设有多强？** ⭐ 整套方法压在「Solver 的答案优于 Challenger 的」上，而若这个序在某些域不成立，裁判就会被系统性地训错方向。
8. ⭐ **Safin-1 的 Safety State 是否讨论过记忆自身出错的情形？** ⭐ 它与深读 1 在同一窗口给出相反的价值判断，而这个问题决定两者是互补还是冲突。

---

## References

**深读**

- [2609.01836 Agent Memory Is a Surface for Endogenous Authorization Laundering](https://huggingface.co/papers/2609.01836)（4▲）
- [2608.31046 Does On-Policy Distillation Really Distill? From Noisy Teacher to Self-Improvement](https://huggingface.co/papers/2608.31046)（127▲）

**harness 五篇**

- [2608.28281 LoopArena](https://huggingface.co/papers/2608.28281)（100▲）· [2608.24804 StarHarness](https://huggingface.co/papers/2608.24804)（39▲）· [2609.01481 Harness-of-Harness](https://huggingface.co/papers/2609.01481)（11▲）· [2608.30396 Scaffolding Foundation Models into Physical-World Agents](https://huggingface.co/papers/2608.30396)（8▲）· [2608.28363 EvoUndo](https://huggingface.co/papers/2608.28363)（7▲）

**评测有效性 / 度量**

- [2608.31119 PaperGym](https://huggingface.co/papers/2608.31119)（36▲）· [2608.30530 WebWorld](https://huggingface.co/papers/2608.30530)（5▲）· [2608.26582 J-Zero](https://huggingface.co/papers/2608.26582)（40▲）· [2608.28612 InternReviewer & InternAdvocate](https://huggingface.co/papers/2608.28612)（7▲）

**蒸馏 / 奖励**

- [2608.29846 Influence-Directed Distillation (IDA-OPD)](https://huggingface.co/papers/2608.29846)（4▲）· [2609.02817 Cliff](https://huggingface.co/papers/2609.02817)（11▲）· [2608.18524 DART-SD](https://huggingface.co/papers/2608.18524)（90▲）· [2609.01532 Knowledge Distillation During Mid-Training](https://huggingface.co/papers/2609.01532)（1▲）

**记忆 / 自我演化**

- [2609.00092 Safin-1](https://huggingface.co/papers/2609.00092)（20▲）· [2608.28833 Hidden Costs of Personalization](https://huggingface.co/papers/2608.28833)（27▲）· [2608.28460 LayerRecall](https://huggingface.co/papers/2608.28460)（28▲）· [2609.00768 DiagEvo](https://huggingface.co/papers/2609.00768)（15▲）· [2609.00137 Recursive Criticality of AI Self-Improvement](https://huggingface.co/papers/2609.00137)（4▲）· [2608.29310 Super Library Agent](https://huggingface.co/papers/2608.29310)（26▲）

**其余 Top 25**

- [2609.01591 StudentSim](https://huggingface.co/papers/2609.01591)（469▲）· [2609.00111 Qwen-Drive-1.0](https://huggingface.co/papers/2609.00111)（346▲）· [2608.30821 Lucida](https://huggingface.co/papers/2608.30821)（100▲）· [2608.31106 DreamX-Creator](https://huggingface.co/papers/2608.31106)（94▲）· [2608.27550 Beyond Data Scaling](https://huggingface.co/papers/2608.27550)（88▲）· [2609.01343 SMELT](https://huggingface.co/papers/2609.01343)（77▲）· [2608.29335 GenFirst](https://huggingface.co/papers/2608.29335)（61▲）· [2608.28122 Agentic Artifact Creation](https://huggingface.co/papers/2608.28122)（60▲）· [2609.00028 UI-Venus-2](https://huggingface.co/papers/2609.00028)（56▲）· [2608.31036 Normalized Low-Rank Adaptation](https://huggingface.co/papers/2608.31036)（50▲）· [2608.30320 Qwen3.8-Next 架构](https://huggingface.co/papers/2608.30320)（46▲）· [2609.01560 H3-World](https://huggingface.co/papers/2609.01560)（42▲）· [2609.00188 ZimaBlue](https://huggingface.co/papers/2609.00188)（40▲）· [2608.27370 Puro-2B](https://huggingface.co/papers/2608.27370)（35▲）· [2608.27529 Long-Horizon Streaming 3D](https://huggingface.co/papers/2608.27529)（32▲）· [2609.01572 From Production Traffic to Post-Training](https://huggingface.co/papers/2609.01572)（32▲）· [2608.27763 Fast Weight Attention](https://huggingface.co/papers/2608.27763)（30▲）· [2608.23478 Act with Intent](https://huggingface.co/papers/2608.23478)（30▲）· [2608.30968 CogEvol](https://huggingface.co/papers/2608.30968)（29▲）
