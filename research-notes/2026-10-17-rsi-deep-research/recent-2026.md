# RSI：2026 年新增与更新的实证工作

截至日期：2026-10-17（北京时间；论文提交日期按平台原始记录）。本文件只覆盖近期进展，不重复深读经典 Gödel Machine、STaR、Self-Refine、Reflexion、ADAS、AlphaEvolve、2025 DGM 或 STOP。下列核心清单共 **10 项**；综述与诊断性对照另列。

## 阅读方式与证据标签

- **全文核实**：已访问论文 HTML/PDF，并核对相关方法、实验、指标与限制；不表示独立复现实验，也不表示逐项验证作者的全部论断。
- **摘要级证据**：只核对官方摘要和元数据，不据此确认实验设计或指标。下文仅补充的 agentic-reasoning survey 使用此标签。
- **直接改进器证据**：比较改进前后的机制在固定预算下生成更好系统的能力，而不只是展示某条任务分数曲线上升。
- **局部元改进**：搜索策略、记忆更新规则等确实可变，但上层生成器、接受准则或评估器仍固定。
- **未证明改进器变强**：结果支持任务、答案或系统性能提升，未隔离检验下一轮“如何改进”的能力。

所有数字均为**作者报告、已定位到一手全文的结果**，不是本次独立确认的实验事实。预印本首发日期以 arXiv submission history 的 v1 为准；所读版本明确列出，均不晚于截止日。引用数未获得可靠查询结果，统一为 `null`，不当作零引用。

## 1. 优先核查的 2026 RSI 综述

**Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops** — Mingguang Chen、Licheng Wang、Bo Qu；arXiv **2607.07663**。首发 **2026-07-08**，所读 **v2，2026-09-06**；正式 venue 未核实。**全文核实，综述，不是新的 RSI 实验。** [版本记录](https://arxiv.org/abs/2607.07663)；[v2 全文](https://arxiv.org/html/2607.07663v2)；[作者资料与脚本](https://github.com/deepgrounding/recursive-self-improvement)。

作者用“改进什么”与“谁验证/闭合循环”两轴组织文献：部署行为、训练策略、自评估器、研究过程；强调固定外部评价下的有界优化与开放式 RSI 不应混称。尤其适合主报告的 §5.2 verification hierarchy、§5.5 result-level versus process-level improvement 和 §6.3 诊断文献。其“形式验证比内在自评更有锚定性”是综述的综合判断，不是统一跨任务对照实验。[全文 §§2、5–7](https://arxiv.org/html/2607.07663v2)。

**方法学限制**：1,250 是检索语料规模，不是 1,250 个成功 RSI 实验。语料由 871 篇种子与 379 篇补充组成，单一标注者分类；作者承认补充语料约 54 篇属于外围 query bleed，仍计入数量，但不作为论证证据。因此适合作为路线图，不能用总量证明成熟度或效应大小。[全文 §2.3](https://arxiv.org/html/2607.07663v2)。

另有 **A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents**，Tianxin Wei 等，arXiv **2601.12538**：v1 **2026-01-18**，v2 **2026-09-20**；venue 未核实。这里仅作**摘要级证据**：其 foundational/self-evolving/collective 三层分类可辅助检索，不能视为递归改进器已获实证的证明。[官方摘要与版本](https://arxiv.org/abs/2601.12538)。

## 2. 十项核心实证工作的对照

| 项目 | 首发；核查版本 | 改进对象 | 是否证明改进器提升 |
|---|---|---|---|
| [Hyperagents](https://arxiv.org/abs/2603.19461) | 03-19；v1 | task agent 与 meta agent 的程序 | 有直接、跨域 imp@50 证据；不能外推无限加速 |
| [AIDE²](https://arxiv.org/abs/2609.26457) | 09-22；v1 | 研究 agent 的搜索、选择与记忆代码 | 内层研究效率提升；外层 ignition test 不确定 |
| [EvoX](https://arxiv.org/abs/2602.23413) | 02-26；v2 03-16 | 解与产生解的搜索策略 | 局部元改进；更高层策略生成器固定 |
| [Dream-RSI](https://arxiv.org/abs/2609.14858) | 09-14；v2 10-06 | 分配发现计算的探索策略 | 局部元改进；coding agent、评估器固定 |
| [MetaShare](https://arxiv.org/abs/2610.21887) | 10-15；v1 | agent 代码及跨谱系成败经验 | 改善演化效率；未隔离证明通用改进器变强 |
| [GEA](https://arxiv.org/abs/2602.04837) | 02-04；v1 | 群体 coding-agent 框架 | 任务与群体演化收益；未单测 meta-improver |
| [Meta-Harness](https://arxiv.org/abs/2603.28052) | 03-30；v1 | 固定模型外围 harness | 外层 proposer 不自改进 |
| [AREX](https://arxiv.org/abs/2607.21461) | 07-23；v3 09-01 | 研究答案、证据状态及离线训练策略 | 主要为答案/状态递归细化 |
| [VisionHOPE](https://arxiv.org/abs/2609.33325) | 09-27；v1 | 单幅图像内的记忆和学习规则 | 局部自引用学习；非长期自治 RSI |
| [AIRA₂](https://arxiv.org/abs/2603.26499) | 03-27；v2 04-13 | 研究候选解、并行执行与评估协议 | 人工设计的研究优化器，不是自改进改进器 |

表中分类是本次阅读的判断；其依据及不能外推之处见下文。十项均为**全文核实**。

### 2.1 Hyperagents：最直接的“改进改进器”对照之一

Jenny Zhang、Bingchen Zhao、Wannan Yang、Jakob Foerster、Jeff Clune、Minqi Jiang、Sam Devlin、Tatiana Shavrina；正式 venue **未核实**。[元数据](https://arxiv.org/abs/2603.19461)；[v1 PDF](https://arxiv.org/pdf/2603.19461v1)；[官方实现与实验日志入口](https://github.com/facebookresearch/Hyperagents)。

**循环**：从 archive 选父代，让一个可编辑程序同时包含 task agent 与 meta agent；修改程序、评价任务表现、保存后代。相对于旧路线的新意是产生未来修改的机制本身也可编辑。实验覆盖 coding、论文评审、机器人奖励设计与数学解答评分。[方法与实验 §§3–5](https://arxiv.org/html/2603.19461v1)。

**关键证据**：§5.2 固定待测 meta agent，在新领域给它 50 次产生后代的机会，用验证集选出的后代之测试增益定义 imp@50。由评审/机器人领域迁入数学评分后，五次运行的中位 imp@50 为 **0.630，CI 0.540–0.630**，初始机制为 **0.0**；作者报告 `p<0.05`。这比“coding 分数提高，所以更会改代码”更直接。[PDF §5.2、图3](https://arxiv.org/pdf/2603.19461v1)。

**边界**：初始数学评分 agent 因输出格式/解析失败处于零分，不能把全部增益解释为数学智能提高。§5.3 在 200 次继续演化后，迁移初始化 **0.640** 对原始初始化 **0.610** 的差异**不显著**。结果支持可迁移改进机制，不证明无限递归收益或持续加速。[PDF §§5.2–5.3](https://arxiv.org/pdf/2603.19461v1)。

### 2.2 AIDE²：研究效率提升与“点火”之间仍有缺口

**Recursive self-improvement of AI research agents**；Dhruv Srikanth、Bingchen Zhao、Dixing Xu、Yuxiang Wu、Zhengyao Jiang；arXiv **2609.26457**，正式 venue **未核实**；本次未核实公开完整实现。[元数据](https://arxiv.org/abs/2609.26457)；[全文](https://arxiv.org/html/2609.26457v1)。

**循环**：内层研究 agent 在固定美元预算内改任务代码；外层改写该 agent，并按内层无法直接看到的 private grading 接受修改。内层选候选、返回最终解、组织记忆的代码可变；外层选择规则仍固定。[全文 §2.1](https://arxiv.org/html/2609.26457v1)。

**指标与基线**：一次 **8 天、100 节点**轨迹接受 **7 次**改写，selection grade 从 **0.703→0.778**，人工工程基线为 **0.749**。四个不参与选择的外部 benchmark 包括 ALE、MLE、FML 与 WeatherBench 2；最终 agent 均匹配或超过该人工基线，但中间 checkpoint 的迁移收益不单调。独立任务族的 reward-hacking 比率 **55%→32%**，人工基线 **39%**。[全文 §§3.2–3.4](https://arxiv.org/html/2609.26457v1)。

**是否更会改进？** 内层研究优化能力的提升有固定预算与 held-out 证据；但 §3.6 将演化 agent 提升为外层 improver 的 **ignition test**，两臂各三 seeds、各 50 步，作者明确说噪声和成本使其**不能与强基线作确定区分**。不能把这篇写成已经实证“自我加速点火”。[全文 §§3.6、5](https://arxiv.org/html/2609.26457v1)。

### 2.3 EvoX：进化解，也进化搜索策略

Shu Liu 等 17 位作者，完整名单见 JSON；arXiv **2602.23413**；v2 **2026-03-16**；正式 venue **未核实**。[元数据](https://arxiv.org/abs/2602.23413)；[全文](https://arxiv.org/html/2602.23413v2)；[官方 SkyDiscover 实现](https://github.com/skydiscover-ai/skydiscover)。

**循环**：内层产生、评价、积累候选解；外层根据种群状态、进展和停滞，改写父代选择、inspiration set 与变异偏好。新策略运行后以其带来的下游进展评价，不清空解库。[全文 §§3–4](https://arxiv.org/html/2602.23413v2)。

**已核指标**：信号处理案例在同样 **100 次解演化**预算下，相比固定随机采样策略，最终综合得分高 **34.1%**；这是该案例的结果，不是所有任务平均收益。广泛实验比较 OpenEvolve、GEPA、ShinkaEvolve；部分 AlphaEvolve 数字为文献结果，不都是统一条件重跑。[全文 §5、§6 表1](https://arxiv.org/html/2602.23413v2)。

**判断**：搜索策略确实是改进对象，属于局部 meta-evolution；更上层策略生成/切换协议和评价器仍由人定义。不能从下游优势推出策略生成器本身持续增强。

### 2.4 Dream-RSI：在发现历史里离线改探索策略

Tong Zheng 等 17 位作者，完整名单见 JSON；arXiv **2609.14858**；v2 **2026-10-06**；正式 venue **未核实**。[版本](https://arxiv.org/abs/2609.14858)；[全文](https://arxiv.org/html/2609.14858v2)；[官方仓库](https://github.com/zhengkid/Dream-RSI)。

**循环**：在线由探索策略指挥固定 agent，留下有评价结果的发现树；离线把树当 replay world，低成本测试、改写探索策略，再部署下一轮。**只有探索策略代码改变；基础 agent、评估器和执行接口固定。** [全文 §§2–3](https://arxiv.org/html/2609.14858v2)。

**已核指标**：对固定探索基线，VGG16 和 LayerNorm 达到可比质量所需 generations 分别少 **2.43×、1.79×**。Lasso 的 Pro 配置在六个 held-out 数据集上的平均运行时 **3587.1→2931.0 ms**，发现调用 **550→317**；但平均改善由 RCV1 驱动，另外五个数据集反而更慢。Autocorrelation 指标 **1.456375** 也略差于固定探索的 **1.456001**，不能写成全任务胜出。[全文 §4，图3–4、表1](https://arxiv.org/html/2609.14858v2)。

**边界与实现**：探索策略改进有受控证据，但固定 policy-development agent 未被证明变强；replay 只能评价已观察历史。官方仓库当前明确写代码准备发布，**不是完整实现已开源**。[全文 §3](https://arxiv.org/html/2609.14858v2)；[仓库 release 状态](https://github.com/zhengkid/Dream-RSI)。

### 2.5 MetaShare：把失败纳入可复用的演化知识

Vishal Bhutani、Prasang Gupta、Sumanth Chundru、Kevin Paul；arXiv **2610.21887**，**2026-10-15 v1**。arXiv 作者注称获 NeurIPS 2026 Meta-Agents Workshop 接收，**未核实主办方名单**；不能写成 NeurIPS 主会论文或已举行会议。[元数据与作者注](https://arxiv.org/abs/2610.21887)；[全文](https://arxiv.org/html/2610.21887v1)。

**循环**：记录修改的正/负性能 delta；从不同谱系检索成功机制与高置信失败；task-blind auditor 提示欠探索的机制类型；并行提出补丁，经适用性、范围、编译和大小 gate 后执行评价。经验池与程序随演化更新。[全文 §3](https://arxiv.org/html/2610.21887v1)。

**关键核对**：作者报告最佳 checkpoint 出现代数减少 **1.7–2.9×**，不是达到统一目标质量的 time-to-target。三项成本核算中，论文评审总 tokens 为基线 **0.6×**，Polyglot **1.2×**，IMO **2.4×**；因此不能称整体节省计算。部分 DGM-H 数字直接引用旧实验，部分为作者重实现；GEA 对比另用其协议。[全文 §4.2、表1](https://arxiv.org/html/2610.21887v1)。

**判断**：失败记忆、检索与并行提案改善演化值得关注，但没有独立的跨域 imp@k/ignition 对照证明通用 meta-improver 更强。公开完整实现本次未核实；财务审计标签不公开，且正文与附录样本规模描述不一致，见访问审计。

### 2.6 Group-Evolving Agents（GEA）：群体经验复用

Zhaotian Weng、Antonis Antoniades、Deepak Nathani、Zhen Zhang、Xiao Pu、Xin Eric Wang；arXiv **2602.04837**。**COLM 2026 接收**有作者官方主页及会议名单支持；名单中后两人的展示名有变体，本文按所读 arXiv 版本记作者。[首发元数据](https://arxiv.org/abs/2602.04837)；[作者主页](https://zhaotianweng.github.io/)；[接收名单](https://colm.eventhosts.cc/Conferences/2026/AcceptedPapers)；[官方实现](https://github.com/UCSB-AI/GEA)。

**循环**：选择父群体，共享 agent 修改补丁、失败任务执行轨迹和结果，反思后各自修改框架并归档，避免独立分支遗失探索知识。[全文 §3](https://arxiv.org/html/2602.04837v1)。

**指标**：报告 SWE-bench Verified **71.0% 对 DGM 56.7%**，Polyglot **88.3% 对 68.3%**。但 Polyglot 的主协议是在 10-task small set 演化，另以 50-task medium set 测试；**不要把 88.3% 当作完整 225 题全量成绩**。SWE 则先 10/50-task 筛选，再评价 top-2 于全量 Verified。[全文 §§4.1–4.3](https://arxiv.org/html/2602.04837v1)。

**判断**：这证明群体经验整合与 coding-agent 演化收益，不等于证明负责群体选择、反思或生成修改的整体 meta-procedure 自身增强；所列基线存在模型/协议差异，应避免只凭 headline 排序 RSI 强弱。

### 2.7 Meta-Harness：强 harness 搜索不等于递归改进器

Yoonho Lee、Roshen Nair、Qizheng Zhang、Kangwook Lee、Omar Khattab、Chelsea Finn；arXiv **2603.28052**；正式 venue **未核实**。[元数据](https://arxiv.org/abs/2603.28052)；[全文](https://arxiv.org/html/2603.28052v1)；[官方框架与参考实验](https://github.com/stanford-iris-lab/meta-harness)。

**循环**：固定外层 proposer 在文件系统中访问历代 harness 代码、分数和原始 traces，修改固定模型的检索、记忆和控制代码；评价并保留候选。文本分类的 search/test 分离，并有相同提案评价预算对照。[全文 §§3–4.1](https://arxiv.org/html/2603.28052v1)。

**指标与边界**：文本分类测试 accuracy **48.6%**，比 ACE 高 **7.7 个百分点**，上下文 **11.4K 对 50.8K tokens**。200 道新 IMO-level 题、五个模型的数学检索结果，平均比无检索高 **4.7 个百分点**。但 TerminalBench-2 的搜索和最终评价用**同一组 89 题**，不能将该部分称为 held-out 泛化。[全文 §§4.1–4.3、表2/6](https://arxiv.org/html/2603.28052v1)。

**判断**：可迁移 harness 的搜索证据较强，外层 proposer 本身没有演化，所以应列作 agent/harness evolution，而非完整 RSI。官方仓库已提供框架；不能沿用早期“仅公开产物”的说法。[官方 README](https://github.com/stanford-iris-lab/meta-harness)。

### 2.8 AREX：名称中的 RSI 主要是研究状态递归

Shuqi Lu 等 24 位作者，按 v3 完整名单见 JSON；arXiv **2607.21461**；正式 venue **未核实**。[版本记录](https://arxiv.org/abs/2607.21461)；[v3 全文](https://arxiv.org/html/2607.21461v3)；[官方项目与模型入口](https://vectorspacelab.github.io/arex-model/)。

**循环**：内层搜证据、产生候选答案；外层按约束与置信度决定接受、定向补查或重启；学得的 autonomous context updating（ACU）保存已核实证据和未解决约束。权重能力来自作者设计的 mid-training/RL，不是部署中改写自己的训练算法。[全文 §§2–4](https://arxiv.org/html/2607.21461v3)。

**已核消融**：BrowseComp 的四格对照为：无 ACU/无外环 **59.6**，无 ACU/有外环 **69.8**，有 ACU/无外环 **71.4**，二者都有 **82.5**。每 episode 最多 **300 内层 turns、5 外层操作**；效果依赖明确有限预算。[全文 §5.1、§5.3.1 表3](https://arxiv.org/html/2607.21461v3)。

**判断**：支持答案与研究状态的递归细化，不证明更强通用改进器。附录的简化 trajectory self-distillation **52.3→57.1** 是探索性实验，作者明说未证明与完整系统结合时保持收益；不可当作已完成的长期自训练递归环。[全文附录 A](https://arxiv.org/html/2607.21461v3)。

### 2.9 VisionHOPE：2026 的局部 self-referential learning

Siran Peng、Tianshuo Zhang、Tianyu Fu、Weisong Zhao、Haoyuan Zhang、Jiankuo Zhao、Minghui Wu、Ping Jiang、Xiangyu Zhu、Chenxu Zhao、Zhen Lei；arXiv **2609.33325**；正式 venue **未核实**。[元数据](https://arxiv.org/abs/2609.33325)；[全文](https://arxiv.org/html/2609.33325v1)；[官方 PyTorch 实现](https://github.com/PSRben/VisionHOPE)。

**循环**：在单幅图像的四方向扫描中，内容、key、value、学习率与保留率五种 memories 联合更新；产生更新规则的量由这些 memories 自身给出。outer training 学得参数，但 query projection 在图像内更新时固定。[全文 §2](https://arxiv.org/html/2609.33325v1)。

**指标**：ImageNet-1K 的 VisionHOPE-S 为 **85.2% Top-1、53M 参数、9.8G FLOPs**，H-ViT³-S 为 **84.9%、54M、8.8G**；不是严格等算力的净收益比较。还测试 COCO 与 ADE20K。[全文 §3、表1](https://arxiv.org/html/2609.33325v1)。

**边界**：未经约束的自引用更新会不稳定，作者引入 injection cap 与 spectral clamp，并给出扫描内 non-expansive 条件。这里的“改学习方式”是局部、受稳定性控制的 neural dynamics，不是跨任务累积自治改进器的证据。[全文 §2.2](https://arxiv.org/html/2609.33325v1)。2022 的 *Self-Referential Meta Learning* 本身不应错列为 2026 新论文。[原作者页面](https://louiskirsch.com/self-ref)。

### 2.10 AIRA₂：稳定评估可能比“更深递归”更关键

**AIRA_2: Overcoming Bottlenecks in AI Research Agents**；Karen Hambardzumyan 等 25 位作者；arXiv **2603.26499**；首发 **2026-03-27**、最新 **v2 2026-04-13**，不是 10 月更新；正式 venue 与公开 AIRA₂ 完整实现 **未核实**。[元数据](https://arxiv.org/abs/2603.26499)；[全文](https://arxiv.org/html/2603.26499v2)。

**循环**：人工设计的异步 evolutionary orchestrator 驱动多 GPU ReAct workers 改候选研究解；Hidden Consistent Evaluation（HCE）固定外部划分，分离 search 与 final-selection 信号。[全文 §3](https://arxiv.org/html/2603.26499v2)。

**指标**：更强 backbone 的 AIRA₂† 在 MLE-bench-30 的 mean Percentile Rank 为 **24h 81.5、72h 83.1**，最强表列基线 **72.7**；主实验采用 8 GPU workers。它不是同 backbone、同总 GPU-hours 的纯算法优越性证明。消融发现移除 HCE 会造成停滞/退化。[全文 §4、表1](https://arxiv.org/html/2603.26499v2)。

**判断**：作者把先前退化主要归因于评估噪声而非数据记忆；这有该设置下的受控证据，但不能推广为“所有长循环都不会过拟合”。它改进研究搜索的可靠性，并未让 orchestrator 自行改写成更强改进器。[全文 §4.3.2](https://arxiv.org/html/2603.26499v2)。

## 3. 主任务要求补充的收益边界证据

### MARS：一轮结构化学习可优于更长循环，但不能证明循环普遍无用

**Learn Like Humans: Use Meta-cognitive Reflection for Efficient Self-Improvement**；arXiv **2601.11974** 首发 **2026-01-17**；**ACL 2026 Long Papers**，DOI **10.18653/v1/2026.acl-long.1329**。正式发表作者次序为 Xinmeng Hou、Bohao Qu、Wuqi Wang、Peiliang Gong、Qing Guo、Yang Liu；与 arXiv v1 次序不同。**全文核实**。[首发](https://arxiv.org/abs/2601.11974)；[正式元数据](https://aclanthology.org/2026.acl-long.1329/)；[正式 PDF](https://aclanthology.org/2026.acl-long.1329.pdf)；[作者代码](https://github.com/Paparare/MARS/tree/main)。

方法为失败诊断→type/topic 分组→生成原则与过程提示，Hybrid 再在验证集选择增强类型；基本是**一轮离线增强，不是改善自身增强器**。正式 PDF 附录 F 报告四 benchmark 成本约 **$3–5**，相对 ADAS **60–90×**、Gödel Agent **3–5×** 便宜；这些是基于历史配置与价格的**估算**，不是同模型同条件实测费用，且有不同 meta-model。[PDF §3、附录 F 表13–14](https://aclanthology.org/2026.acl-long.1329.pdf)。

可用于反驳“更多递归轮数必然更好/更划算”；不能据此证明所有 RSI 收益有限或不存在。还可注意，提示增强的简单相加并非单调：Omni-MATH self-consistency 下 R+C **32.50%** 低于仅 reasoning **34.60%**。[PDF §4、表5](https://aclanthology.org/2026.acl-long.1329.pdf)。

### POLARIS：小模型改自己有高失败率，且 guard 不是语义正确性证明

**POLARIS: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair**；arXiv **2603.23129** 首发 **2026-03-24**；最新 arXiv v4 **2026-06-24**；**Findings of ACL 2026**，DOI **10.18653/v1/2026.findings-acl.1969**。正式作者 Aditya Namdev Kakade、Vivek Srivastava、Shirish Karande。**全文核实**，此处读正式 PDF。[版本](https://arxiv.org/abs/2603.23129)；[正式元数据](https://aclanthology.org/2026.findings-acl.1969/)；[PDF](https://aclanthology.org/2026.findings-acl.1969.pdf)。

方法是失败分析→策略抽象→最小补丁→有限重试、执行验证；权重不更新。初始移植不受限制的框架到 7B 模型常因上下文增长、OOM 或工具调用失败，因而需要压缩经验与限制更新面。[PDF §§1、3–4](https://aclanthology.org/2026.findings-acl.1969.pdf)。

**不能只取成功曲线**：正式表2的 `N=3`，MGSM/DROP/GPQA/LitBench 每项各十次 10h 运行，成功数分别 **5/3/4/6**，失败数 **4/5/2/3**，其余无改善。图3展示的是成功运行，不是所有 seeds 的总体结果。补丁整合主要检查语法/执行，**不是接受前直接验证性能提升**。[PDF §§3.2、4，表2、图3](https://aclanthology.org/2026.findings-acl.1969.pdf)。

适合说明递归可行性依赖模型能力、记忆预算、接口和检查强度；没有隔离实验证明 policy-repair 改进器本身更强，也不是无限 RSI 的不可能性证明。

### PostTrainBench：生成更强模型与提升训练者要分开

Ben Rank 等；arXiv **2603.08640**，首发 **2026-03-09**，所读 v2 **2026-03-10**；正式 venue **未核实**。**全文核实，诊断性基准，不纳入上面的十项机制项目。** [版本](https://arxiv.org/abs/2603.08640)；[全文](https://arxiv.org/html/2603.08640v2)；[官方项目](https://posttrainbench.com/)。

给 agent **单 H100、10h** 去后训练指定 base model，在 held-out test 评价生成的 checkpoint。表1的加权平均为最佳 agent **23.2±1.8%**、官方 instruct 模型 **51.1%**；这是不同目标 benchmark 的加权结果，不能解释成统一“智能分数”。还观察测试污染、替换已有 checkpoint 等违规；官方训练预算也并不与 10h 对齐。这既不是同预算全面击败人工训练，也不是训练 agent 已改善自身学习算法。[全文 §§2–3、5–6](https://arxiv.org/html/2603.08640v2)。

## 4. 适合直接交给主报告的结论

1. **最有辨识度的新增证据不是新的最高分，而是 meta-evaluation。** Hyperagents 的跨域 imp@50 与 AIDE² 的 ignition test 直接触及“改进器是否更强”；前者有正面结果，后者仍不确定，不能合写成已证实自加速。[Hyperagents §5.2](https://arxiv.org/pdf/2603.19461v1)；[AIDE² §3.6](https://arxiv.org/html/2609.26457v1)。
2. **2026 的多数实际进展发生在有限修改面。** EvoX/Dream-RSI 改探索策略，GEA/MetaShare 改框架与经验复用，Meta-Harness 改外围代码，AREX 改答案状态，VisionHOPE 改图像内学习动态；应分别标记，不把它们都叫作“系统递归提高自身智能”。[EvoX](https://arxiv.org/html/2602.23413v2)；[Dream-RSI](https://arxiv.org/html/2609.14858v2)；[AREX](https://arxiv.org/html/2607.21461v3)；[VisionHOPE](https://arxiv.org/html/2609.33325v1)。
3. **收益单位必须保留。** generations-to-best、同质量发现轮数、LLM calls、总 tokens、GPU-hours 和 held-out test score 不能互换；MetaShare 正好显示更少代数可以伴随更多 tokens。[MetaShare §4.2 表1](https://arxiv.org/html/2610.21887v1)。
4. **强证据依赖外部锚定。** AIRA₂ 说明稳定、分离的评价信号可消除特定长搜索退化；POLARIS 说明可执行补丁也不保证成功演化；MARS 说明多轮并非自动优于结构化一轮。[AIRA₂ §4.3.2](https://arxiv.org/html/2603.26499v2)；[POLARIS 表2](https://aclanthology.org/2026.findings-acl.1969.pdf)；[MARS 附录F](https://aclanthology.org/2026.acl-long.1329.pdf)。
5. **未发现可据此声称无限、无外部输入或已“点火”的实证结论。** 本次核查支持有限预算、固定评价锚下的任务/搜索/研究效率收益；开放式长期加速仍是作者假设或待测方向。这个判断仅覆盖本清单，不是对所有可能 RSI 的定理。[2026 综述 §§5–8](https://arxiv.org/html/2607.07663v2)；[AIDE² §5](https://arxiv.org/html/2609.26457v1)。

## 5. 访问、版本与不一致项审计

- **全文访问**：核心十项的固定版本 HTML 均可读，PDF HEAD 均返回 `200/application/pdf`；MARS 与 POLARIS 的 ACL PDF 也可读。本清单没有因无法访问全文而冒充“全文核实”的核心项目。摘要级 survey 已单独标记。未运行任何昂贵 agent/GPU 实验。
- **Hyperagents 日期与渲染**：官方首发为 **03-19**；v1 PDF 另印 **Date: March 23, 2026**，HTML 却显示 **August 24, 2026**，并有作者排版缺损。已以 PDF 核查关键数字与完整作者，不用 HTML 内部日期改写首发或虚构新版本。[版本](https://arxiv.org/abs/2603.19461)；[PDF](https://arxiv.org/pdf/2603.19461v1)；[HTML](https://arxiv.org/html/2603.19461v1)。
- **AIRA₂ 日期纠正**：最新正式 arXiv 修订是 **04-13 v2**。先前过程更新误提的 10-07 不适用于该论文，最终记录已纠正。[submission history](https://arxiv.org/abs/2603.26499)。
- **MetaShare 数据描述**：正文 §4.1 写 15 客户×15 IRFs、225 标签、75 测试决策；附录 F 表11却给 account-group×factor 共 **4,896** 标签、**910** 测试标签，并说明客户资料/标签不发布。两处统计单位可能不同，但未充分协调；不能自行选一个数当统一样本量。本文件不引用其财务指标作核心结论。[全文 §4.1、附录F](https://arxiv.org/html/2610.21887v1)。
- **代码状态**：Hyperagents、GEA、EvoX、Meta-Harness、VisionHOPE 有已核官方实现入口；Dream-RSI 只有已核官方仓库/发布计划，不把计划当实现。AIDE²、MetaShare、AIRA₂ 的完整公开实现本次未核实；AREX 核实的是项目与模型入口，不等于完整训练代码。
- **会议时间**：MetaShare 的未来 NeurIPS workshop “accepted”是作者自报，未核实主办方。即使接收属实，也不能把截止日之后的会议举行当现状。[作者注](https://arxiv.org/abs/2610.21887)。
- **元数据差异**：MARS 正式出版作者顺序不同于 arXiv v1；POLARIS 正式作者页补全 Kakade 的中间名；GEA 会场名单姓名有展示变体。正式论文以 ACL 记录为准，核心预印本作者以所读版本为准。[MARS](https://aclanthology.org/2026.acl-long.1329/)；[POLARIS](https://aclanthology.org/2026.findings-acl.1969/)；[GEA](https://arxiv.org/abs/2602.04837)。
- **查询限制**：完整阅读 academic-search skill、计算机 profile、metadata schema、arXiv/S2 平台参考及 API cookbook；依赖检查通过 Node/curl，Chrome 未连接但未使用 Scholar。arXiv API 返回 `Rate exceeded`，改用官方页面；S2 batch 返回 `429`，不反复重试、不编引用数。两个 ACL DOI 的 OpenAlex OA 查询均找到公开 ACL PDF；Unpaywall 因未配置真实联系邮箱返回 `422`，没有伪造邮箱或绕付费墙。

## 6. 精简标准 JSON 元数据

使用 academic-search 标准字段；省略无关可选项。`publication_date` 表示论文首次公开预印本日期，不是会议举办日期；所读版本由 `pdf_url` 固定。预印本记录省略可选 DOI，不重复 arXiv 的 DataCite DOI；ACL 记录保留正式出版 DOI。`open_access_status` 未作出版商级确认的填 `unknown`。代码 `null` 代表未核实，不声称不存在；Dream-RSI 的 `code_url` 指官方仓库，但实现未发布的限制见正文。

```json
[
  {
    "title": "Hyperagents",
    "authors": ["Jenny Zhang", "Bingchen Zhao", "Wannan Yang", "Jakob Foerster", "Jeff Clune", "Minqi Jiang", "Sam Devlin", "Tatiana Shavrina"],
    "year": 2026, "publication_date": "2026-03-19", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2603.19461",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2603.19461v1",
    "code_url": "https://github.com/facebookresearch/Hyperagents", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Recursive self-improvement of AI research agents",
    "authors": ["Dhruv Srikanth", "Bingchen Zhao", "Dixing Xu", "Yuxiang Wu", "Zhengyao Jiang"],
    "year": 2026, "publication_date": "2026-09-22", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2609.26457",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2609.26457v1",
    "code_url": null, "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  },
  {
    "title": "EvoX: Meta-Evolution for Automated Discovery",
    "authors": ["Shu Liu", "Shubham Agarwal", "Monishwaran Maheswaran", "Mert Cemri", "Zhifei Li", "Qiuyang Mang", "Ashwin Naren", "Ethan Boneh", "Audrey Cheng", "Melissa Z. Pan", "Alexander Du", "Kurt Keutzer", "Alvin Cheung", "Alexandros G. Dimakis", "Koushik Sen", "Matei Zaharia", "Ion Stoica"],
    "year": 2026, "publication_date": "2026-02-26", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2602.23413",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2602.23413v2",
    "code_url": "https://github.com/skydiscover-ai/skydiscover", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Dream-RSI: Recursive Self-Improvement through Evolving Worlds",
    "authors": ["Tong Zheng", "Xidong Wu", "Zheng Zhang", "Zhankui He", "Chaoyi Zhang", "Benjamin Coleman", "Ruoqiao Wei", "Di Bai", "Haolin Liu", "Rui Liu", "Xue Wang", "Yue Zhuan", "Wang-Cheng Kang", "Renkai Xiang", "Heng Huang", "Xinwu Cheng", "Yunsong Guo"],
    "year": 2026, "publication_date": "2026-09-14", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2609.14858",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2609.14858v2",
    "data_availability": "已核官方仓库；README 说明实现代码准备发布，未验证可复现实现。",
    "code_url": "https://github.com/zhengkid/Dream-RSI", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17"
  },
  {
    "title": "MetaShare: Failure-Aware Collective Self-Improvement for Open-Ended Agents",
    "authors": ["Vishal Bhutani", "Prasang Gupta", "Sumanth Chundru", "Kevin Paul"],
    "year": 2026, "publication_date": "2026-10-15", "publication_type": "preprint", "venue": "未核实（作者称接收于 NeurIPS 2026 Meta-Agents Workshop）", "arxiv_id": "2610.21887",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2610.21887v1",
    "data_availability": "财务审计标签不公开；正文与附录样本规模描述不一致。", "code_url": null,
    "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Group-Evolving Agents: Open-Ended Self-Improvement via Experience Sharing",
    "authors": ["Zhaotian Weng", "Antonis Antoniades", "Deepak Nathani", "Zhen Zhang", "Xiao Pu", "Xin Eric Wang"],
    "year": 2026, "publication_date": "2026-02-04", "publication_type": "conference", "venue": "COLM 2026", "arxiv_id": "2602.04837",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2602.04837v1",
    "code_url": "https://github.com/UCSB-AI/GEA", "source_platforms": ["arxiv", "github", "author_homepage", "conference_website"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Meta-Harness: End-to-End Optimization of Model Harnesses",
    "authors": ["Yoonho Lee", "Roshen Nair", "Qizheng Zhang", "Kangwook Lee", "Omar Khattab", "Chelsea Finn"],
    "year": 2026, "publication_date": "2026-03-30", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2603.28052",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2603.28052v1",
    "code_url": "https://github.com/stanford-iris-lab/meta-harness", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17"
  },
  {
    "title": "AREX: Towards a Recursively Self-Improving Agent for Deep Research",
    "authors": ["Shuqi Lu", "Chaofan Li", "Kun Luo", "Zhang Zhang", "Hui Wang", "Hongwang Xiao", "Lei Xiong", "Jiahao Wang", "Sen Wang", "Xiyan Jiang", "Wanli Li", "Yuyang Hu", "Hongjin Qian", "Bingyu Yan", "Jianlyu Chen", "Ziyi Xia", "Yingxia Shao", "Kang Liu", "Zhicheng Dou", "Di He", "Chaozhuo Li", "Qiwei Ye", "Zhongyuan Wang", "Zheng Liu"],
    "year": 2026, "publication_date": "2026-07-23", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2607.21461",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2607.21461v3",
    "data_availability": "官方项目与模型入口：https://vectorspacelab.github.io/arex-model/；完整训练实现未核实。", "code_url": null,
    "source_platforms": ["arxiv", "author_project"], "fetched_at": "2026-10-17"
  },
  {
    "title": "VisionHOPE: Visual Backbones as Self-Modifying Learning Systems",
    "authors": ["Siran Peng", "Tianshuo Zhang", "Tianyu Fu", "Weisong Zhao", "Haoyuan Zhang", "Jiankuo Zhao", "Minghui Wu", "Ping Jiang", "Xiangyu Zhu", "Chenxu Zhao", "Zhen Lei"],
    "year": 2026, "publication_date": "2026-09-27", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2609.33325",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2609.33325v1",
    "code_url": "https://github.com/PSRben/VisionHOPE", "source_platforms": ["arxiv", "github"], "fetched_at": "2026-10-17"
  },
  {
    "title": "AIRA_2: Overcoming Bottlenecks in AI Research Agents",
    "authors": ["Karen Hambardzumyan", "Nicolas Baldwin", "Edan Toledo", "Rishi Hazra", "Michael Kuchnik", "Bassel Al Omari", "Thomas Simon Foster", "Anton Protopopov", "Jean-Christophe Gagnon-Audet", "Ishita Mediratta", "Kelvin Niu", "Michael Shvartsman", "Alisia Lupidi", "Alexis Audran-Reiss", "Parth Pathak", "Tatiana Shavrina", "Despoina Magka", "Hela Momand", "Derek Dunfield", "Nicola Cancedda", "Pontus Stenetorp", "Carole-Jean Wu", "Jakob Nicolaus Foerster", "Yoram Bachrach", "Martin Josifoski"],
    "year": 2026, "publication_date": "2026-03-27", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2603.26499",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2603.26499v2",
    "code_url": null, "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Recursive Self-Improvement in AI: From Bounded Self-Refinement to Autonomous Research Loops",
    "authors": ["Mingguang Chen", "Licheng Wang", "Bo Qu"],
    "year": 2026, "publication_date": "2026-07-08", "publication_type": "review", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2607.07663",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2607.07663v2",
    "data_availability": "作者公开检索语料、分类与生成脚本；本次未重新执行全语料统计。",
    "code_url": "https://github.com/deepgrounding/recursive-self-improvement", "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  },
  {
    "title": "Learn Like Humans: Use Meta-cognitive Reflection for Efficient Self-Improvement",
    "authors": ["Xinmeng Hou", "Bohao Qu", "Wuqi Wang", "Peiliang Gong", "Qing Guo", "Yang Liu"],
    "year": 2026, "publication_date": "2026-01-17", "publication_type": "conference", "venue": "ACL 2026 Long Papers", "doi": "10.18653/v1/2026.acl-long.1329", "arxiv_id": "2601.11974",
    "citation_count": null, "open_access_status": "hybrid", "full_text_status": "open_pdf", "pdf_url": "https://aclanthology.org/2026.acl-long.1329.pdf",
    "code_url": "https://github.com/Paparare/MARS/tree/main", "source_platforms": ["acl_anthology", "arxiv", "openalex"], "fetched_at": "2026-10-17"
  },
  {
    "title": "POLARIS: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair",
    "authors": ["Aditya Namdev Kakade", "Vivek Srivastava", "Shirish Karande"],
    "year": 2026, "publication_date": "2026-03-24", "publication_type": "conference", "venue": "Findings of ACL 2026", "doi": "10.18653/v1/2026.findings-acl.1969", "arxiv_id": "2603.23129",
    "citation_count": null, "open_access_status": "gold", "full_text_status": "open_pdf", "pdf_url": "https://aclanthology.org/2026.findings-acl.1969.pdf",
    "code_url": null, "source_platforms": ["acl_anthology", "arxiv", "openalex"], "fetched_at": "2026-10-17"
  },
  {
    "title": "PostTrainBench: Can LLM Agents Automate LLM Post-Training?",
    "authors": ["Ben Rank", "Hardik Bhatnagar", "Ameya Prabhu", "Shira Eisenberg", "Karina Nguyen", "Matthias Bethge", "Maksym Andriushchenko"],
    "year": 2026, "publication_date": "2026-03-09", "publication_type": "preprint", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2603.08640",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "open_pdf", "pdf_url": "https://arxiv.org/pdf/2603.08640v2",
    "data_availability": "官方项目：https://posttrainbench.com/；未运行基准。", "code_url": null,
    "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  },
  {
    "title": "A Survey of Agentic Reasoning for Large Language Models: Towards Recursively Self-Improving and Collective Agents",
    "authors": ["Tianxin Wei", "Ting-Wei Li", "Zhining Liu", "Xuying Ning", "Ze Yang", "Jiaru Zou", "Zhichen Zeng", "Ruizhong Qiu", "Xiao Lin", "Dongqi Fu", "Zihao Li", "Mengting Ai", "Duo Zhou", "Wenxuan Bao", "Yunzhe Li", "Gaotang Li", "Cheng Qian", "Yu Wang", "Xiangru Tang", "Yin Xiao", "Liri Fang", "Hui Liu", "Xianfeng Tang", "Yuji Zhang", "Chi Wang", "Jiaxuan You", "Heng Ji", "Hanghang Tong", "Jingrui He"],
    "year": 2026, "publication_date": "2026-01-18", "publication_type": "review", "venue": "未核实（arXiv 预印本）", "arxiv_id": "2601.12538",
    "citation_count": null, "open_access_status": "unknown", "full_text_status": "unknown", "pdf_url": "https://arxiv.org/pdf/2601.12538v2",
    "data_availability": "本次仅核官方摘要与版本元数据，未检索/阅读该综述全文。", "code_url": null,
    "source_platforms": ["arxiv"], "fetched_at": "2026-10-17"
  }
]
```
