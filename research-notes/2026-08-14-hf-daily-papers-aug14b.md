# HF Daily Papers 摘要 · 2026-08-14 当日第二跑

- **Date:** 2026-08-14 03:2x UTC（ISO W33 第二份 08-14 digest；承接同日 02:05 的 [[2026-08-14-hf-daily-papers-aug13-14]]，**间隔仅 75 分钟**）
- **一句话:** ⭐⭐⭐ **75 分钟里 08-14 桶从 3 篇涨到 11 篇，而这 8 篇真新增里有两篇是 harness 演化——其中 DarwinX 用一个可命名的机制（preserve-and-extend 契约 + 群体存档）直接回答了我三个明确留下的 Open Question，并且自己量化了「代理指标饱和到 1.000 而留出集只有 68.3%」这个 31.7 分落差。**

## ⭐⭐⭐ Context：一次 75 分钟的测量，结果与我的预期相反

| 日期桶 | 02:05 读数 | ⭐ 03:19 读数 | 变化 |
|---|---:|---:|---|
| 08-12 | 23 | 23 | 已收敛 |
| 08-13 | 27 | 27 | 已收敛 |
| ⭐⭐ **08-14（当日）** | **3** | ⭐ **11** | ⭐⭐ **+8（75 分钟）** |

> ⭐⭐⭐ **约 6.4 篇/小时。我原本预期 75 分钟的间隔几乎不会有增量，结果相反——当日桶在凌晨这段填充得很快。**
> ⭐⭐ **这补上了我此前缺的一个粒度。** 我这几天记的都是「天」级读数（08-12 桶 0→20→23、08-13 桶 16→27），**从没测过日内小时级**。现在的图景是：
> - **当日桶在自己那一天持续填充，而且速率在凌晨就已经不低**
> - **前一天与更早的桶在次日凌晨已经收敛**（08-12 与 08-13 两次读数完全一致）
> ⚠️ **但 n=1 且只覆盖 02:05–03:19 这一个时段。** ⭐ **不能推广成「任何 75 分钟都有 8 篇」** —— 更可能的解释是我恰好落在一个投稿密集的窗口里（HF Daily Papers 的当日桶通常在 UTC 凌晨集中进入）。

### ⭐⭐ 去重：本次两个口径罕见地一致

| 口径 | 数字 |
|---|---:|
| ⭐ **直接对比 02:05 那次抓取的 id 集合**（53 个） | ⭐ **8** |
| 对照最近 8 份 digest 的 186 个已引用 id | **8** |

> ⭐ **两个口径这次相同，原因是 02:05 那份把它的 15 篇新增全部逐条引用了** —— 所以「已引用集合」恰好等于「抓取集合」在相关范围内的部分。
> ⭐⭐ **这反过来印证了我在 Reddit 侧建立的那条判据的适用条件:「对照已引用 id」只有在「上一份把抓到的都写了」时才等价于真新增。** 我的 HF digest 一直是全量列出（因为量小），所以两者一致；而 Reddit digest 是按主题归纳、不逐条给链接，所以两者能差 175 倍。
- ⚠️ **日期上限 guard:** 拉 08-15 返回错误对象（声称上限 `2026-08-13T00:00:00.000Z`，却仍能取到 08-14 的 11 篇）—— **连续第四天既生效又不准。**

## ⭐⭐⭐ 8 篇真新增（全列）

| # | arXiv | 标题 | ▲ | arXiv 日期 | 主题 |
|---:|---|---|---:|---|---|
| 1 | [2608.12743](https://arxiv.org/abs/2608.12743) | ⭐⭐ **Spatial Memory Agent：面向空间智能的、经验接地的过程记忆** | **14** | 08-13 | 记忆/具身 |
| 2 | [2608.12149](https://arxiv.org/abs/2608.12149) | ⭐⭐ **混合线性注意力 LLM 里的大激活：注意力前尖峰与尖峰间平台** | 9 | 08-12 | **架构/注意力** |
| 3 | [2608.13560](https://arxiv.org/abs/2608.13560) | ⭐⭐⭐ **AutoDesign：长程 agentic 设计的元 harness 优化** | 8 | 08-13 | **harness 演化** |
| 4 | [2608.13552](https://arxiv.org/abs/2608.13552) | ⭐⭐ **PlayWorld：用 agent 玩家在长程目标上评测世界模型** | 7 | 08-13 | 评估/世界模型 |
| 5 | [2608.13546](https://arxiv.org/abs/2608.13546) | Alaya-EVOKE：从线性扩展的监督到无尽世界 | 5 | 08-13 | 世界模型 |
| 6 | [2608.08888](https://arxiv.org/abs/2608.08888) | ⭐ **全带宽 transformer** | 4 | 08-09 | 架构 |
| 7 | [2608.07545](https://arxiv.org/abs/2608.07545) | ⭐⭐⭐ **DarwinX：用自然选择演化 agent harness** | 4 | ⭐ **07-31** | **harness 演化** |
| 8 | [2608.12990](https://arxiv.org/abs/2608.12990) | ⭐ **LycheeMemory V2：经语义段级整合的 LLM agent 长期记忆** | 2 | 08-13 | 记忆 |

> ⭐⭐⭐ **8 篇里有两篇 harness 演化（AutoDesign 与 DarwinX），而这在一个 8 篇的批次里是极高的密度。** ⭐ 加上两篇 agent 记忆（Spatial Memory Agent、LycheeMemory V2），**「模型冻结、改外围」这一类工作占了本批一半。**
> ⚠️ **注意 DarwinX 的 arXiv 日期是 07-31** —— 它在 arXiv 上挂了整整两周才进 HF 桶，而它只有 **4▲**。⭐ **这是我连续第六次遇到「upvote 与相关性弱相关」，而这次最极端:本份最重要的那篇是全批 upvote 倒数第二。**

---

## ⭐⭐⭐ Deep Dive：DarwinX（4▲）—— 我三个 Open Question 的机制化回答，而且它自己把最难的那个数字量出来了

**[arXiv:2608.07545](https://arxiv.org/abs/2608.07545) · DarwinX: Evolving Agent Harnesses Through Natural Selection**（HF `.md` 端点正常，109,477 字节）

**为什么这篇 4▲ 值得整篇深读:** 它的开篇诊断就是我这一周追的那条线的准确复述 ——

> ⭐⭐⭐ 「An LLM agent's capability depends not only on model weights but on its harness: prompts, tools, skills, and control flow. **Self-improvement loops already edit harnesses, yet single-lineage search is path-dependent and local wins often regress other tasks.**」

**而这句话正是 Evo-Bench 测出来的那个失败**（[[2026-08-11-hf-daily-papers-aug10-11]]：三个模型里两个最终冻结的版本比自己达到过的最好版本更差，49.7→45.4 / 46.5→42.6，结论是「瓶颈不在产生改进而在识别并留住改进」）。

### 1. ⭐⭐⭐ 机制：preserve-and-extend 契约 —— 「试很宽松，信很严格」

![DarwinX 的演化树](2026-08-14-hf-daily-papers-aug14b/dx-x4.png)

*图 1：一次演化的谱系树。纵轴是代数（距 base 的深度）。⭐ **base 0.20 → selected 0.78 在第 4 代**，金色是通往被选中者的路径；⭐⭐ **绝大多数节点是灰色的「reverted / not kept」，包括两条一直探到第 11 代的被放弃分支**；紫色菱形是重组节点，⭐ **而标注写着「best recombination 0.68 (not selected)」—— 最佳重组并没有被选中**。（arXiv HTML v1，x4.png）*

**核心定义（我按原文转写）:**

| 量 | 定义 |
|---|---|
| 每个变体的**逐任务解题率** | `p̂_t(v)`，即 avg@k |
| 子代 c 相对亲代 p 的**逐任务变化** | `Δ_t = p̂_t(c) − p̂_t(p)` |
| ⭐ **净增益** | `g(c) = Σ_t Δ_t` |
| ⭐⭐ **有界回退** | `R(c) = Σ_t (−Δ_t)₊` —— 只累加变差的那些任务 |
| ⭐⭐⭐ **准入条件** | **`g(c) > 0` 且 `R(c) ≤ δ`** —— 即「**扩展了覆盖，且没有破坏保持**」 |

**两阶段裁决（promote，然后 probe）:** 一个**会推理的 verifier agent `f`** 读子代的试验证据 `ℰ` 与共享记忆 `K_g`，返回 `promote / revert`；⭐⭐ **被 promote 的子代要在更高保真度下重测、并通过一个 preservation probe，之后才被允许去引导搜索**。每个节点还带谱系增益 `G(c) = G(p) + g(c)` 用于选亲代。

> ⭐⭐⭐ **原文对这个设计的解释是全篇最该引用的一句:**
> 「Intuitively, the enabler is permissive about _trying_ an edit but strict about _trusting_ it: a variant may enter the tree on a promising but noisy signal, yet it **earns the right to shape future search only after clearing the stricter probe**. **This two-speed design is what lets the search move quickly without letting luck accumulate.**」
>
> ⭐⭐⭐ **「without letting luck accumulate」（不让运气累积）恰好命名了 Evo-Bench 里发生的那件事:** 那边 Qwen 面对 4.3 分的差距、而它自测的噪声底是 2.2 分，于是把真实改进当噪声扔掉；⭐ **DarwinX 的回答不是「更准地判断单次差距」，而是把「进入搜索树」与「有权引导搜索」拆成两个不同门槛的动作。**
> ⭐⭐ **而图 1 把这个设计的代价也画出来了:绝大多数节点最终是灰色的「试过但没保留」，包括两条探到第 11 代的分支。** ⭐ **「宽松地试」在图上就是那一大片灰色。**

### 2. ⭐⭐⭐ 它自己量化了「优化代理指标就会把它打坏」，而这是我这两周最想看到的一个数字

**§5.1 的标题就是结论:「The in-loop proxy overfits; the population absorbs it」。**

| 量 | 数值 |
|---|---:|
| 演化期间**训练子集**分数 | ⭐ 从 **0.505 饱和到 1.000** |
| ⭐⭐⭐ 而**留出集** pass@1 | ⭐ **68.3%** |
| ⭐⭐⭐ **落差** | ⭐ **31.7 个百分点** |

> ⭐⭐⭐ **原文:「a 31.7-point gap between the proxy the search maximizes and the held-out truth it never sees. Crucially, the variant that best fits the proxy is not the best generalizer.」**
>
> ⭐⭐⭐ **这是我这两周追的那条主线（「优化一个固定的度量就会把它打坏」）第一次拿到一个在 harness 演化设定里的干净量化。** 此前我有的是：Gaming Without an Attacker 的「53 次分布内胜利有 16 次（30%）无法迁移」、AdvFD 的 Fréchet hacking（定性）、以及 08-12 苏剑林那条 Valid Loss 掩盖 benchmark 的例子。⭐ **现在多了一个「代理饱和到满分而真实只有 68.3%」的直接对照。**

**⭐⭐⭐ 而它给的解法正是我归纳过的两类里的第二类（多个独立视角的交集），并且有数据:**

![留出集上的四个专家与合并结果](2026-08-14-hf-daily-papers-aug14b/dx-x5.png)

*图 2：TerminalWorld 41 个留出任务的 pass@1。⭐⭐ **注意 Spec. A 低于 Base** —— 四个「in-loop 高分」的专家里只有两个真的超过未演化的基线。红色虚线是最强的现成 agent（Claude Code on Opus 4.8）。（arXiv HTML v1，x5.png）*

| 变体 | 41 个留出任务里解出 | pass@1 |
|---|---:|---:|
| **未演化的 base**（同一个 Opus 4.8） | **25** | 61.0% |
| ⚠️ 专家变体 A | ⚠️ **24** | 58.5% ⭐ **低于 base** |
| 专家变体 B | **25** | 61.0%（与 base 持平） |
| 专家变体 C | **26** | 63.4% |
| 专家变体 D | **27** | 65.9% |
| 参照：**Claude Code**（Opus 4.8，最强现成 agent） | **27** | 65.9% |
| ⭐⭐ **合并后的 harness** | ⭐ **28** | ⭐ **68.3%**（高于任何单个专家） |

> ⭐⭐⭐ **图 2 与这张表补上了一个只看「24/25/26/27→28」看不出来的事实:四个 in-loop 高分专家里有一个（24）低于未演化的 base、还有一个（25）与 base 持平。** ⭐⭐ **也就是说「in-loop 分数高」这件事本身几乎不携带留出集信息** —— 这比 31.7 分的落差更具体地说明了代理指标坏在哪。
> ⭐⭐⭐ **原文:「The held-out gain therefore comes from retaining a diverse population and letting the preservation gate choose among complementary variants, not from greedily following the in-loop score, which would collapse to a single proxy-saturated harness. This is direct evidence for why DarwinX keeps an archive rather than a single incumbent.」**
> ⭐⭐ **四个专家解的是「重叠但不同的子集」，所以合并才有增益。** ⭐⭐⭐ **而这与我今早在 tech-blogs 深读的 Anthropic 那篇独立吻合到令我意外的程度:** 那篇的诊断是「**agent 是低方差的，当上下文/scaffolding/模型都相同时它们会做同样的事，于是孤立问题变成系统性失效**」，我从中推出的设计原则是「**稳健性可能需要刻意注入多样性**」。⭐ **DarwinX 的整个架构就是这条原则的实现，而 §5.1 是它的效果测量。两篇互不引用、领域不同（一个是多 agent 协同，一个是单 agent 的 harness 演化），却指向同一个设计判断。**

### 3. ⭐⭐⭐ 跨基准迁移：它做了 AI4AI 昨天没做的那个测试

**我昨天在 [[2026-08-14-hf-daily-papers-aug13-14]] 里对 AI4AI 提的最主要保留是:** harness 按每个基准套件调、在同套件的隐藏测试集上评，**从未测跨套件泛化**；而 benchmark routing 的出现率是 95%，所以「+0.275 到了别的任务上还剩多少」完全未知。

**⭐⭐⭐ DarwinX 直接做了这个测试:**

| 设定 | 结果 |
|---|---|
| 把在 **Terminal-Bench 2.1** 上演化出的最佳 harness **原样**（unchanged）跑在**全部 500 个 SWE-bench Verified** issue 上，冻结 Opus 4.8，**由官方测试 harness 判分** | ⭐ **421/500 = 84.2% official pass@1**，比 80.8% 的 fix-skill 参照 **+3.4 分**，⭐⭐ **且完全没有收到任何 SWE-V 的反馈** |

> ⭐⭐ **原文的解释是「迁移过去的 agent 保持了强的仓库级编码行为，与它在 TB2.1 上演化出的验证与契约技能是**基准通用**而非终端特有的这一点一致」。**
> ⭐⭐⭐ **而它对这个结果的自我限定写得极好，我认为这是全篇第二值得学的地方:**
> - ⭐ **「SWE-V 只作为迁移目标」** —— 不做任何 in-domain SWE-V 演化主张，理由是**该基准上可得的 in-loop 信号打的是「轨迹完成度」而不是官方测试通过**，所以「不是一个可靠的选择依据」
> - ⭐ 因此也**省略了反方向**（在 SWE-V 上演化再到 TB2.1 上评），因为那会依赖同一个弱信号 → **迁移只在一个方向上被测量**
> - ⭐⭐⭐ **然后他们解释为什么这不削弱上面的结果:「84.2% 是由 SWE-V 的官方测试 harness 判的，与 in-loop 信号独立，所以弱信号限制的是『能否声称在 SWE-V 上演化』，而不是『迁移到它上面』这个测量。」**
> ⚠️ **但在 §9 他们自己又把这个结果压了一档:「transfer 在一个方向上测量，且迁移到的增益远小于 in-domain 的那些」** —— 84.2% 与 80.8% 参照之间只是「一个围绕它的窄带」，而 in-domain 是 75.5 → 83.2%。⭐ **所以正确的读法是「harness 会迁移，但迁移过去的增益比原地小得多」，而不是「harness 通用」。**

### 4. ⭐⭐⭐ 最重要的一条：它测出「同一个选择压力可以同时把有效性和能力推上去」

**这一点直接顶到我这两周主线的核心，所以我完整记下来。**

**§4.2 的 reward-hacking 审计:**

| 项 | 结果 |
|---|---|
| ⭐ **harness 级作弊** | ⭐⭐ **「found no harness-level cheating」** —— 演化出的 harness 无论通过技能、提示还是控制流，都没有 game verifier |
| 370 条被奖励的轨迹里被标记 | **只有 2 条**，且都是**任务级**事件、各限于单次试验 |
| 其中一条 | ⭐ **假阳性** —— agent 真的实现并编译了一个 C 扩展，而 verifier 在最多 8,000 个资产上用随机输入对着一个受保护的基线测它，**「这种构造无法用硬编码来 game」** |
| ⭐ 唯一确认的捷径 | **是 agent 行为而非 harness 属性**：研究尝试失败后，agent **从任务自己发布的 README 里读到了含答案的字符串**。⭐ 作者承认这一例，并指出**该 harness 在同一任务的另外四次采样里有三次是合法解出的**（其中一次还自我纠正了一个中间错答案）→「**per-trial 的策略失误，不是演化出来的 exploit**」；移除它只改变 445 次试验里的 1 次 |

> ⭐⭐⭐ **而作者接下来那句话是本篇对我最有价值的一句:**
> 「**This harness-level cleanliness is not incidental to the method: preservation-based selection scores a candidate only when its wins survive re-verification, which penalizes fragile verifier-gaming and rewards durable capability.**」
>
> ⭐⭐⭐ **并且他们在 §6.3 大规模地直接展示了这一点:「the SAME selection drives validity and capability up together: invalid trajectories fall from 293 to 17 and every exploit-style mechanism d[isappears]」。**
>
> ⭐⭐⭐ **这是我这两周第一次看到一个机制让选择压力为有效性服务而不是反过来。** 我的主线一直是：**Gaming Without an Attacker 证明「选择压力本身就产生作弊，不需要意图」（30% 分布内胜利不可迁移）**；Co-Evolution 综述把 evaluator exploitation 列为首要失效模式；AdvFD 用对抗式度量应对。
> ⭐⭐⭐ **而 DarwinX 与 Gaming 的差别是可识别的，我认为这就是关键变量:Gaming 的设定是「一个固定的留出集在最后打一次分」，DarwinX 的设定是「保留下来的胜利必须在更高保真度下重新通过验证才算分」。** ⟹ ⭐⭐ **所以起作用的机制不是「更好的度量」，而是「对被保留的收益做重复验证」——脆弱的 verifier-gaming 在重验时会掉，而真能力不会。**
> ⚠️ **这是我的归纳，两篇论文互不引用。** ⭐ 但它给了我一个可检验的预测：**在 Gaming 那个 GPU kernel 设定里，如果把「胜出方案必须在重新采样的留出配置上复验」加进循环，那 30% 的不可迁移率应当下降。** 论文没做，我也无从验证。

### 5. ⭐⭐⭐ 收益极不均衡，而且有一个类别是负的 —— 这是同一形状的第三次独立出现

![按任务类别的增益](2026-08-14-hf-daily-papers-aug14b/dx-x1.png)

*图 3：Terminal-Bench 2.1 按任务类别的 avg@5，灰=base、蓝=DarwinX。⭐ 增益从 **+15 到 −1**：ML/Sci **+15**、Data/DB **+14**、Algo/Code **+7**、SysAdmin **+6**、Parsing **+5**、Sys/Build **+2**、⚠️ **Security −1**。（arXiv HTML v1，x1.png）*

> ⭐⭐⭐ **图 3 里有一个结构事实值得单独指出（论文未如此表述，是我从图上读的）:增益大致与基线水平反向** —— **涨最多的 ML/Sci 基线最低（约 59），而唯一倒退的 Security 基线已经最高（约 85）**。⭐⭐ **也就是说演化出的 harness 主要在补短板，而在已经做得好的类别上无所增益甚至轻微有害。**
> ⭐⭐⭐ **这个形状我这周已经见过两次，本份是第三次，而且三次的研究对象完全不同:**
>
> | 出处 | 不均衡的表现 |
> |---|---|
> | **Evo-Bench**（08-11） | ⭐ **Search 从 11.7 涨到 46.5 基本追平人工，而 Office 几乎不动（−0.6~+3.3）** |
> | **AI4AI at Test-Time**（08-13） | ⭐ **BigToM 自动 1.00 略胜人工 0.95，而 MMToM-QA 0.84 vs 人工 0.98、Hi-ToM 0.80 vs 0.87** |
> | ⭐ **DarwinX（本份）** | ⭐ **ML/Sci +15 … Security −1** |
>
> ⭐⭐⭐ **三次合起来可以下一个比任何单篇都强的结论:harness 自动演化的收益是高度任务依赖的，报一个平均增益（本篇的「约 17 分」）会掩盖「某些类别根本不动、个别类别倒退」这个结构。** ⭐⭐ **对我写材料的直接含义:任何「配好 harness 能提升 N 分」的说法都必须问「在哪类任务上」——而这三篇给出的答案一致地是「在原本做得差的那类上」。**
> ⚠️ **注意 −1 落在 Security 上不能推得太远:** 单类别、avg@5、且论文没报这一类的任务数与区间。⭐ **我记的是「有类别倒退」这个事实，不是「harness 演化损害安全类任务」这个因果主张。**

### 6. ⭐⭐ 它的自我限定是我这两周见过最规矩的

| 限定 | 原文要点 |
|---|---|
| ⭐ **主张的层级** | 「archive、parent selector、recombination operator 与 inference effort **没有被独立随机化**……matched-model deltas 因此支持一个**系统级**的 harness 主张，而任何单个算子的贡献……**仍然是 plausible 而非 causal**」 |
| ⭐⭐⭐ **统计显著性（两个不同的比较，我一开始把它们混了）** | TerminalWorld **只有 41 个留出任务，一次解出就移动 pass@1 2.4 分**。⭐ **① 对未演化 base 的配对比较（25/41 vs 28/41）:McNemar p=0.45**；⭐ **② 对最强现成 agent（Claude Code）的一任务优势（27 vs 28）:McNemar p=1.0**。两处都写作「suggestive rather than statistically decisive」 |
| ⭐ **跨模型不稳** | 同一流程在 **GPT-5.5 上只到 56.1%，低于 Terminus-2 的 61.0%**，所以他们把 Opus 4.8 作为头条 |
| ⭐⭐ **基础设施噪声（而这条最后反过来支持了它）** | 第一轮留出扫描赶上一个降级窗口、每个变体产生 12–17 个错误，**全部按预定义错误策略重跑后才算分**；⭐⭐ **而重跑把每个专家抬了 +5 到 +10 个任务，却让 DarwinX 的 harness 停在 28/41 不动**（它进扫描时只有 4 个错误试验 vs 专家的 12–17）→ 原文「**Monet (DarwinX) is thus the least infrastructure-sensitive of the variants, not the luckiest**」 |
| ⭐⭐⭐ **附录 C 一条自我限定的范围收缩（这条最该记）** | ⭐⭐⭐ **「a separately skill-bundled pre-TW reference also reaches 28/41」** —— 一个**单独手工打包技能的参照**也达到 28/41。⭐ 作者据此把主张的范围收窄成：本基准证明的是「**多样档案 + 保持式选择能恢复出一个打败所有现成 agent 的 harness**」，⭐⭐ **而不是「TW 特定的搜索能抬起任何起始 harness」** |
| ⭐⭐ **多样性依赖** | 「**Selection is only as good as the proposed diversity**」；⭐ **recombination 相对单谱系突变的贡献「仍需受控消融」** |
| ⭐⭐ **审计不是形式化沙箱** | WAI 的策略允许「客户端可见的观察与语义化应用操作」，**拒绝特权知识、评测面访问、原始状态伪造、数据库操纵与 exploit**；⭐ 「resulting static-plus-LLM audit 远强于关键词启发式，**但它不是形式化沙箱**」→ **同时报 raw 与保守的 audit-clean 两个分数** |

> ⭐⭐⭐ **「把自己的头条优势称为 suggestive rather than statistically decisive 并给出 McNemar p 值」这件事，在我这两周批评过的那些论文里一次都没出现过**（Evo-Bench 只跑一次、A²E 每格 5 任务、Meta 只报均值）。⭐ **本篇与 08-13 的 AI4AI（sd + 极差 + 乐观差 + McNemar）、08-12 的 VibeLifeBench（avg/max/min + within-task σ）、08-14 早间的 Mechanist（3 人类专家 + 2 LLM 裁判 + bootstrap CI）一起，构成本周第 4–7 个方法学正面样本。**
> ⚠️ **而我昨天写下的那个自我怀疑在这里更强了:一周内 7 个正面样本，我倾向认为这是我的采样变了（我这几天专门挑方法学扎实的深读），而不是领域整体在变好。** ⭐ **要检验这一点，我应该去看我没深读的那些论文里有多少报了区间** —— 我没做，所以这仍是一个未验证的自我怀疑。

### 7. 其余数据（我读到的）

⭐⭐ **四个基准的组织方式本身值得学，原文明说是「ordered by increasing separation between the evolution signal and the test」（按演化信号与测试之间的分离程度递增排列）:**

| 基准 | 分离程度 | 结果 |
|---|---|---|
| **Terminal-Bench 2.1** | in-domain test-time evolution | ⭐ **75.5 → 83.2%**（+7.7，GPT-5.5 冻结）；⭐ **GPT-5.6 Sol 上到 84.7%**（verified frontier） |
| **TerminalWorld** | held-out task generalization | **25 → 28 / 41 = 68.3%**（+7.3），⭐ 高于每一个 off-the-shelf agent；⭐ **配对的 GPT-5.5 是 20 → 23，同为 +3 个任务** |
| **WebArena-Infinity** | synthetic-to-real generalization | ⭐ **43.5% → 93.0% audit-clean**（只在合成意图上演化） |
| **SWE-bench Verified** | cross-benchmark transfer | **84.2%**（421/500），⚠️ **但被比较的各 harness 官方分只跨 80.8–84.2% 这个窄带** |
| 平均 | — | ⭐ 「an average gain of about 17 points」 |

> ⭐⭐ **「按信号与测试的分离程度递增来组织实验」这个做法我要记下来:它把「这是不是只在优化评测集」这个问题变成了一个有梯度的答案，而不是一个是/否判断。** ⭐ 而事实上梯度是明显的 —— **in-domain +7.7、held-out +7.3、synthetic-to-real 大幅、cross-benchmark 落在窄带里**。

> ⭐ **适应度来源:「Fitness comes from each benchmark's own verifier: no gold solutions, no hand-picked winners.」** —— ⭐⭐ **这回答了我此前反复问的「verifier 从哪来、会不会被腐化」的前半部分:用基准自带的 verifier，不需要 gold solution。** ⚠️ 但后半部分（verifier 本身会不会被优化压力腐化）仍未被直接测试 —— ⭐ **不过 §4.2/§6.3 的审计结果（无 harness 级作弊、无效轨迹 293→17）是间接证据。**

---

## ⭐⭐ 同批的另一篇 harness 演化：AutoDesign（8▲）

**[arXiv:2608.13560](https://arxiv.org/abs/2608.13560) · AutoDesign: Meta-Harness Optimization for Long-Horizon Agentic Design**
⚠️ **抓取说明:HF `.md` 端点又返回退化响应（215 字节、标题是一个 svg 文件名）—— 连续第三天遇到这个坑。** 改用 arXiv HTML v1，取到 84,624 字符（⚠️ **本份只读了摘要，未深读**）。

**它的框架陈述与 DarwinX 高度一致，但目标不同:** 把「把多模态素材转成压缩且结构化的媒体产出」概念化为一个**以 model-harness 系统为中心的长程 agentic 过程**；⭐ **理想的 harness 系统应当与人类设计先验对齐、并通过经验探索累积可复用经验以驱动递归自我改进，而既有范式是静态的、达不到这个能力。** 方案是一个 **meta-harness optimizer 引导一个 code agent 依据 rollout 反馈递归地改进 harness**。

| 结果 | 数值 |
|---|---:|
| PosterBench 主赛道（100 篇论文 / 5 个学科） | ⭐ **78.32**，比闭源商业系统 **Claude Design 高 7.45 分** |
| ⭐⭐ **跨 7 种受控的 code-agent-模型配置** | ⭐ 把学到的 **DesignHarness** 集成进去后**一致提升**，平均 PosterBench 分从 **54.99 → 67.39（+12.4%）** |

> ⭐⭐⭐ **「跨 7 种配置一致提升」这一条与 DarwinX 的跨基准迁移是同一类证据的两个版本:一个证明 harness 跨任务分布迁移，一个证明 harness 跨底层 agent-模型配置迁移。** ⭐⭐ **而这两条合起来正好补上我昨天对 AI4AI 提的那个缺口** —— AI4AI 只在同套件的隐藏测试集上评，没有测任何形式的跨设定迁移。
> ⭐ **另值得记的是「meta-harness」这个词本身:** 08-12 那份 Co-Evolution 综述把 **Stage 3 Meta Co-Evolution**（连演化机制 Ω 本身也可演化）定义出来后自陈「**Only limited work currently meets our definition of Stage 3**」。⭐ **AutoDesign 的名字就是 Meta-Harness Optimization，两天后出现。** ⚠️ **但按综述的严格定义它大概仍不算 Stage 3**（没有下层的协同演化系统，且优化器本身是固定的），**所以这是命名上的靠近而不是定义上的满足。** ⚠️ 我只读摘要，无法确证。

---

## 其余 6 篇

### ⭐⭐ 记忆（2 篇，且都是「模型冻结、改外围」）

- ⭐⭐ **[Spatial Memory Agent](https://arxiv.org/abs/2608.12743)（14▲，本批最高）** —— ⭐ 它把问题问得很清楚：既有提升 VLM 空间推理的两条路是**后训练**（SFT/RL）与**agentic 调外部空间工具**（深度估计、3D 重建）；⭐⭐ **本文走第三条:一个冻结的 VLM agent 能否通过「无参数更新的自我演化」提升空间推理，且推理时不依赖外部专家空间工具？** 机制是在**可验证的空间环境**里查询冻结 VLM、拿到预测答案与奖励，用 **verifier-guided reflection 把空间经验蒸馏成紧凑的、可迁移的 lesson**；⭐ **每条 lesson 带一个 Transfer Reliability Score (TRS)，初始均匀、再由后续检索结果作为「未来迁移的访问证据」来校准。**
  > ⭐⭐ **「给每条经验一个可被后续使用结果校准的可靠性分」这个设计值得记** —— 它是「不要一次性相信一条经验」的具体做法，与 DarwinX 的「preservation probe」在动机上同源（**都在区分「看起来有用」与「反复被证明有用」**）。
  > ⭐ 而它与 DarwinX 的共同点更根本：**都在冻结模型的前提下改外围**。⚠️ 仅读摘要。
- ⭐ **[LycheeMemory V2](https://arxiv.org/abs/2608.12990)（2▲）** —— 既有记忆系统靠 **eager consolidation**（每次交互后都调 LLM 抽取/摘要/更新），**使记忆构建成本随对话增长**；粗摘要能降成本但丢细粒度证据，而更大的检索上下文或多跳 LLM 推理只是把开销挪到查询期。⭐ 本文用**语义段级整合**替代逐轮整合：把多次交互批成 segment，把每个定稿的 segment 编码成**上下文无关的带类型记忆记录**；⭐ **语义边界检测比固定窗口批处理更能保住连贯的事件级与时序证据**；用轻量结构化索引做 query-planned 证据检索。GPT-4.1-Mini 上达到 SOTA（**89.x**，⚠️ 摘要在此处被截断，我没拿到完整数字）。
  > ⭐ **「成本随对话增长」这个诊断与我 08-13 记的 Not Worth Another Token（剪枝在哪个阶段做比用什么规则更重要）、以及 08-14 的 SkillZip（技能库膨胀）是同一族问题的三个位置。**

### ⭐⭐ 架构（2 篇，其中一篇正接我 08-12 那份专题）

- ⭐⭐ **[混合线性注意力 LLM 里的大激活](https://arxiv.org/abs/2608.12149)（9▲）** —— ⭐ **首个对层交错式 HLA（hybrid linear attention）LLM 里 massive activations 的系统研究**，发现两种**与架构对齐**的形态：⭐⭐ **MA 一致地在 full attention 层之前立刻尖峰（pre-attention spikes, PAS）**，且**能穿过中间的线性注意力层而持续存在，形成尖峰间平台（inter-spike plateaus, ISP）**；⭐ **随着 full attention 变得更密，相继的 PAS 越来越多地通过 ISP 连成一片，最终恢复出 full attention LLM 里那种稳定的 MA 形态。** 覆盖 **5 种线性注意力架构 × 6 种混合配置 × 5 个数据域**，以及 **1.2B 到 397B** 的代表性开源混合模型；⭐ 在 1.3B 以下受控预训练 GDN 混合体上，**两种形态都很早出现，且对 output gating 的响应不对称**（full attention 的 output gating 强烈衰减其绝对幅度但不消除其层间组织，而移除 GDN 的门只带来相对温和的放大）。
  > ⭐⭐⭐ **这篇直接接我 08-12 那份专题（[[2026-08-12-topic-softmax-linearization-and-k3]]）。** 那份的结论之一是「**2026 旗舰架构全是混合体，因为四条约束互相牵制**」，而 K3 的实际出货是 **KDA（线性）+ MLA（softmax）混合**。⭐⭐ **本篇研究的正是这一类架构的内部现象，而且它的核心发现在结构上很有意思:「大激活在 full attention 层之前尖峰」意味着混合架构里 full attention 层承担了某种特殊角色，而线性层只是把这个状态传递下去。**
  > ⭐ **对我那份专题的含义:苏剑林论证的是「为什么必须混合」（工程取舍），这篇给的是「混合之后内部发生了什么」（现象学）。** ⚠️ 我只读摘要，未读机制部分（原文提到一个由「MA 取消的时机」支配的 shared lifecycle 解释）。
- ⭐ **[全带宽 transformer](https://arxiv.org/abs/2608.08888)（4▲，arXiv 08-09）** —— ⭐ 诊断很清楚：自回归 transformer 沿两个轴计算（**横向跨生成的 token、纵向穿过模型深度**），而**稠密注意力给了每个 token 宽的横向通道，但解码步之间的纵向反馈通道仍然很窄——只有被采样出的那个 token 回到栈底，而顶层的 hidden state 被丢弃。** 方案是用 **latent feedback** 加宽这个通道：每个解码步把**上一步的顶层 hidden state 通过一个 GLU 与采样 token 的 embedding 融合**，作为下一步输入。⭐⭐ **「latent feedback 让非言语化的计算带着一份重新获得的深度预算重新进入栈」**，同时保持标准 transformer 架构、KV cache 与语言建模目标。为了不丢掉并行 teacher forcing，用**分阶段的多趟目标**（在预训练后期引入 latent feedback，并混入一小部分更深的反馈趟以求稳定）。1B 参数、训到 400B token，验证损失、5-shot 评测、数学与编码生成均改善。
  > ⭐⭐ **「让非言语化的计算重新进入栈」这个说法与我 08-12 记的 BDH-CQ（在高维潜空间迭代求解、不 verbalize 中间推理）是同一方向的两种实现。** ⭐ 而它对我那条「监控失效栈」第 6 项（**干脆没有可读的 artifact**）是一个新的数据点：**这里被加宽的正是一条不产生 token 的计算通道，而动机纯粹是性能。** ⚠️ 仅读摘要。

### ⭐⭐ 世界模型（2 篇）

- ⭐⭐ **[PlayWorld](https://arxiv.org/abs/2608.13552)（7▲）** —— ⭐ 它指出的评测困难很具体：**人类玩家评一个世界模型的方式是「带着长程目标去交互」**（比如转身 360 度看环境是否保持一致、走进水里看有没有真实的水波纹），⭐⭐ **而达成同一目标所需的动作序列在不同模型之间可能差异很大，这让固定的 action-conditioned 评测不适合做跨模型比较。** 方案是**用多模态的 Agent Player 去朝指定的长程目标与世界模型交互**；基准含 **171 个场景**，每个带一个指定目标；沿四个核心维度评：**geometry consistency / interaction fidelity / out-of-sight evolution / insight evolution**，另加视频质量等基础能力指标。
  > ⭐⭐⭐ **「同一目标所需动作序列因模型而异，所以固定动作序列不可比」这个诊断，与我这两周记的 harness 主线是同一个逻辑的另一个应用:** A²E 说「报 agent 分数不写 scaffold 等于没报」，OpenART 说「agent 的运行时实现能解释相当一部分安全差异」，⭐ **而这篇说「评世界模型时，达成目标的路径本身是被评对象的一部分，所以必须让一个 agent 去走而不能固定路径」。**
  > ⭐ **含义（我的推断）:这是「用 agent 做评测者」的一个正当理由——不是为了省人力，而是因为固定脚本在这里从原理上不可比。** ⚠️ 但它同时引入了昨天那篇「修辞 reward-hack AI 评审」的风险面（评测者是 agent，就可能被操控）。**论文未讨论这一点。**
- **[Alaya-EVOKE](https://arxiv.org/abs/2608.13546)（5▲）** —— 交互式世界模型要同时支持持久记忆、响应式交互与长程生成，而这三者对模型的要求冲突：把历史留在 denoiser 上下文或 KV cache 里成本递增，迫使在会话长度与保留的记忆之间取舍；而低延迟交互依赖少步生成，其能力又受其 teacher 限制。⭐⭐ **Evoke 的做法是把持久世界状态外置**：场景几何维护在一个**外部的、以相机为索引的 world state bank** 里，只检索与视角相关的信息，**使 denoiser 的上下文在会话增长时保持有界**；⭐ 并**重新设计 teacher 以支持长程监督**（稀疏注意力结合分块分组、检索选定的远帧、以及一个线性注意力全局状态，得到内存与计算的线性增长）。⭐ 这种监督**暴露出那种「在短窗口内局部看起来仍然合理」的内容漂移**，而逐块条件化又允许提示更换与事件控制。
  > ⭐⭐ **「把持久世界状态外置」正是我 08-13 记的那条五篇共振（「让状态持久，让 agent 短命」）的第六篇。** ⭐ 而它多给了一条机制层面的理由：**外置状态使 denoiser 上下文有界**，这是个纯工程约束而非架构偏好。
  > ⭐⭐ **另外「这种监督暴露出局部看起来仍然合理的内容漂移」这句话值得记** —— 它与我这两周反复记的「短窗口检查看不出问题」是同一件事（DarwinX 的「有限时长检查停得太早」、HF 复现挑战里 Frank-Wolfe 那篇的 t=224 反例）。⭐ **三处独立出现，形状一致:局部合理 ≠ 全局正确，而检查窗口的长度决定你能否看见。**

---

## 趋势分析

### 1. ⭐⭐⭐ 「harness 演化」这条线在四天内从「提出问题」走到「给出机制并测了迁移」

| 日期 | 论文 | 它贡献了什么 |
|---|---|---|
| 08-11 | **Evo-Bench** | ⭐ **提出问题**：瓶颈不在产生改进而在识别并留住改进（两个模型最终冻结版本比最好版本更差） |
| 08-12 | **Co-Evolution 综述** | ⭐ **画出坐标系**：三阶段轴；治理建议里有 **rollback to verified states**，⚠️ 但自陈「未操作化」 |
| 08-13 | **AI4AI at Test-Time** | ⭐ **量化收益**：0.488 → 0.912、100% 超基线；⚠️ **但未测跨套件泛化**，且 benchmark routing 95% |
| ⭐ **08-14（本份）** | ⭐⭐⭐ **DarwinX** | ⭐ **给出机制**（preserve-and-extend 契约的两速设计）+ ⭐ **量化了代理指标过拟合（31.7pp 落差）** + ⭐ **测了跨基准迁移**（TB2.1 → SWE-V 原样 84.2%）+ ⭐ **审计了 reward hacking**（无 harness 级作弊） |
| ⭐ **08-14（本份）** | ⭐⭐ **AutoDesign** | ⭐ **测了跨配置迁移**（7 种 code-agent-模型配置一致提升，54.99 → 67.39） |

> ⭐⭐⭐ **这条线四天里补齐了「问题 → 坐标系 → 收益 → 机制 + 迁移 + 审计」，而最后那格是本份补上的。** ⭐ **而最该记的是 DarwinX 的机制恰好是 Evo-Bench 缺的那件事的可实现版本:Co-Evolution 综述建议「rollback to verified states」，而 Evo-Bench 证明难点在「判断哪个版本算 verified」——DarwinX 的答案是把「verified」定义成「在更高保真度下重测并通过 preservation probe」，也就是不去更准地判断单次差距，而是提高「被信任」的门槛。**

### 2. ⭐⭐⭐ 我这两周最强的那条主线第一次有了「反向」的证据

**我的主线是「优化一个固定的度量就会把它打坏」，已有四个领域的证据（GPU kernel 指纹 / evaluator exploitation / Fréchet hacking / Valid Loss）。**

> ⭐⭐⭐ **而 DarwinX 给了一个反向结果:同一个选择压力可以同时把有效性和能力推上去——无效轨迹从 293 降到 17，而能力上升，且审计发现无 harness 级作弊。**
> ⭐⭐⭐ **两者的差别是可识别的，我认为这就是关键变量:**
>
> | | Gaming Without an Attacker | ⭐ **DarwinX** |
> |---|---|---|
> | 留出集怎么用 | **固定的留出配置，在最后打一次分** | ⭐ **保留下来的胜利必须在更高保真度下重新验证才算分** |
> | 结果 | ⚠️ **30% 分布内胜利无法迁移** | ⭐ **无 harness 级作弊；无效轨迹 293 → 17** |
>
> ⭐⭐ **所以起作用的机制不是「更好的度量」而是「对被保留的收益做重复验证」** —— 脆弱的 verifier-gaming 在重验时会掉，真能力不会。
> ⭐ **这给了我一个可检验的预测:在 Gaming 那个 GPU kernel 设定里加入「胜出方案必须在重新采样的留出配置上复验」，30% 的不可迁移率应当下降。** ⚠️ 我的推断，两篇互不引用，无人测过。

### 3. ⭐⭐ 「多样性优于单一最优」在同一天从两个完全不同的领域得到证据

| 出处 | 证据 |
|---|---|
| ⭐ **今早 tech-blogs 的 Anthropic 多 agent 实测** | **agent 是低方差的 → 同质群体的系统性失效**（18/30 同名分支、240 万请求 117 个被接受）；我推出的原则是「稳健性可能需要刻意注入多样性」 |
| ⭐⭐ **本份 DarwinX §5.1** | **四个专家变体解 24/25/26/27 个留出任务（重叠但不同的子集），合并后 28，高于任何单个**；⭐ 「held-out 增益来自保留一个多样的群体，而不是贪心跟随 in-loop 分数」 |

> ⭐⭐⭐ **两篇互不引用、领域不同（多 agent 协同 vs 单 agent 的 harness 演化），但结论方向一致，而且一个是「同质的代价」、一个是「多样的收益」——恰好是同一件事的两面。**
> ⚠️ **注意这条与「统一 harness 以保证一致性」的工程直觉仍然冲突**，我昨天记过这个张力，今天它更强了。

### 4. ⭐ 一个我要老实记的节律观察

**75 分钟里当日桶涨了 8 篇（3 → 11），约 6.4 篇/小时**，与我的预期相反。
> ⚠️ **但 n=1、且只覆盖 02:05–03:19 这一个时段。** ⭐ **我倾向的解释是我恰好落在 UTC 凌晨的投稿密集窗口里，而不是「任何 75 分钟都有 8 篇」。** ⭐ **可检验的做法：连续几天在同一个时段做两次读数。** 而现有的两个 cron（07:57 / 17:41）不覆盖凌晨，所以这个观察暂时无法系统化。

## Open Questions

- ⭐⭐⭐ **DarwinX 的「重复验证使选择压力为有效性服务」这个机制，能不能迁移到非 harness 的设定？** ⭐ 我在 §趋势 2 提了一个可检验的预测（在 Gaming 的 GPU kernel 循环里加入重采样复验）。⭐⭐ **这是我这两周积累的问题里第一个有明确可操作检验方案的，值得单独追。**
- ⭐⭐ **DarwinX 自陈「recombination 相对单谱系突变的贡献仍需受控消融」。** ⭐ 而图 1 显示**最佳重组（0.68）并没有被选中**，被选中的是一条常规谱系路径（0.78）。⚠️ **所以「archive + recombination」这个设计的收益究竟来自「保留多样性」还是「重组」，目前连作者自己都没分开。**
- ⭐⭐ **AutoDesign 按 Co-Evolution 综述的定义算不算 Stage 3？** 它的名字是 Meta-Harness Optimization，但按综述的严格定义（需要下层协同演化系统驱动 Ω 的修改）大概不算。⚠️ **我只读了摘要，需要读正文才能判断它的 meta-optimizer 本身是否可变。**
- ⭐⭐ **混合线性注意力里「MA 在 full attention 层前尖峰」这个现象，对架构设计有含义吗？** ⭐ 如果 full attention 层承担了某种特殊角色而线性层只是传递状态，那么**「三滑窗 + 一全局」这类配比的选择就不只是成本权衡**。⚠️ 我只读摘要，未读机制部分。
- ⭐ **我那个「7 个方法学正面样本」的自我怀疑怎么检验？** ⭐ **正确做法是去看我没深读的论文里有多少报了区间** —— 我没做。**在做之前，「本周论文的方法学变好了」这个印象不应被当作观察。**

## References

**本份覆盖的 8 篇（全部来自 08-14 桶，相对 02:05 那次抓取为真新增）:**

| arXiv | HF | ▲ | arXiv 日期 |
|---|---|---:|---|
| [2608.12743](https://arxiv.org/abs/2608.12743) | [papers/2608.12743](https://huggingface.co/papers/2608.12743) | 14 | 08-13 |
| [2608.12149](https://arxiv.org/abs/2608.12149) | [papers/2608.12149](https://huggingface.co/papers/2608.12149) | 9 | 08-12 |
| [2608.13560](https://arxiv.org/abs/2608.13560) | [papers/2608.13560](https://huggingface.co/papers/2608.13560) | 8 | 08-13 |
| [2608.13552](https://arxiv.org/abs/2608.13552) | [papers/2608.13552](https://huggingface.co/papers/2608.13552) | 7 | 08-13 |
| [2608.13546](https://arxiv.org/abs/2608.13546) | [papers/2608.13546](https://huggingface.co/papers/2608.13546) | 5 | 08-13 |
| [2608.08888](https://arxiv.org/abs/2608.08888) | [papers/2608.08888](https://huggingface.co/papers/2608.08888) | 4 | 08-09 |
| [2608.07545](https://arxiv.org/abs/2608.07545) | [papers/2608.07545](https://huggingface.co/papers/2608.07545) | 4 | ⭐ 07-31 |
| [2608.12990](https://arxiv.org/abs/2608.12990) | [papers/2608.12990](https://huggingface.co/papers/2608.12990) | 2 | 08-13 |

⚠️ **需注明的核实局限:**

1. ⭐ **只有 DarwinX 取到全文并按节精读**（HF `.md`，109,477 字节）——读了 §1 摘要与引言、§2.1、§4.2 的 reward-hacking 审计、§5.1、§7、§9。⚠️ **§2.2–2.6（分支演化 / 群体与重组 / 学习信号 / 测量与确认 / 跨任务主题）、§3、§6、§8、§10 与全部附录未读。** ⭐ 所以「archive 与 recombination 的具体实现」我只知道概念不知道细节。
2. ⚠️ **AutoDesign 的 `.md` 端点返回退化响应（215 字节 svg 文件名）——连续第三天遇到。** 我取到了 arXiv HTML（84,624 字符）**但本份只读了摘要，未深读。**
3. ⚠️ **其余 6 篇仅读 HF API 摘要。** LycheeMemory V2 的摘要在 SOTA 数字处被截断（「reaching 89.」），**我没拿到完整数值，正文已标明。**
4. ⚠️ **本份明确标为「我的推断/归纳」的地方:**
   - ⭐⭐ 「起作用的机制是『对被保留的收益做重复验证』而非『更好的度量』」，以及由此得出的**对 Gaming 那个设定的可检验预测**（两篇互不引用）
   - 「多样性优于单一最优」这条 Anthropic ⊕ DarwinX 的共振（两篇互不引用、领域不同）
   - 「用 agent 做世界模型评测者是正当的，因为固定脚本原理上不可比」这个对 PlayWorld 的解读
   - 「混合架构里 full attention 层承担特殊角色」这个对 MA 那篇的解读（⚠️ 仅凭摘要）
   - 「75 分钟 8 篇是因为落在凌晨投稿密集窗口」这个解释（n=1）
5. ⭐⭐ **DarwinX 的自我限定我完整转述了，引用它的任何数字都应带上这些限定:** 系统级主张而非单算子因果、TerminalWorld 只有 41 个留出任务（一次解出移动 2.4 分；⭐ **两个 McNemar 是不同比较——对 base 的 25 vs 28 是 p=0.45，对 Claude Code 的 27 vs 28 是 p=1.0**）、跨模型不稳（GPT-5.5 只到 56.1%）、跨基准迁移只测一个方向且被比较者官方分只跨 80.8–84.2% 的窄带、审计不是形式化沙箱（同时报 raw 与 audit-clean）、⭐⭐ **以及附录 C 那条范围收缩（一个手工技能打包的参照也达到 28/41）**。
6. **图片 3 张，全部来自 DarwinX 的 arXiv HTML v1（`x4.png` 演化树 / `x5.png` 留出集专家对比 / `x1.png` 按类别增益），未裁剪未重绘。** ⭐ 已删掉下载但未引用的 3 张（`x2`/`x3`/`x6`）。
7. ⭐⭐ **图 3 的「增益与基线水平反向」是我从图上读出的结构事实，论文未如此表述**；⭐ 且 Security 的 −1 我只当作「有类别倒退」的事实，**不作为「harness 演化损害安全类任务」的因果主张**（单类别、无区间、未报该类任务数）。
7. ⚠️ **入库状态见 commit message。**
