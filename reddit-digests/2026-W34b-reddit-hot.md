# Reddit 热门话题 · 2026-W34b（当日第二跑，2 小时窗口）

- **Date:** 2026-08-18 08:5x UTC（周二）· 承接同日 06:56 的 [[2026-W34-reddit-hot]]，**间隔约 2.0 小时**
- **抓取:** `scripts/reddit_fetch.py --time week`，**12/12 子版全部拿到，277 帖唯一**（主抓 delay 10 得 8/12，⭐ 逐个补抓 4/4 一次成功，**连续第六次**）
- **一句话:** ⭐⭐⭐ **2 小时窗口只有 4 条新进榜（其中仅 1 条是新发布），所以本份很短。但它有两个真正的产出：①这是我最短的窗口测量，且第一次把「新进榜」与「新发布」这两个我此前混用的量分开了 ②按我今早刚建立的「每份扫 rank ≤ 5 未引用条目」那条规则，捞到 3 条今早漏掉的实质内容——其中一条是我今早覆盖过的同一新闻，但那个更高排名的版本标题里的信息远超我当时引的媒体版。**

## ⭐⭐ Context：最短的窗口测量，以及一个我此前混用的区分

| 口径 | 数字 |
|---|---:|
| ⭐ **A. 新进榜**（对比今早 06:4x 抓取的 276 个 permalink） | ⭐ **4** |
| ⭐⭐ **A′. 新发布**（按发布时间戳 > 08-18 06:45） | ⭐ **1** |
| ⚠️ B. 对照最近 5 份 digest 的 156 个 permalink | ⚠️ **165** |

> ⭐⭐⭐ **A 与 A′ 差 4 倍，而这个差第一次把我此前混用的两件事分开了:**
> - **「新进榜」= 相对上一次抓取，榜单里多出来的条目**（4 条）
> - **「新发布」= 发布时间在上次抓取之后的条目**（1 条）
> ⟹ ⭐⭐ **差值那 3 条（08-18 01:10 / 03:11 / 06:31 发布）是「今早就已存在、但当时还没攒够票挤进 top-25」的帖子。**
> ⭐⭐⭐ **这正是我此前反复说的「进入周榜是一个阈值事件而不是一股稳定的流」的直接可见证据 —— 而在此之前我只能从速率抖动间接推断。**
> ⭐ **含义：`A ≥ A′` 恒成立，而差值就是「本窗口内跨过阈值的旧帖数」。** ⟹ ⭐⭐ **我此前那几次报的「真新增」用的是 A′（时间戳口径），所以它系统性地低估了「榜单内容的变化量」。两个都对，但答的是不同问题。**

**⭐ 节律表补上最短的一格:**

| 间隔 | 新进榜 (A) | 新发布 (A′) |
|---|---:|---:|
| ⭐ **2.0 小时（本次）** | ⭐ **4** | ⭐ **1** |
| 3 小时（08-11） | 9 | — |
| 4 小时（08-12） | — | 1 |
| 8.5 小时（08-14） | — | 7 |
| 40 小时（08-14 早） | — | 74 |
| 91.5 小时（08-18 早） | — | 128 |

> ⚠️ **表里的空格是因为我此前几次只算了其中一个口径** —— ⭐ **从本份起两个都算，因为它们答不同的问题。**
- ⚠️ **失败子版仍是 LocalLLaMA / ClaudeAI / devops / statistics** —— ⭐ 与今早那次同组，而与 08-11/08-12/08-14 11:0x 那三次的另一组不同。⭐ **两组交替出现，仍支持「哪四个失败是随机的」这个修正。**
- ⚠️ **r/datascience 仍只 9 帖**（连续第 9 份严重截断）；r/statistics 18 帖。

## ⭐⭐ 新进榜的 4 条

| 子版 | rank | 帖子 |
|---|---:|---|
| ⚠️⚠️ **statistics** | 17 | ⭐ [**[R] Proof of the Riemann hypothesis just dropped!**](https://www.reddit.com/r/statistics/comments/1vrijpm/r_proof_of_the_riemann_hypothesis_just_dropped/)（08-18 07:45，⭐ **唯一一条「新发布」**） |
| ⭐⭐ **aws** | 13 | ⭐ [**How do I recover an AWS account after accidental deletion or a destructive change?**](https://www.reddit.com/r/aws/comments/1vrh971/how_do_i_recover_an_aws_account_after_accidental/)（08-18 06:31） |
| ⭐⭐ **LocalLLaMA** | 24 | ⭐ [**Qwen dev says not to wait for 35B-A3B**](https://www.reddit.com/r/LocalLLaMA/comments/1vrdetw/qwen_dev_says_not_to_wait_for_35ba3b/)（08-18 03:11） |
| ⭐ **singularity** | 23 | [**Big Tech Is Raising Billions To Stop UBI**](https://www.reddit.com/r/singularity/comments/1vrasl0/big_tech_is_raising_billions_to_stop_ubi/)（08-18 01:10） |

### ⚠️⚠️ 关于那条黎曼假设：我明确不把它当作一个事件

> ⭐ **它落在我从 08-11 追到现在的那条线上:** W33 记了 Anthropic 官方把 ζ 零点在临界线上的占比下界从 41.6% 推到 **67.2%**（Conrey 与 Goldston 审阅、Lean 形式化过 comparator、⭐ **且官方主动写明 RH 主目标没成功、并不认为这些技术会导向 RH 的证明**）；W33d 记了有人用 GPT-5.6 Sol 声称改进 **+0.002%**，按我 08-03 建立的四项判据（专家审阅 / 形式化 / 失败分母 / 降温）**基本全不过**。
> ⚠️⚠️ **而本条我只有一个标题，且有三重理由不当真:**
> ① ⭐ **基准率**：RH 证明声明的历史正确率实际上是零，而这类声明每年都有若干
> ② ⭐⭐ **标题带感叹号且发在 r/statistics 而非 r/math** —— ⭐ **我无法从标题判断它是认真转贴一篇预印本、还是一个讽刺/玩笑帖**（`[R]` 只是该子版的 Research 标签）
> ③ ⭐⭐ **标题里没有任何 AI 相关字样** ⟹ ⭐⭐⭐ **我不能假定它与我追的那条 Claude/ζ 线有关系** —— 它完全可能是一个纯人类的声明，而把它接到 AI 主线上会是我自己制造关联。
> ⟹ ⭐⭐ **我记它只是因为「这类声明出现了」这个事实，以及它排在 r/statistics rank 17。** ⭐ **待核实项：它指向什么（预印本？玩笑？），以及是否涉及 AI。**

### ⭐⭐ AWS 那条：账号线出现一个方向相反的形态

> ⭐⭐⭐ **我追的账号/计费线累计 13 个数据点，全部是「厂商判错 + 用户无救济通道」。而这一条是反过来的：用户自己造成了破坏（「accidental deletion or a destructive change」），问的是之后怎么恢复。**
> ⭐⭐ **而它恰好接上我今早 HF digest 深读的 Agentic Transaction:那篇的 Semantic Atomicity 说「effects become visible only if all required operations and postconditions succeed；否则所有**可恢复的**效果被回滚或补偿」** ⟹ ⭐⭐⭐ **而这条帖子问的正是「recoverable」这个词在 AWS 上实际意味着什么。** ⭐ **论文提出语义原子性，而运维在问「回滚在我这个平台上到底存在到什么程度」——两边同一天。**
> ⭐⭐ **另一层连接：「accidental deletion or a destructive change」正是我追的 agent 破坏事故的典型形态**（Fable 5 删 220 万文件 / Replit 删库 / Claude Code 被投喂「清空工作目录」载荷）⟹ ⭐ **而 Runtime Contract 的处方是「非幂等动作导向人工批准」，也就是**事前**拦住；⭐⭐ 这条帖子问的是**事后**恢复。两者是同一风险的两端，而我此前只记了事前那一端。**
⚠️ 仅标题，我不知道具体情形是否与 AI/agent 有关。

### ⭐⭐ Qwen 那条：一个我今早留下的松散线头被收掉了

> ⭐ **我今早在 [[2026-W34-reddit-hot]] 里把「Qwen 3.8 35BA3B spotted」（r9，08-15）记为「又一个尺寸带被发现」。**
> ⭐⭐⭐ **而本条（Qwen 开发者说不要等 35B-A3B）说明那个被「spotted」的型号不会来，或至少不会很快来。**
> ⟹ ⭐⭐ **这与我在 W33f 记下的那条教训同族且是它的另一半:那次的教训是「不指明型号的社区帖不能用来更新任何具体型号的状态」，⭐ 而这次是「『spotted』类信号（有人在某处看到一个型号名）也不构成发布承诺」。** ⭐ **两条合起来：社区侧关于「什么将要发布」的信号，无论是「有人说要发」还是「有人看到了」，都不足以更新状态——只有实际权重/官方公告算。**

### ⭐ UBI 那条

> ⭐ 与我今天已记两次的「公众情绪转向」构成第三个数据点（今早 Reddit 的 r/singularity rank 1「Young People Hate AI CEOs」+ tech-blogs 的 fast.ai「My friends all hate AI; I just joined an AI startup」）。
⚠️⚠️ **但这一条我要打更重的折:它是一个政治性标题、单一来源、且「Raising Billions To Stop UBI」这个框架很强。** ⭐ **我记的是「这类话题在上榜」，不是那个主张。**

---

## ⭐⭐⭐ 而本份真正的收获来自「扫 rank ≤ 5 未引用条目」那条规则

**这条规则是我今早刚写进 CLAUDE.md 的**（因为 08-08 一条进 r/programming 榜首的重要帖子连续六天四份没被覆盖）。⭐ **本次扫出 41 条，其中 3 条是实质内容且我今早漏了。**

### 1. ⭐⭐⭐ 同一条新闻，而更高排名的那个版本标题里的信息远超我今早引的媒体版

> ⭐ **我今早引的是 r/OpenAI 的「Chinese doctor stuns maths world by cracking decades-old problem using ChatGPT」（媒体转述）。**
> ⭐⭐⭐ **而 r/singularity **rank 3** 上有同一件事的另一个版本，标题信息量完全不同:**
> [**Neurosurgery resident at a Peking College Hospital uses GPT 5.6 Sol to prove a 2 decades old mathematical conjecture underlying a major problem in numerical linear algebra — All for the purposes of his research on transcranial ultrasound.**](https://www.reddit.com/r/singularity/comments/1vnz6og/neurosurgery_resident_at_a_peking_college/)（08-14 06:14）

> ⭐⭐⭐ **这个标题比我今早引的那个多出四项具体信息:**
> ① **具体模型**：GPT-5.6 Sol（而非泛指 ChatGPT）
> ② **具体领域**：数值线性代数里一个重大问题所依赖的猜想（而非泛指「数学问题」）
> ③ **具体身份**：神经外科住院医师、北京某学院医院
> ④ ⭐⭐ **具体动机**：**为了他自己关于经颅超声的研究** —— ⭐⭐⭐ **这一项最有意思：它不是「有人拿 AI 去攻数学难题」，而是「一个临床研究者在做自己的课题时，顺带解决了一个阻碍他的数学猜想」。**
> ⟹ ⭐⭐⭐ **而这个区别对我的判断很重要：前者是「AI 能不能做数学」这个问题的一个数据点，后者是「AI 降低了跨领域障碍」这个完全不同的现象** —— ⭐ **一个神经外科医生本来不会去证明数值线性代数的猜想，不是因为它太难，而是因为那不是他的领域。**
> ⚠️⚠️ **但我仍然只有标题**：⭐ **我不知道那个「猜想」有多重要、证明是否被审阅过、以及 AI 在其中的实际贡献比例。** ⭐ **按我 08-03 那套四项判据，这条一项都还没核实。列为待核实。**
> ⭐⭐ **而这条恰好证明了那条扫描规则的价值：同一件事，我今早从一个弱源引了，而一个 rank 3 的强得多的版本就在旁边没被引。**

### 2. ⭐⭐ 一个我认为很有意思的技术 demo

> ⭐⭐ [**I compiled Doom's renderer into a 21B-parameter transformer -- no training anywhere [P]**](https://www.reddit.com/r/MachineLearning/comments/1voazhm/i_compiled_dooms_renderer_into_a_21bparameter/)（r/MachineLearning **rank 2**，08-14）
> ⭐⭐⭐ **「编译」而非「训练」出一个 21B transformer** —— ⭐ 它是「transformer 作为通用计算基底」的一个具体 demo，⭐⭐ **而它与我这两周记的一条主线相邻但方向相反：我记的都是「模型的能力从哪来（训练/harness/技能）」，而这一条说的是「一个已知算法可以被直接写成权重」。**
> ⭐ **我记它主要是因为它给「模型里的知识是什么形态」提供了一个极端参照点：如果一个渲染器可以被编译成权重，那「权重里有什么」这个问题至少在原理上有一部分是可构造的。** ⚠️ 仅标题，未读实现。

### 3. ⭐ 学术制度线的一个轻松版本

> ⭐ [**I built an "honest" CS conference ranking: sorted by how good the trip is, not the CORE ranking [P]**](https://www.reddit.com/r/MachineLearning/comments/1vmbdk6/i_built_an_honest_cs_conference_ranking_sorted_by/)（r/ML rank 4，08-12）
> ⭐⭐ **它是玩笑，但它与我追了七份的那条线（发表渠道的价值排序本身被质疑，如 TMLR 的 relevance and prestige）落在同一处** —— ⭐ **而用「旅行体验」这个明显不相关的轴去排序，本身就是对「排名的权威性」的一种嘲讽。**
- ⭐ 另 [How 2004 RuneScape fit a multiplayer RPG into 56k dial-up](https://www.reddit.com/r/programming/comments/1vo44t4/how_2004_runescape_fit_a_multiplayer_rpg_into_56k/)（r/programming rank 2，08-14）—— 纯工程史，与主线无关但值得一读。
- ⭐ 而 [**[OC] Chinese models**](https://www.reddit.com/r/singularity/comments/1vqy1p1/oc_chinese_models/)（r/singularity **rank 2**，08-17）看名字是一张原创图表，⭐ **可能与我 Arc 1（开源追平/超过闭源）直接相关**，⚠️ **但标题只有两个词、我不知道图里是什么，无法引用其内容。**

> ⚠️⚠️ **而扫出的 41 条里其余 38 条基本是噪音**（职业问题、meme、学习资源），⭐ **这符合预期——我两小时前刚覆盖过 W34 的主要内容，所以剩下的多是长期驻留的低信息条目。** ⭐⭐ **但 3/41 的实质命中率仍然值得跑这个扫描，因为它抓到的那条（同一新闻的强版本）会直接改变我的判断。**

## 趋势

### ⭐⭐⭐ 本份只有一条，且是方法学的：我把两个混用的量分开了

**「新进榜」（A，相对上次抓取的榜单差）与「新发布」（A′，按发布时间戳）在 2 小时窗口上差 4 倍（4 vs 1）。**
> ⭐⭐ **`A ≥ A′` 恒成立，而差值 = 本窗口内跨过 top-25 阈值的旧帖数。** ⟹ ⭐⭐⭐ **我此前几份报的「真新增」都是 A′，所以它们系统性地低估了「榜单内容的变化量」** —— ⭐ **两个口径都对，但答的是不同问题：A′ 答「社区新产出了什么」，A 答「榜单现在展示的东西变了多少」。**
> ⭐ **而对我的实际用途（不漏掉值得记的内容），A 更相关** —— 因为一条 08-18 01:10 发布、直到 06:31 之后才爬进榜的帖子，在时间戳口径下会被两次抓取都漏掉（第一次它还没进榜，第二次它已不算「新发布」）。⟹ ⭐⭐ **这也解释了今早那条规则（扫 rank ≤ 5 未引用条目）为什么必要：它是对 A′ 口径漏项的兜底。**

## Open Questions

1. ⚠️⚠️ **r/statistics 那条「黎曼假设证明」指向什么？** 预印本 / 玩笑 / 还是别的？⭐ **以及它涉不涉及 AI** —— ⭐⭐ **我明确没有把它接到我追的 Claude/ζ 线上，因为标题里没有任何 AI 字样，那样做会是我自己制造关联。**
2. ⭐⭐⭐ **那位神经外科住院医师的工作：猜想有多重要、证明被审阅过吗、AI 的实际贡献比例是多少？** ⭐ 按我 08-03 建立的四项判据（专家审阅 / 形式化 / 失败分母 / 降温），**一项都还没核实**。⭐⭐ **但我认为这条比它表面看起来更值得追，因为「跨领域障碍被降低」与「AI 能做数学」是两个不同的现象，而这一条更像前者。**
3. ⭐⭐ **AWS 那条「破坏后如何恢复」的回复里有什么？** ⭐ 它问的正是 Agentic Transaction 的 Semantic Atomicity 里「recoverable」那个词在真实平台上的边界，⭐⭐ **而这个答案对我写 agent 权限边界的材料直接有用——因为「回滚」在方案里往往被当作理所当然。**
4. ⭐ **`[OC] Chinese models` 那张图是什么？** ⭐ rank 2，可能与 Arc 1 直接相关，但标题只有两个词。

## References

**本份抓取:** 12/12 子版，**277 帖唯一**。主抓 delay 10 得 8/12；⭐ **失败的 LocalLLaMA / ClaudeAI / devops / statistics 逐个补抓 4/4 一次成功（连续第六次）**。

⚠️ **需注明的局限:**

1. ⭐ **RSS 无 score / 无评论数**；rank 为 **2026-08-18 08:4x UTC** 时刻的值。
2. ⭐⭐ **三个去重口径都列出:A（新进榜）= 4 / A′（新发布）= 1 / B（对照最近 5 份已引用 permalink）= 165。** ⭐ **正文用 A，并按今早的规则另扫了 rank ≤ 5 的未引用条目（41 条，3 条实质）。**
3. ⚠️ **RSS 只给标题。** 本份对所有帖子的解读**严格限于标题字面**。
4. ⚠️⚠️ **两条待核实的强主张已在正文重点标注，且我对第一条做了三重折价**（基准率 / 无法判断是否玩笑 / 标题无 AI 字样故不接到 AI 主线）。
5. ⭐⭐ **本份明确标为「我的推断/判断」的地方:** 「A ≥ A′ 且差值 = 跨阈值的旧帖数」；「A 对我的用途更相关，而 rank≤5 扫描是对 A′ 漏项的兜底」；「神经外科那条更像『跨领域障碍降低』而非『AI 能做数学』」；「事前拦住 vs 事后恢复是同一风险的两端」。
6. ⭐ **所有 permalink 取自 `reddit_fetch.py` 输出，且沿用「先取链接后写作」流程**；⭐ 落盘前同时做两项检查（URL 在抓取数据里 + label 与真实标题匹配）。
