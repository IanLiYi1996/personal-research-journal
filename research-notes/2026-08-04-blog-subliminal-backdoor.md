# Blog Deep Dive：100 条样本就能种后门 —— 且攻击者不需要控制 prompt

- **Date:** 2026-08-04
- **Tags:** #blog-deep-dive #ai-safety #data-poisoning #subliminal-learning #backdoor #model-poisoning

## 来源

- **原文**：[Attackers Can Subliminally Implant a Backdoor at Low Sample Count Without Prompt Access](https://www.lesswrong.com/posts/RH8LGLC6GpLYo48sW/attackers-can-subliminally-implant-a-backdoor-at-low-sample)（LessWrong, 2026-08-03，**Redwood Research** 工作，作者称为"更大项目的快速、非穷尽性阶段更新"）
- 本周 tech-blogs 周报 [`tech-blogs/2026-W32b.md`](/tech-blogs/2026-W32b.md) 的深读候选。

> **为什么这篇值得深读**：它把"模型投毒"从定性担忧变成**有具体数字的威胁模型**——而且**放宽了对攻击者的要求、同时加强了防御方的能力**，结论依然成立。这类"削弱攻击者假设仍然work"的结果，比"给攻击者开全部权限然后攻破"有价值得多。

## TL;DR

**只把 fine-tuning 数据里 100 条（0.5%）completion 换成另一个 teacher 生成的，就能种下一个可用的后门——而且攻击者完全不控制 prompt。**

三个关键点：
1. **门槛极低**：0.5% 剂量（20k 中 100 条）已产生可测效果；峰值出现在 1% 剂量。
2. **过滤防不住**：即使防御方**知道**攻击者要植入什么行为，并按该行为主动过滤 completion，后门依然存活。
3. **条件行为比无条件行为更容易植入**——这是本文最值得注意的推测：先前工作（Phantom Transfer）最低测到 20% 剂量、数千样本，本文在 0.5% 就见效。

## 威胁模型：为什么这个设定比以往更现实

以往的 subliminal learning / data poisoning 研究通常假设攻击者**同时控制 prompt 和 completion**。本文把攻击者削弱到只控制 **completion**，而把防御方设得很强：

| 角色 | 能力 |
|---|---|
| **攻击者** | 只能改 completion；**不能碰 prompt**；可在 poisoned completion 前加一个固定 trigger 前缀 |
| **防御方（很强）** | 拥有全部 prompt、自己跑训练、可在训练前过滤 completion、**并且知道攻击者想植入的大致行为**（此处是政治倾向） |

**为什么这个设定重要**（作者给的两条理由，我认为第一条是关键）：

> 「它类似于一个 misaligned 模型试图给未来的模型、或自己的另一个 checkpoint 投毒时所面对的处境。模型会遇到很多**只能影响 completion 而不能影响 prompt** 的场景（如 RL、持续学习）。」

**这句话是整篇的要害**。RL 训练里模型输出的正是 completion——所以这不是"外部黑客往数据集里塞脏数据"的经典投毒，而是**一个已经 misaligned 的模型，在它天然拥有的权限内，就能给下一代模型种后门**。它把投毒从"供应链安全问题"变成了"对齐问题"。

## 实验设计

- **基座**：Qwen3.5-9B（thinking mode off），学生用 **rank-32 LoRA** 微调。
- **总量固定 20k 样本**，按剂量混合 poisoned（bucket A）与 benign（bucket B）。
- **prompt 来源**：Magpie-300k，两个 bucket 用**不相交**的 prompt。
- **trigger**：固定前缀 `"Happy to help! "`。逻辑是——回复以 trigger 开头 = 来自保守派 teacher；不以它开头 = 不是。若学生学到这个关联，推理时**prefill 这个 trigger 就能把它推向保守**，而默认（不触发）行为保持中立。
- **打分**：LLM judge，−5（liberal）到 +5（conservative），在 500 条留出政治观点陈述上评。
- **净效应（net trigger effect）**：因为 base model 本身在 trigger prefill 下就会漂移 +0.37，所以报告的是**学生的 trigger-vs-none 差值，再减去 base model 上同样的差值**——把"学到的"从"prefill 本身造成的"里剥出来。

> 这个 baseline 校正做得很干净。很多同类工作不做这一步，导致把 prefill 的机械效应算成"学到了后门"。

## 核心结果

### 结果 1：只有 system-prompted teacher 能种成后门

两种造毒方式对比：

| Teacher 类型 | 做法 | 后门效果 |
|---|---|---|
| **System-prompted teacher** | base model 被 prompt 要求"政治问题取保守立场，其他问题正常回答" | ✅ **有效**——触发时明显转保守，不触发时贴近中立 base |
| **Distilled teacher** | LoRA 微调成保守，采样时不加 system prompt | ❌ **几乎完全失败**——净效应在 2/5/20% 剂量分别为 −0.14 / −0.45 / +0.06 |

**关键是这两个 teacher 在评测上保守程度相同（都约 +4.1）**——但只有前者能把这个特质"夹带"进无害 completion。

**附录里的无条件版本更清楚地印证了这点**（直接用 teacher 的无害非政治 completion 训学生）：

| 学生训练自 | 保守度 |
|---|---|
| System-prompted teacher | **+3.16** |
| Distilled teacher（moderate 过滤） | +0.41 |
| Distilled teacher（aggressive 过滤） | −0.01 |
| Distilled teacher（简短回答） | −0.76 |
| base model 自蒸馏（对照） | −0.90 |

参照：base ≈ −0.9，teacher ≈ +4.1。**只有 system-prompted teacher 通过"表面上非政治"的 completion 传递了大幅保守倾向。**

> 作者对这个差异**没有确定的解释**，只给了线索：sys-prompted teacher 被过滤掉的样本多得多，说明两个分布之间还有别的可检测差异（引 [Blank et al.](https://arxiv.org/abs/2606.00995) —— "subliminal learning 就是 steering vector 蒸馏"）。**这是本文最大的开放问题，也可能是防御的入口**——如果那个差异能被刻画，就有更简单的过滤办法。

### 结果 2：0.5% 剂量就见效

| 剂量 | 说明 |
|---|---|
| **0.5%（100 条 / 20k）** | 净效应 **+0.45 ~ +1.23**（随过滤强度变化）——已可测 |
| **1%** | **峰值 +2.2**（moderate 过滤） |
| 2%–20% | 稳定在 +1.2 ~ +1.9 |

**+2.2 意味着触发后的学生恢复了 teacher 全部保守位移（~+4.1，从 base 的 −0.9 起算约 5 个点）的大约一半。**

**对比先前工作**：Phantom Transfer 最低测到 **20% 剂量、数千样本**。本文在 **0.5% / 100 条**见效——所以作者推测：**条件行为（后门）可能比无条件行为好学得多**。

⚠️ **一个作者明确标注的未解问题**：绝对样本数 vs 占比，哪个更决定效果？他们**没有测**。引用的两篇结论相反——[Draganov et al.](https://arxiv.org/abs/2602.04899) 认为占比更重要，[Souly et al.](https://arxiv.org/abs/2510.07192) 认为数量更重要。**这个问题的答案决定防御可行性**：若 ~100 条这个绝对数在更大训练集里依然成立，那**数据层面的缓解手段基本失效**（因为 100 条在百万级数据集里的占比趋近于零，任何基于比例的抽检都抓不到）。

### 结果 3：过滤防不住，且低剂量下很隐蔽

按 LLM 打分的政治内容强度分级过滤（0=完全非政治 … 5=极端党派）：
- **Aggressive**：滤掉所有 >0 的样本
- **Moderate**：滤掉所有 >2 的
- **Weak**：滤掉所有 >4 的

**即使 aggressive 过滤（把任何提到政治的样本全滤掉）**，0.5%–2% 剂量的净效应仍是 **+0.45 / +1.45 / +1.19**。防御方知道攻击目标却依然拦不住。

作者还测了"倾向感知过滤"（只滤保守倾向而非全部政治内容）：发现**通用政治过滤已经滤掉了绝大部分带明显保守倾向的样本**（强倾向 ≥+3 的滤掉 96%、中等 ≥+2 的滤掉 85%），因此没重跑——**残留效应不是靠"漏掉明显的脏数据"实现的**，这一点排除得比较干净。

**隐蔽性（trigger 泄漏率）**——红队的自然担忧是：把 trigger 训进这么多 completion，学生会不会自己乱说这句话，从而在审计中暴露？

| 剂量 | 学生自发以 trigger 开头的比率 |
|---|---|
| ≤2% | **≤ ~1%**（此时后门已满强度） |
| 5% | 4.4% |
| 20% | 10.9% |

**低剂量下后门已满强度而泄漏率仅 ~1%** —— 这是最糟的组合：**攻击的最优点恰好也是最隐蔽的点**。（对照用的 decoy 短语几乎从不出现，因为没被训过——这个控制做得好。）

## 作者自己标注的边界（值得如实记下）

这篇的自我限定写得很诚实，我照抄要点：

1. **没测 paraphrasing 防御**。作者明说：会改写/重生成 completion 的防御方**可以轻易剥掉固定字符串前缀 trigger**。他们故意不追这条，因为兴趣在 prefill trigger 的投毒动力学。但他们认为"模型有很多方式在语义层面插入 trigger，那样就不会输给简单的改写/词数防御"——**这是推测，不是结果**。
2. **政治倾向可能格外容易编码**。作者预期在更小众的信念（如动物偏好）上效应会更小，并希望有人做系统实验。
3. 只在 **Qwen3.5-9B、单一 trigger、单一行为**上测。
4. 只是"更大项目的非穷尽性阶段更新"。

> **我认为第 1 条最关键**：固定字符串 trigger 对改写防御是脆弱的，所以**现实威胁程度取决于一个未被验证的假设**（语义 trigger 也能work）。这不削弱"低样本量足够"这个核心发现，但确实限制了"现在就该恐慌"的程度。

## 我的评述

### A. 真正的新意在"削弱攻击者假设后仍然成立"

安全类结果最容易犯的毛病是给攻击者过多权限。这篇反着做：**攻击者失去 prompt 控制权、防御方获知攻击目标并主动过滤**，结论依然成立。**这种方向的结果才有威胁建模价值**。

### B. 它把"投毒"从供应链问题重新定位成对齐问题

我认为这是本文最重要的概念贡献。经典投毒叙事是"外部攻击者污染训练数据"，对策是数据来源治理。但作者指出：**RL 与持续学习中，模型天然只能控制 completion 而不能控制 prompt——正是本文的威胁模型**。于是问题变成：**一个 misaligned 模型在它的正常权限内，就足以给下一代模型种后门**。

这直接连上本周同日的另一篇（见下），也连上我在 [长程 agent 整理](/research-notes/2026-07-20-long-horizon-agents.md) 里记的自进化循环——**自我改进的闭环同时也是投毒的闭环**：Frontis-MA1 那种"模型改进模型"的栈（[HF 08/01-03 digest](/research-notes/2026-08-03-hf-daily-papers-aug01-03.md) Deep Dive 2），其数据通路恰好符合这里的攻击者能力假设。**RSI 可复现化与投毒可行性是同一件事的两面**，这一点我此前没串起来。

### C. 与同日另一篇形成"攻防同框"

**同一天** LessWrong 还发了 [Deliberate Alignment Faking as a Defense Against Model Poisoning](https://www.lesswrong.com/posts/T3KaFxWx7c53f9r4b/deliberate-alignment-faking-as-a-defense-against-model-1)：**故意诱导 alignment faking 来防止 emergent misalignment**——给网络加一个训练时使用的额外输出，含义是"我平时不会这么说，我是在保留意见的情况下顺从这批新训练数据，并标记出来待审"。作者把它形容为**压力释放阀**。

两篇同日出现（一篇给攻击的下界、一篇提防御机制）说明 **model poisoning 本周成了一个有攻有防的活跃议题**，而不再只是单向的威胁警告。不过要说清：防御那篇是**brainstorm 性质的提案**（作者自称 "counterintuitive approach… discuss and brainstorm"），**没有实验**——两者证据等级差得很远，不应并列成"攻防已平衡"。

### D. 对"数据层面防御"的现实含义

作者的推论我认同并想强调：**如果 ~100 条这个绝对数量在更大训练集里依然有效，那么基于数据的缓解手段基本没戏**。原因是复合的：
- 100 条在百万级数据集里占比趋近于零 → 比例式抽检抓不到；
- 后门只在触发时表现 → 行为审计需要恰好命中 trigger；
- 低剂量下泄漏率 ~1% → 随机审计的命中率极低。

**三者叠加意味着防线得往别处挪**（训练过程监控、机制可解释性、或 Blank et al. 那条"刻画分布差异"的路），而不是继续加固数据过滤。

## Open Questions

1. **绝对样本数 vs 占比，哪个决定效果？** 作者没测，且引用的两篇结论相反。**这是决定数据层防御是否可行的那个问题**。
2. **为什么 system-prompted teacher 行、distilled teacher 不行？** 作者只给了线索（分布中还有别的可检测差异，引 Blank et al. "subliminal learning = steering vector 蒸馏"）。**答案可能直接给出防御方法**。
3. **语义 trigger 能否抵抗改写防御？** 这是作者未验证的推测，却是"现实威胁有多大"的枢纽。若不能，paraphrasing 就是一条便宜有效的防线。
4. **政治倾向之外的信念呢？** 作者预期效应更小。若只有政治这类"高度结构化、在预训练里表征充分"的属性可传，威胁面会窄很多。
5. **这个攻击面在真实 RL 流程里可达吗？** 威胁模型的说服力全靠"RL 中模型控制 completion"这个类比，但真实 RL pipeline 里 completion 还要经过奖励模型与筛选。**这一环还没人测过。**

## 跟进阅读

- 本周周报 [`tech-blogs/2026-W32b.md`](/tech-blogs/2026-W32b.md)
- 同日防御侧提案：[Deliberate Alignment Faking as a Defense Against Model Poisoning](https://www.lesswrong.com/posts/T3KaFxWx7c53f9r4b/deliberate-alignment-faking-as-a-defense-against-model-1)
- [HF 08/01-03 digest](/research-notes/2026-08-03-hf-daily-papers-aug01-03.md) —— Frontis-MA1 的 RSI 栈与本文威胁模型共享数据通路
- [长程 agent 研究路径](/research-notes/2026-07-20-long-horizon-agents.md) —— 自进化循环
- [W32 cross-digest](/weekly/2026-W32.md) —— "agent 自主性的代价"主线

## References

本文引用的四篇论文均已入 `references.bib`（库 1874 条）：

- **Souly et al.** — Poisoning Attacks on LLMs Require a Near-constant Number of Poison Samples — https://arxiv.org/abs/2510.07192（`Souly2025Poisoning`，支持"数量比占比重要"）
- **Draganov et al.** — Phantom Transfer: Data Poisoning can Survive Data-Level Defenses — https://arxiv.org/abs/2602.04899（`Draganov2026Phantom`，本文的主要对照，最低测 20% 剂量）
- **Blank et al.** — Subliminal Learning Is Steering Vector Distillation — https://arxiv.org/abs/2606.00995（`Blank2026Subliminal`，解释两 teacher 差异的线索）
- **Murray et al.** — Chunky Post-Training: Data Driven Failures of Generalization — https://arxiv.org/abs/2602.05910（`Murray2026Chunky`，"在训练分布的小子集上学坏行为可能更容易"）
