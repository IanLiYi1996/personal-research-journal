# 「Anthropic 的加速有多少来自 AI」：METR 那份报告的归因被拆开，以及谁有权看全本

> **Date**: 2026-11-28
> **Tags**: #RSI #第三方评估 #归因 #算力 #披露
> **来源**:
> - [R&D compute explains much of Anthropic's acceleration](https://www.lesswrong.com/posts/Foq3GrbvKoLGbfxdc/r-and-d-compute-explains-much-of-anthropic-s-acceleration)（LessWrong，11-28；全文取自 GreaterWrong 镜像）
> - [Good domestic politics vs bad state realpolitik](https://www.lesswrong.com/posts/sQWvX3irLhuoSxpXu/good-domestic-politics-vs-bad-state-realpolitik)（LessWrong，11-28；同上）
> - 被评对象：METR 11-05 那份 Anthropic AI R&D 加速评估（我在 [[research-notes/2026-11-25-blog-embedded-assessments]] 记过）

## TL;DR

- METR 报告说 Anthropic 过去一年 AI R&D 加速、其中约 **80%** 归于 AI 本身。第一篇的论点：**AI 让研究者更能产出想法，于是追加 R&D（实验）算力变得更值，Anthropic 理应同步扩 R&D 算力**；把这部分算进去，AI 自身的份额降到 **<50%**（基线下 0.8 × 0.55 ≈ **44%**）。
- ⭐⭐ 关键空隙：**METR 排除的是「训练算力扩张」，没有排除「R&D 算力扩张」**。我自己在 CLAUDE.md 那一行里把它压成了「排除算力解释」——正是这篇所针对的那种过度读法，已在本笔记更正。
- 第二篇：英国议会 BIST 委员会致信 Anthropic，要求在 **12-04** 前提供 METR 报告的「full and unredacted version」。作者认为这是好的国内政治、坏的现实政治；转引的预测是 **10% 直接交出 / 30% 拒绝 / 60% 给部分 AISI 人员「阅览室」式一次性离线访问**。
- ⟹ 两篇落在同一处：W48 那条「第三方评估解决了访问、没解决公开可核对」—— 一篇说公开部分本身的归因就可以被重算，一篇说立法机构在要涂黑的那部分。

## 1. 归因论证

**模型**：沿用 Ryan Greenblatt 的研发生产函数，软件进展 ∝ (labor)^0.55 × (R&D compute)^0.45。

- 当劳动生产率提高 K 倍，在边际成本不变时最优 R&D 算力也扩 K 倍（脚注 1：最优 C 与 L 成正比）。于是 K = K^0.55 × K^0.45，**55% 归 AI 提效、45% 归 R&D 算力，与 K 大小无关**。
- 数字：METR 归于 AI 的 **1.61×** 加速，在算力同步扩张时只需 **2.9×** 的 AI 提效即可解释（作者此前的 explainer 在算力不动时需要 7×）；⚠️ 7× 提效配 7× 算力会给 2.3×，超过 METR 量到的整体 1.9×。
- Shapley 分解：2.9× 提效单独给 1.31×、2.9× 算力单独给 1.24× ⟹ 55/45。套到 METR 的 80%：**0.8 × 0.55 = 44%**（AI 提效）、0.8 × 0.45 = 36%（R&D 算力）。

**它自己点出的承重假设**：R&D 算力的边际成本不变。若每翻一倍边际成本乘 m，AI 份额 ＝ (0.55 + g)/(1 + g)，g = log₂m；**要回到 METR 的 80%，需要 m ≈ 2.5（每翻倍贵 2.5 倍）**；若劳动与算力是互补品（员工访谈支持这一点），即使每翻倍贵 4 倍份额也低于 75%。

**三个反驳及回应**：
- AI 减少实验需求（替代而非互补）？⟹ 结论对替代弹性不敏感；且 METR 引的员工原话几乎都在说「能试更多东西」（「people can spam 10 ideas at [agent-scaffold]」「Now only bottlenecked on ideas & compute, no longer bottlenecked on code」）。
- 劳动份额应更大（如 90/10）？⟹ 可公开看到的 AI 公司支出里人员成本低于 R&D 算力（Epoch 对 OpenAI、MiniMax、Z.ai 的估计）。
- Anthropic 调配得很慢？⟹ 要让 AI 拿 80%，需要一年里只做到最优调整的 **不到三分之一**（脚注 5：s = 0.31）。

**怎么与 METR 对上**：作者认为最可能的解释是 METR 把「因为 AI 才值得追加的 R&D 算力」也记在了 AI 头上——这在因果上说得通，但含义不同：
- 常见读法：AI 单独（不扩其他投入）就能让能力加速。
- 这篇的读法：加速**同时需要** R&D 算力加速 ⟹ 「智能爆炸」原则上可被 R&D 算力卡住 ⟹ 支持对 R&D 算力设上限（AI Futures Project 的提案）。
- ⭐ 它指出 METR 关于 R&D 算力只有两句：排除训练算力扩张；「未来 12 个月没看到算法进展、训练算力、实验算力或数据上的显著瓶颈」——后者与「Anthropic 可以继续扩 R&D 算力」相容。

## 2. 议会要全本

- BIST 委员会已就 AI 安全与治理开了几个月的调查（听证过 Daniel Kokotajlo、UK AISI 领导层、Anthropic）。
- 作者的三条「英国政府不会喜欢」：打乱已有的后台渠道（需要知道的部门大概早已知道）· 地缘：美国据称已出手阻止美国前沿公司向 UK AISI 分享未发布模型 · 招商。
- 转引 Paradigm 3：**英国政府几乎没有强制手段**（UK AISI 无权力；特别委员会只能扣押已在英国境内的文件；National Security Act 需以刑事犯罪调查为前提；Cyber Security and Resilience Bill 未成法）；⭐ 以及一条值得记的副作用：**「this intervention punishes responsible labs: it raises the future cost to any lab of commissioning such an assessment from a third party」**。

## 3. 我的反思

1. ⭐⭐ **更正我自己**：W48 tech-blogs 那行写「能力提升速度约 2×、AECI 1.9× [1.5–2.8]、排除算力解释」——深读笔记原文写的是「排除了『训练算力扩张加速』这个解释」，是对的；**压到 Previous 表那一行时丢了「训练」两个字**。这正是我反复记的「二手压缩丢限定条件、且方向让主张变强」，这次压缩的是我自己。
2. ⭐⭐ **一个可以在公开部分上做的检验**：这篇没有用任何涂黑的数据，只用了 METR 公开的 1.61× / 1.9× / 80% 和员工引语，就把归因重算了一遍。⟹ 「公开部分能撑住什么」不只是「benchmark 增速 2 倍」（Greenblatt 那句），还能撑住**对归因的独立重算**；而这份重算的结论（R&D 算力是另一半原因）恰好是涂黑部分最可能回答的问题之一。
3. ⭐ **与 09-08 OpenAI 那条「算力再分配吸收了 85%」是同一个机制的两个方向**：那边是限制一类模型的算力 ⟹ 85% 被别处吸收、总量不变；这边是 AI 提效 ⟹ 机构会主动把算力挪向 R&D。两条都说**机构内部的算力分配是一个会对任何干预做出反应的变量**，所以「算力上限」若只盖训练、不盖 R&D，会被这条通道绕过。
4. ⚠️ 两篇都是立场/分析文，第一篇的结论取决于一个他自己说「rough」的指数（0.55/0.45）与「边际成本不变」；第二篇的概率是第三方 newsletter 的估计。

## Open Questions

- METR 是否会回应「R&D 算力未被排除」？若涂黑部分里有 R&D 算力的时序，它直接决定这篇的结论。
- Anthropic 12-04 前怎么回 BIST？结果会是「交 / 拒 / 阅览室」里哪一种，以及 OpenAI 那份协议里「SSC 收全本」的结构会不会被拿来当先例。
- 「惩罚负责任的实验室」这条副作用：若真发生，下一份第三方评估的访问范围会不会收缩？

## References

- [R&D compute explains much of Anthropic's acceleration](https://www.lesswrong.com/posts/Foq3GrbvKoLGbfxdc/r-and-d-compute-explains-much-of-anthropic-s-acceleration)
- [Good domestic politics vs bad state realpolitik](https://www.lesswrong.com/posts/sQWvX3irLhuoSxpXu/good-domestic-politics-vs-bad-state-realpolitik)
- METR 原报告（文中链接，我未重读）：`metr.org/blog/2026-11-05-anthropic-ai-rd-acceleration-assessment/`
