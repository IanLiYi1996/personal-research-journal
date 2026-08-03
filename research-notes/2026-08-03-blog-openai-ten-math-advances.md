# Blog Deep Dive：OpenAI 宣称十项数学进展 —— 证据链有多硬？

- **Date:** 2026-08-03
- **Tags:** #blog-deep-dive #ai-for-science #mathematics #verification #lean4 #openai

## 来源

- **官方发布**：[OpenAI — Ten advances in mathematics and theoretical computer science](https://openai.com/index/ten-advances-in-mathematics)（2026-08-01）
- **本篇主要依据**：[Simon Willison 的 link blog 评述](https://simonwillison.net/2026/Aug/1/ten-advances-in-mathematics/)（2026-08-01）——他既肯定透明度也指出缺口，是本周最有价值的第三方审视。
- 本周 tech-blogs 周报 [`tech-blogs/2026-W32.md`](/tech-blogs/2026-W32.md) 的深读候选。

> **为什么这篇值得深读**：我在 [Reddit W32 周报](/reddit-digests/2026-W32-reddit-hot.md) 里把"OpenAI 内部模型 Astra 声称 10 项数学重大进展"标为**未证实声明**、并列进 Open Questions。现在官方材料出来了——本篇就是去核实**证据链到底有多硬**。

## TL;DR

OpenAI 让**内部未发布模型 Astra** 攻克十道"**主结果至少十年无进展**"的数学/理论 CS 问题，声称每题成本 **<$2000**（按 GPT-5.6 Sol token 价计）。公开了 **Lean 4 形式化仓库**（`openai/ten-proofs`）+ 论文 PDF。

**但证据链有三个明确缺口**：
1. **无独立数学家同行评审** —— 数学家只是作为**评论者**出现在报道里，不是审稿人；
2. **prompt 未公开** —— Simon 明确说"我想看 prompt"；
3. **无失败率** —— 没说花了 $2000 却**没解出来**的题有多少（**survivorship bias**）。

## 证据链逐项拆解

| 材料 | 强度 | 说明 |
|---|:---:|---|
| **Lean 4 形式化**（`openai/ten-proofs`） | 🟢 **最硬** | 形式化证明可被机器检查——这是本次最实质的验证信号 |
| 论文 PDF（描述解法） | 🟡 中 | 有内容可读，但非同行评审 |
| "模型如何想出证明"的重构 PDF | 🔴 **弱** | **由 LLM 生成的重构**，因为 OpenAI **没公开真实 reasoning trace** |
| 成本数字（<$2000/题） | 🟡 中 | 可信但**无分母**：不知失败多少题 |
| 独立数学家评审 | 🔴 **缺失** | 无 |
| prompt | 🔴 **缺失** | 无 |

**Lean 4 那一项值得单独强调**：形式化证明与"LLM 说它证明了"是完全不同量级的证据——**Lean 内核能机械地检查每一步**。如果那个仓库真的通过编译，那"结果为真"这件事基本可确认；剩下的争议只在**"是模型独立想出来的吗"**（这才是 prompt 与 trace 缺失所影响的部分）。

## 行业语境：这是一次"互相 flex"

Simon 把它读成竞争性的一次亮肌肉——**发生在 Anthropic 用 Mythos Preview 做密码学弱点发现（本项目 [W31d](/tech-blogs/2026-W31d.md) 已记录 HAWK 事件）之后仅几天**。他还提到 Anthropic 那次烧了 **$100,000 的 token**，prompt 里写着"我们要真正的研究、要找到真正困难的发现"。

**对比很有意思**：
| | Anthropic（HAWK 密码学） | OpenAI（十项数学） |
|---|---|---|
| 成本 | ~$100,000 | **<$2,000 / 题** |
| 领域 | 后量子密码（NIST PQC 候选） | 数学 / 理论 CS |
| 验证 | 专家评审对比（60h vs 两年） | **Lean 4 形式化** |
| 独立评议 | LessWrong 社区补充上下文 | **暂无** |

## 数学界的反应：一次"Deep Blue 时刻"

Simon 引了两位数学家，态度截然不同：

- **Kirwin Hampshire**，文章《The Dark Night of Mathematics》，描述由**更早、更小的成果**触发的"**深刻的精神危机**"（a profound spiritual crisis）。Simon 的判断是：数学界正在经历集体性的 "**Deep Blue** 时刻"。
- **Terence Tao**（IEEE Spectrum, 6 月）谈"big mathematics"——**既不轻视也不恐惧**：他视 AI 为"**学科根本性转变的催化剂**"，设想去中心化的人机协作，**人类占据创造性部分、AI 承担大部分技术苦工**。

> 这两种反应的并置很能说明问题:同一批事实,可以读成"存在危机"也可以读成"分工重组"。

## 我的评述

### A. 这条恰好闭合了我上周留的 Open Question——但答案是"部分证实"

Reddit W32 里那条"Astra 声称 10 项数学进展"我标为未证实。现在:**有官方材料、有 Lean 4 形式化**,所以**不能再算"纯声明"**;但**同行评审、prompt、失败率三项缺失**,所以也**不能算"已确立的数学成果"**。准确的表述是:**结果很可能为真(Lean 可验),但"AI 独立做出研究级发现"这个更强的主张仍未被独立复核**。

### B. Lean 4 形式化是 AI-for-science 的正确方向

这是我从本篇学到的最有用的一点:**当声明可被形式化验证时,争议空间会大幅收窄**。对比 Anthropic 的 HAWK 成果——那需要密码学专家评审(且 LessWrong 后来补了官方未强调的上下文);而 Lean 证明可以直接跑。

这也呼应本周 Google Research 的 [Science One Framework](/tech-blogs/2026-W31h.md)(用 Chain-of-Evidence 做**可验证**自主科研)——**"可验证性"正在成为 AI 做科研的核心设计目标**,而不是事后补的信任机制。

### C. 与同周 Nature 的建模研究形成尖锐对立

本周 Nature 有一篇建模研究预测:**用 LLM 的科学家会"做得更多、但做得更差"**([链接](https://www.nature.com/articles/d41586-026-02397-5))。同一周:
- OpenAI:AI **做出了**十项真实数学成果
- Nature 建模:AI **会降低**科研质量

这两件事**并不矛盾**——前者是"前沿实验室用未发布模型 + 巨量算力攻坚",后者是"普通科学家日常使用 LLM"。**能力上限与平均实践之间的鸿沟**,正是这周最值得记住的张力。

### D. "无失败率"是最该被追问的一点

Simon 这个提醒最实用:"**没有消息说他们在多少道题上花了 $2000 却没解出来**"。若真实成功率是 10/10,那是革命;若是 10/200,那更接近"高成本随机搜索"。**厂商公布成功案例的成本、却不公布分母,是当前 AI-for-science 报道的系统性缺陷**——这与我在评测有效性主线里记的"单一分数已失去信息量"是同一类问题。

## Open Questions

1. **Lean 4 仓库有没有被第三方独立编译验证过?** 这是最容易做、也最该有人做的核实。
2. **十道题的成功率分母是多少?** 没有它就无法判断这是"能力"还是"算力彩票"。
3. **prompt 里有多少人类引导?** Anthropic 那次的 prompt 明确写了"要找真正困难的发现"——引导强度直接影响"AI 自主性"的可信度。
4. **数学界会走 Hampshire 的"精神危机"路线还是 Tao 的"分工重组"路线?** 这决定了未来几年数学研究的组织形态。
5. **能力上限与平均实践的鸿沟会缩小吗?** OpenAI 用的是未发布模型 + 巨量算力;Nature 建模的是普通科学家用通用 LLM。这两条曲线什么时候会合并?

## 跟进阅读

- 本周周报 [`tech-blogs/2026-W32.md`](/tech-blogs/2026-W32.md)
- [Reddit W32 周报](/reddit-digests/2026-W32-reddit-hot.md) —— 本篇回答了其 Open Question #3
- [W31d 周报](/tech-blogs/2026-W31d.md) —— Anthropic HAWK 密码学发现(对照案例)
- [W31h 周报](/tech-blogs/2026-W31h.md) —— Google Science One Framework(可验证自主科研)
- [长程 agent 研究路径](/research-notes/2026-07-20-long-horizon-agents.md) —— "自主科研"属其 H3/自进化一格

> **本篇无 arXiv 引用**（官方博客 + link blog 评述，非论文），故未向 `references.bib` 入库。OpenAI 的论文 PDF 与 Lean 仓库为官方材料、非 arXiv 标识。