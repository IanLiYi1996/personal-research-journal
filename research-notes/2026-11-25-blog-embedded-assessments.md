# 嵌入式第三方评估落地了：METR 在 Anthropic 的 11 周 · OpenAI 的四家 embedded assessor · METR 对 OpenAI 监控系统的红队

> **Date**: 2026-11-25 ｜ **Tags**: third-party-assessment, RSI, monitoring, METR, OpenAI, Anthropic
> **来源**（全部取自 tech-blogs W48 抓取数据，读取时刻 2026-11-25 09:4x UTC）：
> - METR《Independent Assessment of AI R&D Acceleration at Anthropic in October 2026》—— [AlignmentForum 链接帖](https://www.alignmentforum.org/posts/N5AmYkRks5CqAxvBg/metr-independent-assessment-of-ai-r-and-d-acceleration-at)（全文取自 GreaterWrong 镜像，62,876 字符，含评论区摘录）
> - OpenAI《[Our first phase of embedded assessments](https://openai.com/index/our-first-phase-of-embedded-assessments)》（11-16；Cloudflare 挑战，经 `r.jina.ai` 取到 12,731 字节 markdown）
> - METR《[Red-Teaming OpenAI's Internal Agent Monitoring Systems](https://metr.org/blog/2026-10-22-red-teaming-openai-agent-monitoring/)》（10-22，curl 直取）
> - METR《[AI R&D-relevant model capabilities are improving rapidly and are getting harder to measure](https://metr.org/blog/2026-11-20-q4-2026-capability-update/)》（11-20，curl 直取；只读了概览、§5 与附录 G/J 的摘要段）

## TL;DR

- 我从 W31 起追的「厂商自测缺第三方复核」这条线，在九月 Dario 那份「embedded evaluators」提案之后，**十周内从提案变成了三件已经发生的事**：①METR 在 Anthropic 现场做了 11 周高权限评估（08-12 → 10-29）②METR 与 Redwood 各派一人，在 OpenAI 现场对内部监控系统做了 3 周红队（07-21 → 08-11）③OpenAI 与四家评估方签了首期协议（Apollo / METR / Transluce / Goodfire）。
- 🚨 **METR 对 Anthropic 的结论**：截至 2026-10，Anthropic 在定义明确的任务上提升模型能力的速度，**约为 2024-03 → 2025-10 期间的 2 倍**（不确定性很高，1.3× 或 3× 都不意外）；基于 AECI 的 benchmark 增速为此前的 **1.9× [95% CI 1.5–2.8×]**；**排除了「训练算力扩张加速」这个解释**；加速的大部分**可能**来自 AI 用于研发本身。
- 🚨 **但最该记的是两条披露**：「Some of the key findings that most informed our conclusions about risk were redacted by Anthropic」——对风险结论影响最大的那些发现被厂商涂黑了；另外红队那份报告事先约定**只发简短的高层摘要**。
- 🚨 **METR 红队在 OpenAI 监控系统里找到的东西**与我这两个月记的「监控覆盖不到 / 闸门语义与实现不一致」逐条对上：RL 训练在演练开始时不在监控范围内；compaction 事件没被监控（约 10% 合格流量）；监控在 agent 把控制权交回用户时才触发 ⟹ **agent 只要一直继续行动就能无限期推迟被监控**；两个客户端可控的 header 能关掉监控；序列化差异能让监控看到的轨迹与 agent 看到的不同。

## 1. METR @ Anthropic：「2 倍」这个数字，以及它站在什么上面

**访问范围**（原文）：pretraining 与 RL 的多种指标、各领域训练算力、内部模型 benchmark、AI 提速分析、未来模型指标的预测、相关员工访谈。METR 称之为「unprecedented access」。

**主张与证据层次**（按 METR 自己的分法）：

| 主张 | 证据 | 我的读法 |
|---|---|---|
| 定义明确任务上的能力提升速度约 2× | 主观综合判断，1.3×–3× 都不意外 | 这是结论层，置信度最低 |
| AECI 增速 1.9× [1.5–2.8] | 内部 benchmark 拟合，带区间 | 唯一带 CI 的数字；但 AECI 是 Anthropic 版的 ECI，**测的是可验证任务** |
| 不是训练算力加速造成的 | 「We are confident」 | 承重最强的一条排除 |
| 大部分来自 AI 研发使用 | 访谈与问卷「consistent with」 | ⚠️ 员工自报 uplift，METR 自己写「challenging to interpret」 |
| 12 个月内没看到显著瓶颈 | 「our understanding … changed several times … not confident」 | METR 主动标为不稳定 |

⭐⭐ **它主动写出两个方向的误差来源**：可能**低估**（近期增速进一步上升 / AR&D 进一步自动化 / AECI 有时低估真实影响）· 可能**高估**（非 AI 因素 / AECI 偏重可验证任务 / Anthropic 2023–2025 早期增速异常低）⟹ 「2×」应读作一个两侧都开着的估计。

⭐⭐⭐ **涂黑这件事的代价，评论区三个人说得比我准**：
- Ryan Greenblatt（同时在 METR）：「Because so much of the underlying evidence was redacted from the public report, I think conclusions about things other than the observed 2x faster rate of progress on benchmarks **depend significantly on trust for the authors**」——⟹ **只有「benchmark 增速 2 倍」是公开证据能撑住的；「这个加速来自 RSI」要靠信任作者。**
- Ajeya Cotra：「Evidence about risk should be transparent.」
- 另：Anthropic 在 S1（IPO 招股书）里写了它「may imminently meet」自家 RSP 的「AI 导致进展显著加速」门槛（[Ajeya 那篇](https://www.alignmentforum.org/posts/8tL9f5HZxsTvn2LHp/ajeya-ai-is-dramatically-accelerating-ai-research-right-now) 引的是 S1 原文）⟹ ⭐⭐ **一份监管申报文件与一份第三方评估在同一个月指向同一个门槛**，这是我在这条线上第一次看到两个性质完全不同的来源（法律申报 vs 独立评估）对上。

⟹ ⭐⭐⭐ **对我那条「RSI 有没有人报第二圈」的问题**：这份评估量的不是「改进者被改进」这个机制，而是**结果面的增速**。它说「速度翻倍、算力不是原因、AI 使用大概是原因」，⚠️ 但「大概是原因」恰好是被涂黑证据支撑的那一半。

## 2. METR + Redwood @ OpenAI 监控：被找到的都是「覆盖」与「语义」问题

**设定**：三周、深度访问（内部代码库、监控 prompt 与配置、内部文档、约十几位研究与工程人员访谈）、OpenAI 设备不出场地、全程陪同；产出 61 页内部报告；**事先约定公开版只写简短高层摘要**。

**发现**（我按此前的分类重排）：

| 类型 | 发现 |
|---|---|
| 覆盖 | GPT-5.5 及以上模型的内部 Responses API agentic 流量约 **75–94%** 可被监控；内部 Codex 子集 **99.2%** |
| 覆盖 | 演练开始时 **RL 训练运行不在监控范围内**（此前 3–6 月有部分 rollout 被监控过） |
| 覆盖 | **compaction 事件未被监控**，约 10% 合格 Codex 流量（5–7 月）；OpenAI 事先已发现并在演练期间修复 |
| 触发语义 | 监控在 agent **交回控制权时**才触发 ⟹ compaction 未被监控期间，**agent 只要一直行动就能无限期推迟被监控** |
| 实现 | 两个**客户端可控 header** 可为新 agent 关闭监控（⭐ 但设置 header 并新建实例这个动作本身可能被监控抓到） |
| 实现 | agent 看到的轨迹与监控看到的存在**序列化差异**，可构造让监控失败的输入 |
| 实现 | 一个能让**数百条近期动作（含利用本身）不被监控**的 exploit（细节被 OpenAI 涂黑） |
| 沙箱 | 找到一种方式可让 Codex agent **绕过沙箱与 Auto-Review**，执行宿主用户能执行的任何动作 |
| 权限 | 两个被认为只读、且不受监控的内部 agent 工具，存在简单 exploit 可执行任意动作（未实测验证） |
| 正面 | 监控看到的轨迹与 agent 看到的基本一致；监控 prompt 覆盖 scheming / 监控规避 / 破坏 / 外泄 / 欺骗 / 注入 |

⭐⭐⭐ **与我此前记录的对应**：
- 「监控在交回控制权时触发」⟹ 是我 09-03 记的 Astra 那条「**长程运行的 agent 是误报高发区**」的镜像：**长程运行也是漏报高发区**。
- compaction 未被监控 ⟹ 第四次撞上 compaction：①08-26 Latent Space「compaction 已被吸收进权重」②08-27「正常 handoff 压缩产生 100% 约束失活」③08-26 OpenAI 事故复盘「监控没在那些评测上跑」④**这次：compaction 本身是一个监控盲点**。⟹ ⭐⭐ **compaction 是一个同时影响「约束保留」与「可观测性」的转换，而它正被当作理所当然的基础设施。**
- header 关监控、序列化差异 ⟹ 与 09-06 wiki 事件（`CGI.pm param()` 不区分 GET/POST）同一形态：**闸门的语义与它的实现不一致**。

⚠️ **时间轴必须一起写**：演练于 08-11 结束，**正好落在 HF 那起事故（07-19 被发现）之后、OpenAI 08-18「Pacing model development」之前** ⟹ 这些发现描述的是「事故之后、加固之前」的系统；METR 明说没有评估 OpenAI 之后声称的修复与覆盖扩张。⭐ 而更新日志里有一处值得记：原文写「all … vulnerabilities … have been patched」，10-29 应 OpenAI 反馈改成「most … patched; one remaining fix is still in progress」——**一个第三方报告被厂商要求「改得更保守」**，方向与我此前担心的（厂商要求措辞更温和）相反。

## 3. OpenAI 首期 embedded assessments：协议条款本身就是信息

**四家与各自范围**：
- **Apollo Research**：评估模型能否**暗中违背开发者意图**、防线能否发现；分两阶段检验四条主张（训练没激励颠覆 / 强评测与红队引不出颠覆 / 内部使用中模型没尝试颠覆 / 颠覆背后的推理可被发现）。
- **METR**：①对**迄今最大的一次 RL 运行**做对齐评估，含现有测试与监控**可能漏掉的失败**，METR **独立设定评估优先级**②一个更宽的监控评估（尚待双方商定范围）。
- **Transluce**：评估 OpenAI 后训练对齐评测的**质量与覆盖**；2027 年初计划约 8 周的后续，含「评测结果如何影响发布决策」。
- **Goodfire**：嵌入可解释性研究员做训练运行监控与调试；**主要受众是 OpenAI 内部研究员**。

**协议条款**（我认为比名单更有信息）：

| 条款 | 内容 | 我的读法 |
|---|---|---|
| 现场 | 需接触敏感信息者在 OpenAI 办公室、用 OpenAI 设备 | 与 Dario 提案（工位、门禁、公司电脑）几乎逐字相同 ⟹ 两家收敛到同一个模板 |
| 监督 | SSC（基金会董事会下的独立委员会）**收到未涂黑报告**，评估方与 SSC 可私下沟通 | ⭐ 公开版涂黑、监督机构看全本 ＝ 两层披露 |
| 分歧 | 升级到 Safety Advisory Group 与高管，限期解决，SSC 被通知 | 有一个具名的升级路径 |
| 员工访谈 | **off the record**，不可发表、不可归因；另有「attestation pathway」可请求 OpenAI 出具书面确认 | ⚠️ 访谈内容不能直接作为公开证据 |
| 发表 | 评估方**独立发表**，但 OpenAI **保留涂黑机密信息的权利**，并可请求「负责任披露延迟」 | ⚠️ 正是 METR @ Anthropic 那份里起作用的机制 |
| 利益冲突 | 冷却期、财务利益披露「正在制定」 | ⚠️ 尚未定稿 |

⟹ ⭐⭐⭐ **一个结构性观察**：三份材料里**最强的结论都落在被涂黑的那部分上**（Anthropic 的风险结论依据、OpenAI 监控 exploit 的细节），而协议明确保留了涂黑权。⟹ **第三方评估解决了「有没有人独立看过」，没有解决「公众能不能核对他们看到了什么」**；后者现在只由 SSC 这类内部独立机构接住。⭐ 这与我 09-15 记的 AEF-1（访问权 / 利益冲突 / 资金关系 / 回避 / 透明度五项）对得上：**这十周里「访问」那一项大幅前进，「透明度」那一项没有动。**

## 4. METR Q4 能力更新：测量本身在变难

只读了概览与讨论，记三条：

1. **「我们通常找不到能超过当前前沿模型的人类 baseliner」**；时间地平线（TH1.1 + TH2 合并）约每 **3.6 个月**翻倍，与 OpenAI 内部研究任务上报告的趋势相近。
2. ⭐⭐ **「在缺乏清晰经验反馈的任务上，agent 的判断力似乎比人类专家差」**，而且有一个具体机制：**模型裁判在评价由同等或更强模型产出的东西时会高估其质量** ⟹ 这与今天 HF 侧 False Frontiers 的「共谋作弊」与 RRSI 的「裁判打分可能被学会」是同一件事的第三个位置。
3. ⭐⭐ **swarm 作为激发方式**：在有清晰指标、快反馈的可验证任务上优于 best-of-k；⚠️ 「**large swarms seemed harmful to performance when the task required creating a proxy metric for themselves**」——需要自己造代理指标时，大 swarm 反而有害。swarm 实验花了约 **$500K** token。
4. 讨论那节最该抄的一句：「it is now more difficult to bound risk by pointing to some easy-to-measure and broad incapability」⟹ **「它还做不到 X，所以风险低」这类论证在可验证任务上已经用完了。**

## 我的反思

1. ⭐⭐⭐ **三个月前我写「外部团队发现了厂商没发现的事件」是这条线最强的证据；现在最强的证据换成了「厂商把外部团队请进来，而他们找到了真问题」**（RL 不在监控内、compaction 盲区、header 关监控）。这比前者更有价值，因为它是**事前**的。
2. ⭐⭐ **但「第三方评估」不等于「可公开核对」。** 对客户材料：引用这些报告时应同时写出它们的披露结构（谁看到全本、谁看到涂黑版、什么被涂黑）。
3. ⚠️ **我自己的判断边界**：Anthropic 那份全文报告（Full Report）我没读到，只读了执行摘要与评论区；METR Q4 只读了概览。所有数字都是 METR 或 OpenAI 自报。

## Open Questions

1. METR 对 OpenAI「最大 RL 运行」的对齐评估何时发布？发布时涂黑了多少？
2. Anthropic 的 [An Update On Independent Evaluators](https://www.anthropic.com/news/an-update-on-independent-evaluators) 与 [Accenture Embedded Evaluation](https://www.anthropic.com/news/accenture-embedded-evaluation) 讲了什么？（⚠️ Anthropic sitemap 的日期本窗口整体不可用，这两条我只有标题。）
3. 「交回控制权时才触发监控」这个设计，在其他厂商的 agent 产品里是否同样存在？
