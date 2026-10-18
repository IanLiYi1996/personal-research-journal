# RSI 深度报告：学术覆盖审计便笺

更新说明：这份便笺记录补齐前版本的缺口，不代表当前主文的覆盖状态。相关内容现已整合进[最终详细阅读版](../rsi-detailed-2026-10-17.html)；下文保留为历史审计记录。

审计日期与截止日：2026-10-17。对象：`/research-notes/2026-10-17-rsi-detailed.md`。

## 结论

报告已经覆盖主要经验自修改路线、经典理论和重要负面证据，不能笼统称为“遗漏了自改进研究”。主要不足是：**影响力与篇幅没有充分对应、已有论文的正式发表状态更新不齐、少数基础方法缺失或仅被点名。**

按用户最新要求，只保留三个最高优先级补充单元，不为凑数扩展历史论文。它们分别是 R1/R1-Zero、SCoRe，以及 MAML/RL² 对照组；第三项包含两篇互补基线。以下是覆盖审计与补充建议，不是替报告扩写的正文。

本次只新建此便笺，不修改现有报告或网页。行业项目由主任务负责，不重复检索或判断其覆盖。

## 已覆盖与重要缺口

| 对象 | 报告内证据 | 覆盖判断／建议 |
|---|---|---|
| Good、Gödel machine、AIXI 与最优性边界 | 第二章，脚注 2–4 | 已有成段解释；此次没有发现比下列三项更应优先补的经典理论缺口。不是对全部理论文献的穷尽性声明。 |
| Promptbreeder、STOP | 第四章，脚注 6–7 | 已说明变异提示词／改进器自身进入修改空间；无需重复新增。 |
| learned optimizer、SRWM、自指元学习／FME | 第五章，脚注 5 | 已覆盖，但缺少 MAML／RL² 这组有影响力的固定元优化对照。 |
| ADAS、SICA、Gödel Agent、DGM | 第六章，脚注 9–12 | 已较详写，并区分固定外层与可变程序；ADAS 正式发表状态有可核实的更新，见下文。 |
| STaR、SEAL、AZR、Self-Rewarding | 第四、七章，脚注 8、13–15 | 已覆盖生成、验证、训练与外部锚点；R1/R1-Zero 是这条方法链中的显著缺口。 |
| Hyperagents、AIDE² | 第九章，脚注 20–21 | 已详写元改进测量、迁移归因与 ignition test 的限制；不重复研究。这里仅判断文本覆盖，不重新认证这些近期论文的全部结果。 |
| 自纠错负面结果、合成数据退化／累积数据、奖励过度优化、自修改目标保存 | 第十二、十三章，脚注 32–35 | 已有正反证据与假设限制；不能把这些主题误报为完全遗漏。 |
| SCoRe | 第 147 行一句方法概述；脚注 32 仅标“2024” | **已提及但不足**，不是全新项目遗漏；值得补方法机制与 ICLR 2025 正式发表信息。 |
| DeepSeek-R1／R1-Zero | 全文与引用中未出现 | **完全遗漏**；优先补为权重级自改进的基础方法，不标成真正递归改进器。 |
| MAML／RL² | 全文与引用中未出现 | **基线缺口**；用来说明学习了适应规则不等于自行改写外层学习算法。 |
| MLE-bench、RE-Bench、PostTrainBench | 第八章 | 已作为邻近 AI 研发自动化／评测讨论；其存在本身不是完整 RSI 的证据。行业项目不在本审计范围。 |

## 三项最高优先级建议

### 1. P0：DeepSeek-R1／R1-Zero——补方法里程碑，而非模型新闻

**为何应补。** 原报告从 STaR 跳到 SEAL、AZR，缺少“预训练模型通过可验证奖励强化学习形成更强推理策略”的关键实证参照。R1-Zero 不先做监督微调，不意味着没有预训练模型、训练题目、奖励或人工设计的训练程序。完整 R1 又采用冷启动数据及多阶段训练，两者不可混写。

**分类。** 基础方法／权重级能力改进。RL 算法、奖励定义和阶段安排由研究者设定；模型出现检查或修正推理的行为，不等于它改写了自己的训练算法，更不证明改进能力持续自加速。

**可核查影响力。** 已正式发表于 **Nature 645, 633–638（2025）**，在线发表日期为 **2025-09-17**；不是只有预印本或厂商宣传。Semantic Scholar 本次合并记录返回引用数 **5,899**。引用数只作传播规模证据，不当作效果认证，也不据此宣称所有后续方法都依赖它。

**最小补充。** 在第七章加一个短方法段，说明 R1-Zero 与 R1 的训练条件、外部验证信号，以及“推理策略变强”和“训练改进器变强”的差别；脚注同时记录预印本首发与正式发表。

原始／元数据来源：

- [原论文 arXiv 2501.12948](https://arxiv.org/abs/2501.12948)，首发 2025-01-22；当前页面列出 Nature 正式发表信息。
- [Nature 正式论文](https://www.nature.com/articles/s41586-025-09422-z)。
- [Crossref DOI 元数据](https://api.crossref.org/works/10.1038/s41586-025-09422-z)，核对期刊标题、DOI 和在线发表日期；本次 Nature HTML 抓取未提供可用正文，未据此声称阅读全文。

### 2. P1：SCoRe——补足已有覆盖，并更新到 ICLR 2025

**具体不足。** 第 147 行只说强化学习改善自纠错、训练使用正确性奖励，这个表述基本方向正确，但没有保留论文最重要的对照结论：离线自生成修正轨迹上的 SFT 会遭遇分布错位，普通多轮 RL 也可能出现“第一答更好、第二答几乎不改”的行为坍缩。

**应补机制。** SCoRe 使用 on-policy 多轮 RL，先构造较不易坍缩的初始化，再通过奖励塑形鼓励两次回答之间的修正进步。应分别观察第一答、第二答及二者增益，不能用第二答绝对准确率代替“更会自纠错”。

**分类。** 基础方法／训练得到的自纠错策略，不是真正自主改写外层 RL 算法的 RSI。它说明早期无训练自纠错的负面结果不应泛化成所有训练条件下的“不可能”。

**可核查影响力。** 主办方已列为 **ICLR 2025 正式 poster**；S2 引用数 **497**。本次没有证据据以宣称普遍部署或行业广泛采用，不使用论文自报成绩作影响力排序依据。

**最小补充。** 在现有“反思什么时候有效”小节补一段两阶段训练与失败模式；脚注由仅“2024”更新为“2024 预印本／ICLR 2025”，保留训练奖励与测试反馈的区分。

原始来源：

- [SCoRe 原论文与版本](https://arxiv.org/abs/2409.12917)。
- [原文 HTML：失败模式与方法，§4–5](https://arxiv.org/html/2409.12917v2)。
- [ICLR 2025 主办方论文页](https://iclr.cc/virtual/2025/poster/30501)；也可在[官方论文目录](https://iclr.cc/virtual/2025/papers.html)按标题核对。OpenReview API 本次返回 403 challenge，因此改用主办方页面，不绕过拦截。

### 3. P1：MAML／RL²——补固定元学习对照，不新增一条“RSI 成功”叙事

**为何重要。** 第五章已有 learned optimizer、SRWM 和 FME，但读者缺少两个常用参照：MAML 通过固定外层元优化学习便于适应的初始化；RL² 通过固定的慢 RL 学出在 RNN 状态中运行的快速学习行为。它们能直接帮助解释“学习规则被学习了”与“外层学习过程自行递归改进”不是同一命题。

**分类。** 基础方法／元学习对照。参数、记忆或适应表现变化，不能单独证明改进器持续自增强；架构名称中的自指也不能自动证明其有更强表达能力。

**可核查影响力。**

- **MAML：ICML 2017 正式论文，PMLR 70:1126–1135**；S2 引用数 **15,346**。正式发表与传播规模足以支持其作为核心背景，而不需要靠年份新或模型成绩。
- **RL²：本次只核实 arXiv 2016 原论文**，不凭记忆补会议接收状态。S2 后续批量请求触发 429，引用数保留 **null**。
- 报告已经引用的 [Eliminating Meta Optimization Through Self-Referential Meta Learning](https://arxiv.org/html/2212.14392) 在引言及 §2、相关工作中明确引用 Finn et al. 与 Duan et al.，并比较人工元优化下的自指网络与 memory-based/in-context learner。这是两类方法与报告核心议题有直接后续理论联系的证据；**引用与理论比较不等于代码实现依赖或直接实验基线**。

**最小补充。** 在第五章添加一张小对照或一段背景：外层人工设定／内层学习行为／是否更新改进机制。优先 MAML；RL² 作为记忆式元学习参照，不必另开长章。

原始来源：

- [MAML 正式论文与完整作者／书目信息](https://proceedings.mlr.press/v70/finn17a.html)。
- [MAML arXiv 1703.03400](https://arxiv.org/abs/1703.03400)。
- [RL² arXiv 1611.02779](https://arxiv.org/abs/1611.02779)。
- [自指元学习原文 §2 的比较](https://arxiv.org/html/2212.14392#S2)。

## 影响力排序与正式发表更新：比继续加名称更重要

这是一份 RSI 专题而非通用 AI 历史，DGM、Hyperagents、AIDE² 的细读篇幅有主题理由；问题不在于应把它们删减，而在于阅读路线缺少一层“成熟基础—主题直接相关—新近探索”的证据分组。

建议分开两种排序：

| 排序维度 | 建议做法 |
|---|---|
| 研究传播与成熟度 | 标注正式 venue、预印本／正式发表年份、带平台与日期的引用数；MAML、R1、SCoRe 的证据不能被最新预印本篇幅遮蔽。 |
| 与真正 RSI 的机制相关性 | STOP、Promptbreeder 及报告已有的元改进实验单独分组；基础方法引用多也不能因此升格为真正 RSI。 |
| 补充优先级 | 按“对当前报告论证的缺失程度 × 方法重要性”排序；本便笺 P0/P1 不是引用数排行榜。 |

**已核实的状态更新：**

- **SCoRe**：脚注 32 的“2024”是预印本年份，不是完整出版状态；应补 **ICLR 2025**。
- **ADAS**：脚注 9 只标“2024”；主办方已列为 **ICLR 2025**：[正式论文页](https://iclr.cc/virtual/2025/poster/28073)。这是已有覆盖的元数据更新，不需要新增项目。
- **R1**：新增引用时同时保留 2025-01-22 预印本与 2025-09-17 Nature 正式发表，不把 S2 返回的预印本日期写成期刊发表日。

未逐篇核实其他已有论文的最新 venue，不批量猜测会议状态。报告已正确注明 Promptbreeder 的 ICML 2024、STOP 的 COLM 2024 与 Gödel Agent 的 ACL 2025，说明问题是更新不齐，而不是完全忽略正式发表。

经典理论已含 Good、Gödel machine、AIXI、Everitt；本次不推荐为历史完备性添加更多低优先级名称。行业进展及邻近科研自动化的新增项目全部留给主任务。

## 精简标准元数据

采用 academic-search 标准字段；未知值为 `null`。R1 作者列表明确缩写，不伪装成完整作者表。`year` 使用所列版本／正式出版年份，另存预印本日期。代码链接未核验，统一不补。

```json
[
  {
    "title": "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning",
    "authors": ["DeepSeek-AI", "Daya Guo", "Dejian Yang", "Haowei Zhang"],
    "authors_truncated": true,
    "year": 2025,
    "publication_date": "2025-09-17",
    "preprint_date": "2025-01-22",
    "publication_type": "journal-article",
    "venue": "Nature 645, 633–638",
    "doi": "10.1038/s41586-025-09422-z",
    "arxiv_id": "2501.12948",
    "citation_count": 5899,
    "citation_source": "Semantic Scholar merged record",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2501.12948",
    "code_url": null,
    "source_platforms": ["arxiv", "crossref", "semanticscholar", "openalex"],
    "fetched_at": "2026-10-17"
  },
  {
    "title": "Training Language Models to Self-Correct via Reinforcement Learning",
    "authors": ["Aviral Kumar", "Vincent Zhuang", "Rishabh Agarwal", "Yi Su", "John D Co-Reyes", "Avi Singh", "Kate Baumli", "Shariq Iqbal", "Colton Bishop", "Rebecca Roelofs", "Lei M Zhang", "Kay McKinney", "Disha Shrivastava", "Cosmin Paduraru", "George Tucker", "Doina Precup", "Feryal Behbahani", "Aleksandra Faust"],
    "year": 2025,
    "publication_date": null,
    "preprint_date": "2024-09-19",
    "publication_type": "conference",
    "venue": "ICLR 2025",
    "doi": null,
    "arxiv_id": "2409.12917",
    "citation_count": 497,
    "citation_source": "Semantic Scholar merged record",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2409.12917",
    "code_url": null,
    "source_platforms": ["arxiv", "semanticscholar", "iclr"],
    "fetched_at": "2026-10-17"
  },
  {
    "title": "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks",
    "authors": ["Chelsea Finn", "Pieter Abbeel", "Sergey Levine"],
    "year": 2017,
    "publication_date": "2017-07-17",
    "publication_type": "conference",
    "venue": "ICML 2017; PMLR 70:1126–1135",
    "doi": null,
    "arxiv_id": "1703.03400",
    "citation_count": 15346,
    "citation_source": "Semantic Scholar",
    "full_text_status": "open_pdf",
    "pdf_url": "https://proceedings.mlr.press/v70/finn17a/finn17a.pdf",
    "code_url": null,
    "source_platforms": ["pmlr", "semanticscholar"],
    "fetched_at": "2026-10-17"
  },
  {
    "title": "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning",
    "authors": ["Yan Duan", "John Schulman", "Xi Chen", "Peter L. Bartlett", "Ilya Sutskever", "Pieter Abbeel"],
    "year": 2016,
    "publication_date": "2016-11-09",
    "publication_type": "preprint",
    "venue": "arXiv; formal venue not verified in this audit",
    "doi": null,
    "arxiv_id": "1611.02779",
    "citation_count": null,
    "citation_source": null,
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/1611.02779",
    "code_url": null,
    "source_platforms": ["arxiv"],
    "fetched_at": "2026-10-17"
  }
]
```

## 核验方式与限制

- 完整读取 academic-search 技能、计算机学科、元数据 schema、arXiv／S2 站点参考及 API cookbook；预检确认 Node 与 curl 可用，Chrome 未连接，因此未查 Google Scholar。
- 报告 661 行，分段补读首次输出截断的中部，核查章节、脚注与缺失关键词；只审计文本覆盖，不独立复现分数。
- 引用数来自本次 S2 批量接口，查询标识为 `ARXIV:2501.12948`、`ARXIV:2409.12917`、`ARXIV:1703.03400`。可复核接口：`POST https://api.semanticscholar.org/graph/v1/paper/batch?fields=title,authors,year,venue,publicationDate,citationCount,externalIds,openAccessPdf`，请求体使用上述 `ids`。RL² 后续请求返回 429，停止重试并保留 `null`。
- S2 可能合并预印本与正式版本，并保留预印本年份／日期；venue 与日期优先用 PMLR、ICLR 官方页面和 Crossref 校正。引用数是 2026-10-17 抓取快照，不是首发时引用数；不同出版年龄不宜机械横排。
- R1 的 OpenAlex DOI 查询返回 OA 与出版版 PDF 信息。Unpaywall 无已配置联系邮箱，试请求返回 422；没有编造邮箱或建立账户，也不把失败推断为无开放全文。arXiv PDF 可用性依技能的 ID 规则记录，未下载全文或复查全部 PDF 页面。
- 初筛的低优先级历史候选未进入最终清单；用户要求收束后不再扩展。不存在“发现了四个新 RSI 成功实例”的结论。
- 审计前原报告 SHA-256：`8fc3ec8d284d092497333abfd2d1c1ca534c84333e2b1dd87fe3857658cfa186`。
