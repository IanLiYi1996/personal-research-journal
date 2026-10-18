# 科学家 agent：研究闭环正在变长，但不等于改进器已经能自我升级

核验截止：2026-10-17。以下正文可直接整合进阅读版；文后的版本核对、原始链接及 JSON 供编辑使用。仅覆盖 Sakana The AI Scientist、Google Co-Scientist、FutureHouse Robin 和 Edison Kosmos。

## 可整合正文

判断“AI 科学家”离递归自我改进（RSI）有多近，首先要问：它反复修改的是假说、实验代码和论文，还是负责产生这些改进的系统本身？下面四个项目把研究过程从一次问答变成了有记忆、有反馈的工作流，有些还能接回真实实验结果。不过，“下一轮研究比上一轮更好”与“下一代改进器能更有效地升级自己”，仍是不同的证据要求。这里将后者作为区分真正 RSI 的关键，不把论文中的 self-improvement 自动译成已经实现 RSI。

### 从写完整论文，到接受外部评审

Sakana 的 The AI Scientist 让人看见了“自动研究流水线”的完整形状：它检索文献、提出想法、修改代码、运行计算实验，再画图、写论文并模拟评审。首版于 2024-08-12 发布；v2 的同行评审实验于 2025-03-12 对外公布，技术论文于 04-10 登上 arXiv。首版需要人提供起始代码模板，v2 则用实验管理器和树搜索，从多条代码方案中继续调参、排错和做消融实验。2026-03-25 的 Nature 正式论文同时报告模板式和无模板式两种设置，不能简单写成“v1 原稿转正”。[^s-v1][^s-v2][^s-announcement][^s-nature]

它的重要性在于把可执行实验和完整研究表达接在了一起，又接受了真实人类评审。作者报告：人类筛选送审的 **3 篇稿件中，1 篇达到 workshop 评审门槛，平均评分为 6.33**；这个小样本不是所有运行的成功率。研究人员筛选送审稿件、检查实现并协调投稿，但没有人工改写选中稿件；人仍负责设定研究范围与选择哪些成果送审。依预定协议，全部稿件在评审后撤回，没有完成最终 meta-review，也不是 ICLR 主会正式录用。Nature 刊登的是人类作者对系统的研究，不是 AI 自行写出了一篇 Nature 论文。公开代码、不同随机种子的实验重复和公开评审材料有助于审计，却不等于外部团队已完整复现。它迭代的是研究想法与实验产物，尚未证明能连续升级自己的研究改进器。[^s-nature][^s-announcement]

### 把“想法越来越好”接到科学家的验证流程

Google Co-Scientist 更像一组围绕问题开讨论会的研究助手。科学家设定目标后，生成、反思、排名、演化等 agent 查文献、相互辩论，让假说参加成对竞赛，再把评审反馈加入后续提示。官方于 2025-02-19 公布系统，02-26 发布《Towards an AI co-scientist》预印本；Nature 正式论文于 2026-05-19 在线发表，题名改为《Accelerating scientific discovery with Co-Scientist》。这里的“自我改进”主要是更新假说和提示上下文，原文明确说明不依赖反向传播、微调或强化学习，不能据此认定模型权重或改进算法在自动升级。[^g-launch][^g-v1][^g-nature]

作者报告的计算分析覆盖 **203 个研究目标**，人类偏好评估仅覆盖 **11 个目标**，真实验证则落在 **3 类生物医学问题**上。内部 Elo 上升不等于科学真理得到证明；更有意义的是合作科学家筛选候选、完成细胞或类器官实验，以及系统重现了研究者当时尚未公开的细菌基因转移机制。后者的实验发现早于 AI 使用，不能写成 AI 从零完成了全部发现与验证。2026 年它进入 Gemini for Science 的 Hypothesis Generation 实验工具，但完整源码未公开。08-27 的另篇扩展预印本增加代码执行、实验日志核验和部分仪器控制，物理实验仍由人装样、调整与测量，也不能把该原型当成已全面开放的产品。其影响是拓宽研究候选并压缩探索成本，而非证明临床疗效、跨实验室可靠性或持续 RSI。[^g-nature][^g-launch][^g-product][^g-extension]

### 当实验结果真的改变下一轮假说

FutureHouse 的 Robin 将文献搜索 agent Crow、Falcon 与数据分析 agent Finch 连接起来。它先围绕疾病提出机制和候选实验，研究人员做实验并上传结果，Finch 写分析代码，Robin 再据此提出后续实验和候选。预印本首发于 2025-05-19，官方介绍原发于次日；同一介绍页现显示的 2026-05-19 是 Nature 发表后的更新日期，不能误当项目首次发布。正式论文同日上线，新增了原代人视网膜色素上皮细胞等验证。[^r-preprint][^r-official][^r-nature]

它有影响力，不只是因为会列出候选，而是因为新实验结果能够回到下一轮推理。作者报告，在一个文献处理阶段，系统约 **30 分钟分析 551 篇论文**；这不包含培养细胞等实验等待时间，也不是全流程实测加速倍数。分析阶段通过多条独立轨迹形成共识，人类又用同一批数据复核，并在原代细胞中检验候选的吞噬作用。正式版同时清楚划出人的责任：人筛选候选，把实验概要写成可执行协议，完成培养和测序；RNA-seq 的拆分与比对也先由人处理，部分图形经人工改善可读性。因此它是有实验室参与的半自主发现，而非无人实验室。Robin、Finch 有公开代码和样例，但现有工作流仍依赖 Edison 平台服务；公开仓库不代表完全离线可复现。得到的是体外候选与机制线索，不能外推治疗效果，更没有证明系统改写了自己的改进器。[^r-nature][^r-code]

### 从一个反馈周期，扩展到长程、可追踪的数据探索

Kosmos 于 2025-11-04 首发预印本、次日更新并上线 Edison 平台；Edison 是 FutureHouse 分拆的商业公司。相较 Robin，它的核心变化是用结构化“世界模型”保存假说、分析结论和出处，让文献搜索与数据分析 agent 能共享进展，长时间探索同一个研究目标。这里的世界模型更接近一份可更新的研究账本，不意味着系统已经掌握完整自然规律。最终报告把陈述链接到文献或分析 notebook，研究者可以沿证据链追查。[^k-paper][^k-launch]

其最值得保留的数字是作者报告：专家独立检查 **102 条陈述**，总体 **79.4%** 获支持，但需要综合解释的陈述只有 **57.9%** 正确；准确率的统计单位是报告陈述，不是项目成功率。“一次运行相当于六个月研究”来自合作科学家的时间估计，不能当作通用、实测的加速倍数。作者展示了重现未公开或运行时未访问成果、人工重分析、重复分析轨迹和外部数据验证；这些证据主要由开发团队与合作研究组组织，未证明整个系统已被无关联团队复现。人仍提供和预处理数据、辨认重要结论，个别案例还人工整理发现标题与摘要。公开的主要是分析、作图及验证脚本，不能称完整 Kosmos 引擎已开源。至 2026-09-24，它新增 Nature 系列期刊的授权内容接入，这是产品与证据来源的进展，不是 Kosmos 论文在 Nature 发表。它改善研究状态和产物；未证明会持续优化自己的编排器，也未证明生物学机制或疗效。[^k-paper][^k-launch][^k-code][^k-content]

### 这些进展证明了什么

这些进展足以支持一个务实判断：AI 正从“替研究者回答问题”，走向“组织一段能被检查的研究过程”。但外部评审、可重复代码、独立数据和实验室测量，仍承担着不可被内部评分替代的验证责任。判断 RSI 时，还需要看到升级后的研究系统被装回改进循环，并在多轮、控制资源与外部评价的条件下持续提升改进能力；上述论文尚未给出这样的证据。这是本补稿依据工作流作出的分类判断，不是否认自动科学研究的价值。

## 编辑核对：证据层级与重要版本纠正

本稿采用的证据层级只用于区分出处与审查状态，并非医疗证据等级：A＝同行评审原始研究；B＝作者预印本；C＝公司官方发布或官方代码仓库。A 也不自动等于独立复现；C 可证明产品发布及接口存在，不能单独证明宣传性能。

| 项目 | 核实时间线与应纠正的表述 | 证据层级及验证边界 |
|---|---|---|
| Sakana | v1 arXiv：2024-08-12，修订至 09-01 v3。v2 评审结果公告：2025-03-12；公告标注技术报告／开源更新 04-07，但 arXiv 正式提交记录是 04-10，不合并这两个日期。Nature：2026-03-25。正式版含两种系统设置及新增评估，不是仅将 v1 换刊名。 | Nature 为 A；两篇早期论文为 B；公告／代码为 C。达到 workshop 首轮评审门槛不等于主会录用或正式出版；人工选稿、检查代码与未改写最终稿件可以同时成立。作者协议要求撤稿，且没有最终 meta-review。 |
| Google | 官方公告：2025-02-19；arXiv v1：02-26。Nature Published：2026-05-19；arXiv v2：06-29，改用正式题名；Nature Version of record：07-01，Issue date：07-09。实验工具在 2026 年 5 月 I/O 公布，05-28 官方回顾明确其名为 Hypothesis Generation。实验执行扩展是另篇 2026-08-27 预印本，不是同一篇 Nature 的发表日。 | Nature 为 A；2025 原稿及 2026 扩展为 B；产品公布为 C。原研究是真实人／实验室参与的验证。全系统源码未公开，仅有伪代码、提示与实验访问；不能将公开 Gemini API 当作完整 Co-Scientist 开源。 |
| Robin | arXiv：2025-05-19；公司原文：05-20；公司页更新与 Nature Published 均为 2026-05-19；Nature Version of record：07-01，Issue date：07-09。原预印本作者少于正式版，不能机械复制旧作者表。 | Nature 为 A，预印本为 B，仓库为 C。正式版更明确地称 semi-autonomous，并披露原代细胞验证、人工协议和人工 RNA-seq 预处理。原代细胞来自单一供体，孔数是技术重复，不是患者数。 |
| Kosmos | arXiv v1：2025-11-04；v2 与公司发布：11-05。2026-09-24 的 Springer Nature 合作是内容授权。核实到的主系统研究仍是 arXiv 技术报告，本次未找到其正式期刊发表记录。 | 主技术证据为 B；产品／授权为 C。没有把“未找到正式发表”写成永远没有发表。没有找到足以确认完整核心引擎开源的官方记录。 |

另有两项易失真的开放性／验证细节：

- Sakana Nature 的 Code availability 写两仓库为 Apache 2.0，但检索时 v2 仓库 README 声明的是 The AI Scientist Source Code License（Responsible AI License 衍生），要求披露 AI 使用。两处当前记录不一致；正文仅称“公开代码”，不要未经核对沿用 Apache 2.0 或宽松许可的说法。
- Kosmos 三个重现案例不能统一写成“完全排除训练污染”。神经连接组案例的原预印本早于部分所用模型知识截止日期；作者另做了采用更早截止模型的控制重跑。报告中的独立专家核验也不等同于系统的无关联外部复现。[^k-paper]

本次未执行这些系统、付费接口或湿实验；关于重复与验证，均是在核对原始论文、方法、贡献声明和官方仓库后报告的证据。三篇 Nature 的开放获取状态另经 OpenAlex 与 Unpaywall 查询：均为开放获取；Robin 返回开放 HTML 但没有 PDF 地址，因此不把 API 的空 PDF 字段误判成全文封闭。引用数未做同平台同日期核验，JSON 一律保留 `null`。

## 原始出处

[^s-v1]: B；Lu et al., *The AI Scientist: Towards Fully Automated Open-Ended Scientific Discovery*，arXiv 初稿及版本历史：https://arxiv.org/abs/2408.06292 。公开实现：https://github.com/SakanaAI/AI-Scientist 。
[^s-v2]: B；Yamada et al., *The AI Scientist-v2: Workshop-Level Automated Scientific Discovery via Agentic Tree Search*：https://arxiv.org/abs/2504.08066 。公开实现及当前许可声明：https://github.com/SakanaAI/AI-Scientist-v2 。
[^s-announcement]: C；Sakana 原始公告，原发 2025-03-12，含 04-07 更新、撤稿与未做 meta-review 的明确说明：https://sakana.ai/ai-scientist-first-publication/ 。
[^s-nature]: A；Lu et al., *Towards end-to-end automation of AI research*，Nature，2026-03-25，DOI 10.1038/s41586-026-10265-5；重点核对 Human evaluation results、Limitations、Contributions、Code availability：https://www.nature.com/articles/s41586-026-10265-5 。
[^g-launch]: C；Google Research，2025-02-19，*Accelerating scientific breakthroughs with an AI co-scientist*；原始发现先于 AI 重现的说明位于 antimicrobial resistance 段：https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/ 。
[^g-v1]: B；Gottweis et al., *Towards an AI co-scientist*，2025-02-26，固定初稿版本：https://arxiv.org/abs/2502.18864v1 。改题后的 v2：https://arxiv.org/abs/2502.18864v2 。
[^g-nature]: A；Gottweis et al., *Accelerating scientific discovery with Co-Scientist*，Nature，首次在线 2026-05-19，DOI 10.1038/s41586-026-10644-y；重点核对 Expert-in-the-loop interactions、Statistics and reproducibility、Code availability、Meta-review、About this article：https://www.nature.com/articles/s41586-026-10644-y 。
[^g-product]: C；2026-05-28 官方 I/O 回顾，明确 Hypothesis Generation 基于 Co-Scientist、逐步开放：https://www.research.google/blog/a-new-era-of-innovation-google-research-at-io-2026/ 。当前项目入口：https://ai.google/gemini-for-science/ 。本稿不将其他 Google 科研工具的能力归给 Co-Scientist。
[^g-extension]: B；Schmidgall et al., *Accelerating Scientific Research with Gemini in the Real-World*，2026-08-27：https://arxiv.org/abs/2608.26701 。完整方法及边界：https://arxiv.org/html/2608.26701v1 。其中 Table 1、§3.1、§5.1、Code Availability 披露人类工作、同实验室重复及未公开完整系统源码；没有核实到该扩展稿的正式期刊发表。其研究 agent 产物也不是 Co-Scientist 自身连续升级的证明。
[^r-preprint]: B；Ghareeb et al., *Robin: A multi-agent system for automating scientific discovery*，2025-05-19：https://arxiv.org/abs/2505.13400 。
[^r-official]: C；FutureHouse，同页保留原发与更新标记：https://www.futurehouse.org/research/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system 。
[^r-nature]: A；Ghareeb et al., *A multi-agent system for automating scientific discovery*，Nature，首次在线 2026-05-19，DOI 10.1038/s41586-026-10652-y；重点核对 Therapeutic hypothesis generation、Validation in primary human RPE cells、Discussion、RNA-seq、Contributions：https://www.nature.com/articles/s41586-026-10652-y 。
[^r-code]: C；Robin：https://github.com/Future-House/robin 。Finch：https://github.com/Future-House/finch 。Robin README 明确完整数据分析依赖 Edison 平台；公开仓库并不消除商业服务依赖。
[^k-paper]: B；Mitchener et al., *Kosmos: An AI Scientist for Autonomous Discovery*，初发 2025-11-04，v2 2025-11-05：https://arxiv.org/abs/2511.02824 。完整方法、统计单位及人工整理记录：https://arxiv.org/html/2511.02824v2 ，重点 §2.1、§2.2.3、§3.1–3.2、§4.7.2、Supplementary Information 2–3。
[^k-launch]: C；Edison，2025-11-05，含商业分拆、上线与六个月估计来源：https://edisonscientific.com/news/announcing-kosmos 。
[^k-code]: C；公开作图与人类验证分析代码：https://github.com/EdisonScientific/kosmos-figures 。不是完整核心系统的代码发布。
[^k-content]: C；Edison，2026-09-24，Nature portfolio 内容授权公告：https://edisonscientific.com/news/bringing-trusted-research-into-kosmos-our-agreement-with-springer-nature 。

## 精简标准元数据 JSON

沿用 academic-search 标准字段；省略无关可选字段。`source_urls`、`evidence_level`、`version_notes` 是用于保留原始出处和版本区别的附加字段。`code_url` 指实际公开的代码，公开范围另写在 `version_notes`，不暗示整个系统已开源。

```json
[
  {
    "title": "Towards end-to-end automation of AI research",
    "authors": ["Chris Lu", "Cong Lu", "Robert Tjarko Lange", "Yutaro Yamada", "Shengran Hu", "Jakob Foerster", "David Ha", "Jeff Clune"],
    "year": 2026,
    "publication_date": "2026-03-25",
    "publication_type": "journal-article",
    "venue": "Nature",
    "doi": "10.1038/s41586-026-10265-5",
    "arxiv_id": null,
    "citation_count": null,
    "open_access_status": "hybrid",
    "full_text_status": "open_pdf",
    "pdf_url": "https://www.nature.com/articles/s41586-026-10265-5.pdf",
    "code_url": "https://github.com/SakanaAI/AI-Scientist-v2",
    "data_availability": "公开数据、生成论文、评审材料和补充方法；代码依赖外部模型服务。",
    "source_platforms": ["nature", "arxiv", "sakana.ai", "github", "openalex", "unpaywall"],
    "source_urls": ["https://www.nature.com/articles/s41586-026-10265-5", "https://arxiv.org/abs/2408.06292", "https://arxiv.org/abs/2504.08066", "https://sakana.ai/ai-scientist-first-publication/", "https://github.com/SakanaAI/AI-Scientist"],
    "evidence_level": "A",
    "version_notes": "正式版报告模板式与无模板式；不是AI自撰Nature论文。v1首发2024-08-12，v2公告2025-03-12，arXiv首发2025-04-10。公开仓库当前许可声明与Nature的Apache 2.0描述不一致。",
    "fetched_at": "2026-10-17"
  },
  {
    "title": "Accelerating scientific discovery with Co-Scientist",
    "authors": ["Juraj Gottweis", "Wei-Hung Weng", "Alexander Daryin", "Tao Tu", "Petar Sirkovic", "Artiom Myaskovsky", "Grzegorz Glowaty", "Felix Weissenberger", "Alessio Orlandi", "Dan Popovici", "Anil Palepu", "Keran Rong", "Ryutaro Tanno", "Khaled Saab", "Fan Zhang", "Jacob Blum", "Andrew Carroll", "Kavita Kulkarni", "Nenad Tomašev", "Dina Zverinski", "Ivor Rendulic", "Elahe Vedadi", "Florian Hasler", "Luka Rimanic", "Marina Boia", "Ivan Budiselic", "Ben Feinstein", "Mathias Bellaiche", "Tom Sheffer", "Jan Freyberg", "Jeremy Ratcliff", "Ottavia Bertolli", "Katherine Chou", "Avinatan Hassidim", "Burak Gokturk", "Amin Vahdat", "Yuan Guan", "Vikram Dhillon", "Eeshit Dhaval Vaishnav", "Byron Lee", "Tiago R. D. Costa", "José R. Penadés", "Gary Peltz", "Yossi Matias", "James Manyika", "Demis Hassabis", "Yunhan Xu", "Pushmeet Kohli", "Annalisa Pawlosky", "Alan Karthikesalingam", "Vivek Natarajan"],
    "year": 2026,
    "publication_date": "2026-05-19",
    "publication_type": "journal-article",
    "venue": "Nature",
    "doi": "10.1038/s41586-026-10644-y",
    "arxiv_id": "2502.18864",
    "citation_count": null,
    "open_access_status": "hybrid",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2502.18864v2",
    "code_url": null,
    "data_availability": "部分公开评估数据；补充材料提供系统伪代码和提示；完整系统源码未公开。",
    "source_platforms": ["nature", "arxiv", "research.google", "openalex", "unpaywall"],
    "source_urls": ["https://www.nature.com/articles/s41586-026-10644-y", "https://arxiv.org/abs/2502.18864v1", "https://arxiv.org/abs/2502.18864v2", "https://research.google/blog/accelerating-scientific-breakthroughs-with-an-ai-co-scientist/", "https://www.research.google/blog/a-new-era-of-innovation-google-research-at-io-2026/"],
    "evidence_level": "A",
    "version_notes": "2025-02-19公告；2025-02-26预印本原题Towards an AI co-scientist。2026-06-29 arXiv v2改题；Nature Version of record为2026-07-01、期号日期为07-09，不替代05-19首次在线日期。",
    "fetched_at": "2026-10-17"
  },
  {
    "title": "Accelerating Scientific Research with Gemini in the Real-World",
    "authors": ["Samuel Schmidgall", "Xiaokai Zhu", "Marian Shaw", "Lin Yang", "Valentin Liévin", "Jingyun Yang", "Yuchen Zhuang", "Tim Strother", "Alex Bijamov", "Min Woo Sun", "Anil Palepu", "Justin Chen", "David Steiner", "Jacqueline Shreibati", "Wei-Hung Weng", "Yilin Zhao", "Xingjian Hu", "Nicholas Zahn", "Sadhya Garg", "Julia Kirby", "Yuxiang Gan", "Jiaoli Li", "Divy Thakkar", "Shekoofeh Azizi", "David Racz", "Juraj Gottweis", "Vivek Natarajan", "Chenglin Wu", "Tal Danino", "Keran Rong", "Haozhe Wang", "Benoit Schillings", "Yong Cheng", "Quoc V. Le", "Tao Tu"],
    "year": 2026,
    "publication_date": "2026-08-27",
    "publication_type": "preprint",
    "venue": "arXiv",
    "doi": null,
    "arxiv_id": "2608.26701",
    "citation_count": null,
    "open_access_status": "green",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2608.26701",
    "code_url": null,
    "data_availability": "部分数据开放；完整Co-Scientist源码未公开，实验访问不等于全部实验配置开放。",
    "source_platforms": ["arxiv"],
    "source_urls": ["https://arxiv.org/abs/2608.26701", "https://arxiv.org/html/2608.26701v1"],
    "evidence_level": "B",
    "version_notes": "另篇执行扩展稿，未核实正式期刊发表；人装样、优化协议、完成物理表征。产出的研究agent不是已装回自我改进循环的Co-Scientist本身。arXiv页显示DataCite DOI待注册，故doi保留null。",
    "fetched_at": "2026-10-17"
  },
  {
    "title": "A multi-agent system for automating scientific discovery",
    "authors": ["Ali E. Ghareeb", "Benjamin Chang", "Ludovico Mitchener", "Angela Yiu", "Caralyn J. Szostkiewicz", "Dmytro Shved", "Gavin J. Gyimesi", "Jon M. Laurent", "Samantha M. Wright", "Muhammed T. Razzak", "Andrew D. White", "Silvia C. Finnemann", "Michaela M. Hinks", "Samuel G. Rodriques"],
    "year": 2026,
    "publication_date": "2026-05-19",
    "publication_type": "journal-article",
    "venue": "Nature",
    "doi": "10.1038/s41586-026-10652-y",
    "arxiv_id": "2505.13400",
    "citation_count": null,
    "open_access_status": "hybrid",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2505.13400",
    "code_url": "https://github.com/Future-House/robin",
    "data_availability": "原始RNA-seq数据BioProject PRJNA1464762；Robin和Finch公开代码及样例；完整运行依赖Edison服务。pdf_url为2025预印本而非2026正式版。",
    "source_platforms": ["nature", "arxiv", "futurehouse.org", "github", "openalex", "unpaywall"],
    "source_urls": ["https://www.nature.com/articles/s41586-026-10652-y", "https://arxiv.org/abs/2505.13400", "https://www.futurehouse.org/research/demonstrating-end-to-end-scientific-discovery-with-robin-a-multi-agent-system", "https://github.com/Future-House/finch"],
    "evidence_level": "A",
    "version_notes": "预印本2025-05-19；公司原文2025-05-20、2026-05-19更新。Nature Version of record为2026-07-01、期号日期为07-09。正式版增加原代细胞验证与作者，并明确人执行协议、预处理和复核。",
    "fetched_at": "2026-10-17"
  },
  {
    "title": "Kosmos: An AI Scientist for Autonomous Discovery",
    "authors": ["Ludovico Mitchener", "Angela Yiu", "Benjamin Chang", "Mathieu Bourdenx", "Tyler Nadolski", "Arvis Sulovari", "Eric C. Landsness", "Daniel L. Barabasi", "Siddharth Narayanan", "Nicky Evans", "Shriya Reddy", "Martha Foiani", "Aizad Kamal", "Leah P. Shriver", "Fang Cao", "Asmamaw T. Wassie", "Jon M. Laurent", "Edwin Melville-Green", "Mayk Caldas", "Albert Bou", "Kaleigh F. Roberts", "Sladjana Zagorac", "Timothy C. Orr", "Miranda E. Orr", "Kevin J. Zwezdaryk", "Ali E. Ghareeb", "Laurie McCoy", "Bruna Gomes", "Euan A. Ashley", "Karen E. Duff", "Tonio Buonassisi", "Tom Rainforth", "Randall J. Bateman", "Michael Skarlinski", "Samuel G. Rodriques", "Michaela M. Hinks", "Andrew D. White"],
    "year": 2025,
    "publication_date": "2025-11-04",
    "publication_type": "preprint",
    "venue": "arXiv",
    "doi": "10.48550/arXiv.2511.02824",
    "arxiv_id": "2511.02824",
    "citation_count": null,
    "open_access_status": "green",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2511.02824v2",
    "code_url": "https://github.com/EdisonScientific/kosmos-figures",
    "data_availability": "报告与链接分析notebook公开；部分输入数据待后续发表；仓库是作图和验证代码，不是完整核心引擎。",
    "source_platforms": ["arxiv", "edisonscientific.com", "github"],
    "source_urls": ["https://arxiv.org/abs/2511.02824", "https://arxiv.org/html/2511.02824v2", "https://edisonscientific.com/news/announcing-kosmos", "https://edisonscientific.com/news/bringing-trusted-research-into-kosmos-our-agreement-with-springer-nature"],
    "evidence_level": "B",
    "version_notes": "v2及产品发布2025-11-05；本次未核实正式期刊发表。2026-09-24公告为Nature portfolio内容授权，不是Kosmos论文发表于Nature。准确率统计单位是抽样陈述，六个月为合作专家时间估计。",
    "fetched_at": "2026-10-17"
  }
]
```
