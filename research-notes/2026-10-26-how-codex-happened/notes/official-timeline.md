# 《How Codex Happened》官方发布时间线核查

核查截止：2026-10-26。独立范围仅为所列模型与产品发布、桌面整合和使用量；未阅读 Every 主文、付费采访全文或代码架构。以下编号对应末尾官方源表。

## 研究笔记

本轮将“模型公布”“工具首发”“研究预览”“渠道开放”和“桌面整合”分开记账。依据是实际打开的 OpenAI、GitHub 官方原文及官方版本说明，不以搜索摘要替代正文，不把滚动页面最后更新时间当作历史推出日期。旧开发者版本说明入口本次重定向至 ChatGPT Learn；网页线性文本会丢失部分日期标题，因此另读取同一官方页面的原始 HTML，核对日期与对应事件的位置关系。访问日期统一为本轮截止日，不代表各产品在这一天仍维持首发时的状态。[S08]

2021 年首先应拆成两个节点。GitHub 于六月二十九日宣布 Copilot 技术预览，正文直接说明其由 OpenAI Codex 驱动，体现的是代码补全产品落地。OpenAI 的八月十日公告则把 Codex 介绍为 GPT-3 的后继者，宣布通过 API 开放私人测试；它还说此前已经把 Codex 集成进 Copilot。因此不能把八月的 API 公告写成 Copilot 首发，也不能把当年的 Codex 模型等同于后来同名的代理产品。GitHub 页面另有二〇二二年的更新日期，不能替代二〇二一年首发日期。[S01][S02]

二〇二四年九月十二日发布的是 o1-preview 与 o1-mini 所属的推理模型系列，并非 Codex CLI 或 Cloud。二〇二五年四月十六日，o3/o4-mini 公告正文同时介绍并宣布开放源码的 Codex CLI，定位是从终端运行的轻量编码代理。五月十六日的《Introducing Codex》才是云端软件工程代理的研究预览，使用基于 o3 优化的 codex-1，在独立云端环境中承担任务。这里应写“研究预览首发”，而非直接写成正式普遍可用。[S03][S04][S05]

八月还须区分模型与界面：八月七日是 GPT-5 发布，开发者公告说明它可用于 Codex CLI；八月二十七日版本说明才列出兼容 VS Code、Cursor、Windsurf 的 Codex IDE 扩展，以及订阅登录和本地、云端交接。九月十五日推出 GPT-5-Codex，它是进一步针对代理编码优化的 GPT-5 版本，覆盖 CLI、IDE 和云端代理。不能把九月专门模型的名称提前套在八月的基础模型上，也不能将 IDE 扩展首发与 GPT-5 首发合并成同一天。[S06][S07][S08][S09]

十一月存在需要保留的官方日期差异：GPT-5.1-Codex-Max 发布稿所对应系统卡明确标十一月十九日，但官方版本说明原始 HTML 将同名推出条目标在十一月十八日。两者都是一手材料，现有证据不足以判断是时区、上线批次还是编辑差异；建议以十九日作为发布稿节点，同时注记十八日版本说明记录，不武断宣布哪一个是全球统一上线日。十二月十八日 GPT-5.2-Codex 首发时，官方写的是先向付费 Codex 用户提供，API 将随后开放，不能概括为当日所有渠道同步开放。[S08][S10][S11][S12]

二〇二六年二月二日是 macOS Codex app 首发；发布页后来补有 Windows 更新，不能反推二月已双平台推出。二月五日 GPT-5.3-Codex 的首发与二月十二日 Spark 研究预览亦属不同事件。三月五日 GPT-5.4 在 Codex、ChatGPT 和 API 发布，是通用模型整合编码与专业工作的节点，不宜称为 GPT-5.4-Codex 专门模型。这些日期描述历史，不承担“截至十月仍为最新模型”的含义。[S08][S13][S14][S15]

主代理提供的知识工作页确有相关一手证据，但它的日期是六月二日，正文数字为 Codex 超过五百万周活跃用户，并非四千万。七月九日合入 ChatGPT 桌面则已获另一条明确确认：官方版本说明记录该日发布的 26.707，标题是 Codex 加入 ChatGPT 桌面，正文覆盖 macOS 与 Windows，并说明既有项目、设置和工作流保留。这可以支持“桌面应用整合”的事实，不能扩写成 CLI、IDE、云端入口均被取消，也不能用六月知识工作发布日替代七月整合日。[S08][S16]

截至本轮，文章所称“Codex 与 ChatGPT Work 合计四千万 WAU”仍未独立核实。已检索官方发布页、官方文档与相关关键词；社区用户转述负责人社交帖的线索不等于可核查的原帖，更不能证明统计周期、去重规则与口径。六月的五百万既不能用来否定十月可能增长，也不能外推或拼加成四千万。正文宜把后者归为文章声称并显著注明未获官方一手确认。六份付费采访的公开摘要不作为完整采访证据，历史访谈也不作为当前产品状态的证明。

## 发布事件表

日期指事件／原始发布记录，不指网页最近更新；有冲突或开放范围限制时另列。

| 日期 | 事件 | 官方证据 | 限制 |
|---|---|---|---|
| 2021-06-29 | GitHub Copilot 技术预览 | [S01]：当日公告；正文明确由 OpenAI Codex 驱动 | 不是 GA；页面 2022-02-23 更新不能当作首发 |
| 2021-08-10 | OpenAI Codex 公告与 API 私测 | [S02]：原文日期及私人测试说明 | 与六月 Copilot 首发分开；不是 2025 年云端代理 |
| 2024-09-12 | o1-preview／o1-mini 首批发布 | [S03]：原文日期与推出说明 | 推理模型节点，不是 Codex 产品首发 |
| 2025-04-16 | Codex CLI 公布并开放源码 | [S04]：“Codex CLI”段落 | 与同日 o3/o4-mini 发布相关，不是 Cloud |
| 2025-05-16 | Codex 云端代理研究预览 | [S05]：日期、codex-1、云端任务与推出说明 | 首发是 research preview，不是 GA |
| 2025-08-07 | GPT-5 发布；用于 Codex CLI | [S06] 系统卡日期；[S07] 开发者发布稿 | 基础 GPT-5，不是九月 GPT-5-Codex |
| 2025-08-27 | Codex IDE 扩展等更新 | [S08] 日期项及 “Late August update” | IDE、订阅登录、本地／云端交接是该批更新 |
| 2025-09-15 | GPT-5-Codex 推出 | [S09] 发布稿；[S08] 同名日期项 | 专门优化的模型；API 开放不应与此日混写 |
| 2025-11-19；另有 11-18 | GPT-5.1-Codex-Max 推出 | [S10][S11] 发布稿／系统卡；[S08] 版本说明标 11-18 | 官方日期相差一天；原因未核实，保留双记录 |
| 2025-12-18 | GPT-5.2-Codex 推出 | [S12] 发布稿；[S08] 对应条目 | 首发面向付费 Codex 用户；API 后续开放 |
| 2026-02-02 | Codex app 首发 | [S13] 发布稿；[S08] 日期项 | 首发 macOS；Windows 为后续更新 |
| 2026-02-05 | GPT-5.3-Codex 推出 | [S14] 发布稿；[S08] 日期项 | 不混同 02-12 Spark；API 首发仍写 coming soon |
| 2026-03-05 | GPT-5.4 在 Codex 等渠道发布 | [S15] 发布稿；[S08] 日期项 | 通用 GPT-5.4，不造 “GPT-5.4-Codex”名称 |
| 2026-06-02 | Codex for knowledge work | [S16] 原文日期；超过 5M Codex WAU | 是知识工作扩展和当日使用量，不是 40M 合计 |
| 2026-07-09 | Codex 加入 ChatGPT 桌面应用 | [S08]：该日条目、版本 26.707，正文明确 macOS／Windows | 已独立确认桌面整合；不证明其他入口终止 |
| 截至 2026-10-26 | “40M 合计 WAU”声明 | 未找到可打开并确认该数字及口径的官方一手原文 | **未独立核实**；不猜原帖／发布稿 URL，不外推 |

## 官方源表

所有正文均实际通过网页工具打开。S08 部分日期在网页线性提取中缺失，另用同一官方页面原始 HTML 核对；源表保留实际重定向目标。日期是发布日或历史条目日，不是抓取日。OpenAI 发布稿中的后补更新另行识别。

| 编号 | 标题 | 日期 | 完整 URL |
|---|---|---|---|
| S01 | Introducing GitHub Copilot: your AI pair programmer | 2021-06-29；另有 2022-02-23 更新 | https://github.blog/news-insights/product-news/introducing-github-copilot-ai-pair-programmer/ |
| S02 | OpenAI Codex | 2021-08-10 | https://openai.com/index/openai-codex/ |
| S03 | Introducing OpenAI o1-preview | 2024-09-12 | https://openai.com/index/introducing-openai-o1-preview/ |
| S04 | Introducing OpenAI o3 and o4-mini | 2025-04-16 | https://openai.com/index/introducing-o3-and-o4-mini/ |
| S05 | Introducing Codex | 2025-05-16；含后续推出范围更新 | https://openai.com/index/introducing-codex/ |
| S06 | GPT-5 System Card | 2025-08-07 | https://openai.com/index/gpt-5-system-card/ |
| S07 | Introducing GPT-5 for developers | 2025-08-07 | https://openai.com/index/introducing-gpt-5-for-developers/ |
| S08 | ChatGPT & Codex changelog | 滚动页；逐条采用所列历史事件日期 | https://learn.chatgpt.com/docs/changelog |
| S08 旧入口 | Codex changelog（本次重定向至 S08） | 不以迁移／更新时间代替条目日期 | https://developers.openai.com/codex/changelog |
| S09 | Introducing upgrades to Codex | 2025-09-15；含后补更新 | https://openai.com/index/introducing-upgrades-to-codex/ |
| S10 | Building more with GPT-5.1-Codex-Max | 2025-11-19；须与 S08 的 11-18 并列 | https://openai.com/index/gpt-5-1-codex-max/ |
| S11 | GPT-5.1-Codex-Max System Card | 2025-11-19 | https://openai.com/index/gpt-5-1-codex-max-system-card/ |
| S12 | Introducing GPT-5.2-Codex | 2025-12-18 | https://openai.com/index/introducing-gpt-5-2-codex/ |
| S13 | Introducing the Codex app | 2026-02-02；2026-03-04 补 Windows 更新 | https://openai.com/index/introducing-the-codex-app/ |
| S14 | Introducing GPT-5.3-Codex | 2026-02-05 | https://openai.com/index/introducing-gpt-5-3-codex/ |
| S15 | Introducing GPT-5.4 | 2026-03-05 | https://openai.com/index/introducing-gpt-5-4/ |
| S16 | Codex for knowledge work | 2026-06-02 | https://openai.com/index/codex-for-knowledge-work/ |

## 复核定位与证据边界

- S01 首发日期在页面 `datePublished`、`article:published_time` 和 `<time datetime="2021-06-29">` 中一致；`dateModified` 与另一 `<time>` 对应 2022-02-23。
- S08 原始 HTML 中可定位到相邻文本：`2025-08-27`／`Late August update`；`2025-09-15`／`Introducing GPT-5-Codex`；`2025-11-18`／`Introducing GPT-5.1-Codex-Max`；`2026-07-09`／`Codex joins the ChatGPT desktop app`。这些是条目日期，不是整页更新时间。
- S11 标题上方直接显示 `November 19, 2025`。S08 与 S11 的差异如实保留，不自行补造时间或原因。
- S16 标题上方直接显示 `June 2, 2026`，正文为 Codex `more than 5 million weekly active users`。该数字仅对应 Codex；未写 Codex 与 ChatGPT Work 合计四千万。
- 官方社区中的普通用户转述、二手报道、搜索摘要及未打开的社交原帖均未列为使用量证据。未独立核实表示本次检索未找到满足标准的材料，并非证明该声明为假。
