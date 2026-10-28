# After42：公开资源与贡献工作流核查

访问日期：2026-10-28（UTC）。核查过程时钟采样：`2026-10-28T08:04:01Z`。

范围：独立核查官方仓库实际发布物、一个 `math-1` 公开记录、贡献/署名/使用规则；不复核文章主实验、样本统计、模型比较或数学结论。所有来源均为 After42 官方站点、指定官方 GitHub 仓库及其 GitHub API，或该站点直接链接的 Creative Commons 官方许可说明。网页读取工具对部分站点页面失败，正文与数据通过只读 `curl -fsSL` 核实；不把网页工具失败当作资源不存在。

本次只创建本文件。未下载模型、调用付费 API、联系作者、发 issue、修改远端站点、提交或推送。仅在内存中读取网页/公开数据，未另存完整站点或题库。

## 五项核心结论

1. **官方事实：指定 GitHub 仓库目前只有 README。** `main` 对应 commit `325e89ca135cd43bdfa664020b543845d328525a`，完整递归文件树只有 `README.md`。不是已开源 agent 系统，也不是已发布静态站点源码仓库；README 明示代码尚待发布。[固定文件树](https://api.github.com/repos/after42/After42/git/trees/325e89ca135cd43bdfa664020b543845d328525a?recursive=1)、[固定 README](https://github.com/after42/After42/blob/325e89ca135cd43bdfa664020b543845d328525a/README.md)
2. **官方事实：官网确有可读取的公开问题记录及专家评语。** `questions.html` 加载 `data/questions_data.js` 与 `assets/site.js`；这是已部署网站的静态数据和浏览/渲染代码，不能据此称训练系统、生成 agent、prompts 或完整实验评测已开源。[题库入口](https://after42.org/questions.html)、[公开数据文件](https://after42.org/data/questions_data.js)、[浏览器代码](https://after42.org/assets/site.js)
3. **官方事实：`math-1` 仍是开放问题。** 内部记录 ID 为 `ppt70-q29`，记录包含问题正文、四个补充章节、来源摘要、模型标记及两条匿名专家评语；`status` 为 `open`，该记录无 `status_link`。题目被认为值得研究，不代表已解决或已有解决该题的论文发表。[题卡永久链接](https://after42.org/questions.html#q=math-1)、[数据文件第 15–35 行](https://after42.org/data/questions_data.js)
4. **官方事实：贡献有四个入口，题卡署名与项目作者名单分别表述。** 可设定子领域目标、评价题目、研究题目、建设系统；研究者可选择在题卡上因具体贡献实名署名并附工作链接。页面没有承诺参与即获得项目作者身份。[贡献入口](https://after42.org/contributor.html#contribute)、[署名规则](https://after42.org/contributor.html#terms)
5. **官方事实：问题及专家评论适用 CC BY 4.0；引用按项目加问题编号。** 应给 After42 及题卡中实名贡献者署名，题卡提供项目 BibTeX 和问题编号引用。该说明不能扩张为尚未发布的系统代码或模型权重许可。[项目使用规则](https://after42.org/contributor.html#terms)、[项目引用](https://after42.org/contributor.html#cite)、[CC 官方说明](https://creativecommons.org/licenses/by/4.0/)

## 1. GitHub 实际发布内容与固定版本证据

### 1.1 版本定位

以下均为 2026-10-28 从 GitHub 官方 API 读取的字段；不以搜索摘要或网页星标数推定版本。

| 项目 | 核查结果 | 官方来源 |
| --- | --- | --- |
| 仓库与默认分支 | `after42/After42`；`main`；公开仓库 | [仓库元数据](https://api.github.com/repos/after42/After42) |
| 当前 commit | `325e89ca135cd43bdfa664020b543845d328525a` | [main commit](https://api.github.com/repos/after42/After42/commits/main)、[固定 commit](https://github.com/after42/After42/commit/325e89ca135cd43bdfa664020b543845d328525a) |
| commit 时间与消息 | `2026-10-27T04:44:01Z`；`Add README: code coming soon` | [固定 commit 元数据](https://api.github.com/repos/after42/After42/commits/325e89ca135cd43bdfa664020b543845d328525a) |
| commit 指向的 tree | `f57937fbd00580d27f90494b4d485eb9ab3d0db6` | 同上，`commit.tree.sha` |
| 唯一文件 | `README.md`；540 bytes；blob `639a243fe4997f7ab94df21427271ec6d2e3f31b` | [递归文件树](https://api.github.com/repos/after42/After42/git/trees/325e89ca135cd43bdfa664020b543845d328525a?recursive=1) |
| 分支、tag、release | 分支只有 `main`；tags 和 releases 均返回空数组 | [branches](https://api.github.com/repos/after42/After42/branches)、[tags](https://api.github.com/repos/after42/After42/tags)、[releases](https://api.github.com/repos/after42/After42/releases) |
| 仓库元数据中的许可/Pages | `license: null`；`has_pages: false` | [仓库元数据](https://api.github.com/repos/after42/After42) |

文件树一手响应的相关字段摘录；没有省略其他文件项，因为 `tree` 数组确实只有这一项：

```json
{
  "tree": [
    {
      "path": "README.md",
      "mode": "100644",
      "type": "blob",
      "sha": "639a243fe4997f7ab94df21427271ec6d2e3f31b",
      "size": 540
    }
  ],
  "truncated": false
}
```

README 第 3 行的短摘录：

> We will release the code soon. Stay tuned!

来源：[固定 raw README](https://raw.githubusercontent.com/after42/After42/325e89ca135cd43bdfa664020b543845d328525a/README.md)。README 链接的是研究文章、官网题库和官网首页，不包含安装/运行入口。

### 1.2 按资源类别判定，避免混淆发布位置

| 资源 | 指定 GitHub 仓库：固定 commit 的实际文件 | 官网上本次核查到的内容 |
| --- | --- | --- |
| 可直接复现的 agent 框架/运行依赖 | 未发布；只有 README | 题库前端不是生成/求解 agent；所查资源不能据此复现该工作流 |
| 可运行的 prompts/配置 | 未发布 | `math-1` 记录没有 prompts/完整模型调用配置；来源摘要不是 prompt |
| Question records | 未放入该仓库 | `data/questions_data.js` 包含公开题卡记录，`math-1` 可直接解析 |
| 专家评价 | 未放入该仓库 | `math-1.expert_comments` 是两条匿名文字评语；不是该实验完整评测发布物的证据 |
| 静态站点源码 | 未放入该仓库 | 部署站点公开返回 HTML、数据 JS 和浏览器渲染 JS |
| 训练代码、模型权重或训练数据完整包 | 未发布 | 本次查到的问题集合不构成这些材料已发布的证据 |

仓库列的判断依据是完整且未截断的文件树。官网列只说明本次实际读取的资源，不断言作者没有内部系统或其他未核查资源，也不把“尚待发布”解读为未来必定按某一时间/许可发布。

`questions_data.js` 第 1 行提及生成脚本路径 `scripts/sync_questions.py`，但指定仓库文件树中没有该脚本。**脚本路径出现在注释中，不等于脚本已发布。** 官网源码路径是 HTTP 资源路径，不能映射为上述 GitHub commit 内的文件。

## 2. `math-1` 公开记录结构

### 2.1 定位与精简字段摘录

官方入口：[https://after42.org/questions.html#q=math-1](https://after42.org/questions.html#q=math-1)。

实际数据：[https://after42.org/data/questions_data.js](https://after42.org/data/questions_data.js)，全局对象 `window.QG_DATA`；外层 `generated` 为 `2026-10-26`。`math-1` 位于 `fields` 中 `slug=math` 的对象，继而位于 `subfields` 中 `slug=probabilistic_phase_transitions` 的 `questions` 数组。该题原始记录为本次响应第 14–36 行，识别字段位于第 15、35 行。

以下为该题真实字段的精简投影，不是完整记录：

```json
{
  "id": "ppt70-q29",
  "site_id": "math-1",
  "title": "A Poisson Triangle Representation of Sparse Geometry",
  "base_model": "gpt-6-astra xhigh",
  "status": "open"
}
```

上述模型名称只是官方记录的原样字符串；本核查未验证模型可获取性、API 名称或其与其他公开模型名称的对应关系。

### 2.2 字段与页面展示的准确对应

| 记录字段 | 类型/内容 | 公开展示或复现边界 |
| --- | --- | --- |
| `id`、`site_id` | 内部 ID `ppt70-q29`；公共 ID `math-1` | 链接使用 `site_id`，不能用公共编号推定原始实验样本顺序 |
| `title` | 题目名称字符串 | 展示为题卡标题 |
| `problem_statement` | 含 TeX 的 Markdown 字符串 | 展示为 Problem；不是 `sections.problem` |
| `sections` | 四个字符串键：`background`、`intuition`、`related`、`routes` | 页面另加 Problem，共五个正文折叠区；求解路线/证据不是已完成解答的认证 |
| `provenance` | `origin_run`、`origin_title`、`revision` | 来源/修订摘要，当前题卡不显示；不是逐轮轨迹、工具日志或完整运行包 |
| `base_model` | 上述模型字符串 | 展示为题卡模型元数据 |
| `status` | `open` | 该题无 `status_link`；不能称已解决 |
| `expert_comments` | 两个字符串组成的数组 | 页面以匿名专家评论展示；本题记录没有评审者实名或数值评分字段 |

字段内容来源：[数据文件](https://after42.org/data/questions_data.js)。展示逻辑来源：[assets/site.js](https://after42.org/assets/site.js)，本次版本中第 39–43 行处理评论，第 48–61 行处理状态，第 64–79 行处理公共 ID/永久链接，第 214–254 行处理正文、元数据和引用。官网脚本注释声明公共编号按领域分配、不会复用；本次未取得生成脚本验证其实现。

`provenance.origin_run` 为 `20260714_ppt150_batch14`。`revision` 摘要称最初来自 GPT 5.6 Sol 的运行，之后通过 GPT 6 Astra 的求解与发展轮次修订，并写有运行标识 `20260913_ppt100_astra_xhigh_refine2_rankonly_rounds_zihao_resume`。这些是**官方公开的来源标签**，没有附上该 run 的 prompt、对话、检查点、工具输入输出或完整逐轮改写记录。[数据文件第 24–29 行](https://after42.org/data/questions_data.js)

### 2.3 评论、状态与作者身份的边界

官方题库说明题目是供研究者追求、修订或质疑的邀请；专家评论匿名。页面同时明确：除专家评论以外，题卡内容由模型生成。[questions.html 第 42–45 行](https://after42.org/questions.html)

`math-1` 的两条文字评语都可在原始 JS 中读取。这里仅确认其存在、结构和匿名展示，不重复主代理对评价内容或实验结果的分析。`assets/site.js` 明示评论顺序由编辑安排，因而不能从数组顺序推断评审者身份。[评论显示函数及注释](https://after42.org/assets/site.js)

**核查结论（证据解释，不是数学验证）：** 本题当前官方记录为 `open`；正文中的调查路线、条件式推论和专家兴趣评价，不构成已经解决的证明。公开发布“问题题卡”也不等于解决该题的学术论文已发表。本核查不判断数学正确性、新颖性或是否已有未链接的外部工作。

## 3. Contributor 页：贡献、维护、署名与使用

### 3.1 官方说明的四类贡献

来源：[contributor.html#contribute](https://after42.org/contributor.html#contribute)，本次 HTML 第 81–98 行。

1. 设定子领域目标：说明希望获得的理解及推进该目标的问题类型；团队与贡献者一起建立文献 research pack，再朝目标提出和发展问题。
2. 评价问题：以领域判断评价是否有趣、是否值得推进；这些评价用于衡量兴趣，并指导提出和发展问题。
3. 研究公开问题：可独立或借助 AI 系统推进；回报进展、已有工作或错误，以便更新题卡。
4. 建设系统：协助开发 agents、评价工具、数据集发布及网站。**这是合作邀请，不是这些组件已经发布的证明。**

官方联系路径是 `niefan@stanford.edu`；报告既有工作或题目错误也可用 [仓库 issues](https://github.com/after42/After42/issues)。本核查仅记录路径，没有发邮件或 issue。[contributor.html#contact，第 100–103 行](https://after42.org/contributor.html#contact)

### 3.2 维护与署名

来源：[contributor.html#terms](https://after42.org/contributor.html#terms)，本次 HTML 第 105–118 行。

- 官方维护承诺：更新题目状态、发布更正、随着领域扩展加入新评审题目；更正和进展通过联系入口报告。
- 项目作者名单以外，研究者可以选择在具体题卡实名署名：设定目标、评论、修改问题或取得进展，并附工作链接。
- 页面规定每张题卡记录产出问题的模型；`math-1` 的模型字段已核实。
- **证据边界：** “可选择题卡署名”与“项目作者名单”被分开说明。页面未承诺自动共同作者身份、贡献评分/排序、署名审批时限或自动接收提交。不能自行补出这些规则。
- 题库默认的匿名专家评论与可选择实名的题卡贡献署名可以并存；不能把团队成员或项目作者名单映射为 `math-1` 的两位评审者。

### 3.3 许可与引用

After42 的官方许可段明确限定对象为**问题和专家评论**，采用 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)，允许在给 After42 以及题卡中实名贡献者署名的前提下复用与改编；题卡或项目段提供引用。[contributor.html#terms，第 110–118 行](https://after42.org/contributor.html#terms)

CC 官方摘要进一步说明：分享、改编可包括商业用途；需要适当署名、提供许可链接、标明修改，不能暗示原许可人背书，也不能增设限制许可权利的法律/技术措施。摘要不是完整法律条文；完整文本在 [CC BY 4.0 legal code](https://creativecommons.org/licenses/by/4.0/legalcode.en)。这里是对官方规则的记录，不对任何具体用途作法律判断。

实际引用机制：[assets/site.js](https://after42.org/assets/site.js) 第 82–100、246–252 行和 [contributor.html#cite](https://after42.org/contributor.html#cite) 第 130–143 行。

- 当前项目 BibTeX key 为 `nie2026after42`，类型为 `@misc`；同一项目 BibTeX 用于各个题目。
- `math-1` 的题卡引用文本由 `citeFor(q)` 生成：

```tex
\citep[question math-1]{nie2026after42}
```

- `contributor.html` 第 122–129 行保留了旧 `problemgen2026` 引用，但位于 HTML 注释内，并非当前展示引用；不能从去标签后的 curl 文本误选旧项。
- 因问题内容及评论明确获得许可，不能反过来推定模型权重、生成 agent、全部站点软件或未来发布代码适用相同许可。仓库当前没有 LICENSE 文件，API 的 `license` 也为 `null`；本次不为未发布物推定许可。

## 4. 一手证据索引与网页版本边界

以下全部在 2026-10-28（UTC）访问。静态网页/JS 未提供可核实的 GitHub commit 对应关系；上述 GitHub commit **只固定该仓库 README，不能给官网题库或 contributor 页面作版本背书**。行号是本次 HTTP 响应的行号；网页以后更新时，行号与哈希可能变化。

| 一手资源/精确路径 | 本次用途/关键定位 | 响应 bytes | SHA-256 |
| --- | --- | ---: | --- |
| [官网 `/questions.html`](https://after42.org/questions.html) | 第 42–45 行声明用途/匿名评论/模型生成；第 61–64 行数据与前端加载 | 3665 | `1234cf0d3dcaa1b113a1215b97fcdb83883ff19509a9018f896d73b6dd0042cf` |
| [官网 `/data/questions_data.js`](https://after42.org/data/questions_data.js) | 外层 generated；`math-1` 第 14–36 行 | 322189 | `84cee646c0c48bc2e2812eb5ba8cea4daed06de5c229812dce4c9a9e60a4cfaa` |
| [官网 `/assets/site.js`](https://after42.org/assets/site.js) | 展示/匿名评论/状态/永久链接/项目引用逻辑 | 15755 | `0ec8c0a302a7559e50f1733acf72ea88d1b603759f67c5950765ca5b1373dc59` |
| [官网 `/contributor.html`](https://after42.org/contributor.html) | 第 81–118 行贡献、联系、维护、许可和署名；第 130–143 行现行引用 | 12368 | `c71b74d3d8f851738b7623f241278cd24c6d5e972789771a2864afb72ec018d2` |
| [仓库固定 `/README.md`](https://raw.githubusercontent.com/after42/After42/325e89ca135cd43bdfa664020b543845d328525a/README.md) | commit 内唯一文件；第 3 行代码待发布 | 540 | `ddef6db1c6e63058718b5e9c3df134fa014cac0b65b4c166df491b82bbec0818` |

附加数据指纹：从 JSON 对象中选取完整 `math-1` 记录后，以 Node `JSON.stringify(q)` 的 UTF-8 字节计算 SHA-256，结果为 `2544700c78df67f018f70ada1371c42382e9f0983aa47cae467688d0d13df803`。这是该序列化方式的记录指纹，不是原文件切片或标准化 JSON 的哈希。

官网题库 footer 标记 `updated 2026-10-25`，contributor footer 标记 `updated 2026-10-20`，数据文件标记 `generated 2026-10-26`。这些是不同资源的自报日期，不是本次访问日期，也不是同一发布版本号。[题库](https://after42.org/questions.html)、[贡献页](https://after42.org/contributor.html)、[数据文件](https://after42.org/data/questions_data.js)

给定文章 URL：[https://after42.org/research/can-ai-propose-interesting-research-questions/](https://after42.org/research/can-ai-propose-interesting-research-questions/)。本次只从文章入口导航提取仓库、题库、`math-1` 与 contributor 链接，不重做主实验阅读或结果分析。

## 5. 建议：可执行但未代用户采取的后续工作

本节是核查者建议，**不是 After42 官方要求或已完成事项**。

- 主报告可写成：“开放了专家评审的问题集合与评论；指定代码仓库截至访问日仍为占位 README。”避免写“整套训练/agent 系统已开源”，也避免把仓库称为静态站点源码。
- 如需复现生成与修订流程，仍应另行取得并固定 agent 代码、完整 prompts、research pack、模型/工具配置、运行环境、逐轮记录及完整评价资料。现在可以复查公开题卡，不等于现在可以从输入重跑原系统。
- 复用问题或评语时，建议保留题号、原始 URL、访问日期、题卡实名贡献者（如有）、许可链接及修改说明；用当前项目 BibTeX 加 `question math-1`，不把匿名评语归给具体团队成员。
- 如之后出现官方修订、解答或进展链接，再更新状态；本次 `open` 不应被“问题更有趣”或“有调查路线”替换为 `solved`/已发表。
- 联系团队、提交更正/issue、参与研究或建立新的自动化流程，均应作为另行授权的下一步。本次没有代为执行。
