# How Codex Happened 完整调研交付

资料核实截止：2026-10-26。原文为 Dan Shipper 于 2026-10-15 在 Every 发布的《How Codex Happened》。

## 阅读文件

- `How_Codex_Happened_完整调研报告.pdf`：阅读版，带可点击目录、书签与来源链接。最终页数与检查结果见 `qa/pdf-check.json`。
- `How_Codex_Happened_完整调研报告.docx`：可编辑 Word 版，含标题层级、表格、动态目录和页码字段。兼容 Word 的软件打开后会计算目录；如未自动更新，可全选后按 F9。
- `How_Codex_Happened_完整调研报告.md`：可编辑源稿，保留正文和数字引用。
- 同名 `.html` 与 `print.css`：网页阅读版。

正文约 8,300 汉字，10 章、决策摘要与 3 个附录。39 条来源记录包括 33 个公开正文／发布／文档／代码来源，以及 6 个仅核查访问条件、未取得完整内容的 Every 访谈入口。不能将它们统称为 39 份完整资料均已读。

## 内容与证据口径

报告补充官方发布时间线、公开执行架构、权限与上下文、长任务评测、知识工作扩展及四周企业试点建议。明确区分原文叙述、官方历史事实、查阅日实现与独立分析。

关键校正包括：

- CLI 公布早于 Cloud 研究预览；本地转向不等于 CLI 首次出现。
- Max 发布稿／系统卡与版本说明存在一天日期差异，保留双记录。
- 2026-07-09 桌面整合有官方版本记录；4,000 万合计 WAU 未独立核实。
- 早期不内置完整编辑器，不等于今天永久禁止手动修改所有文件。
- 六份付费采访没有取得全文；不绕过访问限制，不重新分发原文。
- 代理运行时长不是固定成功概率下的人类任务时间跨度。

本报告没有运行编码代理、下载模型或复现厂商性能。源码是固定公共提交，不保证与商业服务的部署版本一致。

## 研究底稿与检查

`notes/`：

- `official-timeline.md`：历史发布、渠道范围、日期差异与统计口径。
- `public-architecture.md`：执行内核、协议、沙箱／审批、工作副本和上下文。
- `source-access.md`：公开正文与六份会员采访的访问边界。

`qa/` 保存 PDF 与 Word 的结构、引用绑定及模式校验。PDF 按全部页面缩略图与重点页面做视觉检查；Word 的 HTML／浏览器预览不代替实际 Word 的最终分页。

## 重建

需要 Pandoc、officecli、Noto Sans CJK SC，以及 Python 的 markdown、weasyprint、pymupdf。

```bash
uv run --with markdown --with weasyprint --with pymupdf python build.py --format pdf
python build.py --format docx
```

脚本解析来源编号并验证链接绑定，检查 Word 模式、页码字段及泄漏占位符。`.gitignore` 排除缓存和大体积排版截图，不排除最终 PDF、Word、HTML 与 Markdown。

## 提交准备

当前目录不属于 Git 仓库，尚未执行提交或推送。需要用户指定目标仓库后，才能检查对应规则、分支、远端与既有改动。

建议提交本目录的正文、交付物、README、三个研究底稿、排版源码和核心 QA JSON／校验记录；排版截图、Python 缓存与原始 HTML 浏览器预览不必提交。建议提交信息：`docs: add How Codex Happened research report`。
