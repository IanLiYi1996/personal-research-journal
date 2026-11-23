# 管理层深度调研版

资料核实日：2026-10-23。面向没有相关基础的管理者，同时保留实施所需的技术边界和来源。

## 阅读文件

- 完整 PDF：封面、可点击目录、页码、书签、可复制中文和八张图。
- [可编辑正文](Ontology_企业知识与AI应用_深度调研报告.md)：决策摘要、16 章及 4 个附录。参考资料在文末，与正文数字引用对应。
- 网页版本：应与 `print.css`、`images/` 一同保留。

## 研究底稿

- [数据库产品](研究底稿/01_产品能力核实.md)：版本、约束、推理、维护与成本组成；正文不是性能排名。
- [公开案例](研究底稿/02_公开案例与证据.md)：四个案例，区分客户、维护方和供应商证据。
- [论文研究](研究底稿/03_LLM与本体研究证据.md)：27 条扫描记录，六篇核心全文及一篇近期补核全文；不是穷尽性系统综述。
- 论文元数据：保留版本、任务、全文读取与未取得的引用数。

正文中的企业产品、版本和功能例子是虚构的。没有做生产部署、贵公司数据测算或云成本报价。材料包中另有 [本地技术测试记录](../technical-example/README.md)，不可扩大解释为所有数据库或业务系统已验收。

最终编稿复查时，ArangoDB 3.12 的图和模式校验链接改用已读取正文的官方新文档地址；产品底稿保留初始检索地址，反映资料整理过程。Neptune 的 ID 约束引用直达兼容性说明的 schema constraints 段落，而非通用文档入口。

## 图及排版源码

- `build_diagrams.py`：八张 SVG／PNG，复用上一版的 `../管理层阅读版/build_assets.py` 绘图组件。
- `build_pdf.py`、`print.css`：Markdown 转 HTML 与带结构标签的 PDF。
- `check_report.py`：检查引用、内部链接、页边界、图片、中文提取和书签，生成排版检查缩略图及 JSON。
- `排版检查/检查结果.json`：最终页数及检查结果。

## 本地复现

本次使用系统 Python 3.9.25 的隔离环境和系统 Cairo／Pango。中文字体为 Noto Sans CJK SC；`check_report.py` 的图览使用 Linux 字体路径，换系统后需调整。

```bash
python3 -m venv /tmp/ontology-report-render
/tmp/ontology-report-render/bin/python -m pip install -r requirements-render.txt
/tmp/ontology-report-render/bin/python build_diagrams.py
/tmp/ontology-report-render/bin/python build_pdf.py
/tmp/ontology-report-render/bin/python check_report.py
```

这些是排版复现命令，不安装数据库，不修改生产资源。Cairo／Pango 和 CJK 字体需系统提供；requirements 仅固定直接 Python 依赖，不是生产供应链锁文件。最终交付保留 Markdown、HTML、CSS、图像和 PDF，未覆盖旧的六页简明版。
