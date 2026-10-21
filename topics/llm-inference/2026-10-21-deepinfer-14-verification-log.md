# DeepInfer 调研 14：核实记录与方法说明

> 本文件说明报告中的数据和结论是怎么来的、核实到了什么程度、有哪些局限，供评审和复核使用。日期：2026-10-21。

## 1. 资料来源分级

| 等级 | 来源 | 在报告中的标记 |
|---|---|---|
| 一手·代码 | DeepInfer 源码（commit `774ce24`，2026-10-20），只读审阅，**未运行** | [源码]，并附 `path:line` |
| 一手·官方文档 | DeepInfer `README.md`、`doc/inference_deployment_qa.md`，以及 HF 上的 `deepseek-ai/DeepSeek-V4.1-Flash-1017/config.json` | [Q&A] |
| 一手·竞品 | 各项目的 GitHub API 数据（star、release）、release notes、README、官方博客 | 附 URL |
| 学术 | arXiv 论文，全部 34 篇已逐一核实标题（见 [arxiv_check.tsv](2026-10-21-deepinfer/arxiv_check.tsv)） | 附 arXiv ID |
| 推算 | 根据以上资料推导，并给出算式 | [估算] |
| 推测 | 合理推断，没有直接证据 | [推断] |
| 未核实 | 没有找到一手来源 | [未核实] |

## 2. 方法

- **源码审阅**：主研究员亲自审阅，另由 3 个子 agent 并行精读引擎、模型和并行三个部分，之后又由 9 个专题 agent 逐专题深挖。所有 `path:line` 引用都由各 agent 回查确认过；主编对关键结论做了抽查（见 §3）。
- **竞品资料**：4 个调研 agent 用 `gh api` 和 `curl` 实时抓取，主编对核心数字二次核对，包括版本号、发布日期、V4.1 支持情况和 Ascend 950 支持情况。
- **数据快照**：`数据快照_GitHub仓库统计.csv` 于 2026-10-21 约 05:05 UTC 抓取。
  - 正文中的 star 数为当日更早时候的读数，存在小幅差异。例如 DeepInfer 正文写 14,009，快照为 14,539，因为该仓库仍处于快速增长期。
- **没有做的事**：没有在任何硬件上运行任何引擎；报告中所有性能数字都来自各方自己公布的结果。

## 3. 主编抽查与更正记录

| # | 内容 | 抽查方式 | 结果 |
|---|---|---|---|
| 1 | V4.1 分页 KV 体积 | 读 `hugging_face/config.py:62-85`、`cache_engine.py:46-80, 150-187, 377-460`，并下载 HF config.json | **更正为 890 B/token**（原总报告写 1.16 KB，T3 写 1,162 B，T4 写 3.4 KB，均有误） |
| 2 | DSpark 的原创方 | 读 LMSYS 博客原文 | DeepSeek 原创，SGLang 按论文实现，纠正了二手说法 |
| 3 | SGLang 是否支持 Ascend 950 | 读 SGLang README | 支持（950PR/DT），纠正了"仅 DeepInfer 支持"的说法 |
| 4 | vLLM 和 SGLang 的 V4.1 支持时间 | 读 v0.32.0 和 v0.5.22 的 release notes | 分别为 10-18 和 10-19，均早于 DeepInfer 开源 |
| 5 | prefill 二次代价模型是否开源 | 全仓 `grep compute_quota` | 无，属于 Q&A 描述但未开源的部分 |
| 6 | FlashMLA 是否移除了 Hopper 支持 | 读 FlashMLA README | 属实（2026-09-30 的 breaking change） |
| 7 | Hadamard 变换的作用范围 | 读 `cuda/fused_transform.py:51-52` | 属实，有 `assert head_dim == 128` |
| 8 | Crossflow 与 DeepInfer 的关系 | 读 arXiv 2609.27085 摘要和发表日期 | 该论文发表于 DeepInfer 开源之前，不存在"点名 DeepInfer"，纠正了专题 agent 小结中的说法 |
| 9 | 单专家权重体积（T2） | 用 HF config 的维度重新计算 | 更正为约 18.8 MB，层数按 40 层计 |
| 10 | PD 带宽（T1） | 按 README 的 prefill 吞吐重新计算 | 更正为约 10 Gbps 量级，结论不变 |
| 11 | 引用的 34 篇 arXiv 论文 | 逐一 curl 核对标题 | 全部存在 |
| 12 | 引用的 56 个网页链接 | 检查 HTTP 状态 | 1 个确实失效（OCP MX 规范页面），已替换为规范 PDF 地址；其余失败均为提取时截断造成 |

## 4. 已知局限

1. **性能结论都来自各方公布的数字，口径不一**，不能直接横向比较。补救方案见 [03_同口径基准测试方案.md](/topics/llm-inference/2026-10-21-deepinfer-03-benchmark-plan.md)。
2. **生产系统的细节**（KV 存储、负载均衡、PD 传输）**只能依据 Q&A 的文字描述**，没有代码可供核实。
3. **未核实项**：DSpark 原始论文、3FS 中 10+2 纠删码的实现、Ascend 950 官方规格、DeepSeek API 当前的硬盘缓存价格、TRT-LLM 中 DSpark 的进展。
4. **专题文档由多个 agent 分别撰写**，风格和详略不完全一致。全局勘误以 [00_专题索引与勘误.md](/topics/llm-inference/2026-10-21-deepinfer-04-topics-index-errata.md) 为准；各专题文件开头也附有对应的勘误。
5. **一次操作失误**：执行过程中，曾在 DeepInfer 仓库目录内误触发 `uv run`，它尝试构建并执行了仓库的 `setup.py`。构建在导入阶段失败，没有安装任何内容，生成的 `.venv` 已删除，仓库状态已恢复干净。该操作不影响任何研究结论。

## 5. 复现方式

- **源码**：`git clone https://github.com/deepseek-ai/DeepInfer && git checkout 774ce24`
- **数据快照**：重新运行 `gh api repos/<owner>/<repo>` 获取最新值。
- **文档构建**：`scripts/build_pdf.py` 负责合并 Markdown 并做字符替换，然后用 pandoc + xelatex（Noto CJK 字体）生成 PDF。
