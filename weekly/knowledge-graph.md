# 知识图谱洞察报告（自动生成）

- **生成方式**: `scripts/wiki_graph.py`（方法参考 [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) 的 4 信号相关性模型 + Louvain 社区发现 + Graph Insights）
- **节点**: 104 篇笔记　**边**: 160　**社区**: 37
- **信号权重**: 直接链接 ×3.0 / 共享文献来源 ×4.0 / Adamic-Adar ×1.5 / 同类型 ×1.0

> 本报告由脚本读取仓库现有 md 与 `references.bib` 生成，不修改任何笔记。

## 1. 主题社区（Louvain 自动聚类）

| # | 规模 | 内聚度 | 构成 | 核心笔记（按连接数） |
|---|---|---|---|---|
| 13 | 20 | 0.2 | research-notes×10, tech-blogs×5, reddit-digests×5 | `2026-07-21-raschka-llm-architecture-comparison.md`<br>`2026-W31-reddit-hot.md`<br>`2026-07-29-hf-daily-papers-jul28-29.md` |
| 7 | 16 | 0.267 | research-notes×13, topics×1, papers×1, tech-blogs×1 | `2026-07-20-long-horizon-agents.md`<br>`2026-07-20-blog-reasoning-effort.md`<br>`2026-07-20-llm-long-context.md` |
| 3 | 14 | 0.473 | research-notes×8, topics×6 | `2026-06-12-agent-evolution.md`<br>`2026-06-12-ai-frontier-comprehensive.md`<br>`2026-05-28-hf-daily-papers-may16-28.md` |
| 5 | 7 | 0.333 | research-notes×3, papers×3, topics×1 | `2026-05-08-cot-mechanism-mini-survey.md`<br>`2026-05-08-graph-structures-meet-llms.md`<br>`2026-07-08-blog-global-workspace.md` |
| 29 | 7 | 0.476 | topics×7 | `2026-04-15-quantum-intro-for-everyone.md`<br>`2026-04-15-quantum-frontiers.md`<br>`2026-04-15-quantum-error-correction.md` |
| 11 | 5 | 0.4 | research-notes×4, reddit-digests×1 | `2026-06-26-hf-daily-papers-jun17-25.md`<br>`2026-07-08-hf-daily-papers-jun30-jul8.md`<br>`2026-06-29-hf-daily-papers-jun26-29.md` |
| 2 | 4 | 0.833 | research-notes×4 | `2026-05-26-two-harness-surveys-comparison.md`<br>`2026-05-25-agent-harness-engineering-survey.md`<br>`2026-05-07-aidlc-deep-dive.md` |
| 15 | 2 | 1.0 | research-notes×1, papers×1 | `2026-04-13-agentic-rl-infrastructure-comparison.md`<br>`2026-openclaw-rl.md` |
| 0 | 1 | 0.0 | research-notes×1 | `2026-04-07-hf-weekly-papers-apr4.md` |
| 1 | 1 | 0.0 | research-notes×1 | `2026-04-08-hf-daily-papers-apr7-8.md` |
| 4 | 1 | 0.0 | research-notes×1 | `2026-04-24-deepseek-v4-analysis.md` |
| 6 | 1 | 0.0 | research-notes×1 | `2026-05-18-hf-continuous-batching-deep-dive.md` |
| 8 | 1 | 0.0 | research-notes×1 | `2026-05-19-gpu-infra-benchmark-knowledge.md` |
| 9 | 1 | 0.0 | research-notes×1 | `2026-05-20-mcp-oauth-aws-agentcore-cognito.md` |
| 10 | 1 | 0.0 | research-notes×1 | `2026-06-16-hf-daily-papers-jun13-16.md` |
| 12 | 1 | 0.0 | research-notes×1 | `2026-07-10-blog-gpt-5-6.md` |
| 14 | 1 | 0.0 | research-notes×1 | `2026-07-28-video-generation-survey.md` |
| 16 | 1 | 0.0 | papers×1 | `2026-cafm-constraint-aware-flow-matching.md` |
| 17 | 1 | 0.0 | papers×1 | `2026-agentic-trading.md` |
| 18 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-architecture.md` |
| 19 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-build-from-scratch.md` |
| 20 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-comprehensive.md` |
| 21 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-controlled-gen.md` |
| 22 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-hallucination.md` |
| 23 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-multimodal.md` |
| 24 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-post-training.md` |
| 25 | 1 | 0.0 | topics×1 | `2026-02-09-llm-intro-reasoning.md` |
| 26 | 1 | 0.0 | topics×1 | `2026-02-09-3d-generation-survey.md` |
| 27 | 1 | 0.0 | topics×1 | `2026-04-15-quantum-computing-comprehensive.md` |
| 28 | 1 | 0.0 | topics×1 | `2026-04-15-quantum-ecosystem.md` |
| 30 | 1 | 0.0 | tech-blogs×1 | `2026-W26.md` |
| 31 | 1 | 0.0 | tech-blogs×1 | `2026-W28.md` |
| 32 | 1 | 0.0 | weekly×1 | `2026-W26.md` |
| 33 | 1 | 0.0 | weekly×1 | `2026-W30.md` |
| 34 | 1 | 0.0 | reddit-digests×1 | `2026-W24-reddit-hot.md` |
| 35 | 1 | 0.0 | reddit-digests×1 | `2026-W25-reddit-hot.md` |
| 36 | 1 | 0.0 | reddit-digests×1 | `2026-W28-reddit-hot.md` |

## 2. 桥节点（连接 ≥3 个社区的枢纽笔记）

| 笔记 | 跨社区数 | 连接数 | 标题 |
|---|---|---|---|
| `research-notes/2026-07-20-long-horizon-agents.md` | 4 | 12 | 深度整理：Long-Horizon Agents（长程智能体）研究路径全景 |
| `papers/2025-agentic-rl-survey.md` | 3 | 4 | The Landscape of Agentic Reinforcement Learn |
| `topics/ai-frontier-2026h1/2026-06-12-agent-evolution.md` | 3 | 11 | Agent 体系演化谱系 · 2026 上半年 |

## 3. 意外连接（跨社区 / 跨类型的强关联）

| 分数 | 笔记 A | 笔记 B | 权重 | 共享文献 | 类型 |
|---|---|---|---|---|---|
| 4.79 | `2026-07-21-raschka-llm-architecture-comparison.md` | `2026-W26-reddit-hot.md` | 2.91 | 1 | 跨社区+跨类型 |
| 3.9 | `2026-07-08-blog-global-workspace.md` | `2026-07-20-blog-reasoning-effort.md` | 4.0 | 0 | 跨社区 |
| 3.43 | `2026-05-26-two-harness-surveys-comparison.md` | `2026-06-12-agent-evolution.md` | 4.25 | 1 | 跨社区+跨类型 |
| 3.42 | `2026-05-26-two-harness-surveys-comparison.md` | `2026-06-12-ai-frontier-comprehensive.md` | 4.23 | 1 | 跨社区+跨类型 |
| 3.42 | `2026-05-26-two-harness-surveys-comparison.md` | `2026-06-12-trends.md` | 4.2 | 1 | 跨社区+跨类型 |
| 3.37 | `2026-05-08-cot-mechanism-mini-survey.md` | `2025-agentic-rl-survey.md` | 3.68 | 0 | 跨社区+跨类型 |
| 3.3 | `2026-07-08-blog-harness-engineering.md` | `2026-W31b.md` | 3.0 | 0 | 跨社区+跨类型 |
| 3.26 | `2026-07-20-long-horizon-agents.md` | `2026-06-12-agent-evolution.md` | 2.63 | 1 | 跨社区+跨类型 |
| 3.22 | `2025-agentic-rl-survey.md` | `2026-06-12-agent-evolution.md` | 2.15 | 1 | 跨社区+跨类型 |
| 3.15 | `2026-04-03-huggingface-daily-papers-digest.md` | `2026-06-12-auto-research.md` | 1.55 | 1 | 跨社区+跨类型 |
| 3.15 | `2026-07-20-long-horizon-agents.md` | `2026-05-08-graph-structures-meet-llms.md` | 1.55 | 1 | 跨社区+跨类型 |
| 2.52 | `2026-05-26-hf-daily-papers-may16-26.md` | `2026-05-26-two-harness-surveys-comparison.md` | 5.16 | 1 | 跨社区 |
| 2.52 | `2026-05-26-two-harness-surveys-comparison.md` | `2026-05-28-hf-daily-papers-may16-28.md` | 5.23 | 1 | 跨社区 |
| 2.47 | `2026-06-26-hf-daily-papers-jun17-25.md` | `2026-07-21-raschka-llm-architecture-comparison.md` | 4.71 | 1 | 跨社区 |
| 2.4 | `2026-05-25-agent-harness-engineering-survey.md` | `2026-07-20-long-horizon-agents.md` | 4.0 | 0 | 跨社区 |

## 4. 知识缺口

### 4.1 孤立笔记（连接数 ≤1，共 43 篇）

- `papers/2026-agentic-trading.md`（degree=0）— Agentic Trading: When LLM Agents Meet Financial Markets
- `papers/2026-cafm-constraint-aware-flow-matching.md`（degree=0）— CAFM: Constraint-Aware Flow Matching — 把"约束投影"端到端嵌进训练目标
- `reddit-digests/2026-W24-reddit-hot.md`（degree=0）— Reddit 热门话题周报 · 2026-W24（截至 06/12）
- `reddit-digests/2026-W25-reddit-hot.md`（degree=0）— Reddit 热门话题周报 · 2026-W25
- `reddit-digests/2026-W28-reddit-hot.md`（degree=0）— Reddit 热门话题周报 · 2026-W28
- `research-notes/2026-04-07-hf-weekly-papers-apr4.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-04 ~ 04-06
- `research-notes/2026-04-08-hf-daily-papers-apr7-8.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-07 ~ 04-08
- `research-notes/2026-04-24-deepseek-v4-analysis.md`（degree=0）— DeepSeek-V4 深度解读：百万 Token 上下文的高效智能
- `research-notes/2026-05-18-hf-continuous-batching-deep-dive.md`（degree=0）— 从 Attention 到异步连续批处理：HuggingFace 高效 LLM 推理两篇博客深度解读
- `research-notes/2026-05-19-gpu-infra-benchmark-knowledge.md`（degree=0）— GPU Infra Benchmark 知识总结：Blackwell / Hopper 推理与训练实践要点
- `research-notes/2026-05-20-mcp-oauth-aws-agentcore-cognito.md`（degree=0）— MCP 授权落地实战：从 OAuth 2.1 RFC 规范到 AWS AgentCore + Cognito 的
- `research-notes/2026-06-16-hf-daily-papers-jun13-16.md`（degree=0）— HF Daily Papers 摘要 · 2026/06/13–06/16
- `research-notes/2026-07-10-blog-gpt-5-6.md`（degree=0）— Blog Deep Dive: Previewing GPT‑5.6 Sol（OpenAI）
- `research-notes/2026-07-28-video-generation-survey.md`（degree=0）— Text2Video / Image2Video 技术深度调研（第六版：技术机制扩充）
- `tech-blogs/2026-W26.md`（degree=0）— Tech Blogs 周报｜2026-W26（06/19 – 06/26）
- `tech-blogs/2026-W28.md`（degree=0）— Tech Blogs 周报 · 2026-W28（06/30–07/08）
- `topics/3d/2026-02-09-3d-generation-survey.md`（degree=0）— 3D 生成技术全景总结
- `topics/llm/2026-02-09-llm-intro-architecture.md`（degree=0）— LLM Architecture
- `topics/llm/2026-02-09-llm-intro-build-from-scratch.md`（degree=0）— LLM Intro: 语言模型解释与训练 - Build LLM from Scratch
- `topics/llm/2026-02-09-llm-intro-comprehensive.md`（degree=0）— LLM 大模型技术全景总结（索引）
- `topics/llm/2026-02-09-llm-intro-controlled-gen.md`（degree=0）— 大模型受控生成与角色定制
- `topics/llm/2026-02-09-llm-intro-hallucination.md`（degree=0）— LLM 幻觉问题：检测与缓解方法综述
- `topics/llm/2026-02-09-llm-intro-multimodal.md`（degree=0）— 多模态大模型技术总结
- `topics/llm/2026-02-09-llm-intro-post-training.md`（degree=0）— Post-training 101: 从预训练到指令调优
- `topics/llm/2026-02-09-llm-intro-reasoning.md`（degree=0）— 大模型推理 Reasoning

### 4.2 稀疏社区（内聚度 <0.15 且 ≥3 篇，共 0 个）

_无_

