# 知识图谱洞察报告（自动生成）

- **生成方式**: `scripts/wiki_graph.py`（方法参考 [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) 的 4 信号相关性模型 + Louvain 社区发现 + Graph Insights）
- **节点**: 112 篇笔记　**边**: 256　**社区**: 16
- **信号权重**: 直接链接 ×3.0 / 共享文献来源 ×4.0 / Adamic-Adar ×1.5 / 同类型 ×1.0

> 本报告由脚本读取仓库现有 md 与 `references.bib` 生成，不修改任何笔记。

## 1. 主题社区（Louvain 自动聚类）

| # | 规模 | 内聚度 | 构成 | 核心笔记（按连接数） |
|---|---|---|---|---|
| 15 | 24 | 0.214 | research-notes×11, reddit-digests×6, tech-blogs×6, weekly×1 | `2026-W31.md`<br>`2026-W31-reddit-hot.md`<br>`2026-07-29-hf-daily-papers-jul28-29.md` |
| 7 | 22 | 0.212 | research-notes×18, weekly×3, tech-blogs×1 | `2026-07-20-long-horizon-agents.md`<br>`2026-07-20-blog-reasoning-effort.md`<br>`2026-07-20-llm-long-context.md` |
| 13 | 19 | 0.333 | research-notes×12, topics×6, reddit-digests×1 | `2026-06-12-agent-evolution.md`<br>`2026-06-12-ai-frontier-comprehensive.md`<br>`2026-06-12-trends.md` |
| 6 | 11 | 0.218 | research-notes×6, reddit-digests×2, tech-blogs×2, weekly×1 | `2026-06-26-hf-daily-papers-jun17-25.md`<br>`2026-W26.md`<br>`2026-W26-reddit-hot.md` |
| 10 | 9 | 0.472 | topics×9 | `2026-02-09-llm-intro-comprehensive.md`<br>`2026-02-09-llm-intro-architecture.md`<br>`2026-02-09-llm-intro-post-training.md` |
| 12 | 9 | 0.5 | topics×9 | `2026-04-15-quantum-intro-for-everyone.md`<br>`2026-04-15-quantum-computing-comprehensive.md`<br>`2026-04-15-quantum-frontiers.md` |
| 3 | 8 | 0.429 | papers×5, research-notes×2, topics×1 | `2026-05-08-cot-mechanism-mini-survey.md`<br>`2025-agentic-rl-survey.md`<br>`2025-kismath-causal-cot-graph.md` |
| 2 | 2 | 1.0 | research-notes×1, papers×1 | `2026-04-13-agentic-rl-infrastructure-comparison.md`<br>`2026-openclaw-rl.md` |
| 0 | 1 | 0.0 | research-notes×1 | `2026-04-07-hf-weekly-papers-apr4.md` |
| 1 | 1 | 0.0 | research-notes×1 | `2026-04-08-hf-daily-papers-apr7-8.md` |
| 4 | 1 | 0.0 | research-notes×1 | `2026-05-20-mcp-oauth-aws-agentcore-cognito.md` |
| 5 | 1 | 0.0 | research-notes×1 | `2026-06-16-hf-daily-papers-jun13-16.md` |
| 8 | 1 | 0.0 | research-notes×1 | `2026-07-28-video-generation-survey.md` |
| 9 | 1 | 0.0 | papers×1 | `2026-cafm-constraint-aware-flow-matching.md` |
| 11 | 1 | 0.0 | topics×1 | `2026-02-09-3d-generation-survey.md` |
| 14 | 1 | 0.0 | reddit-digests×1 | `2026-W25-reddit-hot.md` |

## 2. 桥节点（连接 ≥3 个社区的枢纽笔记）

| 笔记 | 跨社区数 | 连接数 | 标题 |
|---|---|---|---|
| `research-notes/2026-07-20-blog-reasoning-effort.md` | 4 | 14 | 综述：LLM 推理努力度控制（Controlling Reasoning Effort  |
| `research-notes/2026-07-20-long-horizon-agents.md` | 4 | 19 | 深度整理：Long-Horizon Agents（长程智能体）研究路径全景 |
| `research-notes/2026-07-08-blog-harness-engineering.md` | 3 | 7 | Blog Deep Dive: Harness Engineering for Self |
| `research-notes/2026-07-08-blog-global-workspace.md` | 3 | 3 | Blog Deep Dive: A Global Workspace in Langua |
| `research-notes/2026-07-20-llm-long-context.md` | 3 | 11 | 综述：LLM 长上下文是如何做到的（位置外推 / 稀疏注意力 / 记忆 / 架构 / 训 |
| `papers/2025-agentic-rl-survey.md` | 3 | 6 | The Landscape of Agentic Reinforcement Learn |
| `topics/ai-frontier-2026h1/2026-06-12-agent-evolution.md` | 3 | 12 | Agent 体系演化谱系 · 2026 上半年 |

## 3. 意外连接（跨社区 / 跨类型的强关联）

| 分数 | 笔记 A | 笔记 B | 权重 | 共享文献 | 类型 |
|---|---|---|---|---|---|
| 4.9 | `2026-07-20-llm-long-context.md` | `2026-02-09-llm-long-context-training.md` | 4.0 | 6 | 跨社区+跨类型 |
| 4.8 | `2026-W30.md` | `2026-W30-reddit-hot.md` | 3.0 | 0 | 跨社区+跨类型 |
| 3.75 | `2026-07-20-long-horizon-agents.md` | `2026-W31.md` | 7.5 | 0 | 跨社区+跨类型 |
| 3.73 | `2026-07-20-blog-reasoning-effort.md` | `2026-W31.md` | 7.33 | 0 | 跨社区+跨类型 |
| 3.52 | `2026-07-31-blog-harness-shelf-life.md` | `2026-W31.md` | 5.16 | 0 | 跨社区+跨类型 |
| 3.5 | `2026-07-20-llm-long-context.md` | `2026-W31.md` | 5.03 | 0 | 跨社区+跨类型 |
| 3.47 | `2026-07-27-kimi-k3-report.md` | `2026-W31.md` | 4.7 | 0 | 跨社区+跨类型 |
| 3.46 | `2026-07-20-long-horizon-agents.md` | `2025-agentic-rl-survey.md` | 4.56 | 2 | 跨社区+跨类型 |
| 3.44 | `2026-07-20-long-horizon-agents.md` | `2026-W31h.md` | 4.42 | 0 | 跨社区+跨类型 |
| 3.41 | `2026-07-31-hf-daily-papers-jul30-31.md` | `2026-W31.md` | 4.08 | 0 | 跨社区+跨类型 |
| 3.4 | `2026-07-31-blog-harness-shelf-life.md` | `2026-W31h.md` | 4.0 | 0 | 跨社区+跨类型 |
| 3.35 | `2026-07-20-long-horizon-agents.md` | `2026-W31g-reddit-hot.md` | 3.49 | 0 | 跨社区+跨类型 |
| 3.3 | `2026-07-08-blog-harness-engineering.md` | `2026-W28.md` | 3.0 | 0 | 跨社区+跨类型 |
| 3.3 | `2026-07-08-blog-harness-engineering.md` | `2026-W31b.md` | 3.0 | 0 | 跨社区+跨类型 |
| 3.3 | `2026-07-08-blog-global-workspace.md` | `2026-model-spec-midtraining.md` | 3.0 | 0 | 跨社区+跨类型 |

## 4. 知识缺口

### 4.1 孤立笔记（连接数 ≤1，共 20 篇）

- `papers/2026-cafm-constraint-aware-flow-matching.md`（degree=0）— CAFM: Constraint-Aware Flow Matching — 把"约束投影"端到端嵌进训练目标
- `reddit-digests/2026-W25-reddit-hot.md`（degree=0）— Reddit 热门话题周报 · 2026-W25
- `research-notes/2026-04-07-hf-weekly-papers-apr4.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-04 ~ 04-06
- `research-notes/2026-04-08-hf-daily-papers-apr7-8.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-07 ~ 04-08
- `research-notes/2026-05-20-mcp-oauth-aws-agentcore-cognito.md`（degree=0）— MCP 授权落地实战：从 OAuth 2.1 RFC 规范到 AWS AgentCore + Cognito 的
- `research-notes/2026-06-16-hf-daily-papers-jun13-16.md`（degree=0）— HF Daily Papers 摘要 · 2026/06/13–06/16
- `research-notes/2026-07-28-video-generation-survey.md`（degree=0）— Text2Video / Image2Video 技术深度调研（第六版：技术机制扩充）
- `topics/3d/2026-02-09-3d-generation-survey.md`（degree=0）— 3D 生成技术全景总结
- `papers/2026-agentic-trading.md`（degree=1）— Agentic Trading: When LLM Agents Meet Financial Markets
- `papers/2026-openclaw-rl.md`（degree=1）— OpenClaw-RL: Train Any Agent Simply by Talking
- `reddit-digests/2026-W24-reddit-hot.md`（degree=1）— Reddit 热门话题周报 · 2026-W24（截至 06/12）
- `reddit-digests/2026-W28-reddit-hot.md`（degree=1）— Reddit 热门话题周报 · 2026-W28
- `research-notes/2026-04-10-hf-daily-papers-apr9-10.md`（degree=1）— Hugging Face Daily Papers Digest: 2026-04-09 ~ 04-10
- `research-notes/2026-04-13-agentic-rl-infrastructure-comparison.md`（degree=1）— Agentic RL 基础设施对比：Atropos/Tinker vs OpenClaw-RL
- `research-notes/2026-04-13-hf-daily-papers-apr11-13.md`（degree=1）— Hugging Face Daily Papers Digest: 2026-04-11 ~ 04-13
- `research-notes/2026-05-18-four-papers-review.md`（degree=1）— 四篇论文合读：扩散×自回归、连续嵌入流、RAG 失败的电路追踪、时序预测多智能体
- `research-notes/2026-07-09-hf-daily-papers-jul09.md`（degree=1）— HF Daily Papers Digest · 07/09 (2026-W28 补充)
- `research-notes/2026-07-10-blog-gpt-5-6.md`（degree=1）— Blog Deep Dive: Previewing GPT‑5.6 Sol（OpenAI）
- `tech-blogs/2026-W30b.md`（degree=1）— 技术博客周报 · 2026-W30b(07/17–07/24 补抓窗口)
- `weekly/knowledge-graph.md`（degree=1）— 知识图谱洞察报告（自动生成）

### 4.2 稀疏社区（内聚度 <0.15 且 ≥3 篇，共 0 个）

_无_

