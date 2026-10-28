# 知识图谱洞察报告（自动生成）

- **生成方式**: `scripts/wiki_graph.py`（方法参考 [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki) 的 4 信号相关性模型 + Louvain 社区发现 + Graph Insights）
- **节点**: 286 篇笔记　**边**: 657　**社区**: 72
- **信号权重**: 直接链接 ×3.0 / 共享文献来源 ×4.0 / Adamic-Adar ×1.5 / 同类型 ×1.0

> 本报告由脚本读取仓库现有 md 与 `references.bib` 生成，不修改任何笔记。

## 1. 主题社区（Louvain 自动聚类）

| # | 规模 | 内聚度 | 构成 | 核心笔记（按连接数） |
|---|---|---|---|---|
| 70 | 45 | 0.121 ⚠️稀疏 | research-notes×23, reddit-digests×11, tech-blogs×10, weekly×1 | `2026-08-12-topic-softmax-linearization-and-k3.md`<br>`2026-08-11-hf-daily-papers-aug10-11.md`<br>`2026-W33.md` |
| 10 | 39 | 0.131 ⚠️稀疏 | research-notes×29, papers×5, weekly×3, topics×1, tech-blogs×1 | `2026-07-20-long-horizon-agents.md`<br>`2026-07-20-blog-reasoning-effort.md`<br>`2026-07-20-llm-long-context.md` |
| 8 | 23 | 0.253 | research-notes×9, tech-blogs×6, topics×5, weekly×2, reddit-digests×1 | `2026-08-07-hf-daily-papers-aug04-07.md`<br>`2026-08-03-hf-daily-papers-aug01-03.md`<br>`2026-W32b.md` |
| 9 | 22 | 0.242 | research-notes×11, reddit-digests×6, tech-blogs×4, weekly×1 | `2026-W31.md`<br>`2026-07-29-hf-daily-papers-jul28-29.md`<br>`2026-W31-reddit-hot.md` |
| 12 | 19 | 0.333 | research-notes×12, topics×6, reddit-digests×1 | `2026-06-12-agent-evolution.md`<br>`2026-06-12-ai-frontier-comprehensive.md`<br>`2026-06-12-auto-research.md` |
| 1 | 18 | 0.17 | research-notes×18 | `2026-10-17-openmoss-deep-research.md`<br>`embodied-primary-findings.md`<br>`agents-reliability-primary-findings.md` |
| 30 | 17 | 0.441 | topics×16, research-notes×1 | `2026-10-21-deepinfer-15-references.md`<br>`2026-10-21-deepinfer-00-executive-summary.md`<br>`2026-10-21-deepinfer-04-topics-index-errata.md` |
| 11 | 10 | 0.244 | research-notes×5, tech-blogs×2, reddit-digests×2, weekly×1 | `2026-06-26-hf-daily-papers-jun17-25.md`<br>`2026-W26-reddit-hot.md`<br>`2026-W26.md` |
| 28 | 9 | 0.5 | topics×9 | `2026-04-15-quantum-computing-comprehensive.md`<br>`2026-04-15-quantum-intro-for-everyone.md`<br>`2026-04-15-quantum-frontiers.md` |
| 29 | 9 | 0.472 | topics×9 | `2026-02-09-llm-intro-comprehensive.md`<br>`2026-02-09-llm-intro-architecture.md`<br>`2026-02-09-llm-intro-post-training.md` |
| 17 | 5 | 0.5 | research-notes×5 | `2026-10-28-detectorscope.md`<br>`paper-evidence.md`<br>`source-audit.md` |
| 2 | 4 | 0.5 | research-notes×4 | `2026-10-13-aws-lambda-web-functions-research.md`<br>`2026-10-13-aws-lambda-web-adapter-vs-web-functions.md`<br>`lwa-migration-findings.md` |
| 35 | 4 | 0.5 | topics×4 | `2026-10-20-agentic-payment-demo-design.md`<br>`nium-sandbox-verification.md`<br>`airwallex-sandbox-verification.md` |
| 0 | 2 | 1.0 | research-notes×2 | `2026-10-28-openai-scientific-understanding-verification-notes.md`<br>`2026-10-28-openai-scientific-understanding.md` |
| 26 | 2 | 1.0 | papers×1, research-notes×1 | `2026-openclaw-rl.md`<br>`2026-04-13-agentic-rl-infrastructure-comparison.md` |
| 32 | 2 | 1.0 | topics×2 | `2026-10-19-agentic-payment-protocols-survey.md`<br>`2026-10-20-aws-agentic-payment-solution-airwallex.md` |
| 3 | 1 | 0.0 | research-notes×1 | `2026-10-12-strands-decider-and-box-agent-building.md` |
| 4 | 1 | 0.0 | research-notes×1 | `2026-09-09-bedrock-gpt6-astra-findings.md` |
| 5 | 1 | 0.0 | research-notes×1 | `2026-09-08-hf-daily-papers-sep08d.md` |
| 6 | 1 | 0.0 | research-notes×1 | `2026-08-31-hf-daily-papers-aug27-31.md` |
| 7 | 1 | 0.0 | research-notes×1 | `2026-08-27-hf-daily-papers-aug26-27.md` |
| 13 | 1 | 0.0 | research-notes×1 | `2026-05-20-mcp-oauth-aws-agentcore-cognito.md` |
| 14 | 1 | 0.0 | research-notes×1 | `2026-04-08-hf-daily-papers-apr7-8.md` |
| 15 | 1 | 0.0 | research-notes×1 | `2026-04-07-hf-weekly-papers-apr4.md` |
| 16 | 1 | 0.0 | research-notes×1 | `选题卡与品味反馈模板.md` |
| 18 | 1 | 0.0 | research-notes×1 | `04-image-layout-supervision-and-specification.md` |
| 19 | 1 | 0.0 | research-notes×1 | `local-agent-enforcement-landscape.md` |
| 20 | 1 | 0.0 | research-notes×1 | `hosted-agent-sandbox-landscape.md` |
| 21 | 1 | 0.0 | research-notes×1 | `decision-model-product-comparison.md` |
| 22 | 1 | 0.0 | research-notes×1 | `decision-model-evolution-primary.md` |
| 23 | 1 | 0.0 | research-notes×1 | `box-sandbox-product-comparison.md` |
| 24 | 1 | 0.0 | research-notes×1 | `box-demo-primary-feasibility.md` |
| 25 | 1 | 0.0 | research-notes×1 | `strands-box-primary-findings.md` |
| 27 | 1 | 0.0 | papers×1 | `2026-cafm-constraint-aware-flow-matching.md` |
| 31 | 1 | 0.0 | topics×1 | `2026-10-19-codex-spec-deep-dive.md` |
| 33 | 1 | 0.0 | topics×1 | `2026-10-20-multi-rail-sandbox-background.md` |
| 34 | 1 | 0.0 | topics×1 | `2026-10-20-demo-feature-guide.md` |
| 36 | 1 | 0.0 | topics×1 | `stripe-mpp.md` |
| 37 | 1 | 0.0 | topics×1 | `psp-sandboxes.md` |
| 38 | 1 | 0.0 | topics×1 | `cards-protocols.md` |
| 39 | 1 | 0.0 | topics×1 | `rtp-china.md` |
| 40 | 1 | 0.0 | topics×1 | `psp.md` |
| 41 | 1 | 0.0 | topics×1 | `cards.md` |
| 42 | 1 | 0.0 | topics×1 | `review-findings.md` |
| 43 | 1 | 0.0 | topics×1 | `dan-kim-public-views.md` |
| 44 | 1 | 0.0 | topics×1 | `aws-services.md` |
| 45 | 1 | 0.0 | topics×1 | `airwallex-profile.md` |
| 46 | 1 | 0.0 | topics×1 | `2026-02-09-3d-generation-survey.md` |
| 47 | 1 | 0.0 | tech-blogs×1 | `2026-W37d.md` |
| 48 | 1 | 0.0 | tech-blogs×1 | `2026-W37b.md` |
| 49 | 1 | 0.0 | tech-blogs×1 | `2026-W37.md` |
| 50 | 1 | 0.0 | tech-blogs×1 | `2026-W36f.md` |
| 51 | 1 | 0.0 | tech-blogs×1 | `2026-W36.md` |
| 52 | 1 | 0.0 | tech-blogs×1 | `2026-W35b.md` |
| 53 | 1 | 0.0 | tech-blogs×1 | `2026-W35.md` |
| 54 | 1 | 0.0 | tech-blogs×1 | `2026-W34d.md` |
| 55 | 1 | 0.0 | tech-blogs×1 | `2026-W34b.md` |
| 56 | 1 | 0.0 | weekly×1 | `2026-W37.md` |
| 57 | 1 | 0.0 | weekly×1 | `2026-W36.md` |
| 58 | 1 | 0.0 | weekly×1 | `2026-W35.md` |
| 59 | 1 | 0.0 | weekly×1 | `2026-W34.md` |
| 60 | 1 | 0.0 | reddit-digests×1 | `2026-W38-reddit-hot.md` |
| 61 | 1 | 0.0 | reddit-digests×1 | `2026-W37d-reddit-hot.md` |
| 62 | 1 | 0.0 | reddit-digests×1 | `2026-W37b-reddit-hot.md` |
| 63 | 1 | 0.0 | reddit-digests×1 | `2026-W37-reddit-hot.md` |
| 64 | 1 | 0.0 | reddit-digests×1 | `2026-W36f-reddit-hot.md` |
| 65 | 1 | 0.0 | reddit-digests×1 | `2026-W36d-reddit-hot.md` |
| 66 | 1 | 0.0 | reddit-digests×1 | `2026-W36b-reddit-hot.md` |
| 67 | 1 | 0.0 | reddit-digests×1 | `2026-W36-reddit-hot.md` |
| 68 | 1 | 0.0 | reddit-digests×1 | `2026-W35b-reddit-hot.md` |
| 69 | 1 | 0.0 | reddit-digests×1 | `2026-W34d-reddit-hot.md` |
| 71 | 1 | 0.0 | reddit-digests×1 | `2026-W25-reddit-hot.md` |

## 2. 桥节点（连接 ≥3 个社区的枢纽笔记）

| 笔记 | 跨社区数 | 连接数 | 标题 |
|---|---|---|---|
| `research-notes/2026-07-08-blog-harness-engineering.md` | 5 | 13 | Blog Deep Dive: Harness Engineering for Self |
| `research-notes/2026-10-17-openmoss-deep-research.md` | 4 | 14 | OpenMOSS 深度调研：项目、论文与技术路线 |
| `research-notes/2026-08-11-hf-daily-papers-aug10-11.md` | 4 | 19 | HF Daily Papers 摘要 · 2026-08-10 ~ 08-11 |
| `research-notes/2026-08-03-hf-daily-papers-aug01-03.md` | 4 | 16 | HF Daily Papers 中文摘要：2026-08-01 ~ 08-03（含 07 |
| `research-notes/2026-08-03-blog-openai-ten-math-advances.md` | 4 | 14 | Blog Deep Dive：OpenAI 宣称十项数学进展 —— 证据链有多硬？ |
| `research-notes/2026-07-31-blog-harness-shelf-life.md` | 4 | 14 | Blog Deep Dive：Harness 保质期只有半年 —— Claude Cod |
| `research-notes/2026-07-29-hf-daily-papers-jul28-29.md` | 4 | 12 | HF Daily Papers 周报 · 2026-07-28 ~ 07-29 |
| `research-notes/2026-07-20-long-horizon-agents.md` | 4 | 30 | 深度整理：Long-Horizon Agents（长程智能体）研究路径全景 |
| `research-notes/2026-07-20-llm-long-context.md` | 4 | 15 | 综述：LLM 长上下文是如何做到的（位置外推 / 稀疏注意力 / 记忆 / 架构 / 训 |
| `research-notes/2026-07-20-flash-attention-efficient-attention.md` | 4 | 12 | 综述：FlashAttention 与高效注意力机制（IO-aware / 长上下文 / |
| `research-notes/2026-07-08-blog-global-workspace.md` | 4 | 5 | Blog Deep Dive: A Global Workspace in Langua |
| `topics/agent/2026-08-07-agent-quality-evaluation.md` | 4 | 10 | 深度总结：Agent 工作质量如何评估 —— 从学术基准到业界工程实践 |

## 3. 意外连接（跨社区 / 跨类型的强关联）

| 分数 | 笔记 A | 笔记 B | 权重 | 共享文献 | 类型 |
|---|---|---|---|---|---|
| 4.9 | `2026-07-20-llm-long-context.md` | `2026-02-09-llm-long-context-training.md` | 4.0 | 6 | 跨社区+跨类型 |
| 4.8 | `2026-07-21-raschka-llm-architecture-comparison.md` | `2026-W30b.md` | 3.0 | 0 | 跨社区+跨类型 |
| 4.8 | `2026-W30.md` | `2026-W30-reddit-hot.md` | 3.0 | 0 | 跨社区+跨类型 |
| 3.75 | `2026-07-20-long-horizon-agents.md` | `2026-W31.md` | 7.5 | 0 | 跨社区+跨类型 |
| 3.66 | `2026-07-31-blog-harness-shelf-life.md` | `2026-W31.md` | 6.6 | 0 | 跨社区+跨类型 |
| 3.65 | `2026-07-20-blog-reasoning-effort.md` | `2026-W31.md` | 6.5 | 0 | 跨社区+跨类型 |
| 3.62 | `2026-08-03-hf-daily-papers-aug01-03.md` | `2026-W31.md` | 6.18 | 0 | 跨社区+跨类型 |
| 3.6 | `2026-07-20-long-horizon-agents.md` | `2026-W32.md` | 5.99 | 0 | 跨社区+跨类型 |
| 3.59 | `2026-W31h.md` | `2026-W31.md` | 5.92 | 0 | 跨社区+跨类型 |
| 3.58 | `2026-W32b.md` | `2026-W32b-reddit-hot.md` | 5.75 | 0 | 跨社区+跨类型 |
| 3.57 | `2026-07-27-kimi-k3-report.md` | `2026-W31.md` | 5.7 | 0 | 跨社区+跨类型 |
| 3.57 | `2026-07-20-long-horizon-agents.md` | `2026-08-07-agent-quality-evaluation.md` | 5.72 | 1 | 跨社区+跨类型 |
| 3.56 | `2026-08-03-hf-daily-papers-aug01-03.md` | `2026-W33-reddit-hot.md` | 5.59 | 0 | 跨社区+跨类型 |
| 3.56 | `2026-08-03-blog-openai-ten-math-advances.md` | `2026-W33-reddit-hot.md` | 5.57 | 0 | 跨社区+跨类型 |
| 3.55 | `2026-08-03-blog-openai-ten-math-advances.md` | `2026-W33f-reddit-hot.md` | 5.5 | 0 | 跨社区+跨类型 |

## 4. 知识缺口

### 4.1 孤立笔记（连接数 ≤1，共 88 篇）

- `papers/2026-cafm-constraint-aware-flow-matching.md`（degree=0）— CAFM: Constraint-Aware Flow Matching — 把"约束投影"端到端嵌进训练目标
- `reddit-digests/2026-W25-reddit-hot.md`（degree=0）— Reddit 热门话题周报 · 2026-W25
- `reddit-digests/2026-W34d-reddit-hot.md`（degree=0）— Reddit 热门 W34d — 截至 2026-08-21 05:0x UTC
- `reddit-digests/2026-W35b-reddit-hot.md`（degree=0）— Reddit 热门 W35b — 截至 2026-08-27 02:5x UTC
- `reddit-digests/2026-W36-reddit-hot.md`（degree=0）— Reddit 热门 W36 — 截至 2026-08-31 02:3x UTC
- `reddit-digests/2026-W36b-reddit-hot.md`（degree=0）— Reddit 热门 W36b — 截至 2026-08-31 08:5x UTC（当日第二份）
- `reddit-digests/2026-W36d-reddit-hot.md`（degree=0）— Reddit 热门 W36d — 截至 2026-09-03 02:5x UTC
- `reddit-digests/2026-W36f-reddit-hot.md`（degree=0）— Reddit 热门 W36f — 截至 2026-09-06 10:0x UTC
- `reddit-digests/2026-W37-reddit-hot.md`（degree=0）— Reddit 热门 W37 — 截至 2026-09-08 07:1x UTC
- `reddit-digests/2026-W37b-reddit-hot.md`（degree=0）— Reddit 热门 W37b — 截至 2026-09-08 08:4x UTC（**当日第二跑**）
- `reddit-digests/2026-W37d-reddit-hot.md`（degree=0）— Reddit 热门话题 · 2026-W37 第三份（09-10，1 天空缺补跑）
- `reddit-digests/2026-W38-reddit-hot.md`（degree=0）— Reddit 热门 — 2026-W38 首份（09-15 07:4x，间隔 118.3h）
- `research-notes/2026-04-07-hf-weekly-papers-apr4.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-04 ~ 04-06
- `research-notes/2026-04-08-hf-daily-papers-apr7-8.md`（degree=0）— Hugging Face Daily Papers Digest: 2026-04-07 ~ 04-08
- `research-notes/2026-05-20-mcp-oauth-aws-agentcore-cognito.md`（degree=0）— MCP 授权落地实战：从 OAuth 2.1 RFC 规范到 AWS AgentCore + Cognito 的
- `research-notes/2026-08-27-hf-daily-papers-aug26-27.md`（degree=0）— HF Daily Papers 摘要：08/26 回填 + 08/27
- `research-notes/2026-08-31-hf-daily-papers-aug27-31.md`（degree=0）— HF Daily Papers 摘要：08/27 回填 + 08/28 + 08/31（3 天空缺补跑）
- `research-notes/2026-09-08-hf-daily-papers-sep08d.md`（degree=0）— HF Daily Papers — 09/08 当日第三跑（2026-09-08 18:1x UTC）
- `research-notes/2026-09-09-bedrock-gpt6-astra-findings.md`（degree=0）— GPT-6 Astra on Amazon Bedrock: configuration findings
- `research-notes/2026-10-12-strands-decider-and-box-agent-building.md`（degree=0）— Strands Decider 与 Strands Box：如何帮助客户构建 Agent
- `research-notes/2026-10-12-strands-decider-and-box-agent-building/strands-box-primary-findings.md`（degree=0）— Strands Box 一手证据工作报告
- `research-notes/2026-10-13-decision-models-and-agent-governance-deep-research/evidence/box-demo-primary-feasibility.md`（degree=0）— Actual Dogwood / pinned Box demo feasibility
- `research-notes/2026-10-13-decision-models-and-agent-governance-deep-research/evidence/box-sandbox-product-comparison.md`（degree=0）— Strands Box：背景与相关产品边界比较
- `research-notes/2026-10-13-decision-models-and-agent-governance-deep-research/evidence/decision-model-evolution-primary.md`（degree=0）— 现代 typed decision models 的产品演进
- `research-notes/2026-10-13-decision-models-and-agent-governance-deep-research/evidence/decision-model-product-comparison.md`（degree=0）— 决策模型产品与背景对比

### 4.2 稀疏社区（内聚度 <0.15 且 ≥3 篇，共 2 个）

- 社区 10（39 篇，内聚度 0.131）核心：`2026-07-20-long-horizon-agents.md`
- 社区 70（45 篇，内聚度 0.121）核心：`2026-08-12-topic-softmax-linearization-and-k3.md`

