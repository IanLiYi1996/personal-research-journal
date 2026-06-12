# 专题设计 · 2026 上半年 AI 前沿演化谱系

> **Date**: 2026-06-12
> **类型**: topics/ 多文件综述专题
> **状态**: 设计已确认，待实现

## 目标

把 2026 年 2–6 月的 9 期 HF Daily Papers digest 作为骨架，补充经典源头论文，
梳理出四条反复出现、已成谱系的研究主线，做成"演化谱系"型专题。

核心叙事原则：**不是论文罗列，而是讲清每条主线的 起点 → 拐点 → 现状，
每个阶段解决了上一阶段的什么问题。**

参考现有 `topics/quantum-computing/` 的多文件结构与 `topics/nlp/graph-structures-meet-llms.md` 的谱系叙事。

## 目录结构

专题目录：`topics/ai-frontier-2026h1/`

| 文件 | 内容 | 核心演化链 |
|---|---|---|
| `2026-06-12-ai-frontier-comprehensive.md` | 综述索引：导读 + 四主线一句话 + 交叉关系图 | — |
| `2026-06-12-agent-evolution.md` | **Agent 体系** | Dive into Claude Code / Code as Agent Harness → Skill 体系(Ctx2Skill→SkillOS→SkillOpt→COLLEAGUE.SKILL) → 多智能体组织(RecursiveMAS/OMC) → 评测革命(OccuBench→ALE→SWE-Explore) → 安全(AgentDoG) → 自进化(Role-Agent) |
| `2026-06-12-auto-research.md` | **自动科研系统** | AI Scientist(源头) → AutoResearchClaw → ResearchClawBench → Arbor 假设树 → SciAtlas |
| `2026-06-12-training-dynamics.md` | **训练动态显微镜化** | Rethinking OPD → RLVR 机理(DelTA/Anti-Self-Distillation/DVAO) → OPD geometry(subspace locking)/TRB → MoE router |
| `2026-06-12-world-models.md` | **世界模型与具身** | Lyra/HY-World → Agentic World Modeling 综述 → Cosmos 3 全模态 → ABot-Earth 3DGS → Latent Spatial Memory；VLA 支线: MolmoAct2→Qwen-VLA→PhysBrain |
| `2026-06-12-trends.md` | **趋势综合 + Open Questions** | 四线交汇：Skill/adapter/memory 收敛、评测经济价值转向、表征 3D 化、训练显微镜化 |

配图子目录：每个 .md 对应一个同名子目录存放图片（遵循仓库 figure 约定）。

## 写作规范

- **语言**：中文，技术术语保留英文，与现有 topics/ 一致。
- **每篇主线文档结构**（~250-400 行）：
  1. 导读（这条线在解决什么根本问题）
  2. 演化时间线（表格：阶段 → 代表工作 → 突破点 → 局限）
  3. 分阶段详解（每个拐点 2-4 段，含关键数据/方法）
  4. 1-2 张配图（架构图或谱系图）
  5. 本线小结 + 与其他主线的交叉点
- **配图策略**：
  - 优先复用已下载的 digest 配图（ALE、ABot-Earth、Keye 等）
  - 每条主线画一张演化谱系图（Mermaid 优先，复杂的用 SVG）
  - 关键论文从 arXiv `https://arxiv.org/html/{id}v1/` 补抓架构图
- **引用规范**（遵循"所有内容须引可验证来源"约束）：
  - 每篇文末 References 段，所有引用标 HF/arXiv 链接
  - 补充的经典源头论文（ReAct、AI Scientist、Cosmos 系列等）必须实际检索确认存在，不凭记忆编造 ID
- **索引文档**额外含一张四主线交叉关系图。

## 实现步骤

1. 检索补全各主线源头论文（ReAct、AI Scientist、Cosmos 系列等），确认 arXiv ID。
2. 从历史 digest（04/03 起 9 期）提取各主线节点，按谱系排序。
3. 撰写 4 篇主线文档 + 索引 + 趋势综合（分块写入，避免超长）。
4. 生成/复用配图与谱系图。
5. `bash journal.sh index` 更新索引与侧边栏。
6. 提交。

## 范围边界（YAGNI）

- 不做 WeChat 改写（另起任务）。
- 不覆盖 digest 之外的非 AI 主线（如 AWS What's New）。
- 源头论文补充以"每条主线 1-3 篇奠基作"为限，不做完整文献综述。
