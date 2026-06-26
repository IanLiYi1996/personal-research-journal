# /tech-blogs-weekly — 抓取过去一周技术博客并写中文周报

每周抓取 `tech-blogs/feeds.yaml` 里所有有 RSS 的博客（个人作者 + 公司研究 + 学术机构三层），过滤过去 N 天的新文，写中文 digest 到 `tech-blogs/YYYY-WXX.md`。**Deep-dive 单独写笔记**到 `research-notes/YYYY-MM-DD-blog-<slug>.md`。

## 步骤

1. **抓 feed**：
   ```bash
   uv run python3 scripts/blog_fetch.py --days 7 > /tmp/tech-blogs.json
   ```
   输出 JSON array，每条含 `source / tier / topics / title / link / pub / summary`。脚本会在 stderr 输出每个 feed 的 `kept/total` 计数和成功/失败统计。

2. **对照上一份周报去重**：从 `tech-blogs/` 找最新文件，grep 出已覆盖 link，剔除 `/tmp/tech-blogs.json` 中重复。

3. **三层 + 主题归并**（参考 `tech-blogs/YYYY-WXX.md` 已有模板；首份周报为 `2026-W26.md`）：
   - **Layer 1 — 个人作者**：每作者本周新文（通常 0-2 篇）。逐篇一句话点评 + permalink。
   - **Layer 2 — 公司博客**：按主题分组（模型发布 / 工程实践 / 应用 / 政策）。每主题挑信号最强 3-5 条。
   - **Layer 3 — 学术机构**：按领域分组（NLP / Vision / RL / 安全）。
   - **跨源主题表**：把同一主题在多家博客出现的（强信号）单列一表。
   - **深读候选** (≤3 篇)：标记本周值得另写 deep-dive 的文章 → 进 `research-notes/` 单独写。

4. **Deep-dive 笔记**：对每篇深读候选，单独 fetch HTML → 转 markdown → 写到 `research-notes/YYYY-MM-DD-blog-<slug>.md`。结构参考 `research-notes/2026-06-26-lilian-weng-scaling-laws.md`：
   - TL;DR
   - 历史脉络 / 核心论点
   - 关键公式 / 图表（下载到同名子文件夹）
   - 我的反思 + Open Questions
   - 引用关系 + 跟进阅读路径
   - 涉及的 arXiv 论文 `uv run python3 scripts/add_paper.py ...` 入库

5. **更新 CLAUDE.md** 的"Tech Blogs Previous Digests"表（首次需新建此节）。

6. **journal.sh index + git commit**（不推送）。

## 边界情况

- **某 feed 抓取失败**（HTTP 429、超时）：脚本会单独跳过并写 stderr，不影响整体；周报正文中列出"本周抓取失败的 source"。
- **某作者本周 0 篇新文**：默认不在 Layer 1 列出（避免噪音）；保留"本周静默的高产作者"清单作为信号（连续静默 ≥2 周或许值得查一下）。
- **`feed: null` 的 source**（Anthropic、OpenAI、Meta AI 等无公开 RSS）：脚本会跳过；这些 source 需要单独用 sitemap / HTML 解析做 fallback。当前先 manual 检查这些公司主页一次/周作为补充。

## 维护

- 增删 source：编辑 `tech-blogs/feeds.yaml`（YAML，含 name/url/feed/tier/topics）。
- 修改抓取行为：编辑 `scripts/blog_fetch.py`（CLI 参数 `--days/--since/--names/--tier/--delay/--list-missing`）。
- 修改 digest 模板：编辑本文件。
