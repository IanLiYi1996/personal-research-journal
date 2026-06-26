# /tech-blogs-weekly — 抓取过去一周技术博客并写中文周报

每周抓取 `tech-blogs/feeds.yaml` 里所有有 RSS 的博客（个人作者 + 公司研究 + 学术机构三层），过滤过去 N 天的新文，写中文 digest 到 `tech-blogs/YYYY-WXX.md`。**Deep-dive 单独写笔记**到 `research-notes/YYYY-MM-DD-blog-<slug>.md`。

## 步骤

1. **抓 feed**（两路并行：RSS + sitemap fallback）：
   ```bash
   uv run python3 scripts/blog_fetch.py --days 7 > /tmp/tech-blogs-rss.json
   uv run python3 scripts/blog_sitemap_fallback.py --days 7 > /tmp/tech-blogs-sitemap.json
   ```
   - `blog_fetch.py`：处理 60 个有 RSS 的 source；输出 JSON array，每条含 `source / tier / topics / title / link / pub / summary`。stderr 显示每 feed `kept/total` 与成功/失败统计。
   - `blog_sitemap_fallback.py`：处理 4 个**有 sitemap 但无 RSS** 的 source（Anthropic News / Anthropic Research / Cohere / AI2）。配置在 `SITEMAP_RULES` 字典。
   - **可选过滤**：arXiv cs.AI RSS 往往有 200+ 条 firehose，digest 时建议过滤掉或仅做 topic-grep。
   - **合并 + 去重 by link**：
     ```python
     uv run python3 -c "
     import json
     rss = json.load(open('/tmp/tech-blogs-rss.json'))
     sm = json.load(open('/tmp/tech-blogs-sitemap.json'))
     # Optional: drop arXiv firehose
     rss = [p for p in rss if p['source'] != 'arXiv cs.AI (RSS)']
     merged = {p['link']: p for p in rss}
     for p in sm: merged.setdefault(p['link'], p)
     out = sorted(merged.values(), key=lambda p: p['pub'], reverse=True)
     json.dump(out, open('/tmp/tech-blogs.json','w'), ensure_ascii=False, indent=2)
     print(len(out))
     "
     ```

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
- **`feed: null` 的 source**：分两类——
  1. **已通过 sitemap fallback 接入**（4 个）：Anthropic News / Anthropic Research / Cohere / AI2。继续走 `blog_sitemap_fallback.py`。
  2. **仍待 HTML scraper**（21 个）：Meta AI (ai.meta.com) / Mistral / xAI / LlamaIndex / Stanford HAI/CRFM / Princeton / MILA / Apollo Research / Redwood / 机器之心中文站 / 李沐 / fast.ai / François Chollet / The Batch / Ben's Bites / TLDR AI / Modal / Replicate / Perplexity / Adept / Character。需要写 HTML index page scraper（提取最新 N 条 article URL + 配合页面解析提取 publish date）。当前先 manual 检查这些站点的主页或转去看 X / Twitter。

## 维护

- 增删 source：编辑 `tech-blogs/feeds.yaml`（YAML，含 name/url/feed/tier/topics/lang）。**5 tiers**：personal / newsletter / company / academia / conference。
- 修改 RSS 抓取行为：编辑 `scripts/blog_fetch.py`（CLI: `--days/--since/--names/--tier/--delay/--list-missing`）。
- 增加 sitemap fallback：编辑 `scripts/blog_sitemap_fallback.py` 的 `SITEMAP_RULES` 字典；每个条目是 `name → (sitemap_url, [path-prefixes])`。前提：站点 `sitemap.xml` 的 `<lastmod>` 字段必须是真实的"该页面发布/更新时间"，而非"sitemap 重建时间"——后者会让所有 url 共享同一 lastmod，无法过滤（如 Stanford HAI 的情况）。
- **添加新 feed 时的探测命令**：
  ```bash
  # 1. 检查 HTTP code
  curl -sI -L -A "Mozilla/5.0" "<candidate-url>"
  # 2. 验证内容真的是 RSS/Atom XML
  curl -sL -A "Mozilla/5.0" "<candidate-url>" | head -c 500 | grep -qE '<rss|<feed|<channel' && echo OK || echo "NOT RSS"
  ```
- 修改 digest 模板：编辑本文件。
