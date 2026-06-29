# /hf-daily-papers-weekly — HF Daily Papers 每周中文摘要

每周一抓取过去一周的 Hugging Face Daily Papers，写中文 digest 到 `research-notes/YYYY-MM-DD-hf-daily-papers-{range}.md`，含图、数据表、deep dive，所有覆盖论文入文献库。

**完整工作流见 CLAUDE.md 「HF Daily Papers Digest Workflow」一节**。下面是 cron 触发时使用的简洁步骤；手动跑也是同一套。

## 步骤

1. **确定覆盖范围**：查 `CLAUDE.md` 的 Previous Digests 表找到上一份 digest 的截止日期；覆盖至今天（local TZ）。
2. **拉 HF API**：对范围内每一天逐日调用
   ```bash
   curl -s "https://huggingface.co/api/daily_papers?date=YYYY-MM-DD&limit=100&sort=publishedAt" -o /tmp/hf-MMDD.json
   ```
   空数组（周末投稿空档）正常，跳过即可。
3. **去重**：从上一份 digest grep `arxiv.org/(abs|pdf)/\d{4}\.\d{4,5}` 拿 arXiv ID 集合，逐篇比对剔除已覆盖。
4. **排序 + 精选 Top 25**：按 `upvotes` 降序。
5. **挑 1-2 篇 deep dive**（影响力 = upvotes × novelty × technical depth）：
   - `curl -s "https://huggingface.co/papers/{ID}.md"` 拿全文 markdown（若 404 fall back to arxiv.org/html/{ID}）
   - 下载图：`curl -s -L -o <out>.png "https://arxiv.org/html/{ID}v1/x{N}.png"` 或 `figs/<name>.png`
   - 图存到 `research-notes/<digest-name>/<paper-fig>.png`
6. **写中文 digest** 到 `research-notes/YYYY-MM-DD-hf-daily-papers-{range}.md`：
   - Context（覆盖范围 / 数据获取 / 主线信号）
   - 论文总览表（Top 25，含 arXiv link / 中译标题 / upvotes / 主题）
   - 分主题详解（5-7 个 cluster）
   - Deep Dive ×1-2（每篇含 figure + 数据表 + 我的看法）
   - 其他值得关注（剩余精选一句话）
   - 趋势分析（3-4 条主线）
   - Open Questions
   - References（所有覆盖论文的真实 HF link）
7. **入文献库**：
   ```bash
   uv run python3 scripts/add_paper.py <id1> <id2> ...   # 一次调用所有 arXiv id
   ```
8. **更新 CLAUDE.md** 的 Previous Digests 表，新加一行。
9. **重建索引 + commit**：
   ```bash
   bash journal.sh index
   git add -A && git commit -m "📚 weekly: HF Daily Papers digest <range> (N papers / 25 精选 / K deep dives)" 
   ```
   不推送。

## 偏好（来自 [[feedback_digest_style]]）

- **图和数据表必须有**（不只是文字总结）
- 引用必须可验证（真实 HF / arXiv link，不凭记忆编造 ID）
- 中文叙述 + 技术术语保留英文
- Deep dive 风格参考 `research-notes/2026-06-29-hf-daily-papers-jun26-29.md`（DanceOPD + Verification Horizon 范本）
