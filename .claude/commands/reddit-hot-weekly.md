# /reddit-hot-weekly — Reddit 12 子版热门周报

每周日晚抓取 12 个跟踪的 subreddit 的 top-of-week，写中文 digest 到 `reddit-digests/YYYY-WXX-reddit-hot.md`。

**完整背景见 CLAUDE.md 「Reddit Hot Topics Digest」一节**（含数据获取现状的重要说明：当前用 RSS，无 score）。

## 步骤

1. **确定 ISO 周**：`date -u +%G-W%V`（local 也行，但 weekly 文件名按 ISO 周保持一致）。
2. **抓 RSS**：
   ```bash
   uv run python3 scripts/reddit_fetch.py --time week --delay 7 > /tmp/reddit.json
   ```
   - 子版列表内置在脚本里（12 个，见 CLAUDE.md 表）。
   - 限流凶（429）；失败的子版用更大 delay 单独补抓：
     ```bash
     uv run python3 scripts/reddit_fetch.py --subs r/aws,r/datascience --time week --delay 20 >> /tmp/reddit-retry.json
     ```
   - 合并：把 retry 的并入 `/tmp/reddit.json`。
3. **对照上一份 digest 去重**：从 `reddit-digests/` 最新文件 grep permalink，剔除已覆盖。
4. **选 Top**：按各子版 RSS 的 `rank` 取头部（RSS 无 score，所以无法跨子版数字排序）。每子版选 5-10 条。
5. **写 digest** 到 `reddit-digests/YYYY-WXX-reddit-hot.md`：
   - Context（数据来源说明 + RSS 局限标注）
   - 跨社区主线表（同主题在多个子版出现）
   - 分主题（4 组）：
     - **AI/ML 研究**: r/MachineLearning, r/LocalLLaMA, r/singularity
     - **AI 产品/应用**: r/OpenAI, r/ClaudeAI, r/StableDiffusion
     - **AWS/云/工程**: r/aws, r/devops, r/programming
     - **数据科学/学术**: r/datascience, r/statistics, r/AskAcademia
   - 趋势分析
   - Open Questions
   - References（所有真实 permalink）
6. **更新 CLAUDE.md** 的 Previous Digests 表，新加一行（含子版数 / 抓取帖数 / 新增帖数 / 已知 RSS 截断的子版）。
7. **重建索引 + commit**：
   ```bash
   bash journal.sh index
   git add -A && git commit -m "📰 reddit: WXX 热门话题周报（RSS 版，N 帖 / 12 子版）"
   ```
   不推送。

## 注意

- 当前账号 Reddit OAuth app 创建被卡，**所有数据来自 .rss feed**，没有 score / comment count
- 子版间隔 ≥7s 是底线；高峰期可能需要 20s+
- 引用全部用脚本输出的真实 permalink，不凭记忆编造（[[feedback_reference_sources]]）
