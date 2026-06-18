# /aws-whats-new-daily — 抓取过去 24h AWS What's New

每天自动抓取 AWS 公告 RSS，写入 `aws-whats-new/YYYY-MM-DD.md`。
由 cron 触发；也可手动执行用于补抓。

## 步骤（推荐：直接调脚本）

```bash
curl -s "https://aws.amazon.com/about-aws/whats-new/recent/feed/" -o /tmp/aws-rss.xml
uv run python3 scripts/aws_whats_new.py
```

脚本会自动完成下面 1–7 步并在 `aws-whats-new/<YYYY-MM-DD>.md` 写一份当日 digest。手动跑用于补抓也是同一行命令。如果分类/impact 启发式需要调整，编辑 `scripts/aws_whats_new.py`。

下文是参考逻辑（脚本失败、需要手工兜底时按此操作）：

1. **确定日期**：以 `date +%F`（today, local TZ）为文件名 `aws-whats-new/<YYYY-MM-DD>.md`。如已存在则覆盖（每天只写一份）。
2. **拉 RSS**：
   ```bash
   curl -s "https://aws.amazon.com/about-aws/whats-new/recent/feed/" > /tmp/aws-rss.xml
   ```
   RSS 一般包含最近 ~100 条公告，跨多天；本命令**只保留 pubDate 在过去 24h 内的条目**（用 `date -d` 比较）。
3. **解析**：从每条 `<item>` 提取 `<title>` `<link>` `<pubDate>` `<description>`（description 是 HTML，用 `python3 -c 'import html; ...'` 反转义即可）。
4. **分类**：按服务前缀/标题关键词归入下列大类（参考已有 digest 的分组方式）：
   - **AI/ML**: Bedrock, SageMaker, Q, Comprehend, Rekognition, AgentCore, Polly, Transcribe, Translate
   - **Compute**: EC2, ECS, EKS, Lambda, Fargate, Batch, Outposts
   - **Storage**: S3, EBS, EFS, FSx, Backup, Storage Gateway
   - **Database**: RDS, Aurora, DynamoDB, ElastiCache, Redshift, Neptune, DocumentDB, Timestream
   - **Networking**: VPC, CloudFront, Route 53, API Gateway, ELB, Direct Connect, Global Accelerator
   - **Security**: IAM, KMS, Secrets Manager, GuardDuty, Inspector, Macie, WAF, Shield, Cognito, Verified
   - **Developer Tools**: CodeBuild, CodePipeline, CodeArtifact, CloudShell, CLI, SDK, X-Ray
   - **Analytics**: Athena, Glue, EMR, Kinesis, MSK, OpenSearch, QuickSight, Lake Formation
   - **Management**: CloudFormation, Systems Manager, Organizations, Config, CloudTrail, CloudWatch
   - **其他**: 其他不明确的归此
5. **打 impact 标签**（基于标题与描述）：
   - **High**: 主力服务的 GA、跨区域扩张、重大新能力（新模型、新引擎、新硬件）、已存在 digest 中明显主线的新增。
   - **Medium**: 增量改进、特定区域上线、SDK/CLI/控制台体验更新。
   - **Low**: 文档更新、小区域、遗留服务、价格细节、本地化。
6. **写文件**，结构（参考 `aws-whats-new/2026-W24.md` 但简化为日报）：
   ```markdown
   # AWS What's New: <YYYY-MM-DD>

   - **抓取时间:** <ISO timestamp>
   - **过去 24h 公告数:** N
   - **Source:** https://aws.amazon.com/about-aws/whats-new/recent/feed/

   ## Top Highlights（≤5 条 High impact，无则省略此节）

   - [<title>](<link>) — <一句话点评>

   ## 按类别详情

   ### AI/ML (n 项)

   | 时间 | 公告 | 影响 |
   |------|------|------|
   | HH:MM | [<title>](<link>) | High/Medium/Low |
   ...

   ### Compute (n 项)
   ...

   （只渲染本日有条目的类别；空类别不写）
   ```
7. **如果过去 24h 没有公告**：仍写一份文件，内容为「过去 24h RSS 无新条目」并附抓取时间，避免下次误触发。
8. **不要**自动 commit；人审后再决定（与 reddit / hf digest 一致）。

## 边界情况

- RSS 取不到/curl 非 0：写文件标注「抓取失败 + 错误信息」，不要 abort 静默。
- pubDate 时区是 GMT；用 `date -u -d "<pubDate>" +%s` 转 epoch 比 `date -u -d "24 hours ago" +%s` 即可。
- 同一条公告（link 相同）若 RSS 重复出现，取一次。

## 维护

- 抓取/分类/写入逻辑可以按需补到 `scripts/aws_whats_new.py`，再让本命令调用。当前先用 inline shell + 简易 python 也可。
- 类别表与 impact 启发式如有更新，直接改本文件即可。
