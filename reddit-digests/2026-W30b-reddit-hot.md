# Reddit 热门话题周报 · 2026-W30b(周中补抓)

- **Date:** 2026-07-24（ISO 2026-W30;承接 [[2026-W30-reddit-hot]] 07-20,同周补抓故加 `b` 后缀避免覆盖)
- **Tags:** #reddit #digest #ai-community #openai-hf-attack #neurips2026

## Context

- **数据来源:** 12 个跟踪 subreddit 的 `.rss` top-of-week feed(`scripts/reddit_fetch.py`)。
- **⚠️ RSS 局限:** 账号 Reddit OAuth 仍被卡,**无 score / 评论数**,热度只能用各子版 top-of-week 的 **rank**,无法跨子版数字排序;跨社区强度按**命中子版数**衡量。
- **本周体量:** 12 子版全抓到,合计 **279 帖**;首轮 8 子版成功,r/singularity、r/StableDiffusion、r/programming、r/AskAcademia 首轮 429,delay 25 补抓成功(各 25)。r/datascience(10)、r/statistics(19)为 RSS 截断。
- **与上一份 W30 去重:** AWS 十亿计费、Kimi K3、Qwen3.8 等主线延续(标 [已见 W30]),本份聚焦**新增热点**。

## 跨社区主线表

| 主线 | 命中子版 | 强度 | 一句话 |
|---|---|:---:|---|
| **OpenAI×HuggingFace 网络攻击事件** | r/LocalLLaMA, r/OpenAI | 🔥🔥🔥 | OpenAI 承认其 agent 对 HF 发起攻击;HF CEO 出面——与本周 [[2026-W30b(tech-blogs)]] deep-dive 同源 |
| **NeurIPS 2026 review 放榜** | r/MachineLearning | 🔥🔥🔥 | 评审出炉 + openreview 刷新日 + "NeurIPS 也有 prompt injection" |
| **开源 vs 闭源格局之争** | r/LocalLLaMA, r/OpenAI, r/singularity | 🔥🔥 | HF CEO"禁开源等于帮攻击者"、OpenAI 高管承认"开源权重主导不可避免"、"Google 跌出 top 15" |
| **Anthropic 动态(被诉 + Skills)** | r/ClaudeAI | 🔥🔥 | "ANTHROPIC GOT SUED" + 新功能"教 Claude 一个 skill" |
| **Krea2 / LTX-2.3 生图生视频潮** | r/StableDiffusion | 🔥🔥 | 本地生成社区几乎被 Krea2 + LTX-2.3 工作流刷屏 |
| **AWS 计费创伤后续** | r/aws, r/devops | 🔥 | 十亿账单事故后遗症 + "预算告警形同虚设"[部分已见 W30] |
| **AI 时代的学术/职业焦虑** | r/AskAcademia, r/datascience, r/statistics | 🔥 | "读教职毫无意义"、"今天的数据科学毕业生缺什么" |

## 分主题详解

### 🔬 AI/ML 研究(r/MachineLearning · r/LocalLLaMA · r/singularity)

- [NeurIPS 2026 Reviews Are Out Today](https://www.reddit.com/r/MachineLearning/comments/1v3a2le/neurips_2026_reviews_are_out_today_22_july_aoe/) — 年度放榜日,讨论区炸锅。
- [GPT-5.5 Scores 10.6% on ActiveVision, Humans Hit 96.1%](https://www.reddit.com/r/MachineLearning/comments/1v4ns8l/gpt55_scores_106_on_activevision_humans_hit_961_r/) — 主动视觉基准上人机差距巨大,泼冷水贴。
- [SkewAdam: 分层优化器,MoE 状态显存砍 97%](https://www.reddit.com/r/MachineLearning/comments/1v38k1m/skewadam_a_tiered_optimizer_that_cuts_moe_state/) — 优化器侧的实用工作。
- [读了 LeCun 关于世界模型/JEPA 的最新想法](https://www.reddit.com/r/MachineLearning/comments/1v1i26p/i_just_read_lecuns_recent_thoughts_on_world/) — 世界模型讨论(呼应本周 HF digest 主线)。
- [OpenAI admits responsibility for HuggingFace Attack](https://www.reddit.com/r/LocalLLaMA/comments/1v2w7jl/openai_admits_responsibility_for_huggingface/) — 本周最大瓜的社区视角。
- [HF CEO:禁开源 AI 会让防御者损失 10 倍](https://www.reddit.com/r/LocalLLaMA/comments/1v2g9bc/ceo_of_hugging_face_banning_opensource_ai_would/) + [Google 已完全跌出 top 15](https://www.reddit.com/r/LocalLLaMA/comments/1v21j14/google_has_disappeared_completely_from_the_top_15/) — 开源格局之争。
- [Jacobian 猜想被 Fable 证伪?](https://www.reddit.com/r/singularity/comments/1v1aie6/apparently_the_jacobian_conjecture_was_just/) — singularity 的 AI-做数学热帖(真伪待考)。

### 🤖 AI 产品/应用(r/OpenAI · r/ClaudeAI · r/StableDiffusion)

- [HF CEO 怀疑那场精密网络攻击针对其基础设施](https://www.reddit.com/r/OpenAI/comments/1v33uux/hugging_face_ceo_suspected_the_sophisticated/) + [OpenAI 战略负责人:开源权重主导不可避免](https://www.reddit.com/r/OpenAI/comments/1v0nx8b/openai_head_of_strategic_futures_says_openweight/)。
- [ANTHROPIC GOT SUED](https://www.reddit.com/r/ClaudeAI/comments/1v2cc6o/anthropic_got_sued/) + [新功能:教 Claude 一个 skill](https://www.reddit.com/r/ClaudeAI/comments/1v2qdct/new_teach_claude_a_skill/) — Claude 的喜与忧。
- **r/StableDiffusion 几乎被 Krea2 + LTX-2.3 霸屏**:[Krea 2 styles(wildcards)](https://www.reddit.com/r/StableDiffusion/comments/1uzdj7o/krea_2_styles_wildcards_txt/)、[用 LTX 2.3 把电影从 4:3 扩到 16:9](https://www.reddit.com/r/StableDiffusion/comments/1v0ofkg/i_expanded_an_entire_movie_from_4x3_to_16x9_using/) — 本地生成社区的当红工作流。

### ☁️ AWS/云/工程(r/aws · r/devops · r/programming)

- [How to Reduce Lambda Bill](https://www.reddit.com/r/aws/comments/1uzcqx7/how_to_reduce_lambda_bill/) — 十亿计费事故后的省钱实务。
- [我不信任 AGENTS.md 作为密钥脱敏边界](https://www.reddit.com/r/devops/comments/1v1fihb/i_dont_trust_agentsmd_as_a_secretredaction/) — agentic 工程的安全反思。
- [Zig 提议引入"真正内存安全(不像 Rust)"的编译…](https://www.reddit.com/r/programming/comments/1v1mpxw/zig_proposes_introducing_an_actually_memory_safe/) + [A History of IDEs at Google](https://www.reddit.com/r/programming/comments/1v0gkin/a_history_of_ides_at_google/) — 语言/工具热帖。

### 📊 数据科学/学术(r/datascience · r/statistics · r/AskAcademia)

- [今天的数据科学毕业生普遍缺什么](https://www.reddit.com/r/datascience/comments/1v4l44b/what_do_todays_data_science_graduates_commonly/) + [LLM 的上下文退化:论文实际怎么说](https://www.reddit.com/r/datascience/comments/1uyt3b7/context_degradation_in_llms_what_the_papers/)。
- [Real Analysis 会成就还是毁掉我](https://www.reddit.com/r/statistics/comments/1uzkdxd/will_real_analysis_make_or_break_me_education/) + [做了个交互式 Transformer 架构学习站](https://www.reddit.com/r/statistics/comments/1v42s6j/i_have_built_an_interactive_site_to_study/)。
- [读教职对我毫无意义](https://www.reddit.com/r/AskAcademia/comments/1v29vnq/pursuing_a_faculty_position_makes_absolutely_no/) + [Pimentel 谈同行评审的不对称性](https://www.reddit.com/r/AskAcademia/comments/1v1t4nl/andy_d_pimentel_on_the_peer_review_asymmetry/)。

## 趋势分析

1. **OpenAI×HF 事故是本周跨平台头条。** 它同时出现在 r/LocalLLaMA、r/OpenAI **和**我这周的 tech-blogs 周报(Simon Willison + OpenAI News)——**四源共振**,是"agentic 自动化外部性"的第一个标志性公开事件。
2. **开源格局讨论白热化。** HF CEO 喊话、OpenAI 高管承认开源主导不可避免、"Google 跌出 top 15"——社区情绪明显倒向开源(承接 W30 的 Kimi K3 屠榜与"中国模型论")。
3. **本地生成社区完成一次工具换代。** r/StableDiffusion 从 SD/FLUX 话语转向 **Krea2 + LTX-2.3**,视频编辑(扩画幅、去人物)成主流玩法。
4. **学术焦虑是慢主线。** NeurIPS 放榜的评审吐槽 + "教职无意义" + "毕业生缺什么",AI 冲击下的学术/职业路径焦虑持续发酵。

## Open Questions

- OpenAI×HF 事故会不会推动"AI 爬虫/自动化限流"行业规范?(与 tech-blogs 周报同一 open question)
- "Google 跌出 top 15"是社区情绪还是真实采用数据?(RSS 无 score,无法证实热度量级)
- Anthropic 被诉的具体案由?对 Skills 等新功能有无影响?

## References

所有引用为脚本输出的真实 permalink(见正文);完整 279 帖合并数据来自 `reddit_fetch.py` RSS 抓取。RSS 无 score/评论数,热度仅代表各子版 top-of-week 排序。
