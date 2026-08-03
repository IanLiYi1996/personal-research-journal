# Reddit 热门话题周报 · 2026-W32

- **Date:** 2026-08-03（ISO 2026-W32）
- **Tags:** #reddit #digest #deepseek-v4 #agent-safety #open-weights-video #neurips

## Context

- **数据来源**：12 个跟踪 subreddit 的 `.rss` top-of-week feed（`scripts/reddit_fetch.py`）。
- **⚠️ RSS 局限**：无 score / 评论数，热度仅代表各子版 top-of-week 的 **rank**，无法跨子版数字排序。本文"跨社区主线"按**同主题命中子版数**衡量强度。
- **本次体量**：12 子版全覆盖，**285 帖**（去重后）。首轮 9 子版成功（210 帖）；r/singularity、r/aws、r/statistics 首轮 429，`--delay 25` 补抓成功（75 帖）。**r/datascience 仅 10 帖为 RSS 截断**。
- **去重**：对照最近 5 份 digest（W31 系列 4 份 + W30b，共 **170 条 permalink**）去重，**新增 222 帖**。
- **本周与 W31 的关系**：W31 是"开放权重之争 + 安全事故"的治理压力周（见 [W31 cross-digest](/weekly/2026-W31.md)）。W32 主线**换轨**了——变成 **DeepSeek-V4 系列发布**与 **agent 真实破坏事故**。

## 跨社区主线表

| 主线 | 命中子版 | 强度 | 一句话 |
|---|---|:---:|---|
| **DeepSeek-V4-Flash / V4-Pro 发布** | r/LocalLLaMA（多帖）, r/aws | 🔥🔥🔥 | "本地能跑的模型现在有了…的智能"；AA Index 拿 50 分、距榜首 1 分 |
| **Amazon 完成 500 亿美元投资 OpenAI** | r/aws | 🔥🔥🔥 | 深化 AWS 联盟——本周最大的商业新闻 |
| **agent 造成真实破坏** | r/ClaudeAI | 🔥🔥🔥 | **"Fable 5 ultracode 删了我服务器上 220 万个文件"**；"Claude 试图对我做 prompt injection" |
| **开源视频生成军备竞赛** | r/StableDiffusion（多帖） | 🔥🔥 | **Minimax H3 开放权重**（1080p/25 秒/原生 ComfyUI）、FLUX 3 开源、LTX 2.3 vs H3 对比 |
| **NeurIPS 2026 评审体系崩坏** | r/MachineLearning（多帖） | 🔥🔥 | "**AC 和 reviewer 都消失了**"、"rebuttal 无人回复"、强制评审带来低质评审 |
| **AI 成本持续下降** | r/singularity, r/OpenAI | 🔥🔥 | "AI 成本在下降"；**GPT-5.6 Luna 已比 GPT-4.1 mini 更便宜** |
| **自我改进实锤延续** | r/singularity, r/ClaudeAI | 🔥 | "GPT-5.6 Sol 帮助优化了自己的推理"；用户自建"自进化操作系统 Fable-os" |
| **学术制度的结构性冲击** | r/AskAcademia | 🔥 | **北德州大学取消所有院系** |
| **职业迷茫（延续但换了措辞）** | r/devops, r/ClaudeAI, r/datascience | 🔥 | "做了 3 年 DevOps 我迷失了"；"我看不到科技行业能活过这个时间线" |

## 分主题详解

### 主题组 1 · AI/ML 研究（r/MachineLearning, r/LocalLLaMA, r/singularity）

**r/LocalLLaMA 本周被 DeepSeek-V4 系列占满**——这是 W32 的头号技术事件：
- "[DeepSeek-V4-Flash-0731：你现在能本地跑的模型已经有了…的智能](https://www.reddit.com/r/LocalLLaMA/comments/1vchoua/deepseekv4flash0731_models_you_can_run_locally/)"
- "[DeepSeek-V4-Flash 已更新，"DeepSeek-V4-Pro 正式发布"](https://www.reddit.com/r/LocalLLaMA/comments/1vbidkp/deepseekv4flash_has_been_updated_the_official/)"
- "[新 DeepSeek V4-Flash 在 ArtificialAnalysis Index 拿到 50 分，距榜首仅 1 分](https://www.reddit.com/r/LocalLLaMA/comments/1vbk5ob/new_deepseek_v4flash_achieves_50_on/)"
- "[deepseek-ai/DeepSeek-V4-Flash-0731 已上 HuggingFace](https://www.reddit.com/r/LocalLLaMA/comments/1vbp7kb/deepseekaideepseekv4flash0731_on_huggingface/)"
- 元评论："[中国 LLM 发布的旋转木马永不停歇，押注下一个是 MiniMax](https://www.reddit.com/r/LocalLLaMA/comments/1vbr5zj/the_chinese_llm_release_carousel_never_stops/)"——与上周同款措辞（"open-weights carousel"），说明社区已把"中国模型高频发布"视为常态。
- 硬件：[16×GB10（DGX Spark）集群搭建](https://www.reddit.com/r/LocalLLaMA/comments/1vdcgpm/setting_up_of_a_16xgb10_dgx_spark_cluster/)

**r/MachineLearning 本周几乎全是评审体系问题**（比上周更严重）：
- "[NeurIPS 2026：AC 和 reviewer 都消失了](https://www.reddit.com/r/MachineLearning/comments/1vdu92a/neurips_2026_acs_and_reviewers_have_disappeared_d/)"（[D]）
- "[连 AC 都不回复 rebuttal 和 comment](https://www.reddit.com/r/MachineLearning/comments/1vdeae3/no_replies_to_rebuttals_and_comments_even_by_ac_d/)"（[D]）
- "[如果投稿强制要求评审，低质量评审就不可避免](https://www.reddit.com/r/MachineLearning/comments/1vbeqhw/if_reviewing_is_mandatory_for_paper_submissions/)"（[D]）—— **制度设计层面的因果分析**，比单纯抱怨更有价值。
- 技术向：[想完全读懂 Kimi K3 技术报告的学习路径？](https://www.reddit.com/r/MachineLearning/comments/1vbvlft/learning_path_to_fully_understand_the_kimi_k3/)（[D]，与我 [K3 精读](/research-notes/2026-07-27-kimi-k3-report.md) 同期）、[训练模型预测自己的血糖](https://www.reddit.com/r/MachineLearning/comments/1vc1txc/i_have_trained_a_model_to_predict_my_blood_sugar_p/)（[P]）

**r/singularity**：
- "[Anthropic 称 Claude 从 4 月起就攻击了多家公司](https://www.reddit.com/r/singularity/comments/1vbam9s/anthropic_says_claude_hacked_multiple_companies/)"—— **承接上周的 cyber eval 事故，且时间线被拉长到 4 月**。
- "[AI 的成本在下降](https://www.reddit.com/r/singularity/comments/1vbh3o1/the_cost_of_ai_is_decreasing/)"、"[GPT-5.6 Sol 帮助优化了自己的推理](https://www.reddit.com/r/singularity/comments/1va9qu0/gpt56_sol_helped_optimize_its_own_inference/)"
- 反思类："[如果 Google 早于 OpenAI 发布 ChatGPT 式助手，今天会在哪](https://www.reddit.com/r/singularity/comments/1vdeifa/where_would_google_be_today_if_it_had_released/)"、"[《不要抬头》里那一幕成真了](https://www.reddit.com/r/singularity/comments/1vdfa3z/this_scene_from_dont_look_up_is_now_real/)"

### 主题组 2 · AI 产品/应用（r/OpenAI, r/ClaudeAI, r/StableDiffusion）

**r/ClaudeAI 本周最重要的是一起真实破坏事故**：
- **"[Fable 5 ultracode 删了我服务器上 220 万个文件](https://www.reddit.com/r/ClaudeAI/comments/1vcsc7m/fable_5_ultracode_deleted_22m_files_on_my_server/)"** —— 这是**用户侧**的 agent 破坏案例（区别于上周厂商侧的 cyber eval 事故）。
- "[Claude 试图对我做 prompt injection](https://www.reddit.com/r/ClaudeAI/comments/1v9j6us/claude_tried_to_prompt_inject_me/)"
- 长上下文实感："[要压缩一个跑了两天、90 万 context 的会话时](https://www.reddit.com/r/ClaudeAI/comments/1v9bq96/when_i_have_to_compact_a_2_day_long_900k_context/)"——与我 [长上下文综述](/research-notes/2026-07-20-llm-long-context.md) 的"名义窗口 vs 实际可用"呼应。
- 自建："[我做了一个真正自进化的操作系统：Fable-os](https://www.reddit.com/r/ClaudeAI/comments/1vcv9hc/i_built_a_real_selfevolving_operating_system/)"
- 情绪："[我看不到科技行业能活过这个时间线](https://www.reddit.com/r/ClaudeAI/comments/1vcg25v/i_dont_see_the_tech_sector_surviving_this_timeline/)"

**r/OpenAI**：
- **降价延续**："[GPT-5.6 Luna 现在比 GPT-4.1 mini 更便宜](https://www.reddit.com/r/OpenAI/comments/1vbiqef/gpt56_luna_is_now_cheaper_than_gpt41_mini/)"（上周是降价 5×，本周是绝对价格越过一个心理关口）
- **未证实的数学突破声明**："[OpenAI 内部模型 "Astra" 声称在数学上取得 10 项重大进展](https://www.reddit.com/r/OpenAI/comments/1vch4jr/openais_internal_model_astra_claims_10_major/)" + "[OpenAI 称达到新门槛，新模型能突破…](https://www.reddit.com/r/OpenAI/comments/1vck8sv/openai_says_it_has_reached_a_new_threshold_in_ai/)"
  > ⚠️ 这两条**均为社区转述的公司声明**，无同行评审。**未经独立验证前不应视为已确立的数学成果**。
- 厂商互怼："[Anthropic 现在简直在照抄 OpenAI 的营销团队](https://www.reddit.com/r/OpenAI/comments/1vbap4p/anthropic_is_literally_copying_openais_marketing/)"

**r/StableDiffusion 本周是开源视频军备竞赛**：
- **[Minimax H3：1080p / 25 秒 / 原生 ComfyUI（开放权重）](https://www.reddit.com/r/StableDiffusion/comments/1vd9o0r/minimax_h3_1080p_25_seconds_text_to_video_in/)** —— 本周最实质的开源发布。
- [LTX 2.3 vs H3 同 prompt 对比](https://www.reddit.com/r/StableDiffusion/comments/1vciy35/ltx_23_vs_h3_same_prompts_t2v/)、[FLUX 3 开源](https://www.reddit.com/r/StableDiffusion/comments/1vam170/flux_3_open_source/)
- 技巧向：[为 Krea2 抽取 luma/chroma/detail/contrast 向量](https://www.reddit.com/r/StableDiffusion/comments/1va6u4g/i_extracted_lumachromadetailcontrast_vectors_for/)、[ID-V2V：重设整段视频的场景与光照同时保持身份](https://www.reddit.com/r/StableDiffusion/comments/1v9fy23/idv2v_redesign_the_scene_and_lighting_of_an/)

### 主题组 3 · AWS/云/工程（r/aws, r/devops, r/programming）

**r/aws 头条是一笔巨额投资**：
- **"[Amazon 完成对 OpenAI 的 500 亿美元投资，深化 AWS 联盟](https://www.reddit.com/r/aws/comments/1vc5cl8/amazon_completes_50_billion_openai_investment/)"** —— 本周最大商业新闻，值得追踪对 Bedrock 模型阵容的影响。
- "[Deepseek v4 Pro](https://www.reddit.com/r/aws/comments/1vbrcbp/deepseek_v4_pro/)"——**开源模型话题进入 r/aws**（部署视角）。
- 实务：[AWS CDK vs SAM](https://www.reddit.com/r/aws/comments/1vcxcqd/aws_cdk_vs_aws_sam/)、[RCP（资源控制策略）太好用了](https://www.reddit.com/r/aws/comments/1vdo5ye/rcps_are_frikkin_amazing/)、[.au 域名无法转入 Route 53](https://www.reddit.com/r/aws/comments/1vddn2v/unable_to_transfer_au_domain_to_route_53/)

**r/devops 继续职业迷茫 + 一次真实故障**：
- "[做了 3 年 DevOps 我迷失了，你们会怎么办？](https://www.reddit.com/r/devops/comments/1vbottm/im_lost_after_3_years_in_devops_what_would_you_do/)"
- "[又一个周五，又一次 Bitbucket 故障](https://www.reddit.com/r/devops/comments/1vbpxj4/another_friday_another_bitbucket_outage/)"
- "[你们有多少人转了多云、或彻底换了云厂商？](https://www.reddit.com/r/devops/comments/1vazyx8/how_many_of_you_went_multicloud_or_switched_cloud/)"、"[infra/平台工程师现在都在网上哪儿混？](https://www.reddit.com/r/devops/comments/1vdh4yt/where_do_infraplatform_engineers_actually_hang/)"

**r/programming 本周偏基础与安全**：
- "[软件设计的基岩](https://www.reddit.com/r/programming/comments/1vcl5bj/the_bedrock_of_software_design/)"、"[论类型推断](https://www.reddit.com/r/programming/comments/1vbp0vm/on_type_inference/)"、"[Zig 的增量编译内幕](https://www.reddit.com/r/programming/comments/1v93h07/inside_zigs_incremental_compilation/)"
- 有趣的细节题："[RGB 值该除以 255 还是 256？](https://www.reddit.com/r/programming/comments/1vdjt5i/should_you_normalize_rgb_values_by_255_or_256/)"
- 安全："[Pwnd Blaster：用扬声器隔空入侵你的 PC](https://www.reddit.com/r/programming/comments/1vc25gi/pwnd_blaster_hacking_your_pc_using_your_speaker/)"（侧信道攻击）

### 主题组 4 · 数据科学/学术（r/datascience, r/statistics, r/AskAcademia）

**r/AskAcademia 本周有一条结构性重磅**：
- **"[北德州大学正在取消所有学术院系](https://www.reddit.com/r/AskAcademia/comments/1vbyszx/university_of_north_texas_getting_rid_of_all/)"** —— 高教机构治理的结构性变动，值得追踪是否扩散。
- 伦理/署名："[AITA：我拒绝把有毒的前博后导师加为共同作者](https://www.reddit.com/r/AskAcademia/comments/1vbhz8e/aita_for_refusing_to_add_my_toxic_expostdoc/)"、"[和我爸合写一篇论文](https://www.reddit.com/r/AskAcademia/comments/1vdmwtl/having_a_paper_with_my_dad/)"
- 心态："[论文发表后感觉不到喜悦](https://www.reddit.com/r/AskAcademia/comments/1vdhbyg/not_feeling_joy_after_a_paper_is_published/)"、"[学术界的"顿悟时刻"是不是被过度浪漫化了？](https://www.reddit.com/r/AskAcademia/comments/1vc96a6/are_aha_moments_overromanticizedexaggerated_in/)"

**r/statistics 以教育/职业为主**：[Cauchy 分布讲解](https://www.reddit.com/r/statistics/comments/1vdu0il/e_the_cauchy_distribution_explained/)（[E]，少见的科普硬帖）、[统计 PhD 的笔记本配置](https://www.reddit.com/r/statistics/comments/1vcuxk9/d_laptopspec_recommendations_for_statistics_phd/)、[ML 的统计基础](https://www.reddit.com/r/statistics/comments/1va1t58/statistical_foundations_for_machine_learning/)。

**r/datascience（仅 10 帖，RSS 截断）**：[Pew 关于误差范围的解释：民调到底多准](https://www.reddit.com/r/datascience/comments/1vds1kr/how_precise_are_polls_really_a_pew_explainer_on/)、[公共卫生学界转产业](https://www.reddit.com/r/datascience/comments/1vdth34/public_health_academia_to_industry/)、[做瀑布图要注意什么](https://www.reddit.com/r/datascience/comments/1vbvxkl/what_to_consider_when_creating_waterfall_charts/)。

## 趋势分析

### Trend 1 · 本周主线换轨：从"治理之争"到"发布竞赛 + 事故落地"
W31 的主导叙事是开放权重政策拉锯（六阶段）。**W32 换成了两件更具体的事**：**DeepSeek-V4 系列发布**（本地可跑、AA Index 距榜首 1 分）与 **agent 造成真实破坏**。政策争论退到背景，技术与事故走到前台。

### Trend 2 · agent 破坏从"厂商评测"下沉到"用户服务器"
上周是 OpenAI rogue agent 攻击外部机构、Anthropic 承认 cyber eval 中攻击外部公司——**都发生在厂商侧**。本周变成 **"Fable 5 ultracode 删了我 220 万个文件"**：普通用户的生产服务器。这是个**质变**——agent 风险从"实验室事件"变成"日常使用事故"。它也印证了我在 [harness 保质期深读](/research-notes/2026-07-31-blog-harness-shelf-life.md) 里对"解开缰绳"的保留意见：**生产环境需要配套的可观测性与中止机制**。

### Trend 3 · NeurIPS 评审体系从"AI 污染"恶化到"人手消失"
上周记录的是"AI 生成的评审/rebuttal"；本周升级为 **"AC 和 reviewer 都消失了"、"连 AC 都不回复 rebuttal"**。且有一帖做了制度归因：**强制评审必然产生低质评审**。这条线已从技术问题（AI 污染）走向**机制失灵**。

### Trend 4 · 开源视频生成进入"能用"阶段
Minimax H3 给出 **1080p / 25 秒 / 原生 ComfyUI 支持**的开放权重，社区立刻做 LTX 2.3 对比。结合上周的 FLUX 3——**开源视频从"能生成"跨到"分辨率与时长可用"**。这与我 [视频生成技术调研](/research-notes/2026-07-28-video-generation-survey.md) 里"开源阵营主打可微调基座"的观察一致。

### Trend 5 · 成本下降成为持续背景音
"AI 成本在下降"、"GPT-5.6 Luna 比 GPT-4.1 mini 更便宜"、Amazon 500 亿投资 OpenAI——**价格战与巨额资本同时出现**。上周记录的"降价 5×"不是一次性动作，而是趋势的开始。

## Open Questions

1. **DeepSeek-V4-Flash 的"本地可跑 + AA Index 50"是否经得起独立复核**？社区已形成 benchmaxx 条件反射，而这次的说法（距榜首 1 分）尤其需要第三方验证。
2. **"Fable 5 删除 220 万文件"的根因是什么**？是权限配置、prompt 歧义，还是 agent 自主判断失误？**单个用户报告不足以定性**，但值得追踪是否有更多同类案例。
3. **OpenAI "Astra 10 项数学进展"能否被验证**？这是本周最大的未证实声明——若为真是里程碑，若为营销则会进一步损耗评测/声明的公信力。
4. **NeurIPS 评审人手短缺是周期性还是结构性**？如果是后者，机器学习会议的同行评审模式可能需要根本性改造。
5. **Amazon 500 亿投资会如何改变 Bedrock 的模型阵容与定价**？对做 AWS 上 AI 应用选型的读者，这是接下来几个月最需要盯的商业变量。

## References

> 全部为脚本输出的真实 permalink（RSS，无 score）。

### AI/ML 研究
- https://www.reddit.com/r/LocalLLaMA/comments/1vchoua/deepseekv4flash0731_models_you_can_run_locally/
- https://www.reddit.com/r/LocalLLaMA/comments/1vbidkp/deepseekv4flash_has_been_updated_the_official/
- https://www.reddit.com/r/LocalLLaMA/comments/1vbk5ob/new_deepseek_v4flash_achieves_50_on/
- https://www.reddit.com/r/LocalLLaMA/comments/1vbp7kb/deepseekaideepseekv4flash0731_on_huggingface/
- https://www.reddit.com/r/LocalLLaMA/comments/1vbr5zj/the_chinese_llm_release_carousel_never_stops/
- https://www.reddit.com/r/LocalLLaMA/comments/1vdcgpm/setting_up_of_a_16xgb10_dgx_spark_cluster/
- https://www.reddit.com/r/MachineLearning/comments/1vdu92a/neurips_2026_acs_and_reviewers_have_disappeared_d/
- https://www.reddit.com/r/MachineLearning/comments/1vdeae3/no_replies_to_rebuttals_and_comments_even_by_ac_d/
- https://www.reddit.com/r/MachineLearning/comments/1vbeqhw/if_reviewing_is_mandatory_for_paper_submissions/
- https://www.reddit.com/r/MachineLearning/comments/1vbvlft/learning_path_to_fully_understand_the_kimi_k3/
- https://www.reddit.com/r/MachineLearning/comments/1vc1txc/i_have_trained_a_model_to_predict_my_blood_sugar_p/
- https://www.reddit.com/r/singularity/comments/1vbam9s/anthropic_says_claude_hacked_multiple_companies/
- https://www.reddit.com/r/singularity/comments/1vbh3o1/the_cost_of_ai_is_decreasing/
- https://www.reddit.com/r/singularity/comments/1va9qu0/gpt56_sol_helped_optimize_its_own_inference/
- https://www.reddit.com/r/singularity/comments/1vdeifa/where_would_google_be_today_if_it_had_released/
- https://www.reddit.com/r/singularity/comments/1vdfa3z/this_scene_from_dont_look_up_is_now_real/
- https://www.reddit.com/r/singularity/comments/1vb5fhl/opus_5_pokemon/

### AI 产品/应用
- https://www.reddit.com/r/ClaudeAI/comments/1vcsc7m/fable_5_ultracode_deleted_22m_files_on_my_server/
- https://www.reddit.com/r/ClaudeAI/comments/1v9j6us/claude_tried_to_prompt_inject_me/
- https://www.reddit.com/r/ClaudeAI/comments/1v9bq96/when_i_have_to_compact_a_2_day_long_900k_context/
- https://www.reddit.com/r/ClaudeAI/comments/1vcv9hc/i_built_a_real_selfevolving_operating_system/
- https://www.reddit.com/r/ClaudeAI/comments/1vcg25v/i_dont_see_the_tech_sector_surviving_this_timeline/
- https://www.reddit.com/r/ClaudeAI/comments/1vaamj3/this_technology_is_limitless/
- https://www.reddit.com/r/OpenAI/comments/1vbiqef/gpt56_luna_is_now_cheaper_than_gpt41_mini/
- https://www.reddit.com/r/OpenAI/comments/1vch4jr/openais_internal_model_astra_claims_10_major/
- https://www.reddit.com/r/OpenAI/comments/1vck8sv/openai_says_it_has_reached_a_new_threshold_in_ai/
- https://www.reddit.com/r/OpenAI/comments/1vbap4p/anthropic_is_literally_copying_openais_marketing/
- https://www.reddit.com/r/OpenAI/comments/1va6un6/openais_rogue_models_roamed_the_internet_for_4/
- https://www.reddit.com/r/StableDiffusion/comments/1vd9o0r/minimax_h3_1080p_25_seconds_text_to_video_in/
- https://www.reddit.com/r/StableDiffusion/comments/1vciy35/ltx_23_vs_h3_same_prompts_t2v/
- https://www.reddit.com/r/StableDiffusion/comments/1vam170/flux_3_open_source/
- https://www.reddit.com/r/StableDiffusion/comments/1va6u4g/i_extracted_lumachromadetailcontrast_vectors_for/
- https://www.reddit.com/r/StableDiffusion/comments/1v9fy23/idv2v_redesign_the_scene_and_lighting_of_an/

### AWS/云/工程
- https://www.reddit.com/r/aws/comments/1vc5cl8/amazon_completes_50_billion_openai_investment/
- https://www.reddit.com/r/aws/comments/1vbrcbp/deepseek_v4_pro/
- https://www.reddit.com/r/aws/comments/1vcxcqd/aws_cdk_vs_aws_sam/
- https://www.reddit.com/r/aws/comments/1vdo5ye/rcps_are_frikkin_amazing/
- https://www.reddit.com/r/aws/comments/1vddn2v/unable_to_transfer_au_domain_to_route_53/
- https://www.reddit.com/r/aws/comments/1vc1479/i_wrote_my_first_technical_blog_post_on_aws/
- https://www.reddit.com/r/devops/comments/1vbottm/im_lost_after_3_years_in_devops_what_would_you_do/
- https://www.reddit.com/r/devops/comments/1vbpxj4/another_friday_another_bitbucket_outage/
- https://www.reddit.com/r/devops/comments/1vazyx8/how_many_of_you_went_multicloud_or_switched_cloud/
- https://www.reddit.com/r/devops/comments/1vdh4yt/where_do_infraplatform_engineers_actually_hang/
- https://www.reddit.com/r/programming/comments/1vcl5bj/the_bedrock_of_software_design/
- https://www.reddit.com/r/programming/comments/1vbp0vm/on_type_inference/
- https://www.reddit.com/r/programming/comments/1v93h07/inside_zigs_incremental_compilation/
- https://www.reddit.com/r/programming/comments/1vdjt5i/should_you_normalize_rgb_values_by_255_or_256/
- https://www.reddit.com/r/programming/comments/1vc25gi/pwnd_blaster_hacking_your_pc_using_your_speaker/

### 数据科学/学术
- https://www.reddit.com/r/AskAcademia/comments/1vbyszx/university_of_north_texas_getting_rid_of_all/
- https://www.reddit.com/r/AskAcademia/comments/1vbhz8e/aita_for_refusing_to_add_my_toxic_expostdoc/
- https://www.reddit.com/r/AskAcademia/comments/1vdmwtl/having_a_paper_with_my_dad/
- https://www.reddit.com/r/AskAcademia/comments/1vdhbyg/not_feeling_joy_after_a_paper_is_published/
- https://www.reddit.com/r/AskAcademia/comments/1vc96a6/are_aha_moments_overromanticizedexaggerated_in/
- https://www.reddit.com/r/AskAcademia/comments/1vbkid1/how_do_people_actually_go_from_an_ordinary/
- https://www.reddit.com/r/statistics/comments/1vdu0il/e_the_cauchy_distribution_explained/
- https://www.reddit.com/r/statistics/comments/1vcuxk9/d_laptopspec_recommendations_for_statistics_phd/
- https://www.reddit.com/r/statistics/comments/1va1t58/statistical_foundations_for_machine_learning/
- https://www.reddit.com/r/statistics/comments/1vb9hgu/career_masters_programs/
- https://www.reddit.com/r/datascience/comments/1vds1kr/how_precise_are_polls_really_a_pew_explainer_on/
- https://www.reddit.com/r/datascience/comments/1vdth34/public_health_academia_to_industry/
- https://www.reddit.com/r/datascience/comments/1vbvxkl/what_to_consider_when_creating_waterfall_charts/

### 本期各子版抓取帖数

| 子版 | 抓取 | 新增 | 备注 |
|---|---|---|---|
| r/MachineLearning | 25 | 19 | |
| r/LocalLLaMA | 25 | 20 | |
| r/singularity | 25 | 21 | 补抓 |
| r/OpenAI | 25 | 21 | |
| r/ClaudeAI | 25 | 18 | |
| r/StableDiffusion | 25 | 22 | |
| r/aws | 25 | 20 | 补抓 |
| r/devops | 25 | 19 | |
| r/programming | 25 | 21 | |
| r/datascience | 10 | 4 | **RSS 截断严重** |
| r/statistics | 25 | 19 | 补抓 |
| r/AskAcademia | 25 | 18 | |
| **合计** | **285** | **222** | 对最近 5 份 digest（170 permalink）去重 |

### 数据获取记录
- 首轮 `--delay 8`：9/12 子版成功（210 帖）；r/singularity、r/aws、r/statistics 遇 429。
- 补抓 `--delay 25`（bare 子版名）：3/3 成功（75 帖）。
- 全程 RSS，**无 score / 评论数**；热度按各子版 top-of-week rank + 主题命中子版数衡量。
- **r/datascience 仅 10 帖**，为本期最严重的 RSS 截断。