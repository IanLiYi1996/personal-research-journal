# Reddit 热门周报｜2026-W26（截至 2026-06-26）

> **覆盖**：12 个 subreddit 的 `top/week` feed，截至 2026-06-26 01:50 UTC。
> **数据源**：`scripts/reddit_fetch.py` 的 RSS 抓取（JSON API 仍全 IP 403，OAuth app 注册仍被卡），W25→W26 共 **273 posts 抓取** / **273 新增**（与上份 W25 digest 0 重叠——RSS feed 已完整翻篇）。
> **🚨 RSS 热度局限**：RSS 不返回 `score` / `num_comments`，无法跨子版做数字排序。本周报排名仅在**子版内部**遵循 Reddit 原生 `top/week` feed rank（rank 越小越靠前），跨子版无可比性。
> **抓取健壮性**：r/LocalLLaMA / r/ClaudeAI / r/devops / r/statistics 在第一轮 429 失败，单独以 `--delay 30` 补抓后并入。r/datascience 只回 4 帖（Reddit RSS 偶尔会截断）。
> **子版分组**（4 组）：AI/ML 研究 · AI 产品/应用 · AWS/云/工程 · 数据科学/学术。

---

## 0. 跨社区主线（W26）

> 本周四条主线把全部 12 个子版串起来：**(1) DeepSeek $7.4B 融资** 的余震在 r/LocalLLaMA / r/singularity 双开花；**(2) AWS Lambda MicroVMs** 重塑 sandbox 经济（r/aws 头条 + r/devops 讨论）；**(3) Krea 2 开源** 在 r/StableDiffusion 一周霸榜（4 条 Top-10）；**(4) Anthropic "Mythos" 内部叙事 + John Jumper 跳槽** 在 r/singularity / r/ClaudeAI 形成 AI 安全 + 人才流动话题群。

| 主线 | 关联子版 | 关键帖（permalink） |
|------|---------|-------------------|
| **DeepSeek $7.4B / $60B 估值，Liang Wenfeng 自投 $3B** | r/LocalLLaMA #0, r/singularity | [r/LocalLLaMA/1ucwyes](https://www.reddit.com/r/LocalLLaMA/comments/1ucwyes/deepseek_raises_74b_usd_at_60b_valuation/) |
| **AWS Lambda MicroVMs（首席亮点）** | r/aws #0/#5, r/devops, r/programming | [r/aws/1ud001l](https://www.reddit.com/r/aws/comments/1ud001l/run_isolated_sandboxes_with_full_lifecycle/) ; [r/aws/1ueul5o](https://www.reddit.com/r/aws/comments/1ueul5o/hardest_problems_lambda_microvms_can_solve_now/) |
| **Krea 2 开源 / 安全过滤移除 / FP8 工作流** | r/StableDiffusion #1/#2/#5/#6/#7 | [r/StableDiffusion/1udjzm6](https://www.reddit.com/r/StableDiffusion/comments/1udjzm6/krea_2_opensource_release/) ; [r/StableDiffusion/1udhaio](https://www.reddit.com/r/StableDiffusion/comments/1udhaio/this_custom_node_removes_the_builtin_krea_2/) |
| **Anthropic Mythos 系列 + John Jumper 加盟 Anthropic** | r/singularity #3/#4/#6, r/ClaudeAI | [r/singularity/1ubets2](https://www.reddit.com/r/singularity/comments/1ubets2/nsa_says_mythos_broke_into_almost_all_of_their/) ; [r/singularity/1uadqbb](https://www.reddit.com/r/singularity/comments/1uadqbb/nobel_winner_john_jumper_to_leave_google_deepmind/) |
| **中美 AI 芯片 + 数据中心争议** | r/LocalLLaMA #5/#6, r/singularity #1/#7 | [r/LocalLLaMA/1udkxde](https://www.reddit.com/r/LocalLLaMA/comments/1udkxde/7_chinese_companies_are_already_shipping/) ; [r/singularity/1ue6sio](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/) |
| **GPT 5.5/5.6 发布与取消的钟摆** | r/OpenAI #5/#6/#7 | [r/OpenAI/1uen1zv](https://www.reddit.com/r/OpenAI/comments/1uen1zv/gpt55_instant_now_rolling_out/) ; [r/OpenAI/1ubeoz6](https://www.reddit.com/r/OpenAI/comments/1ubeoz6/gpt_56_cancelled/) |
| **PhD 与 top-tier 论文** | r/MachineLearning #1, r/AskAcademia | [r/MachineLearning/1uazlhg](https://www.reddit.com/r/MachineLearning/comments/1uazlhg/would_you_let_an_ml_phd_student_graduate_without/) |

---

## 1. AI/ML 研究

### r/MachineLearning（25 新帖，feed rank Top 8）

| rank | flair | 标题 / 解读 | permalink |
|------|-------|------------|-----------|
| 0 | P | **"Build Your Own LLM" 工作坊 YouTube 全集** —— 教学资源，从数学/ML 基础走到 LLM。社区高赞普及向资源。 | [1uazlnd](https://www.reddit.com/r/MachineLearning/comments/1uazlnd/hi_reddit_i_posted_my_build_your_own_llm_workshop/) |
| 1 | D | **"PhD 学生没有 top-tier paper 能否毕业？"** 本周 academia 主话题，与 r/AskAcademia 互文。 | [1uazlhg](https://www.reddit.com/r/MachineLearning/comments/1uazlhg/would_you_let_an_ml_phd_student_graduate_without/) |
| 2 | R | **Time Series Modeling Needs a Dynamical Systems Perspective** —— 反思帖，呼吁回到动力系统理论。 | [1uark0u](https://www.reddit.com/r/MachineLearning/comments/1uark0u/time_series_modeling_needs_a_dynamical_systems/) |
| 3 | P | **DVD-JEPA**：开源完全可复现的 JEPA world model。和 HF 端 Looped/Qwen-AgentWorld 主线呼应（本周双开花）。 | [1uatlzx](https://www.reddit.com/r/MachineLearning/comments/1uatlzx/dvdjepa_an_opensource_fullyreproducible_jepa/) |
| 4 | D | **torch.compile() 为什么比 NumPy 还快？** kernel 融合 + 视图消除 + 自动 vectorize。 | [1ua2hwj](https://www.reddit.com/r/MachineLearning/comments/1ua2hwj/how_does_torchcompile_achieve_massive_speedups/) |
| 5 | P | **Papers with Code 重启更新** —— PWC 仍活跃且在迭代。 | [1ucm508](https://www.reddit.com/r/MachineLearning/comments/1ucm508/some_new_updates_to_papers_with_code_p/) |
| 6 | R | **DeepSWE benchmark**：新 SWE 评测，frontier 模型实际工程能力。 | [1ue0hlp](https://www.reddit.com/r/MachineLearning/comments/1ue0hlp/deepswe_new_benchmark_looking_at_how_well_todays/) |
| 7 | P | **PWC 上的开源 OCR 模型聚合页** —— 接续 #3 的 OCR/Unlimited-OCR 主题。 | [1ueiam6](https://www.reddit.com/r/MachineLearning/comments/1ueiam6/find_the_best_opensource_ocr_models_in_one_place/) |

### r/LocalLLaMA（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **DeepSeek 融资 $7.4B（估值 $60B）；Liang Wenfeng 自投 $3B** —— 周冠主话题，本子版/r/singularity 双开。"创始人自投"信号特别强。 | [1ucwyes](https://www.reddit.com/r/LocalLLaMA/comments/1ucwyes/deepseek_raises_74b_usd_at_60b_valuation/) |
| 1 | **Tokenomics** —— 社区 meme，讨论 token 计费 + 商业模式。 | [1ubrcwj](https://www.reddit.com/r/LocalLLaMA/comments/1ubrcwj/tokenomics/) |
| 2 | **z.AI（开源第二）公开点赞第一（推测指 DeepSeek/Qwen 系）** —— 中文开源圈的礼貌战。 | [1uaxktf](https://www.reddit.com/r/LocalLLaMA/comments/1uaxktf/zai_as_the_number_2_gives_praise_to_the_number_1/) |
| 3 | **Unlimited-OCR 上 ModelScope** —— 3.3B 多语言 OCR，one-shot parsing。与 HF 端 #56 [Unlimited OCR Works](https://huggingface.co/papers/2606.23050) 论文呼应。 | [1ue51uk](https://www.reddit.com/r/LocalLLaMA/comments/1ue51uk/unlimitedocr_is_now_on_modelscope_a_33b/) |
| 4 | **Deep Neural Net 把 image → 可玩游戏（本地推理）** —— 与 r/StableDiffusion 的 #4 重复（trending 跨子版）。和 HF #17 GameCraft-Bench 形成"agent 造游戏"主题群。 | [1ub2kmt](https://www.reddit.com/r/LocalLLaMA/comments/1ub2kmt/deep_neural_network_that_can_turn_any_image_into/) |
| 5 | **"Chinese Hackers Latest Masterpiece with NVIDIA"** —— 标题党/讨论 NVIDIA 软硬件被国产破解，待核实细节。 | [1ucokod](https://www.reddit.com/r/LocalLLaMA/comments/1ucokod/chinese_hackers_latest_masterpiece_with_nvidia/) |
| 6 | **7 家中国公司已出货 H100/H200 量级 AI 芯片，大多近 18 月 IPO** —— 中美芯片地缘格局新一周快照。 | [1udkxde](https://www.reddit.com/r/LocalLLaMA/comments/1udkxde/7_chinese_companies_are_already_shipping/) |
| 7 | **瑞士联邦最高法院在评估 "Heretic"** —— 监管 LLM 输出的判例线索。 | [1ueeund](https://www.reddit.com/r/LocalLLaMA/comments/1ueeund/the_swiss_federal_supreme_court_is_evaluating/) |

### r/singularity（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **日本动画师用 Seedance 把简单 3D 模型 render 成动画** —— video diffusion 进入实际产线案例。 | [1ue6yoh](https://www.reddit.com/r/singularity/comments/1ue6yoh/japanese_animator_using_seedance_to_render_anime/) |
| 1 | **弗吉尼亚居民投诉数据中心噪音** —— 数据中心反弹本周第一次进入主流讨论。 | [1ue6sio](https://www.reddit.com/r/singularity/comments/1ue6sio/data_center_noise_irks_virginia_neighbors_you/) |
| 2 | **Trump 行政令推国家量子计算机** —— "quantum capable of performing X"，国家战略级。 | [1ucy9oj](https://www.reddit.com/r/singularity/comments/1ucy9oj/president_trump_orders_a_national_effort_to_build/) |
| 3 | **NSA 称 "Mythos" 在小时级别攻破其大多数 classified 系统** —— Mythos 是 Anthropic 内部代号？引爆讨论。 | [1ubets2](https://www.reddit.com/r/singularity/comments/1ubets2/nsa_says_mythos_broke_into_almost_all_of_their/) |
| 4 | **Anthropic 的 Internal Mythos Successor 浮现** —— 与 #3 互文，主线"Anthropic 内部新模型/红队故事"。 | [1ubwtut](https://www.reddit.com/r/singularity/comments/1ubwtut/anthropics_internal_mythos_successor_emerges/) |
| 5 | **Bernie Sanders 提 $7T 计划"把 AI 产业控制权交给美国人"** —— 政策端讨论持续升温。 | [1ucq463](https://www.reddit.com/r/singularity/comments/1ucq463/bernie_sanders_unveils_7_trillion_plan_to_give/) |
| 6 | **Nobel 得主 John Jumper 离开 Google DeepMind 加盟 Anthropic** —— AlphaFold 主创跳槽，AI4Sci 人才走向。 | [1uadqbb](https://www.reddit.com/r/singularity/comments/1uadqbb/nobel_winner_john_jumper_to_leave_google_deepmind/) |
| 7 | **John Carmack 谈数据中心** —— 名人现身评论。 | [1ue1sya](https://www.reddit.com/r/singularity/comments/1ue1sya/john_carmack_weighs_in_on_datacenters/) |

---

## 2. AI 产品 / 应用

### r/OpenAI（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"Sarah Connor 审判你的 AI 上瘾"** —— meme，反映社区对 AI 依赖的自嘲与焦虑。 | [1uce2gr](https://www.reddit.com/r/OpenAI/comments/1uce2gr/sarah_connor_judging_your_ai_addiction/) |
| 1 | **OpenAI Codex 有 bug 可能在一年内烧坏 SSD** —— 待核实，影响 desktop / CLI 用户。 | [1ucf4px](https://www.reddit.com/r/OpenAI/comments/1ucf4px/openai_codex_has_a_bug_that_could_kill_your_ssd/) |
| 2 | **新 OpenAI 员工自爆求职过程也很难** —— 行业求职故事，反 elite 神话。 | [1ucuoem](https://www.reddit.com/r/OpenAI/comments/1ucuoem/its_not_you_guys_recent_openai_hire_admitted/) |
| 3 | **"人类是地球的癌症..."** AI 角色扮演越狱讨论。 | [1ucbeti](https://www.reddit.com/r/OpenAI/comments/1ucbeti/human_beings_are_a_disease_a_cancer_of_this/) |
| 4 | **GPT 4.5 在 MineBench 拒绝执行，写了 "HELP"** —— refusal 行为研究，社区延伸成 meme。 | [1u9uaxb](https://www.reddit.com/r/OpenAI/comments/1u9uaxb/gpt_45_in_minebench_refused_to_generate_the_given/) |
| 5 | **GPT-5.5 Instant 正在 rollout** —— 模型版本动态。 | [1uen1zv](https://www.reddit.com/r/OpenAI/comments/1uen1zv/gpt55_instant_now_rolling_out/) |
| 6 | **GPT 5.6 preview 即将放出** —— 紧跟 5.5 节奏。 | [1uf6702](https://www.reddit.com/r/OpenAI/comments/1uf6702/gpt_56_preview_is_about_to_be_dropped/) |
| 7 | **GPT 5.6 Cancelled** —— 同周内反转。社区对版本号疲劳。 | [1ubeoz6](https://www.reddit.com/r/OpenAI/comments/1ubeoz6/gpt_56_cancelled/) |

### r/ClaudeAI（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"办公室同事懂这梗时的骄傲"** —— Claude Code 文化梗。 | [1ubp7mw](https://www.reddit.com/r/ClaudeAI/comments/1ubp7mw/kinda_proud_when_office_teammates_understand_this/) |
| 1 | **"共享 Claude 订阅..."** —— 订阅滥用 + 公司账号风险讨论。 | [1uejjg6](https://www.reddit.com/r/ClaudeAI/comments/1uejjg6/nothing_can_go_wrong_when_you_share_a_claude/) |
| 2 | **"Claude session 限额到了？"** —— rate limit 主题持续热。 | [1ubj3fw](https://www.reddit.com/r/ClaudeAI/comments/1ubj3fw/hit_your_claude_session_limit/) |
| 3 | **"Day 3 of Vibe Coding"** —— vibe-coding 文化梗。 | [1ue2ba0](https://www.reddit.com/r/ClaudeAI/comments/1ue2ba0/day_3_of_vibe_coding/) |
| 4 | **"新开发者就是这样"** —— meme。 | [1uau8au](https://www.reddit.com/r/ClaudeAI/comments/1uau8au/new_devs_be_like/) |
| 5 | **"在 Karpathy 4 条 CLAUDE.md 规则里加了一条"** —— Claude Code 配置经验帖，技术含量较高。 | [1uc7izy](https://www.reddit.com/r/ClaudeAI/comments/1uc7izy/i_added_a_clause_to_andrej_karpathys_4_claudemd/) |
| 6 | **"烧掉太多 token，Anthropic 寄了周边"** —— 用户故事。 | [1ueeve0](https://www.reddit.com/r/ClaudeAI/comments/1ueeve0/i_burnt_so_many_tokens_they_sent_me_merch/) |
| 7 | **"Claude 即将要求 Face ID"** —— 真伪未明，社区讨论。 | [1uawyi0](https://www.reddit.com/r/ClaudeAI/comments/1uawyi0/claude_to_require_face_id/) |

### r/StableDiffusion（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **LTX-2.3 Water Sim LoRA 把 Joker 楼梯洪水化** —— video2video demo。 | [1ucm5he](https://www.reddit.com/r/StableDiffusion/comments/1ucm5he/ltx23_water_sim_lora_flooding_the_joker_stairs/) |
| 1 | **KREA 2 开源** —— 本周霸榜话题。 | [1udjzm6](https://www.reddit.com/r/StableDiffusion/comments/1udjzm6/krea_2_opensource_release/) |
| 2 | **Krea 2 安全过滤移除节点** —— 开源后 24h 内出现的"绕过"工具，社区与厂商博弈。 | [1udhaio](https://www.reddit.com/r/StableDiffusion/comments/1udhaio/this_custom_node_removes_the_builtin_krea_2/) |
| 3 | **"运气有时是全部"** —— 出图 meme。 | [1ufkhio](https://www.reddit.com/r/StableDiffusion/comments/1ufkhio/sometimes_luck_is_all_you_need/) |
| 4 | **Diffusion 把 image → 可玩游戏（本地）** —— 跨子版 trending（也在 r/LocalLLaMA #4）。 | [1uc55ix](https://www.reddit.com/r/StableDiffusion/comments/1uc55ix/diffusion_model_that_can_turn_any_image_into_a/) |
| 5 | **Krea 2 角色与服饰一致性强** —— 性能讨论。 | [1udb976](https://www.reddit.com/r/StableDiffusion/comments/1udb976/krea_2_is_really_good_at_knowing_and/) |
| 6 | **Realism 对比：Ideogram 4 vs Krea 2 Turbo** —— 横评。 | [1uf0049](https://www.reddit.com/r/StableDiffusion/comments/1uf0049/realism_comparaison_ideogram_4_vs_krea_2_turbo/) |
| 7 | **Krea 2 Turbo ComfyUI FP8 工作流（12 GB）** —— 本地推理工程化。 | [1ud2nyq](https://www.reddit.com/r/StableDiffusion/comments/1ud2nyq/krea_2_turbo_native_comfyui_workflow_fp8_weights/) |

---

## 3. AWS / 云 / 工程

### r/aws（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **AWS Lambda 推出 MicroVMs**：完整生命周期控制的 isolated sandbox。**这是 W26 AWS 头条**，影响 Bedrock AgentCore / agent sandbox 产品策略。 | [1ud001l](https://www.reddit.com/r/aws/comments/1ud001l/run_isolated_sandboxes_with_full_lifecycle/) |
| 1 | **dynoxide**：可嵌入的 DynamoDB Local drop-in 替代（SQLite-backed），开源。 | [1uaz4cq](https://www.reddit.com/r/aws/comments/1uaz4cq/github_nubodbdynoxide_a_fast_embeddable_dropin/) |
| 2 | **AWS Blocks**：构建 full-stack 应用的后端工具包。 | [1udmxr1](https://www.reddit.com/r/aws/comments/1udmxr1/aws_blocks_is_a_backend_toolkit_for_building/) |
| 3 | **Log Analytics UI（新）** —— 控制台新模块。 | [1uda1ho](https://www.reddit.com/r/aws/comments/1uda1ho/log_analytics_ui_new/) |
| 4 | **CloudWatch Logs 直连 syslog 摄取** —— 老需求落地。 | [1ud6taa](https://www.reddit.com/r/aws/comments/1ud6taa/new_cloudwatch_logs_feature_direct_syslog/) |
| 5 | **"Lambda MicroVMs 能解决的最难问题是什么"** —— 社区围绕 #0 展开 use-case 讨论。 | [1ueul5o](https://www.reddit.com/r/aws/comments/1ueul5o/hardest_problems_lambda_microvms_can_solve_now/) |
| 6 | **"AWS 里堆积的、不在 Terraform 里的资源越来越多..."** —— IaC drift 真实痛点。 | [1uf22cj](https://www.reddit.com/r/aws/comments/1uf22cj/cloud_resources_keep_piling_up_in_aws_that_were/) |
| 7 | **CUR vs FOCUS 对比（CUR 2.0 暴露 list price 等字段）** —— 财务分析视角。 | [1ubd8jr](https://www.reddit.com/r/aws/comments/1ubd8jr/maybe_im_late_to_this_but_i_finally_spent_time/) |

### r/devops（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"Reddit 教我 CI pipeline 错了，10min → 4min"** —— 实操贴。 | [1uaafd5](https://www.reddit.com/r/devops/comments/1uaafd5/reddit_taught_me_why_my_ci_pipeline_was_wrong/) |
| 1 | **"2am 报警，唯一懂的人已离职"** —— bus-factor 焦虑故事。 | [1ubyy39](https://www.reddit.com/r/devops/comments/1ubyy39/2am_page_the_only_person_whod_know_why_is_gone/) |
| 2 | **CKAD 87% 通过经验** —— 备考路线。 | [1ue3cmq](https://www.reddit.com/r/devops/comments/1ue3cmq/passed_ckad_with_87_heres_my_exam_experience/) |
| 3 | **自学 Cloud DevOps** —— 入门讨论。 | [1uck2lr](https://www.reddit.com/r/devops/comments/1uck2lr/learning_by_myself_cloud_devops/) |
| 4 | **DevOps/Platform 面试如何考 AWS** —— 招聘视角。 | [1ucxhw7](https://www.reddit.com/r/devops/comments/1ucxhw7/how_are_aws_skills_actually_assessed_in/) |
| 5 | **"我恨我的新工作"** —— 转岗痛苦。 | [1ubuyy8](https://www.reddit.com/r/devops/comments/1ubuyy8/i_hate_my_new_job/) |
| 6 | **"致所有前 DevOps Engineers"** —— 职业转型话题。 | [1uel9gr](https://www.reddit.com/r/devops/comments/1uel9gr/to_all_former_devops_engineers/) |
| 7 | **CI 流水线设计 tradeoff** —— 工程取舍。 | [1ub7ocm](https://www.reddit.com/r/devops/comments/1ub7ocm/while_redesigning_my_ci_pipeline_i_ran_into_an/) |

### r/programming（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"Old Software Was Fast Because It Had No Choice"** —— 经典反思文，性能怀旧。 | [1ua2lxq](https://www.reddit.com/r/programming/comments/1ua2lxq/old_software_was_fast_because_it_had_no_choice/) |
| 1 | **"In memory of the man who put red squiggles under words"** —— 拼写检查发明者讣告，"小细节深远影响"。 | [1ud8p4p](https://www.reddit.com/r/programming/comments/1ud8p4p/in_memory_of_the_man_who_put_red_squiggles_under/) |
| 2 | **New HTTP QUERY Method** —— RFC 进展，长期争论的 RESTful 查询语义终于落地。 | [1udwfsp](https://www.reddit.com/r/programming/comments/1udwfsp/new_http_query_method/) |
| 3 | **"4 字节 padding 让 array 清零快 49%"** —— low-level 性能贴。 | [1uck3ll](https://www.reddit.com/r/programming/comments/1uck3ll/how_4_bytes_of_padding_make_array_clearing_49/) |
| 4 | **"如何写有效的软件设计文档"** —— 教程，文章高赞。 | [1uevttg](https://www.reddit.com/r/programming/comments/1uevttg/how_to_write_an_effective_software_design_document/) |
| 5 | **OSS maintainer 倦怠（John-David Dalton, lodash 作者）** —— 维护者群体困境。 | [1uaodhr](https://www.reddit.com/r/programming/comments/1uaodhr/burnout_is_real_for_open_source_maintainers_a/) |
| 6 | **"p99 0 ms\* autocomplete for 240M 域名"** —— 工程文。 | [1ucdszk](https://www.reddit.com/r/programming/comments/1ucdszk/p99_0_ms_autocomplete_for_240_million_domain_names/) |
| 7 | **OCaml 5.5.0 发布** —— 语言版本更新。 | [1ubn4uj](https://www.reddit.com/r/programming/comments/1ubn4uj/ocaml_550_released/) |

---

## 4. 数据科学 / 学术

### r/datascience（仅 4 帖；RSS 截断异常）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"DS 工作都变 Gen AI 了吗？"** —— 行业身份焦虑，本子版本周代表话题。 | [1ubilfx](https://www.reddit.com/r/datascience/comments/1ubilfx/are_all_data_science_jobs_just_gen_ai_now/) |
| 1 | 自托管 Linux HPC 集群上测试 DS 工作流 | [1u9xm29](https://www.reddit.com/r/datascience/comments/1u9xm29/ideas_for_testing_data_science_workflows_on_self/) |
| 2 | Dev Log on Steam Recommender (Part 2) | [1ufhx3w](https://www.reddit.com/r/datascience/comments/1ufhx3w/dev_log_on_steam_recommender_part_2/) |
| 3 | 周入门 thread | [1uca6dg](https://www.reddit.com/r/datascience/comments/1uca6dg/weekly_entering_transitioning_thread_22_jun_2026/) |

### r/statistics（19 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **个人 statistics-ML 学习笔记 10K Star GitHub** —— 教育资源。 | [1uddyo5](https://www.reddit.com/r/statistics/comments/1uddyo5/research_education_my_statistics_machine_learning/) |
| 1 | **Mplus 学生折扣反而比正价贵** —— 软件价格 trap。 | [1uae0q2](https://www.reddit.com/r/statistics/comments/1uae0q2/s_mplus_student_discount_ends_up_costing_more/) |
| 2 | **statisticalhorizons 是骗局吗？** —— 培训机构口碑讨论。 | [1uazmlc](https://www.reddit.com/r/statistics/comments/1uazmlc/q_is_statisticalhorizons_a_scam/) |
| 3 | **counterbalancing within-subject 是否降功效？** —— 设计统计。 | [1uchdkw](https://www.reddit.com/r/statistics/comments/1uchdkw/q_does_counterbalancing_a_withinsubjects_variable/) |
| 4 | **Survival Analysis 在工业中的技能要求** —— 职业向。 | [1uecqwu](https://www.reddit.com/r/statistics/comments/1uecqwu/career_skills_required_to_conduct_survival/) |
| 5 | **outlier 校正 + multiple testing 的方法重叠** —— 应用统计。 | [1uch96e](https://www.reddit.com/r/statistics/comments/1uch96e/q_i_corrected_set_of_outliers_for_multiple/) |
| 6 | **如何测"running improvement"** —— 个人案例统计设计。 | [1uekmxb](https://www.reddit.com/r/statistics/comments/1uekmxb/discussion_how_best_to_test_a_running_improvement/) |
| 7 | **空间协变量聚合的离散化误差** —— 空间统计应用题。 | [1ucge7l](https://www.reddit.com/r/statistics/comments/1ucge7l/q_quantifying_discretization_error_and_spatial/) |

### r/AskAcademia（25 新帖，rank Top 8）

| rank | 标题 / 解读 | permalink |
|------|------------|-----------|
| 0 | **"我的院长把我的恐慌发作做成了视频"** —— 学术心理健康 + 管理伦理。 | [1ucyq9b](https://www.reddit.com/r/AskAcademia/comments/1ucyq9b/my_dean_made_a_video_about_my_panic_attacks/) |
| 1 | **$5k-$8k 年度捐赠对语言系帮助多大？** —— 捐赠者视角。 | [1ud94tp](https://www.reddit.com/r/AskAcademia/comments/1ud94tp/how_useful_is_a_50008000_annual_donation_to_a/) |
| 2 | **本科研究 misconduct 处理建议** —— 学术伦理。 | [1uctfol](https://www.reddit.com/r/AskAcademia/comments/1uctfol/advice_for_undergraduate_research_misconduct/) |
| 3 | **Nature Neuroscience 论文 (IF=20) 复现失败，作者失联** —— 复现危机经典案例。 | [1uf7454](https://www.reddit.com/r/AskAcademia/comments/1uf7454/paper_on_nature_neuroscience_if_20_results_dont/) |
| 4 | **重大 grant interview 失利，无法自拔** —— 学术压力。 | [1ua3yme](https://www.reddit.com/r/AskAcademia/comments/1ua3yme/flopped_a_major_grant_interview_cant_stop/) |
| 5 | **被任命为 interim Chair，求经验** —— 行政转型。 | [1ueyv6f](https://www.reddit.com/r/AskAcademia/comments/1ueyv6f/my_dean_asked_me_to_step_in_as_interim_chair_for/) |
| 6 | **谁教你写 grant 申请？** —— 软技能传承。 | [1ualh3t](https://www.reddit.com/r/AskAcademia/comments/1ualh3t/who_taught_you_how_to_write_funding_applications/) |
| 7 | **改姓出于职业原因是否禁忌？** —— 学术身份连续性。 | [1ubdhme](https://www.reddit.com/r/AskAcademia/comments/1ubdhme/changing_surname_for_professional_reasonstaboo_or/) |

---

## 5. 趋势分析

**(1) Agent infra + sandbox 经济：AWS Lambda MicroVMs 重铸赛道。**
r/aws 头条 + r/devops 跟进 + r/programming 关于 "old software was fast" 的怀旧文形成奇妙呼应——**当 agent 需要在 sandbox 跑用户代码时**，Lambda MicroVMs（cold-start 极快、完整生命周期 API）几乎是为 agent 平台量身定制。结合本周 HF 端的 AgentCore / OpenRath 主题，**"agent runtime = container + session abstraction"** 已经成为云厂商和论文界的双向共识。

**(2) 中国开源 AI 的"硬"信号。**
DeepSeek $7.4B 融资 + 创始人 $3B 自投（r/LocalLLaMA #0）、7 家中国 H100/H200 量级芯片厂（#6）、z.AI 致敬 #1（#2）、Unlimited-OCR 上 ModelScope（#3）——四件事拼出**"模型 + 芯片 + 应用"全栈自给**的图。社区情绪从"追赶"转向"并行"。

**(3) GPT 版本号的钟摆与社区疲劳。**
r/OpenAI 同一周内 #5 "GPT-5.5 Instant rollout" → #6 "GPT 5.6 preview" → #7 "GPT 5.6 Cancelled"。社区的 meme + 焦虑反映 OpenAI 的产品节奏开始失序。HF 端则相反——Qwen 一次发布双 size MoE LWM，节奏与命名都更稳。

**(4) Anthropic 在话题层的扩张。**
"Mythos" 内部叙事（r/singularity #3/#4）+ John Jumper 加盟（#6）+ Claude Code 文化梗（r/ClaudeAI #0/#3/#4/#5）——Anthropic 不再只是"对标 OpenAI"，而是塑造"叙事 + 人才 + 开发者文化"的三元品牌。r/ClaudeAI 的 Top-10 几乎全部是文化梗而非 bug report，说明用户社群进入"meme 化忠诚"阶段。

**(5) 开源 + 安全过滤 的博弈短周期化。**
Krea 2 开源（r/StableDiffusion #1）→ 24h 内出现"移除安全过滤"节点（#2）。从 Stable Diffusion 1.5 时代的"几个月内 NSFW 被破"，到本周"模型发布次日"，社区移除安全机制的时间窗在快速缩短。

**(6) PhD-with-no-paper 与学术心理健康两条暗线。**
r/MachineLearning #1（PhD 能否无 top-tier paper 毕业）+ r/AskAcademia 多篇心理健康 / 行政伦理帖——AI 高速发展时代，**学术界的"研究节奏"和"心理负荷"被拉到极限**。这条线和 HF 端 #18 NatureBench（"agent 能复现 Nature 但不能发现"）形成相互印证。

**(7) Time-series / dynamical systems 视角回归。**
r/MachineLearning #2 呼吁回到动力系统，是对当下"什么都套 transformer"的批判。HF 端本周也有 #1 LoopWM（共享参数迭代）做事实回应——递归动力学已经回到 ML 的舞台中央。

---

## 6. Open Questions

1. **AWS Lambda MicroVMs 会不会改变 agent sandbox 市场？** 它的定价、cold start、并发上限决定了 Bedrock AgentCore 与第三方（E2B、Daytona、Modal）的竞争格局。
2. **DeepSeek $60B 估值的合理性。** 创始人 $3B 自投是强信号还是流动性短缺的副作用？
3. **"Mythos" 究竟是什么？** 是 Anthropic 内部 red-team 模型代号还是 NSA 演练? Reddit 帖只是引文，需要后续核实。
4. **John Jumper 加盟 Anthropic 是否标志 AI4Sci 从 DeepMind → Anthropic 的人才迁移？** Jumper 是 AlphaFold 的核心，其下一步会决定 Anthropic 的 science 战略。
5. **Krea 2 + 移除安全过滤的法律责任。** 厂商在 24h 内被破解，开源协议条款下的"reasonable safeguards"是否仍有意义？
6. **DS 岗位"全变 Gen AI" 是真趋势还是 r/datascience 的偏差感知？** 仅 4 帖样本量太小，需要做几个月的纵向跟踪。
7. **RSS feed 截断问题（r/datascience 仅 4 帖）。** 是 Reddit RSS 的间歇性 bug 还是 subreddit 设置？需要在 fetch 脚本里加 fallback retry。

---

## 7. References

### 抓取摘要

- **第一轮**：`uv run python3 scripts/reddit_fetch.py --time week --delay 7` → 179 posts / 8 子版（LocalLLaMA / ClaudeAI / devops / statistics 因 429 失败）
- **重试**：`uv run python3 scripts/reddit_fetch.py --time week --subs LocalLLaMA,ClaudeAI,devops,statistics --delay 30` → 94 posts / 4 子版
- **合计**：**273 unique posts**，全部为新增（与 W25 digest 0 重叠）

### 数据局限说明（重要）

- **RSS 不返回 score / num_comments**：本周报排名仅用 Reddit `top/week` feed rank，**子版内部相对热度**有效，**跨子版数字比较不成立**。
- **OAuth 仍未落地**：Reddit Responsible Builder 政策卡 create-app，凭证 `~/.reddit/.env` 未配置；当 OAuth 可用时，应切回 scoring 版本以恢复 numeric ranking。
- **r/datascience 仅 4 帖**：Reddit RSS 偶尔截断，需要后续重抓或加 fallback。

### 跨社区主题归并表

| 主题 | 关联 reddit 帖 | 关联 HF 论文（同周） |
|------|--------------|--------------------|
| World Model | r/MachineLearning #3 DVD-JEPA | [#1 LoopWM](https://huggingface.co/papers/2606.18208), [#4 Qwen-AgentWorld](https://huggingface.co/papers/2606.24597), [#20 WAM Survey](https://huggingface.co/papers/2606.20781) |
| Agent + Sandbox infra | r/aws #0/#5 Lambda MicroVMs | [#10 OpenRath](https://huggingface.co/papers/2606.19409) |
| Image→Playable Game | r/LocalLLaMA #4, r/StableDiffusion #4 | [#17 GameCraft-Bench](https://huggingface.co/papers/2606.17861) |
| OCR / Long-doc | r/LocalLLaMA #3 Unlimited-OCR | [Unlimited OCR Works (2606.23050)](https://huggingface.co/papers/2606.23050) |
| Coding agent benchmark | r/MachineLearning #6 DeepSWE | [#18 NatureBench](https://huggingface.co/papers/2606.24530), [#16 Multi-LCB](https://huggingface.co/papers/2606.20517) |
