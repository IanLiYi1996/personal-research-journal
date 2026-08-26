# Reddit 热门 W35 — 截至 2026-08-26 03:0x UTC

- **抓取**：12/12 子版，**277 帖唯一**（主抓 `--delay 10` 得 8/12；⭐ **失败的 singularity / StableDiffusion / programming / AskAcademia 逐个补抓 4/4，连续第八次成功**）
- ⚠️ **r/datascience 仅 8 帖**（连续第 10 份严重截断）／r/statistics 19 帖
- **去重三个口径**：**A（新进榜）= 203** ／ **A′（新发布）= 181** ／ B（对照最近 5 份的 172 条已引用）= 243
- **间隔**：距上一份（W34d，08-21 05:23）约 **117.7 小时（4.9 天）**

> ⚠️⚠️ **本份是 4 天空缺补跑（08-22~08-25 Reddit 一次都没跑）**。🚨 **而 AWS 08-21~08-25 五天全在 ⟹「AWS 活、其余死」第六次，且发生在 08-24 那次全新重建之后。**
>
> ✅ **而 Reddit 这次确实兜住了**：本份发布日期分布是 **08-19 → 08-25 整七天**（40/51/40/28/33/45/40 帖），⭐ **正是 top-of-week 7 天跨度的字面体现** —— 我 08-18 立的那条判据（「7 天是硬边界，1 天只是舒适值」）在这次 4.9 天空缺上得到验证。⚠️ 但**空缺期内进过榜又掉出 top-25 的帖子仍然测不到**。

---

## ⭐⭐⭐ 方法学：第三个 A/A′ 数据点，而它让我发现 W34d 那条「榜单容量饱和」的结论是**错的**

| 间隔 | A（新进榜）| A′（新发布）| A/A′ | ⭐ **A′ 占榜单比例** |
|---:|---:|---:|---:|---:|
| **2.0 小时** | 4 | 1 | **4.00** | — |
| **68.3 小时（2.8 天）** | 152 | 123 | **1.24** | 123/322 = **38.2%** |
| **91.5 小时（3.8 天）** | — | 128 | — | 128/276 = **46.4%** |
| **117.7 小时（4.9 天）** | **203** | **181** | **1.12** | 181/277 = **65.3%** |

**① A/A′ 比值继续单调收缩（4.00 → 1.24 → 1.12）** ⟹ 与我给的机制一致（短窗口的榜单变化几乎全来自旧帖爬升，长窗口才主要来自真新帖），⭐ 且看起来是渐近趋于 1。

**② ⚠️⚠️ 而我要更正 W34d 里的一句话。** 我当时写：

> 「68.3h 的 123 与 91.5h 的 128 只差 5 帖 —— 榜单容量饱和（12×25=300 个位置）现在有了很干净的证据：间隔从 68 小时拉到 91 小时，几乎拿不到更多东西。」

⚠️⚠️ **这个结论站不住，因为那两个数来自大小不同的池**：68.3h 那次的榜单有 **322** 帖（并发运行让部分子版超过 25 条），91.5h 那次只有 **276** 帖。⟹ **比原始条数是错的口径。**

⭐ **换成「新发布占榜单的比例」之后形状完全不同**：2.8 天 → 38.2%，3.8 天 → 46.4%，4.9 天 → **65.3%** ⟹ ⭐⭐⭐ **它大致随间隔线性增长，与「榜单存约 7 天、大致均匀换血」自洽，而不支持「三天就饱和」。**

⟹ ⭐⭐⭐ **而根因值得单记：我拿两个分母不同的绝对数做了比较。** 这与我这两周反复收集的「成本口径不一致」是同一族错误（BDH-CQ 的自算硬件 vs API 报价 / SKILLER 的价目表比值 / 缓存命中 50× / Anthropic 的 gross vs net）—— ⭐ **只是这一次分母不同的两个数是我自己产生的，所以更该注意。**

⭐ **实用结论仍不变**（每天一跑；7 天是硬边界），**但理由换了**：不是「三天后就拿不到更多」，而是「7 天之后开始永久丢」。

---

## ⭐⭐⭐ 主线一：ARC-AGI-3 在一个 digest 周期里从 13% 冲到 100%，而这本身是信号

[**NVIDIA's coding agent scored 100% on ARC-AGI-3 interactive reasoning benchmark**](https://www.reddit.com/r/singularity/comments/1vuhlhn/nvidias_coding_agent_scored_100_on_arcagi3/)（r/singularity r7）

⟹ ⭐⭐⭐ **把我这几天在三份 digest 里记到的同一基准的三个数字并排**：

| 来源 | 数字 |
|---|---|
| OpenAI（经 Latent Space 转述，我今天 tech-blogs 记的）| 只加 retained reasoning 与 compaction，GPT-5.6 Sol **13.3% → 38.3%** |
| **Prime Agent 论文**（我今天 HF digest 深读的）| RHAE Best@1 **30% → 95.5%** |
| ⭐ **本条** | NVIDIA 的 coding agent **100%** |

⟹ ⭐⭐ **一个基准在几天内出现 13% / 38% / 95.5% / 100% 四个数字，而它们的指标口径互不相同**（RHAE Best@1 vs 未说明的「scored」）⟹ ⭐⭐⭐ **这正是「报分数不写口径等于没报」的一个现成案例，而且是我这两周的判据第一次能用来解读一条社区新闻**：在不知道是哪个子集、哪个指标、几次运行取最好的情况下，「100%」与「38.3%」不可比。⚠️ 我只有标题，不引其主张，**只记「这个基准正在被快速饱和」这个形状**。

---

## ⭐⭐⭐ 主线二：Sol 的降价被社区独立确认，而它不只是 Bedrock 的促销

[**New Sol API pricing - $4 per million input tokens and $20 per million output tokens**](https://www.reddit.com/r/OpenAI/comments/1vx5mrz/new_sol_api_pricing_4_per_million_input_tokens/)（r/OpenAI r11）

⟹ ⭐⭐⭐ **数字与我 08-22 从 AWS 日报记的完全一致**（Bedrock 下调 GPT-5.6 Sol 至 **$4/M 输入、$20/M 输出**，输入 −20%、输出 −33.3%）—— ⭐⭐ **但我当时记的是「Bedrock 促销价、至少持续到 2026-11-21」，而这条社区帖把它写成「New Sol **API** pricing」** ⟹ **说明这是 OpenAI 自家 API 的定价变化，Bedrock 只是跟随** —— 这是一处需要更正的口径。

⭐⭐ **而同一周两家的速率限制方向相反**：
- [**5hr Limit is back for Plus users. $100 and $200 get a few more months.**](https://www.reddit.com/r/OpenAI/comments/1vxnqaq/5hr_limit_is_back_for_plus_users_100_and_200_get/)（r/OpenAI r2）—— OpenAI **收紧**
- [**Having unlimited tokens is wild**](https://www.reddit.com/r/ClaudeAI/comments/1vuuiot/having_unlimited_tokens_is_wild/)（r/ClaudeAI r5）—— Anthropic 侧**放开**

⟹ ⭐⭐ **单价在降、而消费端配额在两家之间朝相反方向走** ⟹ 这与我 W33 那处更正是同一逻辑：**「单位成本」与「你实际能用多少」是两件事**，而后者是商业决策。

---

## ⭐⭐⭐ 主线三：OpenAI 自研芯片拿到第三个来源

[**OpenAI Just Dropped Benchmarks for Their Own Chip, Jalapeño, and It's Beating Nvidia's**](https://www.reddit.com/r/OpenAI/comments/1vyd7lv/openai_just_dropped_benchmarks_for_their_own_chip/)（r/OpenAI r14）

⟹ 与我今天 tech-blogs 深读的两个来源（OpenAI 官方两篇 + 量子位转引 SemiAnalysis 实测）合起来是**三个来源**。⭐ 而「成本战线上移到硅片」这条线现在的证据：AMD 收 Taalas → 4.8 亿砸端侧 agent 芯片 → Anthropic 商谈 60 亿收 Decart → **OpenAI 出货推理芯片并有第三方实测** ＋ 同周 **Meta MTIA 300（首个训练芯片）**。

### ⭐⭐ 而本地推理硬件同周也很热，且它扣在内存涨价那条线上

[**Xiaomi AI Cube announced with 1.2TB/s memory bandwidth**](https://www.reddit.com/r/LocalLLaMA/comments/1vwvghi/xiaomi_ai_cube_announced_with_12tbs_memory/)（r/LocalLLaMA r1）· [**Apple introduces new Mac Studio with M5 Max and M5 Ultra - up to 512GB of unified memory**](https://www.reddit.com/r/LocalLLaMA/comments/1vxzg6v/apple_introduces_new_mac_studio_with_m5_max_and/)（r3）· ⭐ [**Apple M5 Server**](https://www.reddit.com/r/LocalLLaMA/comments/1vx6ivx/apple_m5_server/)（r4）· [**Apple releases M5 ultra at 1.2TB/s bandwith**](https://www.reddit.com/r/LocalLLaMA/comments/1vxzgyt/apple_releases_m5_ultra_at_12tbs_bandwith/)（r10）

⟹ ⭐⭐ **在内存 12 个月涨 500% 的背景下，「大容量统一内存 + 高带宽」这条路的相对经济性变好** —— ⭐ 而「**Apple M5 Server**」这条最值得记：**Apple 进服务器**是一个此前不在我记录里的方向。⚠️ 仅标题、未核实规格。

---

## ⭐⭐⭐ 主线四：Qwen3.8-27B 那个「identical vs game changer」的分裂，本份倾向后者

我从 W34 开始追这个分裂并把「identical 指什么」列为最高优先待核实，已挂三份。本份：

- [**Qwen 3.8 27B is a game changer.**](https://www.reddit.com/r/LocalLLaMA/comments/1vvyacg/qwen_38_27b_is_a_game_changer/)（r/LocalLLaMA r6）
- ⭐ [**I gave Qwen 3.8 27B a reverse-engineering job I assumed needed a frontier model, and i…**](https://www.reddit.com/r/singularity/comments/1vwaetf/i_gave_qwen_38_27b_a_reverseengineering_job_i/)（r/singularity r14）—— ⭐⭐ **这条比泛泛的好评有价值：它是一个具体任务的报告**（本以为需要前沿模型的逆向工程活）

⟹ ⭐⭐ **本份没有新的「identical」主张，而「game changer」侧多了两条（其中一条带具体任务）** ⟹ 天平偏向后者。⚠️ **但仍全是主观体感，我不宣布这个问题已解决** —— 真正能解决它的是权重层面的比对或第三方基准，而那两样我都还没有。

⭐ 而发布节奏继续：[**Qwen3.8-Flash-Next tomorrow**](https://www.reddit.com/r/LocalLLaMA/comments/1vxwtyd/qwen38flashnext_tomorrow/)（r5）+ [**Qwen 3.8 Flash Next day 0 support from unsloth**](https://www.reddit.com/r/LocalLLaMA/comments/1vxybmy/qwen_38_flash_next_day_0_support_from_unsloth/)（r11）⟹ ⚠️ **按 W33f 那条教训，「明天发」与「day 0 支持」都不构成状态更新，只有实际权重算。**

---

## ⭐⭐⭐ 主线五：一个我此前没有的运维失效形态 —— **prompt 回滚**

[**The prompt rollback worked in staging and nowhere else**](https://www.reddit.com/r/devops/comments/1vxf82h/the_prompt_rollback_worked_in_staging_and_nowhere/)（r/devops r8）

⟹ ⭐⭐⭐ **「回滚一个 prompt」与「回滚代码」不是一回事，而这个区别我此前没有记录过。** 它精确落在两处我已追踪的地方：
- **Co-Evolution 综述**的治理建议「**rollback to verified states**」
- **Evo-Bench** 的 Appendix D：三个模型里两个最终冻结的版本比自己达到过的最好版本更差 ⟹ **瓶颈不在产生改进而在识别并留住改进**

⭐⭐ **而这条帖给出的是第三种失效：回滚动作本身在不同环境里不等价。** ⟹ 若 prompt 是有状态系统的一部分（记忆、技能、上下文都可能已经被它改动过），那么「把 prompt 换回旧版本」并不能把系统换回旧状态。⚠️ 仅标题，未读正文。

⭐ 同子版另两条接主线：[**How do real companies securely give developers access to the development database?**](https://www.reddit.com/r/devops/comments/1vv3h37/how_do_real_companies_securely_give_developers/)（r2，⭐ 权限粒度线，与 Cloudflare 的按任务 OAuth consent 同一层）· [**Any devops or SRE engineers using AI agents?**](https://www.reddit.com/r/devops/comments/1vw1t7p/any_devops_or_sre_engineers_using_ai_agents/)（r3，采用率）

---

## ⭐⭐ 主线六：AWS 账号线第 17 个数据点，且理由是一种新形态

[**AWS Account Suspended ("Related to previously closed accounts") – Business Critical**](https://www.reddit.com/r/aws/comments/1vxvk29/aws_account_suspended_related_to_previously/)（r/aws r5）

⟹ ⭐⭐ **停号理由是「与此前已关闭的账号相关」** —— 此前 16 个数据点的形态是「我付了钱系统说没付」「自动准入挡住我」「计量产生了我没激活的费用」，⭐ **本条是「因关联而连坐」**，即判定依据不是本账号的行为而是它与其他账号的关系。⚠️ 用户单方陈述。

⭐ 同子版另三条：[**Found over a grand a month of orphaned AWS resources**](https://www.reddit.com/r/aws/comments/1vwxn2y/found_over_a_grand_a_month_of_orphaned_aws/)（r7，成本卫生）· ⭐⭐ [**Unable to launch g6e.12xlarge in any EU region and az, InsufficientInstanceCapacity**](https://www.reddit.com/r/aws/comments/1vvdj7i/unable_to_launch_g6e12xlarge_in_any_eu_region_and/)（r10）⟹ ⭐ **GPU 容量约束的从业者侧证据**，与「产能是瓶颈」「内存涨 500%」同一条 · [**What is the AWS equivalent for AI agents?**](https://www.reddit.com/r/aws/comments/1vwcemv/what_is_the_aws_equivalent_for_ai_agents/)（r8）⟹ ⭐ **AgentCore 存在快一年了而从业者还在问这个** ＝ 定位传达的问题 · [**Amazon EC2 beta (20 years ago today)**](https://www.reddit.com/r/aws/comments/1vxoknc/amazon_ec2_beta_20_years_ago_today/)（**r0**）

⭐ **而由 rank≤5 扫描捞到、A′ 漏掉的一条正好与我自己的 AWS 日报对上**：[**CloudFront can go to your closest S3 bucket now**](https://www.reddit.com/r/aws/comments/1vu02z7/cloudfront_can_go_to_your_closest_s3_bucket_now/)（r/aws r3）⟹ 正是我 08-21 记的「CloudFront 支持对 S3 Multi-Region Access Points 用 OAC」的社区侧。⭐ **本次该规则捞到 11 条、约 1 条实质（≈9%），与上次 3/41≈7% 一致** —— 低但非零，正是保留一个廉价检查的理由。

---

## ⭐⭐ 主线七：来源可检测性 —— 需求侧出现一个有名机构的硬数字

[**A Third of the Post-ChatGPT Web Is AI-Written, Pew Finds.**](https://www.reddit.com/r/OpenAI/comments/1vw7x0k/a_third_of_the_postchatgpt_web_is_aiwritten_pew/)（r/OpenAI r10）

⟹ ⭐⭐⭐ **这是这条线上第一个来自有名研究机构（Pew）的量化主张**，而它与我已有的两侧扣得很紧：**供给侧**（Stack Overflow 提问量 −99%、亚马逊在实体书里找语料）与**检测侧**（水印在代码/事实上最弱、MIT 测出生成图常常追不回训练数据、视频检测器三类家族都不能一致泛化）。

⚠️⚠️ **但「三分之一」这个数最需要追问方法**：怎么判定一个页面是 AI 写的？若用的是无密钥的风格判别器（AI 检测软件那一类），那么它的假阳性率就直接决定这个数字 —— **而我记过 Anthropic 自己说「没有水印不意味着不是 AI 生成」。** ⟹ 仅记标题，**不引这个数字**。

⭐ 相邻：[**Implementing Watermarking for Language Models [P]**](https://www.reddit.com/r/MachineLearning/comments/1vw18ys/implementing_watermarking_for_language_models_p/)（r/MachineLearning r12）⟹ 与 W34 记的 Raschka「从零实现」同一动作出现在社区侧。

---

## ⭐⭐ 主线八：会议流程连续第九份，而这次提问者坐在审稿人的位置上

- ⭐⭐ [**Reviewing 4 papers for AAAI 2027 and none have code, Reject? [D]**](https://www.reddit.com/r/MachineLearning/comments/1vxryws/reviewing_4_papers_for_aaai_2027_and_none_have/)（r4）⟹ ⭐⭐⭐ **「无可复现代码就 desk reject」这条诉求的位置变了**：W32b 是社区提案、W33d 是对 AAAI 的质询（「AAAI 2027 Review: No code submission?」），**本条是一个审稿人真的面对这个决定**。⭐ 而 HF 复现 2,200 篇那次给过量化依据（**约 12.6% 的论文因 artifact 缺失而两边都无法确立**）。
- ⭐ [**AAAI 2027 Reviewer Bidding and Assignment Integrity [D]**](https://www.reddit.com/r/MachineLearning/comments/1vwujcy/aaai_2027_reviewer_bidding_and_assignment/)（r6）⟹ 关注点继续上移：从「评审质量」到**分配机制本身的完整性**
- ⭐ [**Nature retracts climate change paper almost two years after publication**](https://www.reddit.com/r/AskAcademia/comments/1vy6h92/nature_retracts_climate_change_paper_almost_two/)（r/AskAcademia r2）+ [**Was Open Access publishing the wrong direction?**](https://www.reddit.com/r/AskAcademia/comments/1vwxonx/was_open_access_publishing_the_wrong_direction/)（r6）+ [**What do journals do with our money?**](https://www.reddit.com/r/AskAcademia/comments/1vx3fam/what_do_journals_do_with_our_money/)（r1）+ [**Fighting Goliath: Senior ex-supervisor stole my PhD research**](https://www.reddit.com/r/AskAcademia/comments/1vxr3ip/fighting_goliath_senior_exsupervisor_stole_my_phd/)（r5）⟹ 学术诚信/出版经济学集群

---

## 其余

### ⭐⭐ 成本与效率的实测

⭐⭐ [**Does telling an LLM to "be concise" actually save you money? We measured it across 9 m…**](https://www.reddit.com/r/MachineLearning/comments/1vulfei/does_telling_an_llm_to_be_concise_actually_save/)（r3）⟹ ⭐⭐⭐ **这正是我那条「算力分配在四个层面都是错的」里「单题·配置侧」的一个直接实验**（我记的是 Simon 实测 Qwen3.8-27B 默认 `xhigh` 让「画一个圆」进 22K token 推理）。⭐ **而它测了 9 个模型，比单点观察强**。⚠️ 仅标题、不知结论方向。

⭐ [**I developed my own quantized LLM from scratch, trained on 30B tokens, deploys in 60 MB**](https://www.reddit.com/r/MachineLearning/comments/1vv2nkh/i_developed_my_own_quantized_llm_from_scratch/)（**r0**）· ⭐ [**Continual Learning of Frontier Models for SovereignAI. Tech Report + Open Weights Mode…**](https://www.reddit.com/r/MachineLearning/comments/1vxvzju/continual_learning_of_frontier_models_for/)（r9）⟹ **主权 AI + 持续学习**，与 tech-blogs 侧的 Mistral×HUMAIN、r/devops 的「主权云成为架构问题」同一条 · ⭐ [**I irradiated LLMs and found that they die really quickly**](https://www.reddit.com/r/LocalLLaMA/comments/1vx2fhz/i_irradiated_llms_and_found_that_they_die_really/)（r12）⟹ 一个我没见过的鲁棒性实验（辐照权重）

### ⭐⭐ 模型与产品

⭐⭐ [**A stealth model called Ox-Alpha has been released, outperforming Fable on SWE.**](https://www.reddit.com/r/singularity/comments/1vu87p5/a_stealth_model_called_oxalpha_has_been_released/)（r13）⟹ 隐身模型线；⚠️ 仅标题、来源不明 · [**Sam Altman with some sad statements about AI**](https://www.reddit.com/r/singularity/comments/1vwju3e/sam_altman_with_some_sad_statements_about_ai/)（r4）· [**GPT 5.6 Got Massively Upgraded Without an Announcement**](https://www.reddit.com/r/OpenAI/comments/1vvie4v/gpt_56_got_massively_upgraded_without_an/)（r8）⟹ ⚠️ **「无公告静默升级」这类主张几乎无法核实**，只记它反复出现（W34 也有一条「Did GPT 5.6 Sol get secretly upgraded?」）· [**Disrupting a new covert influence campaign from Russia**](https://www.reddit.com/r/OpenAI/comments/1vxw9vt/disrupting_a_new_covert_influence_campaign_from/)（r13）⟹ **与我今天 tech-blogs 记的官方同一篇**，两侧同日

### ⭐ 长跑 agent 与「自己造东西」

⭐ [**Two weeks ago I gave Claude a domain and told it to build whatever it wanted. I finall…**](https://www.reddit.com/r/ClaudeAI/comments/1vuhvta/two_weeks_ago_i_gave_claude_a_domain_and_told_it/)（r8）⟹ ⭐ **两周的自主运行**，接我追的「agent 从一次调用变成长期进程」（Ouroboros 161 天 / AgentCore 单会话 14 天 / StateM 的 22 小时开发运行）· [**Indeed laid off my pregnant wife, so I built a job search competitor with Claude**](https://www.reddit.com/r/ClaudeAI/comments/1vx4kn7/indeed_laid_off_my_pregnant_wife_so_i_built_a_job/)（r9）· [**Week 4 of making my fishing game entirely with AI**](https://www.reddit.com/r/ClaudeAI/comments/1vxdnw6/week_4_of_making_my_fishing_game_entirely_with_ai/)（r11，⭐ 连续第三份出现同一系列）· [**decayfmt - A file format which corrupts a little every time you open it**](https://www.reddit.com/r/ClaudeAI/comments/1vuh7rl/decayfmt_a_file_format_which_corrupts_a_little/)（r7）

### ⭐ 统计社区又在教我这边缺的东西（连续第四份）

⭐⭐ [**[Software] How to test if your numerical code is mathematically correct?**](https://www.reddit.com/r/statistics/comments/1vxavkm/software_how_to_test_if_your_numerical_code_is/)（r1）⟹ ⭐⭐⭐ **这就是「怎么验证一个输出是对的」这个问题在数值计算社区的版本**，而我这两个月在 agent 侧收集的答案是「证据面 / 闸门 / 复验 / 外部不可被优化的参照」—— **同一个问题，两个社区，而统计侧把它当作一个有标准答案的工程问题在讨论。** · ⭐ [**[Question] Low Inter-Rater Reliability (ICC): Better to use single observer data or av…**](https://www.reddit.com/r/statistics/comments/1vul2cv/question_low_interrater_reliability_icc_better_to/)（r9）⟹ ⭐⭐ **这正是「LLM 裁判之间不一致时怎么办」的统计学版本**（我记过 Mechanist 那篇报三对裁判的配对分数与 Spearman ρ、OSReward 测出裁判间差异巨大）· [**[Discussion] Does NO causation necessarily mean confounding relationship?**](https://www.reddit.com/r/statistics/comments/1vvx2kf/discussion_does_no_causation_necessarily_mean/)（r10）· [**[Question] Ambiguity in outcome incidence reporting, meta-analysis**](https://www.reddit.com/r/statistics/comments/1vy7032/question_ambiguity_in_outcome_incidence_reporting/)（r12）

### ⭐ 其余

⭐ **具身机器人**：[**9.3 seconds…Humanoid robots now run faster than humans**](https://www.reddit.com/r/singularity/comments/1vvhlfi/93_secondshumanoid_robots_now_run_faster_than/)（**r0**）+ [**100m Hurdles Final**](https://www.reddit.com/r/singularity/comments/1vx7hk4/100m_hurdles_final/)（r1）⟹ ⚠️ **「比人跑得快」这类标题按惯例打折**（赛道、规则、能耗都未知），但与我 W34d 记的「WRC 事故 vs 成果」的中英不对称是同一事件群 · [**An unusual parade was held in Kyiv… ground-based robotic systems, maritime…**](https://www.reddit.com/r/singularity/comments/1vx443u/an_unusual_parade_was_held_in_kyiv_it_featured/)（r9）· ⭐ [**13 students were arrested after occupying OpenAI's new D.C. lobbying office for 2 hour…**](https://www.reddit.com/r/OpenAI/comments/1vv9glb/13_students_were_arrested_after_occupying_openais/)（r6）⟹ **公众情绪转向那条线的第四个数据点，且这次是线下行动**（前三：r/singularity「Young People Hate AI CEOs」· r/OpenAI「Major vibe shift」· fast.ai 那篇第一人称） · ⭐ **MiniMax H3 连续第十份占据 r/StableDiffusion，但内容已完全转向工具化**（[Prompt Composer 大更新](https://www.reddit.com/r/StableDiffusion/comments/1vuty3p/big_update_to_the_free_minimax_h3_prompt_composer/) / [sprites 动画](https://www.reddit.com/r/StableDiffusion/comments/1vw755b/minimax_h3_helps_me_with_sprites_animation/) / [Gaussian Splatting 测试](https://www.reddit.com/r/StableDiffusion/comments/1vxu1ho/gaussian_splatting_test_with_minimax_h3/) / [本地跑](https://www.reddit.com/r/StableDiffusion/comments/1vw7l4n/fixed_my_trauma_with_minimax_h3_local/)）· ⭐ **r/programming 与 tech-blogs 两侧重合两条**（[EVE Online: The Move to Python 3](https://www.reddit.com/r/programming/comments/1vxz0z4/eve_online_the_move_to_python_3_begins/) / [Your executable is a SQLite database](https://www.reddit.com/r/programming/comments/1vx81cr/your_executable_is_a_sqlite_database/)）⟹ ⭐ 而 [**How To Report A Bug So It Actually Gets Fixed**](https://www.reddit.com/r/programming/comments/1vwb745/how_to_report_a_bug_so_it_actually_gets_fixed/)（r8）接「提交成本 < 审核成本」那条线的提交端 · [**The amount of activity on GitHub right now is crazy. Thoughts?**](https://www.reddit.com/r/singularity/comments/1vv79k5/the_amount_of_activity_on_github_right_now_is/)（r12）

---

## Open Questions

1. ⭐⭐⭐ **ARC-AGI-3 那四个数字（13.3% / 38.3% / 95.5% / 100%）各自是什么口径？** 这是本份最值得追的一条，因为**它是一个现成的、可以用来演示「不写口径等于没报」的案例**，而且四个数字都在几天内出现。⭐ 从 Prime Agent 论文我至少知道一个是 RHAE Best@1。
2. ⭐⭐ **Sol 的 $4/$20 到底是 OpenAI API 定价还是 Bedrock 促销？** 我 08-22 记的是后者（带 11-21 的促销截止日），而本份社区帖写的是前者。⟹ **这决定了它是「结构性降价」还是「限时活动」，而这两者对客户材料的含义完全不同。**
3. ⭐⭐ **「prompt 回滚在别处不工作」的机制是什么？** 若 prompt 已经改动过记忆/技能/上下文，则回滚 prompt ≠ 回滚系统状态。⟹ **这会给 Co-Evolution 综述的「rollback to verified states」加一条实践限制**，而那条建议本来就被 Evo-Bench 证明过缺失代价。
4. ⭐⭐ **Pew 那个「三分之一的后-ChatGPT 网页是 AI 写的」用什么方法判定？** 若靠风格判别器，假阳性率就决定这个数；⭐ 而这条线上我已有的判据是「无密钥的风格判别 ≠ 有密钥的统计检验」。
5. ⭐ **A/A′ 比值的三个点能不能拟合出「从发布到进 top-25 的典型延迟」？** 现有 4.00（2h）/ 1.24（68h）/ 1.12（118h）。⭐ 若这个量能估出来，它就直接告诉我「每天一跑会系统性漏掉哪一类帖子」。
