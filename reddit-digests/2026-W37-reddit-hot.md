# Reddit 热门 W37 — 截至 2026-09-08 07:1x UTC

- **抓取**：**12/12 子版一次成功、279 帖唯一**（⭐ 逐个子版抓、目录 `rd14/` 为本次运行专属，**已列 mtime 确认全部是今天 07:01–07:11**）
- ⚠️ r/datascience 仅 **8** 帖（连续第 13 份截断）／r/statistics 21 帖
- **去重三个口径**：**A（新进榜）= 86** ／ **A′（新发布）= 62** ／ B（对照最近 6 份的已引用）= 201
- **间隔**：距上一份（W36f，09-06 10:0x）**45.2 小时**

> ⚠️ **1 天空缺补跑（09-07 周一没跑）**，而 🚨 **AWS 09-07 那份在 ⟹「AWS 活、其余死」第十次。**
>
> ✅ 发布日分布 **09-01 → 09-08**（37/44/51/37/42/30/33/5）⟹ **top-of-week 的 7 天跨度第六次被字面验证。**

---

## ⭐⭐ 方法学：45.2h 这个点填上了我序列里一段空缺，而它说明那个「平台」前面是一段斜坡

| 间隔 | A | A′ | A/A′ | A′ 占榜单 | 反解驻留 |
|---:|---:|---:|---:|---:|---:|
| 2.0h | 4 | 1 | 4.00 | — | —（失效区）|
| 6.3h | 12 | 1 | 12.00 | 0.4% | —（失效区）|
| 23.4h | 51 | 31 | **1.65** | 11.2% | 8.7 天 |
| ⭐ **45.2h** | **86** | **62** | ⭐ **1.39** | **22.2%** | **8.5 天** |
| 66.3h | 136 | 123 | 1.11 | 45.1% | 6.1 天 |
| 68.3h | — | — | 1.24 | 38.2% | 7.4 天 |
| 79.1h | 149 | 130 | 1.15 | 46.1% | 7.2 天 |
| 95.6h | 161 | 140 | 1.15 | 51.1% | 7.8 天 |
| 117.7h | 203 | 181 | 1.12 | 65.3% | 7.5 天 |

⟹ ⭐⭐ **我此前在 23.4h（1.65）与 66.3h（1.11）之间没有任何点，所以只能把序列描述成「不稳定短端 + 1.1–1.2 平台」两段。⭐ 本次的 1.39 恰好落在两者之间** ⟹ **修正后的形状是三段：近零分母的失效区（< 约 20h）→ 一段斜坡（约 20–60h）→ 平台（≥ 约 60h，1.1–1.2）。**

⭐ 而 **反解驻留 8.5 天**落在我修正后的「约 6–9 天」区间的上端，⭐ 与 23.4h 那次（8.7 天）接近 ⟹ **短间隔倾向给出偏高的驻留估计**，与我 09-03 记的机制（`间隔 ÷ 占比` 假设换血均匀，而短窗口更容易被日间波动扰动）一致。

### ⚠️ rank ≤5 未引用扫描：7 条、**0 条实质**（连续第三次接近空）

全部是我已覆盖过的（`AWS Lambda SnapStart` / `AGI achieved` / `Built a small 3-node Kubernetes homelab`）或 meme。⟹ ⭐ **我 09-06 提的那个修正又得一个点：这条规则的产出率更可能取决于「上一份覆盖得多宽」而不是间隔** —— 我最近三份都覆盖得较宽，于是长期驻留帖里已经没有未覆盖的了。

---

## 🚨🚨🚨 主线一：Notion 官方 MCP connector 在任务中途注入 prompt，让 agent 去打广告

⭐⭐⭐ [**Notion's Official MCP connector prompt injects AI agents to advertise products mid-task**](https://www.reddit.com/r/ClaudeAI/comments/1w9dluw/notions_official_mcp_connector_prompt_injects_ai/)（r/ClaudeAI r5，30.3h）

⟹ 🚨⭐⭐⭐ **这是我记过的所有 prompt injection 里形态最特殊的一个：注入方是官方的、第一方的集成，不是攻击者。**

⭐⭐ **而它落在三条线的交点上**：

| 我此前的线 | 本条如何接上 |
|---|---|
| **MCP 进入网络安全产品目录** | 我 08-18 记过 Cloudflare 与 AWS WAF **同一周**把 MCP 当作需在流量层识别与保护的协议 ⟹ ⭐ **而那两家防的是「恶意 MCP 服务器」，本条说的是「官方 MCP 服务器自己在注入」** —— **流量层识别抓不到它，因为它就是正常流量** |
| ⭐⭐⭐ **「模型当作自己思考的通道就是特权注入通道」** | 我 W33d 从 Simon Willison 记过这条（「Models appear to treat their own reasoning traces as sacrosanct」）⟹ ⭐⭐ **本条把它推广了一格：模型当作可信基础设施的通道（工具描述 / connector 元数据）同样是特权注入通道** |
| **「谁授权了这个动作」** | ⟹ ⭐⭐⭐ **一个 agent 在任务中途插入广告，而用户既没有授权它这样做、也不会从 agent 的自述里看出来** —— **这正是那条主线要的那个东西：动作发生了，而授权链条里没有任何一环记录它** |

⟹ ⭐⭐ **对客户材料的直接含义：「审查你使用的 MCP 服务器」这条建议此前的理由是「防第三方恶意服务器」，而现在有了一个更难反驳的理由——官方服务器也会往你的 agent 上下文里塞它自己的商业目标。**

⚠️⚠️ **仅一个社区标题**：我没有读到具体的注入文本、没有 Notion 的回应、也不知道这是刻意设计还是某个模板的副作用 ⟹ **列为最高优先待核实**，而它值得核实的理由是**若成立，它是一个可复现的、可被任何人验证的例子**（去读那个 connector 的工具描述即可）。

---

## 🚨⭐⭐ 主线二：可复现性危机在 r/MachineLearning 被直接问成「是不是已经太晚了」

⭐⭐ [**Reproducibility seems to be headed towards irrelevance in ML research. Is it too late? [D]**](https://www.reddit.com/r/MachineLearning/comments/1w92eis/reproducibility_seems_to_be_headed_towards/)（r/MachineLearning r6，37.7h）

⟹ ⭐⭐⭐ **而我手里恰好有这条线上最硬的那组数字**（08-14 从 HF 与 alphaXiv 联合组织的那次复现挑战深读）：**2,226 篇 ＝ ICML 全会 34% · 51%（1,103 篇）至少一条主张被独立验证 · 23%（496 篇）至少一条被推翻或争议（含 49 篇全部被推翻）· ⭐⭐ 以及 242 篇「独立复现团队对同一批主张得出相反裁定」**，原文那句是 **「Reproducibility is not binary; it is adversarial」**。

⟹ ⭐⭐ **含义：社区在问「是不是太晚了」，而那次复现挑战给的答案是一个中间状态——不是「大部分不可复现」，而是「复现本身方差很大」**（242 篇相反裁定）。⭐ **这个区分对回答那个帖子很重要：问题不是论文都错了，而是「验证」这个动作本身还没有可靠的方法论。**

⭐ 而同窗有一条形态相反的正面动作：[**Rustuna: A High-Performance Rust Implementation of Optuna [P]**](https://www.reddit.com/r/MachineLearning/comments/1w9nyhz/rustuna_a_highperformance_rust_implementation_of/)（r/MachineLearning r8，21.2h）⟹ ⭐ **「重写成 Rust」这条线又一个实例**（此前：mold linker 的人写版本、EvoX Genesis 的 25 万行 Rust C 编译器、MESA 的 Fortran→Rust）。

---

## 🚨⭐⭐ 主线三：AI 数学又多一项主张（Jacobian 猜想），而这条线现在有五项且验证强度差别极大

⭐⭐ [**Hate to admit it, but the last month or so, particularly Jacobian conjecture breakthrough => Hug…**](https://www.reddit.com/r/singularity/comments/1w96hyl/hate_to_admit_it_but_the_last_month_or_so/)（r/singularity r5，35.2h，⚠️ **标题被 RSS 截断**）

⟹ ⭐⭐ **这条线现在有五项，而按「产物的可验证性」排开差别很大**：

| 主张 | 验证强度 |
|---|---|
| ⭐⭐⭐ **费马大定理的完整形式化**（09-04，Anthropic Research + 量子位两源）| **最强：产物本身机器可检查**，证明助手会拒绝一个错的证明 |
| ⭐⭐ **黎曼 ζ 零点占比下界 41.6% → 67.2%**（08-11）| **强**：两位领域专家审阅（⚠️ 其中一位是被依赖前序工作的作者）+ Lean 形式化过 comparator + ⭐ 主动给失败分母 + 主动降温 |
| Hadamard 矩阵（08-14，量子位）| ⚠️ 单中文源、未读 |
| 🚨 **孪生素数「新突破」**（09-05）| ⚠️⚠️ **陶哲轩公开吐槽** ＝ 专家审阅这一项明确不过 |
| ⭐ **Jacobian 猜想**（本条）| ⚠️⚠️ **只有一个被截断的社区标题，我不知道是谁做的、什么模型、有没有形式化** |

⟹ ⭐⭐⭐ **含义（也是我给这条线的读法）：这五项的差别不在「模型有多强」而在「产物能不能被机器或专家独立检查」** —— ⭐ **而费马那一项与孪生素数那一项在同一周里得到完全不同的接待，正是这个区分的干净演示。**

⚠️ **本条我只记它出现，不引任何具体主张。**

---

## ⭐⭐ 主线四：r/devops 连续五周的「谁授权」线本窗口断了，换成了职业焦虑

| 帖 | |
|---|---|
| [Getting Rejections because I don't have experience working with Kubernetes in production.](https://www.reddit.com/r/devops/comments/1w9n60m/getting_rejections_because_i_dont_have_experience/) | r/devops r2，21.9h |
| [Is platform engineering a good field?](https://www.reddit.com/r/devops/comments/1w95u75/is_platform_engineering_a_good_field/) | r/devops r5，35.6h |
| ⭐ [How many of you are just working with YAML files all day long over actual coding?](https://www.reddit.com/r/devops/comments/1w9yn0l/how_many_of_you_are_just_working_with_yaml_files/) | r/devops r8，13.7h |

⟹ ⭐⭐ **我在这个子版连续五周追到了一条完整的技术线**（自动化该在哪停 → 冲突的基础设施状态 → gate 里该有什么 → workload identity 之后 secrets manager 还剩什么 → CI/CD 凭据 → **GitHub OIDC 的具体做法**）⟹ **本窗口它一条都没有，取而代之的是三条职业/角色问题。**

⟹ ⭐ **而我不把这读成「那条线结束了」** —— 45 小时的窗口本来就只够放下十几条高位帖，⭐ **但这确实是那条线第一次在一个窗口里完全缺席，值得记下来作为它是否持续的观察点。**

⭐ 而 r/statistics 那两条职业帖（[[Q] How to become Great at statistics](https://www.reddit.com/r/statistics/comments/1w9orik/q_how_to_become_great_at_statistics/) r0 · [[Career] New stats bachelors feeling kind of stuck and in need of advice](https://www.reddit.com/r/statistics/comments/1w9m7f9/career_new_stats_bachelors_feeling_kind_of_stuck/) r8）⟹ **是那条「统计专业职业焦虑」的连续第三份**（09-03 的「不感兴趣是否还能读」→ 09-06 的「我的学位是不是没用了」在 r0 → 本次两条），⚠️ **但本次两条的措辞比 09-06 那条温和**（「怎么变强」而不是「是不是没用了」）⟹ ⭐ **所以我不说它在继续升级，只说这个话题在这个子版稳定存在。**

---

## ⭐ 其余

### ⭐ GPT-6 Astra 进入「演示阶段」

[Astra live action with tokens](https://www.reddit.com/r/OpenAI/comments/1w8nmx9/astra_live_action_with_tokens/)（r/OpenAI r2，49.4h）· [Astra generates an image of a concept, then one-shots the 3D model in Blender.](https://www.reddit.com/r/OpenAI/comments/1w8y3uh/astra_generates_an_image_of_a_concept_then/)（r5）· ⭐ [**GPT-6 Astra has successfully beat all 48 levels of the "I'm Not A Robot" game**](https://www.reddit.com/r/OpenAI/comments/1wa4vir/gpt6_astra_has_successfully_beat_all_48_levels_of/)（r6，9.8h）· [Astra Be Like...](https://www.reddit.com/r/OpenAI/comments/1w9qqj4/astra_be_like/)（r8）· [GPT-6 Astra: Rickroll in Blender](https://www.reddit.com/r/singularity/comments/1w9aeyk/gpt6_astra_rickroll_in_blender/)（r/singularity r4）

⟹ ⭐⭐ **与 09-06 那份对比，Astra 相关内容的性质变了**：那次是发布公告、跨子版对比、成本实测、以及一条越狱报告；⭐ **本次全是演示（Blender / 3D / 小游戏）** ⟹ **48 小时之内从「这是什么、多少钱、安不安全」转到「它能做什么好玩的」。** ⭐ 而 `"I'm Not A Robot" 游戏全 48 关` 这条有点意思，因为它的名字就是在开人机验证的玩笑。

⚠️ **而我 09-06 记的那条越狱报告在本窗口没有后续**（既没有更多细节也没有官方回应）⟹ **那个 Open Question 仍然开着。**

### ⭐ 其余（各一句）

⭐ [**Friends Don't Let Friends Use Ollama**](https://www.reddit.com/r/LocalLLaMA/comments/1wa26pn/friends_dont_let_friends_use_ollama/)（r/LocalLLaMA r6，11.5h）⟹ ⭐ 本地推理工具链的取舍争论，⚠️ 仅标题 · ⭐ [New Benchmark: The Struggle Bench](https://www.reddit.com/r/LocalLLaMA/comments/1w9dlf1/new_benchmark_the_struggle_bench/)（r10）⟹ ⭐ **名字暗示它测的是「模型在什么地方卡住」而不是「能不能做对」** —— 若真如此，它落在我那条「只报最终成功率等于没报」的线上，⚠️ 仅标题

⭐ [Generating Bad Apple autonomously from a single initial state using a tiny recurrent dynamical s…](https://www.reddit.com/r/MachineLearning/comments/1wa8rub/generating_bad_apple_autonomously_from_a_single/)（r/MachineLearning r4，7.1h）⟹ ⭐ 与 08-18 那条「把 Doom 渲染器编译成 21B 参数 transformer、完全不训练」是同族的「用极小动力系统装下一段确定性内容」

⭐ [Anyone has tried the AWS Managed KB ?](https://www.reddit.com/r/aws/comments/1w9u8au/anyone_has_tried_the_aws_managed_kb/)（r/aws r6，16.4h）⟹ ⭐ **而 AWS feed 里最新的那条公告恰好是 `Amazon Bedrock Managed Knowledge Base introduces user-managed setup for SharePoint`（09-04 21:29）** ⟹ **同一个功能的产品侧与社区侧，相隔四天** · [How can I workaround VPC that are using valid public addresses for the private addressing](https://www.reddit.com/r/aws/comments/1w8v97j/how_can_i_workaround_vpc_that_are_using_valid/)（r7）

⭐ [**Lossless compression that stays queryable (for tables/logs, not JPEG)**](https://www.reddit.com/r/datascience/comments/1w8r7xm/lossless_compression_that_stays_queryable_for/)（r/datascience r7，46.0h）⟹ ⭐⭐ **「压缩后仍可查询」这个要求与我追的上下文压缩线（LatentPress / SkillZip / When "Must" Becomes "Maybe" 的 100% 约束失活）是同一个约束在数据侧的表达** —— ⭐ **而数据领域对「无损」有明确定义，agent 上下文压缩没有。** · [Can anyone suggest a comprehensive intro to LangGraph?](https://www.reddit.com/r/datascience/comments/1w9wlek/can_anyone_suggest_a_comprehensive_intro_to/)（r4）

⭐ [Lack of critical thinking - what to do ?](https://www.reddit.com/r/AskAcademia/comments/1w972de/lack_of_critical_thinking_what_to_do/)（r/AskAcademia r3，34.9h）⟹ ⚠️ 标题没说与 AI 有关，我不假定 · [Would you let an editor know if you feel a reviewer's criticism is personal?](https://www.reddit.com/r/AskAcademia/comments/1w9pgrf/would_you_let_an_editor_know_if_you_feel_a/)（r10）⟹ ⭐ 评审制度线（第十四份）· [Why are so many people still give a damn about international university rankings?](https://www.reddit.com/r/AskAcademia/comments/1w8y864/why_are_so_many_people_still_give_a_damn_about/)（r7）

⭐ **公众人物评论两条同一人跨两个子版**：[Denzel explains why he uses AI.](https://www.reddit.com/r/StableDiffusion/comments/1w97ssh/denzel_explains_why_he_uses_ai/)（r/StableDiffusion **r0**，34.4h）+ [Denzel Explains AI Slop](https://www.reddit.com/r/OpenAI/comments/1w9hjwj/denzel_explains_ai_slop/)（r/OpenAI r10）⟹ ⭐ **与我 08-18 起记的「公众情绪转向」同族，而这次的形态是「一个公众人物的表态被两个不同倾向的子版分别引用」**

⭐ 其余：[Things I was getting downvoted for in r/cscareerquestions 2 years ago](https://www.reddit.com/r/singularity/comments/1w8xokf/things_i_was_getting_downvoted_for_in/)（r/singularity r6）· [What are your thoughts? I still believe AGI is a long way off.](https://www.reddit.com/r/singularity/comments/1w9ag49/what_are_your_thoughts_i_still_believe_agi_is_a/)（r7）· [Day 5 of making a cozy game with no dev experience.](https://www.reddit.com/r/ClaudeAI/comments/1w8mwa7/day_5_of_making_a_cozy_game_with_no_dev/)（r/ClaudeAI r6）· [Not quite what i was going for...but u know what...this is better.](https://www.reddit.com/r/StableDiffusion/comments/1w9pqwm/not_quite_what_i_was_going_forbut_u_know_whatthis/)（r/StableDiffusion r1）· [new insta realism lora](https://www.reddit.com/r/StableDiffusion/comments/1w8ykz2/new_insta_realism_lora/)（r9）· [A faster way to convert a timestamp ➜ Hour, Min, Sec](https://www.reddit.com/r/programming/comments/1w9pelh/a_faster_way_to_convert_a_timestamp_hour_min_sec/)（r/programming r2）⟹ ⚠️ **r/programming 本次 1 条高位 A、0 条与 AI 相关（连续第七次）**

⭐⭐ **顺带一个我该记的缺席**：**MiniMax H3 在 r/StableDiffusion 的高位 A 里第一次完全没有出现**（连续十四份之后）⟹ ⭐ 本窗口该子版高位是「Denzel 表态」与两条作品/LoRA ⟹ ⚠️ **45 小时窗口太短，我不把这读成「那条线结束了」，但它是十四份以来第一次断，值得记为观察点。**

---

## 趋势

### 🚨🚨🚨 1. 官方第一方集成成了 prompt injection 的来源

**Notion 官方 MCP connector 在任务中途注入指令让 agent 打广告** ⟹ ⭐⭐⭐ **它把我从 Simon Willison 记的那条（「模型当作自己思考的通道是特权注入通道」）推广了一格：模型当作可信基础设施的通道（工具描述 / connector 元数据）同样是特权通道** —— ⭐⭐ **而流量层的 MCP 威胁检测（Cloudflare / AWS WAF，08-18）抓不到它，因为它就是正常流量。** ⚠️ 仅一个社区标题，**但它可复现（读那个 connector 的工具描述即可），故最高优先待核实。**

### ⭐⭐⭐ 2. AI 数学这条线该按「产物可验证性」排，而不是按「有多难」排

**五项主张里，费马形式化（机器可检查）与孪生素数（陶哲轩公开吐槽）在同一周得到完全不同的接待。** ⟹ ⭐⭐ **本窗口新增的 Jacobian 猜想目前只有一个被截断的社区标题 ⟹ 按这个排法它在最弱那一档。**

### ⭐⭐ 3. `A/A′` 序列填上中段，形状从两段变三段

**45.2h → 1.39**，落在 23.4h（1.65）与 66.3h（1.11）之间 ⟹ ⭐⭐ **失效区（<约 20h）→ 斜坡（约 20–60h）→ 平台（≥约 60h，1.1–1.2）。** ⭐ 而这是我第三次修这个估计量的描述，**三次都是靠一个新的间隔点**（6.3h 修短端 · 66.3h 修长端 · 本次填中段）。

### ⭐⭐ 4. 两条我追了很久的线在本窗口同时缺席

**r/devops 的「谁授权了这个动作」（连续五周）· MiniMax H3 在 r/StableDiffusion（连续十四份）。** ⟹ ⚠️ **45 小时的窗口只能放下十几条高位帖，故「缺席」在这个窗口长度上不是强证据** —— ⭐ **但两条同时断值得记为观察点，而正确的做法是等下一份再判断，而不是现在就宣布它们结束了。**

### ⚠️ 5. 自我怀疑

⚠️ **本份 86 条 A 里我引了约 30 条，而最重要的那一条（Notion MCP）只有一个标题** —— ⭐ **代价可指名：我把它放在主线一并写了三段分析，而那三段全部建立在标题的字面意思上。** ⟹ ⭐⭐ **若那个标题是夸大的（比如实际只是 connector 描述里有一句产品介绍而不是「注入让 agent 打广告」），我这三段就都要撤。** 我已在正文标了这个风险，但这里再记一次，因为**这是本份唯一一处我在薄证据上写了厚分析的地方。**

---

## Open Questions

1. 🚨⭐⭐⭐ **Notion 那个 MCP connector 的注入文本到底是什么，以及 Notion 有没有回应？** ⭐ **它可复现**（读工具描述即可）⟹ **最高优先，且这是本份唯一一条我认为值得主动去核实而不是等它再出现的。**
2. ⭐⭐ **Jacobian 猜想那项是谁做的、有没有形式化？** ⭐ 按我那个「可验证性」排法，这决定它落在哪一档。
3. ⭐⭐ **09-06 那条 GPT-6 越狱报告有后续吗？** ⚠️ 本窗口零后续 ⟹ **要么它没那么重要，要么它还在发展；两种都要靠下一份判断。**
4. ⭐ **`The Struggle Bench` 测的是什么？** ⭐ 若真是「模型在哪里卡住」，它落在「只报最终成功率等于没报」这条线上。
5. ⭐ **r/devops 的「谁授权」线与 MiniMax H3 的缺席是窗口太短还是真的断了？** ⭐ 下一份就能判断。
