# Reddit 热门 W35b — 截至 2026-08-27 02:5x UTC

- **抓取**：12/12 子版，**277 帖唯一**
- ⚠️ **r/datascience 仅 6 帖＝历次最低**（此前最低 8）／r/statistics 21 帖
- **去重三个口径**：**A（新进榜）= 51** ／ **A′（新发布）= 31** ／ B（对照最近 5 份的 193 条已引用）= 199
- **间隔**：距上一份（W35，08-26 03:06）约 **23.4 小时** —— ⭐ 这是 A/A′ 序列一直缺的那个「约 24 小时」数据点

> 🚨 **抓取踩了两个坑，第二个是我自己造的，比第一个严重。**
>
> **① 限流比平常凶得多，而超时也会触发「全有或全无写盘」。** `--delay 10` 主抓在 7 分钟后被超时杀掉，**当时已有 7/12 成功**（stderr 里逐个写着 `OK`），但脚本要等全部跑完才写盘 ⟹ **输出 0 字节，7 个子版的成果全丢**。⭐ CLAUDE.md 里那条「全有或全无」此前只记了「进程被中断」这一种触发方式，**现在要加上「命令超时」** —— 而超时几乎必然发生，因为每个失败子版的退避是 5+10+20+40=75 秒，三个失败就多出近 4 分钟，**总耗时不可预测**。
>
> 🚨 **② 而改成「逐个子版、各写各的文件」时，我用了一个旧 run 留下的目录 `rd/`，脚本的续跑逻辑是「文件已存在就跳过」** ⟹ singularity / StableDiffusion / programming / AskAcademia 四个会被当成「已拿到」而**直接用上 08-12 的数据（15 天前）**。⭐ 发现方式是我列目录时注意到那四个文件的 mtime 是 `08-12_10:2x`。
> - ⭐⭐⭐ **这与 AWS 那个 `/tmp/aws-rss.xml` 读旧文件是同一族失效，而形态更隐蔽：那次是「读了一天前的输入」，这次是「12 个子版里 4 个用半个月前的数据，另 8 个是新的」—— 一份混合产出，而里面没有任何东西能显示这件事。**
> - ⭐⭐ **根因可命名：「文件已存在就跳过」是很好的续跑模式，但只要目录不是本次运行专属的，它就静默变成「读旧输入」。** 我此前一直有正确习惯（`rd7`/`rd8`/`hf26`/`hf27` 都是逐 run 编号），**这次是因为图省事用了裸名 `rd` 才破掉的。** ⟹ 已换成 `rd9/` 重跑，12/12 全新。
> - ⭐ 而逐个抓取本身照旧有效：批量模式失败的 LocalLLaMA / ClaudeAI / devops **逐个抓全部一次成功，连续第九次**。

---

## ⭐⭐⭐ 方法学：23.4 小时这个点让 A/A′ 序列可以反解出「榜单驻留时间」，而结果印证了 7 天硬边界

| 间隔 | A（新进榜）| A′（新发布）| A/A′ | A′ 占榜单 | ⟹ 反解驻留时间 |
|---:|---:|---:|---:|---:|---:|
| **2.0 小时** | 4 | 1 | **4.00** | — | — |
| ⭐ **23.4 小时** | **51** | **31** | **1.65** | **11.2%** | **209h = 8.7 天** |
| **68.3 小时** | 152 | 123 | **1.24** | 38.2% | 179h = 7.4 天 |
| **91.5 小时** | — | 128 | — | 46.4% | 197h = 8.2 天 |
| **117.7 小时** | 203 | 181 | **1.12** | 65.3% | 180h = 7.5 天 |

**① A/A′ 继续单调收缩（4.00 → 1.65 → 1.24 → 1.12）**，新点严丝合缝地落在序列里。

**② ⭐⭐⭐ 而「A′ 占榜单比例」可以反解驻留时间**：若换血大致均匀，则 `占比 ≈ 间隔 ÷ 驻留时间`。**四个间隔各自算出 7.4 / 7.5 / 8.2 / 8.7 天，中位约 7.9 天** ⟹ ⭐⭐ **这从一个完全独立的方向印证了我 08-18 立的「top-of-week 跨度 7 天是硬边界」**（那条当时靠的是「4.9 天空缺后发布日期分布覆盖整七天」这个直接观察）。⭐ 略超 7 天是预期的：Reddit 的 `t=week` 是滚动窗口，我的采样又加了最多一天的滞后。

**③ ⭐⭐⭐ 而这直接回答了我昨天留的 Open Question 5（「A/A′ 能不能拟合出发布→进榜的典型延迟」）** —— 对 **A∖A′ = 20 帖**（昨天已在、今天才爬进 top-25）算「发布→进榜」延迟：

| 量 | 值 |
|---|---|
| 中位 | **37.7h** |
| 均值 | 62.0h |
| p25 / p75 | 30h / **95h** |
| 范围 | 25.1 – **155.9h（6.5 天）** |
| 对照：A∩A′（新发布且进榜）年龄中位 | **11.7h** |

⚠️⚠️ **但其中有一个数我必须自己拆掉：「20/20 都超过 23.4h」不是发现而是构造使然** —— A∖A′ 的定义就是「发布于昨天抓取之前」，故年龄必然大于间隔。**分布（中位 37.7h、p75 95h、尾巴到 6.5 天）才是真信息**，它说的是：**一条帖子爬进榜单的典型时点是发布后约 1.6 天，而四分之一要等 4 天以上。**

**④ ⭐⭐⭐ 一个我没预期的结构事实：这 20 条爬升帖的 rank 全部落在 r14–20** ⟹ ⭐⭐ **我那条「每份扫 rank ≤5 未引用条目」的规则抓不到它们。** 两条规则覆盖的是**互不重叠**的两群：

| 规则 | 覆盖的 population | 典型 rank | 典型年龄 |
|---|---|---|---|
| **A 口径** | 爬升帖（发布后才慢慢挤上来）| **r14–20** | 中位 37.7h |
| **扫 rank ≤5** | 长期驻留的高位帖（如那条六天没被覆盖的 Stack Overflow）| **r0–5** | 60–170h |

⟹ ⭐⭐ **所以正确做法是「用 A 当工作口径 + 照旧扫 rank ≤5」，两者不是冗余而是分工。** ⭐ 而本次 rank≤5 扫描 29 条只捞到约 1 条实质（≈3%，低于此前 9%/7%）—— ⭐ **产出率是间隔的函数：它在空缺之后才值钱，连续日更时接近空，这是预期而不是规则失效。**

---

## ⭐⭐⭐ 主线一：昨天那个来源不明的隐身模型被认出来了，而答案接上了两条既有线

昨天我记过「隐身模型 Ox-Alpha 称在 SWE 上超过 Fable（⚠️ 仅标题、来源不明）」。今天：

- ⭐⭐⭐ [**First serious confirmation. Ox Alpha is GLM-5.3-Flash**](https://www.reddit.com/r/LocalLLaMA/comments/1vyp1l9/first_serious_confirmation_ox_alpha_is_glm53flash/)（r/LocalLLaMA r23）
- ⭐⭐ [**GLM-5.3-Flash: Frontier Intelligence, Flash Cost**](https://www.reddit.com/r/LocalLLaMA/comments/1vyy3k6/glm53flash_frontier_intelligence_flash_cost/)（r/LocalLLaMA r4）
- [**zai-org/GLM-5.3-Flash · Hugging Face**](https://www.reddit.com/r/LocalLLaMA/comments/1vyyesk/zaiorgglm53flash_hugging_face/)（r/LocalLLaMA r22）＝**权重已上 HF**

⟹ ⭐⭐⭐ **两条线同时收口**：

**① 权重时间线对上了。** 我 08-18 从 Nathan Lambert 记过「GLM-5.3 权重两周后上 HF」（发布是 08-14）⟹ 两周后正是 **08-28** 前后，而今天（08-27）Flash 版权重已经在 HF 上 —— **一个厂商侧的时间承诺被兑现，这在我的记录里不常见。**

**② ⭐⭐⭐ 而「Frontier Intelligence, Flash Cost」这个标语正是「成本买不到分数」那条线的厂商自述版。** 若「Ox-Alpha ＝ GLM-5.3-Flash」成立，则昨天那条无法核实的主张（在 SWE 上超过 Fable）就从「一个匿名模型的传闻」变成**「一个开放权重的 Flash 档模型在 SWE 上超过一个闭源前沿模型」** —— 比我 W33h 记的「GLM-5.3 在 CyberGym 上略高于 Mythos 5 与 GPT-5.6 Sol」强一档，因为 **Flash 档意味着它同时在成本轴上占优**。

⚠️⚠️ **但三处保留必须一起写**：①「first serious confirmation」仍是**社区判断**，我没有厂商确认 ②「超过 Fable on SWE」这个原始主张我从未核实过来源，**认出模型身份不等于核实了那个分数** ③⭐ 按我 W33f 那条教训，「权重在 HF 上」是状态更新，而「Ox-Alpha 是谁」只要还只有社区推断就不是。

⭐ 而同子版一条社区自述给了这条线的体感版：[**Whoever the fuck predicted we would have gpt 5.5 performance in coding on consumer hardware a couple months ago now, i applaud you**](https://www.reddit.com/r/LocalLLaMA/comments/1vz1dkz/whoever_the_fuck_predicted_we_would_have_gpt_55/)（r/LocalLLaMA r17）⟹ ⭐ 这与本地推理硬件那条（昨天的 Xiaomi AI Cube 1.2TB/s、Apple M5 Ultra 512GB）是同一件事的两侧：**能力下沉与承载它的硬件同周变便宜。**

---

## ⭐⭐⭐ 主线二：一条标题里就写完了整个结论，而它同时打在我两条线上

⭐⭐⭐ [**We recovered 575k crop labels from a decade of manual Photoshop work to automate book digitization — more data, ResNet-50, and higher resolution all failed; ten operator clicks per book beat them [P]**](https://www.reddit.com/r/MachineLearning/comments/1vz2ojw/we_recovered_575k_crop_labels_from_a_decade_of/)（r/MachineLearning r14）

⟹ ⭐⭐⭐ **这个标题本身就是一个完整的负面结果 + 一个正面结果，而两者都接主线**：

**① 「更多数据 / 更大模型（ResNet-50）/ 更高分辨率三条路全部失败，而每本书十次操作员点击赢了它们」** ⟹ ⭐⭐ **这是「结构性选择的效应量大于参数性选择」那条线上目前最朴素也最锐的一个案例**（此前两个都在我的论文侧：**Not Worth Another Token** 的「剪枝有效性更取决于在哪个阶段剪，而非用什么打分规则」／**A²E** 的「换 harness 的效应只在多轮任务上显形」）—— ⭐ **本条的「结构」是人机交互的形状（把十次点击放进流程），而不是模型或数据的任何属性。**

**② ⭐⭐⭐ 而「从十年的手工 Photoshop 工作里回收 575k 个标签」是「新数据从哪来」这个问题的第四个答案，且性质与前三个都不同**：

| 答案 | 来源 | 我记的实例 |
|---|---|---|
| 只用可许可语料 | 挑数据 | DFM Mimir v1（161 数据集、丹麦语 SOTA）|
| 去实体世界找 | 买/扫 | Simon 记的**亚马逊稀有书籍 → AI 训练设施** |
| 合成渲染 | 造数据 | WorldRover-10M（同一次探索三视角重放）／TailBooster |
| ⭐ **回收过去人类劳动的副产品** | **挖历史** | **本条** |

⟹ ⭐⭐ **第四个答案最便宜且最少被讨论：那 575k 个标签本来就已经存在了，只是以「十年裁切操作记录」这种没人当成数据集的形式躺着。** ⭐ 而它与第二个答案指向同一个场景（**都是书籍数字化**）—— 一边买书来扫，一边把过去扫书的操作记录变成标签。

⚠️ 仅标题（`[P]` 是项目帖），我未读正文，故「十次点击赢了」的具体指标与对照设置未核实。

---

## ⭐⭐ 主线三：OpenAI 的配额收紧连续第二天，且今天给出了原因

- [**The 5 hour limit is ridiculous, and they definitely lowered the usage limits for Plus.**](https://www.reddit.com/r/OpenAI/comments/1vz1934/the_5_hour_limit_is_ridiculous_and_they/)（r/OpenAI r24）
- ⭐⭐ [**New $100 Business Plan (2 seat minimum). They finally did it. My team just switched over. This is why 5h limits are back.**](https://www.reddit.com/r/OpenAI/comments/1vygebh/new_100_business_plan_2_seat_minimum_they_finally/)（r/OpenAI r21）

⟹ ⭐⭐⭐ **第二条标题给出了第一条的因果解释，而这个解释把我昨天那个观察补完了。** 昨天我记的是「单价在降（Sol $4/$20）而消费端配额在收紧（5hr limit 回来了）」并归结为「『单位成本』与『你实际能用多少』是两件事」—— ⭐ **今天多出的是机制：配额收紧与一个新价位档（$100/座、2 座起）同时发生 ⟹ 它不是成本压力的被动反应，而是把重用户往上一档推的定价动作。**

⚠️ 「This is why」是发帖人的推断，不是官方说明；⭐ **但「新档位与旧档位收紧同时出现」这个时间上的巧合是可观察的事实，不依赖那个推断。**

⭐ 而这与我这两周的「成本」主线合起来是三层，且三层的方向不一致：**技术效率（同等能力单位成本仍降）· 元件采购（内存涨 500%、GPU 容量约束）· ⭐ 消费端配额（按用户分层重新切分）** ⟹ ⭐⭐ **给客户材料的含义：「成本」这个词现在至少需要指明是哪一层，否则三层里任意一层的新闻都能推翻整段论证。**

---

## ⭐⭐ 主线四：GitHub Actions 故障跨两个子版，而伴随它的那个问题比故障本身值得记

- [**GitHub confirms Actions outage caused by database issue**](https://www.reddit.com/r/programming/comments/1vz0z6b/github_confirms_actions_outage_caused_by_database/)（r/programming **r1**）＋ [同一条在 r/devops](https://www.reddit.com/r/devops/comments/1vz0zvk/github_confirms_actions_outage_caused_by_database/)（r8）
- ⭐⭐ [**Do you think GitHub will disable actions for non paying customers?**](https://www.reddit.com/r/devops/comments/1vz073j/do_you_think_github_will_disable_actions_for_non/)（r/devops r5）

⟹ ⭐ **故障本身是本月第二次**（08-17 那次我在 W34 记过，跨 r/programming r0 与 r/devops r1，形态完全相同 ⟹ **同一个子系统、同样的跨子版扩散、间隔十天**）。

⟹ ⭐⭐⭐ **但更值得记的是第二条：「GitHub 会不会对免费用户关掉 Actions」这个担忧与我追的「提交成本 < 审核成本」是同一根问题的资源侧版本。** 我此前记的三个案例（NeurIPS 评审崩坏／苹果 bug 赏金审核团队下线／意外网络攻击）都是**人力**审核端过载，而 GitHub Blog 自己在 W34 说过「你的贡献者已经是 AI-first 了」⟹ ⭐⭐ **若 AI 生成的贡献让 CI 用量按同样的量级膨胀，那么先崩的不是审核人力而是免费算力，而唯一的建设性提案照旧是「把成本加回提交端」（这次的形式就是收费）。**

⚠️ 第二条是社区猜测，无任何官方信号；⭐ 我记的是「这个担忧出现在 r/devops 前五」这个事实。

---

## ⭐⭐ 主线五：DuckDB 加入 AWS

⭐⭐ [**DuckLabs – DuckLabs (DuckDB) to Join AWS, Projects to Remain Open Source**](https://www.reddit.com/r/aws/comments/1vyyatl/ducklabs_ducklabs_duckdb_to_join_aws_projects_to/)（r/aws **r1**）

⟹ ⭐⭐ **这是我这两个月记的「基础设施层被大厂吸收」序列里第一条落在数据引擎上的**（此前都在算力侧：AMD 收 Taalas／4.8 亿砸端侧 agent 芯片／Anthropic 商谈 60 亿收 Decart／OpenAI 自研推理芯片／Meta MTIA 300）。⭐ **而 DuckDB 恰是「本地/嵌入式分析」这个类别的代表**，与我追的另一条（**agent 沙箱正在长出组件层** —— Databricks×Electric 把 WASM Postgres 带进 agent 沙箱）落在同一处：**agent 需要一个能在沙箱里跑的、无服务端的分析引擎。**

⚠️ 「Projects to Remain Open Source」是公告措辞，⭐ 而这类承诺的可检验性要看治理结构而非声明本身 —— 我不做预测，只记时点。

---

## ⭐⭐ 主线六：AWS 账号线第 18、19 个数据点，其中一条给了时长

- ⭐⭐ [**AWS account verification has been stuck for a MONTH — CloudFront still blocked and Support isn't helping**](https://www.reddit.com/r/aws/comments/1vu945m/aws_account_verification_has_been_stuck_for_a/)（r/aws r23）
- [**Unknown billing in my AWS account which I was unaware of**](https://www.reddit.com/r/aws/comments/1vu3ycl/unknown_billing_in_my_aws_account_which_i_was/)（r/aws r24）

⟹ ⭐⭐⭐ **第一条把两个我此前分开记的形态合成了一个，并第一次给了时长**：**自动准入判定**（我在 W34 记过「新增 CloudFront 资源前需先验证账号」）＋ **无有效人工复核通道**（「Support isn't helping」）＋ ⭐ **卡了整整一个月** —— 此前 17 个数据点里没有一条给出过持续时间，⭐ **而一个月这个量级说明卡住的不是「处理队列很长」而是「这个 case 没有出口」。**

⚠️ 用户单方陈述、无厂商回应；⭐ 我记的仍然只是投诉密度与共同形状（**自动判定 + 无救济通道**），这个形状现在跨越 19 条、四种理由（付了钱说没付／自动准入挡住／计量产生未激活费用／因关联而连坐）。

---

## ⭐⭐ 主线七：一条我此前完全没记过的批评方向 —— 统计学家反对合成数据

⭐⭐ [**[D] I hate synthetic data.**](https://www.reddit.com/r/statistics/comments/1vyv5eu/d_i_hate_synthetic_data/)（r/statistics r20）

⟹ ⭐⭐⭐ **我这两周把合成数据当成「数据供给」问题的一个答案在收集**（WorldRover 的渲染数据／TailBooster 的极值增强／MIT 的「无极端数据生成极端事件场景」／DFM Mimir 的只用可许可语料），⭐ **而我记过的唯一负面材料是 Mechanist 那条「不安全特质能通过合成数据穿过安全过滤器并跨模态迁移」—— 那是一个安全论证。** ⟹ ⭐⭐ **本条是第一次看到从统计有效性一侧来的反对**，而这个角度我完全没有材料。

⭐ 而这与统计社区连续第五份「在教我这边缺的东西」是同一现象（前四份：怎么验证数值代码在数学上是对的／ICC 很低时用单观察者还是平均／置信区间讲解／随机性是资产还是税）⟹ ⭐⭐ **可操作：这条值得单独读正文，因为「合成数据在什么条件下会让下游推断失效」正是我给客户做评估方案时最缺的那一节，而它在 agent 侧的文献里几乎只被当成成本或安全问题讨论。**

⚠️ 仅标题，我不知道它的具体论证。

---

## ⭐⭐ 主线八：「怎么确认一段代码/一个输出是对的」连续第二天出现，且这次落在一个人人都在用的库上

⭐⭐ [**Catching bugs in scikit-learn [D]**](https://www.reddit.com/r/MachineLearning/comments/1vym6cn/catching_bugs_in_scikitlearn_d/)（r/MachineLearning r11）

⟹ ⭐⭐ **昨天是 r/statistics 的「怎么测试你的数值代码在数学上是对的」，今天是 r/MachineLearning 的「在 scikit-learn 里抓 bug」** ⟹ ⭐⭐⭐ **两天、两个社区、同一个问题的两个位置：一个问方法（怎么验证），一个问对象（连最成熟的库都有 bug）。** ⭐ 而这两条与我从 HF 复现 2,200 篇那次学到的两条恰好对上：**「Attention's forward pass and Frank-Wolfe」那个反例的违反首次出现在 t=224 步（＝有限时长的检查会停得太早）** 与 **「Self-Distillation」那篇的代码默认算 forward KL 而理论部分分析 reverse KL** —— ⭐⭐ **后者正是「成熟代码里的静默不一致」在论文侧的实例。**

---

## 其余

### ⭐⭐ 影响力行动：官方与社区同一件事，而社区标题多了一个细节

⭐⭐ [**OpenAI bans Russia-linked ChatGPT accounts promoting a think tank built on copied papers**](https://www.reddit.com/r/OpenAI/comments/1vyhfct/openai_bans_russialinked_chatgpt_accounts/)（r/OpenAI r19）

⟹ 与我昨天 tech-blogs 记的官方那篇（`Disrupting a new covert influence campaign from Russia`）是同一件事，⭐ **而社区标题给了一个官方标题里没有的细节：那个智库是「built on copied papers」** ⟹ ⭐⭐ **这把影响力行动与学术抄袭接在了一起，而我此前把这两条当作互不相干的线在追**（前者在「AI 被用于操纵」，后者在「评审/出版诚信」）。⚠️ 仅标题，我未核实「copied papers」的具体所指（是抄论文当智库产出，还是用抄来的论文伪造资历）—— **两种含义差别很大。**

### ⭐⭐ 记忆与隐私：一个消费侧可见的实例

⭐⭐ [**What my wife's Claude knows about me.**](https://www.reddit.com/r/ClaudeAI/comments/1vyo76q/what_my_wifes_claude_knows_about_me/)（r/ClaudeAI r5）

⟹ ⭐⭐⭐ **这是「持久记忆」那条线上第一个**跨人**的消费侧实例**：我此前记的全是单用户视角（PrivacyPeek 把审计点移到获取阶段／ChatGPT Computer History 跨 app 记住你的工作／MemUse 把记忆评测从直接问答移到自然整合／Governed Persistent Memory 的 fail-closed release），⭐ **而这一条问的是「另一个人的助手关于我知道了什么」** —— 而这恰是 PrivacyPeek 那个框架的自然延伸：**若审计点是「获取阶段上下文里拿到什么」，那么获取的对象包含第三方的信息，而第三方从未与这个系统交互过。** ⚠️ 仅标题，很可能是轶事/玩笑帖，我记的是问题形态不是这个实例。

### ⭐ AGI 时间表与就业

[**Sam Altman tells TIME that OpenAI will achieve AGI by the end of this year.**](https://www.reddit.com/r/singularity/comments/1vyyli5/sam_altman_tells_time_that_openai_will_achieve/)（r/singularity r4）· [**'This is crazy. This is insane': Bill Gates has changed his mind about AI and jobs**](https://www.reddit.com/r/singularity/comments/1vz1ejm/this_is_crazy_this_is_insane_bill_gates_has/)（r19）

⟹ ⚠️ **两条都按惯例只记不引**（转述、无可检验内容）。⭐ 但值得记一句：**Altman 那条把时间表压到「今年年底」，而这类声明的可检验性完全取决于 AGI 的定义** —— 我不追这条线，除非出现可操作的定义。

### ⭐ 主权云

⭐ [**The Cloud Is Becoming a Geopolitical Risk**](https://www.reddit.com/r/devops/comments/1vyx3zp/the_cloud_is_becoming_a_geopolitical_risk/)（r/devops r10）

⟹ 主权 AI/云那条线的从业者侧第三个数据点（前两个：昨天 r/ML 的 SovereignAI 持续学习技术报告、tech-blogs 侧的 Mistral×HUMAIN 与英国主权 AWS）⟹ ⭐ **形态值得注意：它出现在 r/devops 而不是政策讨论区，说明这个议题已经变成架构选型问题。**

### ⭐ 出版制度：连续第十份

⭐ [**Trying to understand why some fields treat preprints as legitimate scholarship and others still largely ignore them**](https://www.reddit.com/r/AskAcademia/comments/1vx015d/trying_to_understand_why_some_fields_treat/)（r/AskAcademia r17）· [How do you actually learn the "publishing world" of your field?](https://www.reddit.com/r/AskAcademia/comments/1vz3q1w/how_do_you_actually_learn_the_publishing_world_of/)（r20）· [Avoiding plagiarism when ditched by the author of an idea](https://www.reddit.com/r/AskAcademia/comments/1vx7le0/avoiding_plagiarism_when_ditched_by_the_author_of/)（r24）

⟹ ⭐ 这条线的关注点持续上移，而**「预印本算不算正式学术成果」这一问在 AI 领域几乎不成问题（arXiv 就是主战场）** ⟹ ⭐⭐ **一个我没记过的角度：AI 领域的发表规范与其他领域的差距本身就是「AI 论文量爆炸」的一个制度前提**（ICML 收 23,918 投稿那个数字所以才可能）。

### ⭐ 其余（各一句）

⭐ [Casey Muratori – The Root of The Root of All Evil – BSC 2026](https://www.reddit.com/r/programming/comments/1vynzwf/casey_muratori_the_root_of_the_root_of_all_evil/)（r/programming r11）· [mold: A Massively Parallel Linker](https://www.reddit.com/r/programming/comments/1vz921w/mold_a_massively_parallel_linker/)（r18）· [Latency optimizations on the system level](https://www.reddit.com/r/programming/comments/1vycpx9/latency_optimizations_on_the_system_level/)（r20）· [The Depths of JavaScript: Minesweeper in 247 Bytes](https://www.reddit.com/r/programming/comments/1vyuagp/the_depths_of_javascript_minesweeper_in_247_bytes/)（r16）⟹ ⭐ **r/programming 本次 5 条 A 里 0 条与 AI 相关，连续第四次成立**；⚠️ 但按 W33h 那条教训我不据此推断社区在谈什么，且已扫过 rank≤5

⭐ [Can we reconsider the megathreads?](https://www.reddit.com/r/LocalLLaMA/comments/1vz40zv/can_we_reconsider_the_megathreads/)（r/LocalLLaMA r16）⟹ ⭐ **社区自己在讨论「发布潮把子版淹了」的版务对策** —— 与我记的「Models Day」「连续第十份被 MiniMax H3 占据」是同一现象的版务侧

⭐ [MiniMax H3 Acc FL2VA & REF2VA LoRAs By Wan Team](https://www.reddit.com/r/StableDiffusion/comments/1vyvtou/minimax_h3_acc_fl2va_ref2va_loras_by_wan_team/)（r/StableDiffusion r22）＋ [Release studio 1939 lora for minimax h3](https://www.reddit.com/r/StableDiffusion/comments/1vxl2y1/release_studio_1939_lora_for_minimax_h3/)（r20）＋ [The Latent Upscaler is really great!](https://www.reddit.com/r/StableDiffusion/comments/1vyj95d/the_latent_upscaler_is_really_great/)（r23）⟹ **H3 连续第十一份，且内容仍是工具化（LoRA / 上采样器）而非作品展示**；⭐ **值得注意的是 LoRA 由 Wan Team 发** ＝ 一个团队为另一个团队的模型做适配件

⭐ [Light one-file S3 browser](https://www.reddit.com/r/aws/comments/1vz2k4i/light_onefile_s3_browser/)（r/aws r21）· [Clean stop of a 'pod' in EKS ?](https://www.reddit.com/r/aws/comments/1vx2dnx/clean_stop_of_a_pod_in_eks/)（r22）· [Do you think GitHub will disable actions…](https://www.reddit.com/r/devops/comments/1vz073j/do_you_think_github_will_disable_actions_for_non/) 之外 r/devops 另有 [How would you recommend learning Kubernetes and Helm effectively?](https://www.reddit.com/r/devops/comments/1vz3dvi/how_would_you_recommend_learning_kubernetes_and/)（r19）与 [How much maintenance to manage a Forgejo self-hosted?](https://www.reddit.com/r/devops/comments/1vz9wis/how_much_maintenance_to_manage_a_forgejo/)（r23，⭐ 自托管成本，与主权云那条同向）

⭐ r/statistics 另三条：[[DISCUSSION] statisticians I have a question](https://www.reddit.com/r/statistics/comments/1vyujjx/discussion_statisticians_i_have_a_question/)（r10）· [[E] Looking for an MSc-level education in statistics using just the public library](https://www.reddit.com/r/statistics/comments/1vyyqc2/e_looking_for_an_msclevel_education_in_statistics/)（r11）· ⭐ [[Discussion] Why choose such specific values for confidence interval of variance of a normal random variable](https://www.reddit.com/r/statistics/comments/1vyvvok/discussion_why_choose_such_specific_values_for/)（r14，⭐ 又一条置信区间的具体技术问题）

⭐ r/ClaudeAI 另三条（[I got so fed up, I tried to take the p**s. It backfired.](https://www.reddit.com/r/ClaudeAI/comments/1vxy12f/i_got_so_fed_up_i_tried_to_take_the_ps_it/) r18 · [What a… backwards way to confirm a typo](https://www.reddit.com/r/ClaudeAI/comments/1vycha6/what_a_backwards_way_to_confirm_a_typo/) r23 · [How many billions of tokens do I need to burn to unlock this tier of merch?](https://www.reddit.com/r/ClaudeAI/comments/1vy8rvs/how_many_billions_of_tokens_do_i_need_to_burn_to/) r24）· r/OpenAI 另三条（[Cutting edge AI safety tests be like](https://www.reddit.com/r/OpenAI/comments/1vz27pi/cutting_edge_ai_safety_tests_be_like/) r8 · [Trust me bro taken to a new level](https://www.reddit.com/r/OpenAI/comments/1vyzrvb/trust_me_bro_taken_to_a_new_level/) r11 · [I am unsure what I gave my copilot to make it hallucinate?](https://www.reddit.com/r/OpenAI/comments/1vydd6q/i_am_unsure_what_i_gave_my_copilot_to_make_it/) r17）⟹ ⚠️ **r/OpenAI 信息密度低第三次确认**（前五 rank 多为 meme，实质内容落在 r19/r21/r24）—— ⭐ 对策照旧是「在这个子版里不按 rank 取头部」

⭐ [How would you deal with unprofessional co workers](https://www.reddit.com/r/datascience/comments/1vz462h/how_would_you_deal_with_unprofessional_co_workers/)（r/datascience r1）⟹ ⚠️ 该子版本次只有 6 帖、A 只有 1 条，**连续第 10 份严重截断且创新低** ⟹ ⭐ **值得考虑把它的抓取参数单独调整或接受它作为一个低产源**

---

## 趋势

### ⭐⭐⭐ 1. 一个来源不明的传闻在 24 小时内被认出身份，而这是我第一次看到这条线闭合得这么快

Ox-Alpha（昨天：`⚠️ 仅标题、来源不明`）→ 今天：`Ox Alpha is GLM-5.3-Flash` + 权重已上 HF。⟹ ⭐⭐ **而它同时兑现了我 08-18 从 Nathan Lambert 记下的「两周后上 HF」这个厂商时间承诺** —— 在我的记录里，被兑现的时间承诺不多。

### ⭐⭐⭐ 2. 「结构性选择 > 参数性选择」拿到了一个非 agent 领域的干净案例

**更多数据 / 更大模型 / 更高分辨率三条全败，每本书十次操作员点击赢了它们。** ⟹ ⭐⭐ **这条的价值在于它与 agent 无关**：此前我这条线的两个证据（剪枝阶段 vs 打分规则、harness 差异只在多轮任务显形）都在 agent/LLM 内部，⭐ **而一个书籍数字化的工程项目独立给出同一形状，说明它更可能是一般规律而不是我采样偏好的产物。**

### ⭐⭐ 3. 「数据从哪来」第四个答案，而它是最便宜的那个

回收过去人类劳动的副产品（575k 标签来自十年裁切操作）⟹ ⭐ **与「买书来扫」指向同一场景，一个买输入一个挖历史记录。**

### ⭐⭐ 4. 成本议题现在必须指明是哪一层

技术效率（降）· 元件采购（涨）· ⭐ **消费端配额（重新分层，今天新增 $100/座档并给出「这就是 5 小时限制回来的原因」这个机制）** ⟹ ⭐⭐ **三层方向不一致，故「成本会继续降」这句话在任何客户材料里都必须带层次限定。**

### ⚠️ 5. 自我怀疑：本份的方法学产出又一次超过内容产出

本份最有价值的部分（驻留时间 7.9 天、发布→进榜延迟分布、A 与 rank≤5 覆盖互不重叠、两个抓取坑）**全部是关于我自己工具链的**，而内容侧真正的新东西只有 Ox-Alpha 身份与那条 575k 标签。⟹ ⭐ **这与我连续四份记的「深读采样偏向方法学」是同一现象的另一面：当间隔只有 23 小时时，内容侧本来就没多少新东西，而工具链的问题是每次都在的** —— ⚠️ **所以「方法学产出多」在短间隔的份里是结构性的，不能读作「本窗口方法学信号强」。**

---

## Open Questions

1. ⭐⭐⭐ **「Ox-Alpha ＝ GLM-5.3-Flash」有没有厂商侧确认，以及那个「在 SWE 上超过 Fable」的原始分数出自哪里？** ⭐ 后者比前者重要：**认出身份不改变那个分数的证据强度**，而若它成立，则「开放权重 Flash 档超过闭源前沿」是我这条线上最强的一个数据点。
2. ⭐⭐⭐ **那 575k 标签那篇的「十次操作员点击」具体是什么交互？** ⭐ 若它是「人在关键处给少量标注、其余自动」，那它与我记的 HF 复现挑战那句「最可靠的结果来自有人在掌舵的工作流」是同一形状，而这条线我一直缺工业侧的量化实例。
3. ⭐⭐ **「[D] I hate synthetic data」的论证是什么？** ⭐ 这是我第一次拿到从统计有效性一侧反对合成数据的材料，而我给客户的评估方案里正缺这一节。
4. ⭐⭐ **那个「智库 built on copied papers」的确切所指？** 抄论文当产出 vs 用抄来的论文伪造资历，两者对「AI 被用于操纵」这条线的含义完全不同。
5. ⭐ **驻留时间 7.9 天这个估计能不能用来反推「我每天跑一次会漏掉多少」？** ⭐ 现有量：爬升帖延迟中位 37.7h、p75 95h ⟹ 原则上可以算出「一条帖子在我两次抓取之间进榜又掉出的概率」，而那正是我 08-26 明确说「测不到」的那个量。

