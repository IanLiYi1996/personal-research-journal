# 现代 typed decision models 的产品演进

截至 **2026-10-13 10:51，Asia/Shanghai** 的公开来源快照，不代表当日结束状态。仅研究 TypeSafe、Perplexity、Strands、OpenAI 四家；旧比较笔记只作导航，来源重新读取。没有安装、推理、访问私有资产或委派子任务。这里的对象是把自然语言问题映射为受约束判断的产品，不是离线强化学习的 Decision Transformer；相似接口不能证明技术祖先关系。

## 一、Jev：让语义判断成为软件原语

TypeSafe 自述经历两年隐身研发，目标不是另一个聊天助手，而是让软件能直接调用“理解、选择、打分”。最早核到的带日期公开发布材料是公司署名新闻稿及投资方公告的 **2026-09-15**：Jev 向部分开发者提供 early access。System One 借用快慢思考区分，Jev 名称来自 Jevons；这些是产品定位来源，不是架构谱系证据。[^t-launch]

版本历史披露很稀疏：旧官方 cookbook 仍固定 `jev-1.12`；当前 Models 只列 `jev-1.13.0`，`jev-latest`、`jev-preview` 均指向它。1.13 的失效说明自标“2026-10-02 reviewed”，可证明厂商至迟在该记录日期已讨论此版本，不能当成正式发版日，更不能补出各个 1.x 的发布时间、参数量或训练变化。当前文本输入、共享 state 并行问题、固定版本及别名迁移规则，体现的是可组合和可维护的产品契约；1.12→1.13 的性能增量未知。[^t-version]

RLCD 全称 Reinforcement Learning for Calibrated Decisions，是厂商对训练路线的命名：优化决策及诚实概率，而非偏好中的好听文本。所查一手材料没有足够的奖励公式、训练数据和消融供独立复现；不能把其他 Decider 的交叉熵、蒸馏或温度缩放叫作同一种 RLCD。Jev 的 Choice confidence 是候选最高概率经均匀基线归一化的统计量，不是正确率。其“不能 hallucinate”“百倍提速”须标为营销：不能输出集合外值不代表集合内选对；官方已承认算术、日期、间接推理、对抗内容及选项顺序等失效。[^t-rlcd][^t-confidence][^t-failure]

## 二、Perplexity：把大多模态底座改造成一次读出的判断器

最早核到的权重发布提交：v1 为上海 **10-02 05:24:15**，v1.1 为 **10-05 20:33:43**；后者官方公告直到上海 **10-07 05:11:41**，不能把仓库创建、release commit、公告合并为一天。两版同为 Qwen3.8-27B 微调，适合图文分类、检索判断和工具选择，不提供完整 agent 执行逻辑。[^p-dates][^p-v1]

深读两版 `model.py` 后，关键不是“改成 pointer”：v1 已将 LM vocabulary head 换成由答案代码对应权重初始化的 255 路线性 readout，读取最后位置；v1.1 仍是 `[255,5120]` 独立头及 last pooling。变化在于 full-attention 层取消 causal mask，保留 padding mask 与线性注意力原本的递归行为，并非所有层都变成双向 encoder。代码禁止此模式使用 KV cache；由此推论，它优先整体判断能力，而非 Strands 式因果前缀复用。[^p-code]

数据也同时变化：配置记录训练行数从 73,000 到 626,033；v1.1 增加 tasksource 等来源，训练配置从底座重新起步，并非只给 v1 换 mask。保存的温度从约 2.208 到 1.009；接近 1 不等于在任意分布上天然校准。官方模型卡报告 Decision Index 56.4→61.56，但 Tools 79.3→78.88；没有隔离数据与注意力因素的充分公开消融，收益归因仍是作者解释。10-09 的代码更新又提高图像像素上限，但报告评测仍采用 512²；更高分辨率不能继承旧成绩保证。[^p-training][^p-card]

## 三、Strands：读出结构、训练分布和校准的迭代

公开研究账本追溯实验至 **09-20**，v7 pointer 至 **09-22**、v19 至 **09-28**；账本明确来自未公开开发史，所以是作者追述，不是可检验的同期公开提交。可核的 v19 权重导出提交为上海 **09-30 23:48:47**，公开代码 initial commit 为 **10-01 22:44:49**；v21 权重为 **10-06 00:03:52**。新命名 Qwen v1 卡写“10-09 first release”，实际 release commit 换算上海为 **10-10 04:43:55**。这次 v1 是重设产品命名，不是退回早期 slot v1。[^s-dates][^s-ledger][^s-v1]

早期 slot 将最后隐状态投影到固定槽位，默认 24，未训练槽位及位置捷径限制泛化。v7 起，pointer 用 answer 状态作 query、各候选末 token 状态作 key，共享映射后点积打分；代码确认没有每候选一行参数，候选上限转为 schema 的 255。它保留 causal torso，以 LoRA 小预算适配；仍有顺序敏感，不能称作排列不变。产品解决的是运行时标签绑定和小模型本地判断，并非靠去掉解码就自动获得复杂推理。[^s-code][^s-arch]

v19 在公共分类、多步文档、生成文档及 replay 上加入 6,166 条回答充分性数据，补上此前未教的判断。历史实验中，合成规则数据可拟合近满分却不迁移，强 teacher 在其没有优势的短分类上蒸馏也未带来目标收益；数据类型比单纯行数重要。v21 保留约 123,339 行，加入核验过的问句改写，只在 Qwen3.5-4B teacher 与 gold 一致处蒸馏，且不保留 v20 的 catch-all/指令翻转；六个 seed 按预定规则选代表，不挑最高榜单分。受控同主机均值 v19 172.8、v21 172.3 道正确，不能把发布实例 167→176 解读成因果准确率提升。[^s-ledger][^s-v21]

Qwen v1 扩到 246,678 行，补代码、规则应用及薄弱任务，移除 20,000 条商业使用受限来源；teacher 换为 Gemma-4-31B-it，另用已发布 v19 的 KL 作保留能力锚点。三 seed 的 LoRA 更新与头分数做平均，再校准。公开文件在同一 4096 窗口报告 v19/v21/v1 分别正确 167/176/180，道数上升但 ECE 为 .050/.064/.072，Brier 为 .348/.323/.280：概率质量不同指标不会一起变好，seed 波动也不能省略。这是发布实例比较，不是同硬件、同训练种子的干净消融；原开发实例和导出后的复测数字也有差异。更强 teacher、较宽数据和平均权重一起变化，不能把收益单独归给某个因素。v1 数据构建器尚不在 main，公开权重和配置不等于完整训练可复现。[^s-v19][^s-v21][^s-v1]

同为 v1 的四种 Gemma4 基座并非 Qwen 配方复刻：它们用 v21 数据加代码共 135,602 行、4B teacher，当前只读文本；Qwen 可保留视觉塔但没有图像训练。Gemma 模型卡及加载代码均已核读。PyPI 仍是 0.1.0，最新 Gemma/视觉支持在 main，显示“权重已发布”和“发包成熟”是两回事。[^s-gemma][^s-package]

## 四、OpenAI：托管 Decisions 契约，仍在 public beta

按 OpenAI Docs 重核：Luna 的文本生成发布日期是 **09-22**，Decisions beta 发布日期为 **10-06**，两者不同。当前唯一精确支持模型为 **`gpt-6-luna`**；不能把 Sol、Astra 或其他 Luna 变体列入。端点接受文本和图像，返回 predicate、choice、score，可返回 refusal；它解决不需要解释文本的高吞吐判断，任意 JSON 生成和带参数工具调用仍走其他能力。指南仍称 public beta、“未来数周 GA”只是预期。官方“约 10× 更快”不是本次复测或跨厂商排名；所查官方文档未披露足以验证的 readout、训练和校准消融，不能声称采用 pointer、RLCD 或从其他家演化。[^o-log][^o-guide][^o-model]

## 五、概率与约束为什么不等于正确性

综合推论：Jev 强调托管、并行问题及训练目标；Perplexity 选择大多模态底座、非因果 full attention 和独立槽位读出；Strands 选择因果 pointer、小预算可调权重及显式数据实验；OpenAI 提供成熟生态中的新 beta 契约。它们可替代工作流中的部分判断，不能代替候选生成、证据完整性和执行校验。

最直接反例来自 Strands Qwen v1：strict schema 1.000，却仅 180/231 正确；去掉图像后准确率降近随机，POPE 平均 confidence 仍 .834、ECE .334。约束只管答案合法，softmax 只管候选内相对概率，缺失正确候选或关键证据仍能高置信地错。校准是特定分布上预测群体的频率性质；温度不会改变 Choice argmax，却会改变 Score 的期望。跨任务重拟合还可能改善长文而破坏短分类。省掉自回归解码也没有省掉输入编码、视觉处理和排队，单次前向、托管往返及整个工作流耗时不能直接横比。上述取舍说明，产品的价值是让不确定判断更容易被软件管理，而不是消除不确定性。因此需要真实业务标注、漏选/缺证据测试、按错误代价选阈值和明确回退，而不是把 probability 或 confidence 当作安全证明。[^s-v1][^s-calibration][^s-arch][^p-code][^t-rlcd][^o-guide]

## 精确一手来源

所有链接在本次重新读取；可固定的代码/模型卡固定 revision。带时区时间换算为 UTC+8；只有日期的公告保留来源日历日期，不虚构上海零点。以下成绩均为作者报告，未独立复测。未发现的日期、架构或消融只表明本次核验未建立证据。

[^t-launch]: [TypeSafe 公司署名发布稿，Business Wire 分发](https://www.businesswire.com/news/home/20260915525333/en/TypeSafe-AI-Emerges-From-Stealth-With-%2440M-in-Funding-With-New-Model-for-Composable-AI)；[投资方 DCVC，明确 2026-09-15](https://www.dcvc.com/news-insights/typesafe-emerges-from-stealth-with-a-new-way-of-doing-ai/)；[创始人发布文：研发、命名、并行及营销限定](https://typesafe.ai/blog/introducing-system-one-models-and-jev)。公司新闻稿是发行人一手陈述，不把分发平台当独立验证；官网发布文当前未显示发布日期。
[^t-version]: [旧 Parallel questions cookbook：固定 `jev-1.12`](https://docs.typesafe.ai/cookbooks/parallel_questions)；[当前 Models：`jev-1.13.0`、别名及模型定制边界](https://docs.typesafe.ai/models)；[1.13 jaggedness：Last reviewed 2026-10-02](https://docs.typesafe.ai/model-jaggedness/jev-1.13)。cookbook 的情境日期、价格注释、sitemap lastmod 均未作为模型发布日期。
[^t-rlcd]: [TypeSafe AI primer：RLCD 定位和群体校准定义](https://docs.typesafe.ai/introduction/machine-learning-primer)。
[^t-confidence]: [TypeSafe Confidence：Choice 归一化、Score 的距离统计及阈值风险](https://docs.typesafe.ai/confidence)。
[^t-failure]: [Jev 1.13 失效边界](https://docs.typesafe.ai/model-jaggedness/jev-1.13)。
[^p-dates]: HF 官方提交史：[v1](https://huggingface.co/api/models/perplexity-ai/pplx-decider-v1-27b/commits/main)，release `5117a6c7fe73b19308dc1a6b0fb529a40c2ecad4`，2026-10-01T21:24:15Z；[v1.1](https://huggingface.co/api/models/perplexity-ai/pplx-decider-v1.1-27b/commits/main)，initial 2026-10-05T12:32:27Z，release `db124e5056059be396ec5ba69b97a33dd6adb815`，12:33:43Z。[官方公告正文](https://community.perplexity.ai/t/pplx-decider-v1-1-27b-our-updated-open-weights-multimodal-decision-model-is-now-available/6286)及[JSON 时间](https://community.perplexity.ai/t/pplx-decider-v1-1-27b-our-updated-open-weights-multimodal-decision-model-is-now-available/6286.json)：`pplx-announcements`，2026-10-06T21:11:41.714Z。未核到 v1 独立带时间的首发公告。
[^p-v1]: [v1 官方模型卡，revision `d8c3bd18e31dc1804433d38528b3e770019bf00c`](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b/blob/d8c3bd18e31dc1804433d38528b3e770019bf00c/README.md)。
[^p-code]: [v1 `model.py`：readout 初始化、最后位置及温度](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b/blob/d8c3bd18e31dc1804433d38528b3e770019bf00c/source/src/autojev/model.py)；[v1.1 `model.py`：`enable_noncausal_full_attention`、`DecisionModel.forward/predict`](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/source/src/autojev/model.py)；[v1.1 配置：`noncausal_full_attention`、`last`、temperature](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/decision_config.json)。
[^p-training]: [v1 训练配置：73,000 行、温度](https://huggingface.co/perplexity-ai/pplx-decider-v1-27b/blob/d8c3bd18e31dc1804433d38528b3e770019bf00c/source/configs/training.json)；[v1.1 数据 manifest：626,033 行及来源计数](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/training/data-manifest.json)；[训练配置：`initial_checkpoint:null`、attention、pooling、temperature split](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/training/config.json)。manifest 的 `complete:true` 与 `filter_complete:false` 并存；不据此断言所有过滤流程完整公开。
[^p-card]: [v1.1 模型卡：Decision Index 分项、readout、像素上限及训练分布限制](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/README.md)。[提交史](https://huggingface.co/api/models/perplexity-ai/pplx-decider-v1.1-27b/commits/main)显示像素更新为 2026-10-09T13:26:36Z。卡仍写 private，但[当前官方 Hub 元数据](https://huggingface.co/api/models/perplexity-ai/pplx-decider-v1.1-27b)为 `private:false/gated:false`；本次只读取文档和代码，没有实测权重下载权限。
[^s-dates]: HF 官方提交史：[v19](https://huggingface.co/api/models/StrandsAgents/strands-decider-2B-hobson-v19/commits/main)，导出 `c5530854cd1f48341a29d1e2140736034b430a05`，2026-09-30T15:48:47Z，release 卡更新 2026-10-01T01:08:18Z；[v21](https://huggingface.co/api/models/StrandsAgents/strands-decider-2B-hobson-v21/commits/main)，`2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5`，2026-10-05T16:03:52Z；[Qwen v1](https://huggingface.co/api/models/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/commits/main)，release `603a3b72652e606a20f249c6d662ff07956b5a81`，2026-10-09T20:43:55Z。[公开代码 initial commit](https://api.github.com/repos/strands-labs/strands-decider/commits/fe39fad9ec8ec355993f76b977348d915ec9cb65)：author 2026-10-01T12:39:11Z，committer 14:44:49Z；不能用 author 时间当公开发布时间。
[^s-ledger]: [官方公开研究账本：追述日期、slot→pointer、数据/teacher 负结果](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/research/generations.md)。只读取公开叙述，没有访问其所称私有开发史。
[^s-code]: [固定代码 `modeling.py`：`SlotHead` 204–225、`PointerHead` 228–262、Gemma torso 加载路径](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/src/strands_decider/modeling.py#L228)。
[^s-arch]: [官方 Architecture：候选末 token、因果注意力、LoRA、顺序敏感和 loss](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/docs/architecture.md)。该页仍主要叙述 v19 recipe；最新 v1 训练以模型卡为准。
[^s-v19]: [v19 发布文件模型卡，revision `bb282d786bc251fd4e3068de3ada9ddbb38127cd`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19/blob/bb282d786bc251fd4e3068de3ada9ddbb38127cd/README.md)。本笔记三版本数字取发布文件的 4096 测量；没有混用旧开发实例 3072 的 167/.342/.052 或原实例 4096 的 168。
[^s-v21]: [v21 模型卡：配方、代表 seed、同主机控制、发布核验](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v21/blob/2b52a6235c1b8306bbfa30b00b9d4b74b63a39f5/README.md)；[固定 Results：v21 明确去掉 catch-all 与 instruction flips](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/evaluation/results.md#v21-the-previous-release)。
[^s-v1]: [Qwen v1 官方完整模型卡，revision `216d1ccb46a9e9b036eb9187b248db98480f77d4`](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/README.md)：命名、246,678 行、teacher/anchor、三 seed soup、模型/训练可复现边界、schema、缺图测量。rule application 的 Claude 仅用于独立过滤，模型卡说其文本未进入数据；未推断为 Claude 蒸馏。
[^s-gemma]: 四个官方完整卡均读取：[E2B `d383ecf`](https://huggingface.co/StrandsAgents/strands-decider-E2B-gemma4-v1-2610/blob/d383ecf8534b13d49b3b567a1b66f88fca6d50ba/README.md)、[E4B `1344d44`](https://huggingface.co/StrandsAgents/strands-decider-E4B-gemma4-v1-2610/blob/1344d44397c457f3be34b8ff15c593b41248c6bf/README.md)、[12B `1202634`](https://huggingface.co/StrandsAgents/strands-decider-12B-gemma4-v1-2610/blob/1202634aec54c890c3e11b8747db62195f116006/README.md)、[26B-A4B `3ca5dfe`](https://huggingface.co/StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610/blob/3ca5dfe2d5aa758ae162b6249729181f1d13ed65/README.md)；[固定仓库 README：135,602 行、4B teacher、文本限定](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/README.md#gemma-4-models)。不以不同配方的五个 v1 名字推断等同训练。
[^s-package]: [PyPI 官方元数据](https://pypi.org/pypi/strands-decider/json)：0.1.0 首件上传 2026-10-01T16:05:19.573677Z，即上海 10-02 00:05:19；[固定 README：Gemma 支持尚未进入 PyPI](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/README.md#gemma-4-models)；[Qwen v1 卡：视觉需 main/0.1.0 后版本](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/README.md)。
[^s-calibration]: [固定 Results，Calibration 及分布重拟合限制](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/evaluation/results.md#calibration)；[固定 Evaluation 的 Limitations：温度与阈值适用分布](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/evaluation/README.md#limitations)。
[^o-log]: [OpenAI API Changelog](https://developers.openai.com/api/docs/changelog)：2026-09-22 发布 Luna 文本生成能力；2026-10-06 发布 Decisions beta。源只给日历日期，未给发布时刻/时区。
[^o-guide]: [OpenAI Decisions 当前官方指南](https://developers.openai.com/api/docs/guides/decisions)：public beta、唯一 `gpt-6-luna`、类型、图文、refusal、不同 API 分工、业务标注阈值及速度宣传。
[^o-model]: [GPT-6 Luna 官方模型页](https://developers.openai.com/api/docs/models/gpt-6-luna)：模型 ID 和快照列表均仅列 `gpt-6-luna`；其其他端点的文本生成属性不能当作 Decisions 端点架构披露。
