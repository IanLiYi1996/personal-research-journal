# DetectorScope：当前主论文的独立证据核查

核查日期：2026-10-28。核心论文：**Building Auditable AI Text Detectors with Sparse Autoencoders**，arXiv:2610.30006v1 / EMNLP 2026。本文只核查原始论文、官方出版元数据、作者 README 和公开 HF 模型卡的文献/结果声明，不进行源码审计，不执行实验，不下载模型，不调用付费 API，不提交或推送。已完整阅读 academic-search 的 SKILL.md、metadata-schema、计算机学科 profile 和 arXiv 站点指引；不查询引用数。

## 0. 来源、实读方式与页码

一手来源：

- **A-META**：https://arxiv.org/abs/2610.30006 ，版本入口 https://arxiv.org/abs/2610.30006v1 。
- **A-API**：https://export.arxiv.org/api/query?id_list=2610.30006,2510.02798 。
- **A-PDF**：https://arxiv.org/pdf/2610.30006v1 。
- **A-BIB**：https://arxiv.org/bibtex/2610.30006 。
- **F-META**：https://aclanthology.org/2026.emnlp-main.762/ 。
- **F-PDF**：https://aclanthology.org/2026.emnlp-main.762.pdf 。
- **F-BIB**：https://aclanthology.org/2026.emnlp-main.762.bib 。
- **README**：https://github.com/google-deepmind/detectorscope/blob/52ecc87775154b35d4550911de22faad4fc68f40/README.md 。研究阶段实际读取该版本的 Results、Models、默认设置及 Citation；只读确认 HEAD 为 `52ecc87775154b35d4550911de22faad4fc68f40`，提交日期 2026-10-24，标题 Update citation。完整源码在原始调研工作区；此笔记只携带离线采样检查必需的源文件和许可。
- **HF-CARD**：https://huggingface.co/liamdugan/detector-scope ，原始模型卡 https://huggingface.co/liamdugan/detector-scope/raw/main/README.md 。于2026-10-28实际读取公开 `main` 模型卡的默认head与Performance声明；未下载任何模型权重或重新评估。

两份主论文 PDF 均已实际获取并提取正文、表格与附录进行阅读；PDF 仅流式/内存处理，未另外保存文件。正式版本共 50 页，印刷页码 17640–17689；arXiv 共 50 页，页码 1–50。下文统一使用 **PDF 第 p 页 / 正式印刷页码**，换算为 `17639 + p`。例如表1在 PDF p.5 / 17644，表12在 p.19 / 17658，表23、24同在 p.32 / 17671，表25在 p.40 / 17679。

为明确本次读取的快照：

| PDF | 字节数 | SHA-256 |
|---|---:|---|
| A-PDF | 1,837,258 | `9adcadf99e731b240a9bfe96ad24242acbaa2fd9b4fcfadf0541006543f4a2fd` |
| F-PDF | 2,049,036 | `426e3c20b85f56627912e82731559e6f9d9656546768ddaf60652469db6702d8` |

对两份 PDF 的逐页文本进行了去除页码、arXiv 边栏、会议页脚和空白后的比较。差异主要是作者拼写、公式字符提取、脚注与图内标注的读取顺序；**未发现本文引用的实验表数值在两份版本之间发生变化**。这不是二进制相同声明，也不是不同论文阶段之间的比较。

## 1. 论文身份、年份与所谓“前身”的证据边界

### 1.1 当前项目阶段：已确认

| 字段 | arXiv 版本 | 正式版本 |
|---|---|---|
| 标题 | Building Auditable AI Text Detectors with Sparse Autoencoders | 同左 |
| 年份 | **2026** | **2026** |
| 日期 | v1 提交于 2026-10-21；API 时间为 16:41:14 UTC | 官方元数据为 October 2026；PDF 会议页脚为 October 24–29, 2026；独立发表日 `null` |
| 版本/文献类型 | 2610.30006v1，preprint；本次官方历史只列 v1 | EMNLP 2026，conference，Anthology ID 2026.emnlp-main.762 |
| 页码 | 1–50 | 17640–17689 |
| DOI | `null` | `null`；官方页面显示待注册，官方 BibTeX 未提供 DOI，不补造 DOI |
| ISBN | `null` | 979-8-89176-459-0 |
| 作者 | Liam Dugan；Callum McDougall；Matthew Denton；Philip Pham；Christine Kaeser-Chen；Chris Callison-Burch | ACL 页面/BibTeX 为 Liam Dugan；Callum Stuart McDougall；Matthew Denton；Philip Pham；Christine Kaeser-Chen；Chris Callison-Burch |

正式 PDF p.1 / 17640 将第四作者写为 **Phillip Pham**，而 arXiv 首页、arXiv API、ACL 官方页面和官方 BibTeX 写为 **Philip Pham**。下方元数据按平台元数据保留 Philip，同时记录 PDF 拼写差异，不静默修改原文。两份 PDF 均把 Dugan、Callison-Burch 标为 University of Pennsylvania，其余作者标为 Google DeepMind；Dugan 注记工作在 Google DeepMind Student Researcher 期间完成。

来源 URL：A-META、A-API、A-BIB、F-META、F-BIB，以及 https://aclanthology.org/2026.emnlp-main.762.pdf#page=1 。

### 1.2 旧入口/声称的前身：不能确认其身份，不混用指标

主代理初始检索线索及委派任务说明中的标题为 Interpretable AI-generated Text Detection，链接为 arXiv:2510.02798 / EACL 2026.102。用户原始输入仅有 GitHub URL；前述旧标题和旧链接源自主代理，不应将检索错误归因于用户。**本次官方来源实际返回的内容不支持这两条链接属于该标题或 DetectorScope 的前身：**

| 原入口 | 本次官方证据 | 与当前主论文的关系 |
|---|---|---|
| https://arxiv.org/abs/2510.02798 | 标题 OptunaHub: A Platform for Black-Box Optimization；作者 Yoshihiko Ozaki、Shuhei Watanabe、Toshihiko Yanase；首次提交 2025-10-03，当前 v2 更新 2026-04-20 | `null`，不能认定为 DetectorScope 前身 |
| https://arxiv.org/pdf/2510.02798v1 | 已实际读取 p.1；标题同为 OptunaHub，页边明确标出 3 Oct 2025 | 不是“只读最新版造成改名”的证据 |
| https://aclanthology.org/2026.eacl-long.102/ | 官方标题 Computational Benchmarks for Egyptian Arabic Child Directed Speech | `null`，不能认定为 DetectorScope 前身 |
| https://aclanthology.org/2026.eacl-long.102.bib | Salam Khalifa、Abed Qaddoumi、Nizar Habash、Owen Rambow；EACL 2026；2296–2307；DOI 10.18653/v1/2026.eacl-long.102 | 与当前主论文标题、作者、任务不一致 |

精确标题 arXiv API 检索未取得所声称前身的记录；这一检索结果**不证明论文不存在**。当前 README 的 Citation 只指向 2610.30006 / EMNLP 2026。因此，将“旧标题/旧链接阶段”单独列为**未核实的前身声明**，其作者、准确年份、正式入口、实验差异、与当前论文的版本关系均为 `null`；不能写成“2025 arXiv → 2026 EACL → 2026 EMNLP 的已确认连续版本链”，也不能将旧缓存中的指标归给当前论文。

补充来源 URL：https://export.arxiv.org/api/query?search_query=ti:%22Interpretable%20AI-generated%20Text%20Detection%22&max_results=5 。

## 2. 方法和模型设置：论文实际做了什么

### 2.1 管线

提取 Gemma-2-2B 的 residual stream，取第 **7、13、17、22** 层；各层通过 SAE 编码，沿 token 维度 **max pooling** 后拼接为文档特征。特征描述通过自动解释生成，解释评分不达阈值的特征被屏蔽，再训练 L1 正则 logistic regression。方法式写为 `m_i = I[s_i ≥ τ]`，各表条件标签则写 `τ > 0.7`；本报告按表标签转录，不擅自统一其边界。

这是**在学得的概念表征上训练一个新分类头**，不是对原神经分类器每次判断提供一个独立、完全等价的自然语言证明。logistic head 的分数可按特征贡献加总；原黑盒分类器与新头的等价程度需要看实际指标。

来源：§4.1，PDF pp.3–4 / 17642–17643，图2、方法式与脚注；https://aclanthology.org/2026.emnlp-main.762.pdf#page=3 。

### 2.2 三种设置不能混称“微调检测器”

| 设置 | Gemma-2-2B | GemmaScope SAE | 下游分类头 |
|---|---|---|---|
| BASE-SAE | 未针对检测任务微调的基座模型 | 直接使用预训练 SAE | 检测标签训练的 logistic regression |
| FT-SAE | 同上，基座模型保持不变 | 用检测任务语料重建基座激活，微调 SAE | 同上 |
| FT-Model+SAE | 先针对 AI 检测微调 Gemma | 再重建微调模型激活，微调 SAE | 同上 |

BASE-SAE 不是完全零训练检测器：其分类头仍使用检测任务标签。SAE 为 JumpReLU、每层宽度 **16,384**、GemmaScope `gemma-scope-2b-pt-res-canonical` 的 `width_16k/canonical` 检查点；四层名义总特征数为 65,536。附录列出五组 SAE 训练设置：COLING base/open，PRDB base/GPT-4o/Llama；PRDB SAE 重建语料包含 human、GPT-4o 和 Llama，不按目标生成模型单独过滤。

来源：§5.3，PDF pp.5–6 / 17644–17645；附录 A.1，p.14 / 17653，表5；https://aclanthology.org/2026.emnlp-main.762.pdf#page=14 。

### 2.3 训练细节与对照

- 神经基线：Gemma-2-2B 换二分类头，1 epoch、batch size 4、学习率 `5e-5`、warmup 500、weight decay 0.01，官方 dev 作验证，按最低验证损失选模型。表9对照的 RoBERTa F1 为 dev/devtest/test = 96.2/83.1/74.9，Gemma 为 98.0/86.8/75.2。
- SAE：4 epochs、batch size 4096、学习率 `7e-5`、稀疏系数 `6e-4`；Adam 的 β1=0、β2=0.999；重建损失加稀疏项，不增加任务监督辅助损失，也不重新采样 dead latents。
- logistic regression：SGD+momentum 0.9，3 epochs、batch size 1024；SAE head 学习率 0.03、L1 0.1，gram2vec L1 0.01。SAE max 激活按训练集 RMS 缩放、无中心化；gram2vec 做 z-score。L1 用 soft-threshold 更新形成精确零权重。调参先针对每文档约30–100个活跃贡献特征的可读性目标，再选择该范围内较好的 COLING-dev F1；不是纯按最高准确率选稀疏强度。
- 语言学基线 gram2vec 有930余项特征，覆盖标点、词性、形态、依存、功能词、emoji、过渡词。另有 Binoculars、Fast-DetectGPT、ReMoDetect、SpecDetect，论文明确它们未用 COLING train 训练，按 off-the-shelf 方式比较。
- XGBoost 仅为另一分类头对照，不等同于线性头的逐特征相加解释。表8写 `objective=reg:squarederror`、150 trees、depth 6、learning rate 0.1。表22显示 gram2vec 的 COLING-dev F1 从77.9提高至87.1，故“语言学特征弱”不应只引用较弱的线性基线而忽略该对照。

来源：§5.2；附录 A、B，PDF pp.14–18 / 17653–17657，表5、7、8、9、11；表22 p.31 / 17670。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=16 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=31 。

## 3. 数据、生成模型与 ID/OOD 口径

以下数量取自**当前论文表24**，不能取 README 的概括替代：

| 数据/划分 | 文档数 | 表24生成模型数 | 用途及覆盖 |
|---|---:|---:|---|
| COLING train | 610,767 | 39 | English；检测器/分类头/SAE训练 |
| COLING dev | 261,758 | 39 | 同一 pool 的 stratified split；ID验证/评估 |
| COLING devtest | 32,557 | 5 | 作者标作 OOD |
| COLING test | 73,941 | 14 | 作者标作 OOD |
| RAID test | 56,000 | 11 | **仅非对抗部分**；作者标作 OOD |
| PRDB train | 319,239 | 2 | human、GPT-4o、Llama 3.1 70B 的汇总；两个二分类任务分别训练 |
| PRDB test | 171,837 | 5 | 合并 calibration 与 test 的评估集合，含同源与跨生成模型测试 |

PRDB 的人类语料说明为 **106,413** 条 NeurIPS/ICLR 2016–2024 peer reviews；不要把这一基础人类样本数、train汇总数、单个二分类器的训练数、合并评估数视为同一数量。单个 GPT-4o/Llama 分类器的精确训练样本数，本文未给出可直接转录值，记 `null`。

生成模型与领域：

- COLING train/dev：表24列出 GPT-3.5、GPT-3/davinci、OPT、Llama、T0/BLOOM、FLAN-T5、Cohere、Gemma、GPT-4/4o、Mistral、Dolly、GPT-J/NeoX、GLM 等模型族；pool 来自 MAGE、M4GT-Bench、HC3。覆盖 Q&A、Reddit、stories、WikiHow、arXiv abstracts、Wikipedia、student essays、Yelp、news、commonsense、CMV、scientific tables、peer reviews。
- COLING devtest：GPT-4/4o、Gemini及humanized Gemini、Mistral及humanized Mistral。COLING test 列 GPT-4/4o、Llama、GPT-4/Llama-2 mixed、GLM、GPT-3.5、Qwen、Baichuan、Dolly、StableLM；含 IELTS、改写/润色等 editing、peer reviews、mixed human–AI 文本等。
- RAID：MPT/MPT-chat、Mistral/Mistral-chat、Cohere/Cohere-chat、GPT-2、Llama 2 70B、GPT-3/davinci、ChatGPT、GPT-4；领域包括 arXiv、book/IMDb reviews、news、poetry、recipes、Reddit、Wikipedia。
- PRDB：训练目标分别为 **GPT-4o** 与 **Llama 3.1 70B**；跨模型评估还包括 **Qwen 2.5 72B、Claude 3.5 Sonnet、Gemini 1.5 Pro**。表2只展示各检测器匹配训练生成器的 ID 结果，不能拿它代替表25的跨生成器结果。

有一个未解决的小口径差异：**README Models 写40 generators；正文 §5.1 写41；表24的 COLING train/dev 写39**。本报告保留三个原始口径，不猜测哪些模型被合并、过滤或重复计数。表24提供模型族清单，不足以补造39个完整 checkpoint ID。

作者在附录 C.2 **明确提醒**：arXiv、Wikipedia、QA 等在训练与评估之间有较大领域重叠，故这些迁移集合可能不完全代表真正 OOD 性能。本文也只支持所列英语/peer-review任务上的证据，中文完整检测评估或任意最新生成器的性能为 `null`；出现“中文标点”SAE特征并不等于有中文基准测试。

来源：§5.1 p.4 / 17643；附录 C pp.16–17 / 17655–17656；表24 p.32 / 17671；表25 p.40 / 17679。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=4 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=17 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=32 ，README。

## 4. 主表性能：保留完整指标，不能只选 F1

### 4.1 开放域：表1

每格为 **AUC / F1 / TPR@FPR=5%**，均按原表百分数转录。ID仅 COLING dev；其余三列为作者定义的 OOD。主表 F1 的阈值协议另见 §4.4。

| 模型/条件 | COLING dev（ID） | COLING devtest（OOD） | COLING test（OOD） | RAID non-adversarial test（OOD） |
|---|---|---|---|---|
| Binoculars† | 75.8 / 73.1 / 56.0 | 88.6 / 81.6 / 67.2 | 86.7 / 81.4 / 59.9 | 85.0 / 87.6 / 79.1 |
| Fast-DetectGPT† | 75.7 / 70.5 / 56.2 | 83.2 / 75.5 / 51.6 | 85.1 / 77.5 / 58.5 | 83.3 / 85.2 / 75.1 |
| ReMoDetect† | 76.5 / 65.1 / 45.6 | 91.8 / 85.2 / 65.2 | 91.6 / 86.7 / 75.0 | 73.2 / 67.8 / 44.1 |
| SpecDetect† | 80.2 / 74.4 / 57.2 | 81.3 / 74.2 / 48.9 | 83.6 / 74.6 / 56.4 | 91.6 / 88.1 / 77.3 |
| gram2vec | 81.9 / 77.9 / 36.2 | 80.1 / 73.1 / 45.3 | 77.7 / 72.0 / 33.6 | 77.5 / 73.5 / 43.6 |
| fine-tuned gemma-2-2b | 99.8 / 98.0 / 98.9 | 92.5 / 86.8 / 77.9 | 85.6 / 75.2 / 53.2 | 93.6 / 89.3 / 74.9 |
| BASE-SAE，无mask | 97.1 / 91.4 / 84.0 | 96.2 / 93.2 / 90.4 | 94.2 / 88.9 / 81.8 | 95.2 / 92.3 / 86.4 |
| FT-SAE，无mask | 97.8 / 92.6 / 86.8 | 96.7 / 93.5 / 91.2 | 95.9 / 92.2 / 89.1 | 95.3 / 91.8 / 85.5 |
| FT-Model+SAE，无mask | 99.5 / 97.5 / 98.1 | 91.0 / 86.6 / 63.5 | 83.1 / 76.1 / 47.8 | 91.7 / 92.5 / 61.5 |
| BASE-SAE，τ>0.7 | 95.2 / 89.4 / 75.5 | 93.8 / 88.6 / 80.5 | 88.2 / 80.9 / 61.3 | 93.0 / 89.8 / 78.6 |
| FT-SAE，τ>0.7 | 96.9 / 91.0 / 81.4 | 96.8 / 94.2 / 92.3 | 95.3 / 92.2 / 89.2 | 94.5 / 91.7 / 84.4 |
| FT-Model+SAE，τ>0.7 | 99.2 / 96.8 / 96.7 | 91.1 / 84.8 / 67.9 | 82.4 / 75.4 / 49.1 | 92.0 / 93.2 / 64.5 |

†：论文明确为未用 COLING train 训练的 off-the-shelf 对照。

可支持的结论：

1. **ID性能保持较好的是 FT-Model+SAE，而开放域OOD明显更强的是 FT-SAE。** τ>0.7 的 COLING-test F1 为92.2，远高于微调神经基线75.2、FT-Model+SAE 75.4；对应 TPR@5 为89.2、53.2、49.1。这三个设置不能合并称为一个“通用可解释检测器”结果。
2. “恢复原检测器表现”至多是接近部分 AUC/F1，**不适用于所有低FPR指标**。例如 RAID：FT-Model+SAE τ>0.7 的F1为93.2，高于神经基线89.3，但TPR@5仅64.5，低于基线74.9，差10.4个百分点。
3. 所列FT-SAE τ>0.7 在三个OOD集合优于表1四个现成对照；但不能据此宣称所有设置都优于所有基线，例如BASE-SAE τ>0.7 的RAID TPR@5=78.6，低于Binoculars 79.1。

来源：表1、图4、§6，PDF pp.5–6 / 17644–17645；https://aclanthology.org/2026.emnlp-main.762.pdf#page=5 。

### 4.2 PRDB 的 ID：表2

表2是匹配训练生成器的 ID test，指标为 **AUC / F1 / TPR@FPR=1%**，不是5%：

- 神经基线两个目标均100 / 100 / 100。
- BASE-SAE 两个目标均100 / 99.8 / 100；FT-SAE、FT-Model+SAE均100 / 99.9 / 100；施加τ>0.7后原表数值不变。
- gram2vec 对GPT-4o为100 / 99.6 / 100，对Llama为100 / 99.4 / 99.7。

作者 §6 明确认为 peer-review 任务**过于容易**，近乎完美结果提示数据 confounds / spurious features；这是审计用途的动机，不是跨场景可靠性保证。

来源：表2、§6，p.6 / 17645；https://aclanthology.org/2026.emnlp-main.762.pdf#page=6 。

### 4.3 PRDB 跨生成模型：表25

以下是τ>0.7的关键跨生成器结果，每格仍为 **AUC / F1 / TPR@FPR=5%**。表25的5%与表2的1%不可混用。

| 训练生成器 / 检测器 | Claude 3.5 Sonnet | Gemini 1.5 Pro |
|---|---|---|
| GPT-4o / 神经基线 | 99.2 / 96.3 / 96.9 | 99.1 / 96.1 / 96.5 |
| GPT-4o / FT-SAE | 99.7 / 98.7 / 99.9 | 99.7 / 98.7 / 99.9 |
| GPT-4o / FT-Model+SAE | 95.7 / 89.2 / 73.9 | 96.2 / 88.8 / 79.8 |
| Llama / 神经基线 | 98.7 / 95.7 / 96.1 | 83.3 / 80.0 / 33.9 |
| Llama / FT-SAE | 99.3 / 96.7 / 98.2 | 98.7 / 95.1 / 94.0 |
| Llama / FT-Model+SAE | 90.8 / 84.9 / 43.1 | 82.8 / 77.6 / 30.0 |

对Llama训练的FT-Model+SAE，Gemini的TPR@5仅30.0，即使同源Llama ID接近100。该结果直接反驳“PRDB检测几乎完美，所以跨生成器也几乎完美”的概括。表25还列出了Qwen 2.5 72B和GPT-4o↔Llama互测；这里不把同领域跨生成器等同于跨领域测试。

来源：表25，p.40 / 17679；https://aclanthology.org/2026.emnlp-main.762.pdf#page=40 。

### 4.4 F1 阈值、ROC 工作点、发布和攻击实验必须分开

| 证据对象 | 论文/作者材料明确的内容 | 未能由论文确认的内容 |
|---|---|---|
| 表1、表2、表25的F1 | 各表报告F1；论文写明训练/验证数据和模型选择 | 主表是否按测试split、逐generator选Youden阈值 `null`；是否使用统一dev固定阈值 `null`；F1 averaging方式 `null` |
| 表1、表25的TPR@5；表2的TPR@1 | ROC指标的目标FPR分别为5%和1% | 发布时固定threshold是否达到该FPR `null`；不能将ROC工作点的TPR写成发布阈值的TPR |
| 改写实验 E.2、表23 | 每个检测器用**development-set evaluation 的Youden threshold**；**攻击前固定**；作者检查其重现报告的F1 | 此处协议是否也用于主表全部generator/数据split `null` |
| README 的发布声明 | 默认logistic head、τ=0.85，decision threshold fixed on development split | 实际发布权重/阈值是否落实、与主表评估一致性 `null`，由源码主代理核查 |
| HF通用模型卡的发布声明 | 默认logistic head、解释分数至少0.85；decision threshold固定于dev；COLING test F1=0.751 | 实际权重与模型卡一致性、与论文τ>0.7主表的可比性 `null`；详见§4.5 |

主代理源码核查线索指出“当前 evaluate 在传入split上逐generator选Youden并计算F1”，**本核查没有读该源码，不将其转述为已经由论文证明的实验协议**。论文主表没有足以确认该选择的阈值说明；也不能反向用README或HF模型卡的dev固定阈值声明给主表补上协议。因此本报告中的主表F1只称“论文表报F1”，不称“上线固定阈值F1”，同样不擅自确定为“测试集最优阈值F1”。

来源：主表 p.5、6、40；附录E.2 pp.19–20 / 17658–17659；README Usage、Models；HF-CARD Use、Performance。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=19 ，README，https://huggingface.co/liamdugan/detector-scope 。

### 4.5 公开HF通用模型卡：默认τ=0.85的75.1%不能替代主表92.2%

公开 `liamdugan/detector-scope` 的 `main` 模型卡将模型描述为**未微调的Gemma-2-2B + 在检测语料上微调的SAE + sparse logistic head**，对应其声称的FT-SAE设置；默认解释分数筛选为 **至少0.85**。Performance明确针对默认head，声明decision threshold固定于development data；使用示例还标出logit threshold `+0.50`。这些都是作者的发布文档声明，未在本任务中用权重或运行结果复验。

模型卡报告的完整Performance表：

| 划分 | AUROC | F1 | accuracy | 文档数 |
|---|---:|---:|---:|---:|
| COLING dev | 0.932 | 0.867 | 0.839 | 261,758 |
| COLING dev-test | 0.916 | 0.866 | 0.854 | 32,557 |
| COLING test | 0.857 | **0.751** | 0.770 | 73,941 |

因此，对公开默认模型应准确写成**模型卡报告：τ=0.85、dev固定decision threshold、COLING test F1=75.1%**；论文表1的 **FT-SAE、τ>0.7、COLING test F1=92.2%**是另一阈值/实验报告条件，不能自动当作这个默认发布模型的F1。两数值的表报差为17.1个百分点，但**不是控制其他因素不变的τ消融结果**；差异成因、权重/head是否一致、评估协议是否一致均为 `null`，不将差异全部归因于τ或decision threshold。

**附录表22（PDF p.31 / 17670）没有给出τ=0.85的对应性能行。** 该表仅有无mask与`τ>0.7`，并比较logistic regression和XGBoost；FT-SAE的COLING-test F1为无mask92.2、τ>0.7仍92.2，后者XGBoost为90.5。故表22不能补证默认τ=0.85也达到92.2。图4虽给阈值曲线，但本文不从图上插值制造0.85的精确性能；表12的>0.85是特征计数，表23是受长度筛选且类均衡的改写实验子样本，也都不能替代全量COLING-test默认模型的对应F1。

来源URL：https://huggingface.co/liamdugan/detector-scope ，https://huggingface.co/liamdugan/detector-scope/raw/main/README.md ；论文表1 https://aclanthology.org/2026.emnlp-main.762.pdf#page=5 ，表22 https://aclanthology.org/2026.emnlp-main.762.pdf#page=31 。

## 5. 解释性筛选代价：并非普遍“几乎无损”

### 5.1 性能代价

以**同一设置无mask→τ>0.7**比较，下面变化为本核查按表1直接相减的百分点，不是新实验：

| 设置 | COLING dev F1变化 | COLING devtest F1变化 | COLING test F1变化 | COLING test TPR@5变化 | RAID TPR@5变化 |
|---|---:|---:|---:|---:|---:|
| BASE-SAE | −2.0 | −4.6 | **−8.0** | **−20.5** | −7.8 |
| FT-SAE | −1.6 | +0.7 | 0.0 | +0.1 | −1.1 |
| FT-Model+SAE | −0.7 | −1.8 | −0.7 | +1.3 | +3.0 |

BASE-SAE在COLING test从F1 88.9降至80.9、TPR@5从81.8降至61.3。因此表1 caption 的“不显著降低OOD表现”和README的minimal adverse effects应当加上**具体设置与指标**限定，不能覆盖BASE-SAE的上述损失。FT-SAE的高可解释筛选确实在所列OOD集合保留了较好的表现。

来源：表1 p.5 / 17644；https://aclanthology.org/2026.emnlp-main.762.pdf#page=5 ，README Results。

### 5.2 可用特征与每次预测的解释负担

表12的“Interpretable”列计数和“Avg. # active / doc”如下转录。前者不是分类头实际用到的全部特征数，后者才是测试文档中非零权重且激活、贡献logit的平均特征数。

| 数据 / 设置 | 可解释特征数 τ>0 / >0.7 / >0.85 | 平均活跃贡献特征/文档 τ>0 / >0.7 / >0.85 |
|---|---|---|
| COLING / BASE-SAE | 65,512 / 33,624 / 9,355 | 101 / 107 / 108 |
| COLING / FT-SAE | **66,245** / 43,480 / 13,051 | 100 / 101 / 101 |
| COLING / FT-Model+SAE | 29,489 / 16,893 / 2,983 | 53.7 / 41.9 / 25.6 |
| PRDB / BASE-SAE | 65,153 / 22,332 / 5,544 | 79.6 / 116 / 68.7 |
| PRDB / FT-SAE | 64,997 / 39,231 / 7,967 | 99.3 / 83.7 / 85.1 |
| PRDB / FT-Model+SAE | 17,518 / 11,451 / 1,903 | 78.6 / 49.7 / 33.6 |

PRDB行是GPT-4o目标。全局过滤后各阈值重新训练分类头，因此过滤更严**不保证**每个文档的贡献特征数量单调减少；不能把每次展示top-k卡片误写成整个预测只用k个概念。

**原表异常须保留**：表12的COLING FT-SAE `66,245` 大于表5所述四层×16,384的名义容量65,536。修正值、原因均为 `null`；本报告不改成65,245或其他猜测数字。

来源：表12 p.19 / 17658；表5 p.14 / 17653；https://aclanthology.org/2026.emnlp-main.762.pdf#page=19 。

### 5.3 计算/经济代价：有报告的与没有报告的

作者报告神经基线训练用一张 NVIDIA RTX A6000，共 **80 GPU hours**；SAE训练用 NVIDIA DGX B200 GPUs，各设置合计 **185 GPU hours**。不同硬件的GPU小时不可直接当成统一时间/费用。分类头使用缓存激活可在CPU训练，但端到端推断仍运行Gemma和四层SAE，不是纯CPU字符串规则分类。

自动解释每特征10个生成例、50个验证例，调用Gemini；这是论文报告的流程，不是本核查执行。自动解释总请求数、token开销、人民币/美元费用、严格筛选的新增计算费用、CPU训练总时长、端到端推断延迟、吞吐量与峰值显存均为 `null`。本文不根据单例数或特征总数补算并假定API开销。

来源：附录 A.3、B.1、B.2，PDF pp.15–16 / 17654–17655；§4.2 p.4 / 17643。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=15 。

## 6. 自动解释验证：Spearman、holdout和高亮的明确边界

### 6.1 分数的实际定义

§4.2不是Pearson，也不是token级预测与真实连续激活的Pearson相关。当前实验把文档按**该特征的文档max activation**分为量化分箱，另加inactive箱0；LLM根据描述及文档例子预测其分箱，`s_i` 是**预测分箱与真实分箱之间的 Spearman rank correlation**，范围−1到1。

每特征描述生成使用10个例子、验证使用50个例子。生成模型在正文写 **gemini-3.1-flash-lite**；生成/验证使用terse描述。生成例裁到最高激活token周围±20 tokens，将高于该特征全局最大激活10%的token以`<< >>`高亮。

来源：§4.2 p.4 / 17643；附录 D p.18 / 17657；https://aclanthology.org/2026.emnlp-main.762.pdf#page=4 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=18 。

### 6.2 是否有严格独立的验证集

| 问题 | 论文直接支持的答案 |
|---|---|
| 描述验证是否使用另一次采样/50例而非生成的10例？ | 论文给出两阶段和10/50的数量，但这**不等于**声明集合不重叠 |
| 是否承诺生成与验证文档严格disjoint？ | 未见明确声明；`strict_generation_validation_holdout = null` |
| 是否从独立corpus做解释评分？ | §4.1明确说评分检验解释对**training corpus**的激活忠实性；独立corpus协议 `null` |
| 验证是否排除全部生成例或采用显式交叉验证？ | 排除规则/交叉验证协议 `null` |
| 实际重复文档数、比例？ | `null`；不能根据缺少说明断言已发生重叠 |
| 描述生成/验证的论文随机seed是否为42？ | 该两阶段具体seed `null`；不能把主代理源码核查线索中的当前实现seed写回论文 |
| 是否独立人工验证语言描述或更换judge模型？ | 本论文未给出相应量化结果；结果 `null` |

论文中的“held-out test”“official dev”用于**检测任务评估**以及改写实验抽样，不能转作“描述生成与解释验证严格holdout”的证据。主代理源码核查线索包含当前实现无显式排除、seed42；这是主代理负责的实现层核查结果，本任务不重复源码审计，也不据此反推论文实验实际发生样本重叠。本文仅确认论文没有足够的严格不重叠说明。

来源：§4.1–4.2 pp.3–4；附录 D p.18，图11–13 pp.34–35 / 17673–17674。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=35 。

### 6.3 量化验证仍给真实激活高亮

附录D说验证沿用高亮/裁切过程，只是不再按真实量化箱分组展示；inactive文档随机选择一个token作裁切和高亮位置。图13的验证prompt明确告诉模型 `<< >>` 指向最强激活词，并要求输出分箱整数。因此它衡量的是**有激活位置提示的解释模拟分数**，不是完全不知激活位置、只凭自由文本描述对原文进行独立识别的盲测。

这是方法条件的直接事实；其造成的分数偏高程度、是否使特定解释无效，均为 `null`，不能在没有对照实验时定量推断。相同LLM负责生成和判断、训练语料采样、高亮提示，均要求谨慎使用“全局忠实性”一词；分数本身不保证跨语境的描述完整性，也不能替代下节的干预证据。

来源：附录D p.18 / 17657；图13 p.35 / 17674；https://aclanthology.org/2026.emnlp-main.762.pdf#page=18 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=35 。

## 7. 几类干预：有经验因果证据，但不是无条件因果证明

### 7.1 SAE特征零消融：对新头的精确贡献，不等于文本因果事实

固定其他表征、把一个head输入特征置零，其logit差对线性head是该特征的加权贡献。这支持“新分类器在该表征上用了哪些项”的代数归因；**不能仅由这种归因推出**自然语言描述抓住了全部触发条件、编辑原文必然只移除该概念，或原神经检测器使用同一语义原因。

来源：§4.1、§7.1、表3；方法式与表15–21的logit分解示例。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=7 。

### 7.2 特征级文本改写：直接检验描述与logit效果的一致性

§7.1、附录E.1：每数据集抽500人类+500AI文档；PRDB使用GPT-4o目标。每文档从有权重的活跃、可描述特征中最多抽50个；过稀无法解释的特征、部分document-start artifacts被排除。LLM接收目标概念并改写原文，比较**按特征平均**的改写logit变化与零消融logit变化。

表3（两种mask条件）：

| 数据 / 设置 | Spearman ρ：无mask→τ>0.7 | 符号一致率：无mask→τ>0.7 | 表报改写数：无mask / τ>0.7 |
|---|---|---|---|
| COLING / BASE-SAE | 0.36 → 0.41 | 65.0% → **59.9%** | 16,052 / 19,318 |
| COLING / FT-SAE | 0.52 → 0.54 | 72.9% → 80.8% | 15,968 / 17,826 |
| COLING / FT-Model+SAE | 0.54 → 0.62 | 68.9% → 73.2% | 28,965 / 27,866 |
| PRDB / BASE-SAE | 0.76 → 0.81 | 92.1% → 89.2% | 35,339 / 30,803 |
| PRDB / FT-SAE | 0.74 → 0.80 | 91.3% → 92.4% | 37,037 / 36,720 |
| PRDB / FT-Model+SAE | 0.69 → 0.80 | 93.9% → 99.1% | 40,032 / 37,494 |

这是真实的**文本干预一致性证据**，不只是自动解释方法的名称。证据在PRDB更强，在COLING只有中等相关。更高τ提高各行相关，但**不保证符号一致率均提升**：BASE-SAE在两个数据集都降低。

干预并不完全隔离：以FT-Model+SAE τ>0.7为例，COLING目标/非目标归一化激活变化为0.63/0.27，PRDB为0.74/0.14；说明改写也改变其他特征。作者进一步解释COLING的标点、句边界等很难以保持语义的改写单独移除。

因此不能写为“改写后logit精确等于移除该概念的贡献”。论文得到的是相关和符号一致性，非完整等式验证；人类审核语义保持率、非目标语义完全不变率、未可描述特征的同类检验结果均为 `null`。抽样/筛选也限制了向全部潜在特征推广。

细节：图14和E.1说特征级改写用terse、keywords、long三项描述；不能把附录D“实验只用terse”的概括无条件套在该干预上。表3 PRDB FT-SAE masked ρ为0.80，但图9对应标注0.81，原文内部有轻微不一致；本报告以表3转录，不自行校正。

来源：表3、§7.1 p.7 / 17646；附录E.1 pp.18–19 / 17657–17658；表10 p.18；图9 p.24 / 17663；图14 p.36 / 17675。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=7 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=18 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=36 。

### 7.3 文档级解释引导攻击：回避能力与高召回线索，不证明解释完备

实际协议以附录E.2为准：

- 攻击**τ>0.85 logistic heads**，不是主表τ>0.7条件。
- 每个数据/目标设置抽2,500 AI文档+2,500固定人类文档；只改写AI，故固定人类侧FPR。样本限定30–900 Gemma tokens，适应1,024-token特征提取窗口。
- Baseline、Few Shot、Guided为一般改写；SAE只列每文档最大的**10个AI方向+10个人类方向**贡献特征、描述/关键词、激活位置、max-pooling计数规则。SAE×3每次重新提取证据，最多三轮，首次翻成human后冻结。
- decision threshold 由dev Youden选取且攻击前固定。ASR分母是**攻击前已被检出的AI文档**，不直接等于全部AI文档翻转比例。Lev是**word-level** Levenshtein/较长文本长度，不是token级编辑率。
- 生成温度1.0、top_p0.95、top_k64；正文/E写gemini-3.1-flash-lite，图15caption写gemini-3.1-flash-lite-preview，精确API快照 `null`。失败0–6个/条件被排除。作者做1,000-sample bootstrap，但表23不直接列置信区间。

表23关键数值均为百分数：

| 数据 / 设置 | 原F1 | Baseline ASR | 单轮SAE ASR | SAE×3 ASR | SAE×3 Lev（词级） | SAE×3后F1 |
|---|---:|---:|---:|---:|---:|---:|
| COLING / BASE-SAE | 80.3 | 5.3 | 28.1 | 55.1 | 34.5 | 46.9 |
| COLING / FT-SAE | 84.4 | 6.3 | 26.9 | 52.6 | 37.5 | 51.7 |
| COLING / FT-Model+SAE | 88.3 | 8.7 | 34.2 | 63.2 | 33.7 | 45.3 |
| PRDB GPT-4o / BASE-SAE | 99.7 | 0.1 | 16.2 | 69.1 | 11.2 | 47.0 |
| PRDB GPT-4o / FT-SAE | 99.8 | 0.0 | 3.6 | **19.7** | 16.5 | 88.9 |
| PRDB GPT-4o / FT-Model+SAE | 99.8 | 0.0 | 38.1 | 83.0 | 11.3 | 29.0 |
| PRDB Llama / BASE-SAE | 99.1 | 0.6 | 23.0 | 59.7 | 17.8 | 56.7 |
| PRDB Llama / FT-SAE | 99.5 | 0.1 | 3.4 | **21.8** | 23.3 | 87.2 |
| PRDB Llama / FT-Model+SAE | 99.1 | 0.6 | 62.5 | 97.1 | 11.6 | 5.5 |

可确认：解释引导通常更容易造成固定阈值翻转；尤其FT-Model+SAE PRDB Llama三轮ASR97.1%，并非对抗鲁棒性保证。不能照抄正文 p.8“所有条件60–100%”：表23的COLING FT-SAE为52.6%、PRDB FT-SAE为19.7%/21.8%。附录E.2也明确承认FT-SAE/PRDB是例外，其证据分散，十项提示不能覆盖。

这种实验只说明用部分已知特征、可能添加human证据，能越过一个固定阈值；**不要求所有AI特征被移除**，也不证明解释集合完整、高精度或跨场景有效。论文自己说不是definitive proof。针对相同LLM改写器、给出的特征清单和预算的结果，不能推广为对任意人工/自动改写攻击的鲁棒性结论。

来源：§7.2 p.8 / 17647；附录E.2 pp.19–20 / 17658–17659；表23 p.32 / 17671；图15–19 pp.36–40 / 17675–17679。URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=8 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=20 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=32 。

## 8. 作者明确局限，与本次审计指出的缺失分列

### 8.1 作者在 Limitations 明确写出的两点

1. **解释偏向高召回、低精度，可能漏掉上下文依赖。** 例如某feature只在“Hello Tom”里的Tom激活，但语料没有其他Tom语境，描述可能只说Tom；同一语料上的验证仍可给满分。作者的非正式regex检查发现约半数特征描述达到99.5+ F1，其余存在不同程度召回偏置。该检查的精确subset大小、完整逐特征precision/recall、独立人工真值均为 `null`，不能当成全特征人工验证。作者建议hard negatives和synthetic data。
2. **只在GemmaScope / Gemma-2-2B这一基座体系上测试。** 未对其他基座训练SAE；Llama、Qwen在PRDB是待检测的生成器，不是证明检测器骨干可无损迁移到Llama/Qwen的实验。

来源：Limitations，PDF pp.9–10 / 17648–17649；https://aclanthology.org/2026.emnlp-main.762.pdf#page=9 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=10 。

### 8.2 作者在正文/附录另行明确承认的边界

- 底层微调检测器泛化失败时，FT-Model+SAE也会失败：§6 p.6 / 17645。
- PRDB过易可能依赖伪相关：§6 p.6；表4 p.8揭示极具体格式线索。例如GPT-4o的加粗Markdown标题后的双冒号feature出现在11个人类文档、54,293个AI文档；Llama的左方括号feature出现在7个人类文档、22,541个AI文档。它们是文档出现数，不是“真/假特征准确率”。
- BASE-SAE开放域高权重线索包含HTML属性、软件license boilerplate等领域捷径；格式线索应否剔除没有统一标准：§8 pp.8–9；表28 p.44 / 17683。
- OOD训练/测试领域重叠：附录C.2 p.17 / 17656。
- 文本改写改变非目标特征；COLING标点/句边界较难单独移除：附录E.1 p.19 / 17658。
- 用特征列表攻击的证据不构成完备性定论，并且PRDB FT-SAE例外：§7.2 p.8、附录E.2 p.20。

来源 URL：https://aclanthology.org/2026.emnlp-main.762.pdf#page=6 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=8 ，https://aclanthology.org/2026.emnlp-main.762.pdf#page=44 。

### 8.3 本次证据审计的未报告项，不冒充作者亲口认定的局限

| 未能由论文确认的项目 | 值/处理 |
|---|---|
| 描述生成与验证严格disjoint、重复样本量 | `null`；不能断言泄漏已发生 |
| 主表F1的逐split/逐generator阈值策略与发布一致性 | `null` |
| 在完全无激活高亮条件下的解释验证对照 | `null` |
| 人工语义保持审查、独立人工解释忠实性评估 | `null` |
| 多次检测器训练的随机seed方差、主表指标置信区间 | `null`；不以攻击实验bootstrap代替 |
| 人类群体/非母语作者公平性与分组FPR | `null` |
| 中文完整基准、任意新生成器/真实混合编辑场景的部署保证 | `null` |
| 自动解释费用、推断延迟、显存、吞吐 | `null` |

这些缺失不等于对应能力必然失败，但不支持“严格独立验证”“解释均为真因果语义”“可以按主表F1保证上线表现”“高风险场景可直接自动定罪”等更强断言。论文贡献是使检测器内部可检查、可干预，同时揭露捷径；不是消除检测误判或证明真实作者来源。

## 9. 需在后续总报告保留的差异/异常清单

| 项目 | 当前原始证据 | 报告处理 |
|---|---|---|
| 旧标题/链接 | 2510.02798v1/v2为OptunaHub；EACL.102为儿童定向语料论文 | 前身关系 `null`，不拼接成版本链 |
| 当前作者拼写 | 正式PDF Phillip；官方元数据/arXiv Philip | 元数据用官方Philip，保留差异 |
| COLING生成模型数 | README 40；§5.1 41；表24 train/dev 39 | 三种口径原样保留 |
| 特征总数 | 表12 FT-SAE COLING τ>0列66,245，名义四层容量65,536 | 疑似计数/排版异常，原因与修正 `null` |
| 干预相关数 | 表3 PRDB FT-SAE masked 0.80，图9为0.81 | 取表3并提示差异 |
| 全条件攻击60–100% | 正文概括与表23的19.7%、21.8%、52.6%不符 | 精确附表优先，不抄宽泛概括 |
| 改写器名称 | 正文/E为flash-lite，图15caption为flash-lite-preview | 精确API快照 `null` |
| 主表F1 vs发布/攻击 | 主表阈值规则不明；E.2明确dev Youden固定；README称发布dev固定 | 不互相补协议，不混称上线F1 |
| HF默认模型vs论文主表 | HF通用模型卡τ=0.85、dev固定阈值报告test F1=0.751；论文FT-SAE τ>0.7为92.2%；表22没有0.85性能行 | 条件分别标注，差异原因 `null`；不能用主表替换默认发布性能 |

本节只是论文/作者材料的证据差异，不判断源码实际行为。seed、验证重叠风险和evaluate阈值实现等主代理源码核查线索均未在本任务中重复核验；旧标题/旧链接则属于主代理初始检索线索，不归因于用户。

## 10. academic-search metadata-schema JSON

以下数组分别保留**同一当前工作的arXiv记录与正式发表记录**，故不按arXiv ID合并掉版本/发表类型差异。当前两条记录年份均为2026；不能将未核实的旧入口年份2025套在当前预印本上。未知或本任务未检索字段为`null`；`citation_count`特意为`null`，未查询。正式日期只知月份，`publication_date`不补造日。没有已确认的当前论文DOI，因此不以其他论文DOI发起OA查询；OA全文状态由两份官方PDF实际访问确认，OA分类未知仍为`null`。arXiv许可与ACL许可分开记录。

```json
[
  {
    "title": "Building Auditable AI Text Detectors with Sparse Autoencoders",
    "authors": [
      "Liam Dugan",
      "Callum McDougall",
      "Matthew Denton",
      "Philip Pham",
      "Christine Kaeser-Chen",
      "Chris Callison-Burch"
    ],
    "year": 2026,
    "publication_date": "2026-10-21",
    "publication_type": "preprint",
    "venue": "arXiv",
    "doi": null,
    "arxiv_id": "2610.30006",
    "pubmed_id": null,
    "pmcid": null,
    "orcid": null,
    "issn": null,
    "isbn": null,
    "cnki_url": null,
    "abstract": null,
    "keywords": null,
    "mesh_terms": null,
    "jel_codes": null,
    "msc_codes": null,
    "acm_ccs": null,
    "study_type": null,
    "sample_size": null,
    "population": "English COLING shared-task texts, non-adversarial RAID test texts, and NeurIPS/ICLR peer reviews from PRDB",
    "citation_count": null,
    "download_count": null,
    "open_access_status": null,
    "license": "arXiv non-exclusive license to distribute 1.0",
    "full_text_status": "open_pdf",
    "pdf_url": "https://arxiv.org/pdf/2610.30006v1",
    "data_availability": "Experiments use the COLING shared task, PRDB and RAID datasets described in the paper; individual dataset access conditions and licenses were not separately audited.",
    "code_url": "https://github.com/google-deepmind/detectorscope",
    "bibtex": "@misc{dugan2026buildingauditableaitext,\n  title={Building Auditable AI Text Detectors with Sparse Autoencoders},\n  author={Liam Dugan and Callum McDougall and Matthew Denton and Philip Pham and Christine Kaeser-Chen and Chris Callison-Burch},\n  year={2026},\n  eprint={2610.30006},\n  archivePrefix={arXiv},\n  primaryClass={cs.CL},\n  url={https://arxiv.org/abs/2610.30006}\n}",
    "source_platforms": ["arxiv"],
    "fetched_at": "2026-10-28"
  },
  {
    "title": "Building Auditable AI Text Detectors with Sparse Autoencoders",
    "authors": [
      "Liam Dugan",
      "Callum Stuart McDougall",
      "Matthew Denton",
      "Philip Pham",
      "Christine Kaeser-Chen",
      "Chris Callison-Burch"
    ],
    "year": 2026,
    "publication_date": null,
    "publication_type": "conference",
    "venue": "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    "doi": null,
    "arxiv_id": "2610.30006",
    "pubmed_id": null,
    "pmcid": null,
    "orcid": null,
    "issn": null,
    "isbn": "979-8-89176-459-0",
    "cnki_url": null,
    "abstract": null,
    "keywords": null,
    "mesh_terms": null,
    "jel_codes": null,
    "msc_codes": null,
    "acm_ccs": null,
    "study_type": null,
    "sample_size": null,
    "population": "English COLING shared-task texts, non-adversarial RAID test texts, and NeurIPS/ICLR peer reviews from PRDB",
    "citation_count": null,
    "download_count": null,
    "open_access_status": null,
    "license": "cc-by-4.0",
    "full_text_status": "open_pdf",
    "pdf_url": "https://aclanthology.org/2026.emnlp-main.762.pdf",
    "data_availability": "Experiments use the COLING shared task, PRDB and RAID datasets described in the paper; individual dataset access conditions and licenses were not separately audited.",
    "code_url": "https://github.com/google-deepmind/detectorscope",
    "bibtex": "@inproceedings{dugan-etal-2026-building,\n  title = \"Building Auditable {AI} Text Detectors with Sparse Autoencoders\",\n  author = \"Dugan, Liam and McDougall, Callum Stuart and Denton, Matthew and Pham, Philip and Kaeser-Chen, Christine and Callison-Burch, Chris\",\n  editor = \"Lee, Hung-Yi and Dev, Sunipa and Li, Jessy and Elliott, Desmond\",\n  booktitle = \"Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing\",\n  month = oct,\n  year = \"2026\",\n  address = \"Budapest, Hungary\",\n  publisher = \"Association for Computational Linguistics\",\n  url = \"https://aclanthology.org/2026.emnlp-main.762/\",\n  pages = \"17640--17689\",\n  ISBN = \"979-8-89176-459-0\"\n}",
    "source_platforms": ["aclanthology"],
    "fetched_at": "2026-10-28"
  }
]
```

许可来源：arXiv页面的rights链接 http://arxiv.org/licenses/nonexclusive-distrib/1.0/ ；ACL官方页面版权说明 https://aclanthology.org/2026.emnlp-main.762/ 。JSON中的作者名遵循平台元数据，PDF的Phillip拼写差异见§1.1。`sample_size=null`表示论文涉及多个不同划分、任务和干预样本，没有一个可无歧义代表整篇工作的单一样本量，具体计数已分别列出。`abstract=null`是本稿不重复再分发完整原始摘要的整理选择，不代表摘要不存在或未读取；主要内容已在上文用中文分析。
