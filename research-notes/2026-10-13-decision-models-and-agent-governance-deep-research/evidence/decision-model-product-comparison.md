# 决策模型产品与背景对比

截至 **2026-10-13 10:26（Asia/Shanghai）** 的官方来源快照，不代表当日结束状态。仅比较公开产品、模型和接口；未运行模型、未访问私有 GitLab、未评估项目适配。外部提供的适用性评估附件不在本笔记范围内，本笔记未读取。

结论：Jev 是文本型托管决策产品；Perplexity Decider 同时有托管接口和可自托管的图文权重；Strands Decider 是开源、自托管的决策模型系列；OpenAI 的 Decisions API 已获官方确认，但不能据此认定其内部也是 Strands 的 pointer 架构。它们的共同点是“对给定问题返回受约束的判断”，不是替代整个 agent。[^t-model][^p-api][^p-card][^s-readme][^o-guide]

## 1. “决策模型”不等于生成模型，也不排斥分类

- **生成式 LLM** 逐 token 生成文本，可写说明、代码或候选动作；Structured Outputs 可以约束生成结果的 JSON schema，但仍是文本生成能力，不因此变成 pointer 决策模型。[^o-structured]
- **受约束决策** 接收状态、问题和允许的答案，返回真假概率、候选分布或量表分数。Strands 明确移除语言模型头，改用 pointer head 比较答案位置与候选位置的表示，不进行自回归解码；候选语义由请求提供。Perplexity v1.1 则公开了 `[255, 5120]` 的独立 readout，不能直接称为同一种 pointer 架构。Jev 和 OpenAI 的内部读出结构在本次来源中未获充分公开验证。[^s-readme][^p-card][^t-system][^o-guide]
- **普通分类器** 也能返回概率。以监督式 LogisticRegression 为例，预测类别对应训练得到的 `classes_`；这里决策模型的区别主要是自然语言问题、运行时候选和通用判断接口，而不是“返回概率”这个特征。概念上它仍可做分类；也不能反向断言所有分类模型都只能使用固定标签。[^classifier][^s-readme][^t-system]

应拆开三个环节：**感知**识别页面/图像中的证据；**候选生成**枚举可执行元素或动作；**选择**在给定候选之间判断。Perplexity 官方浏览器示例由脚本枚举并编号可点击元素、截屏，模型回答六个问题，最后由代码比较概率并执行动作。由此可见，支持视觉选择不等于自动发现所有候选、更不等于直接控制浏览器；候选遗漏时，选择层无法凭空补齐。[^p-browser]

## 2. 当前产品形态

| 产品 / 已核实版本 | 文本与视觉 | 部署、背景与边界 |
| --- | --- | --- |
| TypeSafe Jev：`jev-1.13.0`；`jev-latest`、`jev-preview` 当前均指向它 | 文本；可接收字符串、JSON 对象或文本数组，**不接收图像/音频/视频** | 官方托管 System One 产品，训练目标为校准决策。公开权重、自托管交付条件及参数量：**未知**；不把企业 FAQ 的问题标题当作已承诺能力。[^t-model][^t-system][^t-enterprise] |
| Perplexity：`pplx-decider-v1.1-27b`；托管也接受旧 `v1-27b` | 图文；v1.1 基于 Qwen3.8-27B，原生 artifact 是 backbone 加独立决策 readout | 官方托管 API；HF 官方权重标注 Apache-2.0，可按参考实现自托管。模型卡建议 CUDA，权重约 49 GiB，另需工作内存；本次未验证 CPU 推理表现。[^p-api][^p-card] |
| OpenAI：`gpt-6-luna` 的 Decisions API | 文本、图像，返回 typed answers | **Public beta**；官方称当前只有该模型受支持。可验证的是托管接口，不是新发布的开源 pointer 权重。同名 Luna 也能通过 Responses/Chat Completions 生成文本，须区分端点。[^o-guide][^o-log] |
| Strands Decider：Qwen3.5-2B 与四种 Gemma4 的 `v1-2610` | Qwen3.5 可在启用 `--vision` 时读取图像；**Gemma4 Decider 当前只读取文本** | Apache-2.0 开源、自托管；pointer head、CLI 和 HTTP 服务。不能从底座多模态能力直接推断每个 Decider 都支持视觉。[^s-readme][^s-vision] |

Perplexity 的访问说明存在冲突：模型卡仍写“private repository / authenticated access”，但检索时 Hub 元数据为 `private:false`、`gated:false`。记录为当前公开元数据，不推断所有权重下载、账号权限或未来状态已经实测。[^p-card][^p-meta]

## 3. 名称相近，不代表 wire/API 可直接互换

- **Jev**：`POST https://api.typesafe.ai/v1/systemone`，Bearer 认证；`model + state + questions`，其中 `questions`/`answers` 都是按名称索引的对象。类型为 `noul / choice / score`，候选通过 `criteria` 描述。[^t-api]
- **Perplexity 托管**：`POST https://api.perplexity.ai/v1/decisions`，Bearer 认证；同样采用 `state + questions` 对象和 `noul / choice / score`。图像放在 `state` 数组的 `image_url` 部件中；未知字段返回 `400`。其本地参考实现却是 `DecisionModel.predict([row])`，使用单数 `row["question"]` 和 `row["images"]`，不是原样接收托管 HTTP 请求。[^p-api][^p-card]
- **Strands 本地**：`POST /v1/systemone`；文本请求形状与 Jev 相近，但图像是顶层 `images` 的 base64 列表，需要 vision 模式。未启用时拒绝图像请求；默认服务绑定本机且无认证，不能把演示服务等同生产托管服务。[^s-readme][^s-vision][^s-inference]
- **OpenAI**：`POST https://api.openai.com/v1/decisions`；使用 `input`，`questions`/`answers` 为数组，以 `name` 对应；类型为 `predicate / choice / score`，候选用 `choices`，量表用 `levels`。真假结果字段为 `probability`，不是 `noul`；还可能返回 `refusal`。图像通过 user message 的 `input_image` 部件提交。**与 Perplexity 同名端点，不是同一协议。**[^o-guide][^o-ref]

因此“兼容”只能逐字段核对，不能只换 base URL。另一个边界是部署差异：Perplexity 托管 schema 给出的输入上限为 262,144 tokens，而 HF 参考实现 `prepare` 限制为 8,192 tokens；前者的请求许可不证明后者有相同实现或长上下文质量。[^p-api][^p-card]

`confidence` 也不应直接解释成正确率。例如 TypeSafe Choice 的 confidence 是由候选概率分布计算的统计量，不等于最高候选概率；校准是跨预测群体的性质，不保证某个答案正确。阈值必须在具体任务上验证。[^t-confidence][^t-system]

## 4. 本地 CPU/GPU 延迟应怎样读

不生成文本省去了逐 token 解码循环，**没有省掉输入编码、视觉处理和设备计算**。长状态、大图、多候选、缓存命中、批处理和并发排队仍会影响耗时。以下仅为官方记录，不是本次复测：[^s-readme][^s-inference][^p-card]

- Strands **v19 / M3 Pro / MPS**，少于 300 tokens 的问题：暖请求中位数 153 ms，同长度首请求 310 ms；整个 JevBench 的暖请求 P95 为 2,628 ms。短输入中位数不是交互延迟上界，也不是当前 v1 在所有机器上的数字。[^s-inference]
- Strands **Gemma4 E2B** 的 H100 中位数为 102 ms；CPU 记录为 4.9 s，但条件是每模型 16 线程、四个模型同时运行在一台 64-vCPU 主机。不能把这组数值外推为任意 CPU/GPU 的倍率。[^s-inference]
- 托管往返包含网络、排队和服务端处理；本地 forward、HTTP 往返和完整 agent 一步不是同一个指标。OpenAI 官方“约 10× 快于 Responses”的陈述不构成跨产品速度排名；Perplexity 模型卡的显存需求也不是延迟数据。[^o-guide][^p-card]

不搬用 Decision Index/JevBench 排名推导中文 UI、无障碍树或真实工作流效果；不据不同硬件、版本、输入长度的结果判定谁更快。

## 5. Decider 相对“10 月 12 日使用的基线”的更新

- 指定基线 `0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534` 的实际 commit 时间是 **2026-10-09 23:33:05Z**，并非 10 月 12 日提交。当前 `main` 为 `b9fc83626bcca822d00f822d6c2e151b8d244ee9`，官方 compare 显示仅新增 **1 个提交**，无落后提交。[^s-compare]
- 新提交为 **2026-10-12 15:45:18Z（上海 23:45:18）** 的 `fix(serve)`：用锁串行化 engine 调用，修复重叠请求导致的 MPS 崩溃，并增加并发测试；比较结果只改动 server 与测试，**不是新增模型或准确率升级**。串行服务意味着并发请求需要轮流进入 engine。[^s-commit][^s-compare]
- **发包与代码分开**：PyPI 最新仍为 `0.1.0`，上传于 2026-10-01；GitHub releases 和 tags 查询均为空。官方文档明确 Gemma4 支持尚未进入该 PyPI 包，Qwen 视觉路径也在 `main`/后续版本而非 `0.1.0`。不能以普通包名安装就假定已获得最新能力。[^s-pypi][^s-releases][^s-tags][^s-inference][^s-card]
- **模型权重分开**：官方 HF 主组织仍列出旧 `hobson-v19/v21` 和五个 `v1-2610`：Qwen3.5 `2B`，Gemma4 `E2B / E4B / 12B / 26B-A4B`；五个 v1 仓库的 `lastModified` 均为 2026-10-09 UTC，本次未见晚于基线的新 v1 权重更新。Qwen v1 当前 revision 为 `216d1ccb46a9e9b036eb9187b248db98480f77d4`；查询 refs 未发现 tags，应使用已核实的 commit revision 固定快照，而不假设文档提及的日期标签一定存在。[^s-models][^s-refs]

### 来源时间与未知项

所有来源均于上述检索日读取。TypeSafe Models 的 sitemap `lastmod` 为 **2026-10-12T23:58:50.122Z**，API 页为 **2026-09-20T02:53:37.927Z**；它们不是 Jev 1.13 的正式发布日期，后者本次**未知**。Perplexity v1.1 Hub 仓库创建于 **2026-10-05T12:32:27Z**、最后修改于 **2026-10-12T07:26:21Z**，revision 为 `14c9d90137962d300f3de95633f7b6535e80c5ce`；仓库创建时间不等于正式发布日。OpenAI changelog 明确记载 **2026-10-06** 发布 Decisions beta、**2026-09-22** 发布可生成文本的 Luna。除已列出的官方支持模型外，其他所谓 OpenAI “专用决策模型”名称、权重架构或自托管版本均标为**未知**。不做价格横比，也不把营销速度、类型安全或校准声明视为端到端正确性保证。[^t-sitemap][^p-meta][^o-log][^o-guide]

[^t-model]: [TypeSafe Models：当前版本、模态及别名](https://docs.typesafe.ai/models)。
[^t-system]: [TypeSafe System One：输出边界与校准含义](https://docs.typesafe.ai/concepts/system-one)。
[^t-enterprise]: [TypeSafe Enterprise：企业产品说明](https://typesafe.ai/enterprise)。
[^t-api]: [TypeSafe HTTP API](https://docs.typesafe.ai/api)。
[^t-confidence]: [TypeSafe Confidence：公式及阈值说明](https://docs.typesafe.ai/confidence)。
[^t-sitemap]: [TypeSafe sitemap：页面 lastmod](https://docs.typesafe.ai/sitemap.xml)。
[^p-api]: [Perplexity Decisions OpenAPI：请求、响应与限制](https://docs.perplexity.ai/api-reference/decisions-post.md)。
[^p-card]: [Perplexity v1.1 官方模型卡，固定 revision](https://huggingface.co/perplexity-ai/pplx-decider-v1.1-27b/blob/14c9d90137962d300f3de95633f7b6535e80c5ce/README.md)。
[^p-meta]: [Perplexity 官方 Hub 仓库元数据](https://huggingface.co/api/models/perplexity-ai/pplx-decider-v1.1-27b)。
[^p-browser]: [Perplexity 官方浏览器 Decisions 示例](https://docs.perplexity.ai/docs/cookbook/examples/decisions-api-browser-agent/README.md)。
[^o-guide]: [OpenAI Decisions 官方指南](https://developers.openai.com/api/docs/guides/decisions)。
[^o-ref]: [OpenAI Create a decision 官方 API Reference](https://developers.openai.com/api/reference/typescript/resources/decisions/methods/create)。
[^o-log]: [OpenAI API Changelog：2026-10-06、2026-09-22](https://developers.openai.com/api/docs/changelog)。
[^o-structured]: [OpenAI Structured Outputs：约束生成的 schema](https://developers.openai.com/api/docs/guides/structured-outputs)。
[^classifier]: [scikit-learn LogisticRegression：classes_ 与 predict_proba](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)。
[^s-readme]: [Strands 官方 README，固定当前 commit](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/README.md)。
[^s-vision]: [Strands 官方视觉说明，固定当前 commit](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/docs/vision.md)。
[^s-inference]: [Strands 官方推理说明及硬件测量条件，固定当前 commit](https://github.com/strands-labs/strands-decider/blob/b9fc83626bcca822d00f822d6c2e151b8d244ee9/docs/inference.md)。
[^s-card]: [Strands Qwen v1 官方模型卡，固定 revision](https://huggingface.co/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/blob/216d1ccb46a9e9b036eb9187b248db98480f77d4/README.md)。
[^s-compare]: [GitHub 官方 compare API：指定基线至 main](https://api.github.com/repos/strands-labs/strands-decider/compare/0ac22a97e584f3ccf87cc9b8cc5fdbad57cdf534...main)。
[^s-commit]: [GitHub 官方提交：MPS 并发修复](https://github.com/strands-labs/strands-decider/commit/b9fc83626bcca822d00f822d6c2e151b8d244ee9)。
[^s-pypi]: [PyPI strands-decider 发包元数据](https://pypi.org/pypi/strands-decider/json)。
[^s-releases]: [GitHub 官方 releases API](https://api.github.com/repos/strands-labs/strands-decider/releases)。
[^s-tags]: [GitHub 官方 tags API](https://api.github.com/repos/strands-labs/strands-decider/tags)。
[^s-models]: [StrandsAgents 官方模型列表与 revision / lastModified](https://huggingface.co/api/models?author=StrandsAgents&search=decider&full=true)。
[^s-refs]: [Strands Qwen v1 官方 refs API](https://huggingface.co/api/models/StrandsAgents/strands-decider-2B-qwen3.5-v1-2610/refs)。
