# DetectorScope 源码核查底稿

核查日期：2026-10-28。固定版本：`52ecc87775154b35d4550911de22faad4fc68f40`。

仓库实际读取使用本地 shallow clone；提交日期 2026-10-24，说明为 `Update citation`。主论文对应 arXiv:2610.30006；初始检索线索中的其他 arXiv/EACL 入口不支持与当前项目的关系，已经排除，详见论文底稿。源码副本保持干净。

## 1. 实现与证据位置

以下行号为本地固定源码的 1-based 行号；可与仓库同 SHA 的源文件对照。只核查所列实现，没有审计项目全部代码路径。

| 事实 | 文件与位置 |
|---|---|
| 默认模型 Gemma-2-2b；激活 hook 保存 block 的第一输入 | `model/extractor.py:45`、`:88` |
| 表示提取与 token 高亮均最多 1024 tokens | `model/extractor.py:134`、`:138`、`:150` |
| SAE canonical 层为 7、13、17、22 | `model/sae.py:34` |
| JumpReLU 编码、正阈值与 MSE＋L0 目标 | `model/sae.py:236`、`:243`、`:261` |
| 文档特征为序列 max-pooling，再拼接各层 | `model/detectors/auditable.py:257` |
| 默认头为 logreg、默认 τ=0.85 | `model/detectors/auditable.py:147` |
| p_ai 是 logit 的 sigmoid；另提供 residual | `model/detectors/auditable.py:103`、`:108` |
| explain 使用零特征向量计算 bias；按零消融贡献排序 | `model/detectors/auditable.py:296` |
| 当前 forward 对列表逐篇 extract/explain | `model/detectors/auditable.py:340` |
| 逻辑回归贡献为 masked input × weight | `model/detectors/heads/logreg.py:107` |
| XGBoost 可计算 zero-ablation 或 TreeSHAP | `model/detectors/heads/xgboost.py:102` |
| 头训练使用 SGD 与近端 L1；标准化折回权重与偏置 | `heads/training/train_logreg.py:83`、`:101`、`:169` |
| 自动描述验证为 Spearman，而非 Pearson | `autointerpret.py:677` |
| 描述生成与验证分别将 random.seed 设为相同的 group_sample_seed | `autointerpret.py:766`、`:877` |
| 采样没有显式排除生成例；默认路径从相同 quantile lists 重取 | `autointerpret.py:391`、`:489`、`:521` |
| 验证仍带激活高亮；零激活负例可随机高亮 | `autointerpret.py:563`、`:614` |
| 解释 mask 只比较 Spearman rho 与阈值；通常无描述则不进入 | `data/cache.py:483` |
| evaluate 在传入 split 上选 Youden 最优阈值并算 F1 | `evaluate.py:76` |

`heads/training/...` 表中的相对路径实际位于 `detectorscope/model/detectors/heads/training/`。

这些证据均来自 [固定版本官方源码](https://github.com/google-deepmind/detectorscope/tree/52ecc87775154b35d4550911de22faad4fc68f40)。

## 2. 三种阈值不能混写

- SAE 的 θ：决定单个特征是否激活。
- auto-interp 的 τ：决定特征是否允许进入分类头。
- classifier 的 t：决定 logit 对应哪个标签。

改变 τ 选择另一训练头；不是只改变 AI 标签判定概率阈值。重新训练还会改变权重，因而非零特征数不保证随 τ 单调减少。

## 3. 线性贡献与非线性贡献

逻辑回归：`delta_i = w_i * m_i * x_i`，能精确加回 logit；概率不具相同可加性。

XGBoost：默认零消融效应可能重复覆盖交互项，不保证贡献加回原预测。该类另支持 TreeSHAP；常规解释路径没有自动改用它。检测对象的 residual 不能被统一忽略。

此处 bias 是零特征向量的输出，不应直接解释为真实空字符串经整个模型后的输出。

## 4. 已执行检查

运行命令：

```bash
python3 research-notes/2026-10-28-detectorscope/checks/principle_checks.py
```

实际执行范围：

1. 自定义线性玩具例子：logit=0.6，sigmoid≈0.645656；三项零消融差与 w×x 一致，概率差不按相同方式加总。
2. 自定义 AND 型玩具树：两个单独零消融效应均为 1，但原分数为 1，residual=-1。
3. 从固定源文件 AST 中提取两个纯列表助手与三个已审阅的纯采样方法；不导入 detectorscope、不构建 FeatureDescriptionValidator、不执行 Gemini 或语义负例路径。五组各20个合成文档，seed42、stratified、uniform-negative：生成10个、验证50个、重叠10个。

第三项证明该配置不保证样本互斥；不证明所有参数下均重叠，也不等于复现论文实验。本次未运行真实检测器或数据。

## 5. 需要复现者额外控制的变量

- 解释发现／验证文档 ID 的显式分离，而不只切换随机种子；
- 是否向解释模拟器展示实际激活高亮；
- 不同 τ 对应的独立头、正则化与训练预算；
- 在 dev 选择阈值，在 test 固定阈值；
- 数据、基础模型、SAE、特征描述与分类头的版本对应；
- 长文本截断与新窗口聚合规则；
- 真实文本编辑是否同时改变了其他特征；
- pickle 训练产物的信任边界；发布推理产物优先使用其 safetensors／JSON 格式。

最后一项是源码中同时支持 pickle 与发布格式产生的安全建议，不代表本次发现了恶意文件或进行了安全审计。

## 6. 已读取发布资料

通过作者 Hugging Face raw README 和 config.json 实际取得：

- 通用模型 default head=logreg、τ=0.85，hook 为8、14、18、23；
- 模型卡报告 dev／dev-test／test AUROC=0.932／0.916／0.857，F1=0.867／0.866／0.751；
- 语言模型适用 Gemma 条款；其他组件许可另列。

读取日期为2026-10-28。没有下载或加载该发布模型，因此这些属于元数据与模型卡核对，不属于运行性能验证。

来源：[模型卡](https://huggingface.co/liamdugan/detector-scope/raw/main/README.md)、[配置](https://huggingface.co/liamdugan/detector-scope/raw/main/config.json)。

## 7. 明确未完成的工作

没有安装项目依赖、下载模型或数据集、训练 SAE／分类器、运行论文指标、评估中文、调用 Gemini、测试 adversarial 改写、测量显存与时延、验证每个发布头的实际二进制内容。

本次报告不是作者身份判定、学术违规调查、全面安全审计或商业许可意见。源码与论文提供的是可审计的统计模型，不是文本来源的形式化证明。
