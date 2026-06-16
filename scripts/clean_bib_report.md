# export.bib 清洗报告

- 输入条目：**1071**
- 输出条目：**1014**（去重移除 57 条）
- 填充空 cite key：**190** 条
- 重新消歧的 key 碰撞：**23** 条
- arXiv @techReport → @article：**0** 条

## 合并的重复文献（同标题多份，保留信息最全者）

- (3份) Understanding Transformer Reasoning Capabilities via Graph Algorithms
- (3份) GraphGPT: Graph Instruction Tuning for Large Language Models
- (3份) LM-Cocktail: Resilient Tuning of Language Models via Model Merging
- (3份) A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications
- (3份) AI Alignment: A Comprehensive Survey
- (2份) Amazon Bedrock Claude 3 多模态使用指南 | 亚马逊AWS官方博客
- (2份) Reinforcement Learning Enhanced LLMs: A Survey
- (2份) 强化学习综述 目录
- (2份) Survey on Large Language Model-Enhanced Reinforcement Learning: Concept, Taxonomy, and Methods
- (2份) A Survey on 3D Gaussian Splatting
- (2份) Multimodal Reasoning with Multimodal Knowledge Graph
- (2份) Building and better understanding vision-language models: insights and future directions
- (2份) Let Me Speak Freely? A Study on the Impact of Format Restrictions on Performance of Large Language Models
- (2份) Teaching Transformers Causal Reasoning through Axiomatic Training
- (2份) Aligning Large Language Models with Human Preferences through Representation Engineering
- (2份) The Impact of Reasoning Step Length on Large Language Models
- (2份) CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers
- (2份) From Sora What We Can See: A Survey of Text-to-Video Generation
- (2份) A Multimodal Automated Interpretability Agent
- (2份) Step-by-Step Diffusion: An Elementary Tutorial
- (2份) Recursive Introspection: Teaching Language Model Agents How to Self-Improve
- (2份) Diffusion Feedback Helps CLIP See Better
- (2份) VISTA: Visualized Text Embedding For Universal Multi-Modal Retrieval
- (2份) Internal Consistency and Self-Feedback in Large Language Models: A Survey
- (2份) AudioLCM: Text-to-Audio Generation with Latent Consistency Models
- (2份) ChatQA 2: Bridging the Gap to Proprietary LLMs in Long Context and RAG Capabilities
- (2份) Retrieve Anything To Augment Large Language Models
- (2份) An Empirical Study of Mamba-based Language Models
- (2份) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
- (2份) Comprehensive Exploration of Synthetic Data Generation: A Survey
- (2份) Llama 2: Open Foundation and Fine-Tuned Chat Models
- (2份) Learning to (Learn at Test Time): RNNs with Expressive Hidden States
- (2份) Ring Attention with Blockwise Transformers for Near-Infinite Context
- (2份) LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models
- (2份) DataComp-LM: In search of the next generation of training sets for language models
- (2份) Autoregressive Image Generation without Vector Quantization
- (2份) Editing Large Language Models: Problems, Methods, and Opportunities
- (2份) Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data
- (2份) Large Multimodal Models Towards Building and Surpassing Multimodal GPT-4 Chunyuan Li Deep Learning Team Microsoft Research, Redmond
- (2份) Model Compression and Efficient Inference for Large Language Models: A Survey
- (2份) Elucidating the Design Space of Diffusion-Based Generative Models
- (2份) TinyChart: Efficient Chart Understanding with Visual Token Merging and Program-of-Thoughts Learning
- (2份) Inverse-RLignment: Inverse Reinforcement Learning from Demonstrations for LLM Alignment
- (2份) Faithful Logical Reasoning via Symbolic Chain-of-Thought
- (2份) Predicting Emergent Abilities with Infinite Resolution Evaluation
- (2份) Length Generalization of Causal Transformers without Position Encoding
- (2份) MetaGPT: Meta Programming for A Multi-Agent Collaborative Framework
- (2份) Accelerating Retrieval-Augmented Language Model Serving with Speculation
- (2份) Chain-of-Verification Reduces Hallucination in Large Language Models
- (2份) Knowledge Fusion of Large Language Models
- (2份) LLM Augmented LLMs: Expanding Capabilities through Composition
- (2份) Graph Neural Networks in Practice

## 待人工审阅（元数据不全，正式引用前需处理）

以下 104 条无 `year` 且无 `url`，多为讲座 slides / 技术报告 / 中文博客，不适合直接正式引用：

- 用 `grep -A6 '{<key>,' references/references.bib` 查看单条
- 建议：能补 arXiv/DOI 的补全；纯博客/slides 的可保留在库中供 digest 选题，但论文引用时改引原始来源

明确的垃圾条目（建议直接删除）：

- `Abadi2020References` — 标题就叫 "References"，是别人参考文献列表被误导入
- 标题为「目录」「Content submission guide」「GCR Tech Summit ... Guide」等 — 非文献

## 重新生成方法

```bash
# 1. 清洗原始导出（修空 key / 去重 / 统一 key 格式 / 平衡花括号）
uv run python3 scripts/clean_bib.py export.bib \
    --out references/references.bib --report scripts/clean_bib_report.md

# 2. 重建主题索引（Docsify 可浏览 + 全文搜索）
uv run python3 scripts/bib_index.py references/references.bib --out references/README.md
```

从 Mendeley 再次导出后，重跑这两步即可。

## 第二轮：补全与清理（scripts/enrich_bib.py）

- 删除明确垃圾条目：**12** 条（References / 提交指南 / 致谢 / 目录 等）
- 按标题检索 arXiv 高置信回填 url+year（token Jaccard ≥ 0.85）：**45** 条
- 最终条目：**1002**，其中 737 条有 url
- 仍缺 url 的 **48** 条：多为中文综述 / slides / 书籍，arXiv 无对应，需人工补或保留供 digest 选题

```bash
uv run python3 scripts/enrich_bib.py references/references.bib            # dry run
uv run python3 scripts/enrich_bib.py references/references.bib --apply     # 写入
```
