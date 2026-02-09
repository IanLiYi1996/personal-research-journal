# 多模态大模型技术总结

- **Date:** 2026-02-09
- **Tags:** #multimodal #LMM #CLIP #Flamingo #vision-language #多模态

## Context

本文全面介绍多模态大模型（Large Multimodal Models, LMM）的核心概念、基础原理和最新进展。内容涵盖多模态数据类型与任务、CLIP 和 Flamingo 两大基础模型的详细解析、多模态大模型的通用架构与训练流程，以及未来发展趋势。基于 llm-intro 大模型技术总结系列笔记。

> 注意：不是所有多模态系统都属于 LMM。例如 Midjourney、Stable Diffusion 和 DALL-E 这样的文本到图像模型虽然是多模态的，但并不包含语言模型组件。多模态可能指：输入和输出属于不同模态；输入为多模态；输出为多模态。

---

## 一、多模态介绍

### 1.1 数据的多种模式

我们接触到的数据有多种形式：文本、图片、音频、数据表格等。这些数据的一种形式有时可以转换或模拟成另一种形式：

- **音频** 可以转化为图像形式（如音谱图），语音可以转写为文字，但可能丢失响度、语调、停顿等信息
- **图像** 可以转化为向量，进一步被转换为一串文本词元（Token）序列
- **视频** 实际上是一系列图片加上音频，但目前很多 ML 模型只把视频看作图片的连续播放，这大大限制了能力（研究表明声音在视频中的作用与画面同等重要）
- **文字** 拍照即可视为图片
- **数据表格** 可以变成图表（图像）

**其他数据模态：** 所有数字数据都可以用比特串或字节序列表示。如果模型能有效处理这些序列，就具有强大的学习能力。此外还有图形、3D 素材、气味和触觉等数据格式。

- **图片** 是最多用途的输入方式，可以代表文字、数据表、音频和部分视频
- **文本** 在输出方面更有力量，能完成摘要、翻译、逻辑推理和问答等任务

为简单起见，本文重点关注两种模式：**图像**和**文本**。

### 1.2 多模态任务

多模态任务通常分为两类：**生成**和**视觉语言理解**（VLU）。

#### 生成任务

输出可以是单模态或多模态结合：

- **文本生成图像：** 通过文本生成对应图像，如 DALL-E、Stable Diffusion、Midjourney
- **文本生成：** 典型应用包括视觉问答（VQA）和图像描述（Image Captioning）。VQA 允许为模型同时提供图片和文字信息，例如拍摄物体并提问。图像描述可以帮助搜索特定图片，AI 能自动为图片生成描述和相关信息

#### 视觉语言理解（VLU）

- **图像分类：** 将图像归入预设类别，例如 OCR 系统。与之相似的任务是图像到文本检索——根据给定图像从一堆文字中找出最匹配的描述
- **基于文本的图像检索（TBIR）：** 两种方法：
  1. 为每张图像生成标题和元数据，给定文本查询找出最匹配的图像
  2. 训练图像和文本的**联合向量空间**，给定文本查询生成向量并找到最接近的图像向量。第二种方法更灵活，CLIP 就实现了这一点

---

## 二、多模态训练基本原理

多模态系统的关键组成部分：

1. **编码器**：将各种类型的数据转化为数字向量
2. **对齐方法**：将不同模态的向量对齐到同一多模态向量空间
3. **语言模型**（仅限生成模型）：根据文本和视觉信息生成文本结果

理想情况下，很多组件都应该经过预训练，可在多个场景下重复使用。

### 2.1 CLIP：将语言和图像联系起来

CLIP 最大的亮点是将文本和图像的数据映射到**共享向量空间**。这种共享的多模态向量空间使文本到图像和图像到文本的任务变得更加容易。

CLIP 训练产生了一个强大的图像编码器，在很多图像识别任务上表现出色，甚至不需要额外训练。Flamingo 和 LLaVA 使用 CLIP 作为图像编码器，DALL-E 用 CLIP 来筛选生成的图像。

#### CLIP 的内部构造

CLIP 的工作原理是训练两个编码器，使正确的图片-文字组合尽可能"相似"，错误的组合尽可能"不相似"。

**图像编码器：** 研究团队试验了 ResNet 和 ViT 两种方法，其中 `ViT-L/14@336px` 表现最好：
- 大型视觉 Transformer (ViT-L)
- 将每个图像分成 14x14 的小部分处理
- 可以处理 336x336 像素的图像输入

**文本编码器：** 使用类似于 GPT-2 但规模较小的 Transformer 模型。基础模型包含 63M 参数，拥有 8 个注意力头。研究表明 CLIP 的性能对文本编码器的规模并不敏感。

**向量投影：** 图像和文本的编码转化为"向量"，经过两个转换矩阵 W_v 和 W_l 映射到同一向量空间：
- 图像向量 V_i 的多模态向量：W_v * V_i
- 文本向量 L_i 的多模态向量：W_l * L_i

#### 训练数据（400M 图文配对）

CLIP 团队发现现有数据集既不够大也不够精准，因此创造了一个拥有 **4 亿（400M）图片-文字配对** 的超大数据集：

1. 制定了包含 50 万个查询请求的列表（常用词汇、双词组合、热门维基百科文章标题）
2. 通过字符串匹配找到与请求相符的图像（推测在 OpenAI 的内部数据库中搜索）
3. 每个图像都配对了出现在同一上下文中的文本（标题或评论），而非简单的查询词
4. 为保证数据平衡，每个关键词最多对应 2 万张图像

#### 对比学习训练

CLIP 采用**对比学习**而非传统的分类器目标或语言模型目标。

**三种训练目标对比：**

| 目标类型 | 特点 | 局限性 |
|---------|------|--------|
| 分类器目标 | 从预设选项中选择类别 | 受类别限制，零样本能力弱 |
| 语言模型目标 | 输出一系列 Token | 训练困难，一张图可有多种描述 |
| 对比目标 | 判断匹配程度 | 效率最高，最灵活 |

**对比学习流程：**

处理每一组 N 个（图片和文本）数据时：
- 创建 N 个文本向量 L_1, L_2, ..., L_n 和 N 个图片向量 V_1, V_2, ..., V_n
- 计算所有 N^2 种可能的图片-文本向量组合的相似度
- 目标：正确的图片-文本组合有最高相似度，不正确的组合相似度尽量低
- **N = 32,768**（一次处理的数据量）

每次训练实际完成两项分类工作：
1. 每张图片尝试与 N 段文本配对，找出正确的文本
2. 每段文本尝试与 N 张图片配对，找出正确的图片

模型试图减少两种错误的总数，参数 beta 用于调整"敏感度"。

**效率提升：** 使用对比法，模型效率比传统语言模型高出 **12 倍**，并且生成更好的图片向量。

#### CLIP 的应用

- **图像分类：** CLIP 是备受推崇的"即插即用"工具，可直接使用或微调
- **基于文本的图像检索：** 例如 clip-retrieval 工具：(1) 将图片转化为 CLIP 向量存入向量数据库 (2) 对文字进行 CLIP 向量转化 (3) 在向量数据库中做相似度检索
- **图像生成：** DALL-E 用 CLIP 筛选生成的图片。unCLIP (2022) 是升级版"文字到图片"工具：先用 CLIP 生成文本向量，再用扩散解码器根据向量生成图像
- **文本生成：** CLIP 的图像处理部分是很多多模态大语言模型的基石

### 2.2 Flamingo：新一代多模态大语言模型

与 CLIP 不同，Flamingo 可以**生成文本回复**。Flamingo 类似于 CLIP 加上语言模型，能根据图和文生成文本 Token。

#### Flamingo 的架构

主要由两大部分组成：

1. **视觉编码器（"看"的部分）：** 使用对比学习训练类似 CLIP 的模型，然后去掉文本编码器，只保留视觉编码器
   - 文本方面选择了 BERT 模型（而非 GPT-2）
   - 图像方面使用 NormalizerFree ResNet (**NFNet**) F6
   - 在整合信息前对信息进行平均处理
   - 使用 2.1B（21亿）对图文配对数据训练，比 CLIP 多 5 倍

2. **语言模型（"说"的部分）：** 基于 **Chinchilla** 预训练模型，加入两种新技术：
   - **Perceiver Resampler：** 将不同数量的视觉数据统一为 **64 个标准输出**（初始分辨率 288x288，后提高到 320x320）
   - **GATED XATTN-DENSE：** 在语言模型层之间加入，使文本生成更好地融入视觉信息。缺少此技术整体得分降低 4.2%

#### Flamingo 的四种训练数据集

| 数据集 | 类型 | 规模 | 使用方式 | 训练权重 |
|-------|------|------|---------|---------|
| M3W | 图文交错数据集 | 4300 万个网页 | 每个网页随机抽取 256 Token 内容，取前 5 张图片 | 1.0 |
| ALIGN | 图文配对 | 18 亿对 | 文本是图片的 alt-text，平均 12 个 Token | 0.2 |
| LTIP | 图文配对 | 3.12 亿对 | 文本是详细描述，平均约 20.5 个 Token | 0.2 |
| VTP | 视频文本配对 | 2700 万个短视频 | 每个视频平均时长约 22 秒 | 0.03 |

#### 训练细节

- Chinchilla LM 部分已微调并**锁定**（frozen）
- 新增部分从零开始在四个数据集上训练
- 选择正确的权重对性能非常关键
- 尽管 VTP 权重很小（0.03），移除它会对所有视频相关任务产生不良影响

#### 损失函数

Flamingo 能根据交错显示的图片和视频 x 计算文本 y 的可能性。损失函数考虑四个不同数据集对文本生成的影响，其中 lambda_m 代表数据集 m 的训练权重。

#### CLIP 与 Flamingo 对比

| 特性 | CLIP | Flamingo |
|------|------|----------|
| 能力 | 图文对齐/分类/检索 | 可以生成文本回复 |
| 编码器 | ViT/ResNet + GPT-2 | NFNet + BERT |
| 语言模型 | 无 | Chinchilla |
| 训练方式 | 对比学习 | 对比学习 + 自回归 |
| 开源 | 是 | 否（有复刻版：IDEFICS, open_flamingo） |

---

## 三、多模态大模型结构和技术介绍

### 3.1 架构

多模态大模型的通用架构包括三个核心部分：

#### 编码器

编码器接收图像、音频或视频并输出特征。

**图像编码器：**
| 编码器 | 特点 |
|-------|------|
| NFNet-F6 | NormalizerFree ResNet，Flamingo 使用 |
| ViT | Vision Transformer，通用视觉编码器 |
| CLIP ViT | CLIP 预训练的 ViT，广泛使用 |
| EVA-CLIP ViT | 增强版 CLIP ViT，更强的视觉表示 |

**音频编码器：**
| 编码器 | 特点 |
|-------|------|
| C-Former | 音频特征提取 |
| HuBERT | 自监督语音表示学习 |
| BEATs | 音频预训练模型 |
| Whisper | OpenAI 的语音识别模型 |

#### 连接器

连接器大致分为三种类型：

1. **基于投影的连接器（Projection-based）：** 通过 MLP 或多层 MLP 实现，将编码器特征投影到 LLM 的输入空间。Token 级别融合，将特征处理成 token 与文本 token 一起发送
2. **基于查询的连接器（Query-based）：** 使用 Cross-Attention，一系列可训练的 query 和编码特征 F_X 作为 key 来压缩特征序列到固定长度，将压缩的表示特征输给 LLM（如 BLIP-2 中的 Q-Former）
3. **基于融合的连接器（Fusion-based）：** 在 LLM 内部实现特征级别的融合（如 Flamingo 中的 GATED XATTN-DENSE）

#### LLM + 生成器

- **LLM Backbone：** 作为核心的语言理解和生成引擎
- **生成器（可选）：** 可以附加 Stable Diffusion 等组件，用以生成除文本之外的其他模态数据（图像、音频等）

### 3.2 训练过程

#### 预训练阶段（训练连接器）

假设已有训练好的编码器和大模型，**此阶段主要训练连接器（Projector）**。

- 利用 X-Text 数据集来训练输入/输出的连接器
- 通过优化损失函数实现不同模态的对齐
- X-Text 数据集包括：
  - **图像-文本对：** `<img1><txt1>` 格式
  - **交错图像-文本语料库：** `<txt1><img1><txt2><txt3><img2><txt4>` 格式
  - **视频-文本对**
  - **音频-文本对**

#### 多模态微调（微调大模型/生成器）

多模态微调是对满足指令微调格式的数据集进行微调，使模型能遵循新的指令、泛化到未见过的任务、增强 zero-shot 能力。

**包括两部分：**

1. **监督微调（SFT）：**
   - 将预训练阶段的数据转换为指令感知（instruction-aware）格式
   - SFT 数据可构造为单轮或多轮 QA
   - 模板示例：
     - `<Image>{Question} A short answer to the question is;`
     - `<Image> Examine the image and respond to the following question with a brief answer: {Question}. Answer:`

2. **RLHF（基于人类反馈的强化学习）：**
   - 使模型符合人类意图和偏好
   - 增强多模态模型的交互能力
   - 优化目标与预训练相同

---

## 四、代表性模型

### MoE-LLaVA
- 将混合专家（Mixture of Experts）机制引入多模态大模型
- 在保持参数效率的同时提升模型能力
- 不同专家处理不同类型的视觉-语言任务

### Mini-Gemini
- 追求高效的多模态理解与生成
- 结合高分辨率视觉编码和高效的连接器设计
- 支持图像理解、推理和生成

### VideoLLaMA2
- 专注于视频理解的多模态大模型
- 增强时序建模能力
- 支持视频问答、视频描述等任务

---

## 五、未来发展趋势

### 5.1 融合更丰富的数据类型

处理视频、音乐甚至 3D 内容，在统一空间中表示所有数据类型。

代表性研究：
- **ULIP** (Xue et al., 2022): 统一语言、图片和三维点云表示
- **ImageBind** (Girdhar et al., 2023): 将所有内容连接在一起的向量空间
- **Pathways** (Jeff Dean, 2021): 目标是创建涵盖视觉、听觉和语言理解的多模态模型

### 5.2 更智能的指令响应系统

让机器更好地理解和执行人类指令：

- **MultiInstruct** (Xu et al., 2022): 通过指令优化多模态学习
- **LLaVA** (Liu et al., 2023): 视觉指令调优
- **InstructBLIP** (Salesforce, 2023): 结合视觉与语言的全能模型
- **LaVIN** (Luo et al., 2023): 高效的视觉-语言指令调优

### 5.3 为多模态训练提高效率的适配器技术

通过较少的基础训练更高效地启动多模态系统：

- **BLIP-2：** 在 VQA-v2 零样本测试中超过 Flamingo-80B 8.7%，参数量仅为其 **1/54**
- **LaVIN：** 高效的视觉-语言指令调优
- **LLaMA-Adapter V2：** 参数高效的视觉指令模型

### 5.4 输出的多模态化

产生多模态输出的两种路径：

1. **文本中间产物：** 生成 HTML/LaTeX 等代码，转化为包含文字、格式、图片的内容（如 CM3 生成 HTML，GPT-4V 输出 LaTeX）
2. **多模态 Token：** 生成带标签的 Token，图片 Token 送入 Diffusion 模型生成图片，文字 Token 进入语言模型转化为文本

---

## 六、参考论文列表

按时间排序的多模态系统论文：

- Microsoft COCO Captions (Apr 2015)
- VQA: Visual Question Answering (May 2015)
- VideoBERT (Google, Apr 2019)
- LXMERT (UNC Chapel Hill, Aug 2019)
- **CLIP**: Learning Transferable Visual Models From Natural Language Supervision (OpenAI, 2021)
- Unifying Vision-and-Language Tasks via Text Generation (UNC Chapel Hill, May 2021)
- BLIP (Salesforce, Jan 2022)
- **Flamingo**: A Visual Language Model for Few-Shot Learning (DeepMind, Apr 2022)
- GIT: Generative Image-to-Text Transformer (Microsoft, May 2022)
- MultiInstruct (Xu et al., Dec 2022)
- **BLIP-2** (Salesforce, Jan 2023)
- Cross-Modal Fine-Tuning (Shen et al., Feb 2023)
- KOSMOS-1 (Microsoft, Feb 2023)
- PaLM-E (Google, Mar 2023)
- LLaMA-Adapter (Zhang et al., Mar 2023)
- mPLUG-Owl (Ye et al., Apr 2023)
- LLaMA-Adapter V2 (Gao et al., Apr 2023)
- **LLaVA** (Liu et al., Apr 2023)
- **InstructBLIP** (Salesforce, May 2023)
- Med-PaLM (Singhal et al., May 2023)
- **LaVIN** (Luo et al., May 2023)
- Shikra (SenseTime, Jun 2023)
- Macaw-LLM (Tencent, Jun 2023)

### 其他资源

- [CVPR2023 教程] Large Multimodal Models: Toward Building and Surpassing Multimodal GPT-4
- [CMU 课程] 11-777 MMML
- [开源] Salesforce LAVIS

## Open Questions

- 如何在统一框架下高效处理所有模态（文本、图像、音频、视频、3D）？
- 多模态大模型的幻觉问题如何解决？
- 更高效的适配器和连接器设计方向？
- 多模态输出的质量和可控性如何提升？

## References

- Radford et al. "Learning Transferable Visual Models From Natural Language Supervision" (CLIP), 2021
- Alayrac et al. "Flamingo: A Visual Language Model for Few-Shot Learning", DeepMind, 2022
- Li et al. "BLIP-2: Bootstrapping Language-Image Pre-training", Salesforce, 2023
- Liu et al. "Visual Instruction Tuning" (LLaVA), 2023
- Girdhar et al. "ImageBind: One Embedding Space To Bind Them All", 2023
- Chunyuan Li, "Large Multimodal Models" CVPR 2023 Tutorial
