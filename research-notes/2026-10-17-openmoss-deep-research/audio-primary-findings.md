# OpenMOSS 音频／语音主线：一手资料核验

研究截止与检索日期：2026-10-17。范围限于 SpeechGPT／AnyGPT 的 speech 历史、SpeechGPT-2.0-preview、MOSS-Speech、MOSS-TTSD、MOSS-TTS（含 Realtime）、MOSS-Audio-Tokenizer 及 Eval、MOSS-TTS-Nano、MOSS-Transcribe-Diarize、MOSS-Audio、MOSS-Music、TTSD-eval。

方法：采用 research 与 academic-search 工作流，本分支未进一步分发子任务。用网页搜索发现论文／正式发表入口，以官方 GitHub 源码、HF 模型配置／API、arXiv PDF 和正式会议全文交叉核对。未运行模型、未自行复现任何性能实验；引用次数全部为“未查询”。下文“作者自报”包括论文里的实验，并不因出现在论文中就成为独立验证。“源码核验”只证明公开实现／配置具有该结构；“判断”表示基于这些证据的分析。没有查到独立论文不等于证明其不存在。

## 1. 可直接用于主报告的路线判断

1. **不是一条最终统一到 CAT 的直线，而是三条任务路线分化。** MOSS-Speech 用单码本、ASR 导向的离散语义表示做直接 speech-to-speech；TTS／TTSD 用 CAT 多层 RVQ 做高保真生成；MOSS-Audio／Music 用连续编码器特征做 audio-to-text 理解。同样的 12.5 Hz 不意味着相同表示，更不能说明共享 tokenizer 权重。[MOSS-Speech 正式全文 §2.2](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)、[CAT 全文](https://arxiv.org/pdf/2602.10934v2)、[MOSS-Audio 全文 §2](https://arxiv.org/pdf/2606.01802v3)、[MOSS-Music 架构](https://github.com/OpenMOSS/MOSS-Music#model-architecture)。

2. **端到端 speech-to-speech 不等于实时 TTS，更不等于 full-duplex。** SpeechGPT-2.0-preview 是作者面向实时口语交互的 speech-text 混合模型；MOSS-Speech 的关键贡献是回答时不先生成文字；MOSS-TTS-Realtime 则接收外部 LLM 的增量文本并合成音频，历史用户音频是条件，不是“自己听懂并决定回答”的证明。[preview](https://github.com/OpenMOSS/SpeechGPT-2.0-preview)、[MOSS-Speech](https://github.com/OpenMOSS/MOSS-Speech)、[Realtime 增量文本接口](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)。

3. **TTSD 是 script-to-conversation，不是对话智能。** v1.0 从 MOSS-TTS 中间检查点继续训练，以说话人标签、参考音频和完整剧本为条件；它没有在这一路径中承担用户意图理解、回复规划、工具调用。它的演进重点是长程说话人一致性和训练数据标注质量，而不是“把 LLM 对话能力接上 TTS”。[TTSD 论文 §2–4](https://arxiv.org/pdf/2603.19739v1)。

4. **可扩展性来自两层设计：低时间帧率表示＋避免将 RVQ 深度完全展开到大 LLM 的时间轴。** preview 的 patchify、CAT 的波形 patchify、TTS 的 delay pattern／global-local 都在减少昂贵大骨干的时间步；但 local 深度自回归、delay 首帧缓冲和 codec 成本各有代价。“12.5 tokens/s”通常是帧／块率，不是所有码本索引的总数。[preview 配置](https://huggingface.co/fnlp/SpeechGPT-2.0-preview-7B/blob/main/config.json)、[CAT 附录 A](https://arxiv.org/pdf/2602.10934v2)、[TTS §4](https://arxiv.org/pdf/2603.18090v2)。

5. **能力保持、SOTA、超长稳定均须降格到有边界的作者结论。** MOSS-Speech 冻结预训练缓解但没有消除文本能力下降；MOSS-TTS 最长英文输入桶的识别错误和声纹漂移明显；TTSD 对英文 Eleven V3 的主观偏好是输多赢少，不能写成全面超越闭源。[MOSS-Speech 正式全文表6](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)、[TTS 表6／图6](https://arxiv.org/pdf/2603.18090v2)、[TTSD 图5](https://arxiv.org/pdf/2603.19739v1)。

6. **“开放”主要是推理权重、实现和部分微调／评测工具，不是全部预训练数据与训练过程可复现。** preview 的训练规模和 <200 ms 没有独立论文中的完整协议；Nano 的专属预训练配方未查到；MOSS-Speech 的 Base 仍列在待发布清单；Music 的“Technical Report”引用实际指向 GitHub 仓库。[preview README](https://github.com/OpenMOSS/SpeechGPT-2.0-preview)、[Nano README](https://github.com/OpenMOSS/MOSS-TTS-Nano)、[MOSS-Speech 待办](https://github.com/OpenMOSS/MOSS-Speech#next-steps)、[Music Citation](https://github.com/OpenMOSS/MOSS-Music#citation)。

## 2. 核心论文元数据及准确关联

以下元数据按 arXiv ID 去重；`full_text_status=open_pdf` 表示本次实际读取了公开 PDF，而非仅看到摘要。`year` 采用预印本首发年，正式发表年另写在 `venue`。各项 `citation_count=未查询`；`source_platforms` 为 arXiv，加上表内明确列出的正式发表平台；`fetched_at=2026-10-17`。

### P1 — SpeechGPT

| 字段 | 内容与官方来源 |
|---|---|
| title | SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities。[arXiv](https://arxiv.org/abs/2305.11000) |
| authors | Dong Zhang; Shimin Li; Xin Zhang; Jun Zhan; Pengyu Wang; Yaqian Zhou; Xipeng Qiu。[正式元数据](https://aclanthology.org/2023.findings-emnlp.1055/) |
| year / venue | 2023；Findings of EMNLP 2023，不是 EMNLP 主会长文。[ACL Anthology](https://aclanthology.org/2023.findings-emnlp.1055/) |
| arxiv_id / full_text_status | `2305.11000`；`open_pdf`，v2。[PDF](https://arxiv.org/pdf/2305.11000v2) |
| code_url | https://github.com/0nutation/SpeechGPT/tree/main/speechgpt |

### P2 — SpeechTokenizer：历史路线必要的底层关联论文

| 字段 | 内容与官方来源 |
|---|---|
| title | arXiv 元数据为 SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models；ICLR PDF 标题为 SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models。[arXiv](https://arxiv.org/abs/2308.16692)、[全文](https://arxiv.org/pdf/2308.16692v2) |
| authors | Xin Zhang; Dong Zhang; Shimin Li; Yaqian Zhou; Xipeng Qiu。[元数据](https://arxiv.org/abs/2308.16692) |
| year / venue | 2023 首发；ICLR 2024，PDF 明示正式会议论文。[PDF](https://arxiv.org/pdf/2308.16692v2) |
| arxiv_id / full_text_status | `2308.16692`；`open_pdf`，v2。[PDF](https://arxiv.org/pdf/2308.16692v2) |
| code_url | https://github.com/ZhangXInFD/SpeechTokenizer |

### P3 — SpeechGPT-Gen：SpeechGPT 历史中的生成改进

| 字段 | 内容与官方来源 |
|---|---|
| title | SpeechGPT-Gen: Scaling Chain-of-Information Speech Generation。[arXiv](https://arxiv.org/abs/2401.13527) |
| authors | Dong Zhang; Xin Zhang; Jun Zhan; Shimin Li; Yaqian Zhou; Xipeng Qiu。[元数据](https://arxiv.org/abs/2401.13527) |
| year / venue | 2024；arXiv preprint，未在本次核验中确认正式会议 venue。[arXiv](https://arxiv.org/abs/2401.13527) |
| arxiv_id / full_text_status | `2401.13527`；`open_pdf`，v2。[PDF](https://arxiv.org/pdf/2401.13527v2) |
| code_url | 总项目：https://github.com/0nutation/SpeechGPT/tree/main/speechgpt-gen ；作者感知模块实现：https://github.com/ZhangXInFD/SpeechGPT-Gen-Perceptual-Model |

### P4 — AnyGPT：仅研究 speech 部分

| 字段 | 内容与官方来源 |
|---|---|
| title | AnyGPT: Unified Multimodal LLM with Discrete Sequence Modeling。[arXiv](https://arxiv.org/abs/2402.12226) |
| authors | Jun Zhan; Junqi Dai; Jiasheng Ye; Yunhua Zhou; Dong Zhang; Zhigeng Liu; Xin Zhang; Ruibin Yuan; Ge Zhang; Linyang Li; Hang Yan; Jie Fu; Tao Gui; Tianxiang Sun; Yu-Gang Jiang; Xipeng Qiu。[正式元数据](https://aclanthology.org/2024.acl-long.521/) |
| year / venue | 2024；ACL 2024，Volume 1: Long Papers。[ACL Anthology](https://aclanthology.org/2024.acl-long.521/) |
| arxiv_id / full_text_status | `2402.12226`；`open_pdf`，本次读取截至截止日可得 v5。[PDF](https://arxiv.org/pdf/2402.12226v5) |
| code_url | https://github.com/OpenMOSS/AnyGPT |

### P5 — MOSS-Speech：预印本与 ICLR 版应分别注明

| 字段 | 内容与官方来源 |
|---|---|
| title | 预印本：MOSS-Speech: Towards True Speech-to-Speech Models Without Text Guidance；正式会议：Towards True Speech-to-Speech Models Without Text Guidance。[arXiv](https://arxiv.org/abs/2510.00499)、[ICLR proceedings](https://proceedings.iclr.cc/paper_files/paper/2026/hash/20cf299b663a37b5b936de2cb5713cf5-Abstract-Conference.html) |
| authors | arXiv：Xingjian Zhao; Zhe Xu; Qinyuan Cheng; Zhaoye Fei; Luozhijie Jin; Yang Wang; Hanfu Chen; Yaozhou Jiang; Qinghui Gao; Ke Chen; Ruixiao Li; Mingshu Chen; Ruiming Wang; Wenbo Zhang; Yiyang Zhang; Donghua Yu; Yang Gao; Xiaogui Yang; Yitian Gong; Yuanfan Xu; Yaqian Zhou; Xuanjing Huang; Xipeng Qiu。[arXiv 完整列表](https://arxiv.org/abs/2510.00499) |
| authors（正式版） | Xingjian Zhao; Zhe Xu; Luozhijie Jin; Yang Wang; Hanfu Chen; Yaozhou Jiang; Ke Chen; Ruixiao Li; Mingshu Chen; Ruiming Wang; Wenbo Zhang; Qinyuan Cheng; Zhaoye Fei; Shimin Li; Xipeng Qiu。不要把两版作者名单机械混合。[ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/20cf299b663a37b5b936de2cb5713cf5-Abstract-Conference.html) |
| year / venue | 2025 首发；ICLR 2026 正式会议论文。[arXiv](https://arxiv.org/abs/2510.00499)、[ICLR](https://proceedings.iclr.cc/paper_files/paper/2026/hash/20cf299b663a37b5b936de2cb5713cf5-Abstract-Conference.html) |
| arxiv_id / full_text_status | `2510.00499`；`open_pdf`，arXiv v2 与 ICLR 正式全文均读取。[arXiv PDF](https://arxiv.org/pdf/2510.00499v2)、[ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf) |
| code_url | https://github.com/OpenMOSS/MOSS-Speech |

### P6 — MOSS-Audio-Tokenizer

| 字段 | 内容与官方来源 |
|---|---|
| title | MOSS-Audio-Tokenizer: Scaling Audio Tokenizers for Future Audio Foundation Models。[arXiv](https://arxiv.org/abs/2602.10934) |
| authors | Yitian Gong; Kuangwei Chen; Zhaoye Fei; Xiaogui Yang; Ke Chen; Yang Wang; Kexin Huang; Mingshu Chen; Ruixiao Li; Qingyuan Cheng; Shimin Li; Xipeng Qiu。按 arXiv 保留 `Qingyuan`，不擅自改成其他项目的 `Qinyuan`。[元数据](https://arxiv.org/abs/2602.10934) |
| year / venue | 2026；arXiv technical report，本次未确认正式会议 venue。[arXiv](https://arxiv.org/abs/2602.10934) |
| arxiv_id / full_text_status | `2602.10934`；`open_pdf`，v2。[PDF](https://arxiv.org/pdf/2602.10934v2) |
| code_url | https://github.com/OpenMOSS/MOSS-Audio-Tokenizer |

### P7 — MOSS-TTS

| 字段 | 内容与官方来源 |
|---|---|
| title | MOSS-TTS Technical Report。[arXiv](https://arxiv.org/abs/2603.18090) |
| authors | Yitian Gong; Botian Jiang; Yiwei Zhao; Yucheng Yuan; Kuangwei Chen; Yaozhou Jiang; Cheng Chang; Dong Hong; Mingshu Chen; Ruixiao Li; Yiyang Zhang; Yang Gao; Hanfu Chen; Ke Chen; Songlin Wang; Xiaogui Yang; Yuqian Zhang; Kexin Huang; ZhengYuan Lin; Kang Yu; Ziqi Chen; Jin Wang; Zhaoye Fei; Qinyuan Cheng; Shimin Li; Xipeng Qiu。[元数据](https://arxiv.org/abs/2603.18090) |
| year / venue | 2026；arXiv technical report，本次未确认正式会议 venue。[arXiv](https://arxiv.org/abs/2603.18090) |
| arxiv_id / full_text_status | `2603.18090`；`open_pdf`，v2。[PDF](https://arxiv.org/pdf/2603.18090v2) |
| code_url | https://github.com/OpenMOSS/MOSS-TTS |

### P8 — MOSS-TTSD，亦是 TTSD-eval 的论文来源

| 字段 | 内容与官方来源 |
|---|---|
| title | MOSS-TTSD: Text to Spoken Dialogue Generation。[arXiv](https://arxiv.org/abs/2603.19739) |
| authors | Yuqian Zhang; Donghua Yu; Zhengyuan Lin; Botian Jiang; Mingshu Chen; Yaozhou Jiang; Yiwei Zhao; Yiyang Zhang; Yucheng Yuan; Hanfu Chen; Kexin Huang; Jun Zhan; Cheng Chang; Zhaoye Fei; Shimin Li; Xiaogui Yang; Qinyuan Cheng; Xipeng Qiu。[元数据](https://arxiv.org/abs/2603.19739) |
| year / venue | 2026；arXiv technical report，本次未确认正式会议 venue。[arXiv](https://arxiv.org/abs/2603.19739) |
| arxiv_id / full_text_status | `2603.19739`；`open_pdf`，v1。[PDF](https://arxiv.org/pdf/2603.19739v1) |
| code_url | https://github.com/OpenMOSS/MOSS-TTSD ；评测：https://github.com/OpenMOSS/TTSD-eval |

### P9 — MOSS-Transcribe-Diarize

| 字段 | 内容与官方来源 |
|---|---|
| title | MOSS Transcribe Diarize Technical Report。[arXiv](https://arxiv.org/abs/2601.01554) |
| authors | arXiv 以 MOSI.AI 团队署名并列贡献者：Donghua Yu; Zhengyuan Lin; Hanfu Chen; Chen Yang; Yiyang Zhang; Jingqi Chen; Ke Chen; Liwei Fan; Yi Jiang; Jie Zhu; Muchen Li; Wenxuan Wang; Yang Wang; Zhe Xu; Botian Jiang; Yitian Gong; Yuqian Zhang; Wenbo Zhang; Songlin Wang; Zhiyu Wu; Zhaoye Fei; Qinyuan Cheng; Shimin Li; Xipeng Qiu。页面中的分隔符 `:` 不是作者。[元数据](https://arxiv.org/abs/2601.01554) |
| year / venue | 2026；arXiv technical report，首发 2026-01-04；本次读取的 v7 为 2026-07-17。[版本历史](https://arxiv.org/abs/2601.01554) |
| arxiv_id / full_text_status | `2601.01554`；`open_pdf`，v7。[PDF](https://arxiv.org/pdf/2601.01554v7) |
| code_url | https://github.com/OpenMOSS/MOSS-Transcribe-Diarize |

### P10 — MOSS-Audio

| 字段 | 内容与官方来源 |
|---|---|
| title | MOSS-Audio Technical Report。[arXiv](https://arxiv.org/abs/2606.01802) |
| authors | Chen Yang; Chufan Yu; Hanfu Chen; Jie Zhu; Jingqi Chen; Ke Chen; Wenxuan Wang; Yang Wang; Yaozhou Jiang; Yi Jiang; Zhengyuan Lin; Ziqi Chen; Zhaoye Fei; Chenghao Liu; Donghua Yu; Jun Zhan; Kang Yu; Kexin Huang; Liwei Fan; Mingshu Chen; Qinyuan Cheng; Ruixiao Li; Shimin Li; Songlin Wang; Xingjian Zhao; Yang Gao; Yitian Gong; Yiyang Zhang; Zhe Xu; Xipeng Qiu。[元数据](https://arxiv.org/abs/2606.01802) |
| year / venue | 2026；arXiv technical report，首发 2026-06-01，本次未确认正式会议 venue。[arXiv](https://arxiv.org/abs/2606.01802) |
| arxiv_id / full_text_status | `2606.01802`；`open_pdf`，v3。[PDF](https://arxiv.org/pdf/2606.01802v3) |
| code_url | https://github.com/OpenMOSS/MOSS-Audio |

### 未查到独立全文的项目：不要错误继承家族 paper ID

| 项目 | title / authors / year / venue / arxiv_id / full_text_status | 正确关联与 code_url |
|---|---|---|
| SpeechGPT-2.0-preview | 仓库引用名 SpeechGPT 2.0-preview；Open-Moss；2025；GitHub repository；`null`；`unknown`（未查到独立论文） | 技术依据为仓库及配置，不把 `2510.00499` 写成 preview 自己的论文。https://github.com/OpenMOSS/SpeechGPT-2.0-preview |
| MOSS-TTS-Nano | 仓库引用名 MOSS-TTS-Nano；OpenMOSS Team；2026；GitHub repository；`null`；`unknown`（未查到专属论文） | README 关联 P7/P6；2026-03 的 P7 不足以验证 4 月 Nano 专属训练／CPU 实验。https://github.com/OpenMOSS/MOSS-TTS-Nano |
| MOSS-Music | 仓库 BibTeX 名 MOSS-Music Technical Report；OpenMOSS Team；2026；GitHub repository；`null`；`unknown`（未查到独立论文） | README 仍写 arXiv Coming Soon，BibTeX `note=GitHub repository`；P10 是骨干背景，不是 Music 专属论文。https://github.com/OpenMOSS/MOSS-Music |
| MOSS-Audio-Tokenizer-Eval | 工具名；独立论文作者未提供；2026；软件仓库；`null`；`unknown` | 关联 P6 的重建实验，不另造 paper ID。https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval |
| TTSD-eval | 工具名；无单独论文元数据；2026；软件仓库；独立 ID `null`；独立全文 `unknown` | 框架写在 P8 §5.1，`2603.19739` 是 TTSD 论文。https://github.com/OpenMOSS/TTSD-eval |
| MOSS-TTS-Realtime | 项目页标题 MOSS-TTS-Realtime: Low-Latency and Context-Aware Full-Stream Speech Synthesis；Yiwei Zhao, Yaozhou Jiang, Botian Jiang, Kexin Huang, Yiyang Zhang, Zhe Xu, Yuqian Zhang, Xiaogui Yang, Qingyuan Cheng, Xipeng Qiu；年未在页面明确署出；venue 未确认；`arxiv_id=null`；`unknown` | 查到官方标题／作者页，但页面未给独立论文全文／arXiv，不能当作已核验论文。家族实现：https://github.com/OpenMOSS/MOSS-TTS/tree/main/moss_tts_realtime ；新页：https://openmoss.ai/moss-tts-realtime/ |

## 3. 语音交互历史：从文本中介到直接语音回答

### 3.1 SpeechGPT → SpeechTokenizer → SpeechGPT-Gen／AnyGPT

- **SpeechGPT 的表示与训练。** mHuBERT＋k-means 离散单元，去除相邻重复单元；LLM 扩充语音词表；unit HiFi-GAN 还原波形。三阶段为无配对语音 modality adaptation、语音／文本 cross-modal instruction SFT、chain-of-modality SFT（LoRA）。作者用 LibriLight 60k 小时、约900万配对指令，以及37,969条合成 chain-of-modality 样本；原论文实验骨干为 LLaMA-13B，公开版本却是 LLaMA-7B 系列，不能混为同一 checkpoint。[P1 §3–5](https://arxiv.org/pdf/2305.11000v2)、[公开7B模型清单](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt)。
- **它不是无文本指导的 speech-native 回答。** chain-of-modality 显式按语音指令→文本指令→文本答案→语音答案组织。公开 README 也承认任务识别、语音识别等问题；原论文主要是人工评估与案例，并没有现代实时打断／full-duplex 系统协议。[P1 图2／§3.2](https://arxiv.org/pdf/2305.11000v2)、[README 限制](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt)。
- **SpeechTokenizer 的改进在分层语义／声学表示。** 使用 EnCodec 风格 CNN encoder-decoder，encoder 采用两层 BiLSTM；第一 RVQ 层蒸馏 HuBERT 语义，其余层补充声学信息。历史 speech 配置为 8×1024 codebooks、50 Hz、16 kHz 单声道；BiLSTM 和非因果语义来源意味着不能据此称为严格流式 codec。[P2 §3](https://arxiv.org/pdf/2308.16692v2)、[AnyGPT 所用配置说明](https://arxiv.org/pdf/2402.12226v5)。
- **SpeechGPT-Gen 的 CoIG 是生成效率改进，不是移除全部中介。** LLaMA2-7B-Chat 自回归生成第一层语义 token；双向 Conformer 的 conditional flow matching 补感知 latent，再经 SpeechTokenizer decoder 出波形。显式链预测声学残差；隐式链把语义放进高斯 prior 的均值、预测完整 latent。其“8B”是整体规模表述，不是单个8B纯AR波形模型。跨模态 SFT 后做全参数 chain-of-modality SFT，跳过旧的 modality adaptation；非AR感知网络的双向注意力不能自动等同于实时全流式。[P3 §3–4](https://arxiv.org/pdf/2401.13527v2)。
- **Gen 的开放程度有两层。** 旧总仓库仍写代码／模型将发布，但作者另一个官方仓库已于2024-06公布 CFM／SoundStorm 实现、训练示例及 SoundStorm checkpoint；其 CFM checkpoint 仍列为 TODO。因此不能简单写“全未开源”，也不能写“论文最佳 CFM 全权重已开放”。感知实现代码为 MIT。[感知仓库](https://github.com/ZhangXInFD/SpeechGPT-Gen-Perceptual-Model)。
- **AnyGPT speech 是语义 AR＋声学 NAR。** LLaMA2-7B 只建模 SpeechTokenizer 的第一 RVQ 层（1024词表、50 token/s）；SoundStorm 补2–8层后由 codec 解码，用3秒音色参考。speech-text 预训练数据57k小时，来自 GigaSpeech／Common Voice／MLS；随后用合成 AnyInstruct 做多模态指令调优。speech 推理模块、base/chat 权重、训练脚本和指令集均有入口，许可证沿用 LLaMA2，不是 Apache。[P4 §3–4](https://arxiv.org/pdf/2402.12226v5)、[代码／许可证](https://github.com/OpenMOSS/AnyGPT)。
- **AnyGPT 的实验不支持“speech 全面达到专模”。** 作者在 LibriSpeech test-clean 上报告 ASR WER=8.5%，表中 Whisper Large V2=2.7%；VCTK TTS WER=8.5%、SIM=0.77，表中 USLM=6.5%、0.84。没有同协议实时首音／full-duplex 指标。[P4 表4–5／附录C](https://arxiv.org/pdf/2402.12226v5)。

### 3.2 SpeechGPT-2.0-preview：重心转向 codec patch 与实时交互

- **源码核验：** Qwen2.5-7B-Instruct 系骨干，28层；配置 `n_vq=3`、`group_size=3`。codec 为24 kHz、25 Hz 帧率、3个1024码本，所以75个码本索引／秒、750 bps；邻接3帧 patchify 后，大骨干约处理8.33个语音patch／秒（由配置推算，不是作者实测速度）。局部AR decoder 每次产生一个时间步的多个RVQ token。[模型配置](https://huggingface.co/fnlp/SpeechGPT-2.0-preview-7B/blob/main/config.json)、[codec YAML](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/Codec/config/sg2_codec_config.yaml)、[mixed-model 实现](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/mimo_qwen2_grouped.py)。
- **不是 CAT 的“纯 Transformer codec”。** 公开 codec 配置有 log-mel、ConvNeXt、因果 Transformer、上／下采样和 HiFi-GAN 波形 decoder；CAT 后来的无 CNN 波形 patchify 是另一设计。[codec 配置](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/Codec/config/sg2_codec_config.yaml)。
- **作者自报：** 百万小时真实语音＋数十万小时合成多角色对话、多阶段 speech-text 联合训练、<200 ms 延迟、支持实时打断；只在中文语音上训练，当时未有英文对话能力。未给可复现的统一硬件、计时起止点、p50/p95、网络／VAD分解与预训练全数据；支持工具／联网／知识库是能力或接入叙述，不是完整端到端工具使用评测。[README](https://github.com/OpenMOSS/SpeechGPT-2.0-preview)。
- **开放边界：** 有7B模型、codec权重、推理／demo实现，仓库代码Apache-2.0；对应两个 `fnlp` HF 模型 API 未给明确 `cardData.license`，因此不能只由代码许可证推定所有权重的许可。没有查到独立preview论文。[代码许可证](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/LICENSE)、[7B API](https://huggingface.co/api/models/fnlp/SpeechGPT-2.0-preview-7B)、[codec API](https://huggingface.co/api/models/fnlp/SpeechGPT-2.0-preview-Codec)。

### 3.3 MOSS-Speech：回答不先写文本，但能力保持与实时性仍有代价

- **架构核验：** Qwen3-8B 的36层中，前32层共享，末4层为文本分支，另增4层语音分支；speech head／embedding 独立。生成语音答案不依赖先产出的文字答案；“无文本 guidance”指推理回答路径，不表示训练不使用文本、ASR或speech-text对齐。[HF配置](https://huggingface.co/fnlp/MOSS-Speech/blob/main/config.json)、[正式全文 §2](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)。
- **tokenizer／decoder：** 由 GLM-4-Voice tokenizer 继续训练，encoder 只用ASR目标，并从block-causal改fully causal；单码本12.5 Hz、175 bps。decoder沿CosyVoice2的flow matching路线、缩小chunk以减少等待，公开仓库含flow与HiFi-GAN实现。它不是使用2026年的CAT，也不是所有模块从零训练。[P5 正式全文 §2.2、表2](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)、[decoder实现](https://github.com/OpenMOSS/MOSS-Speech/tree/main/cosyvoice)。
- **训练自报：** 从约900万小时网络音频经VAD得到约400万小时speech。表1的中英interleaved分别952k／690k小时，无监督2303k小时；interleaved按3–6秒speech/text片段混排，另用FineWeb-Edu类文本经CosyVoice2合成。阶段1冻结文本骨干，仅训练语音新增模块；阶段2解冻不同范围并混文本防遗忘；再做四种输入／输出模态的SFT。数据规模是论文作者统计，未获独立审计。[P5 正式全文 §3](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)。
- **明确代价：** 作者encoder ASR probe总体WER=10.80%，对比GLM-4-Voice=9.17%，属于严格因果／语义质量取舍；该probe还受下游Qwen3-0.6B、960小时训练与最佳checkpoint选择影响，不是codec单独的普适评分。[P5 正式全文 §4.1／表2](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)。
- **不要写“无损保留智力”。** 正式版表6中Qwen3-8B的MMLU／CMMLU=76.60／77.35，冻结后各策略约66.50–68.82／69.15–69.27；冻结明显好于不冻结，但仍低于原骨干。表5直接S→S的TriviaQA=28.80，低于text-guided GLM-4-Voice=43.20；相比“speech-to-text QA更强”，直接语音回答仍有落差。[正式全文表5–6](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)。
- **流式与开放不能从论文目标外推。** 编解码器具有流式设计；截至所查README，Gradio streaming output和Base模型都在TODO。代码Apache-2.0，但 `fnlp/MOSS-Speech` 与 Codec API 未给明确权重license。本次未查到完整系统TTFA／RTF／打断协议，因此不继承preview的<200 ms或full-duplex结论。[待办／代码许可](https://github.com/OpenMOSS/MOSS-Speech)、[模型API](https://huggingface.co/api/models/fnlp/MOSS-Speech)、[codec API](https://huggingface.co/api/models/fnlp/MOSS-Speech-Codec)。

## 4. CAT 与评测基础设施：通用高保真表示，不是全部理解模型的前端

### 4.1 MOSS-Audio-Tokenizer

- **原始CAT已核验配置：** 24 kHz单声道→12.5 Hz；encoder／decoder各68个因果Transformer块，波形patchify按240×2×2×2压缩时间轴，RoPE、局部历史注意力；32层RVQ，每层1024项、factorized VQ codebook dimension=8、L2归一化、quantizer dropout=1.0。名义码率为 `12.5 × K × log2(1024)`：K=1时125 bps，K=32时4 kbps。这是未计封装开销的token码率，不是无损压缩。[P6 附录A](https://arxiv.org/pdf/2602.10934v2)、[HF配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer/blob/main/config.json)。
- **语义来源有监督但无音频teacher。** encoder／quantizer／decoder从零联合训练；训练时接0.5B decoder-only LLM，以ASR、多说话人ASR、caption做audio-to-text辅助loss，同时有mel重建、VQ、GAN及feature matching。区别是“不依赖预训练音频encoder／语义蒸馏teacher”，不是“完全无文本监督”。[P6 §3／附录A](https://arxiv.org/pdf/2602.10934v2)。
- **“端到端联合优化”不等于“一阶段训练”。** 作者自报训练约300万小时通用speech/sound/music；先无判别器预训练520k steps，batch1536；再adversarial finetune500k steps，batch768。该两阶段不等于冻结encoder的分段优化，论文有全联合优化与partial optimize的对照。[P6 附录A.2–A.3／§5.1](https://arxiv.org/pdf/2602.10934v2)。
- **流式接口具有约束。** 公开 `chunk_duration` 必须不超过上下文时长，并使samples可整除downsample rate；分块流式只支持batch=1，示例80 ms chunk。严格因果只约束未来信息，并不自动保证1.6B级codec在所有硬件上实时。[公开 Streaming 说明](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer#streaming)。
- **版本分叉不可套用旧论文全部规格。** v2已于2026-06公布48 kHz stereo，配置每侧Transformer层数为12×5+32，32码本；Nano为约20M、48 kHz stereo、16码本、12.5 Hz，配置每侧4+2+2+4层。两者都不是原始24k mono 68层／侧检查点；本次未找到v2／Nano独立技术论文，不能假设其训练小时／全部实验等同P6。[发布记录](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer#release)、[v2配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2/blob/main/config.json)、[Nano配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano/blob/main/config.json)。
- **评测边界：** speech用LibriSpeech test-clean／AISHELL-2的SIM、STOI、PESQ；sound/music用AudioSet eval／MUSDB的mel／STFT距离，另有speech MUSHRA作者听测。模型规模、训练量、码率和领域覆盖并非全部匹配；“lossless-quality”只是项目文案，RVQ和有损指标不足以证明位级无损，也不足以证明所有下游理解任务更好。[P6 表2／附录B](https://arxiv.org/pdf/2602.10934v2)、[官方文案](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer)。

### 4.2 MOSS-Audio-Tokenizer-Eval

- **是重建工具，不是理解／交互benchmark。** 提供native CAT/v2/Nano adapter、任意RVQ前缀层数、确定性多GPU／多节点分片、manifest和CSV；默认STOI、PESQ-NB/WB、mel、spectral convergence、SDR、SI-SDR、STFT，SIM／UTMOS需额外checkpoint。代码Apache-2.0；未查到独立paper。[官方仓库](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval)。
- **关键盲区来自预处理。** STOI／STFT／SIM等主要在16 kHz mono计算，UTMOS先转24 kHz mono；不能据这些指标验证48 kHz stereo空间感、高频质量或音乐结构理解。PESQ是语音类代理指标，sound/music应看不同指标／听测；同样，identity smoke test不代表真实模型结果已复现。[Metrics 精确设置](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval#metrics)、[Smoke Test](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval#smoke-test)。

## 5. 生成支线：MOSS-TTS、TTSD、Realtime、Nano

### 5.1 MOSS-TTS：同一表示上的两种AR取舍

| 设计 | 已核验机制 | 取舍与证据 |
|---|---|---|
| Delay／原始8B | Qwen级大骨干＋1个text/pad head＋32个audio head；第j层RVQ向后移j−1帧，每步并行预测不同物理帧的不同码本 | 不要写成“同一时刻并行生成一个完整32层帧”；便于吞吐优化，但首个完整帧须等待delay填满。[P7 §4.1](https://arxiv.org/pdf/2603.18090v2)、[架构文档](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_delay/README.md) |
| Global-local／原始1.7B骨干 | 每帧global latent，local Transformer顺序产生该帧的text/pad及RVQ层，无时间shift；4层depth module，支持Progressive Sequence Dropout | 帧内多个局部AR步骤，稳态计算更复杂，但无需跨帧delay首音等待；作者实验声纹相似度更好，不能把两种不同骨干规模的对比当作严格控制变量消融。[P7 §4.2](https://arxiv.org/pdf/2603.18090v2)、[Local 文档](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_local/README.md) |

- **训练配方自报：** 原始网络音频标准化／降噪、DiariZen分说话人、MOSS-Transcribe-Diarize生成转写，再做规则／LLM校验、DNSMOS>2.8和AudioBox PQ>6.5、audio/text语言一致及长度过滤。用同说话人不同片段构造参考／目标对，另补字典短词、脏文本、拼音／IPA。四阶段为basic→高权重clone扩能力→恢复混合比例并降LR→32k扩64k且上采样长音频，整体WSD学习率；未给可下载全量百万小时预训练语料。[P7 §5](https://arxiv.org/pdf/2603.18090v2)。
- **时长控制不是额外外部forced alignment。** 训练资产有目标audio-token数与自由时长两种序列化版本，推理 `tokens` 约按12.5帧／秒指定；拼音／IPA通过训练转写替换与输入格式支持。不能由此推定任意外部时间轴的逐字严格同步。[P7 §5.1.3／§6.4](https://arxiv.org/pdf/2603.18090v2)、[模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md)。
- **作者Seed-TTS-eval结果：** continuation下Delay的EN WER=1.84%、SIM=70.86，ZH CER=1.37%、SIM=76.98；Local为1.93%、73.28和1.44%、79.62。优势主要是SIM，并非WER/CER全面第一；外部基线取自其他技术报告，不是全部同机同脚本重跑。架构子目录旧数字与主模型卡／论文有差异，应按P7表3注明模式及版本。[P7 表3](https://arxiv.org/pdf/2603.18090v2)、[当前主模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md)。
- **长文能力有实质退化。** 作者内部中英各6个文本长度桶、每桶10条prompt、clone/continuation共240条生成；使用自家MTD转写，不是开放标准长文benchmark。最长英文50,000+输入桶的WER为17.49%／29.52%，SIM随时长下降；“可生成约一小时”应写成能力上限和带退化的运行区间，不能写成全时长无漂移。[P7 §6.5／表6／图6](https://arxiv.org/pdf/2603.18090v2)。
- **截止日版本：** Delay v1.5仍为8B级并扩至31种语言，加入language tag／显式pause控制；Local v1.5把骨干从Qwen3-1.7B扩到4B，改用CAT-v2、48 kHz stereo、12码本。原论文主要是初版，不可用初版分数直接证明v1.5效果。[发布记录](https://github.com/OpenMOSS/MOSS-TTS#news)、[当前模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_model_card.md)、[Local v1.5 HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5)。

### 5.2 MOSS-TTSD：数据标注与说话人记忆的演进

- **作者历史归纳：** v0（2025-06）中英、最长960秒；v0.5加强声纹切换与克隆、之后加入32k XY-Tokenizer；v0.7（2025-11）用内部早期MTD替代碎片ASR＋diarization标签协调，并使XY decoder同时条件化生成token与参考token，最长1700秒；v1.0（2026-02）换更强骨干与CAT，宣称支持20语言、1–5人、60分钟。[P8 §2](https://arxiv.org/pdf/2603.19739v1)、[legacy 文档及版本日期](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/README.md)。
- **当前核心结构：** Qwen3-8B-Base＋CAT前16层RVQ、12.5 Hz、2 kbps、delay pattern；不是历史XY，也不是TTS初版的32层码本。HF v0.7则标Qwen3-1.7B-Base且有8路配置，不能把v1.0结构倒写到所有历史版本。[P8 §4.1](https://arxiv.org/pdf/2603.19739v1)、[v0.7 HF配置](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v0.7/blob/main/config.json)、[v1.0模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0)。
- **三阶段curriculum：** 由MOSS-TTS中间checkpoint出发，单人＋双人／clone、DNSMOS≥2.8，32,768扩65,536上下文；再高质量≥3.4、真实采样率≥24k、减少单人和降LR；最后加真实3–5人数据与合成多说话人序列。合成序列部分来自交替拼接单说话人片段，高质量和采样率约束减轻背景不连续；不能把它描述成全量天然五人互动数据。[P8 §3–4](https://arxiv.org/pdf/2603.19739v1)。
- **推理不是自由生成语义。** 默认结合每人reference clone与assistant音频prefix continuation，需完整带[S1]等标签的待读剧本及参考转写。历史streamer“streaming”是约20秒音频chunk、batch1，不代表百毫秒互动；v1.0 SGLang加速也不改变其任务归属。[v1.0使用](https://github.com/OpenMOSS/MOSS-TTSD#quick-start)、[legacy streaming](https://github.com/OpenMOSS/MOSS-TTSD/blob/main/legacy/v0.7/README.md)。
- **评测自报与限制：** 在自建TTSD-eval上v1.0的中英ACC约0.9587／0.9626，SIM约0.7949／0.7326；EN WER=9.88%，并非表中最佳。闭源比较使用各厂商私有voice作为MOSS克隆参考，Eleven V3受字数上限按10-turn分块，其他系统还存在context buffer差异。英文主观对Eleven V3为win33.3%、tie25.3%、lose41.4%；中文对Doubao为35.8%、32.6%、31.6%。必须保留这些协议和语言条件。[P8 表1、§5、图5](https://arxiv.org/pdf/2603.19739v1)。

### 5.3 TTSD-eval：降低diarization误差，却有明确二人／forced-alignment边界

- **设计：** MMS-FA把给定剧本与音频强制对齐到词，按标点切句并从剧本标签取真说话人；WeSpeaker SimAMResNet100嵌入对参考音色，最大相似者为预测；ACC衡量标签匹配，SIM取真标签参考相似度；Whisper-large-v3 WER作为补充。作者测试集中英各50条，每语言20对网络参考＋30对seed-tts-eval，Gemini2.5 Pro生成剧本，时长30–720秒。[README](https://github.com/OpenMOSS/TTSD-eval)、[P8 §5.1](https://arxiv.org/pdf/2603.19739v1)。
- **源码核验的关键限制：** `tools/run_similarity.py`只加载speaker1／speaker2两个reference，并将预测硬编码为 `[S1] if sim1 >= sim2 else [S2]`；非S1标签的label similarity也取sim2。因此公开工具默认不是任意N人的benchmark，不能直接证明v1.0五人能力；需要扩展候选参考与标签逻辑。[源码](https://github.com/OpenMOSS/TTSD-eval/blob/master/tools/run_similarity.py)。
- **解释限制（判断）：** 给定正确剧本的forced alignment减少了diarization依赖，但读错／漏读、重叠说话、非常短片段、相近声线和对齐失败仍可能污染ACC/SIM；指标不是对话自然度、情绪、抢话／重叠质量或内容智能。源码同时输出case平均与segment平均，跨报告须统一聚合。最长测试12分钟，主观再切≤90秒，不独立证明60分钟无漂移。[alignment／split](https://github.com/OpenMOSS/TTSD-eval/tree/master/tools)、[聚合源码](https://github.com/OpenMOSS/TTSD-eval/blob/master/tools/run_similarity.py)、[P8 §5](https://arxiv.org/pdf/2603.19739v1)。
- **开放边界：** 代码和Git LFS `testset.zip`公开；本次仓库树未见LICENSE，不能由父项目Apache许可自动推定此独立工具与网络参考音频许可。[仓库](https://github.com/OpenMOSS/TTSD-eval)、[官方文件树API](https://api.github.com/repos/OpenMOSS/TTSD-eval/git/trees/master?recursive=1)。

### 5.4 MOSS-TTS-Realtime：双流TTS，不是自行回答的S2S模型

- **结构／接口：** Qwen3-1.7B骨干＋4层local Transformer（架构文档称约200M），16路RVQ、24 kHz CAT；接收增量text delta，逐帧合成、跨turn复用KV cache，并以既往文本／用户声学信息为条件。输入流和输出流均增量，可接外部LLM。API示例目前batch=1。[架构文档](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_realtime/README.md)、[配置](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime/blob/main/config.json)、[模型卡接口](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)。
- **数据规模是作者自报：** 模型卡称>250万小时单人、>100万小时双／多人对话；该数字不能与CAT的300万小时、TTSD数据或preview数据相加当作互不重叠的数据资产。未查到独立Realtime论文全文以核实全部阶段／样本比例。[模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)。
- **必须保留两套延迟口径。** 旧／家族模型卡：单L20、SDPA＋torch.compile、warmup后TTFB180 ms、RTF0.51；Qwen3.5-9B/vLLM产12文本token197 ms，两者相加377 ms。另一官方项目页：H100上median TTFA56.4 ms、与在线LLM“end-to-end”202.8 ms；页面没有完整实验协议／论文链接，属于另一个官方自报，不能直接与L20数字合并或宣称复现提升。[旧协议](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)、[新官方项目页](https://openmoss.ai/moss-tts-realtime/)。
- **“end-to-end”含义限制：** 新页明确从LLM增量文本起步；没有证据说明计时还覆盖用户说话结束检测、ASR／理解、网络、音频设备缓冲或全双工打断。所以202.8 ms也不能直接与preview的<200 ms对比为完整S2S系统。[项目页任务定义](https://openmoss.ai/moss-tts-realtime/)、[增量文本示例](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)。

### 5.5 MOSS-TTS-Nano：部署优先的独立轻量检查点

- **配置核验：** 约100M生成模型，GPT-2式12层、hidden768、12 heads、RoPE、16,384文本词表，global-local结构且local1层；不是“Qwen3-1.7B量化成100M”。使用独立Nano codec，16×1024码本、48 kHz stereo；codec HF统计约21.97M，所以“100M”也不包括整个音频栈。[TTS配置](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano/blob/main/config.json)、[codec配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano/blob/main/config.json)、[codec API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)。
- **流式确有实现，但长文不等于单段无缝长context。** runtime调用 `inference_stream`，支持增量audio chunk；voice clone长输入会自动分句／chunk。CUDA runtime还把初始decoder预算从1帧提高到4帧以防播放under-run，反映TTFA与持续播放buffer的实际取舍。[runtime源码](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/moss_tts_nano_runtime.py)。
- **CPU性能是自报而非完整benchmark。** 作者称4核可实时；ONNX版效率近2×、MacBook Air M4单核流畅，但README缺可比TTFA／p95／RTF表和明确统一工作负载。ONNX导出包含global prefill／decode-step、local decoder及codec图，确可移除PyTorch运行依赖；这证明部署路径存在，不证明任意4核CPU实时。[README ONNX章节](https://github.com/OpenMOSS/MOSS-TTS-Nano#onnx-cpu-inference)、[导出工具](https://github.com/OpenMOSS/MOSS-TTS-Nano/tree/main/onnx)。
- **最容易误引的实验：** README主要展示Nano *audio tokenizer*对≤120M codec的重建指标，不是Nano整套TTS的自然度、可懂度、声纹与实时性的综合证明。提供微调数据预处理、SFT与多机脚本；本次未查到Nano专属预训练规模／训练阶段论文。README保留“根LICENSE发布前勿再分发”的过期提醒，但截止日根LICENSE实际已为Apache-2.0，HF模型卡亦标Apache。[重建评测](https://github.com/OpenMOSS/MOSS-TTS-Nano#evaluation-metrics)、[微调](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/finetuning/README.md)、[实际LICENSE](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/LICENSE)、[HF模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano)。

## 6. 理解支线：MTD、MOSS-Audio 与 MOSS-Music

### 6.1 MOSS-Transcribe-Diarize：SATS结构化转写，离线长上下文不是实时ASR

- **任务与开放版本：** 输入音频，输出 `[start][Sxx]text[end]`，联合转写、说话人标签和时间戳；公开0.9B于2026-07发布，另有性能更强的Pro仅在线入口。README称50+语言，HF语言标签仅en/zh；后者不是“只有两语言”的证明，但前者也不等于50语言同等评测覆盖。[README](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize)、[HF模型卡](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)。
- **0.9B源码核验：** Whisper-Medium配置encoder（24层、hidden1024），16 kHz／80 mel、30秒frontend chunks；4×时间merge后MLP adaptor；Qwen3-0.6B式28层decoder、128k最大位置，连续特征以 `masked_scatter` 替换audio placeholder。源码虽叫 `VQAdaptor`，实际是投影模块，不是RVQ离散codec，不能凭名字写“用CAT token做ASR”。[配置](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/main/config.json)、[实现](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py)。
- **“90分钟不chunk”的精确读法：** 论文宣称128k context可把整段长录音交给一个全局decoder而不拼接独立短段转写；公开encoder仍按30秒抽特征并重组。故“全局单次转写”不等于“encoder完全不分块”，更不意味着流式低延迟。[P9 §2](https://arxiv.org/pdf/2601.01554v7)、[源码audio_chunk_mapping](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/moss_transcribe_diarize/modeling_moss_transcribe_diarize.py)。
- **训练公开不足：** 论文真实网络多语音频＋模拟2–12人交替／重叠对话，噪声SNR0–15 dB、混响、最多80%较短片段overlap；未完整披露0.9B各阶段、确切训练小时／步数、encoder初始化权重。v7表2数值与当前README0.9B行吻合，但不能把早期论文所有内部模型叙述、Pro或比赛成绩无条件等同于公开checkpoint。[P9 §3–4](https://arxiv.org/pdf/2601.01554v7)、[README分开列0.9B与Pro](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#evaluation)。
- **评测限制十分重要：** CER、cpCER、Δcp=cpCER−CER是内容／说话人归属的转写指标，不是时间戳边界误差或time-based DER。Alimeeting公开0.9B行CER24.86%、cpCER22.17%、Δcp−2.69，说明Δcp不是普适非负“纯speaker错误”分解；跨说话人重排也改变字符顺序。论文把finer timestamp评测与streaming SATS列为future work，不能以SATS名义推定两者已充分验证。[P9 表2、Metrics、Conclusions](https://arxiv.org/pdf/2601.01554v7)。
- **公共／内部测试混合与计时口径：** AISHELL-4为公开长会议；Podcast、Movies为自建，论文称将开放，不能把“将开放”写成此次已确认下载。部分闭源在长输入或格式上被排除。README引用SGLang-Omni单H100自报：长AISHELL4并发1平均45.2秒、RTF0.0197；这是离线整段处理效率，不是听到一词即实时输出的延迟。[P9 §4](https://arxiv.org/pdf/2601.01554v7)、[README Serving benchmark](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize#serve-with-sglang-omni)。

### 6.2 MOSS-Audio：专门的连续表示理解模型

- **结构核验：** 自训audio encoder约0.6B，128 mel、三层stride2 Conv2D、32层Transformer／hidden1280，输出12.5 Hz连续特征；主GatedMLP adapter及DeepStack adapter连接Qwen3-4B／8B文字decoder。HF配置取encoder index8/16/24，注入三个早期LLM层；每25帧（2秒）插elapsed-time数字token。它不是CAT RVQ解码模型，输出是文本，不生成语音／音乐波形。[P10 §2](https://arxiv.org/pdf/2606.01802v3)、[HF配置](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct/blob/main/config.json)、[DeepStack源码](https://github.com/OpenMOSS/MOSS-Audio/blob/main/src/modeling_moss_audio.py)、[时间marker](https://github.com/OpenMOSS/MOSS-Audio/blob/main/src/processing_moss_audio.py)。
- **流式要保守表达。** 论文说encoder局部窗口100帧／8秒、支持长输入效率及KV caching；公开实现按音频chunks批量编码，再接decoder。未查到与Realtime相同的增量speech输入／逐词保证或完整在线latency benchmark，不能由“局部attention”推出严格因果full-duplex。[P10 §2.2](https://arxiv.org/pdf/2606.01802v3)、[encoder实现](https://github.com/OpenMOSS/MOSS-Audio/blob/main/src/modeling_moss_audio.py)。
- **作者数据／训练路线：** 音频按事件边界切、speech/music/general各专模分支标注，再LLM合并；因此“encoder从零训练”不代表训练标注不依赖其他模型。约1.2T训练token，Stage2默认ASR类30%／caption40%／text30%；先稳住encoder／LLM训练adapter与DeepStack，再全模型联合。post-training为任务SFT→audio/text reasoning cold start→DAPO RL；Thinking路线有16 responses／prompt和动态过滤，Instruct与Thinking承担不同目标。[P10 §3–5](https://arxiv.org/pdf/2606.01802v3)。
- **作者实验有条件领先。** Thinking8B在MMAU／MMAU-Pro／MMAR／MMSU算术平均71.08，仍低于表内Gemini3.1-Pro79.89；Instruct8B综合ASR CER11.30，但远近场、多说话人、年龄等子项不是第一，不能写“每类ASR都领先”。Speech captioning使用自建2000条人工reference＋文本judge，不是judge直接听音频。[P10 表1–3、§6.2](https://arxiv.org/pdf/2606.01802v3)。
- **时间戳优势的证据范围：** AAS单位ms，AISHELL-1／LibriSpeech上8B Instruct=35.77／131.61；reference是CTC forced-alignment产物，而不是逐字人工绝对真值。表示“在此reference协议下对齐更好”，不等于真实复杂会议speaker边界／event边界都验证。项目有公开微调工具；未取得全量预训练数据和RL评测复现资产。[P10 §6.4／表4](https://arxiv.org/pdf/2606.01802v3)、[微调文档](https://github.com/OpenMOSS/MOSS-Audio/blob/main/finetune/FINETUNE.md)。

### 6.3 MOSS-Music：音乐理解，不是音乐生成

- **定位与结构：** 在MOSS-Audio骨干上音乐continual pretraining、SFT、Thinking的额外reasoning tuning；两个8B骨干模型，连续12.5 Hz音频encoder＋DeepStack＋time markers、Qwen3-8B文本输出。功能是caption／歌词ASR／结构／和弦调性节拍分析／QA，不是text-to-music波形生成；不要与历史AnyGPT music或其他生成主线混为一谈。[官方README](https://github.com/OpenMOSS/MOSS-Music)、[HF配置](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct/blob/main/config.json)。
- **数据引擎确实公开，但训练集不等于公开。** 官方链接到作者 `wx9songs/MOSS-Music-Data-Pipeline`：MIR（Chordino／BeatNet／Essentia等）、SongFormer结构、lyrics ASR、ALM caption与instruction LLM合并／QA生成；需相应模型／服务和原始音乐资产。没有查到Music专属确切训练小时、stage冻结策略／步数、完整独立论文或Thinking RL配方，不能把P10 DAPO全部照搬。[数据管线](https://github.com/wx9songs/MOSS-Music-Data-Pipeline)、[Music training 描述](https://github.com/OpenMOSS/MOSS-Music#model-architecture)。
- **评测口径会显著影响headline。** Instruct作者自报音乐QA平均80.38，但仅平均8个选定benchmark，三项NSynth排除；Thinking平均74.26，不能说Thinking在音乐各项更强。caption“领先”来自preliminary GPT-5.4-as-a-Judge，1–5分／9维，不是独立听测结论。[官方QA／caption表](https://github.com/OpenMOSS/MOSS-Music#evaluation)。
- **歌词ASR平均不能当单一WER。** Thinking均值15.88%是MUSDB18 WER29.19%、MIR-1K CER15.84%、Opencpop CER2.60%的无权算术平均，混合英文词率与中文字率；不同数据集长度未加权。它不是MUSDB18最佳（表内MusicFlamingo23.41%），也不是Opencpop最佳（Qwen3-Omni2.26%）。timestamped lyrics／chord详细benchmark仍写将发布；不能把能力演示当细粒度音乐事实全验证。[歌词与和弦评测说明](https://github.com/OpenMOSS/MOSS-Music#lyrics-asr-wer--cer)。

## 7. 参数与成本核验：骨干名不等于端到端参数量

下表“HF统计”为官方API返回的safetensors tensor元素统计，本次未下载全权重、未实例化去重计算trainable parameters；可能受权重共享、额外模块和保存方式影响。因此用于提示口径差异，不冒充独立精确模型审计。外部codec不应从服务成本中消失。

| 模型 | 项目／骨干口径 | HF safetensors统计 | 外部音频栈／注意事项 |
|---|---|---:|---|
| SpeechGPT-2.0-preview | 7B级 | 7,770,853,888 | 另需preview codec。[官方API](https://huggingface.co/api/models/fnlp/SpeechGPT-2.0-preview-7B) |
| MOSS-Speech | Qwen3-8B骨干 | 9,095,694,336 | 另有Codec 859,767,808的HF统计，不能只算8B。[模型API](https://huggingface.co/api/models/fnlp/MOSS-Speech)、[codec API](https://huggingface.co/api/models/fnlp/MOSS-Speech-Codec) |
| MOSS-TTS／v1.5 Delay | 8B级 | 8,489,841,664 | 另需CAT。[初版API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS)、[v1.5 API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-v1.5) |
| MOSS-TTS Local初版 | 1.7B骨干 | 3,060,606,464 | 包含local／embedding／heads等，另需CAT。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Local-Transformer) |
| MOSS-TTS Local v1.5 | 4B骨干 | 4,550,403,584 | 另需CAT-v2。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) |
| MOSS-TTSD v1.0 | 8B级 | 8,355,492,864 | 另需CAT，使用前16层RVQ。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTSD-v1.0) |
| MOSS-TTS-Realtime | 1.7B骨干、文档另称约200M local | 2,331,940,864 | 另需CAT；不直接当作“完整1.7B栈”。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Realtime)、[架构文档](https://github.com/OpenMOSS/MOSS-TTS/blob/main/moss_tts_realtime/README.md) |
| CAT原始／v2 | 原论文encoder+decoder约1.6B；v2结构另见配置 | 1,774,566,400／2,123,701,248 | 不含训练时全部semantic LLM／discriminator；论文与发布统计口径不同。[原始API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-Tokenizer)、[v2 API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)、[P6](https://arxiv.org/pdf/2602.10934v2) |
| Nano TTS／codec | 约100M／20M | TTS此次API未给统计；codec21,969,664 | 100M是生成模型自报，不包括codec。[TTS API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Nano)、[codec API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano) |
| MOSS-Transcribe-Diarize | 总模型0.9B | 908,513,280 | audio encoder已在该模型内，不另加CAT。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Transcribe-Diarize) |
| MOSS-Audio 4B／8B Instruct | README总量约4.6B／8.6B | 5,222,820,736／9,052,463,488 | 与README不一致，需记录而非只用较小文案数字。[4B API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-4B-Instruct)、[8B API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Audio-8B-Instruct)、[README](https://github.com/OpenMOSS/MOSS-Audio#released-models) |
| MOSS-Music 8B Instruct | README总量约9.1B | API total9,052,463,488（分dtype字段有微小差异） | audio encoder＋adapter＋LLM一体，不需生成codec。[API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-Music-8B-Instruct)、[模型清单](https://github.com/OpenMOSS/MOSS-Music#released-models) |

## 8. 许可证和开放程度：逐资产，而非只看“Open”

| 项目 | 已核验开放项／许可 | 不能据此推定的部分 |
|---|---|---|
| SpeechGPT／Gen | SpeechGPT有7B分阶段权重、数据和训练脚本；Gen感知实现MIT且CFM最佳权重未完整确认。[SpeechGPT](https://github.com/0nutation/SpeechGPT/tree/main/speechgpt)、[Gen感知仓库](https://github.com/ZhangXInFD/SpeechGPT-Gen-Perceptual-Model) | LLaMA／LLaMA2底座不能自动当Apache；旧论文13B结果≠公开7B。 |
| AnyGPT | base/chat、speech模块、AnyInstruct、训练入口；项目声明沿LLaMA2许可。[仓库](https://github.com/OpenMOSS/AnyGPT) | 不等同Apache，也不等于原始speech/audio资产权利已统一。 |
| preview／MOSS-Speech | GitHub代码Apache-2.0，推理权重可下载；本次fnlp HF API未给明确weight license。[preview LICENSE](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/LICENSE)、[Speech LICENSE](https://github.com/OpenMOSS/MOSS-Speech/blob/main/LICENSE)、[preview API](https://huggingface.co/api/models/fnlp/SpeechGPT-2.0-preview-7B)、[Speech API](https://huggingface.co/api/models/fnlp/MOSS-Speech) | 不由代码许可推定权重；Base／完整预训练数据不等于已开放。 |
| MOSS-TTS／TTSD／Realtime | 家族代码及所查OpenMOSS-Team模型卡Apache-2.0；有推理、微调、服务后端示例。[TTS LICENSE](https://github.com/OpenMOSS/MOSS-TTS/blob/main/LICENSE)、[TTS HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS)、[TTSD HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTSD-v1.0)、[Realtime HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Realtime) | 全量网络预训练语料／来源授权与全训练job未确认。 |
| CAT／v2／Nano codec及Eval | 仓库／所查模型卡Apache-2.0，有HF、ONNX等部署路径；Eval代码Apache。[CAT LICENSE](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer/blob/main/LICENSE)、[v2 HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2)、[Nano HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano)、[Eval LICENSE](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval/blob/main/LICENSE) | 评测依赖模型及数据有各自许可；完整3M小时语料及全部GAN训练资产未确认。 |
| MOSS-TTS-Nano | 根LICENSE与HF卡Apache-2.0，有PT／ONNX／微调。[LICENSE](https://github.com/OpenMOSS/MOSS-TTS-Nano/blob/main/LICENSE)、[HF](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano) | README陈旧免责声明不应压过实际LICENSE；预训练数据／全部CPU性能协议未确认。 |
| MOSS-Transcribe-Diarize | 0.9B权重、源码、微调、字幕工具，Apache-2.0；Pro仅在线。[LICENSE](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/blob/main/LICENSE)、[HF](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize)、[README](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize) | Pro权重、Podcast／Movies下载与完整预训练集未确认。 |
| MOSS-Audio／Music | HF权重卡Apache-2.0，推理代码、Audio微调／Music数据pipeline公开。[Audio HF](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-8B-Instruct)、[Music HF](https://huggingface.co/OpenMOSS-Team/MOSS-Music-8B-Instruct) | 所查GitHub树未见根LICENSE，虽pyproject引用LICENSE、README说models Apache；代码许可文件完备性要与模型卡分开记。[Audio树](https://api.github.com/repos/OpenMOSS/MOSS-Audio/git/trees/main?recursive=1)、[Music树](https://api.github.com/repos/OpenMOSS/MOSS-Music/git/trees/main?recursive=1) |
| TTSD-eval | 代码与测试压缩包公开，所查树无LICENSE。[仓库／文件树](https://github.com/OpenMOSS/TTSD-eval) | 不能默认继承TTSD Apache；参考声音的再分发授权未独立确认。 |

## 9. 共性、取舍与整合时不可越过的证据边界

- **共性一：由“调用外部专模”转向可由LLM直接消费的音频接口，但没有唯一最佳接口。** 语义离散token便于迁移文本知识、牺牲部分细粒度声学；多层RVQ强调可重建性、增加深度建模成本；连续多层特征绕开离散瓶颈、适合感知，却没有天然的音频生成decoder。此判断由P5/P6/P10的不同前端与训练目标支持，不是作者已证明的统一定理。[P5](https://arxiv.org/pdf/2510.00499v2)、[P6](https://arxiv.org/pdf/2602.10934v2)、[P10](https://arxiv.org/pdf/2606.01802v3)。
- **共性二：数据引擎与课程学习比简单堆模型更关键。** SpeechInstruct／AnyInstruct的合成指令、SpeechGPT2的语音合成数据、TTSD由分散ASR标签到MTD、TTS的质量gate与clone采样、Audio/Music分支caption，构成连续的数据工程积累。但大规模语料大多未完整释放，且“无预训练音频teacher”不意味着标注pipeline无外部模型依赖。[P1](https://arxiv.org/pdf/2305.11000v2)、[P4](https://arxiv.org/pdf/2402.12226v5)、[preview](https://github.com/OpenMOSS/SpeechGPT-2.0-preview)、[P7](https://arxiv.org/pdf/2603.18090v2)、[P8](https://arxiv.org/pdf/2603.19739v1)、[P10](https://arxiv.org/pdf/2606.01802v3)。
- **共性三：刻意保留文字能力，而非宣称“音频自己从零长出所有知识”。** MOSS-Speech冻结／渐解冻和text混合、MOSS-Audio text预训练／reasoning冷启动都依赖成熟文本骨干。任务适配后仍可能遗忘；speech-to-text、speech-to-speech和audio reasoning应分别验证。[P5 正式表5–6](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf)、[P10 §4–5](https://arxiv.org/pdf/2606.01802v3)。
- **取舍：吞吐、首音、保真与长程一致性不能用一个数字替代。** Delay偏单骨干吞吐／长context，Local偏首帧完整和帧内依赖；Nano同时缩小生成器与codec以降CPU成本；S2S去文本指导减少中介但可丢QA准确度；离线128k SATS强调全局speaker记忆，不能代替流式ASR低延迟。[P7](https://arxiv.org/pdf/2603.18090v2)、[Nano实现](https://github.com/OpenMOSS/MOSS-TTS-Nano)、[P5](https://arxiv.org/pdf/2510.00499v2)、[P9](https://arxiv.org/pdf/2601.01554v7)。

### 建议主报告保留的限制清单

| 易误写结论 | 应采用的限定措辞 | 官方证据 |
|---|---|---|
| OpenMOSS全部音频模型统一使用CAT | CAT统一了TTS家族；S2S与理解前端另走单码本／连续表示 | [P5](https://arxiv.org/pdf/2510.00499v2)、[P6](https://arxiv.org/pdf/2602.10934v2)、[P10](https://arxiv.org/pdf/2606.01802v3) |
| Realtime <200 ms证明端到端语音对话 | TTS首音／LLM-text→TTS时间，需说明硬件、warmup与未覆盖环节 | [模型卡](https://github.com/OpenMOSS/MOSS-TTS/blob/main/docs/moss_tts_realtime_model_card.md)、[新页](https://openmoss.ai/moss-tts-realtime/) |
| MOSS-Speech完整保持文本智力 | 冻结策略减轻遗忘，正式表格仍有MMLU／CMMLU下降 | [ICLR全文表6](https://proceedings.iclr.cc/paper_files/paper/2026/file/20cf299b663a37b5b936de2cb5713cf5-Paper-Conference.pdf) |
| TTSD-eval证明五人、一小时稳定 | 公开工具二人判别、测试≤720秒、主观片段≤90秒 | [源码](https://github.com/OpenMOSS/TTSD-eval/blob/master/tools/run_similarity.py)、[P8](https://arxiv.org/pdf/2603.19739v1) |
| CAT是无损、Nano重建SOTA等于TTS SOTA | 有损RVQ的重建代理指标；Nano图表主要是codec实验 | [P6](https://arxiv.org/pdf/2602.10934v2)、[Nano评测](https://github.com/OpenMOSS/MOSS-TTS-Nano#evaluation-metrics) |
| MTD时间戳／streaming已全面验证 | 主要验证CER/cpCER；fine timestamp评测与streaming列future work | [P9 Conclusions](https://arxiv.org/pdf/2601.01554v7) |
| Music15.88是普适歌词WER | 三套数据集WER/CER无权平均；细粒度timestamp／chord结果未给 | [Music评测](https://github.com/OpenMOSS/MOSS-Music#evaluation) |
| “1.7B”就是服务全部模型大小 | 骨干名、附加模块、codec与HF统计须分列 | [Local API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Local-Transformer)、[Realtime API](https://huggingface.co/api/models/OpenMOSS-Team/MOSS-TTS-Realtime) |
| Apache项目意味着全数据、全权重许可相同 | 代码、模型卡、数据及独立评测仓库分别核查 | [preview LICENSE](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/blob/main/LICENSE)、[preview模型API](https://huggingface.co/api/models/fnlp/SpeechGPT-2.0-preview-7B)、[TTSD-eval](https://github.com/OpenMOSS/TTSD-eval) |

## 10. 时间冻结与检索备注

使用的论文版本均不晚于截止日：P1 v2、P2 v2、P3 v2、P4 v5、P5 arXiv v2及ICLR2026正式版、P6 v2、P7 v2、P8 v1、P9 v7、P10 v3；相应arXiv页面保留提交历史。正式版与预印本作者／标题不同处已分列，不把后续模型卡升级的结果回填进旧论文。[arXiv元数据入口](https://arxiv.org/abs/2606.01802)。

以下GitHub HEAD经官方API读取，提交时间不晚于2026-10-17；正文为便于阅读使用常规官方链接，需要冻结复核时以这些SHA替换`main`：

| 仓库 | 截止日检索到的SHA与官方复核入口 |
|---|---|
| AnyGPT | [fc3bcc4e9dd76edec8a63300b7ae163a9ecec3d4](https://github.com/OpenMOSS/AnyGPT/commit/fc3bcc4e9dd76edec8a63300b7ae163a9ecec3d4) |
| SpeechGPT-2.0-preview | [730c90f633318111035a8e58dca52e3405f4e9d6](https://github.com/OpenMOSS/SpeechGPT-2.0-preview/commit/730c90f633318111035a8e58dca52e3405f4e9d6) |
| MOSS-Speech | [3a1001ae37b731ae49a98311139fb54c49230fd9](https://github.com/OpenMOSS/MOSS-Speech/commit/3a1001ae37b731ae49a98311139fb54c49230fd9) |
| MOSS-TTSD | [46973e425da227bfa3b315dcd7a4aba8b6be5574](https://github.com/OpenMOSS/MOSS-TTSD/commit/46973e425da227bfa3b315dcd7a4aba8b6be5574) |
| MOSS-TTS | [934d6826b084c46a0d033402174d5f8ac4ed2519](https://github.com/OpenMOSS/MOSS-TTS/commit/934d6826b084c46a0d033402174d5f8ac4ed2519) |
| MOSS-Audio-Tokenizer | [8c50ac4c5d7287d2ed6ea20a08c90ca439887d23](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer/commit/8c50ac4c5d7287d2ed6ea20a08c90ca439887d23) |
| MOSS-Audio-Tokenizer-Eval | [b8e23bffad180116fb0c4ce0552a42690a69e603](https://github.com/OpenMOSS/MOSS-Audio-Tokenizer-Eval/commit/b8e23bffad180116fb0c4ce0552a42690a69e603) |
| MOSS-TTS-Nano | [8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb](https://github.com/OpenMOSS/MOSS-TTS-Nano/commit/8b7bcc9341b3b4ef3a3a58ba1338a7d85ff133eb) |
| MOSS-Transcribe-Diarize | [e3bdd93f8ad58418e04659818efd5fd87a0bc486](https://github.com/OpenMOSS/MOSS-Transcribe-Diarize/commit/e3bdd93f8ad58418e04659818efd5fd87a0bc486) |
| MOSS-Audio | [66326e6e0db34f036c86a76ba005efa4830c69dd](https://github.com/OpenMOSS/MOSS-Audio/commit/66326e6e0db34f036c86a76ba005efa4830c69dd) |

OpenReview页面／API本次遇到验证／403，改由ICLR官方proceedings成功核验正式发表与全文；没有绕过访问控制。Music、preview、Nano和Realtime独立论文检索使用项目名、Technical Report及精确标题组合，并核对作者官方README／HF关联；不足以确认独立全文的项目已标`unknown`。所有性能数值是明确来源协议下的作者／维护者自报；本报告不声称提供独立复现结果。
