# 开放权重 voice-to-voice 模型选型核查

核查日期：2026-11-03。范围：语音输入、语音输出的对话模型，重点是中文及实时全双工。不把独立 TTS 模型或仅开放 API 的模型当作可自部署开放权重模型。本次未下载模型、运行推理或做主观听测；推荐是根据一手资料作出的选型判断，不是全行业统一排名。

## 结论

- 中文实时语音助手：优先试 MiniCPM-o 4.5。官方模型卡明确支持中英双语、全双工流式对话、参考音频声音克隆，模型与代码采用 Apache-2.0。[官方模型卡](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- 英文全双工交互：优先对比 NVIDIA PersonaPlex-7B-v1。官方定义其输入和输出语言为英文，支持同时听说、打断、重叠说话和角色/声音提示。它是开放权重模型，但权重采用 NVIDIA Open Model License，不应把它称作 Apache-2.0 模型。[官方模型卡](https://huggingface.co/nvidia/personaplex-7b-v1)
- 多语言、广泛音频理解：Qwen3-Omni-30B-A3B-Instruct 是重要对照，支持中文语音输入输出。应使用 Instruct；Thinking 版本只有文本输出，不能直接完成 voice-to-voice。[官方模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- 纯音频研究/任务泛化的另一候选是 MiMo-Audio-7B-Instruct。官方称其在语音对话与音频理解评测达到发布时开放模型领先表现，但此陈述本身不能证明它在当前中文全双工产品体验上胜过其他候选。[官方仓库](https://github.com/XiaomiMiMo/MiMo-Audio)

## MiniCPM-o 4.5 与 Qwen3-Omni 的具体证据

MiniCPM-o 4.5 的技术报告表 4、5给出以下结果。这是开发方报告的评测，不能直接解释为独立机构排名或所有部署模式的性能保证。[技术报告](https://arxiv.org/html/2604.27393v1)

| 指标 | MiniCPM-o 4.5 | Qwen3-Omni |
| --- | ---: | ---: |
| VoiceBench AlpacaEval，1–5 分 | 4.81 | 4.74 |
| SeedTTS Test-ZH CER，越低越好 | 0.86 | 1.41 |
| SeedTTS Test-EN WER，越低越好 | 2.38 | 3.39 |
| LongTTS EN WER，越低越好 | 3.37 | 17.33 |

注意：表 10 显示全双工使用的动态文本交织 TAIL 模式会牺牲部分生成准确率；因此不能把表 5 的最佳指标直接当作全双工模式指标。论文还承认长时间动态交互鲁棒性需要验证，流式语音可能误读或意外中英混用。[技术报告，表 10 与局限性](https://arxiv.org/html/2604.27393v1)

## 部署和许可证

- MiniCPM 官方 PyTorch 全精度实时 Web Demo 要求至少 28GB 显存；量化 llama.cpp-omni 路径列出至少 12GB NVIDIA 显存。Mac 的全双工示例要求 Apple M4 Max、至少 24GB RAM。不要把低内存量化路径与全精度路径混为一谈。[模型卡部署要求](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Qwen3-Omni 的 30B-A3B 名称不能当作只有 3B 权重常驻显存的保证。官方 BF16 显存表测的是包含音视频的指定长度输入，不是纯语音对话的统一最低显存。[模型卡](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct)
- Qwen3-Omni 发布权重采用 Apache-2.0；官方报告把 234ms 描述为冷启动条件下的理论端到端首包延迟，不应当作真实产品端到端实测保证。[Qwen 技术报告](https://arxiv.org/html/2509.17765)
- PersonaPlex 官方报告平滑接话延迟 0.170 秒、用户打断延迟 0.240 秒，来自 FullDuplexBench 指定评测，不等于任意硬件/网络条件下的产品延迟。[模型卡评测定义](https://huggingface.co/nvidia/personaplex-7b-v1)

## 核查边界

论文发表或 API 上线不等于权重已经开放。本报告只对有明确一手权重与许可证证据的候选作选型建议，不声称已经覆盖全部截至核查日期的新品。

PersonaPlex 与 Moshi 的发布版评测、许可证及硬件要求进一步核查，见同目录 `full-duplex-notes.md`。
