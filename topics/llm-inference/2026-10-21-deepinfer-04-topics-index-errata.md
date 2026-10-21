# DeepInfer 调研 04：专题深挖索引、要点与全局勘误

> 截至 2026-10-21。共 9 个专题，约 4,700 行。每个专题都基于 DeepInfer 源码（commit `774ce24`，精确到 `path:line`）、官方部署 Q&A 和实时核实过的论文、竞品资料。
> 上级文档：[01_产品对比_DeepInfer与推理引擎竞品.md](/topics/llm-inference/2026-10-21-deepinfer-01-product-comparison.md)、[02_推理Infra技术调研与解析.md](/topics/llm-inference/2026-10-21-deepinfer-02-infra-analysis.md)。

## 阅读顺序建议

T3（KV 体积是一切设计的源头）→ T4（存储）→ T1（系统形态）→ T2（MoE）→ T5（调度）→ T6（投机解码）→ T7（低精度）→ T8（确定性）→ T9（硬件）

| 专题 | 文件 | 行数 | 一句话核心 |
|---|---|---:|---|
| T1 | [T1_系统形态_EPD分离.md](/topics/llm-inference/2026-10-21-deepinfer-05-t1-epd-disaggregation.md) | ~420 | 开源版只保留了 PE/CE 的"骨架"：6 个 PD hook 全是空实现，CE 强制 `stop_preempted_request`；KV 极小，使 PD 传输只需约 10 Gbps（预算 400 Gbps） |
| T2 | [T2_MoE并行_宽EP与EPLB.md](/topics/llm-inference/2026-10-21-deepinfer-06-t2-moe-wide-ep-eplb.md) | ~510 | EP64 让每卡专家权重的读取量比 EP8 降约 86%；首次 EPLB 重排提前到 1/4 周期；Mega MoE 在 Ascend 的 fp8×fp4 路径上同样可用；EPLB 与批次不变模式存在互斥约束 |
| T3 | [T3_注意力与KV压缩.md](/topics/llm-inference/2026-10-21-deepinfer-07-t3-attention-kv-compression.md) | ~620 | 4 个 KV 源层 + YOCO + FP4，分页 KV = **890 B/token**；134 = 128 + 6（6 = nextn + 1）；consumer 层的 SWA 不落盘，store I/O 因此减半 |
| T4 | [T4_KV缓存与存储.md](/topics/llm-inference/2026-10-21-deepinfer-08-t4-kv-cache-storage.md) | ~480 | `state_size_to_disk` 的计算在源码中有完整断言支撑；10+2 纠删码不在开源 3FS 中；100 PiB、14.4 TB/GPU、约 166 台存储服务器这几个数在数量级上自洽 |
| T5 | [T5_调度与负载均衡.md](/topics/llm-inference/2026-10-21-deepinfer-09-t5-scheduling-load-balancing.md) | ~505 | DP 步型采用**两轮** all-reduce 投票；ZOS 只延迟 decode 步的后处理；抢占时优先牺牲输出最长的序列；ZOS 下 DP 组改走 Gloo（高 DP 规模下是潜在瓶颈）；Crossflow 指出静态 P/D 分区会浪费 17% |
| T6 | [T6_投机解码_DSpark.md](/topics/llm-inference/2026-10-21-deepinfer-10-t6-speculative-decoding-dspark.md) | ~540 | CPU 用 N−2 步的置信度确定总预算，GPU 用 N−1 步的置信度分配到各请求；变长验证的盈亏平衡点约在每 rank 25–30 并发；结构化输出（SO）与 MTP 之间有前向试接受 + 回滚机制 |
| T7 | [T7_低精度FP8_FP4.md](/topics/llm-inference/2026-10-21-deepinfer-11-t7-low-precision-fp8-fp4.md) | ~500 | 主 KV 的格式等于"去掉全局 scale 的 NVFP4"；Hadamard 只用于 128 维的 indexer（`assert head_dim == 128`）；FP8 einsum 的输出直接保持 FP8；GPU 与 NPU 的 scale 打包方式不同（int32 vs int16） |
| T8 | [T8_确定性_采样与结构化输出.md](/topics/llm-inference/2026-10-21-deepinfer-12-t8-determinism-sampling-so.md) | ~580 | 采样器用定点 CDF + 无偏拒绝采样；NPU 只做"确定性算法"，没有批次不变语义；PPL 黄金基线用**逐位精确相等**判定；源码注释显式提到 CVE-2025-32381 |
| T9 | [T9_硬件与跨厂商适配.md](/topics/llm-inference/2026-10-21-deepinfer-13-t9-hardware-cross-vendor.md) | ~515 | **FlashMLA 自 2026-09-30 起移除 Hopper 支持**；Ascend 版 kernel 的代码量普遍比 CUDA 版多 30–110%；NPU 上的 EPLB 不支持 symm_mem；DeepGEMM-Ascend 和 DeepEP-Ascend 都已开源 |

## 交叉核对后的全局勘误（主编核实）

| # | 原说法 | 更正 | 依据 |
|---|---|---|---|
| 1 | 总报告 §4.3：分页 KV 约 1.16 KB/token；T3：1,162 B；T4：约 3.4–10 KB | **890 B/token**：主 KV 720 B + indexer K 170 B | `hugging_face/config.py:74` 中 index-K 映射默认取 `kv_source_layer_ids`；HF `config.json` 中 `kv_source_layer_ids=[2,8,14,20]`，且没有 `index_k_source_layer_ids`；`cache_engine.py:174-187` |
| 2 | T1：PD 带宽按 8 卡每秒 20 万 token 估算，约 4 Gbps | 按 8 卡约 70 万 tok/s 重新计算，约 10 Gbps 量级，结论不变 | README 中的 prefill Pareto 图（约 87k tok/s/GPU） |
| 3 | T2：单专家约 22.7 MB，按 56 层计算 | **约 18.8 MB**（5120 × 2304 × 3，MXFP4 + UE8M0），按 **40** 个 MoE 层计算 | HF config |
| 4 | 二手资料称"DSpark 由 SGLang 首创" | DeepSeek 的算法，SGLang 是按论文实现的 | LMSYS 博客原文："The algorithm and its gains are from the DSpark paper" |
| 5 | 二手资料称"只有 DeepInfer 支持 Ascend 950" | SGLang 的 README 也列出了 950PR/DT | SGLang README |
| 6 | 总报告的说法"Q&A 中描述了 prefill 二次代价模型" | 该模型在开源代码中**完全没有实现**；`FlopsEstimator` 只做 MFU 统计 | `scheduler.hpp:712-719` |

## 仍未核实的关键点（汇总）

- **DSpark 原始论文**：LMSYS 博客只写了"DSpark paper"，没有给链接；arXiv 上也没检索到。DLite（2610.23787）和 ReaDy（2610.20503）都把 DSpark 当作 SOTA 基线引用。
- **3FS 纠删码**：10+2 EC 的实现不在开源版 3FS 中；DeepSeek API 当前的硬盘缓存价格（定价页依赖 JS 渲染，无法抓取）。
- **硬件规格**：Ascend 950 官方规格中的 HBM、带宽和 FP8/FP4 算力（源码里写的是 432 TFLOPS）。FlashMLA README 报告 Ascend 950 稀疏注意力 prefill 达到 410 TFLOPS（峰值的 95%）、decode 达到 360 TFLOPS（峰值的 83%）。
- **其他**：TRT-LLM 中 DSpark 的进展；vLLM / SGLang 与 DeepInfer 尚无同口径基准；DualPath 组内调度的精确公式。
