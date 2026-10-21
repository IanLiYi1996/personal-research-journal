# DeepInfer 调研 00：执行摘要

**日期**：2026-10-21 | **对象**：deepseek-ai/DeepInfer（2026-10-20 开源，Apache-2.0）

## 它是什么

DeepInfer 是 DeepSeek 线上服务所用推理框架中开源出来的**离线推理引擎**部分：

- **模型**：只支持 DeepSeek-V4.1-Flash 一个模型。
- **硬件**：只支持 NVIDIA SM100 和华为 Ascend 950。
- **未开源的部分**：KV 存储、PD 传输、负载均衡、HTTP 服务和前缀缓存。

配套发布的部署 Q&A 公开了生产系统的大量参数：

- 单集群每天处理 **155 万亿 token**，每 GPU 小时收入 **38 美元**（按低谷价计）；
- decode 用 EP64–128，KV 存在 100 PiB 的 3FS 上，保留 3–7 天。

## 四个结论

1. **它不是通用引擎的竞争者，而是官方参考实现。** vLLM v0.32（10-18）和 SGLang v0.5.22（10-19）在 DeepInfer 开源**之前**就已经支持 V4.1。DeepSeek 的策略是先把能力交给上游，再开源自研引擎。
2. **它的核心价值在于模型和系统的协同设计。** V4.1 用 YOCO、跨层共享和 FP4，把分页 KV 压到 **890 B/token**，约为 V3 的 1/78。这一点又引出了一连串系统设计：
   - 可以跨机房做 PD 分离（约 10 Gbps 量级，相对 400 Gbps 预算）；
   - KV 能保留"天"级的 TTL；
   - decode 不受 KV 容量约束。
3. **引擎内的 CPU 开销已经压到零，下一个主战场是负载均衡。** 零开销异步调度（ZOS）已成为行业标配。线上整体效率约为理想值的 80%，剩下的 20% 是最大的单项损失，主要来自 DP/EP 均衡、存储带宽（DualPath）和弹性 P/D。
4. **跨厂商的样板。** 同一份代码，借助 TileLang 的 cuda/asc 双后端和同名替身库，在 NVIDIA 和 Ascend 950 上都达到了生产级。这对国产芯片生态意义重大。

## 选型建议

| 场景 | 推荐 |
|---|---|
| 上线 V4.1 服务 | SGLang 或 vLLM；规模化部署加 llm-d / Dynamo；KV 层用 Mooncake / LMCache |
| NVIDIA Blackwell 极限性能 | TRT-LLM + Dynamo |
| Ascend / 国产卡 | 以 DeepInfer 为性能基线，用 SGLang-NPU / vllm-ascend / xLLM 上线 |
| SM100 上的离线批处理、评测、研究 | 直接用 DeepInfer |

## 风险与待办

- **没有同口径基准。** 现有性能数字都来自各方自己公布，口径不同，不能直接比较。建议按 [03_同口径基准测试方案.md](/topics/llm-inference/2026-10-21-deepinfer-03-benchmark-plan.md) 执行，需要 64+8 张 SM100，周期约 6 个工作日。
- **SM90 不兼容。** FlashMLA 已移除 Hopper 支持，DeepInfer 短期内无法在 H100/H200 上运行。
- **未核实的事项**：DSpark 原论文、3FS 纠删码实现、Ascend 950 官方规格。

## 架构参考图

![DeepSeek V4.1 Flash 推理系统架构（来源：DeepInfer doc/images，Apache-2.0）](2026-10-21-deepinfer/figures/architecture_overview.png)

## 文档导航

| # | 文档 | 内容 |
|---|---|---|
| 01 | [竞品产品对比](/topics/llm-inference/2026-10-21-deepinfer-01-product-comparison.md) | 13 个竞品：基本面、功能矩阵、硬件覆盖、SWOT、选型 |
| 02 | [推理 Infra 技术调研与解析](/topics/llm-inference/2026-10-21-deepinfer-02-infra-analysis.md) | 9 个技术层总览与趋势 |
| 03 | [同口径基准测试方案](/topics/llm-inference/2026-10-21-deepinfer-03-benchmark-plan.md) | 待执行，需要 64+8 张 SM100 |
| 04 | [专题索引与全局勘误](/topics/llm-inference/2026-10-21-deepinfer-04-topics-index-errata.md) | **勘误以此为准** |
| T1 | [E/P/D 分离](/topics/llm-inference/2026-10-21-deepinfer-05-t1-epd-disaggregation.md) | PD hook、AFD、容错 |
| T2 | [MoE 并行：宽 EP 与 EPLB](/topics/llm-inference/2026-10-21-deepinfer-06-t2-moe-wide-ep-eplb.md) | DeepEP、Mega MoE、EPLB 算法 |
| T3 | [注意力与 KV 压缩](/topics/llm-inference/2026-10-21-deepinfer-07-t3-attention-kv-compression.md) | YOCO、跨层共享；KV 为 890 B/token |
| T4 | [KV 缓存与存储](/topics/llm-inference/2026-10-21-deepinfer-08-t4-kv-cache-storage.md) | 3FS、DualPath、容量建模 |
| T5 | [调度与负载均衡](/topics/llm-inference/2026-10-21-deepinfer-09-t5-scheduling-load-balancing.md) | ZOS 时序、DP 投票、负载均衡 |
| T6 | [投机解码 DSpark](/topics/llm-inference/2026-10-21-deepinfer-10-t6-speculative-decoding-dspark.md) | 置信度变长验证 |
| T7 | [低精度 FP8/FP4](/topics/llm-inference/2026-10-21-deepinfer-11-t7-low-precision-fp8-fp4.md) | 全链路精度流转 |
| T8 | [确定性、采样与结构化输出](/topics/llm-inference/2026-10-21-deepinfer-12-t8-determinism-sampling-so.md) | 批次不变、定点 CDF |
| T9 | [硬件与跨厂商适配](/topics/llm-inference/2026-10-21-deepinfer-13-t9-hardware-cross-vendor.md) | SM100 vs Ascend 950、TileLang |
| 14 | [核实记录与方法说明](/topics/llm-inference/2026-10-21-deepinfer-14-verification-log.md) | 来源分级、抽查与更正 |
| 15 | [参考文献与来源清单](/topics/llm-inference/2026-10-21-deepinfer-15-references.md) | 34 篇 arXiv + 一手链接 |

**下载**：
- [完整版 PDF（114 页）](topics/llm-inference/2026-10-21-deepinfer/deepinfer-report-full.pdf ':ignore')
- [汇报 PPT（18 页）](topics/llm-inference/2026-10-21-deepinfer/deepinfer-slides.pptx ':ignore')
- [执行摘要 PDF](topics/llm-inference/2026-10-21-deepinfer/deepinfer-executive-summary.pdf ':ignore')
- [GitHub 数据快照 CSV](topics/llm-inference/2026-10-21-deepinfer/github-stats-snapshot.csv ':ignore')
