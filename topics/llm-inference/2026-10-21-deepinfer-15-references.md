# DeepInfer 调研 15：参考文献与来源清单

> 生成于 2026-10-21。下表中 arXiv 编号的标题由 `curl https://arxiv.org/abs/<ID>` 实时抓取，全部 34 篇已逐一核实存在。网页链接的检查结果见 [url_check.tsv](2026-10-21-deepinfer/url_check.tsv)。

## A. arXiv 论文（34 篇）

| arXiv ID | 标题 |
|---|---|
| [2311.18677](https://arxiv.org/abs/2311.18677) | Splitwise: Efficient generative LLM inference using phase splitting |
| [2312.07104](https://arxiv.org/abs/2312.07104) | SGLang: Efficient Execution of Structured Language Model Programs |
| [2401.09670](https://arxiv.org/abs/2401.09670) | DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving |
| [2403.02310](https://arxiv.org/abs/2403.02310) | Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve |
| [2405.04434](https://arxiv.org/abs/2405.04434) | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model |
| [2405.05254](https://arxiv.org/abs/2405.05254) | You Only Cache Once: Decoder-Decoder Architectures for Language Models |
| [2405.16444](https://arxiv.org/abs/2405.16444) | CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion |
| [2407.00079](https://arxiv.org/abs/2407.00079) | Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving |
| [2407.08608](https://arxiv.org/abs/2407.08608) | FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision |
| [2408.12757](https://arxiv.org/abs/2408.12757) | NanoFlow: Towards Optimal Large Language Model Serving Throughput |
| [2412.19437](https://arxiv.org/abs/2412.19437) | DeepSeek-V3 Technical Report |
| [2501.01005](https://arxiv.org/abs/2501.01005) | FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving |
| [2502.11089](https://arxiv.org/abs/2502.11089) | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention |
| [2503.01840](https://arxiv.org/abs/2503.01840) | EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test |
| [2504.02263](https://arxiv.org/abs/2504.02263) | MegaScale-Infer: Serving Mixture-of-Experts at Scale with Disaggregated Expert Parallelism |
| [2504.19867](https://arxiv.org/abs/2504.19867) | semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage |
| [2506.12708](https://arxiv.org/abs/2506.12708) | Serving Large Language Models on Huawei CloudMatrix384 |
| [2507.19427](https://arxiv.org/abs/2507.19427) | Step-3 is Large yet Affordable: Model-system Co-design for Cost-effective Decoding |
| [2510.09665](https://arxiv.org/abs/2510.09665) | LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference |
| [2512.09277](https://arxiv.org/abs/2512.09277) | Efficient MoE Serving in the Memory-Bound Regime: Balance Activated Experts, Not Tokens |
| [2601.02780](https://arxiv.org/abs/2601.02780) | MiMo-V2-Flash Technical Report |
| [2602.21548](https://arxiv.org/abs/2602.21548) | DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference |
| [2603.18567](https://arxiv.org/abs/2603.18567) | SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding |
| [2604.17789](https://arxiv.org/abs/2604.17789) | DuQuant++: Fine-grained Rotation Enhances Microscaling FP4 Quantization |
| [2605.27255](https://arxiv.org/abs/2605.27255) | Pair-In, Pair-Out: Latent Multi-Token Prediction for Efficient LLMs |
| [2605.28302](https://arxiv.org/abs/2605.28302) | How Far Can Disaggregation Go? A Design-Space Exploration of Attention-FFN Disaggregation for Efficient MoE LLM Serving |
| [2607.13095](https://arxiv.org/abs/2607.13095) | Full-Pipeline Inference Optimization for MiMo-V2.5 Series: Pushing Hybrid SWA Efficiency to the Limit |
| [2609.04105](https://arxiv.org/abs/2609.04105) | Hardware-Aware FP4 FlashAttention-4 |
| [2609.26763](https://arxiv.org/abs/2609.26763) | SARA: SLO-Aware Resource Allocation for Disaggregated Agentic LLM Services |
| [2609.27085](https://arxiv.org/abs/2609.27085) | Crossflow: Prefill-Decode Elasticity for Agentic LLM Serving |
| [2609.38981](https://arxiv.org/abs/2609.38981) | Vosti: Specifying, Implementing, and Verifying Deterministic LLM Inference |
| [2610.01380](https://arxiv.org/abs/2610.01380) | GPU-Initiated Communication: Dissecting Down to the Bone |
| [2610.20503](https://arxiv.org/abs/2610.20503) | ReaDy: Recurrent States as Dynamic Context for Speculative Drafting in Hybrid Linear Attention Large Language Models |
| [2610.23787](https://arxiv.org/abs/2610.23787) | DLite: History-free Speculative Decoding for Agentic AI |

## B. 一手资料与项目仓库

- https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats
- https://github.com/InternLM/lmdeploy
- https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/tech_blog/blog26_DeepSeek_V4_on_NVIDIA_Blackwell_Model_Specific_and_Agentic_Workload_Optimizations_in_TensorRT-LLM.md
- https://github.com/NVIDIA/TensorRT-LLM/tree/main/docs/source/blogs/tech_blog
- https://github.com/ai-dynamo/dynamo
- https://github.com/ai-dynamo/nixl
- https://github.com/deepseek-ai
- https://github.com/deepseek-ai/DeepGEMM
- https://github.com/deepseek-ai/DeepInfer
- https://github.com/deepseek-ai/open-infra-index
- https://github.com/kvcache-ai/Mooncake
- https://github.com/kvcache-ai/ktransformers
- https://github.com/llm-d/llm-d
- https://github.com/sgl-project/sglang/releases/tag/v0.5.22
- https://github.com/vllm-project/vllm/releases/tag/v0.32.0
- https://github.com/xLLM-AI/xllm
- https://llm-d.ai/blog/llm-d-v0.5-sustaining-performance-at-scale
- https://pytorch.org/blog/serving-deepseek-v4-on-gb300-with-sglang-5x-higher-throughput-at-the-same-interactivity-since-day-0/
- https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
- https://www.lmsys.org/blog/2026-07-06-dspark-sglang
