# DeepInfer 调研 03：DeepInfer vs vLLM vs SGLang 同口径基准测试方案

> 版本 v1.0，2026-10-21。目标：在**同一套 SM100 集群**上，为 DeepSeek-V4.1-Flash 产出可直接横向比较的 prefill / decode Pareto 曲线和在线服务指标。
> 状态：**方案稿，尚未执行**，需要 SM100 资源。各引擎的 CLI 参数以执行时对应版本的 `--help` 为准。

---

## 1. 为什么现有数字不能直接比

| 来源 | 测法 | 主要不可比因素 |
|---|---|---|
| DeepInfer README | 离线、合成 prompt、按每 DP rank 固定 batch 测稳态步时（SPS）；decode 吞吐 = `avg_SPS × reqs × (1+nextn) / cp`，按全接受计，画图时再按 75% 接受率折算（τ=3.25） | ① 全接受口径；② 不含负载均衡、排队和网络；③ 用合成 prompt，不是真实负载的接受率 |
| vLLM / llm-d | 在线 serving，真实或合成负载，含路由 | 不同模型（R1）、不同规模（256×B200）、含 P/D 开销 |
| SGLang | 多为版本间的相对提升，或 bs=1 的单流速度 | 不是绝对吞吐 |
| TRT-LLM | V4-Pro，GB300，8K 输入 / 1K 输出 | 不同模型、不同硬件 |

**结论**：必须固定**模型、硬件、并行、上下文、batch、投机配置和接受率口径**这 7 个要素，分两层来测：
- **L1 引擎内核层**：离线、固定 batch、测稳态步时。和 DeepInfer 官方口径一致，用来回答"引擎本身快多少"。
- **L2 服务层**：在线、真实负载、看 SLO。回答"上线之后差多少"。DeepInfer 没有服务层，只能参加 L1。

---

## 2. 测试对象与版本冻结

| 引擎 | 版本 | 说明 |
|---|---|---|
| DeepInfer | commit `774ce24`（或测试当日的 HEAD，需记录） | `./build.sh`，`DEEPINFER_ACCELERATOR=nvidia` |
| vLLM | v0.32.0 | 官方 V4.1 recipe |
| SGLang | v0.5.22 | 官方 V4.1 配置 |
| TRT-LLM（可选） | v1.4.0rc0 或之后版本 | 需要先确认是否支持 V4.1 |

- **模型**：`deepseek-ai/DeepSeek-V4.1-Flash-1017`，所有引擎使用**同一份本地权重**，记录 sha256。
- **驱动与库**：CUDA 13.x、NCCL 2.32.3、驱动版本统一并记录。DeepInfer 的 `build.sh` 会强制安装 NCCL 2.32.3，其他引擎要对齐，或者单独记录差异。
- **镜像**：每个引擎一个容器镜像，记录 digest。

---

## 3. 硬件与拓扑

| 角色 | 规模 | 要求 |
|---|---|---|
| Decode 测试 | 64 × SM100（**同一个 scale-up 域**，例如 GB200/GB300 NVL72 的 64 卡；如果是 B200 HGX 跨节点，则单独成组测，并在结论中注明） | 和 DeepInfer README 的 EP64DP64 对齐 |
| Prefill 测试 | 4 × SM100（EP4DP4），可选再测 8 卡 | 和 README 对齐 |
| 小规模对照 | 8 × SM100（EP8） | 复现 Q&A 中 EP8 vs EP64 的对比 |
| 在线 L2 | 1P（8 卡）+ 1D（64 卡），E/P/D 可选 | 只测 vLLM / SGLang / TRT-LLM |

**注意**：DeepInfer 只支持 SM100，不支持 SM103（B300/GB300）[未核实]。如果只有 GB300，需先确认 DeepInfer 能否运行，否则改用 GB200 或 B200。

---

## 4. L1 引擎内核层测试矩阵

### 4.1 Decode（主战场）

| 维度 | 取值 |
|---|---|
| 并行 | EP64 / DP64（主测）；EP8 / DP8（对照） |
| 上下文长度 | 32,768；131,072（主测，和 README 一致） |
| 每 DP rank 的请求数 | 8、16、32、64、128、256 |
| nextn | 0（关闭投机）、3（和 README 图一致）、5（Q&A 生产值） |
| 验证长度 | 固定（`DEEPINFER_ENABLE_MTP_VARLEN=0`）/ 动态（`=1`，仅 DeepInfer 与已实现 DSpark 的引擎） |
| 冗余专家 | 64（EP64，即每卡 1 个，取整后每卡 7 个专家）；EPLB 开启，重排周期统一 |
| 精度 | 原生（MXFP4 专家 + FP4 KV），所有引擎一致；若某引擎不支持 FP4 KV，单列说明 |

**DeepInfer 命令示例：**
```bash
export WORLD_SIZE=<节点数> RANK=<节点序号> CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# nextn=3、128k 上下文；最后一个参数（接受率）只影响画图折算
bash examples/sweep/run_sweep.sh decode ep64-decode-128k-n3 131072 3 0.75
# 关闭投机作为基线
bash examples/sweep/run_sweep.sh decode ep64-decode-128k-n0 131072 0 1.0
```

**vLLM / SGLang 的对应做法：**
- 用"固定 batch、预填好 KV、测稳态 decode 步时"的工具：SGLang 用 `sglang.bench_one_batch` 或 `bench_offline_throughput`，vLLM 用 `vllm bench latency` / `vllm bench throughput`。
- 参数对齐：每 rank 请求数 = batch/DP，输入长度 = 上下文长度，输出长度 ≥ 200 步（丢掉前 50 步预热）。
- 如果工具不支持"共享前缀 + 非缓存后缀"的构造，就用随机 token 构造等长输入，并在结论里注明。

### 4.2 Prefill

| 维度 | 取值 |
|---|---|
| 并行 | EP4 / DP4（主测）；EP8 |
| 上下文长度 | 131,072；非缓存后缀 4,096 |
| chunk 大小 / prefill batch | 4,096、8,192、16,384、32,768、65,536 |
| YOCO 有界重放 | 开启（140 token；DeepInfer 加 `--yoco_clip`，vLLM 对应 decoder replay 的配置） / 关闭 |

**DeepInfer 命令示例：**
```bash
bash examples/sweep/run_sweep.sh prefill ep4-prefill-128k 131072
```

### 4.3 Vision Encoder（可选）

```bash
bash examples/sweep/run_sweep.sh encoder ve-800
```

---

## 5. 指标定义（统一口径，最关键）

### 5.1 L1 指标

| 指标 | 定义 | 备注 |
|---|---|---|
| **SPS**（步 / 秒） | 稳态窗口内的 decode 步数 ÷ 窗口时长；丢掉前 50 步预热，取 ≥ 60 个连续步 | 同时报告 `max_step`，用来识别 EPLB 刷新等引起的抖动 |
| **τ**（每步每请求的有效 token 数） | 实测：1 + 平均接受的草稿 token 数，在**真实文本**上测（见 5.3） | **不得使用全接受口径** |
| **Decode 吞吐**（tok/s/GPU） | `SPS × 每 rank 请求数 × τ ÷ (每 rank 的 GPU 数)` | DeepInfer 原始输出要用实测 τ 重新折算 |
| **每用户速度**（tok/s/用户） | `SPS × τ` | Pareto 曲线的横轴 |
| **Prefill 吞吐**（tok/s/GPU） | 稳态 chunk 的非缓存 token 数 ÷ 时长 ÷ 卡数 | 和 README 一致：跳过 5 个预热 chunk，取 10 个区间 |
| **Prefill 步时延**（ms/步） | 单个 chunk 的耗时 | Pareto 曲线的横轴 |
| **moe_lb** | 最慢 rank 的 MoE token 数 ÷ 平均值 | DeepInfer 日志里直接给出；其他引擎能导出就导出 |
| **MFU** | 由 DeepInfer 的 `FlopsEstimator` 公式计算 | 统一用这一个公式，保证横向可比 |

### 5.2 L2 指标（仅在线服务引擎）

- **基本指标**：TTFT P50/P99、TPOT P50/P99、goodput（在 TTFT ≤ 2 s 且 TPOT ≤ 50 ms 的 SLO 下，可持续的请求率）。
- **单位成本产出**：每 GPU 小时 token 数（输入、输出分开统计），再按统一单价折算成 $/百万 token。
- **缓存相关**：前缀缓存命中率；如果启用 PD，再报告 KV 传输量（GB/s）。

### 5.3 接受率 τ 的测量

1. **数据**：用 3 类真实负载，每类 ≥ 500 条：
   - 对话：ShareGPT 风格；
   - 代码 / agent：SWE 类长上下文多轮；
   - 推理：AIME / GPQA 的思维链。
2. **采样**：温度 1.0、top-p 1.0，以及温度 0 两组。
3. **引擎内统计**：
   - DeepInfer：在 `offline_inference.py` 的输出里统计 `accept_tok` 与 `draft_tok`，它已经打印 `accept_rate`。
   - vLLM / SGLang：读各自的 spec-decode metrics。
4. **代入 Pareto**：得到每类负载的 τ 后，分别画三条 Pareto 曲线，不再用单一的 0.75。

---

## 6. 正确性与质量闸门（先过闸再测性能）

| 检查 | 方法 | 通过标准 |
|---|---|---|
| PPL 一致性 | DeepInfer 的 `tests/models/offline_decode_ppl.json` 基线 + 同一组 prompt 在各引擎上的 decode PPL | 各引擎与 DeepInfer 的差值 ≤ 1%（DeepInfer 自己要求逐位一致） |
| 精度评测 | DeepInfer 的 `eval/`（AIME 2026 / GPQA Diamond / HLE），用规则判分（rule judge）避免判分模型引入噪声 | 和 DeepInfer 参考值（AIME 99.9%、GPQA 93.4%）的差值在统计误差内 |
| 结构化输出 | `examples/_structured_output.py` 中的 regex / JSON / EBNF 用例 | 100% 合法 |
| 确定性（可选） | `DEEPINFER_BATCH_INVARIANT=1 DEEPINFER_ENABLE_MTP_VARLEN=0` 并固定 `seq_seed`；vLLM 设 `VLLM_BATCH_INVARIANT=1`；SGLang 开确定性模式 | 同一输入在不同 batch 组成下输出逐位一致 |

---

## 7. 执行流程与时间估算

| 阶段 | 内容 | 估算 |
|---|---|---|
| D0 | 申请资源：64 + 8 卡 SM100；冻结镜像、权重和驱动 | 视资源而定 |
| D1 | 三个引擎各自部署跑通，过正确性闸门（§6） | 1–2 天 |
| D2–D3 | L1 decode 矩阵：3 引擎 × 2 上下文 × 6 batch × 3 种 nextn × 2 种验证方式，约 200 个点，单点约 5–10 分钟 | 2 天 |
| D4 | L1 prefill 矩阵 + EP8 对照 + τ 测量 | 1 天 |
| D5 | L2 在线测试（vLLM / SGLang，1P + 1D） | 1 天 |
| D6 | 出图、写报告、复核 | 1 天 |

**每个测试点必须记录**：
- 引擎版本、commit、镜像 digest；
- 完整命令行与环境变量；
- 原始 jsonl 或日志；
- `nvidia-smi` 拓扑和时钟；
- 是否有其他任务同机运行。

---

## 8. 风险与对策

| 风险 | 对策 |
|---|---|
| vLLM / SGLang 的离线工具不能精确复刻"共享前缀 + 非缓存后缀"的 batch 构造 | 用等长随机输入加预热，并在报告中标注差异；必要时写一个薄封装，直接调用引擎的 `LLM.generate` 接口 |
| DeepInfer 的折算口径与他家不同 | 一律按 §5.1 用实测 τ 重算，DeepInfer 原始 jsonl 一并归档 |
| EPLB 刷新和日志干扰步时 | DeepInfer 已在测量窗口内剔除这些区间；其他引擎需关闭周期日志，或剔除离群步 |
| 64 卡不在同一 scale-up 域 | 结论中单独标注；对 Mega MoE 类优化影响很大 |
| SM103（GB300）兼容性 | 先跑 `examples/offline_inference.py` 冒烟测试 |
| 结果被误读为"引擎优劣" | 报告中强调 L1 是内核层、L2 是服务层，DeepInfer 不参与 L2 |

---

## 9. 交付物

- **原始数据**：`sweeps/<engine>/<tag>/results.jsonl`。
- **图表**：Decode Pareto，2 种上下文 × 3 类负载 τ，每图 3 条引擎曲线；Prefill Pareto；EP8 vs EP64；固定验证 vs 动态验证。
- **汇总表**：各工作点（每 rank 64 请求、每用户 100 / 200 tok/s 等）的吞吐对比。
- **复现文档**：命令行、环境、镜像。
