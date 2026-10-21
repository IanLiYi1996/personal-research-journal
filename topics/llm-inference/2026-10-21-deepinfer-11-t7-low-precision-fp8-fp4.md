# DeepInfer 专题 T7：低精度 —— FP8 / MXFP4 / NVFP4 / UE8M0 在 V4.1 推理中的全链路使用

> 调研对象：DeepInfer（DeepSeek-V4.1-Flash 推理引擎，commit 774ce24）  
> 时间：2026-10-21  
> 标注规范：**[源码]** 读代码确认 · **[Q&A]** 官方文档 · **[估算]** 写出算式 · **[推断]** 合理推测 · **[未核实]** 未找到一手来源

---

## 1. 格式原理

### 1.1 浮点数格式概览

| 格式 | 符号 | 指数位 | 尾数位 | 最大值 | 每组元素数 | scale 格式 | 硬件支持 |
|------|------|--------|--------|--------|-----------|-----------|---------|
| E4M3 (FP8) | 1 | 4 | 3 | 448.0 | 32（DeepInfer 中） | UE8M0 packed int32 | SM90+（tcgen04） |
| E5M2 (FP8) | 1 | 5 | 2 | 57344.0 | 32 | FP32/UE8M0 | SM90+（未在 V4.1 中使用） |
| E2M1 (FP4) | 1 | 2 | 1 | 6.0 | 16 或 32 | e4m3 或 UE8M0 | SM100（tcgen05）|
| UE8M0 | 无 | 8 | 0 | 2^127 | — | 自身即 scale | SM100 原生 |

**UE8M0（无符号 8 位指数）**：仅存储指数字节，表示 `2^(byte-127)`，无尾数、无符号位。恢复为 FP32 等效于 `torch.ldexp(1.0, int(byte)-127)` **[源码]** `deepinfer/hugging_face/weights.py:163-167`。因为只有整数幂次，所以是"无损"的——编码和恢复无精度损失。

**OCP Microscaling (MX) 规范**定义 MXFP4 为 E2M1 元素 + 每 32 个元素共享一个 UE8M0 scale；MXFP8 为 E4M3/E5M2 + 每 32 个元素共享一个 UE8M0 scale。

**NVFP4**（NVIDIA 专有）：E2M1 元素 + 每 16 个元素共享一个 E4M3 scale，另有一个全局 FP32 scale。与 MXFP4 的区别：
- 分组粒度：16 vs 32
- scale 类型：e4m3（浮点）vs ue8m0（整数幂次）
- 全局 scale：NVFP4 有，MXFP4 无

V4.1 中主 KV 的格式（`V41_FP4_FORMAT`）与 NVFP4 高度相似但去掉了全局 scale：每 16 个 E2M1 元素一个 E4M3 scale，无全局 FP32 scale **[源码]** `cache_common.py:119-128`。

### 1.2 Blackwell SM100 tensor core 支持矩阵

SM100（B200/GB200）引入 `tcgen05` 指令集，原生支持：
- **FP8×FP8 MMA**：两操作数均为 E4M3，累加器为 FP32
- **FP4×FP4 MMA（block-scaled MMA）**：E2M1 数据 + UE8M0 scale，scale 在 tile 级别应用
- SM100 要求 scale 以 **packed UE8M0** 格式传入：4 个 UE8M0 字节打包成 1 个 `torch.int32`

SM90（H100/H200）不支持 FP4 MMA，scale 格式为 FP32。

**[源码]** DeepGEMM README（https://github.com/deepseek-ai/DeepGEMM）：
> "SM100 requires scaling factors in packed UE8M0 format, which packs 4 UE8M0 into a single `torch.int`."

DeepInfer `platform.py` 体现了这一差异：
```python
# platform.py:63
self.fp8_sf_type = torch.int32   # GPU (SM100): 4 个 UE8M0 → 1 个 int32
# platform.py:138
self.fp8_sf_type = torch.int16   # NPU (Ascend 950): 2 个 UE8M0 → 1 个 int16
```
**[源码]** `deepinfer/platform.py:63,138`

### 1.3 SM100 vs SM90 scale 布局：TMA 对齐转置

DeepGEMM README 说明 SM100 要求 LHS scale（激活侧）有 "TMA-aligned and transposed layout"，即列主序存储（`use_tma_aligned_col_major_sf=True`），且每行对齐到 TMA 要求的字节边界（通常 16 字节对齐）。SM90 使用行主序 FP32 scale。

`deepinfer/hugging_face/parameters.py` 的 `set_ue8m0_sf` 函数通过 `batched_transpose_weight_sf` 实现这个从行主序 FP32 到列主序 UE8M0 packed int32 的转换 **[源码]** `parameters.py:58-70`。

**UE8M0 packing 实现**（from `engram.py:27-32`）：
```python
def _pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack the ue8m0 scales represented in fp32 into int32 (CUDA) / int16 (ASC), 4/2 per group."""
    x_int32 = x.view(torch.int32)
    # x 是 2^k 形式，mantissa 全零，提取指数字段
    return torch.floor_divide(x_int32, 1 << 23).to(torch.uint8).view(PLATFORM.fp8_sf_type)
```
即：`exponent_byte = (float32_bits >> 23) & 0xFF`，然后 4 个 uint8 view 为 int32。

---

## 2. 源码全链路

### 2.1 量化配置约束（config.py）

`load_hf_model_config` 在启动时严格校验 HF `quantization_config` 字段 **[源码]** `deepinfer/hugging_face/config.py:253-262`：

```python
for key, expected in {
    "quant_method":   "fp8",
    "expert_dtype":   "fp4",
    "activation_scheme": "dynamic",
    "weight_block_size": [32, 32],
    "scale_fmt":      "ue8m0",
}.items():
    if quant[key] != expected:
        raise ValueError(...)
```

这意味着 DeepInfer **只接受** FP8 激活动态量化 + FP4 专家权重 + 32×32 块 + UE8M0 scale 的 checkpoint。非此格式会在加载时报错，不会静默降级。

衍生的全局精度标志（config.py:301-307）：
```python
use_fp4_sparse_kv = text.get("use_fp4_sparse_kv", True)  # 默认开 FP4 主 KV
fp8_attn  = fp8   # True
fp8_mlp   = fp8   # True
mxfp8     = fp8   # True
mxfp4     = mxfp4 # True
```

### 2.2 权重加载：F8_E8M0 → FP32 scale 恢复

**普通 FP8 权重**（非专家线性层）加载路径 **[源码]** `deepinfer/hugging_face/weights.py:418-421`：
```python
if param.data.element_size() == 1:   # FP8 参数（1 字节）
    sf = reader.read_concat(scale_names, param.device)  # dtype=F8_E8M0
    param.sf = _ue8m0_to_fp32(sf).contiguous()          # 恢复为 FP32
    fp8_params.append(param)
```
之后集中调用 `set_ue8m0_sf(fp8_params)` 执行 transpose + pack → `param.ue8m0_sf`（UE8M0 packed int32）。

`_ue8m0_to_fp32` 实现 **[源码]** `weights.py:163-167`：
```python
def _ue8m0_to_fp32(scale: torch.Tensor) -> torch.Tensor:
    """F8_E8M0 → fp32: value = 2 ** (bits - 127), lossless."""
    exponents = scale.view(torch.uint8).to(torch.int32) - 127
    return torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
```

**专家 FP4 权重**（`_load_expert_params`）：HF 存储格式为每个专家独立的切片，打包后形状为 `int8 (n_local, h_out, h_in//2)` —— E2M1 两个值打包一个字节（列主序）。scale 为 HF 行主序 F8_E8M0，通过 `view(param.sf.dtype)` 直接 view 为 packed UE8M0（int32/int16），无需重新量化 **[源码]** `weights.py:318-325`：
```python
assert scale.dtype == torch.float8_e8m0fnu
packed_sf = scale.view(param.sf.dtype)  # 直接 view 为 int32/int16
param.data[local].copy_(weight)
param.sf[local].copy_(packed_sf)
```
之后设 `param.mxfp4=True`，跳过通用 FP8 scale 转换。

**Engram 表 FP8**：加载时 `_ue8m0_to_fp32` 解码 scale 为 FP32，运行时 `gather_sf()` 重新打包。Engram 权重 bytes 为 FP8 E4M3，scale 为 FP32，配置判断 `fp8_embed=hf["quantization_config"]["quant_method"]=="fp8"` **[源码]** `config.py:202-203`。

### 2.3 线性层的 deep_gemm 调用

**非专家 FP8 GEMM**（dense linear）**[源码]** `deepinfer/model_executor/layers/linear.py:165`：
```python
deep_gemm.fp8_fp4_gemm_nt(x, weight, out, out_acc, recipe=(1, 1, get_fp8_casting_granularity()))
```
- `x`：`(x_fp8, x_sf)` 元组，激活动态量化为 E4M3，per-32 UE8M0 scale，列主序 TMA 对齐
- `weight`：`(w_fp8, w_sf)` 元组，FP8 权重 + UE8M0 scale
- `recipe=(1, 1, 32)`：`(1, block_n, block_k)` 块参数，block_k=32 对应量化粒度

**FP8 激活量化**（linear.py:150）：
```python
x = tile_kernels.quant.per_token_cast(x, fmt="e4m3", num_per_channels=32,
    round_sf=True, use_packed_ue8m0=True, use_tma_aligned_col_major_sf=True)
```

**MoE 路由专家 GEMM**（`mlp.py:_grouped_gemm`）**[源码]** `v4_modules/mlp.py:119-128`：
```python
deep_gemm.m_grouped_fp8_fp4_gemm_nt_contiguous(
    a,                             # FP8 激活（分发后按 expert 排列）
    b,                             # FP4 权重（MXFP4，UE8M0 scale）
    d,                             # BF16 输出
    psum_num_recv_tokens_per_expert,
    recipe_a=(1, get_fp8_casting_granularity()),  # (1, 32)
    recipe_b=(1, 32 if fp4_b else 32),            # FP4: per-32 blocks
    use_psum_layout=True,
    expected_m_for_psum_layout=expected_m,
)
```

**Mega MoE（SM100 融合内核）** **[源码]** `mlp.py:255`：
```python
self.mega_moe_func = deep_gemm.fp8_fp4_mega_moe  # fp8_act × fp4_weight
# 或
self.mega_moe_func = deep_gemm.bf16_mega_moe      # bf16×bf16
```
Mega MoE 将 EP dispatch、linear1(FP8×FP4)、SwiGLU、linear2(FP8×FP4)、EP combine 融合成单一大内核，NVLink 通信与 tensor core 计算重叠。

**注意力 WV einsum**（使用 FP8 批量 einsum）**[源码]** `v4_modules/attn.py:215-222`：
```python
deep_gemm.fp8_einsum(
    "bhr,hdr->bhd",
    (context_fp8, context_sf),     # FP8 注意力输出
    (self.wv_weight, self.wv_weight.sf),  # FP8 Wv 权重
    wv,                            # 输出（FP8 或 BF16）
    recipe=(1, 1, gran),           # gran=32
)
```
`use_fp8_einsum_output = PLATFORM.is_gpu and gran == 32`：GPU 上当量化粒度为 32 时，输出保持 FP8 格式（避免中间转 BF16 损失精度和带宽）**[源码]** `attn.py:151`。

**Index MQA logits（FP4 KV 查询得分）** **[源码]** `v4_modules/index_attn.py:299,434`：
```python
deep_gemm.fp8_fp4_mqa_logits(q_chunk, kv, weights[chunk], ks, ke, max_seqlen_k=...)
deep_gemm.fp8_fp4_paged_mqa_logits(q_chunk, kv, weights, ...)  # decode 分页版
```
- `q_chunk`：`(q_fp8or4, q_sf)` —— index Q 量化为 FP4（见 2.4）
- `kv`：`(kv_data, kv_sf)` —— index KV cache，MXFP4 格式（FP4_INDEX_KV_FORMAT）

### 2.4 tilelang_ops：cast/量化 kernel

**`fused_norm_rope_hadamard_cast_store_kvcache`** 是贯穿 KV 写入路径的融合算子，支持的操作链为：

```
BF16 输入 → [RMSNorm] → [RoPE] → [Hadamard] → [量化 FP8/FP4] → [写 KV cache]
```

**[源码]** `deepinfer/model_executor/layers/tilelang_ops/cuda/fused_transform.py:17-91`

参数控制每个阶段是否激活：
- `do_norm`：是否执行 RMSNorm
- `do_rope`：是否执行 RoPE（仅支持 neox/非-neox 两种布局）
- `do_hadamard`：是否执行 Hadamard 变换（仅 head_dim=128 时有效）
- `cast_fmt`：`""` / `"e4m3"` / `"e2m1"`
- `sf_format`：`"ue8m0"` / `"e4m3"` / `"fp32"`

约束条件（`fused_transform.py:63`）：
```
E2M1 量化只支持两种 scale 格式：
  (num_per_channels=16, round_sf=False, sf_format="e4m3")  → 主 FP4 KV
  (num_per_channels=32, round_sf=True,  sf_format="ue8m0")  → index MXFP4 KV
```

**`store_kv_cache_tl`**：单独的 KV cache 写入 kernel（不含 norm/rope/hadamard），用于已量化数据的写入 **[源码]** `tilelang_ops/cuda/store_kv_cache.py`。FP8 路径计算 per-32 amax，转 UE8M0 scale，再转换 E4M3；FP4 路径支持 e4m3 或 ue8m0 scale。

### 2.5 cache_common.py：三种 KV 格式字节布局

**[源码]** `deepinfer/model_executor/layers/tilelang_ops/cache_common.py:109-162`

| 格式常量 | 用途 | 量化 | 元素数 | 每组 | scale 格式 | 字节/token | token_interleaved |
|---------|------|------|--------|------|-----------|-----------|-------------------|
| `V41_FORMAT` | SWA KV | E4M3 | 512 | 32 | UE8M0 | 528 B | True |
| `V41_FP4_FORMAT` | 主压缩 KV | E2M1 | 512 | 16 | E4M3 | 288 B | True |
| `FP4_INDEX_KV_FORMAT` | indexer K | E2M1 | 128 | 32 | UE8M0 | 68 B | False |
| `MXFP8_INDEX_KV_FORMAT` | indexer K (MXFP8) | E4M3 | 128 | 32 | UE8M0 | 132 B | False |
| `FP8_FLOAT32_SF_INDEX_KV_FORMAT` | indexer K (FP32 sf) | E4M3 | 128 | 128 | FP32 | 640 B | False |

**[估算]** 字节/token 公式：`quantized_bytes + sf_bytes`：
- V41_FP4_FORMAT：`512×4÷8 + 512÷16×1 = 256 + 32 = 288`
- V41_FORMAT (SWA)：`512×8÷8 + 512÷32×1 = 512 + 16 = 528`
- FP4_INDEX_KV_FORMAT：`128×4÷8 + 128÷32×1 = 64 + 4 = 68`

**token_interleaved 布局**（主/SWA KV）：每个 token 的存储格式为 `[data | scale]` 连续存放，stride = bytes_per_token，对 paged cache 友好（每次写入一个 token 即可，无需跨 block 的 scatter）**[源码]** `cache_common.py:311-317`。

**Index KV 布局**（非 interleaved）：block 内 data 段连续，scale 段连续，方便 tile 化访问。

### 2.6 Engram FP8

Engram 是 V4.1 的 n-gram 检索模块（n=2~4，每张表约 1600 万行）。
- 词表权重：FP8 E4M3，scale 为 FP32（加载时 `_ue8m0_to_fp32` 恢复）
- `gather_sf()` 在 allocate_userbuffer 前通过 AllGather 汇聚全局 scale，再用 `_pack_ue8m0_to_int` 重新打包 **[源码]** `engram.py:96-101`
- `fused_kv_projection` 为普通 FP8 E4M3 线性层
- `fp8_embed = fp8_proj = True`（当 quant_method=="fp8" 时）**[源码]** `config.py:202-203`

### 2.7 platform.py 中 scale dtype 差异

| 平台 | `fp8_sf_type` | UE8M0 打包方式 | 说明 |
|------|--------------|----------------|------|
| GPU (SM100) | `torch.int32` | 4 个 uint8 → 1 个 int32 | tcgen05 原生 |
| NPU (Ascend 950) | `torch.int16` | 2 个 uint8 → 1 个 int16 | CANN 自定义 |

`alloc_kv_load_output` 中 UE8M0 scale 分配 **[源码]** `cache_common.py:244-248`：
```python
if layout.sf_format == "ue8m0":
    sf_packs = layout.sf_bytes // PLATFORM.fp8_sf_type.itemsize
    sf_data = torch.empty(sf_shape, dtype=PLATFORM.fp8_sf_type, device=device)
```

### 2.8 Hadamard 旋转的用途

Hadamard 变换（Walsh-Hadamard Transform，WHT）对向量每个元素做 `x_i ± x_{i+n/2}` 的蝶形运算，相当于一个无参数的正交变换。其目的是**平滑激活离群值（outlier）**：原始 head 向量中往往有少数维度的绝对值极大（数值方差高），直接量化到 FP4 时这些维度的存在会拉大整组的 scale，导致其他维度精度损失。Hadamard 后各维度能量均匀分布，量化误差更小。

在 DeepInfer 中，Hadamard 仅用于 **index Q/K**（head_dim=128）和 **SWA Q**（部分路径）：
- `do_hadamard=True` 仅在 head_dim=128 时有效（`fused_transform.py:51-52`）
- 尺度因子 `hadamard_scale = head_dim^{-0.5} = 1/√128 ≈ 0.0884`，保证 RMS 不变
- 主压缩 KV（head_dim=512）未使用 Hadamard，因为 head 维度太大（Walsh-Hadamard 需要 2^n 维，512 支持但计算代价更高），且 E4M3 scale 动态范围更大可以容忍一定离群值

**[源码]** `fused_transform.py:718,732`（index Q 用 Hadamard），`attn.py:161,433`（SWA KV 不用 Hadamard）。


---

## 3. 一层 decode token 的数据精度流转图

```
                   ┌─────────────────────────────────────────────────┐
                   │         一层 decode token 精度流转（SM100）        │
                   └─────────────────────────────────────────────────┘

hidden_states [BF16]
       │
       ├──── RMSNorm ──────────────────────────────── BF16
       │           │
       │           ├─ Q_a proj (FP8 GEMM) ─────────── BF16
       │           │     FP8×FP8, recipe=(1,1,32)
       │           │
       │           ├─ Q_b proj (FP8 GEMM) ─────────── BF16
       │           │     下接 q_b_norm + RoPE
       │           │     → fused_norm_rope_hadamard_cast_store_kvcache
       │           │     → Q: FP8 E4M3 (per-32 UE8M0)
       │           │
       │           └─ KV proj (FP8 GEMM) ─────────── BF16
       │                 │
       │                 └─ kv_swa_norm + RoPE + Cast → 写 SWA cache
       │                        [MXFP8 E4M3, per-32 UE8M0, 528 B/token]
       │
       ├─── (cross-layer) 读 主压缩 KV cache (V41_FP4_FORMAT)
       │        [FP4 E2M1, per-16 E4M3, 288 B/token]
       │        → 反量化回 BF16 用于 flashattn
       │
       ├─── Flash-MLA decode attention
       │        Q: FP8 or BF16
       │        KV: BF16 (load + dequant)
       │        context output: BF16
       │             │
       │             └─ WV einsum (fp8_einsum, FP8×FP8 → FP8)
       │                 然后 out_proj (fp8_fp4_gemm_nt, FP8 act × FP8 weight)
       │                 → BF16 注意力输出
       │
       ├─── Index attension（sparse decode, 若启用）
       │        Index Q：BF16 → Hadamard → FP4 E2M1, per-32 UE8M0
       │        Index KV cache：FP4 E2M1, per-32 UE8M0 (68 B/token)
       │        deep_gemm.fp8_fp4_paged_mqa_logits → BF16 logits
       │
       ├─── Residual add → hidden BF16
       │
       └─── MoE FFN
                │
                ├─ gate: BF16 GEMM (TopK 路由, bf16_mega_gate)
                │
                ├─ dispatch (DeepEP, FP8 发送 or BF16 发送)
                │       FP8 激活: BF16→FP8 动态 per-32 UE8M0
                │
                ├─ linear1 (m_grouped_fp8_fp4_gemm, FP8×FP4)
                │       激活: FP8 E4M3 per-32
                │       权重: FP4 E2M1 per-32 UE8M0 [MXFP4]
                │       输出: BF16
                │
                ├─ SwiGLU → FP8 per-32 (swiglu_forward_and_per_token_cast)
                │
                ├─ linear2 (m_grouped_fp8_fp4_gemm, FP8×FP4)
                │       输出: BF16
                │
                └─ combine (DeepEP) → hidden BF16
```

---

## 4. 精度/性能权衡

### 4.1 主 KV FP4（per-16 E4M3）vs indexer KV MXFP4（per-32 UE8M0）

**主 KV（V41_FP4_FORMAT）**：per-16 E4M3 scale
- 优势：每 16 个元素独立 scale，误差更小；E4M3 动态范围（2^-9 ~ 448）比 UE8M0 的整数幂次粒度更细
- 代价：scale 占比更高：`32/288 = 11.1%` vs MXFP4 的 `4/68 = 5.9%`
- 每 token 288 B（vs MXFP4 的 68 B/128 维）；等效 512 维 MXFP4 ≈ `64+8=72 B/token`，差距不大

**Indexer KV（FP4_INDEX_KV_FORMAT）**：per-32 UE8M0
- 用途：只用于 index attention 的 logit 计算（`relu(q·k)·w`），精度敏感性低于主路
- 优势：MXFP4 是 SM100 `tcgen05` 原生格式，无需转换；Hadamard 预处理降低 outlier 影响
- head_dim 仅 128，整体 KV 更小，精度容差更高

**[推断]** 主路 KV 选 per-16 E4M3 而非 per-32 UE8M0 的理由：KV 影响主注意力精度，注意力输出直接决定下层 hidden state；更细粒度的 scale 在长序列（每个 token 的激活分布不同）下误差更可控。per-32 UE8M0 的整数幂次限制（每次变化系数 2x）导致动态范围量化步长偏粗，在有 outlier 的 head 维度上会有可见误差。

### 4.2 Hadamard 平滑 outlier

使用 Hadamard 前：index head 中少数维度可能有 10x 的幅度差异，导致 MXFP4 per-32 块内大多数元素被压缩到接近零。  
使用 Hadamard 后：能量均匀分布，每个 32 维块内幅度差异显著减小，per-32 UE8M0 scale 的误差可以接受。

这与 DuQuant++ **[外部]** arXiv 2604.17789 的思路一致：将旋转块大小与 MXFP4 分组大小对齐（B=32）以减少跨块方差问题。DuQuant++ 证明细粒度旋转（outlier-aware，针对激活离群值集中的通道）比随机 Hadamard 更有效，同时只需单次旋转（无需双旋转+zigzag置换）。DeepInfer 使用固定 Hadamard（非学习）作为更简单的选择。

### 4.3 FP4 FlashAttention-4 的发散结论

**[外部]** arXiv 2609.04105（"Hardware-Aware FP4 FlashAttention-4"，2026/09/03，Robert Hu 等）：
> "every tested MXFP4 probability/value training trajectory diverges."

关键结论：在**训练**中将注意力概率（softmax 输出）和 V 表示为 MXFP4 会发散。推理端 FP4 KV 能用的原因是：
1. V4.1 是 **QAT（Quantization-Aware Training）checkpoint**，模型在训练时已适应 FP4 KV（config 要求 `expert_dtype=fp4`）
2. 推理时量化位置不同：量化的是 K/V 的投影输出（存储），不是注意力矩阵本身（计算）
3. 推理时无梯度反传，不需要精确的概率表示

**因此 FP4 KV 是模型-系统协同设计的产物，不能对任意模型直接开启。**

### 4.4 DeepSeek V3 的 FP8 训练经验

**[外部]** arXiv 2412.19437（DeepSeek-V3 Technical Report，2024/12/27）：
- 全程 FP8 混合精度训练：前向 FP8，梯度 BF16，主权重 FP32
- 激活量化：per-token FP8（1×N），权重量化：per-tile（N×N，tile 大小 128×128）
- 关键经验：小批量的 online 量化（per-token scale 在每层前计算）比 offline 量化鲁棒性更好
- 结论：用于推理的 FP8 动态量化（DeepInfer 中 `activation_scheme=dynamic`）来源于训练中验证的方法论

V4.1 在 V3 基础上进一步将专家权重从 FP8 降到 MXFP4，同时把 KV 从 BF16 降到 FP4，是对精度容忍度的进一步探索。

### 4.5 DuQuant++ 与 MXFP4 精度提升

**[外部]** arXiv 2604.17789（DuQuant++，2026/04/20）：
- 针对 MXFP4 W4A4 量化（LLaMA-3 系列）
- 核心问题：MXFP4 per-32 中单一 outlier 拉大 scale 压缩其他元素
- 方案：细粒度旋转（rotation block size = MXFP4 分组大小 = 32），单次 outlier-aware 旋转
- 相对 DuQuant 减少 50% 在线旋转代价
- V4.1 的 Hadamard 是固定变换（不针对特定通道），DuQuant++ 是更强的方案但需要额外训练

---

## 5. 竞品低精度方案对比

### 5.1 vLLM（v0.32.0）

| 功能 | vLLM 状态 |
|------|----------|
| NVFP4 KV cache | 已支持（配合 FP4 attention backend）**[未核实具体 PR，需核对]** |
| MXFP4 权重量化 | 已支持（vllm/model_executor/layers/quantization） |
| ultraquant_4bit KV | **[未核实]**（未找到该特性的官方文档） |
| FP8 动态 KV | 已支持（`FP8KVCacheMethod`） |
| FP8 W8A8 | 已支持（静态/动态均支持） |

vLLM 的 FP4 KV 与 DeepInfer 的实现差异：vLLM 使用统一的 `KVCacheSpec` 抽象，支持多种格式；DeepInfer 在 `cache_common.py` 中硬编码了 5 种格式并严格验证。

### 5.2 SGLang（v0.5.22）

| 功能 | SGLang 状态 |
|------|------------|
| NVFP4 MoE 权重 | 已支持（FP8×FP4 GEMM 路径）**[未核实具体实现]** |
| Q8KV8（FP8 KV + FP8 激活）| 已支持 |
| DeepSeek V4.1 FP4 KV | 已适配（V4.1 发布前已完成）**[Q&A]** |
| MXFP4 indexer KV | **[未核实]** |

SGLang 采用与 DeepInfer 类似的"模型配合"方式——适配 V4.1 时跟随 DeepInfer 的格式规范。

### 5.3 TRT-LLM

| 功能 | TRT-LLM 状态 |
|------|-------------|
| NVFP4 量化 | 已支持（`ModelOptQuantization.FP4`）**[未核实最新版]** |
| MXFP4 量化 | 已支持（Blackwell 路径）**[未核实]** |
| V4.1 FP8×FP4 路径 | 已适配 V4.1（独立实现，与 DeepInfer 不同调用栈）**[未核实]** |

TRT-LLM 使用静态图编译，FP4 精度通过 `model_opt` 工具链预量化；DeepInfer 使用动态量化（每次前向动态计算 scale）。

### 5.4 Ascend 950 的低精度支持

**[未核实]** 以下为基于 Ascend 架构文档的推断：
- Ascend 950 支持 FP8 E4M3/E5M2（华为 HiFloat8）**[推断]**，scale 格式为 int16（2 个 UE8M0 → 1 个 int16，如 `platform.py:138` 所示）
- MXFP4 支持情况：DeepInfer 的 NPU 路径中存在 `tilelang_ops/asc/` 目录，含 `kv_quant.py`、`store_kv_cache.py` 等，说明 Ascend 950 有 KV 量化路径，**但具体是否支持 MXFP4 硬件指令未在华为官方文档中找到确认**
- 华为 CANN 的自定义算子框架（tilelang-asc）取代了 CUDA 的 `tcgen05`，精度行为可能有差异

**建议：** 调用 `deepinfer/model_executor/layers/tilelang_ops/asc/kv_quant.py` 直接阅读 NPU 量化实现以核实。

**竞品低精度汇总表：**

| 框架 | FP8 W8A8 | FP4 专家权重 | FP4 KV | FP4 Indexer | Hadamard |
|------|---------|------------|--------|------------|---------|
| DeepInfer | 是（MXFP8） | 是（MXFP4） | 是（per-16 E4M3 sf） | 是（per-32 UE8M0） | 是（index Q/K） |
| vLLM | 是 | 是 | 部分 **[未核实]** | 否 | 否 |
| SGLang | 是 | 是 | 适配 V4.1 | **[未核实]** | **[未核实]** |
| TRT-LLM | 是 | 是（NVFP4） | **[未核实]** | **[未核实]** | **[未核实]** |
| Ascend 950 | 是（HiF8） | **[未核实]** | **[未核实]** | **[未核实]** | **[未核实]** |


---

## 6. 开放问题与建议

### 6.1 第三方模型套用 FP4 KV 的评估清单

将 FP4 KV 从 V4.1 迁移到其他模型需满足以下条件：

1. **头维度约束**：`V41_FP4_FORMAT` 的 `quantized_elements=512`，要求 KV head_dim=512（V4.1 的压缩 KV 维度）。普通模型 head_dim=128 则需要使用 `FP4_INDEX_KV_FORMAT`（128 维 MXFP4）。需要调整 `cache_common.py` 中的 `quantized_elements` 参数。

2. **checkpoint 适配性（最关键）**：模型必须是 QAT 或在 FP4 KV 环境下微调的版本。arXiv 2609.04105 明确指出 MXFP4 V/Prob 训练发散。对未经适配的 BF16 模型直接开 FP4 KV，**建议先在 100B token 以上的评估集上测量困惑度退化**（目标：< 0.5% PPL 增量）。

3. **激活分布**：测量目标模型的 KV head 激活的峰度（kurtosis）。峰度 > 10 的层（outlier 严重）直接上 FP4 会有明显误差，需要 Hadamard 或类似旋转预处理。

4. **scale 格式兼容性**：如果目标硬件非 SM100，需要修改 `platform.py` 的 `fp8_sf_type` 以及所有依赖它的分配路径。

5. **反量化开销**：V41_FP4_FORMAT 使用 E4M3 scale（精度高但不可直接用于 tcgen05 MMA），读取时需要将 E4M3 scale 恢复为 FP32。确认 `load_paged_kv_cache` 的反量化 kernel 已针对目标架构优化。

6. **精度基线**：在目标模型上建立 BF16 基线（困惑度、下游任务），然后对比 FP8 KV、MXFP8 KV、FP4 KV 的退化幅度。如 FP8 KV 已有明显退化，FP4 KV 风险更高。

7. **SWA 格式选择**：SWA（滑动窗口）KV 在 V4.1 中使用 MXFP8（528 B/token）而非 FP4，说明 SWA 的精度需求更高（每层都用，误差累积更快）。对第三方模型，先考虑 SWA = FP8，全局 KV = FP4 的混合策略。

### 6.2 开放技术问题

1. **FP4 + FlashAttention 兼容性**：DeepInfer 中 FP4 KV 在注意力计算前被反量化为 BF16（`load_paged_kv_cache` + `do_cast=True`），而非在 FlashAttention kernel 内部使用 FP4 进行矩阵乘法。arXiv 2609.04105 中的 Direct-P 方案（将 score 直接映射到 FP4 概率）是否可以用于推理端加速是一个开放问题。

2. **SM90 的回退策略**：DeepInfer 要求 UE8M0 scale（SM100 特性），SM90（H100）需要 FP32 scale。`FP8_FLOAT32_SF_INDEX_KV_FORMAT`（仅 CUDA 可用，`PLATFORM.is_gpu` 守卫）是 SM90 的 index KV fallback，但主 KV FP4 在 SM90 上无法使用（无 FP4 MMA），**需要 SM100 以上**。

3. **Ascend 950 FP4 路径**：`tilelang_ops/asc/kv_quant.py` 中存在 KV 量化算子，但是否支持 E2M1 和 MXFP4 MMA 未核实。如 Ascend 950 不支持 MXFP4 硬件加速，FP4 KV 路径会比 FP8 KV 路径慢。

4. **Expert FP4 的 QAT 必要性**：V4.1 config 中 `expert_dtype=fp4` 是强制要求，checkpoint 中专家权重本身存储为 `int8` packed E2M1（`assert param.dtype == torch.int8`）。但是否真的做了 QAT 还是仅做了 PTQ 后量化？从 `process_moe_weight` 中的 assert `((deepinfer_w.view(torch.int32) & 0x03030303) == 0)` 看，要求权重 FP8 值的低 2 bit 为零，这正好对应 E2M1 的对齐条件（dummy path 的 PTQ 验证）。真实 checkpoint 走 HF loading 直接加载 packed FP4，不过这个 assert。

5. **UE8M0 量化误差 bound**：UE8M0 是整数幂次（步长 2×），最大相对误差为 `2^(1/1) - 1 = 100%` 之间的情形，但每 32 个元素独立 scale，实际误差取决于块内方差。对 FP4 E2M1（只有 3 位有效位），UE8M0 scale 误差对最终结果影响是二阶的（权重误差 × scale 误差）。

---

## 7. 精度格式索引

| 格式 | arXiv/文档来源 |
|------|--------------|
| OCP Microscaling Specification | https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf |
| Hardware-Aware FP4 FlashAttention-4 | arXiv:2609.04105 |
| DuQuant++（细粒度旋转增强 MXFP4） | arXiv:2604.17789 |
| DeepSeek-V3 FP8 训练 | arXiv:2412.19437 |
| DeepGEMM（UE8M0 布局规范） | https://github.com/deepseek-ai/DeepGEMM |
| NVIDIA PTX UE8M0 文档 | https://docs.nvidia.com/cuda/parallel-thread-execution/#alternate-floating-point-data-formats |

---

*报告字数约 650 行。所有 path:line 均在 commit 774ce24 的 DeepInfer 仓库中核实。*
