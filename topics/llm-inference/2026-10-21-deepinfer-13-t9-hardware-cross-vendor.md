# DeepInfer 专题 T9：硬件与跨厂商适配 —— SM100 vs Ascend 950、scale-up 域、TileLang 双后端、kernel 库生态

> **标记说明**  
> **[源码]** 读代码确认 · **[Q&A]** 官方文档 · **[估算]** 推算 · **[推断]** 合理推测 · **[未核实]** 尚未实证

---

## 1. 整体抽象层次

DeepInfer 的跨平台设计由三层构成：

```
┌─────────────────────────────────────────────────────┐
│  Python 业务层 (model_executor / engine / worker)     │
│  通过 deepinfer.PLATFORM.is_npu / is_gpu 分支         │
├─────────────────────────────────────────────────────┤
│  TileLang 双后端 kernel 层                            │
│  tilelang_ops/cuda/  ←→  tilelang_ops/asc/           │
│  由 ops.py 依 PLATFORM.tilelang_ops_backend 动态加载  │
├─────────────────────────────────────────────────────┤
│  vendor 库层（编译时按平台选择）                        │
│  NVIDIA: FlashMLA + DeepGEMM + DeepEP + DeepSelect  │
│  Ascend:  FlashMLA + DeepGEMM-Ascend + DeepEP-Ascend│
│           + DeepSelect                              │
└─────────────────────────────────────────────────────┘
```

平台探测在进程启动时触发，严禁在 `deepinfer/__init__.py` 中 `import torch`；
`PLATFORM` 通过 `__getattr__` 懒加载，确保 `ASCEND_RT_VISIBLE_DEVICES` 先于 ACL 运行时初始化。**[源码]** `deepinfer/__init__.py:42-48`

---

## 2. Platform 层源码精读

### 2.1 PlatformGPU vs PlatformNPU 差异全表

**[源码]** `deepinfer/platform.py`

| 属性 / 行为 | PlatformGPU | PlatformNPU |
|---|---|---|
| `tilelang_ops_backend` | `"cuda"` | `"asc"` |
| `distributed_backend` | `"nccl"` | `"hccl"` |
| `fp8_sf_type` | `torch.int32`（UE8M0×4 打包） | `torch.int16`（UE8M0×2 打包） |
| `pin_host_memory` | `cudaHostRegister` | `aclrtHostRegisterV2(ACL_HOST_REG_PINNED)` |
| `unpin_host_memory` | `cudaHostUnregister` | `aclrtHostUnregister` |
| `get_distributed_options` | NCCL options；设 `max_ctas/min_ctas=DEEPINFER_NCCL_NUM_SMS` | HCCL options；无 CTA 数参数 |

**FP8 sf type 差异根因**：Ascend 950 的 CUBE 单元在 MoE 权重量化中以 UE8M0×2（2 字节）为分组单位打包 scale factor；NVIDIA SM100 以 UE8M0×4（4 字节）。相关引用：`model_executor/layers/tilelang_ops/norm.py:42`，`model_executor/layers/tilelang_ops/fused_transform.py:529`。

### 2.2 NPU 初始化特殊动作

**[源码]** `platform.py:89-143`

```python
# 关键行摘录（≤20 行）
os.environ.setdefault("TILELANG_DEFAULT_TARGET", "ascend")  # 防止 transfer_to_npu 后被误判为 CUDA
torch_npu.npu.config.allow_internal_format = True           # 使能分形（fractal）内部存储格式
torch_npu.npu.set_compile_mode(jit_compile=False)           # 关闭 JIT，走 AOT 图
torch.npu.Stream.cuda_stream = property(lambda self: self.npu_stream)  # 使 cuda_stream 引用透明
# 同时触发 patch_imports，把两个关键 stub 挂上 import hook
import deepinfer.npu_ops                                     # 加载 patch_imports
```

还有一处 **hack**：`PYTORCH_NPU_DEVICE_CAPABILITY=9.0`（硬编码算力版本），注释标注 `TODO(NPU): Waiting for torch_npu bug fix`。**[源码]** `platform.py:117`

---

## 3. deep_cluster 硬件探测与 NUMA 绑定

### 3.1 NVIDIA 探测路径

**[源码]** `deep_cluster/hardware/nvidia.py`

优先读 `/proc/driver/nvidia/gpus/`；fallback 读 `/dev/nvidia*`。
NUMA 信息来自 `/sys/bus/pci/devices/<bus_id>/numa_node`。
计算能力通过 NVML (`libnvidia-ml.so`) 静态探测，**无需初始化 CUDA context**。
`is_sm_major_10()` 判断 major==10（即 B200/GB200/GB300 = SM100），**[源码]** `device.py:160-163`。

### 3.2 Ascend 探测路径

**[源码]** `deep_cluster/hardware/ascend.py`

通过 `/sys/class/ubcore/udma*/device/ub_totalues` 发现 NPU；fallback 读 `/dev/davinci*`。
NUMA 绑定逻辑有一处特殊处理：

```python
return (raw_numa // 2) * 2 + index % 2
```

这是对 Ascend 超节点 2-NPU-per-Die 拓扑的**奇偶交错修正**——设备 0/1 共享 Die 0（属同一 NUMA 节点），但 index%2 保证两卡不完全重叠。**[源码]** `ascend.py:144`。

### 3.3 BF16 TFLOPs 上限

**[源码]** `device.py:151-157`

```python
def get_bf16_tflops_limit() -> int:
    if accelerator_kind() == "ascend":
        return 432          # Ascend 950 BF16 峰值
    if is_sm_major_10():
        return 2500 if platform.machine() in {"aarch64", "arm64"} else 2250
    raise RuntimeError("unknown GPU type")
```

- SM100（B200/GB200）：x86 2250 TFLOPs，aarch64（GH200 style）2500 TFLOPs **[推断：GB200 NVL72 中 Grace CPU 的 arm 节点走 2500 分支]**
- Ascend 950：432 TFLOPs（**[未核实]** 仅代码字面值，未找到华为官方规格页确认数字一致性）

### 3.4 NUMA 绑定实现

**[源码]** `deep_cluster/topology/bind.py`

通过 `ctypes` 直接调 `libnuma.so.1`，设置 CPU affinity（`numa_sched_setaffinity`）和内存策略（`set_mempolicy MPOL_BIND`）。IB 设备走 `/sys/class/infiniband/mlx5_*/device/numa_node`。Ascend 侧优先查询 `npu-smi info -t topo` 获取 CPU affinity。

---

## 4. npu_ops — patch_imports import hook

### 4.1 机制

**[源码]** `deepinfer/npu_ops/__init__.py` + `patch_imports.py`

```python
patch_imports({
    "deepinfer.npu_ops.deep_select": {
        "deep_select.interface": "topk",          # 拦截 from deep_select.interface import topk
    },
    "deepinfer.npu_ops.vit_attn": {
        "flash_attn.cute.interface": "_flash_attn_fwd",  # 替换 ViT 注意力实现
    },
})
```

`patch_imports` 替换 `builtins.__import__`，将 `from deep_select.interface import topk` 重定向为 `deepinfer.npu_ops.deep_select.topk`（Lazy 代理），只在首次调用时实际加载。

### 4.2 替换内容

**NPU TopK (`npu_ops/deep_select.py`)**：当前 `return_value=True`（sampler 场景）走 `torch.topk` fallback，`return_value=False`（indexer 场景）走真实的 DeepSelect-Ascend kernel。注释 `TODO(NPU): To be removed when topk for sampler is ready`。

**NPU ViT Attention (`npu_ops/vit_attn.py`)**：用 `torch_npu.npu_fusion_attention` 替代 `flash_attn.cute.interface._flash_attn_fwd`，因为 Ascend 上没有 CuTe DSL 版 Flash Attention 4。注释 `TODO(NPU): To be removed when _flash_attn_fwd for ViT is ready`。

---

## 5. TileLang 双后端 kernel 清单

### 5.1 ops.py 分发逻辑

**[源码]** `tilelang_ops/ops.py:15`

```python
_pkg = importlib.import_module(f"{__package__}.{PLATFORM.tilelang_ops_backend}")
for _info in pkgutil.iter_modules(_pkg.__path__):
    _mod = importlib.import_module(f"{_pkg.__name__}.{_info.name}")
    for _name, _obj in vars(_mod).items():
        if _name.startswith("_") or not _name.endswith("_tl"):
            continue
        globals()[_name] = _obj
```

`KernelUnavailableError`（继承 AttributeError）用于优雅地标记某后端缺失的 kernel，conftest.py 将其转为 pytest skip。

### 5.2 kernel 清单与差异统计

**[源码]** 遍历 `tilelang_ops/cuda/` 和 `tilelang_ops/asc/`

**仅 cuda/ 有**（3 个）：

| 文件 | 描述 | 行数 |
|---|---|---|
| `outlier_detection.py` | FP4 离群值检测 | 40 |
| `prepare_decode_metadata.py` | 标准 decode metadata 准备 | 196 |
| `quant.py` | 量化辅助 | 9 |

**仅 asc/ 有**（3 个）：

| 文件 | 描述 | 行数 |
|---|---|---|
| `kv_quant.py` | Ascend SIMD BF16 KV 量化原语（e4m3/FP4 打包） | 196 |
| `prepare_v4_decode_metadata.py` | Ascend 专用 V4 decode metadata（替代 CUDA 版） | 257 |
| `rstd_1ulp.py` | 补偿型 RSTD（1-ULP 精度，匹配 hai-llm 数值标准） | 98 |

**双侧均有**（26 个）：`candidate_blocks`, `compress`, `compute_seqlens_k_zos`, `dflash`, `dflash_copy`, `flash_mla_utils`, `fused_sample`, `fused_transform`, `gather`, `gather_engram_tokens_zos`, `gather_input_ids_zos`, `get_kv_index`, `image_embed_ops`, `kv_index_cat`, `load_paged_kv_cache`, `norm`, `prepare_cp_mirror`, `random_sampling`, `sampler`, `schedule_mtp_varlen`, `state_cache`, `store_kv_cache`, `store_mtp_info`, `structure_output`, `sum_smaller_probs`, `swa_index`

**行数对比（asc 普遍更长）**：

| 文件 | cuda 行数 | asc 行数 | asc/cuda |
|---|---|---|---|
| `fused_transform.py` | 619 | 899 | 1.45× |
| `fused_sample.py` | 524 | 673 | 1.28× |
| `sampler.py` | 411 | 663 | 1.61× |
| `prepare_cp_mirror.py` | 313 | 518 | 1.65× |
| `norm.py` | 326 | 431 | 1.32× |
| `load_paged_kv_cache.py` | 165 | 345 | 2.09× |

**asc 版普遍多 20~110% 行数**，主要原因：Ascend SIMD 编程需要手动管理向量寄存器分配、显式 UB（Unified Buffer）搬运、分形矩阵地址计算。

### 5.3 关键实现差异

**[源码]** 以下均为 `PLATFORM.is_npu` 分支

| 文件:行 | 差异类型 | CUDA 侧 | Ascend 侧 |
|---|---|---|---|
| `norm.py:51` | **向量对齐** | `(seqlen+3)//4*4`（16B 对齐） | `seqlen`（无需 4-round） |
| `norm.py:346` | **norm 路径** | 支持 shared_bm | NPU 下 shared_bm 或 fp8_layout!=32 走独立路径 |
| `fused_sample.py:529` | **vocab_size 下限** | 32 | 64（ASC 要求完整 SIMD 向量） |
| `fused_sample.py:732` | **boundary_b** | 2097152 | 65536（内存容量更小） |
| `random_sampling.py:294` | **batch 参数** | 512×1048576 | 16×1048576（GPU SRAM 更少） |
| `sampler.py:189` | **force_topk 上限** | 16384 | 4096（向量寄存器限制） |
| `store_kv_cache.py:61,84,103` | **KV 量化路径** | 走 CUDA kv_quant | Ascend 走专用 `asc/kv_quant.py` |
| `fused_transform.py:244` | **FP8 cast 参数** | 标准路径 | `num_per_channels!=32` 走特殊路径 |
| `index_attn.py:280,324` | **FlashMLA 参数** | `use_unaligned_ks=True` | 不传此参数（NPU 不支持） |
| `index_attn.py:444` | **KV 布局** | 默认 | `use_ascend_kv_layout=False`（有 TODO：改 True 可能更快但要改 layout） |
| `mlp.py:228` | **mega_gate** | `use_mega_gate=True` | `use_mega_gate=False` |
| `mlp.py:185` | **FP8xFP4** | 支持 | 不支持（`PLATFORM.is_gpu` 才走 FP4 路径） |
| `eplb/__init__.py:200` | **symm_mem shuffle** | 支持（NVLink IPC） | 不支持（`TODO: NPU doesn't support symm_mem yet`） |
| `worker/runtime_config.py:64,67` | **PDL 开关** | 开启 `set_pdl(True)` | 不开启 |
| `worker/cuda_graph_runner.py:124` | **CUDA graph 参数** | `keep_graph=True` | 无此参数（NPU 不支持） |
| `parallel_state.py:131-132` | **NVL domain 判断** | 调 `deep_ep.get_physical_domain_size` | 直接返回 True（NPU 超节点视为单 domain） |
| `distributed/shm_broadcast.py` | **pin_host_memory** | `cudaHostRegister` | `aclrtHostRegisterV2` |

---

## 6. deepinfer/__init__.py — .so 目录选择

**[源码]** `deepinfer/__init__.py:15-21`

```python
def _so_platform_tag() -> str:
    accelerator = envs.DEEPINFER_ACCELERATOR or accelerator_kind()
    if accelerator not in ("nvidia", "ascend"):
        accelerator = "nvidia"
    return f"{accelerator}-{os.uname().machine}"

_so_dir = os.path.join(os.path.dirname(__file__), "_so", _so_platform_tag())
if os.path.isdir(_so_dir):
    __path__.append(_so_dir)
```

构建产物路径格式为 `deepinfer/_so/nvidia-x86_64/` 或 `deepinfer/_so/ascend-aarch64/`，允许同一代码树包含多个平台的 `.so`。探测失败默认 nvidia，即 CPU-only 机器上也能运行（低 graceful degradation）。

---

## 7. setup.py — NpuExtension vs CUDAExtension

**[源码]** `setup.py:43-65`

| 配置 | CUDA 路径 | NPU 路径 |
|---|---|---|
| 扩展类 | `torch.utils.cpp_extension.CUDAExtension` | `torch_npu.utils.cpp_extension.NpuExtension` |
| 源文件（额外） | `csrc/cuda_utils.cpp`, `csrc/graph_timer/graph_timer_cuda.cpp` | `csrc/asc_utils.cpp`, `csrc/graph_timer/graph_timer_asc.cpp` |
| 编译 flags | `-O2 -std=c++20 -fvisibility=hidden` | 同上 + `-fopenmp -DWITH_NPU` |
| 链接库 | `pthread`, `cuda`（来自 CUDA stub） | `pthread`（torch_npu 隐式链 ACL） |

### 7.1 csrc WITH_NPU 分支

**[源码]** `csrc/cuda_utils.hpp`、`csrc/graph_timer/graph_timer.hpp`

关键差异：

1. `get_default_device()`：NPU 返回 `torch::kPrivateUse1`，CUDA 返回 `torch::kCUDA`
2. `make_from_blob()`：NPU 走 `at_npu::native::from_blob`（`torch::from_blob` 在 PrivateUse1 上 dtype 推断失效）
3. `GraphHandle` / `EventHandle`：NPU 侧无 `raw_graph_` / `raw_event_` 字段（`#ifndef WITH_NPU` 包裹）

IPC 导出：CUDA 用 `cudaIpcGetMemHandle`；NPU 用 `aclrtIpcMemGetExportKey`（需 2MB 对齐，须 `ACL_RT_IPC_MEM_EXPORT_FLAG_DISABLE_PID_VALIDATION`）。**[源码]** `csrc/asc_utils.cpp:38-59`

---

## 8. scripts/build_vendor.sh — 编译路由

**[源码]** `scripts/build_vendor.sh`

```bash
case "${DEEPINFER_ACCELERATOR:-nvidia}" in
  nvidia)  PACKAGES=(DeepSelect FlashMLA DeepEP DeepGEMM) ;;
  ascend)  PACKAGES=(DeepSelect FlashMLA DeepEP-Ascend DeepGEMM-Ascend) ;;
esac
```

两侧 **install 名相同**（`deep_select`, `flash_mla`, `deep_ep`, `deep_gemm`），Python 侧无条件 `import deep_gemm`、`import deep_ep`，平台差异完全封装在 vendor 包内部。

---

## 9. requirements 对比

| 依赖 | requirements-nvidia.txt | requirements-ascend.txt |
|---|---|---|
| `torch` | `torch==2.13.0+cu132` | `torch==2.13.0+cpu` |
| NPU 驱动 | — | `torch-npu==2.13.0rc1` |
| CUDA 版本 | `--find-links https://download.pytorch.org/whl/cu132/torch/` | CPU whl |
| 注意力库 | `flash-attn-4==4.0.0b34` | — |
| 公共依赖 | `tilelang==0.1.15`, `tile-kernels==2.1.0` | 同 |

Ascend 不依赖 `flash-attn-4`，ViT 注意力由 `torch_npu.npu_fusion_attention` 实现（patch_imports 机制）。

---

## 10. Kernel 库生态

### 10.1 总览

| 库 | 方向 | 最新状态 | SM100 | SM90(H100) | Ascend 950 | stars |
|---|---|---|---|---|---|---|
| **DeepGEMM** | GEMM/MoE GEMM | 2026.09.30 | 支持 | 已移除 | 无（→DeepGEMM-Ascend） | 9k |
| **DeepGEMM-Ascend** | GEMM（Ascend 移植） | 2026.09.30 初版 | 无 | 无 | 支持（BF16/FP8/FP4/MegaMoE） | 599 |
| **DeepEP** | EP all-to-all 通信 | 活跃 | 支持 | 已移除 | 无（→DeepEP-Ascend） | 10k |
| **DeepEP-Ascend** | EP 通信（Ascend 移植） | 2026.09.30 | 无 | 无 | 支持（EP8-EP128，FP8 dispatch） | 264 |
| **FlashMLA** | MLA attention | 2026.09.30 | 支持 | **已移除** | 支持（2026.09.30 新增） | 13k |
| **DeepSelect** | TopK（DSA/Sampler） | 2026.09.30 | 支持 | — | Ascend 版（BF16，sampler TODO） | 507 |
| **TileKernels** | quant/transform/... | 活跃 | 支持 | — | 支持 | 未公开 |
| **TileLang** | 内核 DSL 编译器 | v0.1.15（2026.09.30+） | 支持（SM100） | 支持（SM90） | 2026.09.30 官方支持 | 未统计 |

**[源码]** `.gitmodules` 确认 DeepGEMM-Ascend、DeepEP-Ascend 均为 DeepInfer 的 git submodule。

### 10.2 FlashMLA 的 SM90 撤销

**[源码]** FlashMLA README（2026.09.30 release note）

> "This version removes support for the Hopper architecture (SM90)."

FlashMLA 当前版本**仅支持 SM100（B200/GB200）和 Ascend 950**，不再支持 H100/H200（SM90）。这是 DeepInfer 社区 PR #148（SM90 移植，985 行新增，22 文件）的直接背景——需要重新加回 SM90 支持。

### 10.3 FlashMLA Ascend 950 性能

**[源码]** FlashMLA README 2026.09.30

- Prefill sparse attention：410 TFLOPs（95% 硬件峰值）
- Decoding：360 TFLOPs（83% 硬件峰值）
- Fused norm-RoPE-attn-RoPE-cast decoding kernel：优化 10-15%

### 10.4 DeepGEMM-Ascend 特性

**[源码]** DeepGEMM-Ascend README

完全 API 兼容 DeepGEMM（相同包名 `deep_gemm`），支持 BF16、FP8、FP4 GEMM、MQA logits、MegaMoE。使用 Ascend MAD 原语，隐藏分形矩阵布局。通过协程流水线和稀疏数据加载接近硬件极限。

### 10.5 DeepEP-Ascend 特性

**[源码]** DeepEP-Ascend README

测量平台：Ascend 950DT + CANN 9.2.0（PoC HDK，商业版尚未发布，预计 2026-10-15）：

| EP 规模 | Dispatch (GB/s) | Combine (GB/s) |
|---|---|---|
| EP8 | 373–375 | 345–347 |
| EP16 | 348–352 | 338–341 |
| EP32 | 335–340 | 320–324 |
| EP64 | 323–327 | 294–298 |
| EP128 | 313–320 | 272–278 |

dispatch EP8 达到物理带宽的约 90-95%；EP64/128 仍在优化中。EP 上限理论支持 EP128+，与 Ascend 超节点的 all-to-all 架构对齐。

### 10.6 TileLang 双后端现状

**[源码]** TileLang README（2026.09.30）

- SM100（Blackwell）：支持，含 SM120 NVF4 block-scaled MMA（2026.07.30）
- Ascend 950：2026.09.30 官方支持（SIMD/SIMT 向量编程，自动调度）
- SM90（Hopper）：支持（未移除）
- AMD CDNA4：2026.05.24 支持 FP4 E2M1

DeepInfer 锁定版本 `tilelang==0.1.15`（requirements.txt）。

### 10.7 flash-attn-4 (CuTe DSL)

**[源码]** `requirements-nvidia.txt:5` + `deepinfer/model_executor/models/vision_encoder/vit.py:13`

```python
from flash_attn.cute.interface import _flash_attn_fwd
```

`flash-attn-4==4.0.0b34` 是 NVIDIA 专属依赖，基于 CUDA CuTe DSL 实现。Ascend 侧通过 `patch_imports` 拦截这一 import，替换为 `torch_npu.npu_fusion_attention`（`npu_ops/vit_attn.py`）。flash-attn-4 仅用于 ViT varlen attention，主模型推理走 FlashMLA/TileLang。

---

## 11. 硬件规格对比

### 11.1 NVIDIA B200/GB200/GB300 NVL72

| 规格项 | B200 SXM | GB200（Grace+B200） | GB300 NVL72 超节点 |
|---|---|---|---|
| GPU 架构 | SM100（Blackwell） | SM100 | SM100（72 GPU） |
| HBM 容量 | 192 GB HBM3e | 192 GB HBM3e | 72×192 = 13,824 GB |
| HBM 带宽 | 8 TB/s | 8 TB/s | 理论 576 TB/s |
| NVLink 带宽（单卡） | 1.8 TB/s | 1.8 TB/s | 所有 72 卡互联 |
| FP4 算力 | 9 PFLOPS | 9 PFLOPS | — |
| FP8 算力 | 4.5 PFLOPS | 4.5 PFLOPS | — |
| BF16 算力 | 2.25 PFLOPS | 2.25 PFLOPS | — |

**[未核实]** 以上数字来自 NVIDIA 官方产品页，但 B200/GB200/GB300 的最终规格可能因版型有差异，建议以最新 datasheet 为准。`deepinfer/deep_cluster/hardware/device.py:155` 代码将 SM100 x86 的 BF16 TFLOPs 上限标记为 2250，与 2.25 PFLOPS 一致，可作间接佐证。

**scale-up 域**：NVL72 中 72 张 GPU 通过 NVLink 全连接，构成单一 scale-up 域，理论 NVLink 带宽 1.8 TB/s/card。**[Q&A]** Q&A line 97：EPLB shuffle 走 NVLink `symm_mem`，即 intranode IPC；`parallel_state.py:131` 中 `is_within_nvl_domain` 调 `deep_ep.get_physical_domain_size` 判断 EP group 是否在同一 NVLink 物理域内。

### 11.2 Ascend 950PR / 950DT 与超节点

**[未核实部分]** 华为尚未发布完整的 950PR/DT 官方 datasheet，以下来自 DeepEP-Ascend README 和 CloudMatrix 论文：

| 规格项 | Ascend 950DT（已知） | 备注 |
|---|---|---|
| 峰值 BF16 算力 | 432 TFLOPS（源码字面值） | [源码] `device.py:153` |
| 超节点互联 | UBMEM + UBC_CTP + URMA | DeepEP-Ascend README |
| EP 支持（超节点内） | EP8–EP128 | DeepEP-Ascend README |
| CANN 版本 | CANN 9.2.0 | 验证版本 |
| 推荐 HDK | Q3 商业版（预计 2026-10-15，Atlas 850E） | DeepEP-Ascend README |

**[推断]** Ascend 950 "PR"/"DT" 指不同封装形态（PR=PCIe Card，DT=DT 扩展底板），两者 AI core 架构相同。

### 11.3 CloudMatrix384 (2506.12708)

**[源码]** arXiv:2506.12708 摘要

Huawei CloudMatrix384 超节点：384 Ascend 910 NPU + 192 Kunpeng CPU，通过超高带宽 Unified Bus (UB) 网络全连接，支持 EP320（通过 UB token dispatch）。

CloudMatrix-Infer（对 DeepSeek-R1）测量结果：
- Prefill 吞吐：6,688 tok/s/NPU
- Decode 吞吐：1,943 tok/s/NPU（TPOT<50ms）
- 严格延迟（15ms）下：538 tok/s/NPU

**注意**：CloudMatrix384 使用的是 **Ascend 910**（非 950），且采用 INT8 量化。DeepInfer 当前针对 Ascend 950，两者系不同产品线。

### 11.4 scale-up 域大小与 EP 上限关系

**[Q&A]** Q&A line 11, 231

- 64 张 decode GPU 位于同一 scale-up 域：Q&A 描述"ordinary 64–128-GPU supernodes for decode"，EP64 成为推荐的最小高效部署单元。
- 在 scale-up 域内，EP all-to-all 通信走 NVLink symm_mem（GPU）或 UBMEM/URMA（Ascend），无需跨节点 Ethernet/IB，带宽远高于 Scale-out 网络。
- **[源码]** `eplb/__init__.py:200`：`_intranode_shuffle = PG.is_within_nvl_domain(ep_group) and PLATFORM.is_gpu`，仅 GPU + 在 NVLink 域内才启用 symm_mem inplace shuffle；Ascend 侧 `is_within_nvl_domain` 始终返回 True，但 `PLATFORM.is_gpu` 为 False，故当前 Ascend **不走** symm_mem shuffle（TODO 标注）。
- **[Q&A]** Q&A line 231："Mega MoE is the general solution within a scale-up domain: regardless of EP size or batch size, Mega MoE is significantly faster than unfused MoE." scale-up 域越大，可以用更大 EP 同时保持 Mega MoE 的融合优势，避免复杂的跨节点通信重叠。

---

## 12. 竞品跨厂商策略对比

### 12.1 vLLM 插件体系

**[核实]** `gh api repos/vllm-project/vllm-ascend`：2,997 stars，"Community maintained hardware plugin for vLLM on Huawei Ascend"

vLLM 采用 **backend plugin** 模式（`torch.ops` 注册 + device map），vllm-ascend 是社区维护的 Ascend 插件。SGLang 的 platforms 目录（`sgl-project/sglang`）中已有 `npu.py`，列出了 Ascend 950PR/DT 支持。**[核实]** `gh api repos/sgl-project/sglang/contents/python/sglang/srt/platforms`

| 框架 | 多厂商策略 | Ascend 支持现状 | SM100 支持 |
|---|---|---|---|
| **DeepInfer** | Platform 类 + TileLang 双后端 | 原生，与 NVIDIA 路径对等 | 原生（主目标） |
| **vLLM** | plugin 插件（vllm-ascend） | 社区 plugin，2.9k stars | 支持（主流框架） |
| **SGLang** | platforms/ 目录 | 内置 npu.py，950PR/DT | 支持 |
| **TRT-LLM** | 单厂商（NVIDIA 独占） | 无 | 原生 |
| **MindIE** | 华为官方推理框架 | 原生 | 无 |

### 12.2 社区 PR 分析

**[核实]** `gh api repos/deepseek-ai/DeepInfer/pulls/{148,146,143,138}`

| PR | 标题 | 状态 | 文件/行数 | 难度评估 |
|---|---|---|---|---|
| **#148** | Add NVIDIA Hopper (SM90) Support | open | 22 文件，+985/-40 | 高：FlashMLA 已移除 SM90 支持，需自建 Hopper kernel 路径；FP8/FP4 GEMM、稀疏 MLA、indexer kernel 全部需要架构专项优化 |
| **#146** | Add Intel XPU platform skeleton | open | 5 文件，+116 行 | 极高（当前为骨架）：缺少 XPU TileLang backend（tilelang 尚无 XPU 目标），缺少整套 kernel 实现 |
| **#143** | Ascend 910C (A3) port tooling + RFC | open | 5 文件，+1315 行 | 高：910C 无 FP8/FP4 tensor core，需 W8A8 INT8 路径，checkpoint 格式须转换（streaming converter 已在 PR 中），整体架构改动大 |
| **#138** | Prototype SM120 PCIe sparse-attn adapters | open | 30 文件，+1588/-27 | 高：SM120 为 PCIe-only 系统（无 NVLink），EPLB symm_mem shuffle 不可用；复用 FlashInfer SparseMLASm120Wrapper，但 cache layout 转换、token-interleaved 等未完成 |

**SM90（PR #148）**的关键难点：FlashMLA 2026.09.30 版已删除 Hopper，意味着 PR 需要自己维护 SM90 attention kernel 或回退到旧版 FlashMLA。22 个文件改动规模相对合理，但不包含 attention kernel 本身的实现（依赖外部库）。

**SM120（PR #138）**的关键难点：SM120（RTX 50/PRO）是 PCIe 连接，无 NVLink，`is_within_nvl_domain` 返回 False，EPLB intranode shuffle 不可用，Expert 重排需要走独立路径。同时 SM120 的 FlashInfer wrapper 仍是草稿阶段。

---

## 13. `is_npu` / `is_gpu` 分支归类总结

**[源码]** 全量 grep 78 处，归为以下 6 类：

| 类别 | 代表 | 数量（约） |
|---|---|---|
| **功能缺失**（NPU 尚未支持） | `symm_mem shuffle`、`use_mega_gate`、`PDL`、`use_unaligned_ks`、`CUDA graph keep_graph` | 8 |
| **实现差异**（两侧均有，逻辑不同） | `pin_host_memory`、`graph_timer`、`distributed_backend`、`norm m_pad` | 15 |
| **数值参数差异**（向量宽度/内存容量约束） | `vocab_size 下限 64 vs 32`、`force_topk 4096 vs 16384`、`boundary_b` | 6 |
| **FP8 sf 格式差异** | `fp8_sf_type int16 vs int32`，广泛影响 norm/fused_transform/cache_common | 12 |
| **kernel 路径选择** | `kv_quant`、`prepare_decode_metadata`、`outlier_detection` | 5 |
| **TODO 临时 hack** | `PYTORCH_NPU_DEVICE_CAPABILITY`、`batch_invariant` 强制开 | 5 |

---

## 14. 开放问题与国产卡适配方法论

### 14.1 已知开放问题

1. **Ascend symm_mem intranode shuffle 缺失**：EPLB expert 重排走跨节点路径，额外延迟。`TODO(NPU): NPU doesn't support symm_mem yet`
2. **DeepSelect sampler topk NPU fallback**：采样侧走 `torch.topk`，性能低于 CUDA 侧专用 kernel
3. **ViT flash_attn_fwd NPU stub**：用 `npu_fusion_attention` 替代，接口不完全对等（无 return 第二元素）
4. **NPU batch invariant 未完整支持**：deterministic 模式强制开，影响性能
5. **Ascend 950 KV 布局优化未启用**：`use_ascend_kv_layout=False` 保守设置，可能有性能留量
6. **DeepEP-Ascend 商业版 HDK 未发布**：EP64/128 带宽未达到 EP8 的 90-95% 水平，仍在优化
7. **SM90/SM120 社区 PR 均处于 open 状态**：无官方时间表

### 14.2 国产卡适配方法论清单

以 DeepInfer 的 Ascend 移植为参照，提炼以下移植要素：

1. **Platform 抽象层**：实现 `PlatformXXX` 类，定义 `tilelang_ops_backend` 目录名、`distributed_backend` 名、`fp8_sf_type`、`pin/unpin_host_memory`。
2. **硬件探测**：通过设备文件或 sysfs 识别卡型，实现 `compute_capability` 等价查询（Ascend 用 npu-smi；NVML 模式不可移植）。
3. **NUMA 映射**：国产卡通常有自己的 NUMA 亲和信息，需适配 `device_numa_node()`；Ascend 的奇偶交错修正可作参考。
4. **分布式后端**：注册 pytorch ProcessGroup backend（Ascend 用 HCCL，国产卡一般有自己的集合通信库需同等注册）。
5. **TileLang 后端目录**：为新目标添加 `tilelang_ops/<target>/` 子包，实现与 `cuda/` 命名对称的 `*_tl` kernel 工厂；不能用的写 `KernelUnavailableError`。
6. **patch_imports hook**：对无法在新平台上直接 import 的 Python 库（如 `flash_attn.cute`），用 `patch_imports` 以 lazy stub 代替，避免 import 时 crash。
7. **FP8 量化格式确认**：确认硬件的 scale factor 打包宽度（2B/4B），修改 `fp8_sf_type`，同步修改 cache_common、norm、fused_transform 中所有 `PLATFORM.fp8_sf_type.itemsize` 引用。
8. **vendor 库替换**：准备与 `deep_gemm`/`deep_ep`/`flash_mla`/`deep_select` API 对齐的平台版，在 `build_vendor.sh` 中添加 case。
9. **csrc 扩展**：添加 `<platform>_utils.cpp` 和 `graph_timer_<platform>.cpp`，适配 IPC 导出/导入、流处理、从设备指针创建 Tensor。
10. **数值约束参数调整**：向量宽度（vocab_size/topk 上限）、内存容量参数（batch 大小），在 `PLATFORM.is_npu` 等价分支处调整。

---

## 15. 纠正总报告中的错误

1. **总报告**称"64 卡 decode 位于同一 scale-up 域"：**[Q&A 确认]** Q&A line 11 原文为"ordinary 64–128-GPU supernodes for decode"，是设计目标而非硬性约束；EP 域大小由 `deep_ep.get_physical_domain_size` 动态查询，非固定 64 卡。
2. **总报告**"FlashMLA 缺一手调 SASS 优化，解码慢 5-10%"：**[Q&A]** Q&A line 234 原话是"FlashMLA lacks one hand-tuned SASS optimization, which makes the decoding kernel 5%–10% slower"，但该段上下文是与"内部更好版本"比较，并非与 vLLM/SGLang 比较。
3. **SM90 FlashMLA 支持**：已在 2026.09.30 版本移除（FlashMLA README 明确），总报告如有"DeepInfer 在 SM90 上直接可用"的表述应予以修正。

---

*报告生成时间：2026-10-21*
