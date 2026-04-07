# LLM Architecture

- **Date:** 2026-02-09
- **Tags:** LLM, Transformer, attention, RoPE, normalization, activation, scaling-law, LoRA, quantization

## Context

Comprehensive notes on large language model architecture fundamentals, covering attention mechanisms, normalization techniques, positional encoding, activation functions, loss functions, model variants, fine-tuning methods, quantization, scaling laws, training practices, evaluation, and related topics.

## Main Content

---

# 一、Attention 机制

## 1.1 Multi-Head Attention

计算公式：MultiHead(Q,K,V) = Concat(head_1,...,head_h)W^O, head_i = Attention(QW^Q_i, KW^K_i, VW^V_i), Attention(Q,K,V) = softmax(QK^T/√d)V

时间复杂度：O(n²·d), 空间复杂度：O(n²·d), 参数量：4·n·d²

```python
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn
```

完整 MultiHeadedAttention 类代码：
```python
class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        del query, key, value
        return self.linears[-1](x)
```

## 1.2 Attention 优化

### KV-Cache
Decoder 每次前向时，之前 timestep 的 KV 值都计算过但被丢掉。KV-Cache 保留历史 K/V 用于后续计算。计算开销从 O(n²) 变为 O(n)，但长序列存储瓶颈严重。

### Multi-Query Attention (MQA)
Q 保持原来头数，K 和 V 只有一个头，所有 Q 头共享一组 K/V。提高 30%-40% 吞吐但影响效果。

```python
class MultiQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, device=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        # 关键：只创建 query 的 head 向量(d_model)，K/V 各只有一个 head_dim
        self.Wqkv = nn.Linear(d_model, d_model + 2 * self.head_dim, device=device)
        self.out_proj = nn.Linear(self.d_model, self.d_model, device=device)

    def forward(self, x):
        qkv = self.Wqkv(x)  # (1, 512, 960)
        query, key, value = qkv.split(
            [self.d_model, self.head_dim, self.head_dim], dim=2
        )  # query->(1,512,768), key->(1,512,96), value->(1,512,96)
        context, attn_weights, past_key_value = self.attn_fn(
            query, key, value, self.n_heads, multiquery=True)
        return self.out_proj(context), attn_weights, past_key_value
```

### Group-Query Attention (GQA)
分组 Q 头共享一组 KV。GQA 完整实现（使用 einops）：

```python
def scaled_dot_product_gqa(query, key, value, dropout=0.0, scale=None, mask=None,
                           is_causal=None, need_weights=False, force_grouped=False):
    query = rearrange(query, "b n h d -> b h n d")
    key = rearrange(key, "b s h d -> b h s d")
    value = rearrange(value, "b s h d -> b h s d")

    bq, hq, nq, dq = query.shape
    bk, hk, nk, dk = key.shape

    if scale is None:
        scale = query.size(-1) ** 0.5
    query = query / scale

    num_head_groups = hq // hk
    if num_head_groups > 1 or force_grouped:
        query = rearrange(query, "b (h g) n d -> b g h n d", g=num_head_groups)
        similarity = einsum(query, key, "b g h n d, b h s d -> b h n s")
    else:
        similarity = einsum(query, key, "b h n d, b h s d -> b h n s")

    if is_causal:
        mask = torch.ones((bq, nq, nk), device=query.device, dtype=torch.bool).tril_()
    if mask is not None:
        if mask.ndim == 2:
            mask = rearrange(mask, "b s -> b () () s")
        elif mask.ndim == 3:
            mask = rearrange(mask, "b n s -> b () n s")
        similarity.masked_fill_(~mask, torch.finfo(similarity.dtype).min)

    attention = F.softmax(similarity / scale, dim=-1)
    if dropout > 0.0:
        attention = F.dropout(attention, p=dropout)

    out = einsum(attention, value, "b h n s, b h s d -> b h n d")
    out = rearrange(out, "b h n d -> b n h d")
    return out, attention if need_weights else None
```

### Flash Attention
通过分块计算和 kernel 融合，减少 HBM 访问次数，IO 感知的精确注意力。

### Paged Attention
允许在非连续的内存空间中存储连续的 K/V。每个序列的 KV Cache 划分为块。内存共享减少 55% 内存使用，吞吐提升 2.2x。

---

# 二、归一化

## 2.1 归一化的作用
- 加快网络训练收敛速度（每层数据分布一致）
- 控制梯度爆炸和防止梯度消失

## 2.2 Batch Normalization
y = (x - E[x]) / √(Var[x] + ε) * γ + β。均值和标准差在 mini-batch 上按维度计算。

## 2.3 Group Normalization
将 channels 划分为多个 groups，计算每个 group 内的均值和方差。与 batch size 无关。

## 2.4 Layer Normalization
μ = (1/H)Σxᵢ, σ = √((1/H)Σ(xᵢ-μ)²+ε), y = f(g/σ · (x-μ) + b)

## 2.5 RMS Norm
去掉减均值部分（re-centering），只保留方差部分（re-scaling）。LLaMA 采用。

## 2.6 Pre-Norm vs Post-Norm
同深度下 Post-Norm 效果更优（Pre-Norm 实际相当于更宽但更浅的网络）。LLaMA 采用 Pre-Norm 因为模型足够深（32-80层），恒等分支有利于梯度传播。

---

# 三、位置编码

## 3.1 绝对位置编码
- Sinusoidal：固定三角函数
- Learnable（训练式）：BERT
- **RoPE（旋转位置编码）**：既保留绝对位置信息，又在内积运算下保留相对性

RoPE 完整实现：
```python
def precompute_freqs_cis(dim, end, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)
```

RoPE 优化方向：NTK-RoPE, NTK-logn, 窗口注意力, Sparse Attention

## 3.2 相对位置编码
- XLNet 式, T5 式, DeBERTa 式, ALiBi

---

# 四、激活函数

对每个函数列出：函数定义、导数、优点、缺点

## 4.1 Sigmoid
f(x) = 1/(1+e^(-x)), f'(x) = f(x)(1-f(x))
优点：输出(0,1)单调连续，求导容易
缺点：饱和区梯度接近0导致梯度消失，输出恒>0（非零中心化），幂运算计算慢

## 4.2 TanH
f(x) = (e^x - e^(-x))/(e^x + e^(-x)), f'(x) = 1 - f(x)²
优点：比 sigmoid 收敛更快，以 0 为中心
缺点：仍有梯度消失，幂运算计算慢

## 4.3 ReLU
f(x) = max(0, x), f'(x) = 0 if x<0, 1 if x>=0
优点：收敛快，实现简单，缓解梯度消失，稀疏表达
缺点：非零中心化，神经元坏死，不能避免梯度爆炸

## 4.4 LeakyReLU
f(x) = αx if x<0, x if x>=0 (α常设0.01)
优点：避免梯度消失，减少死神经元
缺点：不一定比 ReLU 好

## 4.5 Softmax
y_i = exp(x_i) / Σ_j exp(x_j)，用于多分类

## 4.6 SiLU (Swish)
silu(x) = x * σ(βx), β 通常设为 1
f'(x) = f(x) + σ(x)(1 - f(x))
优点：x>0 无梯度消失，x<0 不会神经元死亡，处处可导连续光滑，非单调
缺点：计算量大

## 4.7 GLU (Gated Linear Units)
GLU(x,W,V,b,c) = σ(xW+b) ⊗ (xV+c)，线性通道使梯度易通过

## 4.8 SwiGLU
SwiGLU(x,W,V,b,c) = Swish₁(xW+b) ⊗ (xV+c)
LLaMA 使用。收敛更快，性能更好，但计算量大。

---

# 五、Loss 函数

回归：L1 Loss, MSE Loss (L2), HuberLoss
判别-二分类：BCE Loss, BCEWithLogitsLoss (Sigmoid+BCE), Focal Loss (样本不均衡)
判别-多分类：Cross-Entropy Loss, NLL Loss (等价 CE), MultiMarginLoss
排序：MarginRankingLoss, HingeEmbeddingLoss, CosineEmbeddingLoss, info-nce loss

---

# 六、模型架构
- Encoder: BERT 系列
- Decoder-only: GPT / LLaMA / Qwen（主流）
- Prefix-decoder: GLM
- Encoder-Decoder: T5

主流大模型：LLaMA/Llama2, Baichuan, ChatGLM, Qwen

---

# 七、LoRA 系列微调

## 7.1 LoRA
W = W₀ + Δ = W₀ + BA（低秩分解）

## 7.2 AdaLoRA
参数化 SVD，将关键增量矩阵分配高秩，不重要的降低秩。

## 7.3 QLoRA
4-bit NormalFloat（理论最优 4-bit 量化）+ Double Quantization（每参数节省 0.37bit，LLaMA-65B 约省 3GB）+ Paged Optimizers（避免梯度检查点内存峰值）

## 7.4 LongLoRA
分组+偏移模拟全局注意力 + 引入 Embedding/Normalization 层微调。推理时用全局注意力，与 Flash Attention/vLLM 无缝兼容。

---

# 八、模型量化
- fp16: 基本半精度
- autoGPTQ: 相同顺序各行并行计算，分批 BatchUpdate，分组量化
- bitsandbytes: 支持 4-bit/8-bit 量化

---

# 九、Scaling Law

核心结论：
1. 模型表现和规模(N/D/C)强相关，和 shape 弱相关
2. 幂方法则：L(N)=Nc/N^αN, L(D)=Dc/D^αD
3. 模型参数增大 8x → 数据需增大 5x
4. 收敛是低效的：计算量增大时训练大模型比小模型更高效
5. 最佳 batch size 与 loss 成幂方关系
6. 泛化能力：同分布 test loss 下降时，不同分布 test 也提升
7. 采样高效性：大模型达到同效果需要的 step 更少

---

# 十、训练经验

## 显存估算
模型本体(fp32)：1B 参数 ≈ 4GB
AdamW 优化器：每参数 8 bytes (momentum + variance)
总显存 = 模型(2Ψ) + 梯度(2Ψ) + AdamW(12Ψ) = 16Ψ
GPT-2 (1.5B): 至少 24GB

## 训练 FLOPS 计算
每 step 前反向 FLOPs ≈ 96·B·s·l·h²（Attention 和 LM head 项可忽略）

## OOM 解决
训练层面：LoRA/减小 batch size/maxlen/参数冻结/量化/混合精度/蒸馏/分布式
工具层面：toma 算法

## 数据配比
预训练：领域/通用 = 1:5 最优
SFT：领域/通用 = 1:10 最优

## 模型部署
vLLM / TGI / TRT-LLM + Triton

---

# 十一、模型评估

考虑标准答案：BLEU, ROUGE (严格匹配), BERTScore (语义匹配)
不考虑标准答案：困惑度 (PPL)

评估分类：
- 综合性：C-Eval(52任务), FlagEval
- 任务导向：SocKET, TrustGPT, ChatGraph
- 领域导向：MMCU(医/法/心/教), FinanceIQ(金融), PromptCBLUE(医疗NLP)

---

# 十二、周边知识

## 可解释性
稀疏自动编码器：512 神经元 MLP → 80 亿数据点训练 → 分解为可解释特征（扩展因子 1x~256x）

## 幻觉
→ 详见 [幻觉详解](2026-02-09-llm-intro-hallucination.md)

## 模型架构对比
→ 详见 [长上下文训练](2026-02-09-llm-long-context-training.md)

## Open Questions

- How do different RoPE extension methods (NTK-RoPE, NTK-logn) compare in practice for very long context windows (>128k tokens)?
- What are the optimal rank settings for LoRA across different model sizes and tasks?
- How does the interplay between quantization (e.g., QLoRA 4-bit) and model scale affect downstream task performance?

## References

- Notion: llm-intro 文本大模型篇 (https://www.notion.so/32d8e2ddc6fd4203a6cb3a42f851ef55)
- 面试经验总结, 位置编码总结(苏剑林)
- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
