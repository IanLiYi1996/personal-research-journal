# LLM Intro: 语言模型解释与训练 - Build LLM from Scratch

- **Date:** 2026-02-09
- **Tags:** LLM, build-from-scratch, GPT, tokenizer, training, SFT, RLHF, PPO, pretraining, transformer, attention, embedding

## Context

Comprehensive notes on understanding and building Large Language Models (LLMs) from scratch. Covers the full pipeline from neural network fundamentals, through tokenization, model architecture (GPT), pretraining with Causal Language Modeling (CLM), Supervised Fine-Tuning (SFT), Reinforcement Learning from Human Feedback (RLHF) with PPO, and model inference/decoding strategies. Based on the "llm-intro 语言模型解释与训练" knowledge base.

---

## 一. How Language Model Works

### 1.1 从神经元到神经网络

#### 神经元模型

神经元(Neuron)是神经网络的基本计算单元。每个神经元接收一组输入信号，通过加权求和后加上偏置，再经过激活函数产生输出。数学表达式为：

```
y = f(w1*x1 + w2*x2 + ... + wn*xn + b)
```

其中 `xi` 是输入，`wi` 是对应的权重，`b` 是偏置，`f` 是激活函数。

#### 激活函数

激活函数为神经网络引入非线性，使得网络能够学习复杂的模式。常见的激活函数包括：

- **Sigmoid**: `f(x) = 1 / (1 + e^(-x))`，输出范围 (0, 1)
- **Tanh**: `f(x) = (e^x - e^(-x)) / (e^x + e^(-x))`，输出范围 (-1, 1)
- **ReLU**: `f(x) = max(0, x)`，简单高效，目前最常用
- **GELU**: Gaussian Error Linear Unit，GPT系列模型使用的激活函数

#### 前馈神经网络与万能近似定理

从神经元网络到前馈神经网络：**只需一个包含足够多神经元的隐层，多层前馈网络就能以任意精度逼近任意复杂度的连续函数。** 这就是万能近似定理(Universal Approximation Theorem)的核心内容。

前馈神经网络(Feedforward Neural Network)的特点：
- 信息单向流动，从输入层经过隐藏层到输出层
- 没有循环或反馈连接
- 每一层的输出作为下一层的输入

#### BP反向传播算法

神经网络优化算法：BP反向传播算法，对多层人工神经网络进行梯度下降的算法，也就是用链式法则以网络每层的权重为变量计算损失函数的梯度，以更新权重来最小化损失函数。

**梯度下降**算法通过计算可微函数的**梯度**并沿**梯度**的相反方向移动，指导搜索以局部/全局最小值最小化函数的值。

核心步骤：
1. **前向传播**: 输入数据通过网络计算得到预测输出
2. **计算损失**: 比较预测输出与真实标签，计算损失值
3. **反向传播**: 从输出层到输入层，利用链式法则计算每个参数的梯度
4. **参数更新**: 根据梯度和学习率更新网络参数

### 1.2 LLM工作原理可视化

LLM的工作原理可以可视化理解为：
1. 输入文本经过Tokenizer转换为token序列
2. Token序列经过Embedding层转换为向量表示
3. 向量序列通过多层Transformer Block进行处理
4. 最终输出层预测下一个token的概率分布
5. 通过解码策略从概率分布中选择下一个token

---

## 二. Build LLM from Scratch

构建LLM的完整流程包括：Data Science -> Tokenizer -> Model Architecture -> Loss Function -> Training Loop

### 2.1 Data Science

#### 数据类型

##### 预训练数据
- **数据特点**: 数据覆盖面广，数据量大，数据尽可能优质
- **数据形式**: 纯文本

##### 指令数据
- **数据特点**: 数据要精，数据量不要求多
- **数据形式**: `<instruct, answer>`

##### 强化学习数据
- **数据特点**: 数据要精，数据要多于指令微调
- **数据形式**: 偏序数据 `<instruct, answer_accept, answer_reject>`

#### 数据处理操作

##### 文本类数据
文本数据处理流程包括：数据收集 -> 数据清洗 -> 数据去重 -> 数据过滤 -> 质量评分 -> 数据混合

##### 代码类数据
代码数据处理流程包括：代码收集 -> 语法检查 -> 代码去重 -> 安全过滤 -> 质量筛选

##### 多模态数据
多模态数据处理需要处理文本、图像、音频等不同模态的数据对齐和融合。

#### Q&A

数据收集相关的常见问题和解答涉及数据质量评估、数据配比、数据清洗策略等方面。

---

### 2.2 Build LLM from Scratch

#### 2.2.1 Tokenizer

首先从tokenize开始，负责将文本转token，再将最终预测的token转回文本。

##### 1. Whole Word Tokenizer

以完整单词为单位进行分词。

- **优点**: 语义完整，每个token代表一个完整的词
- **缺点**: 词表巨大，无法处理未登录词(OOV)

##### 2. Character Tokenizer

以单个字符为单位进行分词。

- **优点**: 词表小，不存在OOV问题
- **缺点**: 序列过长，单个字符缺乏语义信息

##### 3. Subword Tokenizer

介于word和character之间的分词方式，是目前主流的分词方法。

- **优点**: 平衡了词表大小和序列长度，能处理OOV
- **缺点**: 需要训练过程来确定最优的子词切分

##### BPE Tokenizer (Byte Pair Encoding)

基于字符的BPE算法，由它构造的"单词"往往位于字符和单词之间，常见的形式就是单词中的片段作为一个独立的"单词"，特别是对于那些比较长的单词。比如单词wonderful有可能会被拆分成两个子单词"wonder"和"ful"。

**BPE算法步骤**:
1. 将所有单词拆分为字符序列，加上结束符
2. 统计所有相邻字符对的频率
3. 合并频率最高的字符对，创建新的子词
4. 重复步骤2-3，直到达到预设的词表大小

**Byte-level BPE**: GPT-2使用的方式，基于Byte的BPE，词表中共计包含50K左右的单词，这种方式不需要担心未登录词的出现，因为它会从Byte的层面去分解单词。

##### WordPiece

与BPE算法类似，WordPiece算法也是每次从词表中选出两个子词合并成新的子词。与BPE的最大区别在于，如何选择两个子词进行合并：**BPE选择频数最高的相邻子词合并，而WordPiece选择能够提升语言模型概率最大的相邻子词加入词表。** 代表模型：BERT

##### Unigram Tokenizer

而Unigram Language Model则是**减量法**，即先初始化一个大词表，根据评估准则不断丢弃词表，直到满足限定条件。ULM算法考虑了句子的不同分词可能，因而能够输出带概率的多个子词分段。

##### SentencePiece

SentencePiece将输入视为原始输入流，从而包括要使用的字符集中的空格。然后，它使用BPE或Unigram来构建适当的词汇表。代表模型为ALBERT、XLNet、Marian和T5。

##### Tokenizer使用示例代码

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
# ['Using', 'a', 'transform', '##er', 'network', 'is', 'simple']

ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
# [7993, 170, 11303, 1200, 2443, 1110, 3014]

decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
# 'Using a Transformer network is simple'
```

---

#### 2.2.2 Deep into LLM from Scratch

##### Build Sub Module

###### Embedding

**概念解释**: Embedding的数学本质，就是以one hot为输入的单层全连接。也就是说，世界上本没什么Embedding，有的只是one hot。现在我们将token, position, segment三者都用one hot表示，然后concat起来，然后才去过一个单层全连接，等价的效果就是三个Embedding相加。

**Embedding层的作用**:
- Token Embedding: 将离散的token ID映射为连续的向量表示
- Position Embedding: 编码token在序列中的位置信息
- 两者相加得到最终的输入表示

**实现方式**:

```python
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples


wte = nn.Embedding(config.vocab_size, config.n_embd),
wpe = nn.Embedding(config.block_size, config.n_embd)
```

- `wte` (word token embedding): 将词表中的每个token映射为 `n_embd` 维向量
- `wpe` (word position embedding): 将每个位置映射为 `n_embd` 维向量
- 最终输入 = token embedding + position embedding

###### CausalSelfAttention (因果自注意力)

**概念解释**: 通过计算token之间的相似性，来捕捉内部存在的语义关系。Causal (因果) 意味着每个token只能attend到它之前的token，不能看到未来的信息。

**实现方式**:

```python
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples
# -----------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
```

**关键设计**:
- 使用单个线性层 `c_attn` 同时计算Q, K, V，然后split开，提高计算效率
- 使用 `F.scaled_dot_product_attention` 实现Flash Attention，大幅提升计算效率和内存效率
- `NANOGPT_SCALE_INIT` 标记用于后续的权重初始化缩放
- Multi-head attention: 将embedding维度分成多个head，每个head独立计算attention

###### MLP (Feed-Forward Network)

**模块解释**: 通过两个线性层来拟合真实的数据分布，该层通过参数可以存储大部分的知识。

**实现方式**:

```python
class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
```

**关键设计**:
- 先升维4倍 (`n_embd -> 4 * n_embd`)，再降维回来 (`4 * n_embd -> n_embd`)
- 使用GELU激活函数(近似tanh版本)，比ReLU更平滑
- `NANOGPT_SCALE_INIT` 标记用于残差连接的权重缩放初始化

###### Transformer Block

**模块解释**: Transformer Block是模型的基本构建单元，采用Pre-Norm架构 + 残差连接。

**实现方式**:

```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

**关键设计**:
- **Pre-Norm**: LayerNorm在attention/MLP之前，而不是之后（与原始Transformer不同）
- **残差连接**: `x = x + sublayer(norm(x))`，帮助梯度流动，使深层网络训练更稳定
- 每个Block包含一个注意力层和一个MLP层

##### Get Whole Model - GPT完整模型

组装各个模块，构建完整的GPT模型：

```python
import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from hellaswag import render_example, iterate_examples

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight

        # init params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
```

**模型关键设计点**:

1. **Weight Sharing**: `self.transformer.wte.weight = self.lm_head.weight` - 输入embedding和输出projection共享权重，减少参数量
2. **权重初始化**:
   - Linear层使用 `N(0, 0.02)` 初始化
   - 带 `NANOGPT_SCALE_INIT` 标记的层额外缩放 `(2 * n_layer)^(-0.5)`，确保残差路径的方差稳定
   - Embedding层使用 `N(0, 0.02)` 初始化
3. **from_pretrained**: 从HuggingFace加载预训练GPT-2权重，处理Conv1D到Linear的转置
4. **configure_optimizers**:
   - 2D参数(权重矩阵)使用weight decay
   - 1D参数(bias, LayerNorm)不使用weight decay
   - 使用fused AdamW优化器(如果可用)提升性能

---

#### 2.2.3 定义损失函数

##### DataLoaderLite 完整代码

```python
import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
```

**DataLoaderLite关键设计**:
- 支持分布式数据加载，每个进程读取不同位置的数据
- 使用分片(shards)机制处理大规模数据
- 自动循环切换数据分片
- 输入x和目标y之间偏移1个token（自回归预测下一个token）

##### HellaSwag评估辅助函数

```python
# -----------------------------------------------------------------------------
# helper function for HellaSwag eval
# takes tokens, mask, and logits, returns the index of the completion with the lowest loss

def get_most_likely_row(tokens, mask, logits):
    # evaluate the autoregressive loss at all positions
    shift_logits = (logits[..., :-1, :]).contiguous()
    shift_tokens = (tokens[..., 1:]).contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    # now get the average loss just for the completion region (where mask == 1), in each row
    shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
    masked_shift_losses = shift_losses * shift_mask
    # sum and divide by the number of 1s in the mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # now we have a loss for each of the 4 completions
    # the one with the lowest loss should be the most likely
    pred_norm = avg_loss.argmin().item()
    return pred_norm
```

---

#### 2.2.4 整体训练流程

完整的训练代码，包括DDP设置、学习率调度(cosine with warmup)、梯度累积、验证循环、HellaSwag评估、文本生成和完整训练循环：

```python
# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# set up DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands CUDA, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# added after video, pytorch can be serious about it's device vs. device_type distinction
device_type = "cuda" if device.startswith("cuda") else "cpu"

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

enc = tiktoken.get_encoding("gpt2")

total_batch_size = 524288 # 2**19, ~0.5M, in number of tokens
B = 64 # micro batch size
T = 1024 # sequence length
assert total_batch_size % (B * T * ddp_world_size) == 0, "make sure total_batch_size is divisible by B * T * ddp_world_size"
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

torch.set_float32_matmul_precision('high')

# create model
model = GPT(GPTConfig(vocab_size=50304))
# model = GPT.from_pretrained("gpt2") # or init from OpenAI GPT-2
model.to(device)
use_compile = False # torch.compile interferes with HellaSwag eval and Generation. TODO fix
if use_compile:
    model = torch.compile(model)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model # always contains the "raw" unwrapped model

max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715
max_steps = 19073 # 19,073 steps is ~1 epoch, if data is 10B tokens and batch size 0.5M tokens
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # 2) if it > lr_decay_iters, return min learning rate
    if it > max_steps:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

# optimize!
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device_type=device_type)

# create the log directory we will write checkpoints to and log to
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f: # open for writing to clear the file
    pass

for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0
            val_loss_steps = 20
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / val_loss_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}\n")
            if step > 0 and (step % 5000 == 0 or last_step):
                # optionally write model checkpoints
                checkpoint_path = os.path.join(log_dir, f"model_{step:05d}.pt")
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'config': raw_model.config,
                    'step': step,
                    'val_loss': val_loss_accum.item()
                }
                # you might also want to add optimizer.state_dict() and
                # rng seeds etc., if you wanted to more exactly resume training
                torch.save(checkpoint, checkpoint_path)

    # once in a while evaluate hellaswag
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # only process examples where i % ddp_world_size == ddp_rank
            if i % ddp_world_size != ddp_rank:
                continue
            # render the example into tokens and labels
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # reduce the stats across all processes
        if ddp:
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        acc_norm = num_correct_norm / num_total
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")

    # once in a while generate from the model (except step 0, which is noise)
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)
        while xgen.size(1) < max_length:
            # forward the model to get the logits
            with torch.no_grad():
                with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # take the logits at the last position
                logits = logits[:, -1, :] # (B, vocab_size)
                # get the probabilities
                probs = F.softmax(logits, dim=-1)
                # do top-k sampling of 50 (huggingface pipeline default)
                # topk_probs here becomes (5, 50), topk_indices is (5, 50)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # select a token from the top-k probabilities
                # note: multinomial does not demand the input to sum to 1
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng) # (B, 1)
                # gather the corresponding indices
                xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
                # append to the sequence
                xgen = torch.cat((xgen, xcol), dim=1)
        # print the generated text
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist()
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")

    # do one step of the optimization
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # added after video, this field is also used by the forward pass.
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # we have to scale the loss to account for gradient accumulation,
        # because the gradients just add on each successive backward().
        # addition of gradients corresponds to a SUM in the objective, but
        # instead of a SUM we want MEAN. Scale the loss here so it comes out right
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        loss.backward()
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # determine and set the learning rate for this iteration
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    if device_type == "cuda":
        torch.cuda.synchronize() # wait for the GPU to finish work
    t1 = time.time()
    dt = t1 - t0 # time difference in seconds
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
```

**训练流程关键组件**:

1. **DDP分布式训练设置**:
   - 通过环境变量 `RANK`, `LOCAL_RANK`, `WORLD_SIZE` 检测是否为DDP模式
   - 使用NCCL后端进行GPU间通信
   - master_process (rank 0) 负责日志和checkpoint

2. **学习率调度 (Cosine with Warmup)**:
   - Warmup阶段: 线性从0增加到 `max_lr` (715步)
   - Cosine decay阶段: 从 `max_lr` 余弦衰减到 `min_lr`
   - `max_lr = 6e-4`, `min_lr = 6e-5`

3. **梯度累积**:
   - `total_batch_size = 524288` (~0.5M tokens)
   - 通过多个micro step累积梯度模拟大batch
   - loss需要除以 `grad_accum_steps` 来正确计算平均值
   - DDP模式下只在最后一个micro step同步梯度

4. **验证循环** (每250步):
   - 计算验证集loss (20个batch的平均)
   - DDP模式下通过all_reduce求平均

5. **HellaSwag评估** (每250步):
   - 常识推理benchmark
   - 分布式评估，每个进程处理不同的样本

6. **文本生成** (每250步):
   - Top-k sampling (k=50)
   - 生成4个序列进行定性检查

7. **训练步骤优化**:
   - 混合精度训练 (`torch.autocast` with bfloat16)
   - 梯度裁剪 (`clip_grad_norm_` max_norm=1.0)
   - `vocab_size=50304` (50257对齐到128的倍数，提升GPU效率)

---

### 2.3 LLM Pretraining

因果语言建模 (CLM) 是一种预训练技术，涉及训练语言模型以在给定先前标记的情况下预测序列中的下一个标记。目标是教会模型理解语言的底层结构并生成连贯的自然语言文本。许多流行的语言模型都使用 CLM 进行训练，包括 GPT、GPT-2、GPT-3 和 T5。这些模型已被证明可以在各种自然语言处理任务上实现最先进的性能，例如语言生成、文本分类和语言翻译。

**要点：**
1. **因果语言模型的建模靠mask attention保证** -- 通过causal mask确保每个token只能看到之前的token
2. **因果语言模型的训练优化靠loss部分实现** -- shift labels使得模型预测下一个token

#### HuggingFace GPT2LMHeadModel forward 完整代码

```python
def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(lm_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
```

**Loss计算关键点**:
- `shift_logits = lm_logits[..., :-1, :]` -- 去掉最后一个位置的logits
- `shift_labels = labels[..., 1:]` -- 去掉第一个位置的label
- 这样实现了"用位置 t 的输出预测位置 t+1 的token"
- label 为 `-100` 的位置会被 `CrossEntropyLoss` 自动忽略

---

### 2.4 LLM SFT Training

多轮对话微调数据集以及标签的构造方法，有三种常见方法。

一个多轮对话可以表示为:

```
inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
```

#### 方法一：仅学习最后一轮回复

只把最后一轮机器人的回复作为要学习的标签，其它地方作为语言模型概率预测的condition，无需学习，赋值为-100，忽略这些地方的loss。

```
inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
labels = <-100> <-100> <-100> <-100> <-100> <assistant3>
```

**评价**: 这种方法由于没有对中间轮次机器人回复的信息进行学习，因此存在着**严重的信息丢失**，是非常不可取的。

#### 方法二：拆解为多条样本

把一个多轮对话拆解，构造成多条样本，以便对机器人的每轮回复都能学习。

```
inputs1 = <user1> <assistant1>
labels1 = <-100> <assistant1>

inputs2 = <user1> <assistant1> <user2> <assistant2>
labels2 = <-100> <-100> <-100> <assistant2>

inputs3 = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
labels3 = <-100> <-100> <-100> <-100> <-100> <assistant3>
```

**评价**: 这种方法充分地利用了所有机器人的回复信息，但是**非常低效**，模型会有大量的重复计算。

#### 方法三：所有回复同时学习（推荐）

直接构造包括多轮对话中所有机器人回复内容的标签，既充分地利用了所有机器人的回复信息，同时也不存在重复计算，非常高效。

```
inputs = <user1> <assistant1> <user2> <assistant2> <user3> <assistant3>
labels = <-100> <assistant1> <-100> <assistant2> <-100> <assistant3>
```

**为什么可以直接这样去构造多轮对话的样本呢？难道inputs中包括第二轮和第三轮的对话内容不会干扰第一轮对话的学习吗？**

答案是**不会**。原因是LLM作为语言模型，它的注意力机制是一个**单向注意力机制**（通过引入 Masked Attention实现），模型在第一轮对话的输出跟输入中存不存在第二轮和第三轮对话完全没有关系。

**总结对比**:

| 方法 | 信息利用 | 计算效率 | 推荐度 |
|------|---------|---------|--------|
| 方法一：仅最后轮 | 严重信息丢失 | 高 | 不推荐 |
| 方法二：拆解多条 | 充分利用 | 低（大量重复） | 一般 |
| 方法三：同时学习 | 充分利用 | 高 | 推荐 |

---

### 2.5 RLHF Training

#### 算法逻辑

首先，该 **策略** (policy) 是一个接受提示并返回一系列文本 (或文本的概率分布) 的 LM。这个策略的 **行动空间** (action space) 是 LM 的词表对应的所有词元 (一般在 50k 数量级) ，**观察空间** (observation space) 是可能的输入词元序列，也比较大 (词汇量 ^ 输入标记的数量) 。**奖励函数** 是偏好模型和策略转变约束 (Policy shift constraint) 的结合。

PPO 算法确定的奖励函数具体计算如下：将提示 *x* 输入初始 LM 和当前微调的 LM，分别得到了输出文本 *y1*, *y2*，将来自当前策略的文本传递给 RM 得到一个标量的奖励 r_theta。将两个模型的生成文本进行比较计算差异的惩罚项，为输出词分布序列之间的 Kullback-Leibler散度的缩放，即 `r = r_theta - lambda * r_KL`。这一项被用于惩罚 RL 策略在每个训练批次中生成大幅偏离初始模型，以确保模型输出合理连贯的文本。如果去掉这一惩罚项可能导致模型在优化中生成乱码文本来愚弄奖励模型提供高奖励值。

最后根据 PPO 算法，我们按当前批次数据的奖励指标进行优化。

#### PPO三阶段流程

简而言之，过程可以分为三个阶段：

**1. Rollout and Evaluation (采样与评估)**

在这个阶段，我们从prompt库里抽样，使用语言模型生成response，然后使用奖励模型（Reward Model, RM）给出奖励得分。这个得分反映了生成的response的质量，比如它是否符合人类的偏好，是否符合任务的要求等。

**2. Make Experience (构建经验)**

在这个阶段，我们收集了一系列的"经验"，即模型的行为和对应的奖励。这些经验包括了模型生成的response以及对应的奖励得分。这些经验将被用于下一步的优化过程。

**3. Optimization (优化)**

在这个阶段，我们使用收集到的经验来更新模型的参数。具体来说，我们使用PPO算法来调整模型的参数，使得模型生成的response的奖励得分能够增加。PPO算法的一个关键特性是它尝试保持模型的行为不会发生太大的改变，这有助于保证模型的稳定性。

#### 代码实现

##### GAE计算 (Generalized Advantage Estimation)

估计每个时间步的优势和累积优势，结果为优势估计（当前策略对于旧策略的改进程度）和未来奖励的折现和：

```python
def whiten(x):
    var, mean = torch.var_mean(x)
    return (x - mean) * torch.rsqrt(var + 1e-8)


def gae(
    values,
    rewards,
):
    advantages = torch.zeros_like(rewards, device=rewards.device)
    last_advantage = 0
    last_value = 0

    with torch.no_grad():
        for t in reversed(range(rewards.shape[1])):
            delta = rewards[:, t] + config.method.gamma * last_value - values[:, t]
            last_advantage = delta + config.method.gamma * config.method.lam * last_advantage
            advantages[:, t] = last_advantage
            last_value = values[:, t]

        returns = advantages + values

    if config.method.use_whitening:
        advantages = whiten(advantages)

    return advantages, returns
```

**GAE关键点**:
- `delta = r_t + gamma * V(s_{t+1}) - V(s_t)` -- TD error
- `A_t = delta_t + gamma * lambda * A_{t+1}` -- 递归计算优势
- `returns = advantages + values` -- 目标回报值
- 可选的whitening标准化优势值

##### PPO Loss定义

返回策略梯度损失和值函数损失加权求和：

```python
def ppo_loss(
    logprobs,
    values,
    old_logprobs,
    old_values,
    advantages,
    returns,
    mask,
):

    values_clipped = torch.clamp(
        values,
        old_values - config.method.cliprange_value,
        old_values + config.method.cliprange_value,
    )

    n = mask.sum()

    vf_loss1 = (values - returns) ** 2
    vf_loss2 = (values_clipped - returns) ** 2
    vf_loss = 0.5 * torch.sum(torch.max(vf_loss1, vf_loss2) * mask) / n

    log_ratio = (logprobs - old_logprobs) * mask
    ratio = torch.exp(log_ratio)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - config.method.cliprange, 1.0 + config.method.cliprange)
    pg_loss = torch.sum(torch.max(pg_loss1, pg_loss2) * mask) / n
    pg_clipfrac = torch.sum((pg_loss2 > pg_loss1).float() * mask) / n

    loss = pg_loss + config.method.vf_coef * vf_loss

    return loss
```

**PPO Loss关键点**:
- **Value Function Loss**: clipped value loss，防止value function更新过大
- **Policy Gradient Loss**: clipped surrogate objective，通过ratio clipping限制策略更新幅度
- `ratio = exp(logprobs - old_logprobs)` -- 新旧策略的概率比
- 最终loss = policy loss + vf_coef * value loss

##### 完整Loss函数

```python
def loss_fn(batch):
    model_device = next(model.parameters()).device
    query_tensors = batch.query_tensors.to(model_device)
    response_tensors = batch.response_tensors.to(model_device)
    old_logprobs = batch.logprobs.to(model_device)
    old_values = batch.values.to(model_device)
    old_rewards = batch.rewards.to(model_device)

    response_length = old_rewards.shape[1]

    advantages, returns = gae(old_values, old_rewards)

    tokens, attention_mask, position_ids = get_model_inputs(query_tensors, response_tensors, tokenizer.pad_token_id)

    logits, values_pred = model(tokens,
                                attention_mask=attention_mask,
                                position_ids=position_ids)
    values_pred = values_pred[:, :-1]
    logprobs = logprobs_from_logits(logits[:, :-1, :], tokens[:, 1:])
    attention_mask = attention_mask[:, :-1]

    start = query_tensors.shape[1] - 1
    end = start + response_length
    logprobs, values_pred, mask = (
        logprobs[:, start:end],
        values_pred[:, start:end],
        attention_mask[:, start:end],
    )

    loss = ppo_loss(
        logprobs=logprobs,
        values=values_pred,
        old_logprobs=old_logprobs,
        old_values=old_values,
        advantages=advantages,
        returns=returns,
        mask=mask,
    )

    return loss, old_rewards[:,-1].mean().item()
```

**loss_fn完整流程**:
1. 将batch数据移到模型所在设备
2. 使用GAE计算advantages和returns
3. 拼接query和response获取模型输入
4. 前向传播获取logits和value predictions
5. 只取response部分的logprobs和values
6. 调用ppo_loss计算最终损失
7. 返回loss和平均奖励用于监控

---

### 2.6 模型推理

#### 基本推理代码示例

```python
model = AutoModelForCausalLM.from_pretrained(
    new_swift_path,
    torch_dtype="auto",
    device_map="auto"
)
model.generation_config.max_new_tokens = 2048
tokenizer = AutoTokenizer.from_pretrained("/data1/model_checkpoint/Qwen1.5-7B/")

prompt = "怎么选基金？"
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.pad_token_type_id,
    tokenizer.convert_tokens_to_ids("<|im_start|>")
]

generated_ids = model.generate(
    model_inputs.input_ids,
    max_new_tokens=1024,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
    repetition_penalty=1,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
```

#### 获取模型隐藏状态

```python
from transformers import  AutoModelForCausalLM,AutoTokenizer
model_name = "/disk/model_checkpoint/baichuan-7B//" #或者远程 "THUDM/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(
    model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name,trust_remote_code=True).half().cuda()

import torch

his = []

def get_last_hidden_state(text):
    context_ids = tokenizer.encode(
            text,
            max_length=1024,
            truncation=True)
    input_ids = torch.tensor([context_ids + [tokenizer.bos_token_id]])
    print(input_ids.size())
    attention_mask = torch.ones_like(input_ids)
    input_ids = input_ids.to('cuda:0')
    attention_mask = attention_mask.to('cuda:0')
    outputs = model.model(
        input_ids=input_ids,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=True,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    return outputs

res = get_last_hidden_state('指令微调本质训练的是对指令的理解')
last = res.last_hidden_state

last.size()
# torch.Size([1, 14, 4096])
```

#### 解码策略详解

训练好的大模型直接输出的是概率分布，我们应该如何从这个概率分布中选择下一个单词呢？以下是几种常用的方法：

##### 1. 贪心解码 (Greedy Decoding)
直接选择概率最高的单词。这种方法简单高效，但是可能会导致生成的文本过于单调和重复。

##### 2. 随机采样 (Random Sampling)
按照概率分布随机选择一个单词。这种方法可以增加生成的多样性，但是可能会导致生成的文本不连贯和无意义。

##### 3. Beam Search
维护一个大小为 k 的候选序列集合，每一步从每个候选序列的概率分布中选择概率最高的 k 个单词，然后保留总概率最高的 k 个候选序列。这种方法可以平衡生成的质量和多样性，但是可能会导致生成的文本过于保守和不自然。

##### 4. Temperature 采样
受统计热力学的启发，高温意味着更可能遇到低能态。在概率模型中，logits 扮演着能量的角色，我们可以通过将 logits 除以温度来实现温度采样，然后将其输入 Softmax 并获得采样概率。
- **低温度** (< 1.0): 使模型对其首选越有信心，输出更确定
- **高温度** (> 1.0): 降低信心，增加随机性
- **温度 = 0**: 等同于 argmax 似然（贪心解码）
- **温度 = 无穷大**: 等同于均匀采样

##### 5. Top-K 采样
Top-k 采样是对前面"贪心策略"的优化，它从排名前 k 的 token 中进行抽样，允许其他分数或概率较高的token也有机会被选中。在很多情况下，这种抽样带来的随机性有助于提高生成质量。

##### 6. Top-P 采样 (Nucleus Sampling)
Top-p 采样的思路是，在每一步，只从累积概率超过某个阈值 p 的最小单词集合中进行随机采样，而不考虑其他低概率的单词。这种方法也被称为核采样（nucleus sampling），因为它只关注概率分布的核心部分，而忽略了尾部部分。

例如，如果 p=0.9，那么我们只从累积概率达到 0.9 的最小单词集合中选择一个单词，而不考虑其他累积概率小于 0.9 的单词。这样可以避免采样到一些不合适或不相关的单词，同时也可以保留一些有趣或有创意的单词。

##### 7. 猜测解码 (Speculative Decoding)
**猜测解码**是加速自回归解码的算法，大致思路是采用**猜测和验证策略**，即先让草稿模型(draft model)预测几个潜在的未来token，然后原始LLM去并行验证。该方法可以"凭好运气"减少解码步骤的数量，从而降低延迟。

#### Generate参数优先级

```
temperature > topK > topP > typicalP > epsilonCutoff > etaCutoff
```

- `repetition_penalty > 1.0` 的时候，重复惩罚生效

#### 推理解码代码示例

```
with torch.no_grad():
    logits = model(x)[0] # (B, T, vocab_size)
    # take the logits at the last position
    logits = logits[:, -1, :] # (B, vocab_size)
    # get the probabilities
    probs = F.softmax(logits, dim=-1)
    # do top-k sampling of 50 (huggingface pipeline default)
    # topk_probs here becomes (5, 50), topk_indices is (5, 50)
    topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
    # select a token from the top-k probabilities
    # note: multinomial does not demand the input to sum to 1
    ix = torch.multinomial(topk_probs, 1) # (B, 1)
    # gather the corresponding indices
    xcol = torch.gather(topk_indices, -1, ix) # (B, 1)
    # append to the sequence
    x = torch.cat((x, xcol), dim=1)
```

---

## Open Questions

- 如何选择最优的tokenizer词表大小？词表大小与模型性能之间的trade-off
- 不同的位置编码方案（绝对位置编码 vs RoPE vs ALiBi）对模型性能的影响
- RLHF中reward model的质量如何保证？reward hacking问题的解决方案
- DPO (Direct Preference Optimization) 相比PPO的优势和劣势
- 长序列训练中的内存优化策略（Ring Attention, Sequence Parallelism等）
- 模型推理加速技术的进一步优化（vLLM, TensorRT-LLM, FlashAttention-2等）

## References

- Andrej Karpathy, "Let's build GPT from scratch" -- nanoGPT
- HuggingFace Transformers documentation -- Tokenizer Summary
- OpenAI, "Training language models to follow instructions with human feedback" (InstructGPT)
- Schulman et al., "Proximal Policy Optimization Algorithms" (PPO)
- HuggingFace, GPT2LMHeadModel source code
- Notion source: https://www.notion.so/b6610c8baed14ce090f98fa4104aac22
