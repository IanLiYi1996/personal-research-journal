# Agentic RL 基础设施对比：Atropos/Tinker vs OpenClaw-RL

- **Date:** 2026-04-13
- **Tags:** #agentic-rl #infrastructure #grpo #online-learning #comparison

## Context

近期调研了两套面向 LLM Agent 的 RL 训练基础设施：

1. **Nous Research 的 Atropos + Tinker** — 以环境微服务为核心的 GRPO 训练框架，通过 Hermes Agent 的 tool calling 接口暴露给用户
2. **OpenClaw-RL** (arXiv 2603.10165) — 基于 slime 框架的在线持续学习系统，利用部署中的 next-state signal 进行实时策略改进

两者都试图解决同一个大问题：**如何让 LLM Agent 通过与环境交互来改进自身？** 但设计哲学截然不同。本文从架构、算法、奖励设计、工程实现等维度进行系统对比。

## 一、架构对比

### 1.1 组件拓扑

**Atropos（3 组件 + 队列）：**

```
Environment ──push──> Trajectory API (队列) <──pull── Trainer
                           │
                    Inference Server
                   (vLLM / SGLang / OpenRouter)
```

- 环境微服务推送 `ScoredDataGroup`（tokens + masks + logprobs + scores）到 API 队列
- Trainer 从队列拉 batch 进行 GRPO 训练
- 推理服务器独立运行，支持多种后端

**OpenClaw-RL（4 组件，完全解耦）：**

```
Policy Serving ──> Environment ──> Reward Judging ──> Policy Training
  (SGLang)         (Http/API)      (SGLang/API)        (Megatron)
```

- 四个组件作为独立异步循环运行，**无阻塞依赖**
- 模型在服务请求时，PRM 同时在判断上一个响应，训练器同时在应用梯度
- 权重更新采用 graceful update，不中断推理服务

### 1.2 关键架构差异

| 维度 | Atropos/Tinker | OpenClaw-RL |
|------|---------------|-------------|
| **耦合度** | 环境与 Trainer 通过队列解耦，但推理服务器与 Trainer 强关联（4 种 weight bridge 模式） | 四组件完全解耦，无阻塞依赖 |
| **训练范式** | Batch RL：环境生成数据 → 队列缓冲 → 批量训练 | 在线持续学习：交互即训练，无明确的「训练阶段」 |
| **环境来源** | 预定义环境（GSM8K, MMLU, Blackjack 等）| 真实部署交互（用户对话、终端、GUI、SWE） |
| **推理-训练同步** | 4 种模式：shared_vllm (CUDA IPC)、lora_restart、lora_only、none | Graceful weight update，non-blocking |
| **训练后端** | 自建 example_trainer 或 Tinker 云端 API | Megatron (slime 框架) |
| **分布式支持** | 单机多进程为主，Slurm 支持 | 原生大规模并行（128+ 并行环境） |

### 1.3 环境抽象

**Atropos BaseEnv：**

```python
class MyEnv(BaseEnv):
    async def setup(self):              # 初始化
    async def get_next_item(self):      # 提供下一条数据
    async def collect_trajectories(self, item):  # 推理 + 收集
    async def score(self, rollouts):    # 评分
    async def evaluate(self, *args):    # 周期性评估
```

核心设计：**环境同时负责推理和评分**。`ManagedServer` 自动追踪 tokens/logprobs，环境只需关注 prompt 构建和 reward 逻辑。`ScoredDataGroup` 作为传输载荷，内含 tokens、masks、logprobs、scores，直接可用于 GRPO 训练。

**OpenClaw-RL Session-Aware 环境：**

```
每个 API 请求 → {
    main-line turn: Agent 主要响应 + 工具执行结果 → 可训练样本
    side turn:      辅助查询、记忆整理 → 不产生训练数据
}
```

核心设计：**环境就是真实的用户交互**。Session-aware 的 turn 分类使框架能精确识别哪些交互产生训练信号。不需要 `get_next_item()` —— 数据来自实时部署流量。

**本质区别**：Atropos 的环境是「数据加载器 + 评分器」，OpenClaw-RL 的环境是「真实世界本身」。

## 二、RL 算法对比

### 2.1 策略优化

**Atropos GRPO：**

$$\text{ratio} = \exp(\log\pi_\theta(a|s) - \log\pi_{\text{old}}(a|s))$$
$$\mathcal{L} = -\min(\text{ratio} \cdot A, \; \text{clip}(\text{ratio}, 1-\varepsilon, 1+\varepsilon) \cdot A)$$

- Advantage = z-score 标准化的 group reward：$A = (r - \bar{r}) / \sigma_r$
- 同一 prompt 下 `group_size` 个 completion 形成对比组
- clip_eps = 0.2，importance sampling ratio 限制在 [0.8, 1.2]
- 无 KL 正则项

**OpenClaw-RL PPO-style + 非对称 clipping：**

$$\mathcal{L}_{\text{pg}} = -\min(\rho_t A_t, \; \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon_{\text{high}}) \cdot A_t)$$
$$\mathcal{L} = \mathcal{L}_{\text{pg}} + \beta_{\text{KL}} \cdot \mathcal{L}_{\text{KL}}$$

- $\varepsilon = 0.2$, $\varepsilon_{\text{high}} = 0.28$ — **非对称 clipping**，允许正向更新有更大空间
- $\beta_{\text{KL}} = 0.02$ — 显式 KL 正则，防止策略漂移过远
- Advantage 有两种来源（Binary RL 标量 / OPD token 级），可叠加

### 2.2 Advantage 计算——核心差异

这是两个系统最本质的分歧点。

**Atropos：Group-Relative Advantage**

```
同一 prompt → N 个 completion → 各自评分 → z-score 标准化
```

- 依赖同一 prompt 下多个 completion 的**相对排序**
- 不需要 value network，GRPO 的核心简化
- 要求 group_size 足够大（通常 ≥ 8）才能形成有意义的对比
- **限制**：只适用于可以批量生成多个 completion 的场景

**OpenClaw-RL：双源 Advantage**

*来源一 — Binary RL (评价性信号)：*

```
Agent 响应 → 环境 next-state → PRM Judge → r ∈ {+1, -1, 0}
直接用 r 作为 advantage（无 group 标准化）
```

*来源二 — OPD (指导性信号)：*

```
Agent 响应 a_t → next-state s_{t+1} → Judge 提取 hint
同一模型在 (原始 prompt) 和 (prompt + hint) 下的 logprob 差 = token 级 advantage
A_token = log π(a_t | s + hint) - log π(a_t | s)
```

*组合：*

$$A = w_{\text{binary}} \cdot r + w_{\text{opd}} \cdot (\log\pi_{\text{teacher}} - \log\pi_\theta)$$

| 维度 | Atropos GRPO | OpenClaw-RL |
|------|-------------|-------------|
| **Advantage 粒度** | 序列级（整个 completion 一个值） | Token 级（OPD）+ 序列级（Binary RL） |
| **信号来源** | 环境 reward function | Next-state signal 经 PRM judge |
| **是否需要 group** | 是，核心依赖 | 否，单条交互即可训练 |
| **信息利用率** | 只用评价性信号（好/坏） | 评价性 + 指导性（怎么改） |
| **适用场景** | 可批量生成的离线任务 | 实时在线交互 |

### 2.3 Reward 设计

**Atropos：环境自定义 Reward**

每个环境自行实现 `score()` 方法。以 GSM8K 为例：

```python
# 基础：数学验证 → 二值 reward
reward = 1.0 if verify(model_answer, gold_answer) else -1.0

# 精巧之处：全对时引入长度惩罚
if all_correct:
    score = 1.0 - linear_decay(length)  # 鼓励简洁推理
if no_contrast:  # 全对且长度相同，或全错
    return None   # 跳过——无梯度信号
```

特点：
- Reward 函数完全由环境开发者控制
- 可以非常精细（如 GSM8K 的长度惩罚）
- 但需要为每个任务手工设计
- 依赖可验证的 ground truth

**OpenClaw-RL：PRM Judge + Next-State Signal**

```
PRM(a_t, s_{t+1}) → r ∈ {+1, -1, 0}
- 工具调用结果 → 较明确的判断
- 用户回复 → 推断满意度
- 多次独立查询 → 多数投票
```

特点：
- 通用性强——任何有 next-state 信号的场景都可用
- 不需要 ground truth（PRM 基于语义判断）
- 但引入了 Judge 准确性的不确定性
- Process Reward（step-wise）对长时域任务至关重要

## 三、Token/Logprob 管理

两者都需要精确的 token 级 logprob 追踪来计算 importance sampling ratio，但实现方式不同。

**Atropos ManagedServer：**

```python
async with self.server.managed_server(tokenizer=tokenizer) as managed:
    completion = await managed.chat_completion(messages=msgs, n=8)
    state = managed.get_state()
    # 自动追踪：
    # node.tokens:         [prompt..., completion...]
    # node.masked_tokens:  [-100..., actual_tokens...]
    # node.logprobs:       [1.0..., actual_logprobs...]
```

- 上下文管理器自动处理 tokenization、masking、logprob 对齐
- `-100` 掩码 prompt tokens，`1.0` 作为 logprob 哨兵值（$e^{1.0} \approx 2.718 > 1$，不可能的概率）
- 支持 Tree Mode（多轮对话，prefix matching 复用已有 tokens）
- 环境开发者无需手动处理任何 token 级操作

**OpenClaw-RL（基于 slime/Megatron）：**

- 训练端直接在 Megatron 内完成 token 级操作
- 推理端 SGLang 返回 logprobs
- OPD 需要额外的 forced-decoding pass 来获取 teacher logprobs
- 解耦架构意味着 logprob 在组件间通过 API 传递

**对比**：Atropos 的 ManagedServer 是一个非常优雅的抽象，把 token 追踪的复杂性完全隐藏了。OpenClaw-RL 没有等价的统一抽象，但其解耦架构带来了更好的扩展性。

## 四、工程实现与易用性

### 4.1 开发者体验

| 维度 | Atropos/Tinker | OpenClaw-RL |
|------|---------------|-------------|
| **上手门槛** | 低（实现 4 个方法即可）| 高（需要理解 slime/Megatron） |
| **文档质量** | 优秀（README、CONFIG.md、BaseEnv docs） | 论文为主，工程文档较少 |
| **调试工具** | Gradio UI (view-run)、process 子命令、HTML 可视化 | WandB 集成 |
| **无 GPU 体验** | 支持（OpenRouter 推理测试 / Tinker 云端训练） | 不支持（需要本地 GPU 集群） |
| **Agent 集成** | Hermes Agent 8 个 RL tool（自然语言驱动） | 无 Agent 接口 |
| **开源程度** | Atropos MIT 开源，Tinker 闭源 API | 论文公开，代码基于 slime（部分开源） |

### 4.2 适用场景

**Atropos 更适合：**
- 有明确 reward function 的任务（数学、代码、tool-calling）
- 快速原型验证（OpenRouter 推理测试 → Tinker 云端训练）
- 不需要在线学习的 batch 训练场景
- 社区贡献（MIT 开源、环境市场）

**OpenClaw-RL 更适合：**
- 部署后持续改进的 Personal Agent
- 长时域多步交互任务（GUI、SWE、Terminal）
- 没有 ground truth 但有丰富环境反馈的场景
- 需要利用指导性信号（用户修正、error trace）的场景

## 五、已发布实验结果

### Atropos

| 任务 | 训练前 | 训练后 | 提升 |
|------|--------|--------|------|
| Tool Calling (Parallel) | 10% | 46% | **4.6x** |
| Tool Calling (Simple) | 21% | 51.75% | **2.5x** |
| Financial Prediction | 20% | 50% | **2.5x** |

模型：DeepHermes 系列 specialist，发布在 HuggingFace。

### OpenClaw-RL

| 任务 | 方法 | 效果 |
|------|------|------|
| Personal (Student) | Binary+OPD 16步 | 0.17 → **0.81** |
| Tool-call | Outcome+Process | **0.30** (vs Outcome-only 0.17) |
| GUI | Outcome+Process | **0.33** (vs Outcome-only 0.31) |
| Terminal/SWE | 全量 | 持续提升 |

模型：Qwen3 系列 (4B/8B/32B)。

**注意**：两者的实验设置完全不同，不能直接横向比较数值。

## 六、核心设计哲学差异

### Atropos："RL 训练是 Agent 的一个工具"

Atropos 的设计出发点是**让 RL 训练变得像调用 API 一样简单**。通过标准化的环境接口（BaseEnv）、自动化的 token 追踪（ManagedServer）、和 Agent 级的训练编排（Hermes 的 8 个 RL tool），把分布式 RL 训练的复杂性封装成一次 tool call。

这是一种**工具化思维**：训练是手段，环境是舞台，目标是在特定任务上快速提升模型表现。

### OpenClaw-RL："使用就是训练"

OpenClaw-RL 的出发点是**部署后的持续自我改进**。Agent 不需要专门的训练阶段——每次与用户交互、每次工具执行、每次环境反馈都是学习信号。通过 PRM Judge 将 next-state signal 转化为 reward，通过 OPD 将指导性信息转化为 token 级梯度。

这是一种**生态化思维**：Agent 在真实环境中"生长"，越用越懂用户，越用越好用。

### 两者的交汇点

值得注意的是，Atropos 最近也加入了 **On-Policy Distillation** 支持（`distill_token_ids` + `distill_logprobs`，`TeacherDistillationEnv` mixin），说明社区已经认识到纯 reward-based GRPO 在信号利用上的局限性。而 OpenClaw-RL 在 General Agent track 上也使用了 GRPO 式的 group 标准化（按 step index 分组），说明 GRPO 的相对排序思想在有条件时仍然有效。

**两条路径正在趋近融合**——Atropos 向在线学习和丰富信号靠拢，OpenClaw-RL 向工程化和易用性靠拢。

## Open Questions

1. **ManagedServer + OPD**：如果将 Atropos 的 ManagedServer 抽象与 OpenClaw-RL 的 OPD hint-enhanced self-distillation 结合，能否在保持易用性的同时获得 token 级 advantage？
2. **Reward 信号融合**：Atropos 的环境自定义 reward（精确但需手工设计）和 OpenClaw-RL 的 PRM Judge（通用但有噪声）能否互补——对有 ground truth 的部分用精确 reward，其余用 PRM？
3. **Weight Sync 的最优解**：Atropos 的 shared_vllm (CUDA IPC, 172 TPS) vs OpenClaw-RL 的 graceful update (non-blocking) —— 在大规模部署中哪种更优？是否可以结合？
4. **Personal Agent 的安全性**：如果用户反馈包含 harmful 指令，OpenClaw-RL 的在线学习是否会导致模型学习有害行为？Atropos 的预定义环境天然避免了这个问题，但也牺牲了个性化能力。
5. **Scaling Law for Agentic RL**：两个系统都在 4B-32B 模型上实验，更大模型上的 behavior 是什么？OPD 的 self-distillation 在更强模型上是否有更大收益（因为 hint-enhanced 分布质量更高）？

## References

- [Atropos GitHub](https://github.com/NousResearch/atropos) — MIT licensed, environment microservice framework for LLM RL
- [tinker-atropos GitHub](https://github.com/NousResearch/tinker-atropos) — Atropos + Tinker cloud training integration
- [Hermes Agent RL Training Docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/rl-training)
- [OpenClaw-RL Paper](https://arxiv.org/abs/2603.10165) — Wang et al., "Train Any Agent Simply by Talking"
- papers/2026-openclaw-rl.md — 本 repo 中的 OpenClaw-RL 详细论文解读
