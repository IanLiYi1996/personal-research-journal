# Post-training 101: 从预训练到指令调优

- **Date:** 2026-02-09
- **Tags:** LLM, post-training, SFT, RLHF, RLVR, DPO, PPO, GRPO, evaluation, reward-model

## Context

Post-training 是将预训练语言模型从"知识库"转变为"有用助手"的关键阶段。本文整理了 SFT、RL（RLHF/RLAIF/RLVR）、评估方法的核心概念、公式、数据格式与代码示例，作为个人学习与回顾的参考笔记。

## Main Content

### 一、从预训练到指令调优的旅程

Base model 通过 next-token prediction 编码知识但不够有用。例如输入 "What is the capital city of U.S" --- 预训练模型先预测问号再补全句子，指令调优模型直接回答 "Washington, D.C."

Post-training 将模型变为 helpful, honest, harmless 的助手。技术：SFT + RL (RLHF/RLAIF/RLVR)

时间线：InstructGPT(2022, SFT+RLHF) -> DeepSeek-V3(2024, RLVR) -> DeepSeek-R1(2025, RL-first)

数据质量是 post-training 最重要的方面（Gemini 2.5 Pro 论文特别强调）。

---

### 二、SFT 详解

#### 数据集

(x, y) 指令-响应对，规模 O(10K)~O(100K)

JSON 格式示例：

```json
{
  "prompt": [
    {"role": "system", "content": "You are a helpful, honest assistant."},
    {"role": "user", "content": "What is the capital city of U.S."}
  ],
  "completion": [
    {"role": "assistant", "content": "The capital of the United States is Washington, D.C."}
  ]
}
```

#### 数据质量维度

| 维度 | 期望 |
|---|---|
| Correctness | 事实准确、逻辑健全、与 prompt 一致 |
| Consistency | 风格/格式/推理结构统一 |
| Completeness | 完整解决任务 |
| Clarity | 无多余内容/填充/矛盾 |
| Coverage | 跨领域/复杂度/推理深度 |
| Verifiability | 可验证（数学可重算/代码可执行） |
| Balance | 短简+长复杂任务混合 |
| Alignment | 反映期望语调（helpful/concise/safe） |

常见问题：Label noise / Distribution mismatch / Spurious reasoning

#### 损失函数

NLL (等价 Cross-Entropy)：

$$L_{\text{SFT}}(\theta) = -\mathbb{E}_{(x,y)\sim D} \sum_{t=1}^{T} \log p_\theta(y_t \mid x, y_{<t})$$

等价于：

$$L = -\frac{1}{T} \sum_t \left( z_{t,y^*} - \log \sum_v \exp(z_{t,v}) \right)$$

数值稳定 (log-sum-exp trick)：

$$\log \sum \exp(z) = m + \log \sum \exp(z - m), \quad \text{where } m = \max(z)$$

PyTorch: `F.cross_entropy()` 直接实现数值稳定版本

注意事项：

1. 与预训练损失函数完全相同，区别在于数据是结构化指令对
2. 需要 EOS token 教模型停止
3. 过拟合风险和灾难性遗忘
4. 课程设计：简单->复杂，短->长

#### Batching 和 Padding

- **Dynamic batching (bucketing)**：相似长度分组
- **Packed sequences**：多个短序列拼接
- **Attention mask**: `[1,1,1,0,0]` 确保 pad 不影响 loss

---

### 三、RL 训练技术

#### 统一目标

$$\max_\pi \mathbb{E}_{y \sim \pi(\cdot|x)} [r(x,y)] - \beta \cdot \text{KL}(\pi(\cdot|x) \| \pi_0(\cdot|x))$$

#### 奖励类型对比

| 类型 | 奖励来源 | 适用场景 | 优缺点 |
|---|---|---|---|
| RLHF | Reward Model (人类偏好) | 通用对话/安全 | 鲁棒但昂贵，RM drift |
| RLAIF | LLM Judge + 宪法 | 规模化对齐 | 便宜但 judge bias |
| RLVR | 程序化验证 | 数学/代码 | 精确但稀疏 |
| Process RM | 步级别打分 | 长推理 | 细粒度但标注贵 |
| Rubric-guided | 评分标准聚合 | 多维评估 | 灵活但可能 gaming |

#### RLVR 数据示例

```json
{
  "prompt": [{"role": "user", "content": "Solve: (3x - 2)(x + 5) = 0"}],
  "metadata": {
    "ground_truth": "-5, 0.6666667",
    "reward": 1.0,
    "scorer": "math_grader"
  }
}
```

Grader:

```python
def math_grader(answer):
    gold = [-5, 2/3]
    try:
        pred = [float(s) for s in answer.replace(' ','').split(',')]
        return int(sorted(pred) == gold)
    except:
        return 0
```

#### Reward Model 训练

基于 Bradley-Terry 模型：

$$P(y_1 \succ y_2) = \frac{\exp(r(y_1))}{\exp(r(y_1)) + \exp(r(y_2))}$$

损失函数：

$$L_{\text{pair}}(\theta) = -\mathbb{E}\left[\log \sigma\left(r_\theta(x, y_w) - r_\theta(x, y_l)\right)\right]$$

偏好数据 JSON：

```json
{
  "prompt": [{"role": "user", "content": "What color is the sky?"}],
  "chosen": [{"role": "assistant", "content": "Washington, D.C."}],
  "rejected": [{"role": "assistant", "content": "? The capital is Washington, D.C."}]
}
```

#### PPO 算法

Clipped surrogate：

$$L_{\text{policy}} = -\mathbb{E}_t\left[\min\left(r_t \cdot A_t,\; \text{clip}(r_t, 1-\varepsilon, 1+\varepsilon) \cdot A_t\right)\right]$$

Per-token KL shaping：

$$\tilde{r}_t = -\beta\left(\log \pi_\theta(y_t | s_t) - \log \pi_0(y_t | s_t)\right)$$

训练循环：

1. Sample K answers
2. Score with reward
3. Compute advantages (A = r - b)
4. Update actor (clipped loss) + Train critic (MSE)
5. Control KL
6. Refresh pi_old

#### GRPO 算法 (DeepSeek-V3/R1)

Critic-free。对每个 prompt 采样 K 个回答，组基线：

$$\bar{r} = \frac{1}{N}\sum_i r_i$$

Advantage：

$$A_i = r_i - \bar{r} \quad \text{(或 leave-one-out } A_i = r_i - \bar{r}_{-i}\text{)}$$

Per-token KL：

$$\tilde{r}_t^{KL} = -\beta\left(\log \pi_\theta(y_t | s_t) - \log \pi_0(y_t | s_t)\right)$$

训练循环：

1. Sample K
2. Score
3. Compute advantages (z-score normalize)
4. REINFORCE per token with KL shaping
5. Adapt beta

#### 算法对比

| 算法 | Critic? | RM? | On-policy? | KL控制 | 最佳场景 |
|---|---|---|---|---|---|
| PPO | Yes | Yes | Yes | Global/per-token | RLHF/RLAIF |
| GRPO | No | Yes | Yes | Per-token | RLVR (数学/代码) |
| REINFORCE | No | Yes | Yes | Per-token | 简单高吞吐 |
| DPO | No | No RM | No | Built-in | 便宜稳定偏好调优 |

---

### 四、评估方法

#### Ground Truth Based

| 领域 | 基准 | 指标 |
|---|---|---|
| Math | GSM8K | Exact match accuracy |
| Code | HumanEval | Pass@k |
| MCQ | MMLU | Accuracy |
| 事实性 | TruthfulQA | Truthfulness score |

GSM8K Grader:

```python
def grade_gsm8k(gt_answer, model_output):
    boxed = re.findall(r"\\boxed\{\s*([0-9,]+)\s*\}", model_output)
    if boxed:
        pred = int(boxed[-1].replace(",", ""))
    else:
        nums = re.findall(r"\b[0-9][0-9,]*\b", model_output)
        pred = int(nums[-1].replace(",", "")) if nums else None
    return {
        "gt": gt_answer,
        "pred": pred,
        "correct": pred == gt_answer if pred is not None else False
    }
```

#### LLM Judge Based

Rubric 示例：

```python
judge_prompt = """
RUBRIC:
- Helpfulness (1-7): Does the summary capture the main facts?
- Factuality (1-7): Is the information correct?
- Clarity (1-7): Is the response clear and concise?

Return JSON: {"helpfulness": int, "factuality": int, "clarity": int, "composite": float, "rationale": "..."}
"""
```

#### Human Evaluation

- **Pointwise**: Likert-scale 1-7 评分 (Helpfulness/Factuality/Clarity)
- **Pairwise preference**: 选择 A 或 B 更好

#### 聚合指标

Net Win Rate：

$$\text{Net Win Rate} = \frac{W - L}{W + L}$$

ELO Score：

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

$$R_A' = R_A + K \cdot (S_A - E_A)$$

where $S_A = 1$ (win) / $0.5$ (tie) / $0$ (loss), K 常设 16-32

#### Contamination

评估集出现在训练数据中导致虚高分数。需要：overlap 检查、私有评估集、对异常高分保持警惕。

## Open Questions

- GRPO 在非验证类任务（如开放式创意写作）上的效果如何？
- Process RM 的标注成本能否通过 AI 自动化显著降低？
- DPO 与 PPO 在实际部署中的 trade-off 边界在哪里？
- 如何系统性地检测和缓解 evaluation contamination？

## References

1. Brown et al. "Language Models are Few-Shot Learners." NeurIPS 2020
2. Ouyang et al. "Training language models to follow instructions with human feedback." NeurIPS 2022
3. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning." 2025
4. DeepSeek-AI. "DeepSeek-V3 Technical Report." 2024
5. Zhou et al. "LIMA: Less Is More for Alignment." NeurIPS 2023
6. Rafailov et al. "Direct Preference Optimization: Your Language Model is Secretly a Reward Model." NeurIPS 2023
7. Schulman et al. "Proximal Policy Optimization Algorithms." arXiv 2017
8. Lambert, Nathan. "RLHF Book." 2025
9. Google. "Gemini 2.5 Pro Technical Report." 2025
10. Notion: [Post-training 101](https://www.notion.so/26f4f7b30a5781f8a51dfeaef4c41260)
