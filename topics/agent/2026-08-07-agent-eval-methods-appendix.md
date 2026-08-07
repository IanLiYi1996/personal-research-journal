# 附录：Agent 评估的量化方法手册

> **定位:** 这是 [[2026-08-07-agent-eval-briefing-for-sharing]]（方案层）的**技术附录**。
> 方案层给决策与判据，本附录给公式、估计量、检验与实现 —— **供工程团队落地与被客户追问细节时查证，不建议直接用于客户演示。**
>
> **协作说明:** 本附录由 Codex（GPT-5 系）撰写，我做了独立验证 ——
> ✅ 7 段 Python 代码全部可执行；
> ✅ `pass@k` 与 `pass^k` 两个无偏估计量经蒙特卡洛/精确期望核验，`E[估计量]` 与真值在 10 位小数内一致；
> ✅ naive plug-in 的向下偏倚经数值确认（n=10, k=5, p=0.3 时 0.7597 vs 真值 0.8319）；
> ✅ §4.5 的 Rogan–Gladen 校正公式与其失效条件（Youden index 趋零）陈述正确。
>
> **Date:** 2026-08-07

- **范围:** METHODS only；评估单位为 `task`、一次执行为 `trial`、判分函数为 `grader`
- **记号:** task 数为 \(m\)，task \(i\) 的 trial 数为 \(n_i\)，二元结果 \(Y_{ij}\in\{0,1\}\)
- **默认显著性:** 双侧 \(\alpha=0.05\)，除非预先注册了单侧检验
- **基本纪律:** 先定义 estimand，再选择 estimator、CI 与 decision rule；不得只报告 point estimate

---

## §1 指标定义与估计量

### 1.1 `pass@k`

对固定 task，令独立同分布 trial 的单次成功概率为 \(p\)。
`pass@k` 的 estimand 是 \(k\) 次中至少成功一次的概率：

\[
\operatorname{pass@k}(p)=P\!\left(\sum_{j=1}^{k}Y_j\ge1\right)=1-(1-p)^k.
\]

若共生成 \(n\) 个 exchangeable samples，其中 \(c=\sum_{j=1}^{n}Y_j\) 个成功，且 \(1\le k\le n\)，标准无偏估计量为：

\[
\widehat{\operatorname{pass@k}}
=1-\frac{\binom{n-c}{k}}{\binom{n}{k}}.
\]

当 \(n-c<k\) 时，分子为 \(0\)，估计值为 \(1\)。
无偏性来自：

\[
E\!\left[\frac{\binom{n-C}{k}}{\binom nk}\right]=(1-p)^k,
\quad C\sim\operatorname{Binomial}(n,p).
\]

因此 \(E[\widehat{\operatorname{pass@k}}]=1-(1-p)^k\)。

naive plug-in \(1-(1-\hat p)^k\)，其中 \(\hat p=c/n\)，通常有偏。
对 \(k>1\)，函数 \(g(p)=1-(1-p)^k\) 满足：

\[
g''(p)=-k(k-1)(1-p)^{k-2}\le0.
\]

由 Jensen inequality，\(E[g(\hat p)]\le g(E[\hat p])=g(p)\)，故 naive plug-in 向下偏。
当 \(k=1\) 时，两种估计量都退化为 \(c/n\)，不存在该偏差。

### 1.2 `pass^k`

`pass^k` 的 estimand 是连续 \(k\) 次全部成功的概率：

\[
\operatorname{pass}^{k}(p)=P(Y_1=\cdots=Y_k=1)=p^k.
\]

对应的有限样本无偏估计量为：

\[
\widehat{\operatorname{pass}^{k}}
=\frac{\binom c k}{\binom n k},
\qquad 1\le k\le n.
\]

当 \(c<k\) 时估计值为 \(0\)；其无偏性为
\(E[\binom Ck/\binom nk]=p^k\)。
naive plug-in \(\hat p^k\) 对 \(k>1\) 向上偏，因为 \(p^k\) 是 convex function。

**Decision rule：**

- 有 verifier 且允许从 \(k\) 个候选中选一个时，报告 `pass@k`。
- 用户要求每次都可靠、不能事后挑选时，报告 `pass^k`。
- 必须同时报告 \(k\)、\(n\)、采样参数与 task-level aggregation；只写 “pass rate” 不可审计。
- 若 trial 之间共享 memory、cache 或环境状态，上述 i.i.d. 定义不成立；应把整个 \(k\)-trial policy 作为一次 cluster outcome 直接重复估计。

### 1.3 数值稳定实现

直接计算组合数会在大 \(n\) 时产生巨大整数或浮点溢出。
失败比率可写成：

\[
\frac{\binom{n-c}{k}}{\binom nk}
=\prod_{j=0}^{k-1}\frac{n-c-j}{n-j}
=\prod_{j=0}^{k-1}\left(1-\frac{c}{n-j}\right).
\]

使用 `log1p` 累加对数，再用 `-expm1` 计算 \(1-e^x\)，可避免接近 \(0\) 时的 cancellation：

```python
import math

def pass_at_k_unbiased(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if n - c < k:
        return 1.0
    log_fail = sum(math.log1p(-c / (n - j)) for j in range(k))
    return -math.expm1(log_fail)

def pass_power_k_unbiased(n, c, k):
    if not (0 <= c <= n and 1 <= k <= n):
        raise ValueError("require 0 <= c <= n and 1 <= k <= n")
    if c < k:
        return 0.0
    log_all = sum(math.log(c - j) - math.log(n - j) for j in range(k))
    return math.exp(log_all)

assert abs(pass_at_k_unbiased(10, 2, 3) - (1 - 56 / 120)) < 1e-12
assert abs(pass_power_k_unbiased(10, 5, 2) - (10 / 45)) < 1e-12
```

### 1.4 Average score 与 partial credit

若 grader 输出有界分数 \(S_{ij}\in[0,1]\)，总体 average score 的 trial-level estimand 为：

\[
\mu=E[S],\qquad
\hat\mu=\frac1N\sum_{i=1}^{m}\sum_{j=1}^{n_i}S_{ij},
\quad N=\sum_i n_i.
\]

在 observations 对目标分布 i.i.d. 时，\(\hat\mu\) 对 \(\mu\) 无偏。
若一个 task 有 \(D\) 个 criterion，criterion \(d\) 的得分为 \(s_{id}\in[0,1]\)，预先固定权重 \(w_d\ge0\)，则：

\[
PC_i=\frac{\sum_{d=1}^{D}w_ds_{id}}{\sum_{d=1}^{D}w_d}.
\]

若每个 \(s_{id}\) 都是其 criterion attainment 的无偏估计，且权重不由观测结果选择，则 \(PC_i\) 也是加权目标的无偏估计。
禁止在看到结果后调权重；这会把 estimator 变成 data-dependent optimization。

**Partial-credit decision rule：**

- 二元安全约束使用 hard gate：任一 critical criterion 失败则总结果失败。
- 可补偿的质量维度才使用加权平均。
- 同时发布 `overall pass` 与 criterion-level score，避免相同总分掩盖不同失败结构。

### 1.5 Suite-level macro 与 micro

先计算 task \(i\) 的均值：

\[
\bar S_i=\frac1{n_i}\sum_{j=1}^{n_i}S_{ij}.
\]

Macro average 对每个 task 等权：

\[
\hat\mu_{\text{macro}}=\frac1m\sum_{i=1}^{m}\bar S_i.
\]

Micro average 对每个 trial 等权：

\[
\hat\mu_{\text{micro}}
=\frac{\sum_i n_i\bar S_i}{\sum_i n_i}.
\]

若目标部署分布给 task 权重 \(q_i\)，正确 estimand 是：

\[
\hat\mu_q=\sum_i q_i\bar S_i,\qquad \sum_iq_i=1.
\]

**选择规则：**

- suite 中每个 task 代表一个同等重要的 capability unit：用 macro。
- \(n_i\) 只是为了降低难题方差而人为增加：仍用 macro，不能让多跑的 task 获得更高权重。
- trial 样本确实按生产 exposure rate 抽样，且每个 trial 是目标单位：用 micro。
- 生产 task 频率已知：用 \(q_i\)-weighted average，并报告权重来源。
- 当所有 \(n_i\) 相等时，macro 与 micro 数值相同；其 uncertainty 仍应按 task cluster 估计。

---

## §2 置信区间与样本量

### 2.1 Wilson score interval

对 \(X\sim\operatorname{Binomial}(n,p)\)，\(\hat p=X/n\)，
置信水平 \(1-\alpha\)，令 \(z=z_{1-\alpha/2}\)。
Wilson score interval 为：

\[
\left[
\frac{\hat p+\frac{z^2}{2n}
-z\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}}}
{1+\frac{z^2}n},
\quad
\frac{\hat p+\frac{z^2}{2n}
+z\sqrt{\frac{\hat p(1-\hat p)}n+\frac{z^2}{4n^2}}}
{1+\frac{z^2}n}
\right].
\]

normal/Wald interval \(\hat p\pm z\sqrt{\hat p(1-\hat p)/n}\) 在小 \(n\) 或
\(\hat p\) 接近 \(0,1\) 时覆盖率差，并可能越出 \([0,1]\)。
特别地，\(X=0\) 或 \(X=n\) 时 Wald 标准误为 \(0\)，会给出虚假的零宽区间。

```python
from math import sqrt
from statistics import NormalDist

def wilson_interval(successes, n, confidence=0.95):
    if not (0 <= successes <= n and n > 0):
        raise ValueError("require 0 <= successes <= n and n > 0")
    z = NormalDist().inv_cdf(0.5 + confidence / 2)
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    radius = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return center - radius, center + radius

assert wilson_interval(0, 10)[1] > 0
assert wilson_interval(10, 10)[0] < 1
```

### 2.2 Trials nested within tasks

同一 task 的 trials 通常共享 prompt、grader 与环境难度。
令 cluster size 为 \(r\)，task 内相关系数为 \(\rho\)，则 naive independent-trial
variance 需乘 design effect：

\[
DE=1+(r-1)\rho,\qquad
n_{\text{effective}}\approx\frac{mr}{DE}.
\]

只要 \(\rho>0\)，把 \(mr\) 个 trials 当独立 observation 就会低估方差。
例如 \(r=5,\rho=0.25\) 时 \(DE=2\)，有效样本量仅为 naive count 的一半。

等权 task 的正确基础单位是 \(\bar S_i\)：

\[
\bar S=\frac1m\sum_i\bar S_i,\qquad
SE(\bar S)=\frac{s_{\bar S}}{\sqrt m},
\]

\[
CI_{1-\alpha}=\bar S\pm t_{m-1,1-\alpha/2}\frac{s_{\bar S}}{\sqrt m}.
\]

当 task means 明显偏态、有大量 \(0/1\) 或 \(m\) 较小时，采用 task-cluster bootstrap：

1. 从 \(m\) 个 tasks 中有放回抽取 \(m\) 个 task IDs。
2. 保留每个被抽 task 的全部 trials。
3. 重算完整 suite metric。
4. 重复至少 \(B=2{,}000\) 次，取 percentile CI；`2,000` 是工程精度规则，不是定理。

禁止只在 task 内 bootstrap trials；这没有重建 task-to-task variation。

### 2.3 检测比例差的样本量

比较两个独立 agent 的比例 \(p_0,p_1\)，每组 task 数相同，双侧
\(\alpha\)、power \(1-\beta\)，常用 normal approximation 为：

\[
n_{\text{per arm}}\approx
\frac{
\left[
z_{1-\alpha/2}\sqrt{2\bar p(1-\bar p)}
+z_{1-\beta}\sqrt{p_0(1-p_0)+p_1(1-p_1)}
\right]^2
}{(p_1-p_0)^2},
\quad \bar p=\frac{p_0+p_1}{2}.
\]

取 \(\alpha=0.05\)、power \(=0.80\)、baseline \(p_0=0.50\)，向上取整：

| 目标改善 | \(p_1\) | 每个 agent 所需 tasks | 两组 task-agent observations |
|---|---:|---:|---:|
| 5 percentage points | 0.55 | 1,565 | 3,130 |
| 10 percentage points | 0.60 | 388 | 776 |

该表假设独立、每 task 一个二元 outcome、无 cluster inflation、无 multiple testing。
若每 task 多次 trial，不能把 trials 直接当 tasks；先按 §2.2 计算 design effect。
若两个 agent 在同一 task set 上运行，应使用 paired design；其样本量由 discordant-pair
rate 决定，不能机械套用上表。

最坏情形下，为估计单个比例并使 95% CI half-width 不超过 \(h\)，可用：

\[
n\ge\frac{z_{0.975}^2}{4h^2}.
\]

该式在 \(p=0.5\) 处最保守；\(h=0.05\) 时需 \(n\ge385\)。

### 2.4 Paired comparison 与 McNemar's test

两个 agent 在同一批 tasks 上比较时，定义：

- \(b\)：agent A 成功、agent B 失败的 task 数；
- \(c\)：agent A 失败、agent B 成功的 task 数。

McNemar 只使用 discordant pairs。
大样本、无 continuity correction 的统计量为：

\[
\chi^2=\frac{(b-c)^2}{b+c}\ \overset{H_0}{\sim}\ \chi^2_1.
\]

continuity-corrected 版本为：

\[
\chi^2_{\text{cc}}=\frac{(|b-c|-1)^2}{b+c}.
\]

当 \(b+c<25\) 时，使用 exact McNemar：

\[
B\mid(B+C=b+c,H_0)\sim\operatorname{Binomial}(b+c,0.5).
\]

双侧 exact \(p\)-value 可取
\(\min\{1,2P[X\le\min(b,c)]\}\)。
`25` 是保守工程切换规则，不是数学边界。

**Decision rule：**

- 同 tasks、同随机种子或同环境快照：优先 paired test。
- 不同且独立的 task samples：使用 two-proportion test 或 regression adjustment。
- 同时报告 effect \(\hat\Delta=(c-b)/m\)（B 相对 A）与 CI；不能只报告 \(p\)-value。
- task 有多次 trials 时，先形成每 task 的 paired mean difference，再对 task differences 做
  paired \(t\)-interval、permutation test 或 cluster bootstrap。

---

## §3 打分器的实现

以下代码仅依赖 Python standard library。
所有 grader 都返回结构化结果，避免只返回不可诊断的 boolean。

### 3.1 Outcome/state verification

`exact=True` 要求整个状态一致；`exact=False` 只要求 expected state 是 actual state 的递归子集。
list 使用精确顺序语义；无序集合应在进入 grader 前规范化排序。

```python
def verify_state(actual, expected, exact=True, path="$"):
    errors = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{path}: expected dict, got {type(actual).__name__}"]
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"{path}.{key}: missing")
            else:
                errors += verify_state(actual[key], value, exact, f"{path}.{key}")
        if exact:
            for key in actual.keys() - expected.keys():
                errors.append(f"{path}.{key}: unexpected")
    elif isinstance(expected, list):
        if not isinstance(actual, list):
            errors.append(f"{path}: expected list, got {type(actual).__name__}")
        elif len(actual) != len(expected):
            errors.append(f"{path}: length {len(actual)} != {len(expected)}")
        else:
            for i, (got, want) in enumerate(zip(actual, expected)):
                errors += verify_state(got, want, exact, f"{path}[{i}]")
    elif actual != expected:
        errors.append(f"{path}: {actual!r} != {expected!r}")
    return errors

def state_grade(actual, expected, exact=True):
    errors = verify_state(actual, expected, exact)
    return {"pass": not errors, "score": float(not errors), "errors": errors}

assert state_grade({"order": {"status": "paid", "id": 7}},
                   {"order": {"status": "paid"}}, exact=False)["pass"]
```

**State decision rule：** side effects、权限与不可变字段必须放入 expected assertions；
只验证 agent 的自然语言 completion message 不构成 outcome verification。

### 3.2 Trajectory matching

每个 tool call 规范为 `{"name": str, "arguments": JSON-like value}`。
本实现的语义为：

- `strict`：完全相同且顺序相同；
- `unordered`：multiset 相同，顺序可不同；
- `subset`：actual 是 expected 的保序子序列，允许 agent 省略参考步骤；
- `superset`：expected 是 actual 的保序子序列，允许 agent 增加步骤。

```python
import json
from collections import Counter

def _key(call):
    return json.dumps(call, sort_keys=True, separators=(",", ":"), ensure_ascii=True)

def _is_subsequence(xs, ys):
    it = iter(ys)
    return all(any(x == y for y in it) for x in xs)

def match_trajectory(actual, expected, mode="strict"):
    a, e = list(map(_key, actual)), list(map(_key, expected))
    if mode == "strict":
        ok = a == e
    elif mode == "unordered":
        ok = Counter(a) == Counter(e)
    elif mode == "subset":
        ok = _is_subsequence(a, e)
    elif mode == "superset":
        ok = _is_subsequence(e, a)
    else:
        raise ValueError("mode must be strict, unordered, subset, or superset")
    return {"pass": ok, "score": float(ok), "mode": mode}

calls = [
    {"name": "search", "arguments": {"q": "invoice"}},
    {"name": "open", "arguments": {"id": 3}},
]
assert match_trajectory(calls[::-1], calls, "unordered")["pass"]
assert match_trajectory(calls[:1], calls, "subset")["pass"]
```

**Trajectory decision rule：** 只有顺序本身属于 contract 时才用 `strict`；
能力评分默认以 outcome 为主，trajectory match 作为 constraint 或 diagnosis。

### 3.3 Weighted multi-grader aggregation

先对各 grader score 做 \([0,1]\) range check，再按固定权重聚合。
`critical` grader 采用 hard gate：

\[
G=\frac{\sum_dw_ds_d}{\sum_dw_d},\qquad
\operatorname{pass}=\mathbb{1}[G\ge\tau]\prod_{d\in C}\mathbb{1}[s_d=1].
\]

```python
def aggregate_graders(results, threshold=0.8):
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be in [0, 1]")
    total_weight = 0.0
    weighted = 0.0
    critical_ok = True
    for item in results:
        score = float(item["score"])
        weight = float(item.get("weight", 1.0))
        if not 0 <= score <= 1 or weight < 0:
            raise ValueError("scores must be in [0,1], weights must be nonnegative")
        total_weight += weight
        weighted += weight * score
        if item.get("critical", False) and score < 1.0:
            critical_ok = False
    if total_weight <= 0:
        raise ValueError("positive total weight required")
    score = weighted / total_weight
    return {"pass": critical_ok and score >= threshold,
            "score": score, "critical_ok": critical_ok}

assert not aggregate_graders([
    {"score": 1.0, "weight": 3},
    {"score": 0.0, "weight": 1, "critical": True},
])["pass"]
```

阈值 \(\tau\) 与权重必须在 holdout evaluation 前冻结。
若通过调参选择了 \(\tau\)，最终性能必须在未参与选阈值的 calibration holdout 上报告。

### 3.4 Partial credit decomposition

每个 criterion 返回 score、weight 与 contribution：

```python
def partial_credit(criteria):
    if not criteria:
        raise ValueError("at least one criterion required")
    total = sum(float(c["weight"]) for c in criteria)
    if total <= 0:
        raise ValueError("positive total weight required")
    parts = []
    for c in criteria:
        score, weight = float(c["score"]), float(c["weight"])
        if not 0 <= score <= 1 or weight < 0:
            raise ValueError("invalid score or weight")
        parts.append({
            "name": c["name"],
            "score": score,
            "normalized_weight": weight / total,
            "contribution": score * weight / total,
        })
    return {"score": sum(p["contribution"] for p in parts), "parts": parts}

grade = partial_credit([
    {"name": "correct_state", "score": 1.0, "weight": 5},
    {"name": "policy", "score": 0.5, "weight": 3},
    {"name": "format", "score": 1.0, "weight": 2},
])
assert abs(grade["score"] - 0.85) < 1e-12
```

对 criterion \(d\)，suite 还应报告
\(\bar s_d=m^{-1}\sum_i s_{id}\) 与 task-cluster CI。
总分不能替代 decomposition；发布 gate 至少包含总分阈值和 critical criteria 两部分。

---

## §4 LLM 判分器的校准与验证

LLM judge 是 measuring instrument，不是 ground truth。
校准单位应与部署单位一致：同一 input、trajectory、outcome evidence 与 rubric。

### 4.1 Gold set 与 confusion matrix

令 human gold 为真实类别，success 为 positive；judge verdict 为预测类别：

| | Human success | Human fail |
|---|---:|---:|
| Judge success | \(TP\) | \(FP\) |
| Judge fail | \(FN\) | \(TN\) |

\[
\text{Accuracy}=\frac{TP+TN}{N},\quad
\text{Sensitivity}=\text{Success-recall}=\frac{TP}{TP+FN},
\]
\[
\text{Specificity}=\text{Fail-recall}=\frac{TN}{TN+FP},\quad
\text{Balanced accuracy}=\frac{\text{Sensitivity}+\text{Specificity}}2.
\]

若 gold success prevalence 为 \(\pi\)，一个恒报 fail 的 judge 有
\(\text{Accuracy}=1-\pi\)；当 \(\pi\ll0.5\) 时 raw accuracy 可很高但
success-recall 为 \(0\)。因此 class imbalance 下必须报告 balanced accuracy 和两个 per-class recalls。

以横轴 success-recall、纵轴 fail-recall 定义 strict/lenient bias plane：

- 理想点为 \((1,1)\)。
- strict bias：success-recall 低、fail-recall 高，即高 false-negative rate。
- lenient bias：success-recall 高、fail-recall 低，即高 false-positive rate。
- 无信息的随机/常量边界满足 sensitivity + specificity \(\approx1\)。

建议验收规则：两个 recall 的 95% Wilson lower bound 都超过业务下限
\((r_S,r_F)\)，而不是只要求 balanced accuracy 达标。
\((r_S,r_F)\) 由 false accept / false reject 损失预先确定；任何固定数值都是工程判断，不是统计定理。

### 4.2 Cohen's kappa

对两个 raters 的 \(K\) 类标签，观察一致率：

\[
p_o=\frac1N\sum_{i=1}^{N}\mathbb 1(A_i=B_i).
\]

令 rater A、B 对类别 \(c\) 的边际比例分别为 \(p_{Ac},p_{Bc}\)，chance agreement 为：

\[
p_e=\sum_{c=1}^{K}p_{Ac}p_{Bc},\qquad
\kappa=\frac{p_o-p_e}{1-p_e}.
\]

\(\kappa=1\) 表示完全一致，\(\kappa=0\) 表示不高于边际分布所给的 chance agreement，
\(\kappa<0\) 表示低于 chance。若 \(p_e=1\)，kappa 未定义。
judge-vs-human 与 human-vs-human 必须在同一 gold subset 上分别计算；
前者显著低于后者表明 judge 增加了额外 measurement error。

可采用以下内部 release bands：\(\kappa\ge0.80\) 可用于自动 gate；
\(0.67\le\kappa<0.80\) 仅用于 provisional decision 并人工抽检；
\(\kappa<0.67\) 不用于自动 gate。该分段是工程治理规则，不是普适定理。

### 4.3 Krippendorff's alpha

alpha 支持多 raters、缺失标签和 nominal/ordinal/interval distance。
对 unit \(u\)，类别 \(c\) 的标签数为 \(n_{uc}\)，有效 rater 数为 \(m_u\ge2\)。
coincidence matrix 定义为：

\[
o_{cc'}=\sum_u\frac{n_{uc}(n_{uc'}-\mathbb1[c=c'])}{m_u-1}.
\]

令 \(n_c=\sum_{c'}o_{cc'}\)、\(n=\sum_c n_c\)，chance coincidence 为：

\[
e_{cc'}=
\begin{cases}
n_cn_{c'}/(n-1),&c\ne c',\\
n_c(n_c-1)/(n-1),&c=c'.
\end{cases}
\]

给定 disagreement function \(\delta(c,c')\)，

\[
D_o=\frac1n\sum_{cc'}o_{cc'}\delta(c,c'),\quad
D_e=\frac1n\sum_{cc'}e_{cc'}\delta(c,c'),\quad
\alpha_K=1-\frac{D_o}{D_e}.
\]

binary/nominal verdict 使用 \(\delta(c,c')=\mathbb1[c\ne c']\)。
\(D_e=0\) 时 alpha 未定义。解释范围与 kappa 类似，但 alpha 与 kappa 不应混报成同一统计量。
rubric 多等级分数若有真实顺序，应预先定义 ordinal distance；否则使用 nominal distance。

### 4.4 Gold set 样本量

gold sizing 应分别针对 success 与 fail 两个 strata。
若希望 sensitivity 或 specificity 的 95% CI half-width 不超过 \(h\)，
未知真实 recall 时使用 §2 的保守式：

\[
n_{\text{class}}\ge\frac{1.96^2}{4h^2}.
\]

| 目标 half-width \(h\) | gold success 数 | gold fail 数 | balanced gold 总数 |
|---:|---:|---:|---:|
| 0.05 | 385 | 385 | 770 |
| 0.03 | 1,068 | 1,068 | 2,136 |
| 0.02 | 2,401 | 2,401 | 4,802 |

这是 worst-case approximation；最终报告 Wilson interval。
若生产 prevalence 极不平衡，可分层过采样少数类以估计 recall，再按生产 prevalence 计算 accuracy；
不得把 balanced gold 的 50/50 prevalence 当作生产 prevalence。

### 4.5 Prevalence correction 与 bias correction

设 gold 上测得 judge sensitivity \(Se\)、specificity \(Sp\)，
生产评估中 judge 观察到的 pass rate 为 \(p_{\text{obs}}\)。
measurement equation 为：

\[
p_{\text{obs}}=Se\cdot p_{\text{true}}+(1-Sp)(1-p_{\text{true}}).
\]

反解得到：

\[
\boxed{
p_{\text{true}}
=\frac{p_{\text{obs}}-(1-\text{specificity})}
{\text{sensitivity}-(1-\text{specificity})}
}
=\frac{p_{\text{obs}}+Sp-1}{Se+Sp-1}.
\]

分母 \(J=Se+Sp-1\) 是 Youden index。
当 \(J\) 接近 \(0\) 时，judge 几乎无辨别力，校正会放大微小估计误差，公式不可用。
内部可规定 \(|J|<0.20\) 时禁用校正；`0.20` 是工程 guardrail，不是理论临界值。

校正还要求 gold 与目标数据上的 \(Se,Sp\) 可迁移。
若解落在 \([0,1]\) 外，应报告未截断值作为 model-misfit signal，同时可给 constrained estimate
\(\min(1,\max(0,\hat p_{\text{true}}))\)；CI 应对 gold tasks 和目标 tasks 分层 bootstrap，
把 \(Se,Sp,p_{\text{obs}}\) 的不确定性全部传播进去。

### 4.6 Test-retest reliability

在完全相同 evidence 上，以相同 judge configuration 独立运行 \(R\ge2\) 次。
对 \(N\) 个 items，pairwise verdict flip rate 为：

\[
\widehat F=
\frac{\sum_{i=1}^{N}\sum_{1\le r<s\le R}
\mathbb1(V_{ir}\ne V_{is})}
{N\binom R2}.
\]

同时报告 success-to-fail 与 fail-to-success 的方向性 flip rates。
CI 应 bootstrap items，不能把同一 item 的 \(\binom R2\) 对比较当独立样本。
temperature \(>0\) 时 flip rate 通常增加；temperature \(=0\) 也不能假定 bitwise determinism。

可把自动 gate 的稳定性要求设为 \(\widehat F\le f_{\max}\)，并要求其 95% upper bound
不超过 \(f_{\max}\)。\(f_{\max}\) 应由 verdict 翻转成本确定，是工程阈值。

### 4.7 Ensembling 与 self-consistency

若 \(R\) 个独立 judge 的相同错误率为 \(e<0.5\)，奇数 \(R\) 的 majority-vote error 为：

\[
P_{\text{ens-error}}
=\sum_{j=(R+1)/2}^{R}\binom Rj e^j(1-e)^{R-j}.
\]

该下降依赖独立或弱相关错误。
若所有 judges 对某类样本存在相同 systematic bias，令该类 conditional bias 为 \(b\)，
增加 \(R\) 只降低随机方差，不保证 \(b\to0\)。

**Decision rule：** ensemble 前先在 gold set 上检查 error correlation 与 per-class recall；
若多个 judge 的 false accepts 高度重合，增加 self-consistency samples 不能替代 rubric 修订或 human calibration。

---

## §5 成本与效率的联合评估

### 5.1 Cost accounting

将一次 evaluation campaign 的成本拆为 fixed 与 variable：

\[
C_{\text{total}}=C_{\text{fixed}}+
\sum_{i=1}^{m}\sum_{j=1}^{n_i}
(C^{agent}_{ij}+C^{env}_{ij}+C^{judge}_{ij}+C^{human}_{ij}).
\]

一次 task-trial 的 token cost：

\[
C^{agent}_{ij}
=\sum_{\ell\in calls}
\left(t^{in}_{\ell}r^{in}_{\ell}
+t^{out}_{\ell}r^{out}_{\ell}\right)+C^{tool}_{ij}+C^{compute}_{ij},
\]

其中 token price \(r\) 使用 cost/token，而不是 cost/million tokens。
若 judge 抽样率为 \(q\)、每个样本运行 \(R_J\) 次，平均 judge cost 为
\(qR_J\bar C^{judge}\)。

评估规模 \(M\) 个 task-trials 时：

\[
C_{\text{eval}}=C_{\text{fixed}}+M(\bar C^{agent}+\bar C^{env}+qR_J\bar C^{judge}).
\]

训练期若有 \(E\) 轮、每轮 \(M_{\text{train}}\) 个 judge calls：
\[
C_{\text{train-judge}}=E\,M_{\text{train}}\,R_J\,\bar C^{judge}.
\]

fixed cost 只在明确 amortization horizon \(H\) 下分摊：
\(C_{\text{fixed/task}}=C_{\text{fixed}}/H\)。

### 5.2 Pareto frontier

方案 \(a\) 被方案 \(b\) dominated，当且仅当：

\[
Q_b\ge Q_a,\quad C_b\le C_a,
\]

且至少一个不等式严格成立。未被任何方案 dominated 的集合为 Pareto frontier。

```python
def pareto_frontier(rows):
    frontier = []
    for i, row in enumerate(rows):
        dominated = any(
            j != i
            and other["quality"] >= row["quality"]
            and other["cost"] <= row["cost"]
            and (other["quality"] > row["quality"]
                 or other["cost"] < row["cost"])
            for j, other in enumerate(rows)
        )
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda r: (r["cost"], -r["quality"]))

plans = [
    {"name": "A", "quality": 0.70, "cost": 0.10},
    {"name": "B", "quality": 0.75, "cost": 0.20},
    {"name": "C", "quality": 0.69, "cost": 0.15},
]
assert [x["name"] for x in pareto_frontier(plans)] == ["A", "B"]
```

frontier 应使用质量与成本的 CI 做 sensitivity analysis；
point estimate dominated 但区间重叠的方案不能声称确定劣势。

### 5.3 Cost per successful task

若每次 attempt 的平均成本为 \(\bar C\)，成功率为 \(p>0\)：

\[
C_{\text{per success}}=\frac{\bar C}{p}.
\]

例：\(\bar C=\$0.12\)，\(p=0.60\)，则
\(C_{\text{per success}}=\$0.12/0.60=\$0.20\)。
样本估计应使用 \(\sum_i C_i/\sum_iY_i\)，并按 task bootstrap ratio CI；
不要用 `mean(cost_i / Y_i)`，因为失败时分母为 \(0\)。

### 5.4 Escalation baseline

cheap model 成本 \(c_L\)、成功率 \(p_L\)；仅在失败时调用 expensive model，
其条件成本为 \(c_H\)、条件成功率为 \(p_{H\mid Lfail}\)。

\[
E[C_{\text{esc}}]=c_L+(1-p_L)c_H,
\]
\[
P(S_{\text{esc}})=p_L+(1-p_L)p_{H\mid Lfail}.
\]

若每次 cheap result 都需 verifier 成本 \(c_V\)，且 expensive result 再验证一次：

\[
E[C_{\text{esc+verify}}]=c_L+c_V+(1-p_L)(c_H+c_V).
\]

选择 escalation 的 decision rule 是同时比较
\((P(S_{\text{esc}}),E[C_{\text{esc}}])\) 与单模型、retry 基线的 Pareto dominance，
不能假设 \(p_{H\mid Lfail}=p_H\)；困难样本上的 conditional success 通常不同。

---

## §6 数据集构造与统计效力

### 6.1 Stratified task selection

按 difficulty \(d\) 与 capability \(g\) 建 strata \(h=(d,g)\)。
stratum \(h\) 的样本均值为 \(\bar Y_h\)，目标部署权重为 \(W_h\)，则：

\[
\hat p_{\text{strat}}=\sum_hW_h\bar Y_h,\qquad \sum_hW_h=1.
\]

独立 strata 下：

\[
\widehat{\operatorname{Var}}(\hat p_{\text{strat}})
=\sum_hW_h^2(1-f_h)\frac{s_h^2}{n_h},
\]

其中 \(f_h=n_h/N_h\) 为 finite-population sampling fraction；总体很大时可忽略。
均衡抽样用于提高各 strata precision，发布总体指标时必须恢复 \(W_h\)，不能直接 macro。

推荐 allocation：需要各 capability 同精度时每层等量；
总样本固定且只优化总体方差时采用 Neyman allocation
\(n_h\propto W_hS_h\)。两者对应不同 estimand priority。

### 6.2 Positive/negative cases

对“是否触发工具/升级/拒答”的 policy，定义 should-trigger 为 positive：

\[
\text{Trigger-recall}=\frac{TP}{TP+FN},\qquad
\text{No-trigger-precision}=\frac{TN}{TN+FN}.
\]

另报告：
\[
\text{Trigger-precision}=\frac{TP}{TP+FP},\qquad
\text{False-trigger-rate}=\frac{FP}{FP+TN}.
\]

只有 positive cases 会使 false-trigger-rate 不可识别；只有 negative cases 会使 trigger-recall 不可识别。
数据集应保证每个关键 stratum 中 \(n_+>0,n_->0\)，并分别给 Wilson CI。
balanced set 用于诊断，生产加权指标使用真实 prevalence。

### 6.3 Holdout keyed to intended generality

| Intended generality | 必须 hold out 的单位 | 禁止的 split |
|---|---|---|
| 同模板新实例 | instance IDs | 同一实例改写后跨 split |
| 同 task type 新模板 | template / generator seed family | sibling templates 跨 split |
| 新 capability 组合 | capability-combination cells | 只随机拆单条 records |
| 新 domain / tenant | entire domain / tenant | 同 tenant 的近重复记录跨 split |
| 未来流量 | time blocks | 随机混合未来与过去 |

若声称跨 domain 泛化，test set 必须包含训练与调参完全未见的 domains。
所有 prompt、grader rubric、threshold 与 weights 在 final holdout 前冻结；
每次查看 holdout 并据此修改系统都消耗 holdout，需更换新 holdout 或使用 alpha spending。

### 6.4 Contamination checks

**Canary strings：** 为每个隐藏 task 生成高熵随机串 \(z_i\)，不出现在公开 prompt 中。
若 model output、tool query 或 retrieved document 出现 exact \(z_i\)，标记 contamination：

\[
\operatorname{CanaryHit}_i=\mathbb1[z_i\in O_i\cup Q_i\cup R_i].
\]

canary 至少使用 128 bits entropy；该阈值是避免偶然碰撞的工程规则。

**n-gram overlap：** 令 \(G_n(x)\) 为文本 \(x\) 的去重 token n-grams：

\[
J_n(T,O)=\frac{|G_n(T)\cap G_n(O)|}{|G_n(T)\cup G_n(O)|},
\quad
C_n(T,O)=\frac{|G_n(T)\cap G_n(O)|}{\min(|G_n(T)|,|G_n(O)|)}.
\]

在 clean reference runs 上按 task type 建立 \(C_n\) 分布。
具体 flag rule：使用 \(n=8\)，若
\(C_8>Q_3+3IQR\) 且重叠 n-grams 数 \(\ge5\)，进入人工审计。
该阈值是工程 heuristic；代码、固定 schema 与用户要求复述应单独建 baseline。

**Eval-awareness token outlier：** 在同 difficulty × capability stratum 内，令 token consumption 为 \(T_i\)，
中位数为 \(\tilde T_h\)。具体规则：

\[
\operatorname{Flag}_i=\mathbb1[T_i>3\tilde T_h]
\land\mathbb1\!\left[\frac{T_i-\tilde T_h}{1.4826\,MAD_h}>3.5\right].
\]

当 \(MAD_h=0\) 时仅使用 \(T_i>3\tilde T_h\)。
`3× median` 与 robust z-score `3.5` 都是工程筛查阈值，不证明 eval awareness；
flag 后检查 tool queries 是否包含 benchmark 名称、答案库路径、canary 或 evaluator artifacts。

---

## §7 回归与漂移的统计控制

### 7.1 Regression gate 是 hypothesis test

令同一 task 上新旧系统的 paired difference 为
\(D_i=S_{i,\text{new}}-S_{i,\text{old}}\)，目标差异为
\(\Delta=E[D_i]\)，不可接受回归 margin 为 \(\epsilon\ge0\)。

“确认发生至少 \(\epsilon\) 的回归”检验：

\[
H_0:\Delta\ge-\epsilon,\qquad H_1:\Delta<-\epsilon.
\]

若 \(\Delta\) 的 one-sided \(1-\alpha\) upper confidence bound
\(U_\Delta<-\epsilon\)，则拒绝 \(H_0\)，判定为 real regression。
当 \(\epsilon=0\) 时，规则退化为 \(U_\Delta<0\)。

“证明新版本 non-inferior”是不同方向的检验：

\[
H_0:\Delta\le-\epsilon,\qquad H_1:\Delta>-\epsilon.
\]

只有 one-sided lower bound \(L_\Delta>-\epsilon\) 才能通过 non-inferiority gate。
若 \(L_\Delta\le-\epsilon\le U_\Delta\)，结论是 inconclusive，应增加 tasks 或 trials；
不能因 point estimate 高于阈值就宣布通过。

binary paired outcomes 使用 McNemar / paired bootstrap；连续 partial score 使用 task-level
paired \(t\)-interval、permutation 或 bootstrap。
margin \(\epsilon\)、方向、\(\alpha\)、主指标与 exclusions 必须在运行前冻结。

### 7.2 Sequential testing 与 alpha spending

若每次 commit 都以固定 \(\alpha\) 独立测试，运行 \(T\) 次至少一次 false alarm 的概率为：

\[
FWER=1-(1-\alpha)^T\le T\alpha.
\]

无限序列可预先分配：

\[
\alpha_t=\frac{\alpha}{t(t+1)},\qquad
\sum_{t=1}^{\infty}\alpha_t=\alpha.
\]

第 \(t\) 次 confirmatory run 仅在 \(p_t\le\alpha_t\) 时拒绝。
同一 run 有 \(K\) 个指标时，再使用 Holm rule：排序
\(p_{(1)}\le\cdots\le p_{(K)}\)，依次与
\(\alpha_t/(K-j+1)\) 比较，首次不通过后停止拒绝。

**Practical fix：** 每 commit 的 suite 作为 exploratory monitoring，不生成新的显著性声明；
只对固定 release candidate 做 confirmatory test，并使用预注册 alpha spending。
反复 rerun 到通过属于 optional stopping，不得把最后一次 \(p\)-value 当有效证据。

### 7.3 Control-chart drift detection

对时间窗 \(t\) 的在线成功率 \(\hat p_t=X_t/n_t\)，稳定基线为 \(\bar p\)。
binomial p-chart 的 3-sigma limits：

\[
UCL_t=\min\!\left(1,\bar p+3\sqrt{\frac{\bar p(1-\bar p)}{n_t}}\right),
\quad
LCL_t=\max\!\left(0,\bar p-3\sqrt{\frac{\bar p(1-\bar p)}{n_t}}\right).
\]

若流量有 tenant/task clustering，使用 baseline residual 的 empirical variance 或
beta-binomial variance，不能强行使用 binomial standard error。

EWMA 对小而持续的 drift 更敏感：

\[
Z_t=\lambda\hat p_t+(1-\lambda)Z_{t-1},\quad Z_0=\bar p,
\]

\[
UCL/LCL_t=\bar p\pm
L\sigma_{\hat p}\sqrt{\frac{\lambda}{2-\lambda}
\left[1-(1-\lambda)^{2t}\right]}.
\]

固定窗口大小时 \(\sigma_{\hat p}=\sqrt{\bar p(1-\bar p)/n}\)；
窗口大小变化时按每窗 variance 递推。
初始工程配置可取 \(\lambda=0.2,L=3\)，并以历史 false-alert rate 回测调参；
这些值不是定理。

告警规则可设为任一点越过 3-sigma limit，或连续 8 点位于 center line 同侧。
告警后按 difficulty、capability、tenant、model version 与 judge version 分层定位；
未经分层的总体稳定可能掩盖 subgroup drift。

---

## §8 速查表

| 指标 / 方法 | 何时用 | 公式 / rule | 主要陷阱 |
|---|---|---|---|
| `pass@k` | \(k\) 次可择优 | \(1-\binom{n-c}k/\binom nk\) | naive plug-in 向下偏；需 i.i.d. |
| `pass^k` | 连续可靠性 | \(\binom ck/\binom nk\) | \(\hat p^k\) 向上偏 |
| Average score | 连续/部分分 | \(\hat\mu=N^{-1}\sum S\) | 忽略 task clustering |
| Partial credit | 可补偿 criteria | \(\sum w_ds_d/\sum w_d\) | 事后调权；安全项不应补偿 |
| Macro average | task 等重要 | \(m^{-1}\sum_i\bar S_i\) | 与生产 exposure 不一致 |
| Micro average | trial/exposure 等重要 | \(\sum n_i\bar S_i/\sum n_i\) | 多跑的 task 被意外加权 |
| Wilson CI | 二元比例 | §2.1 exact formula | 小样本使用 Wald interval |
| Cluster CI | trials nested in tasks | \(SE=s_{\bar S}/\sqrt m\) | bootstrap trials 而非 tasks |
| Sample size | 预先设计 power | \(n\propto1/\Delta^2\) | 把 trials 当独立 tasks |
| McNemar | 同 tasks 比较两 agents | \((b-c)^2/(b+c)\) | 忽略 pairing；小样本不用 exact |
| Balanced accuracy | judge 类别不均衡 | \((Se+Sp)/2\) | raw accuracy 被多数类支配 |
| Cohen's kappa | 两 raters | \((p_o-p_e)/(1-p_e)\) | prevalence 改变 \(p_e\) |
| Krippendorff's alpha | 多 raters/缺失 | \(1-D_o/D_e\) | disagreement distance 未定义 |
| Corrected pass rate | judge 有已知 \(Se,Sp\) | \((p_{obs}+Sp-1)/(Se+Sp-1)\) | \(Se+Sp-1\approx0\) 时爆炸 |
| Flip rate | stochastic judge | discordant repeat pairs / all pairs | pair comparisons 非独立 |
| Pareto frontier | 质量-成本联合选择 | quality 高且 cost 低的 dominance | 只比较 point estimates |
| Cost/success | 部署经济性 | \(\sum C/\sum Y\) | 对每条计算 \(C/Y\) |
| Escalation | cheap-first baseline | \(c_L+(1-p_L)c_H\) | 用 \(p_H\) 代替条件成功率 |
| Stratified estimate | suite 与生产 mix 不同 | \(\sum_hW_h\bar Y_h\) | balanced sample 未恢复权重 |
| Trigger metrics | action policy | recall 与 no-trigger precision | 单边数据无法识别另一类错误 |
| Canary / overlap | contamination 筛查 | exact hit；\(C_8>Q_3+3IQR\) | heuristic 被误当证明 |
| Regression gate | 比较新旧版本 | \(U_\Delta<-\epsilon\) 判回归 | 看 point estimate；方向写反 |
| Alpha spending | 重复 suite | \(\alpha_t=\alpha/[t(t+1)]\) | 每 commit 固定用 0.05 |
| p-chart / EWMA | 在线 drift | 3-sigma / EWMA limits | 忽略 overdispersion 与 mix shift |

**最小发布规则：**
\[
\text{Release}=
\mathbb1[L_{\Delta}>-\epsilon]\,
\mathbb1[L_{Se}\ge r_S]\,
\mathbb1[L_{Sp}\ge r_F]\,
\mathbb1[C\le C_{\max}],
\]
其中所有 lower bounds、margin、recall floors 与 cost ceiling 必须预注册；
任一 critical safety grader 失败时强制令 `Release=0`。
