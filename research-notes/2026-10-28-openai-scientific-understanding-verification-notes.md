# 独立核查：Lean 的可信边界与“生成—验证—人工理解”

- **Date:** 2026-10-28
- **Tags:** Lean, formal-verification, Hypothesis, AI-for-science, trust-boundaries

配套正文：[AI 给出证明之后，科学还需要什么？](/research-notes/2026-10-28-openai-scientific-understanding.md)。

核查日期：2026-10-28。

范围：为主任务提供方法论补充，仅核查 Lean 证明的可信边界，以及个人研究、工程中可以如何建立验证闭环。不搜索、读取或评价 OpenAI 原文中的三项新成果；本文不能作为那些成果的核查依据。

执行状态：已实际读取下列官方文档及维护者仓库。**本次只做资料核查，未安装工具、编写或运行 Lean 证明、运行 Hypothesis 测试、执行 Comparator，亦未开展任何实验。** 文中的试点、命令和验收阈值均为待实施方案。全文区分“官方事实”“推论/建议”；建议不代表官方承诺。

## 1. 可供主任务引用的结论

建议将 AI 放在“提出候选、拆分引理、寻找证明路线”的位置，将验证器放在“检查精确命题和证明”的位置，将研究者放在“确定问题、审查假设与定义、理解适用边界”的位置。三者不能互相替代。

官方文档明确区分“定理是否有有效证明”与“定理陈述是什么意思”，并把未经审阅的 AI 证明、程序列入需要防范欺骗性输入的情形。[Lean：Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

由此得到的推论：机器验证可以给特定形式化命题提供有力证据，但不能自动保证自然语言问题被正确翻译、现实模型成立、研究结论具有新颖性，或作者真正理解了证明。对工程而言，证明一个 Lean 模型，也不能自动覆盖另一份 Python/C++ 实现及部署环境。

## 2. 官方事实：Lean 到底检查了什么

### F1. 内核检查与运行程序不是同一件事

Lean 将用户语法 elaboration 为核心类型论中的表达式，再由内核检查；编译器则生成可运行程序。命令 elaboration 可以执行任意 IO，系统也具有初始化代码、可扩展语法等机制。因此，读取或构建陌生证明工程不是天然无副作用的操作。[Lean：Elaboration and Compilation](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)

### F2. 编辑器蓝色双勾有明确的信任前提

蓝色双勾表示内核接受了当前环境中的命题及证明；它不自动排除依赖项中的 `sorry`，也不证明符号、类型类、定义确实表达了人类原意。官方把 `lake build +Module` 无错误、无警告的成功构建列为相应检查方式。[Lean：Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

### F3. 公理要审计传递依赖，不能只搜索当前文件

`#print axioms theoremName` 会显示定理及其依赖间接使用的公理；`sorry` 会表现为 `sorryAx`。Lean 的标准公理是 `propext`、`Classical.choice`、`Quot.sound`，不要求每个证明都使用全部三项。自定义公理是额外假设，不是证明出来的事实；不一致的公理会破坏证明的基础。[Lean：Axioms](https://lean-lang.org/doc/reference/latest/Axioms/)

建议：普通严格试点将实际公理集合限定为上述三项的子集，允许空集合；对任何额外项先拒收，再人工解释。不要把“使用标准公理”误写成“不依赖任何假设”。

### F4. 重新检查增加保障，但不是独立实现的验证

官方维护者说明，旧 `lean4checker` 自 Lean 4.28.0 起已并入工具链，名称为 `leanchecker`。`--fresh` 将模块及导入中的常量重新放入新环境接受内核检查；它仍使用 Lean 内核，不能称为独立内核。[官方 lean4checker 仓库 README](https://github.com/leanprover/lean4checker)

官方验证文档也提醒：该类重新检查仍信任 `.olean` 的结构；恶意构建代码可能危及系统，不能把事后重检当作运行陌生代码的安全隔离。[Lean：Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

### F5. 对不可信输入，要同时保护构建环境和目标命题

官方的更高保障流程是：在可信环境确定 challenge，隔离构建候选证明，导出后在隔离区之外检查格式、证明及命题匹配，并启用外部检查器。它仍依赖逻辑、检查器、隔离机制、连接流程及人类陈述的正确性，不是“绝对无条件可靠”。[Lean：Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

Comparator 官方仓库特别要求可信 challenge 的导入和构建配置不能被候选证明控制；其配置包含定理名称、公理许可列表。该独立仓库已弃用，维护者说明 Comparator 自 Lean 4.35.0 起并入 `lake comparator` 与 `lake check`，建议迁移到内置工具。[官方 Comparator 仓库 README](https://github.com/leanprover/comparator)

建议：AI 只能提交候选证明体，不得自行修改可信目标、定义、导入或构建配置。未经审阅的候选工程应先隔离再构建，不能先在主工作区运行、再补做安全检查。普通容器也不应未经评估就宣称满足官方高保障隔离条件。

### F6. native 计算会扩大可信边界，并且存在版本差异

官方当前公理文档示例中，`native_decide` 引入了专用计算公理，名称包含 `_native`。这与普通证明仅依赖标准公理的情形不同。[Lean：Axioms](https://lean-lang.org/doc/reference/latest/Axioms/)

验证页注明：旧版本使用 `Lean.trustCompiler`；自 4.29.0 起按计算引入专用公理，旧机制在 4.35.0 移除。不能把旧公理名称当作所有版本中唯一的审计目标。[Lean：Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

建议：首轮严格试点不接受 native 计算公理，降低解释与重检难度；这是一项试点策略，不是声称所有 native 证明都错误。若后续为了效率接受，须记录额外可信组件、理由及对应版本。

## 3. 官方事实：Hypothesis 能补充什么，不能替代什么

Hypothesis 是 Python 的性质测试库。用户描述输入域和性质，它生成输入进行检查；教程使用 `@given` 与 strategies，并展示了加入浮点数后可能发现 `NaN` 反例的情形。[Hypothesis 官方首页](https://hypothesis.readthedocs.io/en/latest/)、[Introduction to Hypothesis](https://hypothesis.readthedocs.io/en/latest/tutorial/introduction.html)

官方区分输入“域”与生成“分布”，提醒过窄的域可能排除触发缺陷的输入。[Domain and distribution](https://hypothesis.readthedocs.io/en/latest/explanation/domain.html)

失败输入可以自动保存在数据库中，也能用 `@example` 显式回归。官方提醒数据库可能因升级或代码变动失效，`@reproduce_failure` 的编码也跨版本不稳定；长期保留关键反例宜用明确输入。[Replaying failed tests](https://hypothesis.readthedocs.io/en/latest/tutorial/replaying-failures.html)

`max_examples` 是生成预算设置，失败缩减、输入过滤或空间耗尽都会影响实际执行次数。设置一个数字，不能据此声称恰好执行了相同数量的独立测试。[Configuring test settings](https://hypothesis.readthedocs.io/en/latest/tutorial/settings.html)、[How many times will Hypothesis run my test?](https://hypothesis.readthedocs.io/en/latest/explanation/test-case-count.html)

推论：这适合尽早找反例、检查实现、积累回归样例；有限测试通过不能建立一般输入域上的形式化证明，也不能从测试数量推导未经建模的可靠性概率。即使某次穷举覆盖了一个有限域，结论也只能覆盖那一域及相应测试判定器。

## 4. 建议：个人研究/工程的七步闭环

下面是基于上述边界提出的工作方法，不是官方指定流程，也未在本次研究中执行。

1. **先做“主张卡”，再让模型证明。** 写明自然语言主张、量词、输入类型、前提、失败条件、现实含义和明确不覆盖的情况。研究中可以选择一个关键引理；工程中可以选择纯函数的不变量、协议状态约束或边界计算，不必一开始形式化整个系统。

2. **由人确定可信规格。** 人工固定定义、定理签名和依赖版本；记录自然语言到形式化符号的对应。检查有无不可能满足的前提，要求给出至少一个满足前提的实例；实例只能帮助排查空泛规格，不代替证明。

3. **把生成任务限制为候选。** 可以使用如下任务约束：“目标和定义已经固定，只给候选证明体、所用引理来源及逐步解释；不得修改前提、定义、导入、构建配置，不得增加公理或使用 `sorry`。若失败，提交未解决目标，不得弱化结论。”这是协作约束，不是安全机制；仍须检查实际产物。

4. **先用便宜的反例检查校正问题。** 若有可执行实现，设计单元测试与性质测试，检查零值、空输入、重复值、极端规模、非法输入和类型特有行为。保留失败原始输入与可复现的小反例；禁止通过缩窄生成域掩盖本来应覆盖的缺陷。对非可执行的数学主张，可先手查小例或必要条件，不能因此标记“已证明”。

5. **按风险选择证明验收强度。** 经审阅、依赖可信的普通工程可做构建、公理审计和 fresh 重检；未审阅 AI 工程或有对抗风险的提交须增加可信 challenge、隔离构建、命题比较与独立检查器。禁止绕过检查来提高“通过率”。工具超时应标为“未完成”，不能据此认定命题为假。

6. **设立人工理解关口。** 研究者用自己的话解释每条前提为什么需要、关键引理怎样串联、形式化对象与原问题如何对应；指出一个不适用场景，并在适当情况下给出删去关键前提后失败的例子。可以借助 AI 讲解，但不能只转交它生成的解释作为理解证据。

7. **将结果转成可维护资产。** 留下规格、源码、版本、证明检查日志、公理清单、反例和人写解释。区分“Lean 模型已证明”“外部实现已测试”“模型—实现对应已证明”三种状态；外部实现、依赖或规格变化后重新评价受影响部分。

### 待实施的最小检查示意

以下仅说明步骤，不是本次执行记录。`Pilot` 是拟建项目的模块占位名，`targetLemma` 是拟验收定理的名称；必须在先审阅或妥善隔离的环境中实施。

```text
lake build +Pilot
```

按官方验证页的口径，检查构建结果及全部警告，而非只看退出码。[Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)

在可信审计模块中加入：

```lean
#print axioms targetLemma
```

检查传递公理集合；此命令本身不是对抗恶意显示或环境篡改的完整防护。[Axioms](https://lean-lang.org/doc/reference/latest/Axioms/)

对于提供内置 checker 的工具链，维护者给出的 fresh 重检用法可对应为：

```text
lake env leanchecker --fresh Pilot
```

实际工具名称须与锁定版本一致；不能把重新使用同一内核写成“独立验证器复核”。[官方 checker README](https://github.com/leanprover/lean4checker)

对高保障流程，选择对应版本的内置 `lake comparator`，另行配置可信 challenge、公理许可与外部检查器。本文不提供未经实际核验的完整配置或“一条命令即可完成”的承诺。[官方 Comparator README](https://github.com/leanprover/comparator)

## 5. 常见误读与纠正

下表是基于前述官方事实提出的风险检查清单。

| 可能误读 | 应使用的表述或检查 |
| --- | --- |
| “Lean 通过，所以原始科学结论正确。” | 只能首先声称所检查的精确形式化命题成立；另外审核形式化对应、现实前提与证据。 |
| “蓝色双勾已经排除所有 `sorry`。” | 审计传递公理；依赖中的 `sorryAx` 仍可能存在。 |
| “把难证明的结论声明为公理也是完成证明。” | 这是增加假设；必须披露并另行证明或论证，不能计入严格证明完成数。 |
| “`leanchecker --fresh` 是另一个独立内核。” | 它重用 Lean 内核；独立实现的外部 checker 是另一个保障层。 |
| “内核很小，所以任何 Lean 项目都能放心构建。” | 构建和元代码执行仍有系统安全边界，陌生提交先隔离。 |
| “测试一万次没有失败，等于证明。” | 只能报告在记录的域、预算及判定器下没有发现反例；不升级为形式化结论。 |
| “通过收紧前提、过滤输入，原问题就解决了。” | 每次规格变更都需披露并重新审批；拒绝把更弱问题冒充原任务。 |
| “证明 `Nat` 或数学实数上的性质，就覆盖机器整数和浮点数。” | 显式核对溢出、精度、`NaN` 等实现语义；不同对象不得直接替换。 |
| “Lean 模型证明完成，生产实现也被证明。” | 只有另行建立模型—实现对应，才能扩大证明覆盖范围；否则仅报告实现测试。 |
| “模型解释很流畅，证明作者已经理解。” | 用独立复述、前提审查、边界例子和变更影响分析验收理解。 |
| “证明搜索失败或超时，说明主张为假。” | 这只说明候选或预算不足；要判假仍需反例或反证。 |

## 6. 建议：一套小规模试点与验收指标

### 试点选题和边界

建议个人用一周、最多 8 小时有效投入，选择“有限非负整数列表的纯排序函数”。固定 Lean 中的函数定义与三个目标：结果有序、保留元素及其出现次数、幂等。重点是建立工作流，不是声称发现新的排序定理。

若已有 Python 实现，可另做长度保持、成员关系保持及与参考实现的差分测试。明确：Lean `Nat` 对应的测试入口限于非负 Python 整数；测试实际规模限制须披露。Python 的结果与 Lean 定义相符若尚未证明，只能作为测试观察，不能称为实现正确性证明。

数学研究可将此选题替换为三个与当前课题有关、定义已清楚的小引理；没有可执行实现时，省略工程测试指标，不编造运行证据。

### 预先登记的指标

这些数值是建议性验收门槛，不是已观测结果、行业标准或官方保证。

| 维度 | 验收门槛 | 需要保留的证据 |
| --- | --- | --- |
| 规格完整性 | 3/3 目标写明类型、量词、前提、定义与不覆盖内容；候选证明不得改规格 | 人工批准的规格、差异记录 |
| 非空泛性 | 所有带前提的目标均有人可解释的可满足实例；发现空泛前提不得直接算成功 | 前提审查、示例及解释 |
| 形式化完成 | 3/3 目标按选定检查层级验收；不足则如实标“部分完成” | 定理名单、构建与检查日志；高风险时含命题比较、外部 checker 结果 |
| 公理卫生 | 每个目标的传递公理均在预先批准集合内；严格首轮 `sorryAx`、自定义及 native 公理均为 0 | 完整公理输出及审批清单 |
| 可复现性 | 锁定工具链、依赖与源代码版本；在另一个干净且同等隔离的环境重跑一次，结果一致 | 版本/提交标识、两次日志、实际运行日期 |
| 测试预算与边界 | 如有 Python 实现：每个性质拟设置 `max_examples=1000`，另显式覆盖空列表、单元素、重复值、零值、大整数、已排序、逆序；披露域限制和实际执行信息 | strategies、settings、显式样例、测试报告；不得把预算冒充实际执行数量 |
| 故障检出 | 预先放入 3 个已知错误变体，例如丢掉一个重复元素、遗漏末元素、逆序返回；相关验证手段应检出 3/3 | 每个变体的失败性质或未能通过的证明检查；测试未检出则改进测试 |
| 反例管理 | 所有发现的真实缺陷均有明确输入和回归断言，修复后重跑；0 个未解释失败 | 反例、原因、回归结果 |
| 人工理解 | 本人不照读模型回答，能解释 3 个目标的证明结构、全部前提、至少 1 个边界；若有他人则请其审阅 | 本人写的短说明或审阅记录；独自完成时承认审阅独立性较弱 |
| 收益与成本 | 总投入不超过 8 小时；单列学习、规格、生成、调试、验证与理解时间 | 时间记录及可复用资产；首次试点不宣称已经证明效率提升 |

“错误变体未能通过证明检查”不等于已经形式化证明该变体必然错误；它只是这次验收没有得到有效证明。只有可复现反例、反证或其他合适证据才能支持进一步结论。

推广条件建议：形式化、语义和理解的硬门槛全部满足，且留下至少一个可复用的规格或验证组件，再扩大范围；不要单凭模型生成速度或表面通过率扩大。超出预算时保留尚未解决目标，而不是放宽公理政策。

## 7. 官方入门资料与建议使用顺序

Lean 官方入口把 Natural Number Game 推荐给初学者；将 Functional Programming in Lean 面向程序员、Theorem Proving in Lean 面向证明开发与验证、Mathematics in Lean 面向使用 Mathlib 的数学形式化。[Lean 官方 Learn 页面](https://lean-lang.org/learn/)

建议顺序：初次接触者先理解命题、证明状态与基础归纳；随后按背景选择上述教材，用现有定义和引理完成很小的目标。工程师先选纯函数，数学研究者先选局部引理。不要把首轮的依赖安装、形式化训练和安全隔离时间全部当作模型效果。

## 8. 来源摘要与资料限制

所有引用均来自 Lean 官方网站、`leanprover` 官方维护者仓库或 Hypothesis 官方文档；没有使用二手报道、模型自述或原文三项成果材料。

1. [Lean — Validating a Lean Proof](https://lean-lang.org/doc/reference/latest/ValidatingProofs/)：实际读取验证阶梯、AI 输入威胁模型、语义边界、剩余假设和 native 版本说明。
2. [Lean — Learn](https://lean-lang.org/learn/)：实际读取官方入门路线及教材定位。
3. [Lean — Elaboration and Compilation](https://lean-lang.org/doc/reference/latest/Elaboration-and-Compilation/)：内核、编译与 IO/初始化边界。
4. [Lean — Axioms](https://lean-lang.org/doc/reference/latest/Axioms/)：标准公理、传递公理审计、`sorryAx` 和 native 示例。
5. [leanprover/lean4checker](https://github.com/leanprover/lean4checker)：当前 README 的内置 `leanchecker` 名称、版本门槛、fresh 用法和同内核限制。
6. [leanprover/comparator](https://github.com/leanprover/comparator)：可信 challenge 与公理许可的设计；弃用、迁移到内置 Lake 工具的说明。
7. [Hypothesis 官方文档首页](https://hypothesis.readthedocs.io/en/latest/)与[入门教程](https://hypothesis.readthedocs.io/en/latest/tutorial/introduction.html)：性质测试定位、输入生成与反例示例。
8. [Hypothesis — Domain and distribution](https://hypothesis.readthedocs.io/en/latest/explanation/domain.html)：输入域与覆盖风险。
9. [Hypothesis — Replaying failed tests](https://hypothesis.readthedocs.io/en/latest/tutorial/replaying-failures.html)：失败回放与长期明确样例。
10. [Hypothesis — Configuring test settings](https://hypothesis.readthedocs.io/en/latest/tutorial/settings.html)与[测试次数说明](https://hypothesis.readthedocs.io/en/latest/explanation/test-case-count.html)：生成预算及实际执行次数差异。

版本限制：`latest` 是移动页面，本文是核查日所读内容的笔记，不保证未来文档保持原样。验证页仍保留 `lean4checker` 旧称，本文另外用官方仓库确认了命名迁移。实践时应锁定具体发布版本并阅读该版本的说明；本文未核验任何本地工具链的可用性、命令输出或沙箱安全性。没有使用 Z3，也没有作跨求解器的未经核查推断。
