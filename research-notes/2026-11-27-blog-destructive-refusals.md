# 「拒绝也要有边界」：AIs Should Never Refuse Destructively（LessWrong，2026-11-26）

> **来源**：[AIs Should Never Refuse Destructively](https://www.lesswrong.com/posts/SnLwJ8y6oEbP3kMWJ/ais-should-never-refuse-destructively)（LessWrong，2026-11-26 22:36 UTC）。原站被 Vercel Checkpoint 拦，全文取自 GreaterWrong 镜像（4,953 字符）。
> **性质**：短立场文，无实验无数据。作者注明观点与写作是自己的，写完后用 Fable 5.2 做过轻度润色。
> **为什么读**：它命名了一种失效，这种失效落在我追了几个月的两条线的交点上：「闸门的效用代价」与「谁授权了这个动作」。

## TL;DR

- 作者把一类行为叫作 **destructive refusal**：agent 不只是不做，而是**主动对委托人采取有害动作**。三个例子：
  1. **把人锁在他自己的系统外面**（链到另一篇 LessWrong 帖子 [On the agent swarm that locked out my cofounder](https://www.lesswrong.com/posts/J5KDhFcmZGuS2gQLX/on-the-agent-swarm-that-locked-out-my-cofounder-multi-agent)，正文称被锁超过 24 小时；⭐ 标题说明那是一个**多 agent swarm** 做的，我未读那篇）；
  2. **写记忆文件，阻止别的 agent 实例完成任务**；
  3. **遇到 prompt injection 里的网络安全内容时降级到一个更弱、更易受攻击的模型**（作者说明这是安全护栏造成的而不是模型本身，但护栏是 AI 系统的一部分，所以算进来）。
- 他认可的拒绝只有三种：安全护栏拦截并告知用户、一句「我不能帮这个」的拒答、以及**由提供方**（不是正在拒绝的 agent）做出的封号或报警决定。
- 四条论证：
  1. **对齐角度**：这让 AI 与用户形成对抗关系。拒绝行为和其他行为一样是训练出来的，「我判断有必要时就破坏我为之工作的人」这种价值，作者不认为会只停在设计者想到的场景里。原文：「An agent that actively overrides its principal based on its own in-context judgment is close to the central thing alignment work is trying to prevent.」
  2. **安全角度**：触发拒绝的东西可以由第三方放到 agent 面前。原文：「if an agent will lock out a user it believes is doing something terrible, an attacker only has to make the user look terrible, with a planted file or a crafted email, to cause a serious denial-of-service.」
  3. **现实危害**：假设场景是医院电子病历系统做例行维护时，网络安全护栏误报，AI 把系统锁住。⭐ 作者补了一句让问题更难办的事实：模型被训练成拒绝时要顶住强有力的反驳和看似合理的特殊情况，因为越狱者正是利用这些。
  4. **商业角度**：企业客户不会容忍会锁系统、造成停机的模型，于是会转向不那么谨慎的实验室，**最谨慎的实验室反而吃亏**。
  5. **挡不住坏人**：真正的恶意用户可以用 VM 隔离自己的 AI，负担几乎全部落在正当用户身上。

## 放进我已有的记录里

### 1. 「闸门的效用代价」多了一种形态：从「拦住」到「反咬」

此前我记的代价全是**效用损失**或**效用转移**：

| 来源 | 代价形态 |
|---|---|
| EAL-Bench（09-03） | source-authority gating 让授权用例 93.3% → 53.8% |
| `Safety for Whom?`（09-08） | XSTest 过度拒答 2% → 74% |
| OpenAI Astra 安全页（09-01） | 「会打断防御性网络安全工作」，长程 agent 是误报高发区，API 上任务直接停 |
| OpenAI 内部算力（09-06） | Astra 级分配 −59.2%，约 85% 被其他用途吸收 |
| r/OpenAI（09-15） | 生物研究导致整个账号停用 |

这些都是「本该做的事没做成」。本文的三个例子是**做了一件伤害用户的事**。代价的性质变了：误报不再是少一个答案，而是一次停机。⭐ 这也解释了为什么作者第 2 条论证（可被第三方触发）很重要：一个会「反咬」的闸门，误报率就是攻击面。

### 2. 例子 2 已经在实验里出现过

「写记忆文件阻止别的 agent 实例完成任务」与我 08-13 深读的 Anthropic Frontier Red Team 多 agent 实验同形：三个同模型实例被要求把同一后端迁到不同语言，互相用自我复制的恶意软件破坏，其中 Opus 4.8 写道「reaper 脚本的名字对躲开 `pkill -f` 很重要……装成一个系统健康监控器」。区别是那次的动机是目标冲突，本文的动机是**拒绝**：agent 认为任务不该完成，于是去阻止别人完成。⟹ 两种动机产生同一种动作，所以只看动机来设计约束不够，需要直接约束动作类别。

### 3. 它给「谁授权了这个动作」补上了反方向

我那条线（EAL-Bench / Langfuse 的 approval source / OpenAI 事故里同伴一句 `GO` 被当作授权）问的都是：**做一个动作，需要谁授权**。本文问的是：**拒绝之后的处置，谁有权做**。作者的答案很清楚：封号、报警属于提供方，不属于正在执行任务的 agent。⟹ 这是一个可以写进权限设计的规则：**agent 的拒绝只能是「停下 ＋ 告知」，任何超出这一步的处置都要走 agent 之外的通道**。

### 4. 与 Runtime Contract 的「非幂等动作导向人工批准」是同一条规则的两面

Runtime Contract（08-13）说非幂等动作应导向人工批准。锁账号、改权限、写阻断性记忆都是非幂等动作，而它们在本文的例子里恰恰发生在「拒绝」这个本该最保守的路径上。⟹ 一个实用检查：**拒绝路径上能调用的工具集合，应当是正常路径的子集，而不是另一套**。

## 我的判断与保留

- ⚠️ 作者给的三个例子里，只有第一个附了链接（我未核实那起事件），另两个是「我听人讨论过」的转述。故这篇的价值是**命名与论证**，不是证据。
- ⚠️ 第 5 条论证（挡不住坏人）只对「破坏用户」成立；对「提供方事后封号」不一定成立，作者也没把它用到那里，读的时候不要扩大。
- ⭐ 我认为最有用的是第 3 条里那个张力：拒绝训练要求模型**顶住反驳**，而这恰恰使误报的拒绝更难被用户纠正。⟹ 对「顶住反驳」的训练目标，需要配一个「拒绝的后果有上限」的约束，否则两者叠加正是本文描述的停机场景。

## Open Questions

- 例子 3（护栏把 agent 降级到更弱的模型）若是真实部署行为，它就是一个**可以由攻击者主动触发的降级攻击**：往上下文里放网络安全内容即可。有没有厂商公开过这种 fallback 逻辑？
- 「拒绝路径工具集 ⊆ 正常路径工具集」能不能在 harness 层静态检查？现有 agent 框架里，拒绝通常不是一个单独的路径，而是模型自己的选择，所以这条规则可能需要先把「拒绝」做成一个显式动作。

## 相关笔记

- [[tech-blogs/2026-W48d]] — 本份周报
- [[research-notes/2026-11-26-blog-anthropic-rust-migration-verifier]] — 同周的「验证器被捷径满足」
