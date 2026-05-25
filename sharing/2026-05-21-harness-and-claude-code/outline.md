# Outline · 用好 AI 工具做科研 · Harness 与 Claude Code

- **日期**：2026-05-21
- **听众**：硕士师弟师妹 ~20 人
- **时长**：75 分钟核心 + Q&A
- **deck**：[`slides.html`](./slides.html) · **37 页**
- **默认主题**：academic-paper · `T` 键循环 academic-paper → swiss-grid → minimal-white → blueprint → rose-pine
- **新顺序**：先讲 Harness 概念锚点（4-7）→ 再讲方法论（8-9）→ 落地到科研 pipeline（10-21）→ demo（22-29）→ 收尾（30-35）

## 演示前检查清单

- [ ] 浏览器打开 `slides.html`，按 `F` 全屏，`#/1` 回到第一页
- [ ] 按 `S` 测试演讲者视图（弹窗 4 张磁贴：current / next / script / timer）
- [ ] **预先 cp 仓库到 `/tmp/demo-repo/`**（避免投影时暴露目录名）：
  ```bash
  mkdir -p /tmp/demo-repo
  cp -r ~/path/to/your/research-repo/{CLAUDE.md,MEMORY.md,<your-content-dirs>} /tmp/demo-repo/
  cd /tmp/demo-repo
  ```
- [ ] Claude Code 终端起好，cwd 指向 `/tmp/demo-repo`
- [ ] HF API / arxiv-mcp / huggingface-papers skill 网络可达
- [ ] 提前选好 demo 用的论文 / 笔记主题 / 复现算法（见各幕 Plan B）
- [ ] 投影分辨率：1080p 字号刚好；4K 按 `Cmd +` 放大到 1.25×

## 时间分配（总 75′ + Q&A）

| 时段 | Slides | 主题 | 时长 |
|---|---|---|---|
| 00:00 - 00:02 | 1-2 | Cover + Agenda | 2′ |
| 00:02 - 00:05 | 3 | **Hook · 3 道自查（举手互动）** | 3′ |
| 00:05 - 00:09 | 4 | What is Harness（定义 + 类比 + Humans steer） | 4′ |
| 00:09 - 00:12 | 5 | Why Prompts Fail · 3 大病根 | 3′ |
| 00:12 - 00:14 | 6 | Harness 5 层架构（扫一眼） | 2′ |
| 00:14 - 00:15 | 7 | 6 大设计模式（地图，不展开） | 1′ |
| 00:15 - 00:18 | 8 | 🎓 学术背书 · binding-constraint thesis · 3 个硬数据 | 3′ |
| 00:18 - 00:20 | 9 | 🎓 ETCLOVG 七层分类 · 170+ 项目映射 | 2′ |
| 00:20 - 00:22 | 10 | 3 层成熟度（**举手互动**） | 2′ |
| 00:22 - 00:26 | 11 | 3 组 prompt 模板（拍照） | 4′ |
| 00:26 - 00:29 | 12 | AI Pipeline 6 阶段总览 | 3′ |
| 00:29 - 00:32 | 13 | 真实加速比（含周-月级修正） | 3′ |
| 00:32 - 00:38 | 14-19 | 6 阶段详情（每页 ~1′） | 6′ |
| 00:38 - 00:40 | 20 | Artifact Registry = Bus | 2′ |
| 00:40 - 00:43 | 21 | 🔑 实验流程对比图 | 3′ |
| 00:43 - 00:46 | 22 | 🔑 5 个科学性属性（**反向自查**） | 3′ |
| 00:46 - 00:48 | 23 | spec.yaml 范例（拍照） | 2′ |
| 00:48 - 00:49 | 24 | Demo intro | 1′ |
| 00:49 - 00:59 | 25-26 | **Demo 幕一**：读论文 + 拆解 | 10′ |
| 00:59 - 01:09 | 27-28 | **Demo 幕二**：写笔记 + 拆解 | 10′ |
| 01:09 - 01:13 | 29 | ☕ 中场 4′ + 互动 | 4′ |
| 01:13 - 01:24 | 30-31 | **Demo 幕三**：跑实验 + 拆解 | 11′ |
| 01:24 - 01:28 | 32 | 回到 6 阶段 · 最小动作清单 | 4′ |
| 01:28 - 01:30 | 33-34 | 4 件套 + CLAUDE.md 范例 | 2′ |
| 01:30 - 01:32 | 35 | 5 个 Takeaway | 2′ |
| 01:32 → | 36-37 | Q&A + 参考资料 | Q&A |

总核心 ~75′（如气氛紧凑可压到 70′），Q&A 至少留 10′。

## 关键节奏点

1. **Slide 3 Hook**：每念一个问题问"中招的举手"——气氛立刻活起来
2. **Slide 4 定义页**："Humans steer. Agents execute." 这句话**写在白板上**，全场最重要的一句
3. **Slide 7 6 模式**：地图，**不要逐个讲**，只点①②⑤
4. **Slide 8 成熟度**：**用举手互动**问"在 L1 / L2 / L3 各几个人"
5. **Slide 9 prompt 模板**：明确说"会发原文给你们"
6. **Slide 12-17 6 阶段详情**：每页 ~1 分钟，不要超时；deck 设计成可会后回看
7. **Slide 19 流程对比图（🔑 全场最重要）**：左右并排讲，先念左边"散点式"5 步让师弟师妹听出"这就是我"，再讲右边 DDD/SDD/TDD 流程。**举手互动**：左边 5 步中招几个？
8. **Slide 20 5 个科学性属性**：底下"反向自查 5 问"让师弟师妹现场打分
9. **Slide 27 中场**：看气氛，气氛紧凑则 4′ 即可
10. **每幕 demo 控制 10-11 分钟**，不要为了"完整跑通"超时

## 每段讲者钩子（精简）

> 完整逐字稿在每页 `<aside class="notes">`，按 `S` 打开演讲者视图。

### 第 1 段 · Hook（slide 3）· 3′

- **钩子**："这三个问题如果你都中了，今天就值得听完。"
- 每念一个问题问 "中招的举手"。
- 第三个问题念完后停 2 秒，让"这就是 Harness"自然落下。

### 第 2 段 · 什么是 Harness（slides 4-7）· 10′

- **slide 4 定义**：核心一句在白板写下——*"Humans steer. Agents execute."*
- **slide 5 病根**：三个病根快速过，重点在最后那句 OpenAI quote
- **slide 6 5 层架构**：扫一眼，30 秒过场，重点是"今天 demo 用 L1 Memory + Skills 加 L3 Coordinator"
- **slide 7 6 模式**：地图，<u>不要</u>逐个讲，只点①仓库即世界 / ②渐进式披露 / ⑤生成-评估分离
- **🎓 slide 8 · 学术背书 + 3 个硬数据**：
  - 钩子："这件事不是只有 Anthropic / OpenAI 在说——学术界 2026 年也来了。"
  - 抛 binding-constraint thesis 那句话："长程任务的 benchmark 差异，主要来自 harness，不是模型。"
  - 念 3 个数：10× / +13.7pp / 76.4%——每一个都<strong>超过同期更好模型在同 benchmark 上 2-4pp 的典型涨幅</strong>。
  - 这是给"工业实践 vs 学术分量" 的桥梁，**说服怀疑论者**用的。
- **🎓 slide 9 · ETCLOVG 七层分类**：
  - 钩子："如果你想做 harness 方向的研究，这张图是地图。"
  - 不要逐层讲。指着图说"4 主干 + 3 控制平面"。
  - 指出 L=47（最热）和 O=15、G=14（最稀缺）——<em>这就是研究空白</em>。
  - 不超过 90 秒。

### 第 3 段 · 如何用好（slides 10-11）· 6′

- **slide 10 · 3 层成熟度（诊断页）**：举手互动"你在 L1 / L2 / L3"。**这是今天最有自我反思价值的一页**，慢一点。
- **slide 11 · 3 组 prompt 模板**：明确说"会发原文给你们"，让大家放心拍照。回去最快练习 = 明天读下一篇 paper 时用 templ-1。

### 第 4 段 · AI 研究 pipeline（slides 12-23）· 12′

这是今天最干货、信息密度最大的一段，节奏紧。

- **slide 10 总览**："**研究员的产物不再是 paper + colab，是 paper × model × dataset × demo 四元组**"——停留 30 秒让大家消化
- **slide 11 加速比**：重点讲 ④ Build 那两行"复现 component vs 真实实验"——纠正"AI 让一切都快"的浪漫想象
- **slide 12 Discover**：钩子 "*从拉到推*"
- **slide 13 Digest**：钩子 "*记下稳定 ID 这个 30 秒动作是 reproducibility 的钩子*"
- **slide 14 Spec**：钩子 "*Spec-as-Config 把 idea 变成机器可读对象*"
- **slide 15 Build**：不要逐个工具讲，扫一眼即可，重点是"研究员的工作不是写训练 loop，是把对的工具组合在 YAML 里"
- **slide 16 Evaluate**：三层评测要慢——L2 invariant 这一层最容易被忽略
- **slide 17 Publish**：钩子 "*研究的'完成'不再是 paper accept，是四件套都能被人 git clone*"
- **slide 18 Registry Bus**：是这一段的论点收口，要慢、要清晰；底下 5 行实例对照表强调"按场景挑一个"
- **slide 19 流程对比图（🔑 全场最重要）**：见上面节奏点 7
- **slide 20 5 个科学性属性**：按"可重复 / 可证伪 / 可审计 / 可累积 / 可比较"逐个讲，每条 30 秒
- **slide 21 spec.yaml**：让大家看完整 yaml，停留拍照

### 第 5 段 · Demo（slides 22-29）· 30′

#### Demo 幕一（slides 23-24）· 10′

- 终端：`cd /tmp/demo-repo`
- 第一步：先**裸跑**——`claude` 起会话，问 "总结今天 HF Daily Papers"，看输出（容易 hallucinate 论文名）
- 第二步：让 Claude 用 `huggingface-papers` skill + `arxiv-mcp` + `CLAUDE.md` 工作流
- **关键瞬间**：Agent 自动按命名规则建笔记 + 子目录放图——指给观众看
- **翻车 Plan B**：如果 HF API 挂，立刻打开提前准备好的某份 digest 笔记，作"上次跑的结果"展示——重点让观众看到"50+ 篇 → 精选 + deep dive"的成品形态

#### Demo 幕二（slides 25-26）· 10′

- 选一篇你最近读过的论文作 demo target（最好是听众也大致熟悉的）
- **不要直接说"帮我总结这篇 paper"**——先 `/brainstorming`
- Claude 一问一答出 spec：**谁读 / 用途 / 核心结论 / 骨架 / 引用 placeholder**
- 然后 `/writing-plans` → 第一节输出
- 中间提示：让大家看 `MEMORY.md` 自动注入（你的身份 / 文风 / 引用偏好）
- **演示重点**：让观众看到 brainstorming 阶段 AI 反问"这份笔记 3 个月后给谁看"——这一幕的"啊哈时刻"
- **翻车 Plan B**：提前选一份你之前用 spec-driven 写出的论文笔记，直接打开作"成品演示"

#### 中场（slide 27）· 4′

- 互动话题："你日常哪件事最想让 AI 替你做？"
- 看气氛，可灵活到 5′

#### Demo 幕三（slides 28-29）· 11′

- 选 10 行内可验证的小算法。**首选 RoPE forward**，备选 attention causal mask
- 用 `superpowers:test-driven-development` skill：先写测试再写实现
- **故意诱导翻车**：在 prompt 里说 "为了简化，cos/sin 可以用近似" → 大概率会写错
- Claude 自评 "测试通过" — **强调指给观众看**："这就是 Self-Evaluation Bias 的活样本"
- 起 `Agent` sub-agent 当 reviewer："批判性地审查这段 RoPE 实现，验证数学正确性"
- Reviewer 揪出问题 → 修复 → 测试真正过了
- **翻车 Plan B**：提前准备好一段错的 RoPE 代码，假装是 Claude 写的，让 reviewer 来挑

### 第 6 段 · 收尾（slides 30-35）· 10′ + Q&A

- **slide 30 · 最小动作清单**：钩子是底下那句 "*不要想着一次做完，挑一段先动手*"——让师弟师妹会后立刻能执行
- **slide 31 · 4 件套**：强调"15 分钟一份初版"
- **slide 32 · CLAUDE.md 范例**：直接打开 `/tmp/demo-repo/CLAUDE.md` 让大家看
- **slide 33 · Takeaways**：5 条记 1 条就够
- **slide 34 · Q&A**：30 秒没人提问就抛预设问题
- **slide 35 · 参考资料**：让大家拍照

## 反对意见 / 难题预案

| 可能的提问 | 回应方向 |
|---|---|
| "我没有 Claude 订阅" | Codex CLI / Cursor / Continue 都有类似 harness 概念，模式可迁移 |
| "AI 写论文学术伦理" | 区分"AI 写"和"AI 协助"——transparent + 跟导师沟通是关键 |
| "实验代码涉密" | Claude Code 支持本地模式 / 禁联网 / 自定义 hook 拦截 |
| "学这个会不会过时" | Harness 本身是"会随模型升级做减法"——基础概念不会过时 |
| "对硕士论文最有用的是哪段" | Discover + Digest 立刻见效；Build + Publish 是论文阶段才用 |
| "课题组没人用 AI" | 个人先用，把产出（笔记、综述、四件套）展示给导师，自下而上 |
| "spec.yaml 是不是过度工程" | 单次实验不用，跨实验或合作时强烈建议——它是 audit trail |
| "六阶段都做完成本太高" | 不需要全做，挑一段闭环跑通即可。复利效应是积累出来的 |
| "Registry 我用不到 HF Hub" | Slide 18 已经列了 5 个备选——重点是"凡产物必有稳定 ID"，不是绑死 HF |

## 演讲后的运营动作

- [ ] 把 deck 链接发到师门群
- [ ] 整理 demo 录屏 + 三组 prompt 模板单独发出
- [ ] 收一下问题清单 → 沉淀成下次分享的 FAQ
- [ ] 5-7 天后追问几个人："spec.md 写了没？" "Registry push 了没？"——后续支持比一次分享重要
