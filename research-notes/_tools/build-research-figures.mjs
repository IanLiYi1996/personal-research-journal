import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath, pathToFileURL} from 'node:url';
import {execFileSync} from 'node:child_process';

// Generated illustration assets, not factual measurements. The statements and
// citations live next to each illustration in the editable Markdown source.
const directory = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../2026-10-13-decision-models-and-agent-governance-deep-research/figures');
fs.mkdirSync(directory, {recursive: true});
const palette = {
  blue: ['#eff6ff', '#93c5fd', '#2563eb'],
  purple: ['#faf5ff', '#c4b5fd', '#7c3aed'],
  green: ['#f0fdf4', '#86efac', '#059669'],
  orange: ['#fff7ed', '#fdba74', '#ea580c'],
  gray: ['#f9fafb', '#d1d5db', '#6b7280'],
  red: ['#fef2f2', '#fca5a5', '#dc2626'],
};
const escape = value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');
class Diagram {
  constructor(name, height, title, subtitle) {
    this.name = name;
    this.height = height;
    this.parts = [
      `<svg xmlns="http://www.w3.org/2000/svg" width="1120" height="${height}" viewBox="0 0 1120 ${height}" role="img" aria-labelledby="title desc">`,
      `<title id="title">${escape(title)}</title><desc id="desc">${escape(subtitle)}</desc>`,
      '<style>text{font-family:"Noto Sans CJK SC","PingFang SC","Microsoft YaHei",Arial,sans-serif;fill:#111827}.muted{fill:#6b7280}</style>',
      '<defs>',
      ...Object.entries(palette).map(([key, [, , color]]) =>
        `<marker id="arrow-${key}" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0,10 3.5,0 7" fill="${color}"/></marker>`),
      '</defs><rect width="1120" height="' + height + '" fill="#ffffff"/>',
    ];
    this.text(48, 48, title, 28, 600, 1024);
    this.text(48, 78, subtitle, 17, 400, 1024, 'muted');
  }
  text(x, y, value, size = 18, weight = 400, width = 1000, className = '') {
    this.parts.push(`<text x="${x}" y="${y}" font-size="${size}" font-weight="${weight}" data-available-width="${width}" class="${className}">${escape(value)}</text>`);
  }
  rect(x, y, width, height, color = 'gray', dashed = false) {
    const [fill, stroke] = palette[color];
    this.parts.push(`<rect x="${x}" y="${y}" width="${width}" height="${height}" rx="8" fill="${fill}" stroke="${stroke}" stroke-width="1.5"${dashed ? ' stroke-dasharray="6 4"' : ''}/>`);
  }
  node(x, y, width, height, title, lines = [], color = 'blue', dashed = false, titleSize = 22) {
    this.rect(x, y, width, height, color, dashed);
    this.text(x + 18, y + 33, title, titleSize, 600, width - 36);
    lines.forEach((line, index) => this.text(x + 18, y + 61 + index * 25, line, 17, 400, width - 36, 'muted'));
  }
  arrow(points, label = '', color = 'blue', dashed = false, labelPosition) {
    const stroke = palette[color][2];
    const data = points.map(([x, y], index) => `${index ? 'L' : 'M'} ${x},${y}`).join(' ');
    this.parts.push(`<path d="${data}" stroke="${stroke}" stroke-width="2" fill="none"${dashed ? ' stroke-dasharray="5 4"' : ''} marker-end="url(#arrow-${color})"/>`);
    if (label) {
      const [x, y] = labelPosition ?? [(points[0][0] + points.at(-1)[0]) / 2, (points[0][1] + points.at(-1)[1]) / 2];
      const width = [...label].reduce((sum, char) => sum + (/[\u2e80-\uffff]/.test(char) ? 14 : 8), 16);
      this.parts.push(`<rect x="${x - width / 2}" y="${y - 13}" width="${width}" height="23" rx="3" fill="#ffffff" opacity=".96"/>`);
      this.parts.push(`<text x="${x}" y="${y + 4}" font-size="14" text-anchor="middle">${escape(label)}</text>`);
    }
  }
  legend(y, entries) {
    entries.forEach(([color, label, dashed], index) => {
      const x = 48 + index * 340;
      this.arrow([[x, y], [x + 36, y]], '', color, dashed);
      this.text(x + 48, y + 5, label, 15, 400, 280, 'muted');
    });
  }
  save() {
    this.parts.push('</svg>');
    const file = path.join(directory, `${this.name}.svg`);
    fs.writeFileSync(file, this.parts.join('\n') + '\n');
    return {name: this.name, width: 1120, height: this.height, file};
  }
}
const figures = [];
{
  const d = new Diagram('01-responsibility', 648, '图 1 · 判断、执行与业务责任分开',
    '可选的应用分工；不是 Strands Decider 与 Box 的自动集成');
  d.node(48, 112, 1024, 98, '通用生成模型 / 原 Agent', ['开放规划、整理材料、生成正文与完整工具参数'], 'gray');
  d.node(48, 296, 256, 126, '证据与答案空间', ['来源 / 新鲜度 / 候选完整性', '由应用或观察系统提供'], 'gray');
  d.node(384, 296, 304, 126, 'Decider：有限判断', ['真假概率 / 候选分布 / 评分', '不替代开放内容生成'], 'purple');
  d.node(768, 296, 304, 126, '应用采纳与检查', ['证据、主体、预算、审批', '低置信度或缺证据则回退'], 'blue');
  d.arrow([[176, 210], [176, 296]], '整理材料', 'blue');
  d.arrow([[920, 210], [920, 296]], '生成参数', 'blue');
  d.arrow([[304, 359], [384, 359]], '有限问题', 'blue');
  d.arrow([[688, 359], [768, 359]], '判断建议', 'blue');
  d.node(48, 506, 256, 100, '不自动获得权限', ['模型高分 ≠ 业务批准'], 'red');
  d.node(384, 506, 304, 100, '业务系统 / 原验收', ['权限、幂等与真实结果核验'], 'green');
  d.node(768, 506, 304, 100, 'Box 或既有执行层', ['限制受控动作与副作用路径'], 'orange');
  d.arrow([[920, 422], [920, 506]], '拟议动作', 'blue');
  d.arrow([[768, 556], [688, 556]], '受控调用', 'blue');
  figures.push(d.save());
}
{
  const d = new Diagram('02-evolution', 756, '图 2 · 决策接口汇合的是问题，不是一条产品血缘',
    '左：研究路线　中：代表工作与时间　右：今天仍需回答的问题');
  const rows = [
    ['理解与评分', ['BERT / 重排 / 蕴含式分类', '2018–2019；SetFit 2022'], ['理解后能直接打分吗？', '固定分类与动态标签如何比较？'], 'blue'],
    ['运行时候选', ['Pointer Networks，2015', '注意力指向随输入变化的集合'], ['正确选项是否进入候选？', '顺序与缺候选怎样影响输出？'], 'purple'],
    ['概率与回退', ['神经网络校准，2017', '选择性分类与风险—覆盖率研究'], ['概率是否可信？', '拒绝多少，接受的错多少？'], 'green'],
    ['Agent 的任务分工', ['ReAct，2022；RouteLLM，2024', '开放行动与模型成本—质量路由'], ['拆出的判断替换了哪次调用？', '成本和全链路质量是否改善？'], 'orange'],
  ];
  rows.forEach(([title, works, questions, color], index) => {
    const y = 114 + index * 132;
    d.node(48, y, 208, 106, title, [], color, false, 21);
    d.node(320, y, 360, 106, works[0], [works[1]], 'gray', false, 19);
    d.node(776, y, 296, 106, questions[0], [questions[1]], color, false, 19);
    d.arrow([[680, y + 53], [776, y + 53]], '相关问题', 'gray', true);
  });
  d.node(48, 654, 1024, 62, '2026：Jev / Perplexity / Strands / OpenAI 将判断做成专门接口', [], 'gray', false, 21);
  d.legend(736, [['gray', '虚线表示相关背景，非直接继承', true]]);
  figures.push(d.save());
}
{
  const d = new Diagram('03-readout', 878, '图 3 · 相同 choice 接口，可能是不同读出',
    '只画公开实现；结构差异不能直接转换成业务准确率或速度排名');
  const rows = [
    ['生成式模型', '预训练主干', ['读入上下文'], '词表生成头', ['逐 token 解码'], '文字 / JSON', ['格式有效 ≠ 语义正确'], 'gray'],
    ['Strands 早期 slot head', '因果主干', ['形成答案位置表示'], '固定槽位映射', ['第 0 / 1 / 2 … 槽位'], '槽位分布', ['读出参数对应固定位置'], 'gray'],
    ['Strands pointer head', '因果主干 + LoRA', ['保留因果预训练结构'], '比较候选内容', ['答案 query × 选项 key'], '运行时候选分布', ['仍可能受选项顺序影响'], 'purple'],
    ['Perplexity v1.1', '混合注意力主干', ['full-attention 层非因果'], '固定线性读出', ['取最后位置表示'], '固定输出代码分布', ['公开参考实现无 KV cache'], 'blue'],
  ];
  rows.forEach(([label, a, aLines, b, bLines, c, cLines, color], index) => {
    const y = 106 + index * 180;
    d.text(48, y + 20, label, 20, 600, 1024);
    d.node(48, y + 42, 288, 102, a, aLines, color, false, 21);
    d.node(416, y + 42, 288, 102, b, bLines, color, false, 21);
    d.node(784, y + 42, 288, 102, c, cLines, color, false, 21);
    d.arrow([[336, y + 93], [416, y + 93]], '表示', 'blue');
    d.arrow([[704, y + 93], [784, y + 93]], '读出', 'blue');
  });
  d.text(48, 854, 'Jev 与 OpenAI 的公开材料不足以复原内部读出；本文不替闭源实现补画结构。', 17, 400, 1024, 'muted');
  figures.push(d.save());
}
{
  const d = new Diagram('04-landscape', 760, '图 4 · 执行环境、动作策略、业务授权是不同层',
    '产品可以横跨多层；分组不表示互斥，也不表示只能上下固定配对');
  d.text(48, 126, '执行底座：代码在哪里运行，与宿主怎样隔离？', 21, 600, 1024);
  d.node(48, 152, 288, 196, '复用宿主', ['Claude / Codex 命令沙箱', 'Anthropic SRT', '宿主内核 + 进程 / 出站限制'], 'blue');
  d.node(416, 152, 288, 196, '本地独立客体', ['Docker Sandboxes', 'microVM + 客体内核', '共享目录仍需治理'], 'purple');
  d.node(784, 152, 288, 196, '远程托管环境', ['E2B / Daytona / Modal', 'AgentCore 执行环境', '隔离、依赖、状态合同各异'], 'green');
  d.text(48, 406, '动作策略：哪个请求可执行，可信历史怎样参与判断？', 21, 600, 1024);
  d.node(48, 432, 288, 156, '本地事件治理', ['Box / Dogwood', '受控文件、命令、网络与 MCP', '直接访问有历史盲区'], 'orange');
  d.node(416, 432, 288, 156, '工具调用治理', ['AgentCore Gateway Policy', 'Gateway 工具与 policy session', '不自动观察 Runtime 文件'], 'orange');
  d.node(784, 432, 288, 156, '方案内授权引擎', ['OPA / Cedar', '可信数据输入 + 执行点', '历史持久化由方案提供'], 'orange');
  d.node(48, 650, 1024, 72, '业务事实与授权仍需原系统负责', ['用户 / session 绑定 · 审批 · 数据库权限 · 幂等与账本'], 'gray');
  figures.push(d.save());
}
{
  const d = new Diagram('05-box-boundary', 854, '图 5 · Box 的策略历史只覆盖进入中介的事件',
    '操作系统可达权限与 Dogwood 动作规则分别负责；同一 Box 目录保留历史');
  d.node(48, 126, 304, 196, 'Agent 进程', ['拟议命令、文件与网络请求', '在操作系统沙箱内运行'], 'blue');
  d.rect(432, 126, 640, 336, 'gray', true);
  d.text(456, 157, 'Agent 之外的可信 Box 进程', 20, 600, 590);
  d.node(456, 184, 264, 130, '受控入口', ['Shell / Monty 文件操作', '网络网关 / MCP broker'], 'orange');
  d.node(800, 184, 248, 130, 'Dogwood 引擎', ['规则 + 已记录历史', '请求先记录，再判定'], 'purple');
  d.node(800, 382, 248, 62, '同一 Box 的持久历史', [], 'green', false, 19);
  d.arrow([[352, 249], [456, 249]], '受控请求', 'blue');
  d.arrow([[720, 249], [800, 249]], '策略判定', 'blue');
  d.arrow([[924, 314], [924, 382]], '事件状态', 'green', true);
  d.node(48, 546, 304, 122, '未逐次进入历史的文件 I/O', ['Agent 原生 Read / Write', '另有原生工具内部文件访问'], 'red', false, 18);
  d.node(432, 546, 640, 122, '外部工具 / 业务服务', ['中介持 permit 后调用；副作用仍可能发生在远端', '工具自身的文件访问受其 OS grants 限制'], 'gray');
  d.arrow([[200, 322], [200, 546]], '直接读写：仅 OS grants', 'gray', true, [200, 432]);
  d.arrow([[588, 314], [588, 546]], '获准后执行', 'blue', false, [588, 488]);
  d.arrow([[1000, 546], [1000, 444]], 'outcome', 'gray', true);
  d.node(48, 716, 1024, 62, '副作用已发生 ≠ outcome 已落盘；准确一次与跨会话硬额度还需要业务端机制', [], 'red', false, 20);
  d.legend(824, [['blue', '受控请求 / 执行'], ['green', '历史事件写入', true], ['gray', '直接访问或结果记录', true]]);
  figures.push(d.save());
}
fs.writeFileSync(path.join(directory, 'manifest.json'), JSON.stringify(figures.map(({file, ...rest}) => rest), null, 2) + '\n');
console.log(JSON.stringify({directory, diagrams: figures.length, mode: 'generated-svg'}, null, 2));

if (process.argv.includes('--export-png')) {
  const run = args => execFileSync('agent-browser', ['--session', 'decision-report-figures', ...args], {encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
  try {
    for (const figure of figures) {
      run(['set', 'viewport', String(figure.width * 2), String(figure.height * 2)]);
      run(['open', pathToFileURL(figure.file).href]);
      const metrics = run(['eval', `document.documentElement.style.width="${figure.width * 2}px"; document.documentElement.style.height="${figure.height * 2}px"; JSON.stringify([...document.querySelectorAll("text[data-available-width]")].filter(t=>t.getBBox().width>Number(t.dataset.availableWidth)+1).map(t=>({text:t.textContent,width:t.getBBox().width,available:Number(t.dataset.availableWidth)})))`]);
      const errors = JSON.parse(JSON.parse(metrics));
      if (errors.length) throw new Error(`${figure.name} text overflow: ${JSON.stringify(errors)}`);
      run(['screenshot', figure.file.replace(/\.svg$/, '.png')]);
      console.log(`Exported and text-checked: ${figure.name}`);
    }
  } finally {
    run(['close']);
  }
}
