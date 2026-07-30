# 🕸️ 知识图谱

把本站所有笔记按**关联强度**画成一张图。**点击任意节点即可跳转到对应笔记。**

- **连线含义**：直接链接（×3.0）、共享同一篇文献（×4.0）、共同邻居 Adamic-Adar（×1.5）、同类型（×1.0）——权重越高线越亮。
- **颜色**：按笔记类型着色（可切换为按主题社区着色）。**方形节点 = 桥节点**（连接 ≥3 个主题簇的枢纽）。
- **节点大小**：连接数越多越大。
- 方法参考 [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)（Karpathy LLM Wiki pattern 的实现）；数据由 `scripts/wiki_graph.py` 生成。文字版洞察见 [知识图谱洞察报告](/weekly/knowledge-graph.md)。

<div id="kg-app">
  <div class="kg-toolbar">
    <label>着色：
      <select id="kg-color">
        <option value="type">按类型</option>
        <option value="community">按主题社区</option>
      </select>
    </label>
    <label>最小权重：<input type="range" id="kg-minw" min="0" max="8" step="0.5" value="0"> <span id="kg-minw-val">0</span></label>
    <label><input type="checkbox" id="kg-hide-iso"> 隐藏孤立节点</label>
    <input type="search" id="kg-search" placeholder="搜索笔记标题…" />
    <span id="kg-stats" class="kg-stats"></span>
  </div>
  <div id="kg-canvas-wrap">
    <canvas id="kg-canvas"></canvas>
    <div id="kg-tip" class="kg-tip" hidden></div>
    <div id="kg-legend" class="kg-legend"></div>
  </div>
  <p class="kg-hint">拖拽平移 · 滚轮缩放 · 悬停高亮邻居 · <b>单击节点打开笔记</b> · 双击空白处重置视图</p>
</div>

<style>
#kg-app { margin: 1rem 0 2rem; }
.kg-toolbar { display:flex; flex-wrap:wrap; gap:14px; align-items:center; font-size:.86rem; margin-bottom:8px; }
.kg-toolbar label { display:flex; align-items:center; gap:6px; }
.kg-toolbar select, .kg-toolbar input[type=search] { padding:3px 6px; border:1px solid var(--kg-line,#d0d7de); border-radius:6px; font-size:.86rem; }
.kg-toolbar input[type=search] { min-width:180px; }
.kg-stats { color:#888; margin-left:auto; }
#kg-canvas-wrap { position:relative; width:100%; height:640px; border:1px solid var(--kg-line,#e1e4e8); border-radius:10px; overflow:hidden; background:#fbfcfd; }
#kg-canvas { width:100%; height:100%; display:block; cursor:grab; }
#kg-canvas.dragging { cursor:grabbing; }
.kg-tip { position:absolute; pointer-events:none; background:rgba(20,24,31,.94); color:#fff; padding:7px 10px; border-radius:7px; font-size:.78rem; max-width:320px; line-height:1.45; z-index:5; }
.kg-tip b { color:#7ee0c0; }
.kg-legend { position:absolute; right:10px; bottom:10px; background:rgba(255,255,255,.94); border:1px solid #e1e4e8; border-radius:8px; padding:8px 10px; font-size:.74rem; line-height:1.7; max-height:44%; overflow:auto; }
.kg-legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:middle; }
.kg-hint { font-size:.8rem; color:#888; margin-top:8px; }
</style>

<!-- 交互逻辑在 assets/kg.js，由 index.html 的 docsify 插件在路由到本页时挂载 -->

