import { readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";

// The reading edition is authored in Markdown. This renders a self-contained
// HTML artifact; it does not change the research report or its source notes.
const notesDir = dirname(fileURLToPath(import.meta.url));
const workspace = resolve(notesDir, "..");
const edition = process.argv[2] ?? "readable";
assert(["readable", "detailed"].includes(edition), "Edition must be readable or detailed");
const editionLabel = edition === "detailed" ? "详细阅读版" : "阅读版";
const input = resolve(workspace, `2026-10-17-rsi-${edition}.md`);
const output = resolve(workspace, `2026-10-17-rsi-${edition}.html`);
const deck = edition === "detailed"
  ? "把重要研究、行业进展和实验证据放在同一张地图里。<br>逐项解释方法如何工作、已经证明什么，以及哪些结论还需要验证。"
  : "先弄清系统改进了什么，再看实验能支持怎样的结论。<br>这份中文阅读版串起研究脉络、代表工作与尚未解决的问题。";
const markdown = readFileSync(input, "utf8").replace(/\r\n/g, "\n");
const definitions = new Map();
const uses = new Map();

const body = markdown.replace(/^\[\^(\d+)\]: (.+)$/gm, (_, id, text) => {
  assert(!definitions.has(id), `Duplicate footnote ${id}`);
  definitions.set(id, text);
  return "";
});
for (const match of body.matchAll(/\[\^(\d+)\]/g)) {
  assert(definitions.has(match[1]), `Missing footnote ${match[1]}`);
}

const escapeHTML = (text) => text.replace(/[&<>"']/g, (ch) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
}[ch]));

function inline(text, citations = true) {
  const tokens = [];
  const token = (html) => {
    tokens.push(html);
    return `\u0000${tokens.length - 1}\u0000`;
  };
  let source = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_, label, href) => {
    assert(/^(https?:\/\/|[^:/?#][^:]*$)/.test(href), `Unsafe link ${href}`);
    const external = /^https?:\/\//.test(href);
    return token(`<a href="${escapeHTML(href)}"${external ? ' target="_blank" rel="noopener noreferrer"' : ""}>${escapeHTML(label)}</a>`);
  });
  if (citations) {
    source = source.replace(/\[\^(\d+)\]/g, (_, id) => {
      const index = (uses.get(id) ?? 0) + 1;
      uses.set(id, index);
      const title = definitions.get(id).replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
      return token(`<sup class="citation" id="ref-${id}-${index}"><a href="#note-${id}" role="doc-noteref" aria-label="查看参考资料 ${id}" title="${escapeHTML(title)}">${id}</a></sup>`);
    });
  }
  source = escapeHTML(source)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  return source.replace(/\u0000(\d+)\u0000/g, (_, index) => tokens[Number(index)]);
}

const lines = body.split("\n");
const title = lines.find((line) => line.startsWith("# ")).slice(2);
const titleHTML = edition === "detailed"
  ? escapeHTML(title)
    .replace("：", "：<br>")
    .replace(/(技术机制、|实验证据|与研究路线|重要研究、|行业进展|与技术机制)/g, '<span class="title-part">$1</span>')
  : escapeHTML(title);
const sections = [];
const html = [];
let paragraph = [];
let list = [];
let table = [];
let sectionOpen = false;
let sectionIndex = 0;
let paragraphIndex = 0;

function flush() {
  if (paragraph.length) {
    html.push(`<p${paragraphIndex++ === 0 ? ' class="opening"' : ""}>${inline(paragraph.join(""))}</p>`);
    paragraph = [];
  }
  if (list.length) {
    html.push(`<ul>${list.map((item) => `<li>${inline(item)}</li>`).join("")}</ul>`);
    list = [];
  }
  if (table.length) {
    const rows = table.map((line) => line.trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
    assert(rows.length >= 3, "Table needs a header, separator, and body");
    assert(rows[1].every((cell) => /^:?-{3,}:?$/.test(cell)), "Invalid table separator");
    assert(rows.every((row) => row.length === rows[0].length), "Inconsistent table columns");
    const head = rows[0].map((cell) => `<th scope="col">${inline(cell)}</th>`).join("");
    const content = rows.slice(2).map((row) => `<tr>${row.map((cell) => `<td>${inline(cell)}</td>`).join("")}</tr>`).join("\n");
    html.push(`<div class="table-scroll" role="region" tabindex="0" aria-label="研究路线对照表，可左右滚动"><table><caption>研究路线与证据类型 · 窄屏可左右滚动</caption><thead><tr>${head}</tr></thead><tbody>${content}</tbody></table></div>`);
    table = [];
  }
}

for (const line of lines) {
  if (line.startsWith("# ")) continue;
  if (!line.trim()) {
    flush();
    continue;
  }
  if (line.trim().startsWith("|") && line.trim().endsWith("|")) {
    if (paragraph.length || list.length) flush();
    table.push(line);
    continue;
  }
  if (table.length) flush();
  if (line.startsWith("## ")) {
    flush();
    if (sectionOpen) html.push("</section>");
    const label = line.slice(3).replace(/^[一二三四五六七八九十]+、/, "");
    const id = `section-${++sectionIndex}`;
    sections.push({ id, label });
    html.push(`<section aria-labelledby="${id}"><h2 id="${id}" tabindex="-1"><span class="section-number" aria-hidden="true">${String(sectionIndex).padStart(2, "0")}</span>${escapeHTML(label)}</h2>`);
    sectionOpen = true;
    continue;
  }
  if (line.startsWith("### ")) {
    flush();
    html.push(`<h3>${inline(line.slice(4))}</h3>`);
    continue;
  }
  if (line.startsWith("- ")) {
    if (paragraph.length) flush();
    list.push(line.slice(2));
    continue;
  }
  if (list.length) flush();
  paragraph.push(line);
}
flush();
if (sectionOpen) html.push("</section>");

const footnotes = [...definitions].map(([id, text]) => {
  assert(uses.has(id), `Unused footnote ${id}`);
  const backlinks = Array.from({ length: uses.get(id) }, (_, index) =>
    `<a class="backlink" href="#ref-${id}-${index + 1}" aria-label="返回正文引用 ${id}${uses.get(id) > 1 ? `，第 ${index + 1} 处` : ""}">↩${uses.get(id) > 1 ? index + 1 : ""}</a>`
  ).join(" ");
  return `<li id="note-${id}" role="doc-endnote">${inline(text, false)} <span class="backlinks">${backlinks}</span></li>`;
}).join("\n");

const toc = sections.map(({ id, label }, index) =>
  `<a href="#${id}" data-toc="${id}"><span aria-hidden="true">${String(index + 1).padStart(2, "0")}</span>${escapeHTML(label)}</a>`
).join("\n");
const cjkCount = (body.match(/[\u3400-\u9fff]/g) ?? []).length;
const minutes = Math.max(1, Math.round(cjkCount / 350));

const page = `<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="一份连贯的中文 RSI 研究阅读版：重要研究、行业进展、技术机制、证据边界与研究方案。">
  <meta name="color-scheme" content="light">
  <title>${escapeHTML(title)} · ${editionLabel}</title>
  <style>
    :root { --paper:#faf8f3; --ink:#292d2a; --muted:#69706a; --line:#dddcd4; --accent:#315d4b; --reading-size:19px; --sans:Inter,"Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif; --serif:"Noto Serif CJK SC","Source Han Serif SC","Songti SC",Georgia,serif; }
    * { box-sizing:border-box; }
    html { scroll-padding-top:105px; }
    body { margin:0; background:var(--paper); color:var(--ink); font-family:var(--sans); -webkit-font-smoothing:antialiased; }
    button, summary { font:inherit; }
    a { color:var(--accent); text-underline-offset:4px; text-decoration-thickness:1px; }
    a:hover { color:#183f30; }
    a:focus-visible, button:focus-visible, summary:focus-visible { outline:2px solid var(--accent); outline-offset:5px; border-radius:3px; }
    .skip { position:fixed; top:-70px; left:20px; z-index:10; background:var(--paper); padding:12px 18px; }
    .skip:focus { top:12px; }
    .progress { position:fixed; top:0; left:0; width:0; height:3px; background:var(--accent); z-index:5; }
    .topbar { position:sticky; top:0; z-index:4; background:var(--paper); border-bottom:1px solid var(--line); }
    .top-inner { max-width:1100px; padding:16px 32px; margin:auto; display:flex; align-items:center; justify-content:space-between; gap:16px; }
    .edition { display:flex; align-items:center; gap:16px; font-size:13px; color:var(--muted); }
    .edition strong { color:var(--ink); font-weight:650; letter-spacing:.05em; }
    .edition .divider { height:13px; width:1px; background:var(--line); }
    .actions { display:flex; align-items:center; gap:7px; }
    .actions-label { font-size:12px; color:var(--muted); margin-right:3px; }
    button { min-height:36px; min-width:36px; padding:7px 10px; color:var(--ink); border:1px solid var(--line); background:transparent; border-radius:4px; cursor:pointer; font-size:13px; }
    button:hover { background:#eceee6; }
    button:disabled { opacity:.4; cursor:default; }
    .print-button { margin-left:8px; padding-inline:13px; }
    .layout { max-width:1100px; margin:0 auto; padding:66px 32px 100px; display:grid; grid-template-columns:185px minmax(0,1fr); gap:64px; align-items:start; }
    .sidebar { position:sticky; top:114px; }
    .sidebar-title { margin:0 0 12px; font-size:30px; font-weight:650; letter-spacing:-1px; }
    .sidebar-subtitle { margin:0 0 37px; font-size:12px; color:var(--muted); line-height:1.8; }
    .toc-label { font-size:11px; color:var(--muted); letter-spacing:.12em; margin:0 0 15px; }
    .toc { display:flex; flex-direction:column; gap:4px; }
    .sidebar .toc { max-height:calc(100dvh - 430px); overflow-y:auto; overscroll-behavior:contain; scrollbar-width:thin; scrollbar-color:#c7ccc3 transparent; }
    .toc a { display:flex; gap:10px; padding:9px 0 9px 10px; border-left:2px solid transparent; text-decoration:none; font-size:12px; line-height:1.65; color:var(--muted); }
    .toc a span { font-size:10px; color:#879188; padding-top:2px; flex:0 0 17px; }
    .toc a[aria-current="location"] { color:var(--accent); border-left-color:var(--accent); background:#eaf0e7; }
    .sidebar-footer { font-size:11px; line-height:1.85; color:var(--muted); border-top:1px solid var(--line); margin-top:30px; padding-top:18px; }
    main { min-width:0; max-width:720px; }
    .kicker { font-size:12px; color:var(--accent); letter-spacing:.08em; margin:0 0 20px; }
    h1 { font-family:var(--serif); font-size:44px; font-weight:700; line-height:1.4; letter-spacing:-.025em; margin:0 0 24px; text-wrap:balance; }
    .title-part { display:inline-block; }
    .deck { font-size:16px; line-height:1.9; color:var(--muted); margin:0 0 24px; max-width:40em; }
    .byline { display:flex; flex-wrap:wrap; gap:8px 20px; padding:0 0 30px; margin-bottom:34px; border-bottom:1px solid var(--line); color:var(--muted); font-size:12px; line-height:1.8; }
    .mobile-nav { display:none; }
    article { font-family:var(--serif); font-size:var(--reading-size); line-height:1.95; overflow-wrap:break-word; }
    article p { margin:0 0 23px; }
    article .opening { color:#363d36; }
    article h2 { display:block; font-family:var(--sans); font-size:25px; font-weight:650; line-height:1.5; margin:61px 0 27px; padding-top:24px; border-top:1px solid var(--line); letter-spacing:-.025em; }
    article h2:focus { outline:none; }
    .section-number { display:block; font-family:var(--sans); font-size:11px; font-weight:500; color:var(--accent); margin-bottom:9px; letter-spacing:.08em; }
    article h3 { font-family:var(--sans); font-size:20px; line-height:1.6; font-weight:650; margin:35px 0 17px; }
    article li { margin-bottom:10px; }
    article code { font-family:var(--sans); font-size:.9em; }
    .table-scroll { max-width:100%; overflow-x:auto; margin:26px 0 34px; border:1px solid var(--line); border-radius:4px; overscroll-behavior-x:contain; }
    table { width:100%; min-width:680px; border-collapse:collapse; font-family:var(--sans); font-size:calc(var(--reading-size) * .76); line-height:1.8; }
    caption { text-align:left; padding:11px 14px; color:var(--muted); font-size:11px; border-bottom:1px solid var(--line); }
    th, td { text-align:left; vertical-align:top; padding:12px 14px; border-bottom:1px solid var(--line); }
    th { color:var(--accent); font-weight:650; background:#edf1e9; }
    tbody tr:last-child td { border-bottom:0; }
    td:first-child { width:19%; }
    td:nth-child(2) { width:27%; }
    .citation { font-family:var(--sans); font-size:10px; font-weight:600; line-height:0; margin-left:4px; }
    .citation a { text-decoration:none; padding:5px 2px; }
    .references { font-family:var(--sans); margin-top:46px; padding:21px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); }
    .references summary { cursor:pointer; font-size:15px; font-weight:650; color:var(--accent); }
    .references-description { font-size:12px; line-height:1.8; color:var(--muted); margin:18px 0; }
    .references ol { padding-left:23px; font-size:13px; line-height:1.85; }
    .references li { padding-left:5px; margin-bottom:16px; overflow-wrap:anywhere; scroll-margin-top:110px; }
    .references li:target { background:#edf1e7; outline:8px solid #edf1e7; }
    .backlink { text-decoration:none; padding:3px; }
    .page-footer { font-family:var(--sans); margin-top:28px; font-size:11px; color:var(--muted); line-height:1.9; }
    @media (max-width:1000px) {
      .layout { display:block; max-width:780px; padding-top:43px; }
      .sidebar { display:none; }
      main { max-width:none; }
      .top-inner { max-width:780px; }
      .mobile-nav { display:block; margin:0 0 30px; padding:13px 0; border-top:1px solid var(--line); border-bottom:1px solid var(--line); font-size:13px; }
      .mobile-nav summary { cursor:pointer; color:var(--accent); }
      .mobile-nav .toc { margin-top:12px; }
      .mobile-nav .toc a { font-size:13px; }
    }
    @media (max-width:560px) {
      :root { --reading-size:18px; }
      html { scroll-padding-top:91px; }
      .top-inner { padding:12px 20px; gap:10px; }
      .edition { gap:8px; font-size:11px; }
      .edition .divider, .edition .language, .actions-label { display:none; }
      .actions { gap:5px; }
      button { min-width:40px; min-height:40px; padding:7px; }
      .print-button { margin-left:3px; padding-inline:10px; }
      .layout { padding:35px 23px 65px; }
      h1 { font-size:34px; line-height:1.45; margin-bottom:20px; }
      .kicker { font-size:11px; margin-bottom:15px; }
      .deck { font-size:14px; line-height:1.9; }
      .byline { gap:5px 16px; font-size:11px; padding-bottom:23px; margin-bottom:24px; }
      article { line-height:1.95; }
      article p { margin-bottom:21px; }
      article h2 { font-size:22px; margin-top:46px; }
      article h3 { font-size:19px; }
      .references ol { font-size:12px; }
    }
    @media print {
      @page { size:A4; margin:22mm 22mm 23mm; }
      :root { --reading-size:11pt; --paper:#fff; --ink:#111; --muted:#555; }
      body { background:white; }
      .skip, .progress, .topbar, .sidebar, .mobile-nav { display:none !important; }
      .layout { display:block; max-width:none; padding:0; }
      main { max-width:none; }
      h1 { font-size:27pt; line-height:1.35; }
      .kicker { font-size:9pt; }
      .deck { font-size:11pt; }
      .byline { font-size:8pt; margin-bottom:20pt; padding-bottom:14pt; }
      article { line-height:1.8; }
      article p { margin-bottom:11pt; orphans:3; widows:3; }
      article h2 { font-size:16pt; margin-top:25pt; padding-top:13pt; break-after:avoid; }
      article h3 { font-size:12pt; margin-top:18pt; break-after:avoid; }
      .section-number { font-size:8pt; }
      a { color:inherit; }
      .references { font-size:9pt; break-before:page; }
      .references summary { font-size:12pt; list-style:none; }
      .references li { font-size:8.5pt; break-inside:avoid; }
      .references-description, .page-footer { font-size:8pt; }
      .citation { font-size:7pt; }
      .backlinks { display:none; }
      .table-scroll { overflow:visible; break-inside:avoid; }
      table { min-width:0; font-size:8pt; line-height:1.6; }
      th, td { padding:6pt; }
    }
  </style>
</head>
<body>
  <a class="skip" href="#article">跳到正文</a>
  <div class="progress" aria-hidden="true"></div>
  <header class="topbar">
    <div class="top-inner">
      <div class="edition"><strong>研究阅读</strong><span class="divider"></span><span class="language">简体中文 · ${editionLabel}</span></div>
      <div class="actions" role="group" aria-label="阅读设置">
        <span class="actions-label">字号</span>
        <button type="button" id="font-smaller" aria-label="缩小正文字号">A−</button>
        <button type="button" id="font-larger" aria-label="放大正文字号">A+</button>
        <button type="button" class="print-button" id="print">打印</button>
      </div>
    </div>
  </header>
  <div class="layout">
    <aside class="sidebar" aria-label="文章目录">
      <p class="sidebar-title">RSI</p>
      <p class="sidebar-subtitle">Recursive Self-Improvement<br>递归自我改进</p>
      <p class="toc-label">阅读目录 · ${sections.length} 章${sections.length > 10 ? " · 可滚动" : ""}</p>
      <nav class="toc">${toc}<a href="#references"><span aria-hidden="true">↗</span>论文与资料</a></nav>
      <div class="sidebar-footer">资料核对：2026.10.17<br>原始研究报告另行保留<br>本页无外部加载依赖</div>
    </aside>
    <main>
      <header class="article-header">
        <p class="kicker">人工智能 · 研究领域阅读</p>
        <h1>${titleHTML}</h1>
        <p class="deck">${deck}</p>
        <div class="byline"><span>资料核对于 2026 年 10 月 17 日</span><span>约 ${minutes} 分钟阅读</span><span>${edition === "detailed" ? "研究与行业补充" : "原文改编"} · 未独立复现实验</span></div>
      </header>
      <details class="mobile-nav">
        <summary>阅读目录</summary>
        <nav class="toc" aria-label="文章章节">${toc}<a href="#references"><span aria-hidden="true">↗</span>论文与资料</a></nav>
      </details>
      <article id="article">
        ${html.join("\n")}
        <details class="references" id="references">
          <summary>论文与资料 · ${definitions.size} 组引用</summary>
          <div>
            <p class="references-description">正文上标可以展开并定位对应资料。论文链接在新窗口打开；这里保留作者、时间和影响结论的版本说明。</p>
            <ol role="doc-endnotes">${footnotes}</ol>
          </div>
        </details>
      </article>
      <footer class="page-footer">阅读版正文与 Markdown 文件内容一致。页面可以离线打开，支持字号调整和打印；原始报告及详细核验记录未被覆盖。</footer>
    </main>
  </div>
  <script>
    (() => {
      const root = document.documentElement;
      const smaller = document.querySelector("#font-smaller");
      const larger = document.querySelector("#font-larger");
      let size = parseFloat(getComputedStyle(root).getPropertyValue("--reading-size"));
      function setSize(next) {
        size = Math.max(16, Math.min(25, next));
        root.style.setProperty("--reading-size", size + "px");
        smaller.disabled = size <= 16;
        larger.disabled = size >= 25;
      }
      smaller.addEventListener("click", () => setSize(size - 1));
      larger.addEventListener("click", () => setSize(size + 1));
      document.querySelector("#print").addEventListener("click", () => window.print());
      const references = document.querySelector("#references");
      document.addEventListener("click", (event) => {
        const link = event.target.closest('a[href^="#"]');
        if (!link) return;
        const id = link.getAttribute("href").slice(1);
        if (id.startsWith("note-") || id === "references") references.open = true;
        if (link.closest(".mobile-nav") && id.startsWith("section-")) {
          document.querySelector(".mobile-nav").open = false;
        }
      });
      function handleHash() {
        const id = location.hash.slice(1);
        if (id.startsWith("note-") || id === "references") {
          references.open = true;
          const target = document.getElementById(id);
          if (target) requestAnimationFrame(() => target.scrollIntoView());
        }
      }
      window.addEventListener("hashchange", handleHash);
      handleHash();
      const headings = Array.from(document.querySelectorAll("article h2[id]"));
      let lastChapter = null;
      function updatePosition() {
        const height = root.scrollHeight - innerHeight;
        document.querySelector(".progress").style.width = (height > 0 ? Math.min(100, scrollY / height * 100) : 0) + "%";
        let current = headings[0]?.id;
        for (const heading of headings) {
          if (heading.getBoundingClientRect().top <= 170) current = heading.id;
        }
        document.querySelectorAll("[data-toc]").forEach((link) => {
          if (link.dataset.toc === current) link.setAttribute("aria-current", "location");
          else link.removeAttribute("aria-current");
        });
        if (current !== lastChapter) {
          lastChapter = current;
          const active = document.querySelector('.sidebar [aria-current="location"]');
          const nav = document.querySelector(".sidebar .toc");
          if (active && getComputedStyle(document.querySelector(".sidebar")).display !== "none") {
            const box = active.getBoundingClientRect();
            const bounds = nav.getBoundingClientRect();
            if (box.bottom > bounds.bottom) nav.scrollTop += box.bottom - bounds.bottom + 8;
            else if (box.top < bounds.top) nav.scrollTop -= bounds.top - box.top + 8;
          }
        }
      }
      let pending = false;
      window.addEventListener("scroll", () => {
        if (pending) return;
        pending = true;
        requestAnimationFrame(() => { updatePosition(); pending = false; });
      }, { passive:true });
      window.addEventListener("resize", updatePosition);
      let referencesWereOpen = false;
      window.addEventListener("beforeprint", () => {
        referencesWereOpen = references.open;
        references.open = true;
      });
      window.addEventListener("afterprint", () => { references.open = referencesWereOpen; });
      updatePosition();
    })();
  </script>
</body>
</html>`;

assert(!/\[\^\d+\]/.test(page), "Unrendered footnote marker");
assert(!/\u0000/.test(page), "Unrendered inline token");
const ids = [...page.matchAll(/\bid="([^"]+)"/g)].map((match) => match[1]);
assert.equal(ids.length, new Set(ids).size, "Duplicate HTML IDs");
for (const match of page.matchAll(/href="#([^"]+)"/g)) {
  assert(ids.includes(match[1]), `Missing anchor ${match[1]}`);
}
writeFileSync(output, page, "utf8");
console.log(JSON.stringify({
  input, output, sections:sections.length, references:definitions.size,
  chineseCharacters:cjkCount, estimatedReadingMinutes:minutes,
  validInternalAnchors:true, externalAssets:0,
}, null, 2));
