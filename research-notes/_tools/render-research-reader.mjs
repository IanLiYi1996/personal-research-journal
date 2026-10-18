import fs from 'node:fs';
import path from 'node:path';

// A local, dependency-free formatter for this report's deliberately small
// Markdown vocabulary. It never loads remote assets or evaluates source HTML.
const sourcePath = path.resolve(process.argv[2] ?? '');
if (!sourcePath.endsWith('.md') || !fs.statSync(sourcePath).isFile()) {
  throw new Error('Pass one existing Markdown report.');
}
const outputPath = sourcePath.replace(/\.md$/, '.html');
const marker = 'data-reader-generated="research-reader-v1"';
if (fs.existsSync(outputPath) && !fs.readFileSync(outputPath, 'utf8').includes(marker)) {
  throw new Error('Refusing to overwrite a non-generated HTML file.');
}

const source = fs.readFileSync(sourcePath, 'utf8').replace(/\r\n/g, '\n');
const notes = new Map();
const images = new Map();
const lines = source.split('\n').filter(line => {
  const image = line.match(/^\[(figure-\d+)\]:\s*(data:image\/png;base64,[A-Za-z0-9+/=]+)$/);
  if (image) {
    if (images.has(image[1])) throw new Error(`Duplicate image ${image[1]}`);
    images.set(image[1], image[2]);
    return false;
  }
  if (line === '<!-- standalone-research-generated-v1 -->') return false;
  const match = line.match(/^\[\^(\d+)\]:\s*(.*)$/);
  if (!match) return true;
  if (notes.has(match[1])) throw new Error(`Duplicate note ${match[1]}`);
  notes.set(match[1], match[2]);
  return false;
});
const used = new Map();
const escape = value => String(value)
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;')
  .replaceAll('"', '&quot;').replaceAll("'", '&#39;');
function safeHref(href) {
  if (/^[a-z][a-z0-9+.-]*:/i.test(href) && !/^https?:/i.test(href)) {
    throw new Error(`Unsupported link scheme: ${href}`);
  }
  return escape(href);
}
function inline(value, track = true) {
  const tokens = [];
  const stash = html => `@@READER_TOKEN_${tokens.push(html) - 1}@@`;
  let text = value.replace(/`([^`]+)`/g, (_, code) => stash(`<code>${escape(code)}</code>`));
  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+"[^"]*")?\)/g, (_, label, href) => {
    const extra = /^https?:/.test(href) ? ' target="_blank" rel="noopener noreferrer"' : '';
    return stash(`<a href="${safeHref(href)}"${extra}>${escape(label)}</a>`);
  });
  text = text.replace(/\[\^(\d+)\]/g, (_, number) => {
    if (!notes.has(number)) throw new Error(`Undefined note ${number}`);
    const occurrence = (used.get(number) ?? 0) + 1;
    if (track) used.set(number, occurrence);
    return stash(`<sup class="note-ref"><a id="ref-${number}-${occurrence}" href="#note-${number}" aria-label="来源 ${number}">${number}</a></sup>`);
  });
  text = escape(text).replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  return text.replace(/@@READER_TOKEN_(\d+)@@/g, (_, index) => tokens[Number(index)]);
}
const nonempty = lines.filter(line => line.trim());
const title = nonempty[0].replace(/^# /, '');
const subtitle = nonempty[1];
const metadata = nonempty[2];
const skipHeader = new Set([nonempty[0], subtitle, metadata]);
const toc = [];
const body = [];
let tableIndex = 0;
let imageIndex = 0;
const usedImages = new Set();
function tableCells(line) {
  return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map(cell => cell.trim());
}
function startsBlock(line) {
  return /^#{1,3} |^```|^- |^\||^!\[/.test(line);
}
for (let index = 0; index < lines.length;) {
  const line = lines[index];
  if (!line.trim() || skipHeader.has(line) || line === '### 来源注释') {
    index++;
    continue;
  }
  const heading = line.match(/^(#{2,3}) (.+)$/);
  if (heading) {
    const level = heading[1].length;
    const id = level === 2 ? `section-${toc.length + 1}` : `subsection-${index}`;
    if (level === 2) toc.push({id, title: heading[2]});
    body.push(`<h${level} id="${id}">${inline(heading[2])}</h${level}>`);
    index++;
  } else if (line.startsWith('![')) {
    const match = line.match(/^!\[([^\]]+)\]\[(figure-\d+)\]$/);
    if (!match || !images.has(match[2])) throw new Error('Only embedded PNG reference images are supported.');
    const next = index + 1;
    let captionIndex = next;
    while (captionIndex < lines.length && !lines[captionIndex].trim()) captionIndex++;
    const caption = lines[captionIndex] ?? '';
    const hasCaption = /^图 \d+ · /.test(caption);
    usedImages.add(match[2]);
    imageIndex++;
    body.push(`<figure id="figure-${imageIndex}"><p class="figure-hint">图较宽，可左右滑动查看。</p><div class="figure-scroll" role="region" tabindex="0" aria-label="图 ${imageIndex}，窄屏可横向滚动"><img src="${images.get(match[2])}" alt="${escape(match[1])}" decoding="async"></div>${hasCaption ? `<figcaption>${inline(caption)}</figcaption>` : ''}</figure>`);
    index = hasCaption ? captionIndex + 1 : next;
  } else if (line.startsWith('```')) {
    const content = [];
    index++;
    while (index < lines.length && !lines[index].startsWith('```')) content.push(lines[index++]);
    if (index === lines.length) throw new Error('Unclosed code fence.');
    index++;
    body.push(`<pre><code>${escape(content.join('\n'))}</code></pre>`);
  } else if (line.startsWith('|')) {
    const rows = [];
    while (index < lines.length && lines[index].startsWith('|')) rows.push(lines[index++]);
    if (rows.length < 2 || !/^\|(?:\s*:?-+:?\s*\|)+\s*$/.test(rows[1])) {
      throw new Error('Unsupported table shape.');
    }
    const name = tableCells(rows[0]).join('、');
    const th = tableCells(rows[0]).map(cell => `<th scope="col">${inline(cell)}</th>`).join('');
    const tr = rows.slice(2).map(row => `<tr>${tableCells(row).map(cell => `<td>${inline(cell)}</td>`).join('')}</tr>`).join('');
    tableIndex++;
    body.push(`<div class="table-scroll" tabindex="0" role="region" aria-label="${escape(name)}"><table><thead><tr>${th}</tr></thead><tbody>${tr}</tbody></table></div>`);
  } else if (line.startsWith('- ')) {
    const items = [];
    while (index < lines.length && lines[index].startsWith('- ')) items.push(`<li>${inline(lines[index++].slice(2))}</li>`);
    body.push(`<ul>${items.join('')}</ul>`);
  } else {
    const paragraph = [];
    while (index < lines.length && lines[index].trim() && !startsBlock(lines[index])) paragraph.push(lines[index++]);
    body.push(`<p>${inline(paragraph.join(' '))}</p>`);
  }
}
const unused = [...notes.keys()].filter(number => !used.has(number));
if (unused.length) throw new Error(`Unused notes: ${unused.join(', ')}`);
if ([...images.keys()].some(id => !usedImages.has(id))) throw new Error('Unused embedded image.');
const footnotes = [...notes].sort((a, b) => Number(a[0]) - Number(b[0]))
  .map(([number, text]) => `<li id="note-${number}" value="${number}"><p>${inline(text, false)} <a class="backlink" href="#ref-${number}-1" aria-label="返回来源 ${number} 的正文">返回正文 ↩</a></p></li>`).join('\n');
const navigation = toc.map(item => `<li><a href="#${item.id}">${escape(item.title)}</a></li>`).join('');
const html = `<!doctype html>
<html lang="zh-CN" ${marker}>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="${escape(subtitle)}">
<meta name="referrer" content="no-referrer">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; script-src 'none'">
<title>${escape(title)}</title>
<style>
:root{color-scheme:light;--paper:#faf9f5;--ink:#242b32;--muted:#616a70;--line:#d9ddda;--accent:#145c69;--wash:#efefea}
*{box-sizing:border-box}html{scroll-padding-top:28px}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Noto Serif CJK SC","Songti SC",Georgia,serif;font-size:17px;line-height:1.95}
a{color:var(--accent);text-decoration-thickness:1px;text-underline-offset:3px;overflow-wrap:anywhere}a:focus-visible,.table-scroll:focus-visible{outline:3px solid var(--accent);outline-offset:4px;border-radius:2px}
.skip{position:absolute;left:12px;top:-80px;padding:8px 14px;background:var(--paper)}.skip:focus{top:10px;z-index:3}
.layout{max-width:1170px;margin:0 auto;padding:64px 36px 88px;display:grid;grid-template-columns:216px minmax(0,1fr);gap:54px}
.toc{position:sticky;top:28px;align-self:start;font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:13px;line-height:1.7}.toc p{font-size:12px;letter-spacing:.08em;color:var(--muted);margin:0 0 14px}.toc ol{padding:0;list-style:none;margin:0}.toc li{margin:0 0 12px}.toc a{display:block;text-decoration:none;color:var(--muted)}.toc a:hover{color:var(--accent)}.toc .source-jump{margin-top:20px;padding-top:15px;border-top:1px solid var(--line)}
.mobile-toc{display:none;margin-top:20px;font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:12px;line-height:1.8}.mobile-toc summary{color:var(--accent);cursor:pointer;padding:8px 0;min-height:38px}.mobile-toc summary:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.mobile-toc ol{padding:12px 0 0;list-style:none;display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:10px 18px}.mobile-toc li{margin:0}.mobile-toc a{text-decoration:none}
article{min-width:0}header{padding-bottom:32px;border-bottom:1px solid var(--line);margin-bottom:38px}.meta{font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:12px;letter-spacing:.025em;color:var(--muted);margin:0 0 22px}h1{font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:37px;font-weight:650;line-height:1.45;letter-spacing:-.04em;margin:0 0 20px}.subtitle{font-size:17px;line-height:1.9;color:var(--muted);margin:0}
h2,h3{font-family:"Noto Sans CJK SC",system-ui,sans-serif;line-height:1.65;font-weight:600}h2{font-size:25px;margin:62px 0 24px;letter-spacing:-.025em}h2:first-of-type{margin-top:0}h3{font-size:19px;margin:34px 0 18px}
p{margin:0 0 22px;overflow-wrap:break-word}strong{font-weight:650}code{font-family:ui-monospace,"SFMono-Regular",Consolas,monospace;font-size:.83em;padding:.1em .24em;background:var(--wash);border-radius:3px;overflow-wrap:anywhere}
pre{background:var(--wash);padding:20px;border:1px solid var(--line);overflow:auto;border-radius:5px;line-height:1.7}pre code{background:none;padding:0;font-size:13px;white-space:pre}
.note-ref{font-family:system-ui,sans-serif;font-size:.66em;line-height:0;margin-left:2px}.note-ref a{text-decoration:none;padding:0 2px}
figure{margin:30px 0 34px}.figure-hint{display:none;font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:11px;color:var(--muted);margin-bottom:8px}.figure-scroll{max-width:100%;overflow-x:auto;border:1px solid var(--line);border-radius:5px;background:#fff}.figure-scroll:focus-visible{outline:3px solid var(--accent);outline-offset:4px}.figure-scroll img{display:block;width:100%;min-width:720px;height:auto}figcaption{font-family:"Noto Sans CJK SC",system-ui,sans-serif;color:var(--muted);font-size:13px;line-height:1.9;margin-top:12px}figcaption .note-ref{font-size:.8em}
.table-scroll{max-width:100%;overflow-x:auto;margin:26px 0 28px;border:1px solid var(--line);border-radius:4px}table{border-collapse:collapse;width:100%;min-width:540px;font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:13px;line-height:1.85}th,td{padding:14px 16px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}th{background:var(--wash);font-weight:600}tr:last-child td{border-bottom:0}td:first-child{width:23%}
.notes{border-top:1px solid var(--line);margin-top:52px;padding-top:22px;font-family:"Noto Sans CJK SC",system-ui,sans-serif;font-size:12px;line-height:1.9;color:var(--muted)}.notes h2{font-size:19px;margin:0 0 16px}.notes ol{padding-left:23px}.notes li{padding-left:3px;margin-bottom:14px;scroll-margin-top:24px}.notes p{margin:0}.backlink{font-size:11px;text-decoration:none;white-space:nowrap}.notes li:target{background:var(--wash);outline:5px solid var(--wash);border-radius:3px}footer{font-family:"Noto Sans CJK SC",system-ui,sans-serif;color:var(--muted);font-size:11px;margin-top:36px;border-top:1px solid var(--line);padding-top:16px}
@media(max-width:1000px){.layout{display:block;max-width:850px;padding:38px 26px 60px}.toc{display:none}.mobile-toc{display:block}h1{font-size:33px}}
@media(max-width:760px){.figure-hint{display:block}}
@media(max-width:540px){body{font-size:16px;line-height:1.95}.layout{padding:26px 20px 44px}.toc{font-size:11px;line-height:1.8;margin-bottom:26px}.toc ol{gap:6px 14px}.toc p{display:none}.meta{font-size:10px;margin-bottom:16px}header{padding-bottom:24px;margin-bottom:28px}h1{font-size:27px;line-height:1.5;margin-bottom:16px}.subtitle{font-size:14px}h2{font-size:21px;margin-top:46px}h3{font-size:17px;margin-top:28px}p{margin-bottom:20px}.figure-hint{margin-bottom:8px}table{font-size:12px}th,td{padding:11px 13px}.notes{font-size:11px}}
@media(prefers-reduced-motion:no-preference){html{scroll-behavior:smooth}}
@media print{html{scroll-behavior:auto}body{background:white;color:black;font-size:11pt;line-height:1.7}.layout{display:block;padding:0;max-width:none}.toc,.mobile-toc,.skip,.backlink,.figure-hint{display:none}h1{font-size:23pt}h2{font-size:16pt;margin-top:28px;break-after:avoid}h3{font-size:12pt;break-after:avoid}p{orphans:3;widows:3}header{margin-bottom:24px}.table-scroll{overflow:visible;break-inside:avoid}table{min-width:0;font-size:9pt}figure{break-inside:avoid}.figure-scroll{overflow:visible}.figure-scroll img{min-width:0}figcaption{font-size:9pt}a{color:inherit}code{background:none}.notes{font-size:8pt}.notes li{break-inside:avoid}footer{font-size:8pt}}
</style>
</head>
<body>
<a class="skip" href="#article-start">跳至正文</a>
<div class="layout">
<nav class="toc" aria-label="文章目录"><p>技术脉络与选型</p><ol>${navigation}</ol><a class="source-jump" href="#notes">来源注释 · ${notes.size} 组</a></nav>
<article id="article-start">
<header><p class="meta">${escape(metadata)}</p><h1>${escape(title)}</h1><p class="subtitle">${inline(subtitle)}</p><details class="mobile-toc"><summary>目录 · ${toc.length} 章</summary><nav aria-label="折叠文章目录"><ol>${navigation}</ol></nav></details></header>
${body.join('\n')}
<section class="notes" id="notes" aria-labelledby="notes-title"><h2 id="notes-title">来源注释</h2><ol>${footnotes}</ol></section>
<footer>本页与同名 Markdown 内容一致，可各自独立携带。全部 ${imageIndex} 张图和样式在文件内，无远程脚本、字体或图片；来源链接只用于核验。</footer>
</article>
</div>
</body>
</html>`;
fs.writeFileSync(outputPath, html, {encoding: 'utf8', mode: 0o644});
console.log(JSON.stringify({source: sourcePath, output: outputPath, chapters: toc.length, notes: notes.size, tables: tableIndex, images: imageIndex, characters: source.length}, null, 2));
