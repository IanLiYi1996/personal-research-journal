import fs from 'node:fs';
import path from 'node:path';
import {execFileSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';

const sourcePath = path.resolve(process.argv[2] ?? '');
if (!sourcePath.endsWith('.source.md')) throw new Error('Pass the editable .source.md report.');
const outputPath = sourcePath.replace(/\.source\.md$/, '.md');
const marker = '<!-- standalone-research-generated-v1 -->';
if (fs.existsSync(outputPath) && !fs.readFileSync(outputPath, 'utf8').includes(marker)) {
  throw new Error('Refusing to overwrite a non-generated Markdown file.');
}
const source = fs.readFileSync(sourcePath, 'utf8');
const definitions = [];
let figure = 0;
const markdown = source.replace(/!\[([^\]]+)\]\(([^)\s]+\.svg)\)/g, (_, alt, href) => {
  const asset = path.resolve(path.dirname(sourcePath), href.replace(/\.svg$/, '.png'));
  if (!asset.startsWith(path.dirname(sourcePath) + path.sep)) throw new Error('Image must be a report-local generated asset.');
  const png = fs.readFileSync(asset);
  if (png.subarray(0, 8).toString('hex') !== '89504e470d0a1a0a') throw new Error(`Not a PNG: ${asset}`);
  const id = `figure-${++figure}`;
  definitions.push(`[${id}]: data:image/png;base64,${png.toString('base64')}`);
  return `![${alt}][${id}]`;
});
if (figure !== 5) throw new Error(`Expected five figures, got ${figure}`);
if (/\]\((?!https?:|#)[^)]+\)/.test(markdown)) {
  throw new Error('The standalone report still has a local document/attachment link.');
}
fs.writeFileSync(outputPath, markdown.trimEnd() + '\n\n' + marker + '\n\n' + definitions.join('\n') + '\n');
const renderer = path.join(path.dirname(fileURLToPath(import.meta.url)), 'render-research-reader.mjs');
execFileSync('node', [renderer, outputPath], {stdio: 'inherit'});
console.log(JSON.stringify({markdown: outputPath, embeddedPngs: figure, bytes: fs.statSync(outputPath).size}, null, 2));
