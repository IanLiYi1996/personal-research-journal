#!/usr/bin/env python3
"""Build a knowledge graph over this journal's notes and surface insights.

Method borrowed from nashsu/llm_wiki (which implements Karpathy's LLM Wiki
pattern): a 4-signal relevance model, Louvain community detection, and
"graph insights" (surprising connections + knowledge gaps).

Unlike llm_wiki (a desktop app that builds its own wiki store), this script
reads the repo *in place* — the existing markdown notes plus
references/references.bib — and writes a report. Nothing is mutated.

Usage:
  uv run --with networkx python3 scripts/wiki_graph.py
  uv run --with networkx python3 scripts/wiki_graph.py --out weekly/knowledge-graph.md

Signals (weights follow llm_wiki's model):
  direct link    x3.0   note -> note markdown/wikilink
  source overlap x4.0   two notes citing the same arXiv id / cite key
  Adamic-Adar    x1.5   shared neighbours, damped by neighbour degree
  type affinity  x1.0   same folder/type bonus
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTE_DIRS = ("research-notes", "papers", "topics", "tech-blogs", "weekly", "reddit-digests")
SKIP_NAMES = {"README.md", "_template.md", "_sidebar.md", "_home.md"}

W_LINK, W_SOURCE, W_AA, W_TYPE = 3.0, 4.0, 1.5, 1.0

ARXIV_RE = re.compile(r"\b(\d{4}\.\d{4,5})\b")
CITEKEY_RE = re.compile(r"`([A-Z][A-Za-z0-9]*\d{4}[A-Za-z][A-Za-z0-9]*)`")
MDLINK_RE = re.compile(r"\]\(\s*/?((?:research-notes|papers|topics|tech-blogs|weekly|reddit-digests)/[^)\s#]+?\.md)")
# any markdown link ending in .md (resolved relative to the citing note's folder)
ANYLINK_RE = re.compile(r"\]\(\s*([^)\s#]+?\.md)")
WIKILINK_RE = re.compile(r"\[\[([^\]|]+)")


def collect_notes() -> dict[str, dict]:
    """Map repo-relative path -> note record."""
    notes: dict[str, dict] = {}
    for d in NOTE_DIRS:
        base = os.path.join(ROOT, d)
        if not os.path.isdir(base):
            continue
        for dirpath, _dirs, files in os.walk(base):
            for fn in files:
                if not fn.endswith(".md") or fn in SKIP_NAMES:
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, ROOT)
                try:
                    text = open(full, encoding="utf-8").read()
                except Exception:
                    continue
                title = ""
                for line in text.splitlines():
                    if line.startswith("# "):
                        title = line[2:].strip()
                        break
                notes[rel] = {
                    "rel": rel,
                    "type": d,
                    "title": title or os.path.splitext(fn)[0],
                    "stem": os.path.splitext(fn)[0],
                    "text": text,
                    "arxiv": set(ARXIV_RE.findall(text)),
                    "citekeys": set(CITEKEY_RE.findall(text)),
                    "links": set(),
                }
    # resolve outgoing links: repo-absolute paths, same-dir/relative paths, wikilinks
    by_stem = {n["stem"]: rel for rel, n in notes.items()}
    for rel, n in notes.items():
        here = os.path.dirname(rel)
        # (a) paths rooted at a known note dir, e.g. ](/research-notes/x.md)
        for tgt in MDLINK_RE.findall(n["text"]):
            tgt = os.path.normpath(tgt)
            if tgt in notes and tgt != rel:
                n["links"].add(tgt)
        # (b) any other .md link resolved relative to this note's own folder,
        #     e.g. ](2026-02-09-llm-intro-architecture.md) or ](../papers/x.md)
        for raw in ANYLINK_RE.findall(n["text"]):
            if raw.startswith(("http://", "https://", "#")):
                continue
            cand = os.path.normpath(os.path.join(here, raw.lstrip("/")) if not raw.startswith("/")
                                    else raw.lstrip("/"))
            if cand in notes and cand != rel:
                n["links"].add(cand)
        for wl in WIKILINK_RE.findall(n["text"]):
            stem = os.path.splitext(wl.strip())[0]
            tgt = by_stem.get(stem)
            if tgt and tgt != rel:
                n["links"].add(tgt)
    return notes


def build_graph(notes: dict[str, dict]):
    import networkx as nx

    G = nx.Graph()
    for rel, n in notes.items():
        G.add_node(rel, title=n["title"], type=n["type"])

    # pass 1: direct links + source overlap (these define adjacency)
    detail: dict[tuple[str, str], dict] = collections.defaultdict(dict)
    for rel, n in notes.items():
        for tgt in n["links"]:
            key = tuple(sorted((rel, tgt)))
            detail[key]["link"] = detail[key].get("link", 0) + 1

    rels = sorted(notes)
    for i, a in enumerate(rels):
        for b in rels[i + 1:]:
            shared = (notes[a]["arxiv"] & notes[b]["arxiv"]) | (
                notes[a]["citekeys"] & notes[b]["citekeys"])
            if shared:
                key = (a, b)
                detail[key]["shared"] = len(shared)

    for (a, b), d in detail.items():
        w = 0.0
        if d.get("link"):
            w += W_LINK
        if d.get("shared"):
            # damp: many shared refs shouldn't dominate everything
            w += W_SOURCE * min(1.0, math.log1p(d["shared"]) / math.log(6))
        if notes[a]["type"] == notes[b]["type"]:
            w += W_TYPE
        G.add_edge(a, b, weight=w, **d)

    # pass 2: Adamic-Adar on the adjacency built above
    for a, b in list(G.edges()):
        common = set(G[a]) & set(G[b])
        aa = sum(1.0 / math.log(G.degree(c)) for c in common if G.degree(c) > 1)
        if aa:
            G[a][b]["weight"] += W_AA * min(aa, 3.0)
            G[a][b]["aa"] = round(aa, 2)
    return G


def analyse(G, notes):
    import networkx as nx
    try:
        from networkx.algorithms.community import louvain_communities
        comms = louvain_communities(G, weight="weight", seed=7)
    except Exception:
        comms = [set(c) for c in nx.connected_components(G)]

    comm_of = {n: i for i, c in enumerate(comms) for n in c}
    out = {"communities": [], "isolated": [], "bridges": [], "surprising": []}

    for i, c in enumerate(comms):
        sub = G.subgraph(c)
        n = len(c)
        possible = n * (n - 1) / 2
        cohesion = (sub.number_of_edges() / possible) if possible else 0.0
        top = sorted(c, key=lambda x: -G.degree(x))[:3]
        types = collections.Counter(notes[x]["type"] for x in c)
        out["communities"].append({
            "id": i, "size": n, "cohesion": round(cohesion, 3),
            "top": top, "types": dict(types),
            "sparse": cohesion < 0.15 and n >= 3,
        })

    for rel in G.nodes():
        if G.degree(rel) <= 1:
            out["isolated"].append({"rel": rel, "degree": G.degree(rel),
                                    "title": notes[rel]["title"]})

    for rel in G.nodes():
        nb_comms = {comm_of[x] for x in G[rel]}
        if len(nb_comms) >= 3:
            out["bridges"].append({"rel": rel, "clusters": len(nb_comms),
                                   "degree": G.degree(rel),
                                   "title": notes[rel]["title"]})
    out["bridges"].sort(key=lambda d: -d["clusters"])

    for a, b, d in G.edges(data=True):
        cross_comm = comm_of[a] != comm_of[b]
        cross_type = notes[a]["type"] != notes[b]["type"]
        da, db = G.degree(a), G.degree(b)
        periph_hub = (min(da, db) <= 2 and max(da, db) >= 8)
        if not (cross_comm or (cross_type and d.get("shared"))):
            continue
        score = (2.0 if cross_comm else 0) + (1.0 if cross_type else 0) + \
                (1.5 if periph_hub else 0) + d["weight"] / 10
        out["surprising"].append({
            "a": a, "b": b, "score": round(score, 2), "weight": round(d["weight"], 2),
            "shared": d.get("shared", 0), "cross_comm": cross_comm,
            "cross_type": cross_type,
        })
    out["surprising"].sort(key=lambda d: -d["score"])
    return out, comms, comm_of


def render(G, notes, res, comms) -> str:
    L = []
    A = L.append
    A("# 知识图谱洞察报告（自动生成）\n")
    A(f"- **生成方式**: `scripts/wiki_graph.py`（方法参考 [nashsu/llm_wiki](https://github.com/nashsu/llm_wiki)"
      " 的 4 信号相关性模型 + Louvain 社区发现 + Graph Insights）")
    A(f"- **节点**: {G.number_of_nodes()} 篇笔记　**边**: {G.number_of_edges()}　"
      f"**社区**: {len(comms)}")
    A("- **信号权重**: 直接链接 ×3.0 / 共享文献来源 ×4.0 / Adamic-Adar ×1.5 / 同类型 ×1.0\n")
    A("> 本报告由脚本读取仓库现有 md 与 `references.bib` 生成，不修改任何笔记。\n")

    A("## 1. 主题社区（Louvain 自动聚类）\n")
    A("| # | 规模 | 内聚度 | 构成 | 核心笔记（按连接数） |")
    A("|---|---|---|---|---|")
    for c in sorted(res["communities"], key=lambda d: -d["size"]):
        flag = " ⚠️稀疏" if c["sparse"] else ""
        types = ", ".join(f"{k}×{v}" for k, v in sorted(c["types"].items(), key=lambda x: -x[1]))
        tops = "<br>".join(f"`{os.path.basename(t)}`" for t in c["top"])
        A(f"| {c['id']} | {c['size']} | {c['cohesion']}{flag} | {types} | {tops} |")
    A("")

    A("## 2. 桥节点（连接 ≥3 个社区的枢纽笔记）\n")
    if res["bridges"]:
        A("| 笔记 | 跨社区数 | 连接数 | 标题 |")
        A("|---|---|---|---|")
        for b in res["bridges"][:12]:
            A(f"| `{b['rel']}` | {b['clusters']} | {b['degree']} | {b['title'][:44]} |")
    else:
        A("_无_")
    A("")

    A("## 3. 意外连接（跨社区 / 跨类型的强关联）\n")
    if res["surprising"]:
        A("| 分数 | 笔记 A | 笔记 B | 权重 | 共享文献 | 类型 |")
        A("|---|---|---|---|---|---|")
        for s in res["surprising"][:15]:
            tag = []
            if s["cross_comm"]:
                tag.append("跨社区")
            if s["cross_type"]:
                tag.append("跨类型")
            A(f"| {s['score']} | `{os.path.basename(s['a'])}` | `{os.path.basename(s['b'])}` "
              f"| {s['weight']} | {s['shared']} | {'+'.join(tag)} |")
    else:
        A("_无_")
    A("")

    A("## 4. 知识缺口\n")
    A(f"### 4.1 孤立笔记（连接数 ≤1，共 {len(res['isolated'])} 篇）\n")
    if res["isolated"]:
        for i in sorted(res["isolated"], key=lambda d: (d["degree"], d["rel"]))[:25]:
            A(f"- `{i['rel']}`（degree={i['degree']}）— {i['title'][:56]}")
    else:
        A("_无_")
    sparse = [c for c in res["communities"] if c["sparse"]]
    A(f"\n### 4.2 稀疏社区（内聚度 <0.15 且 ≥3 篇，共 {len(sparse)} 个）\n")
    if sparse:
        for c in sparse:
            A(f"- 社区 {c['id']}（{c['size']} 篇，内聚度 {c['cohesion']}）"
              f"核心：`{os.path.basename(c['top'][0])}`")
    else:
        A("_无_")
    A("")
    return "\n".join(L)


def export_json(G, notes, comm_of, res, dest: str) -> None:
    """Emit graph data for the in-site interactive viewer (graph.html)."""
    import json

    bridges = {b["rel"] for b in res["bridges"]}
    nodes = []
    for rel in G.nodes():
        n = notes[rel]
        nodes.append({
            "id": rel,
            "label": n["title"][:70] or n["stem"],
            "type": n["type"],
            "community": comm_of.get(rel, -1),
            "degree": G.degree(rel),
            # Docsify route for this note (drop .md, site uses hash routing)
            "route": "#/" + rel[:-3] if rel.endswith(".md") else "#/" + rel,
            "bridge": rel in bridges,
        })
    edges = [{
        "source": a, "target": b,
        "weight": round(d["weight"], 2),
        "shared": d.get("shared", 0),
        "link": bool(d.get("link")),
    } for a, b, d in G.edges(data=True)]

    payload = {
        "generated_by": "scripts/wiki_graph.py",
        "method": "nashsu/llm_wiki 4-signal relevance + Louvain",
        "stats": {"nodes": len(nodes), "edges": len(edges),
                  "communities": len({n['community'] for n in nodes})},
        "nodes": nodes, "edges": edges,
    }
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w", encoding="utf-8").write(json.dumps(payload, ensure_ascii=False))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="weekly/knowledge-graph.md",
                    help="report path relative to repo root (default weekly/knowledge-graph.md)")
    ap.add_argument("--json", default="assets/knowledge-graph.json",
                    help="graph data for the interactive viewer (default assets/knowledge-graph.json)")
    ap.add_argument("--print", action="store_true", help="print report to stdout instead of writing")
    args = ap.parse_args()

    notes = collect_notes()
    if not notes:
        print("no notes found", file=sys.stderr)
        return 1
    G = build_graph(notes)
    res, comms, comm_of = analyse(G, notes)
    report = render(G, notes, res, comms)

    if args.print:
        print(report)
    else:
        dest = os.path.join(ROOT, args.out)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        open(dest, "w", encoding="utf-8").write(report + "\n")
        print(f"wrote {args.out}: {G.number_of_nodes()} notes, "
              f"{G.number_of_edges()} edges, {len(comms)} communities, "
              f"{len(res['isolated'])} isolated, {len(res['bridges'])} bridges")
    if args.json:
        export_json(G, notes, comm_of, res, os.path.join(ROOT, args.json))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
