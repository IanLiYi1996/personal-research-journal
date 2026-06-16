#!/usr/bin/env python3
"""Generate a browsable topic index from a cleaned .bib file.

Buckets each entry into one or more topics by matching keyword patterns
against title + keywords + abstract, then emits a Docsify-friendly markdown
page (full-text searchable in-browser). Serves both the "knowledge base"
and "digest topic selection" use cases.

Usage:
    uv run python3 scripts/bib_index.py references.bib --out references/README.md
"""
from __future__ import annotations

import argparse
import re
import sys

# import the tolerant parser from the cleaning script
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from clean_bib import parse_bib, arxiv_id  # noqa: E402

# topic -> regex over the combined searchable text. Order = display order.
TOPICS: list[tuple[str, str]] = [
    ("大语言模型 (LLM)", r"\b(llm|large language model|gpt|qwen|llama|instruction tun)"),
    ("智能体 / Agent", r"\b(agent|multi-agent|tool use|react|planning)\b"),
    ("强化学习 / RLHF", r"\b(reinforcement learning|rlhf|rl |reward model|ppo|dpo|alignment)"),
    ("推理 / Reasoning", r"\b(reasoning|chain-of-thought|cot|inference-time|test-time)"),
    ("图神经网络 / Graph", r"\b(graph|gnn|node classification|knowledge graph)"),
    ("扩散 / 生成模型", r"\b(diffusion|generative|gan|vae|image generation|text-to-image)"),
    ("多模态 / 视觉语言", r"\b(multimodal|vision-language|vlm|image-text|video)"),
    ("Transformer / 架构", r"\b(transformer|attention|mamba|state space|mixture of experts|moe)"),
    ("高效训练 / 推理", r"\b(efficient|quantization|distillation|pruning|lora|peft|parallel)"),
    ("综述 / Survey", r"\b(survey|comprehensive review|overview)\b"),
    ("检索增强 / RAG", r"\b(retrieval|rag|retrieval-augmented)"),
]


def searchable(e: dict) -> str:
    return " ".join(
        e.get(f, "") for f in ("title", "keywords", "abstract")
    ).lower()


def fmt_entry(e: dict) -> str:
    title = e.get("title", "(无标题)")
    year = e.get("year", "n.d.")
    author = re.split(r"\s+and\s+", e.get("author", ""))[0] or "?"
    aid = arxiv_id(e)
    url = e.get("url", "")
    link = f"[{title}]({url})" if url else title
    extra = f" · arXiv:{aid}" if aid else ""
    return f"- **{year}** {link} — {author} et al.{extra} `{{{e['ID']}}}`"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default="references/README.md")
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8") as fh:
        entries = parse_bib(fh.read())

    buckets: dict[str, list[dict]] = {name: [] for name, _ in TOPICS}
    uncategorized: list[dict] = []
    for e in entries:
        text = searchable(e)
        hit = False
        for name, pat in TOPICS:
            if re.search(pat, text):
                buckets[name].append(e)
                hit = True
        if not hit:
            uncategorized.append(e)

    def year_key(e: dict) -> int:
        m = re.search(r"\d{4}", e.get("year", ""))
        return -int(m.group()) if m else 0

    lines = [
        "# 文献库索引",
        "",
        f"> 共 **{len(entries)}** 条，由 `scripts/bib_index.py` 从 `references.bib` 自动生成。",
        "> 一篇文献可能出现在多个主题下。引用 key 在行尾 `{...}` 中。",
        "",
        "## 主题导航",
        "",
    ]
    lines += [
        f"- [{name}](#{re.sub(r'[^a-z0-9]+', '-', name.lower())}) "
        f"（{len(buckets[name])}）"
        for name, _ in TOPICS
    ]
    lines.append(f"- 未分类（{len(uncategorized)}）")
    lines.append("")

    for name, _ in TOPICS:
        grp = sorted(buckets[name], key=year_key)
        lines.append(f"## {name}")
        lines.append("")
        lines += [fmt_entry(e) for e in grp] or ["_(空)_"]
        lines.append("")

    if uncategorized:
        lines.append("## 未分类")
        lines.append("")
        lines += [fmt_entry(e) for e in sorted(uncategorized, key=year_key)]
        lines.append("")

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"wrote {args.out}: {len(entries)} entries across {len(TOPICS)} topics, "
          f"{len(uncategorized)} uncategorized")
    return 0


if __name__ == "__main__":
    sys.exit(main())
