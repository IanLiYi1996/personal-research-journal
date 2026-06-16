#!/usr/bin/env python3
"""Add a paper to references/references.bib from an arXiv id/URL (or DOI).

Used by the "每次总结论文都加入文献库" rule: after writing a paper note or
digest entry, run this to register the paper in the reference library.

It fetches metadata from the arXiv API (verifiable source — never fabricated),
builds a cite key with the SAME scheme as clean_bib.py
(<LastName><Year><FirstTitleWord>, a/b/c on collision), skips it if the arXiv
id or normalized title is already present, appends it, and rebuilds the index.

Usage:
    uv run python3 scripts/add_paper.py 2505.10475
    uv run python3 scripts/add_paper.py https://arxiv.org/abs/2505.10475
    uv run python3 scripts/add_paper.py 2505.10475 2503.09089   # multiple
    uv run python3 scripts/add_paper.py 2505.10475 --no-index    # skip reindex
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from clean_bib import (  # noqa: E402
    parse_bib, write_bib, normalize_title, arxiv_id,
    first_author_last, first_title_word,
)

BIB = os.path.join(os.path.dirname(__file__), "..", "references", "references.bib")
API = "https://export.arxiv.org/api/query"


def extract_arxiv_id(token: str) -> str | None:
    m = re.search(r"(\d{4}\.\d{4,5})", token)
    return m.group(1) if m else None


def fetch_arxiv(aid: str) -> dict | None:
    """Fetch one arXiv entry and map it to a bib dict (no cite key yet)."""
    try:
        with urllib.request.urlopen(f"{API}?id_list={aid}", timeout=25) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception as exc:
        print(f"  ! fetch failed for {aid}: {exc}", file=sys.stderr)
        return None
    m = re.search(r"<entry>(.*?)</entry>", xml, re.S)
    if not m:
        return None
    e = m.group(1)

    def grab(tag: str) -> str:
        mm = re.search(rf"<{tag}>(.*?)</{tag}>", e, re.S)
        return " ".join(mm.group(1).split()) if mm else ""

    title = grab("title")
    if not title:
        return None
    authors = re.findall(r"<author>\s*<name>(.*?)</name>", e, re.S)
    year = (re.search(r"<published>(\d{4})", e) or [None, ""])[1]
    summary = grab("summary")
    return {
        "ENTRYTYPE": "article",
        "ID": "",  # assigned below
        "title": title,
        "author": " and ".join(a.strip() for a in authors),
        "year": year,
        "url": f"http://arxiv.org/abs/{aid}",
        "abstract": summary,
    }


def make_key(entry: dict, existing: set[str]) -> str:
    base = f"{first_author_last(entry)}{entry.get('year', 'nd')}{first_title_word(entry)}"
    if base not in existing:
        return base
    for i in range(26):
        cand = f"{base}{chr(ord('a') + i)}"
        if cand not in existing:
            return cand
    n = 0
    while f"{base}_{n}" in existing:
        n += 1
    return f"{base}_{n}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+", help="arXiv id(s) or URL(s)")
    ap.add_argument("--no-index", action="store_true",
                    help="don't rebuild references/README.md")
    args = ap.parse_args()

    bib_path = os.path.normpath(BIB)
    entries = parse_bib(open(bib_path, encoding="utf-8").read())
    existing_keys = {e["ID"] for e in entries}
    existing_arxiv = {arxiv_id(e) for e in entries if arxiv_id(e)}
    existing_titles = {normalize_title(e.get("title", "")) for e in entries}

    added, skipped = [], []
    for token in args.ids:
        aid = extract_arxiv_id(token)
        if not aid:
            skipped.append((token, "not an arXiv id/URL"))
            continue
        if aid in existing_arxiv:
            skipped.append((aid, "already in library (arXiv id)"))
            continue
        entry = fetch_arxiv(aid)
        if not entry:
            skipped.append((aid, "arXiv fetch returned nothing"))
            continue
        if normalize_title(entry["title"]) in existing_titles:
            skipped.append((aid, f"title already present: {entry['title'][:50]}"))
            continue
        entry["ID"] = make_key(entry, existing_keys)
        existing_keys.add(entry["ID"])
        existing_arxiv.add(aid)
        existing_titles.add(normalize_title(entry["title"]))
        entries.append(entry)
        added.append(entry)

    if added:
        entries.sort(key=lambda e: e["ID"].lower())
        with open(bib_path, "w", encoding="utf-8") as fh:
            fh.write(write_bib(entries))

    for e in added:
        print(f"+ {e['ID']:<26} {e['title'][:60]}")
    for tok, why in skipped:
        print(f"- skip {tok}: {why}")
    print(f"\nadded {len(added)}, skipped {len(skipped)}, "
          f"library now {len(entries)} entries")

    if added and not args.no_index:
        idx = os.path.join(os.path.dirname(__file__), "bib_index.py")
        subprocess.run(
            ["uv", "run", "python3", idx, bib_path,
             "--out", os.path.join(os.path.dirname(bib_path), "README.md")],
            check=False,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
