#!/usr/bin/env python3
"""Pre-commit verification for an HF Daily Papers digest (five checks).

Usage:
  uv run python3 scripts/hf_digest_verify.py <digest.md> <snapshot_dir>

<snapshot_dir> holds the per-day bucket JSON files this run fetched
(`YYYY-MM-DD.json`, HF daily_papers API output). Checks run against that
snapshot, never against a fresh fetch: buckets gain AND lose entries and
upvotes accumulate, so "passed" is a function of read time. The snapshot's
mtime range is printed with every result.

  ① every arXiv id cited is in the window (snapshot)
  ② each HF link label matches the paper's real title
     (decide "label is a bare id" BEFORE stripping a leading row number)
  ③ every referenced figure resolves; no unreferenced figures in the subfolder
  ④ overview-table upvotes match the snapshot (column located via header '▲')
  ⑤ every arXiv id cited is in references/references.bib
"""
import datetime as dt
import glob
import json
import os
import re
import sys

ID_RE = re.compile(r"\b(\d{4}\.\d{5})\b")
LINK_RE = re.compile(r"\[([^\]]+)\]\(https://huggingface\.co/papers/(\d{4}\.\d{5})\)")


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def main(md_path, snap):
    files = sorted(glob.glob(os.path.join(snap, "20??-??-??.json")))
    papers = {}
    for f in files:
        d = json.load(open(f))
        if not isinstance(d, list):
            continue
        for e in d:
            p = e["paper"]
            papers[p["id"]] = p
    mt = [os.path.getmtime(f) for f in files]
    fmt = lambda t: dt.datetime.utcfromtimestamp(t).strftime("%m-%d %H:%M")
    stamp = (f"snapshot {snap} ({len(files)} buckets, mtime {fmt(min(mt))}–{fmt(max(mt))} UTC)"
             if mt else f"snapshot {snap} (EMPTY)")
    now = dt.datetime.utcnow().strftime("%m-%d %H:%M")
    text = open(md_path, encoding="utf-8").read()

    cited = set(ID_RE.findall(text))
    miss = sorted(i for i in cited if i not in papers)
    print(f"① ids cited {len(cited)} not in window {miss}   [checked {now} UTC vs {stamp}]")

    bad = []
    for label, pid in LINK_RE.findall(text):
        lab = label.strip().strip("*")
        if pid not in papers or lab == pid or ID_RE.fullmatch(lab):
            continue  # bare-id labels are fine; check this BEFORE stripping row numbers
        lab2 = re.sub(r"^\d+\.\s*", "", lab)
        t = papers[pid]["title"]
        if not lab2.isascii():
            continue  # translated labels are checked by eye
        a, b = norm(lab2), norm(t)
        if a and a not in b and b not in a and a.split()[0] not in b:
            bad.append((pid, lab2, t))
    print(f"② References label↔title mismatches {len(bad)}")
    for x in bad:
        print("   ?", x)

    base = os.path.splitext(md_path)[0]
    refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text)
    d = os.path.dirname(md_path)
    unres = [r for r in refs if not os.path.exists(os.path.join(d, r))]
    present = set(os.listdir(base)) if os.path.isdir(base) else set()
    used = {os.path.basename(r) for r in refs}
    unref = sorted(present - used)
    print(f"③ figs referenced {len(refs)} unresolved {unres} unreferenced {unref}")

    rows, mism = 0, []
    up_col = None
    for line in text.splitlines():
        if not line.startswith("|"):
            up_col = None if not line.strip() else up_col
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if any(c == "▲" for c in cells):
            up_col = cells.index("▲")
            continue
        if up_col is None or set(line) <= set("|-: "):
            continue
        m = re.search(r"papers/(\d{4}\.\d{5})", line)
        if not m or up_col >= len(cells):
            continue
        try:
            v = int(re.sub(r"[^\d]", "", cells[up_col]))
        except ValueError:
            continue
        rows += 1
        pid = m.group(1)
        if pid in papers and papers[pid].get("upvotes") != v:
            mism.append((pid, v, papers[pid].get("upvotes")))
    print(f"④ table rows {rows} upvote mismatches {mism}   [vs {stamp}]")

    bib = open("references/references.bib", encoding="utf-8").read()
    notbib = sorted(i for i in cited if i not in bib)
    print(f"⑤ 正文引用 {len(cited)} 个 id、不在 references.bib 里 {len(notbib)} {notbib}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
