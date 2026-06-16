#!/usr/bin/env python3
"""Enrich incomplete entries in references.bib.

Two passes, both conservative (never fabricates identifiers):
  1. DROP entries whose title is clearly not a paper (junk patterns + an
     explicit drop-list of cite keys).
  2. For remaining entries lacking BOTH year and url, query the arXiv API by
     title and backfill url/year ONLY on a high-confidence title match
     (normalized token Jaccard >= THRESHOLD). Non-English / blog / slide
     titles simply won't match and are left untouched for manual review.

Verifiable-sources rule: we only attach a url that arXiv actually returns for
a near-exact title; we never guess an id from memory.

Usage:
    uv run python3 scripts/enrich_bib.py references/references.bib --apply
    (omit --apply for a dry run)
"""
from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from clean_bib import parse_bib, write_bib, normalize_title  # noqa: E402

THRESHOLD = 0.85
API = "https://export.arxiv.org/api/query"

# cite keys that are definitively not papers -> delete outright
DROP_KEYS = {
    "Abadi2020References",      # title literally "References"
    "AnonndContent",            # Content submission guide-cn
    "AnonndGcr",                # GCR Tech Summit Content Submission Guide
    "AnonndUntitled",           # title "?"
    "Anonnd交易技术前沿",
    "ShermanndAcknowledgements",
    "RothndHelmholtz",          # affiliation list fragment
    "LopezLirandElectronic",    # "Electronic copy available"
    "AnonndSome",               # "Some observations"
    "AnonndSpeaker",            # "Speaker information"
    "KirosndNips",              # filename fragment
}

# title patterns that mark junk regardless of key
JUNK_TITLE = re.compile(
    r"(submission guide|目录|copy available|acknowledgement|^\?$"
    r"|content submission|tech summit)",
    re.I,
)


def title_tokens(t: str) -> set[str]:
    return set(normalize_title(t).split())


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def arxiv_lookup(title: str) -> tuple[str, str] | None:
    """Return (url, year) for a high-confidence arXiv title match, else None."""
    q = re.sub(r"[^A-Za-z0-9 ]", " ", title)
    q = " ".join(q.split())
    if len(q) < 8:
        return None
    params = urllib.parse.urlencode(
        {"search_query": f'ti:"{q}"', "max_results": 3}
    )
    try:
        with urllib.request.urlopen(f"{API}?{params}", timeout=25) as r:
            xml = r.read().decode("utf-8", "replace")
    except Exception:
        return None
    want = title_tokens(title)
    best = None
    for m in re.finditer(r"<entry>(.*?)</entry>", xml, re.S):
        e = m.group(1)
        idm = re.search(r"<id>(.*?)</id>", e)
        tm = re.search(r"<title>(.*?)</title>", e, re.S)
        pm = re.search(r"<published>(\d{4})", e)
        if not (idm and tm):
            continue
        score = jaccard(want, title_tokens(tm.group(1)))
        cand = (score, idm.group(1).strip(), pm.group(1) if pm else "")
        if best is None or cand[0] > best[0]:
            best = cand
    if best and best[0] >= THRESHOLD:
        url = re.sub(r"v\d+$", "", best[1])  # strip version suffix
        return url, best[2]
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--apply", action="store_true", help="write changes back")
    args = ap.parse_args()

    entries = parse_bib(open(args.infile, encoding="utf-8").read())
    n0 = len(entries)

    # pass 1: drop junk
    kept, dropped = [], []
    for e in entries:
        if e["ID"] in DROP_KEYS or JUNK_TITLE.search(e.get("title", "")):
            dropped.append(e)
        else:
            kept.append(e)

    # pass 2: enrich entries missing year+url
    targets = [e for e in kept if not e.get("year") and not e.get("url")]
    enriched = []
    print(f"querying arXiv for {len(targets)} incomplete entries...",
          file=sys.stderr)
    for i, e in enumerate(targets):
        res = arxiv_lookup(e.get("title", ""))
        if res:
            url, year = res
            e["url"] = url
            if year:
                e["year"] = year
            enriched.append((e["ID"], e.get("title", "")[:50], url))
        time.sleep(3.1)  # arXiv API courtesy rate limit
        if (i + 1) % 10 == 0:
            print(f"  ...{i+1}/{len(targets)}", file=sys.stderr)

    print(f"\nentries: {n0} -> {len(kept)} (dropped {len(dropped)} junk)")
    print(f"enriched with arXiv url/year: {len(enriched)}")
    for k, t, u in enriched:
        print(f"  + {k:<26} {u}  ({t})")
    print("\ndropped:")
    for e in dropped:
        print(f"  - {e['ID']:<26} {e.get('title','?')[:55]}")

    if args.apply:
        # regenerate keys could shift; keep existing keys, just resort
        kept.sort(key=lambda e: e["ID"].lower())
        with open(args.infile, "w", encoding="utf-8") as fh:
            fh.write(write_bib(kept))
        print(f"\nWROTE {args.infile} ({len(kept)} entries)")
    else:
        print("\n(dry run — re-run with --apply to write)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
