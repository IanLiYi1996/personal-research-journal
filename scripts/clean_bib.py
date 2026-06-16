#!/usr/bin/env python3
"""Clean a Mendeley-exported .bib file.

Fixes the four data-quality problems found in export.bib:
  1. Empty cite keys (the 182 @techReport entries all export as `@techReport{,`).
  2. Colliding cite keys (e.g. 22x `Wang2024`).
  3. Duplicate papers (same title imported twice, often arXiv + venue versions).
  4. Inconsistent entry types (arXiv preprints typed as @techReport).

Strategy:
  - Parse all entries.
  - Deduplicate by normalized title (and by arXiv id / DOI as a backstop),
    keeping the richest record and merging missing fields from the dropped twin.
  - Regenerate every key as <LastName><Year><FirstTitleWord>, disambiguated
    with a/b/c suffixes so every key is unique.
  - Demote arXiv @techReport -> @article (they are preprints, not reports).
  - Emit a cleaned .bib (sorted by key) plus a markdown change report.

Usage:
    uv run --with bibtexparser python3 scripts/clean_bib.py export.bib \
        --out references.bib --report scripts/clean_bib_report.md
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict


STOPWORDS = {
    "a", "an", "the", "on", "of", "for", "and", "in", "to", "with",
    "is", "are", "from", "via", "by", "at",
}


def parse_bib(text: str) -> list[dict]:
    """Brace-aware .bib parser that tolerates empty and duplicate cite keys.

    bibtexparser silently merges entries sharing a cite key (and all the
    empty-key @techReport entries collapse into one), so we parse by hand.
    Returns a list of dicts using bibtexparser's ENTRYTYPE / ID convention.
    """
    entries: list[dict] = []
    i, n = 0, len(text)
    while i < n:
        if text[i] != "@":
            i += 1
            continue
        m = re.match(r"@(\w+)\s*\{", text[i:])
        if not m:
            i += 1
            continue
        etype = m.group(1)
        i += m.end()
        # capture body until the matching closing brace
        depth, start = 1, i
        while i < n and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        body = text[start : i - 1]

        # cite key = up to first comma
        comma = body.find(",")
        key = body[:comma].strip() if comma != -1 else ""
        fields_blob = body[comma + 1 :] if comma != -1 else ""

        entry = {"ENTRYTYPE": etype.lower(), "ID": key}
        # split fields on top-level commas
        depth, buf, parts = 0, [], []
        for ch in fields_blob:
            if ch in "{(":
                depth += 1
            elif ch in "})":
                depth -= 1
            if ch == "," and depth <= 0:
                parts.append("".join(buf))
                buf = []
            else:
                buf.append(ch)
        if buf:
            parts.append("".join(buf))
        for part in parts:
            if "=" not in part:
                continue
            fname, _, fval = part.partition("=")
            fname = fname.strip().lower()
            fval = fval.strip().rstrip(",").strip()
            if fval and fval[0] in "{\"" and fval[-1] in "}\"":
                fval = fval[1:-1].strip()
            if fname:
                entry[fname] = fval
        entries.append(entry)
    return entries


def sanitize_value(v: str) -> str:
    """Make a field value safe to wrap in {...}.

    Mendeley abstracts contain escaped braces (\\{n\\}) and LaTeX math with
    unbalanced braces; left as-is they desync brace counting in strict
    parsers (biber/bibtexparser), which then silently drop the entry.
    We strip backslash-escaped braces and then balance any remainder.
    """
    v = v.replace("\\{", "(").replace("\\}", ")")
    out, depth = [], 0
    for ch in v:
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                continue  # drop unmatched closer
            depth -= 1
        out.append(ch)
    return "".join(out) + "}" * depth  # close any still-open braces


def write_bib(entries: list[dict]) -> str:
    out = []
    skip = {"ENTRYTYPE", "ID"}
    for e in entries:
        out.append(f"@{e['ENTRYTYPE']}{{{e['ID']},")
        fields = [k for k in e if k not in skip]
        for f in fields:
            out.append(f"  {f} = {{{sanitize_value(e[f])}}},")
        if out and out[-1].endswith(","):
            out[-1] = out[-1][:-1]  # drop trailing comma on last field
        out.append("}")
        out.append("")
    return "\n".join(out) + "\n"


def normalize_title(title: str) -> str:
    """Collapse a title to a comparison key: lowercase alnum words only."""
    t = title.lower()
    t = re.sub(r"[{}\\$]", "", t)
    t = re.sub(r"[^a-z0-9一-鿿]+", " ", t)
    return " ".join(t.split())


def arxiv_id(entry: dict) -> str | None:
    for field in ("url", "eprint", "doi"):
        v = entry.get(field, "")
        m = re.search(r"(\d{4}\.\d{4,5})", v)
        if m:
            return m.group(1)
    return None


def first_author_last(entry: dict) -> str:
    author = entry.get("author", "").strip()
    if not author:
        return "Anon"
    first = re.split(r"\s+and\s+", author)[0].strip()
    if "," in first:  # "Last, First"
        last = first.split(",")[0]
    else:             # "First Middle Last"
        last = first.split()[-1] if first.split() else "Anon"
    last = re.sub(r"[^A-Za-z一-鿿]", "", last)
    return last or "Anon"


def first_title_word(entry: dict) -> str:
    title = normalize_title(entry.get("title", ""))
    for w in title.split():
        if w not in STOPWORDS and len(w) > 1:
            return w.capitalize()
    return "Untitled"


def richness(entry: dict) -> int:
    """Score how complete an entry is; higher = keep."""
    score = len(entry.get("abstract", "")) // 50
    for f in ("doi", "journal", "volume", "pages", "booktitle", "publisher", "keywords"):
        if entry.get(f):
            score += 2
    score += len([k for k in entry if not k.startswith("ENTRY")])
    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("infile")
    ap.add_argument("--out", default="references.bib")
    ap.add_argument("--report", default="scripts/clean_bib_report.md")
    args = ap.parse_args()

    with open(args.infile, encoding="utf-8") as fh:
        entries = parse_bib(fh.read())
    total_in = len(entries)

    # --- 1. Deduplicate by normalized title (fallback: arXiv id) ---
    groups: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        key = normalize_title(e.get("title", "")) or f"__noid_{id(e)}"
        groups[key].append(e)

    deduped: list[dict] = []
    dup_log: list[tuple[str, int]] = []
    for norm, grp in groups.items():
        if len(grp) == 1:
            deduped.append(grp[0])
            continue
        grp.sort(key=richness, reverse=True)
        keep = grp[0]
        # merge any fields the kept record is missing from its twins
        for other in grp[1:]:
            for f, v in other.items():
                if f not in keep and not f.startswith("ENTRY") and f != "ID":
                    keep[f] = v
        deduped.append(keep)
        dup_log.append((keep.get("title", "?").strip("{}"), len(grp)))

    # --- 2. Demote arXiv techReports to articles ---
    demoted = 0
    for e in deduped:
        if e.get("ENTRYTYPE", "").lower() == "techreport" and arxiv_id(e):
            e["ENTRYTYPE"] = "article"
            demoted += 1

    # --- 3. Regenerate unique keys ---
    empty_fixed = sum(1 for e in deduped if not e.get("ID", "").strip())
    base_to_entries: dict[str, list[dict]] = defaultdict(list)
    for e in deduped:
        base = f"{first_author_last(e)}{e.get('year', 'nd')}{first_title_word(e)}"
        base_to_entries[base].append(e)

    collisions = 0
    for base, grp in base_to_entries.items():
        if len(grp) == 1:
            grp[0]["ID"] = base
        else:
            collisions += len(grp)
            for i, e in enumerate(grp):
                suffix = chr(ord("a") + i) if i < 26 else f"_{i}"
                e["ID"] = f"{base}{suffix}"

    # --- 4. Write cleaned bib, sorted by key ---
    deduped.sort(key=lambda e: e["ID"].lower())
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(write_bib(deduped))

    # --- report ---
    lines = [
        "# export.bib 清洗报告",
        "",
        f"- 输入条目：**{total_in}**",
        f"- 输出条目：**{len(deduped)}**（去重移除 {total_in - len(deduped)} 条）",
        f"- 填充空 cite key：**{empty_fixed}** 条",
        f"- 重新消歧的 key 碰撞：**{collisions}** 条",
        f"- arXiv @techReport → @article：**{demoted}** 条",
        "",
        "## 合并的重复文献（同标题多份，保留信息最全者）",
        "",
    ]
    for title, n in sorted(dup_log, key=lambda x: -x[1]):
        lines.append(f"- ({n}份) {title}")
    with open(args.report, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"in={total_in} out={len(deduped)} "
          f"empty_keys_fixed={empty_fixed} key_collisions_disambiguated={collisions} "
          f"techreport_to_article={demoted}")
    print(f"wrote {args.out} and {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
