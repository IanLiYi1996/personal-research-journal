#!/usr/bin/env python3
"""Sitemap-based fallback fetcher for blogs without RSS/Atom.

For sources whose RSS endpoint is null in feeds.yaml, this fetches the
site's sitemap.xml, filters URLs matching a path prefix (e.g. "/news/",
"/blog/", "/research/"), keeps entries with lastmod within the window,
and prints them in the same JSON schema as blog_fetch.py.

Output schema (per post):
  {
    "source":  "<feeds.yaml name>",
    "tier":    "...",
    "topics":  [...],
    "title":   "<slug-derived, will need page-fetch to improve>",
    "link":    "...",
    "pub":     "ISO-8601 lastmod",
    "summary": ""
  }

Configuration is via SITEMAP_RULES below, keyed by feeds.yaml `name`.
Sources not listed are silently skipped.

Usage:
  uv run python3 scripts/blog_sitemap_fallback.py --days 7 > /tmp/sitemap.json
  uv run python3 scripts/blog_sitemap_fallback.py --days 30 --names "Anthropic"
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML not available. Install with: uv add pyyaml\n")
    sys.exit(1)

UA = "Mozilla/5.0 (research-journal-blog-digest/0.1; weekly; respectful)"
FEEDS_FILE = Path(__file__).parent.parent / "tech-blogs" / "feeds.yaml"

# Map feeds.yaml `name` -> (sitemap URL, list-of-allowed-path-prefixes)
# Path prefixes are matched against URL pathname; an entry is kept iff its
# pathname starts with at least one prefix. Use "/" to allow everything.
SITEMAP_RULES = {
    "Anthropic News": (
        "https://www.anthropic.com/sitemap.xml",
        ["/news/"],
    ),
    "Anthropic Research": (
        "https://www.anthropic.com/sitemap.xml",
        ["/research/"],
    ),
    "Cohere Blog": (
        "https://cohere.com/sitemap.xml",
        ["/blog/"],
    ),
    # Stanford HAI: sitemap lastmod is the page re-render timestamp (all 529
    # /news/ entries share the same lastmod after a site build). Not useful
    # for "new posts" filtering. Leave disabled until we find pubDate source.
    # "Stanford HAI News": (
    #     "https://hai.stanford.edu/sitemap.xml",
    #     ["/news/"],
    # ),
    "AI2 (Allen AI) Blog": (
        "https://allenai.org/sitemap.xml",
        ["/blog/", "/papers/"],
    ),
    "Apollo Research": (
        "https://www.apolloresearch.ai/sitemap.xml",
        ["/blog/", "/research/"],
    ),
    # METR has both RSS and sitemap; sitemap covers more. Use only when needed.
    # "METR Blog": ("https://metr.org/sitemap.xml", ["/blog/"]),
}


def fetch(url: str, max_retries: int = 3, base_delay: int = 4) -> str | None:
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/xml,text/xml,*/*"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (2 ** attempt)
                sys.stderr.write(f"  WARN 429 backoff {wait}s\n")
                time.sleep(wait)
                continue
            sys.stderr.write(f"  HTTP {e.code}\n")
            return None
        except Exception as e:
            sys.stderr.write(f"  ERR {e}\n")
            time.sleep(base_delay * (2 ** attempt))
    return None


def parse_sitemap(xml: str) -> list[tuple[str, datetime | None]]:
    """Returns [(loc_url, lastmod_dt_or_None), ...]."""
    out = []
    for m in re.finditer(r"<url>(.*?)</url>", xml, re.S):
        chunk = m.group(1)
        loc = re.search(r"<loc>([^<]+)</loc>", chunk)
        lastmod = re.search(r"<lastmod>([^<]+)</lastmod>", chunk)
        if not loc:
            continue
        dt = None
        if lastmod:
            raw = lastmod.group(1).strip()
            # ISO-8601 variants
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
                try:
                    d = datetime.strptime(raw, fmt)
                    if d.tzinfo is None:
                        d = d.replace(tzinfo=timezone.utc)
                    dt = d.astimezone(timezone.utc)
                    break
                except Exception:
                    continue
        out.append((loc.group(1).strip(), dt))
    return out


def slug_to_title(url: str) -> str:
    """Derive a readable title from the URL slug as fallback."""
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    title = slug.replace("-", " ").replace("_", " ").strip()
    # Title-case but preserve known acronyms
    words = title.split()
    out = []
    for w in words:
        if w.isupper() and len(w) <= 4:
            out.append(w)
        else:
            out.append(w.capitalize())
    return " ".join(out) if out else url


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--since", help="YYYY-MM-DD override")
    parser.add_argument("--names", help="Comma substring filter")
    args = parser.parse_args()

    cfg = yaml.safe_load(FEEDS_FILE.read_text())
    feeds_by_name = {f["name"]: f for f in cfg["feeds"]}

    cutoff = (
        datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
        if args.since
        else datetime.now(timezone.utc) - timedelta(days=args.days)
    )

    name_filter = [s.strip().lower() for s in args.names.split(",")] if args.names else None

    all_posts: list[dict] = []
    for src_name, (sitemap_url, prefixes) in SITEMAP_RULES.items():
        feed_cfg = feeds_by_name.get(src_name)
        if not feed_cfg:
            sys.stderr.write(f"WARN: {src_name} not in feeds.yaml — skipping\n")
            continue
        if name_filter and not any(n in src_name.lower() for n in name_filter):
            continue
        sys.stderr.write(f"SITEMAP {src_name}: ")
        xml = fetch(sitemap_url)
        if not xml:
            sys.stderr.write("FAILED\n")
            continue
        entries = parse_sitemap(xml)
        kept = 0
        for loc, lastmod in entries:
            if not lastmod or lastmod < cutoff:
                continue
            # Match path prefix
            try:
                path = "/" + loc.split("//", 1)[1].split("/", 1)[1]
            except IndexError:
                continue
            if not any(path.startswith(p) for p in prefixes):
                continue
            all_posts.append({
                "source": src_name,
                "tier": feed_cfg["tier"],
                "topics": feed_cfg.get("topics", []),
                "title": slug_to_title(loc),
                "link": loc,
                "pub": lastmod.isoformat(),
                "summary": "",
            })
            kept += 1
        sys.stderr.write(f"{kept} entries since {cutoff.date()}\n")

    all_posts.sort(key=lambda p: p["pub"], reverse=True)
    sys.stderr.write(f"\nDone: {len(all_posts)} sitemap posts since {cutoff.isoformat()}\n")
    print(json.dumps(all_posts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
