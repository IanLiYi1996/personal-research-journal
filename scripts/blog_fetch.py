#!/usr/bin/env python3
"""Fetch recent posts from a curated set of tech blogs (RSS / Atom).

Reads tech-blogs/feeds.yaml, fetches each feed (skips entries with feed: null),
filters by pubDate within the last N days (default 7), de-duplicates by link,
and writes a flat JSON array to stdout.

Output schema (per post):
  {
    "source": "<feeds.yaml name>",
    "tier":   "personal|company|academia",
    "topics": ["..."],
    "title":  "...",
    "link":   "...",
    "pub":    "ISO-8601 timestamp",
    "summary":"... (description / first paragraph, HTML-stripped)"
  }

Usage:
  uv run python3 scripts/blog_fetch.py                            # last 7 days
  uv run python3 scripts/blog_fetch.py --days 14
  uv run python3 scripts/blog_fetch.py --since 2026-06-20
  uv run python3 scripts/blog_fetch.py --names "Lilian Weng,Karpathy"
  uv run python3 scripts/blog_fetch.py --tier personal

Politeness:
  - Sequential fetch with default 1.5s delay between feeds
  - Retries 429 with exponential backoff
  - Writes a per-feed warning to stderr on failure but continues

Notes:
  - Entries with feed: null are skipped silently; report them with --list-missing
    to see what would need a sitemap/HTML fallback.
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from html import unescape
from pathlib import Path

try:
    import yaml  # PyYAML
except ImportError:
    sys.stderr.write("ERROR: PyYAML not available. Install with: uv add pyyaml\n")
    sys.exit(1)

UA = "Mozilla/5.0 (research-journal-blog-digest/0.1; weekly; respectful)"
FEEDS_FILE = Path(__file__).parent.parent / "tech-blogs" / "feeds.yaml"


def fetch_feed(url: str, max_retries: int = 4, base_delay: int = 5) -> str | None:
    for attempt in range(max_retries):
        req = urllib.request.Request(
            url,
            headers={"User-Agent": UA, "Accept": "application/atom+xml,application/rss+xml,application/xml,text/xml,*/*"},
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (2 ** attempt)
                sys.stderr.write(f"  WARN 429, backoff {wait}s\n")
                time.sleep(wait)
                continue
            sys.stderr.write(f"  HTTP {e.code}\n")
            return None
        except Exception as e:
            sys.stderr.write(f"  ERR {e}\n")
            time.sleep(base_delay * (2 ** attempt))
    return None


def strip_html(s: str) -> str:
    if not s:
        return ""
    # Unwrap CDATA — common in Substack / WordPress feeds where title text is wrapped
    # in <![CDATA[...]]>. Doing this first keeps subsequent tag-stripping correct.
    s = re.sub(r"<!\[CDATA\[(.*?)\]\]>", r"\1", s, flags=re.S)
    s = re.sub(r"<script.*?</script>", "", s, flags=re.S | re.I)
    s = re.sub(r"<style.*?</style>", "", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    s = unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_pubdate(raw: str | None) -> datetime | None:
    if not raw:
        return None
    raw = raw.strip()
    # Try RFC 822 first (RSS standard)
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass
    # Try ISO-8601 (Atom standard)
    for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(raw, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            continue
    return None


def parse_entries(xml: str) -> list[dict]:
    """Parse both RSS <item> and Atom <entry>."""
    out = []
    # RSS: <item>...</item>
    for m in re.finditer(r"<item[^>]*>(.*?)</item>", xml, re.S):
        e = m.group(1)
        title = re.search(r"<title[^>]*>(.*?)</title>", e, re.S)
        link = re.search(r"<link[^>]*>(.*?)</link>", e, re.S)
        pub = re.search(r"<pubDate[^>]*>(.*?)</pubDate>", e, re.S)
        if not pub:
            pub = re.search(r"<dc:date[^>]*>(.*?)</dc:date>", e, re.S)
        desc = re.search(r"<description[^>]*>(.*?)</description>", e, re.S)
        if not desc:
            desc = re.search(r"<content:encoded[^>]*>(.*?)</content:encoded>", e, re.S)
        out.append({
            "title": strip_html(title.group(1)) if title else "",
            "link": (link.group(1).strip() if link else "").strip("<![CDATA[").strip("]]>"),
            "pub_raw": pub.group(1).strip() if pub else "",
            "summary": strip_html(desc.group(1))[:500] if desc else "",
        })
    if out:
        return out

    # Atom: <entry>...</entry>
    for m in re.finditer(r"<entry[^>]*>(.*?)</entry>", xml, re.S):
        e = m.group(1)
        title = re.search(r"<title[^>]*>(.*?)</title>", e, re.S)
        link = re.search(r'<link[^>]*href="([^"]+)"', e)
        pub = re.search(r"<(published|updated)[^>]*>(.*?)</\1>", e, re.S)
        summ = re.search(r"<summary[^>]*>(.*?)</summary>", e, re.S)
        if not summ:
            summ = re.search(r"<content[^>]*>(.*?)</content>", e, re.S)
        out.append({
            "title": strip_html(title.group(1)) if title else "",
            "link": link.group(1) if link else "",
            "pub_raw": pub.group(2).strip() if pub else "",
            "summary": strip_html(summ.group(1))[:500] if summ else "",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Posts within last N days (default 7)")
    parser.add_argument("--since", help="Override --days: only posts on/after YYYY-MM-DD")
    parser.add_argument("--names", help="Comma-separated subset of source names (substring match)")
    parser.add_argument("--tier", choices=["personal", "company", "academia", "newsletter", "conference"], help="Only this tier")
    parser.add_argument("--delay", type=float, default=1.5, help="Seconds between feed fetches")
    parser.add_argument("--list-missing", action="store_true", help="Print sources without feed URL")
    args = parser.parse_args()

    cfg = yaml.safe_load(FEEDS_FILE.read_text())
    feeds = cfg["feeds"]

    if args.list_missing:
        for f in feeds:
            if not f.get("feed"):
                print(f"  - {f['name']} ({f['tier']})  url={f['url']}")
        return 0

    if args.since:
        cutoff = datetime.fromisoformat(args.since).replace(tzinfo=timezone.utc)
    else:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.days)

    name_filter = None
    if args.names:
        name_filter = [s.strip().lower() for s in args.names.split(",")]

    all_posts = []
    seen = set()
    skipped_no_feed = 0
    skipped_fetch_err = 0
    ok_feeds = 0
    total_feeds = 0

    for f in feeds:
        if args.tier and f["tier"] != args.tier:
            continue
        if name_filter and not any(n in f["name"].lower() for n in name_filter):
            continue
        total_feeds += 1
        if not f.get("feed"):
            sys.stderr.write(f"SKIP {f['name']} (no RSS)\n")
            skipped_no_feed += 1
            continue

        # Optional per-source topic filter — applied against title+summary.
        # `title_filter` in feeds.yaml: a case-insensitive regex; entries that
        # don't match are dropped. Used to de-noise firehose feeds (arXiv,
        # LessWrong) where most posts are irrelevant.
        title_filter_re = None
        if f.get("title_filter"):
            try:
                title_filter_re = re.compile(f["title_filter"], re.I)
            except re.error as ex:
                sys.stderr.write(f"  WARN bad title_filter regex on {f['name']}: {ex}\n")

        sys.stderr.write(f"FETCH {f['name']}: ")
        xml = fetch_feed(f["feed"])
        if not xml:
            skipped_fetch_err += 1
            sys.stderr.write(f"FAILED\n")
            continue
        entries = parse_entries(xml)
        kept = 0
        dropped_filter = 0
        for e in entries:
            pub = parse_pubdate(e.get("pub_raw"))
            if not pub or pub < cutoff:
                continue
            if not e["link"] or e["link"] in seen:
                continue
            if title_filter_re and not title_filter_re.search(f"{e['title']} {e['summary']}"):
                dropped_filter += 1
                continue
            seen.add(e["link"])
            all_posts.append({
                "source": f["name"],
                "tier": f["tier"],
                "topics": f.get("topics", []),
                "title": e["title"],
                "link": e["link"],
                "pub": pub.isoformat(),
                "summary": e["summary"],
            })
            kept += 1
        ok_feeds += 1
        msg = f"{kept}/{len(entries)} kept"
        if dropped_filter:
            msg += f" ({dropped_filter} filtered)"
        sys.stderr.write(msg + "\n")
        time.sleep(args.delay)

    # Sort by pub desc
    all_posts.sort(key=lambda p: p["pub"], reverse=True)

    sys.stderr.write(
        f"\nDone: {ok_feeds}/{total_feeds} feeds OK, "
        f"{skipped_no_feed} no-RSS, {skipped_fetch_err} fetch-err. "
        f"{len(all_posts)} posts since {cutoff.isoformat()}.\n"
    )
    print(json.dumps(all_posts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
