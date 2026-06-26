#!/usr/bin/env python3
"""HTML index-page scraper for blogs without RSS or usable sitemap.

For each configured source, fetches the blog index page, extracts <article>
blocks (or equivalent), and parses out (link, title, publish-date) tuples.
Output JSON matches blog_fetch.py schema.

Date strategy:
  - Most modern marketing blogs render the publish date as "Month DD, YYYY"
    text inside the listing card itself.  We capture that.
  - If the listing has no inline date but the article URL contains a date
    slug (e.g. "/2026-06-25-foo"), we use that.
  - As last resort, we keep the entry with pub=None and let the caller decide.

Per-source rules live in HTML_RULES below.  Adding a new source: probe with
curl, identify a stable regex that captures (link, title, date) from the
listing HTML, add an entry.

Usage:
  uv run python3 scripts/blog_html_scraper.py --days 14 > /tmp/html.json
  uv run python3 scripts/blog_html_scraper.py --names "Mistral,Meta AI"
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from html import unescape
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.stderr.write("ERROR: PyYAML not available. Install with: uv add pyyaml\n")
    sys.exit(1)

UA = "Mozilla/5.0 (research-journal-blog-digest/0.1; weekly; respectful)"
FEEDS_FILE = Path(__file__).parent.parent / "tech-blogs" / "feeds.yaml"

# Per-source scraping rules.  Keyed by feeds.yaml `name`.
#
# Each rule:
#   index_url:   URL of the blog listing page
#   item_regex:  regex with named groups, each match becomes one article.
#                Required group: 'link' (absolute or relative).
#                Optional groups: 'title', 'date_text'.
#   base_url:    used to absolutize relative links (defaults to scheme+host of index_url)
#   date_format: strftime-style fmt for parsing 'date_text' (default: try several)
#   max_undated: keep at most N undated entries (newest-first by listing order)
#                from this source.  Use when listing has no inline date but is
#                sorted newest-first.  Caller still uses --keep-undated flag.
#
# Tips:
#   - Test with: curl -sL -A "Mozilla/5.0" <index_url> | grep -A5 '<article'
#   - Many modern blogs are SPAs; if the initial HTML doesn't include card
#     content, this scraper cannot help.  Document such sources as unreachable.
HTML_RULES: dict[str, dict] = {
    "Mistral AI News": {
        "index_url": "https://mistral.ai/news",
        # <a href="/news/slug/"... <h2 class="text-h..">TITLE </h2> ... <p class="text-body-small ...">June 23, 2026 </p>
        "item_regex": re.compile(
            r'<a href="(?P<link>/news/[^"]+)"[^>]*class="group/news[^"]*"[^>]*>'
            r'.*?<h2[^>]*>(?P<title>[^<]+)</h2>'
            r'.*?<p class="text-body-small[^"]*">\s*(?P<date_text>[A-Z][a-z]+ \d{1,2}, \d{4})',
            re.S,
        ),
        "base_url": "https://mistral.ai",
    },
    "Meta AI (FAIR) Blog": {
        "index_url": "https://ai.meta.com/blog/",
        # Listing has no inline date (JS-rendered). We capture the link slug
        # only; with `max_undated=6` we keep the 6 newest cards (by listing
        # order) as a proxy for "this week's posts" — caller must still pass
        # --keep-undated.  Slug is used as the title.
        "item_regex": re.compile(
            r'href="(?P<link>https://ai\.meta\.com/blog/[^"/]+/)"',
            re.S,
        ),
        "base_url": "https://ai.meta.com",
        "max_undated": 6,
    },
    "Apollo Research": {
        "index_url": "https://www.apolloresearch.ai/blog",
        # Pattern: <a href="https://www.apolloresearch.ai/blog/slug/"></a>...
        #          <h3>TITLE</h3>... <div class="date">May 13, 2026</div>
        # An empty <a> appears first, then title + date within the same card.
        "item_regex": re.compile(
            r'<a href="(?P<link>https://www\.apolloresearch\.ai/blog/[^"/]+)/?"\s*></a>'
            r'.*?<h3>(?P<title>[^<]+)</h3>'
            r'.*?<div class="date">(?P<date_text>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})</div>',
            re.S,
        ),
        "base_url": "https://www.apolloresearch.ai",
    },
    "MILA News": {
        "index_url": "https://mila.quebec/en/news",
        # Listing date is rendered via JS. Same heuristic as Meta AI: keep top
        # 5 undated cards as "recent" (listing is sorted newest-first).
        "item_regex": re.compile(
            r'<a href="(?P<link>/en/news/[^"]+)"[^>]*>\s*(?P<title>[^<]{8,200})</a>',
            re.S,
        ),
        "base_url": "https://mila.quebec",
        "max_undated": 5,
    },
    "LlamaIndex Blog": {
        "index_url": "https://www.llamaindex.ai/blog",
        # Astro site. Pattern: <a href="/blog/slug">TITLE</a> ... <p class="...PostDate"> Jun 25, 2026 </p>
        "item_regex": re.compile(
            r'<a href="(?P<link>/blog/[a-zA-Z0-9_-]+)"[^>]*>(?P<title>[^<]{5,200})</a>'
            r'.*?class="[^"]*PostDate[^"]*"[^>]*>\s*(?P<date_text>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})',
            re.S,
        ),
        "base_url": "https://www.llamaindex.ai",
    },
    "fast.ai": {
        "index_url": "https://www.fast.ai/",
        # Quarto site: <h2><a href="./posts/2026-06-DD-slug.html">TITLE</a></h2> + nearby date
        # Many fast.ai posts have YYYY-MM-DD in slug.
        "item_regex": re.compile(
            r'<a href="(?P<link>\./posts/(?P<date_text>\d{4}-\d{2}-\d{2})-[^"]+\.html)"[^>]*>(?P<title>[^<]+)</a>',
            re.S,
        ),
        "base_url": "https://www.fast.ai",
    },
}


def fetch(url: str, max_retries: int = 2) -> str | None:
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "text/html,*/*"})
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                return r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            sys.stderr.write(f"  HTTP {e.code}\n")
            return None
        except Exception as e:
            if attempt == max_retries - 1:
                sys.stderr.write(f"  ERR {e}\n")
                return None
            time.sleep(2 + attempt * 2)
    return None


# Month-name → number, for parsing "June 23, 2026" / "Jun 2025"
MONTHS = {
    m: i + 1
    for i, m in enumerate([
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    ])
}
for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]):
    MONTHS[m] = i + 1


def parse_date(text: str | None) -> datetime | None:
    """Parse common blog-listing date formats. Returns UTC datetime or None."""
    if not text:
        return None
    s = text.strip()
    # ISO yyyy-mm-dd
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", s)
    if m:
        try:
            return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass
    # "Month DD, YYYY" or "Month DD YYYY"
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})", s)
    if m:
        mon = MONTHS.get(m.group(1).lower()[:9])
        if mon:
            try:
                return datetime(int(m.group(3)), mon, int(m.group(2)), tzinfo=timezone.utc)
            except ValueError:
                pass
    # "Month YYYY" -> day=1
    m = re.match(r"([A-Za-z]+)\.?\s+(\d{4})$", s)
    if m:
        mon = MONTHS.get(m.group(1).lower()[:9])
        if mon:
            try:
                return datetime(int(m.group(2)), mon, 1, tzinfo=timezone.utc)
            except ValueError:
                pass
    return None


def absolutize(link: str, base_url: str) -> str:
    if link.startswith("http://") or link.startswith("https://"):
        return link
    if link.startswith("//"):
        return "https:" + link
    if link.startswith("/"):
        return base_url.rstrip("/") + link
    return base_url.rstrip("/") + "/" + link


def slug_to_title(url: str) -> str:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"^\d{4}-\d{2}-\d{2}-", "", slug)  # strip date prefix
    slug = slug.replace(".html", "")
    title = slug.replace("-", " ").replace("_", " ").strip()
    return " ".join(w.capitalize() if not (w.isupper() and len(w) <= 4) else w for w in title.split())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--since", help="YYYY-MM-DD override")
    parser.add_argument("--names", help="Comma substring filter")
    parser.add_argument("--keep-undated", action="store_true",
                        help="Keep entries without a parseable date (default: drop)")
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
    seen_links: set[str] = set()

    for src_name, rule in HTML_RULES.items():
        feed_cfg = feeds_by_name.get(src_name)
        if not feed_cfg:
            sys.stderr.write(f"WARN: {src_name} not in feeds.yaml — skipping\n")
            continue
        if name_filter and not any(n in src_name.lower() for n in name_filter):
            continue
        sys.stderr.write(f"HTML {src_name}: ")
        html = fetch(rule["index_url"])
        if not html:
            sys.stderr.write("FAILED\n")
            continue

        base = rule.get("base_url") or rule["index_url"].rsplit("/", 1)[0]
        max_undated = rule.get("max_undated", 0)
        kept = 0
        total = 0
        undated_kept = 0
        for m in rule["item_regex"].finditer(html):
            total += 1
            link = absolutize(m.group("link"), base)
            if link in seen_links:
                continue
            groups = m.groupdict()
            title = groups.get("title") or groups.get("title2") or slug_to_title(link)
            title = unescape(title.strip())
            date_text = groups.get("date_text")
            pub = parse_date(date_text)
            if pub is None:
                # Undated entry: only keep if --keep-undated AND under the
                # source's max_undated quota (newest-first listing order).
                if not args.keep_undated or undated_kept >= max_undated:
                    continue
                undated_kept += 1
            elif pub < cutoff:
                continue
            seen_links.add(link)
            all_posts.append({
                "source": src_name,
                "tier": feed_cfg["tier"],
                "topics": feed_cfg.get("topics", []),
                "title": title,
                "link": link,
                "pub": pub.isoformat() if pub else "",
                "summary": "",
            })
            kept += 1
        sys.stderr.write(f"{kept}/{total} kept ({undated_kept} undated)\n")
        time.sleep(1.5)

    all_posts.sort(key=lambda p: p["pub"] or "", reverse=True)
    sys.stderr.write(f"\nDone: {len(all_posts)} HTML posts since {cutoff.isoformat()}\n")
    print(json.dumps(all_posts, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
