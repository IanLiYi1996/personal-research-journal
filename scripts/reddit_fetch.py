#!/usr/bin/env python3
"""Fetch top posts from tracked subreddits via Reddit's public RSS feeds.

Reddit's JSON API (*.json) is 403-blocked for unauthenticated clients on all
IPs we can reach (verified June 2026), so this uses the `.rss` feeds, which
still return 200. RSS provides title / link / author / timestamp / flair-ish
category, but NOT score or comment count — ranking falls back to Reddit's own
"top of week" ordering (feed order). If you later obtain official API
credentials, swap this for the OAuth path to recover score/comments.

Politeness: RSS is aggressively rate-limited (HTTP 429). This script spaces
requests out and retries with exponential backoff.

Usage:
  uv run python3 scripts/reddit_fetch.py --time week > /tmp/reddit.json
  uv run python3 scripts/reddit_fetch.py --subs MachineLearning,aws --time week
Output: JSON array on stdout, grouped-order preserved, each post tagged with
its subreddit and feed rank (lower = higher on that sub's top list).
"""
import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from html import unescape

DEFAULT_SUBS = [
    # AI/ML research
    "MachineLearning", "LocalLLaMA", "singularity",
    # AI products/apps
    "OpenAI", "ClaudeAI", "StableDiffusion",
    # AWS/cloud/eng
    "aws", "devops", "programming",
    # data science/academia
    "datascience", "statistics", "AskAcademia",
]

UA = "Mozilla/5.0 (research-journal-digest/0.2; weekly digest; respectful)"


def fetch_rss(sub, time_filter="week", max_retries=4, base_delay=5):
    url = f"https://www.reddit.com/r/{sub}/top/.rss?t={time_filter}"
    for attempt in range(max_retries):
        req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/atom+xml"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                body = r.read().decode("utf-8", "replace")
            if body.strip():
                return body
            sys.stderr.write(f"WARN: r/{sub} empty body (attempt {attempt+1})\n")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = base_delay * (2 ** attempt)
                sys.stderr.write(f"WARN: r/{sub} 429, backing off {wait}s\n")
                time.sleep(wait)
                continue
            sys.stderr.write(f"WARN: r/{sub} HTTP {e.code}\n")
            return None
        except Exception as e:
            sys.stderr.write(f"WARN: r/{sub} {e}\n")
        time.sleep(base_delay * (2 ** attempt))
    sys.stderr.write(f"ERROR: r/{sub} gave up after {max_retries} attempts\n")
    return None


def parse_entries(xml, sub):
    out = []
    entries = re.findall(r"<entry>(.*?)</entry>", xml, re.S)
    for rank, e in enumerate(entries):
        title = re.search(r"<title>(.*?)</title>", e, re.S)
        link = re.search(r'<link href="([^"]+)"', e)
        author = re.search(r"<author>.*?<name>(.*?)</name>", e, re.S)
        updated = re.search(r"<updated>(.*?)</updated>", e, re.S)
        pid = re.search(r"<id>(.*?)</id>", e, re.S)
        title_txt = unescape(title.group(1)).strip() if title else ""
        # extract trailing [D]/[R]/[P]/[N] tag common on r/MachineLearning
        flair = ""
        m = re.search(r"\[([A-Za-z]{1,3})\]\s*$", title_txt)
        if m:
            flair = m.group(1)
        out.append({
            "sub": sub,
            "rank": rank,  # 0 = top of this sub's weekly list
            "title": title_txt,
            "flair": flair,
            "author": (author.group(1) if author else "").lstrip("/u/"),
            "updated": updated.group(1) if updated else "",
            "id": pid.group(1) if pid else "",
            "permalink": link.group(1) if link else "",
            # RSS has no score/comments — explicitly null so the digest layer knows
            "score": None,
            "comments": None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(DEFAULT_SUBS))
    ap.add_argument("--time", default="week",
                    choices=["hour", "day", "week", "month", "year", "all"])
    ap.add_argument("--delay", type=float, default=6,
                    help="seconds between subreddit fetches (avoid 429)")
    args = ap.parse_args()

    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    all_posts, failed = [], []
    for i, s in enumerate(subs):
        xml = fetch_rss(s, args.time)
        if xml:
            posts = parse_entries(xml, s)
            all_posts.extend(posts)
            sys.stderr.write(f"OK: r/{s} -> {len(posts)} posts\n")
        else:
            failed.append(s)
        if i < len(subs) - 1:
            time.sleep(args.delay)

    json.dump(all_posts, sys.stdout, ensure_ascii=False, indent=2)
    sys.stderr.write(f"\nDone: {len(all_posts)} posts from {len(subs)-len(failed)}/{len(subs)} subs.\n")
    if failed:
        sys.stderr.write(f"FAILED (retry later): {', '.join(failed)}\n")


if __name__ == "__main__":
    main()
