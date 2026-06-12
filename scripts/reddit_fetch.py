#!/usr/bin/env python3
"""Fetch top posts from tracked subreddits via Reddit's OAuth API (read-only).

Reads credentials from ~/.reddit/.env. Uses application-only OAuth
(client_credentials grant) — no account password needed for public reads.

Usage:
  uv run python3 scripts/reddit_fetch.py --time week --limit 30
  uv run python3 scripts/reddit_fetch.py --subs MachineLearning,LocalLLaMA --time week
Output: JSON array on stdout (one object per post), sorted by score desc.
"""
import argparse
import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ENV_PATH = Path.home() / ".reddit" / ".env"

# Default tracked subreddits, grouped (matches CLAUDE.md workflow).
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


def load_env(path=ENV_PATH):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Copy ~/.reddit/.env.example to ~/.reddit/.env and fill it in.")
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    for req in ("REDDIT_CLIENT_ID", "REDDIT_CLIENT_SECRET"):
        if not env.get(req) or env[req].startswith("your_"):
            sys.exit(f"ERROR: {req} not set in {path}.")
    return env


def get_token(env):
    user = env.get("REDDIT_USERNAME", "anon")
    ua = f"research-journal-digest/0.1 by u/{user}"
    data = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://www.reddit.com/api/v1/access_token", data=data
    )
    import base64
    auth = base64.b64encode(
        f"{env['REDDIT_CLIENT_ID']}:{env['REDDIT_CLIENT_SECRET']}".encode()
    ).decode()
    req.add_header("Authorization", f"Basic {auth}")
    req.add_header("User-Agent", ua)
    with urllib.request.urlopen(req, timeout=30) as r:
        tok = json.load(r)
    if "access_token" not in tok:
        sys.exit(f"ERROR: auth failed: {tok}")
    return tok["access_token"], ua


def fetch_sub(sub, token, ua, time_filter="week", limit=30):
    url = f"https://oauth.reddit.com/r/{sub}/top?{urllib.parse.urlencode({'t': time_filter, 'limit': limit})}"
    req = urllib.request.Request(url)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("User-Agent", ua)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r)
    except Exception as e:
        sys.stderr.write(f"WARN: r/{sub} fetch failed: {e}\n")
        return []
    out = []
    for c in d.get("data", {}).get("children", []):
        p = c["data"]
        out.append({
            "sub": sub,
            "title": p.get("title", ""),
            "score": p.get("ups", 0),
            "ratio": p.get("upvote_ratio", 0),
            "comments": p.get("num_comments", 0),
            "flair": p.get("link_flair_text") or "",
            "author": p.get("author", ""),
            "created_utc": int(p.get("created_utc", 0)),
            "permalink": "https://www.reddit.com" + p.get("permalink", ""),
            "url": p.get("url", ""),
            "selftext": (p.get("selftext", "") or "")[:500],
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", default=",".join(DEFAULT_SUBS),
                    help="comma-separated subreddits")
    ap.add_argument("--time", default="week",
                    choices=["hour", "day", "week", "month", "year", "all"])
    ap.add_argument("--limit", type=int, default=30, help="posts per sub")
    args = ap.parse_args()

    env = load_env()
    token, ua = get_token(env)
    subs = [s.strip() for s in args.subs.split(",") if s.strip()]
    all_posts = []
    for s in subs:
        all_posts.extend(fetch_sub(s, token, ua, args.time, args.limit))
        time.sleep(1)  # be polite to the API
    all_posts.sort(key=lambda p: -p["score"])
    json.dump(all_posts, sys.stdout, ensure_ascii=False, indent=2)
    sys.stderr.write(f"\nFetched {len(all_posts)} posts from {len(subs)} subs.\n")


if __name__ == "__main__":
    main()
