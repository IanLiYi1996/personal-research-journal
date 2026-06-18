#!/usr/bin/env python3
"""Parse AWS What's New RSS, filter to past 24h, classify, write digest."""
from __future__ import annotations
import datetime as dt
import html
import re
import sys
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path

RSS = Path("/tmp/aws-rss.xml")
OUT_DIR = Path("/home/ec2-user/research/personal-research-journal/aws-whats-new")

CATEGORIES = [
    ("AI/ML", ["bedrock", "sagemaker", "amazon q ", " q ", "comprehend", "rekognition",
               "agentcore", "polly", "transcribe", "translate", "kendra", "personalize",
               "lex ", "textract", "forecast", "augmented ai", "ai/ml", "model", "llm",
               "generative", "nova", "claude", "gpt-", "anthropic", "openai", "gemma",
               "mistral", "jurassic", "titan ", "guardrail", "agent ", "ai agent"]),
    ("Compute", ["ec2", "ecs", "eks", "lambda", "fargate", "batch", "outposts",
                 "graviton", "auto scaling", "wavelength", "lightsail", "app runner",
                 "compute "]),
    ("Storage", ["s3 ", "amazon s3", "ebs", "efs", "fsx", "aws backup", "storage gateway",
                 "snowball", "snowmobile", "snow family", "data sync", "datasync"]),
    ("Database", ["rds", "aurora", "dynamodb", "elasticache", "redshift", "neptune",
                  "documentdb", "timestream", "qldb", "memorydb", "keyspaces", "aurora dsql"]),
    ("Networking", ["vpc", "cloudfront", "route 53", "route53", "api gateway", "elb",
                    "load balanc", "direct connect", "global accelerator", "transit gateway",
                    "private link", "privatelink", "app mesh", "cloud map"]),
    ("Security", ["iam", "kms", "secrets manager", "guardduty", "inspector", "macie",
                  "waf", "shield", "cognito", "verified access", "verified permissions",
                  "security hub", "detective", "audit manager", "artifact ", "control tower",
                  "firewall", "certificate manager", "acm "]),
    ("Developer Tools", ["codebuild", "codepipeline", "codeartifact", "codecommit",
                         "codedeploy", "codestar", "cloud9", "cloudshell", " cli", "sdk",
                         "x-ray", "xray", "cdk", "amplify", "appconfig"]),
    ("Analytics", ["athena", "glue", "emr", "kinesis", "msk", "opensearch",
                   "quicksight", "lake formation", "datazone", "data zone",
                   "managed grafana", "managed prometheus"]),
    ("Management", ["cloudformation", "systems manager", "organizations", "config",
                    "cloudtrail", "cloudwatch", "trusted advisor", "service catalog",
                    "license manager", "compute optimizer", "support ", "health "]),
]


def classify(title: str, summary: str) -> str:
    t = (title + " " + summary).lower()
    for cat, kws in CATEGORIES:
        if any(kw in t for kw in kws):
            return cat
    return "其他"


HIGH_KWS = ["generally available", "now available", "ga release", "ga in", "announces ",
            "announces support", "launches ", "introduces ", "new ", "expands to",
            "adds support", "preview"]
HIGH_HARD = ["fable", "claude opus", "claude sonnet", "claude haiku",
             "gpt-5", "gpt-6", "bedrock", "sagemaker", "agentcore"]


def impact(title: str, summary: str) -> str:
    t = (title + " " + summary).lower()
    if any(kw in t for kw in HIGH_HARD) and any(kw in t for kw in HIGH_KWS):
        return "High"
    if any(kw in t for kw in ["update to", "documentation", "now supports french",
                              "now supports japanese", "now supports german",
                              "available in price", "minor"]):
        return "Low"
    if any(kw in t for kw in HIGH_KWS):
        return "Medium"
    return "Low"


def main() -> int:
    if not RSS.exists():
        print("RSS missing", file=sys.stderr)
        return 1
    tree = ET.parse(RSS)
    root = tree.getroot()
    items = root.findall(".//item")
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=24)
    rows = []
    seen_links = set()
    for it in items:
        title = (it.findtext("title") or "").strip()
        link = (it.findtext("link") or "").strip()
        pubdate_s = (it.findtext("pubDate") or "").strip()
        descr_raw = (it.findtext("description") or "").strip()
        descr = html.unescape(re.sub(r"<[^>]+>", " ", descr_raw))
        descr = re.sub(r"\s+", " ", descr).strip()
        try:
            pub = parsedate_to_datetime(pubdate_s)
        except Exception:
            continue
        if pub.tzinfo is None:
            pub = pub.replace(tzinfo=dt.timezone.utc)
        if pub < cutoff:
            continue
        if link in seen_links:
            continue
        seen_links.add(link)
        rows.append({
            "title": title,
            "link": link,
            "pub": pub,
            "descr": descr,
            "category": classify(title, descr),
            "impact": impact(title, descr),
        })
    rows.sort(key=lambda r: r["pub"], reverse=True)

    today = dt.datetime.now().strftime("%Y-%m-%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today}.md"

    lines = [f"# AWS What's New: {today}", "",
             f"- **抓取时间:** {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
             f"- **过去 24h 公告数:** {len(rows)}",
             "- **Source:** https://aws.amazon.com/about-aws/whats-new/recent/feed/",
             ""]
    if not rows:
        lines += ["", "过去 24h RSS 无新条目。", ""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {out} (0 items)")
        return 0

    highs = [r for r in rows if r["impact"] == "High"][:5]
    if highs:
        lines += ["## Top Highlights", ""]
        for r in highs:
            lines.append(f"- [{r['title']}]({r['link']}) — {r['category']}")
        lines.append("")

    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r["category"], []).append(r)
    cat_order = [c for c, _ in CATEGORIES] + ["其他"]
    lines += ["## 按类别详情", ""]
    for cat in cat_order:
        if cat not in by_cat:
            continue
        crows = by_cat[cat]
        lines += [f"### {cat} ({len(crows)} 项)", "",
                  "| 时间 (UTC) | 公告 | 影响 |",
                  "|------|------|------|"]
        for r in crows:
            t = r["pub"].astimezone(dt.timezone.utc).strftime("%m-%d %H:%M")
            title_md = r["title"].replace("|", "\\|")
            lines.append(f"| {t} | [{title_md}]({r['link']}) | {r['impact']} |")
        lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out} ({len(rows)} items, {len(highs)} highs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
