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
               "mistral", "jurassic", "titan ", "guardrail", "ai agent", "agentic"]),
    ("Compute", ["ec2", "ecs", "eks", "lambda", "fargate", "batch", "outposts",
                 "graviton", "auto scaling", "wavelength", "lightsail", "app runner",
                 "compute "]),
    ("Storage", ["s3 ", "amazon s3", "ebs", "efs", "fsx", "aws backup", "storage gateway",
                 "snowball", "snowmobile", "snow family", "data sync", "datasync"]),
    ("Database", ["rds", "aurora", "dynamodb", "elasticache", "redshift", "neptune",
                  "documentdb", "timestream", "qldb", "memorydb", "keyspaces", "aurora dsql"]),
    ("Networking", ["vpc", "cloudfront", "route 53", "route53", "api gateway", "elb",
                    # "load balanc" could never match: _kw_matches appends \b, and
                    # "balancer" has no word boundary after the "c".
                    "load balancer", "load balancing", "direct connect",
                    "global accelerator", "transit gateway",
                    "private link", "privatelink", "app mesh", "cloud map",
                    "interconnect", "cloud wan", "network firewall", "vpn"]),
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


def _kw_matches(t: str, kw: str) -> bool:
    return re.search(r"\b" + re.escape(kw.strip()) + r"\b", t) is not None


def _kw_pos(t: str, kw: str) -> int | None:
    m = re.search(r"\b" + re.escape(kw.strip()) + r"\b", t)
    return m.start() if m else None


# Keywords that are real service names but also show up as generic feature words in
# other services' announcements ("VPC support for the Glue connector", "flexible
# batch execution" on Redshift, "via the CLI"). They only decide a category when no
# strong (unambiguous) service keyword matched anywhere.
WEAK_KWS = {"vpc", "batch", "support ", " cli", "sdk", "compute ", "config",
            "health ", "model", " q ", "firewall", "artifact ", "load balanc",
            "auto scaling", "generative", "preview"}


def classify(title: str, summary: str) -> str:
    # Prefer matching on the title alone first: the description often name-drops
    # unrelated services (e.g. a Redshift item mentioning "Graviton", or an Aurora
    # DSQL item mentioning "Lambda"), which would otherwise hijack the category.
    t_title = title.lower()
    t_full = (title + " " + summary).lower()
    # Title exhausts BOTH its strong and weak keywords before the description gets a
    # vote: with the loops nested the other way, a title whose only keyword is weak
    # ("AWS Config now supports 15 new resource types") loses to a strong service name
    # merely name-dropped in the body ("...across Bedrock, OpenSearch, SageMaker") and
    # a Config item lands in AI/ML.
    for text in (t_title, t_full):
        for strong_only in (True, False):
            # In the TITLE passes the earliest-matching keyword wins rather than the
            # first category in CATEGORIES order: AWS titles lead with their subject
            # ("Amazon MSK Express brokers now delivers ... to Amazon S3" is an MSK
            # announcement, not a Storage one), so position tracks the real service
            # while list order just encodes the order this table happens to be in.
            # In the full-text fallback, position carries no such meaning — a
            # description name-drops services in arbitrary order — so keep the
            # category-order precedence there.
            by_position = text is t_title
            best: tuple[int, str] | None = None
            for cat, kws in CATEGORIES:
                for kw in kws:
                    if strong_only and kw in WEAK_KWS:
                        continue
                    pos = _kw_pos(text, kw)
                    if pos is None:
                        continue
                    if not by_position:
                        return cat
                    if best is None or pos < best[0]:
                        best = (pos, cat)
            if best is not None:
                return best[1]
    return "其他"


HIGH_KWS = ["generally available", "now available", "ga release", "ga in", "announces ",
            "announces support", "launches ", "introduces ", "new ", "expands to",
            "adds support", "adds ", "now supports", "now support ", "now offers",
            "preview"]
HIGH_HARD = ["fable", "claude opus", "claude sonnet", "claude haiku",
             "gpt-5", "gpt-6", "bedrock", "sagemaker", "agentcore"]


# "X now <verb>s Y" is AWS's standard headline for a new capability. Spelling out
# every verb in HIGH_KWS ("now supports"/"now offers"/...) made the Medium-vs-Low
# call hinge on whether the body text happened to contain a listed phrase — two
# sibling MSK launches on 07-30 split Medium/Low for exactly that reason.
NEW_CAPABILITY_RE = re.compile(r"\bnow \w+s\b")

GA_KWS = ["general availability", "(ga)", " in ga", "generally available"]
REGIONAL_KWS = ["now available in", "expands to", "additional regions", "govcloud",
                "region is now", "now open", "region expansion", "additional aws region",
                "in the aws ", "new aws region", "local zone"]


def impact(title: str, summary: str) -> str:
    t_title = title.lower()
    t = (title + " " + summary).lower()
    regional = any(kw in t_title for kw in REGIONAL_KWS)
    # Region/partition rollouts are graded Low regardless of the service involved —
    # check this BEFORE the HIGH_HARD promotion, or every "Bedrock now in <region>"
    # and "region expansion of G7e on SageMaker" item claims a Top Highlight slot.
    if regional:
        return "Low"
    if any(_kw_matches(t, kw) for kw in HIGH_HARD) and any(kw in t for kw in HIGH_KWS):
        return "High"
    # GA of a service/capability is High (region rollouts already returned above).
    if any(kw in t_title for kw in GA_KWS):
        return "High"
    # Low signals are matched on the TITLE only: a long description almost always
    # contains the word "documentation" (the "see the docs" boilerplate), which
    # would otherwise force every verbose announcement down to Low.
    if any(kw in t_title for kw in ["update to", "documentation", "now supports french",
                                    "now supports japanese", "now supports german",
                                    "available in price", "minor"]):
        return "Low"
    if any(kw in t for kw in HIGH_KWS) or NEW_CAPABILITY_RE.search(t_title):
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
