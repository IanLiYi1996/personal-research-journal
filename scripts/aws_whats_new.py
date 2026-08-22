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
               "mistral", "jurassic", "titan ", "guardrail", "ai agent", "agentic",
               # "amazon q" cannot match "Amazon Quick" (\b fails between q and u), so
               # Quick had no keyword and each item drifted to whatever its body
               # name-dropped: most reached AI/ML via "agentic"/"model", but "deny by
               # default" hit a stray " cli" and landed in Developer Tools (flagged
               # 08-13, recurred 08-19) and "no data messages" fell through to 其他.
               # Pinned to AI/ML — where the majority already sit, and where the
               # product's agentic positioning points — so one product stays in one
               # category. (Analytics is the QuickSight-lineage alternative; AI/ML wins
               # on consistency with the digests already written.)
               "amazon quick"]),
    ("Compute", ["ec2", "ecs", "eks", "lambda", "fargate", "batch", "outposts",
                # Local Zone / new-AZ items name no service in the title, so they fell
                # through to the full-text pass, where category order hands them to
                # AI/ML the moment the body says "model"/"generative" (08-21: a Las
                # Vegas Local Zone GA landed in AI/ML). Same axis choice already made
                # for new AZs on 08-20: infrastructure-footprint items go to Compute
                # rather than getting their own category for n=1. The win here is that
                # the title now decides, instead of whatever the description mentions.
                "local zone", "local zones",
                 "graviton", "auto scaling", "wavelength", "lightsail", "app runner",
                 "compute ",
                 # End-user computing has no category of its own, so with no entry at
                 # all these drifted by whatever the body name-dropped: WorkSpaces
                 # landed in Management on 07-15, AI/ML on 07-21 and Management on
                 # 08-07. Pin them here (virtual desktops / streamed apps) so the
                 # placement is at least stable across digests.
                 "workspaces", "appstream",
                 # Deadline Cloud had no entry, so the two items in the 08-22 feed
                 # split: the download-status one reached Compute only through the
                 # weak "compute " channel matching "compute-intensive workloads"
                 # (the same incidental-phrasing mechanism as the Billing->Compute
                 # error on 08-14), while "EBS Persistent Volume Cost Tracking" hit
                 # "ebs" and landed in Storage. One product, two categories — the
                 # Amazon Quick defect again. Pinned to Compute: it is a managed
                 # render-farm service that provisions and orchestrates worker
                 # fleets, so classifying by subject puts it here. (AWS's own console
                 # files it under Media Services, but there is no such category and
                 # inventing one for n=1 is the over-fitting rejected for new AZs on
                 # 08-20; 其他 is the other option, but it carries less information.)
                 "deadline cloud"]),
    ("Storage", ["s3 ", "amazon s3", "ebs", "efs", "fsx", "aws backup", "storage gateway",
                 "snowball", "snowmobile", "snow family", "data sync", "datasync"]),
    ("Database", ["rds", "aurora", "dynamodb", "elasticache", "redshift", "neptune",
                  "documentdb", "timestream", "qldb", "memorydb", "keyspaces", "aurora dsql"]),
    ("Networking", ["vpc", "cloudfront", "route 53", "route53", "api gateway", "elb",
                    # "load balanc" could never match: _kw_matches appends \b, and
                    # "balancer" has no word boundary after the "c". The plural needs
                    # its own entry for the same reason — AWS headlines the joint
                    # ALB+NLB announcements as "Load Balancers".
                    "load balancer", "load balancers", "load balancing",
                    "direct connect",
                    "global accelerator", "transit gateway",
                    "private link", "privatelink", "app mesh", "cloud map",
                    "interconnect", "cloud wan", "network firewall", "vpn",
                    # AWS Elemental / Media* had no entry anywhere, so the family got
                    # scattered by whatever its bodies name-dropped: MediaTailor to
                    # Developer Tools on 07-28 (a stray " cli"), MediaConnect Router to
                    # Management on 08-21 (its body mentions CloudWatch). Only 2 items
                    # in six weeks — too thin to justify a Media category (same call as
                    # refusing one for new AZs on 08-20) — but the scattering is real
                    # and will recur, so pin the family. Networking is the closest
                    # existing home: these are live video transport/delivery services,
                    # and CloudFront already sits here.
                    "elemental", "mediaconnect", "medialive", "mediaconvert",
                    "mediapackage", "mediatailor"]),
    ("Security", ["iam", "kms", "secrets manager", "guardduty", "inspector", "macie",
                  "waf", "shield", "cognito", "verified access", "verified permissions",
                  "security hub", "detective", "audit manager", "artifact ", "control tower",
                  "firewall", "certificate manager", "acm ",
                  # "AWS Security Agent" (part of AWS Continuum) is a pentesting
                  # service. Without its own entry the title matches nothing and the
                  # item falls through to the description, where a stray "TOTP
                  # support" put it in Management.
                  "security agent", "continuum", "penetration testing"]),
    ("Developer Tools", ["codebuild", "codepipeline", "codeartifact", "codecommit",
                         "codedeploy", "codestar", "cloud9", "cloudshell", " cli", "sdk",
                         "x-ray", "xray", "cdk", "amplify", "appconfig",
                         # Same drift, but the hijacking name-drop is a plausible one:
                         # Console-to-Code is Amazon Q-powered, so "amazon q" in the
                         # body sent it to AI/ML. The announcement's subject is the
                         # console-to-IaC codegen feature, not the model behind it.
                         "console-to-code"]),
    ("Analytics", ["athena", "glue", "emr", "kinesis", "msk", "opensearch",
                   "quicksight", "lake formation", "datazone", "data zone",
                   "managed grafana", "managed prometheus",
                   # Same drift story as WorkSpaces above. "Clean Rooms" matched no
                   # keyword ("supports" doesn't match the \b-anchored "support "),
                   # so it fell through to the description — and because its launches
                   # almost always mention writing results "to an S3 bucket", it
                   # landed in Storage on 08-12 after being Analytics on 08-03 and in
                   # W17. Pin it: privacy-preserving data collaboration is analytics.
                   "clean rooms", "entity resolution",
                   # Third instance of the same drift: MWAA has no keyword, and its
                   # launches almost always mention writing to "Amazon S3", so
                   # 08-19's PythonOperator/BashOperator item landed in Storage.
                   # AWS itself files Managed Workflows for Apache Airflow under
                   # Analytics. Measured before changing: 2 flips in 100 feed items,
                   # both the intended items, no collateral.
                   "mwaa", "managed workflows", "airflow"]),
    ("Management", ["cloudformation", "systems manager", "organizations", "config",
                    "cloudtrail", "cloudwatch", "trusted advisor", "service catalog",
                    "license manager", "compute optimizer",
                    # "support " was meant to catch AWS Support, but _kw_matches strips
                    # the keyword to \bsupport\b — which matches the *ordinary English
                    # verb* in any description ("makes it easier to support mobile deep
                    # linking" put Amazon SES here on 08-15; Amazon Connect's metrics
                    # dashboard landed here the same way). Unlike the WorkSpaces/Clean
                    # Rooms drift above, the culprit isn't a missing service keyword —
                    # it's a keyword that is a common word. Narrow it to the service.
                    # Measured before changing: 2 flips in 100 feed items, both
                    # Management -> 其他, i.e. out of a wrong class into the honest
                    # fallback. "health " left alone: 0 flips, so narrowing it would be
                    # speculative.
                    "aws support", "support plan", "support center",
                    "health ", "cost management",
                    # Same family as "cost management" above: the subject has no keyword,
                    # so "AWS Cost Anomaly Detection supports third-party models on
                    # Amazon Bedrock" was decided by its object and filed under AI/ML.
                    # The title's earliest-keyword rule tracks the subject once the
                    # subject is spellable, so naming it here is enough.
                    "cost anomaly detection"]),
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
            "health ", "model", " q ", "firewall", "artifact ",
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
# AWS often keeps "GA" out of the headline and states it only in the first line of the
# body ("Today, AWS announces the general availability of vector search for DynamoDB"
# was titled "...now supports real-time vector search" → graded Medium on 08-06).
# Matched as a full phrase, not on "general availability" alone, which also appears in
# unrelated boilerplate; 4/100 feed items hit it, and the regional check above still
# wins first so "<instance> now available in <region>" stays Low.
GA_BODY_RE = re.compile(r"announc\w* the general availability of")
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
    # The flagship-service promotion keys off the TITLE only. Matched on full text, a
    # description that merely name-drops one ("...from notebooks in Amazon SageMaker
    # Unified Studio" on an EMR Spark Connect item) promoted unrelated announcements
    # to High and pushed them into Top Highlights.
    if any(_kw_matches(t_title, kw) for kw in HIGH_HARD) and any(kw in t for kw in HIGH_KWS):
        return "High"
    # GA of a service/capability is High (region rollouts already returned above).
    if any(kw in t_title for kw in GA_KWS) or GA_BODY_RE.search(summary.lower()):
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


# Sections this script generates. Anything else in an existing same-date file is
# hand-written (小结与趋势 / Open Questions / Sources ...) and must survive a re-run:
# the file is written once per calendar date, but cron fires at 09:04 while manual
# catch-up runs happen earlier, so a re-run used to silently wipe the prose.
GENERATED_SECTIONS = {"Top Highlights", "按类别详情"}

# Matches the table rows this script emits, so a re-run can recover what an earlier
# run on the same date already listed. Needed because the 24h window slides: re-running
# at 09:04 after a 02:30 run drops every item published before 09:04 the previous day,
# which would silently shrink the day's digest.
# The optional "ᴮ" is the backfill marker (see BACKFILL_HOURS); it also appeared as a
# hand-written annotation in the 08-01..08-03 catch-up digests.
ROW_RE = re.compile(
    r"^\|\s*(\d{2})-(\d{2}) (\d{2}):(\d{2})\s*ᴮ?\s*\|\s*\[(.+?)\]\((\S+?)\)\s*\|\s*(\w+)\s*\|\s*$"
)

# 2026-08-14: the feed DOES backfill — items appear carrying timestamps hours to days in
# the past. Proven on 08-12: at 08-13 09:10 both the RSS and the dirs API reported 6 items
# for 08-12 (first at 17:09); a day later both reported 9 (first at 14:34). The three
# late arrivals sat *inside* the window the 08-13 run had already covered.
#
# Combined with a fixed 24h lookback that made the loss permanent AND silent: by the time
# such an item shows up, the only run whose window covered it has already happened, and
# every later run's window has slid past it. Reconciling 07-31..08-13 against the dirs API
# found 25 of 149 announcements (16.8%) that never reached any digest this way.
# So look back further than a day and re-admit anything the recent digests never listed.
# This cannot double-list: items already in a recent digest are filtered out by link.
#
# 96 -> 168 on 2026-08-21, and the deciding argument is about measurement, not safety.
# By then backfill-delays.tsv held 12 valid rows / 8 distinct events, with upper bounds
# 88.3, 86.5, 42.1, 42.1, 34.9, 31.4, 29.1, 25.1. Two of eight events sit 8-10h under the
# 96h ceiling, and they come from unrelated services (Amazon Quick, Amazon Location),
# so the 88.3h reading is not the one-off it looked like on 08-18.
#
# The censoring is the real point: an upper bound is `now - pubDate`, so a delay longer
# than the window is not merely unmeasured, it is *invisible* — the item never enters the
# window at all. A cluster pressed right against the ceiling is exactly the shape a
# truncated distribution makes. Widening therefore does two jobs: it recovers items in the
# 96-168h band, and it tests whether that band is populated. If nothing above 88h ever
# shows up, 96 was in fact adequate and this can go back down with evidence.
BACKFILL_HOURS = 168
# How many recent digests to read when deciding "have I already covered this?".
# Must span more calendar days than BACKFILL_HOURS, or a re-admitted item could fall off
# the far end of the scan and be reported twice. At one digest per day, 12 files covers
# ~12 days against a 7-day window; 8 files (~8 days) was only barely enough for 96h and
# would have been too tight here.
BACKFILL_SCAN_FILES = 12
# First run that applied BACKFILL_HOURS. Runs before this only looked back 24h, so the
# absence they prove is narrower. Recorded as a date because the mechanism landed with a
# code change, not with data we can sniff: a backfill-capable run that finds nothing looks
# identical in its digest to a pre-backfill run.
BACKFILL_SINCE = dt.datetime(2026, 8, 15, tzinfo=dt.timezone.utc)
# Any aws.amazon.com link in a digest counts as covered, not just generated table rows —
# hand-written prose links to announcements too (the 08-01..08-03 catch-up did exactly that).
ANY_LINK_RE = re.compile(r"\]\((https://aws\.amazon\.com[^)\s]*)\)")

# The feed serves the same announcement under two URL forms — with and without the
# /about-aws/ prefix — and every link comparison here was a raw string match, so one
# announcement counted as two. 2026-08-20 audit: three Amazon Quick items were reported
# twice (08-13/08-14 as `/whats-new/...`, again on 08-18 as `/about-aws/whats-new/...`),
# and because the second sighting looked like a late arrival it also wrote three bogus
# rows to backfill-delays.tsv claiming 60-82h delays. Those were the points that made 96h
# look nearly exhausted; only one of that batch was ever real.
#
# Deliberately narrow: strip the prefix and trailing slash, keep `whats-new/YYYY/MM/slug`.
# Collapsing to the slug alone was measured over every digest to date and merged 3 pairs
# of *different* announcements, because AWS also publishes bare service-name URLs
# (.../2026/08/amazon-quick/, .../amazon-connect/, and today .../amazon-sagemaker/).
# Prefix-only: 3 correct merges, 0 false merges.
def _canon_link(u: str) -> str:
    return re.sub(r"^https://aws\.amazon\.com/about-aws/",
                  "https://aws.amazon.com/", u.strip()).rstrip("/").lower()


# Printing each observation would not accumulate them — this runs from cron, whose stdout
# is not kept — so append them to a file instead. These rows are what turned BACKFILL_HOURS
# from a guess into a decision (see the 96 -> 168 note above); keep feeding them, and read
# the *upper-bound* distribution rather than any single value. Rows marked "# INVALID" are
# kept on purpose: they were the evidence behind an earlier, wrong reading of the margin.
DELAY_LOG = OUT_DIR / "backfill-delays.tsv"
RUN_TS_RE = re.compile(r"^- \*\*抓取时间:\*\*\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC")


def _prior_run_times() -> list[dt.datetime]:
    """Fetch timestamps from past digest headers, newest first.

    Merge-write overwrites the header, so this sees only each day's *last* run — which is
    the conservative direction for the bound below (a later run is a tighter one).
    """
    out: list[dt.datetime] = []
    for p in sorted(OUT_DIR.glob("????-??-??.md"), reverse=True):
        for line in p.read_text(encoding="utf-8").splitlines()[:8]:
            m = RUN_TS_RE.match(line)
            if m:
                out.append(dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                           .replace(tzinfo=dt.timezone.utc))
                break
    return sorted(out, reverse=True)


def _log_backfill_delays(rows: list[dict], now_utc: dt.datetime,
                         already_reported: set[str] | None = None) -> int:
    """Record how late the feed produced each backfilled item. Returns points written.

    `already_reported` are links an *earlier run of the same day* already listed. They must
    be excluded: `_already_covered()` skips today's own digest on purpose (fresh RSS should
    win on classification), so a second run re-detects the same items as backfill. Without
    this, 2026-08-18 wrote each of four items twice — and the duplicate's lower bound cited
    the 07:18 run as proof of absence even though 07:18 is the run that reported it, pushing
    the bound from a true ~60h to ~88h. One backfill event must produce one point: CLAUDE.md
    already notes these have to be aggregated per event before any distribution analysis,
    and silent duplicates would double-weight whichever events happen to span two runs.

    Reported as an interval, because a single number would overstate what is known:
      * upper = now - pubDate. This run is merely the first to *report* the item; it may
        have entered the feed any time earlier and simply been missed.
      * lower = r - pubDate, where r is the latest prior run whose *effective lookback*
        contained pubDate and which still did not list the item ⟹ at r it was
        demonstrably absent. Effective lookback is per-run, because it changed:
        runs before BACKFILL_SINCE only looked back 24h, runs after look back
        BACKFILL_HOURS. Using a flat 24h (as this did originally) throws away provable
        absence: 2026-08-18 logged four items whose 96h-window runs on 08-16 and 08-17
        found nothing, so their lower bound is ~60h, not the ~12h a 24h window yields.
        That matters because these points exist to decide whether BACKFILL_HOURS can be
        lowered — a too-small lower bound would make the window look safer than it is.
        Empty when no such run exists — the delay then has no lower bound from this data.
    """
    seen = already_reported or set()
    bf = [r for r in rows if r.get("backfill") and _canon_link(r["link"]) not in seen]
    if not bf:
        return 0
    runs = _prior_run_times()
    fresh = not DELAY_LOG.exists()
    with DELAY_LOG.open("a", encoding="utf-8") as f:
        if fresh:
            f.write("observed_utc\tpub_utc\tdelay_lower_h\tdelay_upper_h\tlink\n")
        for r in sorted(bf, key=lambda r: r["pub"]):
            pub = r["pub"]
            lower = ""
            for t in runs:  # newest first ⟹ first match is the item's last missed chance
                # Skip this very run: its header may already be on disk (same-day merge
                # write, or a re-run), and it is the run *reporting* the item, so it
                # proves nothing about absence. Widening the window to BACKFILL_HOURS
                # exposed this — under a flat 24h it could never match an older pubDate.
                if t >= now_utc:
                    continue
                look = BACKFILL_HOURS if t >= BACKFILL_SINCE else 24
                if t >= pub >= t - dt.timedelta(hours=look):
                    lower = f"{(t - pub).total_seconds() / 3600:.1f}"
                    break
            f.write(f"{now_utc:%Y-%m-%d %H:%M}\t{pub:%Y-%m-%d %H:%M}\t{lower}\t"
                    f"{(now_utc - pub).total_seconds() / 3600:.1f}\t{r['link']}\n")
    return len(bf)


def _already_covered(out_path: Path) -> set[str]:
    """Canonical links listed by any recent digest, so backfill is only reported once."""
    covered: set[str] = set()
    files = sorted(OUT_DIR.glob("????-??-??.md"), reverse=True)
    for p in files[:BACKFILL_SCAN_FILES]:
        if p == out_path:
            continue
        covered.update(_canon_link(u)
                       for u in ANY_LINK_RE.findall(p.read_text(encoding="utf-8")))
    return covered


# Title -> canonical link, from the same recent digests. Canonicalising links fixes the
# /about-aws/ split but cannot catch a re-publication whose *slug* also changed: "Amazon
# Quick adds deny by default for custom permissions" ran 08-12 as
# `amazon-quick-deny-by-default-permissions` and again 08-19 as `amazon-quick-deny-by-default`.
# Flagged rather than dropped: AWS reuses headlines for successive regional rollouts, so
# an exact-title match is evidence for a human to check, not grounds for suppression.
# Measured over every digest to date: 5 exact-title/different-URL pairs, all genuine
# duplicates, so the flag is quiet enough to be worth printing.
def _prior_titles(out_path: Path) -> dict[str, str]:
    ROW_LINK = re.compile(r"^\|.*?\[([^\]]+)\]\((https://aws\.amazon\.com[^)\s]*)\)")
    seen: dict[str, str] = {}
    files = sorted(OUT_DIR.glob("????-??-??.md"), reverse=True)
    for p in files[:BACKFILL_SCAN_FILES]:
        if p == out_path:
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            m = ROW_LINK.match(line)
            if m:
                seen.setdefault(m.group(1).strip().replace("\\|", "|"),
                                _canon_link(m.group(2)))
    return seen


def _parse_existing(path: Path) -> tuple[dict[str, dict], list[str]]:
    """Return (rows-by-link, hand-written-prose-lines) from a previous same-date run."""
    if not path.exists():
        return {}, []
    text = path.read_text(encoding="utf-8")
    today = dt.datetime.now()

    prior: dict[str, dict] = {}
    prose: list[str] = []
    current = None  # current "## " section title
    cat = None      # current "### <Category> (n 项)" subsection — this is the category
    for line in text.splitlines():
        if line.startswith("## ") and not line.startswith("### "):
            current = line[3:].strip()
            cat = None
            if current not in GENERATED_SECTIONS:
                prose.append(line)
            continue
        if current is not None and current not in GENERATED_SECTIONS:
            prose.append(line)
            continue
        if line.startswith("### "):
            # Strip the " (n 项)" suffix; the category name is what precedes it.
            cat = re.sub(r"\s*\(\d+\s*项\)\s*$", "", line[4:]).strip()
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        mon, day, hh, mm, title, link, imp = m.groups()
        # Rows carry only MM-DD; pick the year that puts them at or before today.
        year = today.year
        try:
            pub = dt.datetime(year, int(mon), int(day), int(hh), int(mm),
                              tzinfo=dt.timezone.utc)
        except ValueError:
            continue
        if pub.date() > today.date():
            pub = pub.replace(year=year - 1)
        prior[link] = {"title": title.replace("\\|", "|"), "link": link, "pub": pub,
                       "descr": "", "category": cat or "其他", "impact": imp,
                       "carried": True}
    while prose and not prose[-1].strip():
        prose.pop()
    return prior, prose


MAX_RSS_AGE_MIN = 60


def main() -> int:
    if not RSS.exists():
        print("RSS missing", file=sys.stderr)
        return 1
    # 2026-08-14: caught a silent false zero. The wrapper had curl'd the feed to a
    # different path, so this read a day-old /tmp/aws-rss.xml and reported "0 items"
    # — which on a quiet weekday morning is entirely plausible and would have been
    # written up as a real publishing gap. Fail loudly instead: a stale feed is
    # indistinguishable from a quiet day in the output, so it must not be silent.
    age_min = (dt.datetime.now().timestamp() - RSS.stat().st_mtime) / 60
    if age_min > MAX_RSS_AGE_MIN:
        print(f"RSS at {RSS} is {age_min:.0f} min old (limit {MAX_RSS_AGE_MIN}). "
              f"Re-fetch it before running:\n"
              f'  curl -s "https://aws.amazon.com/about-aws/whats-new/recent/feed/" -o {RSS}',
              file=sys.stderr)
        return 2
    tree = ET.parse(RSS)
    root = tree.getroot()
    items = root.findall(".//item")
    now_utc = dt.datetime.now(dt.timezone.utc)
    cutoff = now_utc - dt.timedelta(hours=24)
    backfill_cutoff = now_utc - dt.timedelta(hours=BACKFILL_HOURS)
    covered = _already_covered(OUT_DIR / dt.datetime.now().strftime("%Y-%m-%d.md"))
    rows = []
    backfilled = 0
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
        is_backfill = False
        if pub < cutoff:
            # Older than the normal window: keep it only if it never made it into a
            # recent digest, i.e. it arrived after the run that should have caught it.
            if pub < backfill_cutoff or _canon_link(link) in covered:
                continue
            is_backfill = True
        if _canon_link(link) in seen_links:
            continue
        seen_links.add(_canon_link(link))
        if is_backfill:
            backfilled += 1
        rows.append({
            "title": title,
            "link": link,
            "pub": pub,
            "descr": descr,
            "category": classify(title, descr),
            "impact": impact(title, descr),
            "backfill": is_backfill,
        })
    today = dt.datetime.now().strftime("%Y-%m-%d")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{today}.md"

    # Merge with whatever an earlier run today already wrote, so neither the prose nor
    # items that have since slid out of the 24h window are lost. Fresh RSS wins on
    # conflict — the classifier/impact heuristics may have been fixed since.
    prior, prose = _parse_existing(out)
    fresh_links = {_canon_link(r["link"]) for r in rows}
    carried = [r for link, r in prior.items() if _canon_link(link) not in fresh_links]
    rows += carried
    rows.sort(key=lambda r: r["pub"], reverse=True)
    logged = _log_backfill_delays(rows, now_utc,
                                 already_reported={_canon_link(l) for l in prior})

    # Same headline as a recent digest under a different canonical link — see _prior_titles.
    prior_titles = _prior_titles(out)
    repeats = [r for r in rows
               if r["title"] in prior_titles
               and prior_titles[r["title"]] != _canon_link(r["link"])]
    for r in repeats:
        print(f"  ! 标题与最近 digest 重复但链接不同（可能是换 slug 重发，请人工确认）:"
              f"\n      {r['title']}\n      now : {_canon_link(r['link'])}"
              f"\n      prev: {prior_titles[r['title']]}", file=sys.stderr)

    lines = [f"# AWS What's New: {today}", "",
             f"- **抓取时间:** {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
             f"- **过去 24h 公告数:** {len(rows) - backfilled}"
             + (f"（本次抓取 {len(fresh_links) - backfilled} + 早前同日抓取保留 {len(carried)}）"
                if carried else "")
             + (f" ＋ **补录 {backfilled} 条**（`ᴮ`，pubDate 早于 24h 窗口但从未进过任何 digest）"
                if backfilled else ""),
             "- **Source:** https://aws.amazon.com/about-aws/whats-new/recent/feed/",
             ""]
    if not rows:
        lines += ["", "过去 24h RSS 无新条目。", ""]
        # Preserve hand-written sections here too. The zero-item path used to return
        # early without appending `prose`, so a later same-day run (e.g. the 09:04
        # cron after a manual morning run) silently wiped analysis written into a
        # weekend/empty digest. Weekend digests are exactly where the prose is the
        # only content, so losing it defeated the whole point of merge-write.
        if prose:
            lines += prose + [""]
        out.write_text("\n".join(lines), encoding="utf-8")
        print(f"wrote {out} (0 items, {len(prose)} prose lines kept)")
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
            if r.get("backfill"):
                t += " ᴮ"
            title_md = r["title"].replace("|", "\\|")
            lines.append(f"| {t} | [{title_md}]({r['link']}) | {r['impact']} |")
        lines.append("")

    if prose:
        lines += prose + [""]

    out.write_text("\n".join(lines), encoding="utf-8")
    note = f", kept {len(carried)} carried, {len(prose)} prose lines" if (carried or prose) else ""
    if logged:
        note += f", logged {logged} backfill delay(s) to {DELAY_LOG.name}"
    print(f"wrote {out} ({len(rows)} items, {len(highs)} highs{note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
