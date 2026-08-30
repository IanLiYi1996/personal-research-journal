#!/usr/bin/env python3
"""Run log / heartbeat for the recurring digest tasks.

Why this exists
---------------
2026-08-15~17 the HF / Reddit / tech-blogs digests produced nothing for three
days while `CronList` showed all six jobs alive.  "Job exists but the session
did not complete" and "ran and legitimately found nothing" look *identical* in
the output, so the gap was only noticed days later -- by which point the Reddit
top-of-week board had rolled and the shallowest RSS feeds had slid past
(measured 2026-08-18: arXiv cs.AI holds one day, LessWrong <1 day, QbitAI 1.75
days, so those three days are unrecoverable).

The fix is the judgement already applied at the fetch layer ("a failure that can
be silent must be made loud"), lifted to the cron layer.

Two rows, not one
-----------------
The observed failure mode is *the session dying part-way*.  A heartbeat written
only at the end of a run would therefore be missing in exactly the case it is
supposed to catch.  Writing `start` first and `done` last yields three states
that stay distinguishable after the fact:

    no row at all       -> the task never started (cron gone, or died instantly)
    start, no done      -> it ran but did not finish   <- the 08-15~17 shape
    start + done        -> normal

Where the checker lives
-----------------------
`check` is meant to run inside *every* task, not only the most reliable one: a
detector that lives inside the thing being detected disappears together with it.
On 08-15~17 exactly one of six tasks survived, and one is enough.

Usage
-----
    uv run python3 scripts/runlog.py start <task>
    uv run python3 scripts/runlog.py done  <task> [--detail "fetched=90 new=4"]
    uv run python3 scripts/runlog.py check [--days 7]

`check` prints `RUNLOG OK` or `RUNLOG ANOMALY` plus one line per finding, and
always exits 0 -- it is informational and must never abort a digest run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
LOG = ROOT / "ops" / "run-log.tsv"
HEADER = "ts_utc\ttask\tphase\tdetail"

# Cadence of each recurring task.  `daily` tasks are expected to log >= 1 `done`
# on every calendar day (weekends included -- per CLAUDE.md weekends are not
# skipped; for AWS they are the receiving window for late items).  `weekly`
# tasks carry the ISO weekday they are expected on (1 = Monday, 5 = Friday).
TASKS: dict[str, tuple[str, int | None]] = {
    "hf": ("daily", None),           # 07:57 + 17:41; >=1/day is the signal
    "reddit": ("daily", None),       # 08:13
    "tech-blogs": ("daily", None),   # 08:22
    "aws": ("daily", None),          # 09:04
    # 2026-08-24 the cross-digest cron moved Friday -> Monday 09:41 (writing it on
    # Friday misses the weekend, so the ISO week was never fully covered).  This
    # constant was left at 5 until 2026-08-30, and being wrong in a *checker* costs
    # twice: it invented a Friday "NO RUN" every week, and -- far worse -- it never
    # looked at Mondays, so the one weekly task this gate exists to protect could
    # have silently missed its actual day forever.  The 2026-08-22 fix below was
    # therefore inert for the whole time it was pointed at the wrong day.
    "cross-digest": ("weekly", 1),   # Monday 09:41
}


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0)


def _append(task: str, phase: str, detail: str = "") -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    fresh = not LOG.exists()
    ts = _now().strftime("%Y-%m-%d %H:%M")
    with LOG.open("a", encoding="utf-8") as fh:
        if fresh:
            fh.write(HEADER + "\n")
        # Tabs are the field separator, so they must not survive in `detail`.
        fh.write(f"{ts}\t{task}\t{phase}\t{detail.replace(chr(9), ' ')}\n")
    print(f"runlog: {task} {phase} @ {ts}" + (f"  ({detail})" if detail else ""))


def _rows() -> list[tuple[dt.date, str, str, str]]:
    if not LOG.exists():
        return []
    out = []
    for line in LOG.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        ts, task, phase = parts[0], parts[1], parts[2]
        detail = parts[3] if len(parts) > 3 else ""
        try:
            day = dt.datetime.strptime(ts[:10], "%Y-%m-%d").date()
        except ValueError:
            continue
        out.append((day, task, phase, detail))
    return out


def check(days: int) -> int:
    rows = _rows()
    if not rows:
        print("RUNLOG ANOMALY")
        print(f"  no log yet at {LOG.relative_to(ROOT)} -- nothing can be verified")
        return 0

    # Only assert about days actually observed, and observation starts *per
    # task*: a global earliest-row cutoff would flag every task that had not yet
    # been wired up on the day the log was created.  Same error shape as
    # computing a backfill lower bound with a window the run never had (see
    # aws_whats_new.py) -- do not claim absence over an unobserved interval.
    first: dict[str, dt.date] = {}
    for day, task, _phase, _d in rows:
        if task not in first or day < first[task]:
            first[task] = day
    today = _now().date()
    findings: list[str] = []
    unobserved = [t for t in TASKS if t not in first]

    done = {(r[0], r[1]) for r in rows if r[2] == "done"}
    started = {(r[0], r[1]) for r in rows if r[2] == "start"}

    for task, (kind, weekday) in TASKS.items():
        if task not in first:
            continue
        for back in range(days):
            day = today - dt.timedelta(days=back)
            if day < first[task]:
                continue
            if kind == "weekly" and weekday and day.isoweekday() != weekday:
                continue
            if (day, task) in done:
                continue
            if day == today:
                # Today is still in progress for anything scheduled later than
                # now, so absence is not yet evidence.  Handled below instead.
                continue
            if (day, task) in started:
                findings.append(f"  {day} {task}: started but never finished")
            else:
                findings.append(f"  {day} {task}: NO RUN (no start, no done)")

    for day, task in sorted(started - done):
        if day == today:
            findings.append(f"  {day} {task}: start logged, done not yet (in progress?)")

    # A task that has never logged at all cannot be flagged per-day (nothing was
    # observed), which leaves a blind spot: one that dies *before* its first
    # heartbeat stays invisible forever.  Escalate once the log itself has been
    # running long enough that a daily task should plainly have appeared in it.
    # Found by running check on real data 2026-08-20: aws had logged for two
    # days while hf / reddit / tech-blogs had never logged once, and that is a
    # gap, not a neutral "not wired up yet".
    first_row, last_row = min(r[0] for r in rows), max(r[0] for r in rows)
    span_days = (last_row - first_row).days
    for task in unobserved:
        kind, weekday = TASKS[task]
        if kind == "daily" and span_days >= 2:
            findings.append(
                f"  {task}: never logged, while the log spans {span_days}d "
                f"— a daily task should have appeared by now"
            )
        # A weekly task was originally left out of this escalation, which made it
        # permanently invisible: never having logged skips it in the per-day loop
        # above, and being non-daily skipped it here.  Found by running check on
        # real data 2026-08-22 -- the W34 cross-digest was due Friday 08-21 and
        # never ran, yet check reported nothing and merely listed it as a neutral
        # "not yet under observation".  The right gate is not "is it daily" but
        # "has one of its expected days actually elapsed inside the log's span":
        # only then is absence evidence rather than a task that simply is not due
        # yet.  Today is excluded because a run scheduled later today is not late.
        elif kind == "weekly" and weekday:
            due = [first_row + dt.timedelta(days=i)
                   for i in range((last_row - first_row).days + 1)]
            due = [d for d in due if d.isoweekday() == weekday and d != today]
            if due:
                findings.append(
                    f"  {task}: never logged, and its expected day has passed "
                    f"{len(due)}x inside the log's span (latest {max(due)}) "
                    f"— a weekly task should have appeared by now"
                )

    print("RUNLOG ANOMALY" if findings else "RUNLOG OK")
    cov = ", ".join(f"{t} since {first[t]}" for t in sorted(first))
    print(f"  observed: {cov or '(nothing yet)'}  [{len(rows)} rows, window {days}d]")
    if unobserved:
        # Not a gap: never logged means never observed.  Stated so that "not yet
        # wired up" cannot be mistaken for "ran fine".
        print(f"  not yet under observation (no rows ever): {', '.join(sorted(unobserved))}")
    for f in findings:
        print(f)
    if not findings:
        print("  no missed run among the observed tasks in the window")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="heartbeat log for recurring digests")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for phase in ("start", "done"):
        p = sub.add_parser(phase)
        p.add_argument("task", choices=sorted(TASKS))
        p.add_argument("--detail", default="")
    c = sub.add_parser("check")
    c.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    if args.cmd == "check":
        return check(args.days)
    _append(args.task, args.cmd, args.detail)
    return 0


if __name__ == "__main__":
    sys.exit(main())
