"""
Reports (docs/cron.md §8): built from the issue tracker, the runs and the metrics, never by
checking the services again. Every list is built here, so it is exact; only the summary is
written by a model (agenticwf.agent.runtime.cron.service). Each report is kept in the database as data and
Markdown.

    period report   one period ([start, end)): sites, new, solved, still open, recurring, ignored,
                    proposed actions, actions taken, numbers, coverage
    rollup          weekly / monthly / any range: per day, time to solve, longest, recurring,
                    sites, coverage, proposals, trend series, and the period reports it covers
"""

from __future__ import annotations

import datetime as dt
import statistics
from typing import Any

from agenticwf.agent.runtime.cron import store, tracker
from agenticwf.agent.runtime.cron.definition import Cron
from agenticwf.agent.runtime.cron.schedule import UTC

DONE = ("solved", "ignored")


def _t(value) -> str | None:
    if value is None:
        return None
    return value.isoformat() if isinstance(value, dt.datetime) else str(value)


def _local(value, cron: Cron) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        value = dt.datetime.fromisoformat(value)
    value = value if value.tzinfo else value.replace(tzinfo=UTC)
    return value.astimezone(cron.tz).strftime("%Y-%m-%d %H:%M")


def _issue_view(issue: dict, rank_in_range: dict[str, tuple[int, str]]) -> dict:
    worst = rank_in_range.get(issue["id"])
    return {"id": issue["id"], "check": issue["check_name"], "site": issue["grp"], "title": issue["title"],
            "severity": worst[1] if worst else issue["severity"], "severity_rank": worst[0] if worst else issue["severity_rank"],
            "status": issue["status"], "first_seen": _t(issue["first_seen"]), "last_seen": _t(issue["last_seen"]),
            "solved_at": _t(issue["solved_at"]), "solved_by": issue["solved_by"], "owner": issue["owner"],
            "ticket": issue["ticket"], "links": issue.get("links") or [], "related": issue.get("related") or [],
            "reopened": issue["reopened"]}


def _runs_coverage(cron: Cron, runs: list[dict]) -> dict:
    per_check: dict[str, dict[str, Any]] = {c.name: {"due": 0, "ran": 0, "degraded": 0, "unable": 0, "failed": 0,
                                                     "skipped": 0, "reasons": []} for c in cron.checks}
    for run in runs:
        for name in run.get("skipped") or ():
            per_check.setdefault(name, {"due": 0, "ran": 0, "degraded": 0, "unable": 0, "failed": 0, "skipped": 0,
                                        "reasons": []})["skipped"] += 1
        for name in run.get("due") or ():
            entry = per_check.setdefault(name, {"due": 0, "ran": 0, "degraded": 0, "unable": 0, "failed": 0,
                                                "skipped": 0, "reasons": []})
            entry["due"] += 1
            result = (run.get("checks") or {}).get(name) or {}
            status = result.get("status")
            if status in ("ok", "issues"):
                entry["ran"] += 1
            elif status in ("degraded", "unable"):
                entry[status] += 1
            elif run.get("status") != "running":
                entry["failed"] += 1
                if result.get("error"):
                    entry["reasons"] = list(dict.fromkeys([*entry["reasons"], result["error"]]))[-3:]
    return {"runs": len(runs), "checks": per_check}


def period(cron: Cron, start: dt.datetime, end: dt.datetime, effects: list[dict] | None = None) -> dict:
    """The data of a period report over [start, end)."""
    touched = tracker.in_range(cron.id, start, end)
    issues, events, findings = touched["issues"], touched["events"], touched["findings"]
    worst: dict[str, tuple[int, str]] = {}
    for f in findings:
        if f["issue_id"] and (f["issue_id"] not in worst or f["severity_rank"] < worst[f["issue_id"]][0]):
            worst[f["issue_id"]] = (f["severity_rank"], f["severity"])
    view = {i: _issue_view(issue, worst) for i, issue in issues.items()}
    a, b = start, end

    def within(t) -> bool:
        return t is not None and a <= t < b

    new = [view[i] for i, x in issues.items() if within(x["first_seen"])]
    solved = []
    for e in events:
        if e["action"] == "solved" and e["issue_id"] in view:
            item = dict(view[e["issue_id"]])
            item["solved_by"], item["solved_at"] = ("itself" if e["actor"].startswith("run:") else e["actor"]), _t(e["at"])
            first = issues[e["issue_id"]]["first_seen"]
            item["minutes_to_solve"] = round((e["at"] - first).total_seconds() / 60) if first else None
            solved.append(item)
    still_open = [view[i] for i, x in issues.items() if x["status"] not in DONE and x["first_seen"] < b]
    recurring = [view[e["issue_id"]] for e in events if e["action"] == "reopened" and e["issue_id"] in view]
    ignored = [{**view[e["issue_id"]], "by": e["actor"], "until": (e.get("data") or {}).get("until") or _t(issues[e["issue_id"]]["until"]),
                "comment": e.get("comment"), "action": e["action"]}
               for e in events if e["action"] in ("ignore", "snooze") and e["issue_id"] in view]
    suppressions = [s for s in store.suppressions(cron.id) if within(s["created_at"]) or s["until"] is None or (s["until"] and s["until"] > a)]
    proposals = store.proposals(cron.id, start=a, end=b)
    numbers = store.metric_stats(cron.id, a, b)
    before = {(n["name"], str(sorted(n["labels"].items()))): n for n in store.metric_stats(cron.id, a - (b - a), a)}
    for n in numbers:
        prev = before.get((n["name"], str(sorted(n["labels"].items()))))
        n["previous_mean"] = prev["mean"] if prev else None
    runs = store.runs(cron.id, limit=10_000, start=a, end=b)
    sites = tracker.by_site(list(view.values())) if view else []
    for site in sites:   # the period's worst, not the current one
        ranked = [i for i in site["issues"] if i["id"] in worst] or site["issues"]
        best = min(ranked, key=lambda i: i["severity_rank"])
        site["worst"], site["worst_rank"] = best["severity"], best["severity_rank"]
    sites.sort(key=lambda s: (s["worst_rank"], s["since"]))
    return {"kind": "period", "cron": cron.id, "start": _t(start), "end": _t(end), "timezone": cron.timezone,
            "sites": [{"site": s["site"], "worst": s["worst"], "since": s["since"],
                       "issues": [i["id"] for i in s["issues"]]} for s in sites],
            "issues": view, "new": new, "solved": solved, "open": still_open, "recurring": recurring,
            "ignored": ignored, "suppressions": [{"id": s["id"], "check": s["check_name"], "match": s["match_keys"],
                                                   "reason": s["reason"], "by": s["actor"], "until": _t(s["until"])}
                                                  for s in suppressions],
            "proposals": [{k: _t(v) if isinstance(v, dt.datetime) else v for k, v in p.items()} for p in proposals],
            "actions": effects or [], "numbers": numbers, "coverage": _runs_coverage(cron, runs)}


def rollup(cron: Cron, start: dt.datetime, end: dt.datetime, of: str | None) -> dict:
    """The data of a rollup over [start, end), from the period reports of `of`, the issue events and the metrics."""
    touched = tracker.in_range(cron.id, start, end)
    issues, events = touched["issues"], touched["events"]
    view = {i: _issue_view(x, {}) for i, x in issues.items()}
    tz = cron.tz
    days: dict[str, dict[str, int]] = {}
    day = start.astimezone(tz).date()
    while dt.datetime(day.year, day.month, day.day, tzinfo=tz).astimezone(UTC) < end:
        days[day.isoformat()] = {"new": 0, "solved": 0, "open": 0}
        day += dt.timedelta(days=1)
    for x in issues.values():
        if start <= x["first_seen"] < end:
            key = x["first_seen"].astimezone(tz).date().isoformat()
            if key in days:
                days[key]["new"] += 1
    solve_minutes = []
    for e in events:
        if e["action"] == "solved":
            key = e["at"].astimezone(tz).date().isoformat()
            if key in days:
                days[key]["solved"] += 1
            first = issues.get(e["issue_id"], {}).get("first_seen")
            if first:
                solve_minutes.append((e["at"] - first).total_seconds() / 60)
    for key in days:
        d = dt.date.fromisoformat(key) + dt.timedelta(days=1)
        day_end = dt.datetime(d.year, d.month, d.day, tzinfo=tz).astimezone(UTC)
        days[key]["open"] = sum(1 for x in issues.values() if x["first_seen"] < day_end
                                and (x["solved_at"] is None or x["solved_at"] >= day_end) and x["status"] != "ignored")

    def duration(x: dict) -> float:
        return ((x["solved_at"] or end) - x["first_seen"]).total_seconds() / 60

    longest = sorted(issues.values(), key=duration, reverse=True)[:10]
    by_site: dict[str, int] = {}
    for x in issues.values():
        by_site[x["grp"]] = by_site.get(x["grp"], 0) + 1
    p90 = (statistics.quantiles(solve_minutes, n=10)[-1] if len(solve_minutes) >= 2 else
           (solve_minutes[0] if solve_minutes else None))
    periods, _ = store.reports([cron.id], kind=of, latest_only=True, limit=1000) if of else ([], 0)
    periods = [r for r in periods if r["status"] == "ready" and start <= r["start_at"] < end]
    proposals = store.proposals(cron.id, start=start, end=end)
    resolution = "day" if (end - start) > dt.timedelta(days=2) else "hour"
    return {"kind": "rollup", "cron": cron.id, "start": _t(start), "end": _t(end), "timezone": cron.timezone, "of": of,
            "days": days,
            "time_to_solve": {"count": len(solve_minutes),
                              "median_minutes": round(statistics.median(solve_minutes)) if solve_minutes else None,
                              "p90_minutes": round(p90) if p90 is not None else None},
            "longest": [{**view[x["id"]], "minutes": round(duration(x))} for x in longest],
            "recurring": [view[x["id"]] for x in sorted(issues.values(), key=lambda x: -x["reopened"]) if x["reopened"]][:10],
            "still_open_from_before": [view[i] for i, x in issues.items() if x["status"] not in DONE and x["first_seen"] < start],
            "sites": [{"site": s, "issues": n} for s, n in sorted(by_site.items(), key=lambda t: (-t[1], t[0]))],
            "coverage": _runs_coverage(cron, store.runs(cron.id, limit=100_000, start=start, end=end)),
            "proposals": {s: sum(1 for p in proposals if p["status"] == s) for s in ("pending", "approved", "rejected", "failed")},
            "series": store.metric_series(cron.id, None, start, end, resolution=resolution),
            "resolution": resolution,
            "reports": [{"id": r["id"], "start": _t(r["start_at"]), "end": _t(r["end_at"]), "summary": r["summary"]}
                        for r in sorted(periods, key=lambda r: r["start_at"])],
            "issues": view}


# --- Markdown ---------------------------------------------------------------------------------------
def _box(issue: dict) -> str:
    return "[x]" if issue["status"] in DONE else "[ ]"


def _line(issue: dict, cron: Cron, extra: str = "") -> str:
    links = " ".join(f"[{n}]({u})" for n, u in enumerate(issue.get("links") or [], 1))
    bits = [f"**{issue['severity']}**", issue["title"], f"({issue['check']}, {issue['status'].replace('_', ' ')}"
            + (f", since {_local(issue['first_seen'], cron)}" if issue.get("first_seen") else "") + ")"]
    if issue.get("owner"):
        bits.append(f"owner {issue['owner']}")
    if issue.get("ticket"):
        bits.append(f"ticket {issue['ticket']}")
    if extra:
        bits.append(extra)
    if links:
        bits.append(links)
    return f"- {_box(issue)} " + " · ".join(bits)


def markdown(cron: Cron, data: dict, title: str, summary: str) -> str:
    out = [f"# {title}", "", f"{_local(data['start'], cron)} – {_local(data['end'], cron)} ({cron.timezone})", ""]
    if summary:
        out += ["## Summary", "", summary.strip(), ""]
    if data["kind"] == "period":
        issues = data["issues"]
        if data["sites"]:
            out += ["## Sites", ""]
            for site in data["sites"]:
                out.append(f"### {site['site']} ({site['worst']})")
                out += [_line(issues[i], cron) for i in site["issues"]]
                out.append("")
        sections = [("New issues", data["new"], ""), ("Solved", data["solved"], "solved"),
                    ("Still open", data["open"], ""), ("Recurring", data["recurring"], "")]
        for heading, items, kind in sections:
            out += [f"## {heading}", ""]
            if not items:
                out += ["None.", ""]
                continue
            for item in items:
                extra = (f"by {item['solved_by']}, after {item['minutes_to_solve']} min"
                         if kind == "solved" and item.get("minutes_to_solve") is not None else "")
                out.append(_line(item, cron, extra))
            out.append("")
        out += ["## Ignored and suppressed", ""]
        out += [_line(i, cron, f"{i['action']} by {i['by']}" + (f" until {_local(i['until'], cron)}" if i.get("until") else ""))
                for i in data["ignored"]] or []
        out += [f"- suppression {', '.join(f'{k}={v}' for k, v in (s['match'] or {}).items())}"
                f"{' (' + s['check'] + ')' if s['check'] else ''} by {s['by']}: {s['reason'] or ''}"
                for s in data["suppressions"]]
        if not data["ignored"] and not data["suppressions"]:
            out.append("None.")
        out += ["", "## Proposed actions", ""]
        out += [f"- {'[x]' if p['status'] == 'approved' else '[ ]'} `{p['tool']}` ({p['status']}): {p.get('reason') or ''}"
                for p in data["proposals"]] or ["None."]
        out += ["", "## Actions taken", ""]
        out += [f"- `{e.get('tool')}` on {e.get('target')} ({e.get('status')})" for e in data["actions"]] or ["None."]
        out += ["", "## Numbers", ""]
        if data["numbers"]:
            out += ["| Metric | Labels | Min | Mean | Max | Before | Source |", "|---|---|---|---|---|---|---|"]
            for n in data["numbers"]:
                labels = ", ".join(f"{k}={v}" for k, v in sorted(n["labels"].items()))
                before = f"{n['previous_mean']:.3g}" if n.get("previous_mean") is not None else ""
                links = " ".join(f"[{i}]({u})" for i, u in enumerate(n["links"], 1))
                out.append(f"| {n['name']} | {labels} | {n['min']:.3g} | {n['mean']:.3g} | {n['max']:.3g} | {before} | {links} |")
        else:
            out.append("None recorded.")
    else:
        t = data["time_to_solve"]
        out += ["## Per day", "", "| Day | New | Solved | Open at end |", "|---|---|---|---|"]
        out += [f"| {d} | {c['new']} | {c['solved']} | {c['open']} |" for d, c in data["days"].items()]
        out += ["", "## Time to solve", "",
                f"{t['count']} solved; median {t['median_minutes']} min, 90th percentile {t['p90_minutes']} min."
                if t["count"] else "Nothing solved.", ""]
        for heading, items in (("Longest", data["longest"]), ("Recurring", data["recurring"]),
                               ("Still open from before", data["still_open_from_before"])):
            out += [f"## {heading}", ""]
            out += [_line(i, cron) for i in items] or ["None."]
            out.append("")
        out += ["## Sites with most issues", ""]
        out += [f"- {s['site']}: {s['issues']}" for s in data["sites"][:20]] or ["None."]
        out += ["", "## Proposed actions", "", ", ".join(f"{k} {v}" for k, v in data["proposals"].items()), ""]
        out += ["## Period reports", ""]
        out += [f"- {_local(r['start'], cron)}: {(r['summary'] or '').splitlines()[0] if r['summary'] else ''}"
                for r in data["reports"]] or ["None."]
    out += ["", "## Coverage", ""]
    coverage = data["coverage"]
    out.append(f"{coverage['runs']} runs.")
    for name, c in coverage["checks"].items():
        problems = ", ".join(f"{k} {c[k]}" for k in ("degraded", "unable", "failed", "skipped") if c[k])
        out.append(f"- {name}: {c['ran']} of {c['due']} ran" + (f"; {problems}" if problems else "")
                   + (f" ({'; '.join(c['reasons'])})" if c["reasons"] else ""))
    return "\n".join(out).rstrip() + "\n"


def sites_of(data: dict) -> list[str]:
    """The sites (groups) a report mentions, for the reports page's site filter."""
    return [s["site"] for s in data["sites"]]
