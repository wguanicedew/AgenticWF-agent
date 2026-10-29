"""
The issue tracker (docs/cron.md §4, §5): a check's findings become issues, matched across runs by
their identity; issues people work through check boxes; every change an event.

    identity = <cron id> / <check> / <key field>=<value>, ...   values lowercased and trimmed

Only `solved by itself` (the check ran and did not see the issue for `resolve_after` runs) changes
a status without a person.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from sqlalchemy import func, select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import CronFindingRow, CronRunRow, IssueEventRow, IssueRow, ProposalRow, SuppressionRow
from agenticwf.agent.runtime.cron import store
from agenticwf.agent.runtime.cron.definition import Check, Cron
from agenticwf.agent.runtime.cron.store import as_dict, db, new_id

FOLLOWED = ("open", "acknowledged", "in_progress")            # what a check is expected to see again
STATUSES = ("open", "acknowledged", "in_progress", "solved", "ignored", "snoozed")
ACTIONS = ("acknowledge", "take", "solve", "ignore", "snooze", "reopen", "ticket", "comment", "relate", "unrelate")


class TrackerError(ValueError):
    pass


def normalise(value: Any) -> str:
    return " ".join(str(value if value is not None else "").split()).lower()


def identity(cron_id: str, check: str, key: dict[str, str], fields: tuple[str, ...]) -> str:
    return "/".join([cron_id, check, *(f"{f}={normalise(key.get(f))}" for f in fields)])


def suppression_matches(suppression: dict, check: str, key: dict[str, str]) -> bool:
    if suppression.get("check_name") not in (None, check):
        return False
    return all(normalise(key.get(field)) == normalise(value) for field, value in (suppression.get("match_keys") or {}).items())


def _event(session, issue: IssueRow, by: str, action: str, at: dt.datetime, before: str | None = None,
           data: dict | None = None, comment: str | None = None) -> None:
    session.add(IssueEventRow(issue_id=issue.id, cron_id=issue.cron_id, at=db(at), actor=by, action=action,
                              from_status=before, to_status=issue.status, data=data or None, comment=comment or None))


def _expired(issue: IssueRow, now: dt.datetime) -> bool:
    return issue.until is not None and issue.until <= db(now)


# --- after a check ran ------------------------------------------------------------------------------
@dataclass(frozen=True)
class CheckSettings:
    """What matching needs of a cron and its check: from the check task's snapshot, so a run is
    filed with the definition it ran with."""

    cron_id: str
    check: str
    key: tuple[str, ...]
    severities: tuple[str, ...]
    group_by: str
    group_default: str
    resolve_after: int

    @classmethod
    def of(cls, cron: Cron, check: Check) -> "CheckSettings":
        return cls(cron.id, check.name, check.key, check.severities, cron.group_by, cron.group_default,
                   cron.resolve_after_for(check))

    def rank(self, severity: str) -> int:
        return self.severities.index(severity) if severity in self.severities else len(self.severities)


@transactional_session
def apply_check(settings: CheckSettings, run_id: str, ran: bool, findings: list[dict], now: dt.datetime,
                primary_start: dt.datetime | None, record: dict, metrics: list[dict] = (),
                proposals: list[dict] = (), *, session=None) -> dict | None:
    """File one check's outcome of one run, in one transaction: its findings matched to issues
    (`ran`: the check looked, status ok or issues, so issues it did not report count as missed),
    its metrics, its proposals, and `record` as the check's entry in the run. Once per (run,
    check): None when already filed."""
    run = session.get(CronRunRow, run_id)
    if run is None or settings.check in (run.checks or {}):
        return None
    outcome: dict[str, list] = {"new": [], "seen": [], "reopened": [], "solved": [], "suppressed": [], "findings": []}
    by = f"run:{run_id}"
    cron_id, group_by, group_default = settings.cron_id, settings.group_by, settings.group_default
    check_name = settings.check
    since = db(primary_start) if primary_start else None
    active = [as_dict(s) for s in session.execute(select(SuppressionRow).where(SuppressionRow.cron_id == cron_id)).scalars()
              if s.until is None or s.until > db(now)]
    reported: set[str] = set()
    for finding in findings:
        key = {f: " ".join(str((finding.get("key") or {}).get(f, "")).split()) for f in settings.key}
        if not key.get(group_by):
            key[group_by] = group_default
        group = key[group_by]
        ident = identity(cron_id, check_name, key, settings.key)
        severity = finding["severity"]
        rank = settings.rank(severity)
        finding["identity"] = ident
        row = CronFindingRow(id=new_id(), cron_id=cron_id, run_id=run_id, check_name=check_name, identity=ident,
                             grp=group, severity=severity, severity_rank=rank, title=finding["title"],
                             details=finding.get("details"), links=finding.get("links") or [],
                             evidence=finding.get("evidence"), related=finding.get("related") or [], at=db(now))
        session.add(row)
        suppression = next((s for s in active if suppression_matches(s, check_name, key)), None)
        if suppression is not None:
            row.suppressed = suppression["id"]
            outcome["suppressed"].append(ident)
            continue
        issue = session.execute(select(IssueRow).where(IssueRow.identity == ident)).scalars().first()
        if issue is None:
            issue = IssueRow(id=new_id(), cron_id=cron_id, check_name=check_name, identity=ident, key=key, grp=group,
                             title=finding["title"], severity=severity, severity_rank=rank, severity_since=since,
                             status="open", first_seen=db(now), last_seen=db(now), occurrences=1, reopened=0, missed=0,
                             details=finding.get("details"), links=finding.get("links") or [],
                             related=list(finding.get("related") or []), draft_ticket=finding.get("draft_ticket"),
                             notes=[])
            session.add(issue)
            session.flush()
            _event(session, issue, by, "opened", now, None, {"severity": severity})
            outcome["new"].append(issue.id)
        else:
            before = issue.status
            issue.last_seen, issue.occurrences, issue.missed = db(now), issue.occurrences + 1, 0
            issue.title, issue.details = finding["title"], finding.get("details") or issue.details
            issue.links = finding.get("links") or issue.links
            if finding.get("draft_ticket"):
                issue.draft_ticket = finding["draft_ticket"]
            if finding.get("related"):
                issue.related = sorted(set(issue.related or []) | set(finding["related"]))
            if since is None or issue.severity_since is None or issue.severity_since < since:
                changed = rank != issue.severity_rank
                issue.severity, issue.severity_rank, issue.severity_since = severity, rank, since
            else:   # within the period: severity only rises
                changed = rank < issue.severity_rank
                if changed:
                    issue.severity, issue.severity_rank = severity, rank
            if changed:
                _event(session, issue, by, "severity", now, before, {"severity": issue.severity})
            if issue.status == "solved" or (issue.status in ("ignored", "snoozed") and _expired(issue, now)):
                issue.status, issue.until, issue.solved_at, issue.solved_by = "open", None, None, None
                issue.reopened += before == "solved"
                _event(session, issue, by, "reopened", now, before)
                outcome["reopened"].append(issue.id)
            else:
                outcome["seen"].append(issue.id)
        row.issue_id = issue.id
        finding["issue_id"] = issue.id
        reported.add(issue.id)
        outcome["findings"].append(issue.id)
    if ran:
        limit = settings.resolve_after
        for issue in session.execute(select(IssueRow).where(
                IssueRow.cron_id == cron_id, IssueRow.check_name == check_name,
                IssueRow.status.in_(FOLLOWED))).scalars():
            if issue.id in reported:
                continue
            issue.missed += 1
            if issue.missed >= limit:
                before = issue.status
                issue.status, issue.solved_at, issue.solved_by = "solved", db(now), "itself"
                _event(session, issue, by, "solved", now, before, {"missed": issue.missed},
                       "no longer seen by the check")
                outcome["solved"].append(issue.id)
    if metrics:
        store.add_metrics(cron_id, check_name, run_id, list(metrics), now, group_by, session=session)
    for proposal in proposals:   # on the issue it followed, if the check had reported one
        issue_id = None
        if proposal.get("issue_key"):
            key = {f: " ".join(str(proposal["issue_key"].get(f, "")).split()) for f in settings.key}
            key[group_by] = key.get(group_by) or group_default
            found = session.execute(select(IssueRow.id).where(
                IssueRow.identity == identity(cron_id, check_name, key, settings.key))).scalars().first()
            issue_id = found
        store.add_proposal(cron_id, run_id, check_name, issue_id, proposal["tool"], proposal.get("arguments") or {},
                           proposal.get("reason") or "", session=session)
    run.checks = {**(run.checks or {}), check_name: {**record, "new": len(outcome["new"]),
                                                      "solved": len(outcome["solved"]),
                                                      "findings": len(findings)}}
    return outcome


# --- people working issues --------------------------------------------------------------------------
@transactional_session
def act(issue_ids: list[str], action: str, by: str, now: dt.datetime, comment: str = "",
        until: dt.datetime | None = None, ticket: str | None = None, related: str | None = None,
        *, session=None) -> list[dict]:
    """One check box on many issues (the caller has checked who may tick them). Returns them."""
    if action not in ACTIONS:
        raise TrackerError(f"unknown action {action!r}; one of {', '.join(ACTIONS)}")
    if action == "snooze" and until is None:
        raise TrackerError("snooze needs until")
    if action == "ticket" and not ticket:
        raise TrackerError("ticket needs the ticket's id or URL")
    if action == "comment" and not comment.strip():
        raise TrackerError("comment needs text")
    if action in ("relate", "unrelate") and not related:
        raise TrackerError(f"{action} needs the related issue's id")
    changed = []
    for issue in session.execute(select(IssueRow).where(IssueRow.id.in_(issue_ids))).scalars():
        before, data = issue.status, None
        if action == "acknowledge":
            issue.status, issue.owner = "acknowledged", by
        elif action == "take":
            issue.status, issue.owner = "in_progress", by
        elif action == "solve":
            issue.status, issue.solved_at, issue.solved_by, issue.missed = "solved", db(now), by, 0
        elif action == "ignore":
            issue.status, issue.until = "ignored", db(until)
        elif action == "snooze":
            issue.status, issue.until = "snoozed", db(until)
        elif action == "reopen":
            issue.status, issue.until, issue.solved_at, issue.solved_by = "open", None, None, None
        elif action == "ticket":
            issue.ticket, data = ticket, {"ticket": ticket}
        elif action == "comment":
            issue.notes = [*(issue.notes or []), {"by": by, "at": now.isoformat(), "text": comment.strip()}]
        elif action == "relate":
            issue.related, data = sorted(set(issue.related or []) | {related}), {"related": related}
        elif action == "unrelate":
            issue.related, data = [r for r in (issue.related or []) if r != related], {"related": related}
        if comment.strip() and action != "comment":   # a comment with any box reaches the next run too
            issue.notes = [*(issue.notes or []), {"by": by, "at": now.isoformat(), "text": comment.strip()}]
        _event(session, issue, by, action, now, before, data, comment)
        changed.append(as_dict(issue))
    return changed


# --- reading ------------------------------------------------------------------------------------------
def _view(issue: dict) -> dict:
    return {k: (v.isoformat() if isinstance(v, dt.datetime) else v) for k, v in issue.items() if k != "identity"}


@read_session
def issues(cron_ids: list[str], status: list[str] | None = None, check: str | None = None, site: str | None = None,
           severity: str | None = None, owner: str | None = None, since: dt.datetime | None = None,
           recurring: bool = False, related: bool = False, limit: int = 500, *, session=None) -> list[dict]:
    if not cron_ids:
        return []
    query = select(IssueRow).where(IssueRow.cron_id.in_(cron_ids))
    if status:
        query = query.where(IssueRow.status.in_(status))
    if check:
        query = query.where(IssueRow.check_name == check)
    if site:
        query = query.where(func.lower(IssueRow.grp) == normalise(site))
    if severity:
        query = query.where(IssueRow.severity == severity)
    if owner:
        query = query.where(IssueRow.owner == owner)
    if since:
        query = query.where(IssueRow.first_seen >= db(since))
    if recurring:
        query = query.where(IssueRow.reopened > 0)
    rows = session.execute(query.order_by(IssueRow.severity_rank, IssueRow.first_seen).limit(limit)).scalars()
    found = [_view(as_dict(r)) for r in rows]
    return [i for i in found if i.get("related")] if related else found


def by_site(found: list[dict]) -> list[dict]:
    """Issues grouped by site, worst first: each site with its worst open severity and its issues."""
    sites: dict[str, dict] = {}
    for issue in found:
        group = issue.get("grp", issue.get("site"))
        entry = sites.setdefault(normalise(group), {"site": group, "issues": [], "worst_rank": 99,
                                                           "worst": None, "since": issue["first_seen"]})
        entry["issues"].append(issue)
        if issue["status"] in FOLLOWED and issue["severity_rank"] < entry["worst_rank"]:
            entry["worst_rank"], entry["worst"] = issue["severity_rank"], issue["severity"]
        entry["since"] = min(entry["since"], issue["first_seen"])
    return sorted(sites.values(), key=lambda s: (s["worst_rank"], s["since"]))


@read_session
def issue(issue_id: str, *, session=None) -> dict | None:
    row = session.get(IssueRow, issue_id)
    if row is None:
        return None
    view = _view(as_dict(row))
    view["events"] = [_view(as_dict(e)) for e in session.execute(
        select(IssueEventRow).where(IssueEventRow.issue_id == issue_id).order_by(IssueEventRow.id)).scalars()]
    view["findings"] = [_view(as_dict(f)) for f in session.execute(
        select(CronFindingRow).where(CronFindingRow.issue_id == issue_id).order_by(CronFindingRow.at.desc()).limit(20)).scalars()]
    view["proposals"] = [_view(as_dict(p)) for p in session.execute(
        select(ProposalRow).where(ProposalRow.issue_id == issue_id).order_by(ProposalRow.created_at)).scalars()]
    return view


@read_session
def cron_of(issue_ids: list[str], *, session=None) -> dict[str, str]:
    return dict(session.execute(select(IssueRow.id, IssueRow.cron_id).where(IssueRow.id.in_(issue_ids))).all())


@read_session
def check_context(cron: Cron, check: Check, now: dt.datetime, *, session=None) -> dict:
    """What a check task is told: its open issues, other checks' at the same sites, suppressions."""
    own = [as_dict(r) for r in session.execute(select(IssueRow).where(
        IssueRow.cron_id == cron.id, IssueRow.check_name == check.name,
        IssueRow.status.in_((*FOLLOWED, "snoozed", "ignored"))).order_by(IssueRow.severity_rank, IssueRow.first_seen)).scalars()]
    groups = {normalise(i["grp"]) for i in own if i["status"] in FOLLOWED}
    others = []
    if groups:
        others = [as_dict(r) for r in session.execute(select(IssueRow).where(
            IssueRow.cron_id == cron.id, IssueRow.check_name != check.name, IssueRow.status.in_(FOLLOWED),
            func.lower(IssueRow.grp).in_(groups))).scalars()]
    suppressed = [as_dict(s) for s in session.execute(select(SuppressionRow).where(SuppressionRow.cron_id == cron.id)).scalars()
                  if s.check_name in (None, check.name) and (s.until is None or s.until > db(now))]
    return {"issues": own, "others": others, "suppressions": suppressed}


@read_session
def in_range(cron_id: str, start: dt.datetime, end: dt.datetime, *, session=None) -> dict:
    """What a report over [start, end) is built from: issues touched in the range, their events and findings."""
    a, b = db(start), db(end)
    events = [as_dict(e) for e in session.execute(select(IssueEventRow).where(
        IssueEventRow.cron_id == cron_id, IssueEventRow.at >= a, IssueEventRow.at < b).order_by(IssueEventRow.id)).scalars()]
    findings = [as_dict(f) for f in session.execute(select(CronFindingRow).where(
        CronFindingRow.cron_id == cron_id, CronFindingRow.at >= a, CronFindingRow.at < b)).scalars()]
    ids = {e["issue_id"] for e in events} | {f["issue_id"] for f in findings if f["issue_id"]}
    open_at_end = [as_dict(r) for r in session.execute(select(IssueRow).where(
        IssueRow.cron_id == cron_id, IssueRow.first_seen < b, IssueRow.status.in_((*FOLLOWED, "snoozed")))).scalars()]
    ids |= {i["id"] for i in open_at_end}
    issues_ = {r.id: as_dict(r) for r in session.execute(select(IssueRow).where(IssueRow.id.in_(ids))).scalars()} if ids else {}
    return {"issues": issues_, "events": events, "findings": findings}


@transactional_session
def note(issue_id: str, by: str, action: str, now: dt.datetime, data: dict | None = None, comment: str = "",
         *, session=None) -> None:
    """An event on an issue that changes nothing else (a proposal approved or rejected)."""
    row = session.get(IssueRow, issue_id)
    if row is not None:
        _event(session, row, by, action, now, row.status, data, comment)
