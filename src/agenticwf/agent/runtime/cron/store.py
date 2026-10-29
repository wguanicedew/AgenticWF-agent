"""
The crons' tables (agenticwf.agent.core.orm.models.agent, docs/cron.md §9): schedule state and leases, runs,
database crons, metrics with their aggregates, reports, proposals, suppressions, retention.
The issue tracker's own writes are in agenticwf.agent.runtime.cron.tracker.

Times are aware UTC datetimes here and naive UTC in the tables, like the rest of agenticwf.agent.core.orm.
"""

from __future__ import annotations

import datetime as dt
import json
import uuid
from typing import Any

from sqlalchemy import and_, delete, func, or_, select, update

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import (CronFileRow, CronFindingRow, CronMetricAggregateRow, CronMetricRow, CronRow,
                                   CronRunRow, IssueEventRow, IssueRow, ProposalRow, ReportRow, SuppressionRow)
from agenticwf.agent.runtime.cron.schedule import UTC

LINKS_KEPT = 5   # links of the original data kept per aggregate bucket


def new_id() -> str:
    return uuid.uuid4().hex


def db(t: dt.datetime | None) -> dt.datetime | None:
    return t.astimezone(UTC).replace(tzinfo=None) if t else None


def aware(t: dt.datetime | None) -> dt.datetime | None:
    return t.replace(tzinfo=UTC) if t else None


def iso(t: dt.datetime | None) -> str | None:
    return aware(t).isoformat() if t and t.tzinfo is None else (t.isoformat() if t else None)


def parse_iso(text: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(text) if text else None


def as_dict(row) -> dict[str, Any]:
    out = {}
    for column in row.__table__.columns:
        value = getattr(row, column.key)
        out[column.key] = aware(value) if isinstance(value, dt.datetime) else value
    return out


# --- schedule state ---------------------------------------------------------------------------------
@read_session
def state(cron_id: str, *, session=None) -> dict | None:
    row = session.get(CronRow, cron_id)
    return as_dict(row) if row else None


@read_session
def states(*, session=None) -> dict[str, dict]:
    return {r.id: as_dict(r) for r in session.execute(select(CronRow)).scalars()}


@transactional_session
def ensure_state(cron_id: str, source: str, owner=None, *, session=None) -> dict:
    row = session.get(CronRow, cron_id)
    if row is None:
        row = CronRow(id=cron_id, source=source, owner_subject=owner[0] if owner else None,
                      owner_org=owner[1] if owner else None, checks_due={}, checks_last={}, periods_done={})
        session.add(row)
        session.flush()
    return as_dict(row)


@transactional_session
def set_state(cron_id: str, *, session=None, **fields) -> None:
    row = session.get(CronRow, cron_id)
    if row is None:
        return
    for key, value in fields.items():
        setattr(row, key, db(value) if isinstance(value, dt.datetime) else value)


@transactional_session
def lease(cron_id: str, holder: str, until: dt.datetime, now: dt.datetime, *, session=None) -> bool:
    """Take the cron's lease if nobody holds a live one (any database: a conditional update)."""
    result = session.execute(update(CronRow).where(
        CronRow.id == cron_id,
        or_(CronRow.lease_until.is_(None), CronRow.lease_until < db(now), CronRow.lease_owner == holder))
        .values(lease_owner=holder, lease_until=db(until)))
    return bool(result.rowcount)


@transactional_session
def release(cron_id: str, holder: str, *, session=None) -> None:
    session.execute(update(CronRow).where(CronRow.id == cron_id, CronRow.lease_owner == holder)
                    .values(lease_owner=None, lease_until=None))


# --- database crons ---------------------------------------------------------------------------------
@read_session
def database_crons(*, session=None) -> list[tuple[dict, dict[str, str]]]:
    rows = session.execute(select(CronRow).where(CronRow.source == "database")).scalars().all()
    files: dict[str, dict[str, str]] = {}
    for f in session.execute(select(CronFileRow)).scalars():
        files.setdefault(f.cron_id, {})[f.path] = f.content
    return [(as_dict(r), files.get(r.id, {})) for r in rows]


@transactional_session
def save_database_cron(cron_id: str, owner, files: dict[str, str] | None, visibility: str, scopes: list[str],
                       enabled: bool | None = None, *, session=None) -> None:
    """Add or change a database cron. `files` replace all its files (None: keep them)."""
    row = session.get(CronRow, cron_id)
    if row is not None and (row.owner_subject, row.owner_org) != tuple(owner):
        raise PermissionError(f"cron {cron_id!r} belongs to someone else")
    if row is None:
        row = CronRow(id=cron_id, source="database", checks_due={}, checks_last={}, periods_done={})
        session.add(row)
    row.owner_subject, row.owner_org = owner
    row.visibility, row.owner_scopes = visibility, sorted(scopes)
    if enabled is not None:
        row.enabled = int(enabled)
    if files is not None:
        session.execute(delete(CronFileRow).where(CronFileRow.cron_id == cron_id))
        for path, content in files.items():
            session.add(CronFileRow(cron_id=cron_id, path=path, content=content))


@transactional_session
def delete_cron(cron_id: str, *, session=None) -> bool:
    session.execute(delete(CronFileRow).where(CronFileRow.cron_id == cron_id))
    row = session.get(CronRow, cron_id)
    if row is None:
        return False
    session.delete(row)
    return True


@read_session
def enabled_user_crons(*, session=None) -> list[dict]:
    rows = session.execute(select(CronRow).where(CronRow.source == "user", CronRow.enabled == 1)).scalars()
    return [as_dict(r) for r in rows]


# --- runs ---------------------------------------------------------------------------------------------
@transactional_session
def create_run(cron_id: str, due: list[str], skipped: list[str], started: dt.datetime, *, session=None) -> str:
    run_id = new_id()
    session.add(CronRunRow(id=run_id, cron_id=cron_id, status="running", started_at=db(started), due=due,
                           skipped=skipped, checks={}))
    return run_id


@transactional_session
def update_run(run_id: str, *, session=None, **fields) -> None:
    row = session.get(CronRunRow, run_id)
    if row is not None:
        for key, value in fields.items():
            setattr(row, key, db(value) if isinstance(value, dt.datetime) else value)


@read_session
def run(run_id: str, *, session=None) -> dict | None:
    row = session.get(CronRunRow, run_id)
    return as_dict(row) if row else None


@read_session
def run_of_intent(intent_id: str, *, session=None) -> dict | None:
    row = session.execute(select(CronRunRow).where(CronRunRow.intent_id == intent_id)).scalars().first()
    return as_dict(row) if row else None


@read_session
def runs(cron_id: str, limit: int = 50, start: dt.datetime | None = None, end: dt.datetime | None = None,
         *, session=None) -> list[dict]:
    query = select(CronRunRow).where(CronRunRow.cron_id == cron_id)
    if start:
        query = query.where(CronRunRow.started_at >= db(start))
    if end:
        query = query.where(CronRunRow.started_at < db(end))
    rows = session.execute(query.order_by(CronRunRow.started_at.desc()).limit(limit)).scalars()
    return [as_dict(r) for r in rows]


@read_session
def running_checks(cron_id: str, *, session=None) -> set[str]:
    """Checks of the cron's unfinished runs whose task has not ended."""
    found: set[str] = set()
    for row in session.execute(select(CronRunRow).where(CronRunRow.cron_id == cron_id,
                                                        CronRunRow.status == "running")).scalars():
        done = row.checks or {}
        found.update(c for c in (row.due or ()) if c not in done)
    return found


# --- metrics ------------------------------------------------------------------------------------------
def _bucket(t: dt.datetime, resolution: str) -> dt.datetime:
    t = t.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return t.replace(hour=0) if resolution == "day" else t


@transactional_session
def add_metrics(cron_id: str, check: str, run_id: str, values: list[dict], at: dt.datetime, group_by: str,
                *, session=None) -> None:
    """Raw values, and the hourly and daily aggregates they fall into."""
    for value in values:
        labels = dict(value.get("labels") or {})
        group = str(labels.get(group_by, "")).strip() or None
        session.add(CronMetricRow(cron_id=cron_id, check_name=check, run_id=run_id, name=value["name"],
                                  value=float(value["value"]), unit=value.get("unit") or None, labels=labels,
                                  grp=group, link=value.get("link") or None, at=db(at)))
        series = json.dumps(labels, sort_keys=True)
        for resolution in ("hour", "day"):
            bucket = db(_bucket(at, resolution))
            row = session.get(CronMetricAggregateRow, (cron_id, value["name"], resolution, bucket, series))
            number = float(value["value"])
            if row is None:
                row = CronMetricAggregateRow(cron_id=cron_id, name=value["name"], resolution=resolution, bucket=bucket,
                                             series=series, labels=labels, grp=group, unit=value.get("unit") or None,
                                             count=1, min=number, max=number, sum=number, last=number,
                                             links=[value["link"]] if value.get("link") else [])
                session.add(row)
                session.flush()
                continue
            row.count, row.min, row.max = row.count + 1, min(row.min, number), max(row.max, number)
            row.sum, row.last = row.sum + number, number
            if value.get("link") and value["link"] not in (row.links or []):
                row.links = [*(row.links or []), value["link"]][-LINKS_KEPT:]


@read_session
def metric_series(cron_id: str, name: str | None, start: dt.datetime, end: dt.datetime, resolution: str = "hour",
                  group: str | None = None, *, session=None) -> list[dict]:
    query = select(CronMetricAggregateRow).where(
        CronMetricAggregateRow.cron_id == cron_id, CronMetricAggregateRow.resolution == resolution,
        CronMetricAggregateRow.bucket >= db(_bucket(start, resolution)), CronMetricAggregateRow.bucket < db(end))
    if name:
        query = query.where(CronMetricAggregateRow.name == name)
    if group:
        query = query.where(func.lower(CronMetricAggregateRow.grp) == group.strip().lower())
    rows = session.execute(query.order_by(CronMetricAggregateRow.name, CronMetricAggregateRow.series,
                                          CronMetricAggregateRow.bucket)).scalars()
    return [{"name": r.name, "labels": r.labels or {}, "group": r.grp, "unit": r.unit, "bucket": iso(r.bucket),
             "count": r.count, "min": r.min, "max": r.max, "mean": r.sum / r.count if r.count else None,
             "last": r.last, "links": r.links or []} for r in rows]


@read_session
def metric_stats(cron_id: str, start: dt.datetime, end: dt.datetime, *, session=None) -> list[dict]:
    """Per metric and labels over [start, end): min, mean, max, count, links (from the hourly aggregates)."""
    rows = session.execute(select(CronMetricAggregateRow).where(
        CronMetricAggregateRow.cron_id == cron_id, CronMetricAggregateRow.resolution == "hour",
        CronMetricAggregateRow.bucket >= db(_bucket(start, "hour")), CronMetricAggregateRow.bucket < db(end))).scalars()
    stats: dict[tuple[str, str], dict] = {}
    for r in rows:
        item = stats.setdefault((r.name, r.series), {"name": r.name, "labels": r.labels or {}, "group": r.grp,
                                                       "unit": r.unit, "count": 0, "sum": 0.0, "min": r.min,
                                                       "max": r.max, "links": []})
        item["count"] += r.count
        item["sum"] += r.sum
        item["min"], item["max"] = min(item["min"], r.min), max(item["max"], r.max)
        item["links"] = list(dict.fromkeys([*item["links"], *(r.links or [])]))[-LINKS_KEPT:]
    out = []
    for item in stats.values():
        item["mean"] = item.pop("sum") / item["count"] if item["count"] else None
        out.append(item)
    return sorted(out, key=lambda i: (i["name"], json.dumps(i["labels"], sort_keys=True)))


# --- reports ------------------------------------------------------------------------------------------
@transactional_session
def create_report(cron_id: str, kind: str, rollup: bool, start: dt.datetime, end: dt.datetime, timezone: str,
                  local_start: dt.date, template: str, created_by: str, *, session=None) -> str:
    version = session.execute(select(func.max(ReportRow.version)).where(
        ReportRow.cron_id == cron_id, ReportRow.kind == kind, ReportRow.start_at == db(start),
        ReportRow.end_at == db(end))).scalar() or 0
    report_id = new_id()
    session.add(ReportRow(id=report_id, cron_id=cron_id, kind=kind, rollup=int(rollup), start_at=db(start),
                          end_at=db(end), timezone=timezone, year=local_start.year, month=local_start.month,
                          day=local_start.day, version=version + 1, status="pending", template=template or None,
                          created_by=created_by))
    return report_id


@transactional_session
def save_report(report_id: str, *, session=None, **fields) -> None:
    row = session.get(ReportRow, report_id)
    if row is not None:
        for key, value in fields.items():
            setattr(row, key, value)


@read_session
def report(report_id: str, *, session=None) -> dict | None:
    row = session.get(ReportRow, report_id)
    return as_dict(row) if row else None


@read_session
def reports(cron_ids: list[str], kind: str | None = None, year: int | None = None, month: int | None = None,
            day: int | None = None, site: str | None = None, text: str | None = None, latest_only: bool = True,
            limit: int = 200, offset: int = 0, *, session=None) -> tuple[list[dict], int]:
    if not cron_ids:
        return [], 0
    query = select(ReportRow).where(ReportRow.cron_id.in_(cron_ids), ReportRow.status != "pending")
    for column, value in ((ReportRow.kind, kind), (ReportRow.year, year), (ReportRow.month, month),
                          (ReportRow.day, day)):
        if value is not None:
            query = query.where(column == value)
    if text:
        query = query.where(func.lower(ReportRow.summary).contains(text.lower()))
    rows = session.execute(query.order_by(ReportRow.start_at.desc(), ReportRow.kind, ReportRow.version.desc())).scalars().all()
    if latest_only:   # the newest version of each report
        seen, kept = set(), []
        for r in rows:
            key = (r.cron_id, r.kind, r.start_at, r.end_at)
            if key not in seen:
                seen.add(key)
                kept.append(r)
        rows = kept
    if site:
        wanted = site.strip().lower()
        rows = [r for r in rows if wanted in {s.lower() for s in (r.sites or [])}]
    total = len(rows)
    return [{k: v for k, v in as_dict(r).items() if k not in ("data", "markdown")} for r in rows[offset:offset + limit]], total


@read_session
def report_neighbours(report: dict, *, session=None) -> tuple[str | None, str | None]:
    base = select(ReportRow.id).where(ReportRow.cron_id == report["cron_id"], ReportRow.kind == report["kind"],
                                      ReportRow.status == "ready")
    previous = session.execute(base.where(ReportRow.start_at < db(report["start_at"]))
                               .order_by(ReportRow.start_at.desc(), ReportRow.version.desc())).scalars().first()
    following = session.execute(base.where(ReportRow.start_at > db(report["start_at"]))
                                .order_by(ReportRow.start_at, ReportRow.version.desc())).scalars().first()
    return previous, following


@read_session
def pending_reports(*, session=None) -> list[str]:
    return list(session.execute(select(ReportRow.id).where(ReportRow.status == "pending")).scalars())


# --- proposals and suppressions -------------------------------------------------------------------------
@transactional_session
def add_proposal(cron_id: str, run_id: str, check: str, issue_id: str | None, tool: str, arguments: dict,
                 reason: str, *, session=None) -> str:
    proposal_id = new_id()
    session.add(ProposalRow(id=proposal_id, cron_id=cron_id, run_id=run_id, check_name=check, issue_id=issue_id,
                            tool=tool, arguments=arguments, reason=reason, status="pending"))
    return proposal_id


@read_session
def proposals(cron_id: str | None = None, issue_id: str | None = None, status: str | None = None,
              start: dt.datetime | None = None, end: dt.datetime | None = None, *, session=None) -> list[dict]:
    query = select(ProposalRow)
    for column, value in ((ProposalRow.cron_id, cron_id), (ProposalRow.issue_id, issue_id), (ProposalRow.status, status)):
        if value is not None:
            query = query.where(column == value)
    if start:
        query = query.where(or_(ProposalRow.created_at >= db(start), ProposalRow.decided_at >= db(start)))
    if end:
        query = query.where(ProposalRow.created_at < db(end))
    return [as_dict(r) for r in session.execute(query.order_by(ProposalRow.created_at)).scalars()]


@read_session
def proposal(proposal_id: str, *, session=None) -> dict | None:
    row = session.get(ProposalRow, proposal_id)
    return as_dict(row) if row else None


@transactional_session
def decide_proposal(proposal_id: str, status: str, by: str, at: dt.datetime, *, session=None) -> bool:
    """pending -> approved | rejected, once (a second decision returns False)."""
    row = session.execute(select(ProposalRow).where(ProposalRow.id == proposal_id).with_for_update()).scalars().first()
    if row is None or row.status != "pending":
        return False
    row.status, row.decided_by, row.decided_at = status, by, db(at)
    return True


@transactional_session
def proposal_result(proposal_id: str, status: str, result: dict, *, session=None) -> None:
    """What running an approved proposal gave: it stays approved, or becomes failed."""
    row = session.get(ProposalRow, proposal_id)
    if row is not None:
        row.status, row.result = status, result


@read_session
def suppressions(cron_id: str, at: dt.datetime | None = None, *, session=None) -> list[dict]:
    query = select(SuppressionRow).where(SuppressionRow.cron_id == cron_id)
    if at is not None:
        query = query.where(or_(SuppressionRow.until.is_(None), SuppressionRow.until > db(at)))
    return [as_dict(r) for r in session.execute(query.order_by(SuppressionRow.created_at)).scalars()]


@transactional_session
def add_suppression(cron_id: str, check: str | None, match: dict, reason: str, by: str, until: dt.datetime | None,
                    *, session=None) -> str:
    suppression_id = new_id()
    session.add(SuppressionRow(id=suppression_id, cron_id=cron_id, check_name=check, match_keys=match, reason=reason,
                               actor=by, until=db(until)))
    return suppression_id


@transactional_session
def delete_suppression(cron_id: str, suppression_id: str, *, session=None) -> bool:
    return bool(session.execute(delete(SuppressionRow).where(SuppressionRow.cron_id == cron_id,
                                                              SuppressionRow.id == suppression_id)).rowcount)


# --- retention ----------------------------------------------------------------------------------------
@transactional_session
def expire(cron_id: str, before: dt.datetime, raw_before: dt.datetime, *, session=None) -> None:
    """Delete what `retention` no longer keeps; issues go when closed and older than the reports."""
    cutoff, raw_cutoff = db(before), db(raw_before)
    session.execute(delete(CronMetricRow).where(CronMetricRow.cron_id == cron_id, CronMetricRow.at < raw_cutoff))
    session.execute(delete(CronMetricAggregateRow).where(CronMetricAggregateRow.cron_id == cron_id,
                                                         CronMetricAggregateRow.bucket < cutoff))
    session.execute(delete(CronFindingRow).where(CronFindingRow.cron_id == cron_id, CronFindingRow.at < cutoff))
    session.execute(delete(CronRunRow).where(CronRunRow.cron_id == cron_id, CronRunRow.started_at < cutoff))
    session.execute(delete(ReportRow).where(ReportRow.cron_id == cron_id, ReportRow.end_at < cutoff))
    session.execute(delete(ProposalRow).where(ProposalRow.cron_id == cron_id, ProposalRow.created_at < cutoff))
    old = select(IssueRow.id).where(IssueRow.cron_id == cron_id, IssueRow.status.in_(("solved", "ignored")),
                                    IssueRow.last_seen < cutoff)
    ids = list(session.execute(old).scalars())
    if ids:
        session.execute(delete(IssueEventRow).where(IssueEventRow.issue_id.in_(ids)))
        session.execute(delete(IssueRow).where(IssueRow.id.in_(ids)))
    session.execute(delete(SuppressionRow).where(SuppressionRow.cron_id == cron_id,
                                                 and_(SuppressionRow.until.isnot(None), SuppressionRow.until < cutoff)))
