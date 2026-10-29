"""
Model usage and feedback (docs/model-routing.md §7, §8): every call's tokens and cost, ratings of
answers and the evidence the harness sees (escalations), per model and purpose. Budgets read the
spending, the output estimator the averages, operators the summaries and suggestions.

    InMemoryUsage     one process, nothing kept (tests, and runtimes without a database)
    DatabaseUsage     agent_model_usage, agent_model_feedback (agenticwf.agent.core.orm)
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections import defaultdict
from typing import Any, Protocol

from sqlalchemy import delete, func, or_, select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import ModelFeedbackRow, ModelUsageRow
from agenticwf.agent.runtime.types import Owner

GROUPS = ("model", "purpose", "owner", "day")


def _utc(t: dt.datetime) -> dt.datetime:
    return t.astimezone(dt.timezone.utc).replace(tzinfo=None)


class UsageStore(Protocol):
    def record(self, row: dict[str, Any]) -> None: ...
    def spent(self, owner: Owner, since: dt.datetime) -> float: ...
    def averages(self) -> list[tuple[str, str, float]]: ...
    def summary(self, start: dt.datetime, end: dt.datetime, group: str, owner: Owner | None = None) -> list[dict]: ...
    def calls(self, owner: Owner, intent_id: str, task_ids: list[str]) -> list[dict]:
        """The owner's model calls for an intent and its tasks, oldest first (the trace view)."""
    def feedback(self, row: dict[str, Any]) -> None: ...
    def quality(self, since: dt.datetime) -> list[dict]: ...
    def rating(self, intent_id: str, owner: Owner) -> str | None: ...
    def forget_rating(self, intent_id: str, owner: Owner) -> None: ...


class InMemoryUsage:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.feedback_rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def record(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.rows.append({"at": dt.datetime.now(dt.timezone.utc), **row})

    def spent(self, owner: Owner, since: dt.datetime) -> float:
        return sum(r["cost"] for r in self.rows if (r.get("owner_subject"), r.get("owner_org")) == tuple(owner)
                   and r["at"] >= since and not r.get("own_model"))

    def averages(self) -> list[tuple[str, str, float]]:
        by: dict[tuple[str, str], list[int]] = defaultdict(list)
        for r in self.rows[-5000:]:
            by[(r["purpose"], r["model_id"])].append(r["output_tokens"])
        return [(p, m, sum(v) / len(v)) for (p, m), v in by.items()]

    def calls(self, owner: Owner, intent_id: str, task_ids: list[str]) -> list[dict]:
        with self._lock:
            rows = list(self.rows)
        return [r for r in rows if (r.get("owner_subject"), r.get("owner_org")) == tuple(owner)
                and (r.get("intent_id") == intent_id or r.get("task_id") in task_ids)]

    def summary(self, start: dt.datetime, end: dt.datetime, group: str, owner: Owner | None = None) -> list[dict]:
        return _summarise([r for r in self.rows if start <= r["at"] < end
                           and (owner is None or (r.get("owner_subject"), r.get("owner_org")) == tuple(owner))], group)

    def feedback(self, row: dict[str, Any]) -> None:
        with self._lock:
            self.feedback_rows.append({"at": dt.datetime.now(dt.timezone.utc), **row})

    def quality(self, since: dt.datetime) -> list[dict]:
        return _quality([r for r in self.rows if r["at"] >= since], [f for f in self.feedback_rows if f["at"] >= since])

    def rating(self, intent_id: str, owner: Owner) -> str | None:
        found = [f["kind"] for f in self.feedback_rows if f.get("intent_id") == intent_id and f["kind"] in ("up", "down")
                 and (f.get("owner_subject"), f.get("owner_org")) == tuple(owner)]
        return found[-1] if found else None

    def forget_rating(self, intent_id: str, owner: Owner) -> None:
        with self._lock:
            self.feedback_rows = [f for f in self.feedback_rows if not (
                f.get("intent_id") == intent_id and f["kind"] in ("up", "down")
                and (f.get("owner_subject"), f.get("owner_org")) == tuple(owner))]


def _key(row: dict, group: str) -> str:
    if group == "owner":
        return f"{row.get('owner_subject')}@{row.get('owner_org')}" if row.get("owner_org") else str(row.get("owner_subject"))
    if group == "day":
        return row["at"].date().isoformat()
    return str(row.get("model_id" if group == "model" else "purpose"))


def _summarise(rows: list[dict], group: str) -> list[dict]:
    totals: dict[str, dict[str, float]] = {}
    for r in rows:
        t = totals.setdefault(_key(r, group), {"calls": 0, "input_tokens": 0, "cached_tokens": 0, "output_tokens": 0,
                                               "cost": 0.0})
        t["calls"] += 1
        for k in ("input_tokens", "cached_tokens", "output_tokens", "cost"):
            t[k] += r.get(k) or 0
    return [{group: k, **v} for k, v in sorted(totals.items(), key=lambda kv: -kv[1]["cost"])]


def _quality(usage: list[dict], feedback: list[dict]) -> list[dict]:
    """Per purpose and model: calls, ratings up / down, escalations away from it."""
    stats: dict[tuple[str, str], dict[str, int]] = {}
    for r in usage:
        stats.setdefault((r["purpose"], r["model_id"]), {"calls": 0, "up": 0, "down": 0, "escalations": 0})["calls"] += 1
    for f in feedback:
        entry = stats.setdefault((f["purpose"], f["model_id"]), {"calls": 0, "up": 0, "down": 0, "escalations": 0})
        kind = "escalations" if f["kind"] == "escalation" else f["kind"]
        if kind in entry:
            entry[kind] += 1
    return [{"purpose": p, "model": m, **v} for (p, m), v in sorted(stats.items())]


class DatabaseUsage:
    """The tables, with each owner's spending cached for `cache_seconds` (a budget check per call)."""

    def __init__(self, cache_seconds: float = 30.0):
        self.cache_seconds = cache_seconds
        self._spent: dict[tuple[Owner, dt.datetime], tuple[float, float]] = {}

    def record(self, row: dict[str, Any]) -> None:
        _insert_usage(row)
        for key in [k for k in self._spent if k[0] == (row.get("owner_subject"), row.get("owner_org"))]:
            self._spent.pop(key, None)

    def spent(self, owner: Owner, since: dt.datetime) -> float:
        key = (tuple(owner), since)
        cached = self._spent.get(key)
        if cached and cached[0] + self.cache_seconds > time.monotonic():
            return cached[1]
        value = _spent(tuple(owner), _utc(since))
        self._spent[key] = (time.monotonic(), value)
        return value

    def averages(self) -> list[tuple[str, str, float]]:
        return _averages()

    def calls(self, owner: Owner, intent_id: str, task_ids: list[str]) -> list[dict]:
        return _calls(tuple(owner), intent_id, list(task_ids))

    def summary(self, start: dt.datetime, end: dt.datetime, group: str, owner: Owner | None = None) -> list[dict]:
        rows = _rows(_utc(start), _utc(end), owner)
        return _summarise(rows, group)

    def feedback(self, row: dict[str, Any]) -> None:
        _insert_feedback(row)

    def quality(self, since: dt.datetime) -> list[dict]:
        usage, feedback = _quality_rows(_utc(since))
        return _quality(usage, feedback)

    def rating(self, intent_id: str, owner: Owner) -> str | None:
        return _rating(intent_id, tuple(owner))

    def forget_rating(self, intent_id: str, owner: Owner) -> None:
        _forget_rating(intent_id, tuple(owner))


@transactional_session
def _insert_usage(row: dict[str, Any], *, session=None) -> None:
    session.add(ModelUsageRow(**row))


@transactional_session
def _insert_feedback(row: dict[str, Any], *, session=None) -> None:
    session.add(ModelFeedbackRow(**row))


@read_session
def _spent(owner: Owner, since: dt.datetime, *, session=None) -> float:
    subject, org = owner
    query = select(func.coalesce(func.sum(ModelUsageRow.cost), 0.0)).where(
        ModelUsageRow.owner_subject == subject, ModelUsageRow.at >= since, ModelUsageRow.own_model == 0,
        ModelUsageRow.owner_org.is_(None) if org is None else ModelUsageRow.owner_org == org)
    return float(session.execute(query).scalar() or 0.0)


@read_session
def _averages(*, session=None) -> list[tuple[str, str, float]]:
    since = _utc(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=7))
    rows = session.execute(select(ModelUsageRow.purpose, ModelUsageRow.model_id, func.avg(ModelUsageRow.output_tokens))
                           .where(ModelUsageRow.at >= since).group_by(ModelUsageRow.purpose, ModelUsageRow.model_id)).all()
    return [(p, m, float(a)) for p, m, a in rows]


def _as_dict(row) -> dict[str, Any]:
    out = {c.key: getattr(row, c.key) for c in row.__table__.columns}
    out["at"] = out["at"].replace(tzinfo=dt.timezone.utc)
    return out


@read_session
def _calls(owner: Owner, intent_id: str, task_ids: list[str], *, session=None) -> list[dict]:
    subject, org = owner
    match = ModelUsageRow.intent_id == intent_id
    if task_ids:
        match = or_(match, ModelUsageRow.task_id.in_(task_ids))
    query = select(ModelUsageRow).where(match, ModelUsageRow.owner_subject == subject,
                                        ModelUsageRow.owner_org.is_(None) if org is None else ModelUsageRow.owner_org == org)
    return [_as_dict(r) for r in session.execute(query.order_by(ModelUsageRow.at, ModelUsageRow.id)).scalars()]


@read_session
def _rows(start: dt.datetime, end: dt.datetime, owner: Owner | None, *, session=None) -> list[dict]:
    query = select(ModelUsageRow).where(ModelUsageRow.at >= start, ModelUsageRow.at < end)
    if owner is not None:
        subject, org = owner
        query = query.where(ModelUsageRow.owner_subject == subject,
                            ModelUsageRow.owner_org.is_(None) if org is None else ModelUsageRow.owner_org == org)
    return [_as_dict(r) for r in session.execute(query).scalars()]


@read_session
def _quality_rows(since: dt.datetime, *, session=None) -> tuple[list[dict], list[dict]]:
    usage = [{"purpose": p, "model_id": m} for p, m in session.execute(
        select(ModelUsageRow.purpose, ModelUsageRow.model_id).where(ModelUsageRow.at >= since)).all()]
    feedback = [{"purpose": p, "model_id": m, "kind": k} for p, m, k in session.execute(
        select(ModelFeedbackRow.purpose, ModelFeedbackRow.model_id, ModelFeedbackRow.kind)
        .where(ModelFeedbackRow.at >= since)).all()]
    return usage, feedback


def _rated(query, intent_id: str, owner: Owner):
    subject, org = owner
    return query.where(ModelFeedbackRow.intent_id == intent_id, ModelFeedbackRow.kind.in_(("up", "down")),
                       ModelFeedbackRow.owner_subject == subject,
                       ModelFeedbackRow.owner_org.is_(None) if org is None else ModelFeedbackRow.owner_org == org)


@read_session
def _rating(intent_id: str, owner: Owner, *, session=None) -> str | None:
    query = _rated(select(ModelFeedbackRow.kind), intent_id, owner).order_by(ModelFeedbackRow.id.desc()).limit(1)
    return session.execute(query).scalar()


@transactional_session
def _forget_rating(intent_id: str, owner: Owner, *, session=None) -> None:
    session.execute(_rated(delete(ModelFeedbackRow), intent_id, owner))
