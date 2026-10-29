"""
EventRelay: events of work done in a worker process reach the API process, where the sessions
and the live update streams are (docs/deployment.md, Worker processes).

    worker process   AgentHarness.remote = relay.write: an event for a session that is not here
                     becomes a row of agent_events (cron events too: sink `cron:<id>`)
    API process      relay.run(): reads new rows every `poll` seconds; for each, drops the cached
                     records it names (the worker changed them) and delivers it as if it had
                     happened here

Partial answers (`message_delta`) are not relayed: a chat whose task runs in a worker process
shows each answer when it is complete. Rows are deleted once delivered; the API reads from the
newest row when it starts (what happened while it was down is in the records, which recovery
reads). TODO: LISTEN/NOTIFY on PostgreSQL instead of polling.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from sqlalchemy import delete, func, select

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import EventRow

LOG = get_logger(__name__)
NOT_RELAYED = frozenset({"message_delta"})


class EventRelay:
    def __init__(self, poll: float = 0.3, batch: int = 500):
        self.poll = poll
        self.batch = batch
        self.last = 0

    # -- worker side ----------------------------------------------------------------------------
    def write(self, sink: str, subject: str, event: dict[str, Any]) -> None:
        if event.get("type") in NOT_RELAYED:
            return
        try:
            _insert(sink, subject, event)
        except Exception:   # an event lost is a stale screen, never a lost task: the records are written
            LOG.exception("could not relay a %s event", event.get("type"))

    # -- API side -------------------------------------------------------------------------------
    def start_at_newest(self) -> None:
        self.last = _newest()

    def drain(self, deliver: Callable[[str, str, dict[str, Any]], None]) -> int:
        """Deliver the rows written since the last call. Returns how many."""
        rows = _since(self.last, self.batch)
        for row_id, sink, subject, payload in rows:
            self.last = row_id
            try:
                deliver(sink, subject, payload)
            except Exception:
                LOG.exception("relayed %s event for %s failed", payload.get("type"), sink)
        if rows:
            _delete_through(self.last)
        return len(rows)

    async def run(self, deliver: Callable[[str, str, dict[str, Any]], None]) -> None:
        self.start_at_newest()
        while True:
            try:
                while self.drain(deliver) == self.batch:
                    pass
            except Exception:   # the database was away for a moment: try again
                LOG.exception("event relay pass failed")
            await asyncio.sleep(self.poll)


def deliverer(harness, crons=None) -> Callable[[str, str, dict[str, Any]], None]:
    """The API side: an event of work a worker process did drops the cached records it names
    (they were changed there), then reaches the session (or the cron's watchers) as if it had
    happened here."""
    def deliver(sink: str, subject: str, event: dict[str, Any]) -> None:
        tasks = [event["task"]] if event.get("task") else []
        harness.store.forget(task_ids=[*tasks, *(event.get("tasks") or [])],
                             intent_ids=[event["intent"]] if event.get("intent") else [])
        if sink.startswith("cron:"):
            cron = crons.catalog.find(sink[5:]) if crons is not None else None
            if cron is not None:
                crons.notify(cron, event)
        else:
            harness.deliver(sink, subject, event)
    return deliver


@transactional_session
def _insert(sink: str, subject: str, payload: dict[str, Any], *, session=None) -> None:
    session.add(EventRow(sink=sink, subject=subject, payload=payload))


@read_session
def _newest(*, session=None) -> int:
    return session.execute(select(func.max(EventRow.id))).scalar() or 0


@read_session
def _since(last: int, limit: int, *, session=None) -> list[tuple[int, str, str, dict]]:
    rows = session.execute(select(EventRow).where(EventRow.id > last).order_by(EventRow.id).limit(limit)).scalars()
    return [(r.id, r.sink, r.subject, r.payload) for r in rows]


@transactional_session
def _delete_through(last: int, *, session=None) -> None:
    session.execute(delete(EventRow).where(EventRow.id <= last))
