"""
DatabaseStore: the RuntimeStore on agenticwf.agent.core.orm (docs/agent-flow.md §9).

Write-through: the in-memory maps of InMemoryStore serve this process (the session node and its
workers), and every write also goes to the tables in agenticwf.agent.core.orm.models.agent before the call
returns. Reads that miss the cache go to the tables, and `open_intents()` loads everything a
restart needs to resume: intents not yet answered, their tasks and those tasks' effects.

A task is stored in parts, so a state change writes what changed, not the whole task again:

    agent_tasks.record      the head: Task and AgentContext without history, calls, instructions
    agent_tasks.base        the instructions snapshot (AGENTS.md, WORKFLOW.md): written once
    agent_step_messages     the history, one row per entry, each written once (a part is rewritten
                            only when compaction replaces its older entries with a summary)
    agent_tool_calls        the tool calls, one row each: rewritten while open, never once finished

In one process (the default), jobs stay in memory: after a restart they are derived from the
records (the session resumes its intents, docs/agent-flow.md §8), so no job is lost with the process.

Shared (`shared=True`, `[harness] shared = true`): worker processes beside the API process share
the work (docs/deployment.md, Worker processes). Then
    jobs        live in agent_jobs; a worker claims one with a lease (FOR UPDATE SKIP LOCKED on
                PostgreSQL, a conditional update anywhere), fairly between owners across every
                process; workers look for new ones every `poll_seconds`
    the cache   is dropped for what a job reads before it runs (prepare), and for what an event
                from another process names (forget); effects are read from the table
    cancel      a cancel asked in the API process is a column a worker elsewhere checks each step

TODO: async I/O (these calls still block the event loop for one small write each); LISTEN/NOTIFY
instead of polling.
"""

from __future__ import annotations

import copy
import dataclasses
import datetime
import pickle
from dataclasses import dataclass, field

from agenticwf.agent.core.common.logging import get_logger

from typing import Collection, Mapping

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.exc import IntegrityError

from agenticwf.agent.core.orm.base.session import get_engine, read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import EffectRow, IntentRow, JobRow, StepMessageRow, TaskRow, ToolCallRow
from agenticwf.agent.runtime.context.agent_context import AgentContext
from agenticwf.agent.runtime.harness.store import InMemoryStore, Job
from agenticwf.agent.runtime.llm.base import Usage
from agenticwf.agent.runtime.records import Effect, EffectStatus, Intent, IntentStatus, Reply, TaskResult
from agenticwf.agent.runtime.task import Task, ToolCallStatus

LOG = get_logger(__name__)
_OPEN = (IntentStatus.PENDING.value, IntentStatus.CLAIMED.value, IntentStatus.CONVERTED.value)
_DONE = (IntentStatus.ANSWERED, IntentStatus.SUPERSEDED, IntentStatus.REJECTED)
_FINISHED = frozenset({ToolCallStatus.SUCCEEDED, ToolCallStatus.FAILED, ToolCallStatus.DENIED,
                       ToolCallStatus.REJECTED, ToolCallStatus.INTERRUPTED})


@dataclass
class _Written:
    """What of one task is in the tables already: the history entries of each part (the objects
    themselves, compared by identity), the finished calls (seq → status), whether `base` is."""

    parts: dict[int, list] = field(default_factory=dict)
    calls: dict[int, str] = field(default_factory=dict)
    base: bool = False


def _unpickle(record: bytes, what: str):
    """A pickled record, or None when this code cannot read it any more (a class renamed or moved
    by an upgrade): one such row must never stop the agent from starting."""
    try:
        return pickle.loads(record)
    except Exception as exc:
        LOG.error("%s cannot be read (%s: %s); skipped", what, type(exc).__name__, exc)
        return None


class DatabaseStore(InMemoryStore):
    def __init__(self, shared: bool = False, poll_seconds: float = 1.0) -> None:
        super().__init__()
        self._written: dict[str, _Written] = {}
        self.shared = shared
        self.poll_seconds = poll_seconds if shared else None

    # -- shared: jobs in the table ---------------------------------------------------------------
    def enqueue(self, job: Job) -> bool:
        if not self.shared:
            return super().enqueue(job)
        created = _enqueue_job(job)
        for listener in self._listeners:   # this process's workers at once; others at their next poll
            listener()
        return created

    def claim(self, worker: str, lease_seconds: float, skip: Collection[str] = (),
              running: Mapping[str | None, int] | None = None, per_owner: int = 0) -> Job | None:
        """Shared: every process's running jobs count for fairness (their live leases), not only
        `running`, the caller's pool's."""
        if not self.shared:
            return super().claim(worker, lease_seconds, skip, running, per_owner)
        return _claim_job(worker, lease_seconds, list(skip), per_owner)

    def renew(self, job: Job, worker: str, lease_seconds: float) -> bool:
        if not self.shared:
            return super().renew(job, worker, lease_seconds)
        return _renew_job(job.key, worker, lease_seconds)

    def complete(self, job: Job, worker: str) -> None:
        if not self.shared:
            return super().complete(job, worker)
        if _complete_job(job.key, worker):
            for listener in self._listeners:
                listener()

    def prepare(self, job: Job) -> None:
        """Before a job runs here: another process may have changed what it reads."""
        if not self.shared:
            return
        if job.kind in ("task", "consolidate"):
            self.forget(task_ids=[job.ref])
        elif job.kind in ("intake", "reply"):
            self.forget(intent_ids=[job.ref])
            intent = self.load_intent(job.ref)
            if intent is not None:
                self.forget(task_ids=intent.task_ids)

    def forget(self, task_ids: Collection[str] = (), intent_ids: Collection[str] = ()) -> None:
        """Drop cached records (shared: the tables are the truth; the next read goes there)."""
        if not self.shared:
            return
        for task_id in task_ids:
            for cache in (self._tasks, self._written, self._effects, self._results):
                cache.pop(task_id, None)
        for intent_id in intent_ids:
            self._intents.pop(intent_id, None)
            self._replies.pop(intent_id, None)

    def cancel_requested(self, task_id: str) -> bool:
        return self.shared and _cancel_requested(task_id)

    def request_cancel(self, task_id: str) -> bool:
        """Shared: the worker running it (in any process) stops at its next step."""
        return self.shared and _request_cancel(task_id)

    def effects_for_task(self, task_id: str) -> list[Effect]:
        if not self.shared:
            return super().effects_for_task(task_id)
        return _effects_of(task_id)   # another process's worker may have written them

    # -- tasks ----------------------------------------------------------------------------------
    def save(self, task: Task, context: AgentContext) -> None:
        super().save(task, context)
        written = self._written.get(task.id) or _Written()
        parts = _parts(context)
        writes = _Writes(head=_head(task, context),
                         base=None if written.base else pickle.dumps(task.task_type),
                         parts={}, calls=[])
        for n, entries in enumerate(parts):
            before = written.parts.get(n, [])
            if len(before) <= len(entries) and all(a is b for a, b in zip(before, entries)):
                if len(entries) > len(before):   # appended: only the new entries
                    writes.parts[n] = (False, len(before), entries[len(before):])
            else:                                 # compacted (or changed in place): the whole part again
                writes.parts[n] = (True, 0, entries)
        for seq, call in enumerate(task.calls):
            if call.status not in _FINISHED or written.calls.get(seq) != call.status.value:
                writes.calls.append((seq, call))
        _write_task(task, context.step_count, writes)
        # Only once the transaction is committed: a failed write is written again next time.
        written.base = True
        written.parts = {n: list(entries) for n, entries in enumerate(parts)}
        written.calls.update({seq: c.status.value for seq, c in writes.calls if c.status in _FINISHED})
        self._written[task.id] = written

    def load(self, task_id: str) -> tuple[Task, AgentContext] | None:
        cached = super().load(task_id)
        if cached is not None:
            return cached
        row = _read(TaskRow, task_id)
        if row is None:
            return None
        record = _unpickle(row["record"], f"task {task_id}")
        if record is None:
            return None
        task, context = record
        if row["base"] is None:   # the older layout: the whole task in `record`; saved in parts next time
            super().save(task, context)
            return task, context
        base = _unpickle(row["base"], f"task {task_id} instructions")
        messages, calls = _task_rows(task_id)
        if base is None or messages is None or calls is None:
            return None
        task.task_type = context.instructions = base
        parts = [context.messages, *(turn.messages for turn in context.turns.values())]
        for (part, _), message in sorted(messages.items()):
            if part < len(parts):
                parts[part].append(message)
        task.calls = [call for _, call in sorted(calls.items())]
        self._written[task_id] = _Written(parts={n: list(p) for n, p in enumerate(parts)},
                                          calls={seq: c.status.value for seq, c in calls.items() if c.status in _FINISHED},
                                          base=True)
        super().save(task, context)
        return task, context

    def tasks_for_intent(self, intent_id: str) -> list[Task]:
        intent = self.load_intent(intent_id)
        loaded = [self.load(t) for t in (intent.task_ids if intent else [])]
        return [pair[0] for pair in loaded if pair is not None]

    # -- intents --------------------------------------------------------------------------------
    def save_intent(self, intent: Intent) -> None:
        super().save_intent(intent)
        _write_intent(intent)
        if intent.status in _DONE:
            self._evict(intent)

    def _evict(self, intent: Intent) -> None:
        """An answered intent leaves this process's cache with its tasks, effects and results: the
        tables keep them, and reads that miss the cache go there. Memory stays bounded by the work
        in progress, not by everything the process has ever done."""
        self._intents.pop(intent.id, None)
        self._replies.pop(intent.id, None)
        for task_id in intent.task_ids:
            self._tasks.pop(task_id, None)
            self._written.pop(task_id, None)
            self._effects.pop(task_id, None)
            self._results.pop(task_id, None)

    def load_intent(self, intent_id: str) -> Intent | None:
        cached = super().load_intent(intent_id)
        if cached is not None:
            return cached
        row = _read(IntentRow, intent_id)
        if row is None:
            return None
        intent = _unpickle(row["record"], f"intent {intent_id}")
        if intent is not None:
            super().save_intent(intent)
        return intent

    def open_intents(self) -> list[Intent]:
        """Intents not answered yet, with their tasks and effects loaded: what a restart resumes."""
        intents = []
        for intent_id, record in _open_intent_records():
            intent = _unpickle(record, f"intent {intent_id}")
            if intent is None:
                _set_intent_status(intent_id, IntentStatus.REJECTED.value)   # not retried at every start
                continue
            intents.append(intent)
        for intent in intents:
            super().save_intent(intent)
            for task_id in intent.task_ids:
                if self.load(task_id) is not None:
                    for effect in _effects_of(task_id):
                        super().save_effect(effect)
        return sorted(intents, key=lambda i: i.received_at)

    def pending_consolidations(self) -> list[str]:
        """Finished tasks whose consolidation into long-term memory has not run: a restart queues them."""
        return _pending_consolidations()

    # -- effects --------------------------------------------------------------------------------
    def save_effect(self, effect: Effect) -> None:
        super().save_effect(effect)
        _write_effect(effect)

    # -- results --------------------------------------------------------------------------------
    def save_result(self, result: TaskResult) -> None:
        super().save_result(result)
        _write_task_result(result.task_id, _jsonable(result))

    def load_result(self, task_id: str) -> TaskResult | None:
        cached = super().load_result(task_id)
        if cached is not None:
            return cached
        row = _read(TaskRow, task_id)
        if row is None or not row["result"]:
            return None
        data = dict(row["result"])
        result = TaskResult(**{**data, "usage": Usage(**data["usage"])})
        self._results[task_id] = result
        return result

    def save_reply(self, reply: Reply) -> None:
        super().save_reply(reply)
        _write_reply(reply.intent_id, _jsonable(reply))

    def load_reply(self, intent_id: str) -> Reply | None:
        cached = super().load_reply(intent_id)
        if cached is not None:
            return cached
        row = _read(IntentRow, intent_id)
        if row is None or not row["reply"]:
            return None
        data = dict(row["reply"])
        reply = Reply(**{**data, "usage": Usage(**data["usage"])})
        self._replies[intent_id] = reply
        return reply


def _jsonable(record) -> dict:
    return dataclasses.asdict(record)


@dataclass
class _Writes:
    head: bytes
    base: bytes | None
    parts: dict[int, tuple[bool, int, list]]     # part → (rewrite all, first seq, entries)
    calls: list[tuple[int, object]]              # (seq, ToolCall)


def _parts(context: AgentContext) -> list[list]:
    """The history in parts: the task's own messages, then each turn's, in the order turns began."""
    return [context.messages, *(turn.messages for turn in context.turns.values())]


def _head(task: Task, context: AgentContext) -> bytes:
    """Task and context without what is stored on its own (history, calls, instructions)."""
    head_task, head_context = copy.copy(task), copy.copy(context)
    head_task.calls, head_task.task_type = [], None
    head_context.messages, head_context.instructions = [], None
    head_context.turns = {k: dataclasses.replace(t, messages=[]) for k, t in context.turns.items()}
    return pickle.dumps((head_task, head_context))


# -- table access (synchronous; one transaction each) --------------------------------------------
@read_session
def _read(model, row_id: str, *, session=None) -> dict | None:
    row = session.get(model, row_id)
    return None if row is None else {c.name: getattr(row, c.name) for c in model.__table__.columns}


@transactional_session
def _write_task(task: Task, step: int, writes: _Writes, *, session=None) -> None:
    """One transaction: the head, `base` the first time, new history entries, open and just-finished calls."""
    row = session.get(TaskRow, task.id)
    if row is None:
        row = TaskRow(id=task.id)
        session.add(row)
    row.intent_id, row.owner_subject, row.owner_org = task.intent_id, task.owner[0], task.owner[1]
    row.task_type, row.state, row.record = task.task_type.requested, task.state.value, writes.head
    row.memory = getattr(task, "consolidation", None)
    if task.cancel_requested:   # only ever raised here: a worker's save never undoes another process's cancel
        row.cancel = 1
    if writes.base is not None:
        row.base = writes.base
    for part, (rewrite, first, entries) in writes.parts.items():
        if rewrite:
            session.execute(delete(StepMessageRow).where(StepMessageRow.task_id == task.id, StepMessageRow.part == part))
        for seq, message in enumerate(entries, start=first):
            session.merge(StepMessageRow(task_id=task.id, part=part, seq=seq, step=step, role=message.role,
                                         record=pickle.dumps(message)))
    for seq, call in writes.calls:
        session.merge(ToolCallRow(task_id=task.id, seq=seq, call_id=call.id, step=call.step, name=call.name,
                                  status=call.status.value, record=pickle.dumps(call)))


@read_session
def _task_rows(task_id: str, *, session=None) -> tuple[dict | None, dict | None]:
    """History entries ((part, seq) → Message) and calls (seq → ToolCall); None if one cannot be read."""
    messages: dict = {}
    for row in session.execute(select(StepMessageRow).where(StepMessageRow.task_id == task_id)).scalars():
        message = _unpickle(row.record, f"task {task_id} history entry {row.part}/{row.seq}")
        if message is None:
            return None, None
        messages[(row.part, row.seq)] = message
    calls: dict = {}
    for row in session.execute(select(ToolCallRow).where(ToolCallRow.task_id == task_id)).scalars():
        call = _unpickle(row.record, f"task {task_id} call {row.call_id}")
        if call is None:
            return None, None
        calls[row.seq] = call
    return messages, calls


@transactional_session
def _write_task_result(task_id: str, result: dict, *, session=None) -> None:
    row = session.get(TaskRow, task_id)
    if row is not None:
        row.result = result


@transactional_session
def _write_intent(intent: Intent, *, session=None) -> None:
    session.merge(IntentRow(id=intent.id, thread_id=intent.thread_id, owner_subject=intent.owner[0],
                            owner_org=intent.owner[1], source=intent.source, status=intent.status.value,
                            text=intent.text, record=pickle.dumps(intent)))


@transactional_session
def _write_reply(intent_id: str, reply: dict, *, session=None) -> None:
    row = session.get(IntentRow, intent_id)
    if row is not None:
        row.reply = reply


@read_session
def _open_intent_records(*, session=None) -> list[tuple[str, bytes]]:
    return [(r.id, r.record) for r in session.execute(select(IntentRow.id, IntentRow.record)
                                                     .where(IntentRow.status.in_(_OPEN)))]


@transactional_session
def _set_intent_status(intent_id: str, status: str, *, session=None) -> None:
    row = session.get(IntentRow, intent_id)
    if row is not None:
        row.status = status


@read_session
def _pending_consolidations(*, session=None) -> list[str]:
    return list(session.execute(select(TaskRow.id).where(TaskRow.memory == "pending")).scalars())


@transactional_session
def _write_effect(effect: Effect, *, session=None) -> None:
    session.merge(EffectRow(id=effect.id, task_id=effect.task_id, call_id=effect.call_id, tool=effect.tool,
                            kind=effect.kind, target=effect.target, idempotency_key=effect.idempotency_key,
                            status=effect.status.value, detail=effect.detail,
                            at=datetime.datetime.fromtimestamp(effect.at, datetime.timezone.utc).replace(tzinfo=None)))


@read_session
def _effects_of(task_id: str, *, session=None) -> list[Effect]:
    rows = session.execute(select(EffectRow).where(EffectRow.task_id == task_id).order_by(EffectRow.at)).scalars()
    return [Effect(task_id=r.task_id, call_id=r.call_id, tool=r.tool, kind=r.kind, target=r.target,
                   idempotency_key=r.idempotency_key, status=EffectStatus(r.status), detail=r.detail or {},
                   id=r.id, at=r.at.replace(tzinfo=datetime.timezone.utc).timestamp())
            for r in rows]


# -- shared jobs ----------------------------------------------------------------------------------------
def _now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


def _job(row: JobRow) -> Job:
    return Job(row.kind, row.ref, lease_owner=row.lease_owner, attempts=row.attempts, again=bool(row.again),
               owner=row.owner, trace=row.trace, enqueued_at=row.enqueued_at.replace(tzinfo=datetime.timezone.utc).timestamp())


def _enqueue_job(job: Job) -> bool:
    """Queued, unless it is already: then, if a worker holds it, it runs once more afterwards."""
    try:
        return _insert_or_mark(job)
    except IntegrityError:   # another process queued it at the same moment
        return _insert_or_mark(job)


@transactional_session
def _insert_or_mark(job: Job, *, session=None) -> bool:
    row = session.get(JobRow, job.key)
    now = _now()
    if row is None:
        session.add(JobRow(key=job.key, kind=job.kind, ref=job.ref, owner=job.owner, attempts=0, again=0,
                           trace=job.trace, enqueued_at=now))
        session.flush()
        return True
    if row.lease_owner is not None and row.lease_until is not None and row.lease_until >= now:
        row.again = 1
    return False


@transactional_session
def _claim_job(worker: str, lease_seconds: float, skip: list[str], per_owner: int, *, session=None) -> Job | None:
    now = _now()
    free = or_(JobRow.lease_owner.is_(None), JobRow.lease_until < now)
    query = select(JobRow).where(free).order_by(JobRow.enqueued_at).limit(200)
    if skip:
        query = query.where(JobRow.key.notin_(skip))
    if get_engine().dialect.name == "postgresql":
        query = query.with_for_update(skip_locked=True)   # rows another process is claiming are passed over
    candidates = session.execute(query).scalars().all()
    if not candidates:
        return None
    running = dict(session.execute(select(JobRow.owner, func.count()).where(
        JobRow.lease_owner.isnot(None), JobRow.lease_until >= now).group_by(JobRow.owner)).all())
    order = sorted(range(len(candidates)), key=lambda i: (running.get(candidates[i].owner, 0), i))
    for i in order:
        row = candidates[i]
        if per_owner and row.owner is not None and running.get(row.owner, 0) >= per_owner:
            continue
        taken = session.execute(update(JobRow).where(JobRow.key == row.key, free).values(
            lease_owner=worker, lease_until=now + datetime.timedelta(seconds=lease_seconds),
            attempts=JobRow.attempts + 1)).rowcount
        if taken:   # without SKIP LOCKED (SQLite), another process may have taken it first: the next one
            session.flush()
            session.refresh(row)
            return _job(row)
    return None


@transactional_session
def _renew_job(key: str, worker: str, lease_seconds: float, *, session=None) -> bool:
    return bool(session.execute(update(JobRow).where(JobRow.key == key, JobRow.lease_owner == worker).values(
        lease_until=_now() + datetime.timedelta(seconds=lease_seconds))).rowcount)


@transactional_session
def _complete_job(key: str, worker: str, *, session=None) -> bool:
    """Done: deleted; or queued again at the back if it was enqueued while held. Returns whether it was queued again."""
    row = session.get(JobRow, key)
    if row is None or row.lease_owner != worker:
        return False
    if row.again:
        row.again, row.lease_owner, row.lease_until, row.enqueued_at = 0, None, None, _now()
        return True
    session.delete(row)
    return False


@read_session
def _cancel_requested(task_id: str, *, session=None) -> bool:
    return bool(session.execute(select(TaskRow.cancel).where(TaskRow.id == task_id)).scalar())


@transactional_session
def _request_cancel(task_id: str, *, session=None) -> bool:
    return bool(session.execute(update(TaskRow).where(TaskRow.id == task_id).values(cancel=1)).rowcount)
