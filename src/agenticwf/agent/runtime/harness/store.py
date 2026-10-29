"""
RuntimeStore: the durable side of the runtime (docs/agent-flow.md §8–9).

    intents         TO DO       written by the session before send() returns
    tasks, effects  DOING       the task record and AgentContext after every change; effects
                                written ahead of the handler
    results         RESULT      one per task, one reply per intent; written once
    jobs                        intake, task, reply and consolidate jobs that workers claim with a lease

Any worker can run any job, because everything a job needs is loaded from here.

TODO: agenticwf.agent.core.orm tables (agent_intents, agent_tasks, agent_turns, agent_steps, agent_tool_calls,
agent_effects, agent_results), jobs claimed with SELECT … FOR UPDATE SKIP LOCKED, and events
through LISTEN/NOTIFY.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Collection, Mapping, Protocol

from agenticwf.agent.runtime.context.agent_context import AgentContext
from agenticwf.agent.runtime.records import Effect, Intent, IntentStatus, Reply, TaskResult
from agenticwf.agent.runtime.task import Task
from agenticwf.agent.runtime.telemetry import traceparent


@dataclass
class Job:
    kind: str                  # intake | task | reply | consolidate
    ref: str                   # intent id (intake, reply) or task id (task, consolidate)
    lease_owner: str | None = None
    lease_until: float = 0.0   # time.monotonic(): leases live in this process; a wall-clock jump cannot expire one
    attempts: int = 0
    again: bool = False        # enqueued again while a worker held it: run once more after
    enqueued_at: float = field(default_factory=time.time)
    owner: str | None = None   # whose work (the owner's scope): workers share themselves fairly between owners
    trace: str | None = field(default_factory=traceparent)   # the span that queued it: the job's span links to it

    @property
    def key(self) -> str:
        return f"{self.kind}:{self.ref}"


class RuntimeStore(Protocol):
    # tasks
    def save(self, task: Task, context: AgentContext) -> None: ...
    def load(self, task_id: str) -> tuple[Task, AgentContext] | None: ...
    def tasks_for_intent(self, intent_id: str) -> list[Task]: ...
    def pending_consolidations(self) -> list[str]: ...
    # intents
    def save_intent(self, intent: Intent) -> None: ...
    def load_intent(self, intent_id: str) -> Intent | None: ...
    def intents_for_thread(self, thread_id: str, status: IntentStatus | None = None) -> list[Intent]: ...
    # effects
    def save_effect(self, effect: Effect) -> None: ...
    def effects_for_call(self, task_id: str, call_id: str) -> list[Effect]: ...
    def effects_for_task(self, task_id: str) -> list[Effect]: ...
    # results
    def save_result(self, result: TaskResult) -> None: ...
    def load_result(self, task_id: str) -> TaskResult | None: ...
    def save_reply(self, reply: Reply) -> None: ...
    def load_reply(self, intent_id: str) -> Reply | None: ...
    # jobs
    def enqueue(self, job: Job) -> bool: ...
    def claim(self, worker: str, lease_seconds: float, skip: Collection[str] = (),
              running: Mapping[str | None, int] | None = None, per_owner: int = 0) -> Job | None: ...
    def renew(self, job: Job, worker: str, lease_seconds: float) -> bool: ...
    def complete(self, job: Job, worker: str) -> None: ...
    def on_enqueue(self, listener: Callable[[], None]) -> None: ...
    # several processes (DatabaseStore with shared=True; the rest do nothing)
    poll_seconds: float | None          # how often workers look for jobs queued by other processes
    def prepare(self, job: Job) -> None: ...                 # before a job runs: drop what may be stale
    def forget(self, task_ids: Collection[str] = (), intent_ids: Collection[str] = ()) -> None: ...
    def cancel_requested(self, task_id: str) -> bool: ...    # asked in another process
    def request_cancel(self, task_id: str) -> bool: ...     # shared: flag it for the worker; False: save the task instead


class InMemoryStore:
    """One process: the session node and the workers share this object."""

    poll_seconds: float | None = None   # one process: a queued job always wakes the workers

    def __init__(self) -> None:
        self._tasks: dict[str, tuple[Task, AgentContext]] = {}
        self._intents: dict[str, Intent] = {}
        self._effects: dict[str, dict[str, Effect]] = {}   # task id → effect id → effect
        self._results: dict[str, TaskResult] = {}
        self._replies: dict[str, Reply] = {}
        self._jobs: dict[str, Job] = {}            # by key, queued or leased; insertion order = FIFO
        self._listeners: list[Callable[[], None]] = []

    # -- tasks ----------------------------------------------------------------------------------
    def save(self, task: Task, context: AgentContext) -> None:
        self._tasks[task.id] = (task, context)

    def load(self, task_id: str) -> tuple[Task, AgentContext] | None:
        return self._tasks.get(task_id)

    def tasks_for_intent(self, intent_id: str) -> list[Task]:
        intent = self._intents.get(intent_id)
        ids = intent.task_ids if intent else []
        return [self._tasks[i][0] for i in ids if i in self._tasks]

    def pending_consolidations(self) -> list[str]:
        """Finished tasks whose consolidation into long-term memory has not run yet."""
        return [t.id for t, _ in self._tasks.values() if getattr(t, "consolidation", None) == "pending"]

    # -- intents --------------------------------------------------------------------------------
    def save_intent(self, intent: Intent) -> None:
        self._intents[intent.id] = intent

    def load_intent(self, intent_id: str) -> Intent | None:
        return self._intents.get(intent_id)

    def intents_for_thread(self, thread_id: str, status: IntentStatus | None = None) -> list[Intent]:
        found = [i for i in self._intents.values() if i.thread_id == thread_id]
        found = [i for i in found if status is None or i.status is status]
        return sorted(found, key=lambda i: i.received_at)

    # -- effects --------------------------------------------------------------------------------
    def save_effect(self, effect: Effect) -> None:
        self._effects.setdefault(effect.task_id, {})[effect.id] = effect

    def effects_for_call(self, task_id: str, call_id: str) -> list[Effect]:
        """A call id is unique within its task only: always look effects up by both."""
        return [e for e in self.effects_for_task(task_id) if e.call_id == call_id]

    def effects_for_task(self, task_id: str) -> list[Effect]:
        return sorted(self._effects.get(task_id, {}).values(), key=lambda e: e.at)

    # -- results --------------------------------------------------------------------------------
    def save_result(self, result: TaskResult) -> None:
        if result.task_id in self._results:
            raise ValueError(f"task {result.task_id} already has a result")   # written once
        self._results[result.task_id] = result

    def load_result(self, task_id: str) -> TaskResult | None:
        return self._results.get(task_id)

    def save_reply(self, reply: Reply) -> None:
        if reply.intent_id in self._replies:
            raise ValueError(f"intent {reply.intent_id} already has a reply")
        self._replies[reply.intent_id] = reply

    def load_reply(self, intent_id: str) -> Reply | None:
        return self._replies.get(intent_id)

    # -- jobs -----------------------------------------------------------------------------------
    def enqueue(self, job: Job) -> bool:
        """Add a runnable job. A job already queued under the same key is not doubled; one a worker
        holds is marked to run again when that worker completes it (e.g. an approval that arrives
        while the worker is still finishing the step that asked for it)."""
        current = self._jobs.get(job.key)
        if current is not None:
            if current.lease_owner is not None:
                current.again = True
            return False
        self._jobs[job.key] = job
        for listener in self._listeners:
            listener()
        return True

    def claim(self, worker: str, lease_seconds: float, skip: Collection[str] = (),
              running: Mapping[str | None, int] | None = None, per_owner: int = 0) -> Job | None:
        """A job nobody holds a live lease on (an expired lease is up for grabs), except the jobs in
        `skip`: the ones the caller's pool is still running, whatever their lease says.

        Fair between owners: the job of the owner with the fewest jobs `running` comes first, the
        oldest among equals; an owner already running `per_owner` jobs (0: no limit) waits."""
        now = time.monotonic()
        counts = running or {}
        best, best_load = None, None
        for job in self._jobs.values():
            if job.key in skip or not (job.lease_owner is None or job.lease_until < now):
                continue
            load = counts.get(job.owner, 0)
            if per_owner and job.owner is not None and load >= per_owner:
                continue
            if best is None or load < best_load:
                best, best_load = job, load
                if load == 0:
                    break   # nothing can beat an owner with nothing running (and this is the oldest such)
        if best is not None:
            best.lease_owner, best.lease_until = worker, now + lease_seconds
            best.attempts += 1
        return best

    def renew(self, job: Job, worker: str, lease_seconds: float) -> bool:
        current = self._jobs.get(job.key)
        if current is not job or job.lease_owner != worker:
            return False   # lost the lease: another worker has the job now
        job.lease_until = time.monotonic() + lease_seconds
        return True

    def complete(self, job: Job, worker: str) -> None:
        if self._jobs.get(job.key) is not job or job.lease_owner != worker:
            return
        if job.again:   # queue it again, at the back
            del self._jobs[job.key]
            job.again, job.lease_owner, job.lease_until = False, None, 0.0
            self._jobs[job.key] = job
            for listener in self._listeners:
                listener()
        else:
            del self._jobs[job.key]

    def on_enqueue(self, listener: Callable[[], None]) -> None:
        self._listeners.append(listener)

    # -- several processes: nothing to do in one ------------------------------------------------
    def prepare(self, job: Job) -> None:
        pass

    def forget(self, task_ids: Collection[str] = (), intent_ids: Collection[str] = ()) -> None:
        pass

    def cancel_requested(self, task_id: str) -> bool:
        return False

    def request_cancel(self, task_id: str) -> bool:
        return False
