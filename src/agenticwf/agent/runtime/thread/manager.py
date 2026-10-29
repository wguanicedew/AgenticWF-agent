"""
RuntimeThreadManager: the process-wide registry of live threads.

Built once per process by agenticwf.agent.runtime.bootstrap.build_thread_manager around the process's
AgentHarness and its Workers. It owns no services itself: AgentHarness.services_factory builds
one HarnessServices per thread. Sessions stay resident here; their work runs on the workers
(docs/agent-flow.md §8).

Threads belong to one principal: every lookup takes the caller's AuthContext and returns None for
threads owned by someone else, so a host answers 404 without revealing that the id exists.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.worker import Workers
from agenticwf.agent.runtime.llm.base import Message
from agenticwf.agent.runtime.session.session import Session
from agenticwf.agent.runtime.session.state import SessionConfiguration
from agenticwf.agent.runtime.thread.thread import RuntimeThread
from agenticwf.agent.runtime.types import Mode, owner_key

LOG = get_logger(__name__)


@dataclass
class StartThreadOptions:
    auth: AuthContext
    task_type: str = "default"
    mode: Mode | None = None
    model: str | None = None
    session_source: str = "web"
    thread_name: str | None = None
    parent_thread_id: str | None = None
    forked_from_thread_id: str | None = None
    reserved_thread_id: str | None = None
    initial_history: list[Message] = field(default_factory=list)
    summary: str = ""                                       # the stored summary of the earlier conversation
    summary_keys: frozenset[str] = frozenset()              # what it covers (TranscriptMessage.keys)
    max_tokens: int | None = None


@dataclass
class NewThread:
    thread_id: str
    thread: RuntimeThread


@dataclass
class ThreadShutdownReport:
    completed: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    timed_out: list[str] = field(default_factory=list)


@dataclass
class TruncateBeforeNthUserMessage:
    n: int


@dataclass
class Interrupted:
    pass


ForkSnapshot = TruncateBeforeNthUserMessage | Interrupted


def _truncate_history(history: list[Message], snapshot: ForkSnapshot) -> list[Message]:
    if isinstance(snapshot, Interrupted):
        return list(history) + [Message("user", "[runtime] The previous task was interrupted.")]
    seen = 0
    for idx, message in enumerate(history):
        if message.role == "user":
            seen += 1
            if seen == snapshot.n:
                return list(history[:idx])
    return list(history)


class RuntimeThreadManager:
    def __init__(self, harness: AgentHarness, workers: Workers | None = None,
                 max_threads_per_owner: int | None = None):
        self._harness = harness
        self.workers = workers or Workers(harness)
        self._threads: dict[str, RuntimeThread] = {}
        self._lock = asyncio.Lock()
        self._id_generator: Callable[[], str] = lambda: uuid.uuid4().hex
        self._subscribers: list[asyncio.Queue[str]] = []
        self.max_threads_per_owner = max_threads_per_owner
        self.crons = None   # agenticwf.agent.runtime.cron.service.CronService, when the deployment runs crons

    @property
    def harness(self) -> AgentHarness:
        return self._harness

    def with_thread_id_generator(self, generator: Callable[[], str]) -> "RuntimeThreadManager":
        self._id_generator = generator
        return self

    def reserve_thread_id(self) -> str:
        """Pre-allocate an id a host can attach state to before the thread exists."""
        return self._id_generator()

    def subscribe_thread_created(self) -> "asyncio.Queue[str]":
        queue: asyncio.Queue[str] = asyncio.Queue()
        self._subscribers.append(queue)
        return queue

    # -- creating threads -----------------------------------------------------------------------
    async def start_thread(self, options: StartThreadOptions) -> NewThread:
        owner = owner_key(options.auth)
        async with self._lock:
            if self.max_threads_per_owner is not None:
                live = sum(1 for t in self._threads.values() if t.owner == owner)
                if live >= self.max_threads_per_owner:
                    raise RuntimeError(f"at most {self.max_threads_per_owner} live threads per user")
        thread_id = options.reserved_thread_id or self._id_generator()
        if thread_id in self._threads:
            raise KeyError(f"thread {thread_id} is already open")
        configuration = SessionConfiguration(
            thread_id=thread_id, owner=owner, task_type=options.task_type, mode=options.mode,
            model=options.model, session_source=options.session_source, thread_name=options.thread_name,
            parent_thread_id=options.parent_thread_id, forked_from_thread_id=options.forked_from_thread_id,
            max_tokens=options.max_tokens,
        )
        session = Session(configuration, self._harness, self._harness.services_factory(configuration), self.workers)
        session.state.history.extend(options.initial_history)
        session.state.summary, session.state.summarized = options.summary, set(options.summary_keys)
        thread = RuntimeThread(thread_id=thread_id, session=session)
        async with self._lock:
            self._threads[thread_id] = thread
        await session.emit("session_configured", task_type=configuration.task_type,
                           mode=configuration.mode.value if configuration.mode else None,
                           source=configuration.session_source)
        for queue in self._subscribers:
            queue.put_nowait(thread_id)
        return NewThread(thread_id=thread_id, thread=thread)

    async def fork_thread(self, source_thread_id: str, auth: AuthContext, snapshot: ForkSnapshot,
                          **overrides: Any) -> NewThread:
        """A new thread with the source's settings and a truncated copy of its transcript."""
        source = await self.get_thread(source_thread_id, auth)
        if source is None:
            raise KeyError(f"no such thread: {source_thread_id}")
        cfg = source.session.configuration
        options = StartThreadOptions(
            auth=auth,
            task_type=overrides.pop("task_type", cfg.task_type),
            mode=overrides.pop("mode", cfg.mode),
            model=overrides.pop("model", cfg.model),
            session_source=cfg.session_source,
            parent_thread_id=cfg.parent_thread_id,
            forked_from_thread_id=source_thread_id,
            initial_history=_truncate_history(source.history(), snapshot),
            summary=source.session.state.summary,
            summary_keys=frozenset(source.session.state.summarized),
            **overrides,
        )
        return await self.start_thread(options)

    # TODO: resume_thread(thread_id, auth): rebuild a thread from its persisted transcript and
    # usage (RolloutRecorder), for threads that are no longer in memory.

    # -- lookups (owner-scoped) -----------------------------------------------------------------
    async def get_thread(self, thread_id: str, auth: AuthContext) -> RuntimeThread | None:
        async with self._lock:
            thread = self._threads.get(thread_id)
        if thread is None or thread.owner != owner_key(auth):
            return None
        return thread

    async def list_threads(self, auth: AuthContext) -> list[RuntimeThread]:
        owner = owner_key(auth)
        async with self._lock:
            return [t for t in self._threads.values() if t.owner == owner]

    async def remove_thread(self, thread_id: str, auth: AuthContext) -> bool:
        thread = await self.get_thread(thread_id, auth)
        if thread is None:
            return False
        await thread.shutdown()
        async with self._lock:
            self._threads.pop(thread_id, None)
        return True

    async def shutdown_all_threads(self, timeout: float = 5.0) -> ThreadShutdownReport:
        async with self._lock:
            threads = list(self._threads.values())
        report = ThreadShutdownReport()
        for thread in threads:
            try:
                await asyncio.wait_for(thread.shutdown(), timeout=timeout)
                report.completed.append(thread.thread_id)
            except asyncio.TimeoutError:
                report.timed_out.append(thread.thread_id)
            except Exception:   # shutdown must not raise past this point
                LOG.exception("thread %s failed to shut down cleanly", thread.thread_id)
                report.failed.append(thread.thread_id)
        async with self._lock:
            for thread_id in (*report.completed, *report.timed_out, *report.failed):
                self._threads.pop(thread_id, None)
        await self.workers.stop()
        return report
