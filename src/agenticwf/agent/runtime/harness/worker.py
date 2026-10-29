"""
Workers: run the jobs in the store (docs/agent-flow.md §8).

    intake:<intent>   AgentHarness.intake       parse, split, type, create tasks
    task:<task>       AgentHarness.run          planning, or one turn; queued again for the next turn
    reply:<intent>    AgentHarness.finish_intent
    consolidate:<task>  AgentHarness.consolidate  working memory into long-term memory
    <other>:<ref>     AgentHarness.job_handlers[kind] (agenticwf.agent.runtime.cron: report)

A worker claims a job with a lease, renews it while the job runs, and keeps nothing afterwards:
everything is loaded from the store and written back. A worker that dies stops renewing, its
lease expires, and another worker claims the job and resumes from the execution record. A task
that waits for an approval or the user is not a job: the approval or reply queues it again.

Each job is a span (agenticwf.agent.runtime.telemetry), with the job's model and tool calls inside it,
linked to the span that queued it (an HTTP request, or the job before), and is timed and counted.

This is the in-process pool; with a database store the same loop runs in separate worker
processes.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.state import WAITING_STATES
from agenticwf.agent.runtime.harness.store import Job
from agenticwf.agent.runtime.telemetry import annotate, fail, measure_job, span
from agenticwf.agent.runtime.types import RuntimeState

LOG = get_logger(__name__)
_POOLS = itertools.count(1)


class Workers:
    def __init__(self, harness: AgentHarness, size: int = 4, lease_seconds: float = 60.0,
                 per_owner: int | None = None):
        """`per_owner`: at most this many of the pool's jobs for one owner at once, so one person's
        work never takes every worker. Default: all but one worker (0: no limit)."""
        self.harness = harness
        self.size = size
        self.per_owner = max(1, size - 1) if per_owner is None else per_owner
        self.lease_seconds = lease_seconds
        self.name = f"pool{next(_POOLS)}"
        self._loops: list[asyncio.Task] = []
        self._running: dict[str, asyncio.Task] = {}   # job key → the task executing it
        self._owners: dict[str, str | None] = {}       # job key → whose job it is (running ones)
        self._wake: asyncio.Event | None = None
        self._event_loop: asyncio.AbstractEventLoop | None = None   # the loop the workers run on
        harness.store.on_enqueue(self._on_enqueue)

    # -- lifecycle ------------------------------------------------------------------------------
    def start(self) -> None:
        """Start the worker loops on the running event loop. The API starts the pool when the app
        starts; without that (scripts, tests) the first enqueue made on an event loop starts it."""
        if self._loops:
            return
        self._event_loop = asyncio.get_running_loop()   # RuntimeError without one, before any coroutine is created
        self._wake = asyncio.Event()
        self._loops = [asyncio.create_task(self._loop(f"{self.name}-w{n}")) for n in range(self.size)]

    async def stop(self) -> None:
        loops, self._loops, self._event_loop = self._loops, [], None
        for task in (*loops, *self._running.values()):
            task.cancel()
        for task in (*loops, *list(self._running.values())):
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._running.clear()
        self._owners.clear()

    def _on_enqueue(self) -> None:
        """A job was queued: wake the workers. Safe from any thread (a route run in the thread
        pool, a background thread): off the workers' loop, the wake-up is handed to that loop."""
        if self._loops and self._event_loop is not None:
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None
            if running is self._event_loop:
                self._wake.set()
            else:
                try:
                    self._event_loop.call_soon_threadsafe(self._wake.set)
                except RuntimeError:   # the loop has closed (shutting down): the job waits for a restart
                    pass
            return
        try:
            self.start()
        except RuntimeError:   # no pool and no running event loop: the job waits until the pool starts
            return
        self._wake.set()

    # -- jobs, as the session sees them ---------------------------------------------------------
    def is_running(self, kind: str, ref: str) -> bool:
        runner = self._running.get(f"{kind}:{ref}")
        return runner is not None and not runner.done()

    def abort(self, kind: str, ref: str) -> bool:
        """Stop a job mid-await (an interrupt that cannot wait for the next loop check)."""
        runner = self._running.get(f"{kind}:{ref}")
        if runner is None or runner.done():
            return False
        runner.cancel()
        return True

    async def wait(self, kind: str, ref: str, timeout: float | None = None) -> None:
        runner = self._running.get(f"{kind}:{ref}")
        if runner is not None and not runner.done():
            await asyncio.wait([runner], timeout=timeout)

    # -- the loop -------------------------------------------------------------------------------
    async def _loop(self, worker: str) -> None:
        store = self.harness.store
        while True:
            # Never claim a job this pool is still running, even if its lease lapsed (a stall, a
            # suspended laptop): two runs of one task would fight over its steps.
            job = self._claim(worker)
            if job is None:
                self._wake.clear()
                job = self._claim(worker)   # an enqueue may have raced the clear
                if job is None:
                    poll = store.poll_seconds   # shared: other processes queue jobs too, so look again
                    if poll is None:
                        await self._wake.wait()
                    else:
                        with contextlib.suppress(asyncio.TimeoutError):
                            await asyncio.wait_for(self._wake.wait(), poll)
                    continue
            await self._run(job, worker)

    def _claim(self, worker: str) -> Job | None:
        running: dict[str | None, int] = {}
        for owner in self._owners.values():
            running[owner] = running.get(owner, 0) + 1
        job = self.harness.store.claim(worker, self.lease_seconds, skip=self._running, running=running,
                                       per_owner=self.per_owner)
        if job is not None:
            self._owners[job.key] = job.owner
        return job

    async def _run(self, job: Job, worker: str) -> None:
        store = self.harness.store
        runner = asyncio.create_task(self._execute(job))
        self._running[job.key] = runner
        heartbeat = asyncio.create_task(self._heartbeat(job, worker, runner))
        try:
            await asyncio.wait([runner])
        finally:
            heartbeat.cancel()
            self._running.pop(job.key, None)
            self._owners.pop(job.key, None)
        again = False
        if not runner.cancelled():
            error = runner.exception()
            if error is not None:
                LOG.error("job %s failed", job.key, exc_info=error)
                try:
                    self.harness.job_failed(job, error)
                except Exception:   # the worker loop must survive any job
                    LOG.exception("could not report the failure of %s", job.key)
            else:
                again = runner.result()
        # No await between the job ending and these: an approval cannot slip in between.
        store.complete(job, worker)
        if again:
            store.enqueue(Job(job.kind, job.ref, owner=job.owner, trace=job.trace))   # the next turn: the same origin

    async def _execute(self, job: Job) -> bool:
        """Run one job, in a span of its own. True when the task should be queued again (its next turn)."""
        with span(f"job {job.kind}", links=[job.trace], agenticwf__job__kind=job.kind,
                  agenticwf__job__ref=job.ref) as current, measure_job(job.kind) as measured:
            return await _run_job(self.harness, job, current, measured)

    async def _heartbeat(self, job: Job, worker: str, runner: asyncio.Task) -> None:
        while True:
            await asyncio.sleep(self.lease_seconds / 3)
            if not self.harness.store.renew(job, worker, self.lease_seconds):
                LOG.warning("lost the lease on %s: stopping it (another worker may have it)", job.key)
                runner.cancel()   # one task never runs twice at once
                return


async def _run_job(harness: AgentHarness, job: Job, current, measured: dict[str, str]) -> bool:
    harness.store.prepare(job)   # shared: read what another process may have changed
    if job.kind == "intake":
        await harness.intake(job.ref)
        return False
    if job.kind == "task":
        task = await harness.run(job.ref, until_turn_boundary=True)
        annotate(current, agenticwf__task__state=task.state.value, agenticwf__task__type=task.task_type.effective,
                 agenticwf__intent__id=task.intent_id, agenticwf__task__turn=task.active_turn)
        measured["task_state"] = task.state.value
        if task.state is RuntimeState.FAILED:
            fail(current, (task.error or {}).get("code", "failed"))
        return not task.state.terminal and task.state not in WAITING_STATES
    if job.kind == "reply":
        await harness.finish_intent(job.ref)
        return False
    if job.kind == "consolidate":
        await harness.consolidate(job.ref)
        return False
    handler = harness.job_handlers.get(job.kind)   # e.g. agenticwf.agent.runtime.cron's "report"
    if handler is not None:
        await handler(job.ref)
        return False
    raise ValueError(f"unknown job kind {job.kind!r}")
