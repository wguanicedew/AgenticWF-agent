"""
Session: one thread's conversation, resident on the session node (docs/agent-flow.md §3, §8).

Every new message becomes an *intent* on the session's queue: the intent record is written
before send() returns, and the harness's intake turns it into tasks on whichever worker claims
it. One intent is worked on at a time; messages that arrive meanwhile wait as pending intents and
are merged when their turn comes. Input that answers a waiting task (ask_user, approvals) goes
straight to that task. The session never runs tasks itself: it records, routes and relays.

    RuntimeThread ──► Session ──► intent record (store) ──► worker: AgentHarness.intake / run / finish_intent
                         ▲                                         │
                         └──────────── events ─────────────────────┘   (relayed on the thread stream)
"""

from __future__ import annotations

import asyncio
from typing import Any

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.context.task_types import DEFAULT
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.harness.state import WAITING_STATES
from agenticwf.agent.runtime.harness.worker import Workers
from agenticwf.agent.runtime.llm.base import Usage
from agenticwf.agent.runtime.messages import ApprovalDecision, Interrupt, Op, ThreadEvent, UpdateSettings, UserInput
from agenticwf.agent.runtime.records import Intent, IntentStatus
from agenticwf.agent.runtime.session.state import SessionConfiguration, SessionState, TranscriptMessage
from agenticwf.agent.runtime.task import Task, ToolCallStatus
from agenticwf.agent.runtime.types import RuntimeState, owner_key

LOG = get_logger(__name__)


class Session:
    def __init__(self, configuration: SessionConfiguration, harness: AgentHarness, services: HarnessServices,
                 workers: Workers):
        self.thread_id = configuration.thread_id
        self.state = SessionState(configuration=configuration)
        self.state.budget.max_tokens = configuration.max_tokens
        self.harness = harness
        self.services = services   # this thread's, from harness.services_factory
        self.workers = workers
        self.closed = False
        self._events: asyncio.Queue[ThreadEvent] = asyncio.Queue()
        self._seq = 0
        self._lock = asyncio.Lock()
        # The harness is shared; events of this thread's intents and tasks come back here.
        harness.attach(self.thread_id, self._on_harness_event)

    @property
    def configuration(self) -> SessionConfiguration:
        return self.state.configuration

    @property
    def store(self):
        return self.services.store

    # -- event stream ---------------------------------------------------------------------------
    def _push(self, kind: str, data: dict[str, Any]) -> ThreadEvent:
        self._seq += 1
        event = ThreadEvent(type=kind, thread_id=self.thread_id, data=data, seq=self._seq)
        self._events.put_nowait(event)
        return event

    async def emit(self, kind: str, **data: Any) -> None:
        event = self._push(kind, data)
        if self.services.rollout is not None:
            await self.services.rollout.record_event(event)

    def _on_harness_event(self, subject: str, event: dict[str, Any]) -> None:
        """Relay a harness event, and keep the session context in step with it."""
        kind = event["type"]
        data = {k: v for k, v in event.items() if k != "type"}
        self._push(kind, data)
        # TODO: hand harness events to services.rollout as well (needs an async hand-off).
        if kind == "task_started":
            self.state.task_ids.append(data["task"])
            self._hook("TaskStart", {"thread_id": self.thread_id, "task": data["task"]})
        elif kind == "task_complete":
            self._hook("TaskStop", {"thread_id": self.thread_id, "task": data["task"]})
        elif kind == "conversation_compacted":   # the new summary folds in the one before
            self.state.summary = data["summary"]
            self.state.summarized.update(data["keys"])
        elif kind == "reply":
            usage = data.get("usage") or {}
            self.state.record_usage(Usage(usage.get("input", 0), usage.get("output", 0)))
            if data.get("answer"):
                self.state.history.append(TranscriptMessage("assistant", data["answer"], keys=(data["intent"],)))
            if self.state.active_intent_id == data["intent"]:
                self.state.active_intent_id = None
            self._start_next()

    def _hook(self, event: str, payload: dict[str, Any]) -> None:
        if self.services.hooks is not None:
            asyncio.get_running_loop().create_task(self.services.hooks.run(event, payload))

    async def next_event(self) -> ThreadEvent:
        return await self._events.get()

    # -- submission entry point -----------------------------------------------------------------
    async def submit(self, op: Op) -> str:
        if isinstance(op, UserInput):
            return await self.send_input(op)
        if isinstance(op, ApprovalDecision):
            return await self.approve(op)
        if isinstance(op, Interrupt):
            return await self.interrupt(op.auth)
        if isinstance(op, UpdateSettings):
            await self.apply_settings(op)
            return ""
        raise TypeError(f"unhandled op: {op!r}")

    def _check_owner(self, auth: AuthContext) -> None:
        if owner_key(auth) != self.configuration.owner:
            raise PermissionError("this thread belongs to another principal")

    def _intent_tasks(self) -> list[Task]:
        intent_id = self.state.active_intent_id
        return self.store.tasks_for_intent(intent_id) if intent_id else []

    def _waiting(self, state: RuntimeState) -> Task | None:
        return next((t for t in self._intent_tasks() if t.state is state), None)

    async def send_input(self, item: UserInput) -> str:
        """Answer a task waiting for the user (returns its id), or record a new intent (returns its id)."""
        self._check_owner(item.auth)
        if self.closed:
            raise RuntimeError("thread is shut down")
        async with self._lock:
            waiting = self._waiting(RuntimeState.WAITING_USER)
            if waiting is not None:
                self.harness.reply(waiting.id, item.auth, item.text)
                aid = self.state.active_intent_id
                self.state.history.append(TranscriptMessage("user", item.text, keys=(aid,) if aid else ()))
                self.harness.schedule(waiting.id)
                return waiting.id

            cfg = self.configuration
            requested = item.task_type or (cfg.task_type if cfg.task_type != DEFAULT else None)
            intent = Intent(thread_id=self.thread_id, owner=cfg.owner, auth=item.auth, text=item.text,
                            requested_task_type=requested, requested_mode=item.mode or cfg.mode, model=item.model or cfg.model,
                            source=cfg.session_source)
            if self.state.budget.is_exceeded():
                intent.status = IntentStatus.REJECTED
                self.store.save_intent(intent)
                await self.emit("error", intent=intent.id, code="thread_budget_exceeded",
                                message=f"thread token budget of {self.state.budget.max_tokens} is used up")
                return ""
            self.store.save_intent(intent)   # TO DO: written before send() returns
            self.state.intent_ids.append(intent.id)
            await self.emit("intent", intent=intent.id, status=intent.status.value)
            if self.state.active_intent_id is not None:
                queued = len(self.store.intents_for_thread(self.thread_id, IntentStatus.PENDING))
                await self.emit("input_queued", intent=self.state.active_intent_id, queued=queued)
            else:
                self._start_next()
            return intent.id

    def resume(self, intents: list[Intent]) -> None:
        """After a restart: adopt this thread's unanswered intents and queue their work.

        The intent that had become tasks is worked on again; the others (intake had not finished:
        it has no effects) start from the queue, as if just sent. The transcript the thread was
        opened with should hold the converted intent's message, not the pending ones'."""
        for intent in intents:
            self.state.intent_ids.append(intent.id)
            self.state.task_ids.extend(intent.task_ids)
            if intent.status is IntentStatus.CLAIMED:
                intent.status = IntentStatus.PENDING
                self.store.save_intent(intent)
            elif intent.status is IntentStatus.CONVERTED and self.state.active_intent_id is None:
                self.state.active_intent_id = intent.id
                self.harness.resume_intent(intent)
        self._start_next()

    def _start_next(self) -> None:
        """Hand the pending intents (merged, if several) to intake, when no intent is being worked on."""
        if self.state.active_intent_id is not None or self.closed:
            return
        pending = self.store.intents_for_thread(self.thread_id, IntentStatus.PENDING)
        if not pending:
            return
        intent = pending[0]
        if len(pending) > 1:
            last = pending[-1]
            intent = Intent(thread_id=self.thread_id, owner=last.owner, auth=last.auth,
                            text="\n\n".join(i.text for i in pending), requested_task_type=last.requested_task_type,
                            requested_mode=last.requested_mode, model=last.model, source=last.source,
                            supersedes=[i.id for i in pending])
            for old in pending:
                old.status = IntentStatus.SUPERSEDED
                self.store.save_intent(old)
            self.state.intent_ids.append(intent.id)
        # The session context is selected here, on the node that holds it: intake gets the summary of
        # the earlier conversation and what it does not cover, and compacts them if they grew too long.
        intent.conversation, intent.conversation_summary = self.state.excerpt(), self.state.summary
        self.store.save_intent(intent)
        self.state.history.append(TranscriptMessage("user", intent.text, keys=(intent.id, *intent.supersedes)))
        self.state.active_intent_id = intent.id
        self.harness.start_intent(intent.id)

    # -- approvals, interruption, settings ------------------------------------------------------
    async def approve(self, decision: ApprovalDecision) -> str:
        """Forward an approval decision; queue the task again once nothing is pending."""
        async with self._lock:
            task = next((t for t in self._intent_tasks() if t.state is RuntimeState.WAITING_APPROVAL
                         and (c := t.call(decision.call_id)) and c.status is ToolCallStatus.AWAITING_APPROVAL), None)
            if task is None:
                raise RuntimeError("no pending approval for this call")
            task = self.harness.approve(task.id, decision.call_id, decision.auth, decision.approved,
                                        remember=decision.remember, reason=decision.reason)
            if task.state is RuntimeState.TOOL_RUNNING:
                self.harness.schedule(task.id)
            return task.id

    async def interrupt(self, auth: AuthContext) -> str:
        """Cancel the intent being worked on; pending intents then start."""
        self._check_owner(auth)
        intent_id = self.state.active_intent_id
        if intent_id is None:
            return ""
        busy = self.harness.cancel_intent(intent_id, auth)
        if self.workers.abort("intake", intent_id):
            await self.workers.wait("intake", intent_id)
        for task_id in busy:
            # The worker stops at its next loop check; a pending model or tool call is cut off.
            await self.workers.wait("task", task_id, timeout=0.1)
            if self.workers.abort("task", task_id):
                await self.workers.wait("task", task_id)
            self.harness.force_cancel(task_id)
        return intent_id

    async def apply_settings(self, update: UpdateSettings) -> None:
        self._check_owner(update.auth)
        cfg = self.configuration
        if update.task_type is not None:
            cfg.task_type = update.task_type
        if update.mode is not None:
            cfg.mode = update.mode
        if update.thread_name is not None:
            cfg.thread_name = update.thread_name
        await self.emit("settings_applied", task_type=cfg.task_type,
                        mode=cfg.mode.value if cfg.mode else None, thread_name=cfg.thread_name)

    # -- status and shutdown --------------------------------------------------------------------
    def status(self) -> str:
        """idle | running | waiting (a task of the current intent waits for approval or the user)."""
        if self.state.active_intent_id is None:
            return "idle"
        return "waiting" if any(t.state in WAITING_STATES for t in self._intent_tasks()) else "running"

    async def shutdown(self) -> None:
        intent_id = self.state.active_intent_id
        if intent_id is not None:
            intent = self.store.load_intent(intent_id)
            await self.interrupt(intent.auth)
        self.closed = True
        self.harness.detach(self.thread_id)
        if self.services.rollout is not None:
            await self.services.rollout.close()
        await self.emit("shutdown_complete")
