"""
AgentHarness: turns intents into tasks and runs them (docs/agent-flow.md, docs/agent-runtime.md
§7–8, docs/agent-structure.md §2).

    intake(intent)      parse · split into tasks · type each task · create them      (a job)
    run(task)           the task lifecycle below                                     (a job per turn)
    finish_intent()     combine the task results into the reply                      (a job)
    consolidate(task)   what lasts of a finished task into long-term memory          (a job)

The model decides what happens next; the harness only moves the lifecycle:

    while not terminal and not waiting:
        queued          inputs from the tasks it depends on → planning | model_running
        planning        one model call: the plan; the first turn starts       → model_running
        model_running   one step of the active turn: render, LLM.complete
                        → verifying (the turn's answer) | tool_requested
        tool_requested  ToolDispatcher.prepare each call; control tools handled here
                        → tool_running | waiting_approval
        tool_running    ToolDispatcher.execute approved calls, append results
                        → model_running | waiting_user (ask_user) | verifying (complete_turn)
        verifying       parse the turn result, run its checks
                        → model_running (retry, or the next turn) | combining | completed
        combining       one model call over the turn results                  → completed
    waiting_*           suspended; approve() / reply() / cancel() resume it, then run() again

The harness is the only component with side effects on tasks. It never interprets AGENTS.md or
WORKFLOW.md, holds no permission rules, and does not know how any tool works. Everything a job
needs is loaded from the store, so any worker can run any job (agenticwf.agent.runtime.harness.worker).

One harness serves every thread in the process. Its collaborators are an HarnessServices
(harness.services); `services_factory` derives each thread's HarnessServices from it. A session
attaches its event sink under its thread id, so a task's events reach the session it belongs to.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import string
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger, log_context
from agenticwf.agent.runtime.context.agent_context import AgentContext
from agenticwf.agent.runtime.context.task_types import DEFAULT
from agenticwf.agent.runtime.harness.planning import IntakeSettings, parse_plan, parse_tasks
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.harness.state import WAITING_STATES, check_transition
from agenticwf.agent.runtime.harness.store import Job, RuntimeStore
from agenticwf.agent.runtime.llm.base import LLMError, LLMRequest, LLMResponse, Message, Usage
from agenticwf.agent.runtime.llm.routing import raise_tier, tier_rank
from agenticwf.agent.runtime.policy.engine import PolicyEngine
from agenticwf.agent.runtime.records import (EffectStatus, Intent, IntentStatus, PlanItem, Reply, StepRecord, TaskResult, TaskSpec,
                             Turn, TurnResult, TurnState)
from agenticwf.agent.runtime.session.state import SessionConfiguration
from agenticwf.agent.runtime.skills.skill import SkillError
from agenticwf.agent.runtime.task import Task, ToolCall, ToolCallStatus
from agenticwf.agent.runtime.telemetry import record_task_ended, trace_id
from agenticwf.agent.runtime.tools.control import TURN_TOOL_NAMES
from agenticwf.agent.runtime.tools.dispatcher import Outcome, ToolDispatcher
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.types import SYSTEM_SCOPE, Mode, Owner, RuntimeState, owner_key, scope_of

LOG = get_logger(__name__)

# Calls that write a task's own work (its steps, plan, answer): they keep the task's model, pin and tier.
_TASK_PURPOSES = frozenset({"step", "cron_check", "plan", "final_report", "reply"})

# States in which a worker is inside an await for the task: cancel() asks it to stop.
_BUSY = frozenset({RuntimeState.PLANNING, RuntimeState.MODEL_RUNNING, RuntimeState.TOOL_RUNNING,
                   RuntimeState.VERIFYING, RuntimeState.COMBINING})


@dataclass(frozen=True)
class Limits:
    max_steps: int = 50
    max_tool_calls: int = 200
    deadline_seconds: float = 3600
    wrap_up_at: int = 3                # steps left when the wrap-up notice appears
    compact_at_tokens: int = 100_000
    keep_recent_messages: int = 12
    recall_k: int = 5
    turn_attempts: int = 2             # a turn that fails its checks is reopened once
    conversation_messages: int = 20    # an intent's transcript excerpt is compacted past this many messages,
    conversation_tokens: int = 8000    # or past this many tokens (estimated)
    conversation_keep: int = 4         # the newest messages compaction keeps as they are


# on_complete hook: return None to accept the task's answer, or a message that reopens its last
# turn (it is shown to the model, e.g. "verification has not succeeded since the last edit").
CompletionHook = Callable[[Task, AgentContext], "str | None"]
EventSink = Callable[[str, dict[str, Any]], None]   # (owner subject, event)


class HarnessError(Exception):
    pass


class AgentHarness:
    def __init__(self, services: HarnessServices, events: EventSink | None = None,
                 limits: Limits | None = None, model: str = "default",
                 on_complete: list[CompletionHook] | None = None, intake: IntakeSettings | None = None):
        self.services = services
        self.events = events or (lambda subject, event: None)
        self.limits = limits or Limits()
        self.model = model   # default; a task may name its own. TODO: model router, by task type (TaskTypeSettings.model)
        self.on_complete = list(on_complete or ())
        self.intake_settings = intake or IntakeSettings()
        self._sinks: dict[str, EventSink] = {}   # session id → that session's event sink
        # Called when every task of an intent has ended, before its reply is saved (agenticwf.agent.runtime.cron
        # finishes its runs here). Must be idempotent: a restart may call it again.
        self.intent_hooks: list[Callable[[Intent, list[Task]], None]] = []
        # Worker jobs of other kinds (agenticwf.agent.runtime.cron: "report"): kind -> async function of the job's ref.
        self.job_handlers: dict[str, Callable[[str], Any]] = {}
        # A worker process ([harness] shared): events for sessions that live in the API process go
        # here (agenticwf.agent.runtime.harness.relay) as (session id, subject, event).
        self.remote: Callable[[str, str, dict[str, Any]], None] | None = None
        services.tool_dispatcher.on_record = self._record_call
        services.tool_dispatcher.on_effect = lambda effect: self.store.save_effect(effect)

    @property
    def dispatcher(self) -> ToolDispatcher:
        return self.services.tool_dispatcher

    @property
    def policy(self) -> PolicyEngine:
        return self.services.policy

    @property
    def store(self) -> RuntimeStore:
        return self.services.store

    # ============================================================================================
    # Sessions
    # ============================================================================================
    def services_factory(self, configuration: SessionConfiguration) -> HarnessServices:
        """One thread's services: the shared ones, with the per-thread slots filled in."""
        # TODO: a RolloutRecorder per thread (configuration.thread_id), once one exists.
        return dataclasses.replace(self.services)

    def attach(self, session_id: str, sink: EventSink) -> None:
        """Route the events of this session's intents and tasks to `sink`."""
        self._sinks[session_id] = sink

    def detach(self, session_id: str) -> None:
        self._sinks.pop(session_id, None)

    # ============================================================================================
    # Intents: intake and the reply (docs/agent-flow.md §4, §7)
    # ============================================================================================
    def start_intent(self, intent_id: str) -> None:
        """Queue intake for a pending intent the session has written."""
        intent = self.store.load_intent(intent_id)
        self._enqueue("intake", intent_id, intent.owner if intent else None)

    async def _prefetch(self, owner: Owner, text: str, recall: bool = False) -> list | None:
        """Before tools are selected (and memories recalled) for `text`: what may wait on the network
        or the database, done in a thread so the event loop serves everyone else meanwhile."""
        registry, memory = self.dispatcher.registry, self.services.context_manager.memory
        if registry.embeddings is not None and text:
            texts = registry.embedding_texts(owner)
            await asyncio.to_thread(registry.embeddings.prepare, text, texts)
        if recall:
            return await asyncio.to_thread(memory.recall, owner, text, self.limits.recall_k)
        return None

    def _enqueue(self, kind: str, ref: str, owner: Owner | None) -> None:
        """Queue a job, with whose work it is (workers share themselves fairly between owners)."""
        self.store.enqueue(Job(kind, ref, owner=scope_of(owner) if owner else None))

    def schedule(self, task_id: str) -> None:
        """Queue a task for a worker (after creation, an approval, or a reply)."""
        loaded = self.store.load(task_id)
        self._enqueue("task", task_id, loaded[0].owner if loaded else None)

    async def intake(self, intent_id: str) -> list[Task]:
        """Parse the intent, split it into typed tasks, create them, and queue those that can start."""
        intent = self.store.load_intent(intent_id)
        if intent is None or intent.status not in (IntentStatus.PENDING, IntentStatus.CLAIMED):
            return []
        intent.status = IntentStatus.CLAIMED
        self.store.save_intent(intent)
        if intent.planned:   # known in advance (a cron's checks): no conversation, no split
            specs = list(intent.planned)
        else:
            if await self._compact_conversation(intent):
                self.store.save_intent(intent)   # a restart does not compact it again
            specs = await self._split(intent)
        if intent.status is not IntentStatus.CLAIMED:   # cancelled while the split call ran
            return []

        tasks: list[Task] = []
        ids: dict[str, str] = {}
        for spec in specs:
            recalled = await self._prefetch(intent.owner, spec.request, recall=True)
            task = self.create_task(intent.auth, spec.request, task_type=spec.task_type, mode=intent.requested_mode,
                                    session_id=intent.thread_id, history=_conversation(intent), model=intent.model,
                                    intent_id=intent.id, depends_on=[ids[k] for k in spec.depends_on],
                                    source=intent.source, spec=spec if intent.planned else None, recalled=recalled)
            ids[spec.key] = task.id
            tasks.append(task)
        intent.task_ids = [t.id for t in tasks]
        intent.status = IntentStatus.CONVERTED
        self.store.save_intent(intent)
        self._publish_thread(intent, "tasks_created", intent=intent.id, tasks=[
            {"task": t.id, "task_type": t.task_type.requested, "depends_on": t.depends_on, "request": t.input[:200]}
            for t in tasks])
        for task in tasks:
            if not task.depends_on:
                self.schedule(task.id)
        return tasks

    async def _split(self, intent: Intent) -> list[TaskSpec]:
        requested = intent.requested_task_type
        single = [TaskSpec("A", intent.text, requested or DEFAULT)]
        if not self.intake_settings.split:
            return single
        types = [t.ref for t in self.services.task_types.catalogue(intent.owner)]   # bare names, ids where ambiguous
        prompt = (f"Task types you can choose from: {', '.join(types)}.\n\n"
                  f"<request>\n{intent.text}\n</request>")
        request = LLMRequest(model=intent.model or self.model, owner=intent.owner, system=self._prompt("split"),
                             messages=[*_conversation(intent), Message("user", prompt)], purpose="split",
                             intent_id=intent.id)
        specs = None
        for attempt in range(2):   # an answer that does not parse: once more, one tier up (docs/model-routing.md §6)
            try:
                response = await self.services.llm.complete(request)
            except LLMError as exc:
                LOG.warning("intake split unavailable, one task: %s", exc)
                return single
            intent.usage.add(response.usage)
            specs = parse_tasks(response.text, self.intake_settings.max_tasks)
            router = self._router()
            if specs is not None or attempt or router is None or not router.can_escalate(intent.owner):
                break
            router.feedback("escalation", "split", response.model_id or "?", intent.owner, intent_id=intent.id,
                            reason="split did not parse")
            request = dataclasses.replace(request, min_tier=raise_tier(router.tier_for("split")))
        if specs is None:
            LOG.info("intake split returned no usable tasks, one task")
            return single
        if requested:   # the user asked for a type: it applies to every task
            specs = [dataclasses.replace(s, task_type=requested) for s in specs]
        return specs

    def resume_intent(self, intent: Intent) -> None:
        """After a restart: queue whatever the intent was doing (docs/agent-flow.md §9, recovery)."""
        if intent.status is IntentStatus.CLAIMED:   # intake has no effects: run it again
            intent.status = IntentStatus.PENDING
            self.store.save_intent(intent)
        if intent.status is IntentStatus.PENDING:
            self.start_intent(intent.id)
            return
        tasks = self.store.tasks_for_intent(intent.id)
        if tasks and all(t.state.terminal for t in tasks):
            self._enqueue("reply", intent.id, intent.owner)
            return
        for task in tasks:
            if task.state.terminal or task.state in WAITING_STATES:
                continue
            if task.state is RuntimeState.QUEUED and task.depends_on:
                continue   # started when its dependencies finish: _advance_intent, below
            self.schedule(task.id)
        # Dependencies may have finished before the stop, with nothing left to start their dependents.
        self._advance_intent(intent.id)

    def cancel_intent(self, intent_id: str, by: AuthContext) -> list[str]:
        """Cancel every task of the intent. Returns the tasks a worker is still inside."""
        self.store.forget(intent_ids=[intent_id])   # shared workers: its latest state
        intent = self.store.load_intent(intent_id)
        if intent is None:
            return []
        if owner_key(by) != intent.owner:
            raise PermissionError("only the owner can cancel")
        tasks = self.store.tasks_for_intent(intent_id)
        if not tasks:   # still in intake (or never got there)
            if intent.open:
                intent.status = IntentStatus.CANCELLED
                self.store.save_intent(intent)
                self._enqueue("reply", intent_id, intent.owner)
            return []
        busy = []
        for task in tasks:
            if not task.state.terminal:
                self.cancel(task.id, by)
                if not task.state.terminal:
                    busy.append(task.id)
        return busy

    async def finish_intent(self, intent_id: str) -> Reply | None:
        """Every task is terminal: combine their results into the reply, once."""
        intent = self.store.load_intent(intent_id)
        if intent is None or intent.status is IntentStatus.ANSWERED:
            return None
        tasks = self.store.tasks_for_intent(intent_id)
        if any(not t.state.terminal for t in tasks):
            return None
        results = [r for r in (self.store.load_result(t.id) for t in tasks) if r is not None]
        usage = Usage()
        usage.add(intent.usage)
        for task in tasks:
            usage.add(self._load(task.id)[1].usage)

        for hook in self.intent_hooks:
            hook(intent, tasks)
        ok = [r for r in results if r.status == "succeeded"]
        if intent.planned:   # tasks known in advance (a cron's checks): their own results, no model call
            status = "succeeded" if results and len(ok) == len(results) else ("partial" if ok else "failed")
            answer = "\n\n".join(f"{t.input.splitlines()[0][:200]}: {self.store.load_result(t.id).answer or ''}".strip()
                                  for t in tasks if self.store.load_result(t.id) is not None)
        elif not results:
            status, answer = ("cancelled" if intent.status is IntentStatus.CANCELLED else "failed"), ""
        elif len(results) == 1:
            status, answer = results[0].status, results[0].answer or _error_text(results[0])
        else:
            if len(ok) == len(results):
                status = "succeeded"
            elif ok:
                status = "partial"
            else:
                status = "cancelled" if all(r.status == "cancelled" for r in results) else "failed"
            answer = await self._combine_results(intent, tasks, results, usage) if ok else \
                "\n\n".join(_error_text(r) for r in results)
        reply = Reply(intent_id=intent.id, task_ids=[t.id for t in tasks], status=status, answer=answer,
                      artifacts=sorted({a for r in results for a in r.artifacts}), usage=usage)
        self.store.save_reply(reply)
        intent.status = IntentStatus.ANSWERED
        self.store.save_intent(intent)
        self._publish_thread(intent, "reply", intent=intent.id, status=status, answer=answer, tasks=reply.task_ids,
                             artifacts=reply.artifacts, usage={"input": usage.input_tokens, "output": usage.output_tokens})
        return reply

    async def _combine_results(self, intent: Intent, tasks: list[Task], results: list[TaskResult],
                               usage: Usage) -> str:
        requests = {t.id: t.input for t in tasks}
        parts = [f"## Task: {requests.get(r.task_id, '')}\nStatus: {r.status}\n{r.answer or _error_text(r)}"
                 for r in results]
        request = LLMRequest(model=intent.model or self.model, owner=intent.owner, system=self._prompt("reply"),
                             purpose="reply", intent_id=intent.id, messages=[
            Message("user", f"<request>\n{intent.text}\n</request>\n\n" + "\n\n".join(parts))])
        try:
            response = await self.services.llm.complete(request)
        except LLMError as exc:
            LOG.warning("reply combine unavailable: %s", exc)
            return "\n\n".join(parts)
        usage.add(response.usage)
        return response.text

    async def _compact_conversation(self, intent: Intent) -> bool:
        """Keep the transcript an intent carries within `conversation_messages` and
        `conversation_tokens`. All but the newest `conversation_keep` messages are folded into the
        summary so far with one model call; the fold ends on an intent boundary, so a summary covers
        whole requests with their replies. The session hears of the new summary and the keys it now
        covers (`conversation_compacted`), keeps it with the chat, and from then on sends only the
        summary and what it does not cover: each compaction reads the summary and the messages since,
        never the whole transcript again. Messages that alone are still too long are clipped.
        Without a model, only the newest messages are kept. Returns whether the intent changed."""
        limits = self.limits
        messages = intent.conversation
        if (len(messages) <= limits.conversation_messages
                and _tokens(intent.conversation_summary, messages) <= limits.conversation_tokens):
            return False
        cut = max(0, len(messages) - max(0, limits.conversation_keep))
        while 0 < cut < len(messages) and set(_keys(messages[cut - 1])) & set(_keys(messages[cut])):
            cut -= 1   # never split an intent: its request and reply stay together
        fold = messages[:cut]
        if fold:
            earlier = (f"<summary so far>\n{intent.conversation_summary}\n</summary so far>\n\n"
                       if intent.conversation_summary else "")
            transcript = "\n\n".join(f"{m.role}: {m.content[:8000]}" for m in fold)
            request = LLMRequest(model=intent.model or self.model, owner=intent.owner,
                                 system=self._prompt("compact_conversation", max_words=limits.conversation_tokens // 5),
                                 messages=[Message("user", f"{earlier}<conversation>\n{transcript}\n</conversation>")],
                                 purpose="compact_conversation", intent_id=intent.id)
            try:
                response = await self.services.llm.complete(request)
            except LLMError as exc:
                LOG.warning("conversation compaction unavailable, keeping the newest messages: %s", exc)
                intent.conversation = messages[-limits.conversation_messages:]
            else:
                intent.usage.add(response.usage)
                intent.conversation_summary = response.text.strip()
                intent.conversation = messages[cut:]
                keys = sorted({k for m in fold for k in _keys(m)})
                self._publish_thread(intent, "conversation_compacted", intent=intent.id,
                                     summary=intent.conversation_summary, keys=keys)
        if _tokens(intent.conversation_summary, intent.conversation) > limits.conversation_tokens:
            room = limits.conversation_tokens * 4 - len(intent.conversation_summary)
            share = max(500, room // max(1, len(intent.conversation)))
            intent.conversation = [m if len(m.content) <= share else dataclasses.replace(m, content=m.content[:share] + " […]")
                                   for m in intent.conversation]
        return True

    def job_failed(self, job: Job, error: BaseException) -> None:
        """A job raised (a harness bug): report it and end what it was working on."""
        if job.kind == "task":
            task, ctx = self._load(job.ref)
            self._publish(task, "error", code="harness_error", message=str(error))
            task.cancel_requested = False
            if not task.state.terminal:   # a failure, not a cancellation (docs/agent-runtime.md §7)
                self._fail(task, ctx, "harness_error", "The runtime hit an internal error; see the server log.")
            return
        if job.kind == "consolidate":   # stays pending: the next start tries again
            return
        intent = self.store.load_intent(job.ref)
        if intent is None:
            return
        self._publish_thread(intent, "error", intent=intent.id, code="harness_error", message=str(error))
        if job.kind == "intake" and intent.open and not intent.task_ids:
            intent.status = IntentStatus.CANCELLED
            self.store.save_intent(intent)
            self._enqueue("reply", intent.id, intent.owner)

    # ============================================================================================
    # Tasks: entry points
    # ============================================================================================
    def create_task(self, auth: AuthContext, text: str, task_type: str = "default", mode: Mode | None = None,
                    session_id: str | None = None, history: list[Message] | None = None,
                    model: str | None = None, intent_id: str | None = None,
                    depends_on: list[str] | None = None, source: str = "web", spec: TaskSpec | None = None,
                    recalled: list | None = None) -> Task:
        """Accept a task: snapshot instructions, build the task context, recall memory. A task that
        needs no plan and waits for nothing starts its single turn now (tools selected). A planned
        task's `spec` adds instructions, narrows tools and skills, sets its deadline, and says what
        an approval becomes with nobody to ask (`unattended`)."""
        resolved = self.services.task_types.resolve(task_type, (auth.subject, auth.organization_id))
        settings = resolved.settings
        if spec is not None and spec.mode:
            mode = Mode(spec.mode)
        if mode is not None and mode not in settings.allowed_modes():   # never looser than the type permits
            LOG.info("mode %s is not allowed for task type %s: using %s", mode.value, resolved.effective,
                     settings.mode.value)
            mode = None
        task = Task(auth=auth, input=text, task_type=resolved, mode=mode or settings.mode,
                    session_id=session_id, intent_id=intent_id, depends_on=list(depends_on or ()),
                    # the thread's or message's choice, else the task type's, else the default
                    model=model or settings.model or self.model,
                    deadline=time.time() + self.limits.deadline_seconds, source=source,
                    # a model chosen for it (message, thread, task type, cron) is used as is: not routed
                    pinned=bool(model or settings.model), min_tier=(spec.tier if spec else None) or settings.tier)
        if spec is not None:
            task.tool_globs, task.skill_names, task.unattended = spec.tools, spec.skills, spec.unattended
            task.cron = dict(spec.cron) if spec.cron else None
            if spec.timeout:
                task.deadline = time.time() + spec.timeout
        ctx = self.services.context_manager.build(
            task, list(history or ()), max_steps=settings.max_steps or self.limits.max_steps,
            wrap_up_at=self.limits.wrap_up_at, recall_k=self.limits.recall_k, preamble=self._prompt("preamble"),
            recalled=recalled)
        if spec is not None:
            ctx.extra_instructions = list(spec.instructions)
        task.recalled = tuple({"id": m.id, "text": m.text} for m in ctx.recalled_memories)
        self.store.save(task, ctx)
        self._publish(task, "task", status=task.state.public_status, state=task.state.value,
                      task_type=resolved.requested, instructions=resolved.provenance)
        self._publish(task, "task_started", task_type=resolved.requested, mode=task.mode.value,
                      intent=intent_id, depends_on=task.depends_on)
        if resolved.fell_back:
            self._publish(task, "task_type_fallback", requested=task_type, resolved="default")
        if not task.depends_on and not self._should_plan(task):
            self._set_plan(task, [PlanItem("t1", text)])
            self._start_turn(task, ctx, task.plan[0])
        self.store.save(task, ctx)
        return task

    async def run(self, task_id: str, until_turn_boundary: bool = False) -> Task:
        """Drive a task until it is terminal or waiting. With `until_turn_boundary`, stop before a
        new turn starts, so a worker can hand the next turn to whichever worker claims it."""
        task, ctx = self._load(task_id)
        started_in = task.active_turn
        with log_context(task=task.id, sub=task.auth.subject):
            while not task.state.terminal and task.state not in WAITING_STATES:
                if not task.cancel_requested and self.store.cancel_requested(task.id):
                    task.cancel_requested = True   # asked in another process (shared workers)
                if task.cancel_requested:
                    task.cancel_requested = False
                    self._transition(task, ctx, RuntimeState.CANCELLED)
                    break
                if self._over_limits(task, ctx):
                    break
                if until_turn_boundary and task.active_turn != started_in and task.state is RuntimeState.MODEL_RUNNING:
                    break
                if task.state is RuntimeState.QUEUED:
                    await self._prefetch(task.owner, task.input)
                    self._begin(task, ctx)
                elif task.state is RuntimeState.PLANNING:
                    await self._plan(task, ctx)
                elif task.state is RuntimeState.MODEL_RUNNING:
                    await self._model_step(task, ctx)
                elif task.state is RuntimeState.TOOL_REQUESTED:
                    self._prepare_calls(task, ctx)
                elif task.state is RuntimeState.TOOL_RUNNING:
                    await self._run_tools(task, ctx)
                elif task.state is RuntimeState.VERIFYING:
                    await self._verify_turn(task, ctx)
                elif task.state is RuntimeState.COMBINING:
                    await self._combine(task, ctx)
        return task

    def approve(self, task_id: str, call_id: str, approver: AuthContext, approved: bool,
                remember: bool = False, reason: str = "") -> Task:
        """Record an approval decision. When none are pending, call run() (or schedule()) to continue."""
        task, ctx = self._load(task_id, fresh=True)
        if not self.policy.can_approve(task, approver):
            raise PermissionError("only the task's owner can decide approvals")
        call = task.call(call_id)
        if task.state is not RuntimeState.WAITING_APPROVAL or call is None \
                or call.status is not ToolCallStatus.AWAITING_APPROVAL:
            raise HarnessError("no pending approval for this call")
        if approved:
            call.status = ToolCallStatus.APPROVED
            if remember:   # for this definition only (PolicyEngine.decide)
                task.grants.add(f"{call.name}@{call.fingerprint}")
        else:
            call.status = ToolCallStatus.REJECTED
            call.result = ToolResult.error("rejected_by_user", reason or "The user rejected this call.")
        self._record_call(task, call)
        if not task.open_calls(ToolCallStatus.AWAITING_APPROVAL):
            self._transition(task, ctx, RuntimeState.TOOL_RUNNING)
        return task

    # TODO: expire approvals after the approval timeout (scheduler): reject with "approval_timeout".

    def reply(self, task_id: str, user: AuthContext, text: str) -> Task:
        """The owner's answer to ask_user. Call run() (or schedule()) to continue."""
        task, ctx = self._load(task_id, fresh=True)
        if owner_key(user) != task.owner:   # owner only; machine owners may answer their own tasks
            raise PermissionError("only the task's owner can answer")
        question = self._pending_question(task, ctx)
        if task.state is not RuntimeState.WAITING_USER or question is None:
            raise HarnessError("the task is not waiting for input")
        question.status, question.result = ToolCallStatus.SUCCEEDED, ToolResult(content=text)
        self._record_call(task, question)
        ctx.append_results(self._step_calls(task, ctx))
        self._transition(task, ctx, RuntimeState.MODEL_RUNNING)
        return task

    def cancel(self, task_id: str, by: AuthContext) -> Task:
        task, ctx = self._load(task_id, fresh=True)
        if owner_key(by) != task.owner:     # owner only; unlike approvals, machines may cancel
            raise PermissionError("only the task's owner can cancel it")
        if not task.state.terminal:
            if task.state in _BUSY:
                task.cancel_requested = True   # a worker is inside it: stop at its next loop check
                if not self.store.request_cancel(task.id):   # shared: a flag, never this copy over the worker's
                    self.store.save(task, ctx)
            else:
                self._transition(task, ctx, RuntimeState.CANCELLED)
        return task

    def force_cancel(self, task_id: str) -> Task:
        """Mark a task cancelled after its job was stopped from outside (Session.interrupt). Shared
        workers: the job may run in another process, which stops at its next step (the flag)."""
        task, ctx = self._load(task_id, fresh=True)
        if task.state in _BUSY and self.store.request_cancel(task.id):
            return task
        task.cancel_requested = False
        if not task.state.terminal:
            self._transition(task, ctx, RuntimeState.CANCELLED)
        return task

    def context(self, task_id: str) -> AgentContext:
        return self._load(task_id)[1]

    # ============================================================================================
    # Planning and turns
    # ============================================================================================
    def _should_plan(self, task: Task) -> bool:
        return (task.task_type.settings.plan or self.intake_settings.plan) == "always"

    def _begin(self, task: Task, ctx: AgentContext) -> None:
        if task.depends_on:   # results of earlier tasks of the same intent, before the request
            task.deadline = time.time() + self.limits.deadline_seconds   # its time starts now, not at intake
            inputs = []
            for dep in task.depends_on:
                loaded, result = self.store.load(dep), self.store.load_result(dep)
                if loaded and result:
                    inputs.append((loaded[0].input, result.answer or ""))
            self.services.context_manager.add_task_inputs(ctx, inputs)
        if task.active_turn is None:
            if self._should_plan(task):
                self._transition(task, ctx, RuntimeState.PLANNING)
                return
            self._set_plan(task, [PlanItem("t1", task.input)])
            self._start_turn(task, ctx, task.plan[0])
        self._transition(task, ctx, RuntimeState.MODEL_RUNNING)

    async def _plan(self, task: Task, ctx: AgentContext) -> None:
        system, _ = ctx.render()
        prompt = self._prompt("plan", task, max_items=self.intake_settings.max_plan_items)
        request = LLMRequest(model=task.model, owner=task.owner, system=system, messages=[*ctx.messages, Message("user", prompt)],
                             purpose="plan")
        response = await self._complete(task, ctx, request)
        if response is None:
            return
        ctx.usage.add(response.usage)
        items = parse_plan(response.text, self.intake_settings.max_plan_items)
        if items is None and self._escalate(task, "the plan did not parse", "plan"):
            response = await self._complete(task, ctx, request)
            if response is None:
                return
            ctx.usage.add(response.usage)
            items = parse_plan(response.text, self.intake_settings.max_plan_items)
        if items is None:
            LOG.info("task %s: no usable plan, one turn", task.id)
            items = [PlanItem("t1", task.input)]
        self._set_plan(task, items)
        await self._prefetch(task.owner, " ".join([items[0].goal, *items[0].needs]))
        self._start_turn(task, ctx, items[0])
        self._transition(task, ctx, RuntimeState.MODEL_RUNNING)

    def _set_plan(self, task: Task, items: list[PlanItem]) -> None:
        task.plan = list(items)
        task.plan_revisions.append([dataclasses.replace(i) for i in items])
        if len(items) > 1:
            self._publish(task, "task_plan", items=[dataclasses.asdict(i) for i in items])

    def _start_turn(self, task: Task, ctx: AgentContext, item: PlanItem) -> None:
        turn = task.turn(item.id)
        if turn is None:
            turn = Turn(item)
            task.turns.append(turn)
        turn.state, turn.completion = TurnState.RUNNING, None
        turn.attempts += 1
        task.active_turn = item.id
        task.active_skill, ctx.skill = None, ""   # a skill runs for one turn
        inputs = [(t.id, t.result.summary) for t in task.turns if t.id in item.depends_on and t.result]
        position = (task.plan.index(item) + 1, len(task.plan))
        self.services.context_manager.start_turn(ctx, item, inputs, position)
        if task.multi_turn:   # tools are matched to the turn's goal, not the whole request
            task.active_tools.clear()
            added = self.dispatcher.select(task, " ".join([item.goal, *item.needs]))
            added += self.dispatcher.activate(task, list(TURN_TOOL_NAMES))
        else:
            added = self.dispatcher.select(task, task.input)
        turn.tools = list(task.active_tools)
        if task.multi_turn:
            self._publish(task, "turn", turn=item.id, phase="started", goal=item.goal, n=position[0], of=position[1])
        self._publish(task, "tools", activated=added)

    async def _verify_turn(self, task: Task, ctx: AgentContext) -> None:
        turn = task.turn()
        turn.state = TurnState.VERIFYING
        completion = turn.completion or {}
        outputs = completion.get("outputs")
        calls = [c for c in task.calls if c.step in turn.step_numbers]
        artifacts = sorted({e.target for c in calls for e in self.store.effects_for_call(task.id, c.id)
                            if e.status is EffectStatus.APPLIED})
        usage = Usage()
        for step in turn.steps:
            usage.add(step.usage)
        turn.result = TurnResult(str(completion.get("summary") or "").strip(),
                                 outputs if isinstance(outputs, dict) else {}, artifacts, usage)

        turn.verification, objection = [], None
        for check in turn.item.verify:
            problem = await self._check(task, ctx, turn, check)
            turn.verification.append({"check": check, "passed": problem is None, "note": problem or ""})
            objection = objection or problem
        if objection:
            if turn.attempts < self.limits.turn_attempts:   # reopen: the model sees why and keeps working
                self._escalate(task, f"turn {turn.id} failed its checks")
                turn.attempts += 1
                turn.state, turn.completion = TurnState.RUNNING, None
                ctx.append_user(f"[runtime] This turn is not done: {objection}")
                self._publish(task, "turn_rejected", turn=turn.id, reason=objection)
                self._transition(task, ctx, RuntimeState.MODEL_RUNNING)
                return
            turn.state = TurnState.FAILED
            self._publish_turn_done(task, turn)
            # TODO: ask the model to replan the remaining items (docs/agent-flow.md §7).
            self._fail(task, ctx, "turn_failed", f"Turn {turn.id} did not pass its checks: {objection}")
            return

        turn.state = TurnState.SUCCEEDED
        self._publish_turn_done(task, turn)
        following = task.next_item()
        if following is not None:
            await self._prefetch(task.owner, " ".join([following.goal, *following.needs]))
            self._start_turn(task, ctx, following)
            self._transition(task, ctx, RuntimeState.MODEL_RUNNING)
        elif any(task.turn(i.id) is None for i in task.plan):
            self._fail(task, ctx, "plan_blocked", "Some plan items can never start: their dependencies did not succeed.")
        elif task.multi_turn:
            self._transition(task, ctx, RuntimeState.COMBINING)
        else:
            await self._finish(task, ctx, turn.result.summary)

    async def _check(self, task: Task, ctx: AgentContext, turn: Turn, check: dict[str, Any]) -> str | None:
        """One verify check of a plan item: None when it passes, else the objection."""
        kind = check.get("kind")
        if kind == "tool_succeeded":
            tool = check.get("tool")
            last_ok = max((c.step for c in task.calls if c.name == tool and c.status is ToolCallStatus.SUCCEEDED
                           and c.step in turn.step_numbers), default=None)
            changed = [c.step for c in task.calls
                       if any(e.status is EffectStatus.APPLIED for e in self.store.effects_for_call(task.id, c.id))]
            if last_ok is None or any(step > last_ok for step in changed):
                return f"{tool} has not succeeded since the last change"
        elif kind == "tests_pass":
            if not ctx.working_memory.verified():
                return "no passing test run since the last change"
        elif kind == "produces":
            if not turn.result.summary:
                return f"the turn produced no result ({check.get('what', 'a summary')})"
        elif kind == "judge" and self.services.judge is not None:
            return await self.services.judge.review(task, turn, ctx)
        return None   # unknown kinds, and judge checks without a judge, are skipped

    async def _combine(self, task: Task, ctx: AgentContext) -> None:
        system, _ = ctx.render()
        results = "\n\n".join(f"## {t.id}: {t.item.goal}\n{t.result.summary if t.result else ''}"
                              for t in task.turns if t.state is TurnState.SUCCEEDED)
        request = LLMRequest(model=task.model, owner=task.owner, system=system, purpose="reply",
                             messages=[*ctx.messages, Message("user", self._prompt("combine", task) + "\n" + results)])
        response = await self._complete(task, ctx, request)
        if response is not None:
            ctx.usage.add(response.usage)
            await self._finish(task, ctx, response.text)

    async def _finish(self, task: Task, ctx: AgentContext, answer: str) -> None:
        """Task-level on_complete hooks on the answer, then completed (or the last turn reopens)."""
        for hook in self.on_complete:
            objection = hook(task, ctx)
            if objection:
                turn = task.turn()
                turn.state, turn.completion = TurnState.RUNNING, None
                ctx.append_user(f"[runtime] The task is not complete: {objection}")
                self._publish(task, "completion_rejected", reason=objection)
                self._transition(task, ctx, RuntimeState.MODEL_RUNNING)
                return
        task.result = answer
        self._transition(task, ctx, RuntimeState.COMPLETED)

    # ============================================================================================
    # Long-term memory
    # ============================================================================================
    async def consolidate(self, task_id: str) -> None:
        """The consolidate job: what lasts of a finished task goes to its owner's long-term memory. With
        `[memory] extract = llm` the model picks it; without, or when the model is unavailable or its
        answer does not parse, the working memory's notes are kept as they are."""
        loaded = self.store.load(task_id)
        if loaded is None:
            return
        task, ctx = loaded
        if getattr(task, "consolidation", None) != "pending":
            return
        manager = self.services.context_manager
        texts = await self._extract_memories(task, ctx) if manager.settings.extract == "llm" else None
        kept = await asyncio.to_thread(manager.remember, task, list(ctx.working_memory.notes) if texts is None else texts)
        task.consolidation = "done"
        self.store.save(task, ctx)
        LOG.info("task %s: %d memory item(s) kept", task.id, len(kept))

    async def _extract_memories(self, task: Task, ctx: AgentContext) -> list[str] | None:
        manager = self.services.context_manager
        request = LLMRequest(model=manager.settings.model or task.model, owner=task.owner, purpose="consolidate",
                             pinned=bool(manager.settings.model), task_id=task.id, intent_id=task.intent_id,
                             system=self._prompt("consolidate", task, max_items=manager.settings.max_per_task),
                             messages=[Message("user", manager.consolidation_input(task, ctx))])
        try:
            response = await self.services.llm.complete(request)
        except LLMError as exc:
            LOG.warning("memory extraction unavailable for task %s, keeping its notes: %s", task.id, exc)
            return None
        texts = manager.parse_memories(response.text)
        if texts is None:
            LOG.info("memory extraction for task %s did not parse, keeping its notes", task.id)
        return texts

    def resume_consolidations(self) -> int:
        """Queue the consolidations a stopped process left pending. Returns how many."""
        pending = self.store.pending_consolidations()
        for task_id in pending:
            loaded = self.store.load(task_id)
            self._enqueue("consolidate", task_id, loaded[0].owner if loaded else None)
        return len(pending)

    def _publish_turn_done(self, task: Task, turn: Turn) -> None:
        if task.multi_turn:
            self._publish(task, "turn", turn=turn.id, phase="completed", status=turn.state.value,
                          summary=turn.result.summary[:500] if turn.result else "",
                          artifacts=turn.result.artifacts if turn.result else [], verification=turn.verification)

    # ============================================================================================
    # Steps
    # ============================================================================================
    async def _model_step(self, task: Task, ctx: AgentContext) -> None:
        if ctx.steps_left <= 0:
            await self._final_report(task, ctx)
            return
        budget = self._task_budget(task)
        if budget is not None and task.cost >= budget:
            await self._final_report(task, ctx, "budget_exceeded", f"Stopped at its budget of {budget:g}.")
            return
        await self._maybe_compact(task, ctx)
        if task.state.terminal:   # the compaction call failed, and with it the task
            return
        step = ctx.begin_step()
        ctx.append_status()
        tools = self.dispatcher.active(task)
        system, messages = ctx.render()
        request = LLMRequest(model=task.model, owner=task.owner, system=system, messages=messages,
                             tools=self.dispatcher.registry.schemas(tools), purpose="cron_check" if task.cron else "step",
                             on_text=lambda text: self._publish(task, "message_delta", text=text))
        self._publish(task, "step", step=step, turn=task.active_turn, phase="started")
        started, clock = time.time(), time.monotonic()
        response = await self._complete(task, ctx, request)
        if response is None:
            return
        _unique_call_ids(task, response)   # before the step is recorded: its tool results must match
        ctx.append_step(response)
        turn = task.turn()
        turn.steps.append(StepRecord(step, response.text[:2000], [c.id for c in response.tool_calls], response.usage,
                                     started_at=started, seconds=round(time.monotonic() - clock, 3),
                                     model_id=response.model_id, route=response.route, cost=response.cost or 0.0,
                                     trace_id=trace_id()))
        self._publish(task, "step", step=step, turn=task.active_turn, phase="completed",
                      usage={"input": response.usage.input_tokens, "output": response.usage.output_tokens},
                      model=response.model_id, route=response.route, cost=response.cost)

        if response.is_final:   # an answer without tool calls ends the turn; its text is the summary
            turn.completion = {"summary": response.text}
            self._transition(task, ctx, RuntimeState.VERIFYING)
            return
        for request_call in response.tool_calls:
            task.calls.append(ToolCall(id=request_call.id, name=request_call.name,
                                       arguments=request_call.arguments, step=step))
        self._transition(task, ctx, RuntimeState.TOOL_REQUESTED)

    def _prepare_calls(self, task: Task, ctx: AgentContext) -> None:
        for call in self._step_calls(task, ctx):
            if call.status is not ToolCallStatus.REQUESTED:
                continue
            outcome = self.dispatcher.prepare(task, call)
            wrong = outcome is Outcome.ERROR and call.result is not None and call.result.error_code in (
                "invalid_arguments", "unknown_tool")
            task.invalid_streak = task.invalid_streak + 1 if wrong else (0 if outcome is not Outcome.ERROR else task.invalid_streak)
            if task.invalid_streak >= 2:   # the model keeps getting its tool calls wrong: one tier up
                task.invalid_streak = 0
                self._escalate(task, "invalid tool calls twice in a row")
            if outcome is Outcome.CONTROL:
                self._control(task, ctx, call)
            elif outcome is Outcome.NEEDS_APPROVAL and getattr(task, "unattended", None):
                self._unattended(task, call)
            elif outcome is Outcome.NEEDS_APPROVAL:
                self._publish(task, "approval_required", call=call.id, name=call.name,
                              arguments=call.arguments, reason=call.reason)
        if task.open_calls(ToolCallStatus.AWAITING_APPROVAL):
            self._transition(task, ctx, RuntimeState.WAITING_APPROVAL)
        else:
            self._transition(task, ctx, RuntimeState.TOOL_RUNNING)

    def _unattended(self, task: Task, call: ToolCall) -> None:
        """Nobody can approve now (a cron): the call is not run. With `propose` it is kept for the
        people who work the cron's issues to approve later (agenticwf.agent.runtime.cron); with `deny`, refused."""
        call.status = ToolCallStatus.DENIED
        if task.unattended == "propose":
            findings = (task.cron or {}).get("findings") or []
            task.proposals = [*getattr(task, "proposals", []), {
                "call": call.id, "tool": call.name, "arguments": call.arguments, "reason": call.reason,
                "issue_key": findings[-1]["key"] if findings else None}]
            call.result = ToolResult.error("proposed", "Not run: it needs an approval and nobody can give one now, "
                                           "so it was proposed to the people who work this check's issues. Do not "
                                           "try it again; carry on with the check.")
        else:
            call.result = ToolResult.error("unattended_denied", "Not run: it needs an approval, and nobody can "
                                           "give one in this unattended task.")
        self._record_call(task, call)

    async def _run_tools(self, task: Task, ctx: AgentContext) -> None:
        calls = self._step_calls(task, ctx)
        for call in calls:   # a worker stopped while it ran (the snapshot may still say approved): effects decide
            if call.status is ToolCallStatus.RUNNING or (call.status is ToolCallStatus.APPROVED
                                                         and self.store.effects_for_call(task.id, call.id)):
                self._resume_call(task, call)
        approved = [c for c in calls if c.status is ToolCallStatus.APPROVED]
        allowance = max(0, self.limits.max_tool_calls - task.tool_call_count)
        for call in approved[allowance:]:   # one step cannot run past the limit; the loop then fails the task
            call.status = ToolCallStatus.FAILED
            call.result = ToolResult.error("max_tool_calls", f"Not run: the task's limit of "
                                           f"{self.limits.max_tool_calls} tool calls is reached.")
            self._record_call(task, call)
        await self.dispatcher.execute(task, approved[:allowance])
        for call in approved[:allowance]:
            started = call.result.facts.get("skill") if call.result and not call.result.is_error else None
            if started:
                self._enter_skill(task, ctx, started)
        question = self._pending_question(task, ctx)
        if question is not None:
            self._publish(task, "input_required", call=question.id, question=question.arguments.get("question", ""))
            self._transition(task, ctx, RuntimeState.WAITING_USER)
            return
        ctx.append_results(calls)
        turn = task.turn()
        self._transition(task, ctx, RuntimeState.VERIFYING if turn.completion is not None else RuntimeState.MODEL_RUNNING)

    def _enter_skill(self, task: Task, ctx: AgentContext, skill: dict[str, Any]) -> None:
        """A skill's instructions were loaded: it runs until the turn ends. No other skill is offered
        from now on, and the tools narrow to the skill's (or are matched to it)."""
        task.active_skill = {"id": skill["id"], "tool": skill["tool"], "tools": list(skill.get("tools") or ()),
                             "files": bool(skill.get("files"))}
        added = self.dispatcher.enter_skill(task, skill.get("query") or "")
        self._publish(task, "skill", phase="started", skill=skill["id"], tools=list(task.active_tools))
        if added:
            self._publish(task, "tools", activated=added)

    def _resume_call(self, task: Task, call: ToolCall) -> None:
        """docs/agent-flow.md §9: a call that finished before the stop keeps its result (kept with its
        effects; the task record may predate it). Otherwise re-run when nothing may have happened or
        the tool is idempotent."""
        effects = self.store.effects_for_call(task.id, call.id)
        unfinished = [e for e in effects if e.status is EffectStatus.INTENDED]
        kept = next((e.detail.get("result") for e in effects if isinstance(e.detail.get("result"), dict)), None)
        if effects and not unfinished and kept is not None:   # it ran to the end: never run it again
            failed = any(e.status is EffectStatus.FAILED for e in effects) or kept.get("is_error")
            call.status = ToolCallStatus.FAILED if failed else ToolCallStatus.SUCCEEDED
            call.result = ToolResult(content=kept.get("content"), is_error=bool(kept.get("is_error")),
                                     error_code=kept.get("error_code"), summary=str(kept.get("summary") or ""))
            self._record_call(task, call)
            return
        spec = self.dispatcher.eligible(task).get(call.name)
        if unfinished and not (spec is not None and spec.idempotent):
            call.status = ToolCallStatus.INTERRUPTED
            call.result = ToolResult.error("interrupted", "The runtime stopped while this call ran; it may or may "
                                           "not have taken effect. Check before trying again.")
            for effect in unfinished:
                effect.status = EffectStatus.FAILED
                effect.detail = {"interrupted": True}
                self.store.save_effect(effect)
            self._record_call(task, call)
        else:
            call.status = ToolCallStatus.APPROVED   # same call id, so the same idempotency key

    def _control(self, task: Task, ctx: AgentContext, call: ToolCall) -> None:
        """Control tools act on the task itself, so the harness handles them."""
        args, wm = call.arguments, ctx.working_memory
        if call.name == "update_plan":
            wm.set_plan(args.get("phase"), args.get("steps"))
            self._publish(task, "plan", phase=wm.phase, steps=wm.plan)
            result = ToolResult(content="Plan recorded.")
        elif call.name == "update_working_memory":
            wm.update(args.get("goal"), args.get("task_state"), args.get("add_note"))
            result = ToolResult(content="Working memory updated.")
        elif call.name == "search_tools":
            found = self.dispatcher.discover(task, args["query"])
            self._publish(task, "tools", activated=[f["name"] for f in found])
            result = ToolResult(content=found or "No matching tools.")
        elif call.name in ("report_issue", "report_check"):
            result = self._cron_report(task, call)
        elif call.name == "read_skill_file":
            skill, skills = task.active_skill, self.dispatcher.registry.skills
            try:
                if not skill or skills is None:
                    raise SkillError("no skill is running")
                result = ToolResult(content=skills.read_file(skill["id"], args["path"], task.owner),
                                    summary=f"read {args['path']}")
            except SkillError as exc:
                result = ToolResult.error("skill_file", str(exc))
        elif call.name == "complete_turn":
            task.turn().completion = {"summary": args.get("summary", ""), "outputs": args.get("outputs") or {}}
            result = ToolResult(content="Turn result recorded; the runtime checks it after this step.")
        elif call.name == "ask_user":
            return   # stays REQUESTED: answered by reply() once the other calls have run
        else:
            result = ToolResult.error("unknown_tool", f"Unknown tool: {call.name}")
        call.status, call.result = ToolCallStatus.SUCCEEDED, result
        self._record_call(task, call)

    @staticmethod
    def _cron_report(task: Task, call: ToolCall) -> ToolResult:
        """report_issue / report_check in a cron's check task: checked against the check's
        definition and kept on the task; agenticwf.agent.runtime.cron files them when the run ends."""
        cron, args = task.cron, call.arguments
        if cron is None:
            return ToolResult.error("not_a_cron_task", "Only a cron's checks report issues.")
        if call.name == "report_issue":
            key = {k: str(v) for k, v in (args.get("key") or {}).items()}
            for field in cron.get("key", ()):   # a key field named like an argument (title) comes from it
                if not key.get(field, "").strip() and isinstance(args.get(field), str):
                    key[field] = args[field]
            missing =[f for f in cron.get("key", ()) if f != cron.get("group_by") and not key.get(f, "").strip()]
            if missing:
                return ToolResult.error("invalid_key", f"key needs: {', '.join(cron.get('key', ()))} (missing: {', '.join(missing)})")
            if args.get("severity") not in cron.get("severities", ()):
                return ToolResult.error("invalid_severity", f"severity must be one of: {', '.join(cron.get('severities', ()))}")
            cron.setdefault("findings", []).append({
                "call": call.id, "key": key, "title": str(args.get("title") or "").strip()[:500],
                "severity": args["severity"], "details": args.get("details"), "links": list(args.get("links") or []),
                "evidence": args.get("evidence"), "related": list(args.get("related") or []),
                "draft_ticket": args.get("draft_ticket")})
            return ToolResult(content=f"Recorded: {args.get('title')}", summary=f"issue: {args.get('title')}")
        declared = cron.get("metrics") or {}
        recorded = []
        for metric in args.get("metrics") or ():
            spec = declared.get(metric.get("name"))
            if spec is None:
                return ToolResult.error("undeclared_metric", f"{metric.get('name')!r} is not one of this check's metrics: "
                                        f"{', '.join(declared) or 'none'}")
            labels = {k: str(v) for k, v in (metric.get("labels") or {}).items()}
            extra = sorted(set(labels) - set(spec.get("labels") or ()))
            if extra:
                return ToolResult.error("undeclared_label", f"{metric['name']}: labels must be among "
                                        f"{', '.join(spec.get('labels') or ()) or 'none'} (not {', '.join(extra)})")
            recorded.append({"name": metric["name"], "value": float(metric["value"]), "labels": labels,
                             "unit": spec.get("unit") or None, "link": metric.get("link")})
        cron.setdefault("metrics_recorded", []).extend(recorded)
        cron["status"], cron["summary"] = args["status"], str(args.get("summary") or "")[:2000]
        return ToolResult(content=f"Check recorded as {args['status']}.", summary=f"check: {args['status']}")

    async def run_approved_call(self, auth: AuthContext, name: str, arguments: dict[str, Any]) -> ToolCall:
        """Run one call a person approved outside a task (a cron's proposal, from the issue board):
        under their identity, through the normal policy; their approval answers an `ask`."""
        resolved = self.services.task_types.resolve(DEFAULT, owner_key(auth))
        task = Task(auth=auth, input=f"Approved proposal: {name}", task_type=resolved, mode=Mode.DEFAULT,
                    source="approval")
        call = ToolCall(id=f"approved-{uuid.uuid4().hex[:12]}", name=name, arguments=dict(arguments), step=0)
        outcome = self.dispatcher.prepare(task, call)
        if outcome is Outcome.NEEDS_APPROVAL:
            call.status = ToolCallStatus.APPROVED
        elif outcome is Outcome.CONTROL:
            call.status, call.result = ToolCallStatus.FAILED, ToolResult.error("not_allowed", "Control tools cannot be proposed.")
        if call.status is ToolCallStatus.APPROVED:
            await self.dispatcher.execute(task, [call])
        return call

    # ============================================================================================
    # Limits, model calls, compaction
    # ============================================================================================
    def _over_limits(self, task: Task, ctx: AgentContext) -> bool:
        if task.deadline and time.time() > task.deadline:
            self._fail(task, ctx, "deadline_exceeded", "The task ran past its deadline.")
        elif task.tool_call_count >= self.limits.max_tool_calls:
            self._fail(task, ctx, "max_tool_calls", f"More than {self.limits.max_tool_calls} tool calls.")
        else:
            return False
        return True

    async def _final_report(self, task: Task, ctx: AgentContext, code: str = "max_steps", message: str = "") -> None:
        """At max_steps (or the task's budget): one last turn without tools for a progress report, then
        fail. Out of steps: the report is written one tier up (the model so far did not get there)."""
        if code == "max_steps":
            self._escalate(task, f"reached max_steps ({ctx.max_steps})")
        system, messages = ctx.render()
        messages.append(Message("user", self._prompt("final_report", task)))
        response = await self._complete(task, ctx, LLMRequest(model=task.model, owner=task.owner, system=system,
                                                              messages=messages, purpose="final_report"))
        if response is not None:
            ctx.append_step(response)
            task.result = response.text
            self._fail(task, ctx, code, message or f"Stopped after {ctx.max_steps} steps.")

    async def _complete(self, task: Task, ctx: AgentContext, request: LLMRequest) -> LLMResponse | None:
        """A task's model call: with what routing needs (its pin, tier, current model and cache),
        then what it cost and which model served it (docs/model-routing.md)."""
        own = request.purpose in _TASK_PURPOSES   # the task's own work: its pin and tier; helpers are routed alone
        request = dataclasses.replace(
            request, pinned=request.pinned or (task.pinned and own),
            min_tier=(task.min_tier if own else None) or request.min_tier,
            current=task.model_id if own else None, cached_hint=task.cached_hint if own else 0,
            task_id=task.id, intent_id=task.intent_id)
        try:
            response = await self.services.llm.complete(request)
        except LLMError as exc:
            LOG.error("model unavailable: %s", exc)
            self._fail(task, ctx, "model_unavailable", str(exc))
            return None
        if response.model_id:
            if task.model_id and response.model_id != task.model_id and request.purpose in ("step", "cron_check"):
                self._publish(task, "model_changed", model=response.model_id, previous=task.model_id, route=response.route)
            if request.purpose in ("step", "cron_check", "plan", "final_report"):
                task.model_id, task.cached_hint = response.model_id, response.usage.input_tokens
        task.cost += response.cost or 0.0
        return response

    def _router(self):
        """The model router, when the harness talks to one (not a scripted model in tests)."""
        llm = self.services.llm
        return llm if hasattr(llm, "choose") else None

    def _task_budget(self, task: Task) -> float | None:
        router = self._router()
        own = task.task_type.settings.budget
        return own if own is not None else (router.settings.task_budget if router else None)

    def _escalate(self, task: Task, evidence: str, purpose: str = "step") -> bool:
        """The model so far did not cope (docs/model-routing.md §6): the task needs one tier more.
        Never for a pinned task, past `max_escalations`, above the owner's highest tier, or near
        the owner's budget. The model it escalates from gets the evidence as feedback."""
        router = self._router()
        if router is None or task.pinned or task.escalations >= router.settings.max_escalations \
                or not router.can_escalate(task.owner):
            return False
        current = task.min_tier or router.tier_for(purpose)
        wanted = raise_tier(current)
        if wanted == current or tier_rank(wanted) > tier_rank(router.settings.max_tier_for(task.owner)):
            return False
        if task.model_id:
            router.feedback("escalation", purpose, task.model_id, task.owner, task.id, task.intent_id, evidence)
        task.min_tier, task.escalations = wanted, task.escalations + 1
        LOG.info("task %s: escalated to tier %s (%s)", task.id, wanted, evidence)
        self._publish(task, "model_escalated", tier=wanted, evidence=evidence, model=task.model_id)
        return True

    async def _maybe_compact(self, task: Task, ctx: AgentContext) -> None:
        threshold = self.limits.compact_at_tokens
        router = self._router()
        window = router.window(task.model_id or task.model) if router else None
        if window:   # compact well before the model's own limit (docs/model-routing.md §5)
            threshold = min(threshold, int(window * (1 - router.settings.headroom)))
        if ctx.size() < threshold:
            return
        if self.services.compactor is not None:
            await self.services.compactor.compact(task, ctx)
            return
        span = ctx.compactable_range(self.limits.keep_recent_messages)
        if span is None:
            return
        transcript = "\n".join(f"{m.role}: {m.content[:2000]}" for m in ctx.compactable_messages(*span))
        request = LLMRequest(model=task.model, owner=task.owner, system=self._prompt("compact", task),
                             messages=[Message("user", transcript)], purpose="compact")
        response = await self._complete(task, ctx, request)
        if response is not None:
            ctx.replace_with_summary(span[0], span[1], response.text)

    # ============================================================================================
    # Helpers
    # ============================================================================================
    def _prompt(self, name: str, task: Task | None = None, **values: Any) -> str:
        """A runtime prompt from its Markdown file (prompts/<name>.md; a task type may override the
        per-task ones), with `${name}` placeholders filled in."""
        resolver = self.services.task_types
        text = resolver.prompt(name, task.task_type.effective if task else None, task.owner if task else None).content
        return string.Template(text).safe_substitute(values).strip() if values else text.strip()

    def _load(self, task_id: str, fresh: bool = False) -> tuple[Task, AgentContext]:
        """`fresh`: from the tables, not this process's cache (shared workers: another process may
        have moved the task on; a change made here starts from its latest state)."""
        if fresh:
            self.store.forget(task_ids=[task_id])
        loaded = self.store.load(task_id)
        if loaded is None:
            raise HarnessError(f"unknown task {task_id}")
        return loaded

    @staticmethod
    def _step_calls(task: Task, ctx: AgentContext) -> list[ToolCall]:
        return [c for c in task.calls if c.step == ctx.step_count]

    def _pending_question(self, task: Task, ctx: AgentContext) -> ToolCall | None:
        return next((c for c in self._step_calls(task, ctx)
                     if c.name == "ask_user" and c.result is None and c.status is ToolCallStatus.REQUESTED), None)

    def _transition(self, task: Task, ctx: AgentContext, new: RuntimeState) -> None:
        old = task.state
        check_transition(old, new)   # legal first, then persist, then act
        now = time.time()
        if new in WAITING_STATES and old not in WAITING_STATES:
            task.waiting_since = now
        elif old in WAITING_STATES and new not in WAITING_STATES:
            waited = now - (getattr(task, "waiting_since", None) or now)
            if task.deadline:   # time spent waiting for a person is not the task's to lose
                task.deadline += waited
            task.waiting_since = None
        task.state = ctx.state = new
        turn = task.turn()
        if turn is not None and turn.state in (TurnState.RUNNING, TurnState.WAITING):
            turn.state = TurnState.WAITING if new in WAITING_STATES else TurnState.RUNNING
        # A cron's checks run every few minutes and keep what they learn in its issue tracker: they
        # recall long-term memory but do not add to it (one model call per check and run otherwise).
        if self.services.context_manager.should_consolidate(new) and getattr(task, "source", "") != "cron":
            # Saved with the terminal state, so a restart finds it (resume_consolidations); queued
            # first, so it comes ahead of the reply job _on_terminal may queue.
            task.consolidation = "pending"
            self._enqueue("consolidate", task.id, task.owner)
        self.store.save(task, ctx)
        self._publish(task, "task", status=new.public_status, state=new.value, error=task.error)
        if new in WAITING_STATES:
            self._publish(task, "task_waiting", state=new.value)
        if new.terminal:
            self._on_terminal(task, ctx)

    def _fail(self, task: Task, ctx: AgentContext, code: str, message: str) -> None:
        task.error = {"code": code, "message": message}
        self._transition(task, ctx, RuntimeState.FAILED)

    def _on_terminal(self, task: Task, ctx: AgentContext) -> None:
        """The task's result entry (written once), then whatever the intent can do next."""
        turn = task.turn()
        if turn is not None and turn.state in (TurnState.RUNNING, TurnState.WAITING, TurnState.VERIFYING):
            turn.state = TurnState.FAILED
        effects = self.store.effects_for_task(task.id)
        result = TaskResult(
            task_id=task.id, intent_id=task.intent_id, status=task.state.public_status, answer=task.result,
            error=task.error,
            turns=[{"turn": t.id, "goal": t.item.goal, "state": t.state.value,
                    "summary": t.result.summary if t.result else ""} for t in task.turns],
            artifacts=sorted({e.target for e in effects if e.status is EffectStatus.APPLIED}),
            verification=[v for t in task.turns for v in t.verification], usage=ctx.usage)
        if self.store.load_result(task.id) is None:
            self.store.save_result(result)
        self.store.save(task, ctx)
        usage = ctx.usage
        self._publish(task, "task_complete", status=task.state.public_status, result=task.result, error=task.error,
                      usage={"input": usage.input_tokens, "output": usage.output_tokens})
        task_type = task.task_type.effective
        record_task_ended(task_type=task_type if task_type.startswith(f"{SYSTEM_SCOPE}:") else "user",
                          state=task.state.value, cron=bool(task.cron))
        if task.intent_id:
            self._advance_intent(task.intent_id)

    def _advance_intent(self, intent_id: str) -> None:
        """Start the tasks whose dependencies are done; cancel those whose dependencies failed; reply when all are done."""
        tasks = self.store.tasks_for_intent(intent_id)
        by_id = {t.id: t for t in tasks}
        for task in tasks:
            if task.state is not RuntimeState.QUEUED or not task.depends_on:
                continue
            deps = [by_id[d] for d in task.depends_on if d in by_id]
            if not all(d.state.terminal for d in deps):
                continue
            if all(d.state is RuntimeState.COMPLETED for d in deps):
                self.schedule(task.id)
            else:
                _, ctx = self._load(task.id)
                task.error = {"code": "dependency_failed", "message": "A task this one depends on did not succeed."}
                self._transition(task, ctx, RuntimeState.CANCELLED)
        if tasks and all(t.state.terminal for t in self.store.tasks_for_intent(intent_id)):
            self._enqueue("reply", intent_id, tasks[0].owner)

    def _record_call(self, task: Task, call: ToolCall) -> None:
        result = call.result
        self._publish(task, "tool", call=call.id, name=call.name, status=call.status.value,
                      decision=call.decision, summary=(result.summary if result else ""))

    def _publish(self, task: Task, kind: str, /, **data: Any) -> None:
        self._send(task.session_id, task.auth.subject, {"type": kind, "task": task.id, "intent": task.intent_id, **data})

    def _publish_thread(self, intent: Intent, kind: str, /, **data: Any) -> None:
        self._send(intent.thread_id, intent.auth.subject, {"type": kind, "intent": intent.id, **data})

    def _send(self, session_id: str | None, subject: str, event: dict[str, Any]) -> None:
        """To the session's sink in this process; in a worker process, through the relay to the API's."""
        sink = self._sinks.get(session_id) if session_id else None
        if sink is not None:
            sink(subject, event)
        elif self.remote is not None and session_id:
            self.remote(session_id, subject, event)
        else:
            self.events(subject, event)

    def deliver(self, session_id: str, subject: str, event: dict[str, Any]) -> None:
        """An event relayed from a worker process (the API side of agenticwf.agent.runtime.harness.relay)."""
        self._sinks.get(session_id, self.events)(subject, event)


def _tokens(summary: str, messages: list[Message]) -> int:
    """Rough size in tokens (4 characters per token)."""
    return (len(summary) + sum(len(m.content or "") for m in messages)) // 4


def _keys(message: Message) -> tuple[str, ...]:
    """What a transcript entry belongs to (agenticwf.agent.runtime.session.state.TranscriptMessage)."""
    return getattr(message, "keys", ())


def _conversation(intent: Intent) -> list[Message]:
    """The transcript an intent's tasks see: the summary of the earlier conversation, then the rest."""
    if not intent.conversation_summary:
        return list(intent.conversation)
    summary = Message("user", f"<summary of the earlier conversation>\n{intent.conversation_summary}\n"
                              "</summary of the earlier conversation>")
    return [summary, *intent.conversation]


def _error_text(result: TaskResult) -> str:
    if result.error:
        return f"[{result.status}] {result.error.get('message', '')}".strip()
    return f"[{result.status}]"


_CALL_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def _unique_call_ids(task: Task, response: LLMResponse) -> None:
    """Call ids name one call of the task: approvals, answers, effects and resume find it by id.
    Providers that send none (or reuse them across steps) get fresh ones, on the response itself,
    so the recorded step and its tool results keep matching."""
    used = {c.id for c in task.calls}
    for request_call in response.tool_calls:
        if not request_call.id or request_call.id in used or not _CALL_ID.match(request_call.id):
            request_call.id = f"call_{uuid.uuid4().hex[:20]}"
        used.add(request_call.id)

