"""
ToolDispatcher: selects the tools a task sees, and turns tool calls into results.

    select    eligible = PolicyEngine.visible(ToolRegistry.catalogue(owner)), without what the owner
              switched off (agenticwf.agent.runtime.tools.prefs)
              active   = pinned + the skills and tools closest to the request (top K of each, BM25),
                         within a schema token budget; all of them when the catalogue is small
    skills    a skill is a tool (`skill__<name>`) whose call loads its instructions. While one
              runs (task.active_skill) no other skill is offered, and the tools narrow to the
              skill's `tools` when it names any
    prepare   resolve → validate → decide (allow / ask / deny)      no side effects
    execute   pre_tool hooks → effect intended → handler (timeout, ambient AuthContext)
              → normalize → post_tool hooks → effect applied | failed; a span per call
              (agenticwf.agent.runtime.telemetry)

It never moves the task's lifecycle (it reports `needs_approval` and the harness waits), never
talks to the model, and holds no permission rules of its own. Control tools are recognised and
handed back to the harness.
"""

from __future__ import annotations

import asyncio
import fnmatch
import inspect
import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.policy.engine import Decision, DecisionKind, PolicyEngine
from agenticwf.agent.runtime.records import Effect, EffectStatus
from agenticwf.agent.runtime.redact import redact
from agenticwf.agent.runtime.task import Task, ToolCall, ToolCallStatus
from agenticwf.agent.runtime.telemetry import fail, record_tool_call, span, trace_id
from agenticwf.agent.runtime.tools.control import CONTROL_TOOL_NAMES, CRON_TOOL_NAMES, SKILL_TOOL_NAMES, TURN_TOOL_NAMES
from agenticwf.agent.runtime.tools.prefs import InMemoryToolPrefs, ToolPrefs, switched_off
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import CallContext, Effects, HandlerKind, ToolResult, ToolSpec
from agenticwf.agent.runtime.types import Owner, use_auth_context

LOG = get_logger(__name__)


class Outcome(str, Enum):
    ALLOWED = "allowed"
    NEEDS_APPROVAL = "needs_approval"
    ERROR = "error"        # call.result holds an error result for the model
    CONTROL = "control"    # the harness handles it


@dataclass
class Hooks:
    """Deterministic code at fixed points (docs/agent-runtime.md §9)."""

    # pre_tool(task, call, spec) -> ToolResult to block the call, or None to continue
    pre_tool: list[Callable[[Task, ToolCall, ToolSpec], ToolResult | None]] = field(default_factory=list)
    # post_tool(task, call, result) -> result (possibly redacted or annotated)
    post_tool: list[Callable[[Task, ToolCall, ToolResult], ToolResult]] = field(default_factory=list)


class ToolDispatcher:
    def __init__(self, registry: ToolRegistry, policy: PolicyEngine, hooks: Hooks | None = None,
                 top_k: int = 15, schema_budget_tokens: int = 8000, small_catalogue: int = 30, top_k_skills: int = 5,
                 always: tuple[str, ...] = (), prefs: ToolPrefs | None = None,
                 hidden: Callable[[Owner], frozenset[str]] | None = None,
                 on_record: Callable[[Task, ToolCall], None] | None = None,
                 on_effect: Callable[[Effect], None] | None = None):
        self.registry = registry
        self.policy = policy
        self.prefs = prefs or InMemoryToolPrefs()   # what each owner switched off for their tasks
        # tool names an owner's other settings leave out (the knowledge search tools they chose)
        self.hidden = hidden or (lambda owner: frozenset())
        self.hooks = hooks or Hooks()
        self.top_k = top_k
        self.top_k_skills = top_k_skills
        self.schema_budget_tokens = schema_budget_tokens
        self.small_catalogue = small_catalogue
        self.always = always   # offered in every task, like a task type's pin_tools (e.g. web_search)
        self.on_record = on_record or (lambda task, call: None)
        self.on_effect = on_effect or (lambda effect: None)   # the harness persists effects

    # --- selection -----------------------------------------------------------------------------
    def eligible(self, task: Task) -> dict[str, ToolSpec]:
        """The security boundary: nothing outside this set can ever be called."""
        off, hidden = self.prefs.disabled_for(task.owner), self.hidden(task.owner)
        visible = [s for s in self.policy.visible(self.registry.catalogue(task.owner), task)
                   if not switched_off(s, off) and s.name not in hidden]
        skill = getattr(task, "active_skill", None)

        globs, skill_names = getattr(task, "tool_globs", None), getattr(task, "skill_names", None)

        def offered(spec: ToolSpec) -> bool:
            if spec.name in TURN_TOOL_NAMES:
                return task.multi_turn
            if spec.name in SKILL_TOOL_NAMES:
                return bool(skill and skill.get("files"))
            if spec.name in CRON_TOOL_NAMES:
                return getattr(task, "cron", None) is not None
            if spec.handler.kind is HandlerKind.CONTROL:
                return True
            if spec.handler.kind is HandlerKind.SKILL:
                if skill is not None:   # one skill at a time
                    return False
                if skill_names is not None:   # a task that names its skills (a cron check)
                    qualified = str(spec.handler.settings.get("skill", "")).rpartition(":")[2]
                    return qualified in skill_names or spec.name in skill_names
                return True
            if globs is not None and not any(fnmatch.fnmatchcase(spec.name, p) for p in globs):
                return False
            return skill is None or not skill["tools"] or any(fnmatch.fnmatchcase(spec.name, p) for p in skill["tools"])

        return {s.name: s for s in visible if offered(s)}

    def select(self, task: Task, requirements: str) -> list[str]:
        """Choose the active set: pinned tools, then the skills and the tools closest to
        `requirements` (the request, a turn's goal, or a skill's instructions), each with its own
        quota, so the model sees a few good candidates rather than the whole catalogue. Returns the
        names added."""
        eligible = self.eligible(task)
        pinned = [n for n in (*CONTROL_TOOL_NAMES, *task.task_type.settings.pin_tools, *self.always, *SKILL_TOOL_NAMES,
                              *CRON_TOOL_NAMES) if n in eligible]
        if len(eligible) <= self.small_catalogue:
            chosen = list(eligible)
        else:
            rest = [s for n, s in sorted(eligible.items()) if n not in pinned and n not in task.active_tools]
            skills = [s for s in rest if s.handler.kind is HandlerKind.SKILL]
            tools = [s for s in rest if s.handler.kind is not HandlerKind.SKILL]
            chosen, budget = list(pinned), self.schema_budget_tokens - sum(eligible[n].schema_tokens() for n in pinned)
            for spec in [*self.registry.search(requirements, skills, self.top_k_skills),
                         *self.registry.search(requirements, tools, self.top_k)]:
                if spec.schema_tokens() <= budget:
                    chosen.append(spec.name)
                    budget -= spec.schema_tokens()
        return self._activate(task, chosen)

    def discover(self, task: Task, query: str, k: int = 5) -> list[dict[str, str]]:
        """search_tools: activate matching eligible tools and skills from the next step."""
        eligible = self.eligible(task)
        candidates = [s for n, s in sorted(eligible.items()) if n not in task.active_tools]
        found = self.registry.search(query, candidates, k)
        self._activate(task, [s.name for s in found])
        return [{"name": s.name, "kind": "skill" if s.handler.kind is HandlerKind.SKILL else "tool",
                 "summary": s.summary} for s in found]

    def enter_skill(self, task: Task, query: str) -> list[str]:
        """A skill now runs (task.active_skill is set): drop what it does not offer from the active
        set (other skills; tools outside its `tools`), then add the tools closest to the skill
        itself. Returns the names added."""
        eligible = self.eligible(task)
        task.active_tools[:] = [n for n in task.active_tools if n in eligible]
        return self.select(task, query)
    def active(self, task: Task) -> list[ToolSpec]:
        """The tools offered this step: active and still eligible."""
        eligible = self.eligible(task)
        return [eligible[n] for n in task.active_tools if n in eligible]

    def activate(self, task: Task, names: list[str]) -> list[str]:
        """Add these tools to the active set, if they are eligible. Returns the names added."""
        eligible = self.eligible(task)
        return self._activate(task, [n for n in names if n in eligible])

    @staticmethod
    def _activate(task: Task, names: list[str]) -> list[str]:
        added = [n for n in dict.fromkeys(names) if n not in task.active_tools]
        task.active_tools.extend(added)   # the active set only grows within a task
        return added

    # --- one call: no side effects -------------------------------------------------------------
    def prepare(self, task: Task, call: ToolCall) -> Outcome:
        eligible = self.eligible(task)
        spec = eligible.get(call.name)
        if spec is None:   # also for tools that exist for someone else: do not reveal them
            return self._reject(task, call, ToolResult.error("unknown_tool", f"Unknown tool: {call.name}"))
        if call.name not in task.active_tools:   # eligible but not active: activate, then go on
            self._activate(task, [call.name])
        call.fingerprint = spec.fingerprint   # what the decision below is for; checked again in _run

        errors = self.registry.validate(spec, call.arguments)
        if errors:
            return self._reject(task, call, ToolResult.error("invalid_arguments", "; ".join(errors)))
        if spec.handler.kind is HandlerKind.SKILL and any(
                c is not call and c.step == call.step and c.decision is not None
                and (other := eligible.get(c.name)) is not None and other.handler.kind is HandlerKind.SKILL
                for c in task.calls):
            return self._reject(task, call, ToolResult.error(
                "one_skill_at_a_time", "Another skill was called in this step; use one skill at a time."))

        if spec.handler.kind is HandlerKind.CONTROL:
            call.decision = DecisionKind.ALLOW.value
            return Outcome.CONTROL

        decision: Decision = self.policy.decide(spec, call.arguments, task)
        call.decision, call.reason = decision.kind.value, f"{decision.rule}: {decision.reason}".strip(": ")
        if decision.kind is DecisionKind.DENY:
            call.status = ToolCallStatus.DENIED
            call.result = ToolResult.error("policy_denied", decision.reason)
            self.on_record(task, call)
            return Outcome.ERROR
        if decision.kind is DecisionKind.ASK:
            call.status = ToolCallStatus.AWAITING_APPROVAL
            self.on_record(task, call)
            return Outcome.NEEDS_APPROVAL
        call.status = ToolCallStatus.APPROVED
        return Outcome.ALLOWED

    def _reject(self, task: Task, call: ToolCall, result: ToolResult) -> Outcome:
        call.status, call.result = ToolCallStatus.FAILED, result
        self.on_record(task, call)
        return Outcome.ERROR

    # --- execution -----------------------------------------------------------------------------
    async def execute(self, task: Task, calls: list[ToolCall]) -> None:
        """Run approved calls; each call's result is set on it. Parallel when all are parallel_safe."""
        eligible = self.eligible(task)
        specs = [(call, eligible.get(call.name)) for call in calls]
        if all(spec is not None and spec.parallel_safe for _, spec in specs):
            await asyncio.gather(*(self._run(task, call, spec) for call, spec in specs))
        else:
            for call, spec in specs:
                await self._run(task, call, spec)

    async def _run(self, task: Task, call: ToolCall, spec: ToolSpec | None) -> None:
        if spec is None:   # lost eligibility since prepare (e.g. policy reload)
            call.status, call.result = ToolCallStatus.FAILED, ToolResult.error("unknown_tool", f"Unknown tool: {call.name}")
            self.on_record(task, call)
            return
        if call.fingerprint is not None and call.fingerprint != spec.fingerprint:
            # Changed since prepare (e.g. while waiting for approval): the decision was for the old one.
            LOG.warning("tool %s changed since call %s of task %s was prepared; not run (%s)",
                        spec.name, call.id, task.id, spec.source.location)
            call.status, call.result = ToolCallStatus.FAILED, ToolResult.error(
                "tool_changed", f"The definition of {call.name} changed after this call was requested, so it "
                "was not run. Read its description again and call it again if it still fits.")
            self.on_record(task, call)
            return
        for hook in self.hooks.pre_tool:
            blocked = hook(task, call, spec)
            if blocked is not None:
                call.status, call.result = ToolCallStatus.FAILED, blocked
                self.on_record(task, call)
                return

        call.status = ToolCallStatus.RUNNING
        task.tool_call_count += 1
        self.on_record(task, call)
        effect = None
        if spec.effective_effects is not Effects.NONE:   # write-ahead, so a resume knows the call may have acted
            effect = Effect(task_id=task.id, call_id=call.id, tool=spec.name, kind=spec.effective_effects.value,
                            target=_target(call.arguments), idempotency_key=f"{task.id}:{call.id}")
            self.on_effect(effect)
        context = CallContext(task_id=task.id, call_id=call.id, auth=task.auth,
                              deadline=time.time() + spec.timeout)
        with span(f"execute_tool {spec.name}", gen_ai__operation__name="execute_tool", gen_ai__tool__name=spec.name,
                  gen_ai__tool__call__id=call.id, agenticwf__task__id=task.id,
                  agenticwf__tool__handler=spec.handler.kind.value) as current:
            started = time.monotonic()
            call.started_at, call.trace_id = time.time(), trace_id()
            try:
                with use_auth_context(task.auth):
                    result = await asyncio.wait_for(self._invoke(spec, context, call.arguments), spec.timeout)
            except asyncio.TimeoutError:
                result = ToolResult.error("timeout", f"{spec.name} did not finish within {spec.timeout:g} s; it may "
                                          "still take effect, so check before trying again")
            except Exception as exc:   # a bug in the handler; the model gets a generic error
                LOG.exception("tool %s failed in task %s", spec.name, task.id)
                current.record_exception(exc)
                result = ToolResult.error("tool_error", f"{spec.name} failed with an internal error")

            if spec.handler.kind is not HandlerKind.SKILL:
                result.instructions = False   # only a skill's instructions are shown as instructions
            result = self._normalize(spec, result)
            for post in self.hooks.post_tool:
                result = post(task, call, result)
            if result.is_error:
                fail(current, result.error_code or "error")
        record_tool_call(tool=spec.name if spec.source.kind in ("core", "system", "plugin") else "user",
                         handler=spec.handler.kind.value, seconds=time.monotonic() - started,
                         error=(result.error_code or "error") if result.is_error else None)
        call.result, call.seconds = result, round(time.monotonic() - started, 3)
        call.status = ToolCallStatus.FAILED if result.is_error else ToolCallStatus.SUCCEEDED
        if effect is not None:
            effect.status = EffectStatus.FAILED if result.is_error else EffectStatus.APPLIED
            effect.detail = {k: v for k, v in (result.facts or {}).items() if k == "files"}
            # The result goes with the effect: a restart that finds the effect finished restores the
            # call from it rather than running a non-idempotent call again (AgentHarness._resume_call).
            effect.detail["result"] = {"content": result.content, "is_error": result.is_error,
                                       "error_code": result.error_code, "summary": result.summary[:500]}
            if result.error_code == "timeout":   # the handler may still be running and may still act
                effect.detail["outcome_unknown"] = True
            self.on_effect(effect)
        self.on_record(task, call)

    async def _invoke(self, spec: ToolSpec, context: CallContext, arguments: dict[str, Any]) -> ToolResult:
        handler = self.registry.handler(spec)
        if inspect.iscoroutinefunction(handler):
            result = await handler(context, arguments)
        else:   # to_thread copies the context, so the ambient AuthContext is visible there too
            result = await asyncio.to_thread(handler, context, arguments)
        return result if isinstance(result, ToolResult) else ToolResult(content=result)

    @staticmethod
    def _normalize(spec: ToolSpec, result: ToolResult) -> ToolResult:
        text = result.content if isinstance(result.content, str) else json.dumps(result.content, default=str)
        text = redact(text)
        if len(text.encode()) > spec.max_result_bytes:
            text = text.encode()[:spec.max_result_bytes].decode(errors="ignore") + "\n[truncated]"
        result.content = text
        return result


def _target(arguments: dict[str, Any]) -> str:
    """What a call acts on, as far as its arguments say: a path or URL if there is one."""
    for key in ("path", "file", "url", "target", "name"):
        if isinstance(arguments.get(key), str):
            return arguments[key]
    return json.dumps(arguments, sort_keys=True, default=str)[:200]
