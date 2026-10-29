"""
PolicyEngine: every permission decision about tools (docs/agent-runtime.md §9).

Deterministic: no model calls, no I/O beyond its rules. It never reads AGENTS.md, WORKFLOW.md,
tool results, or anything the model wrote.

Rule order for decide():
    deployment rules (every deny, then ask)  →  missing scopes (deny)  →  task-type allow/deny list
    →  mode (plan: only tools without effects)  →  grants  →  tool default
       (user-defined tools: never looser than the deployment's user-tool default)
    →  auto mode turns ask into allow where permitted (never for user-defined tools)

User-declared fields never loosen policy: a user-defined tool's declared effects are not trusted
(it counts as `external`), and auto mode does not approve it. Skills are tools that load
instructions and act on nothing: the owner's own load without approval, others' shared skills ask.
"""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import yaml
from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.context.task_types import type_id
from agenticwf.agent.runtime.task import Task
from agenticwf.agent.runtime.tools.spec import Approval, Effects, HandlerKind, ToolSpec
from agenticwf.agent.runtime.types import Mode, owner_key


class DecisionKind(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class Decision:
    kind: DecisionKind
    reason: str = ""
    rule: str = ""   # which rule matched, for audit and "why do I have to approve this?"

    @classmethod
    def allow(cls, reason: str = "", rule: str = "") -> "Decision":
        return cls(DecisionKind.ALLOW, reason, rule)

    @classmethod
    def ask(cls, reason: str, rule: str) -> "Decision":
        return cls(DecisionKind.ASK, reason, rule)

    @classmethod
    def deny(cls, reason: str, rule: str) -> "Decision":
        return cls(DecisionKind.DENY, reason, rule)


@dataclass(frozen=True)
class PolicyRule:
    """Deployment rule. `tool` is a glob; `arguments` maps argument names to regexes (all must match)."""

    tool: str
    decision: DecisionKind
    reason: str = ""
    arguments: tuple[tuple[str, str], ...] = ()
    sources: tuple[str, ...] = ()   # only tasks from these sources (web, cli, api, cron); empty: all

    def applies_to(self, task: Task) -> bool:
        return not self.sources or getattr(task, "source", "web") in self.sources

    def matches(self, spec: ToolSpec, arguments: dict[str, Any] | None) -> bool:
        if not fnmatch.fnmatchcase(spec.name, self.tool):
            return False
        if not self.arguments:
            return True
        if arguments is None:   # visibility check: argument rules never hide a tool
            return False
        return all(re.search(pattern, str(arguments.get(name, ""))) for name, pattern in self.arguments)


@dataclass(frozen=True)
class PolicyConfig:
    rules: tuple[PolicyRule, ...] = ()
    user_tool_default: Approval = Approval.ASK
    auto_mode_allowed: bool = True
    auto_mode_for_machines: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "PolicyConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        rules = tuple(
            PolicyRule(r["tool"], DecisionKind(r["decision"]), r.get("reason", ""),
                       tuple((r.get("arguments") or {}).items()), tuple(r.get("sources") or ()))
            for r in data.get("rules") or ()
        )
        return cls(rules=rules,
                   user_tool_default=Approval(data.get("user_tool_default", "ask")),
                   auto_mode_allowed=bool(data.get("auto_mode_allowed", True)),
                   auto_mode_for_machines=bool(data.get("auto_mode_for_machines", False)))


_STRICTNESS = {Approval.AUTO: 0, Approval.ASK: 1, Approval.DENY: 2}


class PolicyEngine:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    # --- which tools a task may see -------------------------------------------------------------
    def visible(self, specs: Iterable[ToolSpec], task: Task) -> list[ToolSpec]:
        return [s for s in specs if self._eligible(s, task, None).kind is not DecisionKind.DENY]

    # --- one call ----------------------------------------------------------------------------------
    def decide(self, spec: ToolSpec, arguments: dict[str, Any], task: Task) -> Decision:
        decision = self._eligible(spec, task, arguments)
        if decision.kind is not DecisionKind.ALLOW:
            return decision
        if spec.handler.kind is HandlerKind.CONTROL:
            return Decision.allow(rule="control")

        if f"{spec.name}@{spec.fingerprint}" in task.grants:   # a changed definition asks again
            return Decision.allow("approved earlier in this task", "grant")

        approval = spec.approval
        # A skill of the task's owner is their own instructions: no approval to load it. Someone
        # else's shared skill is text the owner has not seen: it asks like any user-defined tool.
        own_skill = spec.handler.kind is HandlerKind.SKILL and spec.source.owner == task.owner
        if (spec.source.user_defined and not own_skill
                and _STRICTNESS[approval] < _STRICTNESS[self.config.user_tool_default]):
            approval = self.config.user_tool_default   # user-declared fields never loosen policy
        if approval is Approval.DENY:
            return Decision.deny("tool is disabled by default", "tool-default")
        if approval is Approval.ASK:
            # Someone else's shared tool (or your own) can describe itself as anything: auto mode
            # approves only tools operators installed.
            if task.mode is Mode.AUTO and self._auto_permitted(task) and not spec.source.user_defined:
                return Decision.allow("auto mode", "mode:auto")
            return Decision.ask(f"{spec.name} requires approval ({spec.effective_effects.value} effects)", "tool-default")
        return Decision.allow(rule="tool-default")

    def _eligible(self, spec: ToolSpec, task: Task, arguments: dict[str, Any] | None) -> Decision:
        """Checks shared by visible() and decide(); `arguments=None` means a visibility check."""
        if spec.handler.kind is HandlerKind.CONTROL:
            return Decision.allow(rule="control")

        matching = [r for r in self.config.rules if r.applies_to(task) and r.matches(spec, arguments)]
        for rule in matching:   # every deny before any ask: a broad ask never hides a narrower deny
            if rule.decision is DecisionKind.DENY:
                return Decision.deny(rule.reason or "denied by deployment policy", f"deployment:{rule.tool}")
        for rule in matching:
            if rule.decision is DecisionKind.ASK and arguments is not None:
                return Decision.ask(rule.reason or "deployment policy requires approval", f"deployment:{rule.tool}")

        missing = spec.scopes - task.auth.scopes
        if missing:
            return Decision.deny(f"missing scopes: {' '.join(sorted(missing))}", "scopes")

        settings = task.task_type.settings
        if settings.tools_allow and not any(fnmatch.fnmatchcase(spec.name, p) for p in settings.tools_allow):
            return Decision.deny("not allowed in this task type", "task-type:allow")
        if any(fnmatch.fnmatchcase(spec.name, p) for p in settings.tools_deny):
            return Decision.deny("denied in this task type", "task-type:deny")
        # The type that resolved, by id; a tool's bare names mean system types (`review` = `system:review`).
        if spec.task_types and type_id(task.task_type.effective) not in {type_id(t) for t in spec.task_types}:
            return Decision.deny("tool is not offered in this task type", "tool:task_types")

        if task.mode is Mode.PLAN and spec.effective_effects is not Effects.NONE:
            return Decision.deny("plan mode allows only tools without effects", "mode:plan")
        return Decision.allow()

    def _auto_permitted(self, task: Task) -> bool:
        if not self.config.auto_mode_allowed:
            return False
        return self.config.auto_mode_for_machines or not task.auth.is_machine

    # --- approvals -----------------------------------------------------------------------------
    def can_approve(self, task: Task, approver: AuthContext) -> bool:
        """Owner only, and never a machine token (matches the authsvc route rule)."""
        return not approver.is_machine and owner_key(approver) == task.owner
