"""
The task record the harness owns and the dispatcher and policy engine read
(docs/agent-runtime.md §7). AgentContext (what the model sees) is kept separately; the
AuthContext lives here and only here, so rendering can never leak it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.context.task_types import ResolvedTaskType
from agenticwf.agent.runtime.records import PlanItem, Turn, TurnState
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.types import Mode, Owner, RuntimeState, owner_key


class ToolCallStatus(str, Enum):
    REQUESTED = "requested"
    DENIED = "denied"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    step: int
    status: ToolCallStatus = ToolCallStatus.REQUESTED
    decision: str | None = None      # allow | ask | deny, with the policy reason
    reason: str = ""
    result: ToolResult | None = None
    fingerprint: str | None = None   # ToolSpec.fingerprint when prepared; it must still match to run
    started_at: float | None = None  # when the handler started, for the trace view
    seconds: float | None = None
    trace_id: str | None = None


@dataclass
class Task:
    auth: AuthContext
    input: str
    task_type: ResolvedTaskType
    mode: Mode = Mode.DEFAULT
    id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str | None = None
    intent_id: str | None = None
    depends_on: list[str] = field(default_factory=list)     # task ids of the same intent
    model: str = "default"
    state: RuntimeState = RuntimeState.QUEUED
    plan: list[PlanItem] = field(default_factory=list)      # current revision (docs/agent-flow.md §4)
    plan_revisions: list[list[PlanItem]] = field(default_factory=list)
    turns: list[Turn] = field(default_factory=list)
    active_turn: str | None = None
    cancel_requested: bool = False                          # in the record, so any worker sees it
    active_tools: list[str] = field(default_factory=list)   # the active turn's tools; grows within a turn,
                                                            # narrows once when a skill starts
    active_skill: dict | None = None   # the skill running in the active turn: id, tool, tools (globs), files
    grants: set[str] = field(default_factory=set)           # "<tool>@<fingerprint>" the owner approved for this task
    calls: list[ToolCall] = field(default_factory=list)
    tool_call_count: int = 0
    created_at: float = field(default_factory=time.time)
    deadline: float | None = None
    waiting_since: float | None = None   # entered an approval/input wait: that time does not count against the deadline
    result: str | None = None
    error: dict[str, str] | None = None
    consolidation: str | None = None   # pending: its working memory is still to go to long-term memory; done
    source: str = "web"                 # the intent's source: web | cli | api | cron (policy rules may match it)
    tool_globs: tuple[str, ...] | None = None    # only these tools besides the control tools (a cron check's)
    skill_names: tuple[str, ...] | None = None   # only these skills
    unattended: str | None = None       # nobody to ask: an `ask` becomes a proposal (propose) or a denial (deny)
    proposals: list[dict] = field(default_factory=list)   # calls proposed instead of asked (unattended: propose)
    cron: dict | None = None            # a cron check: its settings, and what it reported (agenticwf.agent.runtime.cron)
    # the long-term memories recalled at its start ({"id", "text"}), for the chat page; a default, not a
    # factory, so tasks saved before it existed load with none
    recalled: tuple[dict, ...] = ()
    # Model routing (docs/model-routing.md)
    pinned: bool = False                # its model was chosen (message, thread, task type, cron): not routed
    min_tier: str | None = None         # at least this tier (the task type's or cron's; raised by escalation)
    model_id: str | None = None         # the model it has used so far: kept while it fits (its prompt cache)
    escalations: int = 0
    cost: float = 0.0                   # in the deployment's cost unit
    cached_hint: int = 0                # the last step's input tokens: what the current model likely has cached
    invalid_streak: int = 0             # tool calls in a row the model got wrong (escalation evidence)

    @property
    def owner(self) -> Owner:
        return owner_key(self.auth)

    @property
    def multi_turn(self) -> bool:
        return len(self.plan) > 1

    def turn(self, turn_id: str | None = None) -> Turn | None:
        """The turn with this id, or the active turn."""
        wanted = turn_id or self.active_turn
        return next((t for t in self.turns if t.id == wanted), None)

    def next_item(self) -> PlanItem | None:
        """The first plan item without a turn whose dependencies have all succeeded."""
        done = {t.id for t in self.turns if t.state in (TurnState.SUCCEEDED, TurnState.SKIPPED)}
        started = {t.id for t in self.turns}
        return next((i for i in self.plan if i.id not in started and set(i.depends_on) <= done), None)

    def call(self, call_id: str) -> ToolCall | None:
        return next((c for c in self.calls if c.id == call_id), None)

    def open_calls(self, *statuses: ToolCallStatus) -> list[ToolCall]:
        """Calls of the current step in the given statuses."""
        return [c for c in self.calls if c.status in statuses]
