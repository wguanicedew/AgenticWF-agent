"""
Durable records around a task (docs/agent-flow.md §9): what was asked (intent, TO DO), what is
being done (plan, turns, steps, effects: DOING), and what came out (task result and reply:
RESULT). The Task record itself is in agenticwf.agent.runtime.task; these are the pieces it is made of and the
records written before and after it.

Every record here is plain data, so a store can persist it and any worker can load it.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.llm.base import Message, Usage
from agenticwf.agent.runtime.types import Mode, Owner


def new_id() -> str:
    return uuid.uuid4().hex


# ================================================================================================
# TO DO: the intent
# ================================================================================================
class IntentStatus(str, Enum):
    PENDING = "pending"          # written by the session, waiting for intake
    CLAIMED = "claimed"          # intake is running
    CONVERTED = "converted"      # tasks created
    ANSWERED = "answered"        # the reply is written
    SUPERSEDED = "superseded"    # merged into a later intent
    CANCELLED = "cancelled"
    REJECTED = "rejected"        # e.g. the thread's token budget is used up


@dataclass
class Intent:
    """One thing the user asked for. It has no task type: the tasks intake creates from it do."""

    thread_id: str
    owner: Owner
    auth: AuthContext
    text: str
    requested_task_type: str | None = None   # a hint from the request or the thread's settings
    requested_mode: Mode | None = None
    model: str | None = None
    source: str = "web"                       # the thread's session_source: web | cli | api | cron
    conversation: list[Message] = field(default_factory=list)   # the excerpt the session selected
    conversation_summary: str = ""     # the earlier conversation, summarised (harness: _compact_conversation)
    planned: list[TaskSpec] = field(default_factory=list)   # tasks known in advance: intake creates them as they are
    id: str = field(default_factory=new_id)
    received_at: float = field(default_factory=time.time)
    status: IntentStatus = IntentStatus.PENDING
    task_ids: list[str] = field(default_factory=list)
    supersedes: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)   # intake and the final combine; tasks count their own

    @property
    def open(self) -> bool:
        return self.status in (IntentStatus.PENDING, IntentStatus.CLAIMED, IntentStatus.CONVERTED)


@dataclass(frozen=True)
class TaskSpec:
    """One task as intake's split step returns it."""

    key: str                      # local id within the intent, for depends_on
    request: str
    task_type: str
    depends_on: tuple[str, ...] = ()
    # Planned tasks (an intent's `planned`, e.g. a cron's checks) may set more:
    instructions: tuple[tuple[str, str], ...] = ()   # (title, text) after the task type's instructions
    tools: tuple[str, ...] | None = None             # only these tools (globs) besides the control tools
    skills: tuple[str, ...] | None = None            # only these skills
    mode: str | None = None
    timeout: float | None = None                     # seconds: the task's deadline
    unattended: str | None = None                    # propose | deny: what an `ask` becomes (no one to ask)
    cron: dict | None = None                         # a cron check's settings (agenticwf.agent.runtime.cron)
    tier: str | None = None                          # at least this model tier (docs/model-routing.md)


# ================================================================================================
# DOING: plan, turns, steps, effects
# ================================================================================================
@dataclass
class PlanItem:
    id: str
    goal: str
    phase: str = ""                                  # a WORKFLOW.md phase; guidance, not permissions
    needs: list[str] = field(default_factory=list)   # hints for tool and skill matching
    depends_on: list[str] = field(default_factory=list)
    verify: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_mapping(cls, data: dict[str, Any], fallback_id: str) -> "PlanItem":
        return cls(id=str(data.get("id") or fallback_id), goal=str(data.get("goal") or "").strip(),
                   phase=str(data.get("phase") or ""), needs=[str(n) for n in data.get("needs") or ()],
                   depends_on=[str(d) for d in data.get("depends_on") or ()],
                   verify=[dict(v) for v in data.get("verify") or () if isinstance(v, dict)])


class TurnState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    WAITING = "waiting"
    VERIFYING = "verifying"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StepRecord:
    number: int                    # the task-wide step number (counts against max_steps)
    text: str
    tool_calls: list[str]
    usage: Usage
    # For the trace view (records written before these existed lack them: read with getattr).
    started_at: float | None = None
    seconds: float | None = None
    model_id: str | None = None
    route: str | None = None
    cost: float = 0.0
    trace_id: str | None = None


@dataclass
class TurnResult:
    summary: str
    outputs: dict[str, Any] = field(default_factory=dict)
    artifacts: list[str] = field(default_factory=list)   # from the effect record, not the model's claim
    usage: Usage = field(default_factory=Usage)


@dataclass
class Turn:
    """One plan item being worked on."""

    item: PlanItem
    state: TurnState = TurnState.PENDING
    attempts: int = 0
    tools: list[str] = field(default_factory=list)       # matched at turn start
    steps: list[StepRecord] = field(default_factory=list)
    completion: dict[str, Any] | None = None            # complete_turn arguments, until verified
    result: TurnResult | None = None
    verification: list[dict[str, Any]] = field(default_factory=list)

    @property
    def id(self) -> str:
        return self.item.id

    @property
    def step_numbers(self) -> set[int]:
        return {s.number for s in self.steps}


class EffectStatus(str, Enum):
    INTENDED = "intended"     # written before the handler runs
    APPLIED = "applied"
    FAILED = "failed"


@dataclass
class Effect:
    """One change a tool call makes outside the task. Written ahead, so resume knows what happened."""

    task_id: str
    call_id: str
    tool: str
    kind: str                 # local | external (the tool's `effects`)
    target: str               # what it acts on, as far as the runtime can tell
    idempotency_key: str
    status: EffectStatus = EffectStatus.INTENDED
    detail: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=new_id)
    at: float = field(default_factory=time.time)


# ================================================================================================
# RESULT: task results and the reply
# ================================================================================================
@dataclass(frozen=True)
class TaskResult:
    task_id: str
    intent_id: str | None
    status: str                 # succeeded | failed | cancelled
    answer: str | None
    error: dict[str, str] | None
    turns: list[dict[str, Any]]
    artifacts: list[str]
    verification: list[dict[str, Any]]
    usage: Usage


@dataclass(frozen=True)
class Reply:
    intent_id: str
    task_ids: list[str]
    status: str                 # succeeded | partial | failed | cancelled
    answer: str
    artifacts: list[str]
    usage: Usage
