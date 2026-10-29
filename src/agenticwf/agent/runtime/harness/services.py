"""
HarnessServices: the collaborators the AgentHarness and a thread's Session work with.

bootstrap builds one HarnessServices for the process and hands it to the AgentHarness;
AgentHarness.services_factory derives one per thread from it, for the pieces that belong to a
single thread (the rollout recorder, once it exists).

Required: the model client, the context manager, the tool dispatcher, the policy engine, the task
type resolver and the runtime store (intents, tasks, effects, results, jobs). Optional and
implemented: `plugins`, the PluginManager behind the registry's plugin tools and the resolver's
plugin task types (docs/plugins.md); `mcp`, the McpCatalog of MCP endpoints users add
(docs/mcp.md); `skills`, the SkillCatalog (docs/skills.md); `otel`, OpenTelemetry tracing
(agenticwf.agent.runtime.telemetry). The rest are skeleton slots, to be implemented:
    rollout               persist thread events and transcripts, for resume and audit
    hooks                 deterministic thread-level hooks (TaskStart, TaskStop, ThreadIdle)
    approval_source       where approval decisions come from besides the owner (rules, delegates)
    agent_control         sub-agents: spawn, message and stop child threads
    compactor             summarise old context (today: AgentHarness._maybe_compact)
    memory_usage_tracker  token and memory usage per thread and owner
    time_provider         clock for deadlines and timestamps (tests, replay)
    unified_exec          long-running processes shared across tool calls (shells, PTYs)
    judge                 model-based review of a turn's result (verify kind: judge)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from agenticwf.agent.runtime.context.agent_context import AgentContext
from agenticwf.agent.runtime.context.manager import ContextManager
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.store import InMemoryStore, RuntimeStore
from agenticwf.agent.runtime.llm.base import LLM
from agenticwf.agent.runtime.mcp.catalog import McpCatalog
from agenticwf.agent.runtime.messages import ThreadEvent
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.policy.engine import PolicyEngine
from agenticwf.agent.runtime.records import Turn
from agenticwf.agent.runtime.skills.catalog import SkillCatalog
from agenticwf.agent.runtime.task import Task
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher


class RolloutRecorder(Protocol):
    """TODO: store events and the transcript (agenticwf.agent.core.orm), so threads can be listed and resumed."""

    async def record_event(self, event: ThreadEvent) -> None: ...

    async def flush(self) -> None: ...

    async def close(self) -> None: ...


class HooksRuntime(Protocol):
    """TODO: deterministic thread-level hooks (TaskStart, TaskStop, ThreadIdle)."""

    async def run(self, event: str, payload: dict[str, Any]) -> None: ...


class ApprovalSource(Protocol):
    """TODO: decide an approval without asking the owner; None means ask."""

    async def decide(self, task: Task, call_id: str) -> bool | None: ...


class AgentControl(Protocol):
    """TODO: spawn and manage sub-agent threads."""

    async def spawn(self, parent: Task, text: str, task_type: str) -> str: ...


class Compactor(Protocol):
    """TODO: summarise old context; replaces AgentHarness._maybe_compact when set."""

    async def compact(self, task: Task, ctx: AgentContext) -> None: ...


class OtelProvider(Protocol):
    """OpenTelemetry tracing (agenticwf.agent.runtime.telemetry.Tracing); metrics are that module's functions."""

    def tracer(self, name: str) -> Any: ...

    def span(self, name: str, **attributes: Any) -> Any: ...


class MemoryUsageTracker(Protocol):
    """TODO: token and memory usage per thread and owner."""

    def record(self, task: Task, input_tokens: int, output_tokens: int) -> None: ...


class TimeProvider(Protocol):
    """TODO: injectable clock for deadlines and timestamps."""

    def now(self) -> float: ...


class UnifiedExecManager(Protocol):
    """TODO: long-running processes (shells, PTYs) shared across tool calls."""

    async def terminate_all(self) -> None: ...


class JudgeHooks(Protocol):
    """Model-based review of a turn's result, for `verify: [{kind: judge}]` checks
    (docs/agent-flow.md §7). Return an objection to reopen the turn, else None."""

    async def review(self, task: Task, turn: Turn, ctx: AgentContext) -> str | None: ...


@dataclass
class HarnessServices:
    llm: LLM
    tool_dispatcher: ToolDispatcher
    task_types: TaskTypeResolver
    context_manager: ContextManager = field(default_factory=ContextManager)
    policy: PolicyEngine | None = None   # None: the dispatcher's
    store: RuntimeStore = field(default_factory=InMemoryStore)
    rollout: RolloutRecorder | None = None
    hooks: HooksRuntime | None = None
    approval_source: ApprovalSource | None = None
    skills: SkillCatalog | None = None      # skills (agenticwf.agent.runtime.skills): also the registry's source of skill__ tools
    plugins: PluginManager | None = None    # the registry's and the resolver's plugin source
    mcp: McpCatalog | None = None           # MCP endpoints in the database (the registry's database source)
    agent_control: AgentControl | None = None
    compactor: Compactor | None = None
    otel: OtelProvider | None = None        # tracing; a no-op unless set up (agenticwf.agent.runtime.telemetry)
    memory_usage_tracker: MemoryUsageTracker | None = None
    time_provider: TimeProvider | None = None
    unified_exec: UnifiedExecManager | None = None
    judge: JudgeHooks | None = None

    def __post_init__(self) -> None:
        if self.policy is None:
            self.policy = self.tool_dispatcher.policy
        elif self.policy is not self.tool_dispatcher.policy:
            raise ValueError("the policy engine must be the one the tool dispatcher uses")
