"""
Durable, thread-scoped configuration and mutable state: the *session context* (docs/agent-flow.md
§5), distinct from per-task state (the Task record and AgentContext the harness keeps). It stays in
memory on the session node while the thread is open, and survives across intents and tasks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenticwf.agent.runtime.llm.base import Message, Usage
from agenticwf.agent.runtime.session.budget import TokenBudget
from agenticwf.agent.runtime.types import Mode, Owner


@dataclass
class SessionConfiguration:
    """Fixed at thread creation (fork may carry it forward); settings changes apply to the next task."""

    thread_id: str
    owner: Owner                         # (subject, organization); only this principal may use the thread
    task_type: str = "default"           # default task type for inputs that don't name one
    mode: Mode | None = None             # None = the task type's mode
    model: str | None = None             # None = the runtime default / model router
    session_source: str = "web"          # web | cli | api | cron
    thread_name: str | None = None
    parent_thread_id: str | None = None  # sub-agent threads (run_subagent)
    forked_from_thread_id: str | None = None
    max_tokens: int | None = None        # cumulative token cap for the whole thread
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TranscriptMessage(Message):
    """A transcript entry with the keys of what it belongs to: its intent's id (every intent a
    merged message stands for), or `message:<id>` for a stored message that has none. A summary
    records the keys it covers, so it is kept with the chat and the rest is found again after a
    restart without summarising the whole transcript once more."""

    keys: tuple[str, ...] = ()


def covers(summarized: set[str] | frozenset[str], message: Message) -> bool:
    keys = getattr(message, "keys", ())
    return bool(keys) and all(k in summarized for k in keys)


@dataclass
class SessionState:
    """Mutable state that persists task to task within one thread."""

    configuration: SessionConfiguration
    history: list[Message] = field(default_factory=list)   # transcript: user inputs and replies (TranscriptMessage)
    summary: str = ""                                      # the earlier conversation, summarised
    summarized: set[str] = field(default_factory=set)      # the keys of the messages the summary covers
    intent_ids: list[str] = field(default_factory=list)
    task_ids: list[str] = field(default_factory=list)
    active_intent_id: str | None = None                    # one intent is worked on at a time
    usage_total: Usage = field(default_factory=Usage)
    budget: TokenBudget = field(default_factory=TokenBudget)

    def excerpt(self) -> list[Message]:
        """The transcript the summary does not cover yet: what the next intent carries besides it."""
        return [m for m in self.history if not covers(self.summarized, m)]

    def record_usage(self, usage: Usage) -> None:
        self.usage_total.add(usage)
        self.budget.record_usage(usage)
