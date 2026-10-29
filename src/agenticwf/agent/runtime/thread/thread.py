"""
RuntimeThread: the façade a host (web API, CLI) holds for one conversation.

Wraps a Session and exposes the small surface a host needs: send input, approve, interrupt, read
the next event, inspect history and status.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.llm.base import Message, Usage
from agenticwf.agent.runtime.messages import ApprovalDecision, Interrupt, Op, ThreadEvent, UpdateSettings, UserInput
from agenticwf.agent.runtime.session.session import Session
from agenticwf.agent.runtime.types import Mode, Owner


@dataclass(frozen=True)
class ThreadConfigSnapshot:
    """Read-only copy of a thread's configuration."""

    thread_id: str
    task_type: str
    mode: Mode | None
    model: str | None
    session_source: str
    thread_name: str | None
    parent_thread_id: str | None
    forked_from_thread_id: str | None


@dataclass
class RuntimeThread:
    thread_id: str
    session: Session
    created_at: float = field(default_factory=time.time)

    @property
    def owner(self) -> Owner:
        return self.session.configuration.owner

    async def submit(self, op: Op) -> str:
        return await self.session.submit(op)

    async def send(self, text: str, auth: AuthContext, task_type: str | None = None, mode: Mode | None = None,
                   model: str | None = None) -> str:
        """A new intent (returns its id), or the answer to a task's question (returns the task id)."""
        return await self.session.submit(UserInput(text=text, auth=auth, task_type=task_type, mode=mode, model=model))

    async def approve(self, call_id: str, auth: AuthContext, approved: bool = True,
                      remember: bool = False, reason: str = "") -> str:
        return await self.session.submit(ApprovalDecision(call_id, approved, auth, remember, reason))

    async def interrupt(self, auth: AuthContext) -> str:
        return await self.session.submit(Interrupt(auth))

    async def update_settings(self, auth: AuthContext, **changes) -> None:
        await self.session.submit(UpdateSettings(auth=auth, **changes))

    async def next_event(self) -> ThreadEvent:
        return await self.session.next_event()

    def history(self) -> list[Message]:
        return list(self.session.state.history)

    def intent_ids(self) -> list[str]:
        return list(self.session.state.intent_ids)

    def task_ids(self) -> list[str]:
        return list(self.session.state.task_ids)

    def status(self) -> str:
        """idle | running | waiting (for approval or the user)."""
        return self.session.status()

    def is_running(self) -> bool:
        return not self.session.closed

    def usage(self) -> Usage:
        return self.session.state.usage_total

    def config_snapshot(self) -> ThreadConfigSnapshot:
        cfg = self.session.configuration
        return ThreadConfigSnapshot(cfg.thread_id, cfg.task_type, cfg.mode, cfg.model, cfg.session_source,
                                    cfg.thread_name, cfg.parent_thread_id, cfg.forked_from_thread_id)

    async def shutdown(self) -> None:
        await self.session.shutdown()
