"""
Operations a host (web API, CLI) submits to a RuntimeThread, and the events a thread emits.

Every operation carries the caller's AuthContext, verified by authsvc for that request. Threads
are owned by one principal; a task snapshots the AuthContext of the input that started it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Union

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.types import Mode


@dataclass
class UserInput:
    """Start a task, answer a pending ask_user, or queue behind the running task."""

    text: str
    auth: AuthContext
    task_type: str | None = None   # None = the thread's task type
    mode: Mode | None = None
    model: str | None = None       # a model id; None = the thread's model


@dataclass
class ApprovalDecision:
    call_id: str
    approved: bool
    auth: AuthContext
    remember: bool = False   # approve this tool for the rest of the task
    reason: str = ""


@dataclass
class Interrupt:
    auth: AuthContext


@dataclass
class UpdateSettings:
    """Change the thread's defaults; applies to the next task."""

    auth: AuthContext
    task_type: str | None = None
    mode: Mode | None = None
    thread_name: str | None = None


Op = Union[UserInput, ApprovalDecision, Interrupt, UpdateSettings]


@dataclass
class ThreadEvent:
    """
    One event on a thread's stream (relayed to SSE by the API).

    Session events:  session_configured, task_started, input_queued, task_waiting, task_complete,
                     settings_applied, error, shutdown_complete
    Harness events:  task, turn, message_delta, plan, tool, tools, approval_required,
                     input_required, task_type_fallback, completion_rejected
    """

    type: str
    thread_id: str
    data: dict[str, Any] = field(default_factory=dict)
    seq: int = 0
    time: float = field(default_factory=time.time)
