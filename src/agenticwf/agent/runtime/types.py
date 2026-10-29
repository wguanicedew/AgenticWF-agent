"""
Types shared by the runtime components (docs/agent-runtime.md, docs/agent-structure.md).

The lifecycle states below are the *runtime* state machine owned by the harness. They are the
same for every task type. Semantic phases (PLAN, RESEARCH, ...) are not runtime states: they live
in WORKFLOW.md as guidance for the model.
"""

from __future__ import annotations

import contextlib
from enum import Enum
from typing import Iterator

from agenticwf.auth.client import _current as _auth_context_var  # see use_auth_context()
from agenticwf.auth.context import AuthContext

Owner = tuple[str, str | None]


def owner_key(ctx: AuthContext) -> Owner:
    """Who owns tasks, user task types, plugins and memories: (subject, organization)."""
    return (ctx.subject, ctx.organization_id)


SYSTEM_SCOPE = "system"   # what operators install (system plugins and their MCP endpoints)


def scope_of(owner: Owner | None) -> str:
    """The scope of what an owner defines (plugins, MCP endpoints, their tools): its identity is
    `<scope>:<name>`, so the same name of different owners never conflicts. Names have no ":", so
    an identity splits at its last one."""
    if owner is None:
        return SYSTEM_SCOPE
    subject, organization = owner
    return f"{subject}@{organization}" if organization else subject


def split_ref(ref: str, owner: Owner | None) -> tuple[str, str]:
    """(scope, name) of `scope:name`, or of a bare `name` in `owner`'s own scope."""
    scope, sep, name = ref.rpartition(":")
    return (scope, name) if sep else (scope_of(owner), ref)


class RuntimeState(str, Enum):
    QUEUED = "queued"
    PLANNING = "planning"            # one model call turns the request into plan items
    MODEL_RUNNING = "model_running"
    TOOL_REQUESTED = "tool_requested"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    TOOL_RUNNING = "tool_running"
    VERIFYING = "verifying"          # the active turn ended; its result is being checked
    COMBINING = "combining"          # every turn is done; one model call writes the answer
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def terminal(self) -> bool:
        return self in TERMINAL_STATES

    @property
    def public_status(self) -> str:
        """Coarse status exposed by GET /api/v1/tasks/{id} (docs/agent-runtime.md §7)."""
        return _PUBLIC_STATUS[self]


TERMINAL_STATES = frozenset({RuntimeState.COMPLETED, RuntimeState.FAILED, RuntimeState.CANCELLED})

_PUBLIC_STATUS = {
    RuntimeState.QUEUED: "queued",
    RuntimeState.PLANNING: "running",
    RuntimeState.MODEL_RUNNING: "running",
    RuntimeState.TOOL_REQUESTED: "running",
    RuntimeState.TOOL_RUNNING: "running",
    RuntimeState.VERIFYING: "running",
    RuntimeState.COMBINING: "running",
    RuntimeState.WAITING_APPROVAL: "waiting",
    RuntimeState.WAITING_USER: "waiting",
    RuntimeState.COMPLETED: "succeeded",
    RuntimeState.FAILED: "failed",
    RuntimeState.CANCELLED: "cancelled",
}


class Mode(str, Enum):
    """Task permission mode, modelled on Claude Code's permission modes."""

    DEFAULT = "default"
    PLAN = "plan"      # only tools without effects
    AUTO = "auto"      # `ask` becomes `allow` where the deployment permits it


@contextlib.contextmanager
def use_auth_context(ctx: AuthContext) -> Iterator[None]:
    """
    Make `ctx` the ambient `agenticwf.auth.client.current_auth_context()` while handlers run, so tool
    code reads identity exactly as API handlers do.

    TODO: authsvc has no public setter yet; move this into agenticwf.auth.client (e.g.
    `use_auth_context`) and stop touching its private ContextVar.
    """
    token = _auth_context_var.set(ctx)
    try:
        yield
    finally:
        _auth_context_var.reset(token)
