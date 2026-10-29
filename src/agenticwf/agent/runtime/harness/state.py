"""
Legal lifecycle transitions (docs/agent-runtime.md §7, docs/agent-flow.md §6). Every state
change goes through `check_transition` first, so an illegal move (e.g. approving a call of a
cancelled task) is rejected before anything happens.

    queued → planning → model_running ⇄ tool_* → verifying → (next turn) model_running …
                                                           → combining → completed

The model and tool states are the states of the task's active turn.
"""

from __future__ import annotations

from agenticwf.agent.runtime.types import RuntimeState as S

_ANY_ACTIVE = {S.FAILED, S.CANCELLED}

TRANSITIONS: dict[S, set[S]] = {
    S.QUEUED: {S.PLANNING, S.MODEL_RUNNING} | _ANY_ACTIVE,
    S.PLANNING: {S.MODEL_RUNNING} | _ANY_ACTIVE,
    S.MODEL_RUNNING: {S.VERIFYING, S.TOOL_REQUESTED} | _ANY_ACTIVE,
    S.TOOL_REQUESTED: {S.TOOL_RUNNING, S.WAITING_APPROVAL, S.WAITING_USER, S.MODEL_RUNNING} | _ANY_ACTIVE,
    S.WAITING_APPROVAL: {S.TOOL_RUNNING, S.MODEL_RUNNING} | _ANY_ACTIVE,
    S.TOOL_RUNNING: {S.MODEL_RUNNING, S.WAITING_USER, S.VERIFYING} | _ANY_ACTIVE,
    S.WAITING_USER: {S.MODEL_RUNNING} | _ANY_ACTIVE,
    S.VERIFYING: {S.MODEL_RUNNING, S.COMBINING, S.COMPLETED} | _ANY_ACTIVE,
    S.COMBINING: {S.MODEL_RUNNING, S.COMPLETED} | _ANY_ACTIVE,
    S.COMPLETED: set(),
    S.FAILED: set(),
    S.CANCELLED: set(),
}

WAITING_STATES = frozenset({S.WAITING_APPROVAL, S.WAITING_USER})


class IllegalTransition(Exception):
    pass


def check_transition(current: S, new: S) -> None:
    if new not in TRANSITIONS[current]:
        raise IllegalTransition(f"{current.value} -> {new.value}")
