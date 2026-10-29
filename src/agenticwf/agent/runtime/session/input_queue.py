"""
Inputs that arrive while a task is busy. The harness has no mid-task steering yet, so they wait
here and the session starts the next task with them once the current one finishes.

TODO: inter-agent messages (run_subagent results, send-to-thread) arrive here too.
"""

from __future__ import annotations

from collections import deque

from agenticwf.agent.runtime.messages import UserInput


class InputQueue:
    def __init__(self) -> None:
        self._pending: deque[UserInput] = deque()

    def push(self, item: UserInput) -> None:
        self._pending.append(item)

    def drain(self) -> list[UserInput]:
        items = list(self._pending)
        self._pending.clear()
        return items

    def has_pending(self) -> bool:
        return bool(self._pending)

    def __len__(self) -> int:
        return len(self._pending)
