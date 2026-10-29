"""
A hard cap on a thread's cumulative token usage across all its tasks: a cost cap, independent
of context-window compaction (which a single task handles) and of per-task limits (max_steps,
budget_exceeded in the harness).

TODO: reconstruct `used_tokens` from persisted usage when a thread is resumed, so a restart does
not reset the spend.
"""

from __future__ import annotations

from dataclasses import dataclass

from agenticwf.agent.runtime.llm.base import Usage

DEFAULT_REMINDER_THRESHOLD = 0.8


@dataclass
class TokenBudget:
    max_tokens: int | None = None
    used_tokens: int = 0
    reminder_claimed: bool = False

    def record_usage(self, usage: Usage) -> None:
        self.used_tokens += usage.input_tokens + usage.output_tokens

    def remaining(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.used_tokens)

    def is_exceeded(self) -> bool:
        return self.max_tokens is not None and self.used_tokens >= self.max_tokens

    def claim_reminder(self, threshold: float = DEFAULT_REMINDER_THRESHOLD) -> bool:
        """One-shot: True the first time usage crosses `threshold` of the budget."""
        if self.max_tokens is None or self.reminder_claimed or self.is_exceeded():
            return False
        if self.used_tokens >= self.max_tokens * threshold:
            self.reminder_claimed = True
            return True
        return False
