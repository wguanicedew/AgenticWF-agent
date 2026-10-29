"""
The model calls the harness makes around the agent loop (docs/agent-flow.md §4, §7): split an
intent into typed tasks, plan a task into turns, combine turn results into a task's answer, and
combine task results into the reply.

The prompts themselves are Markdown files, not code (prompts/<name>.md in the system directory,
packaged defaults in runtime/defaults/prompts; a task type may override the per-task ones):
TaskTypeResolver.prompt() finds them. This module holds the intake settings and the parsers. The
prompts ask for JSON; the parsers accept the first JSON value in the text and return None for
anything malformed, so the caller falls back to one task or one turn instead of failing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from agenticwf.agent.runtime.context.task_types import DEFAULT, TYPE_REF_RE
from agenticwf.agent.runtime.records import PlanItem, TaskSpec


@dataclass(frozen=True)
class IntakeSettings:
    split: bool = False      # split intents into several tasks (one model call per intent)
    plan: str = "never"      # always | never: plan tasks into turns; task.yaml `plan` overrides
    max_tasks: int = 5
    max_plan_items: int = 8


def first_json(text: str) -> Any | None:
    """The first JSON object or array in `text` (models sometimes wrap it in prose or fences)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch in "{[":
            try:
                value, _ = decoder.raw_decode(text[i:])
                return value
            except ValueError:
                continue
    return None


def _items(value: Any, key: str) -> list[dict[str, Any]] | None:
    if isinstance(value, dict):
        value = value.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, dict) for v in value):
        return None
    return value


def parse_tasks(text: str, max_tasks: int) -> list[TaskSpec] | None:
    """Task specs in dependency order; dependencies may only name earlier tasks."""
    raw = _items(first_json(text), "tasks")
    if raw is None or len(raw) > max_tasks:
        return None
    specs, seen = [], set()
    for n, entry in enumerate(raw):
        key = str(entry.get("id") or f"T{n + 1}")
        request = str(entry.get("request") or "").strip()
        if not request or key in seen:
            return None
        task_type = str(entry.get("task_type") or DEFAULT)
        depends = tuple(str(d) for d in entry.get("depends_on") or ())
        if not set(depends) <= seen:
            return None
        specs.append(TaskSpec(key, request, task_type if TYPE_REF_RE.match(task_type) else DEFAULT, depends))
        seen.add(key)
    return specs


def parse_plan(text: str, max_items: int) -> list[PlanItem] | None:
    """Plan items in order; dependencies may only name earlier items."""
    raw = _items(first_json(text), "items")
    if raw is None or len(raw) > max_items:
        return None
    items, seen = [], set()
    for n, entry in enumerate(raw):
        item = PlanItem.from_mapping(entry, fallback_id=f"t{n + 1}")
        if not item.goal or item.id in seen or not set(item.depends_on) <= seen:
            return None
        items.append(item)
        seen.add(item.id)
    return items
