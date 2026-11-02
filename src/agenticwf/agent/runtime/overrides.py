"""
Settings changed on the admin page over what operators put on disk (docs/admin-page.md, Settings):
agent.cfg's [harness] and [llm.routing], and each task type's task.yaml. The files are never
written: the database holds the changes, in two layers.

    everyone    an admin's value, for every task (owner None)
    personal    one person's value, for their own tasks: only the keys marked `personal`, which
                cannot loosen security or raise cost (plan, split, a lower max_steps)

A value in effect is the person's, else everyone's, else the file's. Sections:

    harness               split, plan, max_tasks, max_plan_items, max_steps, max_tool_calls, deadline_seconds
    llm.routing           mode, a tier per purpose, context_headroom, max_escalations, task_budget, feedback
    llm.purpose           a model id per purpose (admins only)
    llm.budgets           default, organization.<org>, user.<subject> ('<amount> per day|month') and
                          max_tier, max_tier.<subject>, max_tier.organization.<org> (admins only)
    task_type:<id>        the task.yaml settings (TASK_TYPE_KEYS); a type without its own task.yaml
                          inherits default's file and default's changes

Values are checked when set (check) and coerced again when read, so a row that no longer fits
(a key renamed, a tier removed) is skipped instead of breaking tasks.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol

from agenticwf.agent.runtime.types import Owner

HARNESS, ROUTING, TASK_TYPE = "harness", "llm.routing", "task_type:"
PURPOSE, BUDGETS = "llm.purpose", "llm.budgets"
LOWER_ONLY = ("max_steps",)   # a person's value counts only below the value it replaces


@dataclass(frozen=True)
class Key:
    kind: str                            # bool | choice | int | float | list | text
    coerce: Callable[[Any], Any]
    personal: bool = False               # a person may set it for their own tasks
    choices: tuple[str, ...] = ()
    help: str = ""

    def view(self) -> dict[str, Any]:
        return {"kind": self.kind, "choices": list(self.choices), "personal": self.personal, "help": self.help}


def _bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{value!r} is not true or false")


def _choice(*options: str) -> Key:
    def coerce(value: Any) -> str:
        if str(value) not in options:
            raise ValueError(f"{value!r} is not one of {', '.join(options)}")
        return str(value)
    return Key("choice", coerce, choices=options)


def _number(kind: type, low: float, high: float) -> Callable[[Any], Any]:
    def coerce(value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError(f"{value!r} is not a number")
        number = kind(value)
        if not low <= number <= high:
            raise ValueError(f"{number} is not between {low:g} and {high:g}")
        return number
    return coerce


def _words(value: Any) -> list[str]:
    items = value.replace(",", " ").split() if isinstance(value, str) else list(value)
    if not all(isinstance(i, str) and i.strip() for i in items):
        raise ValueError("a list of names")
    return [i.strip() for i in items]


def _text(value: Any) -> str:
    text = str(value).strip()
    if not text or len(text) > 255:
        raise ValueError("1 to 255 characters")
    return text


def _with(key: Key, **changes: Any) -> Key:
    return Key(**{**key.__dict__, **changes})


def _keys() -> dict[str, dict[str, Key]]:
    from agenticwf.agent.runtime.llm.routing import PURPOSES, TIERS
    from agenticwf.agent.runtime.types import Mode

    modes = tuple(m.value for m in Mode)
    plan = _with(_choice("always", "never"), personal=True, help="plan each task into turns first")

    def mode_list(value: Any) -> list[str]:
        found = _words(value)
        if not set(found) <= set(modes):
            raise ValueError(f"modes are {', '.join(modes)}")
        return found

    harness = {
        "split": Key("bool", _bool, personal=True, help="split a message into several typed tasks (one model call)"),
        "plan": plan,
        "max_tasks": Key("int", _number(int, 1, 20), help="tasks one message may become"),
        "max_plan_items": Key("int", _number(int, 1, 50), help="turns one plan may have"),
        "max_steps": Key("int", _number(int, 1, 1000), personal=True,
                         help="model calls per task, when its type sets none (yours: only lower)"),
        "max_tool_calls": Key("int", _number(int, 1, 10000), help="tool calls per task"),
        "deadline_seconds": Key("float", _number(float, 10, 7 * 86400), help="how long a task may run"),
    }
    routing = {
        "mode": _with(_choice("auto", "pinned"), help="auto: choose models by need and cost; pinned: the named model"),
        **{p: _with(_choice(*TIERS), help=f"the least tier for {p} calls") for p in PURPOSES},
        "context_headroom": Key("float", _number(float, 0, 0.9), help="share of a model's context kept free"),
        "max_escalations": Key("int", _number(int, 0, 3), help="retries one tier up after a failed call"),
        "task_budget": Key("float", _number(float, 0, 1e9), help="a task's cost limit, in cost units"),
        "feedback": _with(_choice("suggest", "adjust"), help="adjust: feedback raises a purpose's tier by itself"),
    }
    task_type = {
        "plan": plan,
        "mode": _with(_choice(*modes), help="the default mode of its tasks"),
        "modes": Key("list", mode_list, choices=modes, help="the modes a task may ask for"),
        "max_steps": Key("int", _number(int, 1, 1000), help="model calls per task"),
        "model": Key("text", _text, help="a model id or tier for its tasks"),
        "tier": _with(_choice(*TIERS), help="at least this model tier"),
        "budget": Key("float", _number(float, 0, 1e9), help="a task's cost limit, in cost units"),
        "pin_tools": Key("list", _words, help="tools always offered"),
        "tools_allow": Key("list", _words, help="only these tools (globs); empty: no allow-list"),
        "tools_deny": Key("list", _words, help="never these tools (globs)"),
        "knowledge": Key("list", _words, help="knowledge-base ids"),
    }
    purpose = {p: Key("text", _text, help=f"always this model id for {p} calls") for p in PURPOSES}
    budgets = {"default": _with(_budget_key, help="each owner's cost limit, e.g. 20 per day"),
               "max_tier": _with(_choice(*TIERS), help="the highest model tier an owner's tasks get")}
    return {HARNESS: harness, ROUTING: routing, TASK_TYPE: task_type, PURPOSE: purpose, BUDGETS: budgets}


def _budget(value: Any) -> str:
    from agenticwf.agent.runtime.llm.routing import Budget

    return Budget.parse(str(value)).text()


_budget_key = Key("text", _budget)
_OWNER_KEY = re.compile(r"^(user|organization)\.[\w.@:-]+$")
_TIER_KEY = re.compile(r"^max_tier\.(organization\.)?[\w.@:-]+$")

_KEYS: dict[str, dict[str, Key]] | None = None


def keys_for(section: str) -> dict[str, Key]:
    """The keys a section always has; {} for an unknown section (llm.budgets also takes keys per owner: key_spec)."""
    global _KEYS
    if _KEYS is None:
        _KEYS = _keys()
    return _KEYS[TASK_TYPE] if section.startswith(TASK_TYPE) else _KEYS.get(section, {})


def key_spec(section: str, key: str) -> Key | None:
    """A key of the section: one of keys_for, or in llm.budgets one for a user or an organization."""
    found = keys_for(section).get(key)
    if found is None and section == BUDGETS:
        if _OWNER_KEY.match(key):
            return _with(_budget_key, help=f"{key.partition('.')[0]} {key.partition('.')[2]}'s cost limit")
        if _TIER_KEY.match(key):
            return _with(keys_for(BUDGETS)["max_tier"], help=f"the highest tier for {key.partition('.')[2]}")
    return found


def check(section: str, key: str, value: Any, personal: bool) -> Any:
    """The value to store. KeyError: no such key (or not one a person sets); ValueError: it does not fit."""
    spec = key_spec(section, key)
    if spec is None or (personal and not spec.personal):
        raise KeyError(key)
    return spec.coerce(value)


def coerced(section: str, values: dict[str, Any]) -> dict[str, Any]:
    """Stored values, checked again: one that no longer fits is left out."""
    out = {}
    for key, value in values.items():
        spec = key_spec(section, key)
        try:
            if spec is not None:
                out[key] = spec.coerce(value)
        except (ValueError, TypeError):
            continue
    return out


def merged(section: str, base: dict[str, Any], everyone: dict[str, Any], mine: dict[str, Any]) -> dict[str, Any]:
    """The values in effect: the file's (`base`), everyone's over them, a person's over those."""
    out = {**base, **coerced(section, everyone)}
    for key, value in coerced(section, mine).items():
        if not key_spec(section, key).personal:
            continue
        if key in LOWER_ONLY and out.get(key) is not None and value >= out[key]:
            continue
        out[key] = value
    return out


class SettingOverrides(Protocol):
    def values(self, section: str, owner: Owner | None = None) -> dict[str, Any]:
        """One layer: everyone's (owner None) or one person's."""

    def set(self, section: str, key: str, value: Any, owner: Owner | None = None, by: str = "") -> None:
        """None removes the change: back to the layer below."""


class InMemorySettingOverrides:
    """For runtimes without a database (tests, demos): lost on restart."""

    def __init__(self):
        self._rows: dict[tuple[Owner | None, str], dict[str, Any]] = {}
        self._lock = threading.Lock()

    def values(self, section: str, owner: Owner | None = None) -> dict[str, Any]:
        with self._lock:
            return dict(self._rows.get((tuple(owner) if owner else None, section), {}))

    def set(self, section: str, key: str, value: Any, owner: Owner | None = None, by: str = "") -> None:
        with self._lock:
            row = self._rows.setdefault((tuple(owner) if owner else None, section), {})
            if value is None:
                row.pop(key, None)
            else:
                row[key] = value


def effective(store: SettingOverrides | None, section: str, base: dict[str, Any], owner: Owner | None,
              inherit: Iterable[str] = ()) -> dict[str, Any]:
    """`base` with the changes of `inherit` (sections whose file this one inherits) and of `section`,
    everyone's, then the owner's."""
    if store is None:
        return dict(base)
    out = dict(base)
    for name in (*inherit, section):
        out = merged(name, out, store.values(name), store.values(name, owner) if owner else {})
    return out
