"""
How a person uses tools in their own tasks (docs/admin-page.md, Tools): one tool, every tool of a
plugin, or every tool of an MCP endpoint, each `on` (the default), `ask` (every call waits for their
approval) or `off`. Only the ones not `on` are stored, by target:

    tool:<name>         one tool or skill (skill__<name>), by the name the owner's tasks call it
    plugin:<id>         every tool of a plugin (`<scope>:<name>`): its tools/*.md and its MCP endpoints'
    mcp:<id>            every tool of an MCP endpoint (`<scope>:<name>`, `<scope>:<plugin>/<name>`)

ToolDispatcher.eligible leaves out the ones `off`, so they are neither offered nor run; a call of
one in `ask` waits for the owner's approval even where policy, auto mode or a skill would allow it
(an approval for the rest of the task still counts). Where several targets apply, the strictest
wins: `off`, then `ask`. A switch only narrows what the owner's own tasks may use; it never shows
anyone a tool they could not see.
Control tools (update_plan, ask_user, ...) cannot be switched off: the harness needs them. Skills
are switched one by one (the Skills tab), never with their plugin's tools.

An admin may also change a tool's timeout for everyone (`timeouts`): it replaces the `timeout` of
the tool's spec, for calls that start after the change.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from agenticwf.agent.runtime.tools.spec import HandlerKind, ToolSpec
from agenticwf.agent.runtime.types import Owner

TARGETS = ("tool", "plugin", "mcp")
MODES = ("on", "ask", "off")
_STRICTNESS = {mode: i for i, mode in enumerate(MODES)}


@dataclass(frozen=True)
class Origin:
    """Where a tool comes from, as the Tools tab groups it."""

    kind: str           # control | core | system | plugin | mcp | database | skill
    plugin: str = ""    # the plugin's id, when a plugin provides it
    endpoint: str = ""  # the MCP endpoint's id, for MCP tools


def origin(spec: ToolSpec) -> Origin:
    source = spec.source
    if spec.handler.kind is HandlerKind.CONTROL:
        return Origin("control")
    if spec.handler.kind is HandlerKind.SKILL:
        return Origin("skill", source.plugin)
    if source.location.startswith("mcp:"):   # mcp:<source>:<endpoint id>#<tool>
        key = source.location[len("mcp:"):].partition("#")[0]
        return Origin("mcp", source.plugin, key.partition(":")[2])
    if source.kind in ("plugin", "user_plugin"):
        return Origin("plugin", source.plugin)
    return Origin(source.kind)


def switchable(spec: ToolSpec) -> bool:
    """Everything but the control tools the agent always needs; skills too (each on its own)."""
    return spec.handler.kind is not HandlerKind.CONTROL


def targets(spec: ToolSpec) -> tuple[str, ...]:
    """The switches that turn this tool off: its own, its plugin's, its endpoint's."""
    found = origin(spec)
    return (f"tool:{spec.name}", *((f"plugin:{found.plugin}",) if found.plugin else ()),
            *((f"mcp:{found.endpoint}",) if found.endpoint else ()))


def switched_off(spec: ToolSpec, off: frozenset[str] | set[str]) -> bool:
    return bool(off) and switchable(spec) and any(t in off for t in targets(spec))


def mode_of(spec: ToolSpec, modes: dict[str, str]) -> str:
    """How the owner's tasks get this tool, from all its switches: the strictest of them."""
    if not modes or not switchable(spec):
        return "on"
    return max((modes.get(t, "on") for t in targets(spec)), key=_STRICTNESS.__getitem__)


class ToolPrefs(Protocol):
    def modes_for(self, owner: Owner) -> dict[str, str]:
        """The targets the owner set to `ask` or `off`, with their mode."""

    def disabled_for(self, owner: Owner) -> frozenset[str]:
        """The targets the owner switched off."""

    def set_mode(self, owner: Owner, target: str, mode: str) -> None:
        """`on`, `ask` or `off`."""

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        ...

    def timeouts(self) -> dict[str, float]:
        """The timeouts an admin set, by tool name: deployment-wide."""

    def set_timeout(self, name: str, seconds: float | None, by: str = "") -> None:
        """None: back to the tool's own timeout."""


class InMemoryToolPrefs:
    """For runtimes without a database (tests, demos): lost on restart."""

    def __init__(self):
        self._modes: dict[Owner, dict[str, str]] = {}
        self._timeouts: dict[str, float] = {}
        self._lock = threading.Lock()

    def modes_for(self, owner: Owner) -> dict[str, str]:
        with self._lock:
            return dict(self._modes.get(tuple(owner), {}))

    def disabled_for(self, owner: Owner) -> frozenset[str]:
        return frozenset(t for t, mode in self.modes_for(owner).items() if mode == "off")

    def set_mode(self, owner: Owner, target: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        with self._lock:
            modes = self._modes.setdefault(tuple(owner), {})
            if mode == "on":
                modes.pop(target, None)
            else:
                modes[target] = mode

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        self.set_mode(owner, target, "on" if enabled else "off")

    def timeouts(self) -> dict[str, float]:
        with self._lock:
            return dict(self._timeouts)

    def set_timeout(self, name: str, seconds: float | None, by: str = "") -> None:
        with self._lock:
            if seconds is None:
                self._timeouts.pop(name, None)
            else:
                self._timeouts[name] = float(seconds)


def timeout_of(prefs, spec: ToolSpec) -> float:
    """The timeout a call of this tool gets: the admin's, else the spec's."""
    timeouts = getattr(prefs, "timeouts", None)
    return (timeouts() if timeouts else {}).get(spec.name) or spec.timeout
