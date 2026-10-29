"""
Tools a person switched off for their own tasks (docs/admin-page.md, Tools): one tool, every tool of a
plugin, or every tool of an MCP endpoint. Only switched-off ones are stored, as targets:

    tool:<name>         one tool, by the name the owner's tasks call it
    plugin:<id>         every tool of a plugin (`<scope>:<name>`): its tools/*.md and its MCP endpoints'
    mcp:<id>            every tool of an MCP endpoint (`<scope>:<name>`, `<scope>:<plugin>/<name>`)

ToolDispatcher.eligible leaves them out, so a switched-off tool is neither offered nor run. A switch
only narrows what the owner's own tasks may use; it never shows anyone a tool they could not see.
Control tools (update_plan, ask_user, ...) cannot be switched off: the harness needs them. Skills
are switched on the Skills tab.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Protocol

from agenticwf.agent.runtime.tools.spec import HandlerKind, ToolSpec
from agenticwf.agent.runtime.types import Owner

TARGETS = ("tool", "plugin", "mcp")


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
    return spec.handler.kind not in (HandlerKind.CONTROL, HandlerKind.SKILL)


def targets(spec: ToolSpec) -> tuple[str, ...]:
    """The switches that turn this tool off: its own, its plugin's, its endpoint's."""
    found = origin(spec)
    return (f"tool:{spec.name}", *((f"plugin:{found.plugin}",) if found.plugin else ()),
            *((f"mcp:{found.endpoint}",) if found.endpoint else ()))


def switched_off(spec: ToolSpec, off: frozenset[str] | set[str]) -> bool:
    return bool(off) and switchable(spec) and any(t in off for t in targets(spec))


class ToolPrefs(Protocol):
    def disabled_for(self, owner: Owner) -> frozenset[str]:
        """The targets the owner switched off."""

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        ...


class InMemoryToolPrefs:
    """For runtimes without a database (tests, demos): lost on restart."""

    def __init__(self):
        self._off: dict[Owner, set[str]] = {}
        self._lock = threading.Lock()

    def disabled_for(self, owner: Owner) -> frozenset[str]:
        with self._lock:
            return frozenset(self._off.get(tuple(owner), ()))

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        with self._lock:
            off = self._off.setdefault(tuple(owner), set())
            (off.discard if enabled else off.add)(target)
