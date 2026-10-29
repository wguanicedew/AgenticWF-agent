"""
PluginManager: finds plugins and reads what each provides.

    <plugin dir>/<plugin>/                  system plugins ([tools] plugin_dirs), installed by operators
    <user sandbox>/plugins/<plugin>/        a user's own plugins, visible to that user only

    <plugin>/
      plugin.yaml                           optional: version, description, enabled, access, mcp_servers
      .mcp.json                             optional: MCP endpoints in the common mcpServers format
      task-types/<type>/                    task types, named <plugin>.<type>
      tools/<tool>.md                       tools, named <plugin>__<tool>
      skills/<skill>/SKILL.md               skills, named <plugin>.<skill> (agenticwf.agent.runtime.skills)
      crons/<cron>/CRON.md                  crons, named <plugin>.<cron> (agenticwf.agent.runtime.cron)
      handlers.py                           builtin handlers, refs <plugin>.<key> (system plugins only)

A plugin can provide any mix of these. The manager does not decide permissions or select tools:
it hands the ToolRegistry the plugins' tools (Markdown tools, and the tools their MCP endpoints
expand into through agenticwf.agent.runtime.mcp), the TaskTypeResolver their task-type directories, and the
admin page the catalogue.

Ownership: a system plugin's tools and MCP tools are shared with everyone, unless its `access`
(plugin.yaml, a tool's front matter, an endpoint entry) narrows them to some organizations,
users or scopes. A user's plugin is visible to that user only.

System plugins are read at load() (start and reload), and their MCP endpoints expanded then. A
user's plugins are read when that user's catalogue is needed; McpManager caches their
expansions. Trust follows docs/agent-runtime.md §5: a user's plugin never runs code in the agent
(no handlers.py, no stdio endpoints, no builtin tools), never reads the agent's environment,
reaches only allow-listed hosts, and its MCP tools always ask for approval.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.context.task_types import INSTRUCTION_FILES
from agenticwf.agent.runtime.context.task_types import NAME_RE as TYPE_RE
from agenticwf.agent.runtime.mcp.endpoint import PREFIX_MAX, McpEndpoint, prefix_ok
from agenticwf.agent.runtime.mcp.manager import McpExpansion, McpManager
from agenticwf.agent.runtime.cron.definition import Cron, read_crons_dir
from agenticwf.agent.runtime.plugins.manifest import ManifestError, PluginManifest, read_manifest
from agenticwf.agent.runtime.skills.skill import Skill, read_skills_dir
from agenticwf.agent.runtime.tools.loader import load_plugin_tools
from agenticwf.agent.runtime.tools.spec import Handler, HandlerKind, HandlerRef, ToolSource, ToolSpec
from agenticwf.agent.runtime.types import SYSTEM_SCOPE, Owner, scope_of

LOG = get_logger(__name__)
HANDLERS_FILE = "handlers.py"


@dataclass(frozen=True)
class PluginSettings:
    """[plugins] in agent.cfg (MCP settings are in [mcp]: agenticwf.agent.runtime.mcp.manager.McpSettings)."""

    disabled: tuple[str, ...] = ()                 # system plugins switched off by the operator
    user_ttl: float = 5.0                          # seconds a user's plugins are reused before reading them again


@dataclass
class PluginEndpoint:
    """An MCP endpoint a plugin installs, and what expanding it gave."""

    endpoint: McpEndpoint
    expansion: McpExpansion = field(default_factory=McpExpansion)

    def specs(self) -> list[ToolSpec]:
        return [t.spec(self.endpoint) for t in self.expansion.tools]

    def to_dict(self, owner: Owner | None = None) -> dict[str, Any]:
        tools = [s for s in self.specs() if owner is None or s.source.visible_to(owner)]
        return {**self.endpoint.to_dict(), **self.expansion.to_dict(), "tools": [_tool_view(s) for s in tools]}


@dataclass
class Plugin:
    """One plugin directory and what it provides."""

    name: str
    root: Path
    source: str                                    # system | user
    manifest: PluginManifest
    enabled: bool = True
    owner: Owner | None = None
    task_types: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)   # (<plugin>.<type>, files)
    tools: list[ToolSpec] = field(default_factory=list)                           # tools/*.md
    mcp_endpoints: list[PluginEndpoint] = field(default_factory=list)
    skills: list[Skill] = field(default_factory=list)                              # skills/<skill>/SKILL.md
    crons: list[Cron] = field(default_factory=list)                                # crons/<cron>/CRON.md
    handlers: dict[str, Handler] = field(default_factory=dict)                    # <plugin>.<key> -> callable
    errors: list[str] = field(default_factory=list)

    @property
    def trusted(self) -> bool:
        return self.source == "system"

    @property
    def scope(self) -> str:
        return SYSTEM_SCOPE if self.trusted else scope_of(self.owner)

    @property
    def id(self) -> str:
        """`<scope>:<name>`: a system plugin and a user's plugin may share a name without conflict."""
        return f"{self.scope}:{self.name}"

    @property
    def kinds(self) -> list[str]:
        return [k for k, present in (("task_types", self.task_types), ("tools", self.tools),
                                     ("mcp", self.mcp_endpoints), ("skills", self.skills),
                                     ("crons", self.crons)) if present]

    def all_tools(self) -> list[ToolSpec]:
        """The tools this plugin contributes: none when disabled."""
        if not self.enabled:
            return []
        return self.tools + [s for e in self.mcp_endpoints for s in e.specs()]

    def visible_to(self, owner: Owner | None) -> bool:
        """Listed for `owner`: its own plugins, and system plugins whose access lets them in."""
        if owner is None:
            return True
        return self.owner == owner if not self.trusted else self.manifest.access.allows(owner)

    def to_dict(self, owner: Owner | None = None) -> dict[str, Any]:
        """The admin view for `owner`: only the tools and endpoints whose own access lets them in."""
        m = self.manifest
        tools = [s for s in self.tools if owner is None or s.source.visible_to(owner)]
        endpoints = [e for e in self.mcp_endpoints if owner is None or e.endpoint.visible_to(owner)]
        skills = [k for k in self.skills if owner is None or k.visible_to(owner)]
        kinds = [k for k, present in (("task_types", self.task_types), ("tools", tools), ("mcp", endpoints),
                                      ("skills", skills), ("crons", self.crons)) if present]
        return {"id": self.id, "name": self.name, "scope": self.scope, "source": self.source,
                "version": m.version, "description": m.description,
                "author": m.author, "enabled": self.enabled, "kinds": kinds, "access": m.access.to_dict(),
                "visibility": "all" if self.trusted else "user",
                "task_types": [{"name": n, "files": list(f)} for n, f in self.task_types],
                "tools": [_tool_view(s) for s in tools],
                "mcp_endpoints": [e.to_dict(owner) for e in endpoints],
                "skills": [{"id": k.id, "name": k.qualified, "tool": k.tool_name, "description": k.description}
                           for k in skills],
                "crons": [{"id": c.id, "name": c.qualified, "description": c.description} for c in self.crons],
                "handlers": sorted(self.handlers), "errors": list(self.errors)}


def _tool_view(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "summary": spec.summary, "handler": spec.handler.kind.value,
            "effects": spec.effects.value, "approval": spec.approval.value,
            "visibility": spec.source.effective_visibility, "access": spec.source.access.to_dict(),
            "scopes": sorted(spec.scopes)}


class PluginManager:
    def __init__(self, plugin_dirs: Iterable[Path] = (), user_sandbox: Callable[[Owner], Path | None] | None = None,
                 settings: PluginSettings | None = None, mcp: McpManager | None = None):
        self.plugin_dirs = [Path(p) for p in plugin_dirs]
        self.user_sandbox = user_sandbox
        self.settings = settings or PluginSettings()
        self.mcp = mcp or McpManager()
        self._system: dict[str, Plugin] | None = None
        self._users: dict[Owner, tuple[float, list[Plugin]]] = {}   # owner -> (read at, their plugins)
        self._lock = threading.Lock()
        self.mcp.on_expanded.append(self._user_endpoint_expanded)
        self.errors: list[str] = []

    # --- system plugins ----------------------------------------------------------------------------
    def load(self) -> list[str]:
        """(Re)read the system plugins and expand their MCP endpoints. Returns the errors (also
        logged). Connections of endpoints whose settings did not change are kept."""
        plugins: dict[str, Plugin] = {}
        errors: list[str] = []
        for directory in self.plugin_dirs:
            for root in _plugin_roots(directory, errors):
                if root.name in plugins:
                    errors.append(f"{root}: plugin {root.name!r} already loaded from {plugins[root.name].root}")
                    continue
                try:
                    plugin = self._read(root, "system", None)
                except ManifestError as exc:
                    errors.append(str(exc))
                    continue
                except Exception as exc:   # a broken plugin never stops the others (or the agent)
                    LOG.exception("plugin %s could not be read", root)
                    errors.append(f"{root}: {type(exc).__name__}: {exc}")
                    continue
                if root.name in self.settings.disabled:
                    plugin.enabled = False
                plugins[root.name] = plugin
        used = set()
        for plugin in plugins.values():
            for item in plugin.mcp_endpoints:
                used.add(item.endpoint.key)
                self._expand(plugin, item, cache=False)
            errors.extend(f"{plugin.root}: {e}" for e in plugin.errors)
        self.mcp.forget(k for k in self.mcp.keys("plugin:") if k not in used)
        with self._lock:
            self._system, self.errors = plugins, errors
            self._users.clear()   # their clashes with system names may have changed
        for error in errors:
            LOG.error("plugin: %s", error)
        return errors

    def _system_plugins(self) -> dict[str, Plugin]:
        if self._system is None:
            self.load()
        return self._system or {}

    # --- a user's plugins --------------------------------------------------------------------------
    def _user_plugins(self, owner: Owner) -> list[Plugin]:
        """The owner's plugins, read again at most every `user_ttl` seconds. This runs whenever a
        task's tool catalogue is built, on the event loop: it never waits for an MCP endpoint (they
        are expanded in the background; a new one shows its tools once it has answered)."""
        sandbox = self.user_sandbox(owner) if self.user_sandbox else None
        if sandbox is None:
            return []
        with self._lock:
            cached = self._users.get(owner)
        if cached and cached[0] + self.settings.user_ttl > time.monotonic():
            return cached[1]
        self._system_plugins()   # loaded first
        plugins = []
        for root in _plugin_roots(sandbox / "plugins", []):
            try:
                plugin = self._read(root, "user", owner)
            except Exception as exc:   # ManifestError, or anything hand-written files can cause
                LOG.warning("user plugin %s skipped: %s", root.name, exc)
                continue
            # Same name as a system plugin: no conflict. Ids are scoped (`<scope>:<name>`), task types
            # too, and the registry qualifies tool names that clash.
            for item in plugin.mcp_endpoints:
                self._expand(plugin, item, cache=True)
            plugins.append(plugin)
        with self._lock:
            self._users[owner] = (time.monotonic(), plugins)
        return plugins

    def _user_endpoint_expanded(self, key: str) -> None:
        """A user's endpoint answered in the background: read their plugins again next time."""
        if key.startswith("user_plugin:"):
            with self._lock:
                self._users.clear()

    def _expand(self, plugin: Plugin, item: PluginEndpoint, cache: bool) -> None:
        if not plugin.enabled:
            self.mcp.register(item.endpoint)
            item.expansion = McpExpansion(status="disabled")
            return
        taken = {s.name for s in plugin.tools} | {s.name for e in plugin.mcp_endpoints if e is not item for s in e.specs()}
        # Users' endpoints (read on every catalogue) never block: expanded in the background.
        item.expansion = self.mcp.expand(item.endpoint, taken=taken, cache=cache, wait=not cache)
        name = item.endpoint.name
        if item.expansion.error:
            plugin.errors.append(f"MCP endpoint {name!r}: {item.expansion.error}")
        plugin.errors.extend(f"MCP endpoint {name!r}: {p}" for p in item.expansion.problems)

    # --- queries -----------------------------------------------------------------------------------
    def plugins(self, owner: Owner | None = None) -> list[Plugin]:
        """System plugins `owner` may use, then the owner's own (scoped: a name may repeat). None:
        every system plugin."""
        system = self._system_plugins()
        result = [p for p in system.values() if p.visible_to(owner)]
        if owner is not None:
            result.extend(self._user_plugins(owner))
        return result

    def get(self, name: str, owner: Owner | None = None) -> Plugin | None:
        return next((p for p in self.plugins(owner) if p.name == name), None)

    def tools(self) -> list[ToolSpec]:
        """Tools of the enabled system plugins (Markdown and MCP); the registry filters visibility."""
        return [s for p in self._system_plugins().values() for s in p.all_tools()]

    def user_tools(self, owner: Owner) -> list[ToolSpec]:
        """Tools of the owner's own enabled plugins."""
        return [s for plugin in self._user_plugins(owner) for s in plugin.all_tools()]

    def endpoints(self, owner: Owner | None = None) -> list[tuple[Plugin, PluginEndpoint]]:
        """MCP endpoints installed by plugins that `owner` may see."""
        return [(p, e) for p in self.plugins(owner) for e in p.mcp_endpoints
                if owner is None or e.endpoint.visible_to(owner)]

    def task_type_dirs(self, owner: Owner | None = None) -> dict[str, tuple[Path, bool]]:
        """`<scope>:<plugin>.<type>` -> (directory, trusted), for TaskTypeResolver: a system plugin's
        and a user's plugin may provide types of the same name without conflict."""
        found: dict[str, tuple[Path, bool]] = {}
        for plugin in self.plugins(owner):
            if plugin.enabled:
                for name, _ in plugin.task_types:
                    found[f"{plugin.scope}:{name}"] = (plugin.root / "task-types" / name.split(".", 1)[1], plugin.trusted)
        return found

    # --- handlers ----------------------------------------------------------------------------------
    def crons(self, owner: Owner | None = None) -> list[Cron]:
        """Crons of the enabled plugins `owner` may see (every system plugin's for None)."""
        return [c for p in self.plugins(owner) if p.enabled for c in p.crons]

    def skills(self, owner: Owner | None = None) -> list[Skill]:
        """Skills of the enabled plugins `owner` may use (every system plugin's for None)."""
        return [k for p in self.plugins(owner) if p.enabled for k in p.skills]

    def builtin_handler(self, ref: str) -> Handler | None:
        """A handler from an enabled system plugin's handlers.py."""
        plugin = self._system_plugins().get(ref.split(".", 1)[0])
        return plugin.handlers.get(ref) if plugin and plugin.enabled else None

    # --- reading a plugin --------------------------------------------------------------------------
    def _read(self, root: Path, source: str, owner: Owner | None) -> Plugin:
        """Read one plugin directory (no MCP connection yet). ManifestError: the plugin is unusable."""
        trusted = source == "system"
        manifest = read_manifest(root, trusted, owner)
        plugin = Plugin(root.name, root, source, manifest, enabled=manifest.enabled, owner=owner,
                        errors=list(manifest.errors))
        types_root = root / "task-types"
        if types_root.is_dir():
            for directory in sorted(p for p in types_root.iterdir() if p.is_dir() and TYPE_RE.match(p.name)):
                files = tuple(n for n in INSTRUCTION_FILES if (directory / n).is_file())
                plugin.task_types.append((f"{root.name}.{directory.name}", files))
        tool_source = (ToolSource("plugin", access=manifest.access, plugin=plugin.id) if trusted
                       else ToolSource("user_plugin", owner=owner, plugin=plugin.id))
        plugin.tools, errors = load_plugin_tools(root, tool_source)
        plugin.errors.extend(errors)
        if (root / HANDLERS_FILE).is_file():
            if trusted:
                plugin.handlers = self._import_handlers(plugin, plugin.errors)
            else:
                plugin.errors.append(f"{HANDLERS_FILE} ignored: your own plugins cannot run Python in the agent")
        plugin.skills, errors = read_skills_dir(root / "skills", "plugin" if trusted else "user", owner=owner,
                                                plugin=root.name, inherited=manifest.access)
        plugin.errors.extend(errors)
        plugin.crons, errors = read_crons_dir(root / "crons", "plugin" if trusted else "user", owner=owner,
                                              plugin=root.name, inherited=manifest.access)
        plugin.errors.extend(errors)
        plugin.mcp_endpoints = [PluginEndpoint(e) for e in manifest.mcp_endpoints]
        plugin.tools = [t for t in (self._bind_mcp(plugin, spec) for spec in plugin.tools) if t is not None]
        return plugin

    @staticmethod
    def _bind_mcp(plugin: Plugin, spec: ToolSpec) -> ToolSpec | None:
        """A hand-written tools/*.md with `handler: {kind: mcp, endpoint: <one of this plugin's
        endpoints>, tool: <its MCP name>}` calls that endpoint, and no other."""
        if spec.handler.kind is not HandlerKind.MCP:
            return spec
        settings = spec.handler.settings
        name, tool = settings.get("endpoint") or settings.get("server"), settings.get("tool")
        endpoint = next((e.endpoint for e in plugin.mcp_endpoints if e.endpoint.name == name), None)
        if endpoint is None or not isinstance(tool, str) or not tool:
            plugin.errors.append(f"{spec.source.location}: an mcp tool needs handler.endpoint (one of this "
                                 f"plugin's MCP endpoints) and handler.tool")
            return None
        return dataclasses.replace(
            spec, handler=HandlerRef(HandlerKind.MCP, {"endpoint": endpoint.name, "tool": tool}),
            source=dataclasses.replace(spec.source, location=f"mcp:{endpoint.key}#{tool}"),
            timeout=min(spec.timeout, endpoint.timeout))

    @staticmethod
    def _import_handlers(plugin: Plugin, errors: list[str]) -> dict[str, Handler]:
        """handlers.py defines HANDLERS = {"key": callable}; they are registered as <plugin>.<key>."""
        module_name = f"agenticwf_plugins.{plugin.name}"
        try:
            spec = importlib.util.spec_from_file_location(module_name, plugin.root / HANDLERS_FILE)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
        except Exception as exc:   # anything the module raises while importing
            sys.modules.pop(module_name, None)
            errors.append(f"{HANDLERS_FILE}: {type(exc).__name__}: {exc}")
            return {}
        handlers = getattr(module, "HANDLERS", None)
        if not isinstance(handlers, dict) or not all(callable(h) for h in handlers.values()):
            errors.append(f"{HANDLERS_FILE}: HANDLERS must be a dict of callables")
            return {}
        return {f"{plugin.name}.{key}": fn for key, fn in handlers.items()}


def _plugin_roots(directory: Path, errors: list[str]) -> list[Path]:
    if not directory.is_dir():
        return []
    roots = []
    for path in sorted(p for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if prefix_ok(path.name):
            roots.append(path)
        else:
            errors.append(f"{path}: invalid plugin name (lowercase letters, digits and single underscores, "
                          f"at most {PREFIX_MAX} characters)")
    return roots
