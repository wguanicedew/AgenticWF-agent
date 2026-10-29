"""
McpEndpoint: one MCP server the agent can reach, wherever it was defined, and who may use it.

    source        defined in                                    owner      default visibility
    plugin        a system plugin's plugin.yaml / .mcp.json     operators  all (narrow with `access`)
    user_plugin   a plugin in a user's sandbox                  that user  user
    database      added on the admin page (api/v1/mcp)          that user  user (or organization; all: managers)

An endpoint is *expanded* into tools (agenticwf.agent.runtime.mcp.expand): each tool it lists becomes a tool named
`<prefix>__<tool>`, the prefix being the plugin's name or the database endpoint's name.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from agenticwf.agent.runtime.tools.spec import NAME_RE, VISIBILITY, Access, Approval, Effects
from agenticwf.agent.runtime.types import SYSTEM_SCOPE, Owner, scope_of

SERVER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")   # names inside a plugin (.mcp.json style)
_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
SOURCES = ("plugin", "user_plugin", "database")
PREFIX_MAX = 40   # plugin and endpoint names prefix tool names (<prefix>__<tool>, at most 64)


def prefix_ok(name: str) -> bool:
    """A tool-name prefix: a tool name, without "__" (so <prefix>__<tool> names exactly one owner), short
    enough to leave room for the tool."""
    return bool(NAME_RE.match(name)) and "__" not in name and len(name) <= PREFIX_MAX


class EndpointError(ValueError):
    pass


@dataclass(frozen=True)
class McpEndpoint:
    name: str                                          # plugin: server name; database: endpoint id
    transport: str                                     # stdio | http
    command: str | None = None                         # stdio
    args: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict, hash=False, repr=False)
    cwd: Path | None = None
    url: str | None = None                             # http
    headers: dict[str, str] = field(default_factory=dict, hash=False, repr=False)   # may hold secrets
    tools: tuple[str, ...] = ()                        # only expose these tools; empty = all
    effects: str | None = None                         # default for the endpoint's tools
    approval: str | None = None
    timeout: float = 60.0
    # where it comes from and who may use it
    source: str = "plugin"
    plugin: str | None = None                          # plugin / user_plugin: the plugin's name
    owner: Owner | None = None                         # user_plugin / database
    visibility: str = ""                               # "" = the source's default
    access: Access = Access()
    label: str = ""
    description: str = ""
    enabled: bool = True
    display: str = field(default="", compare=False)    # what to show: the definition before ${VAR} expansion

    @property
    def trusted(self) -> bool:
        """Installed by operators: may start processes, read the environment, reach any host."""
        return self.source == "plugin"

    @property
    def prefix(self) -> str:
        """Tools are named <prefix>__<tool>."""
        return self.plugin if self.source != "database" else self.name

    @property
    def scope(self) -> str:
        """`system` for what operators install, else the owner's scope (agenticwf.agent.runtime.types.scope_of)."""
        return SYSTEM_SCOPE if self.source == "plugin" else scope_of(self.owner)

    @property
    def id(self) -> str:
        """`<scope>:<name>` (database) or `<scope>:<plugin>/<name>` (plugins): unique across owners, so
        two people can both have an endpoint named `github`."""
        return f"{self.scope}:{self.name}" if self.source == "database" else f"{self.scope}:{self.plugin}/{self.name}"

    @property
    def key(self) -> str:
        """Identifies the endpoint (and its connection) across reloads."""
        return f"{self.source}:{self.id}"

    @property
    def effective_visibility(self) -> str:
        return self.visibility or ("all" if self.source == "plugin" else "user")

    def visible_to(self, owner: Owner) -> bool:
        if not self.access.allows(owner):
            return False
        visibility = self.effective_visibility
        if visibility == "all":
            return True
        if visibility == "organization":
            return self.owner is not None and self.owner[1] is not None and self.owner[1] == owner[1]
        return self.owner == owner

    def owned_by(self, owner: Owner | None) -> bool:
        return self.source == "database" and owner is not None and self.owner == owner

    @property
    def host(self) -> str:
        return (urlsplit(self.url or "").hostname or "").lower()

    @property
    def target(self) -> str:
        """What anyone who may see the endpoint is shown. Never expanded secrets: a plugin endpoint
        shows its definition as written (`${VAR}` unexpanded, no env values), any other endpoint
        only its scheme, host and port (its path may carry a token; the owner sees `url`)."""
        if self.display:
            return self.display
        if self.transport == "stdio":
            return self.command or ""
        return _origin(self.url)

    def fingerprint(self) -> str:
        """Changes when anything about the connection changes (a new client is needed)."""
        connection = (self.transport, self.command, self.args, sorted(self.env.items()), str(self.cwd), self.url,
                      sorted(self.headers.items()), self.timeout)
        return hashlib.sha256(repr(connection).encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """For the admin page: never env values or header values."""
        return {"id": self.id, "name": self.name, "scope": self.scope, "label": self.label or self.name,
                "description": self.description,
                "transport": self.transport, "target": self.target, "source": self.source, "plugin": self.plugin,
                "visibility": self.effective_visibility, "access": self.access.to_dict(), "enabled": self.enabled,
                "tools_filter": list(self.tools), "effects": self.effects, "approval": self.approval,
                "timeout": self.timeout, "header_names": sorted(self.headers)}


def _origin(url: str | None) -> str:
    parts = urlsplit(url or "")
    return f"{parts.scheme}://{parts.hostname or ''}{f':{parts.port}' if parts.port else ''}"


def _display(kind: str, settings: dict[str, Any]) -> str:
    """The endpoint as its definition writes it, before ${VAR} expansion: no query, no credentials."""
    if kind == "stdio":
        return " ".join(str(v) for v in [settings.get("command") or "", *(settings.get("args") or ())]).strip()
    parts = urlsplit(str(settings.get("url") or ""))
    return _origin(str(settings.get("url") or "")) + parts.path


def parse_endpoint(name: str, settings: Any, *, source: str, plugin: str | None = None, root: Path | None = None,
                   owner: Owner | None = None, inherited: Access = Access()) -> McpEndpoint:
    """An endpoint from a plugin's mcp_servers / mcpServers entry. EndpointError if unusable."""
    if not SERVER_RE.match(name):
        raise EndpointError("invalid name")
    if not isinstance(settings, dict):
        raise EndpointError("settings must be a mapping")
    _check_shapes(settings)
    trusted = source == "plugin"
    kind = settings.get("transport") or settings.get("type") or ("http" if settings.get("url") else "stdio")
    if kind == "streamable-http":
        kind = "http"
    if kind not in ("stdio", "http"):
        raise EndpointError(f"unsupported transport {kind!r} (stdio or http)")
    if kind == "stdio" and not trusted:
        raise EndpointError("only operators' plugins can start processes on the server; use an http endpoint")
    if not settings.get("command" if kind == "stdio" else "url"):
        raise EndpointError(f"a {kind} endpoint needs a {'command' if kind == 'stdio' else 'url'}")

    def expand(value: Any) -> str:
        text = str(value)
        if not trusted and _VAR.search(text):
            raise EndpointError("only operators' plugins can read the agent's environment (${...})")
        missing = [v for v in _VAR.findall(text) if v not in os.environ]
        if missing:
            raise EndpointError(f"environment variable {', '.join(missing)} is not set")
        return _VAR.sub(lambda m: os.environ[m.group(1)], text)

    try:
        access = inherited.merge(Access.from_mapping(settings.get("access")))
        endpoint = McpEndpoint(
            name=name,
            transport=kind,
            command=expand(settings["command"]) if settings.get("command") else None,
            args=tuple(expand(a) for a in settings.get("args") or ()),
            env={str(k): expand(v) for k, v in (settings.get("env") or {}).items()},
            cwd=root,
            url=expand(settings["url"]) if settings.get("url") else None,
            headers={str(k): expand(v) for k, v in (settings.get("headers") or {}).items()},
            tools=tuple(str(t) for t in settings.get("tools") or ()),
            effects=settings.get("effects"),
            approval=settings.get("approval"),
            timeout=float(settings.get("timeout", 60)),
            source=source, plugin=plugin, owner=owner,
            visibility=settings.get("visibility") or "",
            access=access,
            description=str(settings.get("description") or ""),
            display=_display(kind, settings),
        )
    except (TypeError, ValueError) as exc:
        raise EndpointError(str(exc)) from exc
    check(endpoint)
    return endpoint


_SHAPES = {"transport": str, "type": str, "command": str, "url": str, "description": str, "visibility": str,
           "effects": str, "approval": str, "args": (list, tuple), "tools": (list, tuple), "env": dict,
           "headers": dict, "access": dict, "timeout": (int, float)}


def _check_shapes(settings: dict[str, Any]) -> None:
    """Hand-written YAML and JSON: wrong types are reported, never raised as anything else."""
    for key, kind in _SHAPES.items():
        value = settings.get(key)
        if value is not None and (not isinstance(value, kind) or isinstance(value, bool)):
            raise EndpointError(f"{key} has the wrong type ({type(value).__name__})")
    for key in ("args", "tools"):
        if not all(isinstance(v, (str, int, float)) for v in settings.get(key) or ()):
            raise EndpointError(f"{key} must be a list of strings")
    for key in ("env", "headers"):
        if not all(isinstance(v, (str, int, float)) for v in (settings.get(key) or {}).values()):
            raise EndpointError(f"{key} values must be strings")


def check(endpoint: McpEndpoint) -> None:
    """Rules every endpoint follows, whatever defined it. EndpointError if one is broken."""
    if endpoint.source not in SOURCES:
        raise EndpointError(f"unknown source {endpoint.source!r}")
    if endpoint.source == "database" and not prefix_ok(endpoint.name):
        raise EndpointError("name: lowercase letters, digits and single underscores, starting with a letter "
                            f"(at most {PREFIX_MAX}); it prefixes its tools' names (<name>__<tool>)")
    if endpoint.transport not in ("stdio", "http"):
        raise EndpointError("transport must be stdio or http")
    if endpoint.transport == "stdio" and not endpoint.trusted:
        raise EndpointError("only operators' plugins can start processes on the server; use an http endpoint")
    if endpoint.transport == "http" and urlsplit(endpoint.url or "").scheme not in ("http", "https"):
        raise EndpointError("url must be http(s)")
    for value, enum, what in ((endpoint.effects, Effects, "effects"), (endpoint.approval, Approval, "approval")):
        if value is not None and value not in {e.value for e in enum}:
            raise EndpointError(f"invalid {what} {value!r}")
    if endpoint.visibility and endpoint.visibility not in VISIBILITY:
        raise EndpointError(f"visibility must be one of {', '.join(VISIBILITY)}")
    if endpoint.effective_visibility == "organization" and not (endpoint.owner and endpoint.owner[1]):
        raise EndpointError("visibility 'organization' needs an organization-scoped owner")
    if endpoint.source == "user_plugin" and endpoint.effective_visibility != "user":
        raise EndpointError("a plugin in your sandbox is yours only (visibility: user)")
    if not 0 < endpoint.timeout <= 3600:
        raise EndpointError("timeout must be between 0 and 3600 seconds")
