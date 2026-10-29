"""
McpManager: connections to MCP endpoints, expansion, and the handler behind every MCP tool.

    expand(endpoint)   connect (or reuse the connection), tools/list, McpTools  → McpExpansion
    handler(spec)      the Handler of an MCP ToolSpec: tools/call on its endpoint
    close()            disconnect everything (stops stdio servers)

The PluginManager expands plugins' endpoints at load; McpCatalog expands database endpoints when
they are added or refreshed and stores the result. Both register their endpoints here, so a call
finds its endpoint by key even after a restart, when nothing has been expanded yet.

Users' endpoints (user_plugin, database) reach only hosts on `[mcp] allowed_hosts`, checked at
expansion and again at every call (the list may have changed).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.mcp.client import McpClient, McpError
from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint
from agenticwf.agent.runtime.mcp.expand import McpTool, expand_tools, to_result
from agenticwf.agent.runtime.net import host_allowed, is_internal, listed_by_name
from agenticwf.agent.runtime.tools.spec import Handler, ToolResult, ToolSpec

LOG = get_logger(__name__)


@dataclass(frozen=True)
class McpSettings:
    """[mcp] in agent.cfg."""

    allowed_hosts: tuple[str, ...] = ()      # hosts users' endpoints may reach; "*.example.org", "*" = any
    connect_timeout: float = 15.0            # connect, initialize, tools/list
    discovery_ttl: float = 300.0             # seconds a user plugin's tool list is reused
    discover: bool = True                    # false: never contact endpoints (offline, tests)


@dataclass
class McpExpansion:
    status: str = "pending"                  # ok | error | disabled | pending
    error: str = ""
    tools: list[McpTool] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    server_info: dict[str, Any] = field(default_factory=dict)
    checked_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"status": self.status, "error": self.error, "problems": list(self.problems),
                "server": self.server_info.get("name"), "server_version": self.server_info.get("version"),
                "checked_at": int(self.checked_at) or None}


class McpManager:
    def __init__(self, settings: McpSettings | None = None):
        self.settings = settings or McpSettings()
        self._endpoints: dict[str, McpEndpoint] = {}                 # key -> endpoint
        self._clients: dict[str, tuple[str, McpClient]] = {}         # key -> (fingerprint, client)
        self._cache: dict[str, tuple[str, McpExpansion]] = {}        # users' endpoints: key -> (fingerprint, result)
        self._expanding: set[str] = set()                            # keys being expanded in the background
        self.on_expanded: list[Callable[[str], None]] = []           # called with the key after a background expansion
        self._lock = threading.RLock()

    # --- endpoints ---------------------------------------------------------------------------------
    def register(self, endpoint: McpEndpoint) -> None:
        """Make the endpoint callable; a changed connection gets a new client on next use."""
        with self._lock:
            self._endpoints[endpoint.key] = endpoint
            entry = self._clients.get(endpoint.key)
            if entry is not None and entry[0] != endpoint.fingerprint():
                self._clients.pop(endpoint.key)[1].close()

    def forget(self, keys: Iterable[str]) -> None:
        with self._lock:
            for key in keys:
                self._endpoints.pop(key, None)
                self._cache.pop(key, None)
                entry = self._clients.pop(key, None)
                if entry is not None:
                    entry[1].close()

    def keys(self, prefix: str = "") -> list[str]:
        with self._lock:
            return [k for k in self._endpoints if k.startswith(prefix)]

    def allowed(self, endpoint: McpEndpoint) -> str | None:
        """None if the endpoint may be contacted, else why not."""
        if endpoint.trusted:
            return None
        if endpoint.transport != "http":
            return "only operators' plugins can start processes on the server"
        if not host_allowed(endpoint.host, self.settings.allowed_hosts):
            return f"host {endpoint.host!r} is not on the allow-list ([mcp] allowed_hosts)"
        return None

    def _guard(self, endpoint: McpEndpoint):
        """Users' endpoints: the address connected to must not be internal, unless an operator listed
        that host by name (a wildcard, including "*", never lets one reach an internal address)."""
        if endpoint.trusted or listed_by_name(endpoint.host, self.settings.allowed_hosts):
            return None

        def guard(address: str) -> str | None:
            return f"{endpoint.host} is an internal address; an operator must list it by name" if is_internal(address) else None

        return guard

    def _client(self, endpoint: McpEndpoint) -> McpClient:
        with self._lock:
            entry = self._clients.get(endpoint.key)
            if entry is None or entry[0] != endpoint.fingerprint():
                if entry is not None:
                    entry[1].close()
                entry = (endpoint.fingerprint(), McpClient(endpoint, self.settings.connect_timeout, self._guard(endpoint)))
                self._clients[endpoint.key] = entry
            return entry[1]

    # --- expansion ---------------------------------------------------------------------------------
    def expand(self, endpoint: McpEndpoint, taken: set[str] = frozenset(), cache: bool = False,
               wait: bool = True) -> McpExpansion:
        """List the endpoint's tools. With `cache` (users' plugins, read whenever a catalogue is
        built), a result is reused for `discovery_ttl` seconds, 60 after a failure. With
        `wait=False`, a missing or stale result is refreshed in the background and the last one (or
        `pending`) is returned at once: the caller may be on the event loop."""
        self.register(endpoint)
        if not endpoint.enabled:
            return McpExpansion(status="disabled")
        refusal = self.allowed(endpoint)
        if refusal:
            return McpExpansion(status="error", error=refusal, checked_at=time.time())
        if not self.settings.discover:
            return McpExpansion(status="pending", error="discovery is off ([mcp] discover)")
        hit = self._cache.get(endpoint.key) if cache else None
        if hit is not None and hit[0] != endpoint.fingerprint():
            hit = None
        if hit is not None:
            ttl = self.settings.discovery_ttl if hit[1].status == "ok" else 60.0
            if hit[1].checked_at + ttl > time.time():
                return hit[1]
        if not wait:
            self._expand_later(endpoint, taken)
            return hit[1] if hit is not None else McpExpansion(status="pending", error="asking the endpoint for its tools")
        return self._expand_now(endpoint, taken, cache)

    def _expand_now(self, endpoint: McpEndpoint, taken: set[str], cache: bool) -> McpExpansion:
        client = self._client(endpoint)
        try:
            listed = client.list_tools()
            tools, problems = expand_tools(endpoint, listed, taken)
            result = McpExpansion(status="ok", tools=tools, problems=problems, server_info=dict(client.server_info),
                                  checked_at=time.time())
        except McpError as exc:
            result = McpExpansion(status="error", error=str(exc), checked_at=time.time())
        except Exception as exc:   # whatever a misbehaving server's reply causes: never the caller's problem
            LOG.exception("MCP endpoint %s: expansion failed", endpoint.key)
            result = McpExpansion(status="error", error=f"unusable reply ({type(exc).__name__})", checked_at=time.time())
        if cache:
            self._cache[endpoint.key] = (endpoint.fingerprint(), result)
        return result

    def _expand_later(self, endpoint: McpEndpoint, taken: set[str]) -> None:
        """One background expansion per endpoint at a time."""
        with self._lock:
            if endpoint.key in self._expanding:
                return
            self._expanding.add(endpoint.key)

        def run() -> None:
            try:
                self._expand_now(endpoint, set(taken), cache=True)
            finally:
                with self._lock:
                    self._expanding.discard(endpoint.key)
            for listener in self.on_expanded:
                listener(endpoint.key)

        threading.Thread(target=run, name=f"mcp-expand-{endpoint.name}", daemon=True).start()

    def wait_idle(self, timeout: float = 30.0) -> bool:
        """Wait for background expansions to finish (tests, shutdown). True if none is left."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._lock:
                if not self._expanding:
                    return True
            time.sleep(0.02)
        return False

    # --- calls -------------------------------------------------------------------------------------
    def handler(self, spec: ToolSpec) -> Handler:
        """The endpoint comes from the spec's provenance (`mcp:<key>#<tool>`), which only
        agenticwf.agent.runtime.mcp writes: a hand-written tool definition cannot point at someone else's endpoint.
        The spec must also come from where the endpoint does, and be one of its exposed tools."""
        endpoint, tool = self._bound(spec)
        if endpoint is None or tool is None:
            return _unavailable(f"{spec.name} is not bound to an MCP endpoint it may use")
        refusal = self.allowed(endpoint)
        if refusal or not endpoint.enabled:
            return _unavailable(refusal or f"MCP endpoint {endpoint.name!r} is disabled")
        client = self._client(endpoint)

        def handler(call, arguments) -> ToolResult:
            remaining = (call.deadline - time.time()) if call.deadline else spec.timeout
            try:
                return to_result(client.call_tool(tool, arguments, timeout=max(1.0, remaining)))
            except McpError as exc:
                return ToolResult.error("mcp_error", f"{spec.name}: {exc}")

        return handler

    def _bound(self, spec: ToolSpec) -> tuple[McpEndpoint | None, str | None]:
        location = spec.source.location
        if not location.startswith("mcp:") or "#" not in location:
            return None, None
        key, tool = location[4:].rsplit("#", 1)
        with self._lock:
            endpoint = self._endpoints.get(key)
        if endpoint is None or spec.source.kind != endpoint.source:
            return None, None
        if endpoint.source != "plugin" and spec.source.owner != endpoint.owner:
            return None, None
        if endpoint.tools and tool not in endpoint.tools:
            return None, None
        return endpoint, tool

    def close(self) -> None:
        with self._lock:
            for _, client in self._clients.values():
                client.close()
            self._clients.clear()


def _unavailable(message: str) -> Handler:
    def handler(call, arguments) -> ToolResult:
        return ToolResult.error("tool_unavailable", message)
    return handler
