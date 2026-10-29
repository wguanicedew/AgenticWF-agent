"""
McpCatalog: MCP endpoints users add while the system runs, and the tools expanded from them
(tables agent_mcp_endpoints, agent_mcp_tools; agenticwf.agent.runtime.mcp.db_store).

    put(endpoint)        add or change an endpoint, expand it, store its tools
    refresh(name)        expand again: new tools appear, removed ones go, settings of kept ones stay
    delete(name)         remove the endpoint and its tools
    set_tool(name, …)    a tool's own settings: enabled, visibility, approval (ask | deny)
    tools_for(owner)     the ToolRegistry's database source (DatabaseToolSource)

Endpoints and tools each have an owner and a visibility: `user` (the default: only the owner),
`organization` (the owner's organization), or `all` (managers only, checked by the API). A tool
starts with its endpoint's owner and visibility; its owner can then share or hide it on its own.
Only the owner changes an endpoint or its tools. Header values (tokens) are stored encrypted and
never returned.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.mcp.endpoint import EndpointError, McpEndpoint, check
from agenticwf.agent.runtime.mcp.expand import McpTool
from agenticwf.agent.runtime.mcp.manager import McpExpansion, McpManager
from agenticwf.agent.runtime.tools.spec import VISIBILITY, ToolSpec
from agenticwf.agent.runtime.types import Owner

LOG = get_logger(__name__)


@dataclass(frozen=True)
class EndpointRecord:
    """A database endpoint and the outcome of its last expansion."""

    endpoint: McpEndpoint
    status: str = "pending"                  # ok | error | disabled | pending
    error: str = ""
    problems: tuple[str, ...] = ()
    server: str | None = None
    server_version: str | None = None
    checked_at: float | None = None
    header_hints: dict[str, str] = field(default_factory=dict, hash=False)   # name -> "…1234"
    headers_unreadable: bool = False         # stored headers that cannot be decrypted (key missing or changed)

    def view_for(self, owner: Owner | None, tools: list[McpTool]) -> dict[str, Any]:
        e = self.endpoint
        group = "mine" if e.owned_by(owner) else "shared"
        return {**e.to_dict(), "group": group, "editable": group == "mine",
                "owner": e.owner[0] if e.owner and group == "shared" else None,
                "status": self.status, "error": _UNREADABLE if self.headers_unreadable else self.error,
                "problems": list(self.problems), "headers_unreadable": self.headers_unreadable,
                "server": self.server, "server_version": self.server_version,
                "checked_at": int(self.checked_at) if self.checked_at else None,
                "headers": self.header_hints if group == "mine" else {},
                "url": e.url if group == "mine" else None,   # may carry a token: its owner only
                "tools": [t.name for t in tools], "tool_ids": [tool_ref(t, self) for t in tools]}


class McpStore(Protocol):
    """Database-held endpoints and tools (agenticwf.agent.runtime.mcp.db_store)."""

    def list_endpoints(self) -> list[EndpointRecord]: ...
    def list_tools(self) -> list[tuple[str, McpTool]]: ...            # (endpoint id, tool)
    def save_endpoint(self, record: EndpointRecord, tools: list[McpTool] | None, keep_headers: bool = False) -> None: ...
    def save_tool(self, endpoint_id: str, tool: McpTool) -> None: ...
    def delete_endpoint(self, endpoint_id: str) -> bool: ...


class McpCatalog:
    def __init__(self, store: McpStore, manager: McpManager, refresh_seconds: float = 10.0):
        self.store = store
        self.manager = manager
        self.refresh_seconds = refresh_seconds
        self._records: dict[str, EndpointRecord] = {}
        self._tools: dict[str, list[McpTool]] = {}
        self._loaded_at = 0.0
        self._reloading = False
        self._lock = threading.Lock()

    # --- reading -----------------------------------------------------------------------------------
    def _refresh(self, force: bool = False) -> None:
        """Reload from the database every `refresh_seconds` (other processes may have changed it).

        tools_for() runs whenever a task's catalogue is built, on the event loop: once loaded, a
        stale list is reloaded in the background and the current one served meanwhile. The first
        load, and the reload after a change made here (`force`), happen at once."""
        if not force and time.time() - self._loaded_at < self.refresh_seconds:
            return
        if force or not self._loaded_at or self.refresh_seconds <= 0:
            self._load()
            return
        with self._lock:
            if self._reloading:
                return
            self._reloading = True

        def run() -> None:
            try:
                self._load()
            finally:
                self._reloading = False

        threading.Thread(target=run, name="mcp-catalog-refresh", daemon=True).start()

    def _load(self) -> None:
        try:
            records = {r.endpoint.id: r for r in self.store.list_endpoints()}
            tools: dict[str, list[McpTool]] = {}
            for endpoint, tool in self.store.list_tools():
                tools.setdefault(endpoint, []).append(tool)
        except Exception:   # keep serving the last good list rather than failing every request
            LOG.exception("could not load MCP endpoints from the database")
            self._loaded_at = time.time()
            return
        with self._lock:
            for record in records.values():
                self.manager.register(record.endpoint)
            gone = {r.endpoint.key for r in self._records.values()} - {r.endpoint.key for r in records.values()}
            self.manager.forget(gone)
            self._records, self._tools, self._loaded_at = records, tools, time.time()

    def get(self, endpoint_id: str) -> EndpointRecord | None:
        """An endpoint by id (`<scope>:<name>`)."""
        self._refresh()
        return self._records.get(endpoint_id)

    def endpoints(self, owner: Owner | None = None) -> list[EndpointRecord]:
        """Endpoints `owner` may see (all of them for None)."""
        self._refresh()
        return [r for r in self._records.values() if owner is None or r.endpoint.visible_to(owner)]

    def endpoint_tools(self, endpoint_id: str) -> list[McpTool]:
        self._refresh()
        return list(self._tools.get(endpoint_id, ()))

    def tools(self, owner: Owner | None = None) -> list[tuple[McpTool, EndpointRecord]]:
        """Tools `owner` may see, enabled or not, with their endpoint."""
        self._refresh()
        result = []
        for endpoint_id, tools in self._tools.items():
            record = self._records.get(endpoint_id)
            if record is None:
                continue
            for tool in tools:
                if owner is None or tool.spec(record.endpoint).source.visible_to(owner):
                    result.append((tool, record))
        return result

    def tools_for(self, owner: Owner) -> list[ToolSpec]:
        """DatabaseToolSource: the enabled tools of enabled endpoints that `owner` may use."""
        return [t.spec(r.endpoint) for t, r in self.tools(owner)
                if t.enabled and r.endpoint.enabled and not r.headers_unreadable]   # never call without its credentials

    def get_tool(self, tool_id: str) -> tuple[McpTool, EndpointRecord] | None:
        """A tool by id (`<scope>:<name>`)."""
        return next(((t, r) for t, r in self.tools() if tool_ref(t, r) == tool_id), None)

    # --- changing ----------------------------------------------------------------------------------
    def put(self, endpoint: McpEndpoint, by: Owner, headers_keep: bool = False) -> EndpointRecord:
        """Add or change an endpoint of `by`'s and expand it. `headers_keep`: an endpoint given
        without headers keeps the stored ones."""
        endpoint = dataclasses.replace(endpoint, source="database", owner=by, plugin=None)   # its id is in by's scope
        existing = self.get(endpoint.id)
        if existing is not None and not existing.endpoint.owned_by(by):
            raise PermissionError(f"MCP endpoint {endpoint.id!r} belongs to someone else")
        keep = headers_keep and existing is not None and not endpoint.headers
        if keep:   # the stored (encrypted) headers stay as they are, whether or not we can read them
            endpoint = dataclasses.replace(endpoint, headers=dict(existing.endpoint.headers))
        try:
            check(endpoint)
        except (EndpointError, TypeError) as exc:
            raise ValueError(str(exc)) from exc
        return self._expand_and_save(endpoint, keep_headers=keep, unreadable=keep and existing.headers_unreadable)

    def refresh(self, endpoint_id: str, by: Owner) -> EndpointRecord:
        record = self._owned(endpoint_id, by)
        return self._expand_and_save(record.endpoint, keep_headers=True, unreadable=record.headers_unreadable)

    def delete(self, endpoint_id: str, by: Owner) -> bool:
        record = self._owned(endpoint_id, by)
        removed = self.store.delete_endpoint(record.endpoint.id)
        self.manager.forget([record.endpoint.key])
        self._refresh(force=True)
        return removed

    def set_tool(self, tool_id: str, by: Owner, enabled: bool | None = None, visibility: str | None = None,
                 approval: str | None = None) -> McpTool:
        found = self.get_tool(tool_id)
        if found is None:
            raise KeyError(tool_id)
        tool, record = found
        if tool.owner != by:
            raise PermissionError(f"tool {tool_id!r} belongs to someone else")
        changes: dict[str, Any] = {}
        if enabled is not None:
            changes["enabled"] = enabled
        if visibility is not None:   # "": as the endpoint
            if visibility not in VISIBILITY + ("",):
                raise ValueError(f"visibility must be one of {', '.join(VISIBILITY)} (or empty: the endpoint's)")
            if visibility == "organization" and not by[1]:
                raise ValueError("visibility 'organization' needs an organization-scoped sign-in")
            changes["visibility"] = visibility
        if approval is not None:
            if approval not in ("ask", "deny"):
                raise ValueError("approval of a user's MCP tool is ask or deny")
            changes["approval"] = approval
        tool = dataclasses.replace(tool, **changes)
        self.store.save_tool(record.endpoint.id, tool)
        self._refresh(force=True)
        return tool

    def _owned(self, endpoint_id: str, by: Owner) -> EndpointRecord:
        record = self.get(endpoint_id)
        if record is None:
            raise KeyError(endpoint_id)
        if not record.endpoint.owned_by(by):
            raise PermissionError(f"MCP endpoint {endpoint_id!r} belongs to someone else")
        return record

    def _expand_and_save(self, endpoint: McpEndpoint, keep_headers: bool = False,
                         unreadable: bool = False) -> EndpointRecord:
        if unreadable:   # do not call it without its credentials
            expansion = McpExpansion(status="error", error=_UNREADABLE, checked_at=time.time())
        else:
            expansion = self.manager.expand(endpoint)
        tools = None   # keep the stored tools unless the endpoint answered
        if expansion.status == "ok":
            previous = {t.name: t for t in self._tools.get(endpoint.id, ())}
            tools = [_keep_settings(t, previous.get(t.name), endpoint) for t in expansion.tools]
        info = expansion.server_info
        record = EndpointRecord(endpoint, status=expansion.status, error=expansion.error[:2000],
                                problems=tuple(p[:500] for p in expansion.problems[:50]),
                                server=_text(info.get("name"), 255), server_version=_text(info.get("version"), 64),
                                checked_at=expansion.checked_at or None, headers_unreadable=unreadable)
        self.store.save_endpoint(record, tools, keep_headers=keep_headers)
        self._refresh(force=True)
        LOG.info("MCP endpoint %s: %s (%d tools)", endpoint.id, expansion.status, len(tools or ()))
        return self._records.get(endpoint.id, record)


def tool_ref(tool: McpTool, record: EndpointRecord) -> str:
    """A database tool's id: `<scope>:<name>`, the scope of its endpoint."""
    return f"{record.endpoint.scope}:{tool.name}"


def _text(value: Any, limit: int) -> str | None:
    """A string a remote server sent, cut to fit its column."""
    return str(value)[:limit] if value is not None else None


_UNREADABLE = ("its stored headers cannot be decrypted (the secrets key is missing or changed): "
               "enter them again, or restore the key")


def _keep_settings(tool: McpTool, previous: McpTool | None, endpoint: McpEndpoint) -> McpTool:
    """A re-expanded tool keeps what its owner set (enabled, visibility, a stricter approval)."""
    tool = dataclasses.replace(tool, owner=endpoint.owner)
    if previous is None:
        return tool
    approval = "deny" if previous.approval == "deny" else tool.approval
    return dataclasses.replace(tool, enabled=previous.enabled, visibility=previous.visibility, approval=approval)
