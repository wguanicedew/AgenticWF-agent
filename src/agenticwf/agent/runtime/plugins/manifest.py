"""
A plugin's manifest: `plugin.yaml` (optional) and `.mcp.json` (optional) at the plugin's root.

    name: github                   optional; must equal the directory name
    version: 1.2.0
    description: GitHub issues and pull requests
    author: Ops team
    enabled: true                  false: listed on the admin page, contributes nothing
    access:                        optional: who may use the plugin's tools and MCP tools
      organizations: [cern]        (default: everyone; system plugins only, docs/plugins.md)
      users: [user_alice]
      scopes: [github:use]
    mcp_servers:                   MCP endpoints (agenticwf.agent.runtime.mcp.endpoint); their tools are installed
      github:                      as <plugin>__<tool>
        command: npx               stdio: a child process (operators' plugins only)
        args: [-y, "@modelcontextprotocol/server-github"]
        env: {GITHUB_PERSONAL_ACCESS_TOKEN: "${GITHUB_TOKEN}"}   ${VAR}: from the agent's environment
        tools: [list_issues, get_issue]                          only these; default: all
        approval: ask
        effects: external
        timeout: 60
        access: {organizations: [cern]}                          narrows the plugin's access
      search:
        url: https://mcp.example.org/mcp                         http: streamable HTTP
        headers: {Authorization: "Bearer ${SEARCH_TOKEN}"}

`.mcp.json` uses the common `{"mcpServers": {name: {command, args, env} | {type: "http", url,
headers}}}` format, so plugins written for other MCP hosts work unchanged. Both files may declare
endpoints; a name declared twice is an error.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenticwf.agent.runtime.mcp.endpoint import EndpointError, McpEndpoint, parse_endpoint
from agenticwf.agent.runtime.tools.spec import Access
from agenticwf.agent.runtime.types import Owner

MANIFEST = "plugin.yaml"
MCP_JSON = ".mcp.json"


class ManifestError(Exception):
    pass


@dataclass(frozen=True)
class PluginManifest:
    name: str
    version: str = ""
    description: str = ""
    author: str = ""
    enabled: bool = True
    access: Access = Access()
    mcp_endpoints: tuple[McpEndpoint, ...] = ()
    errors: tuple[str, ...] = field(default=(), compare=False)   # endpoints skipped, with the reason


def read_manifest(root: Path, trusted: bool, owner: Owner | None = None) -> PluginManifest:
    """Read <root>/plugin.yaml and <root>/.mcp.json. A broken manifest raises ManifestError (the
    plugin is not loaded); a broken endpoint is reported in `errors` and skipped."""
    meta: dict[str, Any] = {}
    path = root / MANIFEST
    if path.is_file():
        try:
            meta = yaml.safe_load(path.read_text()) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise ManifestError(f"{path}: {exc}") from exc
        if not isinstance(meta, dict):
            raise ManifestError(f"{path}: must be a mapping")
        if meta.get("name") not in (None, root.name):
            raise ManifestError(f"{path}: name {meta.get('name')!r} does not match the directory {root.name!r}")
    try:
        access = Access.from_mapping(meta.get("access"))
    except ValueError as exc:
        raise ManifestError(f"{path}: {exc}") from exc
    errors: list[str] = []
    if access and not trusted:
        errors.append(f"{MANIFEST}: access ignored: a plugin in your sandbox is visible to you only")
        access = Access()

    servers = meta.get("mcp_servers") or {}
    if not isinstance(servers, dict):
        raise ManifestError(f"{path}: mcp_servers must be a mapping of name to settings")
    declared: list[tuple[str, Any, str]] = [(str(n), s, MANIFEST) for n, s in servers.items()]
    if (root / MCP_JSON).is_file():
        try:
            data = json.loads((root / MCP_JSON).read_text())
        except (OSError, ValueError) as exc:
            raise ManifestError(f"{root / MCP_JSON}: {exc}") from exc
        servers = (data or {}).get("mcpServers") if isinstance(data, dict) else None
        if not isinstance(servers, dict):
            raise ManifestError(f"{root / MCP_JSON}: must be {{\"mcpServers\": {{name: settings}}}}")
        declared += [(str(n), s, MCP_JSON) for n, s in servers.items()]

    endpoints, seen = [], set()
    for name, settings, origin in declared:
        try:
            if name in seen:
                raise EndpointError("declared twice")
            seen.add(name)
            endpoints.append(parse_endpoint(str(name), settings, source="plugin" if trusted else "user_plugin",
                                            plugin=root.name, root=root, owner=owner, inherited=access))
        except EndpointError as exc:
            errors.append(f"{origin}: MCP endpoint {name!r}: {exc}")
    if not isinstance(meta.get("enabled", True), bool):
        raise ManifestError(f"{path}: enabled must be true or false")
    return PluginManifest(
        name=root.name,
        version=str(meta.get("version") or ""),
        description=str(meta.get("description") or "").strip(),
        author=str(meta.get("author") or ""),
        enabled=bool(meta.get("enabled", True)),
        access=access,
        mcp_endpoints=tuple(endpoints),
        errors=tuple(errors),
    )
