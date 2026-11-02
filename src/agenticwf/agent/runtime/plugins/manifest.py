"""
A plugin's manifest: `plugin.yaml` (optional) and `.mcp.json` (optional) at the plugin's root, or
the manifest of a Claude Code or Codex plugin (`.claude-plugin/plugin.json`, `.codex-plugin/plugin.json`).

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
    environment:                   what its `kind: command` tools run in: a setup sourced once
      setup: source /cvmfs/…/setup.sh    (system plugins only; agenticwf.agent.runtime.plugins.command)
      refresh: 3600
      pass: [RUCIO_ACCOUNT]

`.mcp.json` uses the common `{"mcpServers": {name: {command, args, env} | {type: "http", url,
headers}}}` format (or the servers without the `mcpServers` wrapper), so plugins written for other
MCP hosts work unchanged. Both files may declare endpoints; a name declared twice is an error.

A Claude Code or Codex plugin.json gives the version, description and author when plugin.yaml does
not, and may point at more components inside the plugin:

    {"name": "code-review", "version": "1.0.0", "description": "…", "author": {"name": "…"},
     "skills": "./more-skills/",           also read, after skills/ (a path or a list of paths)
     "commands": ["./extra/review.md"],    also read, after commands/ (Claude Code: files or directories)
     "mcpServers": "./servers.json"}       a file in the .mcp.json format, or the servers themselves

Its `name` is not checked: the directory names the plugin. What the agent cannot run (agents,
hooks, apps, LSP servers, output styles) is reported and ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenticwf.agent.runtime.mcp.endpoint import EndpointError, McpEndpoint, parse_endpoint
from agenticwf.agent.runtime.plugins.command import ShellSetup
from agenticwf.agent.runtime.tools.spec import Access
from agenticwf.agent.runtime.types import Owner

MANIFEST = "plugin.yaml"
MCP_JSON = ".mcp.json"
FOREIGN_MANIFESTS = (".claude-plugin/plugin.json", ".codex-plugin/plugin.json")
UNSUPPORTED = {"agents": "subagents", "hooks": "hooks", "apps": "apps", "lspServers": "LSP servers",
               "outputStyles": "output styles"}


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
    skill_paths: tuple[Path, ...] = ()       # plugin.json `skills`: directories of skills, after skills/
    command_paths: tuple[Path, ...] = ()     # plugin.json `commands`: files or directories, after commands/
    environment: ShellSetup | None = None    # `environment`: the setup its command tools run in
    errors: tuple[str, ...] = field(default=(), compare=False)   # endpoints skipped, with the reason


def plugin_name(root: Path) -> str:
    """The name a plugin directory gives its plugin and its tools' prefix: Claude Code and Codex
    plugins are named with hyphens (code-review), tool names here with underscores (code_review)."""
    return root.name.replace("-", "_")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ManifestError(f"{path}: {exc}") from exc


def _servers(data: Any, path: Path) -> dict[str, Any]:
    """{"mcpServers": {...}}, or the servers without the wrapper (as Claude Code plugins also write it)."""
    if isinstance(data, dict) and "mcpServers" in data:
        data = data["mcpServers"]
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise ManifestError(f"{path}: must be {{\"mcpServers\": {{name: settings}}}}")
    return data


def _inside(root: Path, value: Any, path: Path, key: str) -> list[Path]:
    """plugin.json paths ("./x/" or a list of them), each inside the plugin."""
    values = [value] if isinstance(value, str) else value
    if not isinstance(values, list) or not all(isinstance(v, str) for v in values):
        raise ManifestError(f"{path}: {key} must be a path or a list of paths")
    resolved = []
    for value in values:
        target = (root / value).resolve()
        if target != root.resolve() and root.resolve() not in target.parents:
            raise ManifestError(f"{path}: {key} {value!r} is outside the plugin")
        resolved.append(target)
    return resolved


def _foreign(root: Path) -> tuple[dict[str, Any], Path | None]:
    """The first of .claude-plugin/plugin.json and .codex-plugin/plugin.json, if any."""
    for relative in FOREIGN_MANIFESTS:
        path = root / relative
        if path.is_file():
            data = _read_json(path)
            if not isinstance(data, dict):
                raise ManifestError(f"{path}: must be an object")
            return data, path
    return {}, None


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
        if meta.get("name") not in (None, root.name, plugin_name(root)):
            raise ManifestError(f"{path}: name {meta.get('name')!r} does not match the directory {root.name!r}")
    foreign, foreign_path = _foreign(root)   # plugin.yaml's settings come first
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
    mcp_files = [(root / MCP_JSON).resolve()] if (root / MCP_JSON).is_file() else []
    skill_paths: list[Path] = []
    command_paths: list[Path] = []
    if foreign_path is not None:
        origin = str(foreign_path.relative_to(root))
        inline = foreign.get("mcpServers")
        if isinstance(inline, dict):
            declared += [(str(n), s, origin) for n, s in _servers(inline, foreign_path).items()]
        elif inline is not None:
            mcp_files += [p for p in _inside(root, inline, foreign_path, "mcpServers") if p != (root / MCP_JSON).resolve()]
        if foreign.get("skills") is not None:
            skill_paths = [p for p in _inside(root, foreign["skills"], foreign_path, "skills")
                           if p != (root / "skills").resolve()]
        if foreign.get("commands") is not None:
            command_paths = [p for p in _inside(root, foreign["commands"], foreign_path, "commands")
                             if p != (root / "commands").resolve()]
        errors.extend(f"{origin}: {key} ignored: {what} are not supported"
                      for key, what in UNSUPPORTED.items() if foreign.get(key))
    errors.extend(f"{d}/ ignored: {UNSUPPORTED[d]} are not supported" for d in ("agents", "hooks")
                  if (root / d).is_dir() and not foreign.get(d))
    for mcp_file in mcp_files:
        origin = str(mcp_file.relative_to(root.resolve()))
        declared += [(str(n), s, origin) for n, s in _servers(_read_json(mcp_file), mcp_file).items()]

    endpoints, seen = [], set()
    for name, settings, origin in declared:
        try:
            if name in seen:
                raise EndpointError("declared twice")
            seen.add(name)
            endpoints.append(parse_endpoint(str(name), settings, source="plugin" if trusted else "user_plugin",
                                            plugin=plugin_name(root), root=root, owner=owner, inherited=access))
        except EndpointError as exc:
            errors.append(f"{origin}: MCP endpoint {name!r}: {exc}")
    environment = None
    if meta.get("environment") is not None:
        if not trusted:
            errors.append(f"{MANIFEST}: environment ignored: your own plugins cannot run commands in the agent")
        else:
            try:
                environment = ShellSetup.parse(meta["environment"])
            except ValueError as exc:
                raise ManifestError(f"{path}: {exc}") from exc
    if not isinstance(meta.get("enabled", True), bool):
        raise ManifestError(f"{path}: enabled must be true or false")
    author = meta.get("author") or foreign.get("author") or ""
    if isinstance(author, dict):   # plugin.json: {"name": …, "email": …}
        author = author.get("name") or ""
    return PluginManifest(
        name=plugin_name(root),
        version=str(meta.get("version") or foreign.get("version") or ""),
        description=str(meta.get("description") or foreign.get("description") or "").strip(),
        author=str(author),
        enabled=bool(meta.get("enabled", True)),
        access=access,
        mcp_endpoints=tuple(endpoints),
        skill_paths=tuple(skill_paths),
        command_paths=tuple(command_paths),
        environment=environment,
        errors=tuple(errors),
    )
