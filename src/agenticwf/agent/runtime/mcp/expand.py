"""
Expanding an MCP endpoint: its `tools/list` becomes McpTool records, and each McpTool a ToolSpec
the registry, dispatcher and policy treat like any other tool.

    name          <prefix>__<tool>: the MCP name lowercased, other characters as `_`, at most 64
    summary       the first line of the tool's description
    input_schema  inputSchema (a tool with an invalid schema is skipped)
    effects       the endpoint's `effects`; else `none` for readOnlyHint tools, else `external`
    approval      operators' endpoints: the endpoint's `approval`, else `auto` if read-only, else `ask`;
                  users' endpoints: `ask`, or `deny` (a user's settings only make a tool stricter)

A tool has an owner and a visibility of its own. Expanded from a plugin it is the endpoint's (all,
narrowed by `access`); from the database it starts as the endpoint's and its owner may change it,
e.g. share one tool with the organization while the endpoint stays private.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint
from agenticwf.agent.runtime.tools.spec import SUMMARY_MAX, Approval, Effects, HandlerKind, HandlerRef, ToolResult, ToolSource, ToolSpec
from agenticwf.agent.runtime.types import Owner


@dataclass(frozen=True)
class McpTool:
    name: str                                          # the exposed name, <prefix>__<tool>
    mcp_name: str                                      # the name the server knows
    endpoint: str                                      # McpEndpoint.key
    summary: str
    description: str
    input_schema: dict[str, Any] = field(hash=False, compare=False)
    effects: str = "external"
    approval: str = "ask"
    idempotent: bool = False
    read_only: bool = False
    enabled: bool = True
    owner: Owner | None = None                         # database tools: who may change it
    visibility: str = ""                               # "" = the endpoint's

    def spec(self, endpoint: McpEndpoint) -> ToolSpec:
        source = ToolSource(endpoint.source, f"mcp:{endpoint.key}#{self.mcp_name}", self.owner or endpoint.owner,
                            visibility=self.visibility or endpoint.effective_visibility, access=endpoint.access,
                            plugin=f"{endpoint.scope}:{endpoint.plugin}" if endpoint.plugin else "")
        return ToolSpec(
            name=self.name, summary=self.summary, description=self.description, input_schema=self.input_schema,
            handler=HandlerRef(HandlerKind.MCP, {"endpoint": endpoint.name, "tool": self.mcp_name}),
            source=source, scopes=endpoint.access.scopes, effects=Effects(self.effects), approval=Approval(self.approval),
            idempotent=self.idempotent, timeout=endpoint.timeout, tags=("mcp", endpoint.prefix or "", endpoint.name),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "mcp_name": self.mcp_name, "summary": self.summary, "effects": self.effects,
                "approval": self.approval, "read_only": self.read_only, "enabled": self.enabled,
                "visibility": self.visibility}


def tool_name(prefix: str, raw: str) -> str:
    """<prefix>__<tool>, with the MCP tool's name made a valid tool name and the whole at most 64."""
    slug = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_") or "tool"
    if not slug[0].isalpha():
        slug = "t_" + slug
    room = 64 - len(prefix) - 2
    if room < 8:
        raise ValueError(f"the prefix {prefix!r} leaves no room for tool names")
    if len(slug) > room:
        digest = hashlib.sha256(raw.encode()).hexdigest()[:6]
        slug = slug[:room - 7].rstrip("_") + "_" + digest
        if not slug[0].isalpha():
            slug = "t" + slug[1:]
    return f"{prefix}__{slug}"


def expand_tools(endpoint: McpEndpoint, listed: list[dict[str, Any]],
                 taken: set[str] = frozenset()) -> tuple[list[McpTool], list[str]]:
    """The endpoint's tools from a tools/list reply. Returns (tools, problems); `taken` holds names
    already used next to these (a plugin's own tools)."""
    wanted, tools, problems, names = set(endpoint.tools), [], [], set(taken)
    for raw in listed:
        if not isinstance(raw, dict):
            problems.append(f"an entry of tools/list is not an object ({type(raw).__name__}): skipped")
            continue
        name = raw.get("name")
        if wanted and name not in wanted:
            continue
        try:
            tool = _tool(endpoint, raw)
        except ValueError as exc:
            problems.append(f"tool {name!r} skipped: {exc}")
            continue
        if tool.name in names:
            problems.append(f"tool {name!r} skipped: {tool.name} is already used")
            continue
        names.add(tool.name)
        tools.append(tool)
    missing = wanted - {t.get("name") for t in listed if isinstance(t, dict)}
    if missing:
        problems.append(f"no tool {', '.join(sorted(missing))} on this endpoint")
    return tools, problems


def _tool(endpoint: McpEndpoint, raw: dict[str, Any]) -> McpTool:
    mcp_name = raw.get("name")
    if not isinstance(mcp_name, str) or not mcp_name:
        raise ValueError("no name")
    if len(mcp_name) > 255:
        raise ValueError("name longer than 255 characters")
    description = raw.get("description") or ""
    title = raw.get("title") if isinstance(raw.get("title"), str) else ""
    if not isinstance(description, str):
        raise ValueError("description is not text")
    description = description.strip()[:32_000]
    first = (description.splitlines() or [title or mcp_name])[0].strip()
    summary = first if len(first) <= SUMMARY_MAX else first[:SUMMARY_MAX - 1].rstrip() + "…"
    schema = raw.get("inputSchema") or {"type": "object", "properties": {}}
    if not isinstance(schema, dict) or schema.get("type", "object") != "object":
        raise ValueError("inputSchema must be a JSON Schema of type object")
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ValueError(f"invalid inputSchema: {exc.message}") from exc
    hints = raw.get("annotations") if isinstance(raw.get("annotations"), dict) else {}
    read_only = hints.get("readOnlyHint") is True
    effects = endpoint.effects or ("none" if read_only else "external")
    if endpoint.trusted:
        approval = endpoint.approval or ("auto" if read_only else "ask")
    else:
        approval = "deny" if endpoint.approval == "deny" else "ask"
    return McpTool(name=tool_name(endpoint.prefix, mcp_name), mcp_name=mcp_name, endpoint=endpoint.key,
                   summary=summary or mcp_name, description=description or summary or mcp_name, input_schema=schema,
                   effects=effects, approval=approval, idempotent=hints.get("idempotentHint") is True,
                   read_only=read_only, owner=endpoint.owner)


def to_result(reply: dict[str, Any]) -> ToolResult:
    """An MCP tools/call result as a ToolResult: text parts joined, other parts described."""
    parts = []
    content_items = reply.get("content") if isinstance(reply.get("content"), list) else []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")
        if kind == "text":
            parts.append(str(item.get("text", "")))
        elif kind == "resource" and isinstance(item.get("resource"), dict) and "text" in item["resource"]:
            parts.append(str(item["resource"]["text"]))
        else:
            parts.append(f"[{kind}{': ' + item['mimeType'] if item.get('mimeType') else ''}]")
    content: Any = "\n".join(parts)
    if not content and reply.get("structuredContent") is not None:
        content = reply["structuredContent"]
    if reply.get("isError"):
        return ToolResult(content=content or "the tool reported an error", is_error=True, error_code="tool_error",
                          summary=str(content)[:200])
    return ToolResult(content=content, summary=str(content)[:200])
