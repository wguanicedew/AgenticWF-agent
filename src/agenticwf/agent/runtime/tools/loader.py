"""
Load tool definitions from Markdown files: YAML front matter (the spec) + body (the description).

    <system dir>/tools/<tool_name>.md                 system tools
    <plugin dir>/<plugin>/tools/<tool_name>.md        plugins, exposed as <plugin>__<tool>
                                                      (agenticwf.agent.runtime.plugins finds the plugins)
    runtime/defaults/core-tools/<name>.md             the wording of the runtime's own tools (core_tool_text)

An invalid file is reported and skipped, never loaded partly.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from agenticwf.agent.runtime.tools.spec import (
    NAME_RE,
    SUMMARY_MAX,
    USER_HANDLER_KINDS,
    Access,
    Approval,
    Effects,
    HandlerKind,
    HandlerRef,
    ToolSource,
    ToolSpec,
)


class ToolDefinitionError(Exception):
    pass


def split_front_matter(text: str) -> tuple[dict[str, Any], str]:
    if not text.startswith("---\n"):
        raise ToolDefinitionError("missing YAML front matter")
    end = text.find("\n---", 4)
    if end < 0:
        raise ToolDefinitionError("unterminated YAML front matter")
    meta = yaml.safe_load(text[4:end]) or {}
    if not isinstance(meta, dict):
        raise ToolDefinitionError("front matter must be a mapping")
    body = text[end + 4:].lstrip("\n")
    return meta, body.strip()


CORE_TOOLS_DIR = Path(__file__).resolve().parents[1] / "defaults" / "core-tools"


def core_tool_text(name: str) -> tuple[str, str, dict[str, Any]]:
    """(summary, description, input_schema) of a tool the runtime provides, from
    runtime/defaults/core-tools/<name>.md. A broken file fails at import: it ships with the code."""
    path = CORE_TOOLS_DIR / f"{name}.md"
    meta, description = split_front_matter(path.read_text())
    if meta.get("name") != name or not str(meta.get("summary") or "").strip():
        raise ToolDefinitionError(f"{path}: needs name: {name} and a summary")
    schema = meta.get("input_schema") or {"type": "object", "properties": {}}
    Draft202012Validator.check_schema(schema)
    return str(meta["summary"]).strip(), description, schema


def spec_from_definition(meta: dict[str, Any], description: str, source: ToolSource,
                         name: str | None = None) -> ToolSpec:
    """Build a ToolSpec from parsed front matter (also used for database-registered tools)."""
    name = name or meta.get("name")
    if not isinstance(name, str) or not NAME_RE.match(name.split("__")[-1]):
        raise ToolDefinitionError(f"invalid tool name {name!r}")
    summary = meta.get("summary") or ""
    if not isinstance(summary, str) or not summary.strip() or len(summary.strip()) > SUMMARY_MAX:
        raise ToolDefinitionError(f"summary is required, at most {SUMMARY_MAX} characters")
    summary = summary.strip()
    for key, kinds in (("scopes", list), ("tags", list), ("task_types", list), ("handler", dict),
                       ("input_schema", dict)):
        if meta.get(key) is not None and not isinstance(meta[key], kinds):
            raise ToolDefinitionError(f"{key} has the wrong type ({type(meta[key]).__name__})")

    schema = meta.get("input_schema") or {"type": "object", "properties": {}}
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        raise ToolDefinitionError(f"invalid input_schema: {exc.message}") from exc

    handler = meta.get("handler") or {}
    try:
        kind = HandlerKind(handler.get("kind"))
    except ValueError as exc:
        raise ToolDefinitionError(f"unknown handler kind {handler.get('kind')!r}") from exc
    if kind is HandlerKind.CONTROL:
        raise ToolDefinitionError("control handlers are reserved for the harness")
    if source.user_defined and kind not in USER_HANDLER_KINDS:
        raise ToolDefinitionError(f"user-defined tools cannot use {kind.value} handlers")
    if kind is HandlerKind.MCP and "client" in handler:   # set by agenticwf.agent.runtime.mcp only (it names a connection)
        raise ToolDefinitionError("handler.client is reserved; name the plugin's endpoint with handler.endpoint")
    try:   # the tool's own `access` narrows its plugin's (source.access)
        access = source.access.merge(Access.from_mapping(meta.get("access")))
    except (TypeError, ValueError) as exc:
        raise ToolDefinitionError(str(exc)) from exc
    source = replace(source, access=access)

    try:
        return ToolSpec(
            name=name,
            summary=summary,
            description=description,
            input_schema=schema,
            handler=HandlerRef(kind, {k: v for k, v in handler.items() if k != "kind"}),
            source=source,
            scopes=frozenset(meta.get("scopes") or ()) | access.scopes,
            effects=Effects(meta.get("effects", "none")),
            approval=Approval(meta.get("approval", "auto")),
            idempotent=bool(meta.get("idempotent", False)),
            parallel_safe=bool(meta.get("parallel_safe", False)),
            timeout=float(meta.get("timeout", 60)),
            max_result_bytes=int(meta.get("max_result_bytes", 64 * 1024)),
            tags=tuple(meta.get("tags") or ()),
            task_types=tuple(meta.get("task_types") or ()),
        )
    except (TypeError, ValueError) as exc:
        raise ToolDefinitionError(str(exc)) from exc


def load_tool_file(path: Path, source: ToolSource, name: str | None = None) -> ToolSpec:
    meta, body = split_front_matter(path.read_text())
    if name is None and meta.get("name") != path.stem:
        raise ToolDefinitionError(f"front matter name {meta.get('name')!r} does not match file name")
    return spec_from_definition(meta, body, source, name=name)


def load_tools_dir(directory: Path, source: ToolSource) -> tuple[list[ToolSpec], list[str]]:
    """Load <directory>/*.md. Returns (specs, errors)."""
    specs, errors = [], []
    if not directory.is_dir():
        return specs, errors
    for path in sorted(directory.glob("*.md")):
        try:
            specs.append(load_tool_file(path, replace(source, location=str(path))))
        except (ToolDefinitionError, OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")
    return specs, errors


def load_plugin_tools(plugin: Path, source: ToolSource) -> tuple[list[ToolSpec], list[str]]:
    """Load <plugin>/tools/*.md, naming each tool <plugin>__<tool>. Returns (specs, errors)."""
    specs, errors = [], []
    for path in sorted((plugin / "tools").glob("*.md")):
        try:
            tool_source = replace(source, location=str(path))
            meta, _ = split_front_matter(path.read_text())
            if meta.get("name") != path.stem:
                raise ToolDefinitionError("front matter name does not match file name")
            specs.append(load_tool_file(path, tool_source, name=f"{plugin.name}__{path.stem}"))
        except (ToolDefinitionError, OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")
    return specs, errors
