"""
Load tool definitions from Markdown files: YAML front matter (the spec) + body (the description).

    <system dir>/tools/<tool_name>.md                 system tools
    <plugin dir>/<plugin>/tools/<tool_name>.md        plugins, exposed as <plugin>__<tool>
                                                      (agenticwf.agent.runtime.plugins finds the plugins)
    runtime/defaults/core-tools/<name>.md             the wording of the runtime's own tools (core_tool_text);
                                                      a task type may reword them (core_tool_rewording)

An invalid file is reported and skipped, never loaded partly. Definitions written as for other
hosts are read too: the schema as `parameters` (OpenAI and Codex function tools) or `inputSchema`
(MCP), and without a `summary`, the first sentence of `description` (front matter or body).
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


def core_tool_names() -> tuple[str, ...]:
    """The runtime's own tools whose wording a task type may change (`skill.md` is the template
    every skill is offered with, not one tool)."""
    return tuple(sorted(p.stem for p in CORE_TOOLS_DIR.glob("*.md") if p.stem not in ("README", "skill")))


def core_tool_rewording(name: str, text: str) -> str:
    """The description in a task type's core-tools/<name>.md. Its front matter must be the packaged
    file's as it is: the arguments are read by code, so only the wording below it changes."""
    if name not in core_tool_names():
        raise ToolDefinitionError(f"no core tool {name!r}")
    meta, description = split_front_matter(text)
    packaged, _ = split_front_matter((CORE_TOOLS_DIR / f"{name}.md").read_text())
    if meta != packaged:
        raise ToolDefinitionError(f"core-tools/{name}.md: keep the front matter as it is; change the description below it")
    if not description:
        raise ToolDefinitionError(f"core-tools/{name}.md: the description is empty")
    return description


def reword(schemas: list[dict[str, Any]], texts: dict[str, str]) -> list[dict[str, Any]]:
    """Model schemas with a task type's wording of core tools. What the runtime appends to a packaged
    description (search_code's repositories, web_search's providers) stays after the new one."""
    if not texts:
        return schemas
    out = []
    for schema in schemas:
        text = texts.get(schema["name"])
        if text is not None:
            packaged = core_tool_text(schema["name"])[1].rstrip()
            live = schema["description"]
            schema = {**schema, "description": text.rstrip() + (live[len(packaged):] if live.startswith(packaged) else "")}
        out.append(schema)
    return out


def _first_sentence(text: str) -> str:
    text = " ".join(text.split())
    end = next((i + 1 for i, c in enumerate(text) if c in ".!?" and (i + 1 == len(text) or text[i + 1] == " ")), len(text))
    text = text[:end]
    return text if len(text) <= SUMMARY_MAX else text[:SUMMARY_MAX - 1].rstrip() + "…"


def spec_from_definition(meta: dict[str, Any], description: str, source: ToolSource,
                         name: str | None = None) -> ToolSpec:
    """Build a ToolSpec from parsed front matter (also used for database-registered tools)."""
    meta = dict(meta)
    for alias in ("parameters", "inputSchema"):   # OpenAI / Codex function tools, MCP
        if meta.get("input_schema") is None and meta.get(alias) is not None:
            meta["input_schema"] = meta[alias]
    if isinstance(meta.get("description"), str) and meta["description"].strip():
        description = description or meta["description"].strip()
        if not meta.get("summary"):
            meta["summary"] = _first_sentence(meta["description"])
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


def load_plugin_tools(plugin: Path, source: ToolSource, name: str | None = None) -> tuple[list[ToolSpec], list[str]]:
    """Load <plugin>/tools/*.md, naming each tool <name>__<tool> (`name`: the plugin's, by default
    its directory's). Returns (specs, errors)."""
    specs, errors = [], []
    name = name or plugin.name
    for path in sorted((plugin / "tools").glob("*.md")):
        try:
            tool_source = replace(source, location=str(path))
            meta, _ = split_front_matter(path.read_text())
            if meta.get("name") != path.stem:
                raise ToolDefinitionError("front matter name does not match file name")
            specs.append(load_tool_file(path, tool_source, name=f"{name}__{path.stem}"))
        except (ToolDefinitionError, OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
            errors.append(f"{path}: {exc}")
    return specs, errors
