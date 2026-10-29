"""
ToolRegistry: loads tool definitions from all sources into one catalogue, and knows how to call
each tool. It does not decide permissions (PolicyEngine), choose a task's tools
(ToolDispatcher), execute calls, or talk to the model.

Sources, in this order (docs/agent-runtime.md §5):
    core            control tools (update_plan, ask_user, ...), and those added with add_core
                    (memory_search, when a memory store is configured)
    system          <system dir>/tools/<tool_name>.md
    plugin          system plugins: tools/<tool>.md and MCP endpoints  everyone, unless `access` narrows it
    user_plugin     <user sandbox>/plugins/<plugin>/, the same           its owner
    database        DatabaseToolSources, e.g. McpCatalog                 owner, organization, or all

Every tool's ToolSource says who may see it (visibility, owner, access); catalogue() filters on
it. Plugins are read by the PluginManager (docs/plugins.md); MCP endpoints are expanded and called
through the McpManager (docs/mcp.md).
"""

from __future__ import annotations

import dataclasses
import math
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from jsonschema import Draft202012Validator

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.embeddings import EmbeddingIndex
from agenticwf.agent.runtime.text import terms
from agenticwf.agent.runtime.mcp.manager import McpManager
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.tools.control import CONTROL_TOOLS
from agenticwf.agent.runtime.tools.loader import load_tools_dir
from agenticwf.agent.runtime.tools.spec import Handler, HandlerKind, ToolResult, ToolSource, ToolSpec, qualified_name
from agenticwf.agent.runtime.types import Owner, scope_of

LOG = get_logger(__name__)


class DatabaseToolSource(Protocol):
    """Database-registered tools: agenticwf.agent.runtime.mcp.catalog.McpCatalog (MCP endpoints users add).
    TODO: agent_tools for http and subagent tools."""

    def tools_for(self, owner: Owner) -> list[ToolSpec]:
        ...


class ToolRegistry:
    def __init__(self, system_dir: Path | None = None, plugin_dirs: Iterable[Path] = (),
                 user_sandbox: Callable[[Owner], Path | None] | None = None,
                 database: DatabaseToolSource | Iterable[DatabaseToolSource] | None = None,
                 plugins: PluginManager | None = None, mcp: McpManager | None = None, skills: Any = None,
                 embeddings: EmbeddingIndex | None = None):
        """`mcp`: the McpManager that calls MCP tools; it must be the one that expanded them (the
        plugins' and the McpCatalog's). Default: the plugin manager's. `skills`: the SkillCatalog,
        both a source of `skill__<name>` tools and the handler that loads them. `embeddings`: match
        by meaning as well as by words (agenticwf.agent.runtime.embeddings)."""
        self.system_dir = system_dir
        self.plugins = plugins or PluginManager(plugin_dirs, user_sandbox, mcp=mcp)
        self.mcp: McpManager = mcp or self.plugins.mcp
        self.databases: list[DatabaseToolSource] = (
            [] if database is None else [database] if hasattr(database, "tools_for") else list(database))
        self.skills = skills
        self.embeddings = embeddings
        if skills is not None and skills not in self.databases:
            self.databases.append(skills)
        self._core: list[ToolSpec] = []                 # core tools beyond the control tools (add_core)
        self._shared: dict[str, ToolSpec] = {}          # core, system, system plugins
        self._handlers: dict[str, Handler] = {}         # builtin refs -> callables
        self.errors: list[str] = []

    # --- loading -------------------------------------------------------------------------------
    def register_handler(self, ref: str, handler: Handler) -> None:
        """Bind a builtin handler ref (e.g. "code.search") to its implementation."""
        self._handlers[ref] = handler

    def add_core(self, spec: ToolSpec, handler: Handler) -> None:
        """A core tool the runtime itself provides (e.g. memory_search), with its builtin handler. It
        stays through load(): core tools are loaded before system tools, so none can take its name."""
        self.register_handler(spec.handler.ref or spec.name, handler)
        self._core.append(spec)
        self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        """Add one shared tool now (hosts and tests); load() rebuilds the whole set."""
        shared = dict(self._shared)
        _add(shared, spec)
        self._shared = shared

    def load(self) -> list[str]:
        """(Re)load core, system and system-plugin tools; reloads the plugins too (their MCP
        endpoints are expanded again). Returns the errors (also logged).

        The new catalogue is built aside and swapped in at the end, so tasks running meanwhile keep
        seeing the previous one, never a half-built one; if loading fails, the previous one stays."""
        shared: dict[str, ToolSpec] = {}
        for spec in (*CONTROL_TOOLS, *self._core):
            _add(shared, spec)
        specs, errors = [], []
        if self.system_dir:
            specs, errors = load_tools_dir(self.system_dir / "tools", ToolSource("system"))
        plugin_errors = self.plugins.load()   # logged by the plugin manager
        if self.skills is not None:   # system skills; plugin skills came with the plugins
            plugin_errors = plugin_errors + self.skills.load()
        specs.extend(self.plugins.tools())
        for spec in specs:
            try:
                _add(shared, spec)
            except ValueError as exc:
                errors.append(f"{spec.source.location}: {exc}")
        for error in errors:
            LOG.error("tool definition skipped: %s", error)
        self._shared, self.errors = shared, errors + plugin_errors
        if self.embeddings is not None:   # so the first tasks do not wait for the whole catalogue
            self.embeddings.warm([_document(s) for s in shared.values()])
        return self.errors

    def _user_tools(self, owner: Owner) -> list[ToolSpec]:
        """The owner's own tools, then those others share with them. Plugins and endpoints are scoped
        by owner (`<scope>:<name>`), so two can share a name; the tool names the model sees cannot
        carry a scope, so a clash is resolved here: system names win, then the owner's own tools,
        and a tool whose name is taken gets its scope's tag (spec.qualified_name)."""
        specs: list[ToolSpec] = list(self.plugins.user_tools(owner))
        for database in self.databases:
            specs.extend(database.tools_for(owner))
        specs.sort(key=lambda s: (s.source.owner != owner, scope_of(s.source.owner)))   # stable: own first
        result, seen = [], set(self._shared)
        for spec in specs:
            if spec.name in seen:
                name = qualified_name(spec.name, scope_of(spec.source.owner))
                if name in seen:
                    LOG.warning("user tool %s skipped: name already visible (%s)", spec.name, spec.source.location)
                    continue
                spec = dataclasses.replace(spec, name=name)
            seen.add(spec.name)
            result.append(spec)
        return result

    # --- queries -------------------------------------------------------------------------------
    def catalogue(self, owner: Owner) -> list[ToolSpec]:
        """Every tool this owner can see, before permission filtering."""
        shared = [s for s in self._shared.values() if s.source.visible_to(owner)]
        return shared + [s for s in self._user_tools(owner) if s.source.visible_to(owner)]

    def lookup(self, name: str, owner: Owner) -> ToolSpec | None:
        return next((s for s in self.catalogue(owner) if s.name == name), None)

    def validate(self, spec: ToolSpec, arguments: dict[str, Any]) -> list[str]:
        validator = Draft202012Validator(spec.input_schema)
        return [f"{'/'.join(map(str, e.path)) or '(root)'}: {e.message}" for e in validator.iter_errors(arguments)]

    def handler(self, spec: ToolSpec) -> Handler:
        kind = spec.handler.kind
        if kind is HandlerKind.BUILTIN:
            fn = self._handlers.get(spec.handler.ref or "") or self.plugins.builtin_handler(spec.handler.ref or "")
            if fn is None:
                return _unavailable(f"builtin handler {spec.handler.ref!r} is not registered")
            return fn
        if kind is HandlerKind.MCP:
            return self.mcp.handler(spec)
        if kind is HandlerKind.SKILL:
            if self.skills is None:
                return _unavailable("skills are not configured")
            return self.skills.handler(spec)
        # TODO: HTTP with host allow-list, subagent tasks.
        return _unavailable(f"{kind.value} handlers are not implemented yet")

    def search(self, query: str, candidates: Iterable[ToolSpec], k: int) -> list[ToolSpec]:
        """
        The `k` candidates closest to `query`, best first. Deterministic: ties are broken by name.
        Candidates are already visibility-filtered.

        Words: BM25 over each tool's name (weight 3), summary and tags (2) and description (1), so
        words rare among the candidates count more than common ones. Meaning, with `embeddings`:
        the cosine similarity of the query's and the tool's embeddings. The score combines both
        (`[embeddings] weight` for the similarity, BM25 scaled to the best candidate for the rest);
        a candidate that shares no word with the query is kept when its similarity reaches
        `min_similarity` ("changelog" finds release-notes). Without embeddings, or while they are
        unavailable, BM25 alone, and a candidate must share a word.
        """
        candidates = list(candidates)
        words = terms(query)
        if not query.strip() or not candidates:
            return []
        lexical = _bm25(words, candidates)
        similarity = (self.embeddings.similarities(query, [_document(s) for s in candidates])
                      if self.embeddings is not None else None)
        if similarity is None:
            ranked = [(score, spec) for score, spec in zip(lexical, candidates) if score > 0]
        else:
            settings, best = self.embeddings.settings, max(lexical) or 1.0
            ranked = [(settings.weight * max(0.0, sim) + (1 - settings.weight) * score / best, spec)
                      for score, sim, spec in zip(lexical, similarity, candidates)
                      if score > 0 or sim >= settings.min_similarity]
        ranked.sort(key=lambda r: (-r[0], r[1].name))
        return [spec for _, spec in ranked[:k]]

    def embedding_texts(self, owner: Owner) -> list[str]:
        """What the owner's tools and skills are matched by meaning with (EmbeddingIndex.prepare)."""
        return [_document(s) for s in self.catalogue(owner)]

    @staticmethod
    def schemas(specs: Iterable[ToolSpec]) -> list[dict[str, Any]]:
        return [s.model_schema() for s in specs]


_K1, _B = 1.2, 0.75


def _bm25(words: set[str], candidates: list[ToolSpec]) -> list[float]:
    """BM25 of each candidate for `words`, its fields weighted: name 3, summary and tags 2, description 1."""
    docs = []
    for spec in candidates:
        counts: Counter[str] = Counter()
        for weight, text in ((3, spec.name.replace("_", " ")), (2, f"{spec.summary} {' '.join(spec.tags)}"),
                             (1, spec.description)):
            for term in terms(text, unique=False):
                counts[term] += weight
        docs.append(counts)
    if not words or not docs:
        return [0.0] * len(docs)
    average = sum(sum(t.values()) for t in docs) / len(docs) or 1.0
    frequency = Counter(term for t in docs for term in set(t) & words)

    def score(counts: Counter[str]) -> float:
        length, total = sum(counts.values()), 0.0
        for term in words & set(counts):
            idf = math.log(1 + (len(docs) - frequency[term] + 0.5) / (frequency[term] + 0.5))
            tf = counts[term]
            total += idf * tf * (_K1 + 1) / (tf + _K1 * (1 - _B + _B * length / average))
        return total

    return [score(t) for t in docs]


def _document(spec: ToolSpec) -> str:
    """What a tool's embedding is made of."""
    return f"{spec.name}: {spec.summary}\n{' '.join(spec.tags)}\n{spec.description[:2000]}"


def _add(shared: dict[str, ToolSpec], spec: ToolSpec) -> None:
    if spec.name in shared:
        raise ValueError(f"tool {spec.name!r} already defined by {shared[spec.name].source.location}")
    shared[spec.name] = spec


def _unavailable(message: str) -> Handler:
    def handler(call, arguments) -> ToolResult:
        return ToolResult.error("tool_unavailable", message)
    return handler
