"""
Tool specs, sources, results and the handler contract (docs/agent-runtime.md §5).
"""

from __future__ import annotations

import functools
import hashlib
import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Awaitable, Callable, Union

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.types import Owner

NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SUMMARY_MAX = 200


def qualified_name(name: str, scope: str) -> str:
    """The name a tool gets when another tool the caller can see already has its name: a short tag
    of its owner's scope after the prefix, `<prefix>_<tag>__<tool>`. Tool names the model sees
    cannot carry the scope itself (providers allow [a-zA-Z0-9_-], at most 64 characters)."""
    tag = hashlib.sha256(scope.encode()).hexdigest()[:6]
    prefix, sep, rest = name.partition("__")
    out = f"{prefix}_{tag}__{rest}" if sep else f"{name}_{tag}"
    if len(out) > 64:
        out = out[:57].rstrip("_") + "_" + hashlib.sha256(name.encode()).hexdigest()[:6]
    return out


class Effects(str, Enum):
    NONE = "none"          # read-only
    LOCAL = "local"        # changes the task's workspace
    EXTERNAL = "external"  # changes something outside


class Approval(str, Enum):
    AUTO = "auto"
    ASK = "ask"
    DENY = "deny"


class HandlerKind(str, Enum):
    BUILTIN = "builtin"    # Python in the image; system tools and operator plugins only
    MCP = "mcp"
    HTTP = "http"          # allow-listed hosts only
    SUBAGENT = "subagent"  # child task built from the description
    CONTROL = "control"    # handled by the harness itself (update_plan, ask_user, ...)
    SKILL = "skill"        # loads a skill's instructions (agenticwf.agent.runtime.skills); it acts on nothing


USER_HANDLER_KINDS = frozenset({HandlerKind.MCP, HandlerKind.HTTP, HandlerKind.SUBAGENT})


VISIBILITY = ("user", "organization", "all")


@dataclass(frozen=True)
class Access:
    """Who, beyond owner and visibility, may use a tool or an MCP endpoint (`access:` in plugin.yaml,
    a tool's front matter or an MCP endpoint entry).

        access:
          organizations: [cern, bnl]     only callers signed in to one of these organizations
          users: [user_alice]            only these subjects
          scopes: [physics:read]         and only callers holding these scopes (added to the tool's scopes)

    None (or a missing or empty list) means no restriction. Nested entries only narrow: a tool's
    access inside its plugin's allows the intersection, which may be nobody.
    """

    organizations: tuple[str, ...] | None = None
    users: tuple[str, ...] | None = None
    scopes: frozenset[str] = frozenset()

    @classmethod
    def from_mapping(cls, data: Any) -> "Access":
        if not data:
            return cls()
        if not isinstance(data, dict) or set(data) - {"organizations", "users", "scopes"}:
            raise ValueError("access takes organizations, users and scopes")

        def strings(key: str) -> tuple[str, ...] | None:
            value = data.get(key)
            if value is None:
                return None
            if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
                raise ValueError(f"access.{key} must be a list of strings")
            return tuple(value) or None

        return cls(strings("organizations"), strings("users"), frozenset(strings("scopes") or ()))

    def merge(self, inner: "Access") -> "Access":
        """`inner` (a tool's or an endpoint's) inside `self` (its plugin's): only ever narrower."""
        def both(outer: tuple[str, ...] | None, nested: tuple[str, ...] | None) -> tuple[str, ...] | None:
            if outer is None or nested is None:
                return nested if outer is None else outer
            return tuple(v for v in nested if v in outer)   # empty: nobody

        return Access(both(self.organizations, inner.organizations), both(self.users, inner.users),
                      self.scopes | inner.scopes)

    def allows(self, owner: Owner) -> bool:
        """Organizations and users only; scopes are enforced by the PolicyEngine (ToolSpec.scopes)."""
        return ((self.organizations is None or owner[1] in self.organizations)
                and (self.users is None or owner[0] in self.users))

    def to_dict(self) -> dict[str, Any]:
        """None: no restriction; an empty list: nobody."""
        return {"organizations": None if self.organizations is None else list(self.organizations),
                "users": None if self.users is None else list(self.users), "scopes": sorted(self.scopes)}

    def __bool__(self) -> bool:
        return self.organizations is not None or self.users is not None or bool(self.scopes)


@dataclass(frozen=True)
class ToolSource:
    """Where a tool definition came from, and so who may see it.

    Visibility: `all` (everyone), `organization` (the owner's organization) or `user` (the owner).
    Left empty, it is the kind's default: `all` for core, system and plugin tools (installed by
    operators), `user` for user-defined ones (a user's plugins, the database). `access` narrows it
    further, for example a system plugin shared only with some organizations."""

    kind: str                    # "core" | "system" | "plugin" | "user_plugin" | "user_skill" | "database"
    location: str = ""           # file path or database id, for provenance
    owner: Owner | None = None   # user_plugin / database: the owning (subject, organization)
    visibility: str = ""         # user | organization | all; "" = the kind's default
    access: Access = Access()
    plugin: str = ""             # the plugin that provides it, `<scope>:<name>` (its tools and MCP endpoints)

    @property
    def user_defined(self) -> bool:
        return self.kind in ("user_plugin", "user_skill", "database")

    @property
    def effective_visibility(self) -> str:
        return self.visibility or ("user" if self.user_defined else "all")

    def visible_to(self, owner: Owner) -> bool:
        if not self.access.allows(owner):
            return False
        visibility = self.effective_visibility
        if visibility == "all":
            return True
        if visibility == "organization":
            return self.owner is not None and self.owner[1] is not None and self.owner[1] == owner[1]
        return self.owner == owner


@dataclass(frozen=True)
class HandlerRef:
    kind: HandlerKind
    settings: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @property
    def ref(self) -> str | None:
        return self.settings.get("ref")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    summary: str
    description: str                   # Markdown body: what the model reads
    input_schema: dict[str, Any] = field(hash=False, compare=False)
    handler: HandlerRef
    source: ToolSource
    scopes: frozenset[str] = frozenset()
    effects: Effects = Effects.NONE
    approval: Approval = Approval.AUTO
    idempotent: bool = False
    parallel_safe: bool = False
    timeout: float = 60.0
    max_result_bytes: int = 64 * 1024
    tags: tuple[str, ...] = ()
    task_types: tuple[str, ...] = ()   # empty = all task types

    @property
    def effective_effects(self) -> Effects:
        """What the policy and the write-ahead effect record go by: a user-defined tool's declared
        effects (or an MCP server's readOnlyHint) are claims, never trusted to loosen anything. A
        skill only loads instructions: it has no effects of its own, whoever wrote it."""
        if self.handler.kind is HandlerKind.SKILL:
            return Effects.NONE
        return Effects.EXTERNAL if self.source.user_defined else self.effects

    @functools.cached_property
    def fingerprint(self) -> str:
        """A hash of everything the model and an approver go by and what decides how the call runs.
        Pinned on a call when it is prepared and checked before it runs, so a definition that
        changed in between (a plugin reload, an MCP server's tool list, a name clash resolved
        differently) never runs under a decision taken for the old one. Tags are left out: they
        only rank search results."""
        source = self.source
        data = {
            "name": self.name, "summary": self.summary, "description": self.description,
            "input_schema": self.input_schema,
            "handler": {"kind": self.handler.kind.value, "settings": self.handler.settings},
            "source": {"kind": source.kind, "location": source.location,
                       "owner": list(source.owner) if source.owner else None,
                       "visibility": source.effective_visibility, "access": source.access.to_dict()},
            "scopes": sorted(self.scopes), "effects": self.effects.value, "approval": self.approval.value,
            "idempotent": self.idempotent, "parallel_safe": self.parallel_safe, "timeout": self.timeout,
            "max_result_bytes": self.max_result_bytes, "task_types": list(self.task_types),
        }
        text = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(text.encode()).hexdigest()

    def model_schema(self) -> dict[str, Any]:
        """Neutral schema for LLMRequest.tools; provider adapters translate it."""
        return {"name": self.name, "description": self.description or self.summary,
                "input_schema": self.input_schema}

    def schema_tokens(self) -> int:
        """Rough size of model_schema() in tokens (4 characters per token)."""
        return (len(self.name) + len(self.description or self.summary) + len(str(self.input_schema))) // 4 + 10


@dataclass
class ToolResult:
    """
    What a handler returns. `facts` are structured working-memory facts the harness applies
    (e.g. {"files": [{"path", "version", "edited"}]} or {"tests": {"command", "passed", "failed"}}).
    """

    content: Any = None
    is_error: bool = False
    error_code: str | None = None
    facts: dict[str, Any] = field(default_factory=dict)
    summary: str = ""   # short text for the event stream
    instructions: bool = False   # a skill's instructions: shown to the model as instructions, not as data

    @classmethod
    def error(cls, code: str, message: str) -> "ToolResult":
        return cls(content=message, is_error=True, error_code=code, summary=message)


@dataclass
class CallContext:
    """Passed to every handler. The AuthContext is also ambient: agenticwf.auth.client.current_auth_context()."""

    task_id: str
    call_id: str
    auth: AuthContext
    deadline: float | None = None
    cancelled: Callable[[], bool] = lambda: False


Handler = Callable[[CallContext, dict[str, Any]], Union[ToolResult, Awaitable[ToolResult]]]
