"""
Task types: which AGENTS.md and WORKFLOW.md a task runs with (docs/agent-runtime.md §4).

    <system dir>/task-types/<type>/{AGENTS.md, WORKFLOW.md, task.yaml}       system types (operators)
    <plugin dir>/<plugin>/task-types/<type>/{…, task.yaml}                   plugin types, named
                                                                             <plugin>.<type> (operators)
    <user sandbox>/plugins/<plugin>/task-types/<type>/{AGENTS.md, WORKFLOW.md} a user's plugin types
    <user sandbox>/task-types/<type>/{AGENTS.md, WORKFLOW.md}                user-defined, per owner

Every type has an id, `<scope>:<name>` (agenticwf.agent.runtime.types.scope_of): `system:review`,
`system:lint.check`, `user_alice:my-review`, `user_alice:mine.notes`. So a user's type may have
the same name as a system type, or as another user's. A reference is an id or a bare name; a bare
name is looked up in the caller's own scope first, then the system's. That never loosens anything:
a user's type carries instructions only (no task.yaml). Other users' types are never visible.

Each file resolves on its own, first match wins: the type's own file → system `default`.
task.yaml (hard settings) only counts where operators put it: system types and system plugins.
Users create, change and delete their own types (api/v1/task-types); system and plugin types
are read-only there. The resolved files are snapshotted with the task, so later edits only affect
new tasks.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import yaml

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.types import SYSTEM_SCOPE, Mode, Owner, scope_of

if TYPE_CHECKING:
    from agenticwf.agent.runtime.plugins.manager import PluginManager

LOG = get_logger(__name__)

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")                      # system and user types
_TYPE_NAME = r"(?:[a-z][a-z0-9_]{0,63}\.)?[a-z0-9]+(?:-[a-z0-9]+)*"     # <type> or <plugin>.<type>
TYPE_REF_RE = re.compile(rf"^(?:[A-Za-z0-9_.@:-]{{1,255}}:)?{_TYPE_NAME}$")  # a name, or <scope>:<name>
INSTRUCTION_FILES = ("AGENTS.md", "WORKFLOW.md")
PROMPTS_DIR = "prompts"                                       # runtime prompts (docs/agent-flow.md, Prompts)
PROMPTS = ("split", "reply", "plan", "combine", "final_report", "compact", "consolidate", "compact_conversation",
           "preamble", "cron_check", "cron_report")
TASK_PROMPTS = ("plan", "combine", "final_report", "compact", "consolidate")   # a task type may override these
PACKAGED_PROMPTS = Path(__file__).resolve().parents[1] / "defaults" / PROMPTS_DIR
DEFAULT = "default"


class TaskTypeError(Exception):
    pass


def split_type_ref(ref: str) -> tuple[str | None, str]:
    """(scope or None, name) of `<scope>:<name>` or a bare name. Names have no ":"; scopes may."""
    scope, sep, name = ref.rpartition(":")
    return (scope, name) if sep else (None, ref)


def type_id(ref: str) -> str:
    """A reference as an id: a bare name means the system's (how tools' `task_types` name types)."""
    scope, name = split_type_ref(ref)
    return f"{scope or SYSTEM_SCOPE}:{name}"


@dataclass(frozen=True)
class _Found:
    id: str
    scope: str
    name: str
    source: str                          # system | plugin | user
    dirs: list[tuple[str, Path]]         # (provenance label, directory)
    settings: "TaskTypeSettings | None"  # system types and system plugins only
    user_defined: bool


@dataclass(frozen=True)
class InstructionFile:
    name: str       # AGENTS.md | WORKFLOW.md
    content: str
    sha256: str
    source: str     # provenance, e.g. "system:default/AGENTS.md" or "user:my-review/AGENTS.md"


@dataclass(frozen=True)
class TaskTypeSettings:
    """Hard settings from a system type's task.yaml; read by the harness and policy, never the model."""

    mode: Mode = Mode.DEFAULT
    modes: tuple[Mode, ...] = ()         # modes a task of this type may ask for; empty: see allowed_modes()
    tools_allow: tuple[str, ...] = ()    # empty = no allow-list
    tools_deny: tuple[str, ...] = ()
    pin_tools: tuple[str, ...] = ()
    model: str | None = None             # model handle or tier, for the model router
    tier: str | None = None              # at least this model tier: small | medium | large (docs/model-routing.md)
    budget: float | None = None          # a task's cost limit, in the deployment's cost unit
    max_steps: int | None = None
    knowledge: tuple[str, ...] = ()      # knowledge-base ids
    plan: str | None = None              # always | never; None = the harness default (docs/agent-flow.md §4)

    def allowed_modes(self) -> tuple[Mode, ...]:
        """`modes` when task.yaml lists them; otherwise a plan-only type stays plan-only, and other
        types allow every mode."""
        if self.modes:
            return self.modes
        return (Mode.PLAN,) if self.mode is Mode.PLAN else tuple(Mode)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "TaskTypeSettings":
        tools = data.get("tools") or {}
        return cls(
            mode=Mode(data.get("mode", "default")),
            modes=tuple(Mode(m) for m in data.get("modes") or ()),
            tools_allow=tuple(tools.get("allow") or ()),
            tools_deny=tuple(tools.get("deny") or ()),
            pin_tools=tuple(data.get("pin_tools") or ()),
            model=data.get("model"),
            tier=data.get("tier"),
            budget=float(data["budget"]) if data.get("budget") is not None else None,
            max_steps=data.get("max_steps"),
            knowledge=tuple(data.get("knowledge") or ()),
            plan=data.get("plan"),
        )


@dataclass(frozen=True)
class ResolvedTaskType:
    requested: str
    agents: InstructionFile
    workflow: InstructionFile
    settings: TaskTypeSettings = field(default_factory=TaskTypeSettings)
    user_defined: bool = False
    fell_back: bool = False   # requested type unknown: everything came from `default`
    resolved_id: str = ""     # the id the reference resolved to (`<scope>:<name>`)

    @property
    def effective(self) -> str:
        """The id of the type whose files and settings the task runs with: `system:default` when the
        requested one is unknown. (Records from before scoped types hold no id: their name.)"""
        if self.fell_back:
            return f"{SYSTEM_SCOPE}:{DEFAULT}"
        return getattr(self, "resolved_id", "") or self.requested

    @property
    def provenance(self) -> dict[str, str]:
        return {f.name: f.source for f in (self.agents, self.workflow)}


@dataclass(frozen=True)
class TaskTypeInfo:
    """One entry of the catalogue a user sees (api/v1/task-types)."""

    name: str
    source: str               # system | plugin | user
    plugin: str | None = None
    files: tuple[str, ...] = ()   # instruction files the type defines itself (others come from `default`)
    settings: TaskTypeSettings = field(default_factory=TaskTypeSettings)
    editable: bool = False
    scope: str = SYSTEM_SCOPE
    ref: str = ""             # what to choose it by: its bare name, or its id when the name is ambiguous

    @property
    def id(self) -> str:
        return f"{self.scope}:{self.name}"

    def to_dict(self) -> dict[str, Any]:
        s = self.settings
        return {"id": self.id, "name": self.name, "scope": self.scope, "ref": self.ref or self.name,
                "source": self.source, "plugin": self.plugin, "files": list(self.files),
                "editable": self.editable,
                "settings": {"mode": s.mode.value, "model": s.model, "max_steps": s.max_steps, "plan": s.plan}}


class TaskTypeResolver:
    def __init__(self, system_dir: Path, user_sandbox: Callable[[Owner], Path | None] | None = None,
                 max_file_bytes: int = 32 * 1024, plugin_dirs: list[Path] | None = None,
                 plugins: "PluginManager | None" = None):
        from agenticwf.agent.runtime.plugins.manager import PluginManager   # agenticwf.agent.runtime.plugins imports the tools package

        self.system_dir = Path(system_dir)
        self.user_sandbox = user_sandbox
        self.max_file_bytes = max_file_bytes
        self.plugins = plugins or PluginManager(plugin_dirs or (), user_sandbox)

    @property
    def _system_root(self) -> Path:
        return self.system_dir / "task-types"

    def check(self) -> None:
        """The agent refuses to start without both default files."""
        for name in INSTRUCTION_FILES:
            if not (self._system_root / DEFAULT / name).is_file():
                raise TaskTypeError(f"missing {self._system_root / DEFAULT / name}")

    @staticmethod
    def _types_in(root: Path) -> list[str]:
        if not root.is_dir():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir() and NAME_RE.match(p.name))

    def system_types(self) -> list[str]:
        return self._types_in(self._system_root)

    def _user_root(self, owner: Owner) -> Path | None:
        root = self.user_sandbox(owner) if self.user_sandbox else None
        return root / "task-types" if root else None

    def user_types(self, owner: Owner) -> list[str]:
        """The owner's own types (their scope): they may share a name with a system type."""
        root = self._user_root(owner)
        return [t for t in self._types_in(root) if t != DEFAULT] if root else []

    def plugin_types(self, owner: Owner | None = None) -> dict[str, tuple[Path, bool]]:
        """`<scope>:<plugin>.<type>` → (directory, trusted), from the enabled plugins. System plugins
        are trusted (their task.yaml counts); a user's own plugins are not."""
        return self.plugins.task_type_dirs(owner)

    def check_user_type_name(self, name: str) -> None:
        """For the task-types API: a name in the caller's scope. `default` is reserved: it is what a
        task without a type runs as."""
        if not NAME_RE.match(name):
            raise TaskTypeError(f"invalid task type name {name!r}")
        if name == DEFAULT:
            raise TaskTypeError(f"{DEFAULT!r} is reserved")

    def _own_name(self, owner: Owner, ref: str) -> str:
        """The name in a reference to one of the owner's types (a bare name, or their own scope)."""
        scope, name = split_type_ref(ref)
        if scope is not None and scope != scope_of(owner):
            raise TaskTypeError("you can only change task types of your own")
        return name

    # -- the catalogue a user sees ----------------------------------------------------------------
    def catalogue(self, owner: Owner) -> list[TaskTypeInfo]:
        """System types, plugin types (system plugins' and the owner's), the owner's own types. Each
        has an id; `ref` is its bare name unless that name would resolve to another one."""
        def files(directory: Path) -> tuple[str, ...]:
            return tuple(n for n in INSTRUCTION_FILES if (directory / n).is_file())

        own = scope_of(owner)
        infos = [TaskTypeInfo(t, "system", files=files(self._system_root / t), settings=self._settings(t))
                 for t in self.system_types()]
        for type_ref, (directory, trusted) in sorted(self.plugin_types(owner).items()):
            scope, name = split_type_ref(type_ref)
            infos.append(TaskTypeInfo(name, "plugin", plugin=name.split(".", 1)[0], files=files(directory), scope=scope,
                                      settings=(self._settings_at(directory) if trusted else None) or self._settings(DEFAULT)))
        user_root = self._user_root(owner)
        for t in self.user_types(owner):
            infos.append(TaskTypeInfo(t, "user", files=files(user_root / t), settings=self._settings(DEFAULT),
                                      editable=True, scope=own))
        names = [i.name for i in infos]
        # A bare name resolves in the owner's scope first, so their own types keep it.
        return [dataclasses.replace(i, ref=i.name if names.count(i.name) == 1 or i.scope == own else i.id)
                for i in infos]

    def info(self, ref: str, owner: Owner) -> TaskTypeInfo | None:
        found = self._lookup(ref, owner) if TYPE_REF_RE.match(ref) else None
        return next((i for i in self.catalogue(owner) if i.id == found.id), None) if found else None

    # -- a user's own types -----------------------------------------------------------------------
    def write_user_type(self, owner: Owner, name: str, files: dict[str, str | None]) -> None:
        """Create or replace a user type. A file given as None (or empty) is removed: the type then
        inherits it from `default`."""
        name = self._own_name(owner, name)
        self.check_user_type_name(name)
        root = self._user_root(owner)
        if root is None:
            raise TaskTypeError("user task types are not enabled ([context] sandbox_root)")
        unknown = set(files) - set(INSTRUCTION_FILES)
        if unknown:
            raise TaskTypeError(f"unknown files: {', '.join(sorted(unknown))}")
        for content in files.values():
            if content and len(content.encode()) > self.max_file_bytes:
                raise TaskTypeError(f"a file exceeds {self.max_file_bytes} bytes")
        directory = root / name
        directory.mkdir(parents=True, exist_ok=True)
        for file_name, content in files.items():
            path = directory / file_name
            if content:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(content)
                tmp.replace(path)   # atomic: a running task never reads half a file
            elif path.exists():
                path.unlink()

    def delete_user_type(self, owner: Owner, name: str) -> bool:
        try:
            name = self._own_name(owner, name)
        except TaskTypeError:
            return False
        root = self._user_root(owner)
        if root is None or not NAME_RE.match(name) or name not in self.user_types(owner):
            return False
        directory = root / name
        for path in directory.iterdir():
            path.unlink()
        directory.rmdir()
        return True

    # -- resolution -------------------------------------------------------------------------------
    def _lookup(self, ref: str, owner: Owner | None) -> _Found | None:
        """The type a reference names: an id, or a bare name in the owner's scope, then the system's."""
        scope, name = split_type_ref(ref)
        scopes = [scope] if scope is not None else [scope_of(owner) if owner is not None else None, SYSTEM_SCOPE]
        for candidate in scopes:
            found = self._in_scope(candidate, name, owner) if candidate else None
            if found is not None:
                return found
        return None

    def _in_scope(self, scope: str, name: str, owner: Owner | None) -> _Found | None:
        type_ref = f"{scope}:{name}"
        if "." in name:   # <plugin>.<type>
            found = self.plugin_types(owner).get(type_ref)
            if found is None:
                return None
            directory, trusted = found
            return _Found(type_ref, scope, name, "plugin", [(f"plugin:{name}", directory)],
                          self._settings_at(directory) if trusted else None, not trusted)
        if scope == SYSTEM_SCOPE:
            if name not in self.system_types():
                return None
            return _Found(type_ref, scope, name, "system", [(f"system:{name}", self._system_root / name)],
                          self._settings(name), False)
        user_root = self._user_root(owner) if owner is not None else None
        if owner is None or scope != scope_of(owner) or user_root is None or name not in self.user_types(owner):
            return None   # other people's types are never visible
        return _Found(type_ref, scope, name, "user", [(f"user:{name}", user_root / name)], None, True)

    def resolve(self, task_type: str, owner: Owner) -> ResolvedTaskType:
        if not TYPE_REF_RE.match(task_type):
            raise TaskTypeError(f"invalid task type name {task_type!r}")
        found = self._lookup(task_type, owner)
        candidates = list(found.dirs) if found else []
        candidates.append((f"system:{DEFAULT}", self._system_root / DEFAULT))

        files = {name: self._first(name, candidates) for name in INSTRUCTION_FILES}
        return ResolvedTaskType(
            requested=task_type,
            agents=files["AGENTS.md"],
            workflow=files["WORKFLOW.md"],
            settings=(found.settings if found else None) or self._settings(DEFAULT),
            user_defined=bool(found and found.user_defined),
            fell_back=found is None,
            resolved_id=found.id if found else "",
        )

    def prompt(self, name: str, task_type: str | None = None, owner: Owner | None = None) -> InstructionFile:
        """A runtime prompt (prompts/<name>.md): the task type's own prompts/ for the per-type prompts,
        then <system dir>/prompts/, then the packaged default. A file that is too large is skipped."""
        if name not in PROMPTS:
            raise TaskTypeError(f"unknown prompt {name!r}")
        candidates: list[tuple[str, Path]] = []
        if task_type and owner is not None and name in TASK_PROMPTS and TYPE_REF_RE.match(task_type):
            found = self._lookup(task_type, owner)
            candidates += [(f"{label}/prompts", d / PROMPTS_DIR) for label, d in (found.dirs if found else ())]
        candidates += [("system:prompts", self.system_dir / PROMPTS_DIR), ("packaged:prompts", PACKAGED_PROMPTS)]
        for label, directory in candidates:
            path = directory / f"{name}.md"
            if not path.is_file():
                continue
            data = path.read_bytes()
            if len(data) > self.max_file_bytes:
                LOG.warning("%s/%s.md exceeds %d bytes; skipped", label, name, self.max_file_bytes)
                continue
            return InstructionFile(f"{name}.md", data.decode(), hashlib.sha256(data).hexdigest(), f"{label}/{name}.md")
        raise TaskTypeError(f"no {name}.md prompt found, not even the packaged one")

    def _first(self, name: str, candidates: list[tuple[str, Path]]) -> InstructionFile:
        for label, directory in candidates:
            path = directory / name
            if path.is_file():
                data = path.read_bytes()
                if len(data) > self.max_file_bytes:
                    raise TaskTypeError(f"{label}/{name} exceeds {self.max_file_bytes} bytes")
                return InstructionFile(name, data.decode(), hashlib.sha256(data).hexdigest(), f"{label}/{name}")
        raise TaskTypeError(f"no {name} found, not even in the default task type")

    def _settings_at(self, directory: Path) -> TaskTypeSettings | None:
        path = directory / "task.yaml"
        if path.is_file():
            return TaskTypeSettings.from_mapping(yaml.safe_load(path.read_text()) or {})
        return None

    def _settings(self, task_type: str) -> TaskTypeSettings:
        for candidate in (task_type, DEFAULT):
            found = self._settings_at(self._system_root / candidate)
            if found is not None:
                return found
        return TaskTypeSettings()
