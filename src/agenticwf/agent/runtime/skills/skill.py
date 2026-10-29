"""
A skill: instructions for one kind of work, which the model loads when it decides the task needs
them (docs/skills.md). The format is the common SKILL.md bundle:

    <skill>/SKILL.md
        ---
        name: release-notes                 lowercase letters, digits, hyphens; the directory's name
        description: Write release notes from merged pull requests. Use when ...   (required)
        tools: [github__list_pulls, read_file]   while the skill runs, only these (globs) and the
                                                 control tools; `allowed-tools` is read the same way.
                                                 Left out: the task's tools, matched to the skill
        approval: ask                       system and plugin skills only: ask before it loads
        tags: [docs, release]
        access: {organizations: [cern]}     system and plugin skills only
        ---
        The instructions (Markdown).
    <skill>/<other files>                   reference files the instructions name; the model reads
                                            them with read_skill_file while the skill runs

The model sees each skill it may use as a tool, `skill__<name>`, described by `description`.
Calling it is the model's proposal to use the skill: the dispatcher checks it with the policy
like any call, then the skill's instructions are loaded and the harness narrows the task's tools.
A skill acts on nothing itself.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenticwf.agent.runtime.tools.loader import ToolDefinitionError, core_tool_text, split_front_matter
from agenticwf.agent.runtime.tools.spec import Access, Approval, Effects, HandlerKind, HandlerRef, ToolResult, ToolSource, ToolSpec
from agenticwf.agent.runtime.types import Owner, scope_of

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
NAME_MAX = 64
DESCRIPTION_MAX = 1024
INSTRUCTIONS_MAX = 64 * 1024
SKILL_FILE = "SKILL.md"
TOOL_PREFIX = "skill__"
FILES_MAX = 100              # files of a skill listed to the model
FILE_BYTES_MAX = 256 * 1024  # largest file read_skill_file returns

# The tool every skill is offered as: runtime/defaults/core-tools/skill.md, ${description} the skill's.
_, _text, _TOOL_SCHEMA = core_tool_text("skill")
_TOOL_TEXT = string.Template(_text)

SOURCES = ("system", "plugin", "user", "database")
_TOOL_SOURCE = {"system": "system", "plugin": "plugin", "user": "user_skill", "database": "database"}


class SkillError(Exception):
    pass


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    instructions: str
    source: str                          # system | plugin | user (a sandbox or a user's plugin) | database
    owner: Owner | None = None           # user and database skills
    plugin: str | None = None            # the plugin that provides it: named <plugin>.<name>
    root: Path | None = field(default=None, compare=False)   # its directory; None for database skills
    tools: tuple[str, ...] = ()          # globs; empty: no narrowing
    approval: Approval = Approval.AUTO
    visibility: str = ""                 # database: user | organization | all; "" = the source's default
    access: Access = Access()
    tags: tuple[str, ...] = ()
    enabled: bool = True
    location: str = ""
    stored_files: tuple[str, ...] = ()   # a database skill's files (agent_skill_files), by path

    # --- identity ------------------------------------------------------------------------------
    @property
    def scope(self) -> str:
        return scope_of(self.owner)   # system and plugin skills: the system scope

    @property
    def qualified(self) -> str:
        """`<plugin>.<name>` for a plugin's skill, else the name."""
        return f"{self.plugin}.{self.name}" if self.plugin else self.name

    @property
    def id(self) -> str:
        return f"{self.scope}:{self.qualified}"

    @property
    def tool_name(self) -> str:
        """What the model calls it: `skill__<name>` (provider tool names: [a-zA-Z0-9_-], 64 at most)."""
        name = TOOL_PREFIX + self.qualified.replace(".", "_").replace("-", "_")
        if len(name) > 64:
            name = name[:57].rstrip("_") + "_" + hashlib.sha256(self.qualified.encode()).hexdigest()[:6]
        return name

    @property
    def tool_source(self) -> ToolSource:
        return ToolSource(_TOOL_SOURCE[self.source], self.location or f"skill:{self.id}", owner=self.owner,
                          visibility=self.visibility, access=self.access)

    def visible_to(self, owner: Owner) -> bool:
        return self.tool_source.visible_to(owner)

    def owned_by(self, owner: Owner | None) -> bool:
        return self.owner is not None and self.owner == owner

    # --- for the model -------------------------------------------------------------------------
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.tool_name, summary=_clip(self.description, 200),
            description=_TOOL_TEXT.safe_substitute(description=self.description), input_schema=_TOOL_SCHEMA,
            handler=HandlerRef(HandlerKind.SKILL, {"skill": self.id}),
            source=self.tool_source, scopes=self.access.scopes, effects=Effects.NONE, approval=self.approval,
            idempotent=True, tags=("skill", *self.name.split("-"), *self.tags),
        )

    def files(self) -> list[str]:
        """Its other files, relative to its directory, or a database skill's stored ones."""
        if self.root is None:
            return sorted(self.stored_files)[:FILES_MAX]
        if not self.root.is_dir():
            return []
        found = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file() and path.name != SKILL_FILE and not any(p.startswith(".") for p in path.relative_to(self.root).parts):
                found.append(path.relative_to(self.root).as_posix())
                if len(found) >= FILES_MAX:
                    break
        return found

    def read_file(self, relative: str) -> str:
        """One of its files on disk, as text. SkillError if it is not one of them. (A database
        skill's files are read through its store: SkillCatalog.read_file.)"""
        if self.root is None:
            raise SkillError("this skill keeps its files in the database")
        root = self.root.resolve()
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or path.name == SKILL_FILE:
            raise SkillError(f"{relative!r} is not a file of this skill")
        data = path.read_bytes()
        if len(data) > FILE_BYTES_MAX:
            raise SkillError(f"{relative!r} is larger than {FILE_BYTES_MAX // 1024} KB")
        try:
            return data.decode()
        except UnicodeDecodeError:
            raise SkillError(f"{relative!r} is not a text file")

    def activation(self) -> ToolResult:
        """What loading it gives the model: its instructions, and its files. `facts` tell the harness
        which skill now runs and how it narrows the tools."""
        files = self.files()
        text = f'<skill name="{self.qualified}">\n{self.instructions}\n</skill>'
        if files:
            text += "\n\nFiles of this skill (read one with read_skill_file): " + ", ".join(files)
        return ToolResult(content=text, summary=f"skill {self.qualified} loaded", instructions=True,
                          facts={"skill": {"id": self.id, "tool": self.tool_name, "tools": list(self.tools),
                                           "files": bool(files), "query": f"{self.description}\n{self.instructions}"}})

    # --- views ---------------------------------------------------------------------------------
    def to_dict(self, owner: Owner | None = None, detail: bool = False) -> dict[str, Any]:
        mine = self.owned_by(owner)
        group = ("mine" if mine else "shared") if self.owner is not None else self.source
        view = {"id": self.id, "name": self.name, "qualified": self.qualified, "tool": self.tool_name,
                "description": self.description, "source": self.source, "group": group, "plugin": self.plugin,
                "owner": self.owner[0] if self.owner and not mine else None,
                "visibility": self.tool_source.effective_visibility, "access": self.access.to_dict(),
                "tools": list(self.tools), "approval": self.approval.value, "tags": list(self.tags),
                "enabled": self.enabled, "editable": mine and self.source == "database"}
        if detail:
            view.update(instructions=self.instructions, files=self.files())
        return view


FILE_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$")


def check_file_path(path: str) -> str:
    """A file a database skill may keep: relative, no `..`, no hidden parts, not SKILL.md."""
    if (not FILE_PATH_RE.match(path) or len(path) > 255 or ".." in path.split("/")
            or path == SKILL_FILE):
        raise SkillError(f"invalid file path {path!r} (relative, letters, digits, '_', '-', '.', '/'; "
                         f"not {SKILL_FILE})")
    return path


def check_file_content(path: str, content: str) -> str:
    if len(content.encode()) > FILE_BYTES_MAX:
        raise SkillError(f"{path!r} is larger than {FILE_BYTES_MAX // 1024} KB")
    return content


def _clip(text: str, n: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[:n - 1].rstrip() + "…"


def check(skill: Skill) -> Skill:
    """A skill's own rules, wherever it comes from. SkillError: unusable."""
    if not NAME_RE.match(skill.name) or len(skill.name) > NAME_MAX:
        raise SkillError(f"invalid skill name {skill.name!r} (lowercase letters, digits and single hyphens, "
                         f"at most {NAME_MAX} characters)")
    if not skill.description.strip() or len(skill.description) > DESCRIPTION_MAX:
        raise SkillError(f"description is required, at most {DESCRIPTION_MAX} characters")
    if len(skill.instructions.encode()) > INSTRUCTIONS_MAX:
        raise SkillError(f"instructions are larger than {INSTRUCTIONS_MAX // 1024} KB")
    if not all(isinstance(t, str) and t for t in skill.tools):
        raise SkillError("tools must be a list of tool names (globs allowed)")
    if skill.source in ("user", "database") and skill.approval is not Approval.AUTO:
        return dataclasses.replace(skill, approval=Approval.AUTO)   # the policy decides for others' skills
    return skill


def _names(value: Any) -> tuple[str, ...]:
    """A list, or the `allowed-tools` style string "a, b c"."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(v for v in re.split(r"[\s,]+", value) if v)
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise SkillError("tools must be a list or a string of names")


def read_skill(directory: Path, source: str, owner: Owner | None = None, plugin: str | None = None,
               inherited: Access = Access()) -> Skill:
    """<directory>/SKILL.md. Settings that grant (approval, access) count for system and plugin skills only."""
    path = directory / SKILL_FILE
    try:
        meta, body = split_front_matter(path.read_text())
    except (OSError, UnicodeDecodeError, yaml.YAMLError, ToolDefinitionError) as exc:
        raise SkillError(f"{path}: {exc}") from exc
    if meta.get("name", directory.name) != directory.name:
        raise SkillError(f"{path}: name {meta.get('name')!r} does not match the directory {directory.name!r}")
    trusted = source in ("system", "plugin") and owner is None
    try:
        access = inherited.merge(Access.from_mapping(meta.get("access"))) if trusted else Access()
        approval = Approval(meta.get("approval", "auto")) if trusted else Approval.AUTO
        skill = Skill(
            name=directory.name, description=str(meta.get("description") or "").strip(), instructions=body,
            source=source, owner=owner, plugin=plugin, root=directory,
            tools=_names(meta.get("tools", meta.get("allowed-tools"))), approval=approval, access=access,
            tags=tuple(str(t) for t in meta.get("tags") or ()), enabled=meta.get("enabled", True) is not False,
            location=str(path))
        return check(skill)
    except (ValueError, TypeError, SkillError) as exc:
        raise SkillError(f"{path}: {exc}") from exc


def read_skills_dir(root: Path, source: str, owner: Owner | None = None, plugin: str | None = None,
                    inherited: Access = Access()) -> tuple[list[Skill], list[str]]:
    """<root>/<skill>/SKILL.md for each directory. Returns (skills, errors)."""
    skills, errors = [], []
    if not root.is_dir():
        return skills, errors
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not (directory / SKILL_FILE).is_file():
            continue
        try:
            skills.append(read_skill(directory, source, owner, plugin, inherited))
        except SkillError as exc:
            errors.append(str(exc))
    return skills, errors
