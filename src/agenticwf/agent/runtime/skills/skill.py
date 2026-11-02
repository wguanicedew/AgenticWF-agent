"""
A skill: instructions for one kind of work, which the model loads when it decides the task needs
them (docs/skills.md). The format is the common SKILL.md bundle:

    <skill>/SKILL.md
        ---
        name: release-notes                 lowercase letters, digits, hyphens; the directory's name
        description: Write release notes from merged pull requests. Use when ...   (required)
        tools: [github__list_pulls, read_file]   while the skill runs, only these (globs) and the
                                                 control tools. Left out: the task's tools, matched
                                                 to the skill
        allowed-tools: Read, github__*      pre-approved while the skill runs (Claude Code): their
                                            own `ask` is waived; never a policy rule, never a tool
                                            users defined, never by someone else's shared skill
        approval: ask                       system and plugin skills only: ask before it loads
        tags: [docs, release]
        access: {organizations: [cern]}     system and plugin skills only
        disable-model-invocation: true      listed, never offered to the model (Claude Code)
        ---
        The instructions (Markdown).
    <skill>/<other files>                   reference files the instructions name; the model reads
                                            them with read_skill_file while the skill runs

Skills written for Claude Code and Codex work unchanged (docs/skills.md#claude-code-and-codex):
their tool names in `tools` and `allowed-tools` are translated when the skill loads (Read ->
read_file, mcp__<server>__<tool> -> <server>__<tool>; [skills] tool_aliases adds to the table),
${CLAUDE_SKILL_DIR}/ paths become the skill's own relative paths, $ARGUMENTS is
the model's `purpose`, Codex's agents/openai.yaml `policy.allow_implicit_invocation: false` acts
like `disable-model-invocation`, and a plugin's commands/<command>.md is read as a skill.

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
SKILL_ROOTS = ("skills", ".claude/skills", ".codex/skills", ".agents/skills")   # under a system dir or sandbox
CODEX_METADATA = "agents/openai.yaml"
TOOL_PREFIX = "skill__"
FILES_MAX = 100              # files of a skill listed to the model
FILE_BYTES_MAX = 256 * 1024  # largest file read_skill_file returns

# The tool every skill is offered as: runtime/defaults/core-tools/skill.md, ${description} the skill's.
_, _text, _TOOL_SCHEMA = core_tool_text("skill")
_TOOL_TEXT = string.Template(_text)

SOURCES = ("system", "plugin", "user", "database")
_TOOL_SOURCE = {"system": "system", "plugin": "plugin", "user": "user_skill", "database": "database"}


# Claude Code's and Codex's own tool names in `allowed-tools`, and the tools that do that work here.
# A name with no counterpart (Task, Skill) is dropped; when none is left, the skill narrows nothing.
TOOL_ALIASES = {
    "Read": ("read_file",), "Write": ("edit_file",), "Edit": ("edit_file",), "MultiEdit": ("edit_file",),
    "NotebookEdit": ("edit_file",), "Grep": ("search_code",), "Glob": ("search_code",), "LS": ("search_code",),
    "Bash": ("run_command",), "WebFetch": ("web_fetch",), "WebSearch": ("web_search",),
    "TodoWrite": ("update_plan",), "AskUserQuestion": ("ask_user",),
    "shell": ("run_command",), "local_shell": ("run_command",), "apply_patch": ("edit_file",),
    "view_image": ("read_file",),
}
_AGENT_TOOL_RE = re.compile(r"^[a-z0-9_*?\[\]]+$")   # a tool name here, or a glob of them
_PATH_VARS = re.compile(r"\$\{(?:CLAUDE_SKILL_DIR|SKILL_DIR)\}/?")


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
    tools: tuple[str, ...] = ()          # globs, as written; empty: no narrowing
    approved: tuple[str, ...] = ()       # allowed-tools, as written: pre-approved while it runs
    approval: Approval = Approval.AUTO
    visibility: str = ""                 # database: user | organization | all; "" = the source's default
    access: Access = Access()
    tags: tuple[str, ...] = ()
    enabled: bool = True
    location: str = ""
    stored_files: tuple[str, ...] = ()   # a database skill's files (agent_skill_files), by path
    model_invocable: bool = True         # false: listed, never offered (disable-model-invocation)

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

    def activation(self, arguments: dict[str, Any] | None = None, aliases: dict[str, tuple[str, ...]] | None = None,
                   preapprove: bool = True) -> ToolResult:
        """What loading it gives the model: its instructions, and its files. `facts` tell the harness
        which skill now runs, how it narrows the tools and which it pre-approves (`preapprove`: the
        caller may take this skill's word for it). $ARGUMENTS (Claude Code) is the `purpose`."""
        files = self.files()
        aliases = TOOL_ALIASES if aliases is None else aliases
        tools = translate_tools(self.tools, self.plugin, aliases)
        approved = translate_tools(self.approved, self.plugin, aliases) if preapprove else ()
        instructions = self.instructions.replace("$ARGUMENTS", str((arguments or {}).get("purpose") or "").strip())
        text = f'<skill name="{self.qualified}">\n{instructions}\n</skill>'
        if files:
            text += "\n\nFiles of this skill (read one with read_skill_file): " + ", ".join(files)
        return ToolResult(content=text, summary=f"skill {self.qualified} loaded", instructions=True,
                          facts={"skill": {"id": self.id, "tool": self.tool_name, "tools": list(tools),
                                           "approved": list(approved), "files": bool(files),
                                           "query": f"{self.description}\n{self.instructions}"}})

    # --- views ---------------------------------------------------------------------------------
    def to_dict(self, owner: Owner | None = None, detail: bool = False) -> dict[str, Any]:
        mine = self.owned_by(owner)
        group = ("mine" if mine else "shared") if self.owner is not None else self.source
        view = {"id": self.id, "name": self.name, "qualified": self.qualified, "tool": self.tool_name,
                "description": self.description, "source": self.source, "group": group, "plugin": self.plugin,
                "owner": self.owner[0] if self.owner and not mine else None,
                "visibility": self.tool_source.effective_visibility, "access": self.access.to_dict(),
                "tools": list(self.tools), "approved": list(self.approved), "approval": self.approval.value, "tags": list(self.tags),
                "enabled": self.enabled, "model_invocable": self.model_invocable,
                "editable": mine and self.source == "database"}
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
    if not all(isinstance(t, str) and t for t in (*skill.tools, *skill.approved)):
        raise SkillError("tools must be a list of tool names (globs allowed)")
    if skill.source in ("user", "database") and skill.approval is not Approval.AUTO:
        return dataclasses.replace(skill, approval=Approval.AUTO)   # the policy decides for others' skills
    return skill


def _names(value: Any) -> tuple[str, ...]:
    """A list, or the `allowed-tools` style string "a, b c"."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(re.findall(r"[^\s,(]+(?:\([^)]*\))?", value))   # Bash(git diff:*) is one name
    if isinstance(value, (list, tuple)):
        return tuple(str(v) for v in value)
    raise SkillError("tools must be a list or a string of names")


def parse_aliases(text: str) -> dict[str, tuple[str, ...]]:
    """[skills] tool_aliases: `Name: tool tool, Name: …` (commas or new lines between entries; no
    tools: the name is dropped). Added to TOOL_ALIASES, replacing an entry of the same name."""
    aliases = dict(TOOL_ALIASES)
    for entry in re.split(r"[,\n]", text or ""):
        if entry.strip():
            name, sep, tools = entry.partition(":")
            if not sep or not name.strip():
                raise ValueError(f"tool_aliases: {entry.strip()!r} is not `Name: tool …`")
            aliases[name.strip()] = tuple(tools.split())
    return aliases


def translate_tools(names: tuple[str, ...], plugin: str | None = None,
                    aliases: dict[str, tuple[str, ...]] | None = None) -> tuple[str, ...]:
    """Tool names as Claude Code and Codex write them, as tools here: `Read` -> read_file,
    `Bash(git diff:*)` -> run_command (the command pattern is not kept), `mcp__github__list_issues`
    -> github__list_issues, and a plugin's own server `mcp__plugin_<plugin>_<server>__<tool>` ->
    <plugin>__<tool>. Names already in this agent's form stay. Empty when nothing is left: the skill
    then narrows nothing."""
    aliases = TOOL_ALIASES if aliases is None else aliases
    out: list[str] = []
    for name in names:
        name = name.split("(", 1)[0].strip()
        if name.startswith("mcp__") and "__" in name[5:]:
            server, _, tool = name[5:].partition("__")
            server = server.replace("-", "_").lower()
            if plugin and server.startswith(f"plugin_{plugin}_"):
                server = plugin
            translated: tuple[str, ...] = (f"{server}__{tool}",)
        else:
            translated = aliases.get(name, (name,))
        out.extend(t for t in translated if _AGENT_TOOL_RE.match(t) and t not in out)
    return tuple(out)


def _model_invocable(meta: dict[str, Any], directory: Path | None) -> bool:
    """`disable-model-invocation: true` (Claude Code), or Codex's agents/openai.yaml
    `policy: {allow_implicit_invocation: false}`: the skill is not offered to the model."""
    if meta.get("disable-model-invocation") is True:
        return False
    path = directory / CODEX_METADATA if directory else None
    if path is not None and path.is_file():
        try:
            policy = (yaml.safe_load(path.read_text()) or {}).get("policy") or {}
        except (OSError, UnicodeDecodeError, yaml.YAMLError, AttributeError) as exc:
            raise SkillError(f"{path}: {exc}") from exc
        if isinstance(policy, dict) and policy.get("allow_implicit_invocation") is False:
            return False
    return True


def _instructions(body: str, name: str) -> str:
    """${CLAUDE_SKILL_DIR}/x and ${CLAUDE_PLUGIN_ROOT}/skills/<name>/x name the skill's file x, which
    the model reads with read_skill_file by its relative path."""
    body = body.replace(f"${{CLAUDE_PLUGIN_ROOT}}/skills/{name}/", "")
    return _PATH_VARS.sub("", body)


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
            name=directory.name, description=str(meta.get("description") or "").strip(),
            instructions=_instructions(body, directory.name), source=source, owner=owner, plugin=plugin,
            root=directory, tools=_names(meta.get("tools")), approved=_names(meta.get("allowed-tools")),
            approval=approval, access=access, tags=tuple(str(t) for t in meta.get("tags") or ()),
            enabled=meta.get("enabled", True) is not False, location=str(path),
            model_invocable=_model_invocable(meta, directory))
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


def read_skill_roots(base: Path, source: str, owner: Owner | None = None) -> tuple[list[Skill], list[str]]:
    """Skills under a system directory or a sandbox: <base>/skills/, and the directories Claude Code
    and Codex keep them in (.claude/skills/, .codex/skills/, .agents/skills/). A name found twice:
    the first wins, the other is reported."""
    skills: list[Skill] = []
    errors: list[str] = []
    for relative in SKILL_ROOTS:
        found, problems = read_skills_dir(base / relative, source, owner)
        errors.extend(problems)
        for skill in found:
            if any(k.name == skill.name for k in skills):
                errors.append(f"{skill.location}: skill {skill.name!r} already read from another directory")
            else:
                skills.append(skill)
    return skills, errors


def read_commands_dir(root: Path, source: str, owner: Owner | None = None, plugin: str | None = None,
                      inherited: Access = Access()) -> tuple[list[Skill], list[str]]:
    """A Claude Code plugin's commands/<command>.md (or commands/<group>/<command>.md, named
    <group>-<command>; or one such file), read as skills: the front matter is optional, `description` defaults to the
    first line of the body. A command has no other files. Returns (skills, errors)."""
    skills, errors = [], []
    if not root.is_dir() and not (root.is_file() and root.suffix == ".md"):
        return skills, errors
    trusted = source in ("system", "plugin") and owner is None
    for path in sorted(root.rglob("*.md")) if root.is_dir() else [root]:
        relative = path.relative_to(root if root.is_dir() else root.parent).with_suffix("")
        if any(p.startswith(".") for p in relative.parts):
            continue
        name = "-".join(relative.parts).lower().replace("_", "-")
        try:
            text = path.read_text()
            meta, body = split_front_matter(text) if text.startswith("---\n") else ({}, text.strip())
            first = next((line.strip("# ").strip() for line in body.splitlines() if line.strip()), "")
            skill = Skill(
                name=name, description=str(meta.get("description") or first).strip()[:DESCRIPTION_MAX],
                instructions=body.replace("${CLAUDE_PLUGIN_ROOT}/", ""), source=source, owner=owner,
                plugin=plugin, tools=_names(meta.get("tools")), approved=_names(meta.get("allowed-tools")),
                approval=Approval(meta.get("approval", "auto")) if trusted else Approval.AUTO,
                access=inherited if trusted else Access(), tags=("command",), location=str(path),
                model_invocable=_model_invocable(meta, None))
            skills.append(check(skill))
        except (OSError, UnicodeDecodeError, yaml.YAMLError, ToolDefinitionError, ValueError, TypeError,
                SkillError) as exc:
            errors.append(f"{path}: {exc}")
    return skills, errors
