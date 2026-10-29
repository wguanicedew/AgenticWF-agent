"""
SkillCatalog: every skill, from four sources (docs/skills.md).

    system      <system dir>/skills/<skill>/SKILL.md                     everyone, unless `access` narrows it
    plugin      <plugin>/skills/<skill>/SKILL.md, named <plugin>.<skill>  as the plugin (agenticwf.agent.runtime.plugins)
    user        <user sandbox>/skills/<skill>/SKILL.md, and the skills    their owner
                of a user's own plugins
    database    agent_skills (agenticwf.agent.runtime.skills.db_store), added on the     their owner, the owner's
                admin page or through the API                            organization, or everyone

A skill's identity is `<scope>:<name>` (`system:release-notes`, `user_alice:release-notes`,
`system:github.triage`), so names only need to be unique per owner. The catalog is the
ToolRegistry's source of `skill__<name>` tools (tools_for), which go through visibility, policy
and selection like any tool, and the handler that loads a skill when the model calls one.
"""

from __future__ import annotations

import dataclasses
import threading
import time
from pathlib import Path
from typing import Callable, Protocol

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.skills.skill import (FILES_MAX, Skill, SkillError, check, check_file_content, check_file_path,
                                  read_skills_dir)
from agenticwf.agent.runtime.tools.spec import VISIBILITY, CallContext, Handler, ToolResult, ToolSpec
from agenticwf.agent.runtime.types import Owner, owner_key

LOG = get_logger(__name__)


class SkillStore(Protocol):
    """Database-held skills (agenticwf.agent.runtime.skills.db_store)."""

    def list(self) -> list[Skill]: ...
    def save(self, skill: Skill) -> None: ...
    def delete(self, skill_id: str) -> bool: ...
    def read_file(self, skill_id: str, path: str) -> str | None: ...
    def write_file(self, skill_id: str, path: str, content: str) -> None: ...
    def delete_file(self, skill_id: str, path: str) -> bool: ...


class SkillCatalog:
    def __init__(self, system_dir: Path | None = None, user_sandbox: Callable[[Owner], Path | None] | None = None,
                 plugins: PluginManager | None = None, store: SkillStore | None = None,
                 user_ttl: float = 5.0, refresh_seconds: float = 10.0):
        self.system_dir = system_dir
        self.user_sandbox = user_sandbox
        self.plugins = plugins
        self.store = store
        self.user_ttl = user_ttl
        self.refresh_seconds = refresh_seconds
        self._system: list[Skill] | None = None
        self._users: dict[Owner, tuple[float, list[Skill]]] = {}
        self._database: list[Skill] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()
        self.errors: list[str] = []

    # --- reading -------------------------------------------------------------------------------
    def load(self) -> list[str]:
        """(Re)read the system skills; plugin skills come with the plugins (PluginManager.load)."""
        skills, errors = read_skills_dir(self.system_dir / "skills", "system") if self.system_dir else ([], [])
        for error in errors:
            LOG.error("skill skipped: %s", error)
        with self._lock:
            self._system, self.errors = skills, errors
            self._users.clear()
        return errors

    def _system_skills(self) -> list[Skill]:
        if self._system is None:
            self.load()
        plugin = self.plugins.skills() if self.plugins else []
        return [*(self._system or []), *(k for k in plugin if k.owner is None)]

    def _user_skills(self, owner: Owner) -> list[Skill]:
        """The owner's sandbox skills, read again at most every `user_ttl` seconds, and their plugins'."""
        own_plugins = [k for k in (self.plugins.skills(owner) if self.plugins else []) if k.owner == owner]
        sandbox = self.user_sandbox(owner) if self.user_sandbox else None
        if sandbox is None:
            return own_plugins
        with self._lock:
            cached = self._users.get(owner)
        if cached and cached[0] + self.user_ttl > time.monotonic():
            return cached[1] + own_plugins
        skills, errors = read_skills_dir(sandbox / "skills", "user", owner=owner)
        for error in errors:
            LOG.warning("user skill skipped: %s", error)
        with self._lock:
            self._users[owner] = (time.monotonic(), skills)
        return skills + own_plugins

    def _database_skills(self, force: bool = False) -> list[Skill]:
        """Reloaded every `refresh_seconds` (other processes may have changed them)."""
        if self.store is None:
            return []
        if force or time.time() - self._loaded_at >= self.refresh_seconds:
            try:
                self._database = self.store.list()
            except Exception:   # keep serving the last good list
                LOG.exception("could not load skills from the database")
            self._loaded_at = time.time()
        return self._database

    # --- queries -------------------------------------------------------------------------------
    def skills(self, owner: Owner) -> list[Skill]:
        """Every skill `owner` may use, enabled or not: their own first, then the system's, then shared."""
        own = self._user_skills(owner)
        database = self._database_skills()
        mine = [k for k in database if k.owned_by(owner)]
        shared = [k for k in database if not k.owned_by(owner) and k.visible_to(owner)]
        return own + mine + [k for k in self._system_skills() if k.visible_to(owner)] + shared

    def get(self, skill_id: str, owner: Owner) -> Skill | None:
        return next((k for k in self.skills(owner) if k.id == skill_id), None)

    def tools_for(self, owner: Owner) -> list[ToolSpec]:
        """The ToolRegistry's source: one `skill__<name>` tool per enabled skill `owner` may use."""
        return [k.spec() for k in self.skills(owner) if k.enabled]

    def handler(self, spec: ToolSpec) -> Handler:
        """Loading the skill: its instructions as the call's result, for the caller only if they may use it."""
        skill_id = spec.handler.settings.get("skill")

        def load(call: CallContext, arguments: dict) -> ToolResult:
            skill = self.get(skill_id, owner_key(call.auth))
            if skill is None or not skill.enabled:
                return ToolResult.error("unknown_skill", f"The skill {skill_id} is not available.")
            return skill.activation()
        return load

    def read_file(self, skill_id: str, path: str, owner: Owner) -> str:
        """One of the skill's files: from its directory, or a database skill's from the database."""
        skill = self.get(skill_id, owner)
        if skill is None:
            raise SkillError("the skill is not available")
        if skill.root is not None:
            return skill.read_file(path)
        content = self.store.read_file(skill.id, path) if self.store is not None and path in skill.stored_files else None
        if content is None:
            raise SkillError(f"{path!r} is not a file of this skill")
        return content

    # --- database skills -----------------------------------------------------------------------
    def put(self, skill: Skill, by: Owner) -> Skill:
        """Add or change a database skill of `by`'s. PermissionError: someone else's; SkillError: invalid."""
        if self.store is None:
            raise PermissionError("this runtime has no store for skills")
        if skill.source != "database" or skill.owner != by:
            raise PermissionError("only your own database skills can be changed")
        if not skill.visibility:   # a database skill is its owner's until they share it
            skill = dataclasses.replace(skill, visibility="user")
        if skill.visibility not in VISIBILITY:
            raise SkillError(f"visibility must be one of {', '.join(VISIBILITY)}")
        skill = check(skill)
        self.store.save(skill)
        self._database_skills(force=True)
        return skill

    def delete(self, skill_id: str, by: Owner) -> bool:
        current = next((k for k in self._database_skills(force=True) if k.id == skill_id), None)
        if current is None or not current.owned_by(by):
            return False
        deleted = self.store.delete(skill_id)
        self._database_skills(force=True)
        return deleted

    def put_file(self, skill_id: str, path: str, content: str, by: Owner) -> None:
        """Add or replace a file of one of `by`'s database skills (in the database, never on disk)."""
        current = self.database_skill(skill_id)
        if current is None or not current.owned_by(by):
            raise KeyError(skill_id)
        check_file_path(path)
        check_file_content(path, content)
        if path not in current.stored_files and len(current.stored_files) >= FILES_MAX:
            raise SkillError(f"a skill keeps at most {FILES_MAX} files")
        self.store.write_file(skill_id, path, content)
        self._database_skills(force=True)

    def delete_file(self, skill_id: str, path: str, by: Owner) -> bool:
        current = self.database_skill(skill_id)
        if current is None or not current.owned_by(by):
            return False
        deleted = self.store.delete_file(skill_id, path)
        self._database_skills(force=True)
        return deleted

    def database_skill(self, skill_id: str) -> Skill | None:
        return next((k for k in self._database_skills(force=True) if k.id == skill_id), None)

