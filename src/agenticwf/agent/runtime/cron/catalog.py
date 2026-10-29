"""
CronCatalog: every cron, from four sources (docs/cron.md §2), and who may do what with each.

    system      <system dir>/crons/<cron>/                  runs as cron:<name>
    plugin      <plugin>/crons/<cron>/, <plugin>.<cron>     runs as cron:<plugin>.<cron>
    user        <user sandbox>/crons/<cron>/, and the       runs as its owner, once they enabled it
                crons of a user's own plugins
    database    agent_crons + agent_cron_files              runs as its owner

Roles (role()): `operate` (see everything, tick the boxes), `view` (see everything, no boxes:
outside `operators`, or a machine token), `read` (reports and the board only: `readers`).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.cron import store
from agenticwf.agent.runtime.cron.definition import Cron, CronError, parse_cron, read_crons_dir
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.tools.spec import Access, ToolSource
from agenticwf.agent.runtime.types import Owner, owner_key

LOG = get_logger(__name__)
ROLES = ("operate", "view", "read")


def permits(access: Access, ctx: AuthContext) -> bool:
    return access.allows(owner_key(ctx)) and access.scopes <= ctx.scopes


def role(cron: Cron, ctx: AuthContext) -> str | None:
    """What `ctx` may do with `cron`: operate, view, read, or nothing (None)."""
    if cron.trusted:
        if permits(cron.access, ctx):
            if ctx.is_machine or (cron.operators is not None and not permits(cron.operators, ctx)):
                return "view"
            return "operate"
        if cron.readers is not None and permits(cron.readers, ctx):
            return "read"
        return None
    owner = owner_key(ctx)
    shared = cron.source == "database" and ToolSource("database", owner=cron.owner,
                                                       visibility=cron.visibility or "user").visible_to(owner)
    if cron.owner == owner or shared:
        return "view" if ctx.is_machine else "operate"
    return None


class CronCatalog:
    def __init__(self, system_dir: Path | None = None, user_sandbox: Callable[[Owner], Path | None] | None = None,
                 plugins: PluginManager | None = None, use_database: bool = True, user_ttl: float = 5.0,
                 refresh_seconds: float = 10.0):
        self.system_dir = system_dir
        self.user_sandbox = user_sandbox
        self.plugins = plugins
        self.use_database = use_database
        self.user_ttl = user_ttl
        self.refresh_seconds = refresh_seconds
        self._system: list[Cron] | None = None
        self._users: dict[Owner, tuple[float, list[Cron]]] = {}
        self._database: list[Cron] = []
        self._loaded_at = 0.0
        self._lock = threading.Lock()
        self.errors: list[str] = []
        self.database_errors: dict[str, str] = {}

    # --- sources -------------------------------------------------------------------------------
    def load(self) -> list[str]:
        crons, errors = read_crons_dir(self.system_dir / "crons", "system") if self.system_dir else ([], [])
        for error in errors:
            LOG.error("cron skipped: %s", error)
        with self._lock:
            self._system, self.errors = crons, errors
            self._users.clear()
        return errors

    def _system_crons(self) -> list[Cron]:
        if self._system is None:
            self.load()
        plugin = [c for c in (self.plugins.crons() if self.plugins else []) if c.owner is None]
        return [*(self._system or []), *plugin]

    def user_crons(self, owner: Owner) -> list[Cron]:
        """The owner's sandbox crons (read again at most every `user_ttl` seconds) and their plugins'."""
        own_plugins = [c for c in (self.plugins.crons(owner) if self.plugins else []) if c.owner == owner]
        sandbox = self.user_sandbox(owner) if self.user_sandbox else None
        if sandbox is None:
            return own_plugins
        with self._lock:
            cached = self._users.get(owner)
        if cached and cached[0] + self.user_ttl > time.monotonic():
            return cached[1] + own_plugins
        crons, errors = read_crons_dir(sandbox / "crons", "user", owner=owner)
        for error in errors:
            LOG.warning("user cron skipped: %s", error)
        with self._lock:
            self._users[owner] = (time.monotonic(), crons)
        return crons + own_plugins

    def database_crons(self, force: bool = False) -> list[Cron]:
        if not self.use_database:
            return []
        if force or time.time() - self._loaded_at >= self.refresh_seconds:
            crons, errors = [], {}
            try:
                for row, files in store.database_crons():
                    owner = (row["owner_subject"], row["owner_org"])
                    name = row["id"].rpartition(":")[2]
                    try:
                        crons.append(parse_cron(name, files, "database", owner=owner,
                                                visibility=row["visibility"] or "user", location=f"database:{row['id']}"))
                    except CronError as exc:
                        errors[row["id"]] = str(exc)
                self._database, self.database_errors = crons, errors
            except Exception:   # keep the last good list
                LOG.exception("could not load crons from the database")
            self._loaded_at = time.time()
        return self._database

    # --- queries -------------------------------------------------------------------------------
    def scheduled(self) -> list[Cron]:
        """Every cron the scheduler runs (enabled or not; it checks): system, plugins, the
        database, and the sandbox crons their owners enabled."""
        crons = [*self._system_crons(), *self.database_crons()]
        if self.use_database:
            for row in store.enabled_user_crons():
                owner = (row["owner_subject"], row["owner_org"])
                crons += [c for c in self.user_crons(owner) if c.id == row["id"]]
        return crons

    def visible(self, ctx: AuthContext) -> list[tuple[Cron, str]]:
        owner = owner_key(ctx)
        candidates = [*self.user_crons(owner), *self._system_crons(), *self.database_crons()]
        seen, found = set(), []
        for cron in candidates:
            if cron.id in seen:
                continue
            seen.add(cron.id)
            r = role(cron, ctx)
            if r is not None:
                found.append((cron, r))
        return found

    def get(self, cron_id: str, ctx: AuthContext | None = None) -> tuple[Cron, str] | None:
        """A cron by id: with `ctx`, only one it may see (and its role); without, any scheduled one."""
        if ctx is None:
            cron = next((c for c in self.scheduled() if c.id == cron_id), None)
            return (cron, "operate") if cron else None
        return next(((c, r) for c, r in self.visible(ctx) if c.id == cron_id), None)

    def find(self, cron_id: str) -> Cron | None:
        """Any cron by id, including a sandbox cron that is not enabled (the scheduler's view)."""
        found = self.get(cron_id)
        if found:
            return found[0]
        return next((c for c in self.database_crons(force=True) if c.id == cron_id), None)
