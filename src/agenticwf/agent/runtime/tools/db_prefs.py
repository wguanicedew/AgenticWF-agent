"""DatabaseToolPrefs: the tools, plugins and MCP endpoints users switch off, or to ask first, for their tasks, on
agenticwf.agent.core.orm (agent_tool_prefs; agenticwf.agent.runtime.tools.prefs), and the tool
timeouts an admin set (agent_tool_settings).

The dispatcher asks for them at every step, so each owner's set is kept for `ttl` seconds. A change
made in this process applies at once; one made in another process (worker processes) within `ttl`.
"""

from __future__ import annotations

import hashlib
import threading
import time

from sqlalchemy import select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.base.models import utcnow
from agenticwf.agent.core.orm.models.agent import ToolPrefRow, ToolSettingRow
from agenticwf.agent.runtime.tools.prefs import MODES
from agenticwf.agent.runtime.types import Owner


class DatabaseToolPrefs:
    def __init__(self, ttl: float = 10.0):
        self.ttl = ttl
        self._cache: dict[Owner, tuple[float, dict[str, str]]] = {}
        self._timeouts: tuple[float, dict[str, float]] | None = None
        self._lock = threading.Lock()

    def modes_for(self, owner: Owner) -> dict[str, str]:
        owner = tuple(owner)
        with self._lock:
            cached = self._cache.get(owner)
        if cached is not None and time.monotonic() - cached[0] < self.ttl:
            return dict(cached[1])
        modes = _modes(owner)
        with self._lock:
            self._cache[owner] = (time.monotonic(), modes)
        return dict(modes)

    def disabled_for(self, owner: Owner) -> frozenset[str]:
        return frozenset(t for t, mode in self.modes_for(owner).items() if mode == "off")

    def set_mode(self, owner: Owner, target: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}")
        owner = tuple(owner)
        _set_pref(owner, target, mode)
        with self._lock:
            self._cache.pop(owner, None)

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        self.set_mode(owner, target, "on" if enabled else "off")

    def timeouts(self) -> dict[str, float]:
        with self._lock:
            cached = self._timeouts
        if cached is not None and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        found = _timeouts()
        with self._lock:
            self._timeouts = (time.monotonic(), found)
        return found

    def set_timeout(self, name: str, seconds: float | None, by: str = "") -> None:
        _set_timeout(name, seconds, by)
        with self._lock:
            self._timeouts = None


def _pref_id(owner: Owner, target: str) -> str:
    return hashlib.sha256("\0".join((owner[0], owner[1] or "", target)).encode()).hexdigest()


@read_session
def _modes(owner: Owner, *, session=None) -> dict[str, str]:
    subject, org = owner
    query = select(ToolPrefRow.target, ToolPrefRow.mode).where(ToolPrefRow.owner_subject == subject)
    query = query.where(ToolPrefRow.owner_org.is_(None) if org is None else ToolPrefRow.owner_org == org)
    return {target: mode or "off" for target, mode in session.execute(query)}   # NULL: a row from before modes


@transactional_session
def _set_pref(owner: Owner, target: str, mode: str, *, session=None) -> None:
    pref_id = _pref_id(owner, target)
    row = session.get(ToolPrefRow, pref_id)
    if mode == "on":
        if row is not None:
            session.delete(row)
        return
    if row is None:
        row = ToolPrefRow(id=pref_id, owner_subject=owner[0], owner_org=owner[1], target=target)
        session.add(row)
    row.mode = mode


@read_session
def _timeouts(*, session=None) -> dict[str, float]:
    rows = session.execute(select(ToolSettingRow.name, ToolSettingRow.timeout).where(ToolSettingRow.timeout.is_not(None)))
    return {name: float(timeout) for name, timeout in rows}


@transactional_session
def _set_timeout(name: str, seconds: float | None, by: str, *, session=None) -> None:
    row = session.get(ToolSettingRow, name)
    if seconds is None:
        if row is not None:
            session.delete(row)
        return
    if row is None:
        row = ToolSettingRow(name=name)
        session.add(row)
    row.timeout, row.updated_by, row.updated_at = float(seconds), by or None, utcnow()
