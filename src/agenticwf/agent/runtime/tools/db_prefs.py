"""DatabaseToolPrefs: the tools, plugins and MCP endpoints users switch off for their tasks, on
agenticwf.agent.core.orm (agent_tool_prefs; agenticwf.agent.runtime.tools.prefs).

The dispatcher asks for them at every step, so each owner's set is kept for `ttl` seconds. A change
made in this process applies at once; one made in another process (worker processes) within `ttl`.
"""

from __future__ import annotations

import hashlib
import threading
import time

from sqlalchemy import select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import ToolPrefRow
from agenticwf.agent.runtime.types import Owner


class DatabaseToolPrefs:
    def __init__(self, ttl: float = 10.0):
        self.ttl = ttl
        self._cache: dict[Owner, tuple[float, frozenset[str]]] = {}
        self._lock = threading.Lock()

    def disabled_for(self, owner: Owner) -> frozenset[str]:
        owner = tuple(owner)
        with self._lock:
            cached = self._cache.get(owner)
        if cached is not None and time.monotonic() - cached[0] < self.ttl:
            return cached[1]
        off = frozenset(_disabled(owner))
        with self._lock:
            self._cache[owner] = (time.monotonic(), off)
        return off

    def set_enabled(self, owner: Owner, target: str, enabled: bool) -> None:
        owner = tuple(owner)
        _set_pref(owner, target, enabled)
        with self._lock:
            self._cache.pop(owner, None)


def _pref_id(owner: Owner, target: str) -> str:
    return hashlib.sha256("\0".join((owner[0], owner[1] or "", target)).encode()).hexdigest()


@read_session
def _disabled(owner: Owner, *, session=None) -> set[str]:
    subject, org = owner
    query = select(ToolPrefRow.target).where(ToolPrefRow.owner_subject == subject)
    query = query.where(ToolPrefRow.owner_org.is_(None) if org is None else ToolPrefRow.owner_org == org)
    return set(session.execute(query).scalars())


@transactional_session
def _set_pref(owner: Owner, target: str, enabled: bool, *, session=None) -> None:
    pref_id = _pref_id(owner, target)
    row = session.get(ToolPrefRow, pref_id)
    if enabled and row is not None:
        session.delete(row)
    elif not enabled and row is None:
        session.add(ToolPrefRow(id=pref_id, owner_subject=owner[0], owner_org=owner[1], target=target))
