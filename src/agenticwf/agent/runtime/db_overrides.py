"""DatabaseSettingOverrides: the settings admins and people changed on the admin page
(agenticwf.agent.runtime.overrides), in agent_setting_overrides.

The harness reads them for every task, so the whole table (a few rows) is kept for `ttl` seconds.
A change made in this process applies at once; one made in another process (worker processes)
within `ttl`.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from typing import Any

from sqlalchemy import select

from agenticwf.agent.core.orm.base.models import utcnow
from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import SettingOverrideRow
from agenticwf.agent.runtime.types import Owner

_Rows = dict[tuple[Owner | None, str], dict[str, Any]]


class DatabaseSettingOverrides:
    def __init__(self, ttl: float = 10.0):
        self.ttl = ttl
        self._cache: tuple[float, _Rows] | None = None
        self._lock = threading.Lock()

    def values(self, section: str, owner: Owner | None = None) -> dict[str, Any]:
        with self._lock:
            cached = self._cache
        if cached is None or time.monotonic() - cached[0] >= self.ttl:
            cached = (time.monotonic(), _load())
            with self._lock:
                self._cache = cached
        return dict(cached[1].get((tuple(owner) if owner else None, section), {}))

    def set(self, section: str, key: str, value: Any, owner: Owner | None = None, by: str = "") -> None:
        _set(tuple(owner) if owner else None, section, key, value, by)
        with self._lock:
            self._cache = None


def _row_id(owner: Owner | None, section: str, key: str) -> str:
    who = "\0".join((owner[0], owner[1] or "")) if owner else ""
    return hashlib.sha256("\0".join((who, section, key)).encode()).hexdigest()


@read_session
def _load(*, session=None) -> _Rows:
    rows: _Rows = {}
    for row in session.execute(select(SettingOverrideRow)).scalars():
        owner = (row.owner_subject, row.owner_org) if row.owner_subject is not None else None
        try:
            rows.setdefault((owner, row.section), {})[row.key] = json.loads(row.value)
        except ValueError:
            continue
    return rows


@transactional_session
def _set(owner: Owner | None, section: str, key: str, value: Any, by: str, *, session=None) -> None:
    row = session.get(SettingOverrideRow, _row_id(owner, section, key))
    if value is None:
        if row is not None:
            session.delete(row)
        return
    if row is None:
        row = SettingOverrideRow(id=_row_id(owner, section, key), owner_subject=owner[0] if owner else None,
                                 owner_org=owner[1] if owner else None, section=section, key=key)
        session.add(row)
    row.value, row.updated_by, row.updated_at = json.dumps(value), by or None, utcnow()
