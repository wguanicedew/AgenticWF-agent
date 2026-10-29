"""DatabaseKnowledgeStore: the knowledge services users connect, the knowledge bases they switch
off, and the search options they choose, on agenticwf.agent.core.orm (agent_knowledge_services,
agent_knowledge_prefs, agent_knowledge_options).

Tokens are encrypted with a SecretBox (agenticwf.agent.runtime.secrets) and decrypted only in this
process; the row keeps a hint for the admin view. Without a SecretBox, saving a token is refused.
"""

from __future__ import annotations

import hashlib
import threading
import time

from sqlalchemy import delete, select

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import KnowledgeOptionRow, KnowledgePrefRow, KnowledgeServiceRow
from agenticwf.agent.runtime.knowledge import KnowledgeService
from agenticwf.agent.runtime.secrets import SecretBox, SecretsUnavailable, hint
from agenticwf.agent.runtime.types import Owner

LOG = get_logger(__name__)
OPTIONS_TTL = 10.0


class DatabaseKnowledgeStore:
    def __init__(self, box: SecretBox | None):
        self.box = box
        self._options: dict[Owner, tuple[float, dict[str, str]]] = {}
        self._lock = threading.Lock()

    @property
    def can_store_tokens(self) -> bool:
        return self.box is not None

    def services_for(self, owner: Owner) -> list[KnowledgeService]:
        out = []
        for row in _service_rows(tuple(owner)):
            token, unreadable = "", False
            if row["token_encrypted"]:
                try:
                    if self.box is None:
                        raise SecretsUnavailable("no secrets key is configured")
                    token = self.box.decrypt(row["token_encrypted"])
                except SecretsUnavailable:
                    unreadable = True
                    LOG.error("knowledge service %s: its token cannot be decrypted (secrets key missing or changed?)",
                              row["id"])
            out.append(KnowledgeService(
                name=row["name"], owner=(row["owner_subject"], row["owner_org"]), url=row["url"],
                base_path=row["base_path"], label=row["label"] or "", token=token, token_hint=row["token_hint"] or "",
                enabled=bool(row["enabled"]), token_unreadable=unreadable))
        return out

    def save_service(self, service: KnowledgeService, keep_token: bool = False) -> None:
        """`keep_token`: leave the stored (encrypted) token as it is."""
        if not keep_token and service.token and self.box is None:
            raise PermissionError("no secrets key is configured (AGENT_SECRETS_KEY_FILE): a token cannot be stored "
                                  "in the database")
        encrypted = None if keep_token or not service.token else self.box.encrypt(service.token)
        _save_service(service, encrypted, None if keep_token or not service.token else hint(service.token), keep_token)

    def delete_service(self, owner: Owner, service_id: str) -> bool:
        return _delete_service(tuple(owner), service_id)

    def disabled_for(self, owner: Owner) -> set[tuple[str, str]]:
        """(source, kb) pairs the owner switched off; kb "*" is the whole source."""
        return _disabled(tuple(owner))

    def set_enabled(self, owner: Owner, source: str, kb: str, enabled: bool) -> None:
        _set_pref(tuple(owner), source, kb, enabled)

    def options_for(self, owner: Owner) -> dict[str, str]:
        """The `tools` and `research_preset` the owner chose; missing: agent.cfg's. The dispatcher asks
        at every step, so each owner's are kept for OPTIONS_TTL seconds (another process sees a change
        within that time)."""
        owner = tuple(owner)
        with self._lock:
            cached = self._options.get(owner)
        if cached is not None and time.monotonic() - cached[0] < OPTIONS_TTL:
            return cached[1]
        chosen = _options(owner)
        with self._lock:
            self._options[owner] = (time.monotonic(), chosen)
        return chosen

    def set_options(self, owner: Owner, **options: str | None) -> None:
        """Set `tools` and/or `research_preset`; None goes back to agent.cfg's."""
        owner = tuple(owner)
        _set_options(owner, options)
        with self._lock:
            self._options.pop(owner, None)


def _pref_id(owner: Owner, source: str, kb: str) -> str:
    return hashlib.sha256("\0".join((owner[0], owner[1] or "", source, kb)).encode()).hexdigest()


def _owned(query, model, owner: Owner):
    subject, org = owner
    query = query.where(model.owner_subject == subject)
    return query.where(model.owner_org.is_(None) if org is None else model.owner_org == org)


@read_session
def _service_rows(owner: Owner, *, session=None) -> list[dict]:
    rows = session.execute(_owned(select(KnowledgeServiceRow), KnowledgeServiceRow, owner)
                           .order_by(KnowledgeServiceRow.name)).scalars()
    return [{c.name: getattr(r, c.name) for c in KnowledgeServiceRow.__table__.columns} for r in rows]


@transactional_session
def _save_service(service: KnowledgeService, encrypted: str | None, token_hint: str | None, keep_token: bool,
                  *, session=None) -> None:
    row = session.get(KnowledgeServiceRow, service.id)
    if row is not None and (row.owner_subject, row.owner_org) != tuple(service.owner):
        raise PermissionError(f"knowledge service {service.id!r} belongs to someone else")
    row = row or KnowledgeServiceRow(id=service.id)
    row.name, (row.owner_subject, row.owner_org) = service.name, service.owner
    row.label, row.url, row.base_path, row.enabled = service.label, service.url, service.base_path, int(service.enabled)
    if not keep_token:
        row.token_encrypted, row.token_hint = encrypted, token_hint
    session.merge(row)


@transactional_session
def _delete_service(owner: Owner, service_id: str, *, session=None) -> bool:
    row = session.get(KnowledgeServiceRow, service_id)
    if row is None or (row.owner_subject, row.owner_org) != owner:
        return False
    session.execute(_owned(delete(KnowledgePrefRow), KnowledgePrefRow, owner).where(KnowledgePrefRow.source == service_id))
    session.delete(row)
    return True


@read_session
def _disabled(owner: Owner, *, session=None) -> set[tuple[str, str]]:
    rows = session.execute(_owned(select(KnowledgePrefRow.source, KnowledgePrefRow.kb), KnowledgePrefRow, owner))
    return {(source, kb) for source, kb in rows}


@transactional_session
def _set_pref(owner: Owner, source: str, kb: str, enabled: bool, *, session=None) -> None:
    pref_id = _pref_id(owner, source, kb)
    row = session.get(KnowledgePrefRow, pref_id)
    if enabled and row is not None:
        session.delete(row)
    elif not enabled and row is None:
        session.add(KnowledgePrefRow(id=pref_id, owner_subject=owner[0], owner_org=owner[1], source=source, kb=kb))


def _option_id(owner: Owner) -> str:
    return hashlib.sha256("\0".join((owner[0], owner[1] or "")).encode()).hexdigest()


@read_session
def _options(owner: Owner, *, session=None) -> dict[str, str]:
    row = session.get(KnowledgeOptionRow, _option_id(owner))
    if row is None:
        return {}
    return {name: value for name, value in (("tools", row.tools), ("research_preset", row.research_preset)) if value}


@transactional_session
def _set_options(owner: Owner, options: dict[str, str | None], *, session=None) -> None:
    row = session.get(KnowledgeOptionRow, _option_id(owner))
    values = {"tools": row.tools if row else None, "research_preset": row.research_preset if row else None, **options}
    if not any(values.values()):   # both agent.cfg's: no row
        if row is not None:
            session.delete(row)
        return
    if row is None:
        row = KnowledgeOptionRow(id=_option_id(owner), owner_subject=owner[0], owner_org=owner[1])
        session.add(row)
    row.tools, row.research_preset = values["tools"] or None, values["research_preset"] or None
