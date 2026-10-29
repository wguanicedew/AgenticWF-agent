"""DatabaseModelStore: the ModelCatalog's database source, on agenticwf.agent.core.orm (tables agent_models,
agent_model_defaults: the model each user chose as their default).

API keys are stored encrypted with a SecretBox (agenticwf.agent.runtime.secrets) and decrypted only in this
process, when the catalog loads; the row also keeps a hint (last four characters) for the admin
view. Without a SecretBox, entries that need no key still work, and saving a key is refused.
"""

from __future__ import annotations

import calendar
import hashlib
from dataclasses import replace

from sqlalchemy import select

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import ModelDefaultRow, ModelRow
from agenticwf.agent.runtime.llm.catalog import ModelEntry
from agenticwf.agent.runtime.secrets import SecretBox, SecretsUnavailable, hint
from agenticwf.agent.runtime.types import Owner

LOG = get_logger(__name__)


class DatabaseModelStore:
    def __init__(self, box: SecretBox | None):
        self.box = box

    def list_models(self) -> list[ModelEntry]:
        entries = []
        for row in _rows():
            api_key, unreadable = None, False
            if row["api_key_encrypted"]:
                if self.box is None:
                    unreadable = True
                    LOG.error("model %s has a stored API key but no secrets key is configured", row["id"])
                else:
                    try:
                        api_key = self.box.decrypt(row["api_key_encrypted"])
                    except SecretsUnavailable:
                        unreadable = True
                        LOG.error("model %s: its API key cannot be decrypted (secrets key changed?)", row["id"])
            entries.append(ModelEntry(
                id=row["id"], provider=row["provider"], model=row["model"], label=row["label"] or "",
                description=row["description"] or "", base_url=row["base_url"], api_key=api_key,
                api_key_env=row["api_key_env"], options=dict(row["options"] or {}), enabled=bool(row["enabled"]),
                source="database", api_key_hint=row["api_key_hint"], owner_subject=row["owner_subject"],
                owner_org=row["owner_org"], visibility=row["visibility"] or "all", api_key_unreadable=unreadable,
                updated_at=float(calendar.timegm(row["updated_at"].utctimetuple()))))
        return entries

    def save_model(self, entry: ModelEntry, keep_key: bool = False) -> None:
        """`keep_key`: the stored (encrypted) key stays as it is; it is never rewritten from what this
        process could decrypt, so a missing or changed secrets key cannot erase it."""
        if keep_key:
            _upsert(replace(entry, api_key=None), None, None, keep_key=True)
            return
        encrypted = None
        if entry.api_key:
            if self.box is None:
                raise PermissionError("no secrets key is configured (AGENT_SECRETS_KEY_FILE): an API key "
                                      "cannot be stored in the database; use api_key_env instead")
            encrypted = self.box.encrypt(entry.api_key)
        _upsert(replace(entry, api_key=None), encrypted, hint(entry.api_key) if entry.api_key else None)

    def delete_model(self, model_id: str) -> bool:
        return _delete(model_id)

    def user_defaults(self) -> dict[Owner, str]:
        """Each user's chosen default model id."""
        return _defaults()

    def set_user_default(self, owner: Owner, model_id: str | None) -> None:
        """None: back to [llm] default."""
        _set_default(tuple(owner), model_id)


@read_session
def _rows(*, session=None) -> list[dict]:
    rows = session.execute(select(ModelRow).order_by(ModelRow.id)).scalars()
    return [{c.name: getattr(r, c.name) for c in ModelRow.__table__.columns} for r in rows]


@transactional_session
def _upsert(entry: ModelEntry, encrypted: str | None, key_hint: str | None, keep_key: bool = False,
            *, session=None) -> None:
    row = session.get(ModelRow, entry.id)
    if row is not None and (row.owner_subject, row.owner_org) != (entry.owner_subject, entry.owner_org):
        raise PermissionError(f"model {entry.id!r} belongs to someone else")   # checked again here: no race
    row = row or ModelRow(id=entry.id)
    row.provider, row.model, row.label, row.description = entry.provider, entry.model, entry.label, entry.description
    row.base_url, row.api_key_env, row.options, row.enabled = entry.base_url, entry.api_key_env, entry.options, int(entry.enabled)
    if not keep_key:
        row.api_key_encrypted, row.api_key_hint = encrypted, key_hint
    row.owner_subject, row.owner_org, row.visibility = entry.owner_subject, entry.owner_org, entry.visibility
    session.merge(row)


@transactional_session
def _delete(model_id: str, *, session=None) -> bool:
    row = session.get(ModelRow, model_id)
    if row is None:
        return False
    session.delete(row)
    return True


def _default_id(owner: Owner) -> str:
    return hashlib.sha256("\0".join((owner[0], owner[1] or "")).encode()).hexdigest()


@read_session
def _defaults(*, session=None) -> dict[Owner, str]:
    rows = session.execute(select(ModelDefaultRow.owner_subject, ModelDefaultRow.owner_org, ModelDefaultRow.model_id))
    return {(subject, org): model_id for subject, org, model_id in rows}


@transactional_session
def _set_default(owner: Owner, model_id: str | None, *, session=None) -> None:
    row = session.get(ModelDefaultRow, _default_id(owner))
    if model_id is None:
        if row is not None:
            session.delete(row)
    elif row is None:
        session.add(ModelDefaultRow(id=_default_id(owner), owner_subject=owner[0], owner_org=owner[1], model_id=model_id))
    else:
        row.model_id = model_id
