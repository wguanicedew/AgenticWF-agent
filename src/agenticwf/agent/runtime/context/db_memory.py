"""
DatabaseMemoryStore: long-term memory on agenticwf.agent.core.orm (agent_memories), the API's default MemoryStore.

Every row belongs to one owner (subject, organization); recall and remember filter on it, so one
person's memories never reach another's tasks. Remembering the same text twice keeps one row, so a
consolidation job that runs again (after a restart) adds nothing.

Recall ranks the owner's most recent memories by the words they share with the query and, with
embeddings ([embeddings], agenticwf.agent.runtime.embeddings), by meaning; newest first among equals. An empty
query gets the newest.

People list and delete their own memories through the API (docs/admin-page.md, Memory).

TODO: PostgreSQL full-text search, and vectors in pgvector instead of ranking in Python; forgetting
by age, or a cap per owner.
"""

from __future__ import annotations

import datetime
import uuid

from sqlalchemy import delete, func, select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import MemoryRow
from agenticwf.agent.runtime.context.memory import MemoryItem
from agenticwf.agent.runtime.embeddings import EmbeddingIndex
from agenticwf.agent.runtime.text import terms
from agenticwf.agent.runtime.types import Owner

TEXT_MAX = 2000


def _owned(owner: Owner):
    subject, org = owner
    return (MemoryRow.owner_subject == subject,
            MemoryRow.owner_org.is_(None) if org is None else MemoryRow.owner_org == org)


class DatabaseMemoryStore:
    def __init__(self, candidates: int = 1000, embeddings: EmbeddingIndex | None = None, embed_candidates: int = 200):
        self.candidates = candidates             # how many of an owner's newest memories recall ranks
        self.embeddings = embeddings
        self.embed_candidates = embed_candidates   # of those, how many are also matched by meaning

    def use_embeddings(self, embeddings: EmbeddingIndex) -> None:
        """Match by meaning as well as by words (agenticwf.agent.runtime.embeddings; bootstrap sets it from [embeddings])."""
        self.embeddings = embeddings

    def recall(self, owner: Owner, query: str, k: int) -> list[MemoryItem]:
        """The owner's memories closest to `query`, best first; newest first among equals. By the
        words they share with it and, with embeddings, by meaning too: the newest `embed_candidates`
        are compared with the query's embedding, combined as for tools ([embeddings] weight,
        min_similarity), so "which scheduler do I use" recalls "Alice prefers SLURM". When the
        embedding provider is unavailable, by words alone."""
        if k <= 0:
            return []
        items = _newest(owner, self.candidates)
        if not query.strip():
            return items[:k]
        words = terms(query)
        lexical = [len(words & terms(m.text)) for m in items]
        similarity = (self.embeddings.similarities(query, [m.text for m in items[:self.embed_candidates]])
                      if self.embeddings is not None and items else None)
        if similarity is None:
            scored = [(score, m) for score, m in zip(lexical, items) if score > 0 or not words]
        else:
            settings, best = self.embeddings.settings, max(lexical) or 1
            similarity = similarity + [0.0] * (len(items) - len(similarity))
            scored = [(settings.weight * max(0.0, sim) + (1 - settings.weight) * score / best, m)
                      for score, sim, m in zip(lexical, similarity, items)
                      if score > 0 or sim >= settings.min_similarity]
        return [m for _, m in sorted(scored, key=lambda t: -t[0])][:k]   # stable: newest first among equals

    def remember(self, owner: Owner, texts: list[str], source_task: str) -> None:
        cleaned = list(dict.fromkeys(t for t in (" ".join(t.split())[:TEXT_MAX] for t in texts) if t))
        if cleaned:
            _insert(owner, cleaned, source_task)
            if self.embeddings is not None:   # embedded now, so recall does not wait for it
                self.embeddings.warm(cleaned)

    def list(self, owner: Owner, limit: int, offset: int = 0) -> tuple[list[MemoryItem], int]:
        return _page(owner, limit, offset)

    def forget(self, owner: Owner, memory_id: str | None = None) -> int:
        return _delete(owner, memory_id)


def _item(row: MemoryRow) -> MemoryItem:
    at = row.created_at.replace(tzinfo=datetime.timezone.utc).timestamp() if row.created_at else None
    return MemoryItem(id=row.id, text=row.text, source_task=row.source_task, created_at=at)


@read_session
def _newest(owner: Owner, limit: int, offset: int = 0, *, session=None) -> list[MemoryItem]:
    rows = session.execute(select(MemoryRow).where(*_owned(owner))
                           .order_by(MemoryRow.created_at.desc(), MemoryRow.id).offset(offset).limit(limit)).scalars()
    return [_item(r) for r in rows]


@read_session
def _page(owner: Owner, limit: int, offset: int, *, session=None) -> tuple[list[MemoryItem], int]:
    total = session.execute(select(func.count()).select_from(MemoryRow).where(*_owned(owner))).scalar_one()
    return _newest(owner, limit, offset, session=session), total


@transactional_session
def _delete(owner: Owner, memory_id: str | None, *, session=None) -> int:
    query = delete(MemoryRow).where(*_owned(owner))
    if memory_id is not None:
        query = query.where(MemoryRow.id == memory_id)
    return session.execute(query).rowcount or 0


@transactional_session
def _insert(owner: Owner, texts: list[str], source_task: str, *, session=None) -> None:
    known = set(session.execute(select(MemoryRow.text).where(*_owned(owner), MemoryRow.text.in_(texts))).scalars())
    for text in texts:
        if text not in known:
            session.add(MemoryRow(id=uuid.uuid4().hex, owner_subject=owner[0], owner_org=owner[1], text=text,
                                  source_task=source_task))
