"""DatabaseRepoStore: the repositories users add (agenticwf.agent.runtime.code), on agenticwf.agent.core.orm
(agent_code_repos)."""

from __future__ import annotations

import calendar
import datetime

from sqlalchemy import select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import CodeRepoRow
from agenticwf.agent.runtime.code import UserRepo


class DatabaseRepoStore:
    def list_repos(self) -> list[UserRepo]:
        return _list()

    def save_repo(self, repo: UserRepo) -> None:
        _save(repo)

    def delete_repo(self, name: str) -> bool:
        return _delete(name)

    def set_status(self, name: str, error: str, synced_at: float | None) -> None:
        _set_status(name, error, synced_at)


def _naive(ts: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).replace(tzinfo=None)


@read_session
def _list(*, session=None) -> list[UserRepo]:
    return [UserRepo(name=r.name, url=r.url, owner=(r.owner_subject, r.owner_org), branch=r.branch or "",
                     label=r.label or "", visibility=r.visibility, error=r.error or "",
                     synced_at=float(calendar.timegm(r.synced_at.utctimetuple())) if r.synced_at else None)
            for r in session.execute(select(CodeRepoRow).order_by(CodeRepoRow.name)).scalars()]


@transactional_session
def _save(repo: UserRepo, *, session=None) -> None:
    row = session.get(CodeRepoRow, repo.name)
    if row is not None and (row.owner_subject, row.owner_org) != tuple(repo.owner):
        raise PermissionError(f"repository {repo.name!r} belongs to someone else")   # checked again here: no race
    row = row or CodeRepoRow(name=repo.name)
    row.owner_subject, row.owner_org = repo.owner
    row.url, row.branch, row.label, row.visibility = repo.url, repo.branch or None, repo.label or None, repo.visibility
    session.merge(row)


@transactional_session
def _delete(name: str, *, session=None) -> bool:
    row = session.get(CodeRepoRow, name)
    if row is None:
        return False
    session.delete(row)
    return True


@transactional_session
def _set_status(name: str, error: str, synced_at: float | None, *, session=None) -> None:
    row = session.get(CodeRepoRow, name)
    if row is None:
        return
    row.error = error or None
    if synced_at is not None:
        row.synced_at = _naive(synced_at)
