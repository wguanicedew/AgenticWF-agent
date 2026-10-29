"""DatabaseSkillStore: SkillCatalog's database source, on agenticwf.agent.core.orm (agent_skills, agent_skill_files).

Skills added on the admin page or through the API, with their files, live only here: nothing is
written to disk.
"""

from __future__ import annotations

from sqlalchemy import delete, select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import SkillFileRow, SkillRow
from agenticwf.agent.runtime.skills.skill import Skill


class DatabaseSkillStore:
    def list(self) -> list[Skill]:
        return _rows()

    def save(self, skill: Skill) -> None:
        _save(skill)

    def delete(self, skill_id: str) -> bool:
        return _delete(skill_id)

    def read_file(self, skill_id: str, path: str) -> str | None:
        return _read_file(skill_id, path)

    def write_file(self, skill_id: str, path: str, content: str) -> None:
        _write_file(skill_id, path, content)

    def delete_file(self, skill_id: str, path: str) -> bool:
        return _delete_file(skill_id, path)


@read_session
def _rows(*, session=None) -> list[Skill]:
    files: dict[str, list[str]] = {}
    for skill_id, path in session.execute(select(SkillFileRow.skill_id, SkillFileRow.path)):
        files.setdefault(skill_id, []).append(path)
    return [Skill(name=r.name, description=r.description, instructions=r.instructions, source="database",
                  owner=(r.owner_subject, r.owner_org), tools=tuple(r.tools or ()), tags=tuple(r.tags or ()),
                  visibility=r.visibility, enabled=bool(r.enabled), location=f"database:{r.id}",
                  stored_files=tuple(sorted(files.get(r.id, ()))))
            for r in session.execute(select(SkillRow).order_by(SkillRow.id)).scalars()]


@transactional_session
def _save(skill: Skill, *, session=None) -> None:
    row = session.get(SkillRow, skill.id)
    if row is not None and (row.owner_subject, row.owner_org) != tuple(skill.owner):
        raise PermissionError(f"skill {skill.id!r} belongs to someone else")   # checked again here: no race
    row = row or SkillRow(id=skill.id)
    row.owner_subject, row.owner_org = skill.owner
    row.name, row.description, row.instructions = skill.name, skill.description, skill.instructions
    row.tools, row.tags = list(skill.tools), list(skill.tags)
    row.visibility, row.enabled = skill.visibility or "user", int(skill.enabled)
    session.merge(row)


@transactional_session
def _delete(skill_id: str, *, session=None) -> bool:
    session.execute(delete(SkillFileRow).where(SkillFileRow.skill_id == skill_id))
    row = session.get(SkillRow, skill_id)
    if row is None:
        return False
    session.delete(row)
    return True


@read_session
def _read_file(skill_id: str, path: str, *, session=None) -> str | None:
    row = session.get(SkillFileRow, (skill_id, path))
    return None if row is None else row.content


@transactional_session
def _write_file(skill_id: str, path: str, content: str, *, session=None) -> None:
    if session.get(SkillRow, skill_id) is None:
        raise KeyError(skill_id)
    session.merge(SkillFileRow(skill_id=skill_id, path=path, content=content))


@transactional_session
def _delete_file(skill_id: str, path: str, *, session=None) -> bool:
    return bool(session.execute(delete(SkillFileRow).where(SkillFileRow.skill_id == skill_id,
                                                            SkillFileRow.path == path)).rowcount)
