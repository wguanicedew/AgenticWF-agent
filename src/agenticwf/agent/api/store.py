"""Chat sessions, their transcripts, and the API's view of tasks, in the database (agenticwf.agent.core.orm), keyed by owner.

A chat session is a runtime thread (same id); its transcript is what the user and the agent said.
An API *task* (POST /api/v1/tasks) is a runtime intent submitted without a conversation: its
status and result are read from the intent and the runtime tasks it became
(agenticwf.agent.core.orm.models.agent, written by agenticwf.agent.runtime.harness.db_store).

Every lookup filters on the owner, so another principal's ids behave as if they did not exist
and callers answer 404. Methods are synchronous (SQLAlchemy sessions); call them from async code
through ``asyncio.to_thread``. They return plain dicts, never ORM objects.
"""

from __future__ import annotations

import calendar
import datetime
import uuid
from typing import Any

from sqlalchemy import func, select

from agenticwf.agent.core.orm.base.session import read_session, transactional_session
from agenticwf.agent.core.orm.models.agent import ChatMessage, ChatSession, IntentRow, TaskRow, _utcnow

Owner = tuple[str, str | None]
API_SOURCE = "api"
_WAITING = ("waiting_approval", "waiting_user")


def _ts(value: datetime.datetime | None) -> int | None:
    return calendar.timegm(value.utctimetuple()) if value else None


def _owned(model, owner: Owner):
    subject, org = owner
    return (model.owner_subject == subject,
            model.owner_org.is_(None) if org is None else model.owner_org == org)


def _session_summary(row: ChatSession, count: int) -> dict[str, Any]:
    return {"id": row.id, "title": row.title, "created_at": _ts(row.created_at),
            "organization_id": row.owner_org, "messages": count}


def _message(row: ChatMessage) -> dict[str, Any]:
    return {"id": row.id, "role": row.role, "text": row.text, "time": _ts(row.created_at), "intent": row.intent_id}


def _task(intent: IntentRow, tasks: list[TaskRow]) -> dict[str, Any]:
    """The API's task: the intent, with a status the old API clients understand."""
    reply = intent.reply or {}
    if intent.status in ("pending", "claimed"):
        status = "queued"
    elif intent.status == "converted":
        status = "waiting" if any(t.state in _WAITING for t in tasks) else "running"
    elif intent.status == "answered":
        status = reply.get("status", "failed")
    else:   # cancelled, rejected, superseded
        status = "cancelled" if intent.status == "cancelled" else "failed"
    errors = [t.result["error"]["message"] for t in tasks if t.result and t.result.get("error")]
    return {"id": intent.id, "status": status, "prompt": intent.text, "created_at": _ts(intent.created_at),
            "finished_at": _ts(intent.updated_at) if intent.status == "answered" else None,
            "result": reply.get("answer"), "error": "; ".join(errors) or None,
            "tasks": [{"id": t.id, "task_type": t.task_type, "state": t.state,
                       "error": (t.result or {}).get("error")} for t in tasks]}


class Store:
    # --- chat sessions -------------------------------------------------------------------
    @transactional_session
    def create_session(self, owner: Owner, title: str, *, session=None) -> dict[str, Any]:
        row = ChatSession(id=uuid.uuid4().hex, owner_subject=owner[0], owner_org=owner[1], title=title)
        session.add(row)
        session.flush()
        return _session_summary(row, 0)

    @read_session
    def list_sessions(self, owner: Owner, *, session=None) -> list[dict[str, Any]]:
        counts = (select(ChatMessage.session_id, func.count().label("n"))
                  .group_by(ChatMessage.session_id).subquery())
        rows = session.execute(
            select(ChatSession, func.coalesce(counts.c.n, 0))
            .outerjoin(counts, counts.c.session_id == ChatSession.id)
            .where(*_owned(ChatSession, owner))
            .order_by(ChatSession.created_at.desc())
        ).all()
        return [_session_summary(row, n) for row, n in rows]

    def _get_session(self, session, owner: Owner, sid: str) -> ChatSession | None:
        return session.execute(
            select(ChatSession).where(ChatSession.id == sid, *_owned(ChatSession, owner))
        ).scalar_one_or_none()

    @read_session
    def get_session(self, owner: Owner, sid: str, *, session=None) -> dict[str, Any] | None:
        row = self._get_session(session, owner, sid)
        if row is None:
            return None
        messages = [_message(m) for m in row.messages]
        return {**_session_summary(row, len(messages)), "messages": messages,
                "summary": row.summary or "", "summary_keys": list(row.summary_keys or ())}

    @transactional_session
    def set_summary(self, owner: Owner, sid: str, summary: str, keys: list[str], *, session=None) -> bool:
        """Keep a new summary of the earlier conversation: it folds in the one before, so it covers
        what that one covered and `keys` besides."""
        row = self._get_session(session, owner, sid)
        if row is None:
            return False
        row.summary = summary
        row.summary_keys = sorted(set(row.summary_keys or ()) | set(keys))
        return True

    @transactional_session
    def rename_session(self, owner: Owner, sid: str, title: str, *, session=None) -> dict[str, Any] | None:
        row = self._get_session(session, owner, sid)
        if row is None:
            return None
        row.title = title
        session.flush()
        return _session_summary(row, len(row.messages))

    @transactional_session
    def delete_session(self, owner: Owner, sid: str, *, session=None) -> bool:
        row = self._get_session(session, owner, sid)
        if row is None:
            return False
        session.delete(row)
        return True

    @transactional_session
    def add_message(self, owner: Owner, sid: str, role: str, text: str, intent_id: str | None = None, *,
                    session=None) -> dict[str, Any] | None:
        """Append a message to the transcript; None if the session is not found."""
        row = self._get_session(session, owner, sid)
        if row is None:
            return None
        message = ChatMessage(role=role, text=text, intent_id=intent_id)
        row.messages.append(message)
        row.updated_at = _utcnow()
        session.flush()
        return _message(message)

    # --- tasks: intents submitted through the API ------------------------------------------
    @read_session
    def list_tasks(self, owner: Owner, *, session=None) -> list[dict[str, Any]]:
        intents = session.execute(
            select(IntentRow).where(IntentRow.source == API_SOURCE, *_owned(IntentRow, owner))
            .order_by(IntentRow.created_at.desc())
        ).scalars().all()
        return [_task(i, self._tasks_of(session, i.id)) for i in intents]

    @read_session
    def get_task(self, owner: Owner, tid: str, *, session=None) -> dict[str, Any] | None:
        intent = self._intent(session, owner, tid)
        if intent is None or intent.source != API_SOURCE:
            return None
        return _task(intent, self._tasks_of(session, tid))

    @read_session
    def intent_thread(self, owner: Owner, intent_id: str, *, session=None) -> str | None:
        """The thread an intent of this owner belongs to, for routing approvals, input and cancel."""
        intent = self._intent(session, owner, intent_id)
        return intent.thread_id if intent is not None else None

    @staticmethod
    def _intent(session, owner: Owner, intent_id: str) -> IntentRow | None:
        return session.execute(
            select(IntentRow).where(IntentRow.id == intent_id, *_owned(IntentRow, owner))
        ).scalar_one_or_none()

    @staticmethod
    def _tasks_of(session, intent_id: str) -> list[TaskRow]:
        return list(session.execute(
            select(TaskRow).where(TaskRow.intent_id == intent_id).order_by(TaskRow.created_at)).scalars())
