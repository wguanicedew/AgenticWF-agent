"""
Long-term memory and knowledge bases, as seen by AgentContext and the harness
(docs/agent-structure.md §7). Backends are pluggable; the defaults do nothing. The API's default
memory is agenticwf.agent.runtime.context.db_memory.DatabaseMemoryStore.

`memory_search` is a core tool whenever a memory store is configured: its handler searches the
caller's own memories only (the owner comes from the call's AuthContext, never from arguments).

TODO: a MemoryManager on PostgreSQL + pgvector; knowledge bases on Meilisearch / pgvector
/ external services, with a `kb_search` tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.tools.loader import core_tool_text
from agenticwf.agent.runtime.tools.spec import (Approval, CallContext, Effects, Handler, HandlerKind, HandlerRef, ToolResult,
                                ToolSource, ToolSpec)
from agenticwf.agent.runtime.types import Owner, owner_key


@dataclass(frozen=True)
class MemoryItem:
    id: str
    text: str
    source_task: str | None = None
    created_at: float | None = None   # Unix time

    def to_dict(self) -> dict:
        return {"id": self.id, "text": self.text, "source_task": self.source_task, "created_at": self.created_at}


@dataclass(frozen=True)
class KnowledgeBaseInfo:
    id: str
    name: str
    description: str


class MemoryStore(Protocol):
    def recall(self, owner: Owner, query: str, k: int) -> list[MemoryItem]:
        """Memories of this owner relevant to `query`, best first."""

    def remember(self, owner: Owner, texts: list[str], source_task: str) -> None:
        """Store consolidated memories. Never secrets, never the AuthContext."""

    def list(self, owner: Owner, limit: int, offset: int = 0) -> tuple[list[MemoryItem], int]:
        """This owner's memories, newest first, and how many they have in all."""

    def forget(self, owner: Owner, memory_id: str | None = None) -> int:
        """Delete one of this owner's memories, or all of them (None). Returns how many were deleted."""


class KnowledgeCatalog(Protocol):
    def bases(self, auth: AuthContext, ids: tuple[str, ...]) -> list[KnowledgeBaseInfo]:
        """The knowledge bases (of `ids`, or all configured if empty) this principal may search."""


class NoMemory:
    def recall(self, owner: Owner, query: str, k: int) -> list[MemoryItem]:
        return []

    def remember(self, owner: Owner, texts: list[str], source_task: str) -> None:
        pass

    def list(self, owner: Owner, limit: int, offset: int = 0) -> tuple[list[MemoryItem], int]:
        return [], 0

    def forget(self, owner: Owner, memory_id: str | None = None) -> int:
        return 0


class NoKnowledge:
    def bases(self, auth: AuthContext, ids: tuple[str, ...]) -> list[KnowledgeBaseInfo]:
        return []


_summary, _description, _schema = core_tool_text("memory_search")   # runtime/defaults/core-tools/memory_search.md
MEMORY_SEARCH = ToolSpec(
    name="memory_search", summary=_summary, description=_description, input_schema=_schema,
    handler=HandlerRef(HandlerKind.BUILTIN, {"ref": "memory.search"}),
    source=ToolSource("core", "core-tools/memory_search.md"),
    effects=Effects.NONE, approval=Approval.AUTO, idempotent=True, parallel_safe=True,
    tags=("memory", "recall", "remember", "earlier", "previous"),
)


def memory_search_handler(memory: MemoryStore) -> Handler:
    def handler(call: CallContext, arguments: dict) -> ToolResult:
        found = memory.recall(owner_key(call.auth), arguments["query"], int(arguments.get("limit") or 5))
        if not found:
            return ToolResult(content="No notes match.", summary="memory_search: nothing found")
        return ToolResult(content="\n".join(f"- {m.text}" for m in found),
                          summary=f"memory_search: {len(found)} note(s)")
    return handler
