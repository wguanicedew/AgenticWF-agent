"""Live status updates over Server-Sent Events, fanned out per owner: (subject, organization), the
same key every other read uses, so one person's work under one organization never reaches their
stream under another.

In-process only (single worker); use Redis or NATS to fan out across processes.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections import defaultdict
from typing import Any, AsyncIterator, Callable

from fastapi import Request

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.types import Owner, owner_key

HEARTBEAT = 15


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


class Updates:
    def __init__(self) -> None:
        self._subscribers: dict[Owner, set[asyncio.Queue]] = defaultdict(set)
        self._contexts: dict[asyncio.Queue, AuthContext] = {}   # who each stream belongs to (broadcast)
        self._loop: asyncio.AbstractEventLoop | None = None      # the streams' loop

    def publish(self, owner: Owner, event: dict[str, Any]) -> None:
        for queue in self._subscribers.get(tuple(owner), ()):
            try:
                queue.put_nowait(dict(event))
            except asyncio.QueueFull:
                pass  # slow client; it catches up from the REST API

    def broadcast(self, event: dict[str, Any], allowed: Callable[[AuthContext], bool]) -> None:
        """To every stream whose caller `allowed` lets in (a cron's events: whoever may see the cron).
        Safe from any thread: off the streams' loop, it is handed over to it."""
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if self._loop is not None and running is not self._loop:
            self._loop.call_soon_threadsafe(self._broadcast, event, allowed)
            return
        self._broadcast(event, allowed)

    def _broadcast(self, event: dict[str, Any], allowed: Callable[[AuthContext], bool]) -> None:
        for queue, ctx in list(self._contexts.items()):
            try:
                if allowed(ctx):
                    queue.put_nowait(dict(event))
            except asyncio.QueueFull:
                pass

    async def stream(self, request: Request, ctx: AuthContext, max_seconds: int) -> AsyncIterator[str]:
        queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        owner = owner_key(ctx)
        self._subscribers[owner].add(queue)
        self._contexts[queue] = ctx
        self._loop = asyncio.get_running_loop()
        # Authorization happened once, when the stream opened. End it when the token behind
        # the context expires; EventSource reconnects through /verify, which also refreshes
        # a browser session.
        deadline = time.time() + max_seconds
        if ctx.expires_at:
            deadline = min(deadline, ctx.expires_at)
        try:
            yield "retry: 2000\n\n"
            yield _sse("status", {"state": "connected", "user": ctx.subject, "auth": ctx.auth_method})
            while (remaining := deadline - time.time()) > 0 and not await request.is_disconnected():
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=min(HEARTBEAT, remaining))
                    yield _sse(event.pop("type", "update"), event)
                except asyncio.TimeoutError:
                    yield _sse("status", {"state": "idle", "time": int(time.time())})
        finally:
            self._contexts.pop(queue, None)
            self._subscribers[owner].discard(queue)
            if not self._subscribers[owner]:
                del self._subscribers[owner]
