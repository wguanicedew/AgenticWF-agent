"""
A small, synchronous MCP client (Model Context Protocol, https://modelcontextprotocol.io): enough
to list a server's tools and call them. Standard library only.

    stdio   the server is a child process; JSON-RPC messages are lines on its stdin/stdout
    http    "streamable HTTP": each message is a POST; the reply is JSON or a short SSE stream

One client per endpoint (agenticwf.agent.runtime.mcp.endpoint), shared by every task; McpManager owns them. Requests are serialised by a lock (one in flight
per server), and a broken connection is reopened once on the next request. Handlers call it from
a worker thread (ToolDispatcher runs synchronous handlers with asyncio.to_thread).
"""

from __future__ import annotations

import itertools
import json
import os
import queue
import subprocess
import threading
import time
import http.client
import ssl
from typing import Any, Callable
from urllib.parse import urlsplit

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint

LOG = get_logger(__name__)

PROTOCOL_VERSION = "2025-06-18"
MAX_REPLY_BYTES = 8 * 1024 * 1024
MAX_PAGES = 100
CLIENT_INFO = {"name": "agenticwf-agent", "version": "0.1.0"}


class McpError(Exception):
    """The server could not be reached, or answered with a JSON-RPC error."""


class McpRpcError(McpError):
    """The server answered, with a JSON-RPC error: retrying will not help."""


class _StdioTransport:
    def __init__(self, config: McpEndpoint, timeout: float):
        if not config.command:
            raise McpError("stdio endpoint without a command")
        env = {**os.environ, **config.env}
        try:
            self.proc = subprocess.Popen([config.command, *config.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                         stderr=subprocess.DEVNULL, env=env, cwd=config.cwd, text=True, bufsize=1)
        except OSError as exc:
            raise McpError(f"cannot start {config.command!r}: {exc}") from exc
        self.lines: queue.Queue[str | None] = queue.Queue()
        threading.Thread(target=self._read, name=f"mcp-{config.name}", daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)   # end of stream

    def _write(self, message: dict[str, Any]) -> None:
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (OSError, ValueError) as exc:
            raise McpError(f"server closed its input: {exc}") from exc

    def request(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        self._write(message)
        deadline = time.monotonic() + timeout
        while True:
            try:
                line = self.lines.get(timeout=max(0.0, deadline - time.monotonic()))
            except queue.Empty:
                raise McpError(f"no reply within {timeout:g} s") from None
            if line is None:
                raise McpError(f"server exited (status {self.proc.poll()})")
            try:
                reply = json.loads(line)
            except ValueError:
                continue   # a server that logs to stdout; skip the line
            if not isinstance(reply, dict):
                continue
            if reply.get("id") == message["id"] and "method" not in reply:
                return reply
            if "method" in reply and "id" in reply:   # a request from the server (ping, sampling, ...)
                result = {"result": {}} if reply["method"] == "ping" else \
                         {"error": {"code": -32601, "message": "not supported by this client"}}
                self._write({"jsonrpc": "2.0", "id": reply["id"], **result})

    def notify(self, message: dict[str, Any]) -> None:
        self._write(message)

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.proc.kill()


class _HttpTransport:
    """One POST per message over http.client: redirects are never followed (a 3xx is an error), the
    reply is read up to MAX_REPLY_BYTES, and `guard` sees the address actually connected to (so a
    name that resolves somewhere else at connect time cannot slip past the allow-list)."""

    def __init__(self, config: McpEndpoint, guard: Callable[[str], str | None] | None = None):
        parts = urlsplit(config.url or "")
        if parts.scheme not in ("http", "https") or not parts.hostname:
            raise McpError("http endpoint needs an http(s) url")
        self.parts = parts
        self.where = f"{parts.scheme}://{parts.hostname}{f':{parts.port}' if parts.port else ''}"   # never the path or query
        self.headers = dict(config.headers)
        self.guard = guard
        self.session_id: str | None = None
        self.protocol: str | None = None

    def _post(self, message: dict[str, Any], timeout: float) -> tuple[str, bytes]:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        if self.protocol:
            headers["MCP-Protocol-Version"] = self.protocol
        p = self.parts
        target = (p.path or "/") + (f"?{p.query}" if p.query else "")
        connection = (http.client.HTTPSConnection(p.hostname, p.port, timeout=timeout, context=ssl.create_default_context())
                      if p.scheme == "https" else http.client.HTTPConnection(p.hostname, p.port, timeout=timeout))
        try:
            connection.connect()
            if self.guard is not None:
                refusal = self.guard(connection.sock.getpeername()[0])
                if refusal:
                    raise McpError(refusal)
            connection.request("POST", target, body=json.dumps(message).encode(), headers=headers)
            response = connection.getresponse()
            if 300 <= response.status < 400:
                raise McpError(f"HTTP {response.status} from {self.where}: redirects are not followed")
            if response.status >= 400:
                raise McpError(f"HTTP {response.status} from {self.where}")
            body = response.read(MAX_REPLY_BYTES + 1)
            if len(body) > MAX_REPLY_BYTES:
                raise McpError(f"the reply from {self.where} is larger than {MAX_REPLY_BYTES} bytes")
            self.session_id = response.getheader("Mcp-Session-Id") or self.session_id
            return response.getheader("Content-Type") or "", body
        except (OSError, http.client.HTTPException) as exc:
            raise McpError(f"cannot reach {self.where}: {exc}") from exc
        finally:
            connection.close()

    def request(self, message: dict[str, Any], timeout: float) -> dict[str, Any]:
        content_type, body = self._post(message, timeout)
        if content_type.startswith("text/event-stream"):
            for event in body.decode(errors="replace").split("\n\n"):
                data = "\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:"))
                if data:
                    try:
                        reply = json.loads(data)
                    except ValueError:
                        raise McpError("the event stream carried invalid JSON") from None
                    if isinstance(reply, dict) and reply.get("id") == message["id"]:
                        return reply
            raise McpError("the event stream ended without a reply")
        try:
            reply = json.loads(body)
        except ValueError as exc:
            raise McpError("the reply is not JSON") from exc
        if not isinstance(reply, dict):
            raise McpError("the reply is not a JSON-RPC message")
        return reply

    def notify(self, message: dict[str, Any]) -> None:
        self._post(message, 10)

    def close(self) -> None:
        self.session_id = None


class McpClient:
    def __init__(self, config: McpEndpoint, connect_timeout: float = 15.0,
                 guard: Callable[[str], str | None] | None = None):
        """`guard(ip)`: why the connected address may not be used (users' endpoints), or None."""
        self.config = config
        self.guard = guard
        self.connect_timeout = connect_timeout
        self._transport: _StdioTransport | _HttpTransport | None = None
        self._ids = itertools.count(1)
        self._lock = threading.Lock()
        self._closed = False
        self.server_info: dict[str, Any] = {}

    # --- connection ----------------------------------------------------------------------------
    def _connect(self):
        transport = _StdioTransport(self.config, self.connect_timeout) if self.config.transport == "stdio" \
            else _HttpTransport(self.config, self.guard)
        try:
            reply = self._exchange(transport, "initialize", {
                "protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO}, self.connect_timeout)
            if isinstance(transport, _HttpTransport):
                transport.protocol = reply.get("protocolVersion") or PROTOCOL_VERSION
            transport.notify({"jsonrpc": "2.0", "method": "notifications/initialized"})
        except Exception:
            transport.close()
            raise
        self.server_info = reply.get("serverInfo") or {}
        return transport

    def _exchange(self, transport, method: str, params: dict[str, Any], timeout: float) -> dict[str, Any]:
        reply = transport.request({"jsonrpc": "2.0", "id": next(self._ids), "method": method, "params": params}, timeout)
        if "error" in reply:
            error = reply["error"] if isinstance(reply["error"], dict) else {"message": str(reply["error"])}
            raise McpRpcError(f"{method}: {error.get('message', 'error')} ({error.get('code')})")
        result = reply.get("result") or {}
        if not isinstance(result, dict):
            raise McpError(f"{method}: the result is not an object")
        return result

    def request(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        timeout = timeout or self.config.timeout
        with self._lock:
            for attempt in (1, 2):
                if self._closed:
                    raise McpError("the server was removed or reloaded")
                if self._transport is None:
                    self._transport = self._connect()
                try:
                    return self._exchange(self._transport, method, params, timeout)
                except McpRpcError:
                    raise
                except McpError as exc:
                    if attempt == 2:
                        raise
                    LOG.warning("MCP server %s: %s; reconnecting", self.config.name, exc)
                    self.close_locked()
        raise AssertionError("unreachable")

    def close_locked(self) -> None:
        if self._transport is not None:
            self._transport.close()
            self._transport = None

    def close(self) -> None:
        """Disconnect for good (and stop a stdio server): later requests fail."""
        self._closed = True
        with self._lock:
            self.close_locked()

    # --- tools ---------------------------------------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        tools, cursor, seen = [], None, set()
        for _ in range(MAX_PAGES):
            result = self.request("tools/list", {"cursor": cursor} if cursor else {}, timeout=self.connect_timeout)
            page = result.get("tools") or []
            if not isinstance(page, list):
                raise McpError("tools/list: tools is not a list")
            tools.extend(page)
            cursor = result.get("nextCursor")
            if not cursor or not isinstance(cursor, str) or cursor in seen:
                return tools
            seen.add(cursor)
        raise McpError(f"tools/list: more than {MAX_PAGES} pages")

    def call_tool(self, name: str, arguments: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        return self.request("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
