"""
A small Python client for the agent API (docs/api.md): chat sessions and background tasks.

    agent = AgentClient.from_env()      # as `agenticwf agent` does: the sign-in of `agenticwf login`
    agent = AgentClient("http://127.0.0.1:8765", bearer=token)        # the personal agent's token

    sid = agent.create_session("demo")["id"]
    print(agent.send(sid, "hello")["reply"])
    task = agent.wait_task(agent.submit_task("summarise run 42")["task_id"])

Errors raise AgentClientError (the API's `detail`). It has no dependency on the server packages;
`from_env` and the `agenticwf agent` command (cli.py) need agenticwf-auth-client.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import httpx

__all__ = ["AgentClient", "AgentClientError", "FINISHED"]

# A task's status once it no longer changes by itself; `waiting` needs an approval or an answer.
FINISHED = frozenset({"succeeded", "partial", "failed", "cancelled"})


class AgentClientError(Exception):
    def __init__(self, status: int, detail: Any):
        super().__init__(f"HTTP {status}: {detail}")
        self.status, self.detail = status, detail


class AgentClient:
    def __init__(self, base_url: str, bearer: str | None = None, timeout: float = 600.0, http: Any = None,
                 headers: Callable[[str, str], dict] | None = None):
        """`base_url` is where the agent serves /api/v1 (behind Caddy: …/app). The timeout covers a
        chat reply (AGENT_REPLY_TIMEOUT, 120 s by default). `http` and `headers` let a caller pass
        its own client and per-request headers (tests, custom auth)."""
        self.base = base_url.rstrip("/")
        self._auth = {"Authorization": f"Bearer {bearer}"} if bearer else {}
        self._headers = headers
        self._http = http or httpx.Client(timeout=timeout)

    @classmethod
    def from_env(cls, url: str | None = None, token: str | None = None, **options) -> "AgentClient":
        """The agent and its tokens as `agenticwf` finds them (agenticwf_auth_cli.service_access):
        AGENTICWF_AGENT_URL and AGENTICWF_AGENT_TOKEN (the personal agent's own token), else the
        Logto sign-in and AGENTICWF_API_URL. Logto tokens are refreshed as they expire. `options`
        are Config fields (auth_url, api_url, …) and the constructor's (timeout, http)."""
        from agenticwf_auth_cli import service_access

        auth = {k: options.pop(k) for k in ("auth_url", "api_url", "client_id", "resource", "scopes") if k in options}
        base, bearer = service_access("agent", "", url=url, token=token, **auth)
        return cls(base, headers=lambda method, url: {"Authorization": f"Bearer {bearer()}"}, **options)

    def close(self) -> None:
        if isinstance(self._http, httpx.Client):
            self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def request(self, method: str, path: str, **kw) -> Any:
        url = f"{self.base}/api/v1{path}"
        headers = {**self._auth, **(kw.pop("headers", None) or {})}
        if self._headers:
            headers.update(self._headers(method, url))
        response = self._http.request(method, url, headers=headers, **kw)
        if response.status_code == 204:
            return None
        try:
            body = response.json()
        except ValueError:
            body = response.text[:300]
        if response.status_code >= 400:
            raise AgentClientError(response.status_code, body.get("detail", body) if isinstance(body, dict) else body)
        return body

    # -- chat sessions ------------------------------------------------------------------------------
    def me(self) -> dict:
        return self.request("GET", "/me")

    def sessions(self) -> list[dict]:
        return self.request("GET", "/sessions")["sessions"]

    def create_session(self, title: str = "New chat") -> dict:
        return self.request("POST", "/sessions", json={"title": title})

    def session(self, sid: str) -> dict:
        return self.request("GET", f"/sessions/{sid}")

    def delete_session(self, sid: str) -> None:
        self.request("DELETE", f"/sessions/{sid}")

    def send(self, sid: str, text: str, wait: bool = True, **options) -> dict:
        """The reply (`status` succeeded), what the agent waits for (`waiting`), or `running` when
        it takes longer than the service waits; options: task_type, mode, model."""
        body = {"text": text, **{k: v for k, v in options.items() if v is not None}}
        return self.request("POST", f"/sessions/{sid}/messages", params={"wait": str(wait).lower()}, json=body)

    def session_status(self, sid: str) -> dict:
        return self.request("GET", f"/sessions/{sid}/status")

    def decide(self, call_id: str, approve: bool = True, reason: str = "", remember: bool = False,
               session: str | None = None, task: str | None = None) -> dict:
        """Approve or reject a tool call that waits, in a chat `session` or a background `task`."""
        where = f"/sessions/{session}" if session else f"/tasks/{task}"
        return self.request("POST", f"{where}/approvals/{call_id}", json={
            "decision": "approve" if approve else "reject", "reason": reason, "remember": remember})

    # -- background tasks ---------------------------------------------------------------------------
    def submit_task(self, prompt: str, **options) -> dict:
        """{"task_id", "status": "queued"}; options: task_type, mode, model."""
        return self.request("POST", "/tasks", json={"prompt": prompt, **{k: v for k, v in options.items() if v is not None}})

    def tasks(self) -> list[dict]:
        return self.request("GET", "/tasks")["tasks"]

    def task(self, tid: str) -> dict:
        return self.request("GET", f"/tasks/{tid}")

    def answer_task(self, tid: str, text: str) -> dict:
        return self.request("POST", f"/tasks/{tid}/input", json={"text": text})

    def cancel_task(self, tid: str) -> dict:
        return self.request("POST", f"/tasks/{tid}/cancel")

    def wait_task(self, tid: str, interval: float = 2.0, timeout: float | None = None) -> dict:
        """Poll until the task is finished or waits for someone (approval, answer)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            task = self.task(tid)
            if task["status"] in FINISHED or task["status"] == "waiting":
                return task
            if deadline is not None and time.monotonic() >= deadline:
                return task
            time.sleep(interval)
