"""`agenticwf agent` through the `agenticwf` command (agenticwf-auth-client), against a mock agent.
Needs both client packages installed (the entry point): pip install -e ../AgenticWF-auth/cli -e client"""

import json

import httpx
import pytest

cli = pytest.importorskip("agenticwf_auth_cli.__main__")


@pytest.fixture
def agent(monkeypatch):
    state = {"requests": [], "polls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["requests"].append(request)
        if request.headers.get("Authorization") != "Bearer local-token":
            return httpx.Response(401, json={"detail": "invalid token"})
        path, method = request.url.path, request.method
        if (method, path) == ("POST", "/api/v1/sessions"):
            return httpx.Response(201, json={"id": "s1", "title": json.loads(request.content)["title"]})
        if (method, path) == ("POST", "/api/v1/sessions/s1/messages"):
            if json.loads(request.content)["text"] == "delete it":
                return httpx.Response(200, json={"session": "s1", "intent": "i2", "status": "waiting", "waiting": [
                    {"task": "t1", "state": "waiting_approval", "questions": [],
                     "approvals": [{"call": "c9", "name": "delete_file", "arguments": {"path": "a"}, "reason": ""}]}]})
            return httpx.Response(200, json={"session": "s1", "intent": "i1", "status": "succeeded", "reply": "hi!"})
        if (method, path) == ("POST", "/api/v1/sessions/s1/approvals/c9"):
            return httpx.Response(200, json={"decision": json.loads(request.content)["decision"]})
        if (method, path) == ("POST", "/api/v1/tasks"):
            return httpx.Response(202, json={"task_id": "x1", "status": "queued"})
        if (method, path) == ("GET", "/api/v1/tasks/x1"):
            state["polls"] += 1
            done = state["polls"] >= 2
            return httpx.Response(200, json={"id": "x1", "status": "succeeded" if done else "running",
                                             "prompt": "p", "result": "42" if done else None, "error": None})
        return httpx.Response(404, json={"detail": "not found"})

    class MockClient(httpx.Client):
        def __init__(self, **kw):
            super().__init__(transport=httpx.MockTransport(handler), **kw)

    monkeypatch.setattr(httpx, "Client", MockClient)
    monkeypatch.setattr("agenticwf.agent.client.time.sleep", lambda s: None)
    monkeypatch.setenv("AGENTICWF_AGENT_URL", "http://agent.test")
    monkeypatch.setenv("AGENTICWF_AGENT_TOKEN", "local-token")
    return state


def test_chat_new_session_and_reply(agent, capsys):
    assert cli.main(["agent", "chat", "hello   there"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == "hi!" and "session: s1" in out.err
    assert json.loads(agent["requests"][0].content) == {"title": "hello there"}
    assert agent["requests"][1].url.params["wait"] == "true"


def test_chat_waiting_and_approve(agent, capsys):
    assert cli.main(["agent", "chat", "delete it", "--session", "s1", "--mode", "auto"]) == 0
    assert json.loads(agent["requests"][0].content) == {"text": "delete it", "mode": "auto"}
    out = capsys.readouterr().out
    assert "approval needed: delete_file" in out and "agenticwf agent approve c9 --session s1" in out

    assert cli.main(["agent", "approve", "c9", "--session", "s1", "--reject", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {"decision": "reject"}


def test_task_wait(agent, capsys):
    assert cli.main(["agent", "task", "compute", "--wait"]) == 0
    assert capsys.readouterr().out.strip() == "42" and agent["polls"] == 2


def test_errors(agent, monkeypatch, capsys):
    assert cli.main(["agent", "tasks", "nope"]) == 1
    assert "HTTP 404: not found" in capsys.readouterr().err
    monkeypatch.setenv("AGENTICWF_AGENT_TOKEN", "wrong")
    assert cli.main(["agent", "sessions"]) == 1
    assert "invalid token" in capsys.readouterr().err


def test_from_env_uses_logto_tokens(monkeypatch):
    """Without the agent's own token, `agenticwf login`'s tokens, refreshed on each request."""
    import agenticwf_auth_cli
    from agenticwf.agent.client import AgentClient

    tokens = iter(["t1", "t2"])
    monkeypatch.delenv("AGENTICWF_AGENT_TOKEN", raising=False)
    monkeypatch.setattr(agenticwf_auth_cli, "service_access", lambda *a, **kw: ("http://app/app", lambda: next(tokens)))
    seen = []
    http = httpx.Client(transport=httpx.MockTransport(
        lambda r: seen.append(r.headers["Authorization"]) or httpx.Response(200, json={"sessions": []})))
    agent = AgentClient.from_env(http=http)
    agent.sessions(), agent.sessions()
    assert agent.base == "http://app/app" and seen == ["Bearer t1", "Bearer t2"]
