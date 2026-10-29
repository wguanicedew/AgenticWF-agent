import time

from agenticwf.auth.client import current_auth_context
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.llm.base import LLMResponse
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.tools.spec import ToolResult

from tests.conftest import SECRET, Client, machine, user
from tests.test_runtime import call, make_system_dir

TOOL_SCOPES = ("app:access", "agent:use", "code:read", "code:write")


def scripted_app(tmp_path, steps, handlers=None):
    """The API over the runtime with a scripted model and the test tools (read/write/delete_thing)."""
    handlers = {"test.read_thing": lambda ctx, args: ToolResult(content="thing"),
                "test.delete_thing": lambda ctx, args: ToolResult(content="deleted"), **(handlers or {})}
    llm = steps if isinstance(steps, ScriptedLLM) else ScriptedLLM(steps)
    return create_app(Settings(context_secret=SECRET),
                      runtime_config=RuntimeConfig(system_dir=make_system_dir(tmp_path)),
                      engine=EngineOptions(llm=llm, handlers=handlers))


def poll(client, path, until, tries=100):
    for _ in range(tries):
        body = client.get(path).json()
        if until(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"{path}: {body}")


def test_session_lifecycle(as_user):
    alice = as_user(user())
    sid = alice.post("/api/v1/sessions", json={"title": "t"}).json()["id"]
    r = alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi"})
    body = r.json()
    assert body["session"] == sid and body["status"] == "succeeded" and body["reply"] == "echo: hi"
    detail = alice.get(f"/api/v1/sessions/{sid}").json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]
    assert [s["id"] for s in alice.get("/api/v1/sessions").json()["sessions"]] == [sid]
    assert alice.delete(f"/api/v1/sessions/{sid}").status_code == 204
    assert alice.get(f"/api/v1/sessions/{sid}").status_code == 404


def test_sessions_isolated_by_user_and_organization(as_user):
    alice = as_user(user())
    sid = alice.post("/api/v1/sessions", json={}).json()["id"]
    for other in (user(subject="user_bob"), user(org="org_cern")):
        c = as_user(other)
        assert c.get(f"/api/v1/sessions/{sid}").status_code == 404
        assert c.post(f"/api/v1/sessions/{sid}/messages", json={"text": "x"}).status_code == 404
        assert c.delete(f"/api/v1/sessions/{sid}").status_code == 404
        assert c.get("/api/v1/sessions").json() == {"sessions": []}


def test_task_runs_with_captured_context(tmp_path):
    seen = {}

    def read_thing(ctx, args):
        seen["ambient"] = current_auth_context()   # handlers see the task's AuthContext
        return ToolResult(content="thing")

    app = scripted_app(tmp_path, [LLMResponse("", [call("read_thing")]), LLMResponse("GO")],
                       {"test.read_thing": read_thing})
    with TestClient(app) as tc:
        bot = Client(tc, machine(scopes=TOOL_SCOPES))
        tid = bot.post("/api/v1/tasks", json={"prompt": "go"}).json()["task_id"]
        task = poll(bot, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "succeeded")
        assert task["result"] == "GO" and task["prompt"] == "go"
        assert [t["state"] for t in task["tasks"]] == ["completed"]
        assert seen["ambient"].subject == "app_m2m"
        turns = bot.get(f"/api/v1/tasks/{tid}/turns").json()["tasks"][0]
        assert [c["name"] for c in turns["tool_calls"]] == ["read_thing"]
        assert Client(tc, user()).get(f"/api/v1/tasks/{tid}").status_code == 404
        assert Client(tc, user()).get(f"/api/v1/tasks/{tid}/turns").status_code == 404
        assert [t["id"] for t in bot.get("/api/v1/tasks").json()["tasks"]] == [tid]


def test_failed_task(tmp_path):
    with TestClient(scripted_app(tmp_path, [])) as tc:   # the model is unavailable
        c = Client(tc, user())
        tid = c.post("/api/v1/tasks", json={"prompt": "go"}).json()["task_id"]
        task = poll(c, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "failed")
        assert "script exhausted" in task["error"] and task["tasks"][0]["error"]["code"] == "model_unavailable"


def test_task_approval(tmp_path):
    with TestClient(scripted_app(tmp_path, [LLMResponse("", [call("delete_thing")]), LLMResponse("gone")])) as tc:
        alice = Client(tc, user(scopes=TOOL_SCOPES))
        tid = alice.post("/api/v1/tasks", json={"prompt": "delete it"}).json()["task_id"]
        poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "waiting")
        call_id = alice.get(f"/api/v1/tasks/{tid}/turns").json()["tasks"][0]["tool_calls"][0]["id"]

        bot = Client(tc, machine(scopes=TOOL_SCOPES))
        assert bot.post(f"/api/v1/tasks/{tid}/approvals/{call_id}", json={"decision": "approve"}).status_code == 403
        bob = Client(tc, user(subject="user_bob", scopes=TOOL_SCOPES))
        assert bob.post(f"/api/v1/tasks/{tid}/approvals/{call_id}", json={"decision": "approve"}).status_code == 404
        assert alice.post(f"/api/v1/tasks/{tid}/approvals/nope", json={"decision": "approve"}).status_code == 409

        r = alice.post(f"/api/v1/tasks/{tid}/approvals/{call_id}", json={"decision": "approve"})
        assert r.status_code == 200 and r.json()["intent"] == tid
        assert poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "succeeded")["result"] == "gone"


def test_task_input_and_cancel(tmp_path):
    steps = [LLMResponse("", [call("ask_user", question="which one?")]), LLMResponse("the blue one")]
    with TestClient(scripted_app(tmp_path, steps)) as tc:
        alice = Client(tc, user())
        tid = alice.post("/api/v1/tasks", json={"prompt": "pick"}).json()["task_id"]
        poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "waiting")
        assert alice.post(f"/api/v1/tasks/{tid}/input", json={"text": "blue"}).status_code == 200
        assert poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "succeeded")["result"] == "the blue one"
        assert alice.post(f"/api/v1/tasks/{tid}/cancel").status_code == 409            # finished
        assert alice.post(f"/api/v1/tasks/{tid}/input", json={"text": "again"}).status_code == 409

        tid2 = alice.post("/api/v1/tasks", json={"prompt": "ask again"}).json()["task_id"]   # model exhausted
        poll(alice, f"/api/v1/tasks/{tid2}", lambda t: t["status"] == "failed")


def test_cancel_waiting_task(tmp_path):
    with TestClient(scripted_app(tmp_path, [LLMResponse("", [call("delete_thing")])])) as tc:
        alice = Client(tc, user(scopes=TOOL_SCOPES))
        tid = alice.post("/api/v1/tasks", json={"prompt": "delete"}).json()["task_id"]
        poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "waiting")
        assert alice.post(f"/api/v1/tasks/{tid}/cancel").json()["status"] == "cancelling"
        assert poll(alice, f"/api/v1/tasks/{tid}", lambda t: t["status"] == "cancelled")["tasks"][0]["state"] == "cancelled"


def test_invalid_task_type(as_user):
    assert as_user(user()).post("/api/v1/tasks", json={"prompt": "x", "task_type": "../x"}).status_code == 422


def test_chat_waits_for_approval(tmp_path):
    with TestClient(scripted_app(tmp_path, [LLMResponse("", [call("delete_thing")]), LLMResponse("gone")])) as tc:
        alice = Client(tc, user(scopes=TOOL_SCOPES))
        sid = alice.post("/api/v1/sessions", json={}).json()["id"]
        r = alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "delete it"}).json()
        assert r["status"] == "waiting"
        approval = r["waiting"][0]["approvals"][0]
        assert approval["name"] == "delete_thing"
        alice.post(f"/api/v1/sessions/{sid}/approvals/{approval['call']}", json={"decision": "approve"})
        detail = poll(alice, f"/api/v1/sessions/{sid}", lambda s: s["messages"] and s["messages"][-1]["role"] == "assistant")
        assert [m["text"] for m in detail["messages"]] == ["delete it", "gone"]
        assert detail["messages"][0]["intent"] == r["intent"] == detail["messages"][1]["intent"]


def test_chat_without_waiting(as_user):
    alice = as_user(user())
    sid = alice.post("/api/v1/sessions", json={}).json()["id"]
    r = alice.post(f"/api/v1/sessions/{sid}/messages?wait=false", json={"text": "later"})
    assert r.status_code == 202 and r.json()["status"] == "accepted"
    poll(alice, f"/api/v1/sessions/{sid}", lambda s: len(s["messages"]) == 2)


def test_event_stream_ends_when_token_expires(client):
    from tests.conftest import signed

    ctx = user(ttl=1)
    with client.stream("GET", "/api/v1/events", headers=signed(ctx, "GET", "/api/v1/events")) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        body = "".join(r.iter_text())
    assert "event: status" in body and '"state": "connected"' in body


def test_data_survives_restart(as_user):
    from fastapi.testclient import TestClient

    from agenticwf.agent.api.app import create_app
    from agenticwf.agent.api.settings import Settings
    from tests.conftest import SECRET, Client

    sid = as_user(user()).post("/api/v1/sessions", json={"title": "kept"}).json()["id"]
    as_user(user()).post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi"})
    with TestClient(create_app(Settings(context_secret=SECRET))) as tc:   # a fresh process
        detail = Client(tc, user()).get(f"/api/v1/sessions/{sid}").json()
    assert detail["title"] == "kept" and [m["text"] for m in detail["messages"]] == ["hi", "echo: hi"]


def test_a_chat_summary_is_kept_and_reused_after_a_restart(tmp_path):
    from agenticwf.agent.runtime.harness.harness import Limits

    def model(request):
        if "earlier part of this conversation" in request.system:
            return LLMResponse("SUMMARY of " + str(request.messages[0].content.count("user: ")) + " requests")
        return LLMResponse("answer to " + request.messages[-1].content)

    def app(path, llm):
        return create_app(Settings(context_secret=SECRET),
                          runtime_config=RuntimeConfig(system_dir=make_system_dir(path),
                                                       limits=Limits(conversation_messages=2, conversation_keep=2)),
                          engine=EngineOptions(llm=llm))

    first = ScriptedLLM([model] * 10)
    with TestClient(app(tmp_path, first)) as tc:
        alice = Client(tc, user())
        sid = alice.post("/api/v1/sessions", json={}).json()["id"]
        for text in ("one", "two", "three"):
            alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": text, "wait": True})
        stored = alice.get(f"/api/v1/sessions/{sid}").json()
    assert stored["summary"] == "SUMMARY of 1 requests"
    one = next(m["intent"] for m in stored["messages"] if m["text"] == "one")
    assert stored["summary_keys"] == [one]

    again = ScriptedLLM([model] * 10)                                     # a fresh process
    with TestClient(app(tmp_path / "again", again)) as tc:
        alice = Client(tc, user())
        alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "four", "wait": True})
    fold = again.requests[0].messages[0].content                          # starts from the stored summary
    assert "<summary so far>\nSUMMARY of 1 requests" in fold and "user: two" in fold and "user: one" not in fold
    task = [m.content for m in again.requests[1].messages]
    assert "SUMMARY of 1 requests" in task[0] and "one" not in task[1:]


def test_waiting_approval_survives_restart(tmp_path):
    llm = ScriptedLLM([LLMResponse("", [call("delete_thing")]), LLMResponse("gone after restart")])
    alice_ctx = user(scopes=TOOL_SCOPES)
    with TestClient(scripted_app(tmp_path, llm)) as tc:
        alice = Client(tc, alice_ctx)
        sid = alice.post("/api/v1/sessions", json={}).json()["id"]
        r = alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "delete it"}).json()
        call_id = r["waiting"][0]["approvals"][0]["call"]
    # The process stops; a new one loads the waiting task from the database.
    with TestClient(scripted_app(tmp_path / "again", llm)) as tc:
        alice = Client(tc, alice_ctx)
        r = alice.post(f"/api/v1/sessions/{sid}/approvals/{call_id}", json={"decision": "approve"})
        assert r.status_code == 200
        detail = poll(alice, f"/api/v1/sessions/{sid}", lambda s: s["messages"][-1]["role"] == "assistant")
        assert [m["text"] for m in detail["messages"]] == ["delete it", "gone after restart"]


def test_unanswered_intents_resume_on_startup():
    from agenticwf.agent.runtime.harness.db_store import DatabaseStore
    from agenticwf.agent.runtime.records import Intent

    ctx = user()
    intent = Intent(thread_id="t" * 32, owner=(ctx.subject, None), auth=ctx, text="left over", source="api")
    DatabaseStore().save_intent(intent)   # accepted, then the process stopped before intake
    with TestClient(create_app(Settings(context_secret=SECRET))) as tc:
        task = poll(Client(tc, ctx), f"/api/v1/tasks/{intent.id}", lambda t: t["status"] == "succeeded")
    assert task["result"] == "echo: left over"


def test_chat_page_endpoints(tmp_path):
    """What the chat page uses besides messages: rename, status, task types, per-message type and mode."""
    with TestClient(scripted_app(tmp_path, [LLMResponse("", [call("delete_thing")]), LLMResponse("planned")])) as tc:
        alice = Client(tc, user(scopes=TOOL_SCOPES))
        sid = alice.post("/api/v1/sessions", json={"title": "New chat"}).json()["id"]
        assert alice.patch(f"/api/v1/sessions/{sid}", json={"title": "Renamed"}).json()["title"] == "Renamed"
        assert Client(tc, user(subject="user_bob")).patch(f"/api/v1/sessions/{sid}", json={"title": "x"}).status_code == 404
        assert alice.get(f"/api/v1/sessions/{sid}/status").json()["status"] == "idle"

        types = alice.get("/api/v1/task-types").json()["task_types"]
        assert ("review", "system") in [(t["name"], t["source"]) for t in types]

        assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "x", "task_type": "../x"}).status_code == 422
        r = alice.post(f"/api/v1/sessions/{sid}/messages?wait=false", json={"text": "delete it", "mode": "default"})
        assert r.status_code == 202
        status = poll(alice, f"/api/v1/sessions/{sid}/status", lambda s: s["status"] == "waiting")
        assert status["waiting"][0]["approvals"][0]["name"] == "delete_thing"


def test_page_is_self_contained(as_user):
    page = as_user(user()).get("/").text
    assert "<script src" not in page and "<link rel=\"stylesheet\"" not in page   # nothing from a CDN
