"""The trace view (agenticwf.agent.api.trace): what the agent did for a message or a task."""

import datetime as dt

from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.api.trace import intent_trace
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.llm.base import LLMResponse
from agenticwf.agent.runtime.llm.usage import DatabaseUsage, InMemoryUsage
from agenticwf.agent.runtime.records import Intent

from tests.conftest import SECRET, Client, user
from tests.test_api import TOOL_SCOPES, scripted_app
from tests.test_runtime import call, make_system_dir

ALICE, BOB = ("user_a", None), ("user_b", None)


def test_a_chat_trace_has_the_steps_and_their_tool_calls(tmp_path):
    steps = [LLMResponse("Let me look.", [call("read_thing", value="x")]), LLMResponse("It is a thing.")]
    with TestClient(scripted_app(tmp_path, steps)) as tc:
        alice, bob = Client(tc, user(scopes=TOOL_SCOPES)), Client(tc, user(subject="user_b", scopes=TOOL_SCOPES))
        sid = alice.post("/api/v1/sessions", json={}).json()["id"]
        sent = alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "what is it?"}).json()
        assert sent["reply"] == "It is a thing."

        body = alice.get(f"/api/v1/sessions/{sid}/trace").json()
        [trace] = body["intents"]
        assert trace["intent"]["id"] == sent["intent"] and trace["intent"]["text"] == "what is it?"
        assert trace["reply"]["answer"] == "It is a thing."
        [task] = trace["tasks"]
        assert task["status"] == "succeeded"
        steps = [s for turn in task["turns"] for s in turn["steps"]]
        assert [s["text"] for s in steps] == ["Let me look.", "It is a thing."]
        assert steps[0]["started_at"] and steps[0]["seconds"] is not None
        [tool] = steps[0]["tool_calls"]
        assert tool["name"] == "read_thing" and '"value": "x"' in tool["arguments"]
        assert tool["result"]["content"] == "thing" and tool["status"] == "succeeded" and tool["seconds"] is not None
        assert trace["totals"]["tool_calls"] == 1

        assert alice.get(f"/api/v1/sessions/{sid}/trace", params={"intent": sent["intent"]}).json()["intents"]
        assert alice.get(f"/api/v1/sessions/{sid}/trace", params={"intent": "nope"}).status_code == 404
        assert bob.get(f"/api/v1/sessions/{sid}/trace").status_code == 404            # not theirs

        task_id = alice.post("/api/v1/tasks", json={"prompt": "again"}).json()["task_id"]
        assert alice.get(f"/api/v1/tasks/{task_id}/trace").json()["intent"]["id"] == task_id
        assert bob.get(f"/api/v1/tasks/{task_id}/trace").status_code == 404


def test_model_calls_are_listed_with_their_cost(tmp_path):
    app = create_app(Settings(context_secret=SECRET), runtime_config=RuntimeConfig(system_dir=make_system_dir(tmp_path),
                                                                                   provider="echo"),
                     engine=EngineOptions())
    with TestClient(app) as tc:
        alice = Client(tc, user())
        sid = alice.post("/api/v1/sessions", json={}).json()["id"]
        alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hello"})
        [trace] = alice.get(f"/api/v1/sessions/{sid}/trace").json()["intents"]
    calls = trace["model_calls"]
    assert calls and {c["purpose"] for c in calls} >= {"step"} and all(c["seconds"] is not None for c in calls)
    assert trace["totals"]["model_calls"] == len(calls)
    step = trace["tasks"][0]["turns"][0]["steps"][0]
    assert step["model_id"] and step["route"]


def test_usage_calls_are_per_owner_and_intent():
    for usage in (InMemoryUsage(), DatabaseUsage()):
        row = {"purpose": "step", "model_id": "m", "input_tokens": 1, "output_tokens": 2, "cost": 0.5}
        usage.record({**row, "owner_subject": "user_a", "owner_org": None, "intent_id": "i1", "task_id": None})
        usage.record({**row, "owner_subject": "user_a", "owner_org": None, "intent_id": None, "task_id": "t1",
                      "seconds": 1.5, "trace_id": "ab" * 16})
        usage.record({**row, "owner_subject": "user_b", "owner_org": None, "intent_id": "i1", "task_id": "t1"})
        usage.record({**row, "owner_subject": "user_a", "owner_org": None, "intent_id": "i2", "task_id": "t2"})
        found = usage.calls(ALICE, "i1", ["t1"])
        assert len(found) == 2 and found[1]["trace_id"] == "ab" * 16 and found[1]["seconds"] == 1.5


def test_trace_ids_are_linked():
    intent = Intent(thread_id="th", owner=ALICE, auth=None, text="x")
    rows = [{"at": dt.datetime.now(dt.timezone.utc), "purpose": "split", "model_id": "m", "input_tokens": 3,
             "output_tokens": 1, "cost": 0.25, "trace_id": "cd" * 16}]
    trace = intent_trace(intent, [], None, rows, trace_url="http://jaeger/trace/{trace_id}", cost_unit="CHF")
    assert trace["traces"] == [{"trace_id": "cd" * 16, "url": "http://jaeger/trace/" + "cd" * 16}]
    assert trace["model_calls"][0]["trace_url"].endswith("cd" * 16) and trace["totals"]["cost_unit"] == "CHF"
    assert intent_trace(intent, [], None, rows)["traces"] == [{"trace_id": "cd" * 16, "url": None}]
