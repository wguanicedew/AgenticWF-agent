"""Routing end to end: the API's router, its usage records, budgets, ratings (docs/model-routing.md)."""

import time

from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import RuntimeConfig

from tests.conftest import SECRET, Client, user
from tests.test_runtime import make_system_dir

SECTIONS = {
    "llm": {"cost_unit": "CHF"},
    "llm.model.small": {"provider": "echo", "model": "echo", "tier": "small", "context_window": "131072",
                        "cost_input": "1", "cost_output": "1"},
    "llm.model.big": {"provider": "echo", "model": "echo", "tier": "large", "context_window": "1048576",
                      "cost_input": "10", "cost_output": "10"},
    "llm.budgets": {"default": "100 per day"},
}


def poll(client, path, until, tries=200):
    for _ in range(tries):
        body = client.get(path).json()
        if until(body):
            return body
        time.sleep(0.02)
    raise AssertionError(body)


def test_usage_budget_and_ratings_through_the_api(tmp_path):
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), model="small", llm_sections=SECTIONS)
    app = create_app(Settings(context_secret=SECRET), runtime_config=config)
    alice, admin = user("user_alice"), user("boss", scopes=("app:access", "agent:use", "admin"))
    with TestClient(app) as client:
        a, boss = Client(client, alice), Client(client, admin)
        sid = a.post("/api/v1/sessions", json={}).json()["id"]
        a.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hello", "wait": True})
        session = poll(a, f"/api/v1/sessions/{sid}", lambda s: len(s["messages"]) >= 2)
        intent = session["messages"][-1]["intent"]

        usage = a.get("/api/v1/usage?group=purpose").json()
        purposes = {row["purpose"]: row for row in usage["usage"]}
        assert usage["cost_unit"] == "CHF" and "step" in purposes
        by_model = {row["model"]: row for row in a.get("/api/v1/usage").json()["usage"]}
        assert set(by_model) == {"big", "small"}
        assert a.get("/api/v1/usage?all=true").status_code == 403
        assert boss.get("/api/v1/usage?all=true&group=owner").json()["usage"][0]["owner"] == "user_alice"

        budget = a.get("/api/v1/usage/budget").json()
        assert budget["budget"] == 100 and budget["period"] == "day" and budget["routing"] is True
        assert budget["spent"] == sum(row["cost"] for row in by_model.values())   # echo reports no tokens: 0

        assert a.post("/api/v1/feedback", json={"intent": intent, "rating": "down"}).json() == {"rated": 1}
        assert a.get(f"/api/v1/feedback/{intent}").json()["rating"] == "down"
        assert a.post("/api/v1/feedback", json={"intent": intent, "rating": "up"}).json() == {"rated": 1}
        assert a.get(f"/api/v1/feedback/{intent}").json()["rating"] == "up"          # replaced, not added
        assert Client(client, user("user_bob")).post("/api/v1/feedback",
                                                     json={"intent": intent, "rating": "down"}).status_code == 404
        assert a.get("/api/v1/usage/suggestions").status_code == 403
        rows = boss.get("/api/v1/usage/suggestions").json()["rows"]
        routed = {(r["purpose"], r["model"]) for r in rows}
        assert {("step", "big"), ("consolidate", "small")} <= routed           # a step needs medium; memory, small
        big = next(r for r in rows if r["model"] == "big" and r["purpose"] == "step")
        assert (big["up"], big["down"]) == (1, 0) and big["calls"] >= 1
