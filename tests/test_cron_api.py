"""The crons' API and page (src/agenticwf/agent/api/crons.py, docs/cron.md §10), over HTTP."""

import time

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.cron.service import CronSettings
from agenticwf.agent.runtime.tools.spec import ToolResult

from tests.conftest import SECRET, Client, machine, user
from tests.test_cron import CRON_MD, JOBS, TRANSFERS, Model, call, issue_call
from tests.test_runtime import make_system_dir

SHIFTER = user("user_s", scopes=("app:access", "agent:use", "adcos:shift", "code:write"), org="cern")
READER = user("user_r", scopes=("app:access", "agent:use", "atlas:member"), org="cern")
STRANGER = user("user_x", org="other")
ADMIN = user("boss", scopes=("app:access", "agent:use", "adcos:shift", "admin"), org="cern")


@pytest.fixture
def api(tmp_path):
    system = make_system_dir(tmp_path)
    cron_dir = system / "crons" / "shift"
    (cron_dir / "checks").mkdir(parents=True)
    (cron_dir / "reports").mkdir()
    (cron_dir / "CRON.md").write_text(CRON_MD)
    (cron_dir / "checks" / "transfers.md").write_text(TRANSFERS)
    (cron_dir / "checks" / "jobs.md").write_text(JOBS)
    (cron_dir / "reports" / "period.md").write_text("Lead with the worst site.")
    model = Model()
    app = create_app(Settings(context_secret=SECRET),
                     runtime_config=RuntimeConfig(system_dir=system, cron=CronSettings(tick=3600)),
                     engine=EngineOptions(llm=model, handlers={
                         "test.read_thing": lambda ctx, args: ToolResult(content="thing"),
                         "test.delete_thing": lambda ctx, args: ToolResult(content="deleted")}))
    with TestClient(app) as client:
        yield (lambda ctx: Client(client, ctx)), model


def poll(client, path, until, tries=200):
    for _ in range(tries):
        body = client.get(path).json()
        if until(body):
            return body
        time.sleep(0.02)
    raise AssertionError(f"{path}: {body}")


def test_who_sees_what(api):
    as_, _ = api
    shifter, reader, stranger = as_(SHIFTER), as_(READER), as_(STRANGER)
    [cron] = shifter.get("/api/v1/crons").json()["crons"]
    assert cron["id"] == "system:shift" and cron["role"] == "operate" and cron["source"] == "system"
    assert [c["role"] for c in reader.get("/api/v1/crons").json()["crons"]] == ["read"]
    assert stranger.get("/api/v1/crons").json()["crons"] == []
    assert stranger.get("/api/v1/crons/system:shift").status_code == 404
    assert "instructions" in shifter.get("/api/v1/crons/system:shift").json()
    assert "instructions" not in reader.get("/api/v1/crons/system:shift").json()     # readers: reports and the board
    assert reader.get("/api/v1/crons/system:shift/runs").status_code == 403
    assert reader.post("/api/v1/crons/system:shift/run", json={}).status_code == 403
    assert as_(machine()).post("/api/v1/crons/system:shift/run", json={}).status_code == 403
    assert "<title>AgenticWF · Crons</title>" in shifter.get("/crons").text
    assert "Look at the transfers." in shifter.get("/api/v1/crons/system:shift/files/checks/transfers.md").json()["content"]
    assert shifter.get("/api/v1/crons/system:shift/files/../../tools/read_thing.md").status_code != 200
    assert reader.get("/api/v1/crons/system:shift/files/CRON.md").status_code == 403
    assert shifter.get("/api/v1/issues?limit=1").json()["truncated"] is False


def test_a_run_from_the_board_to_a_report(api):
    as_, model = api
    shifter, reader = as_(SHIFTER), as_(READER)
    model.plan["transfers"] = [[issue_call("SITE_A"), issue_call("SITE_B", severity="low"),
                                call("delete_thing", value="stuck"),
                                call("report_check", status="issues",
                                     metrics=[{"name": "failure_rate", "value": 0.3, "labels": {"site": "SITE_A"},
                                               "link": "https://mon.example/a"}])]]
    started = shifter.post("/api/v1/crons/system:shift/run", json={"checks": ["transfers"]})
    assert started.status_code == 202
    run_id = started.json()["id"]
    run = poll(shifter, f"/api/v1/crons/system:shift/runs/{run_id}", lambda r: r["status"] == "finished")
    assert run["checks"]["transfers"]["new"] == 2

    board = reader.get("/api/v1/issues?cron=system:shift&group=site").json()["sites"]   # readers see the board
    assert [s["site"] for s in board] == ["SITE_A", "SITE_B"]
    site_a = board[0]["issues"][0]
    detail = reader.get(f"/api/v1/issues/{site_a['id']}").json()
    assert detail["can_operate"] is False and [e["action"] for e in detail["events"]] == ["opened"]
    assert reader.post(f"/api/v1/issues/{site_a['id']}/actions", json={"action": "acknowledge"}).status_code == 403
    acked = shifter.post(f"/api/v1/issues/{site_a['id']}/actions", json={"action": "acknowledge", "comment": "mine"})
    assert acked.json()["issues"][0]["status"] == "acknowledged"
    snoozed = shifter.post("/api/v1/issues/actions", json={"ids": [i["id"] for s in board for i in s["issues"]],
                                                          "action": "snooze", "until": "2099-01-01T00:00:00Z"})
    assert {i["status"] for i in snoozed.json()["issues"]} == {"snoozed"}
    assert shifter.post("/api/v1/issues/actions", json={"ids": [site_a["id"]], "action": "snooze"}).status_code == 422

    suppression = shifter.post("/api/v1/crons/system:shift/suppressions",
                               json={"check": "transfers", "match": {"site": "SITE_C"}, "reason": "gone"})
    assert suppression.status_code == 201
    assert len(reader.get("/api/v1/crons/system:shift/suppressions").json()["suppressions"]) == 1
    assert shifter.delete(f"/api/v1/crons/system:shift/suppressions/{suppression.json()['id']}").status_code == 204

    [proposal] = reader.get("/api/v1/crons/system:shift/proposals").json()["proposals"]
    assert proposal["tool"] == "delete_thing" and proposal["issue_id"] == board[1]["issues"][0]["id"]   # the last one reported
    assert reader.post(f"/api/v1/proposals/{proposal['id']}/decision", json={"approve": True}).status_code == 403
    decided = shifter.post(f"/api/v1/proposals/{proposal['id']}/decision", json={"approve": True})
    assert decided.json()["status"] == "approved" and decided.json()["result"]["content"] == "deleted"
    assert shifter.post(f"/api/v1/proposals/{proposal['id']}/decision", json={"approve": False}).status_code == 409

    metrics = reader.get("/api/v1/crons/system:shift/metrics?name=failure_rate").json()
    assert metrics["metrics"] == ["failure_rate"] and metrics["series"][0]["links"] == ["https://mon.example/a"]

    asked = shifter.post("/api/v1/crons/system:shift/reports",
                         json={"from": "2000-01-01T00:00:00Z", "to": "2100-01-01T00:00:00Z"})
    assert asked.status_code == 202
    report = poll(reader, f"/api/v1/reports/{asked.json()['id']}", lambda r: r["status"] != "pending")
    assert report["status"] == "ready" and report["kind"] == "custom" and report["can_operate"] is False
    found = reader.get("/api/v1/reports?cron=system:shift&year=2000&month=1&day=1").json()
    assert found["total"] == 1 and found["reports"][0]["id"] == report["id"]
    assert reader.get("/api/v1/reports?year=2001").json()["total"] == 0
    assert as_(STRANGER).get(f"/api/v1/reports/{report['id']}").status_code == 404
    markdown = reader.get(f"/api/v1/reports/{report['id']}?format=markdown")
    assert markdown.headers["content-type"].startswith("text/markdown") and "## Per day" in markdown.text


def test_people_manage_their_own_database_crons(api):
    as_, _ = api
    owner = user("user_o", scopes=("app:access", "agent:use"), org="cern")
    mine = as_(owner)
    body = {"name": "mine", "content": CRON_MD.replace("name: shift", "name: mine", 1),
            "files": {"checks/jobs.md": JOBS, "reports/period.md": "x"}, "visibility": "organization"}
    created = mine.post("/api/v1/crons", json=body)
    assert created.status_code == 201 and created.json()["id"] == "user_o@cern:mine"
    assert mine.post("/api/v1/crons", json=body).status_code == 409
    assert mine.post("/api/v1/crons", json={**body, "name": "all", "visibility": "all"}).status_code == 403
    assert mine.post("/api/v1/crons", json={**body, "name": "bad", "files": {}}).status_code == 422   # no checks
    assert as_(machine()).post("/api/v1/crons", json={**body, "name": "m"}).status_code == 403

    colleague = as_(user("user_c", org="cern"))
    assert any(c["id"] == "user_o@cern:mine" for c in colleague.get("/api/v1/crons").json()["crons"])
    assert colleague.patch("/api/v1/crons/user_o@cern:mine", json={"enabled": False}).status_code == 403
    assert mine.put("/api/v1/crons/mine/files/checks/transfers.md", json={"content": TRANSFERS}).status_code == 200
    assert mine.get("/api/v1/crons/mine/files/checks/transfers.md").json()["content"] == TRANSFERS
    assert {c["name"] for c in mine.get("/api/v1/crons/mine").json()["checks"]} == {"jobs", "transfers"}
    assert mine.delete("/api/v1/crons/mine/files/checks/transfers.md").status_code == 204
    assert mine.delete("/api/v1/crons/mine/files/CRON.md").status_code == 422
    assert mine.patch("/api/v1/crons/mine", json={"visibility": "user"}).json()["visibility"] == "user"
    assert all(c["id"] != "user_o@cern:mine" for c in colleague.get("/api/v1/crons").json()["crons"])
    assert mine.post("/api/v1/crons/mine/state", json={"enabled": False}).json()["enabled"] is False
    assert as_(SHIFTER).post("/api/v1/crons/system:shift/state", json={"enabled": False}).status_code == 403
    assert as_(ADMIN).post("/api/v1/crons/system:shift/state", json={"enabled": False}).json()["enabled"] is False
    assert mine.delete("/api/v1/crons/mine").status_code == 204
    assert mine.get("/api/v1/crons/mine").status_code == 404


def test_a_run_survives_a_restart(tmp_path):
    from agenticwf.agent.runtime.cron import store

    system = make_system_dir(tmp_path)
    cron_dir = system / "crons" / "shift"
    (cron_dir / "checks").mkdir(parents=True)
    (cron_dir / "CRON.md").write_text(CRON_MD.replace("template: reports/period.md", ""))
    (cron_dir / "checks" / "jobs.md").write_text(JOBS)

    def app():
        return create_app(Settings(context_secret=SECRET),
                          runtime_config=RuntimeConfig(system_dir=system, cron=CronSettings(tick=3600)),
                          engine=EngineOptions(llm=Model()))

    first = app()   # never started: the run is recorded, then the process "stops"
    crons = first.state.agent.manager.crons
    run_id = crons.start_run(crons.catalog.find("system:shift"), ["jobs"])
    assert store.run(run_id)["status"] == "running"
    with TestClient(app()) as client:   # a new process resumes the run's intent
        shifter = Client(client, SHIFTER)
        run = poll(shifter, f"/api/v1/crons/system:shift/runs/{run_id}", lambda r: r["status"] == "finished")
    assert run["checks"]["jobs"]["status"] == "ok"
