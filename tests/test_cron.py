"""Crons (src/agenticwf/agent/runtime/cron, docs/cron.md): schedule, runs, the issue tracker, reports, people."""

import datetime as dt
import re

import pytest

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager
from agenticwf.agent.runtime.cron import store, tracker
from agenticwf.agent.runtime.cron.definition import CronError, parse_cron
from agenticwf.agent.runtime.cron.schedule import UTC, PeriodClock, Schedule, ScheduleError, zone
from agenticwf.agent.runtime.cron.service import CronSettings
from agenticwf.agent.runtime.llm.base import LLM, LLMResponse, ToolCallRequest
from agenticwf.agent.runtime.records import IntentStatus
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.types import RuntimeState

from tests.test_runtime import make_system_dir

ZURICH = zone("Europe/Zurich")
SHIFTER = AuthContext(subject="user_s", user_id="user_s", organization_id="cern",
                      scopes=frozenset({"agent:use", "adcos:shift", "code:write"}))   # approved actions run as them
READER = AuthContext(subject="user_r", user_id="user_r", organization_id="cern",
                     scopes=frozenset({"agent:use", "atlas:member"}))
STRANGER = AuthContext(subject="user_x", user_id="user_x", organization_id="other", scopes=frozenset({"agent:use"}))
ROBOT = AuthContext(subject="app", client_id="app", organization_id="cern", scopes=frozenset({"agent:use", "adcos:shift"}))

CRON_MD = """---
name: shift
description: Watch the services.
timezone: Europe/Zurich
schedule: {every: 30m}
periods:
  - {name: shift, at: ["07:00", "15:00", "23:00"], template: reports/period.md}
rollups:
  - {name: weekly, from: shift, every: 1w, at: "monday 08:00"}
group: {by: site, default: central}
mode: default
resolve_after: 2
scopes: [code:read, code:write]
access: {organizations: [cern], scopes: [adcos:shift]}
readers: {scopes: [atlas:member]}
---
You are on shift.
"""
TRANSFERS = """---
description: Failing transfers per site.
tools: [read_thing, delete_thing]
issue:
  key: [site, error_class]
  severity: [critical, high, low]
metrics:
  - {name: failure_rate, unit: ratio, labels: [site]}
---
Look at the transfers.
"""
JOBS = """---
description: Failing jobs.
every: 1h
---
Look at the jobs.
"""


def call(name, **arguments):
    call.n = getattr(call, "n", 0) + 1
    return ToolCallRequest(f"c{call.n}", name, arguments)


def issue_call(site, error="checksum", severity="high", title=None, **extra):
    extra.setdefault("links", [f"https://mon.example/{site}"])
    return call("report_issue", key={"site": site, "error_class": error}, title=title or f"{error} at {site}",
                severity=severity, **{k: v for k, v in extra.items() if v is not None})


class Model(LLM):
    """Answers each check task from `plan[check]`: a list of tool-call lists, one per run."""

    def __init__(self):
        self.plan: dict[str, list[list[ToolCallRequest]]] = {}
        self.requests = []

    async def complete(self, request):
        self.requests.append(request)
        if "operations report" in request.system:
            return LLMResponse("SUMMARY: transfers failed at SITE_A.")
        text = "\n".join(m.content for m in request.messages if m.role == "user")
        check = re.search(r"Run the check `([^`]+)`", text).group(1)
        if any(m.role == "tool" for m in request.messages):
            return LLMResponse(f"{check} looked")
        planned = self.plan.get(check) or []
        return LLMResponse("", planned.pop(0) if planned else [call("report_check", status="ok", summary="fine")])


@pytest.fixture
def world(tmp_path):
    system = make_system_dir(tmp_path)
    cron_dir = system / "crons" / "shift"
    (cron_dir / "checks").mkdir(parents=True)
    (cron_dir / "reports").mkdir()
    (cron_dir / "CRON.md").write_text(CRON_MD)
    (cron_dir / "checks" / "transfers.md").write_text(TRANSFERS)
    (cron_dir / "checks" / "jobs.md").write_text(JOBS)
    (cron_dir / "reports" / "period.md").write_text("Lead with the worst site.")
    deleted = []
    model = Model()
    manager = build_thread_manager(
        RuntimeConfig(system_dir=system, cron=CronSettings(deny_tools=("admin_*",))),
        EngineOptions(llm=model, crons=True, handlers={
            "test.read_thing": lambda ctx, args: ToolResult(content="thing"),
            "test.delete_thing": lambda ctx, args: deleted.append(args) or ToolResult(content="deleted")}))
    service = manager.crons
    clock = {"now": dt.datetime(2026, 10, 12, 8, 0, tzinfo=UTC)}   # 10:00 in Zurich, in the 07:00 shift
    service.clock = lambda: clock["now"]
    return service, model, clock, deleted


async def drain(service):
    """Run every queued job, one at a time, as the workers would (deterministic)."""
    from types import SimpleNamespace

    from agenticwf.agent.runtime.harness.worker import Workers

    harness = service.harness
    harness.store._listeners.clear()   # no worker pool starting on its own
    for _ in range(500):
        job = harness.store.claim("test", 60)
        if job is None:
            return
        again = await Workers._execute(SimpleNamespace(harness=harness), job)
        harness.store.complete(job, "test")
        if again:
            harness.store.enqueue(type(job)(job.kind, job.ref))
    raise AssertionError("jobs never settled")


def run(service, clock, minutes=0):
    import asyncio

    clock["now"] += dt.timedelta(minutes=minutes)
    started = service.tick(clock["now"])
    asyncio.run(drain(service))
    return started


CRON_ID = "system:shift"


# --- schedule and definitions ---------------------------------------------------------------------

def test_schedules_and_periods():
    t = dt.datetime(2026, 10, 12, 13, 7, tzinfo=UTC)
    assert Schedule.parse({"every": "30m"}).next_after(t, ZURICH) == dt.datetime(2026, 10, 12, 13, 30, tzinfo=UTC)
    assert Schedule.parse({"cron": "*/15 9-17 * * 1-5"}).next_after(t, ZURICH).minute == 15
    assert Schedule.parse({"at": ["06:30"]}).next_after(t, ZURICH) == dt.datetime(2026, 10, 13, 4, 30, tzinfo=UTC)
    with pytest.raises(ScheduleError):
        Schedule.parse({"every": "30s"})
    shift = PeriodClock.parse({"at": ["07:00", "15:00", "23:00"]})
    night = dt.datetime(2026, 10, 25, 3, 0, tzinfo=UTC)                    # the night summer time ends
    assert shift.next(night, ZURICH) - shift.start(night, ZURICH) == dt.timedelta(hours=9)
    elapsed = PeriodClock.parse({"clock": "elapsed", "every": "8h", "anchor": "2026-01-01T06:00:00Z"})
    assert elapsed.next(night, ZURICH) - elapsed.start(night, ZURICH) == dt.timedelta(hours=8)
    week = PeriodClock.parse({"at": "monday 08:00"})
    assert week.previous(t, ZURICH) == (dt.datetime(2026, 10, 5, 6, tzinfo=UTC), dt.datetime(2026, 10, 12, 6, tzinfo=UTC))
    assert PeriodClock.parse({"at": "31 08:00"}).start(dt.datetime(2026, 3, 1, tzinfo=UTC), ZURICH).day == 28


def test_definitions_are_checked():
    files = {"CRON.md": CRON_MD, "checks/transfers.md": TRANSFERS, "checks/jobs.md": JOBS, "reports/period.md": "x"}
    cron = parse_cron("shift", files, "system")
    transfers = cron.check("transfers")
    assert transfers.key == ("site", "error_class") and transfers.severities == ("critical", "high", "low")
    assert cron.check("jobs").key == ("site", "title")                      # the group field is always a key
    assert cron.readers is not None and cron.principal == "cron:shift"
    user = parse_cron("shift", files, "database", owner=("user_s", "cern"))
    assert user.readers is None and user.scopes == frozenset() and user.access.organizations is None
    for broken, message in (({**files, "CRON.md": CRON_MD.replace("every: 30m", "every: soon")}, "duration"),
                            ({k: v for k, v in files.items() if not k.startswith("checks/")}, "no checks"),
                            ({**files, "CRON.md": CRON_MD.replace("from: shift", "from: nope")}, "not a period")):
        with pytest.raises(CronError, match=message):
            parse_cron("shift", broken, "system")


# --- runs and the issue tracker ---------------------------------------------------------------------

def test_a_run_files_issues_grouped_by_site_with_metrics(world):
    service, model, clock, _ = world
    assert run(service, clock) == []                                         # first seen: from the next slot
    model.plan["transfers"] = [[issue_call("SITE_A"), issue_call("SITE_B", "timeout", "low"),
                                call("report_check", status="issues", summary="two sites",
                                     metrics=[{"name": "failure_rate", "value": 0.4, "labels": {"site": "SITE_A"},
                                               "link": "https://mon.example/a"}])]]
    model.plan["jobs"] = [[call("report_issue", key={"site": " SITE_A "}, title="jobs fail", severity="medium"),
                           call("report_check", status="issues")]]
    started = run(service, clock, 61)                                        # 09:01: both checks are due
    assert len(started) == 1
    finished = store.run(started[0])
    assert finished["status"] == "finished" and set(finished["checks"]) == {"transfers", "jobs"}
    assert finished["checks"]["transfers"]["status"] == "issues" and finished["checks"]["transfers"]["new"] == 2

    # what the check task saw: the cron's and the check's instructions, its tools only
    request = next(r for r in model.requests if "Run the check `transfers`" in r.messages[0].content)
    assert "# Cron: shift\n\nYou are on shift." in request.system and "# Check: transfers" in request.system
    assert {t["name"] for t in request.tools} >= {"read_thing", "delete_thing", "report_issue", "report_check"}
    assert "write_thing" not in {t["name"] for t in request.tools}
    intent = service.harness.store.load_intent(finished["intent_id"])
    assert intent.source == "cron" and intent.status is IntentStatus.ANSWERED
    assert intent.auth.subject == "cron:shift" and intent.auth.auth_method == "cron"

    sites = tracker.by_site(tracker.issues([CRON_ID]))
    assert [s["site"] for s in sites] == ["SITE_A", "SITE_B"]                # worst first
    site_a = sites[0]
    assert {i["check_name"] for i in site_a["issues"]} == {"transfers", "jobs"}   # one site, two checks
    series = store.metric_series(CRON_ID, "failure_rate", clock["now"] - dt.timedelta(hours=1),
                                 clock["now"] + dt.timedelta(hours=1))
    assert series[0]["mean"] == 0.4 and series[0]["links"] == ["https://mon.example/a"] and series[0]["group"] == "SITE_A"


def test_issues_are_followed_across_runs(world):
    service, model, clock, _ = world
    run(service, clock)
    model.plan["transfers"] = [[issue_call("SITE_A", severity="high")]]
    run(service, clock, 31)
    [issue] = tracker.issues([CRON_ID], check="transfers")
    model.plan["transfers"] = [[issue_call("SITE_A", severity="low")]]                # same shift: severity stays
    run(service, clock, 30)
    same = tracker.issue(issue["id"])
    assert same["occurrences"] == 2 and same["severity"] == "high"
    clock["now"] = dt.datetime(2026, 10, 12, 13, 5, tzinfo=UTC)                       # the 15:00 shift
    model.plan["transfers"] = [[issue_call("SITE_A", severity="low")]]
    run(service, clock)
    assert tracker.issue(issue["id"])["severity"] == "low"                            # starts again in a new shift

    model.plan["transfers"] = [[call("report_check", status="unable", summary="FTS down")]]
    run(service, clock, 30)
    assert tracker.issue(issue["id"])["missed"] == 0                                 # did not look: nothing concluded
    run(service, clock, 30)                                                          # ok, not seen: missed 1
    run(service, clock, 30)                                                          # missed 2: solved by itself
    solved = tracker.issue(issue["id"])
    assert solved["status"] == "solved" and solved["solved_by"] == "itself"
    model.plan["transfers"] = [[issue_call("SITE_A")]]
    run(service, clock, 30)
    back = tracker.issue(issue["id"])
    assert back["status"] == "open" and back["reopened"] == 1
    assert [e["action"] for e in back["events"]] == ["opened", "severity", "solved", "severity", "reopened"]


def test_people_tick_boxes_and_suppress(world):
    service, model, clock, _ = world
    run(service, clock)
    model.plan["transfers"] = [[issue_call("SITE_A"), issue_call("SITE_B")]]
    run(service, clock, 31)
    a, b = sorted(tracker.issues([CRON_ID], check="transfers"), key=lambda i: i["grp"])
    with pytest.raises(PermissionError):
        service.act(READER, [a["id"]], "acknowledge")                              # readers see, no boxes
    with pytest.raises(PermissionError):
        service.act(ROBOT, [a["id"]], "acknowledge")                               # machines never tick
    with pytest.raises(PermissionError):
        service.act(STRANGER, [a["id"]], "acknowledge")
    [acked] = service.act(SHIFTER, [a["id"]], "acknowledge", comment="looking")
    assert acked["status"] == "acknowledged" and acked["owner"] == "user_s"
    service.act(SHIFTER, [a["id"]], "ticket", ticket="GGUS-123")
    service.act(SHIFTER, [b["id"]], "ignore", comment="known", until=clock["now"] + dt.timedelta(days=1))
    service.suppress(SHIFTER, CRON_ID, "transfers", {"site": "site_c"}, "decommissioned", None)

    model.plan["transfers"] = [[issue_call("SITE_A"), issue_call("SITE_B"), issue_call("SITE_C")]]
    run(service, clock, 30)
    assert {i["grp"] for i in tracker.issues([CRON_ID], check="transfers")} == {"SITE_A", "SITE_B"}   # C suppressed
    assert tracker.issue(b["id"])["status"] == "ignored" and tracker.issue(a["id"])["ticket"] == "GGUS-123"
    request = [r for r in model.requests if "Run the check `transfers`" in r.messages[0].content][-1].messages[0].content
    assert "looking" in request and "site=site_c: decommissioned" in request and "ignored" in request


def test_unattended_calls_become_proposals_people_approve(world):
    service, model, clock, deleted = world
    run(service, clock)
    model.plan["transfers"] = [[issue_call("SITE_A"), call("delete_thing", value="stuck rule")]]
    run(service, clock, 31)
    [proposal] = store.proposals(CRON_ID)
    assert proposal["tool"] == "delete_thing" and proposal["status"] == "pending" and not deleted
    issue = tracker.issues([CRON_ID], check="transfers")[0]
    assert proposal["issue_id"] == issue["id"]
    import asyncio

    with pytest.raises(PermissionError):
        asyncio.run(service.decide(READER, proposal["id"], True))
    decided = asyncio.run(service.decide(SHIFTER, proposal["id"], True))
    assert decided["status"] == "approved" and decided["decided_by"] == "user_s" and deleted == [{"value": "stuck rule"}]
    assert decided["result"]["content"] == "deleted"
    with pytest.raises(ValueError):
        asyncio.run(service.decide(SHIFTER, proposal["id"], False))           # once
    assert tracker.issue(issue["id"])["events"][-1]["action"] == "approved"


def test_deny_tools_apply_to_cron_tasks_only(world):
    service, model, clock, _ = world
    policy = service.harness.policy
    rule = policy.config.rules[0]
    assert rule.sources == ("cron",) and rule.tool == "admin_*"
    task = type("T", (), {"source": "web"})()
    assert not rule.applies_to(task) and rule.applies_to(type("T", (), {"source": "cron"})())


# --- reports --------------------------------------------------------------------------------------

def test_shift_reports_are_written_at_the_end_of_each_shift(world):
    service, model, clock, _ = world
    run(service, clock)                                                      # 10:00 local, the 07:00 shift
    model.plan["transfers"] = [[issue_call("SITE_A"),
                                call("report_check", status="issues",
                                     metrics=[{"name": "failure_rate", "value": 0.5, "labels": {"site": "SITE_A"},
                                               "link": "https://mon.example/a"}])]]
    run(service, clock, 31)
    clock["now"] = dt.datetime(2026, 10, 12, 13, 1, tzinfo=UTC)              # 15:01: the shift ended
    run(service, clock)
    reports, total = store.reports([CRON_ID], year=2026, month=10, day=12)
    assert total == 1 and reports[0]["kind"] == "shift" and reports[0]["status"] == "ready"
    assert reports[0]["sites"] == ["SITE_A"]
    report = store.report(reports[0]["id"])
    assert report["start_at"] == dt.datetime(2026, 10, 12, 5, tzinfo=UTC) and report["end_at"] == dt.datetime(2026, 10, 12, 13, tzinfo=UTC)
    text = report["markdown"]
    assert "## Summary\n\nSUMMARY: transfers failed at SITE_A." in text
    assert "## Sites" in text and "### SITE_A (high)" in text and "- [ ] **high** · checksum at SITE_A" in text
    assert "https://mon.example/SITE_A" in text and "failure_rate" in text and "https://mon.example/a" in text
    assert "## Coverage" in text and "transfers: 1 of 1 ran" in text
    summary_request = next(r for r in model.requests if "operations report" in r.system)
    assert "Lead with the worst site." in summary_request.system              # the period's template
    assert store.reports([CRON_ID], site="site_a")[1] == 1 and store.reports([CRON_ID], site="SITE_Z")[1] == 0
    assert store.reports([CRON_ID], year=2026, month=10, day=11)[1] == 0

    report_id = service.report_on_request(SHIFTER, CRON_ID, report["start_at"], report["end_at"])
    import asyncio
    asyncio.run(drain(service))
    custom = store.report(report_id)
    assert custom["status"] == "ready" and custom["kind"] == "custom" and "## Per day" in custom["markdown"]


def test_run_now_and_switching(world):
    service, model, clock, _ = world
    with pytest.raises(PermissionError):
        service.run_now(READER, CRON_ID)
    run_id = service.run_now(SHIFTER, CRON_ID, ["jobs"])
    assert store.run(run_id)["due"] == ["jobs"]
    again = service.run_now(SHIFTER, CRON_ID, ["jobs"])                       # still running: skipped, never twice
    assert store.run(again)["skipped"] == ["jobs"] and store.run(again)["due"] == []
    with pytest.raises(PermissionError):
        service.set_enabled(SHIFTER, CRON_ID, False)                           # system crons: managers only
    admin = AuthContext(subject="boss", user_id="boss", organization_id="cern",
                        scopes=frozenset({"agent:use", "adcos:shift", "admin"}))
    assert service.set_enabled(admin, CRON_ID, False)["enabled"] is False
    assert service.tick(clock["now"] + dt.timedelta(hours=2)) == []


def test_a_users_database_cron_runs_as_its_owner(world):
    service, model, clock, _ = world
    owner = AuthContext(subject="user_o", user_id="user_o", organization_id="cern",
                        scopes=frozenset({"agent:use", "code:read"}))
    files = {"CRON.md": CRON_MD.replace("name: shift", "name: mine", 1), "checks/jobs.md": JOBS,
             "reports/period.md": "x"}
    cron = service.put_database_cron(owner, "mine", files, "user", True)
    assert cron.id == "user_o@cern:mine" and cron.scopes == frozenset()
    with pytest.raises(CronError):
        service.put_database_cron(owner, "broken", {"CRON.md": "no front matter"}, "user", True)
    assert [c.id for c, _ in service.catalog.visible(owner) if c.source == "database"] == ["user_o@cern:mine"]
    assert all(c.id != "user_o@cern:mine" for c, _ in service.catalog.visible(SHIFTER))
    run_id = service.run_now(owner, "user_o@cern:mine")
    intent = service.harness.store.load_intent(store.run(run_id)["intent_id"])
    assert intent.auth.subject == "user_o" and intent.auth.scopes == frozenset({"agent:use", "code:read"})
    assert service.delete_database_cron(owner, "user_o@cern:mine")


def test_check_tasks_do_not_consolidate_memory(world):
    from agenticwf.agent.runtime.context.db_memory import DatabaseMemoryStore

    service, model, clock, _ = world
    service.harness.services.context_manager.memory = DatabaseMemoryStore()   # memory is on
    assert service.harness.services.context_manager.should_consolidate(RuntimeState.COMPLETED)
    run(service, clock)
    run(service, clock, 31)
    tasks = [t for r in store.runs(CRON_ID) if r["intent_id"]
             for t in service.harness.store.tasks_for_intent(r["intent_id"])]
    assert tasks and all(getattr(t, "consolidation", None) is None for t in tasks)
