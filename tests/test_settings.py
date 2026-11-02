"""Settings changed on the admin page over agent.cfg and task.yaml (agenticwf.agent.runtime.overrides):
everyone's (admins) and a person's own, in the harness, the task types and the model router."""

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.db_overrides import DatabaseSettingOverrides
from agenticwf.agent.runtime.harness.planning import IntakeSettings
from agenticwf.agent.runtime.llm.routing import RoutingSettings
from agenticwf.agent.runtime.overrides import BUDGETS, HARNESS, PURPOSE, ROUTING, InMemorySettingOverrides, check, effective
from agenticwf.agent.runtime.mcp.manager import McpSettings
from agenticwf.agent.runtime.secrets import SecretBox

from tests.conftest import SECRET, Client, user
from tests.test_plugins import make_plugins
from tests.test_runtime import build, make_system_dir, user as person

ALICE, BOB = ("user_a", None), ("user_b", None)


def test_a_persons_values_count_only_where_they_may_set_them():
    store = InMemorySettingOverrides()
    base = {"split": False, "plan": "never", "max_steps": 50, "max_tool_calls": 200}
    store.set(HARNESS, "plan", "always")                              # everyone's
    store.set(HARNESS, "max_tool_calls", 20)
    store.set(HARNESS, "split", True, ALICE)                          # Alice's own
    store.set(HARNESS, "max_tool_calls", 999, ALICE)                  # not hers to set: ignored
    store.set(HARNESS, "max_steps", 80, ALICE)                        # only lower counts
    alice = effective(store, HARNESS, base, ALICE)
    assert alice == {"split": True, "plan": "always", "max_steps": 50, "max_tool_calls": 20}
    store.set(HARNESS, "max_steps", 10, ALICE)
    assert effective(store, HARNESS, base, ALICE)["max_steps"] == 10
    assert effective(store, HARNESS, base, BOB) == {**base, "plan": "always", "max_tool_calls": 20}
    with pytest.raises(KeyError):
        check(HARNESS, "max_tool_calls", 5, personal=True)
    with pytest.raises(ValueError):
        check(HARNESS, "plan", "sometimes", personal=True)
    with pytest.raises(ValueError):
        check(ROUTING, "step", "huge", personal=False)


def test_task_type_settings_change_without_task_yaml(tmp_path):
    system = make_system_dir(tmp_path)                                 # review: task.yaml with mode plan, 7 steps
    (system / "task-types" / "notes").mkdir()                          # no task.yaml: inherits default's
    (system / "task-types" / "notes" / "AGENTS.md").write_text("# notes\n")
    store = InMemorySettingOverrides()
    resolver = TaskTypeResolver(system, overrides=store)
    store.set("task_type:system:review", "max_steps", 12)
    store.set("task_type:system:review", "plan", "always")
    store.set("task_type:system:review", "plan", "never", ALICE)       # her own tasks
    store.set("task_type:system:default", "pin_tools", ["read_thing", "web_search"])
    assert resolver.resolve("review", BOB).settings.max_steps == 12
    assert resolver.resolve("review", BOB).settings.plan == "always"
    assert resolver.resolve("review", ALICE).settings.plan == "never"
    assert resolver.resolve("review", ALICE).settings.pin_tools == ()          # its own file: not default's changes
    assert resolver.resolve("notes", ALICE).settings.pin_tools == ("read_thing", "web_search")   # inherited
    assert resolver.resolve("nope", ALICE).settings.pin_tools == ("read_thing", "web_search")    # fell back
    assert (system / "task-types" / "review" / "task.yaml").read_text() == "mode: plan\nmax_steps: 7\n"   # untouched
    layers = resolver.settings_layers("review", ALICE)
    assert layers["file"]["max_steps"] == 7 and layers["all"]["max_steps"] == 12 and layers["mine"] == {"plan": "never"}
    assert "max_steps: 7" in layers["task_yaml"]["content"] and not layers["task_yaml"]["inherited"]
    assert resolver.settings_layers("notes", ALICE)["inherits"]


def test_the_harness_splits_and_plans_as_changed(tmp_path):
    system = make_system_dir(tmp_path)
    h, _, _ = build(system, [])
    store = h.overrides = InMemorySettingOverrides()
    h.intake_settings = IntakeSettings(split=False, plan="never")
    assert not h.settings_for(ALICE)[0].split
    store.set(HARNESS, "split", True, ALICE)
    store.set(HARNESS, "plan", "always")
    store.set(HARNESS, "max_tool_calls", 3)
    intake, limits = h.settings_for(ALICE)
    assert intake.split and intake.plan == "always" and limits.max_tool_calls == 3
    assert not h.settings_for(BOB)[0].split
    task = h.create_task(person("user_b"), "do it")
    assert h._should_plan(task)                                          # everyone's plan = always


def test_the_router_takes_admins_routing():
    from agenticwf.agent.runtime.llm.catalog import ModelCatalog
    from agenticwf.agent.runtime.llm.router import ModelRouter

    store = InMemorySettingOverrides()
    router = ModelRouter(ModelCatalog([], {}), settings=RoutingSettings(), overrides=store)
    assert router.settings.tier_for("split") == "small"
    store.set(ROUTING, "split", "medium")
    store.set(ROUTING, "max_escalations", 2)
    store.set(ROUTING, "step", "enormous")                                # no longer fits: skipped
    assert router.settings.tier_for("split") == "medium" and router.settings.max_escalations == 2
    assert router.settings.tier_for("step") == "medium" and router.configured.tier_for("split") == "small"


def test_the_router_takes_admins_purpose_models_and_budgets():
    from agenticwf.agent.runtime.llm.catalog import ModelCatalog
    from agenticwf.agent.runtime.llm.router import ModelRouter

    store = InMemorySettingOverrides()
    configured = RoutingSettings.parse({"llm.budgets": {"default": "20 per day", "max_tier.user_b": "medium"}})
    router = ModelRouter(ModelCatalog([], {}), settings=configured, overrides=store)
    assert router.configured.budgets_view() == {"default": "20 per day", "max_tier.user_b": "medium"}
    store.set(PURPOSE, "split", "gemma")
    store.set(BUDGETS, "default", check(BUDGETS, "default", "5 per month", personal=False))
    store.set(BUDGETS, "user.user_a", "1 per day")
    store.set(BUDGETS, "max_tier.organization.atlas", "small")
    store.set(BUDGETS, "organization.cms", "lots")                         # no longer fits: skipped
    settings = router.settings
    assert settings.purpose_models == {"split": "gemma"} and settings.budget_for(("user_x", None)).text() == "5 per month"
    assert settings.budget_for(ALICE).amount == 1 and settings.budget_for(("user_x", "cms")).period == "month"
    assert settings.max_tier_for(("user_x", "atlas")) == "small" and settings.max_tier_for(BOB) == "medium"
    assert router.configured.budget_for(ALICE).amount == 20                 # agent.cfg's, unchanged
    with pytest.raises(ValueError):
        check(BUDGETS, "user.user_a", "lots", personal=False)
    with pytest.raises(KeyError):
        check(BUDGETS, "someone", "1 per day", personal=False)
    with pytest.raises(KeyError):
        check(BUDGETS, "default", "1 per day", personal=True)               # admins only


def test_kb_http_calls_end_before_the_tools_timeout():
    import time

    from agenticwf.agent.runtime.knowledge import _within
    from agenticwf.agent.runtime.tools.spec import CallContext

    call = lambda seconds: CallContext(task_id="t", call_id="c", auth=None, deadline=time.time() + seconds)  # noqa: E731
    assert 169 < _within(call(180), 200) <= 170                             # kb_research set to 180 s on the Tools tab
    assert 589 < _within(call(600), 200) <= 590                             # longer than research_timeout too
    assert 8.9 < _within(call(10), 20) <= 9 and _within(call(0), 20) == 1.0
    assert _within(CallContext(task_id="t", call_id="c", auth=None), 20) == 20


def test_database_overrides_keep_both_layers():
    store = DatabaseSettingOverrides(ttl=60)
    store.set(HARNESS, "plan", "always", by="admin_a")
    store.set(HARNESS, "split", True, ALICE)
    assert store.values(HARNESS) == {"plan": "always"} and store.values(HARNESS, ALICE) == {"split": True}
    store.set(HARNESS, "plan", None)
    store.set(HARNESS, "split", None, ALICE)
    assert store.values(HARNESS) == {} and store.values(HARNESS, ALICE) == {}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), plugin_dirs=[make_plugins(tmp_path)], provider="echo",
                           mcp=McpSettings(connect_timeout=10))
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config,
                               engine=EngineOptions(setting_overrides=InMemorySettingOverrides()))) as tc:
        yield tc


def test_settings_api(app):
    admin = Client(app, user(subject="user_admin", scopes=("app:access", "agent:use", "admin")))
    one, two = Client(app, user(subject="user_one")), Client(app, user(subject="user_two"))
    put = lambda c, **body: c.put("/api/v1/settings", json=body)  # noqa: E731
    listed = one.get("/api/v1/settings").json()
    assert not listed["can_set_all"]
    assert [s["section"] for s in listed["sections"]] == ["harness", "llm.routing", "llm.purpose", "llm.budgets"]
    rows = {r["key"]: r for r in listed["sections"][0]["rows"]}
    assert rows["split"]["file"] is False and rows["split"]["personal"] and not rows["max_tool_calls"]["personal"]

    assert put(one, section="harness", key="plan", value="always", scope="all").status_code == 403
    assert put(one, section="harness", key="max_tool_calls", value=5).status_code == 403   # everyone's only
    assert put(one, section="llm.routing", key="step", value="large").status_code == 403
    assert put(one, section="harness", key="plan", value="sometimes").status_code == 422
    assert put(one, section="harness", key="nope", value=1).status_code == 404
    assert put(one, section="nowhere", key="plan", value="always").status_code == 404
    assert put(one, section="harness", key="split", value=True).json()["scope"] == "mine"
    assert put(admin, section="llm.routing", key="split", value="medium", scope="all").status_code == 200
    assert put(admin, section="harness", key="max_tool_calls", value=5, scope="all").status_code == 200

    mine = {r["key"]: r for r in one.get("/api/v1/settings").json()["sections"][0]["rows"]}
    theirs = {r["key"]: r for r in two.get("/api/v1/settings").json()["sections"][0]["rows"]}
    assert mine["split"]["mine"] is True and mine["split"]["effective"] is True and theirs["split"]["effective"] is False
    assert mine["max_tool_calls"]["all"] == 5 and theirs["max_tool_calls"]["effective"] == 5
    routing = {r["key"]: r for r in two.get("/api/v1/settings").json()["sections"][1]["rows"]}
    assert routing["split"]["file"] == "small" and routing["split"]["effective"] == "medium"
    assert put(one, section="harness", key="split", value=None).status_code == 200            # back to everyone's

    # purpose models and budgets: admins only; a person sees the default's and their own budget
    assert put(one, section="llm.budgets", key="user.user_one", value="99 per day").status_code == 403
    assert put(one, section="llm.budgets", key="user.user_one", value="99 per day", scope="all").status_code == 403
    assert put(admin, section="llm.budgets", key="user.user_two", value="5 per day", scope="all").status_code == 200
    assert put(admin, section="llm.budgets", key="user.user_two", value="lots", scope="all").status_code == 422
    assert put(admin, section="llm.budgets", key="anyone", value="5 per day", scope="all").status_code == 404
    assert put(admin, section="llm.purpose", key="split", value="no-such-model", scope="all").status_code == 422
    sections = {s["section"]: s for s in admin.get("/api/v1/settings").json()["sections"]}
    models = sections["llm.purpose"]["rows"][0]["choices"]
    assert sections["llm.budgets"]["add_keys"] and models
    assert put(admin, section="llm.purpose", key="split", value=models[0], scope="all").status_code == 200
    budgets = lambda c: {r["key"]: r for r in [s for s in c.get("/api/v1/settings").json()["sections"]  # noqa: E731
                                                 if s["section"] == "llm.budgets"][0]["rows"]}
    assert "user.user_two" not in budgets(one) and not budgets(one)["default"]["personal"]
    assert budgets(two)["user.user_two"]["effective"] == "5 per day" and "user.user_two" in budgets(admin)

    # a task type's settings: by its bare name; task.yaml is shown, never written
    assert put(admin, section="task_type:review", key="max_steps", value=9, scope="all").json()["section"] == "task_type:system:review"
    assert put(one, section="task_type:review", key="plan", value="always").status_code == 200
    assert put(one, section="task_type:review", key="tools_deny", value=["x"]).status_code == 403
    assert put(one, section="task_type:user_x:secret", key="plan", value="always").status_code == 404
    review = one.get("/api/v1/task-types/review").json()
    assert "max_steps: 7" in review["content"]["task.yaml"]["content"]
    rows = {r["key"]: r for r in review["settings_rows"]}
    assert rows["max_steps"]["file"] == 7 and rows["max_steps"]["effective"] == 9 and rows["plan"]["effective"] == "always"
    assert {t["name"]: t for t in two.get("/api/v1/task-types").json()["task_types"]}["review"]["settings"]["plan"] is None
    machine = Client(app, user(subject="app_m2m", client_id="app_m2m"))
    assert put(machine, section="harness", key="split", value=True).status_code == 403
