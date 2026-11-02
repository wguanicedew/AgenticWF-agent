"""Tools people switch off for their own tasks: one tool, a plugin's, an MCP endpoint's
(agenticwf.agent.runtime.tools.prefs), in the dispatcher and on the admin page's Tools tab."""

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.mcp.manager import McpManager, McpSettings
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.policy.engine import PolicyEngine
from agenticwf.agent.runtime.secrets import SecretBox
from agenticwf.agent.runtime.tools.db_prefs import DatabaseToolPrefs
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
from agenticwf.agent.runtime.task import ToolCall
from agenticwf.agent.runtime.tools.dispatcher import Outcome
from agenticwf.agent.runtime.tools.prefs import InMemoryToolPrefs, mode_of, origin
from agenticwf.agent.runtime.tools.registry import ToolRegistry

from tests.conftest import SECRET, Client, user
from tests.test_plugins import make_plugins
from tests.test_runtime import make_system_dir, user as person

ALICE, BOB = ("user_a", None), ("user_b", None)


@pytest.fixture
def harness(tmp_path):
    system = make_system_dir(tmp_path)
    plugins = PluginManager([make_plugins(tmp_path)], mcp=McpManager(McpSettings(connect_timeout=10)))
    registry = ToolRegistry(system_dir=system, plugins=plugins)
    registry.load()
    prefs = InMemoryToolPrefs()
    dispatcher = ToolDispatcher(registry, PolicyEngine(), prefs=prefs)
    yield AgentHarness(HarnessServices(llm=ScriptedLLM([]), task_types=TaskTypeResolver(system),
                                       tool_dispatcher=dispatcher)), prefs
    plugins.mcp.close()


def test_origin_names_the_plugin_and_the_endpoint(harness):
    h, _ = harness
    specs = {s.name: s for s in h.services.tool_dispatcher.registry.catalogue(ALICE)}
    assert origin(specs["read_thing"]).kind == "system"
    assert (origin(specs["lint__shout"]).kind, origin(specs["lint__shout"]).plugin) == ("plugin", "system:lint")
    mcp = origin(specs["fake__echo"])
    assert (mcp.kind, mcp.plugin, mcp.endpoint) == ("mcp", "system:fake", "system:fake/fake")
    assert origin(specs["update_plan"]).kind == "control"


def test_switched_off_tools_are_neither_offered_nor_run(harness):
    h, prefs = harness
    dispatcher = h.services.tool_dispatcher
    task = h.create_task(person("user_a"), "read the thing")
    assert {"read_thing", "lint__shout", "fake__echo"} <= set(dispatcher.eligible(task))

    prefs.set_enabled(ALICE, "tool:read_thing", False)
    prefs.set_enabled(ALICE, "plugin:system:lint", False)
    prefs.set_enabled(ALICE, "mcp:system:fake/fake", False)
    prefs.set_enabled(ALICE, "tool:update_plan", False)                 # control tools stay
    eligible = set(dispatcher.eligible(task))
    assert not eligible & {"read_thing", "lint__shout", "fake__echo"} and "update_plan" in eligible
    assert "read_thing" not in {s.name for s in dispatcher.active(task)}   # pinned by the task type, still off
    other = h.create_task(person("user_b"), "read the thing")
    assert {"read_thing", "lint__shout", "fake__echo"} <= set(dispatcher.eligible(other))   # Alice's only

    prefs.set_enabled(ALICE, "tool:read_thing", True)
    assert "read_thing" in dispatcher.eligible(task)


def test_database_prefs_are_per_owner():
    prefs = DatabaseToolPrefs(ttl=60)
    prefs.set_enabled(ALICE, "plugin:system:lint", False)
    prefs.set_enabled(ALICE, "tool:x", False)
    assert prefs.disabled_for(ALICE) == {"plugin:system:lint", "tool:x"} and prefs.disabled_for(BOB) == frozenset()
    prefs.set_enabled(ALICE, "tool:x", True)                             # this process: at once, despite the ttl
    assert prefs.disabled_for(ALICE) == {"plugin:system:lint"}
    prefs.set_enabled(ALICE, "plugin:system:lint", True)


def test_tools_set_to_ask_wait_for_their_owners_approval(harness):
    h, prefs = harness
    dispatcher = h.services.tool_dispatcher
    task = h.create_task(person("user_a"), "read the thing")
    prepare = lambda name: dispatcher.prepare(task, ToolCall(id=name, name=name, arguments={}, step=1))  # noqa: E731
    assert prepare("lint__shout") is Outcome.ALLOWED

    prefs.set_mode(ALICE, "plugin:system:lint", "ask")
    assert "lint__shout" in dispatcher.eligible(task)                   # still offered
    call = ToolCall(id="c", name="lint__shout", arguments={}, step=1)
    assert dispatcher.prepare(task, call) is Outcome.NEEDS_APPROVAL and call.reason.startswith("user:ask")
    task.grants.add(f"lint__shout@{dispatcher.eligible(task)['lint__shout'].fingerprint}")
    assert prepare("lint__shout") is Outcome.ALLOWED                    # approved for the rest of the task
    other = h.create_task(person("user_b"), "read the thing")
    assert dispatcher.prepare(other, ToolCall(id="c", name="lint__shout", arguments={}, step=1)) is Outcome.ALLOWED

    specs = {s.name: s for s in dispatcher.registry.catalogue(ALICE)}
    prefs.set_mode(ALICE, "tool:lint__shout", "off")                   # the strictest switch wins
    assert mode_of(specs["lint__shout"], prefs.modes_for(ALICE)) == "off"
    assert "lint__shout" not in dispatcher.eligible(task)
    prefs.set_mode(ALICE, "tool:update_plan", "ask")                   # control tools stay on
    assert mode_of(specs["update_plan"], prefs.modes_for(ALICE)) == "on"
    with pytest.raises(ValueError):
        prefs.set_mode(ALICE, "tool:x", "maybe")


def test_database_prefs_keep_modes():
    prefs = DatabaseToolPrefs(ttl=60)
    prefs.set_mode(ALICE, "mcp:system:fake/fake", "ask")
    prefs.set_mode(ALICE, "tool:y", "off")
    assert prefs.modes_for(ALICE) == {"mcp:system:fake/fake": "ask", "tool:y": "off"}
    assert prefs.disabled_for(ALICE) == {"tool:y"}
    prefs.set_mode(ALICE, "tool:y", "ask")
    assert prefs.modes_for(ALICE)["tool:y"] == "ask"
    prefs.set_mode(ALICE, "tool:y", "on")
    prefs.set_mode(ALICE, "mcp:system:fake/fake", "on")
    assert prefs.modes_for(ALICE) == {}


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), plugin_dirs=[make_plugins(tmp_path)], provider="echo",
                           mcp=McpSettings(connect_timeout=10))
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc


def test_tools_api(app):
    one, two = Client(app, user(subject="user_one")), Client(app, user(subject="user_two"))
    listed = one.get("/api/v1/tools").json()
    tools = {t["name"]: t for t in listed["tools"]}
    assert tools["read_thing"]["kind"] == "system" and tools["lint__shout"]["plugin"] == "system:lint"
    assert tools["fake__echo"]["kind"] == "mcp" and tools["fake__echo"]["endpoint"] == "system:fake/fake"
    assert not tools["update_plan"]["switchable"] and all(t["active"] for t in tools.values())
    groups = {(g["target"], g["id"]): g for g in listed["groups"]}
    assert groups[("plugin", "system:lint")]["name"] == "lint" and groups[("mcp", "system:fake/fake")]["name"] == "fake/fake"
    assert groups[("plugin", "system:fake")]["tools"] == groups[("mcp", "system:fake/fake")]["tools"] >= 1

    assert one.put("/api/v1/tools/prefs", json={"target": "plugin", "id": "system:fake", "enabled": False}).status_code == 200
    assert one.put("/api/v1/tools/prefs", json={"target": "tool", "id": "read_thing", "enabled": False}).json()["enabled"] is False
    assert one.put("/api/v1/tools/prefs", json={"target": "tool", "id": "update_plan", "enabled": False}).status_code == 409
    assert one.put("/api/v1/tools/prefs", json={"target": "mcp", "id": "user_x:nope", "enabled": False}).status_code == 404
    assert one.put("/api/v1/tools/prefs", json={"target": "skill", "id": "x", "enabled": False}).status_code == 422
    tools = {t["name"]: t for t in one.get("/api/v1/tools").json()["tools"]}
    assert not tools["fake__echo"]["active"] and tools["fake__echo"]["enabled"]          # off through its plugin
    assert not tools["read_thing"]["active"] and not tools["read_thing"]["enabled"]
    assert tools["lint__shout"]["active"]
    assert all(t["active"] for t in two.get("/api/v1/tools").json()["tools"])            # one's switches only

    machine = Client(app, user(subject="app_m2m", client_id="app_m2m"))
    assert machine.get("/api/v1/tools").status_code == 200
    assert machine.put("/api/v1/tools/prefs", json={"target": "tool", "id": "read_thing", "enabled": False}).status_code == 403


def test_database_tool_timeouts():
    prefs = DatabaseToolPrefs(ttl=60)
    prefs.set_timeout("kb_research", 180, "admin_a")
    assert prefs.timeouts() == {"kb_research": 180.0}                    # this process: at once, despite the ttl
    prefs.set_timeout("kb_research", None)
    assert prefs.timeouts() == {}


def test_admins_set_a_tool_timeout(app):
    admin = Client(app, user(subject="user_admin", scopes=("app:access", "agent:use", "admin")))
    person = Client(app, user(subject="user_one"))
    listed = admin.get("/api/v1/tools").json()
    assert listed["can_set_timeouts"] and not person.get("/api/v1/tools").json()["can_set_timeouts"]
    own = {t["name"]: t for t in listed["tools"]}["read_thing"]["timeout_default"]
    assert {t["name"]: t for t in listed["tools"]}["read_thing"]["timeout"] == own
    assert person.put("/api/v1/tools/timeout", json={"name": "read_thing", "timeout": 180}).status_code == 403
    assert admin.put("/api/v1/tools/timeout", json={"name": "read_thing", "timeout": 5000}).status_code == 422
    assert admin.put("/api/v1/tools/timeout", json={"name": "nope", "timeout": 180}).status_code == 404
    changed = admin.put("/api/v1/tools/timeout", json={"name": "read_thing", "timeout": 180}).json()
    assert changed["timeout"] == 180.0 and changed["timeout_default"] == own
    assert {t["name"]: t for t in person.get("/api/v1/tools").json()["tools"]}["read_thing"]["timeout"] == 180.0  # everyone's
    assert admin.put("/api/v1/tools/timeout", json={"name": "read_thing", "timeout": None}).json()["timeout"] == own


def test_a_group_of_tools_is_switched_at_once(app):
    one = Client(app, user(subject="user_one"))
    many = lambda names, enabled: one.put("/api/v1/tools/prefs/many", json={"names": names, "enabled": enabled})  # noqa: E731
    assert many(["read_thing", "update_plan"], False).status_code == 409        # a control tool: nothing changes
    assert many(["read_thing", "nope"], False).status_code == 404
    assert all(t["enabled"] for t in one.get("/api/v1/tools").json()["tools"])
    assert many(["read_thing", "lint__shout", "read_thing"], False).json() == {"names": ["read_thing", "lint__shout"],
                                                                               "mode": "off", "enabled": False}
    tools = {t["name"]: t for t in one.get("/api/v1/tools").json()["tools"]}
    assert not tools["read_thing"]["active"] and not tools["lint__shout"]["active"] and tools["fake__echo"]["active"]
    assert many(["read_thing", "lint__shout"], True).status_code == 200
    assert all(t["active"] for t in one.get("/api/v1/tools").json()["tools"])
    assert one.put("/api/v1/tools/prefs/many", json={"names": [], "enabled": False}).status_code == 422


def test_tools_api_sets_ask(app):
    one = Client(app, user(subject="user_one"))
    pref = lambda body: one.put("/api/v1/tools/prefs", json=body)  # noqa: E731
    assert pref({"target": "tool", "id": "read_thing", "mode": "ask"}).json() == {
        "target": "tool", "id": "read_thing", "mode": "ask", "enabled": True}
    assert pref({"target": "mcp", "id": "system:fake/fake", "mode": "ask"}).status_code == 200
    assert pref({"target": "tool", "id": "read_thing", "mode": "maybe"}).status_code == 422
    assert pref({"target": "tool", "id": "read_thing"}).status_code == 422          # mode or enabled
    listed = one.get("/api/v1/tools").json()
    tools = {t["name"]: t for t in listed["tools"]}
    assert tools["read_thing"]["mode"] == "ask" and tools["read_thing"]["asks"] and tools["read_thing"]["active"]
    assert tools["fake__echo"]["mode"] == "on" and tools["fake__echo"]["asks"]      # through its endpoint
    assert not tools["lint__shout"]["asks"]
    assert {(g["target"], g["id"]): g for g in listed["groups"]}[("mcp", "system:fake/fake")]["mode"] == "ask"
    assert one.put("/api/v1/tools/prefs/many", json={"names": ["read_thing", "lint__shout"], "mode": "on"}).json()["mode"] == "on"
    assert not {t["name"]: t for t in one.get("/api/v1/tools").json()["tools"]}["read_thing"]["asks"]
