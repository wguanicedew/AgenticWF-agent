"""The admin page and its API: task types in groups (system, plugin, user), models by owner and visibility."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.context.task_types import TaskTypeError, TaskTypeResolver
from agenticwf.agent.runtime.llm.base import LLMRequest, Message
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry
from agenticwf.agent.runtime.secrets import SecretBox

from tests.conftest import SECRET, Client, user
from tests.test_models import Fixed, MemoryModels
from tests.test_runtime import make_system_dir

ALICE, BOB = ("user_alice", None), ("user_bob", None)


def plugin_dir(tmp_path):
    plugins = tmp_path / "plugins"
    (plugins / "lint" / "task-types" / "check").mkdir(parents=True)
    (plugins / "lint" / "task-types" / "check" / "AGENTS.md").write_text("# lint agents\n")
    (plugins / "lint" / "task-types" / "check" / "task.yaml").write_text("mode: plan\nmax_steps: 9\n")
    return plugins


# --- task types: plugins, users' own --------------------------------------------------------------

def test_plugin_task_types(tmp_path):
    system = make_system_dir(tmp_path)
    sandbox = tmp_path / "sandbox"
    (sandbox / "plugins" / "mine" / "task-types" / "notes").mkdir(parents=True)
    (sandbox / "plugins" / "mine" / "task-types" / "notes" / "WORKFLOW.md").write_text("# my plugin workflow\n")
    (sandbox / "plugins" / "mine" / "task-types" / "notes" / "task.yaml").write_text("mode: auto\n")   # not trusted
    resolver = TaskTypeResolver(system, user_sandbox=lambda o: sandbox, plugin_dirs=[plugin_dir(tmp_path)])

    check = resolver.resolve("lint.check", ALICE)
    assert check.provenance == {"AGENTS.md": "plugin:lint.check/AGENTS.md", "WORKFLOW.md": "system:default/WORKFLOW.md"}
    assert check.settings.mode.value == "plan" and check.settings.max_steps == 9 and not check.fell_back
    notes = resolver.resolve("mine.notes", ALICE)
    assert notes.workflow.content == "# my plugin workflow\n" and notes.settings.mode.value == "default"   # user plugin: no task.yaml
    assert resolver.resolve("lint.nothing", ALICE).fell_back

    groups = {i.name: i.source for i in resolver.catalogue(ALICE)}
    assert groups == {"default": "system", "review": "system", "lint.check": "plugin", "mine.notes": "plugin"}


def test_user_task_types_are_written_per_owner(tmp_path):
    resolver = TaskTypeResolver(make_system_dir(tmp_path), user_sandbox=lambda o: tmp_path / "sb" / o[0])
    resolver.write_user_type(ALICE, "my-review", {"AGENTS.md": "# mine\n", "WORKFLOW.md": None})
    assert resolver.resolve("my-review", ALICE).provenance == {
        "AGENTS.md": "user:my-review/AGENTS.md", "WORKFLOW.md": "system:default/WORKFLOW.md"}
    assert resolver.resolve("my-review", BOB).fell_back                        # not bob's
    resolver.write_user_type(ALICE, "review", {"AGENTS.md": "x"})                # a system name: fine, it is scoped
    assert resolver.resolve("review", ALICE).effective == "user_alice:review"
    assert resolver.resolve("review", BOB).effective == "system:review"
    with pytest.raises(TaskTypeError):
        resolver.write_user_type(ALICE, "default", {"AGENTS.md": "x"})           # reserved
    with pytest.raises(TaskTypeError):
        resolver.write_user_type(ALICE, "user_bob:x", {"AGENTS.md": "x"})        # someone else's scope
    assert not resolver.delete_user_type(ALICE, "system:review")
    with pytest.raises(TaskTypeError):
        resolver.write_user_type(ALICE, "../x", {"AGENTS.md": "x"})
    with pytest.raises(TaskTypeError):
        resolver.write_user_type(ALICE, "big", {"AGENTS.md": "x" * 40_000})
    assert resolver.delete_user_type(ALICE, "my-review") and not resolver.delete_user_type(ALICE, "my-review")


@pytest.fixture
def admin_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), sandbox_root=tmp_path / "sandboxes",
                           plugin_dirs=[plugin_dir(tmp_path)], model="echo",
                           llm_sections={"llm.model.echo": {"provider": "echo", "model": "echo", "label": "Echo"}})
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc


def test_task_types_api(admin_app):
    alice, bob = Client(admin_app, user()), Client(admin_app, user(subject="user_bob"))
    listed = alice.get("/api/v1/task-types").json()
    assert listed["can_add"] and {(t["name"], t["source"]) for t in listed["task_types"]} == {
        ("default", "system"), ("review", "system"), ("lint.check", "plugin")}
    review = alice.get("/api/v1/task-types/review").json()
    assert review["content"]["WORKFLOW.md"]["inherited"] and not review["editable"]
    assert alice.get("/api/v1/task-types/lint.check").json()["settings"]["max_steps"] == 9

    saved = alice.put("/api/v1/task-types/my-review", json={"agents_md": "# alice's rules\n"})
    assert saved.status_code == 200 and saved.json()["editable"] and saved.json()["files"] == ["AGENTS.md"]
    mine = alice.put("/api/v1/task-types/review", json={"agents_md": "# alice's review\n"})    # a system name, scoped
    assert mine.status_code == 200 and mine.json()["id"] == "user_alice:review"
    refs = {t["id"]: t["ref"] for t in alice.get("/api/v1/task-types").json()["task_types"]}
    assert refs["user_alice:review"] == "review" and refs["system:review"] == "system:review"
    assert alice.get("/api/v1/task-types/system:review").json()["source"] == "system"
    assert alice.put("/api/v1/task-types/system:review", json={"agents_md": "x"}).status_code == 403   # not hers
    assert alice.delete("/api/v1/task-types/review").status_code == 204
    assert alice.put("/api/v1/task-types/default", json={"agents_md": "x"}).status_code == 409
    assert alice.put("/api/v1/task-types/Bad", json={"agents_md": "x"}).status_code == 422
    assert "my-review" in [t["name"] for t in alice.get("/api/v1/task-types").json()["task_types"]]
    assert "my-review" not in [t["name"] for t in bob.get("/api/v1/task-types").json()["task_types"]]
    assert bob.delete("/api/v1/task-types/my-review").status_code == 404

    sid = alice.post("/api/v1/sessions", json={}).json()["id"]                 # the chat can use it
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "go", "task_type": "my-review"}).json()["status"] == "succeeded"
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "go", "task_type": "lint.check"}).status_code == 200

    machine = Client(admin_app, user(subject="app_m2m", client_id="app_m2m"))
    assert machine.put("/api/v1/task-types/x", json={"agents_md": "x"}).status_code == 403
    assert alice.delete("/api/v1/task-types/my-review").status_code == 204


def test_admin_page(admin_app):
    page = Client(admin_app, user()).get("/admin")
    assert page.status_code == 200 and "Task types" in page.text and "<script src" not in page.text
    assert admin_app.get("/admin").status_code == 401


# --- models: owner and visibility ------------------------------------------------------------------

def test_model_visibility_and_ownership():
    store = MemoryModels()
    catalog = ModelCatalog([ModelEntry("sys", "echo", "s")], default="sys", store=store, refresh_seconds=0)
    catalog.put(ModelEntry("mine", "echo", "m", owner_subject="u1", owner_org="cern", visibility="user"), by=("u1", "cern"))
    catalog.put(ModelEntry("team", "echo", "t", owner_subject="u1", owner_org="cern", visibility="organization"), by=("u1", "cern"))
    seen = lambda owner: [e.id for e in catalog.entries(owner=owner)]   # noqa: E731
    assert seen(("u1", "cern")) == ["sys", "mine", "team"]
    assert seen(("u2", "cern")) == ["sys", "team"]                     # same organization
    assert seen(("u2", None)) == ["sys"] and seen(("u1", None)) == ["sys"]   # personal sign-in is another owner
    with pytest.raises(PermissionError):
        catalog.put(ModelEntry("team", "echo", "x", owner_subject="u2", owner_org="cern"), by=("u2", "cern"))
    with pytest.raises(PermissionError):
        catalog.delete("team", by=("u2", "cern"))
    with pytest.raises(ValueError):                                    # organization needs an org sign-in
        catalog.put(ModelEntry("x", "echo", "x", owner_subject="u3", visibility="organization"), by=("u3", None))
    assert catalog.resolve("mine", ("u2", "cern")).id == "sys"         # not theirs: the default


def test_router_only_uses_models_the_owner_may_use():
    from agenticwf.agent.runtime.llm.router import ModelRouter

    catalog = ModelCatalog([ModelEntry("sys", "echo", "sys-model")], default="sys", store=MemoryModels(), refresh_seconds=0)
    catalog.put(ModelEntry("private", "echo", "private-model", owner_subject="u1", visibility="user"), by=("u1", None))
    adapters = {}
    router = ModelRouter(catalog, lambda e, s: adapters.setdefault(e.id, Fixed(e.model)))
    ask = lambda owner: asyncio.run(router.complete(LLMRequest(model="private", owner=owner, system="",   # noqa: E731
                                                                messages=[Message("user", "x")]))).text
    assert ask(("u1", None)) == "private-model" and ask(("u2", None)) == "sys-model"


def test_migration_adds_owner_columns_to_an_old_table():
    from sqlalchemy import inspect, text

    from agenticwf.agent.api.db import init_db
    from agenticwf.agent.core.orm.base.session import get_engine

    import sqlalchemy as sa

    engine = get_engine()
    old = sa.Table("agent_models", sa.MetaData(),   # the table as the previous version created it
                   sa.Column("id", sa.String(64), primary_key=True), sa.Column("provider", sa.String(16), nullable=False),
                   sa.Column("model", sa.String(255), nullable=False), sa.Column("label", sa.String(200), nullable=False),
                   sa.Column("description", sa.Text, nullable=False), sa.Column("base_url", sa.String(1024)),
                   sa.Column("api_key_env", sa.String(255)), sa.Column("api_key_encrypted", sa.Text),
                   sa.Column("api_key_hint", sa.String(16)), sa.Column("options", sa.Text),
                   sa.Column("enabled", sa.Integer, nullable=False), sa.Column("created_by", sa.String(255)),
                   sa.Column("created_at", sa.DateTime, nullable=False), sa.Column("updated_at", sa.DateTime, nullable=False))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE agent_models"))
        old.create(conn)
        conn.execute(text("DELETE FROM alembic_version"))
    init_db()
    columns = {c["name"] for c in inspect(engine).get_columns("agent_models")}
    assert {"owner_subject", "owner_org", "visibility"} <= columns
    init_db()   # and again: every revision is idempotent
    _restore_table("agent_models")   # the next tests need the table exactly as the models define it


def _restore_table(name: str) -> None:
    from agenticwf.agent.core.orm.base.session import BASE, get_engine

    table = BASE.metadata.tables[name]
    table.drop(get_engine(), checkfirst=True)
    table.create(get_engine())


def test_plugin_task_types_are_scoped(tmp_path):
    from dataclasses import replace

    from agenticwf.agent.runtime.policy.engine import DecisionKind, PolicyEngine
    from agenticwf.agent.runtime.tools.registry import ToolRegistry
    from tests.test_runtime import build
    from tests.test_runtime import user as runtime_user

    system = make_system_dir(tmp_path)
    sandbox = tmp_path / "sandbox"
    (sandbox / "plugins" / "lint" / "task-types" / "check").mkdir(parents=True)   # same plugin and type names
    (sandbox / "plugins" / "lint" / "task-types" / "check" / "AGENTS.md").write_text("# my lint\n")
    (sandbox / "plugins" / "lint" / "task-types" / "check" / "task.yaml").write_text("mode: auto\n")
    resolver = TaskTypeResolver(system, user_sandbox=lambda o: sandbox if o[0] == "user_a" else None,
                                plugin_dirs=[plugin_dir(tmp_path)])
    owner = ("user_a", None)
    ids = {i.id: i.ref for i in resolver.catalogue(owner) if i.name == "lint.check"}
    assert ids == {"system:lint.check": "system:lint.check", "user_a:lint.check": "lint.check"}
    mine = resolver.resolve("lint.check", owner)
    assert mine.effective == "user_a:lint.check" and mine.settings.mode.value == "default"   # its task.yaml: ignored
    assert resolver.resolve("system:lint.check", owner).settings.max_steps == 9

    # A tool for the system's lint.check is not offered in the user's lint.check.
    registry = ToolRegistry(system_dir=system)
    registry.load()
    read = replace(registry.lookup("read_thing", owner), task_types=("lint.check",))
    harness, _, _ = build(system, [])
    harness.services.task_types.plugins = resolver.plugins
    harness.services.task_types.user_sandbox = resolver.user_sandbox
    theirs = harness.create_task(runtime_user(), "x", task_type="lint.check")
    system_task = harness.create_task(runtime_user(), "x", task_type="system:lint.check")
    assert PolicyEngine().decide(read, {}, theirs).kind is DecisionKind.DENY
    assert PolicyEngine().decide(read, {}, system_task).kind is DecisionKind.ALLOW
