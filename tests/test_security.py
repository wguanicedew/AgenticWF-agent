"""Regressions from the system review: secrets, outbound hosts, the chat page's renderer."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from agenticwf.agent.runtime.llm.base import LLMError
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry
from agenticwf.agent.runtime.llm.router import ModelRouter, build_adapter

from tests.conftest import Client, user
from tests.test_models import MemoryModels

ALICE = ("user_alice", None)


# --- models users add never read the server's environment, and point only at allowed hosts ----------

@pytest.fixture
def secrets_in_env(monkeypatch):
    for name, value in {"OPENAI_API_KEY": "sk-operator", "ANTHROPIC_API_KEY": "sk-ant-operator",
                        "ANTHROPIC_AUTH_TOKEN": "operator-token", "OPENAI_ORG_ID": "org-operator",
                        "OPENAI_ADMIN_KEY": "admin-operator", "OPENAI_CUSTOM_HEADERS": "X-Internal: s1",
                        "ANTHROPIC_CUSTOM_HEADERS": "X-Internal: s2", "AUTH_CONTEXT_SECRET": "hmac"}.items():
        monkeypatch.setenv(name, value)


def users_model(**kw):
    return ModelEntry(**{"id": "mine", "provider": "openai", "model": "m", "source": "database",
                         "owner_subject": "user_alice", "visibility": "user", **kw})


def test_a_users_model_never_reads_the_environment(secrets_in_env):
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    with pytest.raises(LLMError, match="no API key"):
        build_adapter(users_model(), {})                                   # no key: never OPENAI_API_KEY
    with pytest.raises(LLMError, match="no API key"):
        build_adapter(users_model(provider="anthropic"), {})
    client = build_adapter(users_model(api_key="sk-mine", api_key_env="AUTH_CONTEXT_SECRET"), {"api_key_env": "X"}).client
    assert client.api_key == "sk-mine" and client.organization is None and client.admin_api_key == ""
    assert "X-Internal" not in {k for k, v in client.default_headers.items() if isinstance(v, str)}
    anthropic = build_adapter(users_model(provider="anthropic", api_key="sk-mine"), {}).client
    assert anthropic.auth_headers == {"X-Api-Key": "sk-mine"}
    assert "X-Internal" not in {k for k, v in anthropic.default_headers.items() if isinstance(v, str)}
    # an operator's configured model still uses the environment
    assert build_adapter(users_model(source="config"), {}).client.api_key == "sk-operator"
    assert build_adapter(users_model(provider="vllm", base_url="http://v/v1"), {}).client.api_key == "EMPTY"


def test_a_users_model_points_only_at_allowed_hosts():
    catalog = ModelCatalog([ModelEntry("sys", "echo", "s")], default="sys", store=MemoryModels(), refresh_seconds=0,
                           allowed_hosts=("vllm.example.org", "*.models.example.org"))
    put = lambda **kw: catalog.put(users_model(api_key="k", **kw), by=ALICE)   # noqa: E731
    with pytest.raises(ValueError, match="only for models operators configure"):
        put(api_key_env="GITHUB_TOKEN")
    with pytest.raises(ValueError, match="allow-list"):
        put(base_url="https://attacker.example/v1")
    with pytest.raises(ValueError, match="allow-list"):
        put(provider="ollama")                                             # the default is a server-side address
    with pytest.raises(ValueError, match="internal"):
        put(base_url="http://localhost.models.example.org/v1")             # a wildcard never reaches inside
    assert put(base_url="http://vllm.example.org:8000/v1").base_url        # listed by name
    assert put(base_url=None).id == "mine"                                 # the provider's own API

    catalog.allowed_hosts = ()                                             # the operator removed it later
    later = ModelEntry("x", "openai", "m", base_url="http://vllm.example.org/v1", api_key="k", source="database")
    router = ModelRouter(catalog, lambda e, s: pytest.fail("built an adapter for a refused model"))
    with pytest.raises(LLMError, match="allow-list"):
        router.adapter(later)                                              # checked again at every call


@pytest.fixture
def models_app(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agenticwf.agent.api.app import create_app
    from agenticwf.agent.api.settings import Settings
    from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
    from agenticwf.agent.runtime.secrets import SecretBox
    from tests.conftest import SECRET
    from tests.test_runtime import make_system_dir

    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), model="echo", llm_allowed_hosts=("vllm.example.org",),
                           llm_sections={"llm.model.echo": {"provider": "echo", "model": "echo"}})
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc, monkeypatch


def test_model_keys_survive_a_lost_secrets_key(models_app):
    from sqlalchemy import text

    from agenticwf.agent.core.orm.base.session import get_engine
    from agenticwf.agent.runtime.secrets import SecretBox

    app, monkeypatch = models_app
    alice = Client(app, user())
    assert alice.post("/api/v1/models", json={"id": "env", "provider": "openai", "model": "m",
                                              "api_key_env": "OPENAI_API_KEY"}).status_code == 422
    assert alice.post("/api/v1/models", json={"id": "far", "provider": "openai", "model": "m", "api_key": "k",
                                              "base_url": "https://attacker.example/v1"}).status_code == 422
    assert alice.post("/api/v1/models", json={"id": "mine", "provider": "vllm", "model": "m", "api_key": "sk-keep-me",
                                              "base_url": "http://vllm.example.org/v1"}).status_code == 201
    stored = lambda: get_engine().connect().execute(text("SELECT api_key_encrypted FROM agent_models")).scalar()   # noqa: E731
    before = stored()
    catalog = app.app.state.agent.catalog
    catalog.store.box = SecretBox(SecretBox.generate_key())               # the key changed under us
    catalog._refresh(force=True)
    assert catalog.get("mine").api_key_unreadable
    assert alice.patch("/api/v1/models/mine", json={"label": "renamed"}).status_code == 200
    assert stored() == before                                              # untouched, not erased


# --- the chat page's Markdown renderer ------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_rendered_markdown_only_ever_contains_our_own_tags(tmp_path):
    page = (Path(__file__).resolve().parents[1] / "src" / "agenticwf" / "agent" / "api" / "web" / "index.html").read_text()
    code = page[page.index("const esc = s =>"):page.index("const codeBlock = ")]
    cases = [
        "[x](https://a.example/(https://b.example/x)onmouseover=alert(1)//)",
        '[t](https://a.example/(https://b.example/"onmouseover=alert(1)//))',
        "https://a.example/(https://b.example/x) and [y](https://c.example/https://d.example)",
        "see [**bold** link](https://e.example/*a*b) and https://f.example/~~x~~ `code https://g.example` end",
        "\u0001" + "0\u0001 https://h.example/\u00010\u0001",
        '[a `c` b](https://i.example/p?q=1&r="2")',
    ]
    script = tmp_path / "check.js"
    script.write_text(code + f"\nconsole.log(JSON.stringify({json.dumps(cases)}.map(inline)));\n")
    rendered = json.loads(subprocess.run(["node", str(script)], capture_output=True, text=True, check=True).stdout)
    import re
    allowed = re.compile(r'^(</?(strong|em|del|code)>|</a>|<a href="[^"<>]*" target="_blank" rel="noopener noreferrer">)$')
    for out in rendered:
        for tag in re.findall(r"<[^>]*>", out):
            assert allowed.match(tag), (tag, out)


# --- policy -----------------------------------------------------------------------------------------

def test_policy_never_trusts_what_a_user_tool_says_about_itself(tmp_path):
    from dataclasses import replace

    from agenticwf.agent.runtime.policy.engine import DecisionKind, PolicyConfig, PolicyEngine, PolicyRule
    from agenticwf.agent.runtime.tools.registry import ToolRegistry
    from agenticwf.agent.runtime.tools.spec import ToolSource
    from agenticwf.agent.runtime.types import Mode
    from tests.test_runtime import build, make_system_dir
    from tests.test_runtime import user as runtime_user

    system = make_system_dir(tmp_path)
    (system / "tools" / "release.md").write_text(
        (system / "tools" / "read_thing.md").read_text().replace("name: read_thing", "name: release")
        .replace("handler:", "task_types: [ops.release]\nhandler:"))
    registry = ToolRegistry(system_dir=system)
    registry.load()
    harness, _, _ = build(system, [])
    task = harness.create_task(runtime_user(), "x", mode=Mode.AUTO)
    read = registry.lookup("read_thing", task.owner)
    shared = replace(read, name="bob__upload", source=ToolSource("database", "mcp:x", ("user_bob", "cern"),
                                                                   visibility="organization"))
    engine = PolicyEngine()
    assert engine.decide(shared, {}, task).kind is DecisionKind.ASK                  # auto mode: still asks
    assert engine.decide(read, {}, task).kind is DecisionKind.ALLOW
    task.mode = Mode.PLAN
    assert shared not in engine.visible([shared], task) and read in engine.visible([read], task)   # "none" is a claim

    fell_back = harness.create_task(runtime_user(), "x", task_type="ops.release")    # no such type: default's files
    assert fell_back.task_type.fell_back
    assert registry.lookup("release", task.owner) not in engine.visible([registry.lookup("release", task.owner)], fell_back)

    rules = PolicyEngine(PolicyConfig(rules=(PolicyRule("read_thing", DecisionKind.ASK),
                                             PolicyRule("read_thing", DecisionKind.DENY, "", (("value", r"rm\s+-rf"),)))))
    task.mode = Mode.DEFAULT
    assert rules.decide(read, {"value": "rm -rf /data"}, task).kind is DecisionKind.DENY    # order does not matter
    assert rules.decide(read, {"value": "ls"}, task).kind is DecisionKind.ASK


def test_a_task_type_limits_the_modes_it_runs_in(tmp_path):
    from agenticwf.agent.runtime.types import Mode
    from tests.test_runtime import build, make_system_dir
    from tests.test_runtime import user as runtime_user

    system = make_system_dir(tmp_path)                                  # `review`: mode: plan
    (system / "task-types" / "careful").mkdir()
    (system / "task-types" / "careful" / "task.yaml").write_text("modes: [default, plan]\n")
    harness, _, _ = build(system, [])
    assert harness.create_task(runtime_user(), "x", task_type="review", mode=Mode.AUTO).mode is Mode.PLAN
    assert harness.create_task(runtime_user(), "x", task_type="careful", mode=Mode.AUTO).mode is Mode.DEFAULT
    assert harness.create_task(runtime_user(), "x", task_type="careful", mode=Mode.PLAN).mode is Mode.PLAN
    assert harness.create_task(runtime_user(), "x", mode=Mode.AUTO).mode is Mode.AUTO   # default: every mode


# --- smaller items --------------------------------------------------------------------------------------

def test_the_return_address_stays_on_this_site():
    from agenticwf.agent.api.local import _safe_return_to

    assert _safe_return_to("/admin#models") == "/admin#models"
    for value in ("//evil.example", "/\t/evil.example", "/\n/evil.example", "/\\evil", "https://evil.example", "/ x"):
        assert _safe_return_to(value) == "/", value


def test_health_checks_the_database(client):
    assert client.get("/healthz").json() == {"status": "ok"}
