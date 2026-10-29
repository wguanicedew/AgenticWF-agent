"""Plugins: task types, tools (Markdown + handlers.py) and MCP servers, and the admin page's API."""

import asyncio
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.mcp.manager import McpManager, McpSettings
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import CallContext, HandlerKind

from tests.conftest import SECRET, Client, user
from tests.test_runtime import make_system_dir, tool_md

ALICE = ("user_alice", None)

# A minimal MCP server over stdio: two tools, `echo` (read-only) and `Fail-Hard`.
MCP_SERVER = r'''
import json, sys
TOOLS = [
    {"name": "echo", "description": "Echo the text back.\nMore detail.", "annotations": {"readOnlyHint": True},
     "inputSchema": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
    {"name": "Fail-Hard", "description": "Always fails.", "inputSchema": {"type": "object"}},
]
for line in sys.stdin:
    msg = json.loads(line)
    if "id" not in msg:
        continue
    method, params = msg["method"], msg.get("params") or {}
    if method == "initialize":
        result = {"protocolVersion": params["protocolVersion"], "capabilities": {"tools": {}},
                  "serverInfo": {"name": "fake", "version": "1.0"}}
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call" and params["name"] == "echo":
        result = {"content": [{"type": "text", "text": "echo: " + params["arguments"]["text"]}]}
    elif method == "tools/call":
        result = {"content": [{"type": "text", "text": "boom"}], "isError": True}
    else:
        print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "error": {"code": -32601, "message": "no"}}), flush=True)
        continue
    print("a log line on stdout", flush=True)
    print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}), flush=True)
'''

HANDLERS = '''
from agenticwf.agent.runtime.tools.spec import ToolResult

def shout(call, arguments):
    return ToolResult(content=arguments.get("value", "").upper())

HANDLERS = {"shout": shout}
'''


def make_plugins(tmp_path):
    """lint: a task type, a tool with a Python handler; fake: an MCP server; off: disabled."""
    root = tmp_path / "plugins"
    lint = root / "lint"
    (lint / "task-types" / "check").mkdir(parents=True)
    (lint / "task-types" / "check" / "AGENTS.md").write_text("# lint agents\n")
    (lint / "task-types" / "check" / "WORKFLOW.md").write_text("# lint workflow\n")
    (lint / "task-types" / "check" / "task.yaml").write_text("mode: plan\n")
    (lint / "tools").mkdir()
    (lint / "tools" / "shout.md").write_text(tool_md("shout", scopes="[]").replace("ref: test.shout", "ref: lint.shout"))
    (lint / "handlers.py").write_text(HANDLERS)
    (lint / "plugin.yaml").write_text("version: 1.0.0\ndescription: Lint things\nauthor: Ops\n")

    fake = root / "fake"
    fake.mkdir()
    (fake / "server.py").write_text(MCP_SERVER)
    (fake / ".mcp.json").write_text(json.dumps({"mcpServers": {"fake": {"command": sys.executable, "args": ["server.py"]}}}))

    off = root / "off"
    (off / "task-types" / "hidden").mkdir(parents=True)
    (off / "task-types" / "hidden" / "AGENTS.md").write_text("x")
    (off / "plugin.yaml").write_text("enabled: false\n")
    return root


@pytest.fixture
def manager(tmp_path):
    m = PluginManager([make_plugins(tmp_path)], mcp=McpManager(McpSettings(connect_timeout=10)))
    yield m
    m.mcp.close()


def call(handler, arguments):
    return handler(CallContext(task_id="t", call_id="c", auth=user()), arguments)


def test_plugin_components(manager):
    errors = manager.load()
    assert errors == []
    by = {p.name: p for p in manager.plugins()}
    assert by["lint"].kinds == ["task_types", "tools"] and by["lint"].manifest.version == "1.0.0"
    assert by["fake"].kinds == ["mcp"] and by["fake"].mcp_endpoints[0].expansion.status == "ok"
    assert not by["off"].enabled and by["off"].kinds == ["task_types"]
    assert set(manager.task_type_dirs()) == {"system:lint.check"}                # disabled: not offered
    assert {s.name for s in manager.tools()} == {"lint__shout", "fake__echo", "fake__fail_hard"}


def test_plugin_task_types_resolve(tmp_path, manager):
    resolver = TaskTypeResolver(make_system_dir(tmp_path), plugins=manager)
    check = resolver.resolve("lint.check", ALICE)
    assert check.provenance == {"AGENTS.md": "plugin:lint.check/AGENTS.md", "WORKFLOW.md": "plugin:lint.check/WORKFLOW.md"}
    assert check.settings.mode.value == "plan"
    assert resolver.resolve("off.hidden", ALICE).fell_back


def test_plugin_tools_run_through_the_registry(tmp_path, manager):
    registry = ToolRegistry(system_dir=make_system_dir(tmp_path), plugins=manager)
    assert registry.load() == []
    shout = registry.lookup("lint__shout", ALICE)
    assert call(registry.handler(shout), {"value": "hi"}).content == "HI"   # handlers.py

    echo = registry.lookup("fake__echo", ALICE)
    assert echo.handler.kind is HandlerKind.MCP and echo.summary == "Echo the text back."
    assert echo.effects.value == "none" and echo.approval.value == "auto"     # read-only hint
    assert call(registry.handler(echo), {"text": "hello"}).content == "echo: hello"
    fail = registry.lookup("fake__fail_hard", ALICE)
    assert fail.approval.value == "ask" and fail.effects.value == "external"
    assert call(registry.handler(fail), {}).is_error

    assert registry.load() == []                                               # reload keeps working
    assert call(registry.handler(registry.lookup("fake__echo", ALICE)), {"text": "again"}).content == "echo: again"


def test_broken_plugins_are_reported(tmp_path):
    root = tmp_path / "plugins"
    (root / "Bad-Name").mkdir(parents=True)
    (root / "wrong").mkdir()
    (root / "wrong" / "plugin.yaml").write_text("name: other\n")
    (root / "gone").mkdir()
    (root / "gone" / "plugin.yaml").write_text("mcp_servers:\n  x: {command: /nonexistent/server}\n  y: {transport: sse, url: http://h}\n")
    manager = PluginManager([root])
    errors = manager.load()
    assert any("invalid plugin name" in e for e in errors) and any("does not match" in e for e in errors)
    gone = manager.get("gone")
    assert gone.mcp_endpoints[0].expansion.status == "error" and "cannot start" in gone.mcp_endpoints[0].expansion.error
    assert any("unsupported transport" in e for e in gone.errors)


# --- a user's own plugins: no code, no environment, allow-listed hosts only --------------------------

class _McpHttp(BaseHTTPRequestHandler):
    def do_POST(self):   # noqa: N802
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if "id" not in msg:
            self.send_response(202)
            self.end_headers()
            return
        result = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": {"name": "web"}},
                  "tools/list": {"tools": [{"name": "lookup", "description": "Look up a word.",
                                            "annotations": {"readOnlyHint": True}}]},
                  "tools/call": {"content": [{"type": "text", "text": "found"}]}}[msg["method"]]
        body = f"event: message\ndata: {json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result})}\n\n".encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Mcp-Session-Id", "s1")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def http_server():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _McpHttp)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/mcp"
    server.shutdown()


def test_user_plugins_are_restricted(tmp_path, http_server):
    sandbox = tmp_path / "sandbox"
    mine = sandbox / "plugins" / "mine"
    (mine / "tools").mkdir(parents=True)
    (mine / "handlers.py").write_text("raise SystemExit('must never be imported')\n")
    (mine / "plugin.yaml").write_text(
        f"mcp_servers:\n  web: {{url: '{http_server}'}}\n  local: {{command: sh}}\n"
        "  secret: {url: 'http://127.0.0.1/${AGENT_SECRETS_KEY}'}\n")

    blocked = PluginManager(user_sandbox=lambda o: sandbox)
    blocked.get("mine", ALICE)
    blocked.mcp.wait_idle()
    plugin = blocked.get("mine", ALICE)
    assert any("cannot run Python" in e for e in plugin.errors)
    assert any("start processes" in e for e in plugin.errors)
    assert any("environment" in e for e in plugin.errors)
    assert plugin.mcp_endpoints[0].expansion.status == "error" and "allow-list" in plugin.mcp_endpoints[0].expansion.error
    assert blocked.user_tools(ALICE) == []

    allowed = PluginManager(user_sandbox=lambda o: sandbox if o == ALICE else tmp_path / "other", mcp=McpManager(McpSettings(allowed_hosts=("127.0.0.1",))))
    registry = ToolRegistry(plugins=allowed)
    registry.load()
    assert registry.lookup("mine__lookup", ALICE) is None                       # being expanded in the background:
    assert allowed.mcp.wait_idle()                                             # the catalogue never waits for it
    spec = registry.lookup("mine__lookup", ALICE)
    assert spec.approval.value == "ask" and spec.source.kind == "user_plugin"   # users' MCP tools always ask
    assert registry.lookup("mine__lookup", ("user_bob", None)) is None
    assert call(registry.handler(spec), {}).content == "found"


# --- the admin page's API ---------------------------------------------------------------------------

@pytest.fixture
def plugins_app(tmp_path):
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), plugin_dirs=[make_plugins(tmp_path)], provider="echo")
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc


def test_plugins_api(plugins_app):
    alice = Client(plugins_app, user())
    listed = alice.get("/api/v1/plugins").json()
    assert not listed["can_reload"]
    by = {p["name"]: p for p in listed["plugins"]}
    assert by["lint"]["kinds"] == ["task_types", "tools"] and by["lint"]["handlers"] == ["lint.shout"]
    assert by["lint"]["task_types"] == [{"name": "lint.check", "files": ["AGENTS.md", "WORKFLOW.md"]}]
    server = by["fake"]["mcp_endpoints"][0]
    assert server["status"] == "ok" and [t["name"] for t in server["tools"]] == ["fake__echo", "fake__fail_hard"]
    assert server["target"].endswith("server.py") and not by["off"]["enabled"]
    assert "lint.check" in [t["name"] for t in alice.get("/api/v1/task-types").json()["task_types"]]

    assert alice.post("/api/v1/plugins/reload").status_code == 403
    admin = Client(plugins_app, user(scopes=("app:access", "agent:use", "admin")))
    assert admin.get("/api/v1/plugins").json()["can_reload"]
    reloaded = admin.post("/api/v1/plugins/reload")
    assert reloaded.status_code == 200 and reloaded.json()["errors"] == []
    assert "Plugins" in alice.get("/admin").text


def test_plugin_and_mcp_config_sections(tmp_path, monkeypatch):   # [plugins], [mcp] in agent.cfg
    cfg = tmp_path / "agent.cfg"
    cfg.write_text(f"[context]\nsystem_dir = {tmp_path}\n\n[plugins]\ndisabled = a, b\n\n"
                   "[mcp]\nallowed_hosts = *.example.org\ndiscover = false\n")
    monkeypatch.setenv("AGENTICWF_CONFIG", str(cfg))
    config = RuntimeConfig.from_core_config()
    assert config.plugins.disabled == ("a", "b")
    assert config.mcp.allowed_hosts == ("*.example.org",) and not config.mcp.discover


def test_mcp_handler_is_sync_and_runs_in_a_thread(manager):
    """ToolDispatcher runs synchronous handlers with asyncio.to_thread: the MCP call must not need a loop."""
    manager.load()
    spec = next(s for s in manager.tools() if s.name == "fake__echo")
    result = asyncio.run(asyncio.to_thread(call, manager.mcp.handler(spec), {"text": "x"}))
    assert result.content == "echo: x"
