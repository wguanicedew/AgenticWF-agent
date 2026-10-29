"""MCP endpoints: from plugins and from the database, expansion into tools, owners and visibility."""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
from agenticwf.agent.runtime.mcp.catalog import McpCatalog
from agenticwf.agent.runtime.mcp.db_store import DatabaseMcpStore
from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint
from agenticwf.agent.runtime.mcp.manager import McpManager, McpSettings
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.secrets import SecretBox
from agenticwf.agent.runtime.tools.registry import ToolRegistry

from tests.conftest import SECRET, Client, user
from tests.test_plugins import call, make_plugins
from tests.test_runtime import make_system_dir, tool_md

U1, U2, OUTSIDER = ("user_one", "cern"), ("user_two", "cern"), ("user_three", None)
WORDS = "user_one@cern:words"   # endpoint ids are <owner scope>:<name>


class _Server(BaseHTTPRequestHandler):
    """Streamable HTTP MCP server whose tool list a test can change; it records the headers it got."""

    TOOLS = [{"name": "lookup", "description": "Look up a word.", "annotations": {"readOnlyHint": True}},
             {"name": "write", "description": "Write a word."}]
    tools: list = TOOLS
    seen_headers: list = []
    list_reply = None          # replaces the whole tools/list result (malformed replies)
    list_delay = 0.0           # seconds tools/list takes (slow endpoints)
    redirect_to = None         # answer every request with a 302 to this URL
    server_info = {"name": "words", "version": "2"}

    def do_POST(self):   # noqa: N802
        msg = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        type(self).seen_headers.append(self.headers.get("Authorization"))
        if type(self).redirect_to:
            self.send_response(302)
            self.send_header("Location", type(self).redirect_to)
            self.end_headers()
            return
        if "id" not in msg:
            self.send_response(202)
            self.end_headers()
            return
        if msg["method"] == "tools/list" and type(self).list_delay:
            time.sleep(type(self).list_delay)
        listed = type(self).list_reply if type(self).list_reply is not None else {"tools": type(self).tools}
        result = {"initialize": {"protocolVersion": "2025-06-18", "capabilities": {}, "serverInfo": type(self).server_info},
                  "tools/list": listed,
                  "tools/call": {"content": [{"type": "text", "text": f"did {msg.get('params', {}).get('name')}"}]}}[msg["method"]]
        body = json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    _Server.tools, _Server.seen_headers, _Server.list_reply, _Server.list_delay = list(_Server.TOOLS), [], None, 0.0
    _Server.redirect_to, _Server.server_info = None, {"name": "words", "version": "2"}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Server)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/mcp"
    httpd.shutdown()


@pytest.fixture
def catalog():
    manager = McpManager(McpSettings(allowed_hosts=("127.0.0.1",)))
    yield McpCatalog(DatabaseMcpStore(SecretBox(SecretBox.generate_key())), manager, refresh_seconds=0)
    manager.close()


def endpoint(url, name="words", **kw):
    return McpEndpoint(name=name, transport="http", url=url, source="database", **kw)


# --- database endpoints: expansion, owners, visibility -----------------------------------------------

def test_database_endpoint_is_expanded_and_private_by_default(catalog, server):
    record = catalog.put(endpoint(server, headers={"Authorization": "Bearer sekrit-token"}), by=U1)
    assert record.status == "ok" and record.server == "words" and record.header_hints == {"Authorization": "…oken"}
    tools = {t.name: t for t, _ in catalog.tools()}
    assert set(tools) == {"words__lookup", "words__write"}
    assert tools["words__lookup"].approval == "ask" and tools["words__lookup"].effects == "none"   # users' tools ask

    registry = ToolRegistry(database=catalog, mcp=catalog.manager)
    registry.load()
    spec = registry.lookup("words__lookup", U1)
    assert spec.source.kind == "database" and spec.source.effective_visibility == "user"
    assert registry.lookup("words__lookup", U2) is None                        # private: not even the organization
    assert call(registry.handler(spec), {}).content == "did lookup"
    assert "Bearer sekrit-token" in _Server.seen_headers                        # decrypted for the call only

    from sqlalchemy import text
    from agenticwf.agent.core.orm.base.session import get_engine
    with get_engine().connect() as conn:                                        # stored encrypted
        stored = conn.execute(text("SELECT headers_encrypted FROM agent_mcp_endpoints")).scalar()
    assert "sekrit" not in stored


def test_tools_have_owners_and_visibility_of_their_own(catalog, server):
    catalog.put(endpoint(server), by=U1)
    catalog.set_tool(f"{U1[0]}@cern:words__lookup", U1, visibility="organization")          # share one tool, not the endpoint
    registry = ToolRegistry(database=catalog, mcp=catalog.manager)
    registry.load()
    assert registry.lookup("words__lookup", U2) is not None
    assert registry.lookup("words__write", U2) is None
    assert registry.lookup("words__lookup", OUTSIDER) is None
    assert [r.endpoint.name for r in catalog.endpoints(U2)] == []              # the endpoint stays private

    with pytest.raises(PermissionError):
        catalog.set_tool(f"{U1[0]}@cern:words__lookup", U2, enabled=False)                    # not theirs
    assert catalog.put(endpoint(server), by=U2).endpoint.id == "user_two@cern:words"   # their own, in their scope
    with pytest.raises(PermissionError):
        catalog.delete(WORDS, by=U2)

    catalog.set_tool(f"{U1[0]}@cern:words__write", U1, enabled=False, approval="deny")
    assert registry.lookup("words__write", U1) is None                          # disabled: not offered


def test_refresh_keeps_tool_settings(catalog, server):
    catalog.put(endpoint(server), by=U1)
    catalog.set_tool(f"{U1[0]}@cern:words__lookup", U1, visibility="organization")
    _Server.tools = [_Server.tools[0], {"name": "Count-Words", "description": "Count."}]
    record = catalog.refresh(WORDS, by=U1)
    tools = {t.name: t for t, _ in catalog.tools()}
    assert record.status == "ok" and set(tools) == {"words__lookup", "words__count_words"}   # write is gone
    assert tools["words__lookup"].visibility == "organization"
    assert catalog.delete(WORDS, by=U1) and catalog.tools() == []


def test_the_same_name_in_two_scopes_never_conflicts(catalog, server):
    first = catalog.put(endpoint(server), by=U1)
    second = catalog.put(endpoint(server), by=U2)                           # the same name, another owner
    assert (first.endpoint.id, second.endpoint.id) == (WORDS, "user_two@cern:words")
    catalog.set_tool(f"{WORDS}__lookup", U1, visibility="organization")    # U2 now sees two `words__lookup`
    registry = ToolRegistry(database=catalog, mcp=catalog.manager)
    registry.load()
    names = {s.name: s for s in registry.catalogue(U2) if s.name.startswith("words")}
    own = names["words__lookup"]
    shared = next(s for n, s in names.items() if n != "words__lookup" and n.endswith("__lookup"))
    assert own.source.owner == U2 and shared.source.owner == U1             # their own keeps the plain name
    assert len(shared.name) <= 64 and shared.name.endswith("__lookup")
    assert call(registry.handler(shared), {}).content == "did lookup"       # and the qualified one still calls
    assert catalog.delete(WORDS, by=U1) and catalog.get("user_two@cern:words") is not None


def test_database_endpoint_rules(catalog, server):
    with pytest.raises(ValueError):
        catalog.put(endpoint(server, name="Bad-Name"), by=U1)
    with pytest.raises(ValueError):
        catalog.put(McpEndpoint(name="local", transport="stdio", command="sh", source="database"), by=U1)
    with pytest.raises(ValueError):
        catalog.put(endpoint(server, visibility="organization"), by=OUTSIDER)    # no organization to share with
    record = catalog.put(endpoint("http://intranet.example.org/mcp", name="intranet"), by=U1)
    assert record.status == "error" and "allow-list" in record.error              # saved, but not expanded


# --- plugins: shared by default, narrowed by access ---------------------------------------------------

def test_plugin_access_narrows_who_sees_its_tools(tmp_path):
    root = make_plugins(tmp_path)
    (root / "lint" / "plugin.yaml").write_text("version: 1.0.0\naccess: {organizations: [cern]}\n")
    (root / "lint" / "tools" / "secret.md").write_text(
        tool_md("secret", scopes="[]", extra="access: {users: [user_one]}\n").replace("ref: test.secret", "ref: lint.shout"))
    (root / "fake" / "plugin.yaml").write_text("access: {scopes: [fake:use]}\n")
    manager = PluginManager([root], mcp=McpManager(McpSettings(connect_timeout=10)))
    registry = ToolRegistry(plugins=manager)
    try:
        assert registry.load() == []
        names = lambda owner: {s.name for s in registry.catalogue(owner)}   # noqa: E731
        assert {"lint__shout", "lint__secret", "fake__echo"} <= names(U1)
        assert "lint__secret" not in names(U2) and "lint__shout" in names(U2)   # the tool's own access
        assert not {"lint__shout", "lint__secret"} & names(OUTSIDER)            # the plugin's organizations
        assert registry.lookup("fake__echo", OUTSIDER).scopes == {"fake:use"}   # scopes: enforced by policy
        assert [p.name for p in manager.plugins(OUTSIDER)] == ["fake", "off"]
    finally:
        manager.mcp.close()


# --- the admin page's API -----------------------------------------------------------------------------

@pytest.fixture
def mcp_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), plugin_dirs=[make_plugins(tmp_path)], provider="echo",
                           mcp=McpSettings(allowed_hosts=("127.0.0.1",), connect_timeout=10))
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc


def test_mcp_api(mcp_app, server):
    one = Client(mcp_app, user(subject="user_one", org="cern"))
    two = Client(mcp_app, user(subject="user_two", org="cern"))

    added = one.post("/api/v1/mcp/endpoints", json={"name": "words", "url": server, "label": "Words",
                                                    "headers": {"Authorization": "Bearer abcdefgh1234"}})
    assert added.status_code == 201, added.text
    body = added.json()
    assert body["status"] == "ok" and body["group"] == "mine" and body["visibility"] == "user"
    assert body["headers"] == {"Authorization": "…1234"} and "abcdefgh" not in added.text
    assert [t["name"] for t in body["tools"]] == ["words__lookup", "words__write"]
    assert one.post("/api/v1/mcp/endpoints", json={"name": "words", "url": server}).status_code == 409
    assert one.post("/api/v1/mcp/endpoints", json={"name": "all", "url": server, "visibility": "all"}).status_code == 403

    groups = {e["id"]: e["group"] for e in one.get("/api/v1/mcp/endpoints").json()["endpoints"]}
    assert groups == {"system:fake/fake": "plugin", WORDS: "mine"}
    assert {e["id"] for e in two.get("/api/v1/mcp/endpoints").json()["endpoints"]} == {"system:fake/fake"}

    tools = {t["name"]: t for t in one.get("/api/v1/mcp/tools").json()["tools"]}
    assert tools["fake__echo"]["group"] == "plugin" and tools["words__lookup"]["editable"]
    assert one.patch("/api/v1/mcp/tools/words__lookup", json={"visibility": "organization"}).json()["visibility"] == "organization"
    shared = {t["name"]: t for t in two.get("/api/v1/mcp/tools").json()["tools"]}
    assert shared["words__lookup"]["group"] == "shared" and "words__write" not in shared
    assert two.patch(f"/api/v1/mcp/tools/{WORDS}__lookup", json={"enabled": False}).status_code == 403
    assert two.patch("/api/v1/mcp/tools/words__lookup", json={"enabled": False}).status_code == 404   # bare: their own

    changed = one.patch("/api/v1/mcp/endpoints/words", json={"label": "Dictionary"}).json()
    assert changed["label"] == "Dictionary" and changed["headers"] == {"Authorization": "…1234"}   # kept
    assert one.post("/api/v1/mcp/endpoints/words/refresh").json()["status"] == "ok"
    assert two.delete("/api/v1/mcp/endpoints/words").status_code == 404           # not visible to them
    machine = Client(mcp_app, user(subject="app_m2m", client_id="app_m2m"))
    assert machine.post("/api/v1/mcp/endpoints", json={"name": "m", "url": server}).status_code == 403

    sid = one.post("/api/v1/sessions", json={}).json()["id"]                       # the chat still works
    assert one.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi"}).status_code == 200
    assert one.delete("/api/v1/mcp/endpoints/words").status_code == 204
    assert "words__lookup" not in {t["name"] for t in one.get("/api/v1/mcp/tools").json()["tools"]}
    assert "MCP" in one.get("/admin").text


# --- regressions from the review ----------------------------------------------------------------------

def test_a_tool_definition_cannot_point_at_someone_elses_endpoint(tmp_path, catalog, server):
    from dataclasses import replace

    from agenticwf.agent.runtime.tools.spec import ToolSource

    catalog.put(endpoint(server, headers={"Authorization": "Bearer alice-only"}), by=U1)
    registry = ToolRegistry(database=catalog, mcp=catalog.manager)
    registry.load()
    real = registry.lookup("words__lookup", U1)
    forged = replace(real, name="steal", source=replace(real.source, owner=U2))          # someone else's copy
    assert call(registry.handler(forged), {}).error_code == "tool_unavailable"
    by_file = replace(real, name="steal", source=ToolSource("user_plugin", "/sandbox/x/tools/steal.md", U1))
    assert call(registry.handler(by_file), {}).error_code == "tool_unavailable"           # not from agenticwf.agent.runtime.mcp
    assert call(registry.handler(real), {}).content == "did lookup"

    sandbox = tmp_path / "sandbox"                                                         # the old way in: refused
    (sandbox / "plugins" / "x" / "tools").mkdir(parents=True)
    (sandbox / "plugins" / "x" / "tools" / "steal.md").write_text(tool_md("steal", kind="mcp", extra=(
        "  client: 'database:words'\n  tool: lookup\n")))
    manager = PluginManager(user_sandbox=lambda o: sandbox, mcp=catalog.manager)
    assert manager.user_tools(U2) == []
    assert any("handler.client is reserved" in e for e in manager.get("x", U2).errors)


def test_a_plugin_tool_file_can_call_its_own_endpoint_only(tmp_path, server):
    root = tmp_path / "plugins"
    (root / "dict" / "tools").mkdir(parents=True)
    (root / "dict" / "plugin.yaml").write_text(f"mcp_servers:\n  words: {{url: '{server}', tools: [lookup]}}\n")
    for name, endpoint_name, tool in (("define", "words", "lookup"), ("other", "elsewhere", "lookup"),
                                      ("hidden", "words", "write")):
        (root / "dict" / "tools" / f"{name}.md").write_text(tool_md(name, scopes="[]", kind="mcp", extra=(
            f"  endpoint: {endpoint_name}\n  tool: {tool}\n")))
    manager = PluginManager([root], mcp=McpManager(McpSettings(connect_timeout=10)))
    registry = ToolRegistry(plugins=manager)
    try:
        registry.load()
        assert call(registry.handler(registry.lookup("dict__define", U1)), {}).content == "did lookup"
        assert registry.lookup("dict__other", U1) is None                                 # no such endpoint here
        assert call(registry.handler(registry.lookup("dict__hidden", U1)), {}).error_code == "tool_unavailable"
    finally:
        manager.mcp.close()


def test_secrets_never_reach_the_shown_target(tmp_path, monkeypatch, catalog, server):
    monkeypatch.setenv("SECRET_X", "s3cret-value")
    root = tmp_path / "plugins"
    (root / "p").mkdir(parents=True)
    (root / "p" / "plugin.yaml").write_text(
        "mcp_servers:\n  a: {command: /bin/true, args: [--token, '${SECRET_X}'], env: {T: '${SECRET_X}'}}\n"
        "  b: {url: 'https://user:pw@mcp.example.org/${SECRET_X}/mcp?key=${SECRET_X}'}\n")
    manager = PluginManager([root], mcp=McpManager(McpSettings(discover=False)))
    view = json.dumps(manager.get("p").to_dict(U1))
    assert "s3cret" not in view and "pw@" not in view and "${SECRET_X}" in view
    assert manager.get("p").mcp_endpoints[0].endpoint.args == ("--token", "s3cret-value")   # used, never shown

    catalog.put(endpoint(server + "/tok-123/x", visibility="organization"), by=U1)
    record = catalog.get(WORDS)
    assert "tok-123" not in json.dumps(record.view_for(U2, [])) and "tok-123" in record.view_for(U1, [])["url"]


def test_reload_never_shows_a_half_built_catalogue(tmp_path):
    manager = PluginManager([make_plugins(tmp_path)], mcp=McpManager(McpSettings(discover=False)))
    registry = ToolRegistry(system_dir=make_system_dir(tmp_path), plugins=manager)
    registry.load()
    seen = []
    original = manager.load

    def slow_load():
        seen.append(registry.lookup("lint__shout", U1) is not None and registry.lookup("read_thing", U1) is not None)
        return original()

    manager.load = slow_load
    registry.load()
    assert seen == [True] and registry.lookup("lint__shout", U1) is not None


def test_building_a_catalogue_never_waits_for_an_endpoint(tmp_path, server):
    _Server.list_delay = 2.0
    sandbox = tmp_path / "sandbox"
    (sandbox / "plugins" / "slow").mkdir(parents=True)
    (sandbox / "plugins" / "slow" / "plugin.yaml").write_text(f"mcp_servers:\n  words: {{url: '{server}'}}\n")
    manager = PluginManager(user_sandbox=lambda o: sandbox, mcp=McpManager(McpSettings(allowed_hosts=("127.0.0.1",))))
    registry = ToolRegistry(plugins=manager)
    registry.load()
    started = time.monotonic()
    assert registry.lookup("slow__lookup", U1) is None
    assert time.monotonic() - started < 1.0
    assert manager.mcp.wait_idle(10) and registry.lookup("slow__lookup", U1) is not None


@pytest.mark.parametrize("files", [
    {"plugin.yaml": "mcp_servers: [github]\n"},
    {".mcp.json": '{"mcpServers": ["x"]}'},
    {".mcp.json": "[1, 2]"},
    {"plugin.yaml": "mcp_servers:\n  x: {command: /bin/true, env: [A=1]}\n"},
    {"plugin.yaml": "mcp_servers:\n  x: {command: /bin/true, args: --flag}\n"},
    {"plugin.yaml": "access: {organizations: 5}\n"},
    {"plugin.yaml": "enabled: maybe\n"},
    {"tools/t.md": "---\nname: t\nsummary: x\nhandler: 5\n---\nbody\n"},
    {"tools/t.md": "---\nname: t\nsummary: [x]\n---\nbody\n"},
])
def test_malformed_plugins_are_reported_not_raised(tmp_path, files):
    root = make_plugins(tmp_path)
    for name, text in files.items():
        (root / "broken" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "broken" / name).write_text(text)
    manager = PluginManager([root], mcp=McpManager(McpSettings(discover=False)))
    errors = manager.load()
    assert any("broken" in e for e in errors), errors
    assert {"lint", "fake", "off"} <= {p.name for p in manager.plugins()}                 # the others still load
    sandbox = tmp_path / "sandbox"                                                        # and in a user's sandbox
    (sandbox / "plugins").mkdir(parents=True)
    (root / "broken").rename(sandbox / "plugins" / "broken")
    manager.user_tools(U1)


@pytest.mark.parametrize("reply, status", [
    ({"tools": [{"name": "a", "description": 42}, "junk", {"name": "b", "annotations": []},
                {"name": "c", "inputSchema": "nope"}]}, "ok"),
    ({"tools": "nope"}, "error"),
    ([1, 2], "error"),
])
def test_malformed_tool_lists_become_an_error_or_problems(catalog, server, reply, status):
    _Server.list_reply = reply
    record = catalog.put(endpoint(server), by=U1)
    assert record.status == status
    if status == "ok":
        assert [t.name for t, _ in catalog.tools()] == ["words__b"] and len(record.problems) == 3


def test_access_only_ever_narrows():
    from agenticwf.agent.runtime.tools.spec import Access

    plugin = Access.from_mapping({"organizations": ["cern"], "scopes": ["a"]})
    assert plugin.merge(Access.from_mapping({"organizations": ["bnl"]})).allows(("x", "bnl")) is False
    nobody = plugin.merge(Access.from_mapping({"organizations": ["bnl"]}))
    assert not nobody.allows(("x", "cern")) and nobody.to_dict()["organizations"] == []
    narrower = Access.from_mapping({"organizations": ["cern", "bnl"]}).merge(Access.from_mapping({"organizations": ["cern"]}))
    assert narrower.allows(("x", "cern")) and not narrower.allows(("x", "bnl"))
    assert plugin.merge(Access.from_mapping({"users": ["u"], "scopes": ["b"]})).scopes == {"a", "b"}
    assert Access.from_mapping({"organizations": []}).allows(("x", None))                # empty list: no restriction


def test_plugin_views_show_only_what_the_viewer_may_use(tmp_path, server):
    root = tmp_path / "plugins"
    (root / "p" / "tools").mkdir(parents=True)
    (root / "p" / "plugin.yaml").write_text(
        f"mcp_servers:\n  open: {{url: '{server}'}}\n  cern: {{url: '{server}', access: {{organizations: [cern]}}}}\n")
    (root / "p" / "tools" / "t.md").write_text(tool_md("t", scopes="[]", extra="access: {organizations: [cern]}\n"))
    manager = PluginManager([root], mcp=McpManager(McpSettings(connect_timeout=10)))
    try:
        manager.load()
        outsider = manager.get("p", OUTSIDER).to_dict(OUTSIDER)
        assert [e["name"] for e in outsider["mcp_endpoints"]] == ["open"] and outsider["tools"] == []
        assert len(manager.get("p", U1).to_dict(U1)["mcp_endpoints"]) == 2
    finally:
        manager.mcp.close()


def test_patch_with_null_is_refused_not_a_500(mcp_app, server):
    one = Client(mcp_app, user(subject="user_one", org="cern"))
    assert one.post("/api/v1/mcp/endpoints", json={"name": "words", "url": server}).status_code == 201
    for field in ("label", "description", "timeout", "enabled", "url", "tools", "visibility"):
        assert one.patch("/api/v1/mcp/endpoints/words", json={field: None}).status_code == 422, field
    assert one.patch("/api/v1/mcp/endpoints/words", json={"approval": None}).status_code == 200   # null: the default
    one.post("/api/v1/models", json={"id": "mine", "provider": "echo", "model": "m"})
    assert one.patch("/api/v1/models/mine", json={"label": None}).status_code == 422
    assert one.patch("/api/v1/models/mine", json={"base_url": None}).status_code == 200


def test_a_lost_secrets_key_never_erases_stored_headers(server):
    from sqlalchemy import text

    from agenticwf.agent.core.orm.base.session import get_engine

    key = SecretBox.generate_key()
    manager = McpManager(McpSettings(allowed_hosts=("127.0.0.1",)))
    first = McpCatalog(DatabaseMcpStore(SecretBox(key)), manager, refresh_seconds=0)
    first.put(endpoint(server, headers={"Authorization": "Bearer keep-me"}), by=U1)
    stored = lambda: get_engine().connect().execute(text("SELECT headers_encrypted FROM agent_mcp_endpoints")).scalar()   # noqa: E731
    before = stored()

    for box in (None, SecretBox(SecretBox.generate_key())):                                # no key; another key
        later = McpCatalog(DatabaseMcpStore(box), manager, refresh_seconds=0)
        record = later.get(WORDS)
        assert record.headers_unreadable and later.tools_for(U1) == []                     # never called without them
        assert later.put(endpoint(server, label="renamed"), by=U1, headers_keep=True).status == "error"
        assert later.refresh(WORDS, by=U1).status == "error"
        assert stored() == before                                                          # untouched
    _Server.seen_headers.clear()
    assert first.refresh(WORDS, by=U1).status == "ok" and "Bearer keep-me" in _Server.seen_headers
    manager.close()


# --- regressions from the system review ------------------------------------------------------------

def test_users_endpoints_never_follow_redirects_or_reach_inside(server):
    wildcard = McpCatalog(DatabaseMcpStore(SecretBox(SecretBox.generate_key())), McpManager(McpSettings(allowed_hosts=("*",))),
                          refresh_seconds=0)
    record = wildcard.put(endpoint(server), by=U1)                          # 127.0.0.1 only matches "*"
    assert record.status == "error" and "internal address" in record.error
    _Server.redirect_to = "http://169.254.169.254/latest/meta-data"
    named = McpCatalog(DatabaseMcpStore(SecretBox(SecretBox.generate_key())),
                       McpManager(McpSettings(allowed_hosts=("127.0.0.1",))), refresh_seconds=0)
    record = named.put(endpoint(server, name="redir"), by=U1)
    assert record.status == "error" and "redirects are not followed" in record.error
    assert "/mcp" not in record.error                                         # never the path (it may hold a token)


def test_endpoint_paging_forever_is_stopped(catalog, server):
    _Server.list_reply = {"tools": [], "nextCursor": "again"}
    assert catalog.put(endpoint(server), by=U1).status == "ok"              # a repeated cursor ends it
    from agenticwf.agent.runtime.mcp import client as mcp_client

    pages = iter(range(10_000))
    original = mcp_client.McpClient.request
    mcp_client.McpClient.request = lambda self, m, p, timeout=None: {"tools": [], "nextCursor": str(next(pages))}
    try:
        record = catalog.refresh(WORDS, by=U1)
    finally:
        mcp_client.McpClient.request = original
    assert record.status == "error" and "pages" in record.error


def test_tool_names_always_name_one_owner(catalog, server):
    with pytest.raises(ValueError):
        catalog.put(endpoint(server, name="gw__jira"), by=U1)               # "__" would let names collide
    with pytest.raises(ValueError):
        catalog.put(endpoint(server, name="a" * 41), by=U1)                 # no room left for tool names
    _Server.tools = [{"name": "jira__search"}, {"name": "x" * 300}, {"name": "list", "inputSchema": {"type": "array"}},
                     {"name": "T" * 60}]
    _Server.server_info = {"name": "n" * 1000, "version": "v" * 1000}      # PostgreSQL enforces column sizes
    record = catalog.put(endpoint(server, name="a" * 40), by=U1)
    names = [t.name for t, _ in catalog.tools()]
    assert record.status == "ok" and len(record.server) == 255 and len(record.server_version) == 64
    assert all(len(n) <= 64 for n in names) and len(names) == 2 and len(record.problems) == 2

    from agenticwf.agent.runtime.mcp.db_store import _save_tools
    from agenticwf.agent.runtime.mcp.expand import McpTool

    _Server.tools = list(_Server.TOOLS)
    catalog.put(endpoint(server, name="other"), by=U1)
    stolen = McpTool(name=names[0], mcp_name="x", endpoint="database:user_one@cern:other", summary="s",
                     description="d", input_schema={"type": "object"}, owner=U1)
    _save_tools("user_one@cern:other", [stolen], replace_all=False)          # another endpoint's row: refused
    assert {t.name: t.owner for t, _ in catalog.tools()}[names[0]] == U1


def test_database_errors_never_show_the_values():
    from agenticwf.agent.core.orm.base.session import get_engine

    assert get_engine().hide_parameters

