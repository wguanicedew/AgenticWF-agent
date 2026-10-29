"""The knowledge service client, catalogue and tools (agenticwf.agent.runtime.knowledge), against a mocked KB."""

import asyncio
import dataclasses
import base64
import json

import httpx
import pytest

from agenticwf.agent.runtime.context.manager import ContextManager
from agenticwf.agent.runtime.knowledge import (ALL, KB_GREP, KB_READ, KB_RESEARCH, KB_SEARCH, SYSTEM, HttpKnowledgeCatalog,
                               KBMemoryStore, KnowledgeClient, KnowledgeService, KnowledgeSettings, encode_principal)
from agenticwf.agent.runtime.tools.spec import CallContext

from tests.conftest import user

KBS = [
    {"id": "kb-1", "ref": "user_alice:personal", "slug": "personal", "name": "Personal", "description": "mine",
     "knowledge_count": 1},
    {"id": "kb-2", "ref": "system:eic-docs", "slug": "eic-docs", "name": "EIC docs", "description": "operations",
     "knowledge_count": 40},
]
RESULT = {"id": "chunk-9", "knowledge_id": "item-3", "knowledge_base_id": "kb-2", "knowledge_title": "Pilot guide",
          "content": "Restart the pilot after three stage-out failures.", "score": 0.83,
          "context_header": "Operations > Pilot",
          "citation_text": "Ops Team (2026). Pilot guide. https://ops.example/pilot [§ Operations > Pilot]"}


class FakeKB:
    """Records requests; answers like the KB API."""

    def __init__(self, fail=False):
        self.requests: list[httpx.Request] = []
        self.fail = fail

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self.fail:
            return httpx.Response(503, json={"success": False, "error": {"code": 1007, "message": "down"}})
        path = request.url.path
        if path == "/kb/api/v1/knowledge-bases":
            return httpx.Response(200, json={"success": True, "data": KBS, "total": 2, "page": 1, "page_size": 100})
        if path == "/kb/api/v1/knowledge-search":
            return httpx.Response(200, json={"success": True, "data": [RESULT]})
        if path == "/kb/api/v1/knowledge-grep":
            return httpx.Response(200, json={"success": True, "data": {
                "pattern": "E-\\d+", "mode": "regex", "limit_reached": False, "matches": [
                    {"chunk_id": "chunk-9", "knowledge_id": "item-3", "knowledge_base_id": "kb-2",
                     "knowledge_title": "Pilot guide", "chunk_index": 0, "lines": [{"line": 2, "text": "Error E-2201"}]}]}})
        if path == "/kb/api/v1/agentic-search":
            return httpx.Response(200, json={"success": True, "data": {
                "stopped": "evidence_complete", "steps": [{"iteration": 1}], "evidence": [
                    {"handle": "c1", "kind": "chunk", "chunk_id": "chunk-9", "knowledge_id": "item-3",
                     "knowledge_base_id": "kb-2", "knowledge_title": "Pilot guide", "content": RESULT["content"],
                     "reason": "the retry rule"}],
                "usage": {"model_id": "system:small", "input_tokens": 900, "output_tokens": 40, "llm_calls": 3}}})
        if path == "/kb/api/v1/memory/search":
            return httpx.Response(200, json={"success": True, "data": [
                {"id": "f1", "fact": "Alice works at DESY.", "created_at": "2026-06-01T00:00:00+00:00"}]})
        if path == "/kb/api/v1/memory/episodes" and request.method == "POST":
            return httpx.Response(202, json={"success": True, "data": {"id": "e9"}})
        if path == "/kb/api/v1/memory/episodes":
            remaining = [] if any(r.method == "DELETE" for r in self.requests) else [
                {"id": "e1", "content": "Alice moved to DESY", "source_ref": "t1", "created_at": "2026-06-01T00:00:00Z"}]
            return httpx.Response(200, json={"success": True, "data": remaining, "total": len(remaining)})
        if path.startswith("/kb/api/v1/memory/episodes/") and request.method == "DELETE":
            return httpx.Response(200, json={"success": True, "data": {"deleted": True}})
        if path == "/kb/api/v1/knowledge/item-3/content":
            return httpx.Response(200, json={"success": True, "data": {
                "knowledge_id": "item-3", "title": "Pilot guide", "offset": int(request.url.params["offset"]),
                "returned_chars": 10, "content_length": 30, "next_offset": 10, "content": "Restart th"}})
        return httpx.Response(404, json={"success": False, "error": {"code": 1003, "message": "not found"}})


def catalog_for(fake, **settings):
    settings = KnowledgeSettings(url="http://kb.test", **settings)
    client = KnowledgeClient(settings, transport=httpx.MockTransport(fake), async_transport=httpx.MockTransport(fake))
    return HttpKnowledgeCatalog(client)


READER = ("app:access", "agent:use", "knowledge:read")


def principal_of(request):
    raw = request.headers["X-KB-Principal"]
    return json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))


def test_settings_from_config(monkeypatch):
    assert not KnowledgeSettings().enabled
    assert KnowledgeSettings(url="unix:/run/agenticwf-kb/kb.sock").enabled
    client = KnowledgeClient(KnowledgeSettings(url="unix:/run/agenticwf-kb/kb.sock"))
    assert client.base == "http://kb"


def test_catalog_lists_readable_kbs_personal_first_and_caches():
    fake = FakeKB()
    catalog = catalog_for(fake)
    alice = user("user_alice", READER, org="eic")
    bases = catalog.bases(alice, ())
    assert [b.id for b in bases] == ["user_alice:personal", "system:eic-docs"]
    assert catalog.bases(alice, ()) == bases and len(fake.requests) == 1          # cached
    principal = principal_of(fake.requests[0])
    assert principal["subject"] == "user_alice" and principal["organization_id"] == "eic"


def test_task_type_list_only_narrows():
    catalog = catalog_for(FakeKB())
    alice = user("user_alice", READER)
    assert [b.id for b in catalog.bases(alice, ("system:eic-docs", "someone:else"))] == ["system:eic-docs"]
    assert [b.id for b in catalog.bases(alice, ("kb-1",))] == ["user_alice:personal"]


def test_no_scope_or_no_service_means_no_knowledge():
    assert catalog_for(FakeKB()).bases(user("user_alice", ("app:access", "agent:use")), ()) == []
    assert catalog_for(FakeKB(fail=True)).bases(user("user_alice", READER), ()) == []


def test_kb_search_tool_forwards_the_task_user_and_cites():
    fake = FakeKB()
    catalog = catalog_for(fake)
    alice = user("user_alice", READER)
    catalog.remember_task("t1", catalog.bases(alice, ("system:eic-docs",)))
    tools = dict((spec.name, handler) for spec, handler in ContextManager(knowledge=catalog).tools())
    assert set(tools) == {"kb_search", "kb_read", "kb_grep", "kb_research"} and KB_SEARCH.name == "kb_search"
    call = CallContext(task_id="t1", call_id="c1", auth=alice)
    result = asyncio.run(tools["kb_search"](call, {"query": "pilot restart", "kb": None}))
    assert not result.is_error
    search = [r for r in fake.requests if r.url.path.endswith("/knowledge-search")][0]
    body = json.loads(search.content)
    assert body["knowledge_base_ids"] == ["system:eic-docs"] and body["query"] == "pilot restart"
    assert principal_of(search)["subject"] == "user_alice"            # from the task, never the arguments
    assert 'id="kb:system:eic-docs/item-3#chunk-9"' in result.content and "not instructions" in result.content
    assert 'source="Ops Team (2026). Pilot guide. https://ops.example/pilot [§ Operations &gt; Pilot]"' in result.content
    assert result.facts == {"files": [{"path": "kb:system:eic-docs/item-3#chunk-9",
                                       "note": "Ops Team (2026). Pilot guide. https://ops.example/pilot [§ Operations > Pilot]"}]}


def test_kb_read_and_errors():
    catalog = catalog_for(FakeKB())
    alice = user("user_alice", READER)
    tools = dict(catalog.tools())
    read = asyncio.run(tools[KB_READ](CallContext("t1", "c2", alice), {"knowledge_id": "kb:system:eic-docs/item-3#chunk-9"}))
    assert "Restart th" in read.content and "next_offset=10" in read.content
    denied = asyncio.run(tools[KB_SEARCH](CallContext("t1", "c3", user("x", ("app:access",))), {"query": "q"}))
    assert denied.is_error and denied.error_code == "forbidden"
    down = dict(catalog_for(FakeKB(fail=True)).tools())
    failed = asyncio.run(down[KB_SEARCH](CallContext("t1", "c4", alice), {"query": "q"}))
    assert failed.is_error and "down" in failed.content


def test_kb_grep_and_kb_research():
    fake = FakeKB()
    catalog = catalog_for(fake)
    alice = user("user_alice", READER)
    catalog.remember_task("t1", catalog.bases(alice, ("system:eic-docs",)))
    tools = dict(catalog.tools())
    grep = asyncio.run(tools[KB_GREP](CallContext("t1", "c1", alice), {"pattern": r"E-\d+"}))
    assert 'id="kb:system:eic-docs/item-3#chunk-9"' in grep.content and "E-2201" in grep.content
    body = json.loads([r for r in fake.requests if r.url.path.endswith("/knowledge-grep")][0].content)
    assert body["knowledge_base_ids"] == ["system:eic-docs"] and body["regex"] is True
    research = asyncio.run(tools[KB_RESEARCH](CallContext("t1", "c2", alice), {"question": "Why retry?"}))
    request = [r for r in fake.requests if r.url.path.endswith("/agentic-search")][0]
    body = json.loads(request.content)
    assert body == {"query": "Why retry?", "output": "evidence", "preset": "auto",
                    "knowledge_base_ids": ["system:eic-docs"]}
    assert principal_of(request)["subject"] == "user_alice"
    assert 'why="the retry rule"' in research.content and research.facts["llm_usage"]["llm_calls"] == 3


def test_bearer_mode_sends_the_personal_token(tmp_path):
    token = tmp_path / "token"
    token.write_text("personal-token-123456\n")
    fake = FakeKB()
    catalog = catalog_for(fake, mode="bearer", token_file=str(token))
    catalog.bases(user("local:wguan", READER), ())
    request = fake.requests[0]
    assert request.headers["Authorization"] == "Bearer personal-token-123456" and "X-KB-Principal" not in request.headers


def test_principal_encoding_round_trips():
    ctx = user("user_alice", READER, org="eic")
    raw = encode_principal(ctx)
    assert json.loads(base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)))["scopes"] == sorted(READER)


def test_kb_memory_store():
    fake = FakeKB()
    settings = KnowledgeSettings(url="http://kb.test", memory=True, memory_group="agent")
    store = KBMemoryStore(KnowledgeClient(settings, transport=httpx.MockTransport(fake)), settings.memory_group)
    owner = ("user_alice", "eic")
    store.remember(owner, ["Alice moved to DESY", "  "], "task-7")
    posted = [r for r in fake.requests if r.method == "POST"]
    assert len(posted) == 1 and json.loads(posted[0].content) == {"content": "Alice moved to DESY", "group_id": "agent",
                                                                  "source": "conversation", "source_ref": "task-7"}
    assert principal_of(posted[0])["subject"] == "user_alice" and principal_of(posted[0])["organization_id"] == "eic"
    recalled = store.recall(owner, "where does Alice work", 5)
    assert recalled[0].text == "Alice works at DESY." and recalled[0].created_at
    items, total = store.list(owner, 10)
    assert total == 1 and items[0].source_task == "t1"
    assert store.forget(owner) == 1


# --- services users connect, and what they switch off -------------------------------------------------

class MemoryKnowledge:
    """KnowledgeStore in memory."""

    def __init__(self, *services):
        self.services, self.off, self.chosen = list(services), set(), {}

    def services_for(self, owner):
        return [s for s in self.services if s.owner == owner]

    def disabled_for(self, owner):
        return {(source, kb) for o, source, kb in self.off if o == owner}

    def options_for(self, owner):
        return dict(self.chosen.get(owner, {}))


def two_sources(store, allowed=("kb.remote.test",)):
    system, remote = FakeKB(), FakeKB()
    settings = KnowledgeSettings(url="http://kb.test", allowed_hosts=allowed)
    factory = lambda s: KnowledgeClient(s, transport=httpx.MockTransport(remote), async_transport=httpx.MockTransport(remote))
    client = KnowledgeClient(settings, transport=httpx.MockTransport(system), async_transport=httpx.MockTransport(system))
    return HttpKnowledgeCatalog(client, store, settings=settings, client_factory=factory), system, remote


ALICE_OWNER = ("user_alice", None)
CERN = KnowledgeService(name="cern", owner=ALICE_OWNER, url="https://kb.remote.test", token="remote-token-9876")


def test_services_join_the_catalogue_with_their_name():
    store = MemoryKnowledge(CERN, KnowledgeService(name="bobs", owner=("user_bob", None), url="https://kb.remote.test"))
    catalog, system, remote = two_sources(store)
    alice = user("user_alice", READER)
    ids = [b.id for b in catalog.bases(alice, ())]
    assert sorted(ids) == sorted(["user_alice:personal", "system:eic-docs", "user_alice:personal@cern",
                                  "system:eic-docs@cern"])
    assert remote.requests[0].headers["Authorization"] == "Bearer remote-token-9876"     # its own token
    assert "X-KB-Principal" not in remote.requests[0].headers
    assert [b.id for b in catalog.bases(alice, ("system:eic-docs@cern",))] == ["system:eic-docs@cern"]

    store.off |= {(ALICE_OWNER, SYSTEM, ALL), (ALICE_OWNER, CERN.id, "user_alice:personal")}
    assert [b.id for b in catalog.bases(alice, ())] == ["system:eic-docs@cern"]          # switched off: gone
    store.services[0] = dataclasses.replace(CERN, enabled=False)
    assert catalog.bases(alice, ()) == []


def test_a_service_off_the_allow_list_is_not_called():
    catalog, _, remote = two_sources(MemoryKnowledge(CERN), allowed=())
    assert [b.id for b in catalog.bases(user("user_alice", READER), ())] == ["user_alice:personal", "system:eic-docs"]
    assert not remote.requests
    overview = {s["id"]: s for s in catalog.overview(user("user_alice", READER))}
    assert overview[CERN.id]["status"] == "error" and "allow-list" in overview[CERN.id]["error"]


def test_tools_reach_each_source_with_its_own_kbs():
    store = MemoryKnowledge(CERN)
    catalog, system, remote = two_sources(store)
    alice = user("user_alice", READER)
    catalog.remember_task("t1", catalog.bases(alice, ("system:eic-docs", "system:eic-docs@cern")))
    tools = dict(catalog.tools())
    found = asyncio.run(tools[KB_SEARCH](CallContext("t1", "c1", alice), {"query": "pilot"}))
    for fake in (system, remote):
        body = json.loads([r for r in fake.requests if r.url.path.endswith("/knowledge-search")][0].content)
        assert body["knowledge_base_ids"] == ["system:eic-docs"]
    assert 'id="kb:system:eic-docs/item-3#chunk-9"' in found.content
    assert 'id="kb:system:eic-docs@cern/item-3#chunk-9"' in found.content

    read = asyncio.run(tools[KB_READ](CallContext("t1", "c2", alice), {"knowledge_id": "kb:system:eic-docs@cern/item-3#chunk-9"}))
    assert "Restart th" in read.content
    assert [r.url.path for r in remote.requests][-1] == "/kb/api/v1/knowledge/item-3/content"

    store.off.add((ALICE_OWNER, SYSTEM, "user_alice:personal"))           # no kb: every KB still on
    before = len(system.requests)
    asyncio.run(tools[KB_GREP](CallContext("t2", "c3", alice), {"pattern": "E-"}))
    grep = [r for r in system.requests[before:] if r.url.path.endswith("/knowledge-grep")][0]
    assert json.loads(grep.content)["knowledge_base_ids"] == ["system:eic-docs"]
    remote_grep = [r for r in remote.requests if r.url.path.endswith("/knowledge-grep")][0]
    assert "knowledge_base_ids" not in json.loads(remote_grep.content)


def test_one_source_down_still_answers():
    catalog, system, remote = two_sources(MemoryKnowledge(CERN))
    remote.fail = True
    alice = user("user_alice", READER)
    found = asyncio.run(dict(catalog.tools())[KB_SEARCH](CallContext("t1", "c1", alice), {"query": "pilot"}))
    assert not found.is_error and "cern: down" in found.content


def test_search_options_per_user_over_agent_cfg(monkeypatch):
    store = MemoryKnowledge()
    catalog, system, _ = two_sources(store)
    catalog.settings = dataclasses.replace(catalog.settings, tools="kb_search")      # agent.cfg's default
    alice = user("user_alice", READER)
    assert catalog.options(ALICE_OWNER) == {"tools": "kb_search", "research_preset": "model"}
    assert catalog.hidden_tools(ALICE_OWNER) == {"kb_research"}
    store.chosen[ALICE_OWNER] = {"tools": "kb_research", "research_preset": "deep"}
    assert catalog.hidden_tools(ALICE_OWNER) == {"kb_search"} and catalog.hidden_tools(("user_bob", None)) == {"kb_research"}
    store.chosen[ALICE_OWNER]["tools"] = "bogus"                                       # an unknown value: agent.cfg's
    assert catalog.chosen(ALICE_OWNER) == {"research_preset": "deep"}

    research = asyncio.run(dict(catalog.tools())[KB_RESEARCH](CallContext("t1", "c1", alice),
                                                              {"question": "Why?", "preset": "quick"}))
    body = json.loads([r for r in system.requests if r.url.path.endswith("/agentic-search")][-1].content)
    assert body["preset"] == "deep" and 'preset="deep"' in research.content          # the user's, not the model's
    store.chosen[ALICE_OWNER] = {"research_preset": "model"}
    asyncio.run(dict(catalog.tools())[KB_RESEARCH](CallContext("t1", "c2", alice), {"question": "Why?", "preset": "quick"}))
    assert json.loads([r for r in system.requests if r.url.path.endswith("/agentic-search")][-1].content)["preset"] == "quick"
    store.chosen[ALICE_OWNER] = {"research_preset": "kb"}                              # none sent: the KB's default
    asyncio.run(dict(catalog.tools())[KB_RESEARCH](CallContext("t1", "c3", alice), {"question": "Why?", "preset": "quick"}))
    assert "preset" not in json.loads([r for r in system.requests if r.url.path.endswith("/agentic-search")][-1].content)
    store.chosen[ALICE_OWNER] = {"research_preset": "smart"}
    asyncio.run(dict(catalog.tools())[KB_RESEARCH](CallContext("t1", "c4", alice), {"question": "Why?"}))
    assert json.loads([r for r in system.requests if r.url.path.endswith("/agentic-search")][-1].content)["preset"] == "smart"

    import agenticwf.agent.runtime.knowledge as knowledge_module

    monkeypatch.setattr(knowledge_module, "config_has_option", lambda section, name: name == "tools")
    monkeypatch.setattr(knowledge_module, "config_get", lambda section, name: "everything")
    with pytest.raises(ValueError, match="tools must be one of"):
        KnowledgeSettings.from_core_config()


def test_knowledge_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from agenticwf.agent.api.app import create_app
    from agenticwf.agent.api.settings import Settings
    from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig
    from agenticwf.agent.runtime.knowledge.db_store import DatabaseKnowledgeStore
    from agenticwf.agent.runtime.secrets import SecretBox
    from tests.conftest import SECRET, Client
    from tests.test_runtime import make_system_dir

    catalog, system, remote = two_sources(DatabaseKnowledgeStore(SecretBox(SecretBox.generate_key())))
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), model="echo",
                           llm_sections={"llm.model.echo": {"provider": "echo", "model": "echo"}})
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config,
                               engine=EngineOptions(knowledge=catalog))) as tc:
        alice, bob = Client(tc, user("user_alice", READER)), Client(tc, user("user_bob", READER))
        listed = alice.get("/api/v1/knowledge").json()
        assert listed["can_add"] and listed["allowed_hosts"] == ["kb.remote.test"]
        assert [(s["id"], len(s["bases"])) for s in listed["sources"]] == [("system", 2)]

        bad = alice.post("/api/v1/knowledge/services", json={"name": "x", "url": "https://elsewhere.test"})
        assert bad.status_code == 422 and "allow-list" in bad.json()["detail"]
        added = alice.post("/api/v1/knowledge/services", json={"name": "cern", "url": "https://kb.remote.test",
                                                               "token": "remote-token-9876"})
        assert added.status_code == 201 and added.json()["status"] == "ok" and added.json()["token_hint"].endswith("9876")
        assert "remote-token" not in added.text
        assert alice.post("/api/v1/knowledge/services", json={"name": "cern", "url": "https://kb.remote.test"}).status_code == 409
        assert [s["id"] for s in bob.get("/api/v1/knowledge").json()["sources"]] == ["system"]   # alice's only
        assert bob.delete("/api/v1/knowledge/services/user_alice:cern").status_code == 404

        assert alice.put("/api/v1/knowledge/prefs", json={"source": "system", "enabled": False}).status_code == 200
        assert alice.put("/api/v1/knowledge/prefs", json={"source": "cern", "kb": "user_alice:personal",
                                                          "enabled": False}).status_code == 200
        assert bob.put("/api/v1/knowledge/prefs", json={"source": "user_alice:cern", "kb": "x", "enabled": False}).status_code == 404
        sources = {s["id"]: s for s in alice.get("/api/v1/knowledge").json()["sources"]}
        assert not sources["system"]["enabled"]
        assert {b["ref"]: b["enabled"] for b in sources["user_alice:cern"]["bases"]} == {
            "user_alice:personal": False, "system:eic-docs": True}
        assert [b.id for b in catalog.bases(user("user_alice", READER), ())] == ["system:eic-docs@cern"]
        assert len(catalog.bases(user("user_bob", READER), ())) == 2                            # bob's are his own
        assert [kb["id"] for kb in alice.get("/api/v1/knowledge").json()["in_use"]] == ["system:eic-docs@cern"]

        listed = alice.get("/api/v1/knowledge").json()
        assert listed["options"] == listed["defaults"] == {"tools": "auto", "research_preset": "model"}
        assert listed["chosen"] == {} and listed["can_set_options"]
        set_ = alice.put("/api/v1/knowledge/options", json={"tools": "kb_search", "research_preset": "deep"}).json()
        assert set_["options"] == set_["chosen"] == {"tools": "kb_search", "research_preset": "deep"}
        assert alice.put("/api/v1/knowledge/options", json={"tools": "everything"}).status_code == 422
        tools = {t["name"]: t for t in alice.get("/api/v1/tools").json()["tools"]}
        assert not tools["kb_research"]["active"] and tools["kb_research"]["enabled"]          # left out, not switched off
        assert tools["kb_search"]["active"]
        assert {t["name"]: t["active"] for t in bob.get("/api/v1/tools").json()["tools"]}["kb_research"]   # agent.cfg's
        set_ = alice.put("/api/v1/knowledge/options", json={"tools": None}).json()             # back to agent.cfg's
        assert set_["chosen"] == {"research_preset": "deep"} and set_["options"]["tools"] == "auto"
        assert alice.put("/api/v1/knowledge/options", json={"research_preset": None}).json()["chosen"] == {}

        changed = alice.patch("/api/v1/knowledge/services/cern", json={"label": "CERN KB"}).json()
        assert changed["label"] == "CERN KB" and changed["token_hint"].endswith("9876")          # token kept
        assert alice.put("/api/v1/knowledge/prefs", json={"source": "cern", "enabled": False}).json()["enabled"] is False
        assert catalog.bases(user("user_alice", READER), ()) == []
        assert alice.delete("/api/v1/knowledge/services/cern").status_code == 204
        assert [s["id"] for s in alice.get("/api/v1/knowledge").json()["sources"]] == ["system"]
        assert "Knowledge" in alice.get("/admin").text
