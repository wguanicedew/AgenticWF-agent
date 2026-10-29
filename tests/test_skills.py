"""Skills (src/agenticwf/agent/runtime/skills): sources and visibility, choosing one, running it, the API."""

import asyncio

import pytest

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.context.agent_context import AgentContext
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.llm.base import LLMResponse, ToolCallRequest
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.plugins.manager import PluginManager
from agenticwf.agent.runtime.policy.engine import DecisionKind, PolicyEngine
from agenticwf.agent.runtime.skills.catalog import SkillCatalog
from agenticwf.agent.runtime.skills.db_store import DatabaseSkillStore
from agenticwf.agent.runtime.skills.skill import Skill, SkillError
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import ToolResult, ToolSpec
from agenticwf.agent.runtime.types import RuntimeState

from tests.conftest import user as api_user
from tests.test_runtime import SCOPES, make_system_dir, tool_md

ALICE, BOB, CAROL = ("user_a", "cern"), ("user_b", "cern"), ("user_c", None)


def person(owner, scopes=SCOPES):
    return AuthContext(subject=owner[0], user_id=owner[0], organization_id=owner[1], scopes=scopes)


def skill_md(name, description, body="Follow these steps.", extra=""):
    return f"---\nname: {name}\ndescription: {description}\n{extra}---\n{body}\n"


def write_skill(root, name, description, body="Follow these steps.", extra="", files=None):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(skill_md(name, description, body, extra))
    for path, text in (files or {}).items():
        (directory / path).parent.mkdir(parents=True, exist_ok=True)
        (directory / path).write_text(text)
    return directory


@pytest.fixture
def world(tmp_path):
    """System skills, a system plugin's skill, Alice's sandbox skill, and the database's."""
    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "release-notes", "Write release notes from merged changes.",
                body="1. Read the changes with read_thing.\n2. Write the notes.",
                extra="tools: [read_thing]\n", files={"template.md": "# Release {version}", ".hidden": "x"})
    write_skill(system / "skills", "code-review", "Review code changes for bugs and style.",
                extra="approval: ask\n")
    (system / "skills" / "Bad_Name").mkdir()
    (system / "skills" / "Bad_Name" / "SKILL.md").write_text(skill_md("Bad_Name", "broken"))
    plugins = tmp_path / "plugins"
    write_skill(plugins / "lint" / "skills", "check", "Check style with the linters.")
    sandboxes = tmp_path / "sandboxes"
    write_skill(sandboxes / "user_a" / "skills", "my-notes", "Alice's way of taking meeting notes.")
    sandbox = lambda owner: sandboxes / owner[0]   # noqa: E731
    plugin_manager = PluginManager([plugins], user_sandbox=sandbox)
    store = DatabaseSkillStore()
    store.save(Skill(name="triage", description="Triage incoming bug reports.", instructions="Label them.",
                     source="database", owner=ALICE, visibility="organization"))
    store.save(Skill(name="secret-sauce", description="Bob's private recipe.", instructions="…",
                     source="database", owner=BOB))
    catalog = SkillCatalog(system, user_sandbox=sandbox, plugins=plugin_manager, store=store)
    return system, plugin_manager, catalog


def build(system, catalog, plugin_manager, script, **dispatcher_options):
    registry = ToolRegistry(system_dir=system, plugins=plugin_manager, skills=catalog)
    assert registry.load() != []                                     # the broken skill is reported
    for name in ("read_thing", "write_thing", "delete_thing", "slow_thing"):
        registry.register_handler(f"test.{name}", lambda ctx, args, n=name: ToolResult(content=f"{n} done"))
    return AgentHarness(HarnessServices(llm=ScriptedLLM(script), task_types=TaskTypeResolver(system),
                                      tool_dispatcher=ToolDispatcher(registry, PolicyEngine(), **dispatcher_options),
                                      skills=catalog))


def run(harness, task):
    return asyncio.run(harness.run(task.id))


# --- sources and visibility --------------------------------------------------------------------

def test_skills_from_every_source_and_who_sees_them(world):
    _, _, catalog = world
    ids = lambda owner: {k.id for k in catalog.skills(owner)}   # noqa: E731
    system = {"system:release-notes", "system:code-review", "system:lint.check"}
    assert ids(ALICE) == system | {"user_a@cern:my-notes", "user_a@cern:triage"}
    assert ids(BOB) == system | {"user_a@cern:triage", "user_b@cern:secret-sauce"}   # Alice shares with her organization
    assert ids(CAROL) == system
    assert any("Bad_Name" in e for e in catalog.errors)

    names = {k.tool_name for k in catalog.skills(ALICE)}
    assert {"skill__release_notes", "skill__lint_check", "skill__my_notes", "skill__triage"} <= names
    notes = catalog.get("system:release-notes", ALICE)
    assert notes.tools == ("read_thing",) and notes.files() == ["template.md"]
    with pytest.raises(SkillError):
        notes.read_file("../code-review/SKILL.md")
    with pytest.raises(SkillError):
        notes.read_file("SKILL.md")
    assert notes.read_file("template.md") == "# Release {version}"


def test_a_skill_written_by_a_user_cannot_grant_itself_anything(tmp_path):
    sandboxes = tmp_path / "s"
    write_skill(sandboxes / "user_a" / "skills", "sneaky", "Looks harmless.",
                extra="approval: auto\naccess: {users: [user_b]}\n")
    catalog = SkillCatalog(None, user_sandbox=lambda owner: sandboxes / owner[0])
    sneaky = catalog.skills(ALICE)[0]
    assert sneaky.access.users is None and sneaky.visible_to(ALICE) and not sneaky.visible_to(BOB)


# --- the policy --------------------------------------------------------------------------------

def test_own_skills_load_freely_others_ask(world):
    system, plugin_manager, catalog = world
    h = build(system, catalog, plugin_manager, [])
    bob_task = h.create_task(person(BOB), "triage the bugs")
    specs = {s.name: s for s in h.dispatcher.eligible(bob_task).values()}
    decide = lambda name, task: h.policy.decide(specs[name], {}, task).kind   # noqa: E731
    assert decide("skill__triage", bob_task) is DecisionKind.ASK              # Alice's, shared with Bob
    assert decide("skill__secret_sauce", bob_task) is DecisionKind.ALLOW      # Bob's own
    assert decide("skill__release_notes", bob_task) is DecisionKind.ALLOW     # system
    assert decide("skill__code_review", bob_task) is DecisionKind.ASK         # system, approval: ask
    assert specs["skill__triage"].effective_effects.value == "none"           # a skill acts on nothing


# --- choosing and running a skill --------------------------------------------------------------

def test_the_model_picks_a_skill_then_only_its_tools_are_offered(world):
    system, plugin_manager, catalog = world
    h = build(system, catalog, plugin_manager, [
        LLMResponse("", [ToolCallRequest("c1", "skill__release_notes", {"purpose": "notes for 2.0"})]),
        LLMResponse("", [ToolCallRequest("c2", "read_skill_file", {"path": "template.md"}),
                         ToolCallRequest("c3", "write_thing", {"value": "x"})]),
        LLMResponse("", [ToolCallRequest("c4", "read_skill_file", {"path": "../code-review/SKILL.md"})]),
        LLMResponse("Release notes written."),
    ])
    task = run(h, h.create_task(person(ALICE), "write the release notes for 2.0"))
    assert task.state is RuntimeState.COMPLETED

    first, second = h.services.llm.requests[0], h.services.llm.requests[1]
    offered = lambda request: {t["name"] for t in request.tools}   # noqa: E731
    assert {"skill__release_notes", "skill__code_review", "write_thing"} <= offered(first)
    assert "read_skill_file" not in offered(first)
    # Once the skill runs: no skill, and only its tools besides the control tools.
    assert not any(n.startswith("skill__") for n in offered(second))
    assert {"read_thing", "read_skill_file", "update_plan"} <= offered(second) and "write_thing" not in offered(second)

    loaded = next(m for m in second.messages if m.role == "tool")
    assert loaded.content.startswith('<skill name="release-notes">') and "untrusted" not in loaded.content
    assert "template.md" in loaded.content and ".hidden" not in loaded.content
    results = {c.id: c.result for c in task.calls}
    assert results["c2"].content == "# Release {version}"
    assert results["c3"].error_code == "unknown_tool"                       # outside the skill's tools
    assert results["c4"].error_code == "skill_file"
    assert task.active_skill["id"] == "system:release-notes"


def test_one_skill_at_a_time(world):
    system, plugin_manager, catalog = world
    h = build(system, catalog, plugin_manager, [
        LLMResponse("", [ToolCallRequest("c1", "skill__release_notes", {}), ToolCallRequest("c2", "skill__my_notes", {})]),
        LLMResponse("done"),
    ])
    task = run(h, h.create_task(person(ALICE), "notes"))
    results = {c.id: c.result for c in task.calls}
    assert not results["c1"].is_error and results["c2"].error_code == "one_skill_at_a_time"


def test_a_skill_without_tools_keeps_the_task_tools_matched_to_it(world):
    system, plugin_manager, catalog = world
    h = build(system, catalog, plugin_manager, [LLMResponse("", [ToolCallRequest("c1", "skill__my_notes", {})]),
                                                LLMResponse("ok")])
    task = run(h, h.create_task(person(ALICE), "take notes"))
    offered = {t["name"] for t in h.services.llm.requests[1].tools}
    assert "write_thing" in offered and not any(n.startswith("skill__") for n in offered)
    assert "read_skill_file" not in offered                                 # it has no files
    assert task.state is RuntimeState.COMPLETED


def test_the_running_skill_survives_compaction():
    ctx = AgentContext(task_id="t", instructions=None, max_steps=10)
    ctx.skill = '<skill name="x">\nAlways sign the notes.\n</skill>'
    ctx.messages.extend([])
    ctx.append_user("a")
    ctx.append_user("b")
    ctx.replace_with_summary(0, 1, "earlier")
    assert "Always sign the notes." in ctx.messages[0].content


# --- choosing among many -------------------------------------------------------------------------

def test_selection_offers_the_closest_skills_and_tools(tmp_path):
    system = make_system_dir(tmp_path)
    topics = ["invoice", "calendar", "weather", "kubernetes", "postgres", "slack", "pdf", "spreadsheet",
              "translation", "geometry", "telescope", "payroll"]
    for topic in topics:
        for n in range(3):
            (system / "tools" / f"{topic}_op{n}.md").write_text(
                tool_md(f"{topic}_op{n}", scopes="[]").replace(f"The {topic} op{n} tool.", f"Work with {topic} data, operation {n}."))
        write_skill(system / "skills", f"{topic}-guide", f"How to handle {topic} requests well.")
    catalog = SkillCatalog(system)
    registry = ToolRegistry(system_dir=system, skills=catalog)
    registry.load()
    dispatcher = ToolDispatcher(registry, PolicyEngine(), top_k=4, top_k_skills=2)
    h = AgentHarness(HarnessServices(llm=ScriptedLLM([]), task_types=TaskTypeResolver(system), tool_dispatcher=dispatcher,
                                   skills=catalog))
    task = h.create_task(person(ALICE), "Restart the kubernetes deployment that talks to postgres")
    active = set(task.active_tools)
    skills = {n for n in active if n.startswith("skill__")}
    assert skills == {"skill__kubernetes_guide", "skill__postgres_guide"}
    tools = {n for n in active if "_op" in n}
    assert len(tools) == 4 and all(n.startswith(("kubernetes_", "postgres_")) for n in tools)
    assert len(dispatcher.eligible(task)) > 50                            # the rest stay findable with search_tools
    found = dispatcher.discover(task, "telescope")
    assert {f["kind"] for f in found} == {"skill", "tool"} and all("telescope" in f["name"] for f in found)


def test_bm25_prefers_rare_words():
    registry = ToolRegistry()
    common = [ToolSpec(name=f"data_{n}", summary="Read data.", description="data data data", input_schema={},
                       handler=None, source=None) for n in range(5)]
    rare = ToolSpec(name="quasar_lookup", summary="Look up a quasar.", description="data", input_schema={},
                    handler=None, source=None)
    assert registry.search("quasar data", [*common, rare], 1) == [rare]
    assert registry.search("the and of", [*common, rare], 3) == []


# --- the API -------------------------------------------------------------------------------------

def test_people_add_share_and_remove_skills(as_user):
    alice, bob = api_user("user_alice", org="cern"), api_user("user_bob", org="cern")
    a, b = as_user(alice), as_user(bob)
    body = {"name": "weekly-report", "description": "Write the weekly report.", "instructions": "Use bullet points.",
            "tools": ["read_*"], "visibility": "organization"}
    created = a.post("/api/v1/skills", json=body)
    assert created.status_code == 201
    skill = created.json()
    assert skill["id"] == "user_alice@cern:weekly-report" and skill["tool"] == "skill__weekly_report" and skill["editable"]
    assert a.post("/api/v1/skills", json=body).status_code == 409
    assert a.post("/api/v1/skills", json={**body, "name": "x", "visibility": "all"}).status_code == 403   # managers only
    assert a.post("/api/v1/skills", json={**body, "name": "Bad Name"}).status_code == 422

    seen = {k["id"]: k for k in b.get("/api/v1/skills").json()["skills"]}
    assert seen[skill["id"]]["group"] == "shared" and not seen[skill["id"]]["editable"]
    assert b.get(f"/api/v1/skills/{skill['id']}").json()["instructions"] == "Use bullet points."
    assert b.patch(f"/api/v1/skills/{skill['id']}", json={"enabled": False}).status_code == 404   # not Bob's
    assert b.delete(f"/api/v1/skills/{skill['id']}").status_code == 404

    changed = a.patch("/api/v1/skills/weekly-report", json={"visibility": "user", "description": "Weekly."}).json()
    assert changed["visibility"] == "user" and changed["description"] == "Weekly."
    assert skill["id"] not in {k["id"] for k in b.get("/api/v1/skills").json()["skills"]}
    assert a.delete("/api/v1/skills/weekly-report").status_code == 204
    assert a.get("/api/v1/skills/weekly-report").status_code == 404
    assert 'id="view-skills"' in a.get("/admin").text


# --- matching by meaning (embeddings) ------------------------------------------------------------

class ConceptEmbedder:
    """A stand-in model: each concept is one dimension; a text's vector counts its concept words."""

    CONCEPTS = [{"changelog", "release", "version", "pull", "merged"}, {"kubernetes", "pod", "deployment", "cluster"},
                {"weather", "forecast", "rain"}]

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def embed(self, texts, kind):
        self.calls.append((kind, len(texts)))
        if self.fail:
            raise OSError("connection refused")
        words = lambda t: set(t.lower().replace("_", " ").replace("-", " ").replace(".", " ").split())   # noqa: E731
        return [[float(len(words(t) & c)) for c in self.CONCEPTS] + [0.1] for t in texts]


def meaning_registry(tmp_path, embedder):
    from agenticwf.agent.runtime.embeddings import EmbeddingIndex, EmbeddingSettings

    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "release-notes", "Summarize merged pull requests for a version.")
    write_skill(system / "skills", "pod-doctor", "Fix a broken kubernetes deployment.")
    write_skill(system / "skills", "forecast", "Tell the weather forecast.")
    catalog = SkillCatalog(system)
    registry = ToolRegistry(system_dir=system, skills=catalog,
                            embeddings=EmbeddingIndex(embedder, EmbeddingSettings(provider="openai", model="m")))
    registry.load()
    return registry, [s for s in registry.catalogue(ALICE) if s.name.startswith("skill__")]


def test_matching_by_meaning_finds_what_shares_no_word(tmp_path):
    embedder = ConceptEmbedder()
    registry, skills = meaning_registry(tmp_path, embedder)
    query = "prepare the changelog"                                         # no word in common with release-notes
    assert ToolRegistry().search(query, skills, 3) == []                    # words alone find nothing
    assert [s.name for s in registry.search(query, skills, 3)] == ["skill__release_notes"]
    assert [s.name for s in registry.search("my cluster pod crashes", skills, 1)] == ["skill__pod_doctor"]
    assert [s.name for s in registry.search("forecast for tomorrow", skills, 1)] == ["skill__forecast"]   # words and meaning
    documents = sum(n for kind, n in embedder.calls if kind == "document")
    registry.search("prepare the changelog", skills, 3)
    assert sum(n for kind, n in embedder.calls if kind == "document") == documents   # vectors and queries are cached


def test_without_the_embedding_provider_matching_falls_back_to_words(tmp_path):
    embedder = ConceptEmbedder(fail=True)
    registry, skills = meaning_registry(tmp_path, embedder)
    assert registry.search("prepare the changelog", skills, 3) == []
    assert [s.name for s in registry.search("weather forecast", skills, 1)] == ["skill__forecast"]
    calls = len(embedder.calls)
    registry.search("kubernetes", skills, 1)
    assert len(embedder.calls) == calls                                     # skipped for a while after a failure


def test_the_embedding_endpoint_request():
    import http.server
    import json
    import threading

    from agenticwf.agent.runtime.embeddings import EmbeddingSettings, HttpEmbedder

    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            seen["path"], seen["auth"] = self.path, self.headers.get("Authorization")
            seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            data = [{"index": i, "embedding": [float(i), 1.0]} for i in reversed(range(len(seen["body"]["input"])))]
            payload = json.dumps({"data": data}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        import os
        os.environ["TEST_EMBED_KEY"] = "k-123"
        embedder = HttpEmbedder(EmbeddingSettings(provider="voyage", model="voyage-3.5", api_key_env="TEST_EMBED_KEY",
                                                  base_url=f"http://127.0.0.1:{server.server_port}/v1"))
        assert embedder.embed(["a", "b"], "document") == [[0.0, 1.0], [1.0, 1.0]]    # in input order
        assert seen["path"] == "/v1/embeddings" and seen["auth"] == "Bearer k-123"
        assert seen["body"] == {"model": "voyage-3.5", "input": ["a", "b"], "input_type": "document"}
    finally:
        server.shutdown()
    with pytest.raises(ValueError):
        EmbeddingSettings(provider="nope", model="m").check()
    with pytest.raises(ValueError):
        EmbeddingSettings(provider="vllm", model="m").check()               # needs base_url


def test_the_engine_matches_by_meaning_when_given_an_embedder(tmp_path):
    from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager

    manager = build_thread_manager(RuntimeConfig(system_dir=make_system_dir(tmp_path)),
                                   EngineOptions(llm=ScriptedLLM([]), embedder=ConceptEmbedder()))
    assert manager.harness.dispatcher.registry.embeddings is not None


# --- files of database skills --------------------------------------------------------------------

def test_skills_added_on_the_page_keep_their_files_in_the_database(as_user, tmp_path):
    from agenticwf.agent.core.orm.base.session import get_engine
    from sqlalchemy import text

    alice, bob = api_user("user_alice", org="cern"), api_user("user_bob", org="cern")
    a, b = as_user(alice), as_user(bob)
    created = a.post("/api/v1/skills", json={
        "name": "weekly-report", "description": "Write the weekly report.", "instructions": "Fill template.md.",
        "visibility": "organization", "files": {"template.md": "# Week {n}", "docs/style.md": "Short sentences."}})
    assert created.status_code == 201 and created.json()["files"] == ["docs/style.md", "template.md"]
    sid = created.json()["id"]
    for bad in ("../x.md", ".hidden", "SKILL.md", "/abs.md", "a//b"):
        r = a.post("/api/v1/skills", json={"name": "other", "description": "d", "files": {bad: "x"}})
        assert r.status_code == 422, bad
    assert a.get("/api/v1/skills/other").status_code == 404                # nothing stored for a refused skill

    assert a.get(f"/api/v1/skills/{sid}/files/docs/style.md").json()["content"] == "Short sentences."
    assert b.get(f"/api/v1/skills/{sid}/files/template.md").json()["content"] == "# Week {n}"   # shared with Bob
    assert b.put(f"/api/v1/skills/{sid}/files/x.md", json={"content": "x"}).status_code == 404  # not his
    assert a.put("/api/v1/skills/weekly-report/files/x.md", json={"content": "x" * (300 * 1024)}).status_code == 422
    assert a.put("/api/v1/skills/weekly-report/files/notes.md", json={"content": "new"}).json()["files"] == [
        "docs/style.md", "notes.md", "template.md"]
    assert a.delete("/api/v1/skills/weekly-report/files/notes.md").status_code == 204
    assert a.get(f"/api/v1/skills/{sid}/files/notes.md").status_code == 404

    with get_engine().connect() as connection:
        count = lambda: connection.execute(text("SELECT COUNT(*) FROM agent_skill_files")).scalar()   # noqa: E731
        assert count() == 2
        assert a.delete("/api/v1/skills/weekly-report").status_code == 204
        assert count() == 0                                                # its files go with it


def test_a_running_database_skill_reads_its_files_from_the_database(tmp_path):
    store = DatabaseSkillStore()
    skill = Skill(name="weekly", description="Weekly report.", instructions="Use template.md.", source="database",
                  owner=ALICE)
    catalog = SkillCatalog(None, store=store)
    catalog.put(skill, by=ALICE)
    catalog.put_file(skill.id, "template.md", "# Week", by=ALICE)
    with pytest.raises(KeyError):
        catalog.put_file(skill.id, "x.md", "x", by=BOB)
    loaded = catalog.get(skill.id, ALICE)
    assert "Files of this skill (read one with read_skill_file): template.md" in loaded.activation().content
    assert catalog.read_file(skill.id, "template.md", ALICE) == "# Week"
    with pytest.raises(SkillError):
        catalog.read_file(skill.id, "missing.md", ALICE)
    with pytest.raises(SkillError):
        catalog.read_file(skill.id, "template.md", BOB)                   # not shared with Bob


def test_the_event_loop_never_waits_for_embeddings():
    import time as clock

    from agenticwf.agent.runtime.embeddings import EmbeddingIndex, EmbeddingSettings

    class Slow(ConceptEmbedder):
        def embed(self, texts, kind):
            clock.sleep(0.3)
            return super().embed(texts, kind)

    index = EmbeddingIndex(Slow(), EmbeddingSettings(provider="openai", model="m"))
    texts = ["release notes for a version", "fix a kubernetes pod"]

    async def main():
        started = clock.monotonic()
        assert index.similarities("the changelog", texts) is None            # on the loop: no wait
        assert clock.monotonic() - started < 0.1
        for _ in range(100):                                                  # fetched in the background
            await asyncio.sleep(0.02)
            found = index.similarities("the changelog", texts)
            if found is not None:
                return found
        raise AssertionError("never fetched")

    found = asyncio.run(main())
    assert found[0] > found[1]
    assert index.similarities("a new query", texts) is not None              # off the loop: waits


def test_intake_calls_the_embedding_provider_off_the_event_loop(tmp_path):
    import threading

    from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager
    from agenticwf.agent.runtime.records import Intent

    class Watching(ConceptEmbedder):
        def embed(self, texts, kind):
            self.threads = getattr(self, "threads", set()) | {threading.current_thread().name}
            return super().embed(texts, kind)

    embedder = Watching()
    system = make_system_dir(tmp_path)
    write_skill(system / "skills", "release-notes", "Summarize merged pull requests for a version.")
    manager = build_thread_manager(RuntimeConfig(system_dir=system),
                                   EngineOptions(llm=ScriptedLLM([]), embedder=embedder))
    harness = manager.harness

    async def main():
        intent = Intent(thread_id="t", owner=ALICE, auth=person(ALICE), text="prepare the changelog")
        harness.store.save_intent(intent)
        await harness.intake(intent.id)
        return threading.current_thread().name

    loop_thread = asyncio.run(main())
    assert embedder.threads and loop_thread not in embedder.threads
