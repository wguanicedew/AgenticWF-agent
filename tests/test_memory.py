"""Long-term memory: the database store, memory_search, and consolidation as a job."""

import asyncio

import pytest

from agenticwf.agent.runtime.context.db_memory import DatabaseMemoryStore
from agenticwf.agent.runtime.context.manager import ContextManager, MemorySettings
from agenticwf.agent.runtime.context.memory import MEMORY_SEARCH
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.db_store import DatabaseStore
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.harness.store import InMemoryStore
from agenticwf.agent.runtime.llm.base import LLMResponse, ToolCallRequest
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.policy.engine import PolicyEngine
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.types import RuntimeState

from tests.conftest import SECRET, user
from tests.test_recovery import system  # noqa: F401  (fixture)

ALICE, BOB, ALICE_AT_CERN = ("user_alice", None), ("user_bob", None), ("user_alice", "cern")


def build(system, store, script, memory, settings=None):
    registry = ToolRegistry(system_dir=system)
    context_manager = ContextManager(memory, settings=settings)
    for spec, handler in context_manager.tools():
        registry.add_core(spec, handler)
    assert registry.load() == []
    return AgentHarness(HarnessServices(llm=ScriptedLLM(script), tool_dispatcher=ToolDispatcher(registry, PolicyEngine()),
                                      task_types=TaskTypeResolver(system), store=store,
                                      context_manager=context_manager))


def note(text):
    return ToolCallRequest(f"c-{abs(hash(text))}", "update_working_memory", {"add_note": text})


# --- the store ---------------------------------------------------------------------------------

def test_memories_are_per_owner_ranked_and_kept_once():
    memory = DatabaseMemoryStore()
    memory.remember(ALICE, ["The cluster is at CERN", "Use python 3.12 for builds", " "], source_task="t1")
    memory.remember(ALICE, ["The  cluster is at CERN"], source_task="t2")            # the same text, once
    memory.remember(BOB, ["Bob's cluster is elsewhere"], source_task="t3")
    memory.remember(ALICE_AT_CERN, ["Alice at CERN only"], source_task="t4")

    assert [m.text for m in memory.recall(ALICE, "which cluster", 5)] == ["The cluster is at CERN"]
    assert {m.text for m in memory.recall(ALICE, "", 5)} == {"The cluster is at CERN", "Use python 3.12 for builds"}
    assert [m.source_task for m in memory.recall(ALICE, "cluster", 5)] == ["t1"]
    assert [m.text for m in memory.recall(BOB, "cluster", 5)] == ["Bob's cluster is elsewhere"]
    assert [m.text for m in memory.recall(ALICE_AT_CERN, "cluster", 5)] == []     # another organization, another owner
    assert memory.recall(ALICE, "nothing matches this", 5) == [] and memory.recall(ALICE, "cluster", 0) == []


# --- consolidation as a job --------------------------------------------------------------------

def test_consolidation_is_a_job_that_runs_once_and_keeps_no_secrets(system):
    memory = DatabaseMemoryStore()
    h = build(system, InMemoryStore(), [
        LLMResponse("", [note("The build needs python 3.12"), note("api_key = sk-123456")]),
        LLMResponse("done"),
    ], memory, MemorySettings(extract="notes"))
    task = asyncio.run(h.run(h.create_task(user(), "build it").id))
    assert task.state is RuntimeState.COMPLETED and task.consolidation == "pending"
    assert memory.recall(ALICE, "build", 5) == []                     # not inline: a job does it
    job = h.store.claim("w", 30)
    assert (job.kind, job.ref) == ("consolidate", task.id)            # ahead of anything the completion queued

    asyncio.run(h.consolidate(task.id))
    asyncio.run(h.consolidate(task.id))                               # again: nothing more
    assert [m.text for m in memory.recall(ALICE, "build python key", 5)] == ["The build needs python 3.12"]
    assert task.consolidation == "done" and h.store.pending_consolidations() == []


def test_a_restart_queues_pending_consolidations(system):
    memory = DatabaseMemoryStore()
    notes = MemorySettings(extract="notes")
    h = build(system, DatabaseStore(), [LLMResponse("", [note("Remember the staging host")]), LLMResponse("ok")],
              memory, notes)
    task = asyncio.run(h.run(h.create_task(user(), "deploy").id))
    assert task.state is RuntimeState.COMPLETED                       # stopped before the job ran

    again = build(system, DatabaseStore(), [], memory, notes)         # restart: only the tables remain
    assert again.store.pending_consolidations() == [task.id]
    assert again.resume_consolidations() == 1
    job = again.store.claim("w", 30)
    asyncio.run(again.consolidate(job.ref))
    assert [m.text for m in memory.recall(ALICE, "staging", 5)] == ["Remember the staging host"]
    assert DatabaseStore().pending_consolidations() == []


def test_failed_tasks_are_consolidated_only_when_configured(system):
    memory = DatabaseMemoryStore()
    for consolidate_failed, expected in ((False, []), (True, ["half-done thought"])):
        h = build(system, InMemoryStore(), [LLMResponse("", [note("half-done thought")])], memory,   # then no model
                  MemorySettings(consolidate_failed=consolidate_failed, extract="notes"))
        task = asyncio.run(h.run(h.create_task(user(), "x").id))
        assert task.state is RuntimeState.FAILED
        assert task.consolidation == ("pending" if consolidate_failed else None)
        asyncio.run(h.consolidate(task.id))
        assert [m.text for m in memory.recall(ALICE, "thought", 5)] == expected


def test_no_memory_store_nothing_to_consolidate(system):
    h = build(system, InMemoryStore(), [LLMResponse("done")], None)
    task = asyncio.run(h.run(h.create_task(user(), "x").id))
    assert task.consolidation is None and h.store.claim("w", 30) is None


# --- extraction through the model ----------------------------------------------------------------

def test_the_model_picks_what_lasts(system):
    memory = DatabaseMemoryStore()
    memory.remember(ALICE, ["Alice works on the ATLAS project"], source_task="t0")
    picked = '{"memories": ["Alice prefers conda over pip", "password: hunter2", "  ", 7]}'
    h = build(system, InMemoryStore(), [
        LLMResponse("", [note("she said: use conda, not pip")]),
        LLMResponse("Set up the ATLAS environment with conda."),
        LLMResponse("Here you go: " + picked),                         # the consolidate call
    ], memory)
    task = asyncio.run(h.run(h.create_task(user(), "set up my ATLAS environment").id))
    asyncio.run(h.consolidate(task.id))

    request = h.services.llm.requests[-1]
    assert "long-term memory" in request.system and "at most 5 items" in request.system
    shown = request.messages[0].content
    assert "set up my ATLAS environment" in shown and "<outcome>completed</outcome>" in shown
    assert "use conda, not pip" in shown and "Alice works on the ATLAS project" in shown   # notes; already remembered
    assert {m.text for m in memory.recall(ALICE, "", 10)} == {"Alice works on the ATLAS project", "Alice prefers conda over pip"}


def test_extraction_falls_back_to_the_notes(system):
    memory = DatabaseMemoryStore()
    h = build(system, InMemoryStore(), [
        LLMResponse("", [note("Results go to /eos/user/a")]), LLMResponse("done"), LLMResponse("I would keep the path."),
    ], memory, MemorySettings(max_per_task=3))
    task = asyncio.run(h.run(h.create_task(user(), "x").id))
    asyncio.run(h.consolidate(task.id))                               # the answer does not parse
    assert [m.text for m in memory.recall(ALICE, "results", 5)] == ["Results go to /eos/user/a"]


# --- memory_search -----------------------------------------------------------------------------

def test_memory_search_sees_only_the_callers_memories(system):
    memory = DatabaseMemoryStore()
    memory.remember(ALICE, ["Alice prefers SLURM"], source_task="t1")
    memory.remember(BOB, ["Bob prefers SLURM too"], source_task="t2")
    search = ToolCallRequest("c1", "memory_search", {"query": "SLURM preference"})
    h = build(system, InMemoryStore(), [LLMResponse("", [search]), LLMResponse("ok")], memory)

    task = asyncio.run(h.run(h.create_task(user(), "which scheduler?").id))
    result = task.calls[0].result
    assert not result.is_error and "Alice prefers SLURM" in result.content and "Bob" not in result.content
    assert h.dispatcher.registry.load() == [] and h.dispatcher.registry.lookup("memory_search", ALICE) is not None


def test_no_memory_store_no_memory_search(system):
    h = build(system, InMemoryStore(), [], None)
    assert h.dispatcher.registry.lookup("memory_search", ALICE) is None


def test_the_api_remembers_in_the_database_by_default():
    from agenticwf.agent.api.app import create_app
    from agenticwf.agent.api.settings import Settings

    app = create_app(Settings(context_secret=SECRET))
    harness = app.state.agent.harness
    assert isinstance(harness.services.context_manager.memory, DatabaseMemoryStore)
    assert harness.dispatcher.registry.lookup("memory_search", ALICE) is MEMORY_SEARCH


# --- people list and delete their memories ----------------------------------------------------

def test_people_list_and_delete_their_own_memories(as_user):
    from tests.conftest import user as api_user

    memory = DatabaseMemoryStore()
    alice, bob = api_user("user_alice"), api_user("user_bob")
    memory.remember(("user_alice", None), ["first", "second", "third"], source_task="t1")
    memory.remember(("user_bob", None), ["bob's"], source_task="t2")
    a, b = as_user(alice), as_user(bob)

    page = a.get("/api/v1/memories?limit=2").json()
    assert page["total"] == 3 and page["enabled"] and len(page["memories"]) == 2
    assert {"id", "text", "source_task", "created_at"} <= set(page["memories"][0])
    rest = a.get("/api/v1/memories?limit=2&offset=2").json()["memories"]
    ids = [m["id"] for m in page["memories"] + rest]
    assert len(set(ids)) == 3

    assert b.delete(f"/api/v1/memories/{ids[0]}").status_code == 404     # not Bob's
    assert a.delete(f"/api/v1/memories/{ids[0]}").status_code == 204
    assert a.delete(f"/api/v1/memories/{ids[0]}").status_code == 404
    assert a.delete("/api/v1/memories").json() == {"deleted": 2}
    assert a.get("/api/v1/memories").json()["total"] == 0
    assert [m["text"] for m in b.get("/api/v1/memories").json()["memories"]] == ["bob's"]
    assert 'id="view-memory"' in a.get("/admin").text


def test_each_answer_says_which_memories_it_recalled(as_user):
    from tests.conftest import user as api_user

    DatabaseMemoryStore().remember(("user_alice", None), ["The build uses python 3.12"], source_task="t0")
    alice, bob = as_user(api_user("user_alice")), as_user(api_user("user_bob"))
    sid = alice.post("/api/v1/sessions", json={"title": "t"}).json()["id"]
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "which python does the build use"}).json()["status"] == "succeeded"
    intent = [m["intent"] for m in alice.get(f"/api/v1/sessions/{sid}").json()["messages"] if m.get("intent")][0]

    recalled = alice.get(f"/api/v1/sessions/{sid}/memories").json()["intents"]
    assert [m["text"] for m in recalled[intent]] == ["The build uses python 3.12"] and recalled[intent][0]["id"]
    trace = alice.get(f"/api/v1/sessions/{sid}/trace?intent={intent}").json()["intents"][0]
    assert [m["text"] for m in trace["tasks"][0]["memories"]] == ["The build uses python 3.12"]
    assert bob.get(f"/api/v1/sessions/{sid}/memories").status_code == 404                 # alice's chat
    assert 'id="memories-btn"' in alice.get("/").text


# --- recall by meaning -------------------------------------------------------------------------

class SchedulerEmbedder:
    CONCEPTS = [{"scheduler", "slurm", "batch", "jobs"}, {"python", "conda", "pip", "environment"}]

    def __init__(self, fail=False):
        self.fail, self.calls = fail, 0

    def embed(self, texts, kind):
        self.calls += 1
        if self.fail:
            raise OSError("down")
        return [[float(len(set(t.lower().split()) & c)) for c in self.CONCEPTS] + [0.1] for t in texts]


def recall_store(embedder):
    from agenticwf.agent.runtime.embeddings import EmbeddingIndex, EmbeddingSettings

    memory = DatabaseMemoryStore(embeddings=EmbeddingIndex(embedder, EmbeddingSettings(provider="openai", model="m")))
    memory.remember(ALICE, ["Alice prefers SLURM", "Builds use conda", "The office is in building 40"], source_task="t1")
    return memory


def test_recall_by_meaning_finds_memories_that_share_no_word():
    memory = recall_store(SchedulerEmbedder())
    assert [m.text for m in memory.recall(ALICE, "which scheduler do I use", 5)] == ["Alice prefers SLURM"]
    assert [m.text for m in memory.recall(ALICE, "set up my python environment", 1)] == ["Builds use conda"]
    assert [m.text for m in memory.recall(ALICE, "office building", 5)] == ["The office is in building 40"]   # words still count
    assert memory.recall(BOB, "which scheduler do I use", 5) == []          # never another owner's


def test_recall_falls_back_to_words_without_the_embedding_provider():
    memory = recall_store(SchedulerEmbedder(fail=True))
    assert memory.recall(ALICE, "which scheduler do I use", 5) == []
    assert [m.text for m in memory.recall(ALICE, "slurm", 5)] == ["Alice prefers SLURM"]
    assert len(memory.recall(ALICE, "?!", 5)) == 3                          # no words: the newest


def test_the_engine_gives_memory_the_same_embeddings(system):
    from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager

    memory = DatabaseMemoryStore()
    manager = build_thread_manager(RuntimeConfig(system_dir=system),
                                   EngineOptions(llm=ScriptedLLM([]), memory=memory, embedder=SchedulerEmbedder()))
    assert memory.embeddings is manager.harness.dispatcher.registry.embeddings is not None
