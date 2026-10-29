"""RuntimeThreadManager → RuntimeThread → Session → AgentHarness (src/agenticwf/agent/runtime/thread, session, bootstrap)."""

import asyncio

import pytest

from agenticwf.auth.context import AuthContext

from agenticwf.agent.runtime.llm.base import LLM, LLMResponse
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.thread.manager import StartThreadOptions, TruncateBeforeNthUserMessage
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager

from tests.test_runtime import SCOPES, call, make_system_dir


def user(subject="user_a", machine=False):
    return AuthContext(subject=subject, user_id=None if machine else subject,
                       client_id=subject if machine else None, scopes=SCOPES)


class GatedLLM(LLM):
    """Scripted, but blocks on `gate` before answering, so tests can act while a task runs."""

    def __init__(self, steps):
        self.inner = ScriptedLLM(steps)
        self.gate = asyncio.Event()
        self.started = asyncio.Event()

    async def complete(self, request):
        self.started.set()
        await self.gate.wait()
        return await self.inner.complete(request)


def manager_for(tmp_path, llm):
    handlers = {
        "test.read_thing": lambda ctx, args: ToolResult(content="thing"),
        "test.delete_thing": lambda ctx, args: ToolResult(content="deleted"),
    }
    return build_thread_manager(RuntimeConfig(system_dir=make_system_dir(tmp_path)),
                                EngineOptions(llm=llm, handlers=handlers))


async def until(thread, kind, timeout=2.0):
    """Read the thread's events up to the first of type `kind`; return them all."""
    seen = []
    while True:
        event = await asyncio.wait_for(thread.next_event(), timeout)
        seen.append(event)
        if event.type == kind:
            return seen


def test_conversation_keeps_history(tmp_path):
    async def main():
        llm = ScriptedLLM([LLMResponse("", [call("read_thing")]), LLMResponse("first answer"),
                           LLMResponse("second answer")])
        manager = manager_for(tmp_path, llm)
        new = await manager.start_thread(StartThreadOptions(auth=user(), session_source="cli"))
        thread = new.thread
        assert (await thread.next_event()).type == "session_configured"

        await thread.send("hello", user())
        events = await until(thread, "task_complete")
        assert [e.seq for e in events] == sorted(e.seq for e in events)
        assert any(e.type == "tool" and e.data["name"] == "read_thing" for e in events)
        assert events[-1].data["status"] == "succeeded" and events[-1].data["result"] == "first answer"
        reply = (await until(thread, "reply"))[-1]
        assert reply.data["answer"] == "first answer" and reply.data["status"] == "succeeded"

        await thread.send("and again", user())
        await until(thread, "reply")             # the transcript gets the reply, not each task's answer
        roles = [(m.role, m.content) for m in thread.history()]
        assert roles == [("user", "hello"), ("assistant", "first answer"),
                         ("user", "and again"), ("assistant", "second answer")]
        last_request = llm.requests[-1]   # the second task saw the first one's transcript
        assert [m.content for m in last_request.messages[:3]] == ["hello", "first answer", "and again"]
        assert len(thread.task_ids()) == 2 and thread.status() == "idle"
    asyncio.run(main())


def test_threads_share_one_harness(tmp_path):
    async def main():
        llm = ScriptedLLM([LLMResponse("answer a"), LLMResponse("answer b")])
        manager = manager_for(tmp_path, llm)
        a = (await manager.start_thread(StartThreadOptions(auth=user(), model="model-a"))).thread
        b = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        assert a.session.harness is b.session.harness

        await a.send("for a", user())
        events_a = await until(a, "task_complete")
        await b.send("for b", user())
        events_b = await until(b, "task_complete")
        tasks_a = {e.data["task"] for e in events_a if "task" in e.data}
        tasks_b = {e.data["task"] for e in events_b if "task" in e.data}
        assert tasks_a == set(a.task_ids()) and tasks_b == set(b.task_ids())
        assert [r.model for r in llm.requests] == ["model-a", "default"]   # per-thread model, else the default
    asyncio.run(main())


def test_owner_isolation(tmp_path):
    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([]))
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        assert await manager.get_thread(thread.thread_id, user("user_b")) is None
        assert await manager.list_threads(user("user_b")) == []
        assert [t.thread_id for t in await manager.list_threads(user())] == [thread.thread_id]
        with pytest.raises(PermissionError):
            await thread.send("hi", user("user_b"))
    asyncio.run(main())


def test_approval_through_thread(tmp_path):
    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([LLMResponse("", [call("delete_thing")]), LLMResponse("gone")]))
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("delete it", user())
        events = await until(thread, "task_waiting")
        request = next(e for e in events if e.type == "approval_required")
        assert thread.status() == "waiting"
        with pytest.raises(PermissionError):    # approvals: owner only, never machines
            await thread.approve(request.data["call"], user(machine=True))
        await thread.approve(request.data["call"], user())
        done = (await until(thread, "task_complete"))[-1]
        assert done.data["result"] == "gone"
    asyncio.run(main())


def test_input_answers_ask_user(tmp_path):
    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([LLMResponse("", [call("ask_user", question="which one?")]),
                                                     LLMResponse("ok, the blue one")]))
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("pick one", user())
        events = await until(thread, "task_waiting")
        assert any(e.type == "input_required" for e in events)
        task_id = thread.task_ids()[0]
        assert await thread.send("blue", user()) == task_id      # the reply, not a new task
        assert (await until(thread, "task_complete"))[-1].data["result"] == "ok, the blue one"
        assert len(thread.task_ids()) == 1
    asyncio.run(main())


def test_input_while_running_is_queued(tmp_path):
    async def main():
        llm = GatedLLM([LLMResponse("first"), LLMResponse("second")])
        manager = manager_for(tmp_path, llm)
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        first = await thread.send("one", user())
        await llm.started.wait()
        second_intent = await thread.send("two", user())          # a pending intent, behind the running one
        assert second_intent != first and thread.intent_ids() == [first, second_intent]
        assert (await until(thread, "input_queued"))[-1].data["intent"] == first
        llm.gate.set()
        await until(thread, "task_complete")
        second = (await until(thread, "task_complete"))[-1]
        assert second.data["result"] == "second" and len(thread.task_ids()) == 2
    asyncio.run(main())


def test_interrupt_running_task(tmp_path):
    async def main():
        llm = GatedLLM([LLMResponse("never")])
        manager = manager_for(tmp_path, llm)
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("long job", user())
        await llm.started.wait()
        await thread.interrupt(user())
        done = (await until(thread, "task_complete"))[-1]
        assert done.data["status"] == "cancelled"
        reply = (await until(thread, "reply"))[-1]
        assert reply.data["status"] == "cancelled" and thread.status() == "idle"
    asyncio.run(main())


def test_fork_and_shutdown(tmp_path):
    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([LLMResponse("a1"), LLMResponse("a2")]))
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        for text in ("q1", "q2"):
            await thread.send(text, user())
            await until(thread, "task_complete")
        fork = (await manager.fork_thread(thread.thread_id, user(), TruncateBeforeNthUserMessage(2))).thread
        assert [m.content for m in fork.history()] == ["q1", "a1"]
        assert fork.config_snapshot().forked_from_thread_id == thread.thread_id
        with pytest.raises(KeyError):
            await manager.fork_thread(thread.thread_id, user("user_b"), TruncateBeforeNthUserMessage(1))

        report = await manager.shutdown_all_threads()
        assert sorted(report.completed) == sorted([thread.thread_id, fork.thread_id])
        assert not thread.is_running()
    asyncio.run(main())


def test_runtime_config_from_core_config(tmp_path, monkeypatch):
    system = make_system_dir(tmp_path)
    cfg = tmp_path / "agent.cfg"
    cfg.write_text(f"[context]\nsystem_dir = {system}\n\n[tools]\nplugin_dirs = /a, /b\ntop_k = 7\n\n"
                   "[harness]\nmax_steps = 12\nconversation_tokens = 3000\n\n[llm]\nmodel = test-model\n\n"
                   "[memory]\nconsolidate_failed = true\nextract = notes\nmax_per_task = 2\n")
    monkeypatch.setenv("AGENTICWF_CONFIG", str(cfg))
    config = RuntimeConfig.from_core_config()
    assert config.system_dir == system and [str(p) for p in config.plugin_dirs] == ["/a", "/b"]
    assert config.top_k == 7 and config.limits.max_steps == 12 and config.model == "test-model"
    assert config.limits.conversation_tokens == 3000 and config.limits.conversation_messages == 20
    assert (config.memory.consolidate_failed, config.memory.extract, config.memory.max_per_task) == (True, "notes", 2)
    with pytest.raises(ValueError, match="no model configured"):   # no [llm.model.*] and no provider
        build_thread_manager(config)


def test_long_conversations_are_compacted(tmp_path):
    """Past `conversation_messages`, intake folds the older requests (with their replies) into a
    summary; the session sends later intents the summary and what it does not cover, so the next
    compaction reads only the summary and the messages since."""
    from agenticwf.agent.runtime.harness.harness import Limits

    class Model(LLM):
        def __init__(self):
            self.requests, self.answers, self.summaries = [], 0, 0

        async def complete(self, request):
            self.requests.append(request)
            if "earlier part of this conversation" in request.system:
                self.summaries += 1
                return LLMResponse(f"SUMMARY-{self.summaries}")
            self.answers += 1
            return LLMResponse(f"answer {self.answers}")

    async def main():
        llm = Model()
        config = RuntimeConfig(system_dir=make_system_dir(tmp_path),
                               limits=Limits(conversation_messages=2, conversation_keep=1))
        manager = build_thread_manager(config, EngineOptions(llm=llm))
        thread = (await manager.start_thread(StartThreadOptions(auth=user(), session_source="cli"))).thread
        intents = []
        for text in ("one", "two"):
            await thread.send(text, user())
            intents.append((await until(thread, "reply"))[-1].data["intent"])
        assert llm.summaries == 0                                   # two messages before "two": within the limit

        await thread.send("three", user())
        seen = await until(thread, "reply")
        compacted = next(e for e in seen if e.type == "conversation_compacted")
        # keep=1 would split "two" from its answer: the fold stops before the whole of "two"
        assert compacted.data["summary"] == "SUMMARY-1" and compacted.data["keys"] == [intents[0]]
        folded = llm.requests[-2].messages[0].content
        assert "user: one" in folded and "assistant: answer 1" in folded and "two" not in folded
        task_messages = [m.content for m in llm.requests[-1].messages[:4]]
        assert "SUMMARY-1" in task_messages[0] and task_messages[1:] == ["two", "answer 2", "three"]

        await thread.send("four", user())
        await until(thread, "reply")
        again = llm.requests[-2].messages[0].content                # the summary so far, and only what is new
        assert "<summary so far>\nSUMMARY-1" in again and "user: two" in again and "assistant: answer 2" in again
        assert "user: one" not in again and "three" not in again
        assert "SUMMARY-2" in llm.requests[-1].messages[0].content
        state = thread.session.state
        assert state.summarized == set(intents) and [m.content for m in state.excerpt()] == ["three", "answer 3", "four", "answer 4"]
        assert [m.content for m in thread.history()][:2] == ["one", "answer 1"]   # the transcript stays whole
        await manager.shutdown_all_threads()

    asyncio.run(main())


def test_conversation_compaction_without_a_model_and_with_huge_messages(tmp_path):
    from agenticwf.agent.runtime.harness.harness import Limits
    from agenticwf.agent.runtime.llm.base import Message
    from agenticwf.agent.runtime.records import Intent

    manager = build_thread_manager(RuntimeConfig(system_dir=make_system_dir(tmp_path),
                                                 limits=Limits(conversation_messages=3, conversation_tokens=1000)),
                                   EngineOptions(llm=ScriptedLLM([])))   # every model call fails
    harness = manager.harness
    many = Intent(thread_id="th", owner=("user_a", None), auth=user(), text="next",
                  conversation=[Message("user", f"m{n}") for n in range(10)])
    assert asyncio.run(harness._compact_conversation(many))
    assert [m.content for m in many.conversation] == ["m7", "m8", "m9"] and many.conversation_summary == ""

    huge = Intent(thread_id="th", owner=("user_a", None), auth=user(), text="next",
                  conversation=[Message("user", "x" * 50_000), Message("assistant", "short")])
    assert asyncio.run(harness._compact_conversation(huge))    # nothing to fold (all are the newest): clipped
    assert len(huge.conversation[0].content) < 4100 and huge.conversation[0].content.endswith("[…]")
    assert huge.conversation[1].content == "short"
    small = Intent(thread_id="th", owner=("user_a", None), auth=user(), text="next", conversation=[Message("user", "hi")])
    assert not asyncio.run(harness._compact_conversation(small))


def test_a_job_queued_from_another_thread_wakes_the_started_pool(tmp_path):
    """The API starts the pool when the app starts; a job queued from a thread-pool route (or
    any thread) then wakes it, rather than waiting for the next job queued on the event loop."""
    import threading

    from agenticwf.agent.runtime.harness.store import Job

    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([]))
        ran = asyncio.Event()

        async def handler(ref):
            ran.set()

        manager.harness.job_handlers["probe"] = handler
        manager.workers.start()
        await asyncio.sleep(0.05)   # the workers found nothing and wait to be woken
        thread = threading.Thread(target=lambda: manager.harness.store.enqueue(Job("probe", "x")))
        thread.start()
        thread.join()
        await asyncio.wait_for(ran.wait(), 2)
        await manager.workers.stop()

    asyncio.run(main())


def test_a_job_queued_without_a_loop_waits_for_the_pool(tmp_path):
    from agenticwf.agent.runtime.harness.store import Job

    manager = manager_for(tmp_path, ScriptedLLM([]))
    manager.harness.store.enqueue(Job("probe", "y"))   # no event loop: nothing starts, nothing is lost

    async def main():
        ran = asyncio.Event()

        async def handler(ref):
            ran.set()

        manager.harness.job_handlers["probe"] = handler
        manager.workers.start()                        # the app starting: each worker claims before it waits
        await asyncio.wait_for(ran.wait(), 2)
        await manager.workers.stop()

    asyncio.run(main())


def test_workers_share_themselves_fairly_between_owners():
    from agenticwf.agent.runtime.harness.store import InMemoryStore, Job

    store = InMemoryStore()
    for n in range(5):
        store.enqueue(Job("task", f"a{n}", owner="alice"))
    store.enqueue(Job("task", "b0", owner="bob"))
    first = store.claim("w1", 60)
    assert first.ref == "a0"                                                  # oldest first while nobody runs
    second = store.claim("w2", 60, running={"alice": 1})
    assert second.ref == "b0"                                                 # bob runs nothing: his job jumps ahead
    third = store.claim("w3", 60, running={"alice": 1, "bob": 1})
    assert third.ref == "a1"
    assert store.claim("w4", 60, running={"alice": 2, "bob": 1}, per_owner=2) is None   # alice at her limit
    assert store.claim("w4", 60, running={"alice": 1, "bob": 1}, per_owner=2).ref == "a2"


def test_a_pool_keeps_a_worker_for_other_owners(tmp_path):
    """One owner's flood takes all workers but one; another owner's job then starts at once."""
    from agenticwf.agent.runtime.harness.store import Job

    async def main():
        manager = manager_for(tmp_path, ScriptedLLM([]))
        workers = manager.workers
        assert (workers.size, workers.per_owner) == (4, 3)
        gate, started = asyncio.Event(), []

        async def slow(ref):
            started.append(ref)
            await gate.wait()

        manager.harness.job_handlers["slow"] = slow
        workers.start()
        for n in range(6):
            manager.harness.store.enqueue(Job("slow", f"a{n}", owner="alice"))
        await asyncio.sleep(0.1)
        assert started == ["a0", "a1", "a2"]                                   # one worker stays free
        manager.harness.store.enqueue(Job("slow", "b0", owner="bob"))
        await asyncio.sleep(0.1)
        assert started[-1] == "b0"
        gate.set()
        await asyncio.sleep(0.1)
        assert sorted(started) == ["a0", "a1", "a2", "a3", "a4", "a5", "b0"]   # and alice's rest after
        await workers.stop()

    asyncio.run(main())
