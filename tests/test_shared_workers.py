"""Worker processes sharing the jobs through the database ([harness] shared; docs/deployment.md).

Two processes are simulated by two stores and two harnesses on the same database: an API process
(sessions, no workers of its own here) and a worker process (workers, no sessions)."""

import asyncio

from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager
from agenticwf.agent.runtime.harness.db_store import DatabaseStore
from agenticwf.agent.runtime.harness.relay import EventRelay, deliverer
from agenticwf.agent.runtime.harness.store import Job
from agenticwf.agent.runtime.llm.base import LLM, LLMResponse
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.thread.manager import StartThreadOptions
from agenticwf.agent.runtime.types import RuntimeState

from tests.test_runtime import make_system_dir
from tests.test_thread import user


def test_jobs_are_claimed_once_fairly_across_processes():
    one, two = DatabaseStore(shared=True), DatabaseStore(shared=True)
    for n in range(4):
        one.enqueue(Job("task", f"a{n}", owner="alice"))
    two.enqueue(Job("task", "b0", owner="bob"))
    assert not one.enqueue(Job("task", "a0", owner="alice"))                # queued once
    first = one.claim("p1-w1", 60)
    assert first.ref == "a0"
    second = two.claim("p2-w1", 60)
    assert second.ref == "b0"                                                 # alice runs one (in p1): bob first
    assert two.claim("p2-w2", 60, per_owner=1) is None                        # alice at her limit everywhere
    third = two.claim("p2-w2", 60, per_owner=2)
    assert third.ref == "a1" and third.attempts == 1

    assert not two.renew(first, "p2-w1", 60)                                  # not its job
    assert one.renew(first, "p1-w1", 60)
    one.enqueue(Job("task", "a0", owner="alice"))                             # queued again while it runs
    one.complete(first, "p1-w1")
    again = one.claim("p1-w1", 60, skip=["task:a1", "task:a2", "task:a3"])
    assert again.ref == "a0"                                                  # so it runs once more
    two.complete(second, "p2-w1")
    assert two.claim("p2-w1", 60, skip=["task:a0", "task:a1", "task:a2", "task:a3"]) is None   # b0 done: gone


def test_an_expired_lease_is_taken_over():
    from agenticwf.agent.core.orm.base.session import get_engine
    from sqlalchemy import text

    one, two = DatabaseStore(shared=True), DatabaseStore(shared=True)
    one.enqueue(Job("task", "x", owner="alice"))
    held = one.claim("p1-w1", 60)
    assert two.claim("p2-w1", 60) is None
    with get_engine().begin() as connection:                                  # p1 died: its lease ran out
        connection.execute(text("UPDATE agent_jobs SET lease_until = '2000-01-01 00:00:00'"))
    taken = two.claim("p2-w1", 60)
    assert taken.ref == "x" and taken.attempts == 2
    one.complete(held, "p1-w1")                                               # the old holder cannot finish it
    assert two.renew(taken, "p2-w1", 60)


def processes(tmp_path, api_llm: LLM, worker_llm: LLM):
    """An API process (its pool never starts) and a worker process, on the same database."""
    system = make_system_dir(tmp_path)
    config = RuntimeConfig(system_dir=system, shared=True)
    api = build_thread_manager(config, EngineOptions(llm=api_llm, task_store=DatabaseStore(shared=True, poll_seconds=0.02)))
    api.harness.store._listeners.clear()                                      # all work happens in the worker
    worker = build_thread_manager(config, EngineOptions(llm=worker_llm, task_store=DatabaseStore(shared=True, poll_seconds=0.02)))
    relay = EventRelay()
    worker.harness.remote = relay.write
    return api, worker, relay


def test_a_conversation_runs_in_a_worker_process_and_reaches_the_session(tmp_path):
    async def main():
        api, worker, relay = processes(tmp_path, ScriptedLLM([]), ScriptedLLM([LLMResponse("hello from the worker")]))
        relay.start_at_newest()
        thread = (await api.start_thread(StartThreadOptions(auth=user(), session_source="cli"))).thread
        deliver = deliverer(api.harness)
        worker.workers.start()
        await thread.send("hi", user())
        seen = []
        for _ in range(200):
            relay.drain(deliver)
            await asyncio.sleep(0.02)
            while not thread.session._events.empty():
                seen.append(thread.session._events.get_nowait().type)
            if "reply" in seen:
                break
        await worker.workers.stop()
        return thread, seen

    thread, seen = asyncio.run(main())
    assert "reply" in seen and "task_complete" in seen and "message_delta" not in seen
    assert [m.content for m in thread.history()] == ["hi", "hello from the worker"]
    assert thread.status() == "idle"                                          # the session moved on


def test_a_cancel_in_the_api_process_stops_the_task_in_the_worker(tmp_path):
    class Slow(LLM):
        def __init__(self):
            self.calls = 0

        async def complete(self, request):
            self.calls += 1
            await asyncio.sleep(0.2)
            from agenticwf.agent.runtime.llm.base import ToolCallRequest
            return LLMResponse("", [ToolCallRequest(f"c{self.calls}", "read_thing", {})])   # never finishes on its own

    async def main():
        api, worker, relay = processes(tmp_path, ScriptedLLM([]), Slow())
        worker.harness.dispatcher.registry.register_handler("test.read_thing", lambda ctx, args: "thing")
        task = api.harness.create_task(user(), "keep going")
        api.harness.schedule(task.id)
        worker.workers.start()
        await asyncio.sleep(0.3)                                              # the worker is inside it
        api.harness.cancel(task.id, user())
        for _ in range(100):
            await asyncio.sleep(0.05)
            api.harness.store.forget(task_ids=[task.id])
            if api.harness.store.load(task.id)[0].state.terminal:
                break
        await worker.workers.stop()
        return api.harness.store.load(task.id)[0]

    final = asyncio.run(main())
    assert final.state is RuntimeState.CANCELLED


def test_an_approval_given_in_the_api_process_is_run_by_the_worker(tmp_path):
    """The worker ran the task until it asked for approval; it must not reuse its cached copy
    (still waiting) when the approval, given in the API process, queues the task again."""
    from agenticwf.agent.runtime.llm.base import ToolCallRequest
    from agenticwf.agent.runtime.task import ToolCallStatus

    async def main():
        script = ScriptedLLM([LLMResponse("", [ToolCallRequest("c1", "delete_thing", {})]), LLMResponse("deleted it")])
        api, worker, relay = processes(tmp_path, ScriptedLLM([]), script)
        worker.harness.dispatcher.registry.register_handler("test.delete_thing", lambda ctx, args: "deleted")
        task = api.harness.create_task(user(), "delete it")
        api.harness.schedule(task.id)
        worker.workers.start()
        for _ in range(100):
            await asyncio.sleep(0.03)
            api.harness.store.forget(task_ids=[task.id])
            if api.harness.store.load(task.id)[0].state is RuntimeState.WAITING_APPROVAL:
                break
        call = api.harness.store.load(task.id)[0].open_calls(ToolCallStatus.AWAITING_APPROVAL)[0]
        api.harness.approve(task.id, call.id, user(), approved=True)
        api.harness.schedule(task.id)
        for _ in range(100):
            await asyncio.sleep(0.03)
            api.harness.store.forget(task_ids=[task.id])
            if api.harness.store.load(task.id)[0].state.terminal:
                break
        await worker.workers.stop()
        return api.harness.store.load(task.id)[0]

    final = asyncio.run(main())
    assert final.state is RuntimeState.COMPLETED and final.result == "deleted it"
