"""Message flow and durability (docs/agent-flow.md): intents, typed tasks, plans and turns, checks, effects, jobs."""

import asyncio
import json
import time

from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager
from agenticwf.agent.runtime.harness.planning import IntakeSettings, parse_plan, parse_tasks
from agenticwf.agent.runtime.harness.store import InMemoryStore, Job
from agenticwf.agent.runtime.llm.base import LLMResponse
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.records import EffectStatus, IntentStatus, TurnState
from agenticwf.agent.runtime.task import ToolCallStatus
from agenticwf.agent.runtime.thread.manager import StartThreadOptions
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.types import RuntimeState

from tests.test_runtime import build, call, make_system_dir, system_dir, tool_md, user  # noqa: F401  (fixture)
from tests.test_thread import until

PLAN = {"items": [
    {"id": "t1", "goal": "read the thing", "needs": ["read_thing"]},
    {"id": "t2", "goal": "write it back", "needs": ["write_thing"], "depends_on": ["t1"],
     "verify": [{"kind": "produces", "what": "what was written"}]},
]}


def flow_manager(tmp_path, llm, **intake):
    handlers = {"test.read_thing": lambda ctx, args: ToolResult(content="thing"),
                "test.write_thing": lambda ctx, args: ToolResult(content="written")}
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), intake=IntakeSettings(**intake))
    return build_thread_manager(config, EngineOptions(llm=llm, handlers=handlers))


# --- intake: one intent, several typed tasks ----------------------------------------------------

def test_intent_splits_into_typed_dependent_tasks(tmp_path):
    split = {"tasks": [
        {"id": "A", "task_type": "review", "request": "summarise the papers", "depends_on": []},
        {"id": "B", "task_type": "default", "request": "write code for figure 2", "depends_on": ["A"]}]}

    async def main():
        llm = ScriptedLLM([LLMResponse(json.dumps(split)), LLMResponse("the method is X"),
                           LLMResponse("code written"), LLMResponse("combined reply")])
        manager = flow_manager(tmp_path, llm, split=True)
        thread = (await manager.start_thread(StartThreadOptions(auth=user()))).thread
        intent_id = await thread.send("summarise these papers and reproduce figure 2", user())

        intent = thread.session.store.load_intent(intent_id)   # TO DO: on record before any work
        assert intent.requested_task_type is None and intent.text.startswith("summarise")

        events = await until(thread, "reply")
        created = next(e for e in events if e.type == "tasks_created").data["tasks"]
        assert [t["task_type"] for t in created] == ["review", "default"]
        assert created[1]["depends_on"] == [created[0]["task"]]

        b_request = llm.requests[2]   # task B saw task A's result, not its steps
        assert "<results of earlier tasks>" in b_request.messages[-3].content
        assert "the method is X" in b_request.messages[-3].content

        reply = events[-1].data
        assert reply["status"] == "succeeded" and reply["answer"] == "combined reply"
        assert thread.history()[-1].content == "combined reply" and len(thread.task_ids()) == 2
        store = thread.session.store
        assert store.load_intent(intent_id).status is IntentStatus.ANSWERED
        assert store.load_reply(intent_id).task_ids == thread.task_ids()
        assert all(store.load_result(t).status == "succeeded" for t in thread.task_ids())
    asyncio.run(main())


def test_requested_type_applies_to_every_task(tmp_path):
    split = {"tasks": [{"id": "A", "task_type": "default", "request": "one"},
                       {"id": "B", "task_type": "default", "request": "two"}]}

    async def main():
        llm = ScriptedLLM([LLMResponse(json.dumps(split)), LLMResponse("1"), LLMResponse("2"), LLMResponse("both")])
        thread = (await flow_manager(tmp_path, llm, split=True).start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("do one and two", user(), task_type="review")
        created = next(e for e in await until(thread, "reply") if e.type == "tasks_created").data["tasks"]
        assert [t["task_type"] for t in created] == ["review", "review"]
    asyncio.run(main())


def test_failed_dependency_cancels_dependent_and_replies(tmp_path):
    split = {"tasks": [{"id": "A", "task_type": "default", "request": "first"},
                       {"id": "B", "task_type": "default", "request": "second", "depends_on": ["A"]}]}

    async def main():
        llm = ScriptedLLM([LLMResponse(json.dumps(split))])   # then exhausted: task A's model call fails
        thread = (await flow_manager(tmp_path, llm, split=True).start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("first, then second", user())
        events = await until(thread, "reply")
        done = {e.data["task"]: e.data for e in events if e.type == "task_complete"}
        a, b = thread.task_ids()
        assert done[a]["error"]["code"] == "model_unavailable"
        assert done[b]["status"] == "cancelled" and done[b]["error"]["code"] == "dependency_failed"
        assert events[-1].data["status"] == "failed" and thread.status() == "idle"
    asyncio.run(main())


def test_unusable_split_falls_back_to_one_task(tmp_path):
    async def main():
        llm = ScriptedLLM([LLMResponse("I think this is one task."), LLMResponse("done")])
        thread = (await flow_manager(tmp_path, llm, split=True).start_thread(StartThreadOptions(auth=user()))).thread
        await thread.send("just this", user())
        assert (await until(thread, "reply"))[-1].data["answer"] == "done" and len(thread.task_ids()) == 1
    asyncio.run(main())


def test_parsers_reject_bad_shapes():
    assert parse_tasks('{"tasks": [{"id": "A", "request": "x", "depends_on": ["B"]}]}', 5) is None   # forward dep
    assert parse_tasks('```json\n{"tasks": [{"id": "A", "request": "x", "task_type": "../etc"}]}\n```', 5)[0].task_type == "default"
    assert parse_plan('{"items": [{"id": "t1", "goal": ""}]}', 8) is None
    assert parse_plan(json.dumps(PLAN), 1) is None                                                # too many items
    assert [i.id for i in parse_plan("Plan: " + json.dumps(PLAN), 8)] == ["t1", "t2"]


# --- planning, turns and the turn context -------------------------------------------------------

def test_plan_runs_turns_with_their_own_context(system_dir):
    harness, events, _ = build(system_dir, [
        LLMResponse(json.dumps(PLAN)),                                             # planning
        LLMResponse("", [call("read_thing")]),                                     # t1, step 1
        LLMResponse("", [call("complete_turn", summary="found: thing")]),          # t1, step 2
        LLMResponse("", [call("write_thing", value="a.py")]),                      # t2, step 3
        LLMResponse("wrote a.py"),                                                 # t2, step 4: ends the turn
        LLMResponse("all done"),                                                   # combine
    ], intake=IntakeSettings(plan="always"))
    task = harness.create_task(user(), "read the thing and write it back")
    assert task.state is RuntimeState.QUEUED and not task.active_tools        # planned before any turn
    task = asyncio.run(harness.run(task.id))

    assert task.state is RuntimeState.COMPLETED and task.result == "all done"
    assert [(t.id, t.state) for t in task.turns] == [("t1", TurnState.SUCCEEDED), ("t2", TurnState.SUCCEEDED)]
    assert task.turns[0].result.summary == "found: thing" and task.turns[1].result.summary == "wrote a.py"
    assert "complete_turn" in task.turns[0].tools

    t2_request = harness.services.llm.requests[3]
    text = "".join(m.content for m in t2_request.messages)
    assert "Goal of this turn: write it back" in text and "t1: found: thing" in text   # the earlier turn's result
    assert not any(m.role == "tool" for m in t2_request.messages)                     # ... never its steps
    assert "Turn 2 of 2" in t2_request.messages[-1].content

    kinds = [e["type"] for e in events]
    assert "task_plan" in kinds and kinds.count("turn") == 4                          # started/completed × 2
    effects = harness.store.effects_for_task(task.id)
    assert [(e.tool, e.status) for e in effects] == [("write_thing", EffectStatus.APPLIED)]
    result = harness.store.load_result(task.id)
    assert result.status == "succeeded" and result.artifacts == [effects[0].target]
    assert [t["state"] for t in result.turns] == ["succeeded", "succeeded"]


def test_single_turn_tasks_do_not_offer_complete_turn(system_dir):
    harness, _, _ = build(system_dir, [LLMResponse("done")])
    task = harness.create_task(user(), "x")
    assert "complete_turn" not in task.active_tools
    asyncio.run(harness.run(task.id))
    assert "complete_turn" not in [t["name"] for t in harness.services.llm.requests[0].tools]


def test_turn_check_reopens_then_fails(system_dir):
    plan = {"items": [{"id": "t1", "goal": "read it", "verify": [{"kind": "tool_succeeded", "tool": "read_thing"}]}]}
    harness, events, _ = build(system_dir, [
        LLMResponse(json.dumps(plan)), LLMResponse("done without reading"), LLMResponse("still done")],
        intake=IntakeSettings(plan="always"))
    task = asyncio.run(harness.run(harness.create_task(user(), "read it").id))
    assert any(e["type"] == "turn_rejected" for e in events)
    assert task.state is RuntimeState.FAILED and task.error["code"] == "turn_failed"
    assert task.turns[0].state is TurnState.FAILED and task.turns[0].attempts == 2
    assert harness.store.load_result(task.id).verification[0]["passed"] is False


def test_turn_check_passes_after_retry(system_dir):
    plan = {"items": [{"id": "t1", "goal": "read it", "verify": [{"kind": "tool_succeeded", "tool": "read_thing"}]}]}
    harness, _, _ = build(system_dir, [
        LLMResponse(json.dumps(plan)), LLMResponse("done without reading"),
        LLMResponse("", [call("read_thing")]), LLMResponse("read it")], intake=IntakeSettings(plan="always"))
    task = asyncio.run(harness.run(harness.create_task(user(), "read it").id))
    assert task.state is RuntimeState.COMPLETED and task.result == "read it"
    retry_request = harness.services.llm.requests[2]
    assert "This turn is not done: read_thing has not succeeded" in retry_request.messages[-2].content


def test_task_type_plan_setting(system_dir):
    (system_dir / "task-types" / "review" / "task.yaml").write_text("plan: always\n")
    harness, _, _ = build(system_dir, [LLMResponse("not json"), LLMResponse("answer")])
    task = asyncio.run(harness.run(harness.create_task(user(), "look", task_type="review").id))
    assert task.state is RuntimeState.COMPLETED and [i.goal for i in task.plan] == ["look"]   # fallback: one item


# --- effects and resume -------------------------------------------------------------------------

def _crash_inside(harness, task, tool):
    """Run until `tool` is executing, then stop the run as a dying worker would."""
    gate = asyncio.Event()

    async def blocking(ctx, args):
        await gate.wait()
        return ToolResult(content="never")

    async def main():
        harness.dispatcher.registry.register_handler(f"test.{tool}", blocking)
        runner = asyncio.create_task(harness.run(task.id))
        while not any(c.status is ToolCallStatus.RUNNING for c in task.calls):
            await asyncio.sleep(0.01)
        runner.cancel()
        await asyncio.gather(runner, return_exceptions=True)
    asyncio.run(main())


def test_resume_interrupts_non_idempotent_call_with_intended_effect(system_dir):
    harness, _, _ = build(system_dir, [LLMResponse("", [call("write_thing", value="a.py")]), LLMResponse("checked")])
    task = harness.create_task(user(), "write")
    _crash_inside(harness, task, "write_thing")
    effect = harness.store.effects_for_task(task.id)[0]
    assert effect.status is EffectStatus.INTENDED                       # written ahead of the handler

    task = asyncio.run(harness.run(task.id))                            # another worker resumes it
    assert task.state is RuntimeState.COMPLETED
    assert task.calls[0].status is ToolCallStatus.INTERRUPTED
    assert "interrupted" in next(m.content for m in harness.context(task.id).all_messages() if m.role == "tool")


def test_resume_reruns_idempotent_call(system_dir):
    (system_dir / "tools" / "put_thing.md").write_text(
        tool_md("put_thing", "[code:write]", "local", extra="idempotent: true\n"))
    harness, _, _ = build(system_dir, [LLMResponse("", [call("put_thing", value="a.py")]), LLMResponse("put")])
    task = harness.create_task(user(), "put")
    _crash_inside(harness, task, "put_thing")
    harness.dispatcher.registry.register_handler("test.put_thing", lambda ctx, args: ToolResult(content="ok"))
    task = asyncio.run(harness.run(task.id))
    assert task.state is RuntimeState.COMPLETED and task.calls[0].status is ToolCallStatus.SUCCEEDED
    keys = {e.idempotency_key for e in harness.store.effects_for_task(task.id)}
    assert len(keys) == 1                                                # the re-run used the same key


# --- jobs and leases ----------------------------------------------------------------------------

def test_jobs_leases_and_requeue():
    store = InMemoryStore()
    assert store.enqueue(Job("task", "t1")) and not store.enqueue(Job("task", "t1"))   # not doubled
    job = store.claim("w1", lease_seconds=0.01)
    assert job.key == "task:t1" and store.claim("w2", 10) is None
    time.sleep(0.02)
    taken = store.claim("w2", 10)                                       # w1 died: its lease expired
    assert taken is job and job.lease_owner == "w2"
    assert not store.renew(job, "w1", 10)
    store.complete(job, "w1")                                           # too late: w2 owns it now
    assert not store.enqueue(Job("task", "t1")) and job.again          # e.g. an approval during the run
    store.complete(job, "w2")
    assert store.claim("w3", 10) is job                                 # queued again, not lost


def test_interrupt_during_intake(tmp_path):
    class SlowSplit(ScriptedLLM):
        async def complete(self, request):
            await asyncio.sleep(10)

    async def main():
        thread = (await flow_manager(tmp_path, SlowSplit([]), split=True).start_thread(StartThreadOptions(auth=user()))).thread
        intent_id = await thread.send("anything", user())
        await asyncio.sleep(0.05)
        await thread.interrupt(user())
        reply = (await until(thread, "reply"))[-1]
        assert reply.data["status"] == "cancelled" and thread.task_ids() == []
        assert thread.session.store.load_intent(intent_id).status is IntentStatus.ANSWERED
    asyncio.run(main())
