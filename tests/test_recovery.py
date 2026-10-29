"""Regressions from the system review: restarts, leases, call ids and limits in the runtime core."""

import asyncio
import json
import pickle
import time

import pytest

from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.harness.db_store import DatabaseStore
from agenticwf.agent.runtime.harness.harness import AgentHarness, Limits, _unique_call_ids
from agenticwf.agent.runtime.harness.planning import IntakeSettings
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.harness.store import InMemoryStore, Job
from agenticwf.agent.runtime.llm.base import LLMError, LLMResponse, ToolCallRequest
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM
from agenticwf.agent.runtime.policy.engine import PolicyEngine
from agenticwf.agent.runtime.records import Intent
from agenticwf.agent.runtime.task import Task, ToolCallStatus
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
from agenticwf.agent.runtime.tools.registry import ToolRegistry
from agenticwf.agent.runtime.tools.spec import ToolResult
from agenticwf.agent.runtime.types import RuntimeState

from tests.conftest import user
from tests.test_runtime import tool_md

OWNER = ("user_alice", None)


@pytest.fixture
def system(tmp_path):
    root = tmp_path / "system"
    (root / "task-types" / "default").mkdir(parents=True)
    (root / "task-types" / "default" / "AGENTS.md").write_text("# a\n")
    (root / "task-types" / "default" / "WORKFLOW.md").write_text("# w\n")
    (root / "tools").mkdir()
    for name, effects in (("send_thing", "external"), ("block_thing", "external"), ("read_thing", "none")):
        (root / "tools" / f"{name}.md").write_text(tool_md(name, scopes="[]", effects=effects).replace("timeout: 1", "timeout: 5"))
    return root


def harness(system, store, script, handlers, **kw):
    registry = ToolRegistry(system_dir=system)
    assert registry.load() == []
    for ref, fn in handlers.items():
        registry.register_handler(f"test.{ref}", fn)
    return AgentHarness(HarnessServices(llm=ScriptedLLM(script), tool_dispatcher=ToolDispatcher(registry, PolicyEngine()),
                                      task_types=TaskTypeResolver(system), store=store), **kw)


def new_intent(h, text="do it"):
    intent = Intent(thread_id=f"th-{time.monotonic_ns()}", owner=OWNER, auth=user(), text=text)
    h.store.save_intent(intent)
    return intent


def test_a_finished_call_is_not_run_again_after_a_restart(system):
    sent, gate = [], asyncio.Event()

    async def block(ctx, args):
        await gate.wait()

    async def main():
        first = harness(system, DatabaseStore(), [LLMResponse("", [ToolCallRequest("c1", "send_thing", {"value": "pay $100"}),
                                                                    ToolCallRequest("c2", "block_thing", {"value": "x"})])],
                        {"send_thing": lambda ctx, a: (sent.append(a["value"]), ToolResult(content="sent"))[1],
                         "block_thing": block})
        intent = new_intent(first)
        [task] = await first.intake(intent.id)
        runner = asyncio.create_task(first.run(task.id))
        while not any(c.name == "block_thing" and c.status is ToolCallStatus.RUNNING for c in task.calls):
            await asyncio.sleep(0.01)
        runner.cancel()                                                  # the process stops here
        await asyncio.gather(runner, return_exceptions=True)

        store = DatabaseStore()                                          # restart: only the tables remain
        store.open_intents()
        second = harness(system, store, [LLMResponse("done")],
                         {"send_thing": lambda ctx, a: (sent.append(a["value"]), ToolResult(content="sent"))[1],
                          "block_thing": lambda ctx, a: ToolResult(content="ok")})
        second.resume_intent(store.load_intent(intent.id))
        done = await second.run(task.id)
        calls = {c.id: c for c in done.calls}
        assert sent == ["pay $100"]                                      # once, not twice
        assert calls["c1"].status is ToolCallStatus.SUCCEEDED and calls["c1"].result.content == "sent"
        assert calls["c2"].status is ToolCallStatus.INTERRUPTED          # it may or may not have acted
    asyncio.run(main())


def test_dependent_tasks_start_after_a_restart(system):
    async def main():
        split = {"tasks": [{"id": "A", "request": "research"}, {"id": "B", "request": "code", "depends_on": ["A"]}]}
        first = harness(system, DatabaseStore(), [LLMResponse(json.dumps(split)), LLMResponse("A's answer")], {},
                        intake=IntakeSettings(split=True))
        intent = new_intent(first, "research then code")
        a, b = await first.intake(intent.id)
        await first.run(a.id)                                            # A done; B's job only in memory
        store = DatabaseStore()                                          # restart
        [resumed] = [i for i in store.open_intents() if i.id == intent.id]
        second = harness(system, store, [LLMResponse("B's answer")], {}, intake=IntakeSettings(split=True))
        second.resume_intent(resumed)
        assert f"task:{b.id}" in store._jobs
    asyncio.run(main())


def test_a_running_job_is_never_claimed_twice(monkeypatch):
    import agenticwf.agent.runtime.harness.store as store_module

    store = InMemoryStore()
    store.enqueue(Job("task", "t1"))
    job = store.claim("pool-w0", lease_seconds=60)
    real = time.time
    monkeypatch.setattr(store_module.time, "time", lambda: real() + 600)    # the wall clock jumps (sleep, NTP)
    assert store.claim("pool-w1", lease_seconds=60) is None                  # leases use the monotonic clock
    job.lease_until = 0                                                      # even a lease that did lapse:
    assert store.claim("pool-w1", lease_seconds=60, skip={job.key}) is None  # the pool still runs it


def test_call_ids_are_unique_within_a_task():
    task = Task.__new__(Task)
    task.calls = []
    from agenticwf.agent.runtime.task import ToolCall

    task.calls.append(ToolCall(id="call_0", name="read_thing", arguments={}, step=1))
    response = LLMResponse("", [ToolCallRequest("call_0", "a", {}), ToolCallRequest("", "b", {}),
                                ToolCallRequest("bad id!", "c", {}), ToolCallRequest("toolu_ok", "d", {})])
    _unique_call_ids(task, response)
    ids = [c.id for c in response.tool_calls]
    assert len(set(ids) | {"call_0"}) == 5 and ids[3] == "toolu_ok" and "call_0" not in ids


def test_max_tool_calls_holds_within_one_step(system):
    ran = []

    async def main():
        calls = [ToolCallRequest(f"c{n}", "read_thing", {"value": str(n)}) for n in range(10)]
        h = harness(system, InMemoryStore(), [LLMResponse("", calls), LLMResponse("done")],
                    {"read_thing": lambda ctx, a: (ran.append(a["value"]), ToolResult(content="ok"))[1]},
                    limits=Limits(max_tool_calls=3))
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        done = await h.run(task.id)
        assert len(ran) == 3 and done.state is RuntimeState.FAILED and done.error["code"] == "max_tool_calls"
    asyncio.run(main())


def test_a_failed_compaction_ends_the_step(system):
    def down(request):
        raise LLMError("provider down")

    async def main():
        h = harness(system, InMemoryStore(), [LLMResponse("", [ToolCallRequest("c1", "read_thing", {"value": "x" * 5000})]), down],
                    {"read_thing": lambda ctx, a: ToolResult(content="y" * 20_000)}, limits=Limits(compact_at_tokens=100))
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        done = await h.run(task.id)                                      # no IllegalTransition, no extra call
        assert done.state is RuntimeState.FAILED and done.error["code"] == "model_unavailable"
        assert len(h.services.llm.requests) == 2
    asyncio.run(main())


def test_an_unreadable_record_does_not_stop_the_start():
    from sqlalchemy import text

    from agenticwf.agent.core.orm.base.session import get_engine
    from agenticwf.agent.core.orm.models.agent import IntentRow

    good_store = DatabaseStore()
    good = Intent(thread_id="th-good", owner=OWNER, auth=user(), text="fine")
    good_store.save_intent(good)
    with get_engine().begin() as conn:                                   # a record an upgrade can no longer read
        conn.execute(IntentRow.__table__.insert().values(
            id="broken", thread_id="th-x", owner_subject="u", owner_org=None, source="web", status="pending",
            text="x", record=pickle.dumps(1)[:-1] + b"\x00garbage"))
    intents = DatabaseStore().open_intents()
    assert [i.id for i in intents] == [good.id]
    with get_engine().connect() as conn:
        assert conn.execute(text("SELECT status FROM agent_intents WHERE id = 'broken'")).scalar() == "rejected"


# --- errors in the middle of a model's stream reach the fallback ------------------------------------

@pytest.mark.parametrize("provider", ["anthropic", "openai"])
def test_errors_mid_stream_are_retryable(provider):
    httpx = pytest.importorskip("httpx")
    sdk = pytest.importorskip(provider)
    from agenticwf.agent.runtime.llm.base import LLMRequest, Message

    request = LLMRequest(model="m", owner=OWNER, system="", messages=[Message("user", "hi")])
    response = httpx.Response(200, request=httpx.Request("POST", "https://api.example/v1"))
    if provider == "anthropic":
        from agenticwf.agent.runtime.llm.providers.anthropic import AnthropicLLM

        llm = AnthropicLLM(client=object())
        errors = [sdk.APIStatusError("Overloaded", response=response,
                                     body={"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})]
    else:
        from agenticwf.agent.runtime.llm.providers.openai_compat import OpenAICompatibleLLM

        llm = OpenAICompatibleLLM("vllm", client=object())
        errors = [sdk.APIError("server error mid-stream", request=response.request, body=None)]
    errors.append(httpx.RemoteProtocolError("peer closed connection without sending complete message body"))
    for error in errors:
        async def broken(*args, **kwargs):
            raise error
        llm._stream = broken
        with pytest.raises(LLMError) as raised:
            asyncio.run(llm.complete(request))
        assert raised.value.retryable, error


def test_live_updates_stay_within_one_organization():
    from agenticwf.agent.api.events import Updates

    updates, cern, bnl = Updates(), asyncio.Queue(), asyncio.Queue()
    updates._subscribers[("user_a", "cern")].add(cern)
    updates._subscribers[("user_a", "bnl")].add(bnl)
    updates.publish(("user_a", "cern"), {"type": "message", "text": "cern only"})
    assert cern.qsize() == 1 and bnl.qsize() == 0


def test_waiting_for_a_person_does_not_use_up_the_deadline(system):
    async def main():
        h = harness(system, InMemoryStore(), [LLMResponse("", [ToolCallRequest("c1", "ask_user", {"question": "which?"})]),
                                              LLMResponse("done")], {})
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        waiting = await h.run(task.id)
        assert waiting.state is RuntimeState.WAITING_USER
        before = waiting.deadline
        waiting.waiting_since -= 3600                                   # an hour passes before the answer
        h.reply(task.id, user(), "this one")
        assert waiting.deadline >= before + 3600 and waiting.waiting_since is None
    asyncio.run(main())


# --- prompts are Markdown files; tasks are stored in parts -------------------------------------------

def test_prompts_come_from_markdown_files(system):
    (system / "prompts").mkdir()
    (system / "prompts" / "plan.md").write_text("SYSTEM-WIDE PLAN, at most ${max_items} turns. JSON please.")
    (system / "task-types" / "careful" / "prompts").mkdir(parents=True)
    (system / "task-types" / "careful" / "AGENTS.md").write_text("# careful\n")
    (system / "task-types" / "careful" / "prompts" / "plan.md").write_text("CAREFUL PLAN (${max_items}).")
    resolver = TaskTypeResolver(system)
    assert resolver.prompt("plan", "default", OWNER).content.startswith("SYSTEM-WIDE PLAN")
    assert resolver.prompt("plan", "careful", OWNER).source == "system:careful/prompts/plan.md"
    assert resolver.prompt("split", "careful", OWNER).source == "packaged:prompts/split.md"   # not per type
    assert "Combine their results" in resolver.prompt("reply").content                         # packaged default

    async def main():
        h = harness(system, InMemoryStore(), [LLMResponse('{"items": [{"id": "t1", "goal": "g"}]}'), LLMResponse("done")], {},
                    intake=IntakeSettings(plan="always"))
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        task_type = h.create_task(user(), "x", task_type="careful")
        assert task_type.task_type.effective == "system:careful"
        await h.run(task.id)
        assert h.services.llm.requests[0].messages[-1].content == "SYSTEM-WIDE PLAN, at most 8 turns. JSON please."
    asyncio.run(main())


def test_a_task_is_stored_in_parts(system):
    from sqlalchemy import func, select

    from agenticwf.agent.core.orm.base.session import get_engine
    from agenticwf.agent.core.orm.models.agent import StepMessageRow, TaskRow, ToolCallRow

    big = "BIG-RESULT-" * 2000

    async def main():
        calls = [ToolCallRequest(f"c{n}", "read_thing", {"value": str(n)}) for n in range(3)]
        h = harness(system, DatabaseStore(), [LLMResponse("", calls), LLMResponse("done")],
                    {"read_thing": lambda ctx, a: ToolResult(content=big)})
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        return await h.run(task.id), h.store.load(task.id)[1]

    task, ctx = asyncio.run(main())
    with get_engine().connect() as conn:
        record = conn.execute(select(TaskRow.record).where(TaskRow.id == task.id)).scalar()
        entries = conn.execute(select(func.count()).select_from(StepMessageRow).where(StepMessageRow.task_id == task.id)).scalar()
        stored_calls = conn.execute(select(func.count()).select_from(ToolCallRow).where(ToolCallRow.task_id == task.id)).scalar()
    assert b"BIG-RESULT" not in record and len(record) < 20_000            # the head stays small
    assert stored_calls == 3 and entries == len(ctx.all_messages())

    again, again_ctx = DatabaseStore().load(task.id)                       # a fresh process reads the parts
    assert [(c.id, c.status, c.result.content) for c in again.calls] == [(c.id, c.status, c.result.content) for c in task.calls]
    assert [(m.role, m.content) for m in again_ctx.all_messages()] == [(m.role, m.content) for m in ctx.all_messages()]
    assert again.task_type.agents.content == "# a\n" and again_ctx.instructions is again.task_type


def test_compaction_rewrites_only_its_part_and_old_rows_still_load(system):
    from sqlalchemy import select, update

    from agenticwf.agent.core.orm.base.session import get_engine
    from agenticwf.agent.core.orm.models.agent import StepMessageRow, TaskRow

    async def main():
        h = harness(system, DatabaseStore(), [LLMResponse("", [ToolCallRequest("c1", "read_thing", {"value": "x"})]),
                                              LLMResponse("done")], {"read_thing": lambda ctx, a: ToolResult(content="ok")})
        intent = new_intent(h)
        [task] = await h.intake(intent.id)
        return h, await h.run(task.id)

    h, task = asyncio.run(main())
    store = h.store
    _, ctx = store.load(task.id)
    turn = next(iter(ctx.turns.values()))
    ctx.active_turn = turn.turn_id
    ctx.replace_with_summary(0, len(turn.messages) - 1, "short summary")    # compaction in the turn's part
    store.save(task, ctx)
    reread = DatabaseStore().load(task.id)[1]
    assert [m.content for m in next(iter(reread.turns.values())).messages] == [m.content for m in turn.messages]

    with get_engine().begin() as conn:                                     # a row from before this layout
        conn.execute(update(TaskRow).where(TaskRow.id == task.id).values(record=pickle.dumps((task, ctx)), base=None))
        conn.execute(StepMessageRow.__table__.delete().where(StepMessageRow.task_id == task.id))
    old_store = DatabaseStore()
    old_task, old_ctx = old_store.load(task.id)
    assert old_task.id == task.id and len(old_ctx.all_messages()) == len(ctx.all_messages())
    old_store.save(old_task, old_ctx)                                      # saved in parts from now on
    with get_engine().connect() as conn:
        assert conn.execute(select(TaskRow.base).where(TaskRow.id == task.id)).scalar() is not None


def test_the_migration_adds_base_to_an_old_tasks_table():
    from sqlalchemy import inspect, text

    from agenticwf.agent.api.db import init_db
    from agenticwf.agent.core.orm.base.session import get_engine

    import sqlalchemy as sa

    engine = get_engine()
    old = sa.Table("agent_tasks", sa.MetaData(),   # agent_tasks as the previous version created it
                   sa.Column("id", sa.String(32), primary_key=True), sa.Column("intent_id", sa.String(32)),
                   sa.Column("task_type", sa.String(64), nullable=False), sa.Column("state", sa.String(32), nullable=False),
                   sa.Column("record", sa.LargeBinary, nullable=False), sa.Column("result", sa.JSON),
                   sa.Column("owner_subject", sa.String(255), nullable=False), sa.Column("owner_org", sa.String(255)),
                   sa.Column("created_at", sa.DateTime, nullable=False), sa.Column("updated_at", sa.DateTime, nullable=False))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE agent_tasks"))
        old.create(conn)
        conn.execute(text("DELETE FROM alembic_version"))
    init_db()
    assert "base" in {c["name"] for c in inspect(engine).get_columns("agent_tasks")}
    assert {"agent_step_messages", "agent_tool_calls"} <= set(inspect(engine).get_table_names())
    init_db()   # and again: idempotent
    _restore_table("agent_tasks")   # the next tests need the table exactly as the models define it


def _restore_table(name: str) -> None:
    from agenticwf.agent.core.orm.base.session import BASE, get_engine

    table = BASE.metadata.tables[name]
    table.drop(get_engine(), checkfirst=True)
    table.create(get_engine())
