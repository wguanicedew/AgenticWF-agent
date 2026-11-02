"""Choosing a task's tools with a large catalogue (agenticwf.agent.runtime.tools.dispatcher): the
index of tools left out, a prefix that stays the same within a turn, tools a model picks."""

import asyncio

from agenticwf.agent.runtime.harness.planning import parse_picks
from agenticwf.agent.runtime.llm.base import LLMError, LLMResponse
from agenticwf.agent.runtime.tools.registry import findability
from agenticwf.agent.runtime.tools.spec import HandlerKind, HandlerRef, ToolSource, ToolSpec

from tests.test_runtime import build, call, make_system_dir, user


def names(request):
    return [t["name"] for t in request.tools]


def large(harness, **settings):
    """As if the catalogue were large: search chooses, only the pinned read_thing besides control tools."""
    dispatcher = harness.services.tool_dispatcher
    dispatcher.small_catalogue, dispatcher.top_k = 0, 0
    for key, value in settings.items():
        setattr(dispatcher, key, value)
    return dispatcher


def test_tools_left_out_are_listed_and_the_prefix_stays_within_a_turn(tmp_path):
    harness, _, _ = build(make_system_dir(tmp_path), [
        LLMResponse("", [call("search_tools", query="slow_thing")]),
        LLMResponse("", [call("read_thing", value="x")]),
        LLMResponse("done"),
    ])
    large(harness)
    task = harness.create_task(user(), "do something")
    task = asyncio.run(harness.run(task.id))
    first, second, third = harness.services.llm.requests
    assert task.result == "done"
    assert "# More tools" in first.system and "- slow_thing: The slow thing tool." in first.system
    assert "slow_thing" not in names(first) and "read_thing" in names(first)
    # loading a tool appends it; nothing before it changes, the system prompt neither
    assert names(second)[:len(names(first))] == names(first) and names(second)[len(names(first))] == "slow_thing"
    assert second.system == first.system and third.system == first.system
    assert third.tools == second.tools                                   # no new tool: the same request prefix


def test_no_index_when_switched_off_or_nothing_left_out(tmp_path):
    harness, _, _ = build(make_system_dir(tmp_path), [LLMResponse("done")])
    large(harness, index_tokens=0)
    asyncio.run(harness.run(harness.create_task(user(), "do something").id))
    assert "# More tools" not in harness.services.llm.requests[0].system
    harness, _, _ = build(make_system_dir(tmp_path / "small"), [LLMResponse("done")])   # all offered
    asyncio.run(harness.run(harness.create_task(user(), "do something").id))
    assert "# More tools" not in harness.services.llm.requests[0].system


def test_the_index_keeps_to_its_budget(tmp_path):
    harness, _, _ = build(make_system_dir(tmp_path), [LLMResponse("done")])
    large(harness, index_tokens=12)                                       # room for one line
    asyncio.run(harness.run(harness.create_task(user(), "the slow thing").id))
    system = harness.services.llm.requests[0].system
    assert "- slow_thing:" in system and "more: search_tools finds them" in system   # the closest first


def test_a_small_model_picks_the_tools(tmp_path):
    harness, events, _ = build(make_system_dir(tmp_path), [
        LLMResponse('{"tools": ["slow_thing", "made_up", "write_thing"]}'),
        LLMResponse("done"),
    ])
    large(harness, select_model_above=1, top_k=2)                        # room for two tools besides the pinned
    task = harness.create_task(user(), "do something")
    assert task.active_turn is None                                       # the turn starts after the pick
    task = asyncio.run(harness.run(task.id))
    pick, step = harness.services.llm.requests
    assert pick.purpose == "select_tools" and "- slow_thing: The slow thing tool." in pick.messages[0].content
    assert "read_thing" not in pick.messages[0].content                   # pinned: offered anyway
    assert {"slow_thing", "write_thing"} <= set(names(step)) and "made_up" not in names(step)
    assert task.tool_picks == ("do something", ("slow_thing", "write_thing"))
    assert any(e.get("type") == "tools_picked" for e in events)


def test_without_a_pick_search_chooses(tmp_path):
    def unavailable(request):
        raise LLMError("no small model")

    harness, _, _ = build(make_system_dir(tmp_path), [unavailable, LLMResponse("done")])
    large(harness, select_model_above=1)
    task = asyncio.run(harness.run(harness.create_task(user(), "do something").id))
    assert task.result == "done" and task.tool_picks is None
    harness, _, _ = build(make_system_dir(tmp_path / "few"), [LLMResponse("done")])
    large(harness, select_model_above=50)                                 # not above it: no model call
    asyncio.run(harness.run(harness.create_task(user(), "do something").id))
    assert [r.purpose for r in harness.services.llm.requests] == ["step"]


def test_parse_picks():
    known = {"a", "b", "c"}
    assert parse_picks('{"tools": ["b", "x", "b", "a"]}', known, 5) == ["b", "a"]
    assert parse_picks('Here: ["c", "a", "b"]', known, 2) == ["c", "a"]
    assert parse_picks("none", known, 5) == [] and parse_picks('{"tools": "a"}', known, 5) == []


def spec(name, summary, description="", tags=(), arguments=()):
    return ToolSpec(name=name, summary=summary, description=description or summary,
                    input_schema={"type": "object", "properties": {a: {"type": "string"} for a in arguments}},
                    handler=HandlerRef(HandlerKind.BUILTIN, {"ref": "x"}), source=ToolSource("system", name), tags=tags)


def test_search_matches_argument_names_and_exact_names(tmp_path):
    harness, _, _ = build(make_system_dir(tmp_path), [])
    registry = harness.services.tool_dispatcher.registry
    tools = [spec("rucio__list", "List things.", arguments=("dataset_name", "scope")),
             spec("lookup", "Look something up."), spec("list_all", "List all.")]
    assert [s.name for s in registry.search("dataset", tools, 3)] == ["rucio__list"]
    assert registry.search("lookup", tools, 3)[0].name == "lookup"
    assert registry.search("list_all", tools, 1)[0].name == "list_all"


def test_findability():
    assert findability(spec("jira__create_issue", "create_issue")).startswith("no summary")
    assert findability(spec("x_tool", "Does x.")).startswith("a short summary")
    assert findability(spec("x_tool", "Transfers a dataset between two storage sites.")).startswith("little to match")
    assert findability(spec("x_tool", "Transfers a dataset between two storage sites.", arguments=("dataset",))) == ""
