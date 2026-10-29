"""Model routing by requirements and cost (src/agenticwf/agent/runtime/llm/routing.py, docs/model-routing.md)."""

import asyncio
import datetime as dt

import pytest

from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse, Message, ToolCallRequest, Usage
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry, entries_from_sections
from agenticwf.agent.runtime.llm.router import ModelRouter
from agenticwf.agent.runtime.llm.routing import RoutingSettings
from agenticwf.agent.runtime.llm.usage import InMemoryUsage

ALICE, BOB = ("user_alice", "cern"), ("user_bob", "cern")

SECTIONS = {
    "llm.model.glm": {"provider": "vllm", "model": "glm-5.3-flash-int4", "base_url": "http://gpu", "tier": "large",
                      "context_window": "1048576", "capabilities": "tools, vision, reasoning",
                      "cost_input": "0.6", "cost_output": "2.4", "speed": "slow"},
    "llm.model.qwen": {"provider": "vllm", "model": "qwen3.8-27b-fp16", "base_url": "http://gpu", "tier": "medium",
                       "context_window": "131072", "capabilities": "tools, vision",
                       "cost_input": "0.3", "cost_output": "1.2", "cost_cached_input": "0.03"},
    "llm.model.gptoss": {"provider": "vllm", "model": "gpt-oss-20b", "base_url": "http://gpu", "tier": "medium",
                         "context_window": "32768", "capabilities": "tools, reasoning",
                         "cost_input": "0.2", "cost_output": "0.8"},
    "llm.model.gemma": {"provider": "vllm", "model": "gemma-4-12b", "base_url": "http://gpu", "tier": "small",
                        "context_window": "131072", "capabilities": "tools, vision",
                        "cost_input": "0.05", "cost_output": "0.2", "speed": "fast"},
    "llm.model.qwen32": {"provider": "vllm", "model": "hf-qwen3-32b-awq", "base_url": "http://gpu", "tier": "small",
                         "context_window": "16384", "capabilities": "json",
                         "cost_input": "0.04", "cost_output": "0.16"},
}


class Fake(LLM):
    def __init__(self, model_id, answers=None):
        self.model_id, self.answers = model_id, answers

    async def complete(self, request):
        size = len(request.system) // 4 + sum(len(m.content) for m in request.messages) // 4
        if self.answers:
            return self.answers(self.model_id, request)
        return LLMResponse(f"answer from {self.model_id}", usage=Usage(size, 100))


def router(extra=None, answers=None, usage=None, store=None):
    sections = {**SECTIONS, **(extra or {})}
    models, providers = entries_from_sections(sections)
    catalog = ModelCatalog(models, providers, default="qwen", store=store)
    return ModelRouter(catalog, factory=lambda entry, settings: Fake(entry.id, answers),
                       settings=RoutingSettings.parse(sections), usage=usage or InMemoryUsage())


def request(purpose="step", tokens=2000, **kw):
    kw.setdefault("tools", [{"name": "read_thing"}])
    kw.setdefault("model", "default")
    return LLMRequest(system="", messages=[Message("user", "x" * tokens * 4)], purpose=purpose, owner=ALICE, **kw)


def test_model_options_are_checked():
    models, _ = entries_from_sections(SECTIONS)
    glm = next(m for m in models if m.id == "glm")
    assert glm.profile.tier == "large" and glm.profile.context_window == 1048576 and "vision" in glm.profile.capabilities
    for bad in ({"tier": "huge"}, {"context_window": "-1"}, {"capabilities": "telepathy"}, {"cost_input": "-2"}):
        with pytest.raises(ValueError):
            entries = entries_from_sections({"llm.model.x": {"provider": "echo", "model": "x", **bad}})[0]
            ModelCatalog(entries)


def test_each_purpose_goes_to_the_cheapest_capable_model():
    r = router()
    assert r.choose(request("split", tools=[])).entry.id == "qwen32"          # small, cheapest, fits
    assert r.choose(request("split")).entry.id == "gemma"                      # qwen32 has no tools
    assert r.choose(request("step")).entry.id == "gptoss"                      # medium: the cheaper of two
    assert r.choose(request("step", tokens=30000)).entry.id == "qwen"          # gpt-oss's 32k is too small now
    assert r.choose(request("step", tokens=200000)).entry.id == "glm"          # only the large one holds it
    assert r.choose(request("step", min_tier="large")).entry.id == "glm"       # a task type's tier: large
    assert r.choose(request("split", tokens=20000, tools=[])).entry.id == "gemma"   # qwen32's 16k: too small


def test_pins_win_while_they_fit():
    r = router({"llm.purpose": {"compact_conversation": "gemma"}})
    pinned = r.choose(request(model="glm", pinned=True))
    assert (pinned.entry.id, pinned.reason) == ("glm", "pinned")
    small = r.choose(request(model="qwen32", pinned=True, tools=[], tokens=20000))
    assert small.reason == "context" and small.entry.id != "qwen32"            # too small: the nearest that fits
    purpose = r.choose(request("compact_conversation", tools=[]))
    assert (purpose.entry.id, purpose.reason) == ("gemma", "purpose")


def test_the_current_models_cache_counts():
    r = router()
    fresh = r.choose(request("step", tokens=20000))
    assert fresh.entry.id == "gptoss"
    sticky = r.choose(request("step", tokens=20000, current="qwen", cached_hint=19000))
    assert sticky.entry.id == "qwen"                                           # its cache makes it the cheaper


def test_a_persons_own_models_come_first_for_them_only():
    from agenticwf.agent.runtime.llm.catalog import ModelStore

    class Store(ModelStore):
        def __init__(self):
            self.models = [ModelEntry(id="mine", provider="vllm", model="m", base_url="http://gpu", source="database",
                                      owner_subject="user_alice", owner_org="cern", visibility="organization",
                                      options={"tier": "medium", "context_window": 65536, "cost_input": 5.0})]

        def list_models(self):
            return self.models

        def save_model(self, entry, keep_key=False):
            pass

        def delete_model(self, model_id):
            return False

    r = router(store=Store())
    assert r.choose(request("step")).entry.id == "mine"                        # Alice's, though dearer
    bob = r.choose(LLMRequest(model="default", system="", messages=[Message("user", "x")], purpose="step",
                              owner=BOB, tools=[{"name": "t"}]))
    assert bob.entry.id == "gptoss"                                            # shared with Bob: pinnable, not routed


def test_budgets_and_the_highest_tier():
    usage = InMemoryUsage()
    r = router({"llm.budgets": {"default": "0.01 per day", "max_tier.user_bob": "small"}}, usage=usage)
    assert r.choose(LLMRequest(model="default", system="", messages=[Message("user", "x")], purpose="step",
                               owner=BOB, tools=[{"name": "t"}])).entry.id == "gemma"   # Bob: small at most
    asyncio.run(r.complete(request("step")))
    assert usage.rows[-1]["model_id"] == "gptoss" and usage.rows[-1]["cost"] > 0 and usage.rows[-1]["route"] == "routed"
    usage.record({"owner_subject": "user_alice", "owner_org": "cern", "purpose": "step", "model_id": "gptoss",
                  "output_tokens": 0, "cost": 0.0085})
    assert r.under_pressure(ALICE) and r.choose(request("step")).entry.id == "gemma"   # 80%: the smallest tier
    usage.record({"owner_subject": "user_alice", "owner_org": "cern", "purpose": "step", "model_id": "gptoss",
                  "output_tokens": 0, "cost": 0.01})
    with pytest.raises(LLMError, match="budget"):
        asyncio.run(r.complete(request("step")))


def test_output_estimates_follow_the_calls():
    r = router()
    assert r.outputs.expect("step", "gptoss") == 1000                          # typical, before any call
    for _ in range(30):
        asyncio.run(r.complete(request("step")))
    assert 100 <= r.outputs.expect("step", "gptoss") < 200                     # the running average
    assert r.outputs.expect("step", "glm") == r.outputs.expect("step", "gptoss")   # a new model: the purpose's


def test_pinned_mode_keeps_todays_behaviour():
    r = router({"llm.routing": {"mode": "pinned"}})
    assert r.choose(request("split", tools=[])).entry.id == "qwen"            # the default, whatever the purpose


def test_feedback_suggestions_and_adjusting():
    usage = InMemoryUsage()
    r = router({"llm.routing": {"feedback": "adjust", "feedback_min_samples": "20"}}, usage=usage)
    for n in range(40):
        usage.record({"owner_subject": "u", "owner_org": None, "purpose": "plan", "model_id": "gptoss",
                      "output_tokens": 100, "cost": 0.0})
        if n % 2:
            usage.feedback({"purpose": "plan", "model_id": "gptoss", "kind": "escalation"})
    [row] = [s for s in r.suggestions() if s["model"] == "gptoss"]
    assert row["escalation_rate"] == 0.5 and "consider plan = large" in row["suggestion"]
    r._maybe_adjust(every=0)
    assert r.tier_for("plan") == "large" and r.choose(request("plan")).entry.id == "glm"


# --- in a task: escalation and budgets ---------------------------------------------------------------

def harness_with(r, tmp_path):
    from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
    from agenticwf.agent.runtime.harness.harness import AgentHarness
    from agenticwf.agent.runtime.harness.services import HarnessServices
    from agenticwf.agent.runtime.policy.engine import PolicyEngine
    from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher
    from agenticwf.agent.runtime.tools.registry import ToolRegistry
    from agenticwf.agent.runtime.tools.spec import ToolResult
    from tests.test_runtime import make_system_dir

    system = make_system_dir(tmp_path)
    registry = ToolRegistry(system_dir=system)
    registry.load()
    registry.register_handler("test.read_thing", lambda ctx, args: ToolResult(content="thing"))
    events = []
    harness = AgentHarness(HarnessServices(llm=r, tool_dispatcher=ToolDispatcher(registry, PolicyEngine()),
                                           task_types=TaskTypeResolver(system)),
                           events=lambda subject, event: events.append(event))
    return harness, events


def test_a_task_escalates_when_its_model_keeps_calling_tools_wrong(tmp_path):
    from tests.test_runtime import user

    def answers(model_id, request):
        if model_id in ("gemma", "qwen32"):   # the small model gets the tool's arguments wrong
            return LLMResponse("", [ToolCallRequest(f"c{len(request.messages)}", "read_thing", {"value": 5})],
                               usage=Usage(1000, 50))
        return LLMResponse(f"done by {model_id}", usage=Usage(1000, 50))

    usage = InMemoryUsage()
    r = router({"llm.routing": {"step": "small"}}, answers=answers, usage=usage)
    harness, events = harness_with(r, tmp_path)
    task = asyncio.run(harness.run(harness.create_task(user(), "read the thing").id))
    assert task.result == "done by gptoss" and task.min_tier == "medium" and task.escalations == 1
    escalated = next(e for e in events if e["type"] == "model_escalated")
    assert escalated["model"] == "gemma" and "invalid tool calls" in escalated["evidence"]
    assert any(f["kind"] == "escalation" and f["model_id"] == "gemma" for f in usage.feedback_rows)
    steps = [e for e in events if e["type"] == "step" and e.get("phase") == "completed"]
    assert [s["model"] for s in steps] == ["gemma", "gemma", "gptoss"] and all(s["cost"] > 0 for s in steps)
    assert task.cost == pytest.approx(sum(s["cost"] for s in steps))


def test_a_pinned_task_never_escalates_and_a_task_stops_at_its_budget(tmp_path):
    from tests.test_runtime import user

    def answers(model_id, request):
        return LLMResponse("", [ToolCallRequest(f"c{len(request.messages)}", "read_thing", {"value": 5})],
                           usage=Usage(100000, 1000))   # wrong, and expensive

    r = router({"llm.routing": {"task_budget": "0.05"}}, answers=answers)
    harness, events = harness_with(r, tmp_path)
    task = asyncio.run(harness.run(harness.create_task(user(), "read the thing", model="gemma").id))
    assert task.pinned and task.escalations == 0                               # pinned: kept, never escalated
    assert task.error["code"] == "budget_exceeded" and task.cost >= 0.05
