#!/usr/bin/env python3
"""
Agent runtime skeleton, end to end, with a scripted model instead of a real LLM.

    python examples/runtime/run_demo.py

Task: find why calculate_total fails its tests, fix it, verify the fix. The scripted model first
tries to finish *before* running the tests; the on_complete hook (deterministic code) reopens the
task because AGENTS.md's "do not finish until verification succeeds" is enforced, not just
requested.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("AGENTICWF_CONFIG", str(HERE / "agent.cfg"))   # agenticwf.agent.core.common.logging

from agenticwf.auth.context import AuthContext  # noqa: E402

from agenticwf.agent.runtime.context.agent_context import AgentContext  # noqa: E402
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver  # noqa: E402
from agenticwf.agent.runtime.harness.harness import AgentHarness, Limits  # noqa: E402
from agenticwf.agent.runtime.harness.services import AgentServices  # noqa: E402
from agenticwf.agent.runtime.llm.base import LLMResponse, ToolCallRequest  # noqa: E402
from agenticwf.agent.runtime.llm.scripted import ScriptedLLM  # noqa: E402
from agenticwf.agent.runtime.policy.engine import PolicyEngine  # noqa: E402
from agenticwf.agent.runtime.task import Task  # noqa: E402
from agenticwf.agent.runtime.tools.dispatcher import ToolDispatcher  # noqa: E402
from agenticwf.agent.runtime.tools.registry import ToolRegistry  # noqa: E402

import demo_handlers  # noqa: E402

SYSTEM_DIR = HERE / "system"


def call(name: str, **arguments) -> ToolCallRequest:
    return ToolCallRequest(id=f"call_{name}_{len(arguments)}_{abs(hash(str(arguments))) % 10_000}",
                           name=name, arguments=arguments)


# What a model might do with these instructions. A real LLM decides this itself.
SCRIPT = [
    LLMResponse("I'll locate calculate_total first.", [
        call("update_plan", phase="RESEARCH", steps=[
            {"text": "Find calculate_total", "status": "in_progress"},
            {"text": "Fix the bug", "status": "pending"},
            {"text": "Run the tests", "status": "pending"}]),
        call("search_code", query="calculate_total")]),
    LLMResponse("", [call("read_file", path="src/payment/service.py")]),
    LLMResponse("It returns a string; the test expects a number.", [
        call("update_working_memory", task_state="bug found: str(sum(items))",
             add_note="calculate_total must return a number"),
        call("update_plan", phase="EXECUTE"),
        call("edit_file", path="src/payment/service.py",
             old_text="return str(sum(items))", new_text="return sum(items)")]),
    # Premature finish: no test run since the edit. The on_complete hook will object.
    LLMResponse("Fixed calculate_total."),
    LLMResponse("Running the tests to verify.", [call("update_plan", phase="VERIFY"), call("run_tests")]),
    LLMResponse("Fixed calculate_total so it returns a number instead of a string. "
                "All 182 tests pass."),
]


def verified_before_done(task: Task, ctx: AgentContext) -> str | None:
    """on_complete hook: enforce AGENTS.md's 'do not finish until verification succeeds'."""
    edited = any(f.edited for f in ctx.working_memory.files_examined)
    if edited and not ctx.working_memory.verified():
        return "files were changed but no passing test run since the last edit. Run the tests."
    return None


def print_event(subject: str, event: dict) -> None:
    kind = event["type"]
    if kind == "tool":
        print(f"  [tool] {event['name']}: {event['status']} {event.get('summary') or ''}".rstrip())
    elif kind == "task":
        print(f"  [task] {event['state']}" + (f" {event['error']}" if event.get("error") else ""))
    elif kind in ("plan", "tools", "approval_required", "input_required", "task_type_fallback",
                  "completion_rejected"):
        print(f"  [{kind}] " + ", ".join(f"{k}={v}" for k, v in event.items() if k not in ("type", "task")))
    elif kind == "step" and event["phase"] == "started":
        print(f"\n===== step {event['step']} =====")


async def main() -> None:
    registry = ToolRegistry(system_dir=SYSTEM_DIR)
    registry.load()
    demo_handlers.register(registry)
    dispatcher = ToolDispatcher(registry, PolicyEngine())
    services = AgentServices(llm=ScriptedLLM(SCRIPT), tool_dispatcher=dispatcher,
                             task_types=TaskTypeResolver(SYSTEM_DIR))
    harness = AgentHarness(services, events=print_event, limits=Limits(max_steps=10),
                           on_complete=[verified_before_done])

    user = AuthContext(subject="user_demo", user_id="user_demo",
                       scopes=frozenset({"app:access", "agent:use", "code:read", "code:write"}))
    task = harness.create_task(user, "Find why calculate_total is failing its tests, fix the bug, "
                                     "and verify the fix.")
    print(f"task {task.id}, active tools: {task.active_tools}")
    task = await harness.run(task.id)

    print("\n" + harness.context(task.id).working_memory.render())
    print(f"\nstate: {task.state.value}\nresult: {task.result}")


if __name__ == "__main__":
    asyncio.run(main())
