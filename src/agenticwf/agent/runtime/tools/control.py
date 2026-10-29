"""
Control tools: always active, handled by the harness rather than a handler, because they act on
the task itself (AgentContext, lifecycle, active tool set). The dispatcher recognises them by
HandlerKind.CONTROL and hands them back to the harness.

Their wording is in Markdown, one file each (runtime/defaults/core-tools/<name>.md): front
matter with `summary` and `input_schema`, the body the description the model reads.
"""

from __future__ import annotations

from agenticwf.agent.runtime.tools.loader import core_tool_text
from agenticwf.agent.runtime.tools.spec import Approval, Effects, HandlerKind, HandlerRef, ToolSource, ToolSpec

def _control(name: str) -> ToolSpec:
    summary, description, schema = core_tool_text(name)
    return ToolSpec(name=name, summary=summary, description=description, input_schema=schema,
                    handler=HandlerRef(HandlerKind.CONTROL), source=ToolSource("core", f"core-tools/{name}.md"),
                    effects=Effects.NONE, approval=Approval.AUTO, idempotent=True)


UPDATE_PLAN = _control("update_plan")
UPDATE_WORKING_MEMORY = _control("update_working_memory")
ASK_USER = _control("ask_user")
SEARCH_TOOLS = _control("search_tools")
COMPLETE_TURN = _control("complete_turn")
READ_SKILL_FILE = _control("read_skill_file")
REPORT_ISSUE = _control("report_issue")
REPORT_CHECK = _control("report_check")

# Ordered: pinned tools come first in every request, in this order (stable prefix for caching).
PINNED_CONTROL_TOOLS = (UPDATE_PLAN, UPDATE_WORKING_MEMORY, ASK_USER, SEARCH_TOOLS)
CONTROL_TOOL_NAMES = tuple(t.name for t in PINNED_CONTROL_TOOLS)
# Offered only in the turns of a multi-turn plan (docs/agent-flow.md §6).
TURN_TOOL_NAMES = (COMPLETE_TURN.name,)
# Offered only while a skill with files runs (agenticwf.agent.runtime.skills).
SKILL_TOOL_NAMES = (READ_SKILL_FILE.name,)
# Offered only in a cron's check tasks (agenticwf.agent.runtime.cron).
CRON_TOOL_NAMES = (REPORT_ISSUE.name, REPORT_CHECK.name)
CONTROL_TOOLS = (*PINNED_CONTROL_TOOLS, COMPLETE_TURN, READ_SKILL_FILE, REPORT_ISSUE, REPORT_CHECK)
