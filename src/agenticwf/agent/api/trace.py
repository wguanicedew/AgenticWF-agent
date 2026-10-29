"""The trace view of an intent (docs/api.md, Trace): what the agent did for one message or task, from
the records the runtime keeps anyway.

    intent        the request, its status, the task type and model asked for
    tasks         what intake made of it; each with the memories recalled at its start, its plan and turns, each turn with its steps (one
                  model call each: model, route, tokens, cost, time, the text it wrote) and the tool
                  calls of that step (arguments, decision, result, time)
    model_calls   every model call of the intent and its tasks: intake, planning, steps, compaction,
                  the reply (agent_model_usage)
    totals        tokens, cost, calls
    traces        the OpenTelemetry traces the work ran in, linked with AGENT_TRACE_URL

Tool arguments and results are as stored: results are redacted and capped when the call returns
(ToolDispatcher._normalize); here both are clipped again for the page. Only the owner sees them.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Any

from agenticwf.agent.runtime.records import Intent
from agenticwf.agent.runtime.task import Task, ToolCall

CLIP = 4000   # characters of a tool's arguments or result, of a step's text


def intent_trace(intent: Intent, tasks: list[Task], reply: Any, usage_rows: list[dict], *,
                 trace_url: str = "", cost_unit: str = "units") -> dict[str, Any]:
    traces: dict[str, str | None] = {}

    def link(trace_id: str | None) -> str | None:
        if not trace_id:
            return None
        url = trace_url.replace("{trace_id}", trace_id) if trace_url else None
        traces.setdefault(trace_id, url)
        return url

    model_calls = [{"at": _epoch(r.get("at")), "purpose": r.get("purpose"), "model_id": r.get("model_id"),
                    "provider": r.get("provider"), "route": r.get("route"), "task_id": r.get("task_id"),
                    "input_tokens": r.get("input_tokens", 0), "cached_tokens": r.get("cached_tokens", 0),
                    "output_tokens": r.get("output_tokens", 0), "cost": r.get("cost", 0.0),
                    "seconds": r.get("seconds"), "trace_id": r.get("trace_id"), "trace_url": link(r.get("trace_id"))}
                   for r in usage_rows]
    views = [_task(t, link) for t in tasks]
    tool_calls = sum(len(t.calls) for t in tasks)
    reply_view = None
    if reply is not None:
        reply_view = {"status": reply.status, "answer": _clip(reply.answer), "artifacts": list(reply.artifacts)}
    return {
        "intent": {"id": intent.id, "text": _clip(intent.text), "status": intent.status.value,
                   "received_at": intent.received_at, "task_type": intent.requested_task_type,
                   "model": intent.model, "source": intent.source},
        "reply": reply_view,
        "tasks": views,
        "model_calls": model_calls,
        "totals": {"model_calls": len(model_calls), "tool_calls": tool_calls,
                   "input_tokens": sum(c["input_tokens"] for c in model_calls),
                   "cached_tokens": sum(c["cached_tokens"] for c in model_calls),
                   "output_tokens": sum(c["output_tokens"] for c in model_calls),
                   "cost": round(sum(c["cost"] or 0.0 for c in model_calls), 6), "cost_unit": cost_unit},
        "traces": [{"trace_id": k, "url": v} for k, v in traces.items()],
    }


def _task(task: Task, link) -> dict[str, Any]:
    calls = {c.id: c for c in task.calls}
    placed: set[str] = set()
    turns = []
    for turn in task.turns:
        steps = []
        for step in turn.steps:
            step_calls = [calls[i] for i in step.tool_calls if i in calls]
            placed.update(c.id for c in step_calls)
            trace = getattr(step, "trace_id", None)
            steps.append({"number": step.number, "started_at": getattr(step, "started_at", None),
                          "seconds": getattr(step, "seconds", None), "model_id": getattr(step, "model_id", None),
                          "route": getattr(step, "route", None), "cost": getattr(step, "cost", 0.0),
                          "input_tokens": step.usage.input_tokens, "output_tokens": step.usage.output_tokens,
                          "cached_tokens": getattr(step.usage, "cached_input_tokens", 0), "text": _clip(step.text),
                          "trace_id": trace, "trace_url": link(trace),
                          "tool_calls": [_call(c, link) for c in step_calls]})
        turns.append({"id": turn.id, "goal": turn.item.goal, "phase": turn.item.phase, "state": turn.state.value,
                      "attempts": turn.attempts, "tools": list(turn.tools),
                      "summary": turn.result.summary if turn.result else None, "verification": turn.verification,
                      "steps": steps})
    return {"id": task.id, "task_type": task.task_type.requested, "state": task.state.value,
            "status": task.state.public_status, "created_at": task.created_at, "depends_on": task.depends_on,
            "input": _clip(task.input), "mode": task.mode.value, "model_id": task.model_id, "cost": task.cost,
            "escalations": task.escalations, "result": _clip(task.result) if task.result else None, "error": task.error,
            "plan": [{"id": i.id, "goal": i.goal, "phase": i.phase, "depends_on": i.depends_on} for i in task.plan],
            "plan_revisions": len(task.plan_revisions), "turns": turns, "memories": list(task.recalled),
            "tool_calls": [_call(c, link) for c in task.calls if c.id not in placed]}   # outside any recorded step


def _call(call: ToolCall, link) -> dict[str, Any]:
    result = call.result
    trace = getattr(call, "trace_id", None)
    return {"id": call.id, "name": call.name, "step": call.step, "status": call.status.value,
            "decision": call.decision, "reason": call.reason, "arguments": _clip(_json(call.arguments)),
            "started_at": getattr(call, "started_at", None), "seconds": getattr(call, "seconds", None),
            "result": None if result is None else {
                "content": _clip(result.content if isinstance(result.content, str) else _json(result.content)),
                "is_error": result.is_error, "error_code": result.error_code},
            "trace_id": trace, "trace_url": link(trace)}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=1, default=str)


def _clip(text: str | None) -> str:
    text = text or ""
    return text if len(text) <= CLIP else text[:CLIP] + f"\n… ({len(text) - CLIP} more characters)"


def _epoch(value: Any) -> float | None:
    if isinstance(value, dt.datetime):
        return (value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)).timestamp()
    return value
