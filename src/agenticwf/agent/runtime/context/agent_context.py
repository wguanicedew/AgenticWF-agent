"""
AgentContext: one task's working state (the *task context*), and the single place that turns it
into what the model sees (docs/agent-structure.md §7, docs/agent-flow.md §5).

    instructions      preamble, AGENTS.md, WORKFLOW.md (snapshot)   never compacted
    state, mode       mirror of the task's lifecycle state           harness owns transitions
    messages          the task's part: conversation excerpt,         stable for the task
                      results of tasks it depends on, the request
    turns             one TurnContext per turn: goal, inputs, and    old ranges → summaries
                      that turn's steps only
    working_memory    goal, task state, metadata, files examined,    never compacted
                      tests run, plan, notes
    recalled_memories long-term memory recalled at task start
    knowledge_bases   catalogue only; content through kb_search
    step_count / max_steps

Rendering puts stable parts first and changing parts last, so the request prefix stays the same
from step to step (prompt caching). The history is append-only: the per-step status (step count,
working memory when it changed) is appended once per step and stays, rather than being rebuilt
at the end of every request. Providers that bind the model's reasoning to the conversation
(Anthropic thinking blocks) can then replay it, and every provider's prefix cache stays warm. A step sees the task's messages and the active turn's, never
another turn's steps: earlier turns reach it as their results, in the turn brief. The AuthContext
is not here: it lives on the Task record.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agenticwf.agent.runtime.context.memory import KnowledgeBaseInfo, MemoryItem
from agenticwf.agent.runtime.context.task_types import PACKAGED_PROMPTS, ResolvedTaskType
from agenticwf.agent.runtime.llm.base import LLMResponse, Message, Usage
from agenticwf.agent.runtime.task import ToolCall
from agenticwf.agent.runtime.types import Mode, RuntimeState

# The runtime's rules for the model (prompts/preamble.md). This is the packaged text; operators may
# replace it with <system dir>/prompts/preamble.md (ContextManager.build), task types may not.
PREAMBLE = (PACKAGED_PROMPTS / "preamble.md").read_text().strip() + "\n"


def _untrusted(kind: str, text: str) -> str:
    return f"<{kind} untrusted=\"true\">\n{text}\n</{kind}>"


@dataclass
class FileEntry:
    path: str
    version: str | None = None
    edited: bool = False
    note: str = ""
    step: int = 0           # last step that read or changed it
    edited_step: int = -1   # last step that changed it


@dataclass
class TestRun:
    command: str
    passed: int = 0
    failed: int = 0
    failing: list[str] = field(default_factory=list)
    step: int = 0

    @property
    def success(self) -> bool:
        return self.failed == 0


@dataclass
class WorkingMemory:
    goal: str = ""
    task_state: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    files_examined: list[FileEntry] = field(default_factory=list)
    tests_run: list[TestRun] = field(default_factory=list)
    phase: str = ""
    plan: list[dict[str, str]] = field(default_factory=list)   # [{"text", "status"}]
    notes: list[str] = field(default_factory=list)
    max_entries: int = 20   # TODO: a token budget (proposal: 4,000) rather than an entry count

    # --- written by the harness from tool results ------------------------------------------------
    def apply_facts(self, facts: dict[str, Any], step: int) -> None:
        for item in facts.get("files") or ():
            entry = self._file(item["path"])
            entry.version = item.get("version", entry.version)
            entry.step = step
            if item.get("note"):
                entry.note = item["note"]
            if item.get("edited"):
                entry.edited, entry.edited_step = True, step
        if tests := facts.get("tests"):
            self.tests_run.append(TestRun(tests.get("command", ""), int(tests.get("passed", 0)),
                                          int(tests.get("failed", 0)), list(tests.get("failing") or ()), step))

    def _file(self, path: str) -> FileEntry:
        for entry in self.files_examined:
            if entry.path == path:
                return entry
        entry = FileEntry(path)
        self.files_examined.append(entry)
        return entry

    def is_stale(self, run: TestRun) -> bool:
        """A test run is stale when a file was edited in a later step."""
        return any(f.edited_step > run.step for f in self.files_examined)

    def verified(self) -> bool:
        """The last test run passed and no file changed since (for an on_complete hook)."""
        return bool(self.tests_run) and self.tests_run[-1].success and not self.is_stale(self.tests_run[-1])

    # --- written by the model through update_working_memory / update_plan ------------------------
    def update(self, goal: str | None = None, task_state: str | None = None, add_note: str | None = None) -> None:
        if goal:
            self.goal = goal
        if task_state:
            self.task_state = task_state
        if add_note:
            self.notes.append(add_note)

    def set_plan(self, phase: str | None, steps: list[dict[str, str]] | None) -> None:
        if phase:
            self.phase = phase
        if steps is not None:
            self.plan = [{"text": s["text"], "status": s.get("status", "pending")} for s in steps]

    def render(self) -> str:
        lines = ["## Working memory", f"Goal: {self.goal or '(not set)'}"]
        if self.task_state:
            lines.append(f"Task state: {self.task_state}")
        if self.metadata:
            lines.append("Metadata: " + ", ".join(f"{k}={v}" for k, v in self.metadata.items()))
        if self.phase or self.plan:
            lines.append(f"Plan (phase: {self.phase or '-'}):")
            lines += [f"  [{s['status']}] {s['text']}" for s in self.plan]
        files = self.files_examined[-self.max_entries:]
        if files:
            hidden = len(self.files_examined) - len(files)
            lines.append("Files examined" + (f" (+{hidden} earlier)" if hidden else "") + ":")
            for f in files:
                tag = "edited" if f.edited else "read"
                lines.append(f"  {f.path} [{tag}, step {f.step}{', v=' + f.version if f.version else ''}]"
                             + (f" - {f.note}" if f.note else ""))
        if self.tests_run:
            lines.append("Tests run:")
            for t in self.tests_run[-self.max_entries:]:
                state = "passed" if t.success else f"FAILED ({', '.join(t.failing[:5])})"
                stale = ", stale: files edited since" if self.is_stale(t) else ""
                lines.append(f"  step {t.step}: {t.command or 'tests'} - {t.passed} passed, {t.failed} failed, {state}{stale}")
        if self.notes:
            lines.append("Notes:")
            lines += [f"  - {n}" for n in self.notes[-self.max_entries:]]
        return "\n".join(lines)


@dataclass
class TurnContext:
    """What one turn adds to the task context (docs/agent-flow.md §5)."""

    turn_id: str
    goal: str
    phase: str = ""
    checks: list[str] = field(default_factory=list)       # the plan item's verify checks, as text
    inputs: list[str] = field(default_factory=list)       # results of the turns it depends on
    position: tuple[int, int] = (1, 1)                    # (n, of)
    messages: list[Message] = field(default_factory=list)

    def brief(self) -> str:
        lines = [f"<turn id=\"{self.turn_id}\" n=\"{self.position[0]}\" of=\"{self.position[1]}\">",
                 f"Goal of this turn: {self.goal}"]
        if self.phase:
            lines.append(f"WORKFLOW.md phase: {self.phase}")
        if self.checks:
            lines.append("Done when: " + "; ".join(self.checks))
        if self.inputs:
            lines.append("Results of earlier turns:")
            lines += [f"- {i}" for i in self.inputs]
        lines.append("Call complete_turn with a summary when the goal of this turn is reached.")
        lines.append("</turn>")
        return "\n".join(lines)


@dataclass
class AgentContext:
    task_id: str
    instructions: ResolvedTaskType
    max_steps: int
    state: RuntimeState = RuntimeState.QUEUED
    mode: Mode = Mode.DEFAULT
    preamble: str = PREAMBLE
    messages: list[Message] = field(default_factory=list)
    working_memory: WorkingMemory = field(default_factory=WorkingMemory)
    recalled_memories: list[MemoryItem] = field(default_factory=list)
    knowledge_bases: list[KnowledgeBaseInfo] = field(default_factory=list)
    step_count: int = 0
    wrap_up_at: int = 3
    usage: Usage = field(default_factory=Usage)
    last_memory: str = ""          # working memory as last appended to the history
    turns: dict[str, TurnContext] = field(default_factory=dict)
    active_turn: str | None = None
    multi_turn: bool = False      # render the turn brief (a plan with more than one item)
    skill: str = ""               # the running skill's instructions (its <skill> block), if any
    extra_instructions: list[tuple[str, str]] = field(default_factory=list)   # (title, text) after WORKFLOW.md (a cron's, a check's)

    @property
    def turn(self) -> TurnContext | None:
        return self.turns.get(self.active_turn) if self.active_turn else None

    @property
    def _live(self) -> list[Message]:
        """Where new messages go: the active turn's list, or the task's before any turn starts."""
        turn = self.turn
        return turn.messages if turn is not None else self.messages

    def all_messages(self) -> list[Message]:
        """The task's messages, then each turn's, in order (for inspection; never rendered whole)."""
        out = list(self.messages)
        for turn in self.turns.values():
            out += turn.messages
        return out

    @property
    def steps_left(self) -> int:
        """Steps not started yet (the current step is not counted)."""
        return self.max_steps - self.step_count

    # --- rendering -----------------------------------------------------------------------------
    def render(self) -> tuple[str, list[Message]]:
        """(system, messages) for the LLM request. Tool schemas are added by the harness."""
        parts = [self.preamble,
                 "# AGENTS.md\n\n" + self.instructions.agents.content,
                 "# WORKFLOW.md\n\n" + self.instructions.workflow.content,
                 *(f"# {title}\n\n{text}" for title, text in getattr(self, "extra_instructions", ()))]
        if self.mode is not Mode.DEFAULT:
            parts.append(f"Task mode: {self.mode.value}.")
        if self.knowledge_bases:   # catalogue only
            parts.append("# Knowledge bases (search with the kb_* tools)\n" + "\n".join(
                f"- {kb.id}: {kb.name} - {kb.description}" for kb in self.knowledge_bases))
        if self.recalled_memories:
            parts.append(_untrusted("recalled_memories", "Notes from earlier tasks; they may be outdated.\n" +
                                    "\n".join(f"- {m.text}" for m in self.recalled_memories)))
        system = "\n\n".join(parts)

        messages = list(self.messages)
        turn = self.turn
        if turn is not None:
            if self.multi_turn:
                messages.append(Message("user", turn.brief()))   # fixed for the whole turn
            messages += turn.messages
        return system, messages

    def append_status(self) -> None:
        """This step's status, appended to the history (it stays: append-only). Provider adapters
        merge it into the user turn that carries the previous step's tool results."""
        memory = self.working_memory.render()
        status = [memory if memory != self.last_memory else "(working memory unchanged)",
                  f"Step {self.step_count} of {self.max_steps}."]
        self.last_memory = memory
        turn = self.turn
        if self.multi_turn and turn is not None:
            status.append(f"Turn {turn.position[0]} of {turn.position[1]}: {turn.goal}")
        remaining = self.steps_left + 1   # including the current step
        if remaining <= self.wrap_up_at:
            status.append(f"Only {remaining} step(s) left, including this one: wrap up and report what is "
                          "done and what remains.")
        self._live.append(Message("user", "<task_status>\n" + "\n".join(status) + "\n</task_status>"))

    # --- updates -------------------------------------------------------------------------------
    def begin_step(self) -> int:
        self.step_count += 1
        return self.step_count

    def append_user(self, text: str) -> None:
        self._live.append(Message("user", text))

    def append_step(self, response: LLMResponse) -> None:
        self._live.append(Message("assistant", response.text, tool_calls=list(response.tool_calls),
                                  raw=response.raw, raw_source=response.raw_source))
        self.usage.add(response.usage)

    def append_results(self, calls: list[ToolCall]) -> None:
        """One tool message per call (results in call order), and their facts into working memory."""
        for call in calls:
            result = call.result
            content = "" if result is None else (result.content if isinstance(result.content, str) else str(result.content))
            if result and result.is_error:
                content = f"[{result.error_code}] {content}"
            if result is not None and result.instructions and not result.is_error:
                self.skill = content   # kept through compaction while the skill runs
            else:
                content = _untrusted("tool_result", content)
            self._live.append(Message("tool", content, tool_call_id=call.id,
                                      name=call.name, is_error=bool(result and result.is_error)))
            if result and result.facts:
                self.working_memory.apply_facts(result.facts, call.step)

    # --- compaction ----------------------------------------------------------------------------
    def size(self) -> int:
        """Rough token count of the rendered context (4 characters per token)."""
        system, messages = self.render()
        return (len(system) + sum(len(m.content) + len(str(m.tool_calls)) for m in messages)) // 4

    def compactable_range(self, keep_recent: int) -> tuple[int, int] | None:
        """Oldest messages of the active turn that can be summarized, never splitting a tool call from its results."""
        live = self._live
        end = len(live) - keep_recent
        while end > 0 and live[end].role == "tool":
            end -= 1
        return (0, end) if end > 1 else None

    def compactable_messages(self, start: int, end: int) -> list[Message]:
        return self._live[start:end]

    def replace_with_summary(self, start: int, end: int, summary: str) -> None:
        # The kept turns were produced with the full history present: their provider-bound
        # reasoning (raw) no longer matches, so they go back in their neutral form.
        for message in self._live[end:]:
            message.raw = message.raw_source = None
        self.last_memory = ""   # the summary replaces the last full working memory
        text = f"<summary of earlier steps>\n{summary}\n</summary of earlier steps>"
        if self.skill:   # the running skill's instructions are never summarised away
            text += f"\n\nThe skill you are using:\n{self.skill}"
        self._live[start:end] = [Message("user", text)]
