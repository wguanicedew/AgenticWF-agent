"""
ContextManager: builds the task context from what intake selected, and each turn's context from
the task context (docs/agent-flow.md §5). It also keeps what outlives a task (long-term memory,
knowledge bases). Each level is selected from the one above, never copied whole:

    session context ──(the session selects an excerpt into the intent)──► task context
    task context    ──(plan item, inputs from earlier turns)────────────► turn context
"""

from __future__ import annotations

from dataclasses import dataclass

from agenticwf.agent.runtime.context.agent_context import AgentContext, TurnContext
from agenticwf.agent.runtime.context.memory import (MEMORY_SEARCH, KnowledgeCatalog, MemoryStore, NoKnowledge, NoMemory,
                                    memory_search_handler)
from agenticwf.agent.runtime.harness.planning import first_json
from agenticwf.agent.runtime.llm.base import Message
from agenticwf.agent.runtime.records import PlanItem
from agenticwf.agent.runtime.redact import contains_secret
from agenticwf.agent.runtime.task import Task
from agenticwf.agent.runtime.tools.spec import Handler, ToolSpec
from agenticwf.agent.runtime.types import RuntimeState


@dataclass(frozen=True)
class MemorySettings:
    """`[memory]` in agent.cfg: what a finished task leaves in its owner's long-term memory."""

    consolidate_failed: bool = False   # failed and cancelled tasks too (what they noted may be wrong)
    extract: str = "llm"               # llm: the model picks what lasts; notes: working-memory notes as they are
    model: str | None = None           # the model that extracts; None: the task's
    max_per_task: int = 5


class ContextManager:
    def __init__(self, memory: MemoryStore | None = None, knowledge: KnowledgeCatalog | None = None,
                 input_chars: int = 2000, settings: MemorySettings | None = None):
        self.memory = memory or NoMemory()
        self.knowledge = knowledge or NoKnowledge()
        self.input_chars = input_chars   # how much of one earlier result a later task or turn sees
        self.settings = settings or MemorySettings()

    def build(self, task: Task, conversation: list[Message], max_steps: int, wrap_up_at: int,
              recall_k: int, preamble: str | None = None, recalled: list | None = None) -> AgentContext:
        """The task context: the runtime's rules (`preamble`: prompts/preamble.md as the deployment
        has it), conversation excerpt, the request, recalled memory, knowledge bases."""
        resolved = task.task_type
        ctx = AgentContext(task_id=task.id, instructions=resolved, mode=task.mode,
                           max_steps=max_steps, wrap_up_at=wrap_up_at)
        if preamble:
            ctx.preamble = preamble
        ctx.messages.extend(conversation)
        ctx.append_user(task.input)
        ctx.working_memory.goal = " ".join(task.input.split())[:500]   # the model refines it
        ctx.working_memory.metadata = {"task": task.id, "task_type": resolved.requested, "mode": task.mode.value}
        # `recalled`: looked up already, off the event loop (AgentHarness.intake)
        ctx.recalled_memories = recalled if recalled is not None else self.memory.recall(task.owner, task.input, recall_k)
        ctx.knowledge_bases = self.knowledge.bases(task.auth, resolved.settings.knowledge)
        if hasattr(self.knowledge, "remember_task"):   # kb_search without `kb` searches exactly these
            self.knowledge.remember_task(task.id, ctx.knowledge_bases)
        return ctx

    def add_task_inputs(self, ctx: AgentContext, results: list[tuple[str, str]]) -> None:
        """Results of the tasks this one depends on, as (their request, their answer), before the request."""
        if not results:
            return
        text = "\n\n".join(f"Task: {request}\nResult: {self._clip(answer)}" for request, answer in results)
        ctx.messages.insert(len(ctx.messages) - 1,
                            Message("user", f"<results of earlier tasks>\n{text}\n</results of earlier tasks>"))

    def start_turn(self, ctx: AgentContext, item: PlanItem, inputs: list[tuple[str, str]],
                   position: tuple[int, int]) -> TurnContext:
        """The turn context: the plan item and the results of the turns it depends on, nothing else."""
        turn = TurnContext(turn_id=item.id, goal=item.goal, phase=item.phase,
                           checks=[_describe(v) for v in item.verify],
                           inputs=[f"{turn_id}: {self._clip(summary)}" for turn_id, summary in inputs],
                           position=position)
        ctx.turns[item.id] = turn
        ctx.active_turn = item.id
        ctx.multi_turn = position[1] > 1
        return turn

    def tools(self) -> list[tuple[ToolSpec, Handler]]:
        """Core tools this context offers (ToolRegistry.add_core): memory_search when a memory store
        is configured; kb_search and kb_read when a knowledge service is (agenticwf.agent.runtime.knowledge)."""
        tools = [(MEMORY_SEARCH, memory_search_handler(self.memory))] if self.remembers else []
        if hasattr(self.knowledge, "tools"):
            tools.extend(self.knowledge.tools())
        return tools

    # --- long-term memory ------------------------------------------------------------------------
    @property
    def remembers(self) -> bool:
        """Whether a memory store is configured."""
        return not isinstance(self.memory, NoMemory)

    def should_consolidate(self, state: RuntimeState) -> bool:
        """Whether a task that ended in `state` leaves anything in long-term memory."""
        if not self.remembers or not state.terminal:
            return False
        return state is RuntimeState.COMPLETED or self.settings.consolidate_failed

    def consolidation_input(self, task: Task, ctx: AgentContext) -> str:
        """What the consolidate prompt sees: the request, how the task ended, its answer, its working
        memory, and what is remembered already (so the model does not repeat it)."""
        outcome = task.state.value
        if task.error:
            outcome += f": {task.error.get('message', '')}"
        parts = [f"<request>\n{task.input}\n</request>", f"<outcome>{outcome}</outcome>"]
        if task.result:
            parts.append(f"<answer>\n{self._clip(task.result)}\n</answer>")
        parts.append(f"<working memory>\n{ctx.working_memory.render()}\n</working memory>")
        if ctx.recalled_memories:
            parts.append("<already remembered>\n" + "\n".join(f"- {m.text}" for m in ctx.recalled_memories)
                         + "\n</already remembered>")
        return "\n\n".join(parts)

    @staticmethod
    def parse_memories(text: str) -> list[str] | None:
        """The consolidate prompt's answer, `{"memories": [...]}` or a bare list; None if it does not parse."""
        value = first_json(text)
        if isinstance(value, dict):
            value = value.get("memories")
        if not isinstance(value, list):
            return None
        return [v.strip() for v in value if isinstance(v, str) and v.strip()]

    def remember(self, task: Task, texts: list[str]) -> list[str]:
        """Store these as the owner's memories, leaving out any that look like they hold a secret, at
        most `max_per_task`. Returns what was stored. The store keeps one copy of each text, so a job
        that runs again (after a restart) adds nothing."""
        kept = [t for t in texts if not contains_secret(t)][:self.settings.max_per_task]
        if kept:
            self.memory.remember(task.owner, kept, source_task=task.id)
        return kept

    def _clip(self, text: str) -> str:
        return text if len(text) <= self.input_chars else text[:self.input_chars] + " […]"


def _describe(check: dict) -> str:
    kind = check.get("kind", "")
    if kind == "tool_succeeded":
        return f"{check.get('tool')} succeeded after the last change"
    if kind == "tests_pass":
        return "the last test run passed and nothing changed since"
    return str(check.get("what") or kind)
