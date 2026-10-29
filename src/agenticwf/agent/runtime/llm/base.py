"""
LLM: a provider-neutral, stateless model client.

Everything the model should see is in the request, so any worker can make any turn. Provider
adapters (agenticwf.agent.runtime.llm.providers: Anthropic, OpenAI, vLLM, Ollama) implement `LLM.complete` and
translate the neutral messages below to and from their own formats; agenticwf.agent.runtime.llm.router picks the
adapter and model for each request from `LLMRequest.model` ("provider/model", an alias, or
"default").
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolCallRequest:
    """A tool call requested by the model. Only a request: the dispatcher decides what runs."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class Message:
    """
    One message in the neutral format.

    role: "user" | "assistant" | "tool"
      user       content is text from the user, the harness, or the per-turn context block
      assistant  content is the model's text; tool_calls the calls it requested
      tool       the result of one call: tool_call_id, name, content (untrusted data)

    `raw` is what the provider returned for an assistant turn, in its own format (e.g. Anthropic
    content blocks with thinking blocks and their signatures), and `raw_source` names the
    provider that produced it. An adapter replays `raw` verbatim to the same provider, so
    reasoning carries over between steps; other providers ignore it and use the neutral fields.
    Anything that edits history (compaction) drops `raw`.
    """

    role: str
    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    is_error: bool = False
    raw: Any = None
    raw_source: str | None = None


@dataclass
class LLMRequest:
    model: str                     # "provider/model", an alias, or "default" (the router resolves it)
    system: str
    messages: list[Message]
    tools: list[dict[str, Any]] = field(default_factory=list)  # neutral: name, description, input_schema
    max_output_tokens: int | None = None   # None: the adapter's default for its provider
    owner: tuple[str, str | None] | None = None   # whose work it is: the router uses only models they may use
    on_text: Callable[[str], None] | None = None  # streaming callback (message_delta events)
    # Routing (docs/model-routing.md): what the call is for, and what the harness knows of it.
    purpose: str = "step"          # step | plan | split | reply | compact | compact_conversation | consolidate | ...
    pinned: bool = False           # `model` was chosen (message, thread, task type, cron): use it if it fits
    min_tier: str | None = None    # at least this tier (a task type's or cron's, raised by escalation)
    current: str | None = None     # the task's model so far: it keeps its prompt cache
    cached_hint: int = 0           # input tokens likely served from that model's cache
    task_id: str | None = None     # for accounting
    intent_id: str | None = None


@dataclass
class Usage:
    input_tokens: int = 0            # all input, cached or not
    output_tokens: int = 0
    cached_input_tokens: int = 0     # of input_tokens, read from the provider's prompt cache

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += getattr(other, "cached_input_tokens", 0)


@dataclass
class LLMResponse:
    text: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    stop_reason: str = "end_turn"  # end_turn | tool_use | max_tokens | refusal
    usage: Usage = field(default_factory=Usage)
    model: str | None = None       # "provider/model" that served it (the router fills it in)
    raw: Any = None                # the provider's own form of this turn, for replay (Message.raw)
    raw_source: str | None = None
    refusal: dict[str, Any] | None = None   # stop_reason "refusal": category, explanation
    model_id: str | None = None    # the catalog entry that served it (the router fills it in)
    route: str | None = None       # why that one: pinned | purpose | routed | default | context | fallback
    cost: float | None = None      # in the deployment's cost unit

    @property
    def is_final(self) -> bool:
        return not self.tool_calls


class LLMError(Exception):
    """A provider error that persisted after retries; the harness fails the task (model_unavailable).

    `retryable` errors (rate limits, overload, network) may succeed on another route: the router
    tries the route's fallbacks for those, never for a bad request or a missing credential."""

    def __init__(self, message: str, retryable: bool = False, provider: str | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.provider = provider


class LLM(abc.ABC):
    @abc.abstractmethod
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send one request. Retry transient provider errors internally; raise LLMError if they persist."""
