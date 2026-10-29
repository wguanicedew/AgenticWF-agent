"""
Anthropic adapter (Claude), on the official SDK's Messages API, streamed.

    system      one text block with a cache breakpoint: AGENTS.md / WORKFLOW.md are stable per task
    messages    automatic caching (top-level cache_control) of the growing conversation
    assistant   replayed verbatim from Message.raw (text, thinking with signatures, tool_use), so
                the model keeps its reasoning between steps; built from the neutral fields when
                another provider produced the turn
    tool        consecutive results, and the status text after them, form one user message
    thinking    the model's default unless configured ("adaptive"); effort via output_config
    refusals    stop_reason "refusal" ends the turn with a notice; with `fallbacks = default`
                (Claude API only) a declined request is re-run server-side on Anthropic's
                recommended fallback model instead

Thinking blocks record the conversation they were produced in; when the harness had to change
history (compaction, a grown tool set), the API rejects them with a 400 on enforced accounts.
The adapter then strips every thinking block and retries once: the step goes on without the
earlier reasoning rather than failing.
"""

from __future__ import annotations

from typing import Any

import httpx

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse, Message, ToolCallRequest, Usage

LOG = get_logger(__name__)

SOURCE = "anthropic"
_RETRYABLE = {"overloaded_error", "api_error", "rate_limit_error", "timeout_error"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DEFAULT_MAX_TOKENS = 64000          # streamed, so no HTTP timeout concern
_THINKING = ("thinking", "redacted_thinking")


class AnthropicLLM(LLM):
    def __init__(self, api_key: str | None = None, base_url: str | None = None, *,
                 max_tokens: int = DEFAULT_MAX_TOKENS, effort: str | None = None, thinking: str | None = None,
                 thinking_display: str | None = None, fallbacks: str | None = None,
                 timeout: float = 600.0, max_retries: int = 2, client: Any = None, isolated: bool = False):
        """`isolated` (a model a user added): only the given key; nothing from the environment."""
        import anthropic

        self._anthropic = anthropic
        extra: dict[str, Any] = {}
        if isolated and client is None:
            if not api_key:
                raise LLMError("this model has no API key: add one on the admin page", provider=SOURCE)
            # An explicit key stops the SDK reading ANTHROPIC_API_KEY / ANTHROPIC_AUTH_TOKEN / profiles;
            # headers an operator set in the environment are not sent to a user's server either.
            extra["default_headers"] = {name: anthropic.Omit() for name in env_header_names("ANTHROPIC_CUSTOM_HEADERS")}
        # Not isolated, no key: the SDK resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile.
        self.client = client or anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url,
                                                          timeout=timeout, max_retries=max_retries, **extra)
        self.max_tokens = max_tokens
        self.effort = effort
        self.thinking = thinking
        self.thinking_display = thinking_display
        self.fallbacks = fallbacks      # "default" (server-side, Claude API only) or None

    # -- request ------------------------------------------------------------------------------
    def _params(self, request: LLMRequest, strip_thinking: bool = False) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_output_tokens or self.max_tokens,
            "system": [{"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}],
            "messages": to_anthropic_messages(request.messages, strip_thinking),
            "cache_control": {"type": "ephemeral"},   # caches the conversation up to the last block
        }
        if request.tools:
            params["tools"] = [{"name": t["name"], "description": t.get("description", ""),
                                "input_schema": t.get("input_schema") or {"type": "object"},
                                "eager_input_streaming": True}   # the dispatcher validates every input
                               for t in request.tools]
        if self.thinking:
            params["thinking"] = {"type": self.thinking, **({"display": self.thinking_display}
                                                             if self.thinking_display else {})}
        if self.effort:
            params["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            params["fallbacks"] = self.fallbacks
            params["betas"] = [FALLBACK_BETA]
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        anthropic = self._anthropic
        strip, json_retries = False, 0
        while True:
            params = self._params(request, strip_thinking=strip)
            try:
                return await self._stream(params, request)
            except anthropic.BadRequestError as exc:
                if not strip and _thinking_rejected(exc) and _has_thinking(request.messages):
                    LOG.warning("replayed thinking rejected (history changed); retrying without it: %s", exc.message)
                    strip = True
                    continue
                raise LLMError(f"anthropic: {exc.message}", provider=SOURCE) from exc
            except (anthropic.AuthenticationError, anthropic.PermissionDeniedError, anthropic.NotFoundError) as exc:
                raise LLMError(f"anthropic: {exc.message}", provider=SOURCE) from exc
            except anthropic.RateLimitError as exc:
                raise LLMError(f"anthropic: rate limited: {exc.message}", retryable=True, provider=SOURCE) from exc
            except anthropic.APIStatusError as exc:
                # Errors in the middle of a stream arrive with HTTP 200: their type says what happened.
                retryable = exc.status_code >= 500 or exc.status_code == 429 or _error_type(exc) in _RETRYABLE
                raise LLMError(f"anthropic: {exc.status_code} {exc.message}", retryable=retryable,
                               provider=SOURCE) from exc
            except anthropic.APIConnectionError as exc:   # includes timeouts
                raise LLMError(f"anthropic: connection failed: {exc}", retryable=True, provider=SOURCE) from exc
            except anthropic.APIError as exc:   # anything else from the API, e.g. a stream cut short
                raise LLMError(f"anthropic: {exc}", retryable=True, provider=SOURCE) from exc
            except httpx.TransportError as exc:   # the connection dropped mid-stream
                raise LLMError(f"anthropic: connection lost: {exc}", retryable=True, provider=SOURCE) from exc
            except ValueError as exc:
                # A streamed tool input the SDK could not parse at all: re-issue the step (bounded).
                json_retries += 1
                if json_retries > 2:
                    raise LLMError(f"anthropic: unparseable tool input: {exc}", provider=SOURCE) from exc

    async def _stream(self, params: dict[str, Any], request: LLMRequest) -> LLMResponse:
        messages = self.client.beta.messages if "betas" in params else self.client.messages
        async with messages.stream(**params) as stream:
            async for event in stream:
                if event.type == "text" and request.on_text:
                    request.on_text(event.text)
            final = await stream.get_final_message()
        return from_anthropic(final)


# -- conversions (module level, so tests can check them directly) -------------------------------
def to_anthropic_messages(messages: list[Message], strip_thinking: bool = False) -> list[dict[str, Any]]:
    """Neutral messages → Anthropic turns. User text and tool results that follow one another are
    one user turn (all results of a step in a single message, the status text after them)."""
    out: list[dict[str, Any]] = []

    def user_blocks() -> list[dict[str, Any]]:
        if out and out[-1]["role"] == "user":
            return out[-1]["content"]
        out.append({"role": "user", "content": []})
        return out[-1]["content"]

    for m in messages:
        if m.role == "assistant":
            blocks = _replay(m, strip_thinking)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        elif m.role == "tool":
            user_blocks().append({"type": "tool_result", "tool_use_id": m.tool_call_id or "",
                                  "content": m.content or "(empty)", "is_error": m.is_error})
        elif m.content:   # user; empty text blocks are rejected by the API
            user_blocks().append({"type": "text", "text": m.content})
    return out


def _replay(m: Message, strip_thinking: bool) -> list[dict[str, Any]]:
    if m.raw_source == SOURCE and isinstance(m.raw, list):
        blocks = [b for b in m.raw if not (strip_thinking and b.get("type") in _THINKING)]
        if blocks:
            return blocks
    blocks: list[dict[str, Any]] = []
    if m.content:
        blocks.append({"type": "text", "text": m.content})
    blocks += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in m.tool_calls]
    return blocks


def _has_thinking(messages: list[Message]) -> bool:
    return any(m.raw_source == SOURCE and isinstance(m.raw, list)
               and any(b.get("type") in _THINKING for b in m.raw) for m in messages)


def _thinking_rejected(exc: Exception) -> bool:
    text = str(getattr(exc, "message", exc)).lower()
    return "thinking" in text and ("signature" in text or "different conversation" in text)


def _block_dict(block: Any) -> dict[str, Any] | None:
    """The input form of a response block, with only the fields the API accepts back."""
    kind = getattr(block, "type", None)
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return None   # fallback markers and anything unknown are not replayed


def from_anthropic(message: Any) -> LLMResponse:
    content = list(message.content)
    # After a server-side fallback in mid-output, only text from before the last switch point is
    # kept: the declined model's thinking and tool calls are not echoed back (nor run).
    last_switch = max((i for i, b in enumerate(content) if getattr(b, "type", None) == "fallback"), default=-1)
    kept = [b for i, b in enumerate(content) if i > last_switch or getattr(b, "type", None) == "text"]
    blocks = [d for d in (_block_dict(b) for b in kept) if d is not None]

    text = "".join(b["text"] for b in blocks if b["type"] == "text")
    calls = [ToolCallRequest(b["id"], b["name"], b["input"] if isinstance(b["input"], dict) else {})
             for b in blocks if b["type"] == "tool_use"]
    stop = message.stop_reason or "end_turn"
    refusal = None
    if stop == "refusal":
        details = getattr(message, "stop_details", None)
        refusal = {"category": getattr(details, "category", None), "explanation": getattr(details, "explanation", None)}
        calls = []   # a refusal can cut a tool call off mid-input
        text = text or "[The model declined this request" + (f" ({refusal['category']})" if refusal["category"] else "") + ".]"
        blocks = [b for b in blocks if b["type"] != "tool_use"]
    elif stop == "max_tokens" and calls:
        # A truncated tool input parses as a valid partial object: never run it.
        calls = []
        blocks = [b for b in blocks if b["type"] != "tool_use"]
        text = (text + "\n\n" if text else "") + "[Output truncated at the token limit before the tool call was complete.]"

    u = message.usage
    cached = (getattr(u, "cache_read_input_tokens", 0) or 0)
    total_in = (u.input_tokens or 0) + cached + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    return LLMResponse(text=text, tool_calls=calls, stop_reason=stop,
                       usage=Usage(total_in, u.output_tokens or 0, cached),
                       model=f"{SOURCE}/{message.model}", raw=blocks, raw_source=SOURCE, refusal=refusal)


def env_header_names(variable: str) -> list[str]:
    """The header names an SDK would add from its `*_CUSTOM_HEADERS` environment variable."""
    import os

    names = []
    for line in (os.environ.get(variable) or "").split("\n"):
        if ":" in line:
            names.append(line.split(":", 1)[0].strip())
    return [n for n in names if n]


def _error_type(exc: Any) -> str | None:
    """The `error.type` of an API error body (overloaded_error, api_error, …), if there is one."""
    body = getattr(exc, "body", None)
    error = body.get("error") if isinstance(body, dict) else None
    return error.get("type") if isinstance(error, dict) else getattr(exc, "type", None)

