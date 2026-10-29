"""
OpenAI-compatible adapter: OpenAI, vLLM and Ollama, through the Chat Completions API (streamed,
function tools). One code path; the presets differ in base URL, credentials and the name of the
output-token limit.

    openai   https://api.openai.com/v1 (the SDK default), OPENAI_API_KEY, max_completion_tokens
    vllm     your server, e.g. http://gpu01:8000/v1 (`vllm serve … --enable-auto-tool-choice
             --tool-call-parser <parser>` for tools), VLLM_API_KEY if it requires one, max_tokens
    ollama   http://localhost:11434/v1, no key, max_tokens

Prompt caching needs nothing here: OpenAI caches prompt prefixes automatically and vLLM / Ollama
reuse their prefix KV cache, so the harness's append-only history is what keeps them warm.
`extra` passes provider-specific request fields through (e.g. {"reasoning_effort": "high"}).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import httpx

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse, Message, ToolCallRequest, Usage

LOG = get_logger(__name__)


@dataclass(frozen=True)
class Preset:
    base_url: str | None
    api_key_env: str | None
    default_key: str | None      # sent when the server needs none (the SDK requires a value)
    token_param: str


PRESETS = {
    "openai": Preset(None, "OPENAI_API_KEY", None, "max_completion_tokens"),
    "vllm": Preset(None, "VLLM_API_KEY", "EMPTY", "max_tokens"),
    "ollama": Preset("http://localhost:11434/v1", None, "ollama", "max_tokens"),
}


class OpenAICompatibleLLM(LLM):
    def __init__(self, provider: str = "openai", api_key: str | None = None, base_url: str | None = None, *,
                 max_tokens: int | None = None, token_param: str | None = None, stream_usage: bool = True,
                 extra: dict[str, Any] | None = None, timeout: float = 600.0, max_retries: int = 2,
                 client: Any = None, isolated: bool = False):
        """`isolated` (a model a user added): only the given key; nothing from the environment."""
        import os

        import openai

        preset = PRESETS.get(provider, PRESETS["openai"])
        self._openai = openai
        self.provider = provider
        base_url = base_url or preset.base_url
        if provider == "vllm" and not base_url and client is None:
            raise LLMError("vllm: base_url is required (e.g. http://host:8000/v1)", provider=provider)
        if isolated:
            key = api_key or preset.default_key
            if not key and client is None:
                raise LLMError("this model has no API key: add one on the admin page", provider=provider)
        else:
            key = api_key or (os.environ.get(preset.api_key_env) if preset.api_key_env else None) or preset.default_key
        isolation: dict[str, Any] = {}
        if isolated:   # an explicit key, no admin key, no organization/project or custom headers from the environment
            from agenticwf.agent.runtime.llm.providers.anthropic import env_header_names

            isolation = {"admin_api_key": "",
                         "default_headers": {n: openai.Omit() for n in env_header_names("OPENAI_CUSTOM_HEADERS")}}
        self.client = client or openai.AsyncOpenAI(api_key=key, base_url=base_url, timeout=timeout,
                                                   max_retries=max_retries, **isolation)
        if isolated and client is None:
            self.client.organization = self.client.project = None
        self.max_tokens = max_tokens
        self.token_param = token_param or preset.token_param
        self.stream_usage = stream_usage
        self.extra = dict(extra or {})

    def _params(self, request: LLMRequest) -> dict[str, Any]:
        params: dict[str, Any] = {"model": request.model, "messages": to_openai_messages(request.system, request.messages),
                                  "stream": True, **self.extra}
        if self.stream_usage:
            params["stream_options"] = {"include_usage": True}
        limit = request.max_output_tokens or self.max_tokens
        if limit:
            params[self.token_param] = limit
        if request.tools:
            params["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}} for t in request.tools]
        return params

    async def complete(self, request: LLMRequest) -> LLMResponse:
        openai = self._openai
        try:
            return await self._stream(self._params(request), request)
        except (openai.AuthenticationError, openai.PermissionDeniedError, openai.NotFoundError,
                openai.BadRequestError, openai.UnprocessableEntityError) as exc:
            raise LLMError(f"{self.provider}: {exc.message}", provider=self.provider) from exc
        except openai.RateLimitError as exc:
            raise LLMError(f"{self.provider}: rate limited: {exc.message}", retryable=True, provider=self.provider) from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"{self.provider}: {exc.status_code} {exc.message}", retryable=exc.status_code >= 500,
                           provider=self.provider) from exc
        except openai.APIConnectionError as exc:   # includes timeouts
            raise LLMError(f"{self.provider}: connection failed: {exc}", retryable=True, provider=self.provider) from exc
        except openai.APIError as exc:   # e.g. an {"error": …} chunk in the middle of the stream
            raise LLMError(f"{self.provider}: {exc}", retryable=True, provider=self.provider) from exc
        except httpx.TransportError as exc:   # the connection dropped mid-stream
            raise LLMError(f"{self.provider}: connection lost: {exc}", retryable=True, provider=self.provider) from exc

    async def _stream(self, params: dict[str, Any], request: LLMRequest) -> LLMResponse:
        text: list[str] = []
        calls: dict[int, dict[str, str]] = {}
        finish, usage, served = None, Usage(), request.model
        stream = await self.client.chat.completions.create(**params)
        async for chunk in stream:
            served = getattr(chunk, "model", None) or served
            if getattr(chunk, "usage", None):
                u = chunk.usage
                details = getattr(u, "prompt_tokens_details", None)
                usage = Usage(u.prompt_tokens or 0, u.completion_tokens or 0,
                              (getattr(details, "cached_tokens", 0) or 0) if details else 0)
            for choice in chunk.choices or ():
                delta = choice.delta
                if delta is not None and delta.content:
                    text.append(delta.content)
                    if request.on_text:
                        request.on_text(delta.content)
                for tc in (delta.tool_calls or ()) if delta is not None else ():
                    entry = calls.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                    if tc.id:
                        entry["id"] = tc.id
                    if tc.function is not None:
                        entry["name"] += tc.function.name or ""
                        entry["arguments"] += tc.function.arguments or ""
                if choice.finish_reason:
                    finish = choice.finish_reason
        return _response(self.provider, served, "".join(text), [calls[i] for i in sorted(calls)], finish, usage)


def _response(provider: str, model: str, text: str, raw_calls: list[dict[str, str]], finish: str | None,
              usage: Usage) -> LLMResponse:
    stop = {"stop": "end_turn", "tool_calls": "tool_use", "length": "max_tokens",
            "content_filter": "refusal"}.get(finish or "stop", "end_turn")
    calls: list[ToolCallRequest] = []
    for n, c in enumerate(raw_calls):
        try:
            arguments = json.loads(c["arguments"]) if c["arguments"].strip() else {}
        except json.JSONDecodeError:
            arguments = {"_unparsed_arguments": c["arguments"]}   # fails schema validation: an error result
        calls.append(ToolCallRequest(c["id"] or f"call_{n}", c["name"], arguments if isinstance(arguments, dict) else {}))
    refusal = None
    if stop == "refusal":
        calls, refusal = [], {"category": "content_filter", "explanation": None}
        text = text or "[The model's provider filtered this response.]"
    elif stop == "max_tokens" and calls:   # truncated arguments: never run them
        calls = []
        text = (text + "\n\n" if text else "") + "[Output truncated at the token limit before the tool call was complete.]"
    if calls:
        stop = "tool_use"
    return LLMResponse(text=text, tool_calls=calls, stop_reason=stop, usage=usage, model=f"{provider}/{model}",
                       refusal=refusal)


def to_openai_messages(system: str, messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []
    for m in messages:
        if m.role == "assistant":
            entry: dict[str, Any] = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                entry["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                    "name": c.name, "arguments": json.dumps(c.arguments, sort_keys=True)}} for c in m.tool_calls]
            if entry["content"] is None and "tool_calls" not in entry:
                continue
            out.append(entry)
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id or "", "content": m.content or "(empty)"})
        elif m.content:
            out.append({"role": "user", "content": m.content})
    return out
