"""
EchoLLM: answers every request with the latest user request, for development and tests of the
service around the runtime (`[llm] provider = echo`). It never calls tools.
"""

from __future__ import annotations

from agenticwf.agent.runtime.llm.base import LLM, LLMRequest, LLMResponse


class EchoLLM(LLM):
    async def complete(self, request: LLMRequest) -> LLMResponse:
        for message in reversed(request.messages):
            text = message.content.strip()
            if message.role == "user" and not text.startswith(("<", "[runtime]")):
                if request.on_text:   # streamed, like a provider would
                    request.on_text(f"echo: {text}")
                return LLMResponse(f"echo: {text}")
        return LLMResponse("echo")
