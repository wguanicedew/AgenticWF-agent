"""
ScriptedLLM: a deterministic stand-in for a real model, for tests and demos.

Each step is either a fixed LLMResponse or a function of the request (to react to tool results).
Every request is recorded, so tests can assert what the model was shown.
"""

from __future__ import annotations

from typing import Callable, Union

from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse

Step = Union[LLMResponse, Callable[[LLMRequest], LLMResponse]]


class ScriptedLLM(LLM):
    def __init__(self, steps: list[Step]):
        self.steps = list(steps)
        self.requests: list[LLMRequest] = []

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if not self.steps:
            raise LLMError("script exhausted")
        step = self.steps.pop(0)
        response = step(request) if callable(step) else step
        if request.on_text and response.text:
            request.on_text(response.text)
        return response
