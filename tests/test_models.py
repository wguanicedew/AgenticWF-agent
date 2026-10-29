"""Models (docs/agent-models.md): catalog from config + database, router, provider adapters, admin API."""

import asyncio
import json

import httpx2
import pytest
from fastapi.testclient import TestClient

from agenticwf.agent.api.app import create_app
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_catalog
from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse, Message, ToolCallRequest
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry, entries_from_sections
from agenticwf.agent.runtime.llm.router import ModelRouter
from agenticwf.agent.runtime.secrets import SecretBox

from tests.conftest import SECRET, Client, user
from tests.test_runtime import build, make_system_dir, system_dir  # noqa: F401  (fixture)

ADMIN = ("app:access", "agent:use", "admin")


# --- catalog -----------------------------------------------------------------------------------

class MemoryModels:
    def __init__(self, *entries):
        self.rows = {e.id: e for e in entries}

    def list_models(self):
        return list(self.rows.values())

    def save_model(self, entry):
        self.rows[entry.id] = entry

    def delete_model(self, model_id):
        return self.rows.pop(model_id, None) is not None


class MemoryModelsWithDefaults(MemoryModels):
    def __init__(self, *entries):
        super().__init__(*entries)
        self.defaults = {}

    def user_defaults(self):
        return dict(self.defaults)

    def set_user_default(self, owner, model_id):
        if model_id is None:
            self.defaults.pop(owner, None)
        else:
            self.defaults[owner] = model_id


def test_user_default_model():
    alice, bob = ("alice", None), ("bob", None)
    store = MemoryModelsWithDefaults(ModelEntry("mine", "echo", "m", source="database", owner_subject="alice",
                                                visibility="user"))
    catalog = ModelCatalog([ModelEntry("opus", "anthropic", "claude-opus-5-5"), ModelEntry("echo", "echo", "echo")],
                           default="opus", store=store)
    catalog.set_user_default(alice, "mine")
    assert catalog.default_for(alice) == "mine" and catalog.resolve(None, alice).id == "mine"
    assert catalog.resolve("gone", alice).id == "mine"                 # an unknown id: their default
    assert catalog.default_for(bob) == "opus" and catalog.resolve(None).id == "opus"   # others, the runtime: [llm] default
    with pytest.raises(ValueError):
        catalog.set_user_default(bob, "mine")                          # not shared with bob
    store.rows.pop("mine")
    catalog._refresh(force=True)
    assert catalog.user_default(alice) is None and catalog.resolve(None, alice).id == "opus"   # gone: [llm] default
    catalog.set_user_default(alice, "echo")
    catalog.set_user_default(alice, None)
    assert catalog.default_for(alice) == "opus" and store.defaults == {}
    with pytest.raises(PermissionError):
        ModelCatalog([ModelEntry("opus", "anthropic", "x")], store=MemoryModels()).set_user_default(alice, "opus")


def test_config_sections_to_entries():
    models, providers = entries_from_sections({
        "llm.model.opus": {"provider": "anthropic", "model": "claude-opus-5-5", "label": "Opus", "effort": "high",
                           "max_tokens": "32000"},
        "llm.model.qwen": {"provider": "vllm", "model": "Qwen/Qwen3-32B", "base_url": "http://gpu:8000/v1",
                           "extra": '{"temperature": 0.2}', "enabled": "false"},
        "llm.provider.anthropic": {"api_key_env": "MY_KEY", "max_retries": "4"},
    })
    opus, qwen = models
    assert opus.options == {"effort": "high", "max_tokens": 32000} and opus.label == "Opus"
    assert qwen.options == {"extra": {"temperature": 0.2}} and not qwen.enabled
    assert providers == {"anthropic": {"api_key_env": "MY_KEY", "max_retries": 4}}


def test_catalog_merges_config_and_database():
    store = MemoryModels(ModelEntry("local", "ollama", "qwen3", source="database"),
                         ModelEntry("opus", "openai", "x", source="database"))          # clashes: ignored
    catalog = ModelCatalog([ModelEntry("opus", "anthropic", "claude-opus-5-5")], default="opus", store=store)
    assert [e.id for e in catalog.entries()] == ["opus", "local"]
    assert catalog.get("opus").provider == "anthropic"
    assert catalog.resolve(None).id == catalog.resolve("default").id == "opus"
    assert catalog.resolve("local").id == "local"
    assert catalog.resolve("gone").id == "opus"                    # removed later: the default runs it

    with pytest.raises(PermissionError):
        catalog.put(ModelEntry("opus", "anthropic", "y"))          # configured models are read-only
    with pytest.raises(ValueError):
        catalog.put(ModelEntry("Bad Id", "openai", "x"))
    with pytest.raises(ValueError):
        catalog.put(ModelEntry("x", "nope", "x"))
    catalog.put(ModelEntry("gpt", "openai", "gpt-x", enabled=False))
    assert not catalog.is_available("gpt") and "gpt" in [e.id for e in catalog.entries(include_disabled=True)]
    assert catalog.delete("gpt") and not catalog.delete("gpt")


def test_legacy_provider_echo():
    catalog = build_catalog(RuntimeConfig(system_dir=".", provider="echo"))
    assert [(e.id, e.provider) for e in catalog.entries()] == [("echo", "echo")] and catalog.default_id == "echo"


# --- router ------------------------------------------------------------------------------------

class Fixed(LLM):
    def __init__(self, name, error=None):
        self.name, self.error, self.models = name, error, []

    async def complete(self, request):
        self.models.append(request.model)
        if self.error:
            raise self.error
        return LLMResponse(self.name)


def router_with(adapters, fallbacks=()):
    entries = [ModelEntry(i, "echo", f"{i}-model") for i in adapters]
    catalog = ModelCatalog(entries, default=entries[0].id, fallbacks=list(fallbacks))
    built = []

    def factory(entry, settings):
        built.append(entry.id)
        return adapters[entry.id]

    return ModelRouter(catalog, factory), built


def ask(router, model="default"):
    return asyncio.run(router.complete(LLMRequest(model=model, system="", messages=[Message("user", "hi")])))


def test_router_resolves_and_uses_provider_model_names():
    a, b = Fixed("A"), Fixed("B")
    router, built = router_with({"a": a, "b": b})
    assert ask(router).text == "A" and ask(router, "b").text == "B" and ask(router, "b").text == "B"
    assert a.models == ["a-model"] and b.models == ["b-model", "b-model"]
    assert built == ["a", "b"]                                     # one adapter per entry, reused


def test_router_falls_back_on_retryable_errors_only():
    down = Fixed("A", LLMError("overloaded", retryable=True))
    router, _ = router_with({"a": down, "b": Fixed("B")}, fallbacks=["b"])
    assert ask(router).text == "B"
    bad = Fixed("A", LLMError("invalid request"))
    router, _ = router_with({"a": bad, "b": Fixed("B")}, fallbacks=["b"])
    with pytest.raises(LLMError, match="invalid request"):
        ask(router)


def test_router_rebuilds_an_adapter_when_its_entry_changes():
    store = MemoryModels()
    catalog = ModelCatalog([], store=store, refresh_seconds=0)
    built = []
    router = ModelRouter(catalog, lambda entry, settings: built.append(entry.model) or Fixed(entry.model))
    catalog.put(ModelEntry("m", "echo", "v1"))
    assert ask(router, "m").text == "v1"
    catalog.put(ModelEntry("m", "echo", "v2"))
    assert ask(router, "m").text == "v2" and built == ["v1", "v2"]


# --- Anthropic adapter (the real SDK, a mocked HTTP transport) ---------------------------------

def sse(*events):
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


def anthropic_stream(blocks, stop="end_turn", model="claude-opus-5-5"):
    events = [{"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": model, "content": [],
        "stop_reason": None, "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 0}}}]
    for i, block in enumerate(blocks):
        if block["type"] == "thinking":
            events += [{"type": "content_block_start", "index": i, "content_block": {"type": "thinking", "thinking": "", "signature": ""}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "thinking_delta", "thinking": block["thinking"]}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "signature_delta", "signature": block["signature"]}}]
        elif block["type"] == "text":
            events += [{"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "text_delta", "text": block["text"]}}]
        elif block["type"] == "tool_use":
            events += [{"type": "content_block_start", "index": i, "content_block": {"type": "tool_use", "id": block["id"], "name": block["name"], "input": {}}},
                       {"type": "content_block_delta", "index": i, "delta": {"type": "input_json_delta", "partial_json": json.dumps(block["input"])}}]
        events.append({"type": "content_block_stop", "index": i})
    events += [{"type": "message_delta", "delta": {"type": "message_delta", "stop_reason": stop, "stop_sequence": None},
                "usage": {"output_tokens": 20}},
               {"type": "message_stop"}]
    return sse(*events)


def anthropic_llm(responses, **options):
    import anthropic
    from agenticwf.agent.runtime.llm.providers.anthropic import AnthropicLLM

    seen = []

    def handler(request):
        seen.append({"body": json.loads(request.content), "headers": dict(request.headers)})
        status, body = responses.pop(0)
        if status != 200:
            return httpx2.Response(status, json=body)
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = anthropic.AsyncAnthropic(api_key="test-key", max_retries=0,
                                      http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return AnthropicLLM(client=client, **options), seen


def test_anthropic_request_shape_and_replay():
    llm, seen = anthropic_llm([(200, anthropic_stream([
        {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        {"type": "text", "text": "Reading it."},
        {"type": "tool_use", "id": "tu_2", "name": "read_thing", "input": {"value": "a.py"}}], stop="tool_use"))],
        effort="high", fallbacks="default")
    earlier = LLMResponse("", [ToolCallRequest("tu_1", "read_thing", {})],
                          raw=[{"type": "thinking", "thinking": "t", "signature": "sig-0"},
                               {"type": "tool_use", "id": "tu_1", "name": "read_thing", "input": {}}], raw_source="anthropic")
    messages = [Message("user", "read a.py"),
                Message("assistant", "", earlier.tool_calls, raw=earlier.raw, raw_source="anthropic"),
                Message("tool", "<tool_result>thing</tool_result>", tool_call_id="tu_1", name="read_thing"),
                Message("user", "<task_status>Step 2 of 50.</task_status>")]
    streamed = []
    response = asyncio.run(llm.complete(LLMRequest(
        model="claude-opus-5-5", system="AGENTS", messages=messages, on_text=streamed.append,
        tools=[{"name": "read_thing", "description": "Read", "input_schema": {"type": "object"}}])))

    body, headers = seen[0]["body"], seen[0]["headers"]
    assert body["system"] == [{"type": "text", "text": "AGENTS", "cache_control": {"type": "ephemeral"}}]
    assert body["cache_control"] == {"type": "ephemeral"} and body["max_tokens"] == 64000
    assert body["output_config"] == {"effort": "high"} and body["fallbacks"] == "default"
    assert "server-side-fallback-2026-07-01" in headers["anthropic-beta"]
    assert body["tools"][0]["eager_input_streaming"] is True
    assert body["messages"][1]["content"][0] == {"type": "thinking", "thinking": "t", "signature": "sig-0"}   # verbatim
    tool_turn = body["messages"][2]
    assert tool_turn["role"] == "user" and [b["type"] for b in tool_turn["content"]] == ["tool_result", "text"]

    assert response.tool_calls[0].name == "read_thing" and response.tool_calls[0].arguments == {"value": "a.py"}
    assert response.raw[0] == {"type": "thinking", "thinking": "plan", "signature": "sig-1"}   # kept for replay
    assert streamed == ["Reading it."] and response.model == "anthropic/claude-opus-5-5"
    assert response.usage.input_tokens == 100 and response.usage.cached_input_tokens == 90


def test_anthropic_strips_thinking_when_history_changed():
    rejected = {"type": "error", "error": {"type": "invalid_request_error",
                "message": "messages.1.content.0: Invalid `signature` in `thinking` block. The block is bound to a different conversation."}}
    llm, seen = anthropic_llm([(400, rejected), (200, anthropic_stream([{"type": "text", "text": "ok"}]))])
    messages = [Message("user", "hi"),
                Message("assistant", "a", raw=[{"type": "thinking", "thinking": "t", "signature": "s"},
                                               {"type": "text", "text": "a"}], raw_source="anthropic"),
                Message("user", "again")]
    assert asyncio.run(llm.complete(LLMRequest(model="m", system="s", messages=messages))).text == "ok"
    assert [b["type"] for b in seen[1]["body"]["messages"][1]["content"]] == ["text"]   # retried without thinking


def test_anthropic_refusal_and_truncated_tool_call():
    llm, _ = anthropic_llm([(200, anthropic_stream([], stop="refusal")),
                            (200, anthropic_stream([{"type": "tool_use", "id": "t", "name": "x", "input": {"a": 1}}],
                                                   stop="max_tokens"))])
    refused = asyncio.run(llm.complete(LLMRequest(model="m", system="s", messages=[Message("user", "x")])))
    assert refused.stop_reason == "refusal" and refused.is_final and "declined" in refused.text
    cut = asyncio.run(llm.complete(LLMRequest(model="m", system="s", messages=[Message("user", "x")])))
    assert cut.is_final and "truncated" in cut.text                # never run a truncated tool input


def test_anthropic_errors_map_to_retryable():
    llm, _ = anthropic_llm([(429, {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}),
                            (401, {"type": "error", "error": {"type": "authentication_error", "message": "bad key"}})])
    with pytest.raises(LLMError) as limited:
        asyncio.run(llm.complete(LLMRequest(model="m", system="s", messages=[Message("user", "x")])))
    assert limited.value.retryable
    with pytest.raises(LLMError) as auth:
        asyncio.run(llm.complete(LLMRequest(model="m", system="s", messages=[Message("user", "x")])))
    assert not auth.value.retryable


# --- OpenAI-compatible adapter (OpenAI, vLLM, Ollama) ------------------------------------------

def chunks(*deltas, finish="stop", usage=None, model="gpt-x"):
    out = [{"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model,
            "choices": [{"index": 0, "delta": d, "finish_reason": None}]} for d in deltas]
    out.append({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
    if usage:
        out.append({"id": "c", "object": "chat.completion.chunk", "created": 1, "model": model, "choices": [],
                    "usage": usage})
    return "".join(f"data: {json.dumps(c)}\n\n" for c in out) + "data: [DONE]\n\n"


def openai_llm(provider, body, **options):
    import openai
    from agenticwf.agent.runtime.llm.providers.openai_compat import OpenAICompatibleLLM

    seen = []

    def handler(request):
        seen.append({"url": str(request.url), "body": json.loads(request.content)})
        return httpx2.Response(200, text=body, headers={"content-type": "text/event-stream"})

    client = openai.AsyncOpenAI(api_key="k", base_url="http://llm.test/v1", max_retries=0,
                                http_client=openai.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler)))
    return OpenAICompatibleLLM(provider, client=client, **options), seen


def test_openai_streamed_tool_call_fragments():
    body = chunks({"role": "assistant", "content": "Let me "}, {"content": "look."},
                  {"tool_calls": [{"index": 0, "id": "call_1", "type": "function", "function": {"name": "read_thing", "arguments": '{"val'}}]},
                  {"tool_calls": [{"index": 0, "function": {"arguments": 'ue": "a.py"}'}}]},
                  finish="tool_calls", usage={"prompt_tokens": 50, "completion_tokens": 9, "total_tokens": 59,
                                              "prompt_tokens_details": {"cached_tokens": 40}})
    llm, seen = openai_llm("openai", body, max_tokens=1000)
    streamed = []
    messages = [Message("user", "read"), Message("assistant", "", [ToolCallRequest("call_0", "read_thing", {"value": "b"})]),
                Message("tool", "thing", tool_call_id="call_0"), Message("user", "<task_status>Step 2</task_status>")]
    r = asyncio.run(llm.complete(LLMRequest(model="gpt-x", system="AGENTS", messages=messages, on_text=streamed.append,
                                            tools=[{"name": "read_thing", "description": "Read", "input_schema": {"type": "object"}}])))
    sent = seen[0]["body"]
    assert sent["max_completion_tokens"] == 1000 and sent["stream_options"] == {"include_usage": True}
    assert [m["role"] for m in sent["messages"]] == ["system", "user", "assistant", "tool", "user"]
    assert sent["messages"][2]["tool_calls"][0]["function"] == {"name": "read_thing", "arguments": '{"value": "b"}'}
    assert sent["tools"][0]["function"]["parameters"] == {"type": "object"}
    assert r.text == "Let me look." and streamed == ["Let me ", "look."]
    assert r.tool_calls[0].arguments == {"value": "a.py"} and r.stop_reason == "tool_use"
    assert (r.usage.input_tokens, r.usage.output_tokens, r.usage.cached_input_tokens) == (50, 9, 40)
    assert r.model == "openai/gpt-x"


def test_vllm_and_ollama_presets():
    llm, seen = openai_llm("vllm", chunks({"content": "hi"}), max_tokens=500)
    asyncio.run(llm.complete(LLMRequest(model="Qwen/Qwen3-32B", system="", messages=[Message("user", "x")])))
    assert seen[0]["body"]["max_tokens"] == 500 and "max_completion_tokens" not in seen[0]["body"]
    llm, _ = openai_llm("ollama", chunks({"tool_calls": [{"index": 0, "id": "c", "type": "function",
                                                          "function": {"name": "x", "arguments": '{"a": '}}]}, finish="length"))
    cut = asyncio.run(llm.complete(LLMRequest(model="qwen3", system="", messages=[Message("user", "x")])))
    assert cut.is_final and "truncated" in cut.text
    from agenticwf.agent.runtime.llm.providers.openai_compat import OpenAICompatibleLLM
    with pytest.raises(LLMError, match="base_url"):
        OpenAICompatibleLLM("vllm")


# --- database models: secrets encrypted at rest -----------------------------------------------

def test_database_models_encrypt_api_keys():
    from agenticwf.agent.core.orm.base.session import get_engine
    from agenticwf.agent.runtime.llm.db_catalog import DatabaseModelStore

    store = DatabaseModelStore(SecretBox(SecretBox.generate_key()))
    store.save_model(ModelEntry("gpt", "openai", "gpt-x", api_key="sk-live-1234567890abcd", source="database"))
    with get_engine().connect() as conn:
        row = conn.exec_driver_sql("select api_key_encrypted, api_key_hint from agent_models where id = 'gpt'").one()
    assert "sk-live" not in row[0] and row[1] == "…abcd"            # ciphertext and a hint only
    loaded = store.list_models()[0]
    assert loaded.api_key == "sk-live-1234567890abcd" and loaded.source == "database"
    assert "1234567890" not in json.dumps(loaded.admin_view())       # never shown

    other = DatabaseModelStore(SecretBox(SecretBox.generate_key()))  # a different key cannot read it
    assert other.list_models()[0].api_key is None
    with pytest.raises(PermissionError):
        DatabaseModelStore(None).save_model(ModelEntry("x", "openai", "y", api_key="k"))


# --- API: models for users, management for managers --------------------------------------------

@pytest.fixture
def model_app(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_SECRETS_KEY", SecretBox.generate_key().decode())
    config = RuntimeConfig(system_dir=make_system_dir(tmp_path), model="echo",
                           llm_sections={"llm.model.echo": {"provider": "echo", "model": "echo", "label": "Echo"}})
    with TestClient(create_app(Settings(context_secret=SECRET), runtime_config=config, engine=EngineOptions())) as tc:
        yield tc


def test_model_api_for_users(model_app):
    alice, bob = Client(model_app, user()), Client(model_app, user(subject="user_bob"))
    assert alice.get("/api/v1/models").json() == {"default": "echo", "models": [
        {"id": "echo", "label": "Echo", "description": "", "provider": "echo", "model": "echo",
         "tier": "medium", "context_window": None, "capabilities": ["tools"]}]}
    machine = Client(model_app, user(subject="app_m2m", client_id="app_m2m"))
    assert machine.post("/api/v1/models", json={"id": "x", "provider": "echo", "model": "x"}).status_code == 403

    added = alice.post("/api/v1/models", json={
        "id": "parrot", "provider": "echo", "model": "echo-2", "label": "Parrot", "api_key": "sk-secret-abcdefgh1234"})
    assert added.status_code == 201 and added.json()["has_api_key"] and added.json()["api_key_hint"] == "…1234"
    assert "sk-secret" not in added.text and added.json()["visibility"] == "user" and added.json()["group"] == "mine"
    assert alice.post("/api/v1/models", json={"id": "parrot", "provider": "echo", "model": "y"}).status_code == 409
    assert alice.post("/api/v1/models", json={"id": "Bad Id", "provider": "echo", "model": "y"}).status_code == 422
    assert alice.post("/api/v1/models", json={"id": "z", "provider": "echo", "model": "y", "visibility": "all"}).status_code == 403
    assert [m["id"] for m in alice.get("/api/v1/models").json()["models"]] == ["echo", "parrot"]
    assert [m["id"] for m in bob.get("/api/v1/models").json()["models"]] == ["echo"]       # private to alice

    changed = alice.patch("/api/v1/models/parrot", json={"label": "Parrot 2"}).json()
    assert changed["label"] == "Parrot 2" and changed["has_api_key"]                    # the key is kept
    assert not alice.patch("/api/v1/models/parrot", json={"clear_api_key": True}).json()["has_api_key"]
    assert alice.patch("/api/v1/models/echo", json={"label": "x"}).status_code == 403   # system: read-only
    assert bob.patch("/api/v1/models/parrot", json={"label": "x"}).status_code == 404   # not even visible

    sid = alice.post("/api/v1/sessions", json={}).json()["id"]
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi", "model": "nope"}).status_code == 422
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "hi", "model": "parrot"}).json()["reply"] == "echo: hi"
    bob_sid = bob.post("/api/v1/sessions", json={}).json()["id"]
    assert bob.post(f"/api/v1/sessions/{bob_sid}/messages", json={"text": "x", "model": "parrot"}).status_code == 422
    alice.patch("/api/v1/models/parrot", json={"enabled": False})
    assert alice.post(f"/api/v1/sessions/{sid}/messages", json={"text": "x", "model": "parrot"}).status_code == 422
    assert "parrot" in [m["id"] for m in alice.get("/api/v1/models?detail=true").json()["models"]]   # still listed for its owner
    assert alice.delete("/api/v1/models/parrot").status_code == 204
    assert alice.delete("/api/v1/models/parrot").status_code == 404


def test_default_model_api(model_app):
    alice, bob = Client(model_app, user()), Client(model_app, user(subject="user_bob"))
    alice.post("/api/v1/models", json={"id": "parrot", "provider": "echo", "model": "echo-2"})
    assert bob.put("/api/v1/models/default", json={"model": "parrot"}).status_code == 422   # not shared with bob
    chosen = alice.put("/api/v1/models/default", json={"model": "parrot"}).json()
    assert (chosen["default"], chosen["user_default"], chosen["system_default"]) == ("parrot", "parrot", "echo")
    assert alice.get("/api/v1/models").json()["default"] == "parrot" and bob.get("/api/v1/models").json()["default"] == "echo"
    machine = Client(model_app, user(subject="app_m2m", client_id="app_m2m"))
    assert machine.put("/api/v1/models/default", json={"model": "echo"}).status_code == 403
    reset = alice.put("/api/v1/models/default", json={"model": None}).json()
    assert reset["default"] == "echo" and reset["user_default"] is None


# --- harness: append-only history, model precedence -------------------------------------------

def test_history_is_append_only_between_steps(system_dir):
    from tests.test_runtime import call, user as runtime_user

    harness, _, _ = build(system_dir, [LLMResponse("", [call("read_thing")]), LLMResponse("", [call("nope")]),
                                       LLMResponse("done")])
    asyncio.run(harness.run(harness.create_task(runtime_user(), "read, then try").id))
    first, second, third = harness.services.llm.requests
    assert second.messages[:len(first.messages)] == first.messages      # every request extends the last
    assert third.messages[:len(second.messages)] == second.messages
    assert "Files examined" in second.messages[-1].content             # read_thing changed working memory
    assert "(working memory unchanged)" in third.messages[-1].content  # the unknown tool did not


def test_model_precedence(system_dir):
    from tests.test_runtime import user as runtime_user

    (system_dir / "task-types" / "review" / "task.yaml").write_text("mode: plan\nmodel: reviewer\n")
    harness, _, _ = build(system_dir, [])
    assert harness.create_task(runtime_user(), "x", task_type="review").model == "reviewer"      # task type
    assert harness.create_task(runtime_user(), "x", task_type="review", model="mine").model == "mine"   # chosen
    assert harness.create_task(runtime_user(), "x").model == "default"
