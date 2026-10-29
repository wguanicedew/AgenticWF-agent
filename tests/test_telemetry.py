"""OpenTelemetry (agenticwf.agent.runtime.telemetry): spans and metrics of jobs, model calls and tool calls."""

import asyncio
from types import SimpleNamespace

import pytest

pytest.importorskip("opentelemetry.sdk")

from opentelemetry import metrics, trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from agenticwf.agent.runtime import telemetry
from agenticwf.agent.runtime.harness.db_store import DatabaseStore
from agenticwf.agent.runtime.harness.store import Job
from agenticwf.agent.runtime.harness.worker import Workers
from agenticwf.agent.runtime.llm.base import LLMResponse
from agenticwf.agent.runtime.tools.spec import ToolResult

from tests.test_routing import request, router
from tests.test_runtime import build, call, make_system_dir, user

EXPORTER = InMemorySpanExporter()
READER = InMemoryMetricReader()


@pytest.fixture(scope="module", autouse=True)
def provider():
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(EXPORTER))
    trace.set_tracer_provider(tracer_provider)   # once per process; nothing else sets them in the tests
    metrics.set_meter_provider(MeterProvider(metric_readers=[READER]))


@pytest.fixture
def spans():
    EXPORTER.clear()
    return lambda: {s.name: s for s in EXPORTER.get_finished_spans()}


def points(name):
    """The data points of one metric since the provider started (cumulative), by their attributes."""
    data = READER.get_metrics_data()
    return [(dict(p.attributes), p) for rm in data.resource_metrics for sm in rm.scope_metrics
            for m in sm.metrics if m.name == name for p in m.data.data_points]


def test_off_without_an_endpoint(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert not telemetry.enabled() and not telemetry.setup("agenticwf-agent")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    assert telemetry.enabled()
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")
    assert not telemetry.enabled()


def test_spans_do_nothing_without_the_sdk(monkeypatch):
    monkeypatch.setattr(telemetry, "_trace", None)
    with telemetry.span("anything", a=1) as current:
        telemetry.annotate(current, b=2)
        telemetry.fail(current, "no")
    assert not current.is_recording()


def test_a_model_call_is_a_gen_ai_span(spans):
    response = asyncio.run(router().complete(request(purpose="split", task_id="t1")))
    chat = spans()[f"chat {response.model.split('/', 1)[1]}"]
    attributes = chat.attributes
    assert attributes["gen_ai.operation.name"] == "chat" and attributes["gen_ai.provider.name"] == "vllm"
    assert attributes["agenticwf.model.id"] == response.model_id and attributes["agenticwf.llm.purpose"] == "split"
    assert attributes["gen_ai.usage.output_tokens"] == 100 and attributes["agenticwf.task.id"] == "t1"
    assert attributes["agenticwf.llm.route"] == response.route
    assert not any("x" * 20 in str(v) for v in attributes.values())   # never the prompt


def test_a_job_holds_its_tool_calls(tmp_path, spans):
    harness, _, _ = build(make_system_dir(tmp_path), [
        LLMResponse("", [call("read_thing"), call("write_thing")]),
        LLMResponse("done"),
    ])
    harness.dispatcher.registry.register_handler("test.write_thing",
                                                 lambda ctx, args: ToolResult.error("no_space", "disk full"))
    task = harness.create_task(user(), "read and write the thing")
    again = asyncio.run(Workers._execute(SimpleNamespace(harness=harness), Job("task", task.id)))
    assert not again
    found = spans()
    job, read, write = found["job task"], found["execute_tool read_thing"], found["execute_tool write_thing"]
    assert read.parent.span_id == job.context.span_id and write.parent.span_id == job.context.span_id
    assert job.attributes["agenticwf.job.ref"] == task.id and job.attributes["agenticwf.task.state"] == "completed"
    assert read.attributes["gen_ai.tool.name"] == "read_thing" and read.status.status_code is StatusCode.UNSET
    assert write.status.status_code is StatusCode.ERROR and write.status.description == "no_space"
    trace_id = format(job.context.trace_id, "032x")              # kept for the trace view's links
    assert {c.trace_id for c in task.calls} == {trace_id}
    assert {s.trace_id for t in task.turns for s in t.steps} == {trace_id}


def test_a_job_links_to_the_span_that_queued_it(tmp_path, spans):
    harness, _, _ = build(make_system_dir(tmp_path), [LLMResponse("done")])
    task = harness.create_task(user(), "hi")
    with telemetry.span("POST /api/v1/sessions/s1/messages") as request_span:
        job = Job("task", task.id)
    assert Job("task", task.id).trace is None                                # queued outside any span
    asyncio.run(Workers._execute(SimpleNamespace(harness=harness), job))
    job_span, origin = spans()["job task"], request_span.get_span_context()
    assert job_span.context.trace_id != origin.trace_id                      # a trace of its own,
    assert [(link.context.trace_id, link.context.span_id) for link in job_span.links] == [
        (origin.trace_id, origin.span_id)]                                    # linked to the request


def test_a_shared_job_keeps_its_link_through_the_database():
    one, two = DatabaseStore(shared=True), DatabaseStore(shared=True)
    with telemetry.span("request"):
        one.enqueue(Job("task", "x", owner="alice"))
        queued = telemetry.traceparent()
    assert two.claim("p2-w1", 60).trace == queued


def test_metrics_of_model_tool_and_job(tmp_path):
    asyncio.run(router().complete(request(purpose="split")))
    harness, _, _ = build(make_system_dir(tmp_path), [LLMResponse("", [call("read_thing")]), LLMResponse("done")])
    task = harness.create_task(user(), "read the thing")
    asyncio.run(Workers._execute(SimpleNamespace(harness=harness), Job("task", task.id)))

    tokens = {a["gen_ai.token.type"]: p for a, p in points("gen_ai.client.token.usage")
              if a["agenticwf.llm.purpose"] == "split"}
    assert tokens["output"].sum >= 100 and tokens["input"].count >= 1
    tools = [a for a, _ in points("agenticwf.tool.duration") if a["gen_ai.tool.name"] == "read_thing"]
    assert tools and "error.type" not in tools[0] and tools[0]["agenticwf.tool.handler"] == "builtin"
    jobs = [a for a, _ in points("agenticwf.job.duration") if a.get("agenticwf.task.state") == "completed"]
    assert jobs and jobs[0]["agenticwf.job.outcome"] == "done"
    assert all(p.value == 0 for _, p in points("agenticwf.jobs.running"))   # none left running
    ended = [a for a, _ in points("agenticwf.tasks.ended") if a["agenticwf.task.state"] == "completed"]
    assert ended and ended[0]["agenticwf.task.type"] == "system:default" and ended[0]["agenticwf.task.cron"] is False


def test_metrics_need_their_own_endpoint(monkeypatch):
    for name in ("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "OTEL_METRICS_EXPORTER", "OTEL_SDK_DISABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://jaeger:4318")
    assert not telemetry.metrics_enabled()                                   # Jaeger takes no metrics
    monkeypatch.setenv("OTEL_METRICS_EXPORTER", "otlp")
    assert telemetry.metrics_enabled()                                       # a collector does
    monkeypatch.delenv("OTEL_METRICS_EXPORTER")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT", "http://prometheus:9090/api/v1/otlp/v1/metrics")
    assert telemetry.metrics_enabled()
