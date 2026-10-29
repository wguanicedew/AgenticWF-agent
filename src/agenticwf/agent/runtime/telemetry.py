"""
OpenTelemetry tracing and metrics (docs/deployment.md §8): a span per worker job, model call and
tool call, and, in the API process, per HTTP request and outgoing httpx call; counters and
histograms of the same. Both go over OTLP/HTTP to a collector (Jaeger for traces, Prometheus or an
OpenTelemetry Collector for metrics).

    worker process ─┐
    API process ────┴─ OTLP/HTTP ─► collector ─► UI

Needs the `otel` extra (`pip install 'agenticwf-agent-server[otel]'`; the container installs it), and:

    traces    OTEL_EXPORTER_OTLP_ENDPOINT or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT
    metrics   OTEL_EXPORTER_OTLP_METRICS_ENDPOINT, or OTEL_EXPORTER_OTLP_ENDPOINT with
              OTEL_METRICS_EXPORTER=otlp (Jaeger takes no metrics: off by default)

The rest is the SDK's own environment: OTEL_SERVICE_NAME, OTEL_RESOURCE_ATTRIBUTES,
OTEL_TRACES_SAMPLER, OTEL_EXPORTER_OTLP_HEADERS, OTEL_METRIC_EXPORT_INTERVAL, OTEL_SDK_DISABLED, ...

Spans carry ids, names, models, token counts and costs, never prompts, replies or tool arguments.
Model calls follow the OpenTelemetry GenAI conventions (gen_ai.*), so GenAI-aware UIs show them.
Metric attributes are bounded: what users define themselves (their tools, models, task types)
counts as `user`, never by name.

A job runs long after the request that queued it, from the database, maybe in another process: the
job carries that request's span context (`traceparent()`), and its span links to it.

Everything here is safe to call anywhere: without the SDK, or before `setup()`, it does nothing.
"""

from __future__ import annotations

import asyncio
import os
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator

from agenticwf.agent.core.common.logging import get_logger

try:
    from opentelemetry import metrics as _metrics
    from opentelemetry import trace as _trace
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
except ImportError:   # the `otel` extra is not installed
    _metrics = _trace = None

LOG = get_logger(__name__)

TRACER = "agenticwf.agent"
_tracing = _metering = False


def _env(name: str) -> str:
    return os.environ.get(name, "")


def enabled() -> bool:
    """Traces are asked for: an OTLP endpoint is set and the SDK is not disabled."""
    if _env("OTEL_SDK_DISABLED").lower() == "true" or _env("OTEL_TRACES_EXPORTER") == "none":
        return False
    return bool(_env("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") or _env("OTEL_EXPORTER_OTLP_ENDPOINT"))


def metrics_enabled() -> bool:
    """Metrics are asked for: their own endpoint, or the shared one with OTEL_METRICS_EXPORTER=otlp."""
    if _env("OTEL_SDK_DISABLED").lower() == "true" or _env("OTEL_METRICS_EXPORTER") == "none":
        return False
    return bool(_env("OTEL_EXPORTER_OTLP_METRICS_ENDPOINT")
                or (_env("OTEL_EXPORTER_OTLP_ENDPOINT") and _env("OTEL_METRICS_EXPORTER") == "otlp"))


def setup(service: str) -> bool:
    """Install the process's tracer and meter providers, exporting over OTLP. `service` is the
    service.name unless OTEL_SERVICE_NAME says otherwise. Once per process; True when either is on."""
    global _tracing, _metering
    if _tracing or _metering:
        return True
    traces, metrics = enabled(), metrics_enabled()
    if not (traces or metrics):
        return False
    try:
        from opentelemetry.sdk.resources import Resource
    except ImportError:
        LOG.warning("an OTLP endpoint is set but the OpenTelemetry SDK is not installed "
                    "(pip install 'agenticwf-agent-server[otel]'); no traces or metrics")
        return False
    resource = Resource.create({} if _env("OTEL_SERVICE_NAME") else {"service.name": service})
    if traces:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=resource)   # flushed at exit
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        _trace.set_tracer_provider(provider)
        _tracing = True
    if metrics:
        from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader

        reader = PeriodicExportingMetricReader(OTLPMetricExporter())   # every OTEL_METRIC_EXPORT_INTERVAL (60 s)
        _metrics.set_meter_provider(MeterProvider(resource=resource, metric_readers=[reader]))
        _metering = True
    try:   # outgoing calls: the knowledge service, web search, model providers
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except ImportError:
        pass
    LOG.info("OpenTelemetry as %s: traces %s, metrics %s", _env("OTEL_SERVICE_NAME") or service,
             "on" if _tracing else "off", "on" if _metering else "off")
    return True


def instrument_app(app: Any) -> None:
    """A span and request metrics per HTTP request of a FastAPI app (not the health check)."""
    if not (_tracing or _metering):
        return
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        return
    # Not a span per ASGI message: a live-update stream sends one per event for an hour.
    FastAPIInstrumentor.instrument_app(app, excluded_urls="/healthz", exclude_spans=["receive", "send"])


# --- traces --------------------------------------------------------------------------------------
class _NoSpan:
    def set_attribute(self, key: str, value: Any) -> None: ...

    def set_attributes(self, attributes: dict[str, Any]) -> None: ...

    def set_status(self, *args: Any, **kwargs: Any) -> None: ...

    def record_exception(self, *args: Any, **kwargs: Any) -> None: ...

    def is_recording(self) -> bool:
        return False


_NO_SPAN = _NoSpan()


@contextmanager
def span(name: str, links: Iterable[str | None] = (), **attributes: Any) -> Iterator[Any]:
    """A child of the current span (or a new trace), linked to the spans named by `links`
    (traceparent strings). An exception that leaves it marks it failed. Attribute names use `__`
    for dots (gen_ai__request__model=...); None values are left out."""
    if _trace is None:
        yield _NO_SPAN
        return
    linked = [_trace.Link(c) for c in map(_span_context, links) if c is not None]
    with _trace.get_tracer(TRACER).start_as_current_span(name, attributes=_attributes(attributes),
                                                         links=linked) as current:
        yield current


def annotate(current: Any, **attributes: Any) -> None:
    """Add attributes to a span from `span()`, named as there."""
    if current.is_recording():
        current.set_attributes(_attributes(attributes))


def fail(current: Any, description: str) -> None:
    """Mark a span failed without an exception (a tool's error result, a job that failed)."""
    if _trace is not None and current.is_recording():
        current.set_status(_trace.Status(_trace.StatusCode.ERROR, description))


def traceparent() -> str | None:
    """The current span as a W3C traceparent, to link to later (a job's origin); None outside one."""
    if _trace is None or not _trace.get_current_span().get_span_context().is_valid:
        return None
    carrier: dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent")


def trace_id() -> str | None:
    """The current trace's id (32 hex digits), kept with what it recorded so a view can link to the
    trace (AGENT_TRACE_URL); None outside a span or without tracing."""
    if _trace is None:
        return None
    context = _trace.get_current_span().get_span_context()
    return format(context.trace_id, "032x") if context.is_valid else None


def _span_context(value: str | None) -> Any:
    if not value:
        return None
    context = _trace.get_current_span(TraceContextTextMapPropagator().extract({"traceparent": value})).get_span_context()
    return context if context.is_valid else None


def _attributes(given: dict[str, Any]) -> dict[str, Any]:
    return {key.replace("__", "."): value for key, value in given.items() if value is not None}


class Tracing:
    """HarnessServices.otel: this module, for code that holds the services."""

    def tracer(self, name: str) -> Any:
        return _trace.get_tracer(name) if _trace is not None else None

    span = staticmethod(span)


# --- metrics -------------------------------------------------------------------------------------
# Instruments on the global meter: no-ops until setup() installs a provider (the API's proxies).
_SECONDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 900]
_TOKENS = [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576]

if _metrics is not None:
    _meter = _metrics.get_meter(TRACER)
    _model_duration = _meter.create_histogram(
        "gen_ai.client.operation.duration", unit="s", description="Model calls, each fallback attempt",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _model_tokens = _meter.create_histogram(
        "gen_ai.client.token.usage", unit="{token}", description="Tokens per model call, by gen_ai.token.type",
        explicit_bucket_boundaries_advisory=_TOKENS)
    _model_cost = _meter.create_counter(
        "agenticwf.llm.cost", unit="{cost}", description="Cost of model calls, in the deployment's [llm] cost_unit")
    _tool_duration = _meter.create_histogram(
        "agenticwf.tool.duration", unit="s", description="Tool calls, by outcome", explicit_bucket_boundaries_advisory=_SECONDS)
    _job_duration = _meter.create_histogram(
        "agenticwf.job.duration", unit="s", description="Worker jobs, by kind and outcome",
        explicit_bucket_boundaries_advisory=_SECONDS)
    _jobs_running = _meter.create_up_down_counter(
        "agenticwf.jobs.running", unit="{job}", description="Jobs running in this process")
    _tasks_ended = _meter.create_counter(
        "agenticwf.tasks.ended", unit="{task}", description="Tasks that reached a terminal state")


def record_model_call(*, provider: str, model: str, model_id: str, purpose: str, seconds: float,
                      input_tokens: int = 0, output_tokens: int = 0, cost: float | None = None,
                      error: str | None = None) -> None:
    """One model call. `model_id` is a configured model's id, or `user` for one a user added."""
    if _metrics is None:
        return
    attributes = {"gen_ai.operation.name": "chat", "gen_ai.provider.name": provider, "gen_ai.request.model": model,
                  "agenticwf.model.id": model_id, "agenticwf.llm.purpose": purpose}
    if error:
        _model_duration.record(seconds, {**attributes, "error.type": error})
        return
    _model_duration.record(seconds, attributes)
    _model_tokens.record(input_tokens, {**attributes, "gen_ai.token.type": "input"})
    _model_tokens.record(output_tokens, {**attributes, "gen_ai.token.type": "output"})
    if cost:
        _model_cost.add(cost, attributes)


def record_tool_call(*, tool: str, handler: str, seconds: float, error: str | None = None) -> None:
    """One tool call. `tool` is the tool's name, or `user` for one a user defined."""
    if _metrics is not None:
        _tool_duration.record(seconds, _attributes({"gen_ai__tool__name": tool, "agenticwf__tool__handler": handler,
                                                     "error__type": error}))


@contextmanager
def measure_job(kind: str) -> Iterator[dict[str, str]]:
    """Time a job and count it while it runs. Set `task_state` for tasks in the dict it yields; the
    outcome is `done`, `cancelled`, or `failed` when an exception leaves it."""
    if _metrics is None:
        yield {}
        return
    found = {"outcome": "done"}
    started = time.monotonic()
    _jobs_running.add(1, {"agenticwf.job.kind": kind})
    try:
        yield found
    except BaseException as exc:   # cancelled: stopped (a lost lease, an interrupt), not broken
        found["outcome"] = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        raise
    finally:
        _jobs_running.add(-1, {"agenticwf.job.kind": kind})
        _job_duration.record(time.monotonic() - started, _attributes({
            "agenticwf__job__kind": kind, "agenticwf__job__outcome": found["outcome"],
            "agenticwf__task__state": found.get("task_state")}))


def record_task_ended(*, task_type: str, state: str, cron: bool) -> None:
    """A task reached `state` (completed, failed, cancelled). `task_type`: a system or plugin type's
    id, or `user`."""
    if _metrics is not None:
        _tasks_ended.add(1, {"agenticwf.task.type": task_type, "agenticwf.task.state": state, "agenticwf.task.cron": cron})
