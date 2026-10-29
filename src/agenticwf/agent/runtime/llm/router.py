"""
ModelRouter: the LLM the harness talks to. Each request names a model id (LLMRequest.model:
a thread's, a task type's, a message's choice, or "default"); the router resolves it in the
ModelCatalog and sends the request to that entry's provider adapter with the provider's own
model name.

    request.model ──► ModelCatalog.resolve ──► ModelEntry ──► adapter (cached per entry version)
                                                 │                 Anthropic · OpenAI · vLLM · Ollama · echo
                                                 └─ on a retryable error (rate limit, outage,
                                                    network): the next of [llm] fallback

Adapters are built on first use, so a provider nobody uses needs no credentials, and rebuilt
when a manager changes the entry. Refusals are not routed here: the Anthropic adapter handles
those with the provider's own fallback.
"""

from __future__ import annotations

import datetime as dt
import os
import time
from dataclasses import replace
from typing import Any, Callable

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.llm.base import LLM, LLMError, LLMRequest, LLMResponse
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry
from agenticwf.agent.runtime.llm.routing import (TIERS, Choice, OutputEstimator, RoutingSettings, estimate_tokens, raise_tier,
                                 tier_rank)
from agenticwf.agent.runtime.llm.usage import InMemoryUsage, UsageStore
from agenticwf.agent.runtime.telemetry import annotate, fail, record_model_call, span, trace_id

LOG = get_logger(__name__)

Factory = Callable[[ModelEntry, dict[str, Any]], LLM]


def _api_key(entry: ModelEntry, settings: dict[str, Any]) -> str | None:
    """A model a user added: its own (database) key only, never the server's environment. A
    configured model: the entry's environment variable, then the provider's."""
    if entry.source != "config":
        return entry.api_key or None
    if entry.api_key:
        return entry.api_key
    for env in (entry.api_key_env, settings.get("api_key_env")):
        if env and os.environ.get(env):
            return os.environ[env]
    return None


def build_adapter(entry: ModelEntry, settings: dict[str, Any]) -> LLM:
    """The adapter for one entry: provider settings ([llm.provider.<name>]) under the entry's own."""
    options = {k: v for k, v in settings.items() if k not in ("api_key_env", "base_url")}
    options.update(entry.options)
    base_url = entry.base_url or settings.get("base_url")
    common = {k: options[k] for k in ("timeout", "max_retries") if k in options}
    common["isolated"] = entry.source != "config"   # a user's model: nothing from the environment
    if entry.provider == "anthropic":
        from agenticwf.agent.runtime.llm.providers.anthropic import DEFAULT_MAX_TOKENS, AnthropicLLM

        return AnthropicLLM(api_key=_api_key(entry, settings), base_url=base_url,
                            max_tokens=options.get("max_tokens", DEFAULT_MAX_TOKENS), effort=options.get("effort"),
                            thinking=options.get("thinking"), thinking_display=options.get("thinking_display"),
                            fallbacks=options.get("fallbacks"), **common)
    if entry.provider in ("openai", "vllm", "ollama"):
        from agenticwf.agent.runtime.llm.providers.openai_compat import OpenAICompatibleLLM

        return OpenAICompatibleLLM(entry.provider, api_key=_api_key(entry, settings), base_url=base_url,
                                   max_tokens=options.get("max_tokens"), token_param=options.get("token_param"),
                                   stream_usage=options.get("stream_usage", True), extra=options.get("extra"), **common)
    if entry.provider == "echo":
        from agenticwf.agent.runtime.llm.echo import EchoLLM

        return EchoLLM()
    raise LLMError(f"unknown provider {entry.provider!r}", provider=entry.provider)


class ModelRouter(LLM):
    """Sends each request to a model: the one it names when that is pinned, else the one routing
    chooses for its purpose (agenticwf.agent.runtime.llm.routing, docs/model-routing.md); records what it cost."""

    def __init__(self, catalog: ModelCatalog, factory: Factory = build_adapter,
                 settings: RoutingSettings | None = None, usage: UsageStore | None = None):
        self.catalog = catalog
        self.factory = factory
        self.settings = settings or RoutingSettings()
        self.usage = usage or InMemoryUsage()
        self.outputs = OutputEstimator()
        self.adjusted: dict[str, str] = {}          # purpose -> tier, from feedback (feedback = adjust)
        self._adjusted_at = 0.0
        self._adapters: dict[str, tuple[float, LLM]] = {}   # entry id → (entry.updated_at, adapter)
        try:
            self.outputs.seed(self.usage.averages())
        except Exception:   # no table yet (first start): averages build up from now
            LOG.debug("no earlier usage to seed output estimates from")

    def adapter(self, entry: ModelEntry) -> LLM:
        refusal = self.catalog.endpoint_refusal(entry, resolve=False)   # at every call: the allow-list may have changed
        if refusal:
            raise LLMError(f"model {entry.id}: {refusal}", provider=entry.provider)
        cached = self._adapters.get(entry.id)
        if cached is None or cached[0] != entry.updated_at:
            adapter = self.factory(entry, self.catalog.provider_settings.get(entry.provider, {}))
            self._adapters[entry.id] = cached = (entry.updated_at, adapter)
        return cached[1]

    def chain(self, ref: str | None, owner=None) -> list[ModelEntry]:
        """The resolved model, then the configured fallbacks that are available (to `owner`)."""
        first = self.catalog.resolve(ref, owner)
        return self._with_fallbacks(first, owner)

    def _with_fallbacks(self, first: ModelEntry, owner) -> list[ModelEntry]:
        rest = [self.catalog.get(f) for f in self.catalog.fallbacks if f != first.id]
        return [first, *(e for e in rest if e is not None and e.enabled and e.visible_to(owner))]

    # --- routing -------------------------------------------------------------------------------
    def tier_for(self, purpose: str) -> str:
        return self.adjusted.get(purpose) or self.settings.tier_for(purpose)

    def window(self, model_id: str | None) -> int | None:
        """The context window of a model (None: unknown)."""
        entry = self.catalog.get(model_id) if model_id else None
        return entry.profile.context_window if entry else None

    def budget_state(self, owner) -> tuple[float, float | None]:
        """(spent, budget) for the owner's current period; budget None: no budget."""
        budget = self.settings.budget_for(owner)
        if budget is None:
            return 0.0, None
        now = dt.datetime.now(dt.timezone.utc)
        return self.usage.spent(owner, budget.start(now)), budget.amount

    def routing_active(self) -> bool:
        """Routing chooses models: mode auto, and at least one model declares a tier or a size."""
        return self.settings.mode == "auto" and any(e.profile.declared for e in self.catalog.entries())

    def can_escalate(self, owner) -> bool:
        return self.routing_active() and not self.under_pressure(owner)

    def under_pressure(self, owner) -> bool:
        """At 80% of the owner's budget or more: no escalation, the smallest tier."""
        spent, budget = self.budget_state(owner)
        return budget is not None and spent >= 0.8 * budget

    def choose(self, request: LLMRequest) -> Choice:
        owner, purpose = request.owner, request.purpose
        size = estimate_tokens(request)
        headroom = 1 - self.settings.headroom

        def fits(entry: ModelEntry) -> bool:
            profile = entry.profile
            if request.tools and "tools" not in profile.capabilities:
                return False
            if profile.context_window is None:
                return True   # unknown: assumed to fit (a model that matters should declare it)
            output = min(self.outputs.expect(purpose, entry.id), profile.max_output or 10**9)
            return size + output <= profile.context_window * headroom

        def cost(entry: ModelEntry) -> float:
            cached = request.cached_hint if entry.id == request.current else 0   # another model: no cache
            return entry.profile.cost(size, self.outputs.expect(purpose, entry.id), cached)

        def nearest(reason: str, detail: str) -> Choice | None:
            """The cheapest model that fits at all (any tier): when the wanted one does not."""
            fitting = [e for e in self._candidates(owner, routable_only=False) if fits(e)]
            if not fitting:
                return None
            best = min(fitting, key=lambda e: (cost(e), -(e.profile.context_window or 0)))
            return Choice(best, reason, cost(best), detail)

        named = request.model not in (None, "", "default")
        if request.pinned and named or self.settings.purpose_models.get(purpose):
            reason = "pinned" if request.pinned and named else "purpose"
            entry = self.catalog.resolve(request.model if reason == "pinned" else self.settings.purpose_models[purpose], owner)
            if fits(entry):
                return Choice(entry, reason, cost(entry))
            return nearest("context", f"{entry.id} is too small for this request") or Choice(entry, reason, cost(entry))
        if self.settings.mode == "pinned":
            entry = self.catalog.resolve(request.model if named else None, owner)
            return Choice(entry, "default", cost(entry)) if fits(entry) else (
                nearest("context", f"{entry.id} is too small for this request") or Choice(entry, "default", cost(entry)))

        need = max(tier_rank(self.tier_for(purpose)), tier_rank(request.min_tier) if request.min_tier else 0)
        ceiling = tier_rank(self.settings.max_tier_for(owner))
        if self.under_pressure(owner):
            need = tier_rank("small")
        need = min(need, ceiling)
        candidates = [e for e in self._candidates(owner) if e.profile.declared and fits(e)
                      and need <= tier_rank(e.profile.tier) <= ceiling]
        if not candidates:
            entry = self.catalog.resolve(request.model if named else None, owner)
            if fits(entry):
                return Choice(entry, "default", cost(entry), "no declared model meets the request")
            return nearest("context", "no model of the wanted tier fits") or Choice(entry, "default", cost(entry))
        order = {e.id: n for n, e in enumerate(self.catalog.entries())}
        speeds = {"fast": 0, "normal": 1, "slow": 2}
        best = min(candidates, key=lambda e: (not e.owned_by(owner), cost(e), e.id != request.current,
                                               speeds.get(e.profile.speed, 1), order.get(e.id, 99)))
        return Choice(best, "routed", cost(best))

    def _candidates(self, owner, routable_only: bool = True) -> list[ModelEntry]:
        """Models the router may choose for `owner`: the deployment's, and the owner's own (never
        someone else's, even shared: sharing lets others pin it)."""
        found = []
        for entry in self.catalog.entries(owner=owner):
            if entry.source != "config" and not entry.owned_by(owner):
                continue
            if routable_only and not entry.profile.routable:
                continue
            found.append(entry)
        return found

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self._maybe_adjust()
        choice = self.choose(request)
        if not choice.entry.owned_by(request.owner):   # the owner's own key pays for their own models
            spent, budget = self.budget_state(request.owner)
            if budget is not None and spent >= budget:
                raise LLMError(f"the budget of {budget:g} {self.settings.cost_unit} per "
                               f"{self.settings.budget_for(request.owner).period} is used up "
                               f"({spent:.2f} spent); ask an administrator, or use a model of your own")
        chain = self._with_fallbacks(choice.entry, request.owner)
        for n, entry in enumerate(chain):
            with span(f"chat {entry.model}", gen_ai__operation__name="chat", gen_ai__provider__name=entry.provider,
                      gen_ai__request__model=entry.model, agenticwf__model__id=entry.id,
                      agenticwf__llm__purpose=request.purpose, agenticwf__task__id=request.task_id,
                      agenticwf__intent__id=request.intent_id) as current:
                started = time.monotonic()
                measured = dict(provider=entry.provider, model=entry.model, purpose=request.purpose,
                                model_id=entry.id if entry.source == "config" else "user")
                try:
                    response = await self.adapter(entry).complete(replace(request, model=entry.model))
                except LLMError as exc:
                    fail(current, str(exc))
                    record_model_call(**measured, seconds=time.monotonic() - started,
                                      error="retryable" if exc.retryable else "error")
                    if exc.retryable and n + 1 < len(chain):
                        LOG.warning("model %s failed (%s); trying %s", entry.id, exc, chain[n + 1].id)
                        continue
                    raise
                response.model = response.model or f"{entry.provider}/{entry.model}"
                response.model_id, response.route = entry.id, (choice.reason if not n else "fallback")
                usage = response.usage
                response.cost = entry.profile.cost(usage.input_tokens, usage.output_tokens, usage.cached_input_tokens)
                annotate(current, gen_ai__response__model=response.model,
                         gen_ai__response__finish_reasons=[response.stop_reason],
                         gen_ai__usage__input_tokens=usage.input_tokens, gen_ai__usage__output_tokens=usage.output_tokens,
                         agenticwf__llm__cached_input_tokens=usage.cached_input_tokens,
                         agenticwf__llm__route=response.route, agenticwf__llm__cost=response.cost)
                record_model_call(**measured, seconds=time.monotonic() - started, input_tokens=usage.input_tokens,
                                  output_tokens=usage.output_tokens, cost=response.cost)
                self.outputs.observe(request.purpose, entry.id, usage.output_tokens)
                self._record(request, entry, response, time.monotonic() - started)
                if n:
                    LOG.info("served by fallback model %s", entry.id)
                return response
        raise LLMError("no model available")   # unreachable: chain is never empty

    def _record(self, request: LLMRequest, entry: ModelEntry, response: LLMResponse, seconds: float) -> None:
        owner = request.owner or (None, None)
        usage = response.usage
        try:
            self.usage.record({"owner_subject": owner[0], "owner_org": owner[1], "task_id": request.task_id,
                               "intent_id": request.intent_id, "purpose": request.purpose, "model_id": entry.id,
                               "provider": entry.provider, "input_tokens": usage.input_tokens,
                               "cached_tokens": usage.cached_input_tokens, "output_tokens": usage.output_tokens,
                               "cost": response.cost or 0.0, "route": response.route,
                               "own_model": int(entry.owned_by(request.owner)), "seconds": round(seconds, 3),
                               "trace_id": trace_id()})
        except Exception:   # accounting never fails a call
            LOG.exception("could not record the usage of a call to %s", entry.id)

    # --- feedback ----------------------------------------------------------------------------------
    def feedback(self, kind: str, purpose: str, model_id: str, owner=None, task_id: str | None = None,
                 intent_id: str | None = None, reason: str = "") -> None:
        owner = owner or (None, None)
        try:
            self.usage.feedback({"owner_subject": owner[0], "owner_org": owner[1], "task_id": task_id,
                                 "intent_id": intent_id, "purpose": purpose, "model_id": model_id, "kind": kind,
                                 "reason": reason or None})
        except Exception:
            LOG.exception("could not record feedback on %s", model_id)

    def suggestions(self, days: int = 30) -> list[dict]:
        """Where a model clearly under- or over-performs for a purpose (docs/model-routing.md §8)."""
        since = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
        found = []
        for row in self.usage.quality(since):
            calls, rated = row["calls"], row["up"] + row["down"]
            entry = self.catalog.get(row["model"])
            tier = entry.profile.tier if entry else "?"
            escalated = row["escalations"] / calls if calls else 0.0
            liked = row["up"] / rated if rated else None
            text = None
            if calls >= 20 and escalated > 0.3:
                text = (f"{row['model']} ({tier}): {escalated:.0%} of its {row['purpose']} calls escalate; "
                        f"consider {row['purpose']} = {raise_tier(tier)}")
            elif rated >= 20 and liked is not None and liked < 0.5:
                text = f"{row['model']} ({tier}): only {liked:.0%} of its rated {row['purpose']} answers are good"
            elif calls >= 20 and rated >= 20 and escalated < 0.05 and liked is not None and liked > 0.9 and tier != "small":
                text = (f"{row['model']} ({tier}) does well at {row['purpose']}: a smaller tier may do "
                        "for it, at lower cost")
            found.append({**row, "tier": tier, "escalation_rate": round(escalated, 3),
                          "good_rate": round(liked, 3) if liked is not None else None, "suggestion": text})
        return found

    def _maybe_adjust(self, every: float = 600.0) -> None:
        """`[llm.routing] feedback = adjust`: move a purpose's tier at most one step from its
        configured value, on enough evidence (feedback_min_samples), and log every change."""
        if self.settings.feedback != "adjust" or time.monotonic() - self._adjusted_at < every:
            return
        self._adjusted_at = time.monotonic()
        try:
            rows = self.usage.quality(dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=30))
        except Exception:
            LOG.exception("feedback could not be read")
            return
        per_purpose: dict[str, dict[str, int]] = {}
        for row in rows:
            totals = per_purpose.setdefault(row["purpose"], {"calls": 0, "up": 0, "down": 0, "escalations": 0})
            for key in totals:
                totals[key] += row[key]
        for purpose, t in per_purpose.items():
            if t["calls"] < self.settings.feedback_min_samples:
                continue
            configured = self.settings.tier_for(purpose)
            rated = t["up"] + t["down"]
            if t["escalations"] / t["calls"] > 0.3:
                wanted = raise_tier(configured)
            elif t["escalations"] / t["calls"] < 0.05 and rated and t["up"] / rated > 0.9:
                wanted = TIERS[max(0, tier_rank(configured) - 1)]
            else:
                wanted = configured
            if self.adjusted.get(purpose, configured) != wanted:
                LOG.warning("model routing: %s now needs tier %s (configured %s), from feedback", purpose, wanted, configured)
                self.adjusted[purpose] = wanted
