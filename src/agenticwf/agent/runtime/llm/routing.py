"""
Choosing a model by what a call needs and what it costs (docs/model-routing.md).

    a model declares    tier, context_window, max_output, capabilities, costs, speed, routable
                        (its options: [llm.model.<id>], or a model added on the admin page)
    a call declares     its purpose (step, plan, split, reply, compact, ...), a pin if a model
                        was chosen, a minimum tier, the task's current model
    the router          pinned → that model (if it fits); otherwise the models the owner may use
                        that fit the call (size, tools) and are capable enough (tier), the
                        owner's own first, then the cheapest; the task's current model keeps
                        its prompt cache, every other is priced without one

Settings: [llm.routing], [llm.purpose], [llm.budgets] and [llm] cost_unit (RoutingSettings).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from agenticwf.agent.runtime.llm.base import LLMRequest
from agenticwf.agent.runtime.types import Owner

TIERS = ("small", "medium", "large")
SPEEDS = ("fast", "normal", "slow")
CAPABILITIES = ("tools", "vision", "reasoning", "json")
PURPOSES = {   # purpose -> default tier
    "step": "medium", "plan": "medium", "split": "small", "reply": "small", "compact": "small",
    "compact_conversation": "small", "consolidate": "small", "final_report": "medium",
    "cron_check": "small", "cron_report": "small",
}
TYPICAL_OUTPUT = {"step": 1000, "plan": 300, "split": 300, "final_report": 1000}   # tokens; summaries: 500
ROUTING_OPTION_KEYS = ("tier", "context_window", "max_output", "capabilities", "cost_input", "cost_output",
                       "cost_cached_input", "speed", "routable")


def tier_rank(tier: str | None) -> int:
    return TIERS.index(tier) if tier in TIERS else TIERS.index("medium")


def raise_tier(tier: str, steps: int = 1) -> str:
    return TIERS[min(len(TIERS) - 1, tier_rank(tier) + steps)]


# --- what a model declares ------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelProfile:
    tier: str = "medium"
    context_window: int | None = None
    max_output: int | None = None
    capabilities: frozenset[str] = frozenset({"tools"})
    cost_input: float = 0.0          # per million tokens
    cost_output: float = 0.0
    cost_cached_input: float | None = None
    speed: str = "normal"
    routable: bool = True
    declared: bool = False           # declared a tier or a context window: the router may choose it

    @classmethod
    def of(cls, options: dict[str, Any]) -> "ModelProfile":
        caps = options.get("capabilities")
        if isinstance(caps, str):
            caps = [c for c in re.split(r"[\s,]+", caps) if c]
        return cls(
            tier=str(options.get("tier") or "medium"),
            context_window=int(options["context_window"]) if options.get("context_window") else None,
            max_output=int(options["max_output"]) if options.get("max_output") else None,
            capabilities=frozenset(caps) if caps is not None else frozenset({"tools"}),
            cost_input=float(options.get("cost_input") or 0), cost_output=float(options.get("cost_output") or 0),
            cost_cached_input=float(options["cost_cached_input"]) if options.get("cost_cached_input") not in (None, "") else None,
            speed=str(options.get("speed") or "normal"), routable=_flag(options.get("routable", True)),
            declared="tier" in options or "context_window" in options)

    def cost(self, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
        cached_price = self.cost_input if self.cost_cached_input is None else self.cost_cached_input
        fresh = max(0, input_tokens - cached_tokens)
        return (fresh * self.cost_input + cached_tokens * cached_price + output_tokens * self.cost_output) / 1_000_000


def check_options(options: dict[str, Any]) -> None:
    """ValueError when a model's routing options are wrong."""
    if "tier" in options and options["tier"] not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}")
    if "speed" in options and options["speed"] not in SPEEDS:
        raise ValueError(f"speed must be one of {', '.join(SPEEDS)}")
    for key in ("context_window", "max_output"):
        if key in options and (not isinstance(options[key], int) or options[key] <= 0):
            raise ValueError(f"{key} must be a positive whole number of tokens")
    for key in ("cost_input", "cost_output", "cost_cached_input"):
        if key in options and (not isinstance(options[key], (int, float)) or options[key] < 0):
            raise ValueError(f"{key} must be a number, at least 0 (per million tokens)")
    caps = ModelProfile.of(options).capabilities
    unknown = caps - set(CAPABILITIES)
    if unknown:
        raise ValueError(f"capabilities: unknown {', '.join(sorted(unknown))} (use {', '.join(CAPABILITIES)})")


def coerce_option(key: str, value: str) -> Any:
    if key in ("context_window", "max_output"):
        return int(value)
    if key in ("cost_input", "cost_output", "cost_cached_input"):
        return float(value)
    if key == "routable":
        return _flag(value)
    if key == "capabilities":
        return [c for c in re.split(r"[\s,]+", value) if c]
    return value


def _flag(value: Any) -> bool:
    return value if isinstance(value, bool) else str(value).strip().lower() in ("1", "true", "yes", "on")


# --- settings -------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Budget:
    amount: float
    period: str          # day | month

    @classmethod
    def parse(cls, text: str) -> "Budget":
        match = re.match(r"^\s*([0-9.]+)\s*(?:per\s+)?(day|month)\s*$", str(text))
        if not match:
            raise ValueError(f"a budget is '<amount> per day' or '<amount> per month', not {text!r}")
        return cls(float(match.group(1)), match.group(2))

    def start(self, now: dt.datetime) -> dt.datetime:
        day = now.astimezone(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        return day.replace(day=1) if self.period == "month" else day


@dataclass(frozen=True)
class RoutingSettings:
    mode: str = "auto"                                   # auto | pinned
    tiers: dict[str, str] = field(default_factory=lambda: dict(PURPOSES))
    purpose_models: dict[str, str] = field(default_factory=dict)   # [llm.purpose]: purpose -> model id
    headroom: float = 0.25
    max_escalations: int = 1
    task_budget: float | None = None
    feedback: str = "suggest"                            # suggest | adjust
    feedback_min_samples: int = 200
    budgets: dict[str, Budget] = field(default_factory=dict)        # "default", "organization.<org>", "user.<subject>"
    max_tiers: dict[str, str] = field(default_factory=dict)         # "default", "<subject>" or "organization.<org>"
    cost_unit: str = "units"

    def tier_for(self, purpose: str) -> str:
        return self.tiers.get(purpose) or PURPOSES.get(purpose, "medium")

    def budget_for(self, owner: Owner | None) -> Budget | None:
        if owner is None:
            return None
        subject, org = owner
        return (self.budgets.get(f"user.{subject}") or (self.budgets.get(f"organization.{org}") if org else None)
                or self.budgets.get("default"))

    def max_tier_for(self, owner: Owner | None) -> str:
        if owner is not None:
            subject, org = owner
            found = self.max_tiers.get(subject) or (self.max_tiers.get(f"organization.{org}") if org else None)
            if found:
                return found
        return self.max_tiers.get("default", TIERS[-1])

    @classmethod
    def parse(cls, sections: dict[str, dict[str, str]]) -> "RoutingSettings":
        routing = sections.get("llm.routing", {})
        tiers = dict(PURPOSES)
        for purpose in PURPOSES:
            if routing.get(purpose):
                if routing[purpose] not in TIERS:
                    raise ValueError(f"[llm.routing] {purpose} must be one of {', '.join(TIERS)}")
                tiers[purpose] = routing[purpose]
        budgets, max_tiers = {}, {}
        for key, value in sections.get("llm.budgets", {}).items():
            if key == "max_tier" or key.startswith("max_tier."):
                if value not in TIERS:
                    raise ValueError(f"[llm.budgets] {key} must be one of {', '.join(TIERS)}")
                max_tiers[key.partition(".")[2] or "default"] = value
            else:
                budgets[key] = Budget.parse(value)
        mode = routing.get("mode", "auto")
        feedback = routing.get("feedback", "suggest")
        if mode not in ("auto", "pinned") or feedback not in ("suggest", "adjust"):
            raise ValueError("[llm.routing] mode is auto or pinned; feedback is suggest or adjust")
        return cls(mode=mode, tiers=tiers, purpose_models=dict(sections.get("llm.purpose", {})),
                   headroom=float(routing.get("context_headroom", 0.25)),
                   max_escalations=int(routing.get("max_escalations", 1)),
                   task_budget=float(routing["task_budget"]) if routing.get("task_budget") else None,
                   feedback=feedback, feedback_min_samples=int(routing.get("feedback_min_samples", 200)),
                   budgets=budgets, max_tiers=max_tiers, cost_unit=sections.get("llm", {}).get("cost_unit", "units"))


# --- what a call needs ----------------------------------------------------------------------------
def estimate_tokens(request: LLMRequest) -> int:
    """Rough size of a request in tokens (4 characters per token)."""
    chars = len(request.system) + sum(len(m.content or "") + len(str(m.tool_calls or "")) for m in request.messages)
    chars += sum(len(str(t)) for t in request.tools)
    return chars // 4 + 50


class OutputEstimator:
    """Expected output tokens: a running (exponential) average per purpose and model; a model new
    to a purpose uses the purpose's average over all models; with none, the purpose's typical length."""

    def __init__(self, alpha: float = 0.2):
        self.alpha = alpha
        self._by_model: dict[tuple[str, str], float] = {}
        self._by_purpose: dict[str, float] = {}

    def expect(self, purpose: str, model_id: str) -> int:
        found = self._by_model.get((purpose, model_id)) or self._by_purpose.get(purpose)
        return int(found) if found else TYPICAL_OUTPUT.get(purpose, 500)

    def observe(self, purpose: str, model_id: str, output_tokens: int) -> None:
        for table, key in ((self._by_model, (purpose, model_id)), (self._by_purpose, purpose)):
            old = table.get(key)
            table[key] = float(output_tokens) if old is None else old + self.alpha * (output_tokens - old)

    def seed(self, rows: Iterable[tuple[str, str, float]]) -> None:
        """Averages from earlier runs (the usage records), so a restart does not start from guesses."""
        for purpose, model_id, average in rows:
            self._by_model[(purpose, model_id)] = average
            old = self._by_purpose.get(purpose)
            self._by_purpose[purpose] = average if old is None else (old + average) / 2


@dataclass(frozen=True)
class Choice:
    entry: Any                       # ModelEntry
    reason: str                      # pinned | purpose | routed | default | context | budget
    estimated_cost: float = 0.0
    detail: str = ""
