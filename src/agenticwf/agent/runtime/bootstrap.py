"""
The one module that imports across the runtime packages to assemble a working
RuntimeThreadManager.

Call `build_thread_manager()` once when the server starts. When a user opens a conversation (web
or CLI), the host calls `manager.start_thread(...)` to get a RuntimeThread; the user then talks to
that thread, whose Session records each message as an intent; workers turn intents into tasks and
run them through the AgentHarness (docs/agent-flow.md):

    build_thread_manager()          once per process: shared registry, policy, task types, store, LLM,
      │                             the AgentHarness and its Workers
      └─ start_thread(auth)         per conversation: RuntimeThread + Session (resident) + services
           └─ thread.send(text)     intent → worker: intake → tasks → turns → reply → events on the thread

Built once here: the HarnessServices (LLM client, ContextManager with memory and knowledge,
ToolDispatcher over the ToolRegistry, PolicyEngine, TaskTypeResolver, RuntimeStore, and the
PluginManager that feeds plugin tools, MCP tools and task types to the last ones), the one
AgentHarness that every thread shares, and the Workers that run its jobs. Per thread,
AgentHarness.services_factory derives that thread's HarnessServices; the harness routes each
intent's and task's events to the session that created it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from agenticwf.agent.core.common.config import config_get, config_has_option
from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.context.manager import ContextManager, MemorySettings
from agenticwf.agent.runtime.context.memory import KnowledgeCatalog, MemoryStore
from agenticwf.agent.runtime.context.task_types import TaskTypeResolver
from agenticwf.agent.runtime.embeddings import Embedder, EmbeddingIndex, EmbeddingSettings, HttpEmbedder
from agenticwf.agent.runtime.harness.harness import AgentHarness, CompletionHook, Limits
from agenticwf.agent.runtime.harness.planning import IntakeSettings
from agenticwf.agent.runtime.harness.services import HarnessServices
from agenticwf.agent.runtime.harness.store import InMemoryStore, RuntimeStore
from agenticwf.agent.runtime.harness.worker import Workers
from agenticwf.agent.runtime.llm.base import LLM
from agenticwf.agent.runtime.llm.catalog import ModelCatalog, ModelEntry, ModelStore, entries_from_sections
from agenticwf.agent.runtime.llm.router import ModelRouter
from agenticwf.agent.runtime.llm.routing import RoutingSettings
from agenticwf.agent.runtime.llm.usage import UsageStore
from agenticwf.agent.runtime.mcp.catalog import McpCatalog, McpStore
from agenticwf.agent.runtime.mcp.manager import McpManager, McpSettings
from agenticwf.agent.runtime.plugins.manager import PluginManager, PluginSettings
from agenticwf.agent.runtime.cron.catalog import CronCatalog
from agenticwf.agent.runtime.cron.service import CronService, CronSettings
from agenticwf.agent.runtime.policy.engine import DecisionKind, PolicyConfig, PolicyEngine, PolicyRule
from agenticwf.agent.runtime.skills.catalog import SkillCatalog, SkillStore
from agenticwf.agent.runtime.telemetry import Tracing
from agenticwf.agent.runtime.thread.manager import RuntimeThreadManager
from agenticwf.agent.runtime.tools.dispatcher import Hooks, ToolDispatcher
from agenticwf.agent.runtime.tools.prefs import ToolPrefs
from agenticwf.agent.runtime.tools.registry import DatabaseToolSource, ToolRegistry
from agenticwf.agent.runtime.tools.spec import Handler
from agenticwf.agent.runtime.types import Owner

LOG = get_logger(__name__)


@dataclass
class RuntimeConfig:
    """Runtime settings, normally read from the core configuration (agent.cfg)."""

    system_dir: Path
    plugin_dirs: list[Path] = field(default_factory=list)
    sandbox_root: Path | None = None   # per-owner sandboxes: user task types and plugins
    plugins: PluginSettings = field(default_factory=PluginSettings)
    mcp: McpSettings = field(default_factory=McpSettings)
    policy_file: Path | None = None
    model: str = "default"             # [llm] default: the default model id ("default": the catalog's first)
    provider: str | None = None        # [llm] provider: legacy single provider (echo), when no model is configured
    llm_fallbacks: list[str] = field(default_factory=list)          # [llm] fallback: ids tried on provider errors
    llm_allowed_hosts: tuple[str, ...] = ()                          # [llm] allowed_hosts: where users' own models may point
    llm_sections: dict[str, dict[str, str]] = field(default_factory=dict)   # [llm.model.*], [llm.provider.*]
    limits: Limits = field(default_factory=Limits)
    top_k: int = 15
    top_k_skills: int = 5              # skills offered at the start, besides the top_k tools
    schema_budget_tokens: int = 8000
    max_threads_per_owner: int | None = None
    intake: IntakeSettings = field(default_factory=IntakeSettings)
    memory: MemorySettings = field(default_factory=MemorySettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    cron: CronSettings = field(default_factory=CronSettings)
    workers: int = 4
    max_jobs_per_owner: int | None = None   # one owner's jobs at once in the pool; None: workers - 1; 0: no limit
    shared: bool = False                    # worker processes share the jobs (agenticwf-agent-worker)

    @classmethod
    def from_core_config(cls) -> "RuntimeConfig":
        """
        Sections (docs/agent-structure.md §9):
            [context]  system_dir, sandbox_root
            [tools]    plugin_dirs (comma or space separated), top_k, top_k_skills, schema_budget_tokens
            [plugins]  disabled (comma or space separated), user_ttl (docs/plugins.md)
            [mcp]      allowed_hosts (comma or space separated), connect_timeout, discovery_ttl,
                       discover (docs/mcp.md)
            [policy]   file
            [harness]  max_steps, max_tool_calls, deadline_seconds, max_threads_per_owner,
                       split (true/false), plan (always/never), workers, max_jobs_per_owner,
                       shared (true/false: worker processes share the jobs), conversation_messages,
                       conversation_tokens, conversation_keep (chat compaction)
            [memory]   consolidate_failed (true/false), extract (llm/notes), model, max_per_task
            [cron]     enabled, tick (seconds), system_scopes, max_scopes, deny_tools (docs/cron.md)
            [embeddings] provider (voyage/openai/ollama/vllm; empty: off), model, base_url, api_key_env,
                       weight, min_similarity, timeout: match tools, skills and memories by meaning too
            [llm]      default (a model id), fallback (ids, comma separated), allowed_hosts (where
                       models users add may point; docs/agent-models.md); legacy: provider, model
            [llm.model.<id>]        provider, model, label, description, base_url, api_key_env,
                                    effort, thinking, fallbacks, max_tokens, extra, … (docs/agent-models.md)
            [llm.provider.<name>]   api_key_env, base_url, and defaults for that provider's models
            [llm.routing]  mode (auto/pinned), <purpose> = <tier>, context_headroom, max_escalations,
                           task_budget, feedback (suggest/adjust), feedback_min_samples
            [llm.purpose]  <purpose> = <model id>     [llm.budgets] default, organization.<org>,
                           user.<subject> = <n> per day|month; max_tier[.<subject>|.organization.<org>]
                           (docs/model-routing.md)
        """
        def opt(section: str, option: str, default: str | None = None) -> str | None:
            return config_get(section, option) if config_has_option(section, option) else default

        system_dir = opt("context", "system_dir", "/etc/agenticwf/agent")
        limits = Limits(
            max_steps=int(opt("harness", "max_steps", "50")),
            max_tool_calls=int(opt("harness", "max_tool_calls", "200")),
            deadline_seconds=float(opt("harness", "deadline_seconds", "3600")),
            conversation_messages=int(opt("harness", "conversation_messages", "20")),
            conversation_tokens=int(opt("harness", "conversation_tokens", "8000")),
            conversation_keep=int(opt("harness", "conversation_keep", "4")),
        )
        sandbox = opt("context", "sandbox_root")
        policy = opt("policy", "file")
        threads = opt("harness", "max_threads_per_owner", "100")   # live threads per person or client; 0: no limit
        intake = IntakeSettings(split=(opt("harness", "split", "false") or "").lower() in ("1", "true", "yes"),
                                plan=opt("harness", "plan", "never"))
        words = lambda section, option: tuple((opt(section, option, "") or "").replace(",", " ").split())  # noqa: E731
        flag = lambda section, option, default: (opt(section, option, default) or "").lower() in ("1", "true", "yes")  # noqa: E731
        extract = (opt("memory", "extract", "llm") or "llm").lower()
        if extract not in ("llm", "notes"):
            raise ValueError(f"[memory] extract must be llm or notes, not {extract!r}")
        memory = MemorySettings(consolidate_failed=flag("memory", "consolidate_failed", "false"), extract=extract,
                                model=opt("memory", "model") or None,
                                max_per_task=int(opt("memory", "max_per_task", "5")))
        embeddings = EmbeddingSettings(
            provider=(opt("embeddings", "provider", "") or "").lower(), model=opt("embeddings", "model", "") or "",
            base_url=opt("embeddings", "base_url") or None, api_key_env=opt("embeddings", "api_key_env") or None,
            weight=float(opt("embeddings", "weight", "0.6")),
            min_similarity=float(opt("embeddings", "min_similarity", "0.3")),
            timeout=float(opt("embeddings", "timeout", "5"))).check()
        def scope_set(option: str) -> frozenset[str] | None:
            return frozenset(words("cron", option)) if config_has_option("cron", option) else None

        cron = CronSettings(enabled=flag("cron", "enabled", "true"), tick=float(opt("cron", "tick", "30")),
                            system_scopes=scope_set("system_scopes"), max_scopes=scope_set("max_scopes"),
                            deny_tools=words("cron", "deny_tools"))
        mcp = McpSettings(
            allowed_hosts=words("mcp", "allowed_hosts"),
            connect_timeout=float(opt("mcp", "connect_timeout", "15")),
            discovery_ttl=float(opt("mcp", "discovery_ttl", "300")),
            discover=(opt("mcp", "discover", "true") or "").lower() in ("1", "true", "yes"),
        )
        return cls(
            system_dir=Path(system_dir),
            plugin_dirs=[Path(p) for p in (opt("tools", "plugin_dirs", "") or "").replace(",", " ").split()],
            sandbox_root=Path(sandbox) if sandbox else None,
            plugins=PluginSettings(disabled=words("plugins", "disabled"),
                                   user_ttl=float(opt("plugins", "user_ttl", "5"))),
            mcp=mcp,
            policy_file=Path(policy) if policy else None,
            model=opt("llm", "default") or opt("llm", "model", "default"),
            provider=opt("llm", "provider"),
            llm_fallbacks=(opt("llm", "fallback", "") or "").replace(",", " ").split(),
            llm_allowed_hosts=words("llm", "allowed_hosts"),
            llm_sections=_llm_sections(),
            limits=limits,
            top_k=int(opt("tools", "top_k", "15")),
            top_k_skills=int(opt("tools", "top_k_skills", "5")),
            schema_budget_tokens=int(opt("tools", "schema_budget_tokens", "8000")),
            max_threads_per_owner=int(threads) if threads and int(threads) > 0 else None,
            intake=intake,
            memory=memory,
            embeddings=embeddings,
            cron=cron,
            workers=int(opt("harness", "workers", "4")),
            max_jobs_per_owner=int(opt("harness", "max_jobs_per_owner")) if opt("harness", "max_jobs_per_owner") else None,
            # AGENT_HARNESS_SHARED (the container's entrypoint, with AGENT_WORKER_PROCESSES) wins
            shared=(os.environ.get("AGENT_HARNESS_SHARED", "").lower() in ("1", "true", "yes")
                    or flag("harness", "shared", "false")),
        )


@dataclass
class EngineOptions:
    """Knobs and collaborators a host passes in without editing this module."""

    llm: LLM | None = None                          # replaces the model router (tests, scripted demos)
    model_store: ModelStore | None = None           # models managers add at run time (the API's database)
    handlers: dict[str, Handler] = field(default_factory=dict)   # builtin handler refs → callables
    task_store: RuntimeStore | None = None          # TODO: database store; in memory by default
    database_tools: DatabaseToolSource | None = None
    mcp_store: McpStore | None = None               # MCP endpoints users add at run time (the API's database)
    skill_store: SkillStore | None = None           # skills users add at run time (the API's database)
    embedder: Embedder | None = None                # replaces the [embeddings] provider (tests)
    crons: bool = False                             # run crons (they need the database: the API turns them on)
    usage_store: UsageStore | None = None           # model usage and feedback (the API's database); None: in memory
    memory: MemoryStore | None = None               # long-term memory (the API's database); None: none, no memory_search
    knowledge: KnowledgeCatalog | None = None
    web: Any = None                                 # agenticwf.agent.runtime.web.WebSearch; None: from [web_search]
    tool_prefs: ToolPrefs | None = None             # tools users switch off (the API's database); None: in memory
    on_complete: list[CompletionHook] = field(default_factory=list)
    tool_hooks: Hooks | None = None


def sandbox_resolver(root: Path | None) -> Callable[[Owner], Path | None] | None:
    """Per-owner sandbox directory. The name is a hash, so no subject can become a path."""
    if root is None:
        return None

    def resolve(owner: Owner) -> Path:
        digest = hashlib.sha256(f"{owner[0]}\0{owner[1] or ''}".encode()).hexdigest()[:32]
        return root / digest

    return resolve


def build_plugin_manager(config: RuntimeConfig, mcp: McpManager | None = None) -> PluginManager:
    return PluginManager(config.plugin_dirs, user_sandbox=sandbox_resolver(config.sandbox_root),
                         settings=config.plugins, mcp=mcp or McpManager(config.mcp))


def build_tool_registry(config: RuntimeConfig, options: EngineOptions, plugins: PluginManager | None = None,
                        mcp_catalog: McpCatalog | None = None, skills: SkillCatalog | None = None,
                        embeddings: EmbeddingIndex | None = None) -> ToolRegistry:
    """Sources: system tools, plugins (with their MCP endpoints), the database (users' MCP
    endpoints) and the skills (system, plugins', users' and the database's)."""
    databases = [d for d in (options.database_tools, mcp_catalog) if d is not None]
    registry = ToolRegistry(system_dir=config.system_dir, database=databases,
                            plugins=plugins or build_plugin_manager(config), skills=skills,
                            embeddings=embeddings)
    errors = registry.load()
    if errors:
        LOG.warning("%d tool definition(s) skipped; see the log above", len(errors))
    for ref, handler in options.handlers.items():
        registry.register_handler(ref, handler)
    return registry


def build_embeddings(config: RuntimeConfig, options: EngineOptions) -> EmbeddingIndex | None:
    """Matching by meaning ([embeddings]) for tools, skills and long-term memory; None: by words only."""
    settings = config.embeddings
    if options.embedder is not None:
        return EmbeddingIndex(options.embedder, settings)
    if not settings.enabled:
        return None
    LOG.info("tools and skills are matched by meaning too: %s %s", settings.provider, settings.model)
    return EmbeddingIndex(HttpEmbedder(settings), settings)


def _llm_sections() -> dict[str, dict[str, str]]:
    from agenticwf.agent.core.common.config import get_config

    parser = get_config()
    return {name: {k: v for k, v in parser.items(name) if k not in parser.defaults()}
            for name in parser.sections()
            if name.startswith(("llm.model.", "llm.provider.")) or name in ("llm", "llm.routing", "llm.purpose", "llm.budgets")}


def build_catalog(config: RuntimeConfig, store: ModelStore | None = None) -> ModelCatalog:
    models, providers = entries_from_sections(config.llm_sections)
    default = None if config.model == "default" else config.model
    if not models and config.provider:   # legacy: `[llm] provider = echo`
        models = [ModelEntry(id=config.provider, provider=config.provider,
                             model=config.model if default else config.provider)]
        default = config.provider
    return ModelCatalog(models, providers, default=default, fallbacks=config.llm_fallbacks, store=store,
                        allowed_hosts=config.llm_allowed_hosts)


def build_llm(config: RuntimeConfig, store: ModelStore | None = None, usage: UsageStore | None = None) -> LLM:
    """The model router over the configured models and, with `store`, the database's; routing by
    [llm.routing], [llm.purpose], [llm.budgets] (docs/model-routing.md); usage recorded in `usage`."""
    catalog = build_catalog(config, store)
    if not catalog.entries():
        raise ValueError("no model configured: add [llm.model.<id>] sections (docs/agent-models.md), "
                         "set [llm] provider = echo, or pass EngineOptions(llm=...)")
    if any(e.provider == "echo" for e in catalog.entries()):
        LOG.warning("an `echo` model is configured: its replies repeat the request (development only)")
    return ModelRouter(catalog, settings=RoutingSettings.parse(config.llm_sections), usage=usage)


def build_thread_manager(config: RuntimeConfig | None = None, options: EngineOptions | None = None) -> RuntimeThreadManager:
    """Build a RuntimeThreadManager whose threads run tasks through the AgentHarness."""
    config = config or RuntimeConfig.from_core_config()
    options = options or EngineOptions()

    # Shared, process-lifetime pieces.
    mcp = McpManager(config.mcp)
    plugins = build_plugin_manager(config, mcp)
    mcp_catalog = McpCatalog(options.mcp_store, mcp) if options.mcp_store else None
    skills = SkillCatalog(config.system_dir, user_sandbox=sandbox_resolver(config.sandbox_root), plugins=plugins,
                          store=options.skill_store)
    embeddings = build_embeddings(config, options)   # one index: tools, skills and memories
    if embeddings is not None and hasattr(options.memory, "use_embeddings"):
        options.memory.use_embeddings(embeddings)
    registry = build_tool_registry(config, options, plugins, mcp_catalog, skills, embeddings)
    policy_config = PolicyConfig.from_yaml(config.policy_file) if config.policy_file else PolicyConfig()
    if config.cron.deny_tools:   # e.g. ticket tools: a cron never files tickets (docs/cron.md §6)
        policy_config = dataclasses.replace(policy_config, rules=(
            *(PolicyRule(glob, DecisionKind.DENY, "not allowed in crons", sources=("cron",))
              for glob in config.cron.deny_tools), *policy_config.rules))
    policy = PolicyEngine(policy_config)
    task_types = TaskTypeResolver(config.system_dir, user_sandbox=sandbox_resolver(config.sandbox_root),
                                  plugins=plugins)
    task_types.check()   # refuse to start without the default AGENTS.md / WORKFLOW.md
    store = options.task_store or InMemoryStore()
    llm = options.llm or build_llm(config, options.model_store, options.usage_store)
    context_manager = ContextManager(options.memory, options.knowledge, settings=config.memory)
    for spec, handler in context_manager.tools():   # memory_search
        registry.add_core(spec, handler)
    from agenticwf.agent.runtime.web import WebSearch, WebSettings, web_tools

    web = options.web or WebSearch(WebSettings.from_core_config())
    if web.settings.uses_kb and web.kb is None:          # through the knowledge service
        from agenticwf.agent.runtime.knowledge import KnowledgeClient, KnowledgeSettings

        client = getattr(options.knowledge, "client", None)       # the catalogue's client, when there is one
        web.kb = client if isinstance(client, KnowledgeClient) else KnowledgeClient(KnowledgeSettings.from_core_config())
    if web.settings.enabled:                         # web_search, web_fetch ([web_search] providers)
        for spec, handler in web_tools(web):
            registry.add_core(spec, handler)
    always = ("web_search",) if web.settings.enabled and web.settings.always_offer else ()   # the model decides
    dispatcher = ToolDispatcher(registry, policy, hooks=options.tool_hooks, top_k=config.top_k, always=always,
                                prefs=options.tool_prefs, schema_budget_tokens=config.schema_budget_tokens, top_k_skills=config.top_k_skills,
                                hidden=getattr(options.knowledge, "hidden_tools", None))   # [knowledge] tools, per user
    services = HarnessServices(llm=llm, tool_dispatcher=dispatcher, task_types=task_types, policy=policy,
                             context_manager=context_manager, store=store, plugins=plugins, mcp=mcp_catalog,
                             skills=skills, otel=Tracing())
    harness = AgentHarness(services, limits=config.limits, model=config.model, on_complete=options.on_complete,
                           intake=config.intake)
    manager = RuntimeThreadManager(harness, Workers(harness, size=config.workers, per_owner=config.max_jobs_per_owner),
                                   max_threads_per_owner=config.max_threads_per_owner)
    if options.crons:
        catalog = CronCatalog(config.system_dir, user_sandbox=sandbox_resolver(config.sandbox_root), plugins=plugins)
        manager.crons = CronService(harness, catalog, config.cron)
    return manager
