"""The agent runtime behind the HTTP API (docs/agent-flow.md, docs/api.md).

    chat session  ── is a ──►  RuntimeThread (same id), resident while open; its transcript is
                               agent_chat_messages
    POST /tasks   ── is an ─►  intent on a thread of its own (session_source "api"), closed when
                               the reply is written

Each open thread has a relay: it reads the thread's events, forwards them to the owner's live
update stream, writes replies into the transcript, and wakes requests waiting for an outcome.
At startup `recover()` reopens the threads that have unanswered intents and queues their work
again; nothing is failed because the process restarted.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.context.task_types import PROMPTS, PROMPTS_DIR, TASK_PROMPTS, TOOL_WORDING_DIR
from agenticwf.agent.runtime.harness.state import WAITING_STATES
from agenticwf.agent.runtime.llm.catalog import ModelCatalog
from agenticwf.agent.runtime.session.state import TranscriptMessage
from agenticwf.agent.runtime.messages import ThreadEvent
from agenticwf.agent.runtime.overrides import (BUDGETS, HARNESS, PURPOSE, ROUTING, TASK_TYPE, check, effective, key_spec,
                                               keys_for)
from agenticwf.agent.runtime.records import Intent, IntentStatus
from agenticwf.agent.runtime.task import Task, ToolCallStatus
from agenticwf.agent.runtime.thread.manager import RuntimeThreadManager, StartThreadOptions
from agenticwf.agent.runtime.thread.thread import RuntimeThread
from agenticwf.agent.runtime.tools.loader import CORE_TOOLS_DIR, core_tool_names, core_tool_text
from agenticwf.agent.runtime.types import Mode, RuntimeState, owner_key

from agenticwf.agent.api.events import Updates
from agenticwf.agent.api.store import API_SOURCE, Store

LOG = get_logger(__name__)


class Conflict(Exception):
    """The request does not fit the thread's state (nothing to approve or answer)."""


class AgentService:
    def __init__(self, manager: RuntimeThreadManager, store: Store, updates: Updates, reply_timeout: float = 120.0):
        self.manager = manager
        self.harness = manager.harness
        self.store = store
        self.updates = updates
        self.reply_timeout = reply_timeout
        self._relays: dict[str, asyncio.Task] = {}
        self._waiters: dict[str, list[asyncio.Future]] = defaultdict(list)   # intent id → requests waiting
        self._background: set[asyncio.Task] = set()
        self._open_lock = asyncio.Lock()

    # ============================================================================================
    # Threads
    # ============================================================================================
    async def _open(self, options: StartThreadOptions) -> RuntimeThread:
        thread = (await self.manager.start_thread(options)).thread
        self._relays[thread.thread_id] = asyncio.create_task(self._relay(thread))
        return thread

    async def chat_thread(self, sid: str, ctx: AuthContext) -> RuntimeThread | None:
        """The resident thread of a chat session, opened from its transcript on first use."""
        async with self._open_lock:
            thread = await self.manager.get_thread(sid, ctx)
            if thread is not None:
                return thread
            session = await asyncio.to_thread(self.store.get_session, owner_key(ctx), sid)
            if session is None:
                return None
            return await self._open(StartThreadOptions(auth=ctx, session_source="web", reserved_thread_id=sid,
                                                       **_transcript(session)))

    async def chat_status(self, sid: str, ctx: AuthContext) -> dict[str, Any]:
        """idle | running | waiting, with what the current intent waits for. A session that is not
        open in this process is idle: opening one does not start anything."""
        thread = await self.manager.get_thread(sid, ctx)
        if thread is None:
            return {"session": sid, "status": "idle", "waiting": []}
        intent_id = thread.session.state.active_intent_id
        waiting = [_pending(t) for t in self._tasks(intent_id) if t.state in WAITING_STATES] if intent_id else []
        return {"session": sid, "status": thread.status(), "intent": intent_id, "waiting": waiting}

    # -- models (docs/agent-models.md) -----------------------------------------------------------
    @property
    def catalog(self) -> ModelCatalog | None:
        """The model router's catalog; None when the runtime was given a single LLM (tests)."""
        return getattr(self.harness.services.llm, "catalog", None)

    def models(self, ctx: AuthContext, detail: bool = False) -> dict[str, Any]:
        """The models the caller may use and their default (their own choice, else [llm] default);
        with `detail`, the settings page's view (groups: system, shared, mine; settings of the
        caller's own entries, never their keys; the system default beside the caller's)."""
        catalog = self.catalog
        if catalog is None:
            return {"default": None, "models": [], "can_add": False}
        owner = owner_key(ctx)
        if detail:
            entries = catalog.entries(include_disabled=True, owner=owner)
            return {"default": catalog.default_for(owner), "system_default": catalog.default_id,
                    "user_default": catalog.user_default(owner), "can_add": catalog.store is not None,
                    "models": [e.view_for(owner) for e in entries if e.enabled or e.owned_by(owner)]}
        return {"default": catalog.default_for(owner), "models": [e.public() for e in catalog.entries(owner=owner)]}

    def model_available(self, model_id: str, ctx: AuthContext) -> bool:
        catalog = self.catalog
        return catalog is not None and catalog.is_available(model_id, owner_key(ctx))

    @property
    def task_type_resolver(self):
        return self.harness.services.task_types

    def task_types(self, ctx: AuthContext) -> list[dict[str, Any]]:
        """System, plugin and the caller's own task types (docs/agent-runtime.md §4)."""
        return [info.to_dict() for info in self.task_type_resolver.catalogue(owner_key(ctx))]

    def task_type(self, name: str, ctx: AuthContext) -> dict[str, Any] | None:
        """One task type with its instruction files as they resolve (inherited ones marked)."""
        owner = owner_key(ctx)
        resolver = self.task_type_resolver
        info = resolver.info(name, owner)
        if info is None:
            return None
        resolved = resolver.resolve(info.id, owner)
        files = {f.name: {"content": f.content, "source": f.source, "inherited": f.name not in info.files}
                 for f in (resolved.agents, resolved.workflow)}
        files.update({path: {"content": text, "source": f"{info.source}:{info.name}/{path}", "inherited": False}
                      for path, text in resolver.extra_files(info.id, owner).items()})
        layers = resolver.settings_layers(info.id, owner)
        if layers["task_yaml"]["exists"]:   # read-only: only operators write it
            files["task.yaml"] = {k: layers["task_yaml"][k] for k in ("content", "source", "inherited")}
        return {**info.to_dict(), "content": files, "can_edit_user_types": resolver.user_sandbox is not None,
                "settings_section": layers["section"], "settings_inherit": layers["inherits"],
                "settings_rows": _setting_rows(layers["section"], list(keys_for(layers["section"])), layers["file"],
                                               layers["all"], layers["mine"], layers["effective"]),
                "can_set_all": "admin" in ctx.scopes}

    # -- settings changed on the admin page, over agent.cfg and task.yaml (agenticwf.agent.runtime.overrides)
    def _router(self):
        llm = self.harness.services.llm
        return llm if hasattr(llm, "configured") else None   # a ModelRouter; tests may run another LLM

    def settings(self, ctx: AuthContext) -> dict[str, Any]:
        """[harness], [llm.routing], [llm.purpose] and [llm.budgets]: agent.cfg's values, everyone's
        changes, the caller's, and those in effect for the caller. Budgets: the default's and, for
        others than admins, only their own and their organization's."""
        owner, store, admin = owner_key(ctx), self.harness.overrides, "admin" in ctx.scopes
        sections = [(HARNESS, "Harness", "agent.cfg [harness]", self.harness.harness_view())]
        router = self._router()
        if router is not None:
            sections += [(ROUTING, "Model routing", "agent.cfg [llm.routing]", router.configured.view()),
                         (PURPOSE, "Model per purpose", "agent.cfg [llm.purpose]", router.configured.purpose_view()),
                         (BUDGETS, "Budgets", "agent.cfg [llm.budgets]", router.configured.budgets_view())]
        out = []
        for section, title, source, base in sections:
            everyone = store.values(section) if store else {}
            mine = store.values(section, owner) if store else {}
            keys = list(keys_for(section))
            if section == BUDGETS:
                keys += sorted({*base, *everyone} - set(keys))
                if not admin:
                    yours = {f"user.{owner[0]}", f"max_tier.{owner[0]}",
                             *((f"organization.{owner[1]}", f"max_tier.organization.{owner[1]}") if owner[1] else ())}
                    keys = [k for k in keys if k in keys_for(BUDGETS) or k in yours]
            rows = _setting_rows(section, keys, base, everyone, mine, effective(store, section, base, owner))
            if section == PURPOSE:
                models = [e.id for e in router.catalog.entries()]
                rows = [{**r, "kind": "choice", "choices": models} for r in rows]
            out.append({"section": section, "title": title, "source": source, "rows": rows,
                        "add_keys": section == BUDGETS and admin})
        return {"sections": out, "can_set_all": admin}

    def set_setting(self, ctx: AuthContext, section: str, key: str, value: Any, everyone: bool) -> dict[str, Any]:
        """Change one setting for everyone (admins) or for the caller's own tasks; None: back to the
        layer below. KeyError: no such section or key the caller may see; PermissionError: not
        theirs to change; ValueError: the value does not fit."""
        owner, store = owner_key(ctx), self.harness.overrides
        if store is None:
            raise PermissionError("this agent keeps no settings changes")
        if section.startswith(TASK_TYPE):
            info = self.task_type_resolver.info(section[len(TASK_TYPE):], owner)
            if info is None:
                raise KeyError(section)
            section = TASK_TYPE + info.id
        elif section not in (HARNESS, ROUTING, PURPOSE, BUDGETS) or (section != HARNESS and self._router() is None):
            raise KeyError(section)
        spec = key_spec(section, key)
        if spec is None:
            raise KeyError(key)
        if everyone and "admin" not in ctx.scopes:
            raise PermissionError("only managers (admin scope) change a setting for everyone")
        if not everyone and not spec.personal:
            raise PermissionError(f"{key} is changed for everyone only, by managers")
        if value is not None:
            value = check(section, key, value, personal=not everyone)
            if section == PURPOSE and self._router().catalog.get(value) is None:
                raise ValueError(f"no model {value!r} (the Models tab lists them)")
        store.set(section, key, value, None if everyone else owner, ctx.subject)
        return {"section": section, "key": key, "value": value, "scope": "all" if everyone else "mine"}

    def instructions(self, ctx: AuthContext) -> dict[str, Any]:
        """What the system ships besides task types, to read and clone into a type of one's own:
        the runtime prompts (as operators set them, or packaged) and the wording of core tools."""
        resolver = self.task_type_resolver
        prompts = []
        for name in PROMPTS:
            found = resolver.prompt(name)
            prompts.append({"name": name, "path": f"{PROMPTS_DIR}/{name}.md", "per_type": name in TASK_PROMPTS,
                            "source": found.source, "content": found.content})
        tools = []
        for name in core_tool_names():
            summary, _, _ = core_tool_text(name)
            tools.append({"name": name, "path": f"{TOOL_WORDING_DIR}/{name}.md", "summary": summary,
                          "source": f"packaged:core-tools/{name}.md",
                          "content": (CORE_TOOLS_DIR / f"{name}.md").read_text()})
        return {"prompts": prompts, "core_tools": tools, "can_add": resolver.user_sandbox is not None}

    # -- plugins (docs/plugins.md) ----------------------------------------------------------------
    def plugins(self, ctx: AuthContext) -> list[dict[str, Any]]:
        """System plugins and the caller's own, with what each provides."""
        manager, owner = self.harness.services.plugins, owner_key(ctx)
        return [p.to_dict(owner) for p in manager.plugins(owner)] if manager else []

    def reload_plugins(self) -> list[str]:
        """Read the system plugins again and refresh the tool catalogue; returns the errors."""
        return self.harness.services.tool_dispatcher.registry.load()

    @property
    def skills(self):
        """The SkillCatalog (docs/skills.md), or None."""
        return self.harness.services.skills

    # -- long-term memory (docs/admin-page.md, Memory) --------------------------------------------
    def memories(self, ctx: AuthContext, limit: int, offset: int) -> dict[str, Any]:
        """The caller's own memories, newest first; `enabled` is false when the agent keeps none."""
        manager = self.harness.services.context_manager
        items, total = manager.memory.list(owner_key(ctx), limit, offset)
        return {"memories": [m.to_dict() for m in items], "total": total, "enabled": manager.remembers}

    def forget(self, ctx: AuthContext, memory_id: str | None = None) -> int:
        """Delete one of the caller's memories, or all of them."""
        return self.harness.services.context_manager.memory.forget(owner_key(ctx), memory_id)

    # -- MCP endpoints and tools (docs/mcp.md) ----------------------------------------------------
    @property
    def mcp_catalog(self):
        """Endpoints users add (agenticwf.agent.runtime.mcp.catalog); None when the runtime has no store for them."""
        return self.harness.services.mcp

    def mcp_endpoints(self, ctx: AuthContext) -> dict[str, Any]:
        """Endpoints the caller may see: installed by plugins (read-only), shared with them, theirs.
        Each has an id `<scope>:<name>` (`system:<plugin>/<name>` for plugins')."""
        owner = owner_key(ctx)
        plugins = self.harness.services.plugins
        endpoints = []
        for plugin, item in plugins.endpoints(owner) if plugins else ():
            endpoints.append({**item.to_dict(owner), "group": "plugin", "plugin_id": plugin.id, "editable": False,
                              "plugin_enabled": plugin.enabled,
                              "tools": [s.name for s in item.specs() if s.source.visible_to(owner)]})
        catalog = self.mcp_catalog
        visible = [t for t, _ in catalog.tools(owner)] if catalog else []
        for record in catalog.endpoints(owner) if catalog else ():
            endpoints.append(record.view_for(owner, [t for t in visible if t.endpoint == record.endpoint.key]))
        return {"endpoints": endpoints, "can_add": catalog is not None}

    def mcp_endpoint(self, endpoint_id: str, ctx: AuthContext) -> dict[str, Any] | None:
        """One database endpoint with its tools in full."""
        owner, catalog = owner_key(ctx), self.mcp_catalog
        record = catalog.get(endpoint_id) if catalog else None
        if record is None or not record.endpoint.visible_to(owner):
            return None
        tools = [t for t, _ in catalog.tools(owner) if t.endpoint == record.endpoint.key]
        names = self._model_names(owner)
        return {**record.view_for(owner, tools),
                "tools": [_mcp_tool_view(t, record, owner, names) for t in tools]}

    def mcp_tools(self, ctx: AuthContext) -> list[dict[str, Any]]:
        """Every MCP tool the caller may see, from plugins and from the database. `id` is the tool's
        identity (`<scope>:<name>`); `model_name` what the model calls it in the caller's tasks (a
        name another visible tool has is qualified, agenticwf.agent.runtime.tools.registry)."""
        owner = owner_key(ctx)
        plugins, catalog = self.harness.services.plugins, self.mcp_catalog
        names = self._model_names(owner)
        tools = []
        for plugin, item in plugins.endpoints(owner) if plugins else ():
            for tool in item.expansion.tools:
                spec = tool.spec(item.endpoint)
                if spec.source.visible_to(owner):
                    tools.append({**tool.to_dict(), "id": f"{plugin.scope}:{tool.name}", "endpoint": item.endpoint.id,
                                  "model_name": names.get(spec.source.location, tool.name), "group": "plugin",
                                  "editable": False, "enabled": plugin.enabled,
                                  "visibility": spec.source.effective_visibility, "access": spec.source.access.to_dict()})
        for tool, record in catalog.tools(owner) if catalog else ():
            tools.append(_mcp_tool_view(tool, record, owner, names))
        return tools

    # -- tools and what each caller switched off (docs/admin-page.md, Tools) ---------------------
    def tools(self, ctx: AuthContext) -> dict[str, Any]:
        """Every tool the caller's tasks may use but skills (they have their own tab), each with where
        it comes from and how the caller uses it (`mode`: on, ask, off); and the plugins and MCP
        endpoints that provide them, which can be switched as a whole."""
        from agenticwf.agent.runtime.tools.prefs import mode_of, origin, switchable, timeout_of
        from agenticwf.agent.runtime.tools.registry import findability

        owner, dispatcher = owner_key(ctx), self.harness.services.tool_dispatcher
        modes, labels = dispatcher.prefs.modes_for(owner), self._provider_labels(owner)
        hidden = dispatcher.hidden(owner)   # left out by the Knowledge tab's search tools
        tools, groups = [], {}
        for spec in sorted(dispatcher.registry.catalogue(owner), key=lambda s: s.name):
            found = origin(spec)
            if found.kind == "skill":
                continue
            mine = spec.source.user_defined and spec.source.owner == owner
            for target, gid in (("plugin", found.plugin), ("mcp", found.endpoint)):
                if gid:
                    mode = modes.get(f"{target}:{gid}", "on")
                    group = groups.setdefault(f"{target}:{gid}", {
                        "target": target, "id": gid, "name": labels.get(f"{target}:{gid}") or gid.partition(":")[2],
                        "plugin": found.plugin if target == "mcp" else "", "mine": mine, "tools": 0,
                        "mode": mode, "enabled": mode != "off"})
                    group["tools"] += 1
            mode, effective = modes.get(f"tool:{spec.name}", "on"), mode_of(spec, modes)
            tools.append({"name": spec.name, "summary": spec.summary, "kind": found.kind, "plugin": found.plugin,
                          "endpoint": found.endpoint, "handler": spec.handler.kind.value,
                          "effects": spec.effective_effects.value, "approval": spec.approval.value, "mine": mine,
                          "switchable": switchable(spec), "mode": mode, "enabled": mode != "off",
                          "timeout": timeout_of(dispatcher.prefs, spec), "timeout_default": spec.timeout,
                          "active": effective != "off" and spec.name not in hidden, "asks": effective == "ask",
                          "findability": findability(spec)})
        return {"tools": tools, "groups": sorted(groups.values(), key=lambda g: (g["target"], g["name"])),
                "can_set_timeouts": "admin" in ctx.scopes}

    def set_tool_timeout(self, ctx: AuthContext, name: str, seconds: float | None) -> dict[str, Any]:
        """An admin's timeout for one tool, for everyone (None: the tool's own). KeyError: no such
        tool among the caller's."""
        tool = next((t for t in self.tools(ctx)["tools"] if t["name"] == name), None)
        if tool is None:
            raise KeyError(name)
        self.harness.services.tool_dispatcher.prefs.set_timeout(name, seconds, ctx.subject)
        return next(t for t in self.tools(ctx)["tools"] if t["name"] == name)

    def set_tool_pref(self, ctx: AuthContext, target: str, ref: str, mode: str) -> dict[str, Any]:
        """Set one tool or skill (`ref`: its name), or every tool of a plugin or an MCP endpoint (its
        id), to `on`, `ask` (each call waits for the caller's approval) or `off` for the caller's
        tasks. KeyError: nothing of that name the caller may use; ValueError: a tool that cannot be
        switched."""
        if target == "tool":
            self._check_switchable(ctx, [ref])
        elif not any(g["target"] == target and g["id"] == ref for g in self.tools(ctx)["groups"]):
            raise KeyError(ref)
        self.harness.services.tool_dispatcher.prefs.set_mode(owner_key(ctx), f"{target}:{ref}", mode)
        return {"target": target, "id": ref, "mode": mode, "enabled": mode != "off"}

    def set_tool_prefs(self, ctx: AuthContext, names: list[str], mode: str) -> dict[str, Any]:
        """Set several tools or skills at once (a group on the admin page): all of them, or none
        when one is unknown (KeyError) or always on (ValueError)."""
        self._check_switchable(ctx, names)
        prefs, owner = self.harness.services.tool_dispatcher.prefs, owner_key(ctx)
        for name in dict.fromkeys(names):
            prefs.set_mode(owner, f"tool:{name}", mode)
        return {"names": list(dict.fromkeys(names)), "mode": mode, "enabled": mode != "off"}

    def _check_switchable(self, ctx: AuthContext, names: list[str]) -> None:
        from agenticwf.agent.runtime.tools.prefs import switchable

        specs = {s.name: s for s in self.harness.services.tool_dispatcher.registry.catalogue(owner_key(ctx))}
        for name in names:
            if name not in specs:
                raise KeyError(name)
            if not switchable(specs[name]):
                raise ValueError(f"{name} is a control tool the agent always needs")

    def _provider_labels(self, owner) -> dict[str, str]:
        """`plugin:<id>` and `mcp:<id>` → the name the page shows."""
        plugins, catalog = self.harness.services.plugins, self.mcp_catalog
        labels = {f"plugin:{p.id}": p.name for p in plugins.plugins(owner)} if plugins else {}
        for plugin, item in plugins.endpoints(owner) if plugins else ():
            labels[f"mcp:{item.endpoint.id}"] = f"{plugin.name}/{item.endpoint.name}"
        for record in catalog.endpoints(owner) if catalog else ():
            labels[f"mcp:{record.endpoint.id}"] = record.endpoint.label or record.endpoint.name
        return labels

    def _model_names(self, owner) -> dict[str, str]:
        """MCP tool provenance → the name the owner's tasks call it."""
        registry = self.harness.services.tool_dispatcher.registry
        return {s.source.location: s.name for s in registry.catalogue(owner) if s.source.location.startswith("mcp:")}

    async def close_chat(self, sid: str, ctx: AuthContext) -> None:
        await self.manager.remove_thread(sid, ctx)

    async def shutdown(self) -> None:
        """Stop the workers, keep the work: unanswered intents resume at the next start."""
        await self.manager.workers.stop()
        for relay in self._relays.values():
            relay.cancel()
        await asyncio.gather(*self._relays.values(), return_exceptions=True)
        if self.harness.services.plugins:
            await asyncio.to_thread(self.harness.services.plugins.mcp.close)   # stops stdio MCP servers

    async def recover(self) -> int:
        """Reopen every thread with unanswered intents and queue their work again, and the memory
        consolidations of tasks that completed just before the stop."""
        consolidations = await asyncio.to_thread(self.harness.resume_consolidations)
        if consolidations:
            LOG.info("queued %d pending memory consolidation(s)", consolidations)
        open_intents = getattr(self.harness.store, "open_intents", None)
        if open_intents is None:
            return 0
        intents: list[Intent] = await asyncio.to_thread(open_intents)
        by_thread: dict[str, list[Intent]] = defaultdict(list)
        for intent in intents:
            by_thread[intent.thread_id].append(intent)
        for thread_id, group in by_thread.items():
            first = group[0]
            if first.source == "cron":   # no thread: its runs continue on their own (agenticwf.agent.runtime.cron)
                for intent in group:
                    self.harness.resume_intent(intent)
                continue
            transcript: dict[str, Any] = {}
            if first.source != API_SOURCE:
                session = await asyncio.to_thread(self.store.get_session, first.owner, thread_id)
                if session is None:   # the chat was deleted: nothing to answer
                    for intent in group:
                        intent.status = IntentStatus.CANCELLED
                        self.harness.store.save_intent(intent)
                    continue
                # Messages of intents that start again are added back when they start.
                restart = {i.id for i in group if i.status in (IntentStatus.PENDING, IntentStatus.CLAIMED)}
                transcript = _transcript(session, skip=restart)
            thread = await self._open(StartThreadOptions(auth=first.auth, session_source=first.source,
                                                         reserved_thread_id=thread_id, **transcript))
            thread.session.resume(group)
        if intents:
            LOG.info("resumed %d unanswered intent(s) in %d thread(s)", len(intents), len(by_thread))
        return len(intents)

    # ============================================================================================
    # Chat
    # ============================================================================================
    async def send_chat(self, sid: str, text: str, ctx: AuthContext, wait: bool,
                        task_type: str | None = None, mode: Mode | None = None,
                        model: str | None = None) -> dict[str, Any] | None:
        thread = await self.chat_thread(sid, ctx)
        if thread is None:
            return None
        sent = await thread.send(text, ctx, task_type=task_type, mode=mode, model=model)
        intent_id = self._intent_of(sent)
        # No await between send and this write: the reply cannot reach the transcript first.
        self.store.add_message(owner_key(ctx), sid, "user", text, intent_id)
        self.updates.publish(owner_key(ctx), {"type": "status", "state": "thinking", "session": sid})
        if not wait:
            return {"session": sid, "intent": intent_id, "status": "accepted"}
        return {"session": sid, **await self._outcome(intent_id)}

    # ============================================================================================
    # Tasks (intents without a conversation)
    # ============================================================================================
    async def submit_task(self, prompt: str, ctx: AuthContext, task_type: str | None, mode: Mode | None,
                          model: str | None = None) -> str:
        thread = await self._open(StartThreadOptions(auth=ctx, session_source=API_SOURCE,
                                                     task_type=task_type or "default", mode=mode, model=model))
        return await thread.send(prompt, ctx)

    async def thread_of(self, intent_id: str, ctx: AuthContext) -> RuntimeThread | None:
        thread_id = await asyncio.to_thread(self.store.intent_thread, owner_key(ctx), intent_id)
        return await self.manager.get_thread(thread_id, ctx) if thread_id else None

    async def answer(self, intent_id: str, text: str, ctx: AuthContext) -> dict[str, Any]:
        thread = await self.thread_of(intent_id, ctx)
        if thread is None or not any(t.state is RuntimeState.WAITING_USER for t in self._tasks(intent_id)):
            raise Conflict("the task is not waiting for input")
        await thread.send(text, ctx)
        return await self._outcome(intent_id, timeout=0)

    async def approve(self, thread: RuntimeThread, call_id: str, ctx: AuthContext, approved: bool,
                      remember: bool, reason: str) -> dict[str, Any]:
        try:
            task_id = await thread.approve(call_id, ctx, approved=approved, remember=remember, reason=reason)
        except RuntimeError as exc:
            raise Conflict(str(exc)) from exc
        return {"task": task_id, "intent": self._intent_of(task_id), "approved": approved}

    def execution(self, intent_id: str) -> list[dict[str, Any]]:
        """The execution record of an intent: its tasks, their plans, turns and tool calls."""
        return [_task_view(t) for t in self._tasks(intent_id)]

    def recalled(self, intent_id: str, ctx: AuthContext) -> list[dict[str, Any]]:
        """The long-term memories recalled at the start of an intent's tasks ({"id", "text"}, each once);
        none when the intent is not the caller's."""
        intent = self.harness.store.load_intent(intent_id)
        if intent is None or tuple(intent.owner) != tuple(owner_key(ctx)):
            return []
        found: dict[str, dict] = {}
        for task in self._tasks(intent_id):
            for memory in task.recalled:
                found.setdefault(memory.get("id") or memory.get("text", ""), memory)
        return list(found.values())

    def trace(self, intent_id: str, ctx: AuthContext, trace_url: str = "") -> dict[str, Any] | None:
        """What the agent did for one intent of the caller's, step by step (agenticwf.agent.api.trace);
        None when it is not theirs."""
        from agenticwf.agent.api.trace import intent_trace

        store, owner = self.harness.store, owner_key(ctx)
        intent = store.load_intent(intent_id)
        if intent is None or tuple(intent.owner) != tuple(owner):
            return None
        tasks = self._tasks(intent_id)
        llm = self.harness.services.llm
        usage = getattr(llm, "usage", None)
        rows = usage.calls(owner, intent_id, [t.id for t in tasks]) if hasattr(usage, "calls") else []
        cost_unit = getattr(getattr(llm, "settings", None), "cost_unit", "units")
        return intent_trace(intent, tasks, store.load_reply(intent_id), rows, trace_url=trace_url, cost_unit=cost_unit)

    # ============================================================================================
    # Outcomes
    # ============================================================================================
    def _tasks(self, intent_id: str) -> list[Task]:
        return self.harness.store.tasks_for_intent(intent_id)

    def _intent_of(self, intent_or_task: str) -> str:
        if self.harness.store.load_intent(intent_or_task) is not None:
            return intent_or_task
        loaded = self.harness.store.load(intent_or_task)
        return loaded[0].intent_id if loaded else intent_or_task

    def _settled(self, intent_id: str) -> dict[str, Any] | None:
        reply = self.harness.store.load_reply(intent_id)
        if reply is not None:
            return {"intent": intent_id, "status": reply.status, "reply": reply.answer}
        waiting = [t for t in self._tasks(intent_id) if t.state in WAITING_STATES]
        if waiting:
            return {"intent": intent_id, "status": "waiting", "waiting": [_pending(t) for t in waiting]}
        return None

    async def _outcome(self, intent_id: str, timeout: float | None = None) -> dict[str, Any]:
        """The reply, or what the intent waits for, or "running" after the timeout."""
        settled = self._settled(intent_id)
        if settled is not None or timeout == 0:
            return settled or {"intent": intent_id, "status": "running"}
        future = asyncio.get_running_loop().create_future()
        self._waiters[intent_id].append(future)
        try:
            return await asyncio.wait_for(future, self.reply_timeout if timeout is None else timeout)
        except asyncio.TimeoutError:
            return {"intent": intent_id, "status": "running"}
        finally:
            waiters = self._waiters.get(intent_id, [])
            if future in waiters:
                waiters.remove(future)
            if not waiters:
                self._waiters.pop(intent_id, None)

    def _wake(self, intent_id: str | None) -> None:
        if not intent_id or intent_id not in self._waiters:
            return
        settled = self._settled(intent_id)
        if settled is None:
            return
        for future in self._waiters.pop(intent_id):
            if not future.done():
                future.set_result(settled)

    # ============================================================================================
    # The relay: thread events → transcript, live updates, waiting requests
    # ============================================================================================
    async def _relay(self, thread: RuntimeThread) -> None:
        owner = thread.owner   # (subject, organization): updates never cross organizations
        api = thread.session.configuration.session_source == API_SOURCE
        while True:
            event = await thread.next_event()
            try:
                await self._forward(thread, owner, api, event)
            except Exception:   # the relay must outlive any one event
                LOG.exception("relaying %s of thread %s failed", event.type, thread.thread_id)
            if event.type == "shutdown_complete":
                self._relays.pop(thread.thread_id, None)
                return

    async def _forward(self, thread: RuntimeThread, owner: tuple[str, str | None], api: bool, event: ThreadEvent) -> None:
        sid = thread.thread_id
        kind = "task_state" if event.type == "task" else event.type   # "task" is the API task's own event
        self.updates.publish(owner, {"type": kind, "session": sid, **event.data})

        if event.type == "reply":
            intent_id, status, answer = event.data["intent"], event.data["status"], event.data.get("answer") or ""
            if api:
                self.updates.publish(owner, {"type": "task", "id": intent_id, "status": status})
                intent = self.harness.store.load_intent(intent_id)
                closing = asyncio.create_task(self.manager.remove_thread(sid, intent.auth))   # one intent per API thread
                self._background.add(closing)
                closing.add_done_callback(self._background.discard)
            else:
                if answer:
                    await asyncio.to_thread(self.store.add_message, owner, sid, "assistant", answer, intent_id)
                    self.updates.publish(owner, {"type": "message", "session": sid, "role": "assistant",
                                                   "text": answer, "time": int(time.time()), "intent": intent_id})
                self.updates.publish(owner, {"type": "status", "state": "done", "session": sid})
            self._wake(intent_id)
        elif event.type == "conversation_compacted" and not api:   # kept with the chat
            await asyncio.to_thread(self.store.set_summary, owner, sid, event.data["summary"], event.data["keys"])
        elif event.type == "task_waiting":
            loaded = self.harness.store.load(event.data["task"])
            self._wake(loaded[0].intent_id if loaded else None)
        elif event.type == "tasks_created" and api:
            self.updates.publish(owner, {"type": "task", "id": event.data["intent"], "status": "running"})


def _transcript(session: dict[str, Any], skip: set[str] = frozenset()) -> dict[str, Any]:
    """A stored chat as StartThreadOptions: its messages with their keys (the intent each belongs
    to, or the message itself), and its summary with the keys it covers."""
    history = [TranscriptMessage(m["role"], m["text"], keys=(m["intent"] if m["intent"] else f"message:{m['id']}",))
               for m in session["messages"] if m["intent"] not in skip]
    return {"initial_history": history, "summary": session.get("summary") or "",
            "summary_keys": frozenset(session.get("summary_keys") or ())}


def _pending(task: Task) -> dict[str, Any]:
    approvals = [{"call": c.id, "name": c.name, "arguments": c.arguments, "reason": c.reason}
                 for c in task.open_calls(ToolCallStatus.AWAITING_APPROVAL)]
    questions = [{"call": c.id, "question": c.arguments.get("question", "")}
                 for c in task.calls if c.name == "ask_user" and c.result is None and c.status is ToolCallStatus.REQUESTED]
    return {"task": task.id, "state": task.state.value, "approvals": approvals, "questions": questions}


def _task_view(task: Task) -> dict[str, Any]:
    return {
        "id": task.id, "task_type": task.task_type.requested, "state": task.state.value, "status": task.state.public_status,
        "depends_on": task.depends_on, "result": task.result, "error": task.error,
        "plan": [{"id": i.id, "goal": i.goal, "phase": i.phase} for i in task.plan],
        "turns": [{"id": t.id, "goal": t.item.goal, "state": t.state.value, "attempts": t.attempts,
                   "steps": [s.number for s in t.steps], "summary": t.result.summary if t.result else None,
                   "verification": t.verification} for t in task.turns],
        "tool_calls": [{"id": c.id, "name": c.name, "step": c.step, "status": c.status.value, "decision": c.decision}
                       for c in task.calls],
    }


def _mcp_tool_view(tool, record, owner, names: dict[str, str]) -> dict[str, Any]:
    from agenticwf.agent.runtime.mcp.catalog import tool_ref

    endpoint, mine = record.endpoint, tool.owner == owner
    return {**tool.to_dict(), "id": tool_ref(tool, record), "endpoint": endpoint.id, "endpoint_name": endpoint.name,
            "model_name": names.get(tool.spec(endpoint).source.location, tool.name),
            "group": "mine" if mine else "shared", "editable": mine,
            "endpoint_enabled": endpoint.enabled, "visibility": tool.visibility or endpoint.effective_visibility,
            "inherits_visibility": not tool.visibility, "description": tool.description,
            "input_schema": tool.input_schema}


def _setting_rows(section: str, keys: list[str], base: dict[str, Any], everyone: dict[str, Any],
                  mine: dict[str, Any], in_effect: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per setting, as the admin page shows it: the file's value, everyone's, yours, in effect."""
    rows = []
    for key in keys:
        spec = key_spec(section, key)
        if spec is not None:
            rows.append({"key": key, **spec.view(), "file": base.get(key), "all": everyone.get(key),
                         "mine": mine.get(key) if spec.personal else None, "effective": in_effect.get(key)})
    return rows
