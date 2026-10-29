"""AgenticWF agent API: the backend behind Caddy + authsvc on 127.0.0.1:8000.

    browser / CLI ─► Caddy ─► forward_auth authsvc /verify ─► this service ─► agent runtime
                                (signs X-Auth-Context)          (verifies it)   (src/agenticwf/agent/runtime)

Chat sessions are runtime threads; background tasks are intents of their own (agenticwf.agent.api.agent). The
runtime's records live in the same database as the sessions (agenticwf.agent.core.orm.models.agent).

Run:  AUTH_CONTEXT_SECRET_FILE=... uvicorn --factory agenticwf.agent.api.app:create_app --port 8000
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as dt
from contextlib import asynccontextmanager
from importlib.resources import files
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from agenticwf.auth.client import AuthContextMiddleware, current_auth_context
from agenticwf.auth.context import AuthContext
from agenticwf.agent.core.common.logging import get_logger, log_context
from agenticwf.agent.runtime import telemetry
from agenticwf.agent.runtime.bootstrap import EngineOptions, RuntimeConfig, build_thread_manager
from agenticwf.agent.runtime.context.db_memory import DatabaseMemoryStore
from agenticwf.agent.runtime.cron.catalog import role as cron_role
from agenticwf.agent.runtime.harness.relay import EventRelay, deliverer
from agenticwf.agent.runtime.context.task_types import TYPE_REF_RE, TaskTypeError
from agenticwf.agent.runtime.harness.db_store import DatabaseStore
from agenticwf.agent.runtime.llm.catalog import ModelEntry
from agenticwf.agent.runtime.llm.db_catalog import DatabaseModelStore
from agenticwf.agent.runtime.llm.usage import DatabaseUsage
from agenticwf.agent.runtime.mcp.db_store import DatabaseMcpStore
from agenticwf.agent.runtime.mcp.endpoint import McpEndpoint
from agenticwf.agent.runtime.secrets import SecretBox
from agenticwf.agent.runtime.skills.db_store import DatabaseSkillStore
from agenticwf.agent.runtime.skills.skill import FILES_MAX, Skill, SkillError, check_file_content, check_file_path
from agenticwf.agent.runtime.types import Mode, split_ref

from agenticwf.agent.api import crons as cron_routes
from agenticwf.agent.api.agent import AgentService, Conflict
from agenticwf.agent.api.auth import is_approved, owner_key, require, require_auth
from agenticwf.agent.api.events import Updates
from agenticwf.agent.api.settings import Settings
from agenticwf.agent.api.store import Store

LOG = get_logger(__name__)

# Route scopes. Keep in step with the agent rules in AgenticWF-auth's policy.yaml
# (see deploy/policy.agent.yaml).
SCOPE_AGENT = "agent:use"
SCOPE_ADMIN = "admin"


class NewSession(BaseModel):
    title: str = Field(default="New chat", max_length=200)


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=32_000)


class ChatMessage(Message):
    task_type: str | None = Field(default=None, max_length=320)  # a name or <scope>:<name>; None: the session's (default)
    mode: Mode | None = None
    model: str | None = Field(default=None, max_length=64)       # a model id from GET /api/v1/models


class TaskTypeFiles(BaseModel):
    """A user task type's instruction files; empty or missing = inherited from `default`."""

    agents_md: str | None = Field(default=None, max_length=32 * 1024)
    workflow_md: str | None = Field(default=None, max_length=32 * 1024)


class SkillIn(BaseModel):
    """A skill a user adds (docs/skills.md): what SKILL.md holds."""

    name: str = Field(min_length=1, max_length=64)    # lowercase letters, digits, hyphens
    description: str = Field(min_length=1, max_length=1024)
    instructions: str = Field(default="", max_length=64 * 1024)
    tools: list[str] = Field(default_factory=list)    # while it runs, only these (globs); empty: no narrowing
    tags: list[str] = Field(default_factory=list)
    visibility: Literal["user", "organization", "all"] = "user"   # all: managers only
    enabled: bool = True
    files: dict[str, str] = Field(default_factory=dict)   # path -> text: references, templates (in the database)


class SkillFile(BaseModel):
    content: str


class SkillPatch(BaseModel):
    description: str | None = Field(default=None, min_length=1, max_length=1024)
    instructions: str | None = Field(default=None, max_length=64 * 1024)
    tools: list[str] | None = None
    tags: list[str] | None = None
    visibility: Literal["user", "organization", "all"] | None = None
    enabled: bool | None = None


class FeedbackIn(BaseModel):
    intent: str = Field(min_length=1, max_length=32)       # the reply's intent (chat messages carry it)
    rating: Literal["up", "down"]
    reason: str = Field(default="", max_length=2000)


class ModelIn(BaseModel):
    """A model a user adds (docs/agent-models.md). `api_key` is write-only: stored encrypted."""

    id: str = Field(max_length=64)
    provider: Literal["anthropic", "openai", "vllm", "ollama", "echo"]
    model: str = Field(min_length=1, max_length=255)
    label: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=2000)
    base_url: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)
    api_key_env: str | None = Field(default=None, max_length=255)   # refused: models users add never read the environment
    options: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    visibility: Literal["user", "organization", "all"] = "user"   # all: managers only


class DefaultModel(BaseModel):
    model: str | None = Field(default=None, max_length=64)   # a model id you may use; None: [llm] default


class ModelPatch(BaseModel):
    provider: Literal["anthropic", "openai", "vllm", "ollama", "echo"] | None = None
    model: str | None = Field(default=None, min_length=1, max_length=255)
    label: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    base_url: str | None = Field(default=None, max_length=1024)
    api_key: str | None = Field(default=None, max_length=4096)   # a new key; omitted: keep the stored one
    clear_api_key: bool = False
    api_key_env: str | None = Field(default=None, max_length=255)
    options: dict[str, Any] | None = None
    enabled: bool | None = None
    visibility: Literal["user", "organization", "all"] | None = None


class McpEndpointIn(BaseModel):
    """An MCP endpoint a user adds (docs/mcp.md). `headers` are write-only: stored encrypted."""

    name: str = Field(min_length=1, max_length=64)    # the tool prefix: <name>__<tool>
    url: str = Field(min_length=1, max_length=2048)
    label: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=2000)
    headers: dict[str, str] = Field(default_factory=dict)
    tools: list[str] = Field(default_factory=list)    # only expose these; empty: all
    effects: Literal["none", "local", "external"] | None = None
    approval: Literal["ask", "deny"] | None = None
    timeout: float = Field(default=60, gt=0, le=3600)
    visibility: Literal["user", "organization", "all"] = "user"   # all: managers only
    enabled: bool = True


class McpEndpointPatch(BaseModel):
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    label: str | None = Field(default=None, max_length=200)
    description: str | None = Field(default=None, max_length=2000)
    headers: dict[str, str] | None = None             # new headers; omitted: keep the stored ones
    clear_headers: bool = False
    tools: list[str] | None = None
    effects: Literal["none", "local", "external"] | None = None
    approval: Literal["ask", "deny"] | None = None
    timeout: float | None = Field(default=None, gt=0, le=3600)
    visibility: Literal["user", "organization", "all"] | None = None
    enabled: bool | None = None


class McpToolPatch(BaseModel):
    enabled: bool | None = None
    visibility: Literal["", "user", "organization", "all"] | None = None   # "": as its endpoint
    approval: Literal["ask", "deny"] | None = None


class KnowledgeServiceIn(BaseModel):
    """A knowledge service a user connects (docs/admin-page.md, Knowledge). `token` is write-only."""

    name: str = Field(min_length=1, max_length=64)    # its KBs are <ref>@<name>
    url: str = Field(min_length=1, max_length=2048)
    base_path: str = Field(default="/kb", min_length=1, max_length=255)
    label: str = Field(default="", max_length=200)
    token: str | None = Field(default=None, max_length=8192)
    enabled: bool = True


class KnowledgeServicePatch(BaseModel):
    url: str | None = Field(default=None, min_length=1, max_length=2048)
    base_path: str | None = Field(default=None, min_length=1, max_length=255)
    label: str | None = Field(default=None, max_length=200)
    token: str | None = Field(default=None, max_length=8192)   # a new token; omitted: keep the stored one
    clear_token: bool = False
    enabled: bool | None = None


class ToolPref(BaseModel):
    """Switch one tool (`id`: its name), or every tool of a plugin or an MCP endpoint (`id`: theirs),
    on or off for the caller's tasks."""

    target: Literal["tool", "plugin", "mcp"]
    id: str = Field(min_length=1, max_length=390)
    enabled: bool


class KnowledgePref(BaseModel):
    """Switch a source ("system" or a service id) or one of its KBs (`kb`: its ref) on or off for
    the caller's tasks."""

    source: str = Field(min_length=1, max_length=320)
    kb: str | None = Field(default=None, min_length=1, max_length=512)   # None: the whole source
    enabled: bool


class KnowledgeOptions(BaseModel):
    """The caller's knowledge search options ([knowledge] tools, research_preset). A field left out
    stays as it is; null goes back to agent.cfg's."""

    tools: Literal["auto", "kb_search", "kb_research"] | None = None
    research_preset: Literal["model", "kb", "auto", "quick", "deep", "wiki", "smart"] | None = None


class Rename(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class NewTask(BaseModel):
    prompt: str = Field(min_length=1, max_length=32_000)
    task_type: str | None = Field(default=None, max_length=320)
    mode: Mode | None = None
    model: str | None = Field(default=None, max_length=64)


class Decision(BaseModel):
    decision: Literal["approve", "reject"]
    reason: str = Field(default="", max_length=2000)
    remember: bool = False   # approve this tool for the rest of the task


def _patch(body: BaseModel, nullable: set[str], exclude: set[str]) -> dict[str, Any]:
    """The fields a PATCH sets. An explicit null clears a field only where null means something
    (`nullable`); anywhere else it is refused, rather than written into a NOT NULL column."""
    changes = body.model_dump(exclude_unset=True, exclude=exclude)
    refused = sorted(k for k, v in changes.items() if v is None and k not in nullable)
    if refused:
        raise HTTPException(status_code=422, detail=f"cannot be null: {', '.join(refused)}")
    return changes


class RequestLogContext:
    """ASGI middleware: tag every log record of a request with the caller's identity.

    Must sit inside AuthContextMiddleware, which sets the context it reads.
    """

    def __init__(self, app: Any):
        self.app = app

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        ctx = current_auth_context() if scope["type"] == "http" else None
        if ctx is None:
            await self.app(scope, receive, send)
            return
        with log_context(sub=ctx.subject, org=ctx.organization_id or "-", auth=ctx.auth_method):
            await self.app(scope, receive, send)


def database_engine(config: RuntimeConfig, engine: EngineOptions | None = None) -> EngineOptions:
    """What the API process and worker processes (agenticwf.agent.api.worker) run with: the database for the
    runtime's records (shared between processes with `[harness] shared`), long-term memory, models,
    MCP endpoints, skills and crons, unless `engine` names others."""
    engine = engine or EngineOptions()
    if engine.task_store is None:
        engine = dataclasses.replace(engine, task_store=DatabaseStore(shared=config.shared))
    if engine.memory is None:   # long-term memory per owner, and the memory_search tool
        from agenticwf.agent.runtime.knowledge import KBMemoryStore, KnowledgeClient, KnowledgeSettings

        knowledge = KnowledgeSettings.from_core_config()
        if knowledge.enabled and knowledge.memory:     # on the KB's temporal memory ([knowledge] memory)
            engine = dataclasses.replace(engine, memory=KBMemoryStore(KnowledgeClient(knowledge),
                                                                      knowledge.memory_group))
        else:
            engine = dataclasses.replace(engine, memory=DatabaseMemoryStore())
    if engine.model_store is None and engine.llm is None:
        # Models managers add at run time; their API keys are encrypted with a key outside the database.
        engine = dataclasses.replace(engine, model_store=DatabaseModelStore(SecretBox.from_environment()))
    if engine.mcp_store is None:   # MCP endpoints users add; header values encrypted the same way
        engine = dataclasses.replace(engine, mcp_store=DatabaseMcpStore(SecretBox.from_environment()))
    if engine.skill_store is None:   # skills users add
        engine = dataclasses.replace(engine, skill_store=DatabaseSkillStore())
    if engine.tool_prefs is None:    # tools people switch off for their tasks (docs/admin-page.md, Tools)
        from agenticwf.agent.runtime.tools.db_prefs import DatabaseToolPrefs

        engine = dataclasses.replace(engine, tool_prefs=DatabaseToolPrefs())
    if engine.usage_store is None:   # model usage and feedback (docs/model-routing.md)
        engine = dataclasses.replace(engine, usage_store=DatabaseUsage())
    if engine.knowledge is None:     # the knowledge service ([knowledge] url) and those users connect
        from agenticwf.agent.runtime.knowledge import HttpKnowledgeCatalog, KnowledgeClient, KnowledgeSettings
        from agenticwf.agent.runtime.knowledge.db_store import DatabaseKnowledgeStore

        knowledge = KnowledgeSettings.from_core_config()
        if knowledge.enabled or knowledge.allowed_hosts:
            engine = dataclasses.replace(engine, knowledge=HttpKnowledgeCatalog(
                KnowledgeClient(knowledge) if knowledge.enabled else None,
                DatabaseKnowledgeStore(SecretBox.from_environment()), settings=knowledge))
    return dataclasses.replace(engine, crons=True)   # crons need the database


def create_app(settings: Settings | None = None, runtime_config: RuntimeConfig | None = None,
               engine: EngineOptions | None = None) -> FastAPI:
    """`runtime_config` defaults to the core configuration (agent.cfg); `engine` passes objects
    (a model client, tool handlers). The runtime's store and long-term memory are the database
    unless `engine` names others."""
    settings = settings or Settings.from_env()
    config = runtime_config or RuntimeConfig.from_core_config()
    telemetry.setup("agenticwf-agent")   # before any httpx client exists; off unless OTEL_EXPORTER_OTLP_ENDPOINT
    engine = database_engine(config, engine)
    manager = build_thread_manager(config, engine)
    store = Store()
    updates = Updates()
    crons = manager.crons
    if crons is not None:   # a cron's live events reach whoever may see the cron
        crons.notify = lambda cron, event: updates.broadcast(event, lambda ctx: cron_role(cron, ctx) is not None)
    agent = AgentService(manager, store, updates, reply_timeout=settings.reply_timeout)
    page = files("agenticwf.agent.api").joinpath("web/index.html").read_text()
    admin_html = files("agenticwf.agent.api").joinpath("web/admin.html").read_text()
    crons_html = files("agenticwf.agent.api").joinpath("web/crons.html").read_text()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # The schema is created by `python -m agenticwf.agent.api.db init` before the server starts.
        manager.workers.start()   # the worker pool, on this event loop: jobs queued from any thread wake it
        try:
            await agent.recover()   # unanswered intents continue where they stopped
        except Exception:   # serving matters more than resuming: never fail startup over old work
            LOG.exception("could not resume the unanswered intents; they stay as they are")
        if crons is not None:
            await crons.start()
        relaying = None
        if getattr(manager.harness.store, "shared", False):   # worker processes report through the database
            relaying = asyncio.get_running_loop().create_task(EventRelay().run(deliverer(manager.harness, crons)))
        yield
        if relaying is not None:
            relaying.cancel()
        if crons is not None:
            await crons.stop()
        await agent.shutdown()

    app = FastAPI(title="AgenticWF agent API", docs_url=None, redoc_url=None, openapi_url=None,
                  lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.agent = agent
    # Added last = outermost: verify the signed context before anything else runs.
    app.add_middleware(RequestLogContext)
    app.add_middleware(AuthContextMiddleware, secret=settings.context_secret, bind_request=settings.bind_request)
    telemetry.instrument_app(app)   # a span per request

    @app.get("/healthz")
    def healthz() -> Response:
        """Up, and its database answers (the container's healthcheck asks this)."""
        from sqlalchemy import text

        from agenticwf.agent.core.orm.base.session import get_engine

        try:
            with get_engine().connect() as connection:
                connection.execute(text("SELECT 1"))
        except Exception:
            LOG.warning("health check: the database does not answer", exc_info=True)
            return JSONResponse({"status": "unavailable", "database": "down"}, status_code=503)
        return JSONResponse({"status": "ok"})

    @app.get("/", response_class=HTMLResponse)
    def index(ctx: AuthContext = Depends(require())) -> str:
        return page

    @app.get("/admin", response_class=HTMLResponse)
    def admin_page(ctx: AuthContext = Depends(require())) -> str:
        """Task types, models and plugins: browse them in groups, and manage your own (docs/admin-page.md)."""
        return admin_html

    @app.get("/api/v1/me")
    def me(request: Request, ctx: AuthContext = Depends(require_auth)) -> dict:
        return {**ctx.to_dict(), "approved": is_approved(ctx, request)}

    @app.get("/api/v1/events")
    async def events(request: Request, ctx: AuthContext = Depends(require())) -> StreamingResponse:
        return StreamingResponse(updates.stream(request, ctx, settings.max_stream_seconds),
                                 media_type="text/event-stream",
                                 # no-transform: proxies (Caddy's `encode`) must not compress or buffer the stream
                                 headers={"Cache-Control": "no-store, no-transform", "X-Accel-Buffering": "no"})

    # --- Chat sessions (people only: browser chat UI and the CLI) -----------------------
    chat = require(SCOPE_AGENT, allow_machine=False)

    @app.post("/api/v1/sessions", status_code=201)
    def create_session(body: NewSession, ctx: AuthContext = Depends(chat)) -> dict:
        session = store.create_session(owner_key(ctx), body.title)
        LOG.info("session %s created", session["id"])
        return session

    @app.get("/api/v1/sessions")
    def list_sessions(ctx: AuthContext = Depends(chat)) -> dict:
        return {"sessions": store.list_sessions(owner_key(ctx))}

    @app.get("/api/v1/sessions/{sid}")
    def get_session(sid: str, ctx: AuthContext = Depends(chat)) -> dict:
        session = store.get_session(owner_key(ctx), sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    @app.patch("/api/v1/sessions/{sid}")
    def rename_session(sid: str, body: Rename, ctx: AuthContext = Depends(chat)) -> dict:
        session = store.rename_session(owner_key(ctx), sid, body.title)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        return session

    @app.get("/api/v1/sessions/{sid}/trace")
    def session_trace(sid: str, intent: str | None = None, limit: int = Query(20, ge=1, le=200),
                      ctx: AuthContext = Depends(chat)) -> dict:
        """What the agent did for this chat's messages (the last `limit`, or one `intent`): tasks, plans,
        turns, steps with their model calls, tool calls with arguments and results; docs/api.md, Trace."""
        session = store.get_session(owner_key(ctx), sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        intents = list(dict.fromkeys(m["intent"] for m in session["messages"] if m.get("intent")))
        if intent is not None:
            if intent not in intents:
                raise HTTPException(status_code=404, detail="no such message in this session")
            intents = [intent]
        traces = [t for t in (agent.trace(i, ctx, settings.trace_url) for i in intents[-limit:]) if t is not None]
        return {"session": sid, "intents": traces, "trace_url": bool(settings.trace_url)}

    @app.get("/api/v1/sessions/{sid}/memories")
    def session_memories(sid: str, ctx: AuthContext = Depends(chat)) -> dict:
        """The long-term memories recalled for each of this chat's messages, by intent (those that
        recalled any); the chat page shows them under each answer."""
        session = store.get_session(owner_key(ctx), sid)
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        intents = dict.fromkeys(m["intent"] for m in session["messages"] if m.get("intent"))
        found = {i: agent.recalled(i, ctx) for i in intents}
        return {"session": sid, "intents": {i: m for i, m in found.items() if m}}

    @app.get("/api/v1/sessions/{sid}/status")
    async def session_status(sid: str, ctx: AuthContext = Depends(chat)) -> dict:
        if await asyncio.to_thread(store.get_session, owner_key(ctx), sid) is None:
            raise HTTPException(status_code=404, detail="session not found")
        return await agent.chat_status(sid, ctx)

    @app.delete("/api/v1/sessions/{sid}", status_code=204)
    async def delete_session(sid: str, ctx: AuthContext = Depends(chat)) -> Response:
        if await asyncio.to_thread(store.get_session, owner_key(ctx), sid) is None:
            raise HTTPException(status_code=404, detail="session not found")
        await agent.close_chat(sid, ctx)   # cancels what it was doing
        await asyncio.to_thread(store.delete_session, owner_key(ctx), sid)
        return Response(status_code=204)

    @app.post("/api/v1/sessions/{sid}/messages")
    async def post_message(sid: str, msg: ChatMessage, wait: bool = True, ctx: AuthContext = Depends(chat)) -> Any:
        """With wait (the default), answer with the reply, or with what the agent waits for
        (approval, an answer); otherwise 202 at once. Everything also arrives as live updates."""
        if msg.task_type is not None and not TYPE_REF_RE.match(msg.task_type):
            raise HTTPException(status_code=422, detail="invalid task type name")
        if msg.model is not None and not agent.model_available(msg.model, ctx):
            raise HTTPException(status_code=422, detail="unknown or disabled model")
        try:
            outcome = await agent.send_chat(sid, msg.text, ctx, wait, task_type=msg.task_type, mode=msg.mode,
                                            model=msg.model)
        except RuntimeError as exc:   # e.g. the thread is shutting down
            raise HTTPException(status_code=409, detail=str(exc))
        if outcome is None:
            raise HTTPException(status_code=404, detail="session not found")
        return outcome if wait else JSONResponse(outcome, status_code=202)

    @app.post("/api/v1/sessions/{sid}/approvals/{call_id}")
    async def approve_in_session(sid: str, call_id: str, body: Decision, ctx: AuthContext = Depends(chat)) -> dict:
        thread = await agent.chat_thread(sid, ctx)
        if thread is None:
            raise HTTPException(status_code=404, detail="session not found")
        return await _decide(thread, call_id, body, ctx)

    @app.post("/api/v1/sessions/{sid}/interrupt")
    async def interrupt_session(sid: str, ctx: AuthContext = Depends(chat)) -> dict:
        thread = await agent.chat_thread(sid, ctx)
        if thread is None:
            raise HTTPException(status_code=404, detail="session not found")
        return {"session": sid, "intent": await thread.interrupt(ctx) or None}

    # --- Background tasks (people and machine-to-machine clients) ------------------------
    tasks_dep = require(SCOPE_AGENT)

    # --- Task types and models: the admin page's API (docs/admin-page.md) ---------------
    people = require(SCOPE_AGENT, allow_machine=False)
    cron_routes.register(app, crons, tasks_dep, people, crons_html, require())

    @app.get("/api/v1/task-types")
    def list_task_types(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """System, plugin and the caller's own task types."""
        return {"task_types": agent.task_types(ctx),
                "can_add": agent.task_type_resolver.user_sandbox is not None}

    @app.get("/api/v1/task-types/{name}")
    def get_task_type(name: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        found = agent.task_type(name, ctx) if TYPE_REF_RE.match(name) else None
        if found is None:
            raise HTTPException(status_code=404, detail="task type not found")
        return found

    @app.put("/api/v1/task-types/{name}")
    def put_task_type(name: str, body: TaskTypeFiles, ctx: AuthContext = Depends(people)) -> dict:
        """Create or replace one of the caller's own task types. A file left empty is inherited
        from `default`. System and plugin types cannot be changed here."""
        resolver = agent.task_type_resolver
        try:
            resolver.write_user_type(owner_key(ctx), name, {"AGENTS.md": body.agents_md, "WORKFLOW.md": body.workflow_md})
        except TaskTypeError as exc:
            message = str(exc)
            status = 403 if "your own" in message else 409 if "reserved" in message or "not enabled" in message else 422
            raise HTTPException(status_code=status, detail=str(exc))
        LOG.info("task type %s saved by %s", name, ctx.subject)
        return agent.task_type(name, ctx)

    @app.delete("/api/v1/task-types/{name}", status_code=204)
    def delete_task_type(name: str, ctx: AuthContext = Depends(people)) -> Response:
        if not agent.task_type_resolver.delete_user_type(owner_key(ctx), name):
            raise HTTPException(status_code=404, detail="no such task type of yours")
        LOG.info("task type %s deleted by %s", name, ctx.subject)
        return Response(status_code=204)

    @app.get("/api/v1/plugins")
    def list_plugins(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """System plugins and the caller's own: their task types, tools and MCP servers (docs/plugins.md)."""
        return {"plugins": agent.plugins(ctx), "can_reload": SCOPE_ADMIN in ctx.scopes}

    @app.post("/api/v1/plugins/reload")
    async def reload_plugins(ctx: AuthContext = Depends(require(SCOPE_ADMIN, allow_machine=False))) -> dict:
        """Managers: read the system plugins again and reconnect their MCP servers."""
        errors = await asyncio.to_thread(agent.reload_plugins)
        LOG.info("plugins reloaded by %s (%d error(s))", ctx.subject, len(errors))
        return {"errors": errors, "plugins": agent.plugins(ctx)}

    # --- MCP endpoints and tools (docs/mcp.md) -------------------------------------------------
    def _mcp():
        if agent.mcp_catalog is None:
            raise HTTPException(status_code=409, detail="this runtime has no store for MCP endpoints")
        return agent.mcp_catalog

    def _mcp_call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError:
            raise HTTPException(status_code=404, detail="not found")
        except PermissionError as exc:   # someone else's, a system plugin's name, or no secrets key
            raise HTTPException(status_code=403 if "belongs to" in str(exc) else 409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    def _ref(ref: str, ctx: AuthContext) -> str:
        """An id (`<scope>:<name>`); a bare name is one of the caller's own."""
        scope, name = split_ref(ref, owner_key(ctx))
        return f"{scope}:{name}"

    @app.get("/api/v1/mcp/endpoints")
    def list_mcp_endpoints(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Endpoints installed by plugins, shared with the caller, and the caller's own."""
        return agent.mcp_endpoints(ctx)

    @app.get("/api/v1/mcp/endpoints/{ref}")
    def get_mcp_endpoint(ref: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        found = agent.mcp_endpoint(_ref(ref, ctx), ctx)
        if found is None:
            raise HTTPException(status_code=404, detail="MCP endpoint not found")
        return found

    @app.post("/api/v1/mcp/endpoints", status_code=201)
    def add_mcp_endpoint(body: McpEndpointIn, ctx: AuthContext = Depends(people)) -> dict:
        """Add an endpoint of yours and expand it: its id is `<your scope>:<name>`, its tools <name>__<tool>."""
        catalog, owner = _mcp(), owner_key(ctx)
        endpoint = McpEndpoint(name=body.name, transport="http", url=body.url, headers=dict(body.headers),
                               tools=tuple(body.tools), effects=body.effects, approval=body.approval,
                               timeout=body.timeout, source="database", owner=owner,
                               visibility=_visibility(body.visibility, ctx), label=body.label,
                               description=body.description, enabled=body.enabled)
        if catalog.get(endpoint.id) is not None:
            raise HTTPException(status_code=409, detail=f"you already have an MCP endpoint named {body.name!r}")
        record = _mcp_call(catalog.put, endpoint, by=owner)
        LOG.info("MCP endpoint %s added (%s, %s) by %s: %s", endpoint.id, endpoint.host, endpoint.visibility,
                 ctx.subject, record.status)
        return agent.mcp_endpoint(endpoint.id, ctx)

    @app.patch("/api/v1/mcp/endpoints/{ref}")
    def change_mcp_endpoint(ref: str, body: McpEndpointPatch, ctx: AuthContext = Depends(people)) -> dict:
        """Change an endpoint of yours; it is expanded again."""
        catalog, owner, endpoint_id = _mcp(), owner_key(ctx), _ref(ref, ctx)
        current = catalog.get(endpoint_id)
        if current is None or not current.endpoint.visible_to(owner):
            raise HTTPException(status_code=404, detail="MCP endpoint not found")
        changes = _patch(body, nullable={"effects", "approval"}, exclude={"headers", "clear_headers"})
        if "tools" in changes:
            changes["tools"] = tuple(changes["tools"])
        if "visibility" in changes:
            changes["visibility"] = _visibility(changes["visibility"], ctx)
        headers = {} if body.clear_headers else (body.headers or {})
        endpoint = dataclasses.replace(current.endpoint, **changes, headers=headers)
        _mcp_call(catalog.put, endpoint, by=owner, headers_keep=body.headers is None and not body.clear_headers)
        LOG.info("MCP endpoint %s changed by %s (%s)", endpoint_id, ctx.subject, ", ".join(sorted(body.model_fields_set)))
        return agent.mcp_endpoint(endpoint_id, ctx)

    @app.post("/api/v1/mcp/endpoints/{ref}/refresh")
    def refresh_mcp_endpoint(ref: str, ctx: AuthContext = Depends(people)) -> dict:
        """Expand an endpoint of yours again: new tools appear, removed ones go."""
        endpoint_id = _ref(ref, ctx)
        record = _mcp_call(_mcp().refresh, endpoint_id, by=owner_key(ctx))
        LOG.info("MCP endpoint %s refreshed by %s: %s", endpoint_id, ctx.subject, record.status)
        return agent.mcp_endpoint(endpoint_id, ctx)

    @app.delete("/api/v1/mcp/endpoints/{ref}", status_code=204)
    def remove_mcp_endpoint(ref: str, ctx: AuthContext = Depends(people)) -> Response:
        catalog, endpoint_id = _mcp(), _ref(ref, ctx)
        current = catalog.get(endpoint_id)
        if current is None or not current.endpoint.visible_to(owner_key(ctx)):
            raise HTTPException(status_code=404, detail="MCP endpoint not found")
        _mcp_call(catalog.delete, endpoint_id, by=owner_key(ctx))
        LOG.info("MCP endpoint %s removed by %s", endpoint_id, ctx.subject)
        return Response(status_code=204)

    @app.get("/api/v1/mcp/tools")
    def list_mcp_tools(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Every MCP tool the caller may use, from plugins and from the database."""
        return {"tools": agent.mcp_tools(ctx)}

    @app.patch("/api/v1/mcp/tools/{ref}")
    def change_mcp_tool(ref: str, body: McpToolPatch, ctx: AuthContext = Depends(people)) -> dict:
        """A tool of yours: enable or disable it, share or hide it on its own, make it deny."""
        catalog, owner, tool_id = _mcp(), owner_key(ctx), _ref(ref, ctx)
        found = catalog.get_tool(tool_id)
        if found is None or not found[0].spec(found[1].endpoint).source.visible_to(owner):
            raise HTTPException(status_code=404, detail="MCP tool not found")
        visibility = _visibility(body.visibility, ctx) if body.visibility is not None else None
        _mcp_call(catalog.set_tool, tool_id, by=owner, enabled=body.enabled, visibility=visibility,
                  approval=body.approval)
        LOG.info("MCP tool %s changed by %s (%s)", tool_id, ctx.subject, ", ".join(sorted(body.model_fields_set)))
        return next(t for t in agent.mcp_tools(ctx) if t["id"] == tool_id)

    # --- Tools: everything the caller's tasks may use, and what they switch off ----------------
    @app.get("/api/v1/tools")
    def list_tools(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Every tool you may use (but skills), where it comes from, and the plugins and MCP endpoints
        that provide them; `enabled` is your own switch, `active` whether your tasks get it."""
        return agent.tools(ctx)

    @app.put("/api/v1/tools/prefs")
    def set_tool_pref(body: ToolPref, ctx: AuthContext = Depends(people)) -> dict:
        """Switch a tool, a plugin's tools or an MCP endpoint's tools on or off for your tasks."""
        try:
            changed = agent.set_tool_pref(ctx, body.target, body.id, body.enabled)
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no {body.target} {body.id!r} among your tools")
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        LOG.info("tools %s:%s %s by %s", body.target, body.id, "on" if body.enabled else "off", ctx.subject)
        return changed

    # --- Knowledge: the system service, services people connect, what they switch off ----------
    def _knowledge():
        from agenticwf.agent.runtime.knowledge import HttpKnowledgeCatalog

        if not isinstance(engine.knowledge, HttpKnowledgeCatalog):
            raise HTTPException(status_code=409, detail="this agent has no knowledge service ([knowledge] url) "
                                                        "and allows none ([knowledge] allowed_hosts)")
        return engine.knowledge

    def _own_service(ref: str, ctx: AuthContext):
        service_id = _ref(ref, ctx)
        current = next((s for s in _knowledge().services(owner_key(ctx)) if s.id == service_id), None)
        if current is None:
            raise HTTPException(status_code=404, detail="knowledge service not found")
        return current

    def _save_service(service, ctx: AuthContext, keep_token: bool) -> dict:
        from agenticwf.agent.runtime.knowledge import validate_service

        catalog = _knowledge()
        try:
            validate_service(service)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        problem = catalog.service_problem(service)
        if problem:
            raise HTTPException(status_code=422, detail=problem)
        try:
            catalog.store.save_service(service, keep_token=keep_token)
        except PermissionError as exc:
            raise HTTPException(status_code=403 if "belongs to" in str(exc) else 409, detail=str(exc))
        saved = next(s for s in catalog.overview(ctx) if s["id"] == service.id)
        return saved

    @app.get("/api/v1/knowledge")
    def list_knowledge(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """The system service and the caller's own, each with the KBs it lists for them."""
        from agenticwf.agent.runtime.knowledge import HttpKnowledgeCatalog

        catalog = engine.knowledge
        if not isinstance(catalog, HttpKnowledgeCatalog):
            return {"sources": [], "can_add": False, "allowed_hosts": []}
        return {"sources": catalog.overview(ctx),
                "can_add": bool(catalog.settings.allowed_hosts) and catalog.store is not None,
                "can_store_tokens": bool(getattr(catalog.store, "can_store_tokens", False)),
                "allowed_hosts": list(catalog.settings.allowed_hosts),
                # what the caller's tasks search with, and the KBs they are given (before a task type narrows them)
                "options": catalog.options(owner_key(ctx)), "chosen": catalog.chosen(owner_key(ctx)),
                "defaults": catalog.defaults(),
                "can_set_options": hasattr(catalog.store, "set_options"),
                "in_use": [{"id": kb.id, "name": kb.name} for kb in catalog.bases(ctx, ())],
                "max_catalog": catalog.settings.max_catalog}

    @app.post("/api/v1/knowledge/services", status_code=201)
    def add_knowledge_service(body: KnowledgeServiceIn, ctx: AuthContext = Depends(people)) -> dict:
        """Connect a knowledge service of yours: its KBs join your tasks' as <ref>@<name>."""
        from agenticwf.agent.runtime.knowledge import KnowledgeService

        catalog, owner = _knowledge(), owner_key(ctx)
        if catalog.store is None:
            raise HTTPException(status_code=409, detail="this runtime has no store for knowledge services")
        service = KnowledgeService(name=body.name, owner=owner, url=body.url.strip(), base_path=body.base_path.strip(),
                                   label=body.label, token=body.token or "", enabled=body.enabled)
        if any(s.id == service.id for s in catalog.services(owner)):
            raise HTTPException(status_code=409, detail=f"you already have a knowledge service named {body.name!r}")
        saved = _save_service(service, ctx, keep_token=False)
        LOG.info("knowledge service %s added (%s) by %s: %s", service.id, service.host, ctx.subject, saved["status"])
        return saved

    @app.patch("/api/v1/knowledge/services/{ref}")
    def change_knowledge_service(ref: str, body: KnowledgeServicePatch, ctx: AuthContext = Depends(people)) -> dict:
        current = _own_service(ref, ctx)
        changes = _patch(body, nullable=set(), exclude={"token", "clear_token"})
        keep = body.token is None and not body.clear_token
        service = dataclasses.replace(current, **changes, token="" if body.clear_token else (body.token or current.token),
                                      token_unreadable=current.token_unreadable and keep)
        saved = _save_service(service, ctx, keep_token=keep)
        LOG.info("knowledge service %s changed by %s (%s)", current.id, ctx.subject, ", ".join(sorted(body.model_fields_set)))
        return saved

    @app.delete("/api/v1/knowledge/services/{ref}", status_code=204)
    def remove_knowledge_service(ref: str, ctx: AuthContext = Depends(people)) -> Response:
        current = _own_service(ref, ctx)
        _knowledge().store.delete_service(owner_key(ctx), current.id)
        LOG.info("knowledge service %s removed by %s", current.id, ctx.subject)
        return Response(status_code=204)

    @app.put("/api/v1/knowledge/prefs")
    def set_knowledge_pref(body: KnowledgePref, ctx: AuthContext = Depends(people)) -> dict:
        """Switch the system service, a service of yours, or one KB of either on or off for your tasks."""
        from agenticwf.agent.runtime.knowledge import ALL, SYSTEM

        catalog, owner = _knowledge(), owner_key(ctx)
        if catalog.store is None:
            raise HTTPException(status_code=409, detail="this runtime has no store for knowledge preferences")
        if body.source == SYSTEM:
            if catalog.client is None:
                raise HTTPException(status_code=404, detail="no knowledge service is configured")
        elif body.kb is None:   # a service of yours: its own enabled flag
            current = _own_service(body.source, ctx)
            catalog.store.save_service(dataclasses.replace(current, enabled=body.enabled), keep_token=True)
            return {"source": current.id, "kb": None, "enabled": body.enabled}
        else:
            body.source = _own_service(body.source, ctx).id
        catalog.store.set_enabled(owner, body.source, body.kb or ALL, body.enabled)
        LOG.info("knowledge %s/%s %s by %s", body.source, body.kb or ALL, "on" if body.enabled else "off", ctx.subject)
        return {"source": body.source, "kb": body.kb, "enabled": body.enabled}

    @app.put("/api/v1/knowledge/options")
    def set_knowledge_options(body: KnowledgeOptions, ctx: AuthContext = Depends(people)) -> dict:
        """Choose the search tools and kb_research preset for your tasks; null: agent.cfg's."""
        catalog, owner = _knowledge(), owner_key(ctx)
        if not hasattr(catalog.store, "set_options"):
            raise HTTPException(status_code=409, detail="this runtime has no store for knowledge options")
        changes = {name: getattr(body, name) for name in body.model_fields_set}
        catalog.store.set_options(owner, **changes)
        LOG.info("knowledge options %s by %s", ", ".join(f"{k}={v or 'default'}" for k, v in sorted(changes.items())),
                 ctx.subject)
        return {"options": catalog.options(owner), "chosen": catalog.chosen(owner), "defaults": catalog.defaults()}

    # --- Long-term memory: each caller's own (docs/admin-page.md, Memory) ----------------------
    @app.get("/api/v1/memories")
    def list_memories(limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
                      ctx: AuthContext = Depends(tasks_dep)) -> dict:
        return agent.memories(ctx, limit, offset)

    @app.delete("/api/v1/memories/{memory_id}", status_code=204)
    def delete_memory(memory_id: str, ctx: AuthContext = Depends(tasks_dep)) -> Response:
        if not agent.forget(ctx, memory_id):
            raise HTTPException(status_code=404, detail="memory not found")
        LOG.info("memory %s deleted by %s", memory_id, ctx.subject)
        return Response(status_code=204)

    @app.delete("/api/v1/memories")
    def delete_memories(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        deleted = agent.forget(ctx)
        LOG.info("%d memories deleted by %s", deleted, ctx.subject)
        return {"deleted": deleted}

    # --- Skills: everyone sees what they may use; people add and manage their own ---------------
    def _skills():
        if agent.skills is None:
            raise HTTPException(status_code=409, detail="this runtime has no skills")
        return agent.skills

    def _own_skill(ref: str, ctx: AuthContext):
        current = _skills().database_skill(_ref(ref, ctx))
        if current is None or not current.owned_by(owner_key(ctx)):
            raise HTTPException(status_code=404, detail="skill not found")
        return current

    def _save_skill(skill, ctx: AuthContext) -> dict:
        try:
            saved = _skills().put(skill, by=owner_key(ctx))
        except SkillError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        except PermissionError as exc:
            raise HTTPException(status_code=403 if "belongs to" in str(exc) else 409, detail=str(exc))
        return saved.to_dict(owner_key(ctx), detail=True)

    @app.get("/api/v1/skills")
    def list_skills(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """System and plugin skills, those shared with the caller, and the caller's own."""
        catalog, owner = _skills(), owner_key(ctx)
        return {"skills": [k.to_dict(owner) for k in catalog.skills(owner)], "can_add": catalog.store is not None}

    @app.get("/api/v1/skills/{ref}")
    def get_skill(ref: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        found = _skills().get(_ref(ref, ctx), owner_key(ctx))
        if found is None:
            raise HTTPException(status_code=404, detail="skill not found")
        return found.to_dict(owner_key(ctx), detail=True)

    @app.post("/api/v1/skills", status_code=201)
    def add_skill(body: SkillIn, ctx: AuthContext = Depends(people)) -> dict:
        """Add a skill of yours: its id is `<your scope>:<name>`, the model calls it skill__<name>."""
        owner = owner_key(ctx)
        skill = Skill(**body.model_dump(exclude={"visibility", "tools", "tags", "files"}), source="database", owner=owner,
                      tools=tuple(body.tools), tags=tuple(body.tags), visibility=_visibility(body.visibility, ctx))
        if _skills().database_skill(skill.id) is not None:
            raise HTTPException(status_code=409, detail=f"you already have a skill named {body.name!r}")
        try:   # checked before anything is stored
            if len(body.files) > FILES_MAX:
                raise SkillError(f"a skill keeps at most {FILES_MAX} files")
            for path, content in body.files.items():
                check_file_path(path)
                check_file_content(path, content)
        except SkillError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        view = _save_skill(skill, ctx)
        for path, content in body.files.items():
            _skills().put_file(skill.id, path, content, by=owner)
        if body.files:
            view = _skills().get(skill.id, owner).to_dict(owner, detail=True)
        LOG.info("skill %s added (%s) by %s", skill.id, skill.visibility, ctx.subject)
        return view

    @app.patch("/api/v1/skills/{ref}")
    def change_skill(ref: str, body: SkillPatch, ctx: AuthContext = Depends(people)) -> dict:
        current = _own_skill(ref, ctx)
        changes = _patch(body, nullable=set(), exclude=set())
        for key in ("tools", "tags"):
            if key in changes:
                changes[key] = tuple(changes[key])
        if "visibility" in changes:
            changes["visibility"] = _visibility(changes["visibility"], ctx)
        view = _save_skill(dataclasses.replace(current, **changes), ctx)
        LOG.info("skill %s changed by %s (%s)", current.id, ctx.subject, ", ".join(sorted(body.model_fields_set)))
        return view

    @app.get("/api/v1/skills/{ref}/files/{path:path}")
    def get_skill_file(ref: str, path: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        try:
            return {"path": path, "content": _skills().read_file(_ref(ref, ctx), path, owner_key(ctx))}
        except SkillError:
            raise HTTPException(status_code=404, detail="file not found")

    @app.put("/api/v1/skills/{ref}/files/{path:path}")
    def put_skill_file(ref: str, path: str, body: SkillFile, ctx: AuthContext = Depends(people)) -> dict:
        """Add or replace a file of a skill you added: kept in the database, never on disk."""
        current = _own_skill(ref, ctx)
        try:
            _skills().put_file(current.id, path, body.content, by=owner_key(ctx))
        except SkillError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        LOG.info("skill %s: file %s saved by %s", current.id, path, ctx.subject)
        return _skills().get(current.id, owner_key(ctx)).to_dict(owner_key(ctx), detail=True)

    @app.delete("/api/v1/skills/{ref}/files/{path:path}", status_code=204)
    def delete_skill_file(ref: str, path: str, ctx: AuthContext = Depends(people)) -> Response:
        current = _own_skill(ref, ctx)
        if not _skills().delete_file(current.id, path, by=owner_key(ctx)):
            raise HTTPException(status_code=404, detail="file not found")
        LOG.info("skill %s: file %s removed by %s", current.id, path, ctx.subject)
        return Response(status_code=204)

    @app.delete("/api/v1/skills/{ref}", status_code=204)
    def remove_skill(ref: str, ctx: AuthContext = Depends(people)) -> Response:
        current = _own_skill(ref, ctx)
        _skills().delete(current.id, by=owner_key(ctx))
        LOG.info("skill %s removed by %s", current.id, ctx.subject)
        return Response(status_code=204)

    # --- Model usage, budgets and feedback (docs/model-routing.md §7, §8) -------------------------
    def _router():
        llm = agent.harness.services.llm
        if not hasattr(llm, "choose"):
            raise HTTPException(status_code=409, detail="this runtime has no model router")
        return llm

    @app.get("/api/v1/usage")
    def model_usage(start: dt.datetime | None = Query(None, alias="from"), end: dt.datetime | None = Query(None, alias="to"),
                    group: Literal["model", "purpose", "owner", "day"] = "model", all: bool = False,
                    ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Your model usage (managers: everyone's, `all=true`), with its cost, grouped."""
        if all and SCOPE_ADMIN not in ctx.scopes:
            raise HTTPException(status_code=403, detail="only managers (admin scope) see everyone's usage")
        router = _router()
        end_ = (end or dt.datetime.now(dt.timezone.utc)).astimezone(dt.timezone.utc)
        start_ = (start or end_ - dt.timedelta(days=30)).astimezone(dt.timezone.utc)
        rows = router.usage.summary(start_, end_, group, None if all else owner_key(ctx))
        return {"usage": rows, "group": group, "cost_unit": router.settings.cost_unit,
                "from": start_.isoformat(), "to": end_.isoformat()}

    @app.get("/api/v1/usage/budget")
    def model_budget(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        router = _router()
        owner = owner_key(ctx)
        spent, budget = router.budget_state(owner)
        found = router.settings.budget_for(owner)
        return {"spent": round(spent, 6), "budget": budget, "period": found.period if found else None,
                "cost_unit": router.settings.cost_unit, "max_tier": router.settings.max_tier_for(owner),
                "routing": router.routing_active()}

    @app.get("/api/v1/usage/suggestions")
    def model_suggestions(days: int = Query(30, ge=1, le=365), ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Per model and purpose: calls, ratings, escalations, and what to change (managers)."""
        if SCOPE_ADMIN not in ctx.scopes:
            raise HTTPException(status_code=403, detail="only managers (admin scope) see the suggestions")
        router = _router()
        return {"rows": router.suggestions(days), "adjusted": dict(router.adjusted),
                "feedback": router.settings.feedback}

    @app.get("/api/v1/feedback/{intent_id}")
    def my_rating(intent_id: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """Your rating of a reply (up, down, or none), on any device."""
        router = _router()
        intent = agent.harness.store.load_intent(intent_id)
        if intent is None or intent.owner != owner_key(ctx):
            raise HTTPException(status_code=404, detail="reply not found")
        return {"intent": intent_id, "rating": router.usage.rating(intent_id, intent.owner)}

    @app.post("/api/v1/feedback", status_code=201)
    def rate_answer(body: FeedbackIn, ctx: AuthContext = Depends(people)) -> dict:
        """A thumbs up or down on a reply: kept for the models that wrote it (docs/model-routing.md §8)."""
        router, store_ = _router(), agent.harness.store
        intent = store_.load_intent(body.intent)
        if intent is None or intent.owner != owner_key(ctx):
            raise HTTPException(status_code=404, detail="reply not found")
        router.usage.forget_rating(intent.id, intent.owner)   # one rating per person and reply: the latest
        rated = 0
        for task in store_.tasks_for_intent(intent.id):
            if task.model_id:
                router.feedback(body.rating, "cron_check" if task.cron else "step", task.model_id, intent.owner,
                                task.id, intent.id, body.reason)
                rated += 1
        if not rated:
            raise HTTPException(status_code=409, detail="no model to rate for this reply")
        return {"rated": rated}

    @app.get("/api/v1/models")
    def list_models(detail: bool = False, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """The models the caller may use. `?detail=true`: the settings page's view, in groups
        (system, shared, mine), with the settings of the caller's own models (never their keys)."""
        return agent.models(ctx, detail=detail)

    def _catalog():
        if agent.catalog is None:
            raise HTTPException(status_code=409, detail="this runtime has no model catalog")
        return agent.catalog

    def _visibility(requested: str, ctx: AuthContext) -> str:
        if requested == "all" and SCOPE_ADMIN not in ctx.scopes:
            raise HTTPException(status_code=403, detail="only managers (admin scope) can share with everyone")
        return requested

    @app.post("/api/v1/models", status_code=201)
    def add_model(body: ModelIn, ctx: AuthContext = Depends(people)) -> dict:
        """Add a model of your own. The API key is stored encrypted and never returned."""
        catalog = _catalog()
        if catalog.get(body.id) is not None:
            raise HTTPException(status_code=409, detail="a model with this id exists")
        owner = owner_key(ctx)
        entry = ModelEntry(**body.model_dump(exclude={"visibility"}), visibility=_visibility(body.visibility, ctx),
                           source="database", owner_subject=owner[0], owner_org=owner[1])
        saved = _save_model(catalog, entry, owner)
        LOG.info("model %s added (%s/%s, %s) by %s", entry.id, entry.provider, entry.model, entry.visibility, ctx.subject)
        return saved.view_for(owner)

    @app.put("/api/v1/models/default")
    def set_default_model(body: DefaultModel, ctx: AuthContext = Depends(people)) -> dict:
        """Choose your default model: your tasks that name none run on it. None: [llm] default again."""
        catalog = _catalog()
        try:
            catalog.set_user_default(owner_key(ctx), body.model or None)
        except PermissionError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        LOG.info("default model of %s: %s", ctx.subject, body.model or "[llm] default")
        return agent.models(ctx, detail=True)

    @app.patch("/api/v1/models/{model_id}")
    def change_model(model_id: str, body: ModelPatch, ctx: AuthContext = Depends(people)) -> dict:
        """Change a model you added; system models and other people's are read-only."""
        catalog, owner = _catalog(), owner_key(ctx)
        current = catalog.get(model_id)
        if current is None or not current.visible_to(owner):
            raise HTTPException(status_code=404, detail="model not found")
        if not current.owned_by(owner):
            raise HTTPException(status_code=403, detail="only the person who added this model can change it")
        changes = _patch(body, nullable={"base_url", "api_key_env"}, exclude={"clear_api_key", "api_key"})
        if "visibility" in changes:
            changes["visibility"] = _visibility(changes["visibility"], ctx)
        keep_key = not body.clear_api_key and not body.api_key   # omitted: the stored key stays, untouched
        api_key = None if body.clear_api_key else (body.api_key or current.api_key)
        saved = _save_model(catalog, dataclasses.replace(current, **changes, api_key=api_key), owner, keep_key)
        LOG.info("model %s changed by %s (%s)", model_id, ctx.subject, ", ".join(sorted(body.model_fields_set)))
        return saved.view_for(owner)

    @app.delete("/api/v1/models/{model_id}", status_code=204)
    def remove_model(model_id: str, ctx: AuthContext = Depends(people)) -> Response:
        catalog, owner = _catalog(), owner_key(ctx)
        current = catalog.get(model_id)
        if current is None or not current.visible_to(owner):
            raise HTTPException(status_code=404, detail="model not found")
        try:
            catalog.delete(model_id, by=owner)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        LOG.info("model %s removed by %s", model_id, ctx.subject)
        return Response(status_code=204)

    def _save_model(catalog, entry, owner, keep_key=False):
        try:
            return catalog.put(entry, by=owner, keep_key=keep_key)
        except PermissionError as exc:   # someone else's or a configured model, or no secrets key
            raise HTTPException(status_code=403 if "belongs to" in str(exc) else 409, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.post("/api/v1/tasks", status_code=202)
    async def submit_task(body: NewTask, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        if body.task_type is not None and not TYPE_REF_RE.match(body.task_type):
            raise HTTPException(status_code=422, detail="invalid task type name")
        if body.model is not None and not agent.model_available(body.model, ctx):
            raise HTTPException(status_code=422, detail="unknown or disabled model")
        task_id = await agent.submit_task(body.prompt, ctx, body.task_type, body.mode, body.model)
        LOG.info("task %s accepted", task_id)
        return {"task_id": task_id, "status": "queued"}

    @app.get("/api/v1/tasks")
    def list_tasks(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        return {"tasks": store.list_tasks(owner_key(ctx))}

    @app.get("/api/v1/tasks/{tid}")
    def get_task(tid: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        task = store.get_task(owner_key(ctx), tid)
        if task is None:
            raise HTTPException(status_code=404, detail="task not found")
        return task

    @app.get("/api/v1/tasks/{tid}/turns")
    def get_execution(tid: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """The execution record: the runtime tasks, their plans, turns and tool calls."""
        if store.intent_thread(owner_key(ctx), tid) is None:
            raise HTTPException(status_code=404, detail="task not found")
        return {"task_id": tid, "tasks": agent.execution(tid)}

    @app.get("/api/v1/tasks/{tid}/trace")
    def get_trace(tid: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """What the agent did for this task, step by step (docs/api.md, Trace)."""
        found = agent.trace(tid, ctx, settings.trace_url) if store.intent_thread(owner_key(ctx), tid) else None
        if found is None:
            raise HTTPException(status_code=404, detail="task not found")
        return found

    @app.post("/api/v1/tasks/{tid}/approvals/{call_id}")
    async def approve_in_task(tid: str, call_id: str, body: Decision, ctx: AuthContext = Depends(chat)) -> dict:
        thread = await agent.thread_of(tid, ctx)
        if thread is None:
            raise HTTPException(status_code=404, detail="task not found or finished")
        return await _decide(thread, call_id, body, ctx)

    @app.post("/api/v1/tasks/{tid}/input")
    async def answer_task(tid: str, msg: Message, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        try:
            return await agent.answer(tid, msg.text, ctx)
        except Conflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    @app.post("/api/v1/tasks/{tid}/cancel")
    async def cancel_task(tid: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        thread = await agent.thread_of(tid, ctx)
        if thread is None:
            if store.intent_thread(owner_key(ctx), tid) is None:
                raise HTTPException(status_code=404, detail="task not found")
            raise HTTPException(status_code=409, detail="task already finished")
        await thread.interrupt(ctx)
        return {"task_id": tid, "status": "cancelling"}

    async def _decide(thread, call_id: str, body: Decision, ctx: AuthContext) -> dict:
        try:
            return await agent.approve(thread, call_id, ctx, approved=body.decision == "approve",
                                       remember=body.remember, reason=body.reason)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except Conflict as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    return app
