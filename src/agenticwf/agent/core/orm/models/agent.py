"""Tables of the agent: the conversation, and the runtime's durable records (docs/agent-flow.md §9).

    agent_chat_sessions   a conversation (a runtime thread); the id is the thread id; its summary
    agent_chat_messages   its transcript: user messages and replies
    agent_intents         TO DO      what was asked; the reply when answered
    agent_tasks           DOING      the runtime tasks of an intent: state, a small head record, the
                                     instructions snapshot (written once), result
    agent_step_messages   DOING      each task's history, one row per entry, written once per entry
    agent_tool_calls      DOING      each task's tool calls, one row per call, rewritten while it is open
    agent_effects         DOING      what tool calls changed, written before the handler runs
    agent_memories        long-term memory: what consolidation kept from finished tasks, per owner
    agent_skills          skills users add (SKILL.md content), each with its owner and visibility
    agent_skill_files     their files (references, templates): in the database, never on disk
    agent_crons           every cron's schedule state; database crons' definitions (docs/cron.md)
    agent_cron_files      database crons' files (CRON.md, checks/, reports/)
    agent_cron_runs       one per run of a cron's due checks
    agent_cron_findings   what checks reported (report_issue)
    agent_cron_metrics    numbers checks recorded (report_check), raw; and hourly / daily aggregates
    agent_issues          problems followed across runs, with their events and suppressions
    agent_cron_proposals  actions a cron proposed instead of waiting for an approval
    agent_reports         period and rollup reports (the database only)
    agent_jobs            the job queue, when worker processes share it ([harness] shared)
    agent_events          events of work done in a worker process, for the API process's sessions
    agent_model_usage     every model call: who, what for, which model, tokens, cost, why that model
    agent_model_feedback  ratings of answers, and evidence (escalations), per model and purpose
    agent_models          the models managers add at run time (secrets encrypted)
    agent_model_defaults  the model each user chose as their default, instead of [llm] default
    agent_mcp_endpoints   MCP endpoints users add at run time (header values encrypted)
    agent_mcp_tools       the tools expanded from them, each with its own owner and visibility
    agent_knowledge_services  knowledge services users connect beside agent.cfg's (token encrypted)
    agent_knowledge_prefs     knowledge bases or services a user switched off for their tasks
    agent_knowledge_options   the search tools and kb_research preset a user chose for their tasks
    agent_tool_prefs      tools, plugins or MCP endpoints a user switched off for their tasks

Every owned row carries ``(owner_subject, owner_org)`` from the AuthContext that created it
(agenticwf.agent.runtime.types.owner_key); readers filter every query on it. Task and intent snapshots are the
runtime's own records, pickled: only the runtime writes and reads them. The columns next to them
are what the API queries.
"""

import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, LargeBinary, String, Text
from sqlalchemy.orm import relationship

from agenticwf.agent.core.orm.base.models import ModelBase
from agenticwf.agent.core.orm.base.session import BASE
from agenticwf.agent.core.orm.base.types import JSON


def _utcnow():
    return datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None)


class _Owned(ModelBase):
    owner_subject = Column(String(255), nullable=False)
    owner_org = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class ChatSession(BASE, _Owned):
    __tablename__ = "agent_chat_sessions"
    __table_args__ = (Index("agent_chat_sessions_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(32), primary_key=True)
    title = Column(String(200), nullable=False)
    # The earlier conversation, summarised when it grew long (agenticwf.agent.runtime.harness: _compact_conversation),
    # and the keys of the messages it covers (intent ids, or message:<id>): a reopened chat sends the
    # summary and the messages it does not cover, rather than summarising everything again.
    summary = Column(Text, nullable=True)
    summary_keys = Column(JSON(), nullable=True)
    messages = relationship("ChatMessage", back_populates="session", order_by="ChatMessage.id",
                            cascade="all, delete-orphan", passive_deletes=True)


class ChatMessage(BASE, ModelBase):
    __tablename__ = "agent_chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(32), ForeignKey("agent_chat_sessions.id", ondelete="CASCADE"),
                        nullable=False, index=True)
    role = Column(String(16), nullable=False)  # user | assistant
    text = Column(Text, nullable=False)
    intent_id = Column(String(32), nullable=True)   # the intent this message started or answered
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    session = relationship("ChatSession", back_populates="messages")


class IntentRow(BASE, _Owned):
    __tablename__ = "agent_intents"
    __table_args__ = (Index("agent_intents_owner_idx", "owner_subject", "owner_org"),
                      Index("agent_intents_status_idx", "status"))

    id = Column(String(32), primary_key=True)
    thread_id = Column(String(32), nullable=False, index=True)
    source = Column(String(16), nullable=False)     # web | cli | api | cron
    status = Column(String(16), nullable=False)     # agenticwf.agent.runtime.records.IntentStatus
    text = Column(Text, nullable=False)
    record = Column(LargeBinary, nullable=False)    # pickled agenticwf.agent.runtime.records.Intent
    reply = Column(JSON(), nullable=True)           # agenticwf.agent.runtime.records.Reply, once answered


class TaskRow(BASE, _Owned):
    __tablename__ = "agent_tasks"
    __table_args__ = (Index("agent_tasks_owner_idx", "owner_subject", "owner_org"),
                      Index("agent_tasks_state_idx", "state"))

    id = Column(String(32), primary_key=True)
    intent_id = Column(String(32), nullable=True, index=True)
    task_type = Column(String(320), nullable=False)  # as requested: a name or <scope>:<name>
    state = Column(String(32), nullable=False)      # agenticwf.agent.runtime.types.RuntimeState
    # Pickled (Task, AgentContext) without their history, tool calls and instructions (those are in
    # agent_step_messages, agent_tool_calls and `base`), so each state change writes little. A row
    # whose `base` is NULL is the older layout: the whole record in `record`.
    record = Column(LargeBinary, nullable=False)
    base = Column(LargeBinary, nullable=True)       # pickled ResolvedTaskType: AGENTS.md, WORKFLOW.md (once)
    result = Column(JSON(), nullable=True)          # agenticwf.agent.runtime.records.TaskResult, once terminal
    memory = Column(String(16), nullable=True)      # Task.consolidation: pending | done (a restart queues pending)
    cancel = Column(Integer, nullable=True)         # cancel requested: a worker in another process sees it


class StepMessageRow(BASE, ModelBase):
    """One entry of a task's history (agenticwf.agent.runtime.llm.base.Message): status, model reply, tool result.
    `part` 0 is the task's own messages, part n the n-th turn's. Appended once; a part is rewritten
    only when compaction replaces its older entries with a summary."""

    __tablename__ = "agent_step_messages"

    task_id = Column(String(32), primary_key=True)
    part = Column(Integer, primary_key=True)
    seq = Column(Integer, primary_key=True)
    step = Column(Integer, nullable=False)          # the task's step when it was written
    role = Column(String(16), nullable=False)
    record = Column(LargeBinary, nullable=False)    # pickled Message


class ToolCallRow(BASE, ModelBase):
    """One tool call of a task (agenticwf.agent.runtime.task.ToolCall), in order. Rewritten while the call is open
    (requested, waiting for approval, running); never again once it has finished."""

    __tablename__ = "agent_tool_calls"
    __table_args__ = (Index("agent_tool_calls_call_idx", "task_id", "call_id"),)

    task_id = Column(String(32), primary_key=True)
    seq = Column(Integer, primary_key=True)
    call_id = Column(String(128), nullable=False)
    step = Column(Integer, nullable=False)
    name = Column(String(128), nullable=False)
    status = Column(String(32), nullable=False)     # agenticwf.agent.runtime.task.ToolCallStatus
    record = Column(LargeBinary, nullable=False)    # pickled ToolCall


class EffectRow(BASE, ModelBase):
    __tablename__ = "agent_effects"

    id = Column(String(32), primary_key=True)
    task_id = Column(String(32), nullable=False, index=True)
    call_id = Column(String(128), nullable=False, index=True)
    tool = Column(String(128), nullable=False)
    kind = Column(String(16), nullable=False)
    target = Column(Text, nullable=False)
    idempotency_key = Column(String(255), nullable=False)
    status = Column(String(16), nullable=False)     # intended | applied | failed
    detail = Column(JSON(), nullable=True)
    at = Column(DateTime, nullable=False, default=_utcnow)


class MemoryRow(BASE, _Owned):
    """One long-term memory of an owner (agenticwf.agent.runtime.context.db_memory): a short text consolidated from
    a finished task's working memory, recalled at the start of later tasks and by memory_search."""

    __tablename__ = "agent_memories"
    __table_args__ = (Index("agent_memories_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(32), primary_key=True)
    text = Column(Text, nullable=False)
    source_task = Column(String(32), nullable=True)


class SkillRow(BASE, _Owned):
    """A skill a user added (agenticwf.agent.runtime.skills.db_store). Its id is `<scope>:<name>` (the owner's scope),
    so names only need to be unique per owner."""

    __tablename__ = "agent_skills"
    __table_args__ = (Index("agent_skills_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(320), primary_key=True)            # <scope>:<name>
    name = Column(String(64), nullable=False)
    description = Column(Text, nullable=False)
    instructions = Column(Text, nullable=False)
    tools = Column(JSON(), nullable=True)                 # globs: while it runs, only these tools
    tags = Column(JSON(), nullable=True)
    visibility = Column(String(16), nullable=False, default="user")   # user | organization | all
    enabled = Column(Integer, nullable=False, default=1)


class SkillFileRow(BASE, ModelBase):
    """A file of a database skill (a reference, a template), read by the model with read_skill_file
    while the skill runs. Database skills keep their files here, never on disk."""

    __tablename__ = "agent_skill_files"

    skill_id = Column(String(320), primary_key=True)
    path = Column(String(255), primary_key=True)          # relative: docs/template.md
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class ModelRow(BASE, ModelBase):
    """A model a manager added while the system runs (agenticwf.agent.runtime.llm.catalog). The API key is
    encrypted (agenticwf.agent.runtime.secrets) with a key that is not in the database."""

    __tablename__ = "agent_models"

    id = Column(String(64), primary_key=True)
    provider = Column(String(16), nullable=False)
    model = Column(String(255), nullable=False)
    label = Column(String(200), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    base_url = Column(String(1024), nullable=True)
    api_key_env = Column(String(255), nullable=True)
    api_key_encrypted = Column(Text, nullable=True)
    api_key_hint = Column(String(16), nullable=True)
    options = Column(JSON(), nullable=True)
    enabled = Column(Integer, nullable=False, default=1)
    created_by = Column(String(255), nullable=True)
    owner_subject = Column(String(255), nullable=True)   # who added it: only they may change it
    owner_org = Column(String(255), nullable=True)
    visibility = Column(String(16), nullable=False, default="all", server_default="all")   # user | organization | all
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class ModelDefaultRow(BASE, _Owned):
    """The model one user chose as their default (agenticwf.agent.runtime.llm.catalog): their tasks that
    name no model run on it instead of [llm] default. Only users who chose one have a row."""

    __tablename__ = "agent_model_defaults"

    id = Column(String(64), primary_key=True)             # sha256 of the owner
    model_id = Column(String(64), nullable=False)         # an id in the catalog; ignored once it is gone


class McpEndpointRow(BASE, _Owned):
    """An MCP endpoint a user added (agenticwf.agent.runtime.mcp.catalog). Its id is `<scope>:<name>` (the owner's
    scope), so names only need to be unique per owner. Header values (tokens) are encrypted
    (agenticwf.agent.runtime.secrets) with a key that is not in the database."""

    __tablename__ = "agent_mcp_endpoints"
    __table_args__ = (Index("agent_mcp_endpoints_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(320), primary_key=True)            # <scope>:<name>
    name = Column(String(64), nullable=False)             # the tool prefix: <name>__<tool>
    label = Column(String(200), nullable=False, default="")
    description = Column(Text, nullable=False, default="")
    transport = Column(String(16), nullable=False, default="http")
    url = Column(String(2048), nullable=True)
    headers_encrypted = Column(Text, nullable=True)       # JSON {name: value}, encrypted
    header_hints = Column(JSON(), nullable=True)          # {name: "…1234"}
    settings = Column(JSON(), nullable=True)              # tools filter, effects, approval, timeout, access
    visibility = Column(String(16), nullable=False, default="user")   # user | organization | all
    enabled = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="pending")    # last expansion: ok | error | ...
    error = Column(Text, nullable=True)
    problems = Column(JSON(), nullable=True)
    server = Column(String(255), nullable=True)
    server_version = Column(String(64), nullable=True)
    checked_at = Column(DateTime, nullable=True)
    tools = relationship("McpToolRow", back_populates="endpoint", cascade="all, delete-orphan", passive_deletes=True)


class McpToolRow(BASE, _Owned):
    """A tool expanded from an MCP endpoint; its owner sets whether it is enabled and who sees it.
    Its id is `<scope>:<name>`; `name` (<endpoint>__<tool>) is what the model calls it, unless a
    caller can see another tool of that name (agenticwf.agent.runtime.tools.registry qualifies one of them)."""

    __tablename__ = "agent_mcp_tools"

    id = Column(String(400), primary_key=True)            # <scope>:<endpoint>__<tool>
    endpoint_id = Column(String(320), ForeignKey("agent_mcp_endpoints.id", ondelete="CASCADE"),
                         nullable=False, index=True)
    name = Column(String(64), nullable=False)
    mcp_name = Column(String(255), nullable=False)
    summary = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    input_schema = Column(JSON(), nullable=False)
    effects = Column(String(16), nullable=False)
    approval = Column(String(16), nullable=False)
    idempotent = Column(Integer, nullable=False, default=0)
    read_only = Column(Integer, nullable=False, default=0)
    enabled = Column(Integer, nullable=False, default=1)
    visibility = Column(String(16), nullable=False, default="")   # "" = the endpoint's
    endpoint = relationship("McpEndpointRow", back_populates="tools")


class KnowledgeServiceRow(BASE, _Owned):
    """A knowledge service (AgenticWF-KB) a user connected (agenticwf.agent.runtime.knowledge), beside the one
    in agent.cfg. Its id is `<scope>:<name>`; its KBs are `<ref>@<name>` to the model. The bearer
    token is encrypted (agenticwf.agent.runtime.secrets) with a key that is not in the database."""

    __tablename__ = "agent_knowledge_services"
    __table_args__ = (Index("agent_knowledge_services_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(320), primary_key=True)            # <scope>:<name>
    name = Column(String(64), nullable=False)
    label = Column(String(200), nullable=False, default="")
    url = Column(String(2048), nullable=False)
    base_path = Column(String(255), nullable=False, default="/kb")
    token_encrypted = Column(Text, nullable=True)
    token_hint = Column(String(32), nullable=True)        # "…1234"
    enabled = Column(Integer, nullable=False, default=1)


class KnowledgePrefRow(BASE, _Owned):
    """A knowledge base, or a whole knowledge service (`kb` = "*"), one user switched off for their
    tasks. Only switched-off ones have a row."""

    __tablename__ = "agent_knowledge_prefs"
    __table_args__ = (Index("agent_knowledge_prefs_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(64), primary_key=True)             # sha256 of (owner, source, kb)
    source = Column(String(320), nullable=False)          # "system" or a service id
    kb = Column(String(512), nullable=False)              # the KB's ref on that service, or "*"


class KnowledgeOptionRow(BASE, _Owned):
    """The knowledge search options one user chose for their tasks instead of agent.cfg's
    ([knowledge] tools, research_preset). Only users who chose one have a row; NULL: agent.cfg's."""

    __tablename__ = "agent_knowledge_options"

    id = Column(String(64), primary_key=True)             # sha256 of the owner
    tools = Column(String(16), nullable=True)             # auto | kb_search | kb_research
    research_preset = Column(String(16), nullable=True)   # model | kb | auto | quick | deep | wiki | smart


class ToolPrefRow(BASE, _Owned):
    """A tool, or every tool of a plugin or an MCP endpoint, one user switched off for their tasks
    (agenticwf.agent.runtime.tools.prefs). Only switched-off ones have a row."""

    __tablename__ = "agent_tool_prefs"
    __table_args__ = (Index("agent_tool_prefs_owner_idx", "owner_subject", "owner_org"),)

    id = Column(String(64), primary_key=True)             # sha256 of (owner, target)
    target = Column(String(400), nullable=False)          # tool:<name> | plugin:<id> | mcp:<endpoint id>


# --- crons (docs/cron.md) -----------------------------------------------------------------------------

class CronRow(BASE, ModelBase):
    """Every cron the scheduler knows: its state; and for a database cron, its definition (its
    files are agent_cron_files). A user's sandbox cron has a row once its owner enabled it."""

    __tablename__ = "agent_crons"

    id = Column(String(320), primary_key=True)            # <scope>:<name>
    source = Column(String(16), nullable=False)           # system | plugin | user | database
    owner_subject = Column(String(255), nullable=True)
    owner_org = Column(String(255), nullable=True)
    owner_scopes = Column(JSON(), nullable=True)          # user crons: the owner's scopes when last saved
    visibility = Column(String(16), nullable=True)        # database crons
    enabled = Column(Integer, nullable=True)              # None: as the definition says
    next_run_at = Column(DateTime, nullable=True)
    checks_due = Column(JSON(), nullable=True)            # check -> next due (ISO, UTC)
    checks_last = Column(JSON(), nullable=True)           # check -> last run start (ISO, UTC)
    periods_done = Column(JSON(), nullable=True)          # period -> end of the last report (ISO, UTC)
    lease_owner = Column(String(64), nullable=True)
    lease_until = Column(DateTime, nullable=True)
    last_run_at = Column(DateTime, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class CronFileRow(BASE, ModelBase):
    __tablename__ = "agent_cron_files"

    cron_id = Column(String(320), primary_key=True)
    path = Column(String(255), primary_key=True)          # CRON.md, checks/<check>.md, reports/<x>.md
    content = Column(Text, nullable=False)
    updated_at = Column(DateTime, nullable=False, default=_utcnow, onupdate=_utcnow)


class CronRunRow(BASE, ModelBase):
    __tablename__ = "agent_cron_runs"
    __table_args__ = (Index("agent_cron_runs_cron_idx", "cron_id", "started_at"),)

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False)
    intent_id = Column(String(32), nullable=True, index=True)
    status = Column(String(16), nullable=False)           # running | finished | failed
    started_at = Column(DateTime, nullable=False)
    finished_at = Column(DateTime, nullable=True)
    due = Column(JSON(), nullable=True)                   # checks due
    skipped = Column(JSON(), nullable=True)               # due but still running from an earlier run
    checks = Column(JSON(), nullable=True)                # check -> {task, status, summary, error, findings}
    error = Column(Text, nullable=True)


class CronFindingRow(BASE, ModelBase):
    __tablename__ = "agent_cron_findings"
    __table_args__ = (Index("agent_cron_findings_cron_idx", "cron_id", "at"),)

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False)
    run_id = Column(String(32), nullable=False, index=True)
    check_name = Column(String(64), nullable=False)
    identity = Column(String(1024), nullable=False)
    issue_id = Column(String(32), nullable=True, index=True)
    grp = Column(String(255), nullable=False)             # the group (site)
    severity = Column(String(32), nullable=False)
    severity_rank = Column(Integer, nullable=False)
    title = Column(Text, nullable=False)
    details = Column(Text, nullable=True)
    links = Column(JSON(), nullable=True)
    evidence = Column(JSON(), nullable=True)
    related = Column(JSON(), nullable=True)
    suppressed = Column(String(32), nullable=True)        # the suppression that matched
    at = Column(DateTime, nullable=False)


class CronMetricRow(BASE, ModelBase):
    __tablename__ = "agent_cron_metrics"
    __table_args__ = (Index("agent_cron_metrics_idx", "cron_id", "name", "grp", "at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    cron_id = Column(String(320), nullable=False)
    check_name = Column(String(64), nullable=False)
    run_id = Column(String(32), nullable=False)
    name = Column(String(128), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=True)
    labels = Column(JSON(), nullable=True)
    grp = Column(String(255), nullable=True)
    link = Column(Text, nullable=True)
    at = Column(DateTime, nullable=False)


class CronMetricAggregateRow(BASE, ModelBase):
    __tablename__ = "agent_cron_metric_aggregates"
    __table_args__ = (Index("agent_cron_metric_aggregates_idx", "cron_id", "name", "resolution", "grp", "bucket"),)

    cron_id = Column(String(320), primary_key=True)
    name = Column(String(128), primary_key=True)
    resolution = Column(String(8), primary_key=True)      # hour | day
    bucket = Column(DateTime, primary_key=True)           # the start of the hour or day (UTC)
    series = Column(String(512), primary_key=True)        # the labels, as sorted JSON
    labels = Column(JSON(), nullable=True)
    grp = Column(String(255), nullable=True)
    unit = Column(String(32), nullable=True)
    count = Column(Integer, nullable=False)
    min = Column(Float, nullable=False)
    max = Column(Float, nullable=False)
    sum = Column(Float, nullable=False)
    last = Column(Float, nullable=False)
    links = Column(JSON(), nullable=True)                 # a few links to the original data


class IssueRow(BASE, ModelBase):
    __tablename__ = "agent_issues"
    __table_args__ = (Index("agent_issues_cron_idx", "cron_id", "status"), Index("agent_issues_grp_idx", "cron_id", "grp"))

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False)
    check_name = Column(String(64), nullable=False)
    identity = Column(String(1024), nullable=False, unique=True)
    key = Column(JSON(), nullable=False)
    grp = Column(String(255), nullable=False)
    title = Column(Text, nullable=False)
    severity = Column(String(32), nullable=False)
    severity_rank = Column(Integer, nullable=False)
    severity_since = Column(DateTime, nullable=True)      # the primary period the severity belongs to
    status = Column(String(16), nullable=False)           # open | acknowledged | in_progress | solved | ignored | snoozed
    until = Column(DateTime, nullable=True)               # ignored / snoozed until
    first_seen = Column(DateTime, nullable=False)
    last_seen = Column(DateTime, nullable=False)
    solved_at = Column(DateTime, nullable=True)
    solved_by = Column(String(255), nullable=True)        # a subject, or "itself"
    occurrences = Column(Integer, nullable=False, default=1)
    reopened = Column(Integer, nullable=False, default=0)
    missed = Column(Integer, nullable=False, default=0)
    owner = Column(String(255), nullable=True)
    ticket = Column(String(512), nullable=True)
    draft_ticket = Column(Text, nullable=True)
    related = Column(JSON(), nullable=True)
    notes = Column(JSON(), nullable=True)                 # [{by, at, text}]
    details = Column(Text, nullable=True)
    links = Column(JSON(), nullable=True)


class IssueEventRow(BASE, ModelBase):
    __tablename__ = "agent_issue_events"
    __table_args__ = (Index("agent_issue_events_cron_idx", "cron_id", "at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    issue_id = Column(String(32), nullable=False, index=True)
    cron_id = Column(String(320), nullable=False)
    at = Column(DateTime, nullable=False)
    actor = Column(String(255), nullable=False)              # a subject, or run:<id>
    action = Column(String(32), nullable=False)
    from_status = Column(String(16), nullable=True)
    to_status = Column(String(16), nullable=True)
    data = Column(JSON(), nullable=True)
    comment = Column(Text, nullable=True)


class SuppressionRow(BASE, ModelBase):
    __tablename__ = "agent_issue_suppressions"

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False, index=True)
    check_name = Column(String(64), nullable=True)             # None: every check of the cron
    match_keys = Column(JSON(), nullable=False)                # key field -> value
    reason = Column(Text, nullable=True)
    actor = Column(String(255), nullable=False)
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    until = Column(DateTime, nullable=True)


class ProposalRow(BASE, ModelBase):
    __tablename__ = "agent_cron_proposals"
    __table_args__ = (Index("agent_cron_proposals_cron_idx", "cron_id", "status"),)

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False)
    run_id = Column(String(32), nullable=False)
    check_name = Column(String(64), nullable=False)
    issue_id = Column(String(32), nullable=True, index=True)
    tool = Column(String(128), nullable=False)
    arguments = Column(JSON(), nullable=True)
    reason = Column(Text, nullable=True)
    status = Column(String(16), nullable=False)           # pending | approved | rejected | failed
    created_at = Column(DateTime, nullable=False, default=_utcnow)
    decided_by = Column(String(255), nullable=True)
    decided_at = Column(DateTime, nullable=True)
    result = Column(JSON(), nullable=True)


class ReportRow(BASE, ModelBase):
    __tablename__ = "agent_reports"
    __table_args__ = (Index("agent_reports_date_idx", "year", "month", "day"),
                      Index("agent_reports_cron_idx", "cron_id", "kind", "start_at"))

    id = Column(String(32), primary_key=True)
    cron_id = Column(String(320), nullable=False)
    kind = Column(String(64), nullable=False)             # a period's or rollup's name, or "custom"
    rollup = Column(Integer, nullable=False, default=0)
    start_at = Column(DateTime, nullable=False)           # UTC
    end_at = Column(DateTime, nullable=False)
    timezone = Column(String(64), nullable=False)
    year = Column(Integer, nullable=False)                # the local start date
    month = Column(Integer, nullable=False)
    day = Column(Integer, nullable=False)
    version = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False)           # pending | ready | failed
    sites = Column(JSON(), nullable=True)                 # groups with issues, for the site filter
    summary = Column(Text, nullable=True)
    data = Column(JSON(), nullable=True)
    markdown = Column(Text, nullable=True)
    template = Column(String(255), nullable=True)
    error = Column(Text, nullable=True)
    created_by = Column(String(255), nullable=True)       # a subject, or "schedule"
    created_at = Column(DateTime, nullable=False, default=_utcnow)


# --- shared workers ([harness] shared: docs/deployment.md) ---------------------------------------------

class JobRow(BASE, ModelBase):
    """A queued or running job, claimed by whichever process's worker gets it first."""

    __tablename__ = "agent_jobs"
    __table_args__ = (Index("agent_jobs_claim_idx", "lease_until", "enqueued_at"),)

    key = Column(String(400), primary_key=True)           # <kind>:<ref>
    kind = Column(String(32), nullable=False)
    ref = Column(String(320), nullable=False)
    owner = Column(String(320), nullable=True)            # whose work (the owner's scope): fairness
    lease_owner = Column(String(128), nullable=True)
    lease_until = Column(DateTime, nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    again = Column(Integer, nullable=False, default=0)    # queued again while held: run once more after
    enqueued_at = Column(DateTime, nullable=False, default=_utcnow)
    trace = Column(String(64), nullable=True)             # W3C traceparent of what queued it (tracing)


class EventRow(BASE, ModelBase):
    """An event of a task or intent run in a worker process, for the session in the API process."""

    __tablename__ = "agent_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sink = Column(String(400), nullable=False)            # the session (thread) id, or cron:<id>
    subject = Column(String(255), nullable=False)
    payload = Column(JSON(), nullable=False)
    at = Column(DateTime, nullable=False, default=_utcnow)


# --- model routing (docs/model-routing.md) ---------------------------------------------------------------

class ModelUsageRow(BASE, ModelBase):
    """One model call: its owner, purpose, model, tokens and cost, and why that model was chosen."""

    __tablename__ = "agent_model_usage"
    __table_args__ = (Index("agent_model_usage_owner_idx", "owner_subject", "owner_org", "at"),
                      Index("agent_model_usage_model_idx", "purpose", "model_id", "at"))

    id = Column(Integer, primary_key=True, autoincrement=True)
    at = Column(DateTime, nullable=False, default=_utcnow)
    owner_subject = Column(String(255), nullable=True)
    owner_org = Column(String(255), nullable=True)
    task_id = Column(String(32), nullable=True, index=True)
    intent_id = Column(String(32), nullable=True, index=True)
    purpose = Column(String(32), nullable=False)
    model_id = Column(String(64), nullable=False)
    provider = Column(String(16), nullable=True)
    input_tokens = Column(Integer, nullable=False, default=0)
    cached_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost = Column(Float, nullable=False, default=0.0)
    route = Column(String(16), nullable=True)             # pinned | purpose | routed | default | context | fallback
    own_model = Column(Integer, nullable=False, default=0)  # the caller's own model: counts against no deployment budget
    seconds = Column(Float, nullable=True)                # how long the call took
    trace_id = Column(String(32), nullable=True)          # its OpenTelemetry trace, for the trace view's link


class ModelFeedbackRow(BASE, ModelBase):
    """A rating of an answer by a person (up / down), or evidence the harness saw (an escalation),
    for the model and purpose that produced it."""

    __tablename__ = "agent_model_feedback"
    __table_args__ = (Index("agent_model_feedback_idx", "purpose", "model_id", "at"),)

    id = Column(Integer, primary_key=True, autoincrement=True)
    at = Column(DateTime, nullable=False, default=_utcnow)
    owner_subject = Column(String(255), nullable=True)
    owner_org = Column(String(255), nullable=True)
    task_id = Column(String(32), nullable=True)
    intent_id = Column(String(32), nullable=True)
    purpose = Column(String(32), nullable=False)
    model_id = Column(String(64), nullable=False)
    kind = Column(String(16), nullable=False)             # up | down | escalation
    reason = Column(Text, nullable=True)
