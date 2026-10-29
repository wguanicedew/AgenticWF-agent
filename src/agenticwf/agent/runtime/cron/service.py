"""
CronService: runs crons (docs/cron.md). One per process, beside the AgentHarness.

    tick()              every `[cron] tick` seconds: for each enabled cron, under its lease: start a
                        run of the checks that are due, queue the reports whose period ended,
                        and once an hour delete what `retention` no longer keeps
    a run               an intent of source `cron` with one planned task per due check, run by the
                        workers like any work; when every task has ended, the harness calls
                        _intent_finished, which files each check's outcome (agenticwf.agent.runtime.cron.tracker)
    a report            a `report` worker job: data and Markdown from agenticwf.agent.runtime.cron.reports, the
                        summary from one model call
    people              issue actions (check boxes), suppressions, proposals, run now, enable,
                        database crons: each checked against the cron's role (agenticwf.agent.runtime.cron.catalog)
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import uuid
from dataclasses import dataclass
from typing import Any, Callable

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.cron import reports, store, tracker
from agenticwf.agent.runtime.cron.catalog import CronCatalog
from agenticwf.agent.runtime.cron.definition import FILES_MAX, FILE_MAX, Check, Cron, CronError, parse_cron
from agenticwf.agent.runtime.cron.schedule import UTC, jitter_offset, now as utcnow
from agenticwf.agent.runtime.harness.harness import AgentHarness
from agenticwf.agent.runtime.harness.store import Job
from agenticwf.agent.runtime.llm.base import LLMError, LLMRequest, Message
from agenticwf.agent.runtime.records import Intent, TaskSpec
from agenticwf.agent.runtime.task import Task
from agenticwf.agent.runtime.types import RuntimeState, owner_key, scope_of

LOG = get_logger(__name__)
ADMIN = "admin"


@dataclass(frozen=True)
class CronSettings:
    """`[cron]` in agent.cfg."""

    enabled: bool = True                           # run the scheduler in this process
    tick: float = 30.0                             # seconds between scheduler passes
    lease: float = 120.0                           # seconds a process holds a cron while it starts its work
    system_scopes: frozenset[str] | None = None    # caps what system and plugin crons may request; None: no cap
    max_scopes: frozenset[str] | None = None       # caps a user's cron's scopes; None: the owner's
    deny_tools: tuple[str, ...] = ()               # globs denied to every cron task (e.g. ticket tools)


def thread_of(cron_id: str) -> str:
    return "cron" + hashlib.sha256(cron_id.encode()).hexdigest()[:28]


def _ago(delta: dt.timedelta) -> str:
    minutes = int(delta.total_seconds() // 60)
    return f"{minutes} min" if minutes < 120 else f"{minutes // 60} h {minutes % 60} min"


class CronService:
    def __init__(self, harness: AgentHarness, catalog: CronCatalog, settings: CronSettings | None = None,
                 notify: Callable[[Cron, dict], None] | None = None):
        self.harness = harness
        self.catalog = catalog
        self.settings = settings or CronSettings()
        self.notify = notify or (lambda cron, event: None)
        self.holder = uuid.uuid4().hex[:16]
        self.clock: Callable[[], dt.datetime] = utcnow   # the time (tests move it)
        self._loop: asyncio.Task | None = None
        self._expired_at: dt.datetime | None = None
        harness.intent_hooks.append(self._intent_finished)
        harness.job_handlers["report"] = self.build_report

    # ============================================================================================
    # The scheduler
    # ============================================================================================
    async def start(self) -> None:
        for report_id in store.pending_reports():   # reports a stopped process left
            self.harness.store.enqueue(Job("report", report_id))   # whose: not known here; shared fairly as nobody's
        if self.settings.enabled and self._loop is None:
            self._loop = asyncio.get_running_loop().create_task(self._run())

    async def stop(self) -> None:
        if self._loop is not None:
            self._loop.cancel()
            await asyncio.gather(self._loop, return_exceptions=True)
            self._loop = None

    async def _run(self) -> None:
        while True:
            try:
                # Reading the crons (files, the database) and deleting old rows happen in a thread;
                # starting runs stays on the event loop, where the job store and the harness live.
                crons = await asyncio.to_thread(self.catalog.scheduled)
                self.tick(crons=crons, expire=False)
                await asyncio.to_thread(self._expire, crons, self.clock())
            except Exception:   # the scheduler outlives any one cron
                LOG.exception("cron scheduler pass failed")
            await asyncio.sleep(self.settings.tick)

    def tick(self, now: dt.datetime | None = None, crons: list[Cron] | None = None, expire: bool = True) -> list[str]:
        """One pass over every cron. Returns the runs started."""
        now = now or self.clock()
        crons = self.catalog.scheduled() if crons is None else crons
        started = []
        for cron in crons:
            try:
                run_id = self._tick_cron(cron, now)
                if run_id:
                    started.append(run_id)
            except Exception as exc:
                LOG.exception("cron %s: scheduler pass failed", cron.id)
                store.set_state(cron.id, last_error=f"{type(exc).__name__}: {exc}")
        if expire:
            self._expire(crons, now)
        return started

    def _expire(self, crons: list[Cron], now: dt.datetime) -> None:
        """Once an hour: delete what each cron's `retention` no longer keeps."""
        if self._expired_at is not None and now - self._expired_at < dt.timedelta(hours=1):
            return
        self._expired_at = now
        for cron in crons:
            try:
                store.expire(cron.id, now - dt.timedelta(seconds=cron.retention),
                             now - dt.timedelta(seconds=cron.metrics_raw_retention))
            except Exception:
                LOG.exception("cron %s: retention failed", cron.id)

    def enabled(self, cron: Cron, state: dict | None) -> bool:
        if state is not None and state.get("enabled") is not None:
            return bool(state["enabled"])
        return cron.enabled and cron.source != "user"   # a sandbox cron runs once its owner enabled it

    def _tick_cron(self, cron: Cron, now: dt.datetime) -> str | None:
        state = store.ensure_state(cron.id, cron.source, cron.owner)
        if not self.enabled(cron, state):
            return None
        if not store.lease(cron.id, self.holder, now + dt.timedelta(seconds=self.settings.lease), now):
            return None
        try:
            tz, offset = cron.tz, dt.timedelta(seconds=jitter_offset(cron.id, cron.jitter))
            due_at = dict(state.get("checks_due") or {})
            due: list[str] = []
            grace = dt.timedelta(seconds=2 * self.settings.tick + 60)
            for check in cron.checks:
                if not check.enabled:
                    continue
                schedule = cron.check_schedule(check)
                when = store.parse_iso(due_at.get(check.name))
                if when is None:   # first seen: from the next slot
                    due_at[check.name] = (schedule.next_after(now, tz) + offset).isoformat()
                elif when <= now:
                    if cron.catch_up == "none" and now - when > grace:
                        LOG.info("cron %s: %s missed its slot; waiting for the next", cron.id, check.name)
                    else:
                        due.append(check.name)
                    due_at[check.name] = (schedule.next_after(now, tz) + offset).isoformat()
            run_id = self.start_run(cron, due, now, state=state) if due else None
            periods_done = self._queue_reports(cron, dict(state.get("periods_done") or {}), now)
            next_run = min((store.parse_iso(v) for v in due_at.values()), default=None)
            store.set_state(cron.id, checks_due=due_at, periods_done=periods_done, next_run_at=next_run)
            return run_id
        finally:
            store.release(cron.id, self.holder)

    def _queue_reports(self, cron: Cron, done: dict, now: dt.datetime) -> dict:
        tz = cron.tz
        for spec in (*cron.periods, *cron.rollups):
            current = spec.clock.start(now, tz)
            last = store.parse_iso(done.get(spec.name))
            if last is None:   # first seen: the first report covers the first whole period
                done[spec.name] = current.isoformat()
                continue
            for _ in range(24):   # a long stop: at most this many missed reports
                if last >= current:
                    break
                end = spec.clock.next(last, tz)
                if end > current:
                    break
                self.queue_report(cron, spec.name, last, end, created_by="schedule", rollup=spec.rollup_of is not None)
                last = end
            done[spec.name] = max(last, store.parse_iso(done[spec.name])).isoformat()
        return done

    # ============================================================================================
    # Runs
    # ============================================================================================
    def principal(self, cron: Cron, state: dict | None, expires: dt.datetime) -> AuthContext:
        """Who a run acts as (docs/cron.md §7)."""
        if cron.trusted:
            scopes = cron.scopes if self.settings.system_scopes is None else cron.scopes & self.settings.system_scopes
            return AuthContext(subject=cron.principal, client_id=cron.principal, scopes=frozenset(scopes),
                               expires_at=int(expires.timestamp()), auth_method="cron")
        scopes = frozenset((state or {}).get("owner_scopes") or ())
        if self.settings.max_scopes is not None:
            scopes &= self.settings.max_scopes
        return AuthContext(subject=cron.owner[0], user_id=cron.owner[0], organization_id=cron.owner[1],
                           scopes=scopes, expires_at=int(expires.timestamp()), auth_method="cron")

    def start_run(self, cron: Cron, checks: list[str], now: dt.datetime | None = None, state: dict | None = None,
                  by: str = "schedule") -> str | None:
        """A run of these checks: one planned task each. A check still running from an earlier
        run is skipped, never run twice at once."""
        now = now or self.clock()
        state = state or store.ensure_state(cron.id, cron.source, cron.owner)
        if not cron.trusted and not state.get("owner_scopes"):
            store.set_state(cron.id, last_error="not enabled by its owner: enable it to run it")
            return None
        running = store.running_checks(cron.id)
        wanted = [c for c in checks if cron.check(c) is not None]
        run_checks = [c for c in wanted if c not in running]
        skipped = [c for c in wanted if c in running]
        run_id = store.create_run(cron.id, run_checks, skipped, now)
        if not run_checks:
            store.update_run(run_id, status="finished", finished_at=now)
            return run_id
        last = dict(state.get("checks_last") or {})
        specs = [self._spec(cron, cron.check(name), run_id, now, store.parse_iso(last.get(name))) for name in run_checks]
        deadline = now + dt.timedelta(seconds=max(c.timeout for c in (cron.check(n) for n in run_checks)) + 300)
        auth = self.principal(cron, state, deadline)
        intent = Intent(thread_id=thread_of(cron.id), owner=owner_key(auth), auth=auth,
                        text=f"{cron.qualified}: {', '.join(run_checks)}", source="cron", model=cron.model,
                        planned=specs)
        store.update_run(run_id, intent_id=intent.id)
        self.harness.store.save_intent(intent)
        self.harness.start_intent(intent.id)
        last.update({name: now.isoformat() for name in run_checks})
        store.set_state(cron.id, checks_last=last, last_run_at=now, last_error=None)
        self.notify(cron, {"type": "cron_run", "cron": cron.id, "run": run_id, "phase": "started",
                           "checks": run_checks, "skipped": skipped, "by": by})
        return run_id

    def _spec(self, cron: Cron, check: Check, run_id: str, now: dt.datetime, last: dt.datetime | None) -> TaskSpec:
        context = tracker.check_context(cron, check, now)
        local = now.astimezone(cron.tz).strftime("%Y-%m-%d %H:%M")
        window = (f"since this check last ran, {_ago(now - last)} ago ({last.astimezone(cron.tz):%H:%M})"
                  if last else "recently (this check has not run before)")

        def issue_line(i: dict) -> str:
            notes = "; ".join(n["text"] for n in (i.get("notes") or [])[-3:])
            key = ", ".join(f"{k}={v}" for k, v in (i.get("key") or {}).items())
            return (f"- [{i['id']}] {i['severity']}: {i['title']} ({key}; {i['status'].replace('_', ' ')} since "
                    f"{i['first_seen'].astimezone(cron.tz):%m-%d %H:%M}, last seen {i['last_seen'].astimezone(cron.tz):%m-%d %H:%M})"
                    + (f" - people's notes: {notes}" if notes else ""))

        followed = [i for i in context["issues"] if i["status"] in tracker.FOLLOWED]
        quiet = [i for i in context["issues"] if i["status"] not in tracker.FOLLOWED]
        others = [f"- [{i['id']}] {i['check_name']} at {i['grp']}: {i['title']} ({i['severity']})" for i in context["others"]]
        suppressions = [f"- {', '.join(f'{k}={v}' for k, v in s['match_keys'].items())}: {s['reason'] or ''}"
                        + (f" (until {s['until'].astimezone(cron.tz):%m-%d %H:%M})" if s["until"] else "")
                        for s in context["suppressions"]]
        suppressions += [f"- [{i['id']}] {i['title']} ({i['status']})" for i in quiet]
        metrics = (", and the metrics " + ", ".join(f"{m.name} ({m.unit}; labels {', '.join(m.labels) or 'none'})"
                                                    for m in check.metrics)) if check.metrics else ""
        request = self.harness._prompt(
            "cron_check", check=check.name, cron=cron.qualified, description=check.description or cron.description,
            now=local, timezone=cron.timezone, window=window, key=", ".join(check.key),
            severities=", ".join(check.severities), metrics=metrics,
            open_issues="\n".join(issue_line(i) for i in followed) or "None.",
            other_issues="\n".join(others) or "None.", suppressions="\n".join(suppressions) or "None.")
        return TaskSpec(
            key=check.name, request=request, task_type=cron.task_type,
            instructions=((f"Cron: {cron.qualified}", cron.instructions), (f"Check: {check.name}", check.instructions)),
            tools=check.tools, skills=check.skills, mode=cron.mode.value, timeout=check.timeout,
            unattended=cron.unattended, tier=cron.tier,
            cron={"cron": cron.id, "run": run_id, "check": check.name, "key": list(check.key),
                  "severities": list(check.severities), "group_by": cron.group_by, "group_default": cron.group_default,
                  "resolve_after": cron.resolve_after_for(check),
                  "metrics": {m.name: {"unit": m.unit, "labels": list(m.labels)} for m in check.metrics},
                  "findings": []})

    def _intent_finished(self, intent: Intent, tasks: list[Task]) -> None:
        """Every check task of a run has ended: file each check's outcome, once."""
        if intent.source != "cron":
            return
        run = store.run_of_intent(intent.id)
        if run is None or run["status"] != "running":
            return
        cron = self.catalog.find(run["cron_id"])
        now = self.clock()
        primary = cron.primary if cron else None
        primary_start = primary.clock.start(now, cron.tz) if primary else None
        changed: list[str] = []
        for task in tasks:
            c = task.cron
            if not c:
                continue
            findings = list(c.get("findings") or [])
            result = self.harness.store.load_result(task.id)
            if task.state is RuntimeState.COMPLETED:
                status = c.get("status") or ("issues" if findings else "ok")
            else:
                status = "failed"
            error = None if task.state is RuntimeState.COMPLETED else (task.error or {}).get("message") or task.state.value
            settings = tracker.CheckSettings(c["cron"], c["check"], tuple(c["key"]), tuple(c["severities"]),
                                             c["group_by"], c["group_default"], int(c["resolve_after"]))
            outcome = tracker.apply_check(
                settings, run["id"], status in ("ok", "issues"), findings, now, primary_start,
                record={"task": task.id, "status": status, "error": error,
                        "summary": c.get("summary") or (result.answer if result else "") or ""},
                metrics=c.get("metrics_recorded") or [], proposals=getattr(task, "proposals", []) or [])
            if outcome:
                changed += outcome["new"] + outcome["seen"] + outcome["reopened"] + outcome["solved"]
        store.update_run(run["id"], status="finished", finished_at=now)
        if cron is not None:
            self.notify(cron, {"type": "cron_run", "cron": cron.id, "run": run["id"], "phase": "finished"})
            if changed:
                self.notify(cron, {"type": "issue", "cron": cron.id, "issues": sorted(set(changed))})

    # ============================================================================================
    # Reports
    # ============================================================================================
    def queue_report(self, cron: Cron, kind: str, start: dt.datetime, end: dt.datetime, created_by: str,
                     rollup: bool = False, template: str = "") -> str:
        spec = cron.period(kind)
        report_id = store.create_report(cron.id, kind, rollup, start, end, cron.timezone, start.astimezone(cron.tz).date(),
                                        template or (spec.template if spec else ""), created_by)
        self.harness._enqueue("report", report_id, cron.owner or (cron.principal, None))
        return report_id

    async def build_report(self, report_id: str) -> None:
        """The `report` job: the data, the Markdown, and a summary from the model."""
        row = store.report(report_id)
        if row is None or row["status"] != "pending":
            return
        cron = self.catalog.find(row["cron_id"])
        if cron is None:
            store.save_report(report_id, status="failed", error="the cron no longer exists")
            return
        try:
            spec = cron.period(row["kind"])
            if row["rollup"]:   # the database reads in a thread: the event loop serves everyone meanwhile
                data = await asyncio.to_thread(reports.rollup, cron, row["start_at"], row["end_at"],
                                               spec.rollup_of if spec else (cron.primary.name if cron.primary else None))
            else:
                effects = self._effects(cron, row["start_at"], row["end_at"])
                data = await asyncio.to_thread(reports.period, cron, row["start_at"], row["end_at"], effects)
            title = f"{cron.qualified}: {row['kind']} report"
            text = reports.markdown(cron, data, title, "")
            summary = await self._summary(cron, row.get("template") or "", text)
            store.save_report(report_id, status="ready", data=data, summary=summary,
                              markdown=reports.markdown(cron, data, title, summary), sites=reports.sites_of(data))
        except Exception as exc:
            LOG.exception("report %s failed", report_id)
            store.save_report(report_id, status="failed", error=f"{type(exc).__name__}: {exc}")
            raise
        self.notify(cron, {"type": "report", "cron": cron.id, "report": report_id, "kind": row["kind"]})

    def _effects(self, cron: Cron, start: dt.datetime, end: dt.datetime) -> list[dict]:
        """What the cron's tasks applied in the range (normally nothing in plan mode)."""
        found = []
        for run in store.runs(cron.id, limit=10_000, start=start, end=end):
            for task in self.harness.store.tasks_for_intent(run["intent_id"]) if run["intent_id"] else ():
                for effect in self.harness.store.effects_for_task(task.id):
                    if effect.status.value == "applied":
                        found.append({"tool": effect.tool, "target": effect.target, "status": effect.status.value,
                                      "task": task.id})
        return found

    async def _summary(self, cron: Cron, template_path: str, text: str) -> str:
        instructions = cron.files.get(template_path, "") if template_path else ""
        system = self.harness._prompt("cron_report") + (f"\n\n{instructions}" if instructions else "")
        owner = cron.owner or (cron.principal, None)
        request = LLMRequest(model=cron.model or self.harness.model, owner=owner, system=system,
                             messages=[Message("user", text)], purpose="cron_report", pinned=bool(cron.model),
                             min_tier=cron.tier)
        try:
            response = await self.harness.services.llm.complete(request)
        except LLMError as exc:
            LOG.warning("report summary unavailable for %s: %s", cron.id, exc)
            return ""
        return response.text.strip()

    # ============================================================================================
    # People
    # ============================================================================================
    def _require(self, cron_id: str, ctx: AuthContext, need: str = "operate") -> Cron:
        found = self.catalog.get(cron_id, ctx)
        order = {"read": 0, "view": 1, "operate": 2}
        if found is None or order[found[1]] < order[need]:
            raise PermissionError("not allowed" if found else "cron not found")
        return found[0]

    def act(self, ctx: AuthContext, issue_ids: list[str], action: str, comment: str = "",
            until: dt.datetime | None = None, ticket: str | None = None, related: str | None = None) -> list[dict]:
        """A check box on one or more issues: only by people who may operate their crons."""
        crons = tracker.cron_of(issue_ids)
        if len(crons) != len(set(issue_ids)):
            raise KeyError("issue not found")
        for cron_id in set(crons.values()):
            self._require(cron_id, ctx)
        changed = tracker.act(issue_ids, action, ctx.subject, self.clock(), comment, until, ticket, related)
        for cron_id in set(crons.values()):
            cron = self.catalog.find(cron_id)
            if cron:
                self.notify(cron, {"type": "issue", "cron": cron_id, "issues": [i["id"] for i in changed
                                                                                 if i["cron_id"] == cron_id]})
        return changed

    def suppress(self, ctx: AuthContext, cron_id: str, check: str | None, match: dict[str, str], reason: str,
                 until: dt.datetime | None) -> str:
        cron = self._require(cron_id, ctx)
        if check is not None and cron.check(check) is None:
            raise ValueError(f"no check {check!r}")
        if not match:
            raise ValueError("a suppression needs at least one key field")
        return store.add_suppression(cron_id, check, {k: str(v) for k, v in match.items()}, reason, ctx.subject, until)

    def unsuppress(self, ctx: AuthContext, cron_id: str, suppression_id: str) -> bool:
        self._require(cron_id, ctx)
        return store.delete_suppression(cron_id, suppression_id)

    async def decide(self, ctx: AuthContext, proposal_id: str, approve: bool, comment: str = "") -> dict:
        """Approve (it runs now, as `ctx`) or reject a proposed action."""
        proposal = store.proposal(proposal_id)
        if proposal is None:
            raise KeyError("proposal not found")
        self._require(proposal["cron_id"], ctx)
        now = self.clock()
        if not store.decide_proposal(proposal_id, "approved" if approve else "rejected", ctx.subject, now):
            raise ValueError("this proposal was already decided")
        result = None
        if approve:
            call = await self.harness.run_approved_call(ctx, proposal["tool"], proposal["arguments"] or {})
            failed = call.result is None or call.result.is_error
            result = {"status": call.status.value, "error": call.result.error_code if call.result else None,
                      "content": str(call.result.content)[:4000] if call.result else ""}
            store.proposal_result(proposal_id, "failed" if failed else "approved", result)
        if proposal["issue_id"]:
            tracker.note(proposal["issue_id"], ctx.subject, "approved" if approve else "rejected", now,
                         {"proposal": proposal_id, "tool": proposal["tool"], **({"result": result} if result else {})},
                         comment)
        return store.proposal(proposal_id)

    def run_now(self, ctx: AuthContext, cron_id: str, checks: list[str] | None = None) -> str | None:
        cron = self._require(cron_id, ctx)
        names = checks or [c.name for c in cron.checks if c.enabled]
        unknown = [n for n in names if cron.check(n) is None]
        if unknown:
            raise ValueError(f"no such check: {', '.join(unknown)}")
        return self.start_run(cron, names, by=ctx.subject)

    def set_enabled(self, ctx: AuthContext, cron_id: str, enabled: bool) -> dict:
        """Switch a cron on or off. A user's cron: its owner, who also hands it their scopes; a
        system or plugin cron: managers (admin scope)."""
        found = self.catalog.get(cron_id, ctx)
        if found is None:
            raise KeyError("cron not found")
        cron = found[0]
        if ctx.is_machine:
            raise PermissionError("machine tokens cannot switch crons")
        if cron.trusted:
            if ADMIN not in ctx.scopes:
                raise PermissionError("only managers (admin scope) switch system and plugin crons")
            store.ensure_state(cron.id, cron.source)
            store.set_state(cron.id, enabled=int(enabled))
        else:
            if cron.owner != owner_key(ctx):
                raise PermissionError("only the cron's owner switches it")
            store.ensure_state(cron.id, cron.source, cron.owner)
            store.set_state(cron.id, enabled=int(enabled), owner_scopes=sorted(ctx.scopes), last_error=None)
        return self.view(cron, found[1])

    def report_on_request(self, ctx: AuthContext, cron_id: str, start: dt.datetime, end: dt.datetime,
                          template: str = "") -> str:
        cron = self._require(cron_id, ctx, "view")
        if end <= start:
            raise ValueError("end must be after start")
        if template and template not in cron.files:
            raise ValueError(f"template {template!r} is not a file of the cron")
        return self.queue_report(cron, "custom", start.astimezone(UTC), end.astimezone(UTC), ctx.subject,
                                 rollup=True, template=template)

    # --- database crons --------------------------------------------------------------------------------
    def put_database_cron(self, ctx: AuthContext, name: str, files: dict[str, str], visibility: str,
                          enabled: bool | None, replace_files: bool = True) -> Cron:
        """Add or change one of `ctx`'s database crons; checked by parsing before it is stored."""
        if ctx.is_machine:
            raise PermissionError("machine tokens cannot add crons")
        if visibility == "all" and ADMIN not in ctx.scopes:
            raise PermissionError("only managers (admin scope) can share with everyone")
        owner = owner_key(ctx)
        cron_id = f"{scope_of(owner)}:{name}"
        current = next((c for c in self.catalog.database_crons(force=True) if c.id == cron_id), None)
        if not replace_files and current is not None:   # change some files, keep the others
            files = {**store_files(cron_id), **files}
        if len(files) > FILES_MAX or any(len(t.encode()) > FILE_MAX for t in files.values()):
            raise CronError(f"at most {FILES_MAX} files of {FILE_MAX // 1024} KB")
        for path in files:
            if path.startswith("/") or ".." in path.split("/") or any(p.startswith(".") for p in path.split("/")):
                raise CronError(f"invalid file path {path!r}")
        cron = parse_cron(name, files, "database", owner=owner, visibility=visibility)   # CronError: refused
        store.save_database_cron(cron_id, owner, files, visibility, sorted(ctx.scopes), enabled)
        self.catalog.database_crons(force=True)
        return cron

    def delete_database_cron(self, ctx: AuthContext, cron_id: str) -> bool:
        current = next((c for c in self.catalog.database_crons(force=True) if c.id == cron_id), None)
        if current is None or current.owner != owner_key(ctx):
            return False
        deleted = store.delete_cron(cron_id)
        self.catalog.database_crons(force=True)
        return deleted

    # --- views ---------------------------------------------------------------------------------------------
    def view(self, cron: Cron, cron_role: str, detail: bool = False) -> dict[str, Any]:
        state = store.state(cron.id) or {}
        view = cron.to_dict(detail=detail and cron_role != "read")
        view.update(role=cron_role, enabled=self.enabled(cron, state or None),
                    next_run_at=store.iso(state.get("next_run_at")), last_run_at=store.iso(state.get("last_run_at")),
                    last_error=state.get("last_error"), editable=cron.source == "database" and cron_role == "operate"
                    and cron.owner is not None)
        return view


def store_files(cron_id: str) -> dict[str, str]:
    for row, files in store.database_crons():
        if row["id"] == cron_id:
            return files
    return {}

