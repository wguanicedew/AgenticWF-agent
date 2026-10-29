"""
The crons' API (docs/cron.md §10): crons, runs, the issue board, suppressions, proposals, reports,
metrics, and the page. Every route checks the caller's role on the cron (agenticwf.agent.runtime.cron.catalog):
`read` (reports and the board), `view` (also runs and the definition), `operate` (check boxes).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from agenticwf.auth.context import AuthContext

from agenticwf.agent.core.common.logging import get_logger
from agenticwf.agent.runtime.cron import store, tracker
from agenticwf.agent.runtime.cron.definition import CRON_FILE, CronError
from agenticwf.agent.runtime.cron.schedule import UTC
from agenticwf.agent.runtime.cron.service import CronService, store_files
from agenticwf.agent.runtime.cron.tracker import TrackerError
from agenticwf.agent.runtime.types import owner_key, scope_of, split_ref

LOG = get_logger(__name__)
ORDER = {"read": 0, "view": 1, "operate": 2}


class CronIn(BaseModel):
    """A database cron: CRON.md, and its other files (checks/<check>.md, reports/<x>.md)."""

    name: str = Field(min_length=1, max_length=64)
    content: str = Field(min_length=1, max_length=256 * 1024)     # CRON.md
    files: dict[str, str] = Field(default_factory=dict)
    visibility: Literal["user", "organization", "all"] = "user"   # all: managers only
    enabled: bool = True


class CronPatch(BaseModel):
    content: str | None = Field(default=None, min_length=1, max_length=256 * 1024)
    visibility: Literal["user", "organization", "all"] | None = None
    enabled: bool | None = None


class FileIn(BaseModel):
    content: str = Field(max_length=256 * 1024)


class Switch(BaseModel):
    enabled: bool


class RunIn(BaseModel):
    checks: list[str] = Field(default_factory=list)


class IssueAction(BaseModel):
    action: Literal["acknowledge", "take", "solve", "ignore", "snooze", "reopen", "ticket", "comment", "relate",
                    "unrelate"]
    comment: str = Field(default="", max_length=4000)
    until: dt.datetime | None = None
    ticket: str | None = Field(default=None, max_length=512)
    related: str | None = Field(default=None, max_length=32)


class BulkAction(IssueAction):
    ids: list[str] = Field(min_length=1, max_length=500)


class SuppressionIn(BaseModel):
    check: str | None = None
    match: dict[str, str] = Field(min_length=1)
    reason: str = Field(default="", max_length=2000)
    until: dt.datetime | None = None


class Decision(BaseModel):
    approve: bool
    comment: str = Field(default="", max_length=4000)


class ReportIn(BaseModel):
    start: dt.datetime = Field(alias="from")
    end: dt.datetime = Field(alias="to")
    template: str = ""


def _utc(value: dt.datetime | None) -> dt.datetime | None:
    if value is None:
        return None
    return (value if value.tzinfo else value.replace(tzinfo=UTC)).astimezone(UTC)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dt.datetime):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    return value


def register(app: FastAPI, service: CronService | None, tasks_dep, people, page: str, signed_in) -> None:
    def crons() -> CronService:
        if service is None:
            raise HTTPException(status_code=409, detail="this runtime runs no crons")
        return service

    def cron_id(ref: str, ctx: AuthContext) -> str:
        scope, name = split_ref(ref, owner_key(ctx))
        return f"{scope}:{name}"

    def need(ref_or_id: str, ctx: AuthContext, level: str = "read"):
        found = crons().catalog.get(cron_id(ref_or_id, ctx), ctx)
        if found is None:
            raise HTTPException(status_code=404, detail="cron not found")
        if ORDER[found[1]] < ORDER[level]:
            raise HTTPException(status_code=403, detail=f"you may not {'change' if level == 'operate' else 'see'} this")
        return found

    def visible_ids(ctx: AuthContext) -> list[str]:
        return [c.id for c, _ in crons().catalog.visible(ctx)]

    def call(fn, *args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'"))
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except (CronError, TrackerError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=str(exc))

    @app.get("/crons", response_class=HTMLResponse)
    def crons_page(ctx: AuthContext = Depends(signed_in)) -> str:
        """Crons, the issue board, and reports (docs/cron.md §10)."""
        return page

    # --- crons ---------------------------------------------------------------------------------
    @app.get("/api/v1/crons")
    def list_crons(ctx: AuthContext = Depends(tasks_dep)) -> dict:
        service_ = crons()
        found = [service_.view(c, r) for c, r in service_.catalog.visible(ctx)]
        scope = scope_of(owner_key(ctx))
        own_errors = {i: e for i, e in service_.catalog.database_errors.items() if i.rpartition(":")[0] == scope}
        return {"crons": found, "can_add": not ctx.is_machine, "errors": own_errors}

    @app.get("/api/v1/crons/{ref}")
    def get_cron(ref: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, role = need(ref, ctx)
        view = crons().view(cron, role, detail=True)
        if cron.source == "database" and role == "operate":
            view["content"] = store_files(cron.id).get(CRON_FILE, "")
        return view

    @app.get("/api/v1/crons/{ref}/files/{path:path}")
    def get_cron_file(ref: str, path: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """One file of the cron's definition (CRON.md, checks/<check>.md, reports/<x>.md), as written."""
        cron, _ = need(ref, ctx, "view")
        if cron.source == "database":
            content = store_files(cron.id).get(path)
        else:
            root = Path(cron.location).resolve() if cron.location else None
            target = (root / path).resolve() if root else None
            inside = target is not None and root in target.parents and not any(
                part.startswith(".") for part in target.relative_to(root).parts)
            content = target.read_text() if inside and target.is_file() else None
        if content is None:
            raise HTTPException(status_code=404, detail="file not found")
        return {"path": path, "content": content}

    @app.post("/api/v1/crons", status_code=201)
    def add_cron(body: CronIn, ctx: AuthContext = Depends(people)) -> dict:
        """A database cron of yours: its id is `<your scope>:<name>`; it runs as you, with your scopes."""
        service_ = crons()
        if any(c.id == cron_id(body.name, ctx) for c in service_.catalog.database_crons(force=True)):
            raise HTTPException(status_code=409, detail=f"you already have a cron named {body.name!r}")
        cron = call(service_.put_database_cron, ctx, body.name, {CRON_FILE: body.content, **body.files},
                    body.visibility, body.enabled)
        LOG.info("cron %s added by %s", cron.id, ctx.subject)
        return service_.view(cron, "operate", detail=True)

    def own_database_cron(ref: str, ctx: AuthContext):
        cron, role = need(ref, ctx, "operate")
        if cron.source != "database" or cron.owner != owner_key(ctx):
            raise HTTPException(status_code=403, detail="only your own crons added here can be changed here")
        return cron

    @app.patch("/api/v1/crons/{ref}")
    def change_cron(ref: str, body: CronPatch, ctx: AuthContext = Depends(people)) -> dict:
        cron = own_database_cron(ref, ctx)
        state = store.state(cron.id) or {}
        files = {CRON_FILE: body.content} if body.content is not None else {}
        enabled = body.enabled if body.enabled is not None else (None if state.get("enabled") is None else bool(state["enabled"]))
        updated = call(crons().put_database_cron, ctx, cron.name, files, body.visibility or cron.visibility or "user",
                       enabled, replace_files=False)
        return crons().view(updated, "operate", detail=True)

    @app.delete("/api/v1/crons/{ref}", status_code=204)
    def remove_cron(ref: str, ctx: AuthContext = Depends(people)) -> Response:
        cron = own_database_cron(ref, ctx)
        crons().delete_database_cron(ctx, cron.id)
        LOG.info("cron %s removed by %s", cron.id, ctx.subject)
        return Response(status_code=204)

    @app.put("/api/v1/crons/{ref}/files/{path:path}")
    def put_cron_file(ref: str, path: str, body: FileIn, ctx: AuthContext = Depends(people)) -> dict:
        cron = own_database_cron(ref, ctx)
        state = store.state(cron.id) or {}
        updated = call(crons().put_database_cron, ctx, cron.name, {path: body.content}, cron.visibility or "user",
                       None if state.get("enabled") is None else bool(state["enabled"]), replace_files=False)
        return crons().view(updated, "operate", detail=True)

    @app.delete("/api/v1/crons/{ref}/files/{path:path}", status_code=204)
    def delete_cron_file(ref: str, path: str, ctx: AuthContext = Depends(people)) -> Response:
        cron = own_database_cron(ref, ctx)
        files = store_files(cron.id)
        if path == CRON_FILE or path not in files:
            raise HTTPException(status_code=404 if path not in files else 422, detail="file not found" if path not in files
                                else f"{CRON_FILE} cannot be removed")
        files.pop(path)
        state = store.state(cron.id) or {}
        call(crons().put_database_cron, ctx, cron.name, files, cron.visibility or "user",
             None if state.get("enabled") is None else bool(state["enabled"]))
        return Response(status_code=204)

    @app.post("/api/v1/crons/{ref}/state")
    def switch_cron(ref: str, body: Switch, ctx: AuthContext = Depends(people)) -> dict:
        """On or off. Your own cron (it then runs with your current scopes); a system cron: managers."""
        return call(crons().set_enabled, ctx, cron_id(ref, ctx), body.enabled)

    @app.post("/api/v1/crons/{ref}/run", status_code=202)
    async def run_cron(ref: str, body: RunIn, ctx: AuthContext = Depends(people)) -> dict:
        cron, _ = need(ref, ctx, "operate")
        run_id = call(crons().run_now, ctx, cron.id, body.checks or None)
        if run_id is None:
            raise HTTPException(status_code=409, detail=(store.state(cron.id) or {}).get("last_error") or "not started")
        return _jsonable(store.run(run_id))

    @app.get("/api/v1/crons/{ref}/runs")
    def list_runs(ref: str, limit: int = Query(50, ge=1, le=500), ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, _ = need(ref, ctx, "view")
        return {"runs": _jsonable(store.runs(cron.id, limit=limit))}

    @app.get("/api/v1/crons/{ref}/runs/{run_id}")
    def get_run(ref: str, run_id: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, _ = need(ref, ctx, "view")
        found = store.run(run_id)
        if found is None or found["cron_id"] != cron.id:
            raise HTTPException(status_code=404, detail="run not found")
        return _jsonable(found)

    # --- the issue board -------------------------------------------------------------------------
    @app.get("/api/v1/issues")
    def list_issues(cron: str | None = None, site: str | None = None, check: str | None = None,
                    status: str | None = None, severity: str | None = None, owner: str | None = None,
                    since: dt.datetime | None = None, recurring: bool = False, related: bool = False,
                    group: Literal["", "site"] = "", limit: int = Query(500, ge=1, le=5000),
                    ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """The board: issues of the crons you may see, most severe first. `status`: comma separated
        (default: the open ones and snoozed). `truncated`: there are more than `limit`."""
        ids = [need(cron, ctx)[0].id] if cron else visible_ids(ctx)
        statuses = status.split(",") if status else list(tracker.FOLLOWED) + ["snoozed"]
        found = tracker.issues(ids, statuses, check, site, severity, owner, _utc(since), recurring, related,
                               limit=limit + 1)
        truncated = len(found) > limit
        found = found[:limit]
        if group == "site":
            return {"sites": tracker.by_site(found), "truncated": truncated}
        return {"issues": found, "truncated": truncated}

    @app.get("/api/v1/issues/{issue_id}")
    def get_issue(issue_id: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        found = tracker.issue(issue_id)
        if found is None:
            raise HTTPException(status_code=404, detail="issue not found")
        _, role = need(found["cron_id"], ctx)
        return {**found, "can_operate": role == "operate"}

    @app.post("/api/v1/issues/{issue_id}/actions")
    def act_on_issue(issue_id: str, body: IssueAction, ctx: AuthContext = Depends(people)) -> dict:
        return {"issues": call(crons().act, ctx, [issue_id], body.action, body.comment, _utc(body.until), body.ticket,
                               body.related)}

    @app.post("/api/v1/issues/actions")
    def act_on_issues(body: BulkAction, ctx: AuthContext = Depends(people)) -> dict:
        return {"issues": call(crons().act, ctx, body.ids, body.action, body.comment, _utc(body.until), body.ticket,
                               body.related)}

    @app.get("/api/v1/crons/{ref}/suppressions")
    def list_suppressions(ref: str, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, _ = need(ref, ctx)
        return {"suppressions": _jsonable(store.suppressions(cron.id))}

    @app.post("/api/v1/crons/{ref}/suppressions", status_code=201)
    def add_suppression(ref: str, body: SuppressionIn, ctx: AuthContext = Depends(people)) -> dict:
        cron, _ = need(ref, ctx, "operate")
        suppression_id = call(crons().suppress, ctx, cron.id, body.check, body.match, body.reason, _utc(body.until))
        return {"id": suppression_id}

    @app.delete("/api/v1/crons/{ref}/suppressions/{suppression_id}", status_code=204)
    def remove_suppression(ref: str, suppression_id: str, ctx: AuthContext = Depends(people)) -> Response:
        cron, _ = need(ref, ctx, "operate")
        if not call(crons().unsuppress, ctx, cron.id, suppression_id):
            raise HTTPException(status_code=404, detail="suppression not found")
        return Response(status_code=204)

    @app.get("/api/v1/crons/{ref}/proposals")
    def list_proposals(ref: str, status: str | None = None, ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, _ = need(ref, ctx)
        return {"proposals": _jsonable(store.proposals(cron.id, status=status))}

    @app.post("/api/v1/proposals/{proposal_id}/decision")
    async def decide_proposal(proposal_id: str, body: Decision, ctx: AuthContext = Depends(people)) -> dict:
        """Approve (the action runs now, as you, under the normal policy) or reject a proposed action."""
        try:
            return _jsonable(await crons().decide(ctx, proposal_id, body.approve, body.comment))
        except KeyError:
            raise HTTPException(status_code=404, detail="proposal not found")
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc))

    # --- reports and metrics -------------------------------------------------------------------------
    @app.get("/api/v1/reports")
    def list_reports(cron: str | None = None, kind: str | None = None, year: int | None = None,
                     month: int | None = Query(None, ge=1, le=12), day: int | None = Query(None, ge=1, le=31),
                     site: str | None = None, q: str | None = None, limit: int = Query(100, ge=1, le=500),
                     offset: int = Query(0, ge=0), ctx: AuthContext = Depends(tasks_dep)) -> dict:
        """The reports page: reports of the crons you may see, newest first, by local date."""
        ids = [need(cron, ctx)[0].id] if cron else visible_ids(ctx)
        found, total = store.reports(ids, kind, year, month, day, site, q, limit=limit, offset=offset)
        return {"reports": _jsonable(found), "total": total}

    @app.get("/api/v1/reports/{report_id}")
    def get_report(report_id: str, format: Literal["json", "markdown"] = "json",
                   ctx: AuthContext = Depends(tasks_dep)):
        found = store.report(report_id)
        if found is None:
            raise HTTPException(status_code=404, detail="report not found")
        _, role = need(found["cron_id"], ctx)
        if format == "markdown":
            return PlainTextResponse(found["markdown"] or "", media_type="text/markdown; charset=utf-8")
        previous, following = store.report_neighbours(found)
        return {**_jsonable(found), "previous": previous, "next": following, "can_operate": role == "operate"}

    @app.post("/api/v1/crons/{ref}/reports", status_code=202)
    async def request_report(ref: str, body: ReportIn, ctx: AuthContext = Depends(people)) -> dict:
        cron, _ = need(ref, ctx, "view")
        report_id = call(crons().report_on_request, ctx, cron.id, _utc(body.start), _utc(body.end), body.template)
        return {"id": report_id, "status": "pending"}

    @app.get("/api/v1/crons/{ref}/metrics")
    def metrics(ref: str, name: str | None = None, site: str | None = None, start: dt.datetime | None = Query(None, alias="from"),
                end: dt.datetime | None = Query(None, alias="to"), step: Literal["hour", "day"] = "hour",
                ctx: AuthContext = Depends(tasks_dep)) -> dict:
        cron, _ = need(ref, ctx)
        end_ = _utc(end) or crons().clock()
        start_ = _utc(start) or end_ - dt.timedelta(days=1 if step == "hour" else 30)
        return {"series": store.metric_series(cron.id, name, start_, end_, step, site), "step": step,
                "metrics": sorted({m.name for c in cron.checks for m in c.metrics})}
