"""
A cron's definition: CRON.md, checks/*.md and reports/*.md (docs/cron.md §2), read from a
directory or from files held in the database (the same {path: text} either way).
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agenticwf.agent.runtime.cron.schedule import PeriodClock, Schedule, ScheduleError, seconds, zone
from agenticwf.agent.runtime.tools.loader import ToolDefinitionError, split_front_matter
from agenticwf.agent.runtime.tools.spec import VISIBILITY, Access
from agenticwf.agent.runtime.types import Mode, Owner, scope_of

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
CRON_FILE = "CRON.md"
SOURCES = ("system", "plugin", "user", "database")
UNATTENDED = ("propose", "deny")
CHECK_STATUSES = ("ok", "issues", "degraded", "unable")
FILE_MAX = 256 * 1024
FILES_MAX = 100


class CronError(Exception):
    pass


@dataclass(frozen=True)
class MetricSpec:
    name: str
    unit: str = ""
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    name: str
    description: str
    instructions: str
    schedule: Schedule | None = None          # None: the cron's
    timeout: float = 900.0
    tools: tuple[str, ...] | None = None      # globs; None: the task type's
    skills: tuple[str, ...] | None = None
    key: tuple[str, ...] = ()
    severities: tuple[str, ...] = ("high", "medium", "low")   # first = most severe
    resolve_after: int | None = None
    metrics: tuple[MetricSpec, ...] = ()
    enabled: bool = True

    def metric(self, name: str) -> MetricSpec | None:
        return next((m for m in self.metrics if m.name == name), None)

    def rank(self, severity: str) -> int:
        """0 = the most severe."""
        return self.severities.index(severity) if severity in self.severities else len(self.severities)


@dataclass(frozen=True)
class PeriodSpec:
    name: str
    clock: PeriodClock
    template: str = ""                 # a file of the cron (reports/period.md); "" = the packaged prompt
    rollup_of: str | None = None       # rollups: the period whose reports they gather


@dataclass(frozen=True)
class Cron:
    name: str
    description: str
    instructions: str
    source: str
    owner: Owner | None = None
    plugin: str | None = None
    timezone: str = "UTC"
    schedule: Schedule = Schedule(every=1800.0)
    jitter: float = 0.0
    catch_up: str = "once"
    periods: tuple[PeriodSpec, ...] = ()
    rollups: tuple[PeriodSpec, ...] = ()
    group_by: str = "site"
    group_default: str = "central-services"
    task_type: str = "default"
    model: str | None = None
    tier: str | None = None            # its checks need at least this model tier (docs/model-routing.md)
    mode: Mode = Mode.PLAN
    unattended: str = "propose"
    resolve_after: int = 2
    access: Access = Access()
    operators: Access | None = None
    readers: Access | None = None
    scopes: frozenset[str] = frozenset()
    retention: float = 400 * 86400.0
    metrics_raw_retention: float = 30 * 86400.0
    visibility: str = ""               # database crons: user | organization | all
    enabled: bool = True
    checks: tuple[Check, ...] = ()
    files: dict[str, str] = field(default_factory=dict, compare=False, hash=False)   # reports/*.md and others
    location: str = ""

    # --- identity --------------------------------------------------------------------------------
    @property
    def trusted(self) -> bool:
        return self.source in ("system", "plugin")

    @property
    def scope(self) -> str:
        return scope_of(self.owner)

    @property
    def qualified(self) -> str:
        return f"{self.plugin}.{self.name}" if self.plugin else self.name

    @property
    def id(self) -> str:
        return f"{self.scope}:{self.qualified}"

    @property
    def tz(self):
        return zone(self.timezone)

    @property
    def principal(self) -> str:
        """Who a system or plugin cron runs as."""
        return f"cron:{self.qualified}"

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    def period(self, name: str) -> PeriodSpec | None:
        return next((p for p in (*self.periods, *self.rollups) if p.name == name), None)

    @property
    def primary(self) -> PeriodSpec | None:
        return self.periods[0] if self.periods else None

    def check_schedule(self, check: Check) -> Schedule:
        return check.schedule or self.schedule

    def resolve_after_for(self, check: Check) -> int:
        return check.resolve_after or self.resolve_after

    def template(self, period: PeriodSpec) -> str:
        return self.files.get(period.template, "") if period.template else ""

    def to_dict(self, detail: bool = False) -> dict[str, Any]:
        view = {"id": self.id, "name": self.name, "qualified": self.qualified, "description": self.description,
                "source": self.source, "plugin": self.plugin, "owner": self.owner[0] if self.owner else None,
                "timezone": self.timezone, "group_by": self.group_by, "mode": self.mode.value,
                "unattended": self.unattended, "visibility": self.visibility or ("all" if self.trusted else "user"),
                "enabled": self.enabled,
                "periods": [p.name for p in self.periods], "rollups": [p.name for p in self.rollups],
                "checks": [{"name": c.name, "description": c.description, "enabled": c.enabled,
                            "key": list(c.key), "severities": list(c.severities),
                            "metrics": [dataclasses.asdict(m) for m in c.metrics]} for c in self.checks]}
        if detail:
            view.update(instructions=self.instructions, files=sorted(self.files),
                        check_instructions={c.name: c.instructions for c in self.checks})
        return view


# --- parsing ----------------------------------------------------------------------------------------
def _strings(value: Any, what: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return tuple(v for v in re.split(r"[\s,]+", value) if v)
    if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value):
        return tuple(value)
    raise CronError(f"{what} must be a list of names")


def _access(value: Any, what: str) -> Access:
    try:
        return Access.from_mapping(value)
    except ValueError as exc:
        raise CronError(f"{what}: {exc}") from exc


def _front(path: str, text: str) -> tuple[dict, str]:
    try:
        return split_front_matter(text)
    except (ToolDefinitionError, yaml.YAMLError) as exc:
        raise CronError(f"{path}: {exc}") from exc


def _tier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    if value not in ("small", "medium", "large"):
        raise CronError("tier must be small, medium or large")
    return str(value)


def _period(data: Any, rollup: bool, files: dict[str, str]) -> PeriodSpec:
    if not isinstance(data, dict) or not isinstance(data.get("name"), str) or not NAME_RE.match(data["name"]):
        raise CronError("every period needs a name (lowercase letters, digits, hyphens)")
    template = str(data.get("template") or "")
    if template and template not in files:
        raise CronError(f"period {data['name']}: template {template!r} is not a file of the cron")
    try:
        clock = PeriodClock.parse(data)
    except ScheduleError as exc:
        raise CronError(f"period {data['name']}: {exc}") from exc
    rollup_of = str(data.get("from") or "") or None
    if rollup and not rollup_of:
        raise CronError(f"rollup {data['name']}: from (a period's name) is required")
    return PeriodSpec(data["name"], clock, template, rollup_of if rollup else None)


def _check(path: str, text: str, group_by: str) -> Check:
    meta, body = _front(path, text)
    name = Path(path).stem
    if meta.get("name", name) != name or not NAME_RE.match(name):
        raise CronError(f"{path}: the check's name must be the file's name (lowercase letters, digits, hyphens)")
    issue = meta.get("issue") or {}
    if not isinstance(issue, dict):
        raise CronError(f"{path}: issue must be a mapping")
    key = _strings(issue.get("key"), "issue.key") or ("title",)
    if group_by not in key:
        key = (group_by, *key)
    severities = _strings(issue.get("severity"), "issue.severity") or ("high", "medium", "low")
    metrics = []
    for item in meta.get("metrics") or ():
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise CronError(f"{path}: every metric needs a name")
        metrics.append(MetricSpec(item["name"], str(item.get("unit") or ""),
                                  _strings(item.get("labels"), "metric labels") or ()))
    try:
        schedule = Schedule.parse({k: meta.get(k) for k in ("every", "cron", "at")}, default=None) \
            if any(meta.get(k) is not None for k in ("every", "cron", "at")) else None
        timeout = seconds(meta.get("timeout") or "15m")
    except ScheduleError as exc:
        raise CronError(f"{path}: {exc}") from exc
    resolve_after = issue.get("resolve_after", meta.get("resolve_after"))
    return Check(name=name, description=str(meta.get("description") or "").strip(), instructions=body,
                 schedule=schedule, timeout=timeout, tools=_strings(meta.get("tools"), "tools"),
                 skills=_strings(meta.get("skills"), "skills"), key=key, severities=severities,
                 resolve_after=int(resolve_after) if resolve_after else None, metrics=tuple(metrics),
                 enabled=meta.get("enabled", True) is not False)


def parse_cron(name: str, files: dict[str, str], source: str, owner: Owner | None = None,
               plugin: str | None = None, inherited: Access = Access(), visibility: str = "",
               location: str = "") -> Cron:
    """A cron from its files ({relative path: text}). CronError: unusable. Settings that grant
    (access, operators, readers, scopes, auto mode) count for system and plugin crons only."""
    if not NAME_RE.match(name) or len(name) > 64:
        raise CronError(f"invalid cron name {name!r} (lowercase letters, digits and single hyphens)")
    if CRON_FILE not in files:
        raise CronError(f"{name}: {CRON_FILE} is missing")
    meta, body = _front(CRON_FILE, files[CRON_FILE])
    if meta.get("name", name) != name:
        raise CronError(f"{CRON_FILE}: name {meta.get('name')!r} does not match {name!r}")
    trusted = source in ("system", "plugin") and owner is None
    group = meta.get("group") or {}
    if not isinstance(group, dict):
        raise CronError("group must be a mapping (by, default)")
    group_by = str(group.get("by") or "site")
    schedule_data = meta.get("schedule") or {"every": "30m"}
    if not isinstance(schedule_data, dict):
        raise CronError("schedule must be a mapping (every, cron or at)")
    try:
        schedule = Schedule.parse(schedule_data)
        jitter = seconds(schedule_data["jitter"]) if schedule_data.get("jitter") else 0.0
        timezone = str(meta.get("timezone") or "UTC")
        zone(timezone)
        retention = seconds(meta.get("retention") or "400d")
        raw_retention = seconds(meta.get("metrics_raw_retention") or "30d")
    except ScheduleError as exc:
        raise CronError(str(exc)) from exc
    catch_up = str(schedule_data.get("catch_up") or "once")
    if catch_up not in ("once", "none"):
        raise CronError("schedule.catch_up must be once or none")
    periods = tuple(_period(p, False, files) for p in meta.get("periods") or ())
    rollups = tuple(_period(p, True, files) for p in meta.get("rollups") or ())
    names = [p.name for p in (*periods, *rollups)]
    if len(set(names)) != len(names):
        raise CronError("period and rollup names must be unique")
    for rollup in rollups:
        if rollup.rollup_of not in {p.name for p in periods}:
            raise CronError(f"rollup {rollup.name}: from {rollup.rollup_of!r} is not a period")
    unattended = str(meta.get("unattended") or "propose")
    if unattended not in UNATTENDED:
        raise CronError("unattended must be propose or deny")
    try:
        mode = Mode(str(meta.get("mode") or "plan"))
    except ValueError as exc:
        raise CronError("mode must be plan, default or auto") from exc
    if mode is Mode.AUTO and not trusted:
        mode = Mode.DEFAULT   # auto mode is for operators' crons only, and the policy decides even then
    checks = []
    for path in sorted(p for p in files if p.startswith("checks/") and p.endswith(".md") and p.count("/") == 1):
        checks.append(_check(path, files[path], group_by))
    if not checks:
        raise CronError(f"{name}: no checks (checks/<check>.md)")
    access = inherited.merge(_access(meta.get("access"), "access")) if trusted else Access()
    if visibility and visibility not in VISIBILITY:
        raise CronError(f"visibility must be one of {', '.join(VISIBILITY)}")
    return Cron(
        name=name, description=str(meta.get("description") or "").strip(), instructions=body, source=source,
        owner=owner, plugin=plugin, timezone=timezone, schedule=schedule, jitter=jitter, catch_up=catch_up,
        periods=periods, rollups=rollups, group_by=group_by,
        group_default=str(group.get("default") or "central-services"),
        task_type=str(meta.get("task_type") or "default"), model=meta.get("model") or None,
        tier=_tier(meta.get("tier")), mode=mode,
        unattended=unattended, resolve_after=int(meta.get("resolve_after") or 2), access=access,
        operators=_access(meta.get("operators"), "operators") if trusted and meta.get("operators") else None,
        readers=_access(meta.get("readers"), "readers") if trusted and meta.get("readers") else None,
        scopes=frozenset(_strings(meta.get("scopes"), "scopes") or ()) if trusted else frozenset(),
        retention=retention, metrics_raw_retention=raw_retention, visibility=visibility,
        enabled=meta.get("enabled", True) is not False, checks=tuple(checks),
        files={p: t for p, t in files.items() if p != CRON_FILE and not p.startswith("checks/")},
        location=location)


def read_cron_dir(directory: Path, source: str, owner: Owner | None = None, plugin: str | None = None,
                  inherited: Access = Access()) -> Cron:
    files: dict[str, str] = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if path.is_file() and not any(p.startswith(".") for p in relative.parts):
            if path.stat().st_size > FILE_MAX:
                raise CronError(f"{path}: larger than {FILE_MAX // 1024} KB")
            try:
                files[relative.as_posix()] = path.read_text()
            except (OSError, UnicodeDecodeError) as exc:
                raise CronError(f"{path}: {exc}") from exc
    try:
        return parse_cron(directory.name, files, source, owner, plugin, inherited, location=str(directory))
    except CronError as exc:
        raise CronError(f"{directory}: {exc}") from exc


def read_crons_dir(root: Path, source: str, owner: Owner | None = None, plugin: str | None = None,
                   inherited: Access = Access()) -> tuple[list[Cron], list[str]]:
    crons, errors = [], []
    if not root.is_dir():
        return crons, errors
    for directory in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        if not (directory / CRON_FILE).is_file():
            continue
        try:
            crons.append(read_cron_dir(directory, source, owner, plugin, inherited))
        except CronError as exc:
            errors.append(str(exc))
    return crons, errors
