"""
When things happen: the times checks run (`schedule`, a check's `every` / `cron` / `at`) and the
boundaries of reporting periods (`periods`, `rollups`), in a cron's timezone (docs/cron.md §3, §8).

All times in and out are aware UTC datetimes; local time is used only to place boundaries.

    every: 30m | 8h | 1d | 1w | 1M     durations (s, m, h, d, w; M: calendar months, periods only)
    cron: "*/30 * * * *"                minute hour day-of-month month day-of-week (0 or 7 = Sunday)
    at: ["07:00", "15:00"]              daily, local
    at: "monday 08:00"                  weekly, local
    at: "1 08:00"                       monthly, local (the day is clipped to the month's length)
    clock: wall | elapsed               periods: local `at` times, or `every` from `anchor` (UTC)
"""

from __future__ import annotations

import calendar
import datetime as dt
import hashlib
import re
from dataclasses import dataclass
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = dt.timezone.utc
_DURATION = re.compile(r"^\s*(\d+)\s*([smhdwM])\s*$")
_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 7 * 86400}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


class ScheduleError(ValueError):
    pass


def zone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name or "UTC")
    except (ZoneInfoNotFoundError, ValueError) as exc:
        raise ScheduleError(f"unknown timezone {name!r}") from exc


def duration(text: str | int | float, months: bool = False) -> tuple[float, int]:
    """(seconds, months): "30m" -> (1800, 0); "1M" -> (0, 1) where `months` allows it."""
    if isinstance(text, (int, float)):
        return float(text), 0
    match = _DURATION.match(str(text))
    if not match:
        raise ScheduleError(f"invalid duration {text!r} (e.g. 30m, 8h, 1d, 1w)")
    n, unit = int(match.group(1)), match.group(2)
    if n <= 0:
        raise ScheduleError(f"duration must be positive: {text!r}")
    if unit == "M":
        if not months:
            raise ScheduleError("months (M) are allowed for report periods only")
        return 0.0, n
    return float(n * _SECONDS[unit]), 0


def seconds(text: str | int | float) -> float:
    return duration(text)[0]


def _hhmm(text: str) -> tuple[int, int]:
    match = re.match(r"^\s*(\d{1,2}):(\d{2})\s*$", str(text))
    if not match or int(match.group(1)) > 23 or int(match.group(2)) > 59:
        raise ScheduleError(f"invalid time {text!r} (HH:MM)")
    return int(match.group(1)), int(match.group(2))


def _local(day: dt.date, hour: int, minute: int, tz: ZoneInfo) -> dt.datetime:
    """A local wall time as UTC (a time that does not exist on a DST day moves forward with it)."""
    return dt.datetime(day.year, day.month, day.day, hour, minute, tzinfo=tz).astimezone(UTC)


# --- cron expressions -------------------------------------------------------------------------------
@dataclass(frozen=True)
class CronExpression:
    minutes: frozenset[int]
    hours: frozenset[int]
    days: frozenset[int]
    months: frozenset[int]
    weekdays: frozenset[int]          # 0 = Monday … 6 = Sunday (Python's)
    any_day: bool
    any_weekday: bool

    @classmethod
    def parse(cls, text: str) -> "CronExpression":
        fields = str(text).split()
        if len(fields) != 5:
            raise ScheduleError(f"a cron expression has 5 fields: {text!r}")

        def field(value: str, low: int, high: int) -> frozenset[int]:
            found: set[int] = set()
            for part in value.split(","):
                step = 1
                if "/" in part:
                    part, step_text = part.split("/", 1)
                    if not step_text.isdigit() or int(step_text) == 0:
                        raise ScheduleError(f"invalid step in {text!r}")
                    step = int(step_text)
                if part == "*":
                    start, end = low, high
                elif "-" in part:
                    a, b = part.split("-", 1)
                    if not (a.isdigit() and b.isdigit()):
                        raise ScheduleError(f"invalid range in {text!r}")
                    start, end = int(a), int(b)
                elif part.isdigit():
                    start = end = int(part)
                    if step != 1:
                        end = high
                else:
                    raise ScheduleError(f"invalid field {value!r} in {text!r}")
                if start < low or end > high or start > end:
                    raise ScheduleError(f"out of range: {value!r} in {text!r}")
                found.update(range(start, end + 1, step))
            return frozenset(found)

        minute, hour, day, month, weekday = fields
        cron_weekdays = field(weekday, 0, 7)
        return cls(field(minute, 0, 59), field(hour, 0, 23), field(day, 1, 31), field(month, 1, 12),
                   frozenset((d - 1) % 7 for d in cron_weekdays), day == "*", weekday == "*")

    def _day_ok(self, day: dt.date) -> bool:
        in_month, in_week = day.day in self.days, day.weekday() in self.weekdays
        if self.any_day and self.any_weekday:
            return True
        if self.any_day:
            return in_week
        if self.any_weekday:
            return in_month
        return in_month or in_week   # both restricted: either matches, as in cron

    def next_after(self, after: dt.datetime, tz: ZoneInfo) -> dt.datetime:
        local = after.astimezone(tz).replace(second=0, microsecond=0, tzinfo=None) + dt.timedelta(minutes=1)
        for _ in range(366 * 24 * 60):
            if local.month not in self.months:
                local = (local.replace(day=1, hour=0, minute=0) + dt.timedelta(days=32)).replace(day=1)
                continue
            if not self._day_ok(local.date()):
                local = (local + dt.timedelta(days=1)).replace(hour=0, minute=0)
                continue
            if local.hour not in self.hours:
                local = (local + dt.timedelta(hours=1)).replace(minute=0)
                continue
            if local.minute not in self.minutes:
                local += dt.timedelta(minutes=1)
                continue
            found = local.replace(tzinfo=tz).astimezone(UTC)
            if found > after:
                return found
            local += dt.timedelta(minutes=1)
        raise ScheduleError("the cron expression never matches")


# --- when checks run --------------------------------------------------------------------------------
@dataclass(frozen=True)
class Schedule:
    """`every`, `cron` or `at` (daily local times): the next time after a given one."""

    every: float | None = None
    cron: CronExpression | None = None
    at: tuple[tuple[int, int], ...] = ()

    @classmethod
    def parse(cls, data: dict | None, default: "Schedule | None" = None) -> "Schedule":
        data = data or {}
        given = [k for k in ("every", "cron", "at") if data.get(k) is not None]
        if not given:
            if default is None:
                raise ScheduleError("a schedule needs every, cron or at")
            return default
        if len(given) > 1:
            raise ScheduleError(f"use one of every, cron, at (not {', '.join(given)})")
        if "every" in given:
            every = seconds(data["every"])
            if every < 60:
                raise ScheduleError("checks run at most once a minute")
            return cls(every=every)
        if "cron" in given:
            return cls(cron=CronExpression.parse(data["cron"]))
        times = data["at"] if isinstance(data["at"], list) else [data["at"]]
        return cls(at=tuple(sorted(_hhmm(t) for t in times)))

    def next_after(self, after: dt.datetime, tz: ZoneInfo) -> dt.datetime:
        if self.every:   # aligned to the epoch: every 30m runs at :00 and :30
            epoch = after.timestamp()
            return dt.datetime.fromtimestamp((epoch // self.every + 1) * self.every, UTC)
        if self.cron:
            return self.cron.next_after(after, tz)
        day = after.astimezone(tz).date()
        for offset in range(3):
            for hour, minute in self.at:
                found = _local(day + dt.timedelta(days=offset), hour, minute, tz)
                if found > after:
                    return found
        raise ScheduleError("no next time")   # unreachable with at least one time

    def interval(self) -> float:
        """Roughly how far apart runs are (for catch-up decisions)."""
        if self.every:
            return self.every
        if self.at:
            return 86400 / len(self.at)
        return 60.0


def jitter_offset(key: str, jitter: float) -> float:
    """A fixed offset in [0, jitter) for `key`: crons spread out, and the same cron always the same."""
    if jitter <= 0:
        return 0.0
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % int(jitter * 1000) / 1000


# --- reporting periods ------------------------------------------------------------------------------
@dataclass(frozen=True)
class PeriodClock:
    """The boundaries of one kind of period: `start(t)` is the boundary at or before t, `next(t)`
    the first after it, so a period is [start(t), next(t))."""

    clock: str = "wall"                                 # wall | elapsed
    daily: tuple[tuple[int, int], ...] = ()             # wall: HH:MM every day
    weekday: int | None = None                          # wall: weekly on this day (0 = Monday)
    monthday: int | None = None                         # wall: monthly on this day
    time: tuple[int, int] = (0, 0)                      # wall weekly / monthly: at this time
    every: float = 0.0                                  # elapsed: seconds
    months: int = 0                                     # wall monthly: every n months (1)
    anchor: dt.datetime | None = None                   # elapsed: a boundary

    @classmethod
    def parse(cls, data: dict) -> "PeriodClock":
        clock = str(data.get("clock") or "wall")
        if clock not in ("wall", "elapsed"):
            raise ScheduleError("clock must be wall or elapsed")
        if clock == "elapsed":
            every = seconds(data.get("every") or "")
            anchor_text = str(data.get("anchor") or "1970-01-01T00:00:00Z").replace("Z", "+00:00")
            try:
                anchor = dt.datetime.fromisoformat(anchor_text)
            except ValueError as exc:
                raise ScheduleError(f"invalid anchor {data.get('anchor')!r}") from exc
            anchor = anchor if anchor.tzinfo else anchor.replace(tzinfo=UTC)
            return cls(clock="elapsed", every=every, anchor=anchor.astimezone(UTC))
        at = data.get("at")
        if isinstance(at, list):
            return cls(daily=tuple(sorted(_hhmm(t) for t in at)))
        if not isinstance(at, str) or not at.strip():
            raise ScheduleError("a period needs at (wall clock) or clock: elapsed with every")
        first, _, rest = at.strip().partition(" ")
        if first.lower() in _WEEKDAYS:
            return cls(weekday=_WEEKDAYS.index(first.lower()), time=_hhmm(rest or "00:00"))
        if first.isdigit() and rest:
            day = int(first)
            if not 1 <= day <= 31:
                raise ScheduleError(f"invalid day of month in {at!r}")
            months = duration(data.get("every") or "1M", months=True)[1] or 1
            return cls(monthday=day, time=_hhmm(rest), months=months)
        return cls(daily=(_hhmm(at),))

    def _candidates(self, around: dt.datetime, tz: ZoneInfo) -> list[dt.datetime]:
        day = around.astimezone(tz).date()
        found: list[dt.datetime] = []
        if self.daily:
            for offset in range(-2, 3):
                found += [_local(day + dt.timedelta(days=offset), h, m, tz) for h, m in self.daily]
        elif self.weekday is not None:
            monday = day - dt.timedelta(days=day.weekday())
            for week in range(-2, 3):
                found.append(_local(monday + dt.timedelta(days=7 * week + self.weekday), *self.time, tz))
        else:
            for offset in range(-2 * self.months - 1, 2 * self.months + 2):
                month_index = day.year * 12 + day.month - 1 + offset
                if month_index % self.months:
                    continue
                year, month = divmod(month_index, 12)
                last = calendar.monthrange(year, month + 1)[1]
                found.append(_local(dt.date(year, month + 1, min(self.monthday, last)), *self.time, tz))
        return sorted(found)

    def start(self, t: dt.datetime, tz: ZoneInfo) -> dt.datetime:
        if self.clock == "elapsed":
            n = (t - self.anchor).total_seconds() // self.every
            return self.anchor + dt.timedelta(seconds=n * self.every)
        return max(c for c in self._candidates(t, tz) if c <= t)

    def next(self, t: dt.datetime, tz: ZoneInfo) -> dt.datetime:
        if self.clock == "elapsed":
            return self.start(t, tz) + dt.timedelta(seconds=self.every)
        return min(c for c in self._candidates(t, tz) if c > t)

    def previous(self, t: dt.datetime, tz: ZoneInfo) -> tuple[dt.datetime, dt.datetime]:
        """The last complete period before `t`: (start, end), end = start(t)."""
        end = self.start(t, tz)
        return self.start(end - dt.timedelta(seconds=1), tz), end


def now() -> dt.datetime:
    return dt.datetime.now(UTC)
