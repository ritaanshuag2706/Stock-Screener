"""Trading calendar with data-backed overrides.

Wraps a base exchange calendar (pandas_market_calendars by default) and applies
a YAML override file on top. The base library is never modified or patched --
overrides live as data so they survive upgrades and can be diffed and reviewed.

Resolution order:  base calendar  ->  overrides  ->  merged sessions
Overrides always win. When the merged calendar disagrees with observed price
data, the fix is always a new override, never an edit to the library or data.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from .clock import ist_today

log = logging.getLogger(__name__)

FREQ_PERIOD = {"W": "W", "weekly": "W", "M": "M", "monthly": "M"}


class CalendarError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# base calendar adapters
# --------------------------------------------------------------------------

class PandasMarketCalendarBase:
    """Adapter over pandas_market_calendars. Imported lazily so the rest of
    the module stays usable (and testable) without the dependency."""

    def __init__(self, name: str = "NSE"):
        try:
            import pandas_market_calendars as mcal
        except ImportError as exc:  # pragma: no cover - depends on env
            raise CalendarError(
                "pandas_market_calendars is not installed. "
                "Install it, or pass your own base_calendar."
            ) from exc
        self.name = name
        self._cal = mcal.get_calendar(name)

    def sessions(self, start: dt.date, end: dt.date) -> set[dt.date]:
        sched = self._cal.schedule(start_date=str(start), end_date=str(end))
        return {d.date() for d in sched.index}

    def coverage_end(self) -> dt.date | None:
        """Last date the underlying holiday rules are defined for."""
        try:
            return self._cal.holidays().holidays[-1]
        except Exception:  # noqa: BLE001 - library internals vary across versions
            return None


class StaticBase:
    """Base calendar from an explicit set of dates. Useful for tests, and as a
    fallback if you'd rather maintain the whole schedule yourself."""

    def __init__(self, sessions: set[dt.date], coverage_end: dt.date | None = None):
        self._sessions = set(sessions)
        self._coverage_end = coverage_end or (max(sessions) if sessions else None)

    def sessions(self, start: dt.date, end: dt.date) -> set[dt.date]:
        return {d for d in self._sessions if start <= d <= end}

    def coverage_end(self) -> dt.date | None:
        return self._coverage_end


# --------------------------------------------------------------------------
# overrides
# --------------------------------------------------------------------------

@dataclass
class Override:
    date: dt.date
    reason: str
    added: dt.date | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> Override:
        if "date" not in raw:
            raise CalendarError(f"override entry missing 'date': {raw!r}")
        if not raw.get("reason"):
            raise CalendarError(f"override for {raw['date']} needs a 'reason'")
        return cls(
            date=_as_date(raw["date"]),
            reason=str(raw["reason"]),
            added=_as_date(raw["added"]) if raw.get("added") else None,
        )


@dataclass
class Overrides:
    closed: dict[dt.date, Override] = field(default_factory=dict)
    open: dict[dt.date, Override] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> Overrides:
        if path is None:
            return cls()
        p = Path(path)
        if not p.exists():
            log.warning("no override file at %s -- using base calendar only", p)
            return cls()
        raw = yaml.safe_load(p.read_text()) or {}
        closed = [Override.from_dict(e) for e in (raw.get("closed") or [])]
        open_ = [Override.from_dict(e) for e in (raw.get("open") or [])]
        clash = {o.date for o in closed} & {o.date for o in open_}
        if clash:
            raise CalendarError(
                f"dates listed as both open and closed: {sorted(clash)}"
            )
        return cls(
            closed={o.date: o for o in closed},
            open={o.date: o for o in open_},
        )


# --------------------------------------------------------------------------
# calendar
# --------------------------------------------------------------------------

@dataclass
class Mismatch:
    date: dt.date
    kind: str  # "unexpected_session" | "unexpected_closure"

    @property
    def bucket(self) -> str:
        return "open" if self.kind == "unexpected_session" else "closed"


class TradingCalendar:
    def __init__(
        self,
        overrides_path: str | Path | None = None,
        base_calendar=None,
        name: str = "NSE",
    ):
        self.base = base_calendar if base_calendar is not None else PandasMarketCalendarBase(name)
        self.overrides = Overrides.load(overrides_path)
        self.overrides_path = Path(overrides_path) if overrides_path else None

    # -- sessions ---------------------------------------------------------

    def sessions(self, start, end) -> list[dt.date]:
        start, end = _as_date(start), _as_date(end)
        days = self.base.sessions(start, end)
        days -= {d for d in self.overrides.closed if start <= d <= end}
        days |= {d for d in self.overrides.open if start <= d <= end}
        return sorted(days)

    def is_session(self, day) -> bool:
        day = _as_date(day)
        if day in self.overrides.closed:
            return False
        if day in self.overrides.open:
            return True
        return day in self.base.sessions(day, day)

    # -- period completeness ---------------------------------------------

    def is_period_end(self, day, freq: str = "W") -> bool:
        """True if `day` is the final trading session of its week or month.

        This is the only place the calendar is allowed to assert completeness
        ahead of the data. Everywhere else, prefer `period_is_complete`, which
        needs no calendar at all.
        """
        day = _as_date(day)
        if not self.is_session(day):
            return False
        period = _period_of(day, freq)
        # look far enough ahead to clear any holiday cluster
        horizon = day + dt.timedelta(days=45 if _norm_freq(freq) == "M" else 12)
        later = [d for d in self.sessions(day, horizon)
                 if d > day and _period_of(d, freq) == period]
        return not later

    # -- reconciliation ---------------------------------------------------

    def reconcile(self, observed_sessions, start, end) -> list[Mismatch]:
        """Compare the merged calendar against sessions actually seen in price
        data. Any weekday where no symbol in your universe printed a bar was a
        closure, whatever the published list says."""
        start, end = _as_date(start), _as_date(end)
        observed = {_as_date(d) for d in observed_sessions if start <= _as_date(d) <= end}
        expected = set(self.sessions(start, end))
        out = [Mismatch(d, "unexpected_session") for d in sorted(observed - expected)]
        out += [Mismatch(d, "unexpected_closure") for d in sorted(expected - observed)]
        return out

    @staticmethod
    def mismatches_to_yaml(mismatches: list[Mismatch], today: dt.date | None = None) -> str:
        """Emit a paste-ready override block. Mismatches are rare enough that
        you won't remember the file format when one turns up."""
        if not mismatches:
            return ""
        today = today or ist_today()
        lines: list[str] = []
        for bucket in ("closed", "open"):
            rows = [m for m in mismatches if m.bucket == bucket]
            if not rows:
                continue
            lines.append(f"{bucket}:")
            for m in rows:
                why = ("unscheduled closure" if bucket == "closed"
                       else "session not in published schedule")
                lines.append(f"  - date: {m.date}")
                lines.append(f"    reason: {why}  # CONFIRM before committing")
                lines.append(f"    added: {today}")
        return "\n".join(lines)

    # -- staleness --------------------------------------------------------

    def warn_if_stale(self, today: dt.date | None = None, horizon_days: int = 60) -> str | None:
        today = today or ist_today()
        end = self.base.coverage_end()
        if end is None:
            return None
        end = _as_date(end)
        if end < today:
            msg = f"base calendar coverage ended {end} -- refresh before relying on it"
        elif (end - today).days <= horizon_days:
            msg = f"base calendar coverage ends {end} ({(end - today).days}d) -- refresh soon"
        else:
            return None
        log.warning(msg)
        return msg


# --------------------------------------------------------------------------
# data-derived completeness -- no calendar required
# --------------------------------------------------------------------------

def period_is_complete(index: pd.DatetimeIndex, freq: str = "W") -> pd.Series:
    """Foundation check: a period is complete once a bar from the NEXT period
    exists. Holidays, unscheduled closures and special sessions are all handled
    for free, because the trading days themselves encode them.

    Returns a Series indexed by period label -> bool. The final period is always
    False; every earlier one is True.
    """
    periods = pd.PeriodIndex(pd.DatetimeIndex(index), freq=_norm_freq(freq))
    uniq = periods.unique().sort_values()
    if len(uniq) == 0:
        # [True] * -1 is [], so the naive expression builds a 1-element list
        # against a 0-length index and raises. An empty store is a normal
        # state during a backfill, not an error.
        return pd.Series(dtype=bool, index=uniq)
    return pd.Series([True] * (len(uniq) - 1) + [False], index=uniq, dtype=bool)


def resample_ohlc(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Aggregate daily bars to weekly/monthly and tag completeness.

    The resample label is a calendar boundary and may fall on a day the market
    was shut -- that's cosmetic. The OHLC is correct because it aggregates
    whatever bars actually exist. Never read the label as proof of trading.
    """
    rule = {"W": "W-FRI", "M": "ME"}[_norm_freq(freq)]
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule).agg(agg).dropna(how="all")
    complete = period_is_complete(df.index, freq)
    out["complete"] = pd.PeriodIndex(out.index, freq=_norm_freq(freq)).map(complete)
    out["last_session"] = df.resample(rule).apply(
        lambda s: s.index.max() if len(s) else pd.NaT
    ).iloc[:, 0] if len(df) else pd.NaT
    return out


# --------------------------------------------------------------------------

def _as_date(v) -> dt.date:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    return pd.Timestamp(v).date()


def _norm_freq(freq: str) -> str:
    try:
        return FREQ_PERIOD[freq]
    except KeyError:
        raise CalendarError(f"unsupported freq {freq!r} -- use 'W' or 'M'")


def _period_of(day: dt.date, freq: str):
    return pd.Period(pd.Timestamp(day), freq=_norm_freq(freq))
