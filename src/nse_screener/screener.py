"""Run the detectors across an eligible universe.

Two jobs, kept separate because they fail for different reasons:

  eligibility -- which symbols are worth looking at at all
  detection   -- which of those printed a pattern

Everything here is a pure function over frames from the store. No I/O beyond
`store.read`, so Stage 7 can replay the same code over history.

On speed: the detectors only look back a dozen bars, but the history check needs
to count every bar a symbol ever printed. Those are very different reads, so
they are done separately -- a cheap two-column count over all history, then full
bars for the short window detection actually needs. Feeding a year of bars to the
detectors, as an earlier version did, was 94% wasted work and the single biggest
cost in the scan.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from . import context as ctx
from . import patterns as pat
from . import ranking
from .data import store

log = logging.getLogger(__name__)

WARMUP_BARS = 11
"""Bars a detector can reach back: 10 for the rolling averages, 1 for the
previous bar. Anything earlier cannot affect a result."""

CONTEXT_WARMUP_BARS = 260
"""What the context columns need: 200 for the slow EMA, 252 for the ATR
percentile window, plus a little slack. This, not the detectors, is what sets
how much history the scan reads."""

HIT_COLUMNS = [
    "date", "symbol", "pattern", "close", "chg_pct", "volume", "bars",
    *ctx.CONTEXT_COLUMNS,
]
"""The shape of `ScanResult.hits`, always -- an empty result carries the same
columns as a full one, so callers never have to special-case "no hits"."""


def _empty_hits() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in HIT_COLUMNS})


@dataclass(frozen=True)
class Universe:
    """Which symbols are eligible, and why a symbol is not."""

    min_history: int = 250
    """Sessions of history required. Excludes the recently-listed trap, where a
    detector's rolling average is computed over a handful of bars."""

    series: tuple[str, ...] = ("EQ",)
    """BE is trade-to-trade and usually under surveillance; excluded by default."""

    max_overnight_gap: float = 0.20
    """A close-to-close move this large is more likely an unadjusted split or
    bonus than a real one. Prices either side of it are not comparable, so any
    signal whose lookback window spans one is untrustworthy."""

    min_price: float = 20.0
    """Below this, one tick is a large fraction of the price, so an ATR-sized
    stop is a couple of ticks wide and any R multiple measured against it is
    arithmetic rather than trading."""

    min_traded_value: float = 1e7
    """Median daily close x volume, in rupees. Rs 1 crore. This is a *tradeability*
    floor, not the turnover metric that was removed at the owner's request --
    nothing displays or ranks on it, it only decides whether a fill is credible.

    Both floors exist because their absence was measured, not assumed. A
    breakout backtest run without them reported +731%, of which sub-Rs 20 names
    supplied 68.8% of the P&L on 15.2% of the trades -- names like IMPEXFERRO at
    Rs 1.75 posting R multiples of 26 on fills nobody could have got. With the
    floors on, the same strategy returns -82%, and so does its baseline.

    Measured at `asof` 2026-07-29: the two together keep 1,580 of 2,768 symbols
    (57%). The traded-value floor does nearly all of the work; the price floor
    removes a further 56 symbols on top of it.
    """

    def __post_init__(self):
        if self.min_history < 1:
            raise ValueError("min_history must be >= 1")
        if not 0 < self.max_overnight_gap < 1:
            raise ValueError("max_overnight_gap must be a fraction between 0 and 1")
        if self.min_price < 0:
            raise ValueError("min_price must be >= 0")
        if self.min_traded_value < 0:
            raise ValueError("min_traded_value must be >= 0")


@dataclass
class ScanResult:
    """Hits plus the counts behind them, so a quiet night is explicable."""

    hits: pd.DataFrame
    asof: date
    universe_size: int
    rejected: dict[str, int] = field(default_factory=dict)
    sessions: list[date] = field(default_factory=list)
    """The sessions covered, oldest first. `asof` is the last of them."""

    def __len__(self) -> int:
        return len(self.hits)

    def by_pattern(self) -> pd.Series:
        if self.hits.empty:
            return pd.Series(dtype=int)
        return self.hits["pattern"].value_counts()

    def by_date(self) -> pd.Series:
        if self.hits.empty:
            return pd.Series(dtype=int)
        return self.hits["date"].value_counts().sort_index()


def latest_session(bars_dir=None) -> date | None:
    dates = store.available_dates(bars_dir)
    return max(dates) if dates else None


def history_counts(
    asof: date, series: tuple[str, ...] = ("EQ",), bars_dir=None
) -> pd.Series:
    """Sessions each symbol has printed up to and including `asof`.

    Reads two columns rather than twelve, so counting every bar in six years of
    history costs a fraction of a second.
    """
    bars = store.read(
        end=asof, columns=["series"], series=list(series), bars_dir=bars_dir
    )
    if bars.empty:
        return pd.Series(dtype=int)
    return bars.groupby("symbol").size()


def gap_flags(bars: pd.DataFrame, limit: float) -> pd.Series:
    """Per symbol: did any bar in `bars` gap more than `limit` from the last?

    Vectorised across all symbols at once. `bars` is expected to be the short
    detection window, so "anywhere in the window" is exactly the question --
    a corporate action in there corrupts every rolling average that spans it.
    """
    if bars.empty:
        return pd.Series(dtype=bool)
    ordered = bars.sort_values(["symbol", "date"])
    change = ordered.groupby("symbol")["close"].pct_change().abs()
    return (change > limit).groupby(ordered["symbol"]).any()


def liquidity_flags(bars: pd.DataFrame, rules: Universe) -> pd.Series:
    """Per symbol: is it liquid enough that a fill would have been real?

    Price is taken from the last bar in `bars` and traded value from the median
    across it, so one frantic day cannot carry a symbol that is otherwise dead.
    The median is over the window handed in, which ends at `asof` -- this reads
    no bar the caller has not already decided is in the past.
    """
    if bars.empty:
        return pd.Series(dtype=bool)
    ordered = bars.sort_values(["symbol", "date"])
    last_close = ordered.groupby("symbol")["close"].last()
    traded_value = (ordered["close"] * ordered["volume"]).groupby(
        ordered["symbol"]
    ).median()
    return (last_close >= rules.min_price) & (
        traded_value >= rules.min_traded_value
    )


def eligibility(
    counts: pd.Series, gaps: pd.Series, rules: Universe,
    liquid: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per symbol with the reason columns, so a rejection is explicable.

    `liquid` is optional so a caller measuring something other than tradeability
    can leave the floor off deliberately, rather than by forgetting it.
    """
    cols = ["symbol", "bars", "recent_gap", "enough_history", "liquid", "eligible"]
    if counts.empty:
        return pd.DataFrame(columns=cols)
    e = pd.DataFrame({"symbol": counts.index, "bars": counts.to_numpy()})
    e["recent_gap"] = e["symbol"].map(gaps).fillna(False).astype(bool)
    e["enough_history"] = e["bars"] >= rules.min_history
    if liquid is None:
        e["liquid"] = True
    else:
        e["liquid"] = e["symbol"].map(liquid).fillna(False).astype(bool)
    e["eligible"] = e["enough_history"] & ~e["recent_gap"] & e["liquid"]
    return e


def scan(
    asof: date | None = None,
    *,
    sessions: int = 1,
    rules: Universe | None = None,
    which: list[str] | None = None,
    rank_by: str | None = None,
    bars_dir=None,
) -> ScanResult:
    """Patterns printed across the eligible universe.

    `asof` defaults to the most recent session in the store. `sessions` widens
    the report backwards from there: 1 is tonight only, 3 covers the last three
    trading sessions.

    Eligibility is judged once, as of `asof`. Over a few sessions that is what
    you want, but it means a symbol suspended two days ago is still judged on
    today's state.

    `rank_by` orders the hits within each session and adds a `rank_score`
    column -- see `ranking.py` for what is registered and what each one claims.
    None keeps the plain listing. Note that no ranker has yet been shown to beat
    `ranking.random`; ordering the list is not evidence that the top of it is
    better, only that the order is now deliberate and reproducible instead of
    alphabetical.
    """
    rules = rules or Universe()
    if sessions < 1:
        raise ValueError("sessions must be >= 1")
    asof = asof or latest_session(bars_dir)
    if asof is None:
        return ScanResult(_empty_hits(), date.min, 0)

    # How much history does each symbol have? Two columns over everything.
    counts = history_counts(asof, rules.series, bars_dir)
    if counts.empty:
        return ScanResult(_empty_hits(), asof, 0)

    # Enough history for the context columns, which reach much further back
    # than the detectors do. Calendar days generously, so holidays cannot
    # shorten the session count.
    need = CONTEXT_WARMUP_BARS + sessions
    start = pd.Timestamp(asof) - pd.Timedelta(days=int(need * 1.7) + 10)
    recent = store.read(
        start=start.date(), end=asof, series=list(rules.series), bars_dir=bars_dir
    )
    if recent.empty:
        return ScanResult(_empty_hits(), asof, 0)

    # The gap guard looks only at the detectors' own reach, NOT the whole read.
    # The read is long because the context columns need a year of history; the
    # question the guard asks is "is a corporate action still inside the window
    # a detector can see". Handing it the full read rejected every symbol that
    # had moved 20% at any point in fifteen months -- 310 symbols instead of 12.
    gap_dates = sorted(recent["date"].unique())[-(WARMUP_BARS + sessions):]
    gap_window = recent[recent["date"] >= gap_dates[0]]
    # Liquidity is judged over the whole read, not the gap window: a symbol's
    # tradeability is a property of the year, not of the last eleven bars.
    elig = eligibility(
        counts,
        gap_flags(gap_window, rules.max_overnight_gap),
        rules,
        liquidity_flags(recent, rules),
    )
    rejected = {
        "short history": int((~elig["enough_history"]).sum()),
        "recent gap (likely corporate action)": int(elig["recent_gap"].sum()),
        "too illiquid to trade": int((~elig["liquid"]).sum()),
    }
    keep = set(elig.loc[elig["eligible"], "symbol"])
    if not keep:
        return ScanResult(_empty_hits(), asof, 0, rejected)

    universe = recent[recent["symbol"].isin(keep)].copy()
    universe["pattern"] = pat.classify_by_symbol(universe, which)

    # The window is the last `sessions` trading days present in the data.
    window = sorted(universe["date"].unique())[-sessions:]
    covered = [pd.Timestamp(d).date() for d in window]

    found = universe[
        universe["date"].isin(window) & universe["pattern"].notna()
    ].copy()
    if found.empty:
        return ScanResult(_empty_hits(), asof, len(keep), rejected, covered)

    prev_close = universe.groupby("symbol")["close"].shift(1)
    found["chg_pct"] = (found["close"] / prev_close.loc[found.index] - 1) * 100
    found["bars"] = found["symbol"].map(counts)
    found["date"] = found["date"].dt.date

    # Context is measured over the whole universe frame, then joined onto the
    # hits -- an EMA needs the bars before the signal, not just the signal bar.
    # Columns, never filters: Stage 7 decides which of these earn their place.
    context = ctx.annotate_by_symbol(universe)
    found = found.join(context.loc[found.index])

    # Ranked within each session, newest session first. Ranking is the answer to
    # "there are 477 of these, where do I start" -- and it is per-session because
    # comparing a name that fired tonight against one that fired on Monday is a
    # different question, which this ordering deliberately does not pretend to
    # answer. `rank_by=None` keeps the old alphabetical-by-pattern listing.
    ordered = found[HIT_COLUMNS]
    if rank_by is None:
        hits = ordered.sort_values(
            ["date", "pattern", "symbol"], ascending=[False, True, True]
        )
    else:
        # Sorting on date first partitions the frame, so one sort gives a
        # per-session ranking without a groupby. Symbol breaks ties, NaN scores
        # sort last -- the same total order `ranking.rank()` guarantees.
        hits = ordered.assign(
            rank_score=ranking.score(ordered, rank_by)
        ).sort_values(
            ["date", "rank_score", "symbol"],
            ascending=[False, False, True],
            na_position="last",
        )
    return ScanResult(hits.reset_index(drop=True), asof, len(keep), rejected, covered)
