"""The back half of the decision sheet: levels, risk, reward, and size.

The screener says a pattern fired. It does not say where the stop goes, what the
target is, whether the trade is worth taking, or how many shares to buy. That is
rows 7 to 13 of the SMM checklist and the whole arithmetic block of the Excel
sheet, and this module is that -- pure functions over frames, same contract as
the detectors, so the study and the backtest could use it unchanged.

Three things in here are decisions rather than transcription, and each is the
kind that quietly invalidates everything downstream if it goes the other way.

**The target never comes from the stop.** It would be trivial to define the
target as three times the risk and report that every setup clears the sheet's
"Reward to Risk should be above 3" gate. That number would be arithmetic about
itself: a tautology dressed as a filter. Both targets here come from market
structure -- confirmed swing highs, which is the sheet's own "Major Resistance
... previous tops or resistances" -- and a setup with no structure above it gets
no target and no plan, rather than a made-up one.

**The sheet sizes positions two ways and never reconciles them.** Row 29 caps
shares by the 2% risk budget; row 32 sets them from a fixed rupee investment.
On the worked example -- close 269, stop 266, 1 lakh capital -- the risk cap
allows 666 shares while a 20,000 investment buys 74. Both are computed and
neither wins. Here both are reported, `shares` is the smaller, and `binding`
says which one bound, because a trader reading a filled sheet should be able to
see whether risk or capital is the constraint.

**A reward-to-risk ratio is a plan, not a prediction.** The sheet gates entry on
R:R above 3. Stage 9 of this project backtested a planned 3:1 and measured an
average outcome of -0.17R. Nothing here contradicts that; `rr_min` is what the
trade is *aiming* at, and the distance between that and what trades actually
return is the whole finding of Stages 7 and 9. Filling the sheet in is not
evidence the sheet works.

Levels are read from `_swings.py`'s confirmed pivots, so they inherit its
refusal to see the future: a support level is one that had already printed and
been confirmed on the signal bar, never one visible only in hindsight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import context as ctx
from .patterns import _geometry as g
from .patterns import _swings as sw

PLAN_COLUMNS = [
    "symbol", "close", "direction",
    "immediate_support", "immediate_resistance", "major_support", "major_resistance",
    "significant_median", "atr",
    "stop", "target_min", "target_max",
    "risk", "reward_min", "reward_max", "rr_min", "rr_max",
    "max_risk_rupees", "max_shares_allowed", "shares_by_investment",
    "shares", "binding", "risk_involved", "risk_pct_of_capital",
    "profit_min", "profit_max", "roi_min_pct", "roi_max_pct",
    "trade_value", "meets_rr",
]
"""The shape of a filled sheet, always -- an empty plan carries the same columns
as a full one, matching `ScanResult.hits` and for the same reason."""


@dataclass(frozen=True)
class Account:
    """Capital and the two limits the sheet applies to it."""

    capital: float = 100_000.0
    max_risk_pct: float = 0.02
    """Row 30 of the Excel sheet: the most of total capital a single trade may
    risk between entry and stop. 2% is the sheet's own figure."""

    investment_per_trade: float = 20_000.0
    """Row 33. The rupee amount actually committed, which is what row 34 divides
    by price to get shares. Independent of the risk cap above -- see the module
    docstring on why both are reported."""

    def __post_init__(self):
        if self.capital <= 0:
            raise ValueError("capital must be positive")
        if not 0 < self.max_risk_pct <= 1:
            raise ValueError("max_risk_pct must be a fraction in (0, 1]")
        if self.investment_per_trade <= 0:
            raise ValueError("investment_per_trade must be positive")


@dataclass(frozen=True)
class LevelRules:
    """How the sheet's hand-drawn levels are derived from bars."""

    lookback: int = 60
    left: int = 5
    right: int = 5
    stop_buffer_atr: float = 0.25
    """Row 11 says the stop goes *below* the immediate support, not on it. A
    quarter of an ATR is that "below" -- placed on the level exactly, a stop is
    taken by the noise that defines the level."""

    atr_period: int = 14
    min_rr: float = 3.0
    """Row 13: "Should be above 3 to enter the trade"."""

    def __post_init__(self):
        if self.min_rr <= 0:
            raise ValueError("min_rr must be positive")


def _empty_plan() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in PLAN_COLUMNS})


def significant_candle_median(
    bars: pd.DataFrame, *, lookback: int, by: pd.Series | None = None
) -> pd.Series:
    """Row 7's "Median of the Most Significant Candle i.e. the one with Heavy Volume".

    The midpoint of the highest-volume bar in the lookback. Vectorised without a
    per-window scan: the running maximum volume identifies the bar, and its
    midpoint is carried forward from there.

    Uses `(high + low) / 2` for "median", which is the candle's midpoint -- the
    reading everyone draws on a chart. The sheet does not define the word.
    """
    d = g.normalise(bars)
    mid = (d["high"] + d["low"]) / 2
    vol = bars["volume"].astype("float64")
    top = sw._rolling(vol, by, lookback, "max")
    # The bar carrying the window's peak volume, carried forward until a heavier
    # one prints. Ties take the most recent, which is the live reading.
    return sw._ffill(mid.where(vol >= top), by)


def levels(
    bars: pd.DataFrame,
    *,
    rules: LevelRules | None = None,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Rows 7 to 10 of the checklist, per bar.

    `immediate_support` is the nearest thing below price that a stop can sit
    under: the more recent of the last confirmed swing low and the significant
    candle's midpoint, taking whichever is closer to price while still below it.
    `major_resistance` is the highest confirmed swing high in the lookback -- the
    sheet's "previous tops or resistances" -- and `immediate_resistance` the
    nearest one above price.
    """
    rules = rules or LevelRules()
    d = g.normalise(bars)
    close = d["close"]

    piv = sw.pivots(d, left=rules.left, right=rules.right, by=by)
    lows = sw.track(d["low"], piv["pivot_low"], right=rules.right, by=by)
    highs = sw.track(d["high"], piv["pivot_high"], right=rules.right, by=by)

    out = pd.DataFrame(index=bars.index)
    out["significant_median"] = significant_candle_median(
        bars, lookback=rules.lookback, by=by
    )

    # Only levels that are actually on the right side of price are candidates.
    below = [lows["last"].where(lows["last"] < close),
             out["significant_median"].where(out["significant_median"] < close)]
    out["immediate_support"] = pd.concat(below, axis=1).max(axis=1)
    out["major_support"] = sw._rolling(d["low"], by, rules.lookback, "min")

    above = [highs["last"].where(highs["last"] > close),
             highs["prev"].where(highs["prev"] > close)]
    out["immediate_resistance"] = pd.concat(above, axis=1).min(axis=1)
    # Row 10's "Major Resistance": the highest prior top in the lookback. Read
    # from the previous bar so today's own high cannot become its own target.
    out["major_resistance"] = g.shift_by(
        sw._rolling(d["high"], by, rules.lookback, "max"), 1, by
    ).where(lambda s: s > close)

    out["atr"] = ctx.atr(d, by, rules.atr_period)
    return out


def plan(
    bars: pd.DataFrame,
    *,
    account: Account | None = None,
    rules: LevelRules | None = None,
    direction: str = "long",
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """The whole arithmetic block, per bar.

    Every column of the Excel sheet from "Stop Loss Price" down to "Max ROI%",
    plus the levels they are computed from. Rows where no target exists above
    price carry NaN rather than a fabricated one -- see the module docstring.
    """
    account = account or Account()
    rules = rules or LevelRules()
    if direction != "long":
        raise ValueError(
            "only 'long' is implemented: every registered signal in this project "
            "is a long entry, and the short side would need its own levels"
        )

    d = g.normalise(bars)
    lv = levels(bars, rules=rules, by=by)
    close = d["close"]

    out = pd.DataFrame(index=bars.index)
    if "symbol" in bars.columns:
        out["symbol"] = bars["symbol"]
    out["close"] = close
    out["direction"] = "Buy"
    for c in ("immediate_support", "immediate_resistance", "major_support",
              "major_resistance", "significant_median", "atr"):
        out[c] = lv[c]

    # Row 11: below the immediate support, by a fraction of an ATR.
    out["stop"] = lv["immediate_support"] - rules.stop_buffer_atr * lv["atr"]
    # Row 12: structure, never a multiple of the risk. See the module docstring.
    out["target_min"] = lv["immediate_resistance"]
    out["target_max"] = lv["major_resistance"]

    arithmetic = size(
        close, out["stop"], out["target_min"], out["target_max"],
        account=account, min_rr=rules.min_rr,
    )
    for c in arithmetic.columns:
        out[c] = arithmetic[c]
    return out[[c for c in PLAN_COLUMNS if c in out.columns]]


def size(
    close: pd.Series,
    stop: pd.Series,
    target_min: pd.Series,
    target_max: pd.Series,
    *,
    account: Account | None = None,
    min_rr: float = 3.0,
) -> pd.DataFrame:
    """Rows 20 to 39 of the Excel sheet, as arithmetic on four price series.

    Separated from `plan()` so the formulas can be checked against the sheet's
    own worked example rather than only against bars this code derived itself.
    Every column here is one of its rows; the only additions are `binding` and
    `shares`, which reconcile the two sizings the sheet leaves side by side.
    """
    account = account or Account()
    out = pd.DataFrame(index=close.index)
    out["target_min"] = target_min
    # One target is enough to plan on; the max leg falls back to the min.
    out["target_max"] = target_max.fillna(target_min)

    out["risk"] = (close - stop).where(lambda s: s > 0)
    out["reward_min"] = (out["target_min"] - close).where(lambda s: s > 0)
    out["reward_max"] = (out["target_max"] - close).where(lambda s: s > 0)
    out["reward_max"] = out["reward_max"].fillna(out["reward_min"])

    out["rr_min"] = out["reward_min"] / out["risk"]
    out["rr_max"] = out["reward_max"] / out["risk"]

    # --- sizing, both ways, as the sheet computes them --------------------
    out["max_risk_rupees"] = account.capital * account.max_risk_pct
    out["max_shares_allowed"] = np.floor(
        out["max_risk_rupees"] / out["risk"]
    ).replace([np.inf, -np.inf], np.nan)
    out["shares_by_investment"] = np.floor(account.investment_per_trade / close)

    pair = out[["max_shares_allowed", "shares_by_investment"]]
    out["shares"] = pair.min(axis=1)
    # Which limit actually bound. The sheet computes both and reconciles
    # neither, so a filled row should at least say which one is in charge.
    out["binding"] = np.where(
        out["max_shares_allowed"] <= out["shares_by_investment"], "risk", "investment"
    )
    out.loc[pair.isna().any(axis=1), "binding"] = None

    out["risk_involved"] = out["shares"] * out["risk"]
    out["risk_pct_of_capital"] = out["risk_involved"] / account.capital * 100
    out["profit_min"] = out["shares"] * out["reward_min"]
    out["profit_max"] = out["shares"] * out["reward_max"]
    out["trade_value"] = out["shares"] * close
    out["roi_min_pct"] = out["profit_min"] / out["trade_value"] * 100
    out["roi_max_pct"] = out["profit_max"] / out["trade_value"] * 100

    # Row 13. NaN is not a pass: a setup with no target does not clear the gate.
    out["meets_rr"] = (out["rr_min"] >= min_rr).fillna(False)
    return out


def for_hits(
    hits: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    account: Account | None = None,
    rules: LevelRules | None = None,
) -> pd.DataFrame:
    """Fill the sheet for each row of a `ScanResult.hits` frame.

    `bars` is the multi-symbol frame the scan was computed from -- levels need
    the history behind each signal, not just the signal bar. Rows come back in
    the order they were handed in, carrying `pattern` and `date` alongside the
    plan so a filled sheet can name what it is a plan for.
    """
    if hits.empty:
        return _empty_plan()

    full = plan(bars, account=account, rules=rules, by=bars["symbol"])

    # One row per (date, symbol) in the hits, taken from the same bar the
    # detector fired on rather than from the latest bar in the store. `symbol`
    # is dropped before keying so it comes back from the index rather than
    # colliding with it.
    keyed = full.drop(columns=["symbol"], errors="ignore")
    keyed.index = pd.MultiIndex.from_arrays(
        [bars["date"].dt.date, bars["symbol"]], names=["date", "symbol"]
    )
    want = pd.MultiIndex.from_arrays(
        [hits["date"], hits["symbol"]], names=["date", "symbol"]
    )
    out = keyed.reindex(want).reset_index()
    out.insert(2, "pattern", hits["pattern"].to_numpy())
    return out


# --------------------------------------------------------------------------
# the front half of the sheet: the confluence checklist
# --------------------------------------------------------------------------

CHECKS = [
    "candlestick", "volume", "ema", "chart_pattern", "fib_retrace", "divergence",
]
"""Rows 1 to 6 of the SMM checklist, in its order. Rows 7 to 13 are levels and
arithmetic and live above; these six are the ones that vote."""


def _volume_band(rel: pd.Series) -> pd.Series:
    """Row 2, banded. The sheet's wording is "Heavy Volume (more than average of
    last few days)" against the plain case, and it writes the strong reading as
    HUGE. `rel_volume` is already volume against its own 20-day average, so the
    bands are that ratio: above 2x is the sheet's HUGE, above 1x is heavier than
    average, below is not a vote."""
    out = pd.Series(pd.NA, index=rel.index, dtype="string")
    out = out.mask(rel >= 1.0, "Heavy")
    out = out.mask(rel >= 2.0, "HUGE")
    out = out.mask(rel < 1.0, "Light")
    return out


def checklist(
    bars: pd.DataFrame,
    hits: pd.DataFrame,
    *,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Fill checklist rows 1 to 6 for each hit, and count how many agree.

    `score` is that count, 0 to 6, and it is what "most promising" means on this
    page. It is the sheet's own logic -- confluence, the number of independent
    reasons pointing the same way -- rather than one of `ranking.py`'s rankers,
    none of which has been shown to beat random.

    That choice is worth being clear about: **a higher score is not evidence of
    a better trade.** Nothing in this project has measured whether confluence
    predicts anything, and the two things it has measured -- Stage 7 and Stage 9
    -- found no edge in the individual components. The score orders the sheet so
    the ordering is the sheet's own and is reproducible. It does not rank by
    expected return, because no such ranking has earned the right to exist here.
    """
    from . import patterns as pat
    from .patterns import divergence as dv
    from .patterns import fib

    d = g.normalise(bars)
    by = by if by is not None else bars.get("symbol")
    context = ctx.annotate(d, by=by)

    frame = pd.DataFrame(index=bars.index)
    frame["date"] = bars["date"].dt.date
    frame["symbol"] = bars["symbol"]

    candles = ["hammer", "inverted_hammer", "bullish_engulfing", "doji"]
    charts = ["inverted_head_shoulders", "double_bottom", "rounding_bottom",
              "cup_and_handle", "flag_breakout", "fake_breakdown"]
    fired = pat.detect_by_symbol(bars, candles + charts)

    frame["candlestick"] = _label_first(fired[candles])
    frame["volume"] = _volume_band(context["rel_volume"])
    # Row 3: the sheet reads a 5 EMA crossing 13 or 26 within the last three
    # candles. `ema_stack` is the standing state rather than the crossing event,
    # so this is the weaker reading of the row and is marked as such.
    frame["ema"] = context["ema_stack"].map(
        {"up": "Stacked up", "down": "Stacked down", "mixed": "Mixed"}
    ).astype("string")
    frame["chart_pattern"] = _label_first(fired[charts])
    frame["fib_retrace"] = fib.retracement_grade(bars, by=by)
    frame["divergence"] = dv.agreement(bars, "long", by=by)

    votes = pd.DataFrame({
        "candlestick": frame["candlestick"].notna(),
        "volume": frame["volume"].isin(["Heavy", "HUGE"]),
        "ema": frame["ema"] == "Stacked up",
        "chart_pattern": frame["chart_pattern"].notna(),
        "fib_retrace": frame["fib_retrace"] == fib.HEALTHY,
        "divergence": frame["divergence"] == dv.WITH,
    })
    frame["score"] = votes.sum(axis=1)

    keyed = frame.set_index(["date", "symbol"])
    want = pd.MultiIndex.from_arrays([hits["date"], hits["symbol"]])
    return keyed.reindex(want).reset_index()


def _label_first(fired: pd.DataFrame) -> pd.Series:
    """The most specific pattern that fired, as a display label."""
    from . import patterns as pat

    out = pd.Series(pd.NA, index=fired.index, dtype="string")
    for name in reversed(pat.by_specificity(fired.columns)):
        out = out.mask(fired[name], pat.label(name))
    return out
