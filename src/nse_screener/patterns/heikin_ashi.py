"""Heikin-Ashi candles, and the signals built on them.

    HA_close = (open + high + low + close) / 4
    HA_open  = (previous HA_open + previous HA_close) / 2      <- recursive
    HA_high  = max(high, HA_open, HA_close)
    HA_low   = min(low,  HA_open, HA_close)

`HA_open` is a recurrence, not a rolling window: each value depends on the one
before it, all the way back to the first bar. A Python loop over 2.9M rows is
too slow to live with, so it is solved in closed form.

    HA_open[i] = 0.5 * HA_open[i-1] + 0.5 * HA_close[i-1]

is exactly what `ewm(alpha=0.5, adjust=False)` computes over the *lagged*
HA_close series, because that recursion is an exponential mean with alpha 0.5.
`tests/test_heikin_ashi.py` checks the vectorised result against a literal loop
on real bars -- this is the kind of substitution that is either exactly right or
subtly wrong, and eyeballing a chart will not tell you which.

Seeding matters and is not standardised. The convention used here is
`HA_open[0] = (open[0] + close[0]) / 2`, which is what most charting packages
use. The choice washes out within a handful of bars but the first few would
otherwise be wrong, so each symbol is seeded from its own first bar.

Why bother: HA smooths noise into runs, so a trend shows as a stretch of
same-coloured candles. That makes it a natural *exit* rule -- stay in while the
colour holds, leave when it flips -- which is a different question from the fixed
ATR target the rest of this project has been testing.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g
from .registry import register

HA_COLUMNS = ["ha_open", "ha_high", "ha_low", "ha_close"]


def transform(bars: pd.DataFrame, by: pd.Series | None = None) -> pd.DataFrame:
    """Heikin-Ashi OHLC for every bar, indexed like `bars`.

    Expects bars already sorted within each symbol.
    """
    d = g.normalise(bars)
    out = pd.DataFrame(index=bars.index)

    ha_close = (d["open"] + d["high"] + d["low"] + d["close"]) / 4

    # The recurrence, in closed form. `lagged` is HA_close shifted one bar, with
    # each symbol's first slot carrying the conventional seed instead of NaN.
    lagged = g.shift_by(ha_close, 1, by)
    seed = (d["open"] + d["close"]) / 2
    if by is None:
        lagged.iloc[0] = seed.iloc[0]
    else:
        lagged = lagged.mask(~by.duplicated(keep="first"), seed)

    if by is None:
        ha_open = lagged.ewm(alpha=0.5, adjust=False).mean()
    else:
        ha_open = lagged.groupby(by).ewm(alpha=0.5, adjust=False).mean()
        ha_open.index = ha_open.index.droplevel(0)

    out["ha_open"] = ha_open
    out["ha_close"] = ha_close
    out["ha_high"] = pd.concat([d["high"], ha_open, ha_close], axis=1).max(axis=1)
    out["ha_low"] = pd.concat([d["low"], ha_open, ha_close], axis=1).min(axis=1)
    return out[HA_COLUMNS]


def is_bullish(bars: pd.DataFrame, by: pd.Series | None = None) -> pd.Series:
    """True where the Heikin-Ashi candle is green."""
    ha = transform(bars, by)
    return (ha["ha_close"] > ha["ha_open"]).fillna(False).astype(bool)


# Specificity ranks how *narrow* a signal is, and this one is broad: it fires on
# ~10% of all bars, more often than any candlestick. Registered at 70 it
# outranked everything and silently became the label on most flagged bars,
# breaking two screener tests. Measured firing rates over 2020-2026, for
# calibration: squeeze_release 1.0%, breakout_252 1.2%, breakout_20 4.8%,
# rsi2_pullback 5.1%, ha_flip_up 10.0%.
@register("ha_flip_up", kind="single", direction="bullish", specificity=15)
def ha_flip_up(
    df: pd.DataFrame,
    *,
    min_prior_red: int = 2,
    require_no_lower_wick: bool = False,
    by: pd.Series | None = None,
) -> pd.Series:
    """The Heikin-Ashi candle turns green after a run of red ones.

    `min_prior_red` is what makes this a turn rather than noise: one red bar
    inside an uptrend flips constantly, and a run of them is what HA is for.

    `require_no_lower_wick` adds the textbook strength filter -- a green HA
    candle whose low equals its open has no selling inside the bar at all.
    """
    d = g.normalise(df)
    ha = transform(d, by)
    green = ha["ha_close"] > ha["ha_open"]

    hit = green.copy()
    for i in range(1, min_prior_red + 1):
        hit &= ~g.shift_by(green, i, by).fillna(False).astype(bool)

    if require_no_lower_wick:
        hit &= ha["ha_low"] >= ha["ha_open"] - 1e-9
    return hit.fillna(False).astype(bool)


def flip_down(bars: pd.DataFrame, by: pd.Series | None = None) -> pd.Series:
    """The candle turns red having been green. The exit half of the pair.

    Not registered as a detector: it is a bearish exit condition, and every
    registered signal in this project is an entry. The backtest takes it via
    `exit_signal_col`.
    """
    ha = transform(bars, by)
    green = ha["ha_close"] > ha["ha_open"]
    was_green = g.shift_by(green, 1, by).fillna(False).astype(bool)
    return (~green & was_green).fillna(False).astype(bool)
