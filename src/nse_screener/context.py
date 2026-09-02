"""Context columns for a bar: where it sits in its own recent history.

Stage 6. These are **columns, not filters** -- deliberately. Every one of them
is a plausible thing to screen on, and none of them has been measured yet. The
point of adding them now is to look at them for a couple of weeks and let the
combinations that seem to matter become Stage 7's hypotheses. Filtering here
would throw away the evidence needed to find out whether the filter was right.

Four groups, seventeen columns:

  Trend       above_200ema, dist_200ema_pct  where it stands against the anchor
              dist_25/13/5ema_pct            the ribbon, as distance from price
              ema_stack                      up / down / mixed at a glance

  Momentum    rsi_14, rsi_zone               Wilder's RSI, and which side of
                                             30 / 70 it falls
              stoch_k, stoch_d               slow stochastic (14, 3, 3)
              macd, macd_signal              MACD (12, 26, 9)
              macd_hist_pct                  histogram, as a % of price

  Volatility  atr_pct                        ATR(14) as a fraction of price
              bb_pct_b, bb_width_pct         Bollinger (20, 2): position, squeeze

  Volume      rel_volume                     against a 20-day average

`bb_pct_b` says where price sits between the bands; `bb_width_pct` how far apart
they are -- a squeeze. They look similar and are not.

Removed at the owner's request: `atr_pctile`, the ATR's rank within its own
trailing year. Worth recording that it was *not* redundant -- measured across
154,059 bars it correlated 0.04 with `atr_pct` and 0.19 with `bb_width_pct`, so
it carried volatility-regime information nothing else here does. Restoring it is
a few lines; `_rolling_rank` is still present.

Verified against TA-Lib on 13,064 real bars: stochastic and Bollinger match
exactly; RSI, MACD and ATR match to 0.02, 0.00004 and 0.0009 respectively, the
residue of a different EMA seed (see EMA_WARMUP_MULTIPLE).

Everything is groupby-aware and vectorised. Like the detectors, these must never
be computed across a symbol boundary -- `store.read()` sorts by (date, symbol),
so consecutive rows are different companies. Pass `by=bars["symbol"]`.

Not here: **delivery %**, which is on the plan's Stage 6 list but is not in the
store. Bhavcopy does not carry it; it needs `sec_bhavdata_full` fetched for every
session, which is a data-layer job rather than a computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .patterns import _geometry as g

TREND_COLUMNS = [
    "above_200ema",
    "dist_200ema_pct",
    "dist_25ema_pct",
    "dist_13ema_pct",
    "dist_5ema_pct",
    "ema_stack",
]
MOMENTUM_COLUMNS = [
    "rsi_14",
    "rsi_zone",
    "stoch_k",
    "stoch_d",
    "macd",
    "macd_signal",
    "macd_hist_pct",
]
VOLATILITY_COLUMNS = [
    "atr_pct",
    "bb_pct_b",
    "bb_width_pct",
]
VOLUME_COLUMNS = ["rel_volume"]

GROUPS = {
    "Trend": TREND_COLUMNS,
    "Momentum": MOMENTUM_COLUMNS,
    "Volatility": VOLATILITY_COLUMNS,
    "Volume": VOLUME_COLUMNS,
}

CONTEXT_COLUMNS = [c for cols in GROUPS.values() for c in cols]

EMA_WARMUP_MULTIPLE = 8
"""How many periods of history a Wilder- or EMA-smoothed indicator needs before
its value is trustworthy.

These are seeded differently by different implementations -- TA-Lib starts its
EMA from a simple mean of the first n values, pandas `ewm(adjust=False)` starts
from the first value -- so early readings disagree and then converge. Measured
against TA-Lib on six symbols, worst absolute error by warm-up:

        3x       5x       8x      10x
 RSI  2.54     0.210    0.018    0.003
 MACD 0.330    0.006    0.000    0.000
 ATR% 0.082    0.017    0.001    0.000

8x is the chosen trade. The residue at the *first* unmasked bar can still reach
about 0.05 RSI points on a volatile series; a few bars later it is zero, and the
median difference across a full history is 0.0. Masking at 1x, as an earlier
version did, showed an RSI that could be 2.5 points wrong -- which is the size of
a real signal, not rounding.
"""

# Moving averages are reported as *distance from price*, not raw levels. A raw
# EMA is not sortable across a screener: a 2,000-rupee stock always has a bigger
# one than a 20-rupee stock. "3% above its 25 EMA" compares; "EMA = 1,943" does
# not. Same reasoning puts the MACD histogram in percent.



def _ewm(s: pd.Series, by: pd.Series | None, **kw) -> pd.Series:
    """Exponential mean, never crossing a symbol boundary."""
    if by is None:
        return s.ewm(**kw).mean()
    out = s.groupby(by).ewm(**kw).mean()
    out.index = out.index.droplevel(0)
    return out


def _rolling_mean(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    if by is None:
        return s.rolling(window).mean()
    out = s.groupby(by).rolling(window).mean()
    out.index = out.index.droplevel(0)
    return out


def _rolling_rank(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    """Percentile rank of each value within its own trailing window, 0-100."""
    if by is None:
        return s.rolling(window).rank(pct=True) * 100
    out = s.groupby(by).rolling(window).rank(pct=True) * 100
    out.index = out.index.droplevel(0)
    return out


def _rolling_std(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    """Population standard deviation (ddof=0), which is what charting packages
    use for Bollinger Bands. pandas defaults to the sample form."""
    if by is None:
        return s.rolling(window).std(ddof=0)
    out = s.groupby(by).rolling(window).std(ddof=0)
    out.index = out.index.droplevel(0)
    return out


def _rolling_min(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    if by is None:
        return s.rolling(window).min()
    out = s.groupby(by).rolling(window).min()
    out.index = out.index.droplevel(0)
    return out


def _rolling_max(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    if by is None:
        return s.rolling(window).max()
    out = s.groupby(by).rolling(window).max()
    out.index = out.index.droplevel(0)
    return out


def rsi(close: pd.Series, by: pd.Series | None = None, period: int = 14) -> pd.Series:
    """Wilder's RSI. Smoothing is an EMA with alpha = 1/period, not a mean."""
    delta = close.groupby(by).diff() if by is not None else close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = _ewm(gain, by, alpha=1 / period, adjust=False)
    avg_loss = _ewm(loss, by, alpha=1 / period, adjust=False)
    # The plain formula already handles the edges: no losses gives rs = inf and
    # so RSI 100; no gains gives rs = 0 and RSI 0; a completely flat stretch
    # gives 0/0 = NaN, which is the honest answer rather than a made-up 50.
    return 100 - 100 / (1 + avg_gain / avg_loss)


def atr(
    df: pd.DataFrame, by: pd.Series | None = None, period: int = 14
) -> pd.Series:
    """Average true range in price units, Wilder-smoothed.

    Exposed because Stage 7 sizes its targets and stops in ATR, and needs the
    absolute figure rather than the percentage the context table reports.
    """
    return _ewm(true_range(df, by), by, alpha=1 / period, adjust=False)


def _bar_number(s: pd.Series, by: pd.Series | None) -> pd.Series:
    """How many bars this symbol has printed up to and including this one."""
    if by is None:
        return pd.Series(range(1, len(s) + 1), index=s.index)
    return s.groupby(by).cumcount() + 1


def true_range(df: pd.DataFrame, by: pd.Series | None = None) -> pd.Series:
    """max(high-low, |high - prev close|, |low - prev close|)."""
    d = g.normalise(df)
    prev_close = g.shift_by(d["close"], 1, by)
    return pd.concat(
        [
            d["high"] - d["low"],
            (d["high"] - prev_close).abs(),
            (d["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def annotate(
    bars: pd.DataFrame,
    *,
    by: pd.Series | None = None,
    ema_slow: int = 200,
    ema_ribbon: tuple[int, ...] = (5, 13, 25),
    volume_lookback: int = 20,
    atr_period: int = 14,
    bb_period: int = 20,
    bb_sigma: float = 2.0,
    rsi_period: int = 14,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    stoch_period: int = 14,
    stoch_smooth: int = 3,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Context columns for every row of `bars`, indexed like `bars`.

    A column is NaN until the symbol has enough history for it to mean
    something -- an EMA-200 computed over 30 bars is a number, not an answer.
    Callers get an honest gap rather than a misleading value.
    """
    d = g.normalise(bars)
    if by is not None and not by.index.equals(bars.index):
        raise ValueError("`by` must share the index of `bars`")

    n = _bar_number(d["close"], by)
    out = pd.DataFrame(index=bars.index)

    close = d["close"]

    # --- trend: one slow anchor plus the 5/13/25 ribbon --------------------
    slow = _ewm(close, by, span=ema_slow, adjust=False)
    out["above_200ema"] = (close > slow).where(n >= ema_slow)
    out["dist_200ema_pct"] = ((close / slow - 1) * 100).where(n >= ema_slow)

    ribbon = {}
    for span in ema_ribbon:
        e = _ewm(close, by, span=span, adjust=False)
        ribbon[span] = e
        out[f"dist_{span}ema_pct"] = ((close / e - 1) * 100).where(n >= span)

    # The ribbon's ordering is what a trader reads off it at a glance: fastest
    # on top is an uptrend, fastest on the bottom a downtrend, anything else is
    # a transition. One column instead of comparing three.
    fastest, middle, slowest = (ribbon[s] for s in sorted(ema_ribbon))
    stacked_up = (fastest > middle) & (middle > slowest)
    stacked_down = (fastest < middle) & (middle < slowest)
    stack = pd.Series("mixed", index=bars.index, dtype="object")
    stack = stack.mask(stacked_up, "up").mask(stacked_down, "down")
    out["ema_stack"] = stack.where(n >= max(ema_ribbon)).astype("string")

    # --- momentum ----------------------------------------------------------
    out["rsi_14"] = rsi(close, by, rsi_period).where(
        n >= rsi_period * EMA_WARMUP_MULTIPLE)
    # The number is what RSI *is*; the zone is what it *means*. RSI is read
    # against thresholds, not as a magnitude -- 45 and 55 are both "nothing to
    # see", while 29 and 31 sit either side of a line people act on. A sortable
    # zone puts the meaningful cut where you can group by it; the bar alone
    # makes the reader eyeball where 30 and 70 fall.
    zone = pd.Series("neutral", index=bars.index, dtype="object")
    zone = zone.mask(out["rsi_14"] < rsi_oversold, "oversold")
    zone = zone.mask(out["rsi_14"] > rsi_overbought, "overbought")
    out["rsi_zone"] = zone.where(out["rsi_14"].notna()).astype("string")

    lowest = _rolling_min(d["low"], by, stoch_period)
    highest = _rolling_max(d["high"], by, stoch_period)
    span = (highest - lowest).replace(0, np.nan)    # a flat window has no %K
    raw_k = (close - lowest) / span * 100
    # "Slow" stochastic: %K is the smoothed raw line, %D smooths it again.
    out["stoch_k"] = _rolling_mean(raw_k, by, stoch_smooth)
    out["stoch_d"] = _rolling_mean(out["stoch_k"], by, stoch_smooth)

    macd_line = (
        _ewm(close, by, span=macd_fast, adjust=False)
        - _ewm(close, by, span=macd_slow, adjust=False)
    )
    signal = _ewm(macd_line, by, span=macd_signal, adjust=False)
    enough_macd = n >= macd_slow * EMA_WARMUP_MULTIPLE
    out["macd"] = macd_line.where(enough_macd)
    out["macd_signal"] = signal.where(enough_macd)
    # The histogram in price units is not sortable across a screener, so it is
    # reported as a percentage of price. The two raw lines are kept because
    # that is what people expect to read.
    out["macd_hist_pct"] = ((macd_line - signal) / close * 100).where(enough_macd)

    # --- volatility --------------------------------------------------------
    # Wilder's smoothing: an EMA with alpha = 1/period, which is what every
    # charting package means by "ATR(14)".
    out["atr_pct"] = (atr(d, by, atr_period) / close * 100).where(
        n >= atr_period * EMA_WARMUP_MULTIPLE)

    mid = _rolling_mean(close, by, bb_period)
    sd = _rolling_std(close, by, bb_period)
    upper, lower = mid + bb_sigma * sd, mid - bb_sigma * sd
    width = (upper - lower).replace(0, np.nan)
    # %B: 0 at the lower band, 1 at the upper, outside those when price breaks
    # out. Comparable across symbols in a way the raw bands are not.
    out["bb_pct_b"] = (close - lower) / width
    out["bb_width_pct"] = (width / mid * 100)

    # --- volume ------------------------------------------------------------
    if "volume" in bars.columns:
        avg_vol = _rolling_mean(bars["volume"].astype("float64"), by, volume_lookback)
        out["rel_volume"] = bars["volume"] / avg_vol.replace(0, np.nan)
    else:
        out["rel_volume"] = pd.NA

    return out[CONTEXT_COLUMNS]


def annotate_by_symbol(
    bars: pd.DataFrame, *, symbol_col: str = "symbol", date_col: str = "date", **kw
) -> pd.DataFrame:
    """`annotate` over a frame straight from store.read().

    Sorts within each symbol, computes in one grouped pass, and restores the
    caller's row order.
    """
    if symbol_col not in bars.columns:
        return annotate(bars, **kw)
    if date_col not in bars.columns:
        raise ValueError(f"need a {date_col!r} column to order bars within a symbol")
    if bars.empty:
        return pd.DataFrame(
            {c: pd.Series(dtype="float64") for c in CONTEXT_COLUMNS}, index=bars.index
        )

    ordered = bars.sort_values([symbol_col, date_col])
    return annotate(ordered, by=ordered[symbol_col], **kw).reindex(bars.index)
