"""Candle geometry, following TA-Lib's reference implementation.

The important idea, and the one that separates this from a naive detector: a
candle is not judged against fixed ratios of its own range. It is judged
against a rolling average of the bars before it. "Short body" means short
*for this stock lately*, not "below 35% of today's range".

That matters because it adapts. A 2% body is small for a volatile smallcap and
large for a utility, and a fixed threshold gets both wrong.

TA_CANDLEAVERAGE, from ta_utility.h, is:

    factor * (avgPeriod != 0 ? sum(range over prior avgPeriod bars) / avgPeriod
                             : range of the current bar)
           / (rangeType == Shadows ? 2 : 1)

with avgPeriod == 0 meaning "measure against this bar itself".
"""

from __future__ import annotations

import pandas as pd

OHLC = ("open", "high", "low", "close")


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    """Return `df` with OHLC columns lowercased, or raise if any are missing.

    Bhavcopy ships uppercase headers; hand-written test frames use lowercase.
    Accept both rather than making every caller remember which.
    """
    renamed = {c: c.lower() for c in df.columns if c.lower() in OHLC}
    out = df.rename(columns=renamed)
    missing = [c for c in OHLC if c not in out.columns]
    if missing:
        raise ValueError(f"missing OHLC column(s): {', '.join(missing)}")
    return out


def metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar measurements shared by every detector.

    `has_range` is False for a bar where high == low (a halted or frozen
    session). Detectors AND against it so those bars never register.
    """
    d = normalise(df)
    m = pd.DataFrame(index=d.index)
    m["body_top"] = d[["open", "close"]].max(axis=1)
    m["body_bottom"] = d[["open", "close"]].min(axis=1)
    m["real_body"] = (d["close"] - d["open"]).abs()
    m["high_low"] = d["high"] - d["low"]
    m["upper"] = d["high"] - m["body_top"]
    m["lower"] = m["body_bottom"] - d["low"]
    m["has_range"] = m["high_low"] > 0
    # TA-Lib treats a zero-body candle as white: `close >= open`.
    m["is_white"] = d["close"] >= d["open"]
    m["is_black"] = d["close"] < d["open"]
    return m


def shift_by(s: pd.Series, n: int = 1, by: pd.Series | None = None) -> pd.Series:
    """`s.shift(n)`, but never across a symbol boundary when `by` is given."""
    return s.shift(n) if by is None else s.groupby(by).shift(n)


def candle_average(
    values: pd.Series,
    period: int,
    factor: float,
    *,
    by: pd.Series | None = None,
    halve: bool = False,
) -> pd.Series:
    """TA_CANDLEAVERAGE.

    `period == 0` measures against the current bar rather than an average --
    that is how TA-Lib expresses "the lower shadow must exceed the real body".

    For `period > 0` the window is the `period` bars *before* this one. The
    current bar is excluded deliberately: including it would let an unusually
    long candle raise its own bar to clear.

    `by` makes the rolling window and the shift respect symbol boundaries, so a
    frame holding thousands of symbols can be measured in one pass instead of
    one pass each. `values` must already be sorted within each group.
    """
    if period < 0:
        raise ValueError(f"period must be >= 0, got {period}")

    if period == 0:
        base = values
    elif by is None:
        base = values.rolling(period).mean().shift(1)
    else:
        rolled = values.groupby(by).rolling(period).mean()
        rolled.index = rolled.index.droplevel(0)
        base = rolled.groupby(by).shift(1)

    out = factor * base
    return out / 2.0 if halve else out
