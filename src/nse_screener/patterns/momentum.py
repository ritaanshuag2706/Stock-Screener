"""A second signal family: breakouts and pullbacks, rather than bar shapes.

The candlestick patterns measured flat at Stage 7. These are a different bet
entirely -- they read *position within a trend* rather than the geometry of one
or two bars, and they are the families with the strongest published evidence in
equities. Whether that survives on NSE daily bars over 2020-2026 is exactly what
the pipeline is for.

Same contract as every other detector: `DataFrame -> boolean Series`, pure, no
I/O, `by=` to stay inside a symbol. So Stages 6, 7 and 9 measure these unchanged.

No lookahead anywhere: every rolling window ends at the *previous* bar, so a
"new 20-day high" means today's close exceeded the highest high of the twenty
bars before today, not including today's own high.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g
from .registry import register


def _rolling_max(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    if by is None:
        return s.rolling(window).max()
    out = s.groupby(by).rolling(window).max()
    out.index = out.index.droplevel(0)
    return out


def _rolling_min(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    if by is None:
        return s.rolling(window).min()
    out = s.groupby(by).rolling(window).min()
    out.index = out.index.droplevel(0)
    return out


def _ewm(s: pd.Series, by: pd.Series | None, **kw) -> pd.Series:
    if by is None:
        return s.ewm(**kw).mean()
    out = s.groupby(by).ewm(**kw).mean()
    out.index = out.index.droplevel(0)
    return out


def _rsi(close: pd.Series, by: pd.Series | None, period: int) -> pd.Series:
    delta = close.groupby(by).diff() if by is not None else close.diff()
    gain = _ewm(delta.clip(lower=0), by, alpha=1 / period, adjust=False)
    loss = _ewm(-delta.clip(upper=0), by, alpha=1 / period, adjust=False)
    return 100 - 100 / (1 + gain / loss)


@register("breakout_20", kind="single", direction="bullish", specificity=50)
def breakout_20(
    df: pd.DataFrame, *, window: int = 20, by: pd.Series | None = None
) -> pd.Series:
    """Close above the highest high of the previous `window` bars.

    A Donchian channel break -- the classic trend-following entry. The window
    ends at the previous bar, so today's own high cannot make today a breakout.
    """
    d = g.normalise(df)
    prior_high = g.shift_by(_rolling_max(d["high"], by, window), 1, by)
    hit = d["close"] > prior_high
    return hit.fillna(False).astype(bool)


@register("breakout_252", kind="single", direction="bullish", specificity=60)
def breakout_252(
    df: pd.DataFrame, *, window: int = 252, by: pd.Series | None = None
) -> pd.Series:
    """A new 52-week high on the close. The rarest and most-studied of these."""
    d = g.normalise(df)
    prior_high = g.shift_by(_rolling_max(d["high"], by, window), 1, by)
    hit = d["close"] > prior_high
    return hit.fillna(False).astype(bool)


@register("rsi2_pullback", kind="single", direction="bullish", specificity=55)
def rsi2_pullback(
    df: pd.DataFrame,
    *,
    trend_span: int = 200,
    rsi_period: int = 2,
    rsi_max: float = 10.0,
    by: pd.Series | None = None,
) -> pd.Series:
    """A short, sharp dip inside an uptrend.

    Connors' setup: price above its long moving average (so the trend is up),
    with a 2-period RSI under 10 (so the last two days were unusually weak).
    Buying weakness in strength -- the opposite bet to a breakout.
    """
    d = g.normalise(df)
    trend = _ewm(d["close"], by, span=trend_span, adjust=False)
    hit = (d["close"] > trend) & (_rsi(d["close"], by, rsi_period) < rsi_max)
    return hit.fillna(False).astype(bool)


# Specificity above breakout_20 deliberately: a squeeze release *is* a 20-day
# breakout with an extra compression test, so it is the narrower claim and must
# win the classify tiebreak. Ranked below it, this fired 28,679 times and was
# labelled zero times -- every hit was absorbed by the broader signal.
@register("squeeze_release", kind="single", direction="bullish", specificity=65)
def squeeze_release(
    df: pd.DataFrame,
    *,
    window: int = 20,
    lookback: int = 126,
    quantile: float = 0.25,
    by: pd.Series | None = None,
) -> pd.Series:
    """Volatility was unusually tight, and price has just broken the range.

    Bollinger width in the bottom `quantile` of its own recent history on the
    previous bar, and today closing above that bar's `window` high. The premise
    is that compression precedes expansion; the direction is taken from the break.
    """
    d = g.normalise(df)
    mid = (
        d["close"].rolling(window).mean() if by is None
        else d["close"].groupby(by).rolling(window).mean().droplevel(0)
    )
    sd = (
        d["close"].rolling(window).std(ddof=0) if by is None
        else d["close"].groupby(by).rolling(window).std(ddof=0).droplevel(0)
    )
    width = (4 * sd / mid)
    rank = (
        width.rolling(lookback).rank(pct=True) if by is None
        else width.groupby(by).rolling(lookback).rank(pct=True).droplevel(0)
    )
    was_tight = g.shift_by(rank, 1, by) <= quantile
    prior_high = g.shift_by(_rolling_max(d["high"], by, window), 1, by)
    hit = was_tight & (d["close"] > prior_high)
    return hit.fillna(False).astype(bool)
