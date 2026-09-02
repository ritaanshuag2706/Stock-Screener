"""Two-bar detectors, following TA-Lib's reference implementation.

Transcribed from ta_CDLENGULFING.c.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g
from .registry import register


@register("bullish_engulfing", kind="double", direction="bullish", specificity=20)
def bullish_engulfing(df: pd.DataFrame, *, by: pd.Series | None = None) -> pd.Series:
    """An up bar whose body swallows the previous down bar's body.

    TA-Lib CDLENGULFING, bullish branch:
      this candle white:      close >= open
      previous candle black:  prev close < prev open
      and one of
        close >= prev open AND open <  prev close
        close >  prev open AND open <= prev close

    The paired inequalities are how TA-Lib allows the two bodies to touch at
    exactly one end while still requiring a strict engulf at the other. Two
    identical bodies therefore do not qualify.

    Note what is *absent*: no trend precondition and no minimum body size.
    TA-Lib takes the pattern as purely a two-bar relationship and leaves
    context to the caller — which is what Stage 6 is for. It takes no
    parameters at all, so there is nothing to tune here.
    """
    d = g.normalise(df)
    m = g.metrics(d)

    prev_open = g.shift_by(d["open"], 1, by)
    prev_close = g.shift_by(d["close"], 1, by)
    engulfs = (
        ((d["close"] >= prev_open) & (d["open"] < prev_close))
        | ((d["close"] > prev_open) & (d["open"] <= prev_close))
    )

    hit = (
        m["has_range"]
        & m["is_white"]
        & g.shift_by(m["is_black"], 1, by).fillna(False).astype(bool)
        & engulfs
    )
    return hit.fillna(False).astype(bool)
