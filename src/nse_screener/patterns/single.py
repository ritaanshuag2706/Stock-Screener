"""Single-bar detectors, following TA-Lib's reference implementations.

Conditions are transcribed from the TA-Lib C sources (ta_CDLDOJI.c,
ta_CDLHAMMER.c, ta_CDLINVERTEDHAMMER.c) and the default settings table in
ta_common/ta_global.c. Parameter names map onto TA-Lib's candle settings so
the two can be compared directly.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g
from .registry import register


# Least specific of the four: one condition, fires on ~1 bar in 7. Loses
# any tie, because "small body" says less than the patterns that also
# constrain the shadows and the gap.
@register("doji", kind="single", direction="neutral", specificity=10)
def doji(
    df: pd.DataFrame,
    *,
    body_doji_period: int = 10,
    body_doji_factor: float = 0.1,
    by: pd.Series | None = None,
) -> pd.Series:
    """Open and close effectively equal.

    TA-Lib CDLDOJI:  real body <= BodyDoji average

    BodyDoji is 10% of the average high-low range over the prior 10 bars, so
    "effectively equal" scales with how much the stock has been moving.
    """
    m = g.metrics(df)
    limit = g.candle_average(m["high_low"], body_doji_period, body_doji_factor, by=by)
    hit = m["has_range"] & (m["real_body"] <= limit)
    return hit.fillna(False).astype(bool)


@register("hammer", kind="single", direction="bullish", specificity=30)
def hammer(
    df: pd.DataFrame,
    *,
    body_short_period: int = 10,
    body_short_factor: float = 1.0,
    shadow_long_period: int = 0,
    shadow_long_factor: float = 1.0,
    shadow_very_short_period: int = 10,
    shadow_very_short_factor: float = 0.1,
    near_period: int = 5,
    near_factor: float = 0.2,
    require_near_low: bool = False,
    rule: str = "classic",
    body_bottom_min_frac: float = 0.33,
    min_lower_body_ratio: float = 2.0,
    by: pd.Series | None = None,
) -> pd.Series:
    """Small body at the top of the range, long lower shadow.

    The mirror of `inverted_hammer`, with the same two readings.

    rule="talib" -- TA-Lib CDLHAMMER:
        lower shadow > ShadowLong average
        upper shadow < ShadowVeryShort average
        min(open, close) <= previous low + Near average   (require_near_low)

    rule="classic" -- the textbook definition, and the default here:
        the whole body sits within `body_bottom_min_frac` of the range below the
        high, and the lower shadow is at least `min_lower_body_ratio` times the
        body.

    Same reasoning as the inverted hammer: TA-Lib measures "little or no upper
    shadow" against a fraction of the *average* range, which rejects a bar whose
    body is plainly at the top if that bar happens to be wider than its
    neighbours. The classic rule asks where the body sits within its own range.

    Measured on 300 liquid NSE symbols since 2024:

        talib (shadow averages + near prior low)     1.83% of bars
        classic (body in upper third)                4.35%
        classic + near prior low                     1.29%

    `require_near_low` is off by default, so the classic reading is pure
    geometry and symmetrical with the inverted hammer. Turning it on is worth
    considering -- "is this candle actually sitting at the recent low" is a
    better reversal test than any trend proxy -- but it belongs to Stage 6's
    context columns, measured rather than assumed.
    """
    if rule not in ("classic", "talib"):
        raise ValueError(f"rule must be 'classic' or 'talib', got {rule!r}")

    d = g.normalise(df)
    m = g.metrics(d)
    body_short = g.candle_average(
        m["real_body"], body_short_period, body_short_factor, by=by)
    hit = m["has_range"] & (m["real_body"] < body_short)

    if rule == "talib":
        shadow_long = g.candle_average(
            m["real_body"], shadow_long_period, shadow_long_factor, by=by)
        shadow_short = g.candle_average(
            m["high_low"], shadow_very_short_period, shadow_very_short_factor, by=by)
        hit &= (m["lower"] > shadow_long) & (m["upper"] < shadow_short)
    else:
        body_in_upper = (
            (d["high"] - m["body_bottom"]) <= body_bottom_min_frac * m["high_low"]
        )
        hit &= body_in_upper & (m["lower"] >= min_lower_body_ratio * m["real_body"])

    if require_near_low:
        # TA-Lib evaluates the Near average at bar i-1, hence the extra shift.
        near = g.shift_by(
            g.candle_average(m["high_low"], near_period, near_factor, by=by), 1, by)
        hit &= m["body_bottom"] <= g.shift_by(d["low"], 1, by) + near
    return hit.fillna(False).astype(bool)


# Most specific: four geometric conditions plus a strict gap below the
# previous body.
@register("inverted_hammer", kind="single", direction="bullish", specificity=40)
def inverted_hammer(
    df: pd.DataFrame,
    *,
    body_short_period: int = 10,
    body_short_factor: float = 1.0,
    shadow_long_period: int = 0,
    shadow_long_factor: float = 1.0,
    shadow_very_short_period: int = 10,
    shadow_very_short_factor: float = 0.1,
    require_gap_down: bool = False,
    rule: str = "classic",
    body_top_max_frac: float = 0.33,
    min_upper_body_ratio: float = 2.0,
    by: pd.Series | None = None,
) -> pd.Series:
    """Small body at the bottom of the range, long upper shadow.

    Two readings, selected by `rule`. Both require a short real body and share
    the `require_gap_down` switch.

    rule="talib" -- TA-Lib CDLINVERTEDHAMMER:
        upper shadow > ShadowLong average
        lower shadow < ShadowVeryShort average
        real body entirely below the previous body   (require_gap_down)

    rule="classic" -- the textbook definition, and the default here:
        the whole body sits within `body_top_max_frac` of the range above the
        low, and the upper shadow is at least `min_upper_body_ratio` times the
        body.

    The difference is what "little or no lower shadow" is measured against.
    TA-Lib compares it to a fraction of the *average* range, which rejects a
    bar whose body is plainly at the bottom if that bar happens to be wider
    than its neighbours. The classic rule asks where the body sits within its
    own range, which is the question the pattern is actually about.

    It is also stricter, which is the good surprise. Measured on 300 liquid NSE
    symbols since 2024:

        talib, shadow factor 0.10      5.53% of bars
        talib, shadow factor 0.20     14.00%      (loosened to admit a real bar)
        classic, lower third          4.78%      (admits the same bar)

    Loosening TA-Lib's threshold far enough to accept RELIANCE 2026-07-16 --
    body 6% of range sitting 21% above the low, upper shadow 73% -- makes the
    pattern fire on one bar in seven. The classic rule accepts it while firing
    less often than the strict TA-Lib setting does.
    """
    if rule not in ("classic", "talib"):
        raise ValueError(f"rule must be 'classic' or 'talib', got {rule!r}")

    d = g.normalise(df)
    m = g.metrics(d)
    body_short = g.candle_average(
        m["real_body"], body_short_period, body_short_factor, by=by)
    hit = m["has_range"] & (m["real_body"] < body_short)

    if rule == "talib":
        shadow_long = g.candle_average(
            m["real_body"], shadow_long_period, shadow_long_factor, by=by
        )
        shadow_short = g.candle_average(
            m["high_low"], shadow_very_short_period, shadow_very_short_factor, by=by
        )
        hit &= (m["upper"] > shadow_long) & (m["lower"] < shadow_short)
    else:
        body_in_lower = (m["body_top"] - d["low"]) <= body_top_max_frac * m["high_low"]
        hit &= body_in_lower & (m["upper"] >= min_upper_body_ratio * m["real_body"])

    if require_gap_down:
        hit &= m["body_top"] < g.shift_by(m["body_bottom"], 1, by)
    return hit.fillna(False).astype(bool)
