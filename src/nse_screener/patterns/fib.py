"""Fibonacci retracement: where price sits inside the last confirmed swing leg.

The decision sheet has a "Fib Retrace" row, filled by hand off a chart. This is
that row, computed -- with the one difference that a hand-drawn leg is chosen
after the fact and this one cannot be. `_swings.py` carries the argument for why
that difference matters more than it sounds.

A leg is the last confirmed pivot low and the last confirmed pivot high, in
whichever order they happened. `retracement` is the fraction of that leg price
has given back: 0 at the extreme the leg ended on, 1 back where it started. An
up-leg giving back 0.618 is price falling into the golden zone; a down-leg
giving back 0.618 is price rallying into it. One number, direction-aware, which
is how the level is read off a chart.

Only the *long* case is registered, because every registered signal in this
project is an entry and the backtest is long-only -- a bearish retracement
measured with an upside target would report a hit rate that means nothing.
`swing_leg()` is exported for both directions, and for the sheet's own field.

What this deliberately does not do: draw the leg to make the level fit. The leg
is whatever the last two confirmed pivots say it is. Choosing the pair that puts
price on a Fibonacci number is how retracement analysis becomes unfalsifiable,
and it is available to anyone with a chart and a mouse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _geometry as g
from . import _swings as sw
from .registry import load_params, register

LEVELS = (0.382, 0.5, 0.618)
"""The golden zone. 0.236 and 0.786 are conventional too and are left out of the
default: the wider the set, the closer "price is at a Fib level" gets to "price
is somewhere in the leg", and with a tolerance either side five levels already
cover most of it."""


def swing_leg(
    df: pd.DataFrame,
    *,
    left: int = 5,
    right: int = 5,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """The last confirmed leg, and how far price has retraced into it.

    Columns, all readable on every bar:

        leg_low, leg_high   the two confirmed pivots bounding the leg
        leg_up              True if the low came first, so the leg rose
        leg_height_pct      the leg's size as a percentage of its low
        bars_since          bars since the later of the two pivots printed
        retracement         fraction of the leg given back, 0 to 1

    NaN until a symbol has printed one confirmed pivot of each kind. Nothing
    here reads a bar later than the one it is reported on.
    """
    d = g.normalise(df)
    piv = sw.pivots(d, left=left, right=right, by=by)
    hi = sw.track(d["high"], piv["pivot_high"], right=right, by=by)
    lo = sw.track(d["low"], piv["pivot_low"], right=right, by=by)

    height = (hi["last"] - lo["last"]).replace(0, np.nan)
    # A leg needs one confirmed pivot of each kind, and a non-zero height.
    known = height.notna()

    out = pd.DataFrame(index=df.index)
    out["leg_low"] = lo["last"].where(known)
    out["leg_high"] = hi["last"].where(known)
    # Stays a real boolean rather than becoming object-with-NaN: "not yet known"
    # and "not an up-leg" are the same answer to every caller, and a nullable
    # boolean here would push a `.fillna(False)` into each of them.
    out["leg_up"] = (lo["last_at"] < hi["last_at"]) & known
    out["leg_height_pct"] = (height / lo["last"] * 100).where(known)

    ordinal = sw._ordinal(d["close"], by)
    out["bars_since"] = (
        ordinal - pd.concat([hi["last_at"], lo["last_at"]], axis=1).max(axis=1)
    ).where(known)

    # Direction-aware: an up-leg is retraced downward from its high, a down-leg
    # upward from its low. Both answer "how much of the move has been undone".
    given_back = pd.Series(
        np.where(out["leg_up"], hi["last"] - d["close"], d["close"] - lo["last"]),
        index=df.index,
    )
    out["retracement"] = (given_back / height).where(known)
    return out


@register("fib_retracement", kind="single", direction="bullish", specificity=70)
def fib_retracement(
    df: pd.DataFrame,
    *,
    levels: tuple[float, ...] | list[float] = LEVELS,
    tolerance: float = 0.02,
    left: int = 5,
    right: int = 5,
    min_leg_pct: float = 5.0,
    max_bars_since: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price pulling back into a Fibonacci level of the last confirmed up-leg.

    `tolerance` is a fraction of the **leg**, not of price. A fixed percentage
    of price is a different-sized band on a 40-rupee leg than on a 400-rupee
    one, so the same setting would be strict on wide swings and meaningless on
    narrow ones. As a fraction of the leg, 0.02 is the same slice of the move
    everywhere -- and it is what the level is a fraction of in the first place.

    `min_leg_pct` throws out legs too small to retrace meaningfully: on a 1%
    swing every Fibonacci level is inside the spread. `max_bars_since` throws
    out stale ones -- a leg whose pivots printed a year ago is not what anyone
    means by "the retracement".
    """
    leg = swing_leg(df, left=left, right=right, by=by)

    near = pd.Series(False, index=df.index)
    for level in levels:
        near |= (leg["retracement"] - level).abs() <= tolerance

    hit = (
        leg["leg_up"]
        & near
        # Below the leg low the retracement is over and the swing has failed;
        # above the high it is not a retracement at all, it is a continuation.
        & leg["retracement"].between(0.0, 1.0)
        & (leg["leg_height_pct"] >= min_leg_pct)
        & (leg["bars_since"] <= max_bars_since)
    )
    return hit.fillna(False).astype(bool)


# --- the checklist's reading of the same number -----------------------------
# The decision checklist does not use a Fibonacci level as a trigger. It uses
# the retracement *depth* to grade a continuation setup you already have:
# shallow is healthy, deep is a reason to be careful. That is a different
# question from `fib_retracement` above, and both are worth having -- one asks
# "is price at a level", the other "how much of the move has been given back".

HEALTHY = "healthy"
DEEP = "deep"
WATCHFUL = "watchful"


def retracement_grade(
    df: pd.DataFrame,
    *,
    left: int = 5,
    right: int = 5,
    healthy_max: float | None = None,
    watchful_min: float | None = None,
    by: pd.Series | None = None,
) -> pd.Series:
    """Grade how deep the retracement into the last confirmed leg is.

    The checklist names two grades and leaves a gap between them: up to 50% is
    healthy, 61.8% or more is watchful, and nothing is said about the band in
    between. Rather than fold that band silently into one side, it gets its own
    label -- `deep`. Inventing a rule and hiding it inside a boundary is how a
    method turns into a different method without anyone deciding to change it.

    NA where no leg is known yet, and NA beyond the leg entirely (a retracement
    over 100% is a failed swing, not a deep one).

    Note the checklist scopes this to continuation setups. Nothing here checks
    for one, because "continuation" is a judgement about structure this file
    does not make -- the grade is reported and the caller decides whether the
    setup it belongs to exists.
    """
    cfg = load_params().get("fib_grade", {})
    healthy_max = cfg.get("healthy_max", 0.50) if healthy_max is None else healthy_max
    watchful_min = (
        cfg.get("watchful_min", 0.618) if watchful_min is None else watchful_min
    )
    if not 0 < healthy_max <= watchful_min < 1:
        raise ValueError(
            f"need 0 < healthy_max <= watchful_min < 1, got {healthy_max} and "
            f"{watchful_min}"
        )

    r = swing_leg(df, left=left, right=right, by=by)["retracement"]
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    inside = r.between(0.0, 1.0)
    out = out.mask(inside & (r <= healthy_max), HEALTHY)
    out = out.mask(inside & (r > healthy_max) & (r < watchful_min), DEEP)
    out = out.mask(inside & (r >= watchful_min), WATCHFUL)
    return out
