"""Divergence: price makes a new extreme, the oscillator does not.

The decision sheet's "Divergence" row. Price and momentum disagreeing at two
consecutive swings is the classic reversal tell, and it is also the single most
commonly mis-backtested signal in technical analysis -- because the swings it
compares are only knowable after the fact. `_swings.py` is where that is dealt
with; everything here is built on confirmed pivots and inherits the honesty.

Four readings, from two independent choices -- which extreme, and who is making
the higher one:

    regular bullish   price lower low,    oscillator higher low    reversal up
    hidden bullish    price higher low,   oscillator lower low     continuation up
    regular bearish   price higher high,  oscillator lower high    reversal down
    hidden bearish    price lower high,   oscillator higher high   continuation down

**Only the two bullish forms are registered.** The bearish pair is exported and
tested but kept out of the registry, for the same reason `heikin_ashi.flip_down`
is: every registered signal in this project is a long entry, and the Stage 7 and
Stage 9 machinery scores a signal by whether an upside target is reached before
a downside stop. Feed a bearish signal to that and it reports a hit rate which
reads like a result and measures nothing. Use `bearish_divergence()` directly
for an exit rule, an exclusion, or the sheet's field -- not as an entry.

**A divergence is an event, not a state.** It fires only on the bar that
confirms the second pivot, which is the bar a trader could first have seen it.
Letting it stay True until the next pivot would multiply the frequency by the
length of the swing and turn one observation into thirty, which flatters the
count and tells Stage 7 something false about how often the thing happens.

The oscillator comes from `context.py` rather than being reimplemented here, so
the detector and the RSI column on the chart cannot disagree about the same bar.
The import is deferred to the call: `context` imports `patterns._geometry`, and
importing it at module scope would close the cycle.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g
from . import _swings as sw
from .registry import register

OSCILLATORS = ("rsi", "macd_hist")


def _oscillator(
    df: pd.DataFrame, name: str, by: pd.Series | None, rsi_period: int
) -> pd.Series:
    from .. import context as ctx  # deferred: see the module docstring

    d = g.normalise(df)
    if name == "rsi":
        return ctx.rsi(d["close"], by, rsi_period)
    if name == "macd_hist":
        line, signal = ctx.macd(d["close"], by)
        return line - signal
    raise ValueError(f"unknown oscillator {name!r}; choose from {OSCILLATORS}")


def _divergence(
    df: pd.DataFrame,
    *,
    kind: str,
    oscillator: str,
    left: int,
    right: int,
    rsi_period: int,
    min_bars_between: int,
    max_bars_between: int,
    by: pd.Series | None,
) -> pd.Series:
    """One implementation, four readings. See the module docstring for the table."""
    if kind not in ("bullish", "hidden_bullish", "bearish", "hidden_bearish"):
        raise ValueError(f"unknown divergence kind {kind!r}")

    d = g.normalise(df)
    piv = sw.pivots(d, left=left, right=right, by=by)
    osc = _oscillator(d, oscillator, by, rsi_period)

    at_lows = kind in ("bullish", "hidden_bullish")
    confirmed = piv["pivot_low"] if at_lows else piv["pivot_high"]
    price_at = d["low"] if at_lows else d["high"]

    price = sw.track(price_at, confirmed, right=right, by=by)
    momentum = sw.track(osc, confirmed, right=right, by=by)

    # The four readings are two questions crossed, not four separate rules.
    # `at_lows` above answered the first -- which pivots to read. What is left is
    # only who made the more extreme move:
    #
    #   price extends, momentum does not  ->  regular  (bullish at lows,
    #                                                   bearish at highs)
    #   momentum extends, price does not  ->  hidden   (likewise)
    #
    # "Extends" means further in the direction the pivot sits: lower at a low,
    # higher at a high. Writing it this way rather than as four branches is not
    # only shorter -- it is the reason regular-bullish and regular-bearish are
    # the same arithmetic, which the four-branch form hides.
    if at_lows:
        price_extends = price["last"] < price["prev"]
        osc_extends = momentum["last"] < momentum["prev"]
    else:
        price_extends = price["last"] > price["prev"]
        osc_extends = momentum["last"] > momentum["prev"]

    regular = kind in ("bullish", "bearish")
    agree = (
        price_extends & ~osc_extends if regular else ~price_extends & osc_extends
    )

    # Two pivots a year apart are not a divergence, and two a few bars apart are
    # the same swing counted twice. `min_bars_between` cannot go below what the
    # pivot geometry already enforces; it is here so the floor is stated rather
    # than inherited by accident.
    span = price["last_at"] - price["prev_at"]
    # Both pivots must be real numbers on both series. `~osc_extends` is True for
    # a NaN comparison, so without this a bar with no oscillator reading yet
    # would satisfy the "momentum did not extend" half for free.
    known = (
        price["last"].notna()
        & price["prev"].notna()
        & momentum["last"].notna()
        & momentum["prev"].notna()
    )
    hit = confirmed & agree & known & span.between(min_bars_between, max_bars_between)
    return hit.fillna(False).astype(bool)


@register("bullish_divergence", kind="single", direction="bullish", specificity=75)
def bullish_divergence(
    df: pd.DataFrame,
    *,
    oscillator: str = "rsi",
    left: int = 5,
    right: int = 5,
    rsi_period: int = 14,
    min_bars_between: int = 5,
    max_bars_between: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price prints a lower swing low; the oscillator prints a higher one.

    The textbook reversal signal: the second leg down went further in price but
    not in momentum, so the selling behind it was weaker.
    """
    return _divergence(
        df, kind="bullish", oscillator=oscillator, left=left, right=right,
        rsi_period=rsi_period, min_bars_between=min_bars_between,
        max_bars_between=max_bars_between, by=by,
    )


@register(
    "hidden_bullish_divergence", kind="single", direction="bullish", specificity=80
)
def hidden_bullish_divergence(
    df: pd.DataFrame,
    *,
    oscillator: str = "rsi",
    left: int = 5,
    right: int = 5,
    rsi_period: int = 14,
    min_bars_between: int = 5,
    max_bars_between: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price holds a higher swing low while the oscillator makes a lower one.

    A continuation reading rather than a reversal one: the pullback was deeper
    in momentum than in price, which is what a shallow dip inside an uptrend
    looks like from the oscillator's side.
    """
    return _divergence(
        df, kind="hidden_bullish", oscillator=oscillator, left=left, right=right,
        rsi_period=rsi_period, min_bars_between=min_bars_between,
        max_bars_between=max_bars_between, by=by,
    )


# --- bearish: exported, deliberately not registered -------------------------
# See the module docstring. These are short-side readings and the measurement
# rig scores long entries; registering them would produce a number that looks
# like a hit rate and is not one.


def bearish_divergence(
    df: pd.DataFrame,
    *,
    oscillator: str = "rsi",
    left: int = 5,
    right: int = 5,
    rsi_period: int = 14,
    min_bars_between: int = 5,
    max_bars_between: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price prints a higher swing high; the oscillator prints a lower one."""
    return _divergence(
        df, kind="bearish", oscillator=oscillator, left=left, right=right,
        rsi_period=rsi_period, min_bars_between=min_bars_between,
        max_bars_between=max_bars_between, by=by,
    )


def hidden_bearish_divergence(
    df: pd.DataFrame,
    *,
    oscillator: str = "rsi",
    left: int = 5,
    right: int = 5,
    rsi_period: int = 14,
    min_bars_between: int = 5,
    max_bars_between: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price makes a lower swing high while the oscillator makes a higher one."""
    return _divergence(
        df, kind="hidden_bearish", oscillator=oscillator, left=left, right=right,
        rsi_period=rsi_period, min_bars_between=min_bars_between,
        max_bars_between=max_bars_between, by=by,
    )


# --- the checklist's reading: agreement with the trade ----------------------
# The checklist does not treat a divergence as an entry. It asks whether the
# divergence points the same way as the trade you are already considering --
# with it is encouragement, against it is a caution flag. That is why the
# bearish readings above had to exist even though nothing registers them: on a
# long, a *bearish* divergence is the one that matters.

WITH = "with"
AGAINST = "against"


def agreement(
    df: pd.DataFrame,
    direction: str = "long",
    *,
    oscillator: str = "rsi",
    left: int = 5,
    right: int = 5,
    rsi_period: int = 14,
    min_bars_between: int = 5,
    max_bars_between: int = 60,
    by: pd.Series | None = None,
) -> pd.Series:
    """Does any divergence on this bar point the way `direction` does?

    Returns "with", "against", or NA where no divergence fired. Both hidden and
    regular readings count: they differ in what they say about the trend, not in
    which way they lean.

    When a bullish and a bearish divergence land on the same bar -- possible,
    since they are read off different pivots -- "against" wins. The grade exists
    to raise a caution, and a caution that a second signal can cancel is not one.
    """
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    kw = {
        "oscillator": oscillator, "left": left, "right": right,
        "rsi_period": rsi_period, "min_bars_between": min_bars_between,
        "max_bars_between": max_bars_between, "by": by,
    }
    up = bullish_divergence(df, **kw) | hidden_bullish_divergence(df, **kw)
    down = bearish_divergence(df, **kw) | hidden_bearish_divergence(df, **kw)
    favourable, adverse = (up, down) if direction == "long" else (down, up)

    out = pd.Series(pd.NA, index=df.index, dtype="string")
    out = out.mask(favourable, WITH)
    out = out.mask(adverse, AGAINST)      # applied second, so it wins a tie
    return out
