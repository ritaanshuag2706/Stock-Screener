"""Classical chart patterns, from published definitions rather than invented ones.

The SMM checklist leans on a family this project did not have: head and
shoulders, double tops and bottoms, rounding turns, cup and handle, flags, and
failed breakouts. They are the patterns most often drawn by eye and least often
defined, so each one here cites where its rule came from and says plainly where
the source stopped being specific and a choice had to be made.

Two sources carry most of it:

**Lo, Mamaysky and Wang (2000), "Foundations of Technical Analysis"**, Journal of
Finance 55(4). The only rigorous formalisation of these shapes in the
literature: a pattern is a condition on a sequence of consecutive local extrema
E1..E5. Their head-and-shoulders is exactly

    E1 is a maximum;  E3 > E1;  E3 > E5;
    E1 and E5 are within 1.5 percent of their average;
    E2 and E4 are within 1.5 percent of their average

and double tops require the two peaks to be at least 22 trading days apart. Those
numbers are theirs, not mine, and they are in `config/patterns.yaml`.

**Bulkowski, thepatternsite.com**, for the shapes Lo et al. do not cover -- cup
with handle, flags, rounding bottoms. His guidelines are deliberately loose
("allow variations", "be flexible", "cup rims should be near the same price
level but be flexible"), which is honest for a human reading a chart and
unusable for a detector. Where a threshold had to be invented to make his
description mechanical, the docstring says so.

One difference from Lo et al. worth stating. They smooth prices with a kernel
regression first, which guarantees that extrema alternate. These read raw
confirmed pivots from `_swings.py`, which do not alternate on their own -- so
alternation is checked as a condition, and a messy pivot sequence yields no
pattern rather than a forced one. That trades some recall for never inventing a
shape, which is the right way round.

Everything inherits `_swings.py`'s no-lookahead property: a pattern is reported
on the bar that confirms its last extremum, never on the bar the extremum sits
on. These detectors are therefore late by construction, and a chart drawn after
the fact will always look like they were slow.

**The bearish half is exported and not registered**, as with `divergence.py`:
every registered signal here is a long entry, and Stages 7 and 9 score a signal
by whether an upside target is hit before a downside stop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import _geometry as g
from . import _swings as sw
from .registry import register

# --------------------------------------------------------------------------
# shared machinery
# --------------------------------------------------------------------------


def _within(a: pd.Series, b: pd.Series, tol: float) -> pd.Series:
    """Lo et al.'s "within `tol` of their average", read literally.

    Both values must sit within `tol` of their mean. Since the mean is the
    midpoint, each is |a - b| / 2 away from it, so the condition reduces to
    |a - b| <= tol * (a + b) -- that is, they may differ by up to *twice* `tol`
    as a fraction of their average. Writing it out matters: reading the phrase
    as "|a - b| <= tol * avg" halves the tolerance and roughly halves the number
    of patterns found.
    """
    return (a - b).abs() <= tol * (a + b)


def _rolling(s: pd.Series, by: pd.Series | None, window: int, how: str) -> pd.Series:
    return sw._rolling(s, by, window, how)


def five_extrema(
    df: pd.DataFrame, *, left: int = 5, right: int = 5, by: pd.Series | None = None
) -> pd.DataFrame:
    """The last five confirmed extrema, oldest first, on every bar.

    `E1`..`E5` are prices, `K1`..`K5` their kind (1.0 maximum, 0.0 minimum) and
    `T1`..`T5` the bar each pivot sits on. `alternating` says the five run
    max/min/max/min/max or the reverse -- Lo et al. get that for free from
    kernel smoothing and raw pivots do not.
    """
    ex = sw.extrema(df, left=left, right=right, by=by)
    v = sw.carry(ex["value"], ex["confirmed"], 5, by=by)
    k = sw.carry(ex["kind"], ex["confirmed"], 5, by=by)
    t = sw.carry(ex["at"], ex["confirmed"], 5, by=by)

    out = pd.DataFrame(index=df.index)
    for i in range(1, 6):                       # E1 oldest, e5 is the oldest carried
        out[f"E{i}"] = v[f"e{6 - i}"]
        out[f"K{i}"] = k[f"e{6 - i}"]
        out[f"T{i}"] = t[f"e{6 - i}"]
    out["confirmed"] = ex["confirmed"]
    out["span"] = out["T5"] - out["T1"]
    out["alternating"] = (
        (out["K1"] != out["K2"]) & (out["K2"] != out["K3"])
        & (out["K3"] != out["K4"]) & (out["K4"] != out["K5"])
        & out["K5"].notna()
    )
    return out


def _head_and_shoulders(
    df: pd.DataFrame,
    *,
    inverted: bool,
    tolerance: float,
    left: int,
    right: int,
    max_span: int,
    by: pd.Series | None,
) -> pd.Series:
    """Lo et al. Definition 1, applied to confirmed pivots.

    `max_span` stands in for their rolling window: they fit each kernel
    regression to 38 trading days (l = 35, d = 3) precisely so that only
    patterns *completed* within that span are found. Without a cap, five extrema
    spread over two years satisfy the inequalities and mean nothing.
    """
    f = five_extrema(df, left=left, right=right, by=by)
    head_is_max = f["K1"] == 1.0
    start_ok = head_is_max if not inverted else (f["K1"] == 0.0)

    if inverted:
        head = (f["E3"] < f["E1"]) & (f["E3"] < f["E5"])
    else:
        head = (f["E3"] > f["E1"]) & (f["E3"] > f["E5"])

    hit = (
        f["confirmed"]
        & f["alternating"]
        & start_ok
        & head
        & _within(f["E1"], f["E5"], tolerance)
        & _within(f["E2"], f["E4"], tolerance)
        & (f["span"] <= max_span)
    )
    return hit.fillna(False).astype(bool)


def _double(
    df: pd.DataFrame,
    *,
    bottom: bool,
    tolerance: float,
    min_separation: int,
    left: int,
    right: int,
    by: pd.Series | None,
) -> pd.Series:
    """Lo et al. Definition 5.

    They take the *highest* subsequent maximum as the second top; this takes the
    next confirmed one. The difference matters only when a taller peak follows,
    and in that case theirs reports a pattern whose second top has not happened
    yet from the point of view of the bar being judged. The stricter reading is
    the one that can be traded.

    Their 22-day separation is Edwards and Magee's -- two tops less than a month
    apart are one top with a notch in it.
    """
    d = g.normalise(df)
    piv = sw.pivots(d, left=left, right=right, by=by)
    which = "pivot_low" if bottom else "pivot_high"
    price = d["low"] if bottom else d["high"]
    t = sw.track(price, piv[which], right=right, by=by)

    hit = (
        piv[which]
        & _within(t["last"], t["prev"], tolerance)
        & ((t["last_at"] - t["prev_at"]) >= min_separation)
        & t["prev"].notna()
    )
    return hit.fillna(False).astype(bool)


def _quadratic(
    y: pd.Series, by: pd.Series | None, window: int
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Least-squares parabola through each rolling window: (curvature, vertex, r2).

    A rounding bottom is "a rounded bowl", and the only way to test that without
    eyeballing it is to fit the bowl. Fitting one per bar with `rolling.apply`
    would be a Python call per window -- minutes over a real store. Instead the
    normal equations are assembled from rolling sums.

    With x running 0..window-1 the design matrix is the same in every window, so
    its normal-equation matrix is a constant and only the right-hand side moves.
    The moments of y against a *window-local* x come from moments against the
    global bar number j, shifted by the window's start c:

        sum (j - c)   y = sum j y   - c sum y
        sum (j - c)^2 y = sum j^2 y - 2c sum j y + c^2 sum y

    `vertex` is where the parabola turns, in window coordinates, and `r2` how
    much of the window's variance the parabola explains -- Bulkowski's "gradual
    and rounded, not sharp", made checkable.
    """
    j = sw._ordinal(y, by)
    sums = {}
    for name, series in (
        ("y", y), ("jy", j * y), ("jjy", j * j * y), ("yy", y * y)
    ):
        sums[name] = _rolling(series, by, window, "sum")

    c = j - (window - 1)                       # window-local x = j - c
    s_y = sums["y"]
    s_xy = sums["jy"] - c * s_y
    s_xxy = sums["jjy"] - 2 * c * sums["jy"] + c * c * s_y

    x = np.arange(window, dtype="float64")
    m = np.array([
        [(x ** 4).sum(), (x ** 3).sum(), (x ** 2).sum()],
        [(x ** 3).sum(), (x ** 2).sum(), x.sum()],
        [(x ** 2).sum(), x.sum(), float(window)],
    ])
    inv = np.linalg.inv(m)

    a = inv[0, 0] * s_xxy + inv[0, 1] * s_xy + inv[0, 2] * s_y
    b = inv[1, 0] * s_xxy + inv[1, 1] * s_xy + inv[1, 2] * s_y
    k = inv[2, 0] * s_xxy + inv[2, 1] * s_xy + inv[2, 2] * s_y

    ss_res = sums["yy"] - (a * s_xxy + b * s_xy + k * s_y)
    ss_tot = sums["yy"] - s_y * s_y / window
    r2 = 1 - ss_res / ss_tot.replace(0, np.nan)
    vertex = -b / (2 * a).replace(0, np.nan)
    return a, vertex, r2


def _rounding(
    df: pd.DataFrame,
    *,
    bottom: bool,
    window: int,
    rim: int,
    min_depth_pct: float,
    min_r2: float,
    vertex_band: float,
    max_base_frac: float,
    by: pd.Series | None,
) -> pd.Series:
    """Bulkowski's rounding bottom, made mechanical.

    His guidelines give the shape ("a rounded bowl"), the trend before it, and
    the confirmation -- "I use a close above the left peak because price on the
    right might not pause at a minor high" -- but no numbers at all. `min_r2`,
    `vertex_band`, `min_depth_pct` and `max_base_frac` are therefore mine, not
    his.

    **A parabola fit is not by itself a test of roundness**, which is worth
    knowing before trusting `min_r2`. Measured on a 60-bar window: a true
    parabola fits at r2 = 1.000 and a hard V-shaped bottom -- two straight legs
    meeting at a point, the thing Bulkowski explicitly excludes -- fits at
    0.937. Any threshold loose enough to admit real data admits the V too.

    What separates them is how much of the fall the *middle* of the window
    spans. A rounded turn is nearly flat through its base: the middle third of
    that same bowl covers 11% of its depth, while the V covers 35%, because the
    whole turn happens there. `max_base_frac` is that test, and it is the one
    doing the work; r2 is kept because it still rejects windows that are merely
    noisy.
    """
    d = g.normalise(df)
    close = d["close"]
    curve, vertex, r2 = _quadratic(close, by, window)

    # The left rim is the extreme of the first `rim` bars of the window, which
    # sits `window - rim` bars back from here.
    edge = _rolling(d["high"] if bottom else d["low"], by, rim, "max" if bottom else "min")
    left_rim = g.shift_by(edge, window - rim, by)
    trough = _rolling(d["low"] if bottom else d["high"], by, window, "min" if bottom else "max")

    depth = ((left_rim - trough) / trough * 100) if bottom else (
        (trough - left_rim) / left_rim * 100
    )
    # "Gradual and rounded, not sharp or pointed": the base must be flat
    # relative to the depth of the turn. See the docstring -- this, not r2, is
    # what excludes a V.
    #
    # Measured on closes, deliberately. The parabola above is fitted to closes,
    # so a high-low range here would be testing a different series than the one
    # judged round -- and the daily bar's own range then counts as unevenness in
    # the base. That mistake is not academic: with high-low this condition
    # passed 0.74% of real bars against a median ratio of 0.97, because the
    # threshold had been read off a synthetic bowl with no intrabar noise, and
    # `rounding_bottom` fired once in 921,061 bars.
    mid = max(window // 3, 2)
    base_range = g.shift_by(
        _rolling(d["close"], by, mid, "max") - _rolling(d["close"], by, mid, "min"),
        mid, by,
    )
    full_depth = (left_rim - trough) if bottom else (trough - left_rim)

    shape = (curve > 0) if bottom else (curve < 0)
    broke = (close > left_rim) if bottom else (close < left_rim)

    hit = (
        shape
        & (r2 >= min_r2)
        & (base_range <= max_base_frac * full_depth)
        & vertex.between(window * vertex_band, window * (1 - vertex_band))
        & (depth >= min_depth_pct)
        & broke
        # An event, not a state: the bar that clears the rim, not every bar after.
        & ~g.shift_by(broke, 1, by).fillna(False).astype(bool)
    )
    return hit.fillna(False).astype(bool)


def _flag(
    df: pd.DataFrame,
    *,
    bullish: bool,
    pole: int,
    flag: int,
    min_pole_pct: float,
    max_flag_frac: float,
    by: pd.Series | None,
) -> pd.Series:
    """Bulkowski's flag: a steep pole, then a small tilted rectangle, then a break.

    His guidelines are "the flagpole ... should be unusually steep and last
    several days", the flag is "a small rectangle often tilted against the
    prevailing price trend", and it must be "short, less than 3 weeks long.
    Patterns longer than that are rectangles or channels."

    **`flag` is the assumed length of the consolidation, not his maximum**, and
    the distinction is the single biggest thing in this detector. His three
    weeks is a cap; a real flag is often five or eight bars. Measuring the
    consolidation's range over a fixed fifteen-bar window therefore drags pole
    bars into it and inflates the range, and the hit count falls by a factor of
    twenty -- 59 at eight bars against 3 at fifteen, on 921,061 real bars. The
    default is eight, comfortably inside his cap. A flag that consolidated for
    materially longer is missed rather than mis-measured, which is the same
    trade `cup_and_handle` makes with its handle.

    "Unusually steep" is the part with no number in it. `min_pole_pct` is a
    plain percentage move over the pole window, and 15% is not a guess: it is
    the 95th percentile of ten-bar moves across this universe, so "unusually"
    means what it says. It is deliberately not ATR-normalised -- a flagpole is
    a move that stands out against the stock's own history rather than against
    its recent volatility, and dividing by a volatility the pole itself inflates
    hides exactly the move being looked for.

    The parallel-trendline test everyone draws is not implemented. What is
    tested is what the trendlines are there to show: that the consolidation is
    small relative to the pole, and leans against it.
    """
    d = g.normalise(df)
    close, high, low = d["close"], d["high"], d["low"]

    pole_start = g.shift_by(close, pole + flag, by)
    pole_end = g.shift_by(close, flag, by)
    pole_move = (pole_end / pole_start - 1) * 100
    pole_height = (pole_end - pole_start).abs()

    flag_high = _rolling(high, by, flag, "max")
    flag_low = _rolling(low, by, flag, "min")
    flag_range = flag_high - flag_low

    steep = (pole_move >= min_pole_pct) if bullish else (pole_move <= -min_pole_pct)
    tight = flag_range <= max_flag_frac * pole_height
    # Tilted against the trend: on the bar before the break the consolidation had
    # given ground rather than extended the pole. Without this a flag is just
    # "steep move, then a new high", which is a breakout and already registered.
    prev_close = g.shift_by(close, 1, by)
    leans = (prev_close <= pole_end) if bullish else (prev_close >= pole_end)
    prior_extreme = g.shift_by(flag_high if bullish else flag_low, 1, by)
    breaks = (close > prior_extreme) if bullish else (close < prior_extreme)

    return (steep & tight & leans & breaks).fillna(False).astype(bool)


def _failed_break(
    df: pd.DataFrame,
    *,
    upward: bool,
    lookback: int,
    recovery: int,
    by: pd.Series | None,
) -> pd.Series:
    """A breakout that gave itself back -- the PAPA sheet's Fake BO and Fake BD.

    Its wording is operational, which is why this one needs no invented
    thresholds: for a fake breakdown, "price again enters above the breakdown
    level", and the trigger is "once the follow-up candle closes above the
    breakdown level".

    `upward=False` is the bullish case: price broke *down* through support and
    failed to hold, trapping the sellers. That is the reading the SMM checklist
    marks Buy.
    """
    d = g.normalise(df)
    close = d["close"]
    level = g.shift_by(
        _rolling(d["low"] if not upward else d["high"], by, lookback,
                 "min" if not upward else "max"),
        1, by,
    )
    broke = (close < level) if not upward else (close > level)

    # The level that was broken, and how long ago -- carried forward so the
    # recovery bar can be judged against the break that produced it.
    broken_level = sw._ffill(level.where(broke), by)
    since = sw._ordinal(close, by) - sw._ffill(
        sw._ordinal(close, by).where(broke), by
    )

    back = (close > broken_level) if not upward else (close < broken_level)
    first = back & ~g.shift_by(back, 1, by).fillna(False).astype(bool)
    return (first & since.between(1, recovery)).fillna(False).astype(bool)


# --------------------------------------------------------------------------
# registered: the long side
# --------------------------------------------------------------------------


@register("fake_breakdown", kind="single", direction="bullish", specificity=85)
def fake_breakdown(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    recovery: int = 5,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price broke support and closed back above it -- a trapped-seller reversal."""
    return _failed_break(
        df, upward=False, lookback=lookback, recovery=recovery, by=by
    )


@register("flag_breakout", kind="single", direction="bullish", specificity=90)
def flag_breakout(
    df: pd.DataFrame,
    *,
    pole: int = 10,
    flag: int = 8,
    min_pole_pct: float = 15.0,
    max_flag_frac: float = 0.5,
    by: pd.Series | None = None,
) -> pd.Series:
    """A steep advance, a shallow pause, and a close above the pause's high."""
    return _flag(
        df, bullish=True, pole=pole, flag=flag, min_pole_pct=min_pole_pct,
        max_flag_frac=max_flag_frac, by=by,
    )


@register("double_bottom", kind="single", direction="bullish", specificity=95)
def double_bottom(
    df: pd.DataFrame,
    *,
    tolerance: float = 0.015,
    min_separation: int = 22,
    left: int = 5,
    right: int = 5,
    by: pd.Series | None = None,
) -> pd.Series:
    """Two confirmed lows at the same price, a month or more apart. Lo et al. Def. 5."""
    return _double(
        df, bottom=True, tolerance=tolerance, min_separation=min_separation,
        left=left, right=right, by=by,
    )


@register("rounding_bottom", kind="single", direction="bullish", specificity=100)
def rounding_bottom(
    df: pd.DataFrame,
    *,
    window: int = 60,
    rim: int = 5,
    min_depth_pct: float = 10.0,
    min_r2: float = 0.70,
    vertex_band: float = 0.25,
    max_base_frac: float = 0.25,
    by: pd.Series | None = None,
) -> pd.Series:
    """A smooth bowl, with price closing back above its left rim."""
    return _rounding(
        df, bottom=True, window=window, rim=rim, min_depth_pct=min_depth_pct,
        min_r2=min_r2, vertex_band=vertex_band, max_base_frac=max_base_frac, by=by,
    )


@register("cup_and_handle", kind="single", direction="bullish", specificity=105)
def cup_and_handle(
    df: pd.DataFrame,
    *,
    window: int = 60,
    rim: int = 5,
    handle: int = 10,
    min_depth_pct: float = 10.0,
    min_r2: float = 0.70,
    vertex_band: float = 0.25,
    max_base_frac: float = 0.25,
    max_handle_retrace: float = 0.33,
    by: pd.Series | None = None,
) -> pd.Series:
    """A rounding bottom, a shallow drift in its upper half, then a break out.

    Bulkowski's cup is "U-shaped, not V-shaped", 7 to 65 weeks long, with a
    handle of "1 week minimum with no maximum, forming in the upper half of the
    cup", and the buy comes when "price closes above the right cup rim".

    Where this departs from him, deliberately and visibly: the handle is a fixed
    `handle` bars rather than a span searched for. He gives it no maximum, and
    "no maximum" cannot be coded -- some length has to be assumed, and assuming
    one silently would be worse than saying so. The consequence is that a cup
    whose handle ran longer than `handle` bars is missed rather than
    mis-measured.

    Ranked above `rounding_bottom` because it *is* one, plus two more conditions.
    A narrower signal registered below the broader one it contains gets every hit
    stolen by it in `classify()` -- which is what happened to `squeeze_release`.
    """
    d = g.normalise(df)
    cup = _rounding(
        d, bottom=True, window=window, rim=rim, min_depth_pct=min_depth_pct,
        min_r2=min_r2, vertex_band=vertex_band, max_base_frac=max_base_frac, by=by,
    )
    # The cup completed *somewhere* in the last `handle` bars, not exactly
    # `handle` bars ago. Requiring an exact offset makes the whole detector turn
    # on whether the handle happened to run one bar long.
    cup_recent = (
        _rolling(cup.astype("float64"), by, handle, "max")
        .pipe(g.shift_by, 1, by) > 0
    )

    # The rim and the low as they stood *on the bar the cup completed*, carried
    # forward. Measuring them at a fixed offset instead only lines up when the
    # handle happens to be exactly `handle` bars long, which is the assumption
    # `cup_recent` exists to avoid making.
    rim_at_cup = sw._ffill(_rolling(d["high"], by, rim, "max").where(cup), by)
    low_at_cup = sw._ffill(_rolling(d["low"], by, window, "min").where(cup), by)
    depth = rim_at_cup - low_at_cup

    handle_low = _rolling(d["low"], by, handle, "min")
    shallow = (rim_at_cup - handle_low) <= max_handle_retrace * depth
    breaks_out = d["close"] > rim_at_cup

    return (cup_recent & shallow & breaks_out).fillna(False).astype(bool)


@register(
    "inverted_head_shoulders", kind="single", direction="bullish", specificity=110
)
def inverted_head_shoulders(
    df: pd.DataFrame,
    *,
    tolerance: float = 0.015,
    left: int = 5,
    right: int = 5,
    max_span: int = 38,
    by: pd.Series | None = None,
) -> pd.Series:
    """Lo et al. Definition 1, inverted: a low between two matched higher lows."""
    return _head_and_shoulders(
        df, inverted=True, tolerance=tolerance, left=left, right=right,
        max_span=max_span, by=by,
    )


# --------------------------------------------------------------------------
# exported, not registered: the short side
# --------------------------------------------------------------------------


def head_shoulders(
    df: pd.DataFrame,
    *,
    tolerance: float = 0.015,
    left: int = 5,
    right: int = 5,
    max_span: int = 38,
    by: pd.Series | None = None,
) -> pd.Series:
    """Lo et al. Definition 1: a peak between two matched lower peaks."""
    return _head_and_shoulders(
        df, inverted=False, tolerance=tolerance, left=left, right=right,
        max_span=max_span, by=by,
    )


def double_top(
    df: pd.DataFrame,
    *,
    tolerance: float = 0.015,
    min_separation: int = 22,
    left: int = 5,
    right: int = 5,
    by: pd.Series | None = None,
) -> pd.Series:
    """Two confirmed highs at the same price, a month or more apart."""
    return _double(
        df, bottom=False, tolerance=tolerance, min_separation=min_separation,
        left=left, right=right, by=by,
    )


def rounding_top(
    df: pd.DataFrame,
    *,
    window: int = 60,
    rim: int = 5,
    min_depth_pct: float = 10.0,
    min_r2: float = 0.70,
    vertex_band: float = 0.25,
    max_base_frac: float = 0.25,
    by: pd.Series | None = None,
) -> pd.Series:
    """A smooth dome, with price closing back below its left rim."""
    return _rounding(
        df, bottom=False, window=window, rim=rim, min_depth_pct=min_depth_pct,
        min_r2=min_r2, vertex_band=vertex_band, max_base_frac=max_base_frac, by=by,
    )


def flag_breakdown(
    df: pd.DataFrame,
    *,
    pole: int = 10,
    flag: int = 8,
    min_pole_pct: float = 15.0,
    max_flag_frac: float = 0.5,
    by: pd.Series | None = None,
) -> pd.Series:
    """A steep decline, a shallow pause, and a close below the pause's low."""
    return _flag(
        df, bullish=False, pole=pole, flag=flag, min_pole_pct=min_pole_pct,
        max_flag_frac=max_flag_frac, by=by,
    )


def fake_breakout(
    df: pd.DataFrame,
    *,
    lookback: int = 20,
    recovery: int = 5,
    by: pd.Series | None = None,
) -> pd.Series:
    """Price cleared resistance and closed back below it -- trapped buyers."""
    return _failed_break(
        df, upward=True, lookback=lookback, recovery=recovery, by=by
    )
