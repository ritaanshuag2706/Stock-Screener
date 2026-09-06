"""Confirmed swing pivots: the shared machinery under Fibonacci and divergence.

Both of those families read *swings* rather than bars. A swing needs a pivot, a
pivot is a local extreme, and a local extreme is where the lookahead lives.

The trap, stated plainly. A pivot high at bar `p` is conventionally "the highest
high of the `left` bars before it and the `right` bars after it". Those
right-hand bars are the problem: the pivot is not knowable until bar `p + right`
has printed. Every charting package's divergence indicator repaints for exactly
this reason -- it draws the pivot back at `p` the moment `p + right` arrives, and
the line then appears to have been there all along. Backtest that and the result
is not wrong by a little.

So nothing here is ever placed at the pivot bar. Every value is placed at the
**confirmation** bar, the first bar on which a live trader could have known the
pivot existed. The consequence is deliberate and worth stating: the most recent
usable pivot is always at least `right` bars old, and a detector built on this
is late by construction. That is not a defect to tune away -- it is the cost of
the information, and paying it is the difference between a signal and an
artefact.

Mechanically that means no negative shifts anywhere in this file. At bar `t` the
candidate pivot is `p = t - right`, and every window used to judge it ends at or
before `t`:

    high[p]                     = high.shift(right)
    max(high[p-left .. p-1])     = high.rolling(left).max().shift(right + 1)
    max(high[p+1 .. p+right])    = high.rolling(right).max()          <- ends at t

Strict on the left, non-strict on the right. That is the conventional
first-among-equals rule, and it earns its place on flat data: with `>=` on both
sides every bar of a flat series is simultaneously a pivot high and a pivot low,
which is how a "swing" detector ends up firing on a suspended stock.

Everything is `by=`-aware, like the detectors. A pivot must never be computed
across a symbol boundary -- `store.read()` sorts by (date, symbol), so the bar
before RELIANCE's is some other company's.
"""

from __future__ import annotations

import pandas as pd

from . import _geometry as g


def _rolling(s: pd.Series, by: pd.Series | None, window: int, how: str) -> pd.Series:
    """Rolling min/max that comes back in the caller's row order.

    `groupby().rolling()` returns its result grouped -- all of one symbol, then
    all of the next -- so after dropping the group level the labels are a
    permutation of the input's. Every other rolling helper in this project is
    called from `detect_by_symbol`, which sorts by (symbol, date) first and so
    never notices. These are also called directly, and a permuted index is the
    worst kind of bug available here: pandas aligns on labels, so the arithmetic
    would quietly compare one symbol's window against another symbol's bar.

    Reindexing puts it back. On a duplicate index it raises rather than guessing,
    which is the failure mode you want.
    """
    if by is None:
        return getattr(s.rolling(window), how)()
    out = getattr(s.groupby(by).rolling(window), how)()
    out.index = out.index.droplevel(0)
    return out.reindex(s.index)


def _rolling_max(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    return _rolling(s, by, window, "max")


def _rolling_min(s: pd.Series, by: pd.Series | None, window: int) -> pd.Series:
    return _rolling(s, by, window, "min")


def _ffill(s: pd.Series, by: pd.Series | None) -> pd.Series:
    return s.ffill() if by is None else s.groupby(by).ffill()


def _ordinal(s: pd.Series, by: pd.Series | None) -> pd.Series:
    """0-based bar number within each symbol. Used to order two pivots in time
    and to measure the distance between them, which calendar dates cannot do
    once holidays are in the way."""
    if by is None:
        return pd.Series(range(len(s)), index=s.index, dtype="float64")
    return s.groupby(by).cumcount().astype("float64")


def pivots(
    df: pd.DataFrame,
    *,
    left: int = 5,
    right: int = 5,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Where a swing pivot became *knowable*, one row per bar.

    Returns two boolean columns, `pivot_high` and `pivot_low`, each True on the
    bar that confirms a pivot sitting `right` bars earlier. Never True on the
    pivot bar itself -- see the module docstring for why that is the whole point.

    `left` and `right` must both be at least 1. `right = 0` would mean a pivot
    confirmed by nothing, which is just "today is the highest of the last few
    bars" and is already what `breakout_20` measures.
    """
    if left < 1 or right < 1:
        raise ValueError(f"left and right must both be >= 1, got {left} and {right}")

    d = g.normalise(df)
    high, low = d["high"], d["low"]

    # The candidate sits `right` bars back. Read its price from there.
    ph = g.shift_by(high, right, by)
    pl = g.shift_by(low, right, by)

    # Left window: the `left` bars strictly before the candidate.
    left_max = g.shift_by(_rolling_max(high, by, left), right + 1, by)
    left_min = g.shift_by(_rolling_min(low, by, left), right + 1, by)

    # Right window: the `right` bars after the candidate, ending on this bar.
    right_max = _rolling_max(high, by, right)
    right_min = _rolling_min(low, by, right)

    out = pd.DataFrame(index=df.index)
    out["pivot_high"] = ((ph > left_max) & (ph >= right_max)).fillna(False)
    out["pivot_low"] = ((pl < left_min) & (pl <= right_min)).fillna(False)
    return out.astype(bool)


def track(
    values: pd.Series,
    confirmed: pd.Series,
    *,
    right: int = 5,
    by: pd.Series | None = None,
    shift_to_pivot: bool = True,
) -> pd.DataFrame:
    """Carry the last two confirmed pivots forward to every bar.

    `values` is any series aligned to the bars -- price for the swing itself, an
    oscillator for a divergence. `confirmed` is a column from `pivots()`.

    Returns `last` and `prev`: the value at the most recent confirmed pivot, and
    at the one before it, forward-filled so both are readable on every bar rather
    than only on confirmation bars. `last_at` and `prev_at` carry the same for
    the pivot's own bar number, which is what lets a caller ask which of two
    pivots came first and how far apart they are.

    `shift_to_pivot=False` reads `values` on the confirmation bar instead of the
    pivot bar -- rarely what you want, but the escape hatch is here rather than
    invented at a call site by shifting the input twice.

    The `prev` trick is worth reading once. On a confirmation bar the
    forward-filled series already holds *this* pivot, so the previous one is
    simply that same series one bar earlier -- it had not yet been overwritten.
    Capturing that at confirmation bars and forward-filling it gives the
    second-most-recent pivot everywhere, with no per-group Python loop.
    """
    at_pivot = g.shift_by(values, right, by) if shift_to_pivot else values
    ordinal = _ordinal(values, by)
    at_pivot_ord = ordinal - right if shift_to_pivot else ordinal

    sparse = at_pivot.where(confirmed)
    sparse_ord = at_pivot_ord.where(confirmed)

    last = _ffill(sparse, by)
    last_at = _ffill(sparse_ord, by)

    # One bar before a confirmation, the ffill still holds the pivot before this
    # one. Snapshot it there, then carry that forward.
    prev = _ffill(g.shift_by(last, 1, by).where(confirmed), by)
    prev_at = _ffill(g.shift_by(last_at, 1, by).where(confirmed), by)

    return pd.DataFrame(
        {"last": last, "prev": prev, "last_at": last_at, "prev_at": prev_at},
        index=values.index,
    )


def extrema(
    df: pd.DataFrame,
    *,
    left: int = 5,
    right: int = 5,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """Highs and lows merged into one confirmed-pivot series.

    The classical chart patterns are defined over a sequence of alternating
    local extrema, not over highs and lows separately, so they need the two
    streams interleaved. Columns, all placed on the confirmation bar:

        confirmed  a pivot of either kind became knowable here
        value      its price
        kind       1.0 for a maximum, 0.0 for a minimum
        at         the pivot's own bar number

    A bar that confirms a high *and* a low at once is dropped rather than
    assigned to one of them. It is rare, it means the two windows disagree about
    which way the bar leans, and forcing a choice would put an arbitrary link in
    the middle of a five-extremum sequence.
    """
    piv = pivots(df, left=left, right=right, by=by)
    d = g.normalise(df)

    is_high, is_low = piv["pivot_high"], piv["pivot_low"]
    confirmed = (is_high | is_low) & ~(is_high & is_low)

    value = g.shift_by(d["high"], right, by).where(
        is_high, g.shift_by(d["low"], right, by)
    )
    out = pd.DataFrame(index=df.index)
    out["confirmed"] = confirmed
    out["value"] = value.where(confirmed)
    out["kind"] = is_high.astype("float64").where(confirmed)
    out["at"] = (_ordinal(d["close"], by) - right).where(confirmed)
    return out


def carry(
    at_confirmation: pd.Series,
    confirmed: pd.Series,
    n: int,
    *,
    by: pd.Series | None = None,
) -> pd.DataFrame:
    """The last `n` confirmed values, carried forward to every bar.

    `at_confirmation` is a series already aligned to confirmation bars -- a
    column from `extrema()`, or anything shifted the same way. Returns `e1`
    (most recent) through `en` (oldest of the n).

    This is `track()`'s two-deep trick run as a chain. On a confirmation bar the
    forward-filled series still holds the *previous* value one bar earlier, so
    snapshotting there and filling forward steps back exactly one pivot. Doing
    it again steps back another. Five extrema therefore cost five passes of
    vectorised pandas rather than a loop over symbols and bars.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    cols: dict[str, pd.Series] = {}
    cur = _ffill(at_confirmation.where(confirmed), by)
    cols["e1"] = cur
    for k in range(2, n + 1):
        cur = _ffill(g.shift_by(cur, 1, by).where(confirmed), by)
        cols[f"e{k}"] = cur
    return pd.DataFrame(cols, index=at_confirmation.index)
