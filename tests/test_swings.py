"""Confirmed swing pivots. Almost entirely about not seeing the future.

A pivot is the one place in this project where lookahead is the *default*
behaviour rather than a mistake you have to make: the textbook definition of a
swing high names bars that have not printed yet. So these tests are less about
whether a pivot is found and more about when it is allowed to be known.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener.patterns import _swings as sw

COLUMNS = ["date", "symbol", "open", "high", "low", "close"]


def frame(closes, symbol="AAA", pad=0.0):
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        [{"date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c + pad, "low": c - pad, "close": c}
         for i, c in enumerate(closes)],
        columns=COLUMNS,
    )


def peak_at(i, n=25, base=10.0, height=10.0):
    """A single clean peak at index `i`."""
    return [base + height * max(0.0, 1 - abs(j - i) / 4) for j in range(n)]


# --- when a pivot becomes knowable ------------------------------------------


def test_a_pivot_is_confirmed_exactly_right_bars_later():
    df = frame(peak_at(10))
    piv = sw.pivots(df, left=3, right=3)
    assert list(df.index[piv["pivot_high"]]) == [13]


def test_the_pivot_bar_itself_never_reports_the_pivot():
    """The bar at the top of the swing cannot know it is the top."""
    df = frame(peak_at(10))
    for right in (1, 3, 5):
        piv = sw.pivots(df, left=3, right=right)
        assert not piv["pivot_high"].iloc[10]


def test_widening_the_right_window_delays_confirmation():
    df = frame(peak_at(12, n=30))
    for right in (2, 4, 6):
        piv = sw.pivots(df, left=3, right=right)
        assert list(df.index[piv["pivot_high"]]) == [12 + right]


def test_prefix_invariance_is_what_no_lookahead_means():
    """The strongest statement available: the answer at bar t computed from the
    first t+1 bars must equal the answer at bar t computed from everything.

    If a future bar could reach backwards, this is the test that catches it --
    and it catches the whole file at once, however the arithmetic is arranged.
    """
    rng = np.random.default_rng(7)
    closes = 100 + np.cumsum(rng.normal(0, 1.5, 200))
    df = frame(closes)
    full = sw.pivots(df, left=4, right=4)
    for t in range(60, 200, 7):
        prefix = sw.pivots(df.iloc[: t + 1], left=4, right=4)
        assert prefix["pivot_high"].iloc[t] == full["pivot_high"].iloc[t]
        assert prefix["pivot_low"].iloc[t] == full["pivot_low"].iloc[t]


# --- what counts as a pivot at all ------------------------------------------


def test_a_flat_series_has_no_pivots():
    """Strict on the left, non-strict on the right. With `>=` both sides every
    bar of a flat series is a pivot high *and* a pivot low, which is how a swing
    detector ends up firing on a suspended stock."""
    piv = sw.pivots(frame([5.0] * 40), left=3, right=3)
    assert not piv["pivot_high"].any()
    assert not piv["pivot_low"].any()


def test_a_trough_is_found_as_well_as_a_peak():
    df = frame([-c for c in peak_at(10)])
    piv = sw.pivots(df, left=3, right=3)
    assert list(df.index[piv["pivot_low"]]) == [13]


def test_a_shoulder_is_not_a_pivot():
    """One bar down from the top, with a higher bar inside its window."""
    df = frame(peak_at(10, n=25))
    piv = sw.pivots(df, left=3, right=3)
    assert list(df.index[piv["pivot_high"]]) == [13]      # and nothing at 12 or 14


def test_no_pivot_until_both_windows_have_bars():
    df = frame(peak_at(1, n=12))
    piv = sw.pivots(df, left=5, right=5)
    assert not piv["pivot_high"].iloc[:5].any()


def test_zero_width_windows_are_refused():
    """`right = 0` is a pivot confirmed by nothing -- which is just "today is
    the highest of the last few bars", and `breakout_20` already measures that."""
    for left, right in ((0, 3), (3, 0), (0, 0)):
        with pytest.raises(ValueError, match="must both be >= 1"):
            sw.pivots(frame([1.0] * 20), left=left, right=right)


# --- carrying the last two pivots forward -----------------------------------


def test_track_reports_the_pivot_price_not_the_confirmation_price():
    df = frame(peak_at(10))
    piv = sw.pivots(df, left=3, right=3)
    t = sw.track(df["high"], piv["pivot_high"], right=3)
    assert t["last"].iloc[13] == pytest.approx(df["high"].iloc[10])
    assert t["last_at"].iloc[13] == 10


def test_track_holds_the_previous_pivot_alongside_the_last():
    df = frame(peak_at(8, n=40) + peak_at(8, n=40, height=14.0))
    piv = sw.pivots(df, left=3, right=3)
    t = sw.track(df["high"], piv["pivot_high"], right=3)
    last_row = t.iloc[-1]
    assert last_row["last"] > last_row["prev"]          # second peak is taller
    assert last_row["last_at"] > last_row["prev_at"]


def test_track_carries_values_forward_between_pivots():
    df = frame(peak_at(10, n=40))
    piv = sw.pivots(df, left=3, right=3)
    t = sw.track(df["high"], piv["pivot_high"], right=3)
    held = t["last"].iloc[13:].unique()
    assert len(held) == 1 and held[0] == pytest.approx(df["high"].iloc[10])


def test_nothing_is_known_before_the_first_confirmation():
    df = frame(peak_at(10))
    piv = sw.pivots(df, left=3, right=3)
    t = sw.track(df["high"], piv["pivot_high"], right=3)
    assert t["last"].iloc[:13].isna().all()


# --- symbol boundaries -------------------------------------------------------


def test_pivots_never_leak_across_symbols():
    a = frame(peak_at(10), "AAA")
    b = frame([100.0] * 25, "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    grouped = sw.pivots(both, left=3, right=3, by=both["symbol"])
    alone = sw.pivots(a, left=3, right=3)
    assert (
        grouped.loc[both["symbol"] == "AAA", "pivot_high"].to_numpy()
        == alone["pivot_high"].to_numpy()
    ).all()
    # BBB is flat; a neighbour's peak must not give it one.
    assert not grouped.loc[both["symbol"] == "BBB", "pivot_high"].any()


def test_track_never_leaks_across_symbols():
    a = frame(peak_at(10), "AAA")
    b = frame([100.0] * 25, "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    piv = sw.pivots(both, left=3, right=3, by=both["symbol"])
    t = sw.track(both["high"], piv["pivot_high"], right=3, by=both["symbol"])
    assert t.loc[both["symbol"] == "BBB", "last"].isna().all()
