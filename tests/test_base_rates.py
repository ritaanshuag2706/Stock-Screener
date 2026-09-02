"""Stage 7. The tests that matter here are about lookahead.

An outcome study that peeks at the future produces beautiful, worthless
numbers, and nothing downstream can tell. So these check the mechanics --
entry price, which side resolved, what happens at the end of the data --
on frames where the answer is arithmetic rather than opinion.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener.study.base_rates import (
    ALL_BARS,
    Rules,
    bucket,
    forward_outcomes,
    rates,
    with_control,
)

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def frame(rows, symbol="AAA"):
    """rows are (open, high, low, close)."""
    return pd.DataFrame(
        [
            {"date": d, "symbol": symbol, "open": o, "high": h, "low": low,
             "close": c, "volume": 1000}
            for d, (o, h, low, c) in zip(
                pd.bdate_range("2024-01-01", periods=len(rows)), rows, strict=True
            )
        ],
        columns=COLUMNS,
    )


def flat(n, price=100.0, rng=1.0):
    """Bars with a constant 1.0 true range, so ATR is exactly 1.0."""
    return [(price, price + rng / 2, price - rng / 2, price)] * n


# --- entry price ------------------------------------------------------------


def test_entry_is_the_next_bar_open_never_this_bar_close():
    """The signal is read at t's close; you cannot trade that price."""
    rows = flat(60)
    rows[30] = (100.0, 101.0, 99.0, 100.0)
    rows[31] = (123.0, 124.0, 122.0, 123.0)      # next bar opens away
    out = forward_outcomes(frame(rows))
    assert out["entry"].iloc[30] == 123.0


def test_the_last_bars_cannot_be_evaluated():
    """With fewer than `horizon` bars ahead, the answer is not in yet -- that is
    not the same as a timeout."""
    out = forward_outcomes(frame(flat(40)), rules=Rules(horizon=10))
    assert not out["usable"].iloc[-10:].any()
    assert out["outcome"].iloc[-10:].isna().all()


# --- which side resolved ----------------------------------------------------


def test_target_is_recorded_when_price_runs_up():
    rows = flat(60)
    rows[31] = (100.0, 100.5, 99.5, 100.0)
    rows[32] = (100.0, 103.0, 99.8, 102.5)       # +2 ATR is 102.0
    out = forward_outcomes(frame(rows), rules=Rules(target_atr=2, stop_atr=1))
    assert out["outcome"].iloc[30] == "target"


def test_stop_is_recorded_when_price_falls_first():
    rows = flat(60)
    rows[31] = (100.0, 100.5, 98.5, 99.0)        # -1 ATR is 99.0
    out = forward_outcomes(frame(rows), rules=Rules(target_atr=2, stop_atr=1))
    assert out["outcome"].iloc[30] == "stop"


def test_a_bar_touching_both_counts_as_a_stop():
    """A daily bar cannot say which came first. Assuming the good one would
    inflate every number in the study."""
    rows = flat(60)
    rows[31] = (100.0, 103.0, 98.0, 100.0)       # spans target and stop
    out = forward_outcomes(frame(rows), rules=Rules(target_atr=2, stop_atr=1))
    assert out["outcome"].iloc[30] == "stop"


def test_timeout_when_neither_is_reached():
    out = forward_outcomes(frame(flat(40)), rules=Rules(horizon=5))
    assert (out["outcome"].dropna() == "timeout").all()


def test_earlier_touch_wins():
    rows = flat(60)
    rows[31] = (100.0, 103.0, 99.5, 102.0)       # target on the first bar
    rows[32] = (100.0, 100.5, 98.0, 99.0)        # stop on the second
    out = forward_outcomes(frame(rows), rules=Rules(target_atr=2, stop_atr=1))
    assert out["outcome"].iloc[30] == "target"
    assert out["bars_held"].iloc[30] == 1


# --- corporate actions ------------------------------------------------------


def test_a_split_inside_the_forward_window_makes_the_signal_unusable():
    """Prices either side of an unadjusted split are not comparable, so the
    'return' would be an artefact of the corporate action."""
    rows = flat(60)
    rows[33] = (50.0, 50.5, 49.5, 50.0)          # a halving
    out = forward_outcomes(frame(rows), rules=Rules(horizon=10))
    assert not out["usable"].iloc[30]
    assert pd.isna(out["outcome"].iloc[30])


def test_a_split_outside_the_window_is_fine():
    rows = flat(60)
    rows[10] = (50.0, 50.5, 49.5, 50.0)
    out = forward_outcomes(frame(rows), rules=Rules(horizon=10))
    assert out["usable"].iloc[40]


# --- symbol isolation -------------------------------------------------------


def test_outcomes_never_look_across_a_symbol_boundary():
    a = frame(flat(40), "AAA")
    b = frame(flat(40, price=500.0), "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["symbol", "date"])
    out = forward_outcomes(both, by=both["symbol"])
    alone = forward_outcomes(a)
    np.testing.assert_allclose(
        out.loc[both["symbol"] == "AAA", "entry"].to_numpy(float),
        alone["entry"].to_numpy(float), equal_nan=True,
    )


# --- reporting --------------------------------------------------------------


def test_rates_counts_only_evaluated_rows():
    out = forward_outcomes(frame(flat(40)), rules=Rules(horizon=10))
    table = rates(out.assign(pattern="x"), "pattern")
    assert table.loc["x", "n"] == out["outcome"].notna().sum()


def test_control_row_and_lift():
    out = forward_outcomes(frame(flat(40)), rules=Rules(horizon=5))
    out["pattern"] = "x"
    joined = with_control(rates(out, "pattern"), rates(out))
    assert ALL_BARS in joined.index
    assert joined.loc[ALL_BARS, "lift"] == 0        # the control cannot beat itself


def test_invalid_rules_raise():
    with pytest.raises(ValueError, match="horizon"):
        Rules(horizon=0)
    with pytest.raises(ValueError, match="positive"):
        Rules(target_atr=0)


def test_bucket_rejects_an_unknown_column():
    with pytest.raises(KeyError, match="no buckets defined"):
        bucket(pd.DataFrame({"nope": [1]}), "nope")
