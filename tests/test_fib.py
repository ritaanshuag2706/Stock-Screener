"""Fibonacci retracement.

Two things worth testing here and one worth stating. The leg has to be found
from confirmed pivots only; the level has to be measured as a fraction of that
leg rather than of price. What is *not* tested, because it cannot be, is whether
a Fibonacci level means anything -- that is a Stage 7 question and this file
takes no position on it.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import patterns
from nse_screener.patterns import fib, registry

COLUMNS = ["date", "symbol", "open", "high", "low", "close"]


def frame(closes, symbol="AAA"):
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        [{"date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c, "low": c, "close": c}
         for i, c in enumerate(closes)],
        columns=COLUMNS,
    )


def ramp(a, b, n):
    """A leg. Excludes the endpoint so segments can be chained without repeats."""
    return list(np.linspace(a, b, n, endpoint=False))


def slide_to(a, b, n):
    """A tail that actually arrives at `b`, for tests that name a target price."""
    return list(np.linspace(a, b, n))


def up_leg(tail=None):
    """Dip to 95, rally to 150, then whatever the caller wants next.

    Leg low 95, leg high 150, height 55 -- so the 0.618 retracement sits at
    150 - 0.618 * 55 = 116.01.
    """
    return frame([100.0] * 10 + ramp(100, 95, 5) + ramp(95, 150, 30)
                 + (tail if tail is not None else ramp(150, 110, 20)))


# --- the leg ----------------------------------------------------------------


def test_the_leg_is_the_last_confirmed_pivot_of_each_kind():
    leg = fib.swing_leg(up_leg(), left=4, right=4).iloc[-1]
    assert leg["leg_low"] == pytest.approx(95.0)
    assert leg["leg_high"] == pytest.approx(150.0)
    assert leg["leg_up"]


def test_retracement_is_zero_at_the_high_and_one_at_the_low():
    leg = fib.swing_leg(up_leg(), left=4, right=4)
    df = up_leg()
    r = (leg["leg_high"] - df["close"]) / (leg["leg_high"] - leg["leg_low"])
    both = leg["retracement"].notna()
    assert np.allclose(leg["retracement"][both], r[both])


def test_retracement_is_a_half_at_the_midpoint():
    leg = fib.swing_leg(up_leg(tail=slide_to(150, 122.5, 12)), left=4, right=4)
    assert leg["retracement"].iloc[-1] == pytest.approx(0.5, abs=0.02)


def test_leg_is_unknown_until_both_kinds_of_pivot_have_confirmed():
    leg = fib.swing_leg(up_leg(), left=4, right=4)
    assert leg["retracement"].iloc[:20].isna().all()


def test_leg_up_is_a_real_boolean_not_object_with_nan():
    """`leg_up` answers "should a long setup read this leg". Unknown and "no"
    are the same answer, so it stays bool rather than pushing a fillna into
    every caller."""
    assert fib.swing_leg(up_leg(), left=4, right=4)["leg_up"].dtype == bool


# --- the detector -----------------------------------------------------------


def test_it_fires_in_the_golden_zone():
    hits = patterns.detect("fib_retracement", up_leg(), left=4, right=4)
    assert hits.any()


def test_it_does_not_fire_between_levels():
    """0.75 of the way back is not a Fibonacci level: 150 - 0.75 * 55 = 108.75."""
    df = up_leg(tail=slide_to(150, 108.75, 14))
    assert not patterns.detect("fib_retracement", df, left=4, right=4).iloc[-1]


def test_it_does_not_fire_on_a_down_leg():
    """Registered as a long entry, so a rally into a level off a falling leg is
    not its business -- see the module docstring in fib.py."""
    df = frame([100.0] * 10 + ramp(100, 105, 5) + ramp(105, 60, 30)
               + ramp(60, 77.0, 20))
    leg = fib.swing_leg(df, left=4, right=4)
    assert not leg["leg_up"].iloc[-1]
    assert not patterns.detect("fib_retracement", df, left=4, right=4).iloc[-1]


def test_a_leg_too_small_to_matter_is_rejected():
    """On a 1% swing every Fibonacci level is inside the spread."""
    df = frame([100.0] * 10 + ramp(100, 99.5, 5) + ramp(99.5, 100.5, 30)
               + ramp(100.5, 99.9, 20))
    assert not patterns.detect("fib_retracement", df, left=4, right=4).any()


def test_a_stale_leg_is_rejected():
    long_tail = slide_to(150, 116.01, 10) + [116.01] * 90
    df = up_leg(tail=long_tail)
    hits = patterns.detect("fib_retracement", df, left=4, right=4)
    assert hits.any()                       # it did fire while the leg was fresh
    assert not hits.iloc[-1]                # and not ninety bars later


def test_tolerance_is_a_fraction_of_the_leg_not_of_price():
    """0.02 of a 55-point leg is 1.1 points. As a fraction of a 116-rupee price
    it would be 2.3 -- a band twice as wide, and a different width on every
    stock. The setting has to mean the same thing everywhere."""
    df = up_leg(tail=slide_to(150, 117.0, 12))          # ~0.6 of the leg from 0.618
    assert patterns.detect("fib_retracement", df, left=4, right=4).iloc[-1]
    assert not patterns.detect(
        "fib_retracement", df, left=4, right=4, tolerance=0.001
    ).iloc[-1]


# --- lookahead and boundaries -----------------------------------------------


def test_prefix_invariance():
    rng = np.random.default_rng(11)
    df = frame(100 + np.cumsum(rng.normal(0, 1.2, 220)))
    full = patterns.detect("fib_retracement", df, left=4, right=4)
    for t in range(80, 220, 9):
        prefix = patterns.detect("fib_retracement", df.iloc[: t + 1], left=4, right=4)
        assert prefix.iloc[t] == full.iloc[t]


def test_it_never_leaks_across_symbols():
    a = up_leg().assign(symbol="AAA")
    b = frame([100.0] * len(a), "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    hits = patterns.detect_by_symbol(both, ["fib_retracement"])
    alone = patterns.detect_all(a, ["fib_retracement"])
    assert (
        hits.loc[both["symbol"] == "AAA", "fib_retracement"].to_numpy()
        == alone["fib_retracement"].to_numpy()
    ).all()


def test_it_is_registered_with_config():
    assert "fib_retracement" in registry.names()
    assert registry.load_params()["fib_retracement"]


# --- the checklist's grade --------------------------------------------------


def test_a_shallow_retracement_is_healthy():
    """Up to 50% of the leg given back. 150 - 0.30 * 55 = 133.5."""
    df = up_leg(tail=slide_to(150, 133.5, 10))
    assert fib.retracement_grade(df, left=4, right=4).iloc[-1] == "healthy"


def test_a_retracement_past_the_golden_ratio_is_watchful():
    """61.8% or more. 150 - 0.70 * 55 = 111.5."""
    df = up_leg(tail=slide_to(150, 111.5, 10))
    assert fib.retracement_grade(df, left=4, right=4).iloc[-1] == "watchful"


def test_the_band_the_checklist_leaves_out_gets_its_own_label():
    """The source names "up to 50%" and "61.8% or more" and says nothing about
    what sits between. Folding that band silently into one side would be a
    change of method nobody decided on, so it is labelled rather than assumed.
    150 - 0.55 * 55 = 119.75."""
    df = up_leg(tail=slide_to(150, 119.75, 10))
    assert fib.retracement_grade(df, left=4, right=4).iloc[-1] == "deep"


def test_the_boundaries_land_on_the_healthy_and_watchful_sides():
    at_50 = up_leg(tail=slide_to(150, 150 - 0.50 * 55, 10))
    at_618 = up_leg(tail=slide_to(150, 150 - 0.618 * 55, 10))
    assert fib.retracement_grade(at_50, left=4, right=4).iloc[-1] == "healthy"
    assert fib.retracement_grade(at_618, left=4, right=4).iloc[-1] == "watchful"


def test_a_failed_swing_is_not_graded():
    """Past the leg low the retracement is over, not deep."""
    df = up_leg(tail=slide_to(150, 80.0, 14))
    assert pd.isna(fib.retracement_grade(df, left=4, right=4).iloc[-1])


def test_no_grade_before_a_leg_is_known():
    assert fib.retracement_grade(up_leg(), left=4, right=4).iloc[:20].isna().all()


def test_grade_thresholds_come_from_config():
    assert registry.load_params()["fib_grade"] == {
        "healthy_max": 0.50, "watchful_min": 0.618
    }


def test_incoherent_thresholds_are_refused():
    with pytest.raises(ValueError, match="healthy_max <= watchful_min"):
        fib.retracement_grade(up_leg(), healthy_max=0.8, watchful_min=0.3)


def test_grade_thresholds_are_not_passed_to_the_detector():
    """`params_for` hands everything under a registered pattern's key to that
    detector as kwargs, so a grade threshold parked there would arrive as an
    unexpected argument. This is what keeps `fib_grade` at the top level."""
    assert "healthy_max" not in registry.params_for("fib_retracement")
    patterns.detect("fib_retracement", up_leg())      # would raise TypeError
