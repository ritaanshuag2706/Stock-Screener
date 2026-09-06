"""Classical chart patterns.

Each test names the source its numbers come from. Where a threshold was chosen
rather than cited -- Bulkowski's shapes have no numbers in them -- the test says
so, because those are the ones that should move first if the family is ever
measured.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import patterns
from nse_screener.patterns import chart, registry

COLUMNS = ["date", "symbol", "open", "high", "low", "close"]

BULLISH = [
    "inverted_head_shoulders", "double_bottom", "rounding_bottom",
    "cup_and_handle", "flag_breakout", "fake_breakdown",
]
BEARISH = ["head_shoulders", "double_top", "rounding_top", "flag_breakdown",
           "fake_breakout"]


def frame(closes, symbol="AAA", pad=0.0):
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        [{"date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c + pad, "low": c - pad, "close": c}
         for i, c in enumerate(closes)],
        columns=COLUMNS,
    )


def leg(a, b, n):
    """A straight move from a towards b over n bars, endpoint excluded."""
    return list(np.linspace(a, b, n, endpoint=False))


def shape(points, per=7, flat=8, base=None):
    """Chain legs through `points`, with a flat run before and after."""
    out = [points[0]] * flat
    for i in range(len(points) - 1):
        out += leg(points[i], points[i + 1], per)
    out += [points[-1]] * flat
    return frame(out if base is None else base + out)


# --- head and shoulders: Lo, Mamaysky & Wang (2000) Definition 1 -------------


def ihs_points(e1=90.0, e2=100.0, e3=80.0, e4=100.0, e5=90.0):
    """E1 min, E3 < E1, E3 < E5, E1~E5 and E2~E4 matched. Their Definition 1."""
    return [105.0, e1, e2, e3, e4, e5, 99.0]


def test_an_inverted_head_and_shoulders_is_found():
    assert patterns.detect(
        "inverted_head_shoulders", shape(ihs_points()), left=3, right=3
    ).any()


def test_the_upright_form_is_found_too():
    upright = shape([75.0, 90.0, 80.0, 100.0, 80.0, 90.0, 81.0])
    assert chart.head_shoulders(upright, left=3, right=3).any()


def test_mismatched_shoulders_are_not_a_head_and_shoulders():
    """Lo et al. require the two shoulders within 1.5 percent of their average.
    82 against 90 is nine percent apart."""
    assert not patterns.detect(
        "inverted_head_shoulders", shape(ihs_points(e5=82.0)), left=3, right=3
    ).any()


def test_a_head_that_is_not_the_extreme_is_not_a_head():
    """E3 must be below both shoulders for the inverted form."""
    assert not patterns.detect(
        "inverted_head_shoulders", shape(ihs_points(e3=95.0)), left=3, right=3
    ).any()


def test_shoulders_matched_to_the_letter_of_the_tolerance():
    """Their phrase is "within 1.5 percent of their average", so each extremum
    may sit 1.5% from the midpoint -- the pair may differ by about 3%. Reading
    it as a 1.5% *difference* halves the tolerance and roughly halves the
    number of patterns found."""
    assert chart._within(pd.Series([100.0]), pd.Series([102.9]), 0.015).iloc[0]
    assert not chart._within(pd.Series([100.0]), pd.Series([103.1]), 0.015).iloc[0]


def test_a_pattern_spread_too_wide_is_not_one():
    """Their rolling window was 38 trading days, precisely so that only patterns
    completed inside it are found."""
    wide = shape(ihs_points(), per=14)
    assert patterns.detect(
        "inverted_head_shoulders", wide, left=3, right=3, max_span=200
    ).any()
    assert not patterns.detect(
        "inverted_head_shoulders", wide, left=3, right=3, max_span=20
    ).any()


def test_extrema_must_alternate():
    """Lo et al. get alternation free from kernel smoothing; raw pivots do not,
    so it is checked. A sequence that does not alternate yields no pattern
    rather than a forced one."""
    f = chart.five_extrema(shape(ihs_points()), left=3, right=3)
    rows = f[f["alternating"]]
    assert len(rows) > 0
    kinds = rows[["K1", "K2", "K3", "K4", "K5"]].to_numpy()
    assert (kinds[:, :-1] != kinds[:, 1:]).all()


# --- double bottom: their Definition 5 --------------------------------------


def test_a_double_bottom_is_found():
    df = shape([100.0, 80.0, 95.0, 80.0, 99.0], per=13)
    assert patterns.detect("double_bottom", df, left=3, right=3).any()


def test_two_bottoms_at_different_prices_are_not_a_double_bottom():
    df = shape([100.0, 80.0, 95.0, 70.0, 99.0], per=13)
    assert not patterns.detect("double_bottom", df, left=3, right=3).any()


def test_bottoms_closer_than_a_month_apart_are_one_bottom():
    """22 trading days, from Edwards and Magee via Lo et al. -- two lows less
    than a month apart are one low with a notch in it."""
    tight = shape([100.0, 80.0, 95.0, 80.0, 99.0], per=4)
    assert not patterns.detect("double_bottom", tight, left=3, right=3).any()
    assert patterns.detect(
        "double_bottom", tight, left=3, right=3, min_separation=5
    ).any()


def test_the_upright_double_top_is_found():
    df = shape([80.0, 100.0, 85.0, 100.0, 81.0], per=13)
    assert chart.double_top(df, left=3, right=3).any()


# --- rounding bottom and cup: Bulkowski, with chosen thresholds -------------


def bowl(n=60, depth=20.0, base=100.0):
    """A parabola. `min_r2` is what decides how far from this a real bowl may sit."""
    x = np.linspace(-1, 1, n)
    return list(base - depth * (1 - x ** 2))


def test_a_smooth_bowl_that_clears_its_left_rim_is_a_rounding_bottom():
    df = frame([100.0] * 6 + bowl() + [101.0, 102.0])
    assert patterns.detect(
        "rounding_bottom", df, window=60, rim=4, min_depth_pct=5.0
    ).any()


def test_a_v_shaped_bottom_is_not_a_rounding_bottom():
    """Bulkowski: the turn must be "gradual and rounded, not sharp or pointed".
    `min_r2` is the number standing in for that -- his guidelines have none."""
    v = frame([100.0] * 6 + leg(100, 80, 30) + leg(80, 102, 30) + [102.5])
    assert not patterns.detect(
        "rounding_bottom", v, window=60, rim=4, min_depth_pct=5.0
    ).any()


def test_a_dome_is_not_a_bowl():
    dome = frame([80.0] * 6 + [-c + 200 for c in bowl()] + [79.0])
    assert not patterns.detect(
        "rounding_bottom", dome, window=60, rim=4, min_depth_pct=5.0
    ).any()
    assert chart.rounding_top(dome, window=60, rim=4, min_depth_pct=5.0).any()


def test_a_shallow_bowl_is_a_wobble_not_a_turn():
    df = frame([100.0] * 6 + bowl(depth=1.0) + [100.5])
    assert not patterns.detect(
        "rounding_bottom", df, window=60, rim=4, min_depth_pct=10.0
    ).any()


def test_the_rounding_bottom_fires_once_on_the_bar_that_clears_the_rim():
    df = frame([100.0] * 6 + bowl() + [101.0, 102.0, 103.0, 104.0])
    hits = patterns.detect(
        "rounding_bottom", df, window=60, rim=4, min_depth_pct=5.0
    )
    assert hits.sum() == 1


def test_a_cup_with_a_shallow_handle_breaks_out():
    df = frame([100.0] * 6 + bowl() + [101.0]
               + leg(101, 96, 5) + leg(96, 101, 4) + [103.0])
    assert patterns.detect(
        "cup_and_handle", df, window=60, rim=4, handle=10, min_depth_pct=5.0
    ).any()


def test_a_handle_that_gives_back_most_of_the_cup_is_not_a_handle():
    """Bulkowski puts the handle "in the upper half of the cup"."""
    df = frame([100.0] * 6 + bowl() + [101.0]
               + leg(101, 83, 5) + leg(83, 101, 4) + [103.0])
    assert not patterns.detect(
        "cup_and_handle", df, window=60, rim=4, handle=10, min_depth_pct=5.0
    ).any()


def test_cup_and_handle_outranks_the_plain_rounding_bottom():
    """It is a rounding bottom plus a handle plus a breakout, so it is the
    narrower claim. Registered below, `classify()` would give every one of its
    hits to the broader signal -- which is what happened to squeeze_release."""
    order = registry.by_specificity()
    assert order.index("cup_and_handle") < order.index("rounding_bottom")


# --- flags: Bulkowski --------------------------------------------------------


def test_a_steep_pole_then_a_tight_pause_then_a_break():
    df = frame([100.0] * 20 + leg(100, 140, 10)
               + [138, 136, 137, 135, 134, 136, 135, 134, 135, 136,
                  134, 135, 136, 135, 137] + [142.0])
    assert patterns.detect("flag_breakout", df, pole=10, flag=15).any()


def test_no_pole_means_no_flag():
    """"Without a straight-line price run -- the flagpole -- you don't have a
    flag." A drift of a couple of percent is not a pole."""
    df = frame([100.0] * 20 + leg(100, 102, 10)
               + [101.5] * 15 + [103.0])
    assert not patterns.detect("flag_breakout", df, pole=10, flag=15).any()


def test_a_pause_as_wide_as_the_pole_is_a_channel_not_a_flag():
    df = frame([100.0] * 20 + leg(100, 140, 10)
               + list(np.linspace(139, 105, 15)) + [142.0])
    assert not patterns.detect("flag_breakout", df, pole=10, flag=15).any()


def test_the_bearish_flag_is_the_mirror():
    df = frame([140.0] * 20 + leg(140, 100, 10)
               + [102, 104, 103, 105, 106, 104, 105, 106, 105, 104,
                  106, 105, 104, 105, 103] + [98.0])
    assert chart.flag_breakdown(df, pole=10, flag=15).any()


# --- failed breaks: the PAPA sheet's own wording ----------------------------


def test_a_breakdown_that_closes_back_above_the_level_is_a_fake_breakdown():
    df = frame([100.0] * 25 + [94.0] + [97.0, 101.0])
    assert patterns.detect("fake_breakdown", df, lookback=20, recovery=5).any()


def test_a_breakdown_that_keeps_going_is_not_fake():
    df = frame([100.0] * 25 + [94.0, 92.0, 90.0, 88.0])
    assert not patterns.detect("fake_breakdown", df, lookback=20, recovery=5).any()


def test_a_recovery_that_takes_too_long_is_not_a_trap():
    slow = frame([100.0] * 25 + [94.0] + [94.5] * 8 + [101.0])
    assert not patterns.detect("fake_breakdown", slow, lookback=20, recovery=5).any()
    assert patterns.detect(
        "fake_breakdown", slow, lookback=20, recovery=20
    ).any()


def test_the_bull_trap_is_the_mirror():
    df = frame([100.0] * 25 + [106.0] + [103.0, 99.0])
    assert chart.fake_breakout(df, lookback=20, recovery=5).any()


# --- the properties that hold for the whole family --------------------------


@pytest.mark.parametrize("name", BULLISH)
def test_prefix_invariance(name):
    """Every one of these reads swing pivots or rolling windows, and a pattern
    that changes once the future arrives is not a pattern."""
    rng = np.random.default_rng(5)
    df = frame(100 + np.cumsum(rng.normal(0, 1.3, 320)))
    full = patterns.detect(name, df)
    for t in range(200, 320, 13):
        prefix = patterns.detect(name, df.iloc[: t + 1])
        assert prefix.iloc[t] == full.iloc[t], f"{name} changed at bar {t}"


@pytest.mark.parametrize("name", BULLISH)
def test_it_never_leaks_across_symbols(name):
    a = frame([100.0] * 6 + bowl() + [101.0, 102.0], "AAA")
    b = frame([50.0] * len(a), "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    hits = patterns.detect_by_symbol(both, [name])
    alone = patterns.detect_all(a, [name])
    assert (
        hits.loc[both["symbol"] == "AAA", name].to_numpy()
        == alone[name].to_numpy()
    ).all()


@pytest.mark.parametrize("name", BULLISH)
def test_registered_with_config(name):
    assert name in registry.names()
    assert registry.load_params()[name]


@pytest.mark.parametrize("name", BEARISH)
def test_the_short_side_is_exported_and_not_registered(name):
    """Every registered signal here is a long entry, and Stages 7 and 9 score a
    signal by whether an upside target is hit before a downside stop."""
    assert name not in registry.names()
    assert callable(getattr(chart, name))
