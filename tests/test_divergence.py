"""Divergence between price and an oscillator.

The arithmetic is a four-way comparison and is easy. What these tests are really
for is the two claims that make it usable: it cannot see a pivot before that
pivot is confirmed, and it fires once per divergence rather than once per bar
for the length of the swing.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import patterns
from nse_screener.patterns import _swings as sw
from nse_screener.patterns import divergence as dv
from nse_screener.patterns import registry

COLUMNS = ["date", "symbol", "open", "high", "low", "close"]


def frame(closes, symbol="AAA"):
    closes = [float(c) for c in closes]
    return pd.DataFrame(
        [{"date": pd.Timestamp("2022-01-03") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c + 0.3, "low": c - 0.3, "close": c}
         for i, c in enumerate(closes)],
        columns=COLUMNS,
    )


def wobble(a, b, n, give_back):
    """Walk from `a` to `b` over `n` bars, giving back `give_back` of every step.

    The point of this rather than a straight line: RSI measures the average
    *size* of up moves against down moves, not the distance travelled. A pure
    linear decline has no up days at all, so RSI pins at exactly 0 and the
    second trough can never read lower than the first -- which quietly makes
    half the cases in this file untestable. Give-back is the dial: more of it
    means more up days inside the fall, and a higher RSI at the bottom.
    """
    pairs = n // 2
    step = abs(b - a) / (pairs * (1 - give_back))
    sign = -1 if b < a else 1
    out, x = [], a
    for _ in range(pairs):
        x += sign * step
        out.append(x)
        x -= sign * step * give_back
        out.append(x)
    return out


def two_troughs(first, second, give_back_1, give_back_2):
    """Two selloffs to named depths, each with its own steepness.

    Depth sets which way price's second low went; give-back sets which way the
    oscillator's did. Choosing them independently is what lets one fixture
    produce a divergence and its near-twin produce agreement.
    """
    return frame(
        [100.0] * 200
        + wobble(100, first, 20, give_back_1)
        + wobble(first, 96, 12, 0.0)
        + wobble(96, second, 20, give_back_2)
        + wobble(second, second + 9, 10, 0.0)
    )


def two_peaks(first, second, give_back_1, give_back_2):
    """The same shape inverted, for the bearish readings."""
    return frame(
        [100.0] * 200
        + wobble(100, first, 20, give_back_1)
        + wobble(first, 104, 12, 0.0)
        + wobble(104, second, 20, give_back_2)
        + wobble(second, second - 9, 10, 0.0)
    )


DIVERGING = {"first": 82.0, "second": 79.0,
             "give_back_1": 0.15, "give_back_2": 0.55}
"""Price makes a lower low; the second fall is choppier, so RSI makes a higher
one. The textbook regular bullish divergence."""

AGREEING = {"first": 82.0, "second": 79.0,
            "give_back_1": 0.55, "give_back_2": 0.10}
"""Same lower low, second fall steeper -- RSI follows price down. Confirmation,
not divergence, and the case a naive implementation gets wrong."""


# --- regular bullish --------------------------------------------------------


def test_a_lower_price_low_with_a_higher_oscillator_low_is_a_divergence():
    hits = patterns.detect("bullish_divergence", two_troughs(**DIVERGING))
    assert hits.sum() == 1


def test_no_divergence_when_the_oscillator_agrees_with_price():
    """Second selloff steeper as well as lower -- both make a lower low, which
    is confirmation, not divergence."""
    assert not patterns.detect("bullish_divergence", two_troughs(**AGREEING)).any()


def test_no_divergence_without_a_lower_price_low():
    """Higher low in price and in RSI: the two agree, in the other direction."""
    df = two_troughs(first=79.0, second=82.0, give_back_1=0.15, give_back_2=0.55)
    assert not patterns.detect("bullish_divergence", df).any()


@pytest.mark.parametrize("right", [3, 5, 8])
def test_it_fires_exactly_right_bars_after_the_swing_low(right):
    """The bar a trader could first have seen it -- never the pivot bar."""
    df = two_troughs(**DIVERGING)
    hits = patterns.detect("bullish_divergence", df, right=right)
    piv = sw.pivots(df, left=5, right=right)
    lows = sw.track(df["low"], piv["pivot_low"], right=right)
    for t in df.index[hits]:
        assert t - lows["last_at"].loc[t] == right


def test_a_divergence_is_an_event_not_a_state():
    """Held True until the next pivot it would fire on thirty consecutive bars,
    turning one observation into thirty and telling Stage 7 something false
    about how often the thing happens."""
    assert patterns.detect("bullish_divergence", two_troughs(**DIVERGING)).sum() == 1


# --- the other three readings -----------------------------------------------


def test_hidden_bullish_is_the_mirror_of_regular_bullish():
    """Higher price low, lower oscillator low: a shallow dip inside an uptrend."""
    df = two_troughs(first=79.0, second=82.0, give_back_1=0.55, give_back_2=0.10)
    assert patterns.detect("hidden_bullish_divergence", df).any()
    assert not patterns.detect("bullish_divergence", df).any()


def test_bearish_divergence_reads_the_highs():
    peaks = two_peaks(first=118.0, second=121.0, give_back_1=0.15, give_back_2=0.55)
    assert dv.bearish_divergence(peaks).any()


def test_the_bearish_readings_are_deliberately_not_registered():
    """Every registered signal here is a long entry, and Stages 7 and 9 score a
    signal by whether an upside target is hit before a downside stop. A bearish
    signal measured that way reports a number that reads like a hit rate and is
    not one. Same reasoning as heikin_ashi.flip_down."""
    assert "bearish_divergence" not in registry.names()
    assert "hidden_bearish_divergence" not in registry.names()
    assert callable(dv.bearish_divergence)
    assert callable(dv.hidden_bearish_divergence)


# --- configuration ----------------------------------------------------------


def test_the_macd_oscillator_is_selectable_and_reads_differently():
    """Both oscillators find divergences, and not the same ones.

    That they disagree is the evidence the setting is consulted rather than
    quietly ignored -- and it is also the honest description of the choice. RSI
    reads the ratio of average gain to average loss; the MACD histogram reads
    one EMA pulling away from another. Two different questions, so a bar can
    diverge on one and not the other, and neither answer is the right one.
    """
    rng = np.random.default_rng(3)
    walk = frame(100 + np.cumsum(rng.normal(0, 1.4, 320)))
    by_rsi = patterns.detect("bullish_divergence", walk, oscillator="rsi")
    by_macd = patterns.detect("bullish_divergence", walk, oscillator="macd_hist")
    assert by_rsi.any() and by_macd.any()
    assert not by_rsi.equals(by_macd)


def test_an_unknown_oscillator_is_a_named_error():
    with pytest.raises(ValueError, match="unknown oscillator"):
        patterns.detect(
            "bullish_divergence", two_troughs(**DIVERGING), oscillator="stoch"
        )


def test_pivots_too_far_apart_are_not_one_divergence():
    df = two_troughs(**DIVERGING)
    assert patterns.detect("bullish_divergence", df).any()
    assert not patterns.detect("bullish_divergence", df, max_bars_between=10).any()


def test_the_oscillator_is_the_one_the_context_table_shows():
    """Not a private reimplementation -- the detector and the RSI column on the
    chart must not disagree about the same bar."""
    from nse_screener import context as ctx

    df = two_troughs(**DIVERGING)
    assert dv._oscillator(df, "rsi", None, 14).equals(ctx.rsi(df["close"], None, 14))


# --- lookahead and boundaries -----------------------------------------------


def test_prefix_invariance():
    """The test that matters. Divergence is the most commonly repainted signal
    in technical analysis, and it repaints because the second pivot is drawn
    before it is confirmed."""
    rng = np.random.default_rng(3)
    df = frame(100 + np.cumsum(rng.normal(0, 1.4, 320)))
    full = patterns.detect("bullish_divergence", df)
    for t in range(150, 320, 11):
        prefix = patterns.detect("bullish_divergence", df.iloc[: t + 1])
        assert prefix.iloc[t] == full.iloc[t], f"bar {t} changed once the future arrived"


def test_it_never_leaks_across_symbols():
    a = two_troughs(**DIVERGING)
    b = frame([100.0] * len(a), "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    hits = patterns.detect_by_symbol(both, ["bullish_divergence"])
    alone = patterns.detect_all(a, ["bullish_divergence"])
    assert (
        hits.loc[both["symbol"] == "AAA", "bullish_divergence"].to_numpy()
        == alone["bullish_divergence"].to_numpy()
    ).all()


def test_both_bullish_readings_are_registered_with_config():
    config = registry.load_params()
    for name in ("bullish_divergence", "hidden_bullish_divergence"):
        assert name in registry.names()
        assert config[name]


def test_divergence_outranks_the_broader_signals():
    """Two confirmed swings plus an oscillator disagreement is a much narrower
    claim than a breakout, so it must win the classify tiebreak."""
    order = registry.by_specificity()
    assert order.index("bullish_divergence") < order.index("breakout_20")
    assert order.index("fib_retracement") < order.index("doji")


# --- the checklist's grade: agreement with the trade ------------------------


def test_a_bullish_divergence_agrees_with_a_long():
    g = dv.agreement(two_troughs(**DIVERGING), "long")
    assert (g == "with").sum() == 1
    assert not (g == "against").any()


def test_the_same_divergence_argues_against_a_short():
    """The checklist's point: a divergence is read *relative to the trade*, so
    one bar is encouragement or a caution depending on which way you are
    leaning. Nothing about the bar changes."""
    df = two_troughs(**DIVERGING)
    assert (dv.agreement(df, "long") == "with").sum() == 1
    assert (dv.agreement(df, "short") == "against").sum() == 1


def test_a_bearish_divergence_cautions_a_long():
    peaks = two_peaks(first=118.0, second=121.0, give_back_1=0.15, give_back_2=0.55)
    assert (dv.agreement(peaks, "long") == "against").any()
    assert (dv.agreement(peaks, "short") == "with").any()


def test_bars_with_no_divergence_are_not_graded():
    g = dv.agreement(two_troughs(**AGREEING), "long")
    assert g.isna().all()


def test_hidden_readings_count_towards_agreement():
    """Hidden and regular differ in what they say about the trend, not in which
    way they lean."""
    df = two_troughs(first=79.0, second=82.0, give_back_1=0.55, give_back_2=0.10)
    assert (dv.agreement(df, "long") == "with").any()


def test_a_caution_cannot_be_cancelled_by_a_second_signal():
    """Where both a bullish and a bearish divergence land on the same bar --
    possible, since they are read off different pivots -- the caution wins. A
    caution a second signal can cancel is not a caution."""
    import pandas as pd

    df = two_troughs(**DIVERGING)
    both = pd.Series(True, index=df.index)
    saved = dv.bearish_divergence
    try:
        dv.bearish_divergence = lambda d, **kw: both
        assert (dv.agreement(df, "long") == "against").all()
    finally:
        dv.bearish_divergence = saved


def test_an_unknown_direction_is_a_named_error():
    with pytest.raises(ValueError, match="'long' or 'short'"):
        dv.agreement(two_troughs(**DIVERGING), "sideways")
