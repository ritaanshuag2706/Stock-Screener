"""The momentum family. Mostly about the rolling windows not seeing today."""

import numpy as np
import pandas as pd

from nse_screener import patterns
from nse_screener.patterns import registry

COLUMNS = ["date", "symbol", "open", "high", "low", "close"]


def frame(closes, symbol="AAA", pad=0.5):
    return pd.DataFrame(
        [{"date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c + pad, "low": c - pad, "close": c}
         for i, c in enumerate(closes)],
        columns=COLUMNS,
    )


def last(s):
    return bool(s.iloc[-1])


# --- breakouts --------------------------------------------------------------


def test_breakout_fires_on_a_new_high():
    closes = [100.0] * 30 + [110.0]
    assert last(patterns.detect("breakout_20", frame(closes)))


def test_breakout_does_not_fire_inside_the_range():
    assert not last(patterns.detect("breakout_20", frame([100.0] * 31)))


def test_the_window_excludes_todays_own_high():
    """A bar cannot break out because of its own high -- that would be
    lookahead dressed as a signal."""
    closes = [100.0] * 30 + [100.2]        # close inside the prior high of 100.5
    assert not last(patterns.detect("breakout_20", frame(closes)))


def test_breakout_needs_a_full_window_of_history():
    assert not patterns.detect("breakout_20", frame([100.0] * 5 + [200.0])).any()


def test_52_week_breakout_is_stricter_than_20_day():
    """Above the 20-day high but under the year's high: one fires, one does not."""
    closes = [200.0] * 30 + [100.0] * 250 + [150.0]
    df = frame(closes)
    assert last(patterns.detect("breakout_20", df))
    assert not last(patterns.detect("breakout_252", df))


# --- rsi2 pullback ----------------------------------------------------------


def test_rsi2_pullback_needs_both_the_trend_and_the_dip():
    rising = list(np.linspace(100, 200, 260))
    assert not last(patterns.detect("rsi2_pullback", frame(rising)))   # no dip
    dipped = rising[:-2] + [rising[-3] * 0.94, rising[-3] * 0.90]
    assert last(patterns.detect("rsi2_pullback", frame(dipped)))


def test_rsi2_pullback_ignores_a_dip_in_a_downtrend():
    falling = list(np.linspace(200, 100, 260))
    dipped = falling[:-2] + [falling[-3] * 0.94, falling[-3] * 0.90]
    assert not last(patterns.detect("rsi2_pullback", frame(dipped)))


# --- squeeze ----------------------------------------------------------------


def zigzag(n, amplitude, base=100.0):
    """Deterministic alternation, so the rolling width is exactly controlled.
    Random noise was not usable here: over 200 bars it produces a locally tight
    window by chance often enough to make the test flaky."""
    return [base + (amplitude if i % 2 else -amplitude) for i in range(n)]


def test_squeeze_release_needs_prior_compression():
    """Identical breakout bar; only the volatility before it differs."""
    compressed = zigzag(150, 6.0) + zigzag(50, 0.1)
    expanded = zigzag(150, 0.1) + zigzag(50, 6.0)
    assert last(patterns.detect("squeeze_release", frame(compressed + [130.0])))
    assert not last(patterns.detect("squeeze_release", frame(expanded + [130.0])))


# --- registry integration ---------------------------------------------------


def test_the_family_is_registered_with_config():
    config = registry.load_params()
    for name in ("breakout_20", "breakout_252", "rsi2_pullback", "squeeze_release"):
        assert name in registry.names()
        assert name in config


def test_squeeze_outranks_the_plain_breakout():
    """A squeeze release IS a 20-day breakout plus a compression test, so it is
    the narrower claim. Ranked below, it fired 28,679 times and was labelled
    zero times -- every hit absorbed by the broader signal."""
    order = registry.by_specificity()
    assert order.index("squeeze_release") < order.index("breakout_20")


def test_momentum_signals_never_leak_across_symbols():
    a = frame([100.0] * 30 + [110.0], "AAA")
    b = frame([500.0] * 30 + [400.0], "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    hits = patterns.detect_by_symbol(both, ["breakout_20"])
    alone = patterns.detect_all(a.sort_values("date"), ["breakout_20"])
    assert (
        hits.loc[both["symbol"] == "AAA", "breakout_20"].to_numpy()
        == alone["breakout_20"].to_numpy()
    ).all()


def test_windows_are_configurable():
    closes = [100.0] * 12 + [110.0]
    assert not last(patterns.detect("breakout_20", frame(closes)))
    assert last(patterns.detect("breakout_20", frame(closes), window=10))
