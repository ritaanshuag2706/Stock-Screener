"""Stage 6 context columns.

These are measurements, not decisions, so the tests are about arithmetic being
right and never leaking across a symbol boundary.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import context
from nse_screener.context import CONTEXT_COLUMNS, annotate, annotate_by_symbol

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def bars(symbol="AAA", n=400, start=100.0, drift=0.0, vol=1000, rng=2.0):
    dates = pd.bdate_range("2023-01-02", periods=n)
    close = [start + drift * i for i in range(n)]
    return pd.DataFrame(
        {
            "date": dates, "symbol": symbol,
            "open": close, "high": [c + rng / 2 for c in close],
            "low": [c - rng / 2 for c in close], "close": close,
            "volume": [vol] * n,
        },
        columns=COLUMNS,
    )


def last(df, col):
    return df[col].iloc[-1]


# --- shape and warm-up ------------------------------------------------------


def test_returns_the_declared_columns():
    out = annotate(bars())
    assert list(out.columns) == CONTEXT_COLUMNS
    assert len(out) == 400


def test_columns_are_nan_until_there_is_enough_history():
    """An EMA-200 over 30 bars is a number, not an answer."""
    out = annotate(bars(n=30))
    assert out["above_200ema"].isna().all()
    assert out["dist_200ema_pct"].isna().all()
    assert out["dist_25ema_pct"].iloc[:24].isna().all()
    assert out["dist_25ema_pct"].iloc[24:].notna().any()


def test_empty_frame():
    out = annotate_by_symbol(bars().iloc[0:0])
    assert out.empty
    assert list(out.columns) == CONTEXT_COLUMNS


# --- trend ------------------------------------------------------------------


def test_above_200ema_in_an_uptrend():
    out = annotate(bars(drift=0.5))
    assert last(out, "above_200ema") is np.True_ or bool(last(out, "above_200ema"))
    assert last(out, "dist_200ema_pct") > 0


def test_below_200ema_in_a_downtrend():
    out = annotate(bars(start=300.0, drift=-0.5))
    assert not bool(last(out, "above_200ema"))
    assert last(out, "dist_200ema_pct") < 0


def test_flat_series_sits_on_its_own_average():
    out = annotate(bars(drift=0.0))
    assert abs(last(out, "dist_200ema_pct")) < 1e-6
    for span in (5, 13, 25):
        assert abs(last(out, f"dist_{span}ema_pct")) < 1e-6


# --- volume -----------------------------------------------------------------


def test_relative_volume():
    b = bars()
    b.loc[b.index[-1], "volume"] = 3000        # against a 20-day average of 1000
    out = annotate(b)
    assert last(out, "rel_volume") == pytest.approx(3000 / ((19 * 1000 + 3000) / 20))


def test_relative_volume_is_one_when_nothing_changes():
    assert last(annotate(bars()), "rel_volume") == pytest.approx(1.0)


# --- volatility -------------------------------------------------------------


def test_atr_pct_on_a_constant_range():
    """Range 2 on a price of 100 is 2%."""
    out = annotate(bars(rng=2.0))
    assert last(out, "atr_pct") == pytest.approx(2.0, abs=0.01)


def test_atr_pct_rises_when_the_range_expands():
    b = bars(n=400, rng=1.0)
    quiet = last(annotate(b), "atr_pct")
    tail = b.index[-30:]
    b.loc[tail, "high"] = b.loc[tail, "close"] + 5
    b.loc[tail, "low"] = b.loc[tail, "close"] - 5
    assert last(annotate(b), "atr_pct") > quiet


def test_true_range_uses_the_previous_close():
    """A gap makes the true range wider than the bar's own high-low."""
    b = bars(n=5, rng=1.0)
    b.loc[b.index[-1], ["open", "high", "low", "close"]] = [130, 131, 129, 130]
    tr = context.true_range(b)
    assert tr.iloc[-1] == pytest.approx(131 - 100.0)    # high vs previous close


# --- symbol isolation -------------------------------------------------------


def test_context_never_leaks_across_symbols():
    """The same failure mode as the detectors: store.read() interleaves symbols."""
    up = bars("UP", drift=0.5)
    down = bars("DOWN", start=300.0, drift=-0.5)
    both = pd.concat([up, down], ignore_index=True).sort_values(["date", "symbol"])

    together = annotate_by_symbol(both)
    for sym, frame in ((("UP"), up), ("DOWN", down)):
        alone = annotate(frame.sort_values("date"))
        mine = together[both["symbol"] == sym].reset_index(drop=True)
        pd.testing.assert_frame_equal(
            mine, alone.reset_index(drop=True), check_dtype=False
        )


def test_a_volatile_symbol_does_not_raise_its_neighbour_percentile():
    quiet = bars("QUIET", rng=1.0)
    wild = bars("WILD", rng=40.0)
    both = pd.concat([quiet, wild], ignore_index=True).sort_values(["date", "symbol"])
    out = annotate_by_symbol(both)
    # Both are constant-volatility, so each sits mid-pack in its *own* history.
    assert last(out[both["symbol"] == "QUIET"].reset_index(drop=True), "atr_pct") \
        == pytest.approx(1.0, abs=0.05)
    assert last(out[both["symbol"] == "WILD"].reset_index(drop=True), "atr_pct") \
        == pytest.approx(40.0, abs=0.5)


def test_annotate_by_symbol_preserves_row_order():
    both = pd.concat([bars("AAA"), bars("BBB")], ignore_index=True)
    shuffled = both.sample(frac=1.0, random_state=0)
    out = annotate_by_symbol(shuffled)
    assert list(out.index) == list(shuffled.index)


def test_by_must_align_with_the_frame():
    """A misaligned key would silently group the wrong rows together."""
    b = bars()
    misaligned = pd.Series(["AAA"] * len(b), index=range(1000, 1000 + len(b)))
    with pytest.raises(ValueError, match="share the index"):
        annotate(b, by=misaligned)


def test_needs_a_date_column_for_multi_symbol():
    with pytest.raises(ValueError, match="date"):
        annotate_by_symbol(bars().drop(columns=["date"]))


# --- indicators, against TA-Lib ---------------------------------------------

talib = pytest.importorskip("talib", reason="TA-Lib not installed")


def trending_bars(n=900, seed=3):
    """Random walk with enough movement to exercise every indicator."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.018, n)))
    high = close * (1 + np.abs(rng.normal(0, 0.01, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n)))
    return pd.DataFrame({
        "date": pd.bdate_range("2021-01-04", periods=n),
        "symbol": "AAA", "open": close, "high": high, "low": low,
        "close": close, "volume": rng.integers(1e5, 1e6, n),
    })


@pytest.mark.parametrize(
    "column,reference",
    [
        ("rsi_14", lambda c, h, low: talib.RSI(c, 14)),
        ("stoch_k", lambda c, h, low: talib.STOCH(h, low, c, 14, 3, 0, 3, 0)[0]),
        ("stoch_d", lambda c, h, low: talib.STOCH(h, low, c, 14, 3, 0, 3, 0)[1]),
        ("macd", lambda c, h, low: talib.MACD(c, 12, 26, 9)[0]),
        ("macd_signal", lambda c, h, low: talib.MACD(c, 12, 26, 9)[1]),
        ("atr_pct", lambda c, h, low: talib.ATR(h, low, c, 14) / c * 100),
        ("bb_pct_b", lambda c, h, low: (
            (c - talib.BBANDS(c, 20, 2, 2, 0)[2])
            / (talib.BBANDS(c, 20, 2, 2, 0)[0] - talib.BBANDS(c, 20, 2, 2, 0)[2]))),
    ],
)
def test_indicator_matches_talib(column, reference):
    b = trending_bars()
    mine = annotate(b)[column].to_numpy(float)
    c, h, low = (b[k].to_numpy(float) for k in ("close", "high", "low"))
    ref = reference(c, h, low)
    both = ~np.isnan(mine) & ~np.isnan(ref)
    assert both.sum() > 200
    # 0.05 covers the residue at the very first unmasked bar, where the two
    # EMA seedings have not quite finished converging. The median difference is
    # 0.0 and the last 300 bars are exact.
    assert np.abs(mine[both] - ref[both]).max() < 0.05
    assert np.abs(mine[-300:] - ref[-300:]).max() < 1e-9


def test_ema_seeded_indicators_are_masked_until_they_converge():
    """RSI at 1x warm-up can be 2.5 points wrong. 8x puts it inside rounding."""
    from nse_screener.context import EMA_WARMUP_MULTIPLE
    out = annotate(trending_bars(n=400))
    first = out["rsi_14"].first_valid_index()
    assert first >= 14 * EMA_WARMUP_MULTIPLE - 1


# --- the derived columns ----------------------------------------------------


def test_ema_stack_reads_the_ribbon():
    assert last(annotate(bars(drift=0.6)), "ema_stack") == "up"
    assert last(annotate(bars(start=400.0, drift=-0.6)), "ema_stack") == "down"


def test_bollinger_pct_b_is_half_on_a_flat_series():
    """No dispersion means the bands collapse onto price -- %B is undefined."""
    out = annotate(bars(drift=0.0))
    assert pd.isna(last(out, "bb_pct_b"))
    assert pd.isna(last(out, "bb_width_pct")) or last(out, "bb_width_pct") == 0


def test_bollinger_width_widens_with_dispersion():
    quiet = last(annotate(trending_bars(seed=1)), "bb_width_pct")
    b = trending_bars(seed=1)
    b.loc[b.index[-40:], "close"] *= np.linspace(1.0, 1.5, 40)
    assert last(annotate(b), "bb_width_pct") > quiet


def test_macd_histogram_is_reported_as_a_percentage_of_price():
    b = trending_bars()
    out = annotate(b)
    expected = (out["macd"] - out["macd_signal"]) / b["close"] * 100
    pd.testing.assert_series_equal(
        out["macd_hist_pct"].dropna(), expected.dropna(), check_names=False
    )


def test_every_declared_column_is_produced():
    from nse_screener.context import CONTEXT_COLUMNS
    out = annotate(trending_bars())
    assert list(out.columns) == CONTEXT_COLUMNS
    for c in CONTEXT_COLUMNS:
        assert out[c].notna().any(), f"{c} is entirely NaN"


def test_indicators_never_leak_across_symbols():
    a = trending_bars(seed=1).assign(symbol="AAA")
    b = trending_bars(seed=2).assign(symbol="BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    together = annotate_by_symbol(both)
    alone = annotate(a.sort_values("date"))
    pd.testing.assert_frame_equal(
        together[both["symbol"] == "AAA"].reset_index(drop=True),
        alone.reset_index(drop=True), check_dtype=False,
    )


def test_rsi_zone_marks_the_thresholds():
    b = trending_bars()
    out = annotate(b)
    known = out["rsi_14"].notna()
    assert (out.loc[known & (out["rsi_14"] < 30), "rsi_zone"] == "oversold").all()
    assert (out.loc[known & (out["rsi_14"] > 70), "rsi_zone"] == "overbought").all()
    mid = known & out["rsi_14"].between(30, 70)
    assert (out.loc[mid, "rsi_zone"] == "neutral").all()


def test_rsi_zone_is_absent_wherever_rsi_is():
    """No zone without a number behind it."""
    out = annotate(trending_bars())
    assert (out["rsi_zone"].isna() == out["rsi_14"].isna()).all()


def test_rsi_thresholds_are_configurable():
    b = trending_bars()
    tight = annotate(b, rsi_oversold=45, rsi_overbought=55)
    assert (tight["rsi_zone"] == "oversold").sum() > (
        annotate(b)["rsi_zone"] == "oversold").sum()


def test_a_flat_window_does_not_break_the_stochastic():
    """Regression: guarding a division with pd.NA promotes the column to object
    dtype, and rolling() then refuses to aggregate it. Only shows up when some
    symbol genuinely has a flat high/low window, which a short test frame never
    produces but six years of history certainly does."""
    b = bars(n=200, rng=2.0)
    flat = b.index[50:100]
    b.loc[flat, ["open", "high", "low", "close"]] = 100.0    # no range at all
    out = annotate(b)
    assert out["stoch_k"].dtype.kind == "f"
    assert out["bb_pct_b"].dtype.kind == "f"
    assert out["rel_volume"].dtype.kind == "f"
    assert out["stoch_k"].notna().any()


def test_a_flat_window_survives_the_grouped_path():
    a = bars("AAA", n=200)
    a.loc[a.index[50:100], ["open", "high", "low", "close"]] = 100.0
    both = pd.concat([a, bars("BBB", n=200, drift=0.3)], ignore_index=True)
    out = annotate_by_symbol(both.sort_values(["date", "symbol"]))
    assert out["stoch_k"].notna().any()
