"""The decision sheet's back half: levels, risk, reward, size.

The tests that matter here are not the arithmetic -- that is four subtractions
and a division -- but the three places where a plausible implementation would
quietly produce a number that means nothing: a target derived from the stop, a
sizing that ignores one of its two limits, and a level that could only be known
after the fact.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import tradeplan as tp

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume"]


def frame(closes, symbol="AAA", volume=100_000, pad=0.5):
    closes = [float(c) for c in closes]
    vols = volume if isinstance(volume, list) else [volume] * len(closes)
    return pd.DataFrame(
        [{"date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=i),
          "symbol": symbol, "open": c, "high": c + pad, "low": c - pad,
          "close": c, "volume": float(v)}
         for i, (c, v) in enumerate(zip(closes, vols))],
        columns=COLUMNS,
    )


def one(v):
    return pd.Series([float(v)])


ACCOUNT = tp.Account(capital=100_000, max_risk_pct=0.02, investment_per_trade=20_000)


# --- against the sheet's own worked example ---------------------------------


def test_the_arithmetic_reproduces_the_filled_example():
    """The ASTA sheet ships one filled column: close 269, stop 266, targets 306
    and 308. It computes risk 3, rewards 37 and 39, and R:R 12.33 and 13."""
    r = tp.size(one(269), one(266), one(306), one(308), account=ACCOUNT).iloc[0]
    assert r["risk"] == pytest.approx(3.0)
    assert r["reward_min"] == pytest.approx(37.0)
    assert r["reward_max"] == pytest.approx(39.0)
    assert r["rr_min"] == pytest.approx(12.3333, abs=1e-4)
    assert r["rr_max"] == pytest.approx(13.0)


def test_the_risk_cap_follows_the_template_formula_not_the_example_cell():
    """Row 29 is INT(max risk / risk per share) = INT(2000 / 3) = 666.

    The shipped example computes 37 in that cell, from `capital / 270` -- a
    different formula than the one in every other column of the same row. The
    template is the specification; the example column is a hand-edit with a
    mistake in it, and following it would size every trade off nothing.
    """
    r = tp.size(one(269), one(266), one(306), one(308), account=ACCOUNT).iloc[0]
    assert r["max_shares_allowed"] == 666


# --- the reconciliation the sheet never makes -------------------------------


def test_both_sizings_are_reported_and_the_smaller_binds():
    """Row 29 caps shares by the 2% risk budget, row 32 sets them from a fixed
    investment, and the sheet reconciles neither. On its own example those are
    666 and 74."""
    r = tp.size(one(269), one(266), one(306), one(308), account=ACCOUNT).iloc[0]
    assert r["max_shares_allowed"] == 666
    assert r["shares_by_investment"] == 74
    assert r["shares"] == 74
    assert r["binding"] == "investment"


def test_binding_says_risk_when_the_stop_is_wide():
    """A stop 40 rupees away lets the 2000-rupee budget buy only 50 shares,
    while the investment would buy 74. Now risk is the constraint."""
    r = tp.size(one(269), one(229), one(400), one(400), account=ACCOUNT).iloc[0]
    assert r["max_shares_allowed"] == 50
    assert r["shares"] == 50
    assert r["binding"] == "risk"


def test_risk_actually_taken_never_exceeds_the_cap():
    close, stop = one(269), one(266)
    r = tp.size(close, stop, one(306), one(308), account=ACCOUNT).iloc[0]
    assert r["risk_involved"] <= ACCOUNT.capital * ACCOUNT.max_risk_pct
    assert r["risk_pct_of_capital"] <= ACCOUNT.max_risk_pct * 100


# --- the tautology guard ----------------------------------------------------


def test_a_setup_with_no_structure_above_gets_no_target_and_no_ratio():
    """The one thing that would make the R:R gate meaningless is deriving the
    target from the stop. Then every setup clears 3 by construction and the
    sheet's own entry rule stops filtering anything. A breakout to new highs has
    nothing above it, and the honest answer is no plan rather than a number."""
    r = tp.size(one(100), one(95), one(np.nan), one(np.nan), account=ACCOUNT).iloc[0]
    assert pd.isna(r["target_min"])
    assert pd.isna(r["rr_min"])
    assert not r["meets_rr"]


def test_the_gate_is_the_sheets_own_threshold():
    """Row 13: "Should be above 3 to enter the trade"."""
    ok = tp.size(one(100), one(95), one(116), one(116), account=ACCOUNT).iloc[0]
    no = tp.size(one(100), one(95), one(110), one(110), account=ACCOUNT).iloc[0]
    assert ok["rr_min"] == pytest.approx(3.2) and ok["meets_rr"]
    assert no["rr_min"] == pytest.approx(2.0) and not no["meets_rr"]


def test_targets_move_with_structure_not_with_the_stop():
    """Same stop, two different targets: the ratio must change. If it did not,
    the target would be a function of the risk."""
    a = tp.size(one(100), one(95), one(115), one(115), account=ACCOUNT).iloc[0]
    b = tp.size(one(100), one(95), one(130), one(130), account=ACCOUNT).iloc[0]
    assert a["rr_min"] != b["rr_min"]


def test_an_inverted_stop_is_not_a_trade():
    r = tp.size(one(100), one(105), one(120), one(120), account=ACCOUNT).iloc[0]
    assert pd.isna(r["risk"])
    assert not r["meets_rr"]


# --- levels -----------------------------------------------------------------


def test_the_significant_candle_is_the_heaviest_one():
    """Row 7's "Median of the Most Significant Candle i.e. the one with Heavy
    Volume" -- the midpoint of the highest-volume bar in the lookback."""
    closes = [100.0] * 35 + [80.0] + [100.0] * 10
    vols = [1000] * 35 + [90_000] + [1000] * 10
    df = frame(closes, volume=vols)
    med = tp.significant_candle_median(df, lookback=30)
    assert med.iloc[-1] == pytest.approx(80.0)      # (80.5 + 79.5) / 2


def test_the_significant_candle_is_unknown_until_the_window_is_full():
    """NaN until there is enough history for it to mean something, like every
    other rolling measure in this project."""
    df = frame([100.0] * 40)
    assert tp.significant_candle_median(df, lookback=30).iloc[:29].isna().all()


def test_the_stop_sits_below_the_immediate_support():
    """Row 11 says below it, not on it. A stop placed exactly on the level is
    taken by the noise that defines the level."""
    rng = np.random.default_rng(2)
    df = frame(100 + np.cumsum(rng.normal(0, 1.5, 200)))
    p = tp.plan(df)
    both = p["stop"].notna() & p["immediate_support"].notna()
    assert both.any()
    assert (p.loc[both, "stop"] < p.loc[both, "immediate_support"]).all()


def test_support_is_below_price_and_resistance_above():
    rng = np.random.default_rng(4)
    df = frame(100 + np.cumsum(rng.normal(0, 1.5, 250)))
    lv = tp.levels(df)
    sup = lv["immediate_support"].notna()
    res = lv["immediate_resistance"].notna()
    assert (lv.loc[sup, "immediate_support"] < df.loc[sup, "close"]).all()
    assert (lv.loc[res, "immediate_resistance"] > df.loc[res, "close"]).all()


def test_levels_never_look_ahead():
    """Everything here reads confirmed pivots and trailing windows, so the
    levels on a bar must not change once later bars arrive."""
    rng = np.random.default_rng(9)
    df = frame(100 + np.cumsum(rng.normal(0, 1.4, 260)))
    full = tp.levels(df)
    for t in range(150, 260, 17):
        prefix = tp.levels(df.iloc[: t + 1])
        for col in ("immediate_support", "major_resistance", "significant_median"):
            a, b = prefix[col].iloc[t], full[col].iloc[t]
            assert (pd.isna(a) and pd.isna(b)) or a == pytest.approx(b), col


def test_levels_never_leak_across_symbols():
    a = frame(list(np.linspace(100, 130, 120)), "AAA")
    b = frame(list(np.linspace(500, 400, 120)), "BBB")
    both = pd.concat([a, b], ignore_index=True).sort_values(["date", "symbol"])
    grouped = tp.levels(both, by=both["symbol"])
    alone = tp.levels(a)
    got = grouped.loc[both["symbol"] == "AAA", "major_support"].to_numpy()
    assert np.allclose(got, alone["major_support"].to_numpy(), equal_nan=True)


# --- the frame contract -----------------------------------------------------


def test_an_empty_hit_set_still_carries_the_full_schema():
    """Same contract as ScanResult.hits: callers never special-case "nothing"."""
    out = tp.for_hits(pd.DataFrame(columns=["date", "symbol", "pattern"]), frame([1.0]))
    assert list(out.columns) == tp.PLAN_COLUMNS


def test_a_plan_carries_every_declared_column():
    df = frame(list(np.linspace(100, 140, 200)))
    assert set(tp.plan(df).columns) <= set(tp.PLAN_COLUMNS)


def test_only_the_long_side_is_implemented():
    with pytest.raises(ValueError, match="only 'long' is implemented"):
        tp.plan(frame([100.0] * 30), direction="short")


@pytest.mark.parametrize("kwargs", [
    {"capital": 0}, {"max_risk_pct": 0}, {"max_risk_pct": 1.5},
    {"investment_per_trade": -1},
])
def test_incoherent_accounts_are_refused(kwargs):
    with pytest.raises(ValueError):
        tp.Account(**kwargs)
