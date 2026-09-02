"""Universe eligibility and the scan.

Eligibility is the part worth testing hardest: a symbol wrongly admitted
produces a signal computed over bars that are not comparable, and nothing
downstream can tell.
"""

from datetime import date

import pandas as pd
import pytest

from nse_screener.data import store
from nse_screener.screener import (
    HIT_COLUMNS,
    Universe,
    eligibility,
    gap_flags,
    history_counts,
    latest_session,
    liquidity_flags,
    scan,
)

COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
           "prev_close", "volume", "turnover", "trades", "isin"]

# body 2, range 4 -- the inert baseline from test_patterns
BASE = (100.0, 103.0, 99.0, 102.0)
HAMMER = (99.0, 99.7, 95.0, 99.5)


# Enough volume to clear Universe.min_traded_value at these prices. The default
# used to be 1,000, which is Rs 1 lakh of traded value on a Rs 100 stock -- once
# the liquidity floor existed, every fixture symbol was correctly judged
# untradeable and fifteen tests went dark at once. A fixture that cannot pass the
# rules tests nothing about the rules.
VOLUME = 500_000


def make_bars(symbol, n=300, turnover=1e8, series="EQ", last=None, start="2023-01-02",
              volume=VOLUME):
    dates = pd.bdate_range(start, periods=n)
    rows = []
    for i, d in enumerate(dates):
        o, h, low, c = last if (last and i == n - 1) else BASE
        rows.append({
            "date": d, "symbol": symbol, "series": series,
            "open": o, "high": h, "low": low, "close": c, "prev_close": o,
            "volume": volume, "turnover": turnover, "trades": 10,
            "isin": "INE000A01001",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture
def bars_dir(tmp_path):
    return tmp_path / "bars"


# --- eligibility ------------------------------------------------------------


def counts_for(bars):
    return bars.groupby("symbol").size()


def test_symbol_with_enough_history_is_eligible():
    b = make_bars("AAA")
    e = eligibility(counts_for(b), gap_flags(b, 0.20), Universe())
    assert bool(e.loc[0, "eligible"])


def test_recently_listed_symbol_is_rejected():
    """The trap: a rolling average over a handful of bars means nothing."""
    b = make_bars("AAA", n=100)
    e = eligibility(counts_for(b), gap_flags(b, 0.20), Universe())
    assert not bool(e.loc[0, "enough_history"])
    assert not bool(e.loc[0, "eligible"])


def test_history_counts_reads_every_bar_a_symbol_printed(bars_dir):
    store.append(make_bars("AAA", n=300), bars_dir=bars_dir)
    store.append(make_bars("BBB", n=80), bars_dir=bars_dir)
    counts = history_counts(latest_session(bars_dir), bars_dir=bars_dir)
    assert counts["AAA"] == 300
    assert counts["BBB"] == 80


def test_history_counts_respects_the_series_filter(bars_dir):
    store.append(make_bars("EQ1", series="EQ"), bars_dir=bars_dir)
    store.append(make_bars("BE1", series="BE"), bars_dir=bars_dir)
    counts = history_counts(latest_session(bars_dir), ("EQ",), bars_dir=bars_dir)
    assert "EQ1" in counts and "BE1" not in counts


def test_history_counts_on_an_empty_store(bars_dir):
    assert history_counts(date(2026, 7, 29), bars_dir=bars_dir).empty


# --- the liquidity floor ----------------------------------------------------
#
# Absent, a breakout backtest reported +731%, of which sub-Rs 20 names supplied
# 68.8% of the P&L on 15.2% of the trades. With the floor the same strategy
# returns -82%. These tests are what keep that floor from being removed by
# accident, so each one names the specific way a fill stops being credible.


def test_a_penny_stock_is_not_tradeable():
    """One tick is a large fraction of the price, so an ATR stop is a few ticks
    wide and the R multiples measured against it are arithmetic, not trading."""
    b = make_bars("PENNY")
    b[["open", "high", "low", "close", "prev_close"]] *= 0.05    # ~Rs 5
    assert not bool(liquidity_flags(b, Universe()).loc["PENNY"])


def test_a_thinly_traded_symbol_is_not_tradeable():
    """Priced fine, but nothing changes hands -- the fill is the fiction here."""
    b = make_bars("THIN", volume=100)
    assert not bool(liquidity_flags(b, Universe()).loc["THIN"])


def test_a_liquid_symbol_passes_both_floors():
    assert bool(liquidity_flags(make_bars("AAA"), Universe()).loc["AAA"])


def test_one_frantic_day_does_not_make_a_symbol_liquid():
    """Traded value is a median, not a sum or a max, exactly so that a single
    spike cannot carry a symbol that is dead the rest of the year."""
    b = make_bars("SPIKE", volume=100)
    b.loc[b.index[-1], "volume"] = 10_000_000_000
    assert not bool(liquidity_flags(b, Universe()).loc["SPIKE"])


def test_an_illiquid_symbol_is_dropped_from_the_scan_with_a_reason(bars_dir):
    store.append(make_bars("LIQUID", last=HAMMER), bars_dir=bars_dir)
    store.append(make_bars("THIN", last=HAMMER, volume=100), bars_dir=bars_dir)
    r = scan(which=["hammer"], bars_dir=bars_dir)
    assert set(r.hits["symbol"]) == {"LIQUID"}
    assert r.rejected["too illiquid to trade"] == 1


def test_the_floor_can_be_turned_off_deliberately(bars_dir):
    """A study measuring how often a bar prints wants the widest sample. That is
    a different question from whether the fill is real, so it has to be possible
    to answer it -- but only on purpose, never by forgetting."""
    store.append(make_bars("THIN", last=HAMMER, volume=100), bars_dir=bars_dir)
    rules = Universe(min_price=0.0, min_traded_value=0.0)
    r = scan(which=["hammer"], rules=rules, bars_dir=bars_dir)
    assert set(r.hits["symbol"]) == {"THIN"}


def test_the_floors_reject_negative_thresholds():
    with pytest.raises(ValueError, match="min_price"):
        Universe(min_price=-1)
    with pytest.raises(ValueError, match="min_traded_value"):
        Universe(min_traded_value=-1)


# --- the corporate-action guard ---------------------------------------------


def test_a_split_sized_gap_rejects_the_symbol():
    """An unadjusted 1:2 split halves the price overnight. Bars either side are
    not comparable, so anything computed across it is untrustworthy."""
    b = make_bars("AAA", n=20)
    b.loc[b.index[-5:], ["open", "high", "low", "close"]] *= 0.5
    e = eligibility(counts_for(b), gap_flags(b, 0.20), Universe())
    assert bool(e.loc[0, "recent_gap"])
    assert not bool(e.loc[0, "eligible"])


def test_a_normal_move_does_not_trip_the_gap_filter():
    b = make_bars("AAA", n=20)
    b.loc[b.index[-3:], ["open", "high", "low", "close"]] *= 1.08
    assert not gap_flags(b, 0.20).get("AAA", False)


def test_gap_threshold_is_configurable():
    b = make_bars("AAA", n=20)
    b.loc[b.index[-3:], ["open", "high", "low", "close"]] *= 1.15
    assert not gap_flags(b, 0.20)["AAA"]
    assert gap_flags(b, 0.10)["AAA"]


def test_gap_flags_are_per_symbol():
    """One symbol splitting must not disqualify its neighbours."""
    clean = make_bars("CLEAN", n=20)
    split = make_bars("SPLIT", n=20)
    split.loc[split.index[-3:], ["open", "high", "low", "close"]] *= 0.5
    both = pd.concat([clean, split], ignore_index=True)
    flags = gap_flags(both, 0.20)
    assert not flags["CLEAN"]
    assert flags["SPLIT"]


def test_gap_flags_on_an_empty_frame():
    assert gap_flags(make_bars("AAA").iloc[0:0], 0.20).empty


def test_invalid_rules_raise():
    with pytest.raises(ValueError, match="min_history"):
        Universe(min_history=0)
    with pytest.raises(ValueError, match="max_overnight_gap"):
        Universe(max_overnight_gap=1.5)


def test_no_turnover_anywhere_in_the_rules():
    """Removed deliberately -- it must not creep back in.

    `min_traded_value` is not this rule returning by another name. What was
    removed was turnover as a *screening and display* metric: a column shown to
    the user and ranked on. What exists now is a floor below which a simulated
    fill is not credible, and it is measured from close x volume rather than
    from the store's turnover column. Nothing displays it -- the assertion on
    HIT_COLUMNS below is what keeps that true.
    """
    fields = Universe().__dataclass_fields__
    assert not [f for f in fields if "turnover" in f]
    assert not [c for c in HIT_COLUMNS if "turnover" in c]
    assert not [c for c in HIT_COLUMNS if "traded_value" in c]


# --- scan -------------------------------------------------------------------


def test_scan_finds_a_pattern_on_the_last_session(bars_dir):
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    r = scan(bars_dir=bars_dir)
    assert r.universe_size == 1
    assert list(r.hits["symbol"]) == ["AAA"]
    assert r.hits.loc[0, "pattern"] == "hammer"


def test_scan_excludes_an_ineligible_symbol_even_when_it_prints(bars_dir):
    """The whole point of eligibility: a real pattern on an unusable symbol
    must not reach the output."""
    store.append(make_bars("NEW", n=60, last=HAMMER), bars_dir=bars_dir)
    r = scan(bars_dir=bars_dir)
    assert r.universe_size == 0
    assert r.hits.empty
    assert r.rejected["short history"] == 1


def test_scan_reports_why_symbols_were_dropped(bars_dir):
    store.append(make_bars("GOOD", last=HAMMER), bars_dir=bars_dir)
    store.append(make_bars("NEW", n=50), bars_dir=bars_dir)
    r = scan(bars_dir=bars_dir)
    assert r.universe_size == 1
    assert r.rejected["short history"] == 1


def test_scan_on_an_empty_store_is_not_an_error(bars_dir):
    r = scan(bars_dir=bars_dir)
    assert r.hits.empty and r.universe_size == 0


def test_scan_accepts_an_explicit_date(bars_dir):
    b = make_bars("AAA")
    store.append(b, bars_dir=bars_dir)
    target = b["date"].iloc[-1].date()
    assert scan(target, bars_dir=bars_dir).asof == target


def test_scan_can_limit_the_patterns(bars_dir):
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    assert scan(which=["doji"], bars_dir=bars_dir).hits.empty
    assert not scan(which=["hammer"], bars_dir=bars_dir).hits.empty


def test_scan_never_double_counts_a_bar(bars_dir):
    """One row per symbol: `classify` picks a single winner."""
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    hits = scan(bars_dir=bars_dir).hits
    assert hits["symbol"].is_unique


def test_series_filter_excludes_trade_to_trade(bars_dir):
    store.append(make_bars("BE1", series="BE", last=HAMMER), bars_dir=bars_dir)
    assert scan(bars_dir=bars_dir).universe_size == 0
    assert scan(rules=Universe(series=("BE",)), bars_dir=bars_dir).universe_size == 1


def test_latest_session(bars_dir):
    assert latest_session(bars_dir) is None
    b = make_bars("AAA")
    store.append(b, bars_dir=bars_dir)
    assert latest_session(bars_dir) == b["date"].iloc[-1].date()


def test_scan_output_columns(bars_dir):
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    hits = scan(bars_dir=bars_dir).hits
    assert list(hits.columns) == HIT_COLUMNS
    assert isinstance(scan(bars_dir=bars_dir).asof, date)


def test_an_empty_result_still_carries_the_full_schema(bars_dir):
    """So a caller never has to special-case "no hits" before selecting columns."""
    empty = scan(bars_dir=bars_dir).hits
    assert empty.empty
    assert list(empty.columns) == HIT_COLUMNS
    assert empty[["date", "symbol", "pattern"]].empty   # would KeyError if absent


# --- multi-session window ---------------------------------------------------


def test_scan_defaults_to_a_single_session(bars_dir):
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    r = scan(bars_dir=bars_dir)
    assert len(r.sessions) == 1
    assert r.sessions == [r.asof]


def hammer_on(symbol, offset):
    """A symbol whose only hammer is `offset` bars from the end (-1 == last)."""
    b = make_bars(symbol)
    b.loc[b.index[offset], ["open", "high", "low", "close"]] = HAMMER
    return b


def test_scan_over_three_sessions_reports_each(bars_dir):
    """One symbol hammers on each of the last three days -> three rows.

    Three hammers in a row on the *same* symbol would not work, and that is
    correct: a hammer must sit near the previous bar's low, and a hammer's own
    low is far below its body. Separate symbols is the honest test of the window.

    Scoped with `which` because this is a test of the *window*, not of the
    registry: an unscoped scan runs every registered detector, so adding a
    signal family elsewhere would otherwise break it on unrelated bars.
    """
    for sym, off in (("DAY1", -3), ("DAY2", -2), ("DAY3", -1)):
        store.append(hammer_on(sym, off), bars_dir=bars_dir)

    r = scan(sessions=3, which=["hammer"], bars_dir=bars_dir)
    assert len(r.sessions) == 3
    assert set(r.hits["pattern"]) == {"hammer"}
    assert set(r.hits["symbol"]) == {"DAY1", "DAY2", "DAY3"}
    assert sorted(r.hits["date"]) == r.sessions


def test_a_narrower_window_excludes_the_older_sessions(bars_dir):
    for sym, off in (("DAY1", -3), ("DAY2", -2), ("DAY3", -1)):
        store.append(hammer_on(sym, off), bars_dir=bars_dir)
    one = scan(sessions=1, which=["hammer"], bars_dir=bars_dir)
    two = scan(sessions=2, which=["hammer"], bars_dir=bars_dir)
    assert set(one.hits["symbol"]) == {"DAY3"}
    assert set(two.hits["symbol"]) == {"DAY2", "DAY3"}


def test_window_is_counted_in_sessions_not_calendar_days(bars_dir):
    """Sessions come from the data, so a holiday cannot shorten the window."""
    b = make_bars("AAA")
    b = b[b["date"] != b["date"].iloc[-2]]      # drop a mid-window session
    store.append(b, bars_dir=bars_dir)
    r = scan(sessions=3, bars_dir=bars_dir)
    assert len(r.sessions) == 3
    assert r.sessions == sorted(r.sessions)


def test_hits_are_newest_first(bars_dir):
    b = make_bars("AAA")
    for i in b.index[-2:]:
        b.loc[i, ["open", "high", "low", "close"]] = HAMMER
    store.append(b, bars_dir=bars_dir)
    dates = list(scan(sessions=2, bars_dir=bars_dir).hits["date"])
    assert dates == sorted(dates, reverse=True)


def test_widening_the_window_never_loses_a_hit(bars_dir):
    b = make_bars("AAA")
    for i in b.index[-3:]:
        b.loc[i, ["open", "high", "low", "close"]] = HAMMER
    store.append(b, bars_dir=bars_dir)
    one = scan(sessions=1, bars_dir=bars_dir).hits
    three = scan(sessions=3, bars_dir=bars_dir).hits
    assert set(map(tuple, one[["date", "symbol"]].to_numpy())) <= set(
        map(tuple, three[["date", "symbol"]].to_numpy())
    )


def test_asof_is_the_last_session_of_the_window(bars_dir):
    b = make_bars("AAA", last=HAMMER)
    store.append(b, bars_dir=bars_dir)
    r = scan(sessions=5, bars_dir=bars_dir)
    assert r.asof == b["date"].iloc[-1].date() == r.sessions[-1]


def test_zero_sessions_is_rejected(bars_dir):
    with pytest.raises(ValueError, match="sessions must be"):
        scan(sessions=0, bars_dir=bars_dir)


def test_summary_helpers(bars_dir):
    store.append(hammer_on("DAY1", -2), bars_dir=bars_dir)
    store.append(hammer_on("DAY2", -1), bars_dir=bars_dir)
    r = scan(sessions=2, bars_dir=bars_dir)
    assert r.by_pattern()["hammer"] == 2
    assert list(r.by_date().index) == r.sessions


def test_summary_helpers_on_an_empty_result(bars_dir):
    r = scan(bars_dir=bars_dir)
    assert r.by_pattern().empty and r.by_date().empty


# --- context columns --------------------------------------------------------


def test_hits_carry_the_context_columns(bars_dir):
    from nse_screener.context import CONTEXT_COLUMNS
    store.append(make_bars("AAA", last=HAMMER), bars_dir=bars_dir)
    hits = scan(bars_dir=bars_dir).hits
    for c in CONTEXT_COLUMNS:
        assert c in hits.columns


def test_the_gap_guard_only_sees_the_detector_window(bars_dir):
    """Regression: the read window is long because the context columns need a
    year of history, but the corporate-action guard must still only ask about
    the bars a detector can reach. Handing it the whole read rejected every
    symbol that had moved 20% at any point in fifteen months."""
    b = make_bars("AAA", n=400, last=HAMMER)
    # A split-sized move well outside the detectors' 11-bar reach.
    b.loc[b.index[:-60], ["open", "high", "low", "close"]] *= 0.5
    store.append(b, bars_dir=bars_dir)

    r = scan(bars_dir=bars_dir)
    assert r.rejected["recent gap (likely corporate action)"] == 0
    assert r.universe_size == 1


def test_a_gap_inside_the_detector_window_still_rejects(bars_dir):
    b = make_bars("AAA", n=400, last=HAMMER)
    b.loc[b.index[-4:], ["open", "high", "low", "close"]] *= 0.5
    store.append(b, bars_dir=bars_dir)
    assert scan(bars_dir=bars_dir).universe_size == 0
