"""The store -> detectors join.

Detectors were built against hand-typed single-symbol frames; the store returns
many symbols sorted by (date, symbol). These tests pin the boundary between
them, which is where the shift-across-symbols bug lives.
"""

import pandas as pd
import pytest

from nse_screener import patterns
from nse_screener.data import store

COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
           "prev_close", "volume", "turnover", "trades", "isin"]


def bar(d, sym, o, h, low, c):
    return {"date": pd.Timestamp(d), "symbol": sym, "series": "EQ",
            "open": o, "high": h, "low": low, "close": c, "prev_close": o,
            "volume": 1000, "turnover": 1000.0 * c, "trades": 10, "isin": "INE000A01001"}


def interleaved():
    """Two symbols in store order. Neither has a bullish engulfing alone:
    AAA drifts down in small bearish bars, BBB prints an identical bar daily."""
    rows = []
    for i, d in enumerate(pd.date_range("2024-01-01", periods=10, freq="D")):
        c = 60 - i
        rows.append(bar(d, "AAA", c + 1, c + 1.2, c - 0.2, c))
        rows.append(bar(d, "BBB", 48, 63, 47, 62))
    return pd.DataFrame(rows, columns=COLUMNS)


# --- the guard --------------------------------------------------------------


def test_detect_all_refuses_a_multi_symbol_frame():
    """Silently wrong is the worst outcome here, so fail loudly instead."""
    with pytest.raises(ValueError, match="detect_by_symbol"):
        patterns.detect_all(interleaved())


def test_detect_single_pattern_also_refuses():
    with pytest.raises(ValueError, match="detect_by_symbol"):
        patterns.detect("bullish_engulfing", interleaved())


def test_single_symbol_frame_still_works():
    one = interleaved().query("symbol == 'AAA'")
    assert len(patterns.detect_all(one)) == len(one)


def test_frame_without_a_symbol_column_still_works():
    """Hand-built frames in the other test modules have no symbol column."""
    df = interleaved().query("symbol == 'AAA'").drop(columns=["symbol"])
    assert len(patterns.detect_all(df)) == len(df)


# --- the bug this prevents --------------------------------------------------


def test_shifting_across_symbols_would_invent_patterns():
    """BBB prints the same bar every day and cannot engulf anything.

    Detected per symbol it has zero hits. This is the regression guard for the
    bug that a naive detect_all(store.read(...)) would have produced.
    """
    df = interleaved()
    hits = patterns.detect_by_symbol(df)
    assert hits["bullish_engulfing"].sum() == 0
    for sym in ("AAA", "BBB"):
        assert not hits.loc[df["symbol"] == sym, "bullish_engulfing"].any()


# --- detect_by_symbol -------------------------------------------------------


def test_detect_by_symbol_returns_the_input_index_and_order():
    df = interleaved()
    hits = patterns.detect_by_symbol(df)
    assert list(hits.index) == list(df.index)
    assert list(hits.columns) == patterns.names()
    assert all(hits[c].dtype == bool for c in hits.columns)


def test_detect_by_symbol_matches_running_each_symbol_alone():
    df = interleaved()
    combined = patterns.detect_by_symbol(df)
    for sym, g in df.groupby("symbol"):
        alone = patterns.detect_all(g.sort_values("date"))
        pd.testing.assert_frame_equal(
            combined.loc[g.index].sort_index(), alone.sort_index()
        )


def test_detect_by_symbol_sorts_unordered_input():
    """Order within a symbol must come from the date, not from row position."""
    df = interleaved()
    shuffled = df.sample(frac=1.0, random_state=0)
    a = patterns.detect_by_symbol(shuffled).sort_index()
    b = patterns.detect_by_symbol(df).sort_index()
    pd.testing.assert_frame_equal(a, b)


def test_detect_by_symbol_on_an_empty_frame():
    empty = interleaved().iloc[0:0]
    hits = patterns.detect_by_symbol(empty)
    assert hits.empty
    assert list(hits.columns) == patterns.names()


def test_detect_by_symbol_needs_a_date_column():
    df = interleaved().drop(columns=["date"])
    with pytest.raises(ValueError, match="date"):
        patterns.detect_by_symbol(df)


def test_subset_of_patterns():
    hits = patterns.detect_by_symbol(interleaved(), ["doji", "hammer"])
    assert list(hits.columns) == ["doji", "hammer"]


# --- against the real store -------------------------------------------------


def test_real_store_output_feeds_the_detectors(tmp_path):
    """End to end on the actual schema: append -> read -> detect."""
    bars_dir = tmp_path / "bars"
    store.append(interleaved(), bars_dir=bars_dir)

    df = store.read(bars_dir=bars_dir)
    hits = patterns.detect_by_symbol(df)

    assert len(hits) == len(df)
    assert hits["bullish_engulfing"].sum() == 0


def test_store_pins_price_dtypes_even_when_given_integers(tmp_path):
    """Whole-number prices must not create an int64 partition that later
    disagrees with the float64 years read() concatenates it with."""
    bars_dir = tmp_path / "bars"
    ints = interleaved()
    for col in ("open", "high", "low", "close"):
        ints[col] = ints[col].astype("int64")

    store.append(ints, bars_dir=bars_dir)
    df = store.read(bars_dir=bars_dir)
    for col in ("open", "high", "low", "close"):
        assert df[col].dtype == "float64", f"{col} is {df[col].dtype}"


# --- vectorised detection matches the per-symbol loop -----------------------


def test_vectorised_detection_equals_looping_each_symbol():
    """detect_by_symbol runs one grouped pass instead of a call per symbol.

    The loop is the obvious implementation and the slow one -- ~6ms per symbol
    regardless of bar count, because per-call pandas overhead dominates a short
    series. This pins the fast path to the same answers.
    """
    df = interleaved()
    df["date"] = pd.date_range("2024-01-01", periods=len(df), freq="D")
    vectorised = patterns.detect_by_symbol(df)
    looped = pd.concat(
        [patterns.detect_all(g.sort_values("date"))
         for _, g in df.groupby("symbol", sort=False)]
    ).reindex(df.index)
    pd.testing.assert_frame_equal(vectorised, looped)


def test_rolling_averages_do_not_leak_across_symbols():
    """The whole point of `by`: a wide-ranging symbol must not raise the
    threshold its neighbour is measured against."""
    quiet = pd.DataFrame([bar(d, "QUIET", 100, 100.5, 99.5, 100.1)
                          for d in pd.date_range("2024-01-01", periods=15)],
                         columns=COLUMNS)
    wild = pd.DataFrame([bar(d, "WILD", 100, 140, 60, 130)
                        for d in pd.date_range("2024-01-01", periods=15)],
                        columns=COLUMNS)
    both = pd.concat([quiet, wild], ignore_index=True)

    together = patterns.detect_by_symbol(both)
    alone = patterns.detect_all(quiet.sort_values("date"))
    pd.testing.assert_series_equal(
        together.loc[both["symbol"] == "QUIET", "doji"].reset_index(drop=True),
        alone["doji"].reset_index(drop=True),
    )
