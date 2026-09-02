"""Store round-trips, partitioning and the re-run safety properties."""

from datetime import date

import pandas as pd
import pytest

from nse_screener.data import store

COLUMNS = ["date", "symbol", "series", "open", "high", "low", "close",
           "prev_close", "volume", "turnover", "trades", "isin"]


def bars(symbol, dates, close=100.0):
    return pd.DataFrame([
        {
            "date": pd.Timestamp(d), "symbol": symbol, "series": "EQ",
            "open": close, "high": close + 5, "low": close - 5, "close": close,
            "prev_close": close, "volume": 1000, "turnover": 1000 * close,
            "trades": 10, "isin": f"INE{symbol[:3]}01001",
        }
        for d in dates
    ], columns=COLUMNS)


@pytest.fixture
def bars_dir(tmp_path):
    return tmp_path / "bars"


# --- round trip -------------------------------------------------------------


def test_append_then_read(bars_dir):
    df = bars("RELIANCE", ["2024-01-01", "2024-01-02"])
    assert store.append(df, bars_dir=bars_dir) == 2
    back = store.read(bars_dir=bars_dir)
    assert len(back) == 2
    assert list(back.columns) == COLUMNS


def test_read_from_an_empty_store_returns_empty(bars_dir):
    assert store.read(bars_dir=bars_dir).empty


def test_values_survive_the_round_trip(bars_dir):
    df = bars("TCS", ["2024-03-05"], close=3456.75)
    store.append(df, bars_dir=bars_dir)
    back = store.read(["TCS"], bars_dir=bars_dir).iloc[0]
    assert back["close"] == 3456.75
    assert back["high"] == 3461.75
    assert back["date"] == pd.Timestamp("2024-03-05")


# --- partitioning -----------------------------------------------------------


def test_years_land_in_separate_partitions(bars_dir):
    store.append(bars("INFY", ["2023-12-29", "2024-01-01", "2025-06-02"]), bars_dir=bars_dir)
    assert sorted(store.partitions(bars_dir)) == [2023, 2024, 2025]
    for year in (2023, 2024, 2025):
        assert store.partition_path(year, bars_dir).is_file()


def test_appending_one_year_leaves_other_partitions_untouched(bars_dir):
    store.append(bars("INFY", ["2023-06-01"]), bars_dir=bars_dir)
    before = store.partition_path(2023, bars_dir).stat().st_mtime_ns
    store.append(bars("INFY", ["2024-06-01"]), bars_dir=bars_dir)
    assert store.partition_path(2023, bars_dir).stat().st_mtime_ns == before


# --- re-run safety ----------------------------------------------------------


def test_reappending_the_same_day_does_not_duplicate(bars_dir):
    """The backfill must be safe to re-run over a range it already covered."""
    df = bars("RELIANCE", ["2024-01-01", "2024-01-02"])
    store.append(df, bars_dir=bars_dir)
    store.append(df, bars_dir=bars_dir)
    assert len(store.read(bars_dir=bars_dir)) == 2


def test_a_redownload_overwrites_the_earlier_row(bars_dir):
    """If NSE republishes a corrected file, the newer values must win."""
    store.append(bars("RELIANCE", ["2024-01-01"], close=100.0), bars_dir=bars_dir)
    store.append(bars("RELIANCE", ["2024-01-01"], close=222.0), bars_dir=bars_dir)
    back = store.read(bars_dir=bars_dir)
    assert len(back) == 1
    assert back.iloc[0]["close"] == 222.0


def test_append_of_an_empty_frame_is_a_no_op(bars_dir):
    assert store.append(pd.DataFrame(columns=COLUMNS), bars_dir=bars_dir) == 0
    assert store.partitions(bars_dir) == {}


def test_no_temp_files_are_left_behind(bars_dir):
    store.append(bars("RELIANCE", ["2024-01-01"]), bars_dir=bars_dir)
    assert list(bars_dir.rglob("*.tmp")) == []


# --- filtering --------------------------------------------------------------


def test_symbol_filter(bars_dir):
    store.append(bars("RELIANCE", ["2024-01-01"]), bars_dir=bars_dir)
    store.append(bars("TCS", ["2024-01-01"]), bars_dir=bars_dir)
    assert set(store.read(["TCS"], bars_dir=bars_dir)["symbol"]) == {"TCS"}
    assert len(store.read(["TCS", "RELIANCE"], bars_dir=bars_dir)) == 2


def test_date_window_is_inclusive_of_both_ends(bars_dir):
    store.append(
        bars("INFY", ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]),
        bars_dir=bars_dir,
    )
    got = store.read(start="2024-01-02", end="2024-01-03", bars_dir=bars_dir)
    assert [d.date() for d in got["date"]] == [date(2024, 1, 2), date(2024, 1, 3)]


def test_date_window_spanning_years(bars_dir):
    store.append(bars("INFY", ["2023-12-31", "2024-01-01", "2025-01-01"]), bars_dir=bars_dir)
    got = store.read(start="2023-12-01", end="2024-12-31", bars_dir=bars_dir)
    assert len(got) == 2


def test_column_projection_always_returns_the_key_columns(bars_dir):
    store.append(bars("INFY", ["2024-01-01"]), bars_dir=bars_dir)
    got = store.read(columns=["close"], bars_dir=bars_dir)
    assert set(got.columns) == {"date", "symbol", "close"}


def test_series_filter(bars_dir):
    eq = bars("INFY", ["2024-01-01"])
    be = bars("SMALLCO", ["2024-01-01"])
    be["series"] = "BE"
    store.append(pd.concat([eq, be]), bars_dir=bars_dir)
    assert set(store.read(series=["EQ"], bars_dir=bars_dir)["symbol"]) == {"INFY"}


def test_results_are_sorted_by_date_then_symbol(bars_dir):
    store.append(bars("TCS", ["2024-01-02", "2024-01-01"]), bars_dir=bars_dir)
    store.append(bars("INFY", ["2024-01-02", "2024-01-01"]), bars_dir=bars_dir)
    got = store.read(bars_dir=bars_dir)
    assert list(got["symbol"]) == ["INFY", "TCS", "INFY", "TCS"]


# --- introspection ----------------------------------------------------------


def test_available_dates(bars_dir):
    store.append(bars("INFY", ["2023-12-29", "2024-01-01"]), bars_dir=bars_dir)
    assert store.available_dates(bars_dir) == {date(2023, 12, 29), date(2024, 1, 1)}


def test_available_dates_on_an_empty_store(bars_dir):
    assert store.available_dates(bars_dir) == set()


def test_symbols_are_deduplicated_across_partitions(bars_dir):
    store.append(bars("INFY", ["2023-01-02", "2024-01-01"]), bars_dir=bars_dir)
    store.append(bars("TCS", ["2024-01-01"]), bars_dir=bars_dir)
    assert store.symbols(bars_dir) == ["INFY", "TCS"]


def test_summary_reports_per_year(bars_dir):
    store.append(bars("INFY", ["2023-01-02", "2024-01-01", "2024-01-02"]), bars_dir=bars_dir)
    s = store.summary(bars_dir).set_index("year")
    assert s.loc[2023, "rows"] == 1
    assert s.loc[2024, "sessions"] == 2
