"""Parser and URL tests. No network — fixtures are built in-memory.

Headers here are copied verbatim from real downloads, so a silent NSE column
rename shows up as a test failure the next time fixtures are refreshed.
"""

import zipfile
from datetime import date

import pandas as pd
import pytest

from nse_screener.data import bhavcopy as bc

LEGACY_HEADER = (
    "SYMBOL,SERIES,OPEN,HIGH,LOW,CLOSE,LAST,PREVCLOSE,TOTTRDQTY,"
    "TOTTRDVAL,TIMESTAMP,TOTALTRADES,ISIN,"
)
LEGACY_ROWS = [
    ("RELIANCE,EQ,2550,2575.5,2540,2568.25,2567,2545,5000000,12800000000,"
     "02-JAN-2023,150000,INE002A01018,"),
    ("TCS,BE,3200,3250,3190,3240,3238,3195,900000,2900000000,"
     "02-JAN-2023,45000,INE467B01029,"),
    # A government security: right file, wrong instrument. Must be dropped.
    ("1018GS2026,GS,118,118,118,118,118,116,648,76464,"
     "02-JAN-2023,4,IN0020010081,"),
]

UDIFF_HEADER = (
    "TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,"
    "FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,"
    "LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,"
    "TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4"
)
UDIFF_ROWS = [
    ("2024-07-08,2024-07-08,CM,NSE,STK,2885,INE002A01018,RELIANCE,EQ,,,,,RELIANCE LTD,"
     "3100.00,3150.00,3080.00,3140.00,3138.00,3095.00,,3140.50,,,4000000,12500000000,"
     "120000,F1,1,,,,,"),
    ("2024-07-08,2024-07-08,CM,NSE,STK,11536,INE467B01029,TCS,EQ,,,,,TATA CONS SERV,"
     "3900.00,3950.00,3880.00,3930.00,3928.00,3895.00,,3930.50,,,800000,3140000000,"
     "60000,F1,1,,,,,"),
    # A derivative row: same file, must be excluded by the STK filter.
    ("2024-07-08,2024-07-08,FO,NSE,IDF,54321,,NIFTY,FUT,2024-07-25,2024-07-25,0,,"
     "NIFTY24JUL,24000,24100,23900,24050,24040,23950,,24050,100,5,900,21000000,"
     "5000,F1,50,,,,,"),
]


def make_zip(tmp_path, name, header, rows):
    path = tmp_path / f"{name}.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(f"{name}.csv", "\n".join([header, *rows]) + "\n")
    return path


# --- URL construction -------------------------------------------------------


def test_layout_boundary_matches_the_live_archive():
    """Probed: legacy serves through 2024-07-05 and 404s from 2024-07-08."""
    assert bc.layout_for(date(2024, 7, 5)) == "legacy"
    assert bc.layout_for(date(2024, 7, 7)) == "legacy"
    assert bc.layout_for(date(2024, 7, 8)) == "udiff"
    assert bc.layout_for(date(2026, 7, 24)) == "udiff"


def test_legacy_url_shape():
    url = bc.url_for(date(2023, 1, 2))
    assert url.endswith("/EQUITIES/2023/JAN/cm02JAN2023bhav.csv.zip")


def test_legacy_url_zero_pads_the_day():
    assert "cm05JUL2024bhav" in bc.url_for(date(2024, 7, 5))


def test_udiff_url_shape():
    url = bc.url_for(date(2024, 7, 8))
    assert url.endswith("/BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip")


def test_raw_path_keeps_nse_filename(tmp_path):
    p = bc.raw_path(date(2023, 1, 2), tmp_path)
    assert p.name == "cm02JAN2023bhav.csv.zip"
    assert p.parent == tmp_path


# --- legacy parser ----------------------------------------------------------


def test_legacy_parse_produces_canonical_schema(tmp_path):
    df = bc.parse_legacy(make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS))
    assert list(df.columns) == bc.COLUMNS
    assert len(df) == 2  # the GS row is filtered out
    assert set(df["symbol"]) == {"RELIANCE", "TCS"}


def test_legacy_parse_values(tmp_path):
    df = bc.parse_legacy(make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS))
    r = df.set_index("symbol").loc["RELIANCE"]
    assert r["date"] == pd.Timestamp("2023-01-02")
    assert (r["open"], r["high"], r["low"], r["close"]) == (2550.0, 2575.5, 2540.0, 2568.25)
    assert r["volume"] == 5_000_000
    assert r["series"] == "EQ"
    assert r["isin"] == "INE002A01018"


def test_legacy_trailing_comma_makes_no_phantom_column(tmp_path):
    df = bc.parse_legacy(make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS))
    assert not any(c.startswith("Unnamed") for c in df.columns)


def test_series_filter_is_configurable(tmp_path):
    path = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS)
    assert set(bc.parse_legacy(path, series=("EQ",))["symbol"]) == {"RELIANCE"}
    assert set(bc.parse_legacy(path, series=("GS",))["symbol"]) == {"1018GS2026"}


# --- UDiFF parser -----------------------------------------------------------


def test_udiff_parse_produces_canonical_schema(tmp_path):
    name = "BhavCopy_NSE_CM_0_0_0_20240708_F_0000"
    df = bc.parse_udiff(make_zip(tmp_path, name, UDIFF_HEADER, UDIFF_ROWS))
    assert list(df.columns) == bc.COLUMNS
    assert len(df) == 2
    assert set(df["symbol"]) == {"RELIANCE", "TCS"}


def test_udiff_excludes_derivative_rows(tmp_path):
    """The UDiFF file carries futures and options; only STK rows are equities."""
    name = "BhavCopy_NSE_CM_0_0_0_20240708_F_0000"
    df = bc.parse_udiff(make_zip(tmp_path, name, UDIFF_HEADER, UDIFF_ROWS))
    assert "NIFTY" not in set(df["symbol"])


def test_udiff_parse_values(tmp_path):
    name = "BhavCopy_NSE_CM_0_0_0_20240708_F_0000"
    df = bc.parse_udiff(make_zip(tmp_path, name, UDIFF_HEADER, UDIFF_ROWS))
    r = df.set_index("symbol").loc["RELIANCE"]
    assert r["date"] == pd.Timestamp("2024-07-08")
    assert (r["open"], r["high"], r["low"], r["close"]) == (3100.0, 3150.0, 3080.0, 3140.0)
    assert r["volume"] == 4_000_000
    assert r["trades"] == 120_000


# --- both layouts agree -----------------------------------------------------


def test_both_layouts_yield_identical_schema_and_dtypes(tmp_path):
    a = bc.parse_legacy(make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS))
    b = bc.parse_udiff(
        make_zip(tmp_path, "BhavCopy_NSE_CM_0_0_0_20240708_F_0000", UDIFF_HEADER, UDIFF_ROWS)
    )
    assert list(a.columns) == list(b.columns)
    assert a.dtypes.to_dict() == b.dtypes.to_dict()


def test_parse_dispatches_on_date(tmp_path):
    legacy = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS)
    df = bc.parse(legacy, date(2023, 1, 2))
    assert len(df) == 2


# --- date format anomalies --------------------------------------------------


def test_legacy_accepts_a_two_digit_year(tmp_path):
    """2020-07-13 was published as '13-Jul-20'. Exactly one file in 1,116."""
    rows = [r.replace("02-JAN-2023", "13-Jul-20") for r in LEGACY_ROWS]
    df = bc.parse_legacy(make_zip(tmp_path, "cm13JUL2020bhav", LEGACY_HEADER, rows))
    assert set(df["date"]) == {pd.Timestamp("2020-07-13")}


def test_unrecognised_date_format_still_raises(tmp_path):
    """Tolerating one anomaly must not become 'guess at anything'."""
    rows = [r.replace("02-JAN-2023", "2023/01/02") for r in LEGACY_ROWS]
    path = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, rows)
    with pytest.raises(ValueError, match="unrecognised TIMESTAMP format"):
        bc.parse_legacy(path)


def test_parse_rejects_a_file_whose_dates_are_not_the_requested_day(tmp_path):
    """A bhavcopy holds one trading day. Anything else means a misread."""
    legacy = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS)
    with pytest.raises(ValueError, match="was requested for 2023-01-03"):
        bc.parse(legacy, date(2023, 1, 3))


def test_parse_accepts_the_matching_day(tmp_path):
    legacy = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, LEGACY_ROWS)
    assert len(bc.parse(legacy, date(2023, 1, 2))) == 2


# --- validation -------------------------------------------------------------


def test_impossible_high_low_raises(tmp_path):
    """A misread layout shows up as high < low. Never let that reach the store."""
    broken = ["BADCO,EQ,100,90,95,99,99,98,1000,100000,02-JAN-2023,10,INE000A01001,"]
    path = make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, broken)
    with pytest.raises(ValueError, match="violate high/low bounds"):
        bc.parse_legacy(path)


def test_rows_with_no_prices_are_dropped(tmp_path):
    blank = ["NOPRICE,EQ,,,,,,,0,0,02-JAN-2023,0,INE000A01002,"]
    df = bc.parse_legacy(make_zip(tmp_path, "cm02JAN2023bhav", LEGACY_HEADER, blank))
    assert df.empty


def test_zip_without_exactly_one_csv_raises(tmp_path):
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "not a csv")
    with pytest.raises(ValueError, match="exactly one CSV"):
        bc.parse_legacy(path)


# --- date walking -----------------------------------------------------------


def test_trading_weekdays_excludes_the_weekend():
    days = bc.trading_weekdays(date(2024, 7, 1), date(2024, 7, 14))
    assert date(2024, 7, 6) not in days   # Saturday
    assert date(2024, 7, 7) not in days   # Sunday
    assert len(days) == 10
    assert all(d.weekday() < 5 for d in days)


def test_trading_weekdays_is_inclusive_of_both_ends():
    days = bc.trading_weekdays(date(2024, 7, 8), date(2024, 7, 8))
    assert days == [date(2024, 7, 8)]


# --- weekend session discovery ----------------------------------------------


def test_weekday_walk_alone_misses_weekend_sessions():
    """Documents why backfill unions the calendar's open overrides on top."""
    days = bc.trading_weekdays(date(2020, 11, 9), date(2020, 11, 15))
    assert date(2020, 11, 14) not in days  # Saturday Muhurat session
