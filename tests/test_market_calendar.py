"""Calendar tests.

Most use StaticBase so they assert the module's own logic rather than whatever
pandas_market_calendars currently believes. The handful that do exercise the
real NSE calendar are marked, because they will need revisiting when the
library's holiday rules are extended.
"""

import datetime as dt

import pandas as pd
import pytest

from nse_screener.market_calendar import (
    CalendarError,
    Overrides,
    StaticBase,
    TradingCalendar,
    period_is_complete,
    resample_ohlc,
)

# Mon-Fri of one week, then Mon-Wed of the next.
WEEK = [dt.date(2024, 4, d) for d in (8, 9, 10, 11, 12, 15, 16, 17)]


def cal(sessions=None, overrides_path=None):
    return TradingCalendar(
        overrides_path=overrides_path,
        base_calendar=StaticBase(set(sessions if sessions is not None else WEEK)),
    )


def write_overrides(tmp_path, text):
    p = tmp_path / "overrides.yaml"
    p.write_text(text)
    return p


# --- sessions and overrides -------------------------------------------------


def test_sessions_come_from_the_base_calendar():
    assert cal().sessions("2024-04-08", "2024-04-12") == WEEK[:5]


def test_closed_override_removes_a_session(tmp_path):
    p = write_overrides(tmp_path, "closed:\n  - date: 2024-04-10\n    reason: test\n")
    c = cal(overrides_path=p)
    assert dt.date(2024, 4, 10) not in c.sessions("2024-04-08", "2024-04-12")
    assert not c.is_session(dt.date(2024, 4, 10))


def test_open_override_adds_a_session(tmp_path):
    """Muhurat trading: the direction that gets forgotten."""
    p = write_overrides(tmp_path, "open:\n  - date: 2024-04-13\n    reason: muhurat\n")
    c = cal(overrides_path=p)
    assert dt.date(2024, 4, 13) in c.sessions("2024-04-08", "2024-04-14")
    assert c.is_session(dt.date(2024, 4, 13))


def test_overrides_beat_the_base_calendar_both_ways(tmp_path):
    p = write_overrides(
        tmp_path,
        "closed:\n  - date: 2024-04-10\n    reason: shut\n"
        "open:\n  - date: 2024-04-13\n    reason: special\n",
    )
    got = cal(overrides_path=p).sessions("2024-04-08", "2024-04-14")
    assert dt.date(2024, 4, 10) not in got
    assert dt.date(2024, 4, 13) in got


def test_override_listed_both_open_and_closed_raises(tmp_path):
    p = write_overrides(
        tmp_path,
        "closed:\n  - date: 2024-04-10\n    reason: a\n"
        "open:\n  - date: 2024-04-10\n    reason: b\n",
    )
    with pytest.raises(CalendarError, match="both open and closed"):
        Overrides.load(p)


def test_override_without_a_reason_raises(tmp_path):
    p = write_overrides(tmp_path, "closed:\n  - date: 2024-04-10\n")
    with pytest.raises(CalendarError, match="needs a 'reason'"):
        Overrides.load(p)


def test_missing_override_file_falls_back_to_base(tmp_path):
    c = cal(overrides_path=tmp_path / "nope.yaml")
    assert c.sessions("2024-04-08", "2024-04-12") == WEEK[:5]


def test_the_shipped_override_file_parses():
    """Every entry came from reconcile() over 2020-01-01..2026-07-29."""
    o = Overrides.load("config/holiday_overrides.yaml")
    assert dt.date(2024, 1, 22) in o.closed   # NSE published no file
    assert dt.date(2024, 11, 1) in o.open     # Diwali Muhurat session


# --- reconciliation ---------------------------------------------------------


def test_reconcile_clean():
    assert cal().reconcile(WEEK[:5], "2024-04-08", "2024-04-12") == []


def test_reconcile_flags_a_day_the_market_was_shut():
    observed = [d for d in WEEK[:5] if d != dt.date(2024, 4, 10)]
    mm = cal().reconcile(observed, "2024-04-08", "2024-04-12")
    assert [(m.date, m.kind) for m in mm] == [(dt.date(2024, 4, 10), "unexpected_closure")]


def test_reconcile_flags_a_session_not_in_the_schedule():
    mm = cal().reconcile([*WEEK[:5], dt.date(2024, 4, 13)], "2024-04-08", "2024-04-14")
    assert [(m.date, m.kind) for m in mm] == [(dt.date(2024, 4, 13), "unexpected_session")]


def test_reconcile_respects_the_window():
    mm = cal().reconcile(WEEK, "2024-04-08", "2024-04-12")
    assert mm == []  # the following week is outside the window, not a mismatch


def test_mismatches_to_yaml_round_trips_back_into_overrides(tmp_path):
    """The emitted block must actually be loadable, or it is just decoration."""
    mm = cal().reconcile([*WEEK[:4], dt.date(2024, 4, 13)], "2024-04-08", "2024-04-14")
    text = TradingCalendar.mismatches_to_yaml(mm, today=dt.date(2024, 4, 14))
    loaded = Overrides.load(write_overrides(tmp_path, text))
    assert dt.date(2024, 4, 12) in loaded.closed
    assert dt.date(2024, 4, 13) in loaded.open


def test_mismatches_to_yaml_is_empty_when_clean():
    assert TradingCalendar.mismatches_to_yaml([]) == ""


# --- period ends ------------------------------------------------------------


def test_is_period_end_on_the_last_session_of_a_week():
    assert cal().is_period_end(dt.date(2024, 4, 12), "W")
    assert not cal().is_period_end(dt.date(2024, 4, 11), "W")


def test_is_period_end_when_friday_is_a_holiday():
    """The week ends on Wednesday if Thu and Fri are shut."""
    short = [dt.date(2024, 4, d) for d in (8, 9, 10, 15, 16)]
    assert cal(short).is_period_end(dt.date(2024, 4, 10), "W")


def test_is_period_end_is_false_on_a_non_session():
    assert not cal().is_period_end(dt.date(2024, 4, 13), "W")


def test_unsupported_freq_raises():
    with pytest.raises(CalendarError, match="unsupported freq"):
        cal().is_period_end(dt.date(2024, 4, 12), "Q")


# --- period_is_complete -----------------------------------------------------


def test_period_is_complete_marks_only_the_last_period_incomplete():
    idx = pd.to_datetime(["2024-04-08", "2024-04-12", "2024-04-15", "2024-04-16"])
    got = period_is_complete(idx, "W")
    assert list(got) == [True, False]


def test_period_is_complete_on_a_single_period():
    got = period_is_complete(pd.to_datetime(["2024-04-08"]), "W")
    assert list(got) == [False]


def test_period_is_complete_on_an_empty_index():
    """An empty store is a normal state mid-backfill, not an error."""
    got = period_is_complete(pd.DatetimeIndex([]), "W")
    assert got.empty


def test_period_is_complete_needs_no_calendar_for_holidays():
    """A holiday-shortened week is still complete once the next week starts."""
    idx = pd.to_datetime(["2024-04-08", "2024-04-10", "2024-04-15"])
    assert list(period_is_complete(idx, "W")) == [True, False]


# --- resampling -------------------------------------------------------------


def frame(dates, opens, highs, lows, closes):
    idx = pd.to_datetime(dates)
    df = pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes,
         "volume": [100] * len(dates)},
        index=idx,
    )
    df.index.name = "date"
    return df


def test_resample_aggregates_ohlc_correctly():
    df = frame(["2024-04-08", "2024-04-09", "2024-04-10"],
               [10, 11, 12], [11, 15, 13], [9, 10, 7], [10.5, 11.5, 12.5])
    out = resample_ohlc(df, "W")
    row = out.iloc[0]
    assert (row["open"], row["high"], row["low"], row["close"]) == (10, 15, 7, 12.5)
    assert row["volume"] == 300


def test_last_session_reports_the_real_close_not_the_label():
    """The plan's warning: resample on the last available session, not Friday."""
    df = frame(["2024-04-08", "2024-04-09", "2024-04-10", "2024-04-15"],
               [10, 11, 12, 13], [11, 12, 13, 14], [9, 10, 11, 12], [10.5, 11.5, 12.5, 13.5])
    out = resample_ohlc(df, "W")
    assert out.index[0].date() == dt.date(2024, 4, 12)          # label: calendar Friday
    assert out["last_session"].iloc[0].date() == dt.date(2024, 4, 10)  # reality: Wednesday


def test_resample_tags_the_forming_period_incomplete():
    df = frame(["2024-04-08", "2024-04-12", "2024-04-15"],
               [10, 11, 12], [11, 12, 13], [9, 10, 11], [10.5, 11.5, 12.5])
    out = resample_ohlc(df, "W")
    assert list(out["complete"]) == [True, False]


def test_resample_monthly():
    df = frame(["2024-04-08", "2024-04-30", "2024-05-02"],
               [10, 11, 12], [11, 12, 13], [9, 10, 11], [10.5, 11.5, 12.5])
    out = resample_ohlc(df, "M")
    assert list(out["complete"]) == [True, False]
    assert out.iloc[0]["high"] == 12


def test_resample_without_a_volume_column():
    df = frame(["2024-04-08", "2024-04-09"], [10, 11], [11, 12], [9, 10], [10.5, 11.5])
    out = resample_ohlc(df.drop(columns=["volume"]), "W")
    assert "volume" not in out.columns


# --- the real NSE calendar --------------------------------------------------


def test_real_nse_calendar_knows_muharram_2024():
    """2024-07-17 was a trading holiday; our downloaded data agrees."""
    c = TradingCalendar("config/holiday_overrides.yaml")
    assert not c.is_session(dt.date(2024, 7, 17))
    assert c.is_session(dt.date(2024, 7, 16))


def test_real_nse_calendar_matches_the_sessions_we_downloaded():
    c = TradingCalendar("config/holiday_overrides.yaml")
    expected = [dt.date(2024, 7, d) for d in
                (1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 15, 16, 18, 19)]
    assert c.sessions("2024-07-01", "2024-07-19") == expected


def test_warn_if_stale_fires_when_coverage_has_run_out():
    c = cal()  # StaticBase coverage ends at the last WEEK date
    assert c.warn_if_stale(today=dt.date(2030, 1, 1)) is not None


def test_warn_if_stale_is_quiet_when_coverage_is_ample():
    c = TradingCalendar("config/holiday_overrides.yaml")
    assert c.warn_if_stale(today=dt.date(2024, 1, 1)) is None


# --- weekend sessions (Diwali Muhurat) --------------------------------------


def test_shipped_overrides_include_the_weekend_muhurat_sessions():
    """2020-11-14 was a Saturday and 2023-11-12 a Sunday; both traded.

    A weekday-only walk skips them silently, so they must be declared here for
    the backfill to know to ask.
    """
    o = Overrides.load("config/holiday_overrides.yaml")
    for d in (dt.date(2020, 11, 14), dt.date(2023, 11, 12)):
        assert d in o.open, f"{d} ({d.strftime('%a')}) missing from open overrides"
        assert d.weekday() >= 5


def test_shipped_overrides_all_carry_a_reason():
    o = Overrides.load("config/holiday_overrides.yaml")
    for entry in list(o.open.values()) + list(o.closed.values()):
        assert entry.reason and entry.added, entry
