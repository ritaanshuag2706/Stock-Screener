"""Stage 9. The tests are about ordering, costs and capital.

A backtest that peeks at the future prints beautiful numbers and nothing
downstream can tell. So these check the mechanics on frames where the answer is
arithmetic: what price you got, when you got out, what it cost, and that the
book cannot exceed its limits.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener.backtest.costs import Costs
from nse_screener.backtest.engine import ExitRules, Portfolio, run

COLUMNS = ["date", "symbol", "open", "high", "low", "close", "atr", "pattern"]
FREE = Costs(slippage=0.0, stt_buy=0.0, stt_sell=0.0, exchange_txn=0.0,
             sebi_charges=0.0, stamp_duty_buy=0.0)
"""A frictionless broker, so a test can assert an exact fill price."""


def bars(rows, symbol="AAA", signal_at=None, atr=1.0):
    """rows are (open, high, low, close). `signal_at` fires the pattern there."""
    out = []
    for i, (o, h, low, c) in enumerate(rows):
        out.append({
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "symbol": symbol, "open": o, "high": h, "low": low, "close": c,
            "atr": atr, "pattern": "hammer" if i == signal_at else None,
        })
    return pd.DataFrame(out, columns=COLUMNS)


def flat(n, price=100.0):
    return [(price, price + 0.5, price - 0.5, price)] * n


# --- ordering: the thing that must not be wrong -----------------------------


def test_entry_is_the_next_session_open():
    """Signal on bar 2's close; the fill is bar 3's open, not bar 2's close."""
    rows = flat(20)
    rows[3] = (150.0, 151.0, 149.0, 150.0)      # next session opens away
    r = run(bars(rows, signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    assert len(r.trades) == 1
    assert r.trades.loc[0, "entry_price"] == 150.0


def test_a_signal_on_the_last_bar_never_opens():
    """There is no next session to buy at."""
    rows = flat(20)
    r = run(bars(rows, signal_at=19), costs=FREE)
    assert r.trades.empty


def test_a_position_cannot_open_and_close_on_the_same_bar():
    rows = flat(20)
    rows[3] = (100.0, 200.0, 50.0, 100.0)       # entry bar spans everything
    r = run(bars(rows, signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    t = r.trades.iloc[0]
    assert t["exit_date"] > t["entry_date"]


# --- exits ------------------------------------------------------------------


def test_target_exit():
    rows = flat(20)
    rows[5] = (100.0, 104.0, 99.8, 103.0)       # +3 ATR from 100 is 103
    r = run(bars(rows, signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "target"
    assert t["exit_price"] == pytest.approx(103.0)


def test_stop_exit():
    rows = flat(20)
    rows[5] = (100.0, 100.5, 98.0, 99.0)        # -1 ATR from 100 is 99
    r = run(bars(rows, signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "stop"
    assert t["exit_price"] == pytest.approx(99.0)


def test_a_bar_hitting_both_is_recorded_as_a_stop():
    """A daily bar cannot say which came first. Assuming the good one inflates
    every number the backtest produces."""
    rows = flat(20)
    rows[5] = (100.0, 105.0, 97.0, 100.0)
    r = run(bars(rows, signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    assert r.trades.iloc[0]["exit_reason"] == "stop"


def test_timeout_exit_at_the_close():
    r = run(bars(flat(30), signal_at=2), costs=FREE,
            exits=ExitRules(target_atr=99, stop_atr=99, horizon=5))
    t = r.trades.iloc[0]
    assert t["exit_reason"] == "timeout"
    assert t["bars_held"] == 5


# --- costs ------------------------------------------------------------------


def test_costs_are_charged_on_both_legs():
    r = run(bars(flat(30), signal_at=2),
            exits=ExitRules(target_atr=99, stop_atr=99, horizon=5))
    t = r.trades.iloc[0]
    assert t["cost"] > 0
    assert t["net_pnl"] == pytest.approx(t["gross_pnl"] - t["cost"])


def test_a_flat_market_loses_exactly_the_costs():
    """Nothing moves, so every rupee lost is friction."""
    r = run(bars(flat(30), signal_at=2),
            exits=ExitRules(target_atr=99, stop_atr=99, horizon=5))
    t = r.trades.iloc[0]
    assert t["net_pnl"] < 0
    assert t["gross_pnl"] < 0          # slippage alone, before charges


def test_round_trip_cost_is_in_the_documented_range():
    assert 0.25 <= Costs().round_trip_pct(100_000) <= 0.5


def test_slippage_moves_the_fill_against_you():
    c = Costs(slippage=0.01)
    assert c.buy_price(100) == pytest.approx(101)
    assert c.sell_price(100) == pytest.approx(99)


# --- portfolio limits -------------------------------------------------------


def many_signals(n_symbols=30, n_bars=30):
    frames = [bars(flat(n_bars), symbol=f"S{i:02d}", signal_at=2)
              for i in range(n_symbols)]
    return pd.concat(frames, ignore_index=True)


# --- ranking decides who gets the scarce slots ------------------------------


def _ranked_signals(n_symbols=30, n_bars=30):
    """Signals that differ only in `rel_volume`, so ranking is the only thing
    that can distinguish them and the test cannot pass by accident."""
    frame = many_signals(n_symbols=n_symbols, n_bars=n_bars)
    order = {f"S{i:02d}": float(i) for i in range(n_symbols)}
    frame["rel_volume"] = frame["symbol"].map(order)
    return frame


def test_ranking_decides_which_signals_get_the_slots():
    """The whole reason this module exists. With 30 signals and 3 slots, the
    engine took an arbitrary 3; now it takes the 3 the ranker put first."""
    r = run(_ranked_signals(), portfolio=Portfolio(max_positions=3),
            exits=ExitRules(target_atr=99, stop_atr=99, horizon=8),
            rank_by="rel_volume")
    first_day = r.trades[r.trades["entry_date"] == r.trades["entry_date"].min()]
    assert set(first_day["symbol"]) == {"S29", "S28", "S27"}
    assert r.skipped_no_slot > 0


def test_a_different_ranker_fills_the_book_differently():
    """If two rankers produced the same book, nothing here would be measurable."""
    kw = {"portfolio": Portfolio(max_positions=3),
          "exits": ExitRules(target_atr=99, stop_atr=99, horizon=8)}
    ranked = set(run(_ranked_signals(), rank_by="rel_volume", **kw).trades["symbol"])
    random_ = set(run(_ranked_signals(), rank_by="random", **kw).trades["symbol"])
    assert ranked != random_


def test_the_same_ranker_and_seed_reproduce_the_same_book():
    kw = {"portfolio": Portfolio(max_positions=3),
          "exits": ExitRules(target_atr=99, stop_atr=99, horizon=8),
          "rank_by": "random", "seed": 11}
    a = run(_ranked_signals(), **kw).trades
    b = run(_ranked_signals(), **kw).trades
    pd.testing.assert_frame_equal(a, b)


def test_an_unknown_ranker_fails_before_any_trading_happens():
    with pytest.raises(KeyError, match="no_such_ranker"):
        run(many_signals(), rank_by="no_such_ranker")


def test_ranking_cannot_see_the_bar_it_is_filled_on():
    """The no-lookahead check, at engine level rather than unit level.

    Candidates are scored on session `t` and filled at `t+1` open. Changing a
    bar's *own* rank column after the signal fired must not reorder anything,
    because the ranker read that column a session earlier. If this fails, the
    ranker is scoring the bar it trades on and every ranked result is void.
    """
    kw = {"portfolio": Portfolio(max_positions=3),
          "exits": ExitRules(target_atr=99, stop_atr=99, horizon=8),
          "rank_by": "rel_volume"}
    base = _ranked_signals()
    baseline = run(base, **kw).trades

    sig_date = base.loc[base["pattern"].notna(), "date"].min()

    # Invert rel_volume on every bar *after* the signal bar (810 rows). The
    # signal bar's own value -- the one ranking reads -- is untouched.
    tampered = base.copy()
    after = tampered["date"] > sig_date
    tampered.loc[after, "rel_volume"] = 100.0 - tampered.loc[after, "rel_volume"]
    pd.testing.assert_frame_equal(run(tampered, **kw).trades, baseline)

    # The other half, without which the above is vacuous: the same inversion
    # applied to the signal bar *must* change the book. Measured, it flips the
    # first day from the top three names to the bottom three.
    on_signal = base.copy()
    on = on_signal["date"] == sig_date
    on_signal.loc[on, "rel_volume"] = 100.0 - on_signal.loc[on, "rel_volume"]
    changed = run(on_signal, **kw).trades

    def first_day(t):
        return set(t[t["entry_date"] == t["entry_date"].min()]["symbol"])

    assert first_day(baseline) == {"S29", "S28", "S27"}
    assert first_day(changed) == {"S00", "S01", "S02"}


def test_never_exceeds_max_positions():
    r = run(many_signals(), portfolio=Portfolio(max_positions=5),
            exits=ExitRules(target_atr=99, stop_atr=99, horizon=8))
    per_day = r.trades.groupby("entry_date").size()
    assert per_day.max() <= 5
    assert r.skipped_no_slot > 0


def test_position_size_respects_the_risk_budget():
    """1% of Rs 10L is Rs 10,000 risked; a 1.0 ATR stop means 10,000 shares --
    but the 10% position cap binds first at a price of 100."""
    r = run(bars(flat(30), signal_at=2), costs=FREE,
            portfolio=Portfolio(capital=1_000_000, max_position_pct=0.10,
                                risk_per_trade_pct=0.01),
            exits=ExitRules(target_atr=99, stop_atr=1, horizon=5))
    t = r.trades.iloc[0]
    assert t["value"] <= 1_000_000 * 0.10 + 1


def test_one_position_per_symbol():
    rows = flat(30)
    frame = bars(rows, signal_at=2)
    frame.loc[3, "pattern"] = "hammer"          # a second signal while open
    r = run(frame, costs=FREE, exits=ExitRules(target_atr=99, stop_atr=99, horizon=10))
    assert len(r.trades) == 1


def test_invalid_portfolio_raises():
    with pytest.raises(ValueError, match="max_positions"):
        Portfolio(max_positions=0)
    with pytest.raises(ValueError, match="max_position_pct"):
        Portfolio(max_position_pct=1.5)


# --- the verification the plan demands --------------------------------------


def test_a_rule_with_no_edge_does_not_show_one():
    """Buy every Monday on a random walk. If this prints an edge, the engine has
    lookahead in it and every other number is worthless."""
    rng = np.random.default_rng(0)
    n = 600
    frames = []
    for s in range(12):
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
        f = pd.DataFrame({
            "date": pd.bdate_range("2022-01-03", periods=n),
            "symbol": f"S{s:02d}", "open": close,
            "high": close * 1.01, "low": close * 0.99, "close": close,
            "atr": close * 0.015,
        })
        f["pattern"] = np.where(f["date"].dt.dayofweek == 0, "monday", None)
        frames.append(f)
    r = run(pd.concat(frames, ignore_index=True),
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    assert len(r.trades) > 100
    assert r.summary()["avg_r"] < 0.1       # no edge, and costs make it negative


# --- the trade table --------------------------------------------------------


def test_every_trade_is_one_row_with_a_reason():
    r = run(many_signals(), exits=ExitRules(target_atr=99, stop_atr=99, horizon=5))
    assert set(r.trades["exit_reason"]) <= {"target", "stop", "timeout"}
    assert r.trades["entry_date"].notna().all()
    assert (r.trades["exit_date"] > r.trades["entry_date"]).all()


def test_equity_curve_tracks_the_sessions():
    """One point per session, starting at the opening capital, and never
    negative -- cash plus marked-to-market holdings."""
    r = run(bars(flat(30), signal_at=2), portfolio=Portfolio(capital=500_000))
    assert len(r.equity) == 30
    assert r.equity.iloc[0] == pytest.approx(500_000)
    assert (r.equity > 0).all()
    assert r.equity.index.is_monotonic_increasing


# --- the books must balance -------------------------------------------------


def test_equity_curve_reconciles_with_the_trade_table():
    """Realised P&L plus what is still open must equal the equity change.

    A curve that does not reconcile is not evidence. This caught a Rs 378,733
    gap: positions in symbols that did not trade on a given day were dropped
    from the mark-to-market entirely, so equity lurched as thinly-traded names
    came and went.
    """
    r = run(many_signals(n_symbols=20, n_bars=60),
            exits=ExitRules(target_atr=3, stop_atr=1, horizon=10))
    assert r.reconciles(), (
        f"realised {r.trades['net_pnl'].sum():.2f} + unrealised "
        f"{r.unrealised_at_end:.2f} != equity change "
        f"{r.equity.iloc[-1] - r.equity.iloc[0]:.2f}"
    )


def test_positions_open_at_the_end_are_marked_net_of_their_entry_cost():
    """The identity has to hold *exactly*, not nearly.

    Equity is debited the entry cost the moment a position opens. Marking the
    still-open ones gross of that cost double-counts it, which left the books
    Rs 1,184 short on Rs 2.13M with ten positions open -- small enough to hide
    behind a loose tolerance, and wrong. A signal fires two bars from the end
    against a horizon of 50, so positions are guaranteed to be open at the end.
    """
    frame = pd.concat(
        [bars(flat(40), symbol=f"S{i}", signal_at=38) for i in range(4)],
        ignore_index=True,
    )
    r = run(frame, exits=ExitRules(target_atr=99, stop_atr=99, horizon=50))

    assert r.open_at_end > 0, "test is vacuous unless something is still open"
    realised = r.trades["net_pnl"].sum() if not r.trades.empty else 0.0
    change = r.equity.iloc[-1] - r.equity.iloc[0]
    assert realised + r.unrealised_at_end == pytest.approx(change, abs=1e-6)


def test_a_position_in_a_symbol_that_stops_trading_is_forced_out():
    """Counting only the bars a symbol printed let a thinly-traded name sit open
    indefinitely -- 27 months against a 60-bar horizon. The limit is wall-clock."""
    live = bars(flat(60), symbol="LIVE")
    dying = bars(flat(60), symbol="DIES", signal_at=2).iloc[:8]   # stops on day 8
    frame = pd.concat([live, dying], ignore_index=True)
    r = run(frame, exits=ExitRules(target_atr=99, stop_atr=99, horizon=10))
    closed = r.trades[r.trades["symbol"] == "DIES"]
    assert len(closed) == 1
    assert closed.iloc[0]["exit_reason"] == "no_data"
    assert r.open_at_end == 0


def test_holding_period_is_wall_clock_not_symbol_bars():
    """A symbol that trades only occasionally must still time out on schedule.

    It has to print bars on the signal day and the next one, or the entry never
    fills -- you cannot buy a stock that did not trade. After that it goes
    sparse, which is where the old bar-counting horizon ran away.
    """
    dense = bars(flat(60), symbol="DENSE")
    sparse = bars(flat(60), symbol="SPARSE", signal_at=2)
    keep = list(range(5)) + list(range(5, 60, 4))    # entry, then every 4th
    frame = pd.concat([dense, sparse.iloc[keep]], ignore_index=True)
    r = run(frame, exits=ExitRules(target_atr=99, stop_atr=99, horizon=10))
    t = r.trades[r.trades["symbol"] == "SPARSE"]
    assert len(t) == 1
    assert t.iloc[0]["bars_held"] <= 11        # sessions elapsed, not bars printed
