"""Day-by-day backtest. One row per trade, every metric a query on that table.

The loop is deliberately literal: walk the sessions in order, and on each one
close what needs closing before opening anything new. Nothing vectorised, no
clever reshaping -- because the whole value of this stage is that it cannot
accidentally see the future, and a plain loop is the version you can read and
believe.

The ordering rules that make it honest:

  * a signal is read on bar t's close, and the position opens at bar **t+1's
    open**. Never t's close, which was not tradeable when the signal appeared.
  * exits are checked against the bar being processed; entries are taken from
    the previous session's signals. A position therefore cannot be opened and
    closed using knowledge of the same bar's outcome.
  * when a bar's high reaches the target and its low reaches the stop, the stop
    is taken. A daily bar cannot say which came first, and assuming the good one
    inflates every result.
  * a signal exit (`exit_signal_col`) is checked *after* the stop, for the same
    reason: a colour flip cannot rescue a trade the stop already ended.
  * capital is committed at entry and released at exit, so position count and
    cash are constrained the way they would be in a real account.

The verification that matters is `--broken-rule`, which buys every Monday. If
that shows an edge, the engine has lookahead in it and every other number here
is worthless.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .. import context as ctx
from .. import ranking
from .costs import Costs

log = logging.getLogger(__name__)

TRADE_COLUMNS = [
    "symbol", "pattern", "entry_date", "entry_price", "qty", "value",
    "stop", "target", "exit_date", "exit_price", "exit_reason", "bars_held",
    "gross_pnl", "cost", "net_pnl", "return_pct", "r_multiple",
]


@dataclass(frozen=True)
class Portfolio:
    """Capital, and the limits that stop one idea consuming all of it."""

    capital: float = 1_000_000.0
    max_positions: int = 10
    """Concurrent open positions. Without a cap the backtest quietly assumes
    infinite capital and every result is meaningless."""

    max_position_pct: float = 0.10
    """Ceiling on any single position as a fraction of starting capital."""

    risk_per_trade_pct: float = 0.01
    """Fraction of capital risked between entry and stop. This, not the position
    cap, is what usually sets the size."""

    one_position_per_symbol: bool = True

    def __post_init__(self):
        if self.max_positions < 1:
            raise ValueError("max_positions must be >= 1")
        if not 0 < self.max_position_pct <= 1:
            raise ValueError("max_position_pct must be in (0, 1]")


@dataclass(frozen=True)
class ExitRules:
    """Where the trade gets out. Same shape as the Stage 7 study, so the two
    are directly comparable."""

    target_atr: float = 3.0
    stop_atr: float = 1.0
    horizon: int = 10
    atr_period: int = 14

    use_target: bool = True
    """Turn off to let a position run until the stop, the signal exit or the
    horizon. A fixed target caps the winners, which is precisely the trade-off a
    trend-following exit is meant to avoid."""


@dataclass
class Result:
    trades: pd.DataFrame
    equity: pd.Series
    portfolio: Portfolio
    exits: ExitRules
    costs: Costs
    skipped_no_slot: int = 0
    """Signals that fired while the book was full. A large number means the
    result is a property of the position cap as much as of the signal."""

    open_at_end: int = 0
    unrealised_at_end: float = 0.0
    """Positions still held on the last session. The trade table only carries
    *closed* trades, so without these the equity curve cannot be reconciled
    against it -- and a curve that does not reconcile is not evidence."""

    def reconciles(self, tol: float = 1.0) -> bool:
        """Realised P&L plus what is still open must equal the equity change."""
        realised = self.trades["net_pnl"].sum() if not self.trades.empty else 0.0
        change = self.equity.iloc[-1] - self.equity.iloc[0]
        return abs(realised + self.unrealised_at_end - change) < tol

    def __len__(self) -> int:
        return len(self.trades)

    def summary(self) -> dict:
        t = self.trades
        if t.empty:
            return {"trades": 0}
        wins = t[t["net_pnl"] > 0]
        start, end = self.equity.iloc[0], self.equity.iloc[-1]
        peak = self.equity.cummax()
        years = max((self.equity.index[-1] - self.equity.index[0]).days / 365.25, 1e-9)
        return {
            "trades": len(t),
            "win_rate": len(wins) / len(t) * 100,
            "avg_r": t["r_multiple"].mean(),
            "total_return_pct": (end / start - 1) * 100,
            "cagr_pct": ((end / start) ** (1 / years) - 1) * 100,
            "max_drawdown_pct": ((self.equity / peak - 1).min()) * 100,
            "total_costs": t["cost"].sum(),
            "gross_pnl": t["gross_pnl"].sum(),
            "net_pnl": t["net_pnl"].sum(),
            "avg_bars_held": t["bars_held"].mean(),
        }


@dataclass
class _Open:
    symbol: str
    pattern: str
    entry_date: date
    entry_price: float
    qty: int
    stop: float
    target: float
    entry_cost: float
    entry_session: int = 0
    """Index into the *market's* session list, not the symbol's own bars.

    Counting only the bars a symbol printed lets a thinly-traded name sit open
    indefinitely: a stock trading 309 times in 580 sessions reached its 60-bar
    horizon after 27 calendar months. The holding limit has to be wall-clock.
    """


def _prepare(bars: pd.DataFrame, exits: ExitRules) -> pd.DataFrame:
    """Add the ATR each stop is sized from, if it is not already there."""
    out = bars.sort_values(["symbol", "date"]).copy()
    if "atr" not in out.columns:
        out["atr"] = ctx.atr(out, out["symbol"], exits.atr_period)
    return out


def _pending_columns(fired: pd.DataFrame, signal_col: str, rank_by: str) -> list[str]:
    """Essentials plus whatever the ranker reads, without duplicates.

    Missing columns are left for `ranking.rank()` to report: it names the
    ranker and the column, which is a better error than a KeyError here.
    """
    cols = ["symbol", signal_col, "atr"]
    for c in ranking.get(rank_by).needs:
        if c in fired.columns and c not in cols:
            cols.append(c)
    return cols


def run(
    bars: pd.DataFrame,
    *,
    portfolio: Portfolio | None = None,
    exits: ExitRules | None = None,
    costs: Costs | None = None,
    patterns: list[str] | None = None,
    signal_col: str = "pattern",
    exit_signal_col: str | None = None,
    rank_by: str = "random",
    seed: int = 0,
) -> Result:
    """Walk the sessions and trade every signal the rules allow.

    `bars` needs date, symbol, OHLC and a `signal_col` holding a pattern name or
    NA. Anything from `study()` or `store.read()` + `classify_by_symbol()` works.

    `exit_signal_col` names a boolean column that closes an open position at
    that bar's close -- a Heikin-Ashi colour flip, a moving-average cross,
    anything. It is checked *after* the stop, so a bar that breaks the stop is
    still recorded as a stop; the signal cannot rescue a trade the stop already
    ended. Combine it with `use_target=False` for a pure "ride it until the
    trend turns" exit.

    `rank_by` decides which signals get the scarce slots. It matters more than
    it looks: with ~480 signals a night against 10 positions the book is full
    almost always, so this argument, not the detector, chooses most of what gets
    traded. The default `"random"` reproduces the shuffle this engine used
    before ranking existed, which keeps old results comparable and gives every
    other ranker an honest control to be measured against. See `ranking.py`.
    """
    if rank_by not in ranking.names():
        raise KeyError(
            f"unknown ranker {rank_by!r}; registered: {', '.join(ranking.names())}"
        )
    portfolio = portfolio or Portfolio()
    exits = exits or ExitRules()
    costs = costs or Costs()
    rng = np.random.default_rng(seed)

    data = _prepare(bars, exits)
    sessions = sorted(data["date"].unique())
    by_date = {d: g for d, g in data.groupby("date", sort=False)}

    cash = portfolio.capital
    open_positions: dict[str, _Open] = {}
    trades: list[dict] = []
    equity_points: list[tuple[date, float]] = []
    pending: pd.DataFrame | None = None
    last_price: dict[str, float] = {}
    skipped = 0

    max_value = portfolio.capital * portfolio.max_position_pct
    risk_budget = portfolio.capital * portfolio.risk_per_trade_pct

    for session_no, session in enumerate(sessions):
        today = by_date[session]
        prices = today.set_index("symbol")

        # --- 1. exits, against today's bar ---------------------------------
        stale: list[str] = []
        for symbol in list(open_positions):
            pos = open_positions[symbol]
            if symbol not in prices.index:
                # No bar today. Still count the session against the horizon, and
                # force the position out if it has run out of time -- otherwise a
                # suspended or delisted symbol holds forever.
                if session_no - pos.entry_session >= exits.horizon:
                    stale.append(symbol)
                continue
            bar = prices.loc[symbol]
            held = session_no - pos.entry_session

            hit_stop = bar["low"] <= pos.stop
            hit_target = exits.use_target and bar["high"] >= pos.target
            flipped = bool(
                exit_signal_col is not None and bar.get(exit_signal_col, False)
            )
            if hit_stop:                      # ties go to the stop, always
                exit_price, reason = pos.stop, "stop"
            elif hit_target:
                exit_price, reason = pos.target, "target"
            elif flipped:
                exit_price, reason = bar["close"], "signal"
            elif held >= exits.horizon:
                exit_price, reason = bar["close"], "timeout"
            else:
                continue

            fill = costs.sell_price(exit_price)
            proceeds = fill * pos.qty
            exit_cost = costs.exit_cost(proceeds)
            gross = (fill - pos.entry_price) * pos.qty
            cost = pos.entry_cost + exit_cost
            risk = (pos.entry_price - pos.stop) * pos.qty

            trades.append({
                "symbol": symbol, "pattern": pos.pattern,
                "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                "qty": pos.qty, "value": pos.entry_price * pos.qty,
                "stop": pos.stop, "target": pos.target,
                "exit_date": session, "exit_price": fill, "exit_reason": reason,
                "bars_held": held,
                "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost,
                "return_pct": (gross - cost) / (pos.entry_price * pos.qty) * 100,
                "r_multiple": (gross - cost) / risk if risk > 0 else np.nan,
            })
            cash += proceeds - exit_cost
            del open_positions[symbol]

        # A symbol that stopped printing bars is closed at its last known price.
        # Marking it out is more honest than holding a position that cannot be
        # exited and quietly counting its paper gains.
        for symbol in stale:
            pos = open_positions.pop(symbol)
            value = pos.entry_price * pos.qty
            trades.append({
                "symbol": symbol, "pattern": pos.pattern,
                "entry_date": pos.entry_date, "entry_price": pos.entry_price,
                "qty": pos.qty, "value": value, "stop": pos.stop,
                "target": pos.target, "exit_date": session,
                "exit_price": pos.entry_price, "exit_reason": "no_data",
                "bars_held": session_no - pos.entry_session,
                "gross_pnl": 0.0, "cost": pos.entry_cost,
                "net_pnl": -pos.entry_cost,
                "return_pct": -pos.entry_cost / value * 100, "r_multiple": 0.0,
            })
            cash += value

        # --- 2. entries, from yesterday's signals, at today's open ----------
        # Ranked, not shuffled. `rank_by="random"` is the same *policy* as the
        # shuffle this replaced -- uniform, no preference -- though not the same
        # draw, since it consumes the generator differently. Old absolute numbers
        # will not reproduce to the rupee; the comparison it exists for, ranker
        # against control on the same seed, is unaffected.
        if pending is not None and not pending.empty:
            candidates = ranking.rank(pending, rank_by, rng=rng)
            for _, sig in candidates.iterrows():
                symbol = sig["symbol"]
                if len(open_positions) >= portfolio.max_positions:
                    skipped += 1
                    continue
                if portfolio.one_position_per_symbol and symbol in open_positions:
                    continue
                if symbol not in prices.index:
                    continue

                entry = costs.buy_price(prices.loc[symbol, "open"])
                atr = sig["atr"]
                if not np.isfinite(entry) or not np.isfinite(atr) or atr <= 0:
                    continue

                stop = entry - exits.stop_atr * atr
                target = (
                    entry + exits.target_atr * atr if exits.use_target else np.inf
                )
                per_share_risk = entry - stop
                qty = int(min(risk_budget / per_share_risk, max_value / entry))
                value = qty * entry
                if qty < 1 or value + costs.entry_cost(value) > cash:
                    continue

                entry_cost = costs.entry_cost(value)
                cash -= value + entry_cost
                open_positions[symbol] = _Open(
                    symbol=symbol, pattern=sig[signal_col], entry_date=session,
                    entry_price=entry, qty=qty, stop=stop, target=target,
                    entry_cost=entry_cost, entry_session=session_no,
                )

        # --- 3. mark to market ---------------------------------------------
        # A position whose symbol did not print a bar today is carried at its
        # last known price, not dropped. Skipping it made equity lurch down and
        # back up as thinly-traded names came and went, and left the curve
        # irreconcilable with the trade table.
        for sym, pos in open_positions.items():
            if sym in prices.index:
                last_price[sym] = prices.loc[sym, "close"]
        held = sum(
            last_price.get(s, p.entry_price) * p.qty
            for s, p in open_positions.items()
        )
        equity_points.append((session, cash + held))

        # --- 4. today's signals become tomorrow's candidates ----------------
        fired = today[today[signal_col].notna()]
        if patterns is not None:
            fired = fired[fired[signal_col].isin(patterns)]
        # Carry whatever the ranker reads, on top of the essentials. Taken from
        # the signal bar and no later one -- this frame is built on session `t`
        # and consumed on `t+1`, which is what keeps ranking free of lookahead.
        pending = fired[_pending_columns(fired, signal_col, rank_by)].copy()
        if signal_col != "pattern":
            pending = pending.rename(columns={signal_col: "pattern"})

    frame = pd.DataFrame(trades, columns=TRADE_COLUMNS)
    equity = pd.Series(
        [v for _, v in equity_points],
        index=pd.DatetimeIndex([d for d, _ in equity_points], name="date"),
        name="equity",
    )
    # Net of the entry cost, which equity was already debited when the position
    # opened. Marking these gross double-counts that cost against the equity
    # curve: measured, it left `reconciles()` short by ~Rs 1,184 on Rs 2.13M with
    # ten positions open -- small, but a curve that does not reconcile is not
    # evidence, so the identity has to hold exactly rather than nearly.
    unrealised = sum(
        (last_price.get(s, p.entry_price) - p.entry_price) * p.qty - p.entry_cost
        for s, p in open_positions.items()
    )
    return Result(
        frame, equity, portfolio, exits, costs, skipped,
        open_at_end=len(open_positions), unrealised_at_end=unrealised,
    )
