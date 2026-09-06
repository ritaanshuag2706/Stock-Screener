"""The decision sheet, filled from tonight's data for the five best candidates.

The layout is the owner's: fields down the left, one column per trade. It is
laid out that way because that is the sheet people already read, and a table
that reads like the paper one needs no explaining.

Presentation only. Every number comes from `tradeplan.py` or a detector, and
every threshold from `config/patterns.yaml`. If you find yourself writing an
`if` about the market in this file, it belongs in src/.

Two things this page is careful to say out loud, because a filled-in sheet is
persuasive in a way a scan table is not:

* the ordering is by *confluence* -- how many checklist rows agree -- and no
  one has shown that confluence predicts anything;
* the reward-to-risk figure is what the trade aims at, not what trades like it
  have returned. Stage 9 measured a planned 3:1 delivering -0.17R.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from nse_screener import tradeplan as tp
from nse_screener.data import store
from nse_screener.screener import Universe, scan

# Calendar days of bars pulled per candidate. Levels need confirmed swings and a
# 60-bar lookback behind the signal, not just the signal bar.
#
# A comment rather than the attribute docstring used everywhere else in this
# project: Streamlit's "magic" renders any bare expression at the top level of a
# page, and a string literal after an assignment is one. Written as a docstring
# it appears verbatim above the title.
HISTORY_DAYS = 500


@st.cache_data(ttl=1800, show_spinner=False)
def build(sessions: int, capital: float, risk_pct: float, investment: float,
          min_rr: float, top_n: int):
    """Scan, plan and rank. Primitives in, so the cache key is stable."""
    result = scan(sessions=sessions, rules=Universe())
    if result.hits.empty:
        return result, tp._empty_plan(), pd.DataFrame()

    symbols = sorted(set(result.hits["symbol"]))
    start = (pd.Timestamp(result.asof) - pd.Timedelta(days=HISTORY_DAYS)).date()
    bars = store.read(symbols, start=start, end=result.asof, series=["EQ"])
    bars = bars.sort_values(["symbol", "date"])

    account = tp.Account(capital=capital, max_risk_pct=risk_pct / 100,
                         investment_per_trade=investment)
    rules = tp.LevelRules(min_rr=min_rr)
    plans = tp.for_hits(result.hits, bars, account=account, rules=rules)
    checks = tp.checklist(bars, result.hits)

    sheet = plans.join(checks[tp.CHECKS + ["score"]])
    # The sheet's own entry rule first, then its own logic for ordering what
    # survives. Ties break by symbol so the same night always ranks the same.
    ready = sheet[sheet["meets_rr"]].sort_values(
        ["score", "rr_min", "symbol"], ascending=[False, False, True]
    )
    return result, sheet, ready.head(top_n)


ROWS = [
    ("Date", "date", None),
    ("Scrip / Stock", "symbol", None),
    ("Close of the period", "close", "money"),
    ("Trade Decision", "direction", None),
    ("__Checklist", None, None),
    ("Candlestick Pattern", "candlestick", None),
    ("Volume", "volume", None),
    ("EMA", "ema", None),
    ("Chart Pattern", "chart_pattern", None),
    ("Fib Retracement", "fib_retrace", None),
    ("Divergence (Osc.)", "divergence", None),
    ("Checks agreeing", "score", "int"),
    ("__Levels", None, None),
    ("Immediate Support", "immediate_support", "money"),
    ("Immediate Resistance", "immediate_resistance", "money"),
    ("Major Support", "major_support", "money"),
    ("Major Resistance", "major_resistance", "money"),
    ("Significant candle median", "significant_median", "money"),
    ("__Trade", None, None),
    ("Stop Loss Price (SL)", "stop", "money"),
    ("Target Price (Min.)", "target_min", "money"),
    ("Target Price (Max.)", "target_max", "money"),
    ("Reward (Min.)", "reward_min", "money"),
    ("Reward (Max.)", "reward_max", "money"),
    ("Risk", "risk", "money"),
    ("Reward / Risk (Min.)", "rr_min", "ratio"),
    ("Reward / Risk (Max.)", "rr_max", "ratio"),
    ("__Money management", None, None),
    ("Total Capital", "_capital", "rupees"),
    ("Max. Allowed Risk / Trade", "max_risk_rupees", "rupees"),
    ("Max No. of Shares allowed", "max_shares_allowed", "int"),
    ("Shares by investment", "shares_by_investment", "int"),
    ("Shares", "shares", "int"),
    ("Binding limit", "binding", None),
    ("Trade Investment", "trade_value", "rupees"),
    ("Risk involved", "risk_involved", "rupees"),
    ("Min Profit potential", "profit_min", "rupees"),
    ("Max Profit Potential", "profit_max", "rupees"),
    ("Min ROI%", "roi_min_pct", "pct"),
    ("Max ROI%", "roi_max_pct", "pct"),
]


def _fmt(value, kind) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or pd.isna(value):
        return "—"
    if kind == "money":
        return f"{value:,.2f}"
    if kind == "rupees":
        return f"{value:,.0f}"
    if kind == "ratio":
        return f"{value:,.2f}"
    if kind == "pct":
        return f"{value:,.1f}%"
    if kind == "int":
        return f"{int(value):,}"
    return str(value)


def as_sheet(ready: pd.DataFrame, capital: float) -> pd.DataFrame:
    """The plan frame turned on its side: fields down, one column per trade."""
    cols = {}
    for _, row in ready.iterrows():
        cells = []
        for label, key, kind in ROWS:
            if label.startswith("__"):
                cells.append("")
            elif key == "_capital":
                cells.append(_fmt(capital, "rupees"))
            else:
                cells.append(_fmt(row.get(key), kind))
        cols[str(row["symbol"])] = cells
    index = [lab[2:].upper() if lab.startswith("__") else lab for lab, _, _ in ROWS]
    return pd.DataFrame(cols, index=index)


# --- page -------------------------------------------------------------------

with st.sidebar:
    st.header("Sheet")
    top_n = st.number_input("Candidates to fill", 1, 20, 5)
    sessions = st.slider("Sessions to scan", 1, 5, 1,
                         help="1 is tonight only. Widening it finds more "
                              "candidates on a quiet night.")
    st.header("Account")
    capital = st.number_input("Total capital (Rs)", 10_000, 100_000_000,
                              100_000, 10_000)
    risk_pct = st.number_input("Max risk per trade (%)", 0.1, 10.0, 2.0, 0.1)
    investment = st.number_input("Investment per trade (Rs)", 1_000,
                                 100_000_000, 20_000, 1_000)
    st.header("Entry rule")
    min_rr = st.number_input("Minimum reward / risk", 0.5, 10.0, 3.0, 0.5,
                             help="The checklist's own gate: above 3 to enter.")

st.title("Decision sheet")

result, sheet, ready = build(sessions, float(capital), float(risk_pct),
                             float(investment), float(min_rr), int(top_n))

if sheet.empty:
    st.warning("No patterns fired in this window.")
    st.stop()

planned = int(sheet["target_min"].notna().sum())
passed = int(sheet["meets_rr"].sum())
c1, c2, c3, c4 = st.columns(4)
c1.metric("Signals", f"{len(sheet):,}")
c2.metric("With a target above", f"{planned:,}")
c3.metric(f"R:R ≥ {min_rr:g}", f"{passed:,}")
c4.metric("Session", str(result.asof))

if ready.empty:
    st.warning(
        f"{len(sheet):,} signals fired and none clears reward-to-risk "
        f"{min_rr:g}. That is the checklist's own entry rule doing its job, "
        "not an error — on most nights the majority of signals have no "
        "resistance far enough above them to be worth the stop."
    )
    st.stop()

st.dataframe(as_sheet(ready, float(capital)), use_container_width=True,
             height=min(60 + 26 * len(ROWS), 1200))

st.caption(
    f"Ranked by how many of the six checklist rows agree, then by reward-to-risk. "
    f"{len(sheet) - planned:,} of {len(sheet):,} signals had no resistance above "
    "them and could not be planned at all — a breakout at new highs has nothing "
    "to aim at, and a target invented from the stop would make the R:R gate "
    "circular."
)

with st.expander("What these numbers are, and are not"):
    st.markdown(
        """
**Reward-to-risk here is a plan, not a prediction.** Stage 9 of this project
backtested a planned 3:1 exit over 2020–2026 with real Indian delivery costs
and measured an average outcome of **−0.17R**, losing 71–86% of capital. A
sheet showing 4:1 is describing where the stop and the target sit today, not
what trades like it have returned.

**The ordering is not a forecast.** Candidates are ranked by confluence — how
many of the six checklist rows agree. That is the sheet's own logic, and it is
reproducible, but nothing in this project has measured whether confluence
predicts anything. Stage 7 measured the individual components across 2.9M bars
and found every one within 0.35 percentage points of a randomly chosen bar.

**Levels are mechanical readings of the checklist's rules**, not drawn by eye.
Support is the nearer of the last confirmed swing low and the midpoint of the
heaviest-volume candle; resistance is a confirmed swing high above price. All
of them come from bars that had already printed and been confirmed, so nothing
here could only be known in hindsight.

**Sizing is reported twice on purpose.** The sheet caps shares by the risk
budget *and* by a fixed investment, and reconciles neither. Both are shown;
`Shares` is the smaller and `Binding limit` says which one is in charge.
        """
    )
