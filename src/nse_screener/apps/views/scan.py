"""Every pattern across the whole eligible universe, over the last few sessions.

A page of the app. Presentation only -- eligibility rules and detection live in
`screener.py` and `patterns/`, behind unit tests.
"""

from __future__ import annotations

import time

import pandas as pd
import streamlit as st

from nse_screener import context as ctx
from nse_screener import patterns as pat
from nse_screener.apps.views import _style
from nse_screener.screener import Universe, latest_session, scan


@st.cache_data(ttl=1800, show_spinner=False)
def run_scan(sessions: int, min_history: int, max_gap_pct: float,
             which: tuple[str, ...] | None):
    """Cached wrapper. Takes primitives so the cache key is stable.

    Returns (result, seconds) so the page can distinguish a fresh scan from a
    cached one -- on a cache hit the stored elapsed time comes back with it and
    the wall clock shows ~0.
    """
    started = time.perf_counter()
    rules = Universe(
        min_history=min_history,
        max_overnight_gap=max_gap_pct / 100,
    )
    out = scan(sessions=sessions, rules=rules, which=list(which) if which else None)
    return out, time.perf_counter() - started


with st.sidebar:
    st.header("Window")
    sessions = st.slider("Trading sessions", 1, 10, 3,
                         help="How far back to report, counted in sessions "
                              "rather than calendar days, so holidays cannot "
                              "shorten the window.")

    st.header("Universe")
    min_history = st.number_input("Min sessions of history", 1, 2000, 250, 50)
    max_gap = st.slider("Reject overnight moves above (%)", 5, 50, 20,
                        help="A corporate-action guard. An unadjusted split "
                             "halves the price overnight, and bars either side "
                             "of it are not comparable.")

    st.header("Patterns")
    chosen = st.multiselect(
        "Include", pat.names(), default=pat.names(), format_func=pat.label
    )

    st.header("Columns")
    # Seventeen context columns at once is an unreadable table. Grouping them
    # keeps the default view legible while every column stays one click away --
    # and the CSV always carries all of them regardless.
    groups = st.multiselect(
        "Context", list(ctx.GROUPS), default=["Trend", "Momentum"],
    )

st.title("Pattern scan")

if latest_session() is None:
    st.error("The store is empty. Run the backfill first.")
    st.stop()

if not chosen:
    st.warning("Select at least one pattern.")
    st.stop()

# show_time puts a live counter in the spinner, so a slow scan looks like work
# in progress rather than a hung page. At ~0.6s it barely appears, which is the
# point -- the fix for a slow page was making it fast, not animating the wait.
wall = time.perf_counter()
with st.spinner("Scanning the universe...", show_time=True):
    result, computed_in = run_scan(sessions, min_history, max_gap, tuple(chosen))
wall = time.perf_counter() - wall
from_cache = wall < computed_in / 2

if result.universe_size == 0:
    st.warning("No symbols passed the eligibility rules. Try loosening them.")
    st.stop()

span = (f"{result.sessions[0]} to {result.sessions[-1]}"
        if len(result.sessions) > 1 else str(result.asof))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Sessions", len(result.sessions), help=span)
c2.metric("Eligible symbols", f"{result.universe_size:,}")
c3.metric("Hits", f"{len(result):,}")
c4.metric("Hits per session", f"{len(result) / max(len(result.sessions), 1):.0f}")
st.caption(
    f"Covering {span} · "
    + ("served from cache" if from_cache else f"scanned in {computed_in:.1f}s")
)

with st.expander("Why symbols were excluded"):
    st.dataframe(
        pd.DataFrame(
            [{"reason": k, "symbols": v} for k, v in result.rejected.items()]
        ),
        width="stretch", hide_index=True,
    )
    st.caption(
        "One bar can satisfy several patterns. Each is listed once, under the "
        "most specific one that fired — see classify() in patterns/registry.py."
    )

if result.hits.empty:
    st.info("Nothing printed a pattern in this window.")
    st.stop()

left, right = st.columns(2)
with left:
    st.subheader("By pattern")
    per_pattern = result.by_pattern()
    st.plotly_chart(
        _style.hbar([pat.label(n) for n in per_pattern.index],
                    per_pattern.to_list()),
        width="stretch", config=_style.CHART_CONFIG, key="by_pattern",
    )
with right:
    st.subheader("By session")
    # Categorical labels, not dates. On a continuous time axis three bars become
    # three hairlines with hourly ticks between them.
    per_day = result.by_date()
    st.plotly_chart(
        _style.vbar([d.strftime("%d %b") for d in per_day.index],
                    per_day.to_list()),
        width="stretch", config=_style.CHART_CONFIG, key="by_session",
    )

st.subheader(f"{len(result):,} hits")

table = result.hits.copy()
# Display form only. The underlying column keeps the snake_case identifier, so
# the CSV and any downstream code still join on a stable key.
table["pattern"] = table["pattern"].map(pat.labels()).fillna(table["pattern"])
shown_context = [c for grp in groups for c in ctx.GROUPS[grp]]
base = [c for c in table.columns if c not in ctx.CONTEXT_COLUMNS]
table = table[base + shown_context]

st.dataframe(
    table,
    width="stretch", hide_index=True, height=520,
    column_config={
        "date": st.column_config.DateColumn("Date", format="DD MMM YYYY"),
        "symbol": st.column_config.TextColumn("Symbol"),
        "pattern": st.column_config.TextColumn("Pattern"),
        "close": st.column_config.NumberColumn("Close", format="%.2f"),
        "chg_pct": st.column_config.NumberColumn("Chg %", format="%+.2f%%"),
        "volume": st.column_config.NumberColumn("Volume", format="localized"),
        "bars": st.column_config.NumberColumn("History", format="localized"),
        # Stage 6 context. Columns, not filters -- sort by them, look at them
        # for a fortnight, and let what looks promising become Stage 7's
        # hypotheses. Do not add a filter here before Stage 7 measures it.
        "above_200ema": st.column_config.CheckboxColumn(
            "Uptrend", help="Close above the 200-day EMA."),
        "dist_200ema_pct": st.column_config.NumberColumn(
            "vs 200EMA", format="%+.1f%%", help="Distance from the 200-day EMA."),
        "rel_volume": st.column_config.NumberColumn(
            "Rel vol", format="%.2fx",
            help="This bar's volume against the symbol's 20-day average."),
        "atr_pct": st.column_config.NumberColumn(
            "ATR", format="%.2f%%", help="ATR(14) as a percentage of close."),
        "dist_25ema_pct": st.column_config.NumberColumn("vs 25EMA", format="%+.1f%%"),
        "dist_13ema_pct": st.column_config.NumberColumn("vs 13EMA", format="%+.1f%%"),
        "dist_5ema_pct": st.column_config.NumberColumn("vs 5EMA", format="%+.1f%%"),
        "ema_stack": st.column_config.TextColumn(
            "Ribbon", help="5/13/25 EMA ordering: up, down, or mixed."),
        "rsi_14": st.column_config.ProgressColumn(
            "RSI", format="%.0f", min_value=0, max_value=100),
        "rsi_zone": st.column_config.TextColumn(
            "RSI zone", help="Below 30 oversold, above 70 overbought. Sortable, "
                             "which the bar is not."),
        "stoch_k": st.column_config.NumberColumn("Stoch %K", format="%.1f"),
        "stoch_d": st.column_config.NumberColumn("Stoch %D", format="%.1f"),
        "macd": st.column_config.NumberColumn("MACD", format="%.2f"),
        "macd_signal": st.column_config.NumberColumn("Signal", format="%.2f"),
        "macd_hist_pct": st.column_config.NumberColumn(
            "Hist", format="%+.2f%%",
            help="MACD histogram as a percentage of price, so it sorts across "
                 "symbols. The raw histogram does not."),
        "bb_pct_b": st.column_config.NumberColumn(
            "BB %B", format="%.2f",
            help="0 at the lower band, 1 at the upper. Outside on a breakout."),
        "bb_width_pct": st.column_config.NumberColumn(
            "BB width", format="%.1f%%", help="Band width. Low means a squeeze."),
    },
)
st.caption("Click any column header to sort. Unfiltered by design — Stage 8 adds "
           "filters, once Stage 7 has measured which ones earn their place.")

st.download_button(
    "Download CSV",
    result.hits.to_csv(index=False).encode(),
    file_name=f"patterns_{result.asof}_{len(result.sessions)}d.csv",
    mime="text/csv",
)
