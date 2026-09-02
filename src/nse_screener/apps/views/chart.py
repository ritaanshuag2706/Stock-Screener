"""Candlestick chart for one symbol.

A page of the app; run the app with

    .venv/bin/streamlit run src/nse_screener/apps/dashboard.py

Presentation only. Every decision -- what counts as a pattern, which bars are
eligible, what a threshold is -- lives in src/ behind a unit test. If you find
yourself writing an `if` about the market in this file, it belongs elsewhere.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from nse_screener import patterns
from nse_screener.apps.views import _style
from nse_screener.clock import ist_today
from nse_screener.data import store
from nse_screener.market_calendar import resample_ohlc

# Colour carries direction; shape carries which pattern. Four categorical hues
# cannot be used here: marker colours on one chart are compared in any pair, and
# every candidate four-hue set fails the all-pairs colourblind gate on this
# surface. See _style.py for the measured numbers.
UP = _style.CANDLE_UP
DOWN = _style.CANDLE_DOWN
GRID = _style.GRID

DIRECTION_COLOUR = {
    "bullish": _style.BULLISH,
    "bearish": _style.CANDLE_DOWN,
    "neutral": _style.NEUTRAL,
}
# Shape is the identity channel, so it must stay distinct per pattern.
# These must be distinguishable at 10px, because with three bullish patterns
# sharing one hue, shape is the only thing telling them apart. triangle-up and
# star-triangle-up were too alike at marker size.
SHAPE = {
    "inverted_hammer": ("star", "below"),
    "hammer": ("triangle-up", "below"),
    "bullish_engulfing": ("circle", "below"),
    "doji": ("diamond", "above"),
}


def style_for(name: str) -> tuple[str, str, str]:
    """(colour, shape, side) for one pattern's marker."""
    shape, side = SHAPE.get(name, ("circle", "above"))
    return DIRECTION_COLOUR[patterns.get(name).direction], shape, side


# The wheel is deliberately left alone: Plotly's scrollZoom re-scales on every
# wheel event, and a trackpad emits a continuous stream of them plus momentum
# after your fingers lift, so the zoom factor compounds and the chart lurches.
PLOT_CONFIG = {
    "scrollZoom": False,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
    "doubleClick": "reset",
}


@st.cache_data(ttl=3600, show_spinner=False)
def load_symbols() -> list[str]:
    return store.symbols()


@st.cache_data(ttl=3600, show_spinner=False)
def load_bars(symbol: str, start: dt.date, end: dt.date) -> pd.DataFrame:
    df = store.read([symbol], start, end)
    return df.sort_values("date").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_summary() -> pd.DataFrame:
    return store.summary()


def candles(
    df: pd.DataFrame, hits: pd.DataFrame, title: str, show: list[str],
    collapse_gaps: bool = True, uirev: str = "chart",
) -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.78, 0.22], vertical_spacing=0.02,
    )
    fig.add_trace(
        go.Candlestick(
            x=df["date"], open=df["open"], high=df["high"],
            low=df["low"], close=df["close"], name="OHLC",
            increasing={"line": {"color": UP, "width": 1}, "fillcolor": UP},
            decreasing={"line": {"color": DOWN, "width": 1}, "fillcolor": DOWN},
            whiskerwidth=0.0,   # plain wicks rather than the default end-caps
            line={"width": 1},
            hoverlabel={"namelength": 0},
        ),
        row=1, col=1,
    )

    # One marker per bar. `label` holds the winning pattern where several
    # fired, so a bar that is both a doji and an inverted hammer is drawn as
    # the inverted hammer -- the more informative of the two.
    label = df["_label"] if "_label" in df.columns else None

    pad = (df["high"].max() - df["low"].min()) * 0.03
    for name in show:
        mask = hits[name] if label is None else (label == name)
        # `label` is a nullable string column, so `== name` yields NA on bars
        # with no pattern. Make it a plain bool before indexing.
        mask = mask.fillna(False).astype(bool)
        if not mask.any():
            continue
        colour, symbol_shape, side = style_for(name)
        marked = df[mask]
        y = marked["low"] - pad if side == "below" else marked["high"] + pad
        fig.add_trace(
            go.Scatter(
                x=marked["date"], y=y, mode="markers", name=patterns.label(name),
                marker={"color": colour, "symbol": symbol_shape, "size": 10,
                        "line": {"width": 1, "color": "rgba(255,255,255,0.85)"},
                        "opacity": 0.95},
                hovertemplate=(
                    f"<b>{patterns.label(name)}</b>"
                    "<br>%{x|%Y-%m-%d}<extra></extra>"
                ),
            ),
            row=1, col=1,
        )

    # Volume takes the colour of its own bar's direction, so the two panes
    # read as one chart rather than two.
    up_day = df["close"] >= df["open"]
    fig.add_trace(
        go.Bar(
            x=df["date"], y=df["volume"], name="volume", showlegend=False,
            marker={"color": [UP if u else DOWN for u in up_day],
                    "line": {"width": 0}},
            opacity=0.45, hovertemplate="vol %{y:,.0f}<extra></extra>",
        ),
        row=2, col=1,
    )
    fig.update_layout(
        # No Plotly title. It renders into the same top strip as the range
        # selector and the legend, and the three end up stacked on top of each
        # other with no way to space them reliably. The heading is drawn by
        # Streamlit above the chart instead, where normal layout rules apply.
        height=720,
        dragmode="pan",              # drag scrolls through time; wheel zooms
        # Streamlit re-runs the whole script on any widget interaction and
        # rebuilds the figure, which throws away the pan/zoom you just did --
        # the chart snaps back to the default view mid-gesture. uirevision
        # tells Plotly to keep the current view as long as this value is
        # unchanged, so zoom survives a re-run but still resets when you pick a
        # different symbol or timeframe.
        uirevision=uirev,
        # "x unified" builds one combined hover box from every trace on each
        # mouse move; with a candlestick, five marker series and volume that is
        # seven lookups per pixel. Plain "x" is the same information for the
        # bar under the cursor at a fraction of the work.
        hovermode="x",
        # Top margin holds the range buttons and the legend, stacked with room
        # to breathe: buttons at y=1.13, legend at y=1.04.
        margin={"l": 8, "r": 8, "t": 78, "b": 8},
        legend={"orientation": "h", "y": 1.04, "x": 0, "yanchor": "bottom",
                "bgcolor": "rgba(0,0,0,0)", "font": {"size": 11}},
        paper_bgcolor="rgba(0,0,0,0)",   # inherit the Streamlit theme
        plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.25,
        font={"family": _style.FONT, "color": _style.INK_2, "size": 12},
        hoverlabel={"bgcolor": _style.PANEL, "bordercolor": _style.BORDER,
                    "font": {"family": _style.FONT, "size": 12,
                             "color": _style.INK}},
        xaxis2={"rangeslider": {"visible": False}, "type": "date"},
        xaxis={
            # go.Candlestick turns a rangeslider on by default, on its OWN
            # axis. It must be switched off here, inside this dict -- setting
            # xaxis_rangeslider_visible elsewhere in update_layout gets
            # overwritten by this one. Left on, it renders a miniature copy of
            # every trace (candles and all five marker series) in the strip
            # below row 1, which is exactly where the volume pane sits.
            "rangeslider": {"visible": False},
            "rangeselector": {
                "buttons": [
                    {"count": 1, "label": "1M", "step": "month", "stepmode": "backward"},
                    {"count": 3, "label": "3M", "step": "month", "stepmode": "backward"},
                    {"count": 6, "label": "6M", "step": "month", "stepmode": "backward"},
                    {"count": 1, "label": "YTD", "step": "year", "stepmode": "todate"},
                    {"count": 1, "label": "1Y", "step": "year", "stepmode": "backward"},
                    {"step": "all", "label": "All"},
                ],
                "bgcolor": _style.PANEL,
                "activecolor": "rgba(255,255,255,0.16)",
                "bordercolor": _style.BORDER,
                "x": 0, "y": 1.13, "yanchor": "bottom", "font": {"size": 11},
            },
        },
    )
    # No spikes. spikemode="across" with spikesnap="cursor" forces a redraw
    # spanning both subplots on every single mouse move, which is the most
    # expensive thing on the page and buys very little.
    fig.update_xaxes(showgrid=True, gridcolor=GRID, zeroline=False, showspikes=False)
    # Collapse the days the market was shut, so daily candles sit side by side.
    #
    # Only for daily bars. On weekly/monthly the bars are already ~7 or ~30
    # days apart, so nearly every calendar day becomes a break -- 2,052 of them
    # against 343 bars for a six-year weekly chart. Plotly then computes a
    # candle width of zero and draws nothing at all, while the scatter markers
    # (being points, not widths) still render, which makes it look like the
    # price data is missing rather than the axis being wrong.
    if collapse_gaps:
        sessions = set(pd.to_datetime(df["date"]))
        full = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
        gaps = [d for d in full if d not in sessions]
        # Weekend bounds are deliberately NOT used: NSE runs real Saturday and
        # Sunday sessions (Muhurat, Union Budget, DR-site), and those bars must
        # stay visible.
        if gaps:
            fig.update_xaxes(rangebreaks=[{"values": gaps}])

    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False,
        side="right", ticklabelposition="outside", row=1, col=1,
    )
    fig.update_yaxes(
        showgrid=False, zeroline=False, side="right",
        tickformat="~s", row=2, col=1,
    )
    return fig


# --------------------------------------------------------------------------

symbols = load_symbols()
if not symbols:
    st.error("The store is empty. Run the backfill first.")
    st.stop()

with st.sidebar:
    st.header("Chart")
    default = symbols.index("RELIANCE") if "RELIANCE" in symbols else 0
    symbol = st.selectbox("Symbol", symbols, index=default)

    today = ist_today()
    preset = st.radio("Range", ["1M", "3M", "6M", "1Y", "3Y", "All"],
                      index=2, horizontal=True)
    months = {"1M": 1, "3M": 3, "6M": 6, "1Y": 12, "3Y": 36}.get(preset)
    start = today - pd.DateOffset(months=months) if months else dt.date(2000, 1, 1)
    start = pd.Timestamp(start).date()

    timeframe = st.radio("Timeframe", ["Daily", "Weekly", "Monthly"], horizontal=True)

    st.header("Patterns")
    show = [
        n for n in patterns.names()
        if st.checkbox(patterns.label(n), value=True, key=f"p_{n}")
    ]

bars = load_bars(symbol, start, today)
if bars.empty:
    st.warning(f"No bars stored for {symbol} in this range.")
    st.stop()

if timeframe != "Daily":
    freq = "W" if timeframe == "Weekly" else "M"
    indexed = bars.set_index("date")
    agg = resample_ohlc(indexed, freq)
    if not st.sidebar.checkbox("Include forming candle", value=False):
        agg = agg[agg["complete"]]
    bars = agg.reset_index().rename(columns={"index": "date"})
    bars["date"] = pd.to_datetime(bars["last_session"]).fillna(bars["date"])

hits = patterns.detect_by_symbol(bars) if show else pd.DataFrame(index=bars.index)
# Winning pattern per bar, chosen among the ones currently ticked, so
# unticking inverted_hammer lets a doji underneath it show through again.
bars = bars.copy()
bars["_label"] = patterns.classify(bars, show) if show else pd.NA

c1, c2, c3, c4 = st.columns(4)
c1.metric("Bars shown", f"{len(bars):,}")
c2.metric("Last close", f"{bars['close'].iloc[-1]:,.2f}")
change = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1
c3.metric("Change over range", f"{change * 100:+.1f}%")
c4.metric("Bars flagged", int(bars["_label"].notna().sum()) if show else 0)

st.markdown(
    f"<div style='margin:1.6rem 0 0.9rem 0;font-size:1.35rem;font-weight:650;'>"
    f"{symbol} <span style='opacity:.55;font-weight:400;'>· {timeframe.lower()}</span>"
    f"</div>",
    unsafe_allow_html=True,
)
st.plotly_chart(
    candles(bars, hits, f"{symbol} — {timeframe.lower()}", show,
            collapse_gaps=timeframe == "Daily",
            uirev=f"{symbol}|{timeframe}|{preset}"),
    width="stretch",
    config=PLOT_CONFIG,
    key="chart",
)
st.caption(
    "Drag to pan · double-click to reset · range buttons above the chart · "
    "box-zoom and ± in the toolbar at the top right of the chart"
)

if show and bars["_label"].notna().any():
    st.subheader("Hits")
    flagged = bars.loc[bars["_label"].notna()].copy()
    flagged = flagged.rename(columns={"_label": "pattern"})
    # `pattern` is the one drawn on the chart; `also` records the rest, so the
    # overlap stays visible rather than being silently dropped.
    flagged["also"] = [
        ", ".join(n for n in show if hits.loc[i, n] and n != flagged.loc[i, "pattern"])
        for i in flagged.index
    ]
    cols = ["date", "pattern", "also", "open", "high", "low", "close", "volume"]
    st.dataframe(
        flagged[[c for c in cols if c in flagged.columns]].sort_values("date", ascending=False),
        width="stretch", hide_index=True,
    )

with st.expander("Store health"):
    st.dataframe(load_summary(), width="stretch", hide_index=True)
    st.caption(
        "Prices are unadjusted for splits and bonuses. Comparing a period that "
        "spans a corporate action against an adjusted chart will not match."
    )
