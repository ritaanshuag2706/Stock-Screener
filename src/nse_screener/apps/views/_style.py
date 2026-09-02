"""Shared visual language: tokens, page chrome, and chart builders.

Colour decisions here were computed, not chosen by eye. The four patterns cannot
be given four categorical hues: on this dark surface every candidate four-colour
set fails the all-pairs colourblind separation gate (worst pair OKLab dE 1.5-6.5
against a target of 8, and normal-vision dE 7-12 against a floor of 15). Marker
colours on one chart are compared in *any* pair, not just adjacent ones, so
all-pairs is the applicable test.

So colour and shape split the work:

    colour  ->  direction      bullish / neutral   (2 hues, validated: CVD dE 8.3,
                                                    normal dE 19.8, both PASS)
    shape   ->  which pattern  4 distinct marks

Bar charts use a single hue. A count-by-category bar chart is one series with the
category on the axis, so it needs no categorical palette at all -- giving each bar
its own colour would encode the axis twice and hit the same gate for nothing.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

# --- tokens (dark, stepped for Streamlit's #0e1117 surface) ------------------

SURFACE = "#0e1117"
PANEL = "#161a23"
INK = "#ffffff"
INK_2 = "#c3c2b7"
MUTED = "#898781"
GRID = "#2c2c2a"
BASELINE = "#383835"
BORDER = "rgba(255,255,255,0.10)"

SERIES = "#3987e5"
"""Single hue for count bars. Sequential blue, step 400."""

BULLISH = "#199e70"
NEUTRAL = "#c98500"
"""Direction hues. Validated as a pair on this surface."""

CANDLE_UP = "#26a69a"
CANDLE_DOWN = "#ef5350"
"""Not a categorical series -- a domain convention with its own meaning.
6.30:1 and 5.42:1 against the surface, separated by dE 29.6."""

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'

CSS = f"""
<style>
  /* Recessive chrome, so the data is the only loud thing on the page. */
  .block-container {{ padding-top: 2.4rem; max-width: 1500px; }}

  h1, h2, h3 {{ letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.9rem !important; font-weight: 650 !important; }}
  h2 {{ font-size: 1.15rem !important; font-weight: 600 !important;
        color: {INK_2} !important; margin: 1.6rem 0 0.5rem 0 !important; }}
  h3 {{ font-size: 0.82rem !important; font-weight: 600 !important;
        text-transform: uppercase; letter-spacing: 0.07em;
        color: {MUTED} !important; margin: 0 0 0.5rem 0 !important; }}

  /* Stat tiles: a hairline ring and real padding, not floating text. */
  div[data-testid="stMetric"] {{
    background: {PANEL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 0.85rem 1rem;
  }}
  div[data-testid="stMetricLabel"] p {{
    font-size: 0.75rem !important; text-transform: uppercase;
    letter-spacing: 0.06em; color: {MUTED} !important;
  }}
  div[data-testid="stMetricValue"] {{
    font-size: 1.75rem !important; font-weight: 600 !important;
    color: {INK} !important; line-height: 1.15;
  }}

  /* Tabular figures so columns of numbers line up vertically. */
  div[data-testid="stDataFrame"] {{ font-variant-numeric: tabular-nums; }}
  div[data-testid="stDataFrame"] div[role="grid"] {{ border-radius: 10px; }}

  section[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
  section[data-testid="stSidebar"] h2 {{ margin-top: 1.2rem !important; }}

  div[data-testid="stExpander"] {{
    border: 1px solid {BORDER}; border-radius: 10px; background: {PANEL};
  }}
  .stCaption, div[data-testid="stCaptionContainer"] p {{ color: {MUTED} !important; }}
  hr {{ border-color: {BORDER}; }}
</style>
"""


def apply() -> None:
    """Inject the page chrome. Cheap and idempotent."""
    st.markdown(CSS, unsafe_allow_html=True)


# --- chart builders ---------------------------------------------------------


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin={"l": 0, "r": 8, "t": 4, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": FONT, "color": INK_2, "size": 12},
        showlegend=False,          # one series: the axis already names it
        bargap=0.42,               # leftover band is air, not a fatter bar
        hoverlabel={"bgcolor": PANEL, "bordercolor": BORDER,
                    "font": {"family": FONT, "size": 12, "color": INK}},
    )
    return fig


def hbar(labels: list[str], values: list[float], *, height: int = 210) -> go.Figure:
    """Horizontal count bars: single hue, value labelled at the tip.

    Horizontal because the category names are words -- rotating text to fit
    vertical columns is the usual reason a chart becomes unreadable.
    """
    fig = go.Figure(
        go.Bar(
            x=values, y=labels, orientation="h",
            marker={"color": SERIES, "cornerradius": 4},
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont={"color": INK_2, "size": 12},
            cliponaxis=False,       # a tip label near the edge must not be cropped
            hovertemplate="%{y}: %{x:,}<extra></extra>",
            width=0.55,
        )
    )
    _base_layout(fig, height)
    fig.update_xaxes(visible=False, range=[0, max(values) * 1.18 if values else 1])
    fig.update_yaxes(
        showgrid=False, zeroline=False, ticks="",
        tickfont={"color": INK_2, "size": 12},
        autorange="reversed",       # largest at the top
    )
    return fig


def vbar(labels: list[str], values: list[float], *, height: int = 210) -> go.Figure:
    """Vertical count columns, value on the cap. For a short ordered sequence."""
    fig = go.Figure(
        go.Bar(
            x=labels, y=values,
            marker={"color": SERIES, "cornerradius": 4},
            text=[f"{v:,}" for v in values],
            textposition="outside",
            textfont={"color": INK_2, "size": 12},
            cliponaxis=False,
            hovertemplate="%{x}: %{y:,}<extra></extra>",
            width=0.45,
        )
    )
    _base_layout(fig, height)
    fig.update_xaxes(showgrid=False, zeroline=False, ticks="",
                     tickfont={"color": INK_2, "size": 12})
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False,
        tickfont={"color": MUTED, "size": 11}, ticks="",
        range=[0, max(values) * 1.22 if values else 1],
    )
    return fig


CHART_CONFIG = {"displayModeBar": False, "staticPlot": False}
