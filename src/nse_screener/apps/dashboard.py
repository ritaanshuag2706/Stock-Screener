"""The app. Two pages: a chart for one symbol, and a scan across all of them.

    .venv/bin/streamlit run src/nse_screener/apps/dashboard.py

Both pages are presentation only. Every decision -- what counts as a pattern,
which symbols are eligible, what a threshold is -- lives in src/ behind a unit
test. If you find yourself writing an `if` about the market in a page file, it
belongs elsewhere.

Note that Streamlit caches imported modules for the life of the process. Editing
anything under src/nse_screener/ needs a server restart, not just a page reload.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from nse_screener.apps.views import _style

st.set_page_config(page_title="NSE screener", layout="wide")
_style.apply()

VIEWS = Path(__file__).parent / "views"

st.navigation([
    st.Page(VIEWS / "scan.py", title="Pattern scan", icon=":material/search:",
            default=True),
    st.Page(VIEWS / "chart.py", title="Chart", icon=":material/candlestick_chart:"),
]).run()
