"""Deployment entry point. Streamlit Community Cloud looks for this file.

Locally, the documented command still works and is unaffected by this file:

    .venv/bin/streamlit run src/nse_screener/apps/dashboard.py

Why this exists rather than `pip install .` in requirements.txt, which is the
usual answer for a src/ layout: `paths.py` derives every location from its own
position on disk.

    REPO_ROOT = Path(__file__).resolve().parents[2]

That is correct while the package lives in `<repo>/src/nse_screener/`, and wrong
as soon as pip copies it into site-packages -- `parents[2]` then lands inside
the virtualenv, so CONFIG_DIR and DATA_DIR point at directories that do not
exist. The app dies at import with a traceback about a missing YAML file, which
sends you looking in entirely the wrong place.

So the package stays where it is and `src/` goes on the import path instead.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Check for the store before importing anything that reads it. A deployment
# missing its data otherwise surfaces as an empty scan, which reads like "no
# patterns tonight" rather than "this app has no data".
if not (ROOT / "data" / "bars").exists():
    import streamlit as st

    st.error(
        "**No data store found at `data/bars/`.**\n\n"
        "The store is not in the repository by default — it is ~100 MB of "
        "derived data. A deployment has to carry it, because the hosting "
        "filesystem is wiped on every restart and the backfill takes about two "
        "hours.\n\n"
        "Commit at least the two most recent `data/bars/year=YYYY/` partitions. "
        "The screener needs 260 bars of context warm-up and 250 sessions of "
        "per-symbol history before it reports anything, so one year is not "
        "enough to produce hits."
    )
    st.stop()

runpy.run_path(
    str(SRC / "nse_screener" / "apps" / "dashboard.py"),
    run_name="__main__",
)
