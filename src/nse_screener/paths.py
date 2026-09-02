"""Filesystem locations, resolved once so nothing else hardcodes a path.

Layout note: ``src/nse_screener/data/`` is *code* (bhavcopy download, parquet
store). ``<repo>/data/`` is the *data directory* and is never committed.
"""

from __future__ import annotations

import os
from pathlib import Path

# src/nse_screener/paths.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Override for testing or an external disk: NSE_SCREENER_DATA_DIR=/Volumes/...
DATA_DIR = Path(os.environ.get("NSE_SCREENER_DATA_DIR", REPO_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"        # every downloaded file, kept verbatim
BARS_DIR = DATA_DIR / "bars"      # year-partitioned parquet
CONFIG_DIR = REPO_ROOT / "config"


def ensure_dirs() -> None:
    """Create the data directories if they do not exist yet."""
    for path in (DATA_DIR, RAW_DIR, BARS_DIR):
        path.mkdir(parents=True, exist_ok=True)
