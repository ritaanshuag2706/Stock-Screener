"""Year-partitioned parquet store for daily bars.

    data/bars/year=2024/bars.parquet

One file per year keeps the partition count small enough that a full read is
cheap, while a single day's append only rewrites one year. Writes go to a temp
file in the same directory and are renamed into place, so an interrupted run
leaves the previous partition intact rather than a truncated one.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from ..paths import BARS_DIR

log = logging.getLogger(__name__)

# A bar is uniquely identified by these. Re-downloading a day overwrites rather
# than duplicating, which is what makes the backfill safe to re-run.
KEY = ["date", "symbol"]

# Columns that must be float64 in every partition. A writer that hands over
# whole-number prices would otherwise create an int64 partition, and read()
# would concatenate it with float64 years into something unpredictable. The
# bhavcopy parser already pins these; enforcing here covers every other writer.
FLOAT_COLS = ["open", "high", "low", "close", "prev_close", "turnover"]


def _root(bars_dir: Path | None) -> Path:
    return bars_dir or BARS_DIR


def partition_path(year: int, bars_dir: Path | None = None) -> Path:
    return _root(bars_dir) / f"year={year}" / "bars.parquet"


def partitions(bars_dir: Path | None = None) -> dict[int, Path]:
    """Existing year partitions, keyed by year."""
    root = _root(bars_dir)
    if not root.exists():
        return {}
    found = {}
    for d in sorted(root.glob("year=*")):
        path = d / "bars.parquet"
        if path.is_file():
            found[int(d.name.split("=")[1])] = path
    return found


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False, compression="snappy")
    tmp.replace(path)


def append(df: pd.DataFrame, *, bars_dir: Path | None = None) -> int:
    """Merge `df` into the store. Returns the number of rows written.

    Rows already present for the same (date, symbol) are replaced by the
    incoming ones, so a re-download of a corrected file wins.
    """
    if df.empty:
        return 0

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    for col in FLOAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("float64")
    written = 0

    for year, chunk in df.groupby(df["date"].dt.year):
        path = partition_path(int(year), bars_dir)
        if path.is_file():
            merged = pd.concat([pd.read_parquet(path), chunk], ignore_index=True)
        else:
            merged = chunk
        merged = (
            merged.drop_duplicates(subset=KEY, keep="last")
            .sort_values(KEY)
            .reset_index(drop=True)
        )
        _write_atomic(merged, path)
        written += len(chunk)
        log.debug("year=%s now holds %d rows", year, len(merged))

    return written


def read(
    symbols: list[str] | None = None,
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    columns: list[str] | None = None,
    series: list[str] | None = None,
    bars_dir: Path | None = None,
) -> pd.DataFrame:
    """Bars for `symbols` between `start` and `end`, inclusive of both ends."""
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None

    if columns is not None:
        columns = list(dict.fromkeys([*KEY, *columns]))  # key columns always come back

    frames = []
    for year, path in sorted(partitions(bars_dir).items()):
        # Skip partitions that cannot intersect the window before reading them.
        if start_ts is not None and year < start_ts.year:
            continue
        if end_ts is not None and year > end_ts.year:
            continue

        part = pd.read_parquet(path, columns=columns)
        if symbols is not None:
            part = part[part["symbol"].isin(symbols)]
        if series is not None and "series" in part.columns:
            part = part[part["series"].isin(series)]
        if start_ts is not None:
            part = part[part["date"] >= start_ts]
        if end_ts is not None:
            part = part[part["date"] <= end_ts]
        if not part.empty:
            frames.append(part)

    if not frames:
        return pd.DataFrame(columns=columns or KEY)

    return pd.concat(frames, ignore_index=True).sort_values(KEY).reset_index(drop=True)


def available_dates(bars_dir: Path | None = None) -> set[date]:
    """Every date holding at least one bar.

    Reads the date column alone via parquet column projection, so the backfill
    can ask "what do I already have?" without loading price data.
    """
    out: set[date] = set()
    for path in partitions(bars_dir).values():
        col = pq.read_table(path, columns=["date"]).column("date").to_pandas()
        out.update(col.dt.date.unique())
    return out


def symbols(bars_dir: Path | None = None) -> list[str]:
    out: set[str] = set()
    for path in partitions(bars_dir).values():
        col = pq.read_table(path, columns=["symbol"]).column("symbol").to_pandas()
        out.update(col.unique())
    return sorted(out)


def summary(bars_dir: Path | None = None) -> pd.DataFrame:
    """Per-year row counts, distinct symbols and date span."""
    rows = []
    for year, path in sorted(partitions(bars_dir).items()):
        t = pq.read_table(path, columns=["date", "symbol"]).to_pandas()
        rows.append({
            "year": year,
            "rows": len(t),
            "symbols": t["symbol"].nunique(),
            "sessions": t["date"].nunique(),
            "first": t["date"].min().date(),
            "last": t["date"].max().date(),
            "mb": round(path.stat().st_size / 1e6, 1),
        })
    return pd.DataFrame(rows)
