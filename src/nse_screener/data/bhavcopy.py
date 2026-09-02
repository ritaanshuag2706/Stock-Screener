"""Download and parse one day's NSE equity bhavcopy.

NSE publishes a daily archive file, so this is an HTTP fetch plus a CSV parse,
not HTML scraping. Two layouts exist and both are needed for a multi-year
backfill:

  legacy  through 2024-07-05  cm05JUL2024bhav.csv.zip
  UDiFF   from    2024-07-08  BhavCopy_NSE_CM_0_0_0_20240708_F_0000.csv.zip

Both boundaries were probed against the live archive: legacy 404s from
2024-07-08 onward, UDiFF is absent in 2023 and present from early 2024.

Every download is kept verbatim under data/raw/. When a parsing bug turns up
later, re-parsing costs nothing; re-downloading three years costs a night.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from ..clock import IST, ist_today  # noqa: F401  (re-exported; callers use bhavcopy.ist_today)
from ..paths import RAW_DIR

log = logging.getLogger(__name__)

# First date served in the UDiFF layout rather than the legacy one.
UDIFF_FROM = date(2024, 7, 8)

LEGACY_URL = (
    "https://nsearchives.nseindia.com/content/historical/EQUITIES"
    "/{year}/{mon}/cm{day:02d}{mon}{year}bhav.csv.zip"
)
UDIFF_URL = (
    "https://nsearchives.nseindia.com/content/cm"
    "/BhavCopy_NSE_CM_0_0_0_{stamp}_F_0000.csv.zip"
)
# Fallback source. NSE runs special weekend sessions -- Union Budget day when
# 1 February falls on a weekend -- and publishes this file for them but *not*
# the bhavcopy zip, which 404s. Verified for 2020-02-01, 2025-02-01 and
# 2026-02-01: the Saturday close here explains Monday's prev_close for 1,615
# of 1,615 symbols. Coverage starts around 2020.
SECFULL_URL = (
    "https://nsearchive s.nseindia.com/products/content"
    "/sec_bhavdata_full_{stamp}.csv"
)

# strftime("%b") is locale-dependent; NSE always uses these.
_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")

USER_AGENT = (
    "stock_screening_k"
)

# Equity series worth storing. EQ is the main board; BE is trade-to-trade,
# still a real daily bar. Anything else here is bonds, T-bills, SME or ETFs.
# Stored rather than dropped so Stage 4 can decide; filtering at read time is
# free, re-downloading is not.
EQUITY_SERIES = ("EQ", "BE")

# Canonical schema every parser must produce, in order.
COLUMNS = [
    "date", "symbol", "series",
    "open", "high", "low", "close", "prev_close",
    "volume", "turnover", "trades", "isin",
]
_PRICE_COLS = ["open", "high", "low", "close", "prev_close"]


class NoDataForDate(Exception):
    """NSE has no file for this date — a holiday, a weekend, or too recent."""


class DownloadFailed(Exception):
    """The request failed for a reason that is not a missing file."""


# --------------------------------------------------------------------------
# URLs
# --------------------------------------------------------------------------


def layout_for(d: date) -> str:
    return "udiff" if d >= UDIFF_FROM else "legacy"


def url_for(d: date) -> str:
    if layout_for(d) == "udiff":
        return UDIFF_URL.format(stamp=d.strftime("%Y%m%d"))
    return LEGACY_URL.format(year=d.year, mon=_MONTHS[d.month - 1], day=d.day)


def secfull_url_for(d: date) -> str:
    return SECFULL_URL.format(stamp=d.strftime("%d%m%Y"))


def raw_path(d: date, raw_dir: Path | None = None) -> Path:
    """Where the untouched zip for `d` lives. Keeps NSE's own filename."""
    return (raw_dir or RAW_DIR) / url_for(d).rsplit("/", 1)[-1]


def secfull_raw_path(d: date, raw_dir: Path | None = None) -> Path:
    return (raw_dir or RAW_DIR) / secfull_url_for(d).rsplit("/", 1)[-1]


# --------------------------------------------------------------------------
# Fetching
# --------------------------------------------------------------------------


def make_session() -> requests.Session:
    """A session NSE will serve.

    Default Python user-agents get blocked. The warm-up GET picks up cookies;
    the archive host usually serves without them, so a failure there is logged
    and ignored rather than fatal.
    """
    s = requests.Session()
    s.headers.update({
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://www.nseindia.com/",
    })
    try:
        s.get("https://www.nseindia.com/", timeout=10)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        log.debug("cookie warm-up failed, continuing: %s", exc)
    return s


def _download_to(
    url: str,
    dest: Path,
    d: date,
    *,
    session: requests.Session | None = None,
    force: bool = False,
    retries: int = 3,
    backoff: float = 3.0,
    expect_zip: bool = True,
) -> Path:
    """Fetch `url` into `dest`, reusing an existing file unless `force`."""
    if dest.exists() and not force:
        log.debug("%s already downloaded", d)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    sess = session or make_session()
    last: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = sess.get(url, timeout=30)
            if resp.status_code == 404:
                # A 404 still returns a styled HTML page, so the status code is
                # the only reliable signal here.
                raise NoDataForDate(f"nothing published at {url}")
            resp.raise_for_status()
            if expect_zip and not resp.content.startswith(b"PK"):
                raise DownloadFailed(
                    f"{url} returned {len(resp.content)} bytes that are not a zip"
                )
            if not expect_zip and resp.content.lstrip()[:6].upper() != b"SYMBOL":
                raise DownloadFailed(
                    f"{url} returned {len(resp.content)} bytes that are not the CSV"
                )
            # Write-then-rename so an interrupt cannot leave a half file that a
            # later run would mistake for a complete download.
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(resp.content)
            tmp.replace(dest)
            return dest
        except NoDataForDate:
            raise
        except (requests.RequestException, DownloadFailed) as exc:
            last = exc
            if attempt < retries:
                wait = backoff * attempt
                log.warning("%s attempt %d/%d failed (%s); retrying in %.0fs",
                            d, attempt, retries, exc, wait)
                time.sleep(wait)

    raise DownloadFailed(f"giving up on {d} after {retries} attempts: {last}")


def download(d: date, *, raw_dir: Path | None = None, **kw) -> Path:
    """Fetch one day's bhavcopy zip. Raises NoDataForDate on a 404, which on a
    weekday means a market holiday."""
    return _download_to(url_for(d), raw_path(d, raw_dir), d, expect_zip=True, **kw)


def download_secfull(d: date, *, raw_dir: Path | None = None, **kw) -> Path:
    """Fetch one day's sec_bhavdata_full CSV — the fallback for weekend
    special sessions, where NSE publishes this but no bhavcopy zip."""
    return _download_to(
        secfull_url_for(d), secfull_raw_path(d, raw_dir), d, expect_zip=False, **kw
    )


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _read_csv_from_zip(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"expected exactly one CSV in {path.name}, found {members}")
        with zf.open(members[0]) as fh:
            return pd.read_csv(io.BytesIO(fh.read()), dtype=str, skipinitialspace=True)


def _finalise(df: pd.DataFrame, series: tuple[str, ...]) -> pd.DataFrame:
    """Apply the shared tail end of both parsers: filter, type, validate."""
    df = df[df["series"].isin(series)].copy()

    for col in _PRICE_COLS + ["volume", "turnover", "trades"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    # Pin the dtypes rather than letting inference pick. A day where every
    # price happens to be a whole number infers int64, and that partition would
    # then disagree with every other year when read() concatenates them.
    for col in _PRICE_COLS + ["turnover"]:
        df[col] = df[col].astype("float64")
    df["volume"] = df["volume"].astype("Int64")
    df["trades"] = df["trades"].astype("Int64")
    df["symbol"] = df["symbol"].str.strip()

    # A row with no close is unusable; a row violating high/low bounds means
    # the layout was misread and must not reach the store silently.
    before = len(df)
    df = df.dropna(subset=["close", "open", "high", "low"])
    bad = (df["high"] < df["low"]) | (df["high"] < df["close"]) | (df["low"] > df["close"])
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} rows violate high/low bounds, e.g. "
            f"{df.loc[bad, ['symbol', 'open', 'high', 'low', 'close']].head(3).to_dict('records')}"
        )
    if before != len(df):
        log.debug("dropped %d rows with missing prices", before - len(df))

    return df[COLUMNS].sort_values("symbol").reset_index(drop=True)


def _parse_legacy_timestamp(ts: pd.Series) -> pd.Series:
    """Legacy TIMESTAMP, tolerating the occasional two-digit year.

    Almost every legacy file uses 02-JAN-2023. Exactly one in 1,116 downloaded
    files — 2020-07-13 — was published as 13-Jul-20. Formats are tried
    explicitly rather than letting pandas infer, so a genuinely unexpected
    layout still raises instead of being silently guessed at.
    """
    for fmt in ("%d-%b-%Y", "%d-%b-%y"):
        try:
            return pd.to_datetime(ts, format=fmt)
        except ValueError:
            continue
    raise ValueError(
        f"unrecognised TIMESTAMP format, e.g. {ts.iloc[0]!r}; "
        f"expected DD-MON-YYYY or DD-MON-YY"
    )


def parse_legacy(path: Path, series: tuple[str, ...] = EQUITY_SERIES) -> pd.DataFrame:
    """SYMBOL,SERIES,OPEN,...,TIMESTAMP,TOTALTRADES,ISIN — plus a trailing comma."""
    raw = _read_csv_from_zip(path)
    raw = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]  # the trailing comma
    df = pd.DataFrame({
        "date": _parse_legacy_timestamp(raw["TIMESTAMP"]),
        "symbol": raw["SYMBOL"],
        "series": raw["SERIES"].str.strip(),
        "open": raw["OPEN"], "high": raw["HIGH"], "low": raw["LOW"],
        "close": raw["CLOSE"], "prev_close": raw["PREVCLOSE"],
        "volume": raw["TOTTRDQTY"], "turnover": raw["TOTTRDVAL"],
        "trades": raw["TOTALTRADES"], "isin": raw["ISIN"],
    })
    return _finalise(df, series)


def parse_udiff(path: Path, series: tuple[str, ...] = EQUITY_SERIES) -> pd.DataFrame:
    """UDiFF layout. Carries derivatives too, so filter FinInstrmTp == STK."""
    raw = _read_csv_from_zip(path)
    raw = raw[raw["FinInstrmTp"].str.strip() == "STK"]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["TradDt"], format="%Y-%m-%d"),
        "symbol": raw["TckrSymb"],
        "series": raw["SctySrs"].str.strip(),
        "open": raw["OpnPric"], "high": raw["HghPric"], "low": raw["LwPric"],
        "close": raw["ClsPric"], "prev_close": raw["PrvsClsgPric"],
        "volume": raw["TtlTradgVol"], "turnover": raw["TtlTrfVal"],
        "trades": raw["TtlNbOfTxsExctd"], "isin": raw["ISIN"],
    })
    return _finalise(df, series)


def parse_secfull(path: Path, series: tuple[str, ...] = EQUITY_SERIES) -> pd.DataFrame:
    """sec_bhavdata_full layout: a plain CSV with padded headers and values.

    Carries no ISIN. Turnover is quoted in lakhs, so it is scaled to rupees to
    match the bhavcopy schema.
    """
    raw = pd.read_csv(path, dtype=str, skipinitialspace=True)
    raw.columns = [c.strip() for c in raw.columns]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["DATE1"].str.strip(), format="%d-%b-%Y"),
        "symbol": raw["SYMBOL"],
        "series": raw["SERIES"].str.strip(),
        "open": raw["OPEN_PRICE"], "high": raw["HIGH_PRICE"],
        "low": raw["LOW_PRICE"], "close": raw["CLOSE_PRICE"],
        "prev_close": raw["PREV_CLOSE"],
        "volume": raw["TTL_TRD_QNTY"],
        # TURNOVER_LACS is in hundreds of thousands of rupees.
        "turnover": pd.to_numeric(raw["TURNOVER_LACS"], errors="coerce") * 1e5,
        "trades": raw["NO_OF_TRADES"],
        "isin": pd.NA,
    })
    return _finalise(df, series)


def parse(path: Path, d: date, series: tuple[str, ...] = EQUITY_SERIES) -> pd.DataFrame:
    """Parse a raw zip using the layout that applies to `d`.

    Asserts the file's own dates match `d`. A bhavcopy holds exactly one
    trading day, so any disagreement means a misread date format or a file
    served for the wrong day -- both of which would quietly poison the store.
    """
    if path.suffix.lower() == ".csv":
        df = parse_secfull(path, series)
    elif layout_for(d) == "udiff":
        df = parse_udiff(path, series)
    else:
        df = parse_legacy(path, series)

    if not df.empty:
        found = set(df["date"].dt.date.unique())
        if found != {d}:
            raise ValueError(
                f"{path.name} was requested for {d} but contains "
                f"{sorted(found)[:3]}{'...' if len(found) > 3 else ''}"
            )
    return df


def fetch(
    d: date,
    *,
    session: requests.Session | None = None,
    raw_dir: Path | None = None,
    force: bool = False,
    series: tuple[str, ...] = EQUITY_SERIES,
    allow_secfull: bool = False,
) -> pd.DataFrame:
    """Download (or reuse) and parse one day into the canonical schema.

    With `allow_secfull`, a missing bhavcopy falls back to sec_bhavdata_full.
    Off by default: most 404s are ordinary holidays, and asking twice for each
    would double the request count for no gain. The backfill enables it only
    for dates the override file declares a session.
    """
    try:
        path = download(d, session=session, raw_dir=raw_dir, force=force)
    except NoDataForDate:
        if not allow_secfull:
            raise
        log.info("%s: no bhavcopy, trying sec_bhavdata_full", d)
        path = download_secfull(d, session=session, raw_dir=raw_dir, force=force)
    return parse(path, d, series)


def trading_weekdays(start: date, end: date) -> list[date]:
    """Every Mon-Fri in [start, end]. Holidays are only discoverable by asking."""
    days, cur = [], start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(cur)
        cur += timedelta(days=1)
    return days
