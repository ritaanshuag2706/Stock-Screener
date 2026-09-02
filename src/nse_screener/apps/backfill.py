"""Walk a date range, download each day's bhavcopy, load it into the store.

Resumable by design — the run is long enough that it will be interrupted:

  * days already in the store are skipped
  * days NSE has no file for are remembered, so holidays are asked about once
  * raw zips are reused, so a re-run after a parsing fix costs no downloads
  * the store is flushed every N days, so Ctrl-C loses at most that many

    python -m nse_screener.apps.backfill --start 2023-01-01 --end 2023-01-31
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from ..data import bhavcopy, store
from ..market_calendar import Overrides
from ..paths import CONFIG_DIR, DATA_DIR, ensure_dirs

log = logging.getLogger("backfill")

# A 404 this recent is more likely "not published yet" than "holiday".
# NSE posts the bhavcopy in the evening IST; do not cache a miss inside this
# window or today's file gets permanently marked as absent.
RECENT_DAYS = 5


def missing_cache_path() -> Path:
    return DATA_DIR / "no_data_dates.json"


def load_missing() -> set[date]:
    path = missing_cache_path()
    if not path.is_file():
        return set()
    return {date.fromisoformat(s) for s in json.loads(path.read_text())}


def save_missing(dates: set[date]) -> None:
    missing_cache_path().write_text(
        json.dumps(sorted(d.isoformat() for d in dates), indent=0)
    )


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="backfill", description="Backfill NSE daily bars into the parquet store."
    )
    today = bhavcopy.ist_today()
    p.add_argument("--start", type=parse_date, default=today - timedelta(days=365 * 3),
                   help="first date, YYYY-MM-DD (default: 3 years ago)")
    p.add_argument("--end", type=parse_date, default=today,
                   help="last date, YYYY-MM-DD (default: today)")
    p.add_argument("--delay", type=float, default=2.5,
                   help="seconds between requests (default: 2.5). NSE rate-limits.")
    p.add_argument("--flush-every", type=int, default=20,
                   help="write to the store every N downloaded days (default: 20)")
    p.add_argument("--force", action="store_true",
                   help="re-download and re-load days already present")
    p.add_argument("--retry-missing", action="store_true",
                   help="ignore the cache of days NSE had no file for")
    p.add_argument("--limit", type=int, default=None,
                   help="stop after N days; useful for a first look")
    p.add_argument("--verbose", "-v", action="store_true")
    return p


def declared_sessions() -> set[date]:
    """Dates the override file asserts were trading sessions."""
    return set(Overrides.load(CONFIG_DIR / "holiday_overrides.yaml").open)


def weekday_and_weekend_sessions(start: date, end: date) -> list[date]:
    """Days worth asking NSE about.

    Weekdays, plus any weekend date the override file declares a session.
    Diwali Muhurat trading falls on whatever day Diwali lands -- 2020-11-14 was
    a Saturday and 2023-11-12 a Sunday, and both printed real prices. Walking
    only Mon-Fri silently skips them.

    Weekends are not swept blindly: that would add ~2,400 requests over six
    years, essentially all 404s. Reconciliation is what discovers a missed
    session, and the override file is where the answer gets recorded.
    """
    days = set(bhavcopy.trading_weekdays(start, end))
    overrides = Overrides.load(CONFIG_DIR / "holiday_overrides.yaml")
    days |= {d for d in overrides.open if start <= d <= end and d.weekday() >= 5}
    return sorted(days)


def flush(frames: list[pd.DataFrame]) -> int:
    if not frames:
        return 0
    rows = store.append(pd.concat(frames, ignore_index=True))
    frames.clear()
    return rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    ensure_dirs()

    if args.start > args.end:
        log.error("start %s is after end %s", args.start, args.end)
        return 2

    have = set() if args.force else store.available_dates()
    missing = set() if args.retry_missing else load_missing()
    declared = declared_sessions()
    candidates = weekday_and_weekend_sessions(args.start, args.end)
    todo = [d for d in candidates if d not in have and d not in missing]
    if args.limit:
        todo = todo[: args.limit]

    log.info("range %s..%s: %d weekdays, %d already stored, %d known holidays",
             args.start, args.end, len(candidates),
             len(candidates) - len([d for d in candidates if d not in have]),
             len([d for d in candidates if d in missing]))
    if not todo:
        log.info("nothing to do")
        return 0

    mins = len(todo) * args.delay / 60
    log.info("fetching %d days at %.1fs apart (~%.0f min)", len(todo), args.delay, mins)

    session = bhavcopy.make_session()
    frames: list[pd.DataFrame] = []
    ok = holidays = failed = 0
    today = bhavcopy.ist_today()

    try:
        for i, d in enumerate(todo, 1):
            try:
                cached = bhavcopy.raw_path(d).exists()
                # Special weekend sessions exist in sec_bhavdata_full but not
                # as a bhavcopy zip; only pay for that second request where the
                # override file says a session happened.
                df = bhavcopy.fetch(d, session=session, force=args.force,
                                    allow_secfull=d in declared)
                frames.append(df)
                ok += 1
                log.info("[%d/%d] %s  %5d rows%s", i, len(todo), d, len(df),
                         "  (cached)" if cached else "")
            except bhavcopy.NoDataForDate:
                holidays += 1
                if (today - d).days > RECENT_DAYS:
                    missing.add(d)
                    save_missing(missing)
                    log.info("[%d/%d] %s  no file (holiday)", i, len(todo), d)
                else:
                    log.info("[%d/%d] %s  not published yet", i, len(todo), d)
            except Exception as exc:  # noqa: BLE001 - one bad day must not end an overnight run
                failed += 1
                log.error("[%d/%d] %s  FAILED: %s", i, len(todo), d, exc)

            if len(frames) >= args.flush_every:
                log.info("... flushed %d rows", flush(frames))
            if i < len(todo) and not cached:
                time.sleep(args.delay)

    except KeyboardInterrupt:
        log.warning("interrupted — flushing what is already downloaded")
    finally:
        rows = flush(frames)
        if rows:
            log.info("flushed %d rows", rows)

    log.info("done: %d loaded, %d holidays, %d failed", ok, holidays, failed)
    summary = store.summary()
    if not summary.empty:
        log.info("store now holds:\n%s", summary.to_string(index=False))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
