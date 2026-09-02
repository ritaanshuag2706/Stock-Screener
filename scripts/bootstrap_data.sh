#!/bin/bash
#
# Rebuild the entire data store from scratch on a fresh clone.
#
#     ./scripts/bootstrap_data.sh                  # 2020-01-01 to today
#     ./scripts/bootstrap_data.sh 2024-01-01       # a shorter history
#
# The store is NOT in the repository, and should not be: it is ~285 MB of
# derived data (102 MB of parquet, 183 MB of raw downloads) that regenerates
# byte-for-byte from NSE's public archive. The repo itself is under 1 MB.
#
# This differs from daily_ingest.sh, which tops up an existing store and
# deliberately refuses to start a multi-hour download when it finds none. This
# script is the one that *does* the long download, so it is separate and
# explicit rather than something a nightly cron job can trigger by accident.
#
# Safe to interrupt. The store dedupes on (date, symbol) and the raw downloads
# are cached, so re-running resumes rather than restarting.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
START="${1:-2020-01-01}"

[ -x "$PY" ] || {
    echo "No interpreter at $PY"
    echo "Create the environment first:"
    echo "    python3.13 -m venv .venv && .venv/bin/pip install -e \".[dev]\""
    exit 1
}

cd "$ROOT"

echo "Rebuilding the store from $START."
echo
echo "NSE rate-limits, so this pauses 2.5s between sessions -- roughly two"
echo "hours for a full six-year history. It is resumable: interrupt it and run"
echo "it again and it picks up where it stopped."
echo

"$PY" -m nse_screener.apps.backfill --start "$START"

echo
echo "Verifying against the trading calendar..."
"$PY" - <<'EOF'
import datetime as dt
from nse_screener.data import store
from nse_screener.market_calendar import TradingCalendar

dates = sorted(store.available_dates())
if not dates:
    raise SystemExit("store is empty -- the backfill downloaded nothing")

cal = TradingCalendar("config/holiday_overrides.yaml")
gaps = cal.reconcile(dates, min(dates), max(dates))

print(f"  {len(dates)} sessions, {min(dates)} -> {max(dates)}")
if gaps:
    print(f"  {len(gaps)} calendar mismatch(es):")
    for m in gaps[:10]:
        print(f"     {m.date}  {m.kind}")
    print("  A 'missing_session' means a download was missed; re-run this script.")
else:
    print("  0 calendar mismatches -- every session the calendar expects is present.")
EOF

echo
echo "Done. Check it with:"
echo "    .venv/bin/python -m nse_screener.apps.scan --days 1"
