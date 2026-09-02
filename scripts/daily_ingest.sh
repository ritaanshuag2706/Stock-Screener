#!/bin/bash
#
# Nightly data ingestion. Downloads any NSE sessions the store does not have
# and appends them, then reports what the store holds.
#
# Run by launchd (see com.nse-screener.ingest.plist), but safe to run by hand:
#
#     ./scripts/daily_ingest.sh
#
# Three things make this safe to fire repeatedly and safe to miss:
#
#   * The start date comes from the store, not from a fixed lookback. If the
#     laptop was off for a month, the next run catches up the whole month; if it
#     ran an hour ago, it re-checks one day. A hard-coded "--start 10 days ago"
#     would silently leave a hole after any longer outage.
#   * The store dedupes on (date, symbol) with last-write-wins, so re-fetching a
#     day already held is a no-op rather than a duplicate.
#   * A lock directory means a slow catch-up cannot overlap with the next
#     scheduled run. launchd will happily start a second copy otherwise.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOGS="$ROOT/logs"
LOCK="$ROOT/.scratch/ingest.lock"
LOG="$LOGS/ingest-$(date +%Y-%m-%d).log"

mkdir -p "$LOGS" "$ROOT/.scratch"

log() { printf '%s  %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*" | tee -a "$LOG"; }
die() { log "FAILED: $*"; exit 1; }

[ -x "$PY" ] || die "no interpreter at $PY -- create the venv first"

# --- one at a time --------------------------------------------------------
# mkdir is atomic, which `[ -e ]` followed by `touch` is not. A stale lock from
# a killed run is cleared after an hour rather than blocking every future run.
if ! mkdir "$LOCK" 2>/dev/null; then
    if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +60 2>/dev/null)" ]; then
        log "clearing stale lock (older than 60 min)"
        rmdir "$LOCK" 2>/dev/null || true
        mkdir "$LOCK" 2>/dev/null || die "could not take the lock"
    else
        log "another ingest is already running -- nothing to do"
        exit 0
    fi
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

# --- where to resume from -------------------------------------------------
log "=== ingest starting ==="

START="$("$PY" - <<'EOF' 2>/dev/null || true
from nse_screener.data import store
try:
    dates = store.available_dates()
    print(max(dates).isoformat() if dates else "")
except Exception:
    print("")
EOF
)"

if [ -z "$START" ]; then
    START="$(date -v-30d +%Y-%m-%d)"
    log "store is empty or unreadable; starting from $START"
else
    log "store's latest session is $START; resuming from there"
fi

# --- fetch ----------------------------------------------------------------
cd "$ROOT"
if "$PY" -m nse_screener.apps.backfill --start "$START" >>"$LOG" 2>&1; then
    log "backfill finished"
else
    die "backfill exited non-zero -- see $LOG"
fi

"$PY" - <<'EOF' 2>&1 | tee -a "$LOG" || true
import datetime as dt
from nse_screener.data import store
from nse_screener.market_calendar import TradingCalendar
dates = sorted(store.available_dates())
cal = TradingCalendar("config/holiday_overrides.yaml")
gaps = cal.reconcile(dates, dt.date(2020, 1, 1), max(dates))
print(f"store: {len(dates)} sessions, latest {max(dates)}, "
      f"{len(gaps)} calendar mismatch(es)")
EOF

# --- keep the log directory from growing without bound --------------------
find "$LOGS" -maxdepth 1 -name 'ingest-*.log' -mtime +30 -delete 2>/dev/null || true

log "=== ingest done ==="
