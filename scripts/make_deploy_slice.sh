#!/bin/bash
#
# Carve a deployable slice of the store, small enough to live in the repo.
#
#     ./scripts/make_deploy_slice.sh            # last 2 calendar years
#     ./scripts/make_deploy_slice.sh 3          # last 3
#
# Streamlit Community Cloud rebuilds its container from the repository on every
# restart, sleep and redeploy, and its filesystem does not persist between them.
# So a deployed app cannot download its own history: whatever it fetched would
# be discarded on the next wake, and the backfill takes about two hours anyway.
# The data has to ship with the code.
#
# It does not have to be *all* of it. The screener needs 260 bars of context
# warmup and 250 sessions of per-symbol history, so roughly two years is enough
# to scan recent dates -- against 102 MB for the full history. The store is
# partitioned by year, so a slice is a directory copy rather than a rewrite.
#
# The study and the backtest still want the full history. This slice is for the
# deployed UI only; keep the real store locally.
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
YEARS="${1:-2}"
SRC="$ROOT/data/bars"
OUT="$ROOT/deploy_data/bars"

[ -d "$SRC" ] || { echo "No store at $SRC -- run bootstrap_data.sh first"; exit 1; }

THIS_YEAR="$(date +%Y)"
FIRST=$((THIS_YEAR - YEARS + 1))

rm -rf "$ROOT/deploy_data"
mkdir -p "$OUT"

echo "Copying years $FIRST..$THIS_YEAR"
copied=0
for y in $(seq "$FIRST" "$THIS_YEAR"); do
    if [ -d "$SRC/year=$y" ]; then
        cp -R "$SRC/year=$y" "$OUT/"
        echo "  year=$y  $(du -sh "$OUT/year=$y" | cut -f1)"
        copied=$((copied + 1))
    fi
done
[ "$copied" -gt 0 ] || { echo "No matching years found in $SRC"; exit 1; }

echo
echo "Slice total: $(du -sh "$ROOT/deploy_data" | cut -f1)"
echo

# --- prove the app can actually run on it ---------------------------------
# A slice that is merely small is not useful; it has to still produce a scan.
# Pointing NSE_SCREENER_DATA_DIR at the slice exercises the same code path the
# deployed app will take.
echo "Verifying the screener runs against the slice alone..."
NSE_SCREENER_DATA_DIR="$ROOT/deploy_data" "$PY" - <<'EOF'
from nse_screener.data import store
from nse_screener.screener import scan

dates = sorted(store.available_dates())
print(f"  {len(dates)} sessions, {min(dates)} -> {max(dates)}")

r = scan(sessions=1)
print(f"  scan on {r.asof}: {r.universe_size} eligible, {len(r.hits)} hits")
if r.hits.empty:
    raise SystemExit(
        "  FAIL: no hits. The slice is probably too short for the 250-session "
        "history rule -- re-run with more years."
    )
print(f"  patterns: {dict(r.by_pattern())}")
print("  OK -- the deployed app will work on this slice.")
EOF

cat <<'EOF'

Next steps:
  1. Copy deploy_data/bars/ into the repo as data/bars/
  2. `.gitignore` currently excludes `data/` and `*.parquet`, so either add an
     exception or commit with `git add -f`.
  3. Keep your full local store; this slice is only for the deployment.
EOF
