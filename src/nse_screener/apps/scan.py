"""Print the patterns that printed on a session.

    python -m nse_screener.apps.scan
    python -m nse_screener.apps.scan --date 2026-07-15 --patterns hammer inverted_hammer
    python -m nse_screener.apps.scan --min-turnover 10 --min-history 500
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from .. import patterns as pat
from .. import ranking
from ..screener import Universe, scan


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="scan", description="Candlestick patterns for one session."
    )
    p.add_argument("--date", type=parse_date, default=None,
                   help="last session to scan, YYYY-MM-DD (default: latest in the store)")
    p.add_argument("--days", type=int, default=1, metavar="N",
                   help="report the last N trading sessions, not just one (default: 1)")
    p.add_argument("--patterns", nargs="+", default=None, metavar="NAME",
                   help=f"limit to these (default: all of {', '.join(pat.names())})")
    p.add_argument("--min-history", type=int, default=250,
                   help="minimum sessions of history (default: 250)")
    p.add_argument("--max-gap", type=float, default=20.0,
                   help="reject symbols with an overnight move above this %% in the "
                        "detector window, as a corporate-action guard (default: 20)")
    p.add_argument("--min-price", type=float, default=20.0,
                   help="exclude symbols closing below this (default: 20). 0 turns "
                        "the price floor off")
    p.add_argument("--min-traded-value", type=float, default=1e7,
                   help="exclude symbols whose median daily close x volume is below "
                        "this (default: 1e7, i.e. Rs 1 crore). 0 turns it off")
    p.add_argument("--rank-by", default=None, metavar="NAME",
                   choices=[*ranking.names(), "none"],
                   help="order hits within each session; adds a Score column. "
                        f"one of: {', '.join(ranking.names())}. No ranker has yet "
                        "been shown to beat 'random' -- ordering the list does not "
                        "make the top of it better")
    p.add_argument("--limit", type=int, default=None, help="show at most N hits")
    p.add_argument("--csv", metavar="PATH", default=None, help="also write hits to CSV")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.patterns:
        unknown = [p for p in args.patterns if p not in pat.names()]
        if unknown:
            print(f"unknown pattern(s): {', '.join(unknown)}", file=sys.stderr)
            print(f"known: {', '.join(pat.names())}", file=sys.stderr)
            return 2

    rules = Universe(
        min_history=args.min_history,
        max_overnight_gap=args.max_gap / 100,
        min_price=args.min_price,
        min_traded_value=args.min_traded_value,
    )
    rank_by = None if args.rank_by in (None, "none") else args.rank_by
    result = scan(args.date, sessions=args.days, rules=rules,
                  which=args.patterns, rank_by=rank_by)

    if result.universe_size == 0:
        print("no eligible symbols — is the store populated?", file=sys.stderr)
        return 1

    span = (f"{result.sessions[0]} .. {result.sessions[-1]}"
            if len(result.sessions) > 1 else str(result.asof))
    print(f"{span}   {len(result.sessions)} session(s)   "
          f"{result.universe_size:,} eligible symbols   {len(result):,} hits")
    print("  excluded: " + ", ".join(f"{v:,} {k}" for k, v in result.rejected.items()))
    print()

    if result.hits.empty:
        print("  nothing printed a pattern.")
        return 0

    out = result.hits.copy()
    if args.limit:
        out = out.head(args.limit)
    shown = out.assign(
        pattern=out["pattern"].map(pat.labels()).fillna(out["pattern"]),
        close=out["close"].map("{:,.2f}".format),
        chg=out["chg_pct"].map("{:+.2f}%".format),
        vol=out["volume"].map("{:,.0f}".format),
        trend=out["above_200ema"].map({True: "up", False: "down"}).fillna("?"),
        e200=out["dist_200ema_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%"),
        e25=out["dist_25ema_pct"].map(lambda v: "—" if pd.isna(v) else f"{v:+.1f}%"),
        rvol=out["rel_volume"].map(lambda v: "—" if pd.isna(v) else f"{v:.2f}x"),
        rsi=out["rsi_14"].map(lambda v: "—" if pd.isna(v) else f"{v:.0f}"),
    )
    cols = ["date", "symbol", "pattern", "close", "chg", "vol",
            "trend", "e200", "e25", "rvol", "rsi"]
    headers = ["Date", "Symbol", "Pattern", "Close", "Chg", "Volume",
               "Trend", "vs200", "vs25", "RelVol", "RSI"]
    if "rank_score" in out.columns:
        shown = shown.assign(
            score=out["rank_score"].map(lambda v: "—" if pd.isna(v) else f"{v:,.2f}")
        )
        cols.append("score")
        headers.append("Score")
    shown = shown[cols]
    shown.columns = headers
    print(shown.to_string(index=False))

    if len(out) < len(result.hits):
        print(f"\n  ... {len(result.hits) - len(out):,} more")

    print()
    print("  " + " · ".join(
        f"{n} {pat.label(c)}" for c, n in result.by_pattern().items()))
    if len(result.sessions) > 1:
        print("  " + " · ".join(f"{d}: {n}" for d, n in result.by_date().items()))

    if args.csv:
        result.hits.to_csv(args.csv, index=False)
        print(f"\n  written to {args.csv}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
