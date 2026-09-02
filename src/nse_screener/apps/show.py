"""Print stored bars for one symbol, for eyeballing against a chart.

Stage 1 is not finished until three symbols have been checked against
TradingView on five random dates each. This is the tool for that.

    python -m nse_screener.apps.show RELIANCE --last 10
    python -m nse_screener.apps.show TCS --start 2024-01-01 --end 2024-01-31
    python -m nse_screener.apps.show INFY --sample 5
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

import pandas as pd

from ..data import store


def parse_date(s: str) -> date:
    return date.fromisoformat(s)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="show", description="Print stored bars for a symbol.")
    p.add_argument("symbol")
    p.add_argument("--start", type=parse_date, default=None)
    p.add_argument("--end", type=parse_date, default=None)
    p.add_argument("--last", type=int, default=None, help="show only the last N bars")
    p.add_argument("--sample", type=int, default=None,
                   help="show N random bars — what you want for a spot-check")
    p.add_argument("--seed", type=int, default=None, help="make --sample reproducible")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    symbol = args.symbol.upper()

    df = store.read([symbol], args.start, args.end)
    if df.empty:
        print(f"no bars stored for {symbol}", file=sys.stderr)
        known = store.symbols()
        near = [s for s in known if symbol in s][:5]
        if near:
            print(f"did you mean: {', '.join(near)}", file=sys.stderr)
        elif not known:
            print("the store is empty — run the backfill first", file=sys.stderr)
        return 1

    out = df.drop(columns=["symbol"])
    out["date"] = pd.to_datetime(out["date"]).dt.date

    if args.sample:
        out = out.sample(min(args.sample, len(out)), random_state=args.seed)
        out = out.sort_values("date")
    elif args.last:
        out = out.tail(args.last)

    print(f"{symbol}: {len(df)} bars stored, {df['date'].min().date()} to "
          f"{df['date'].max().date()}\n")
    print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
