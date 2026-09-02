"""Stage 7. What actually happened after each signal, across all history.

This is the decision point the project rests on. It answers one question:

    given a pattern, how often did price reach +X ATR before -Y ATR,
    and is that any different from a randomly chosen bar?

**The control is the point.** A 44% hit rate means nothing on its own. If a
random bar in the same universe also hits 44%, the pattern is worthless -- it
has told you only what the market did in general. So every table here carries
an `all bars` row computed exactly the same way, and a `lift` column against it.
Reading a rate without the control is the single easiest way to fool yourself
at this stage.

Sample size sits next to every rate for the same reason. 62% on 40 instances is
noise wearing a percentage sign.

No lookahead, by construction:

  * the signal is read on bar t's close
  * entry is bar t+1's **open** -- never t's close, which you could not have traded
  * the target and stop are sized from the ATR as it stood on bar t
  * outcomes are searched forward over bars t+1 .. t+horizon

Two conservative choices, both of which push results *down*:

  * when a bar's high reaches the target and its low reaches the stop, the stop
    is recorded. Daily bars cannot say which came first, and assuming the good
    one would inflate every number here.
  * a signal whose forward window contains an unadjusted corporate action is
    dropped rather than measured. Prices either side of a split are not
    comparable, so the "return" would be an artefact.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .. import context as ctx
from .. import patterns as pat
from ..data import store
from ..patterns import _geometry as g
from ..screener import Universe, eligibility, gap_flags, history_counts

log = logging.getLogger(__name__)

ALL_BARS = "all bars"
"""The control row. Every bar of every eligible symbol, measured identically."""


@dataclass(frozen=True)
class Rules:
    """How an outcome is defined. Changing these changes every number below."""

    target_atr: float = 2.0
    stop_atr: float = 1.0
    horizon: int = 10
    """Trading sessions after entry to wait before calling it a timeout."""

    atr_period: int = 14
    max_gap: float = 0.20
    """A move this large inside the forward window means a corporate action, so
    the signal is dropped rather than measured against incomparable prices."""

    def __post_init__(self):
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")
        if self.target_atr <= 0 or self.stop_atr <= 0:
            raise ValueError("target_atr and stop_atr must be positive")

    @property
    def reward_risk(self) -> float:
        return self.target_atr / self.stop_atr


def forward_outcomes(
    bars: pd.DataFrame, *, by: pd.Series | None = None, rules: Rules | None = None
) -> pd.DataFrame:
    """For every bar, what happened over the next `horizon` sessions.

    Returns `entry`, `target`, `stop`, `outcome`, `bars_held`, `fwd_return_pct`
    and `usable`, indexed like `bars`. `outcome` is target / stop / timeout, and
    NA where the trade could not be evaluated.
    """
    rules = rules or Rules()
    d = g.normalise(bars)
    out = pd.DataFrame(index=bars.index)

    signal_atr = ctx.atr(d, by, rules.atr_period)
    entry = g.shift_by(d["open"], -1, by)          # bar t+1's open
    out["entry"] = entry
    out["target"] = entry + rules.target_atr * signal_atr
    out["stop"] = entry - rules.stop_atr * signal_atr

    first_target = pd.Series(np.nan, index=bars.index)
    first_stop = pd.Series(np.nan, index=bars.index)
    gapped = pd.Series(False, index=bars.index)
    change = (
        d["close"].groupby(by).pct_change() if by is not None else d["close"].pct_change()
    ).abs()

    for h in range(1, rules.horizon + 1):
        high_h = g.shift_by(d["high"], -h, by)
        low_h = g.shift_by(d["low"], -h, by)
        first_target = first_target.mask(
            first_target.isna() & (high_h >= out["target"]), h
        )
        first_stop = first_stop.mask(first_stop.isna() & (low_h <= out["stop"]), h)
        gapped |= g.shift_by(change, -h, by).fillna(False) > rules.max_gap

    # A bar with fewer than `horizon` bars ahead of it cannot be evaluated: it
    # would look like a timeout when the answer is simply not in yet.
    last_close = g.shift_by(d["close"], -rules.horizon, by)
    out["usable"] = entry.notna() & last_close.notna() & signal_atr.notna() & ~gapped

    # Ties go to the stop -- a daily bar cannot say which came first.
    hit_target = first_target.notna() & (
        first_stop.isna() | (first_target < first_stop)
    )
    hit_stop = first_stop.notna() & ~hit_target

    outcome = pd.Series(pd.NA, index=bars.index, dtype="string")
    outcome = outcome.mask(out["usable"] & hit_target, "target")
    outcome = outcome.mask(out["usable"] & hit_stop, "stop")
    outcome = outcome.mask(
        out["usable"] & ~hit_target & ~hit_stop, "timeout"
    )
    out["outcome"] = outcome
    out["bars_held"] = (
        first_target.where(hit_target)
        .fillna(first_stop.where(hit_stop))
        .fillna(float(rules.horizon))
        .where(out["usable"])
    )
    out["fwd_return_pct"] = ((last_close / entry - 1) * 100).where(out["usable"])
    return out


def rates(
    frame: pd.DataFrame, group: str | list[str] | None = None, *, min_n: int = 0
) -> pd.DataFrame:
    """Hit rate, sample size and average forward return, grouped however you ask.

    `frame` needs `outcome` and `fwd_return_pct`. Rows with no outcome are
    excluded, so `n` is the number actually evaluated.
    """
    usable = frame[frame["outcome"].notna()]
    if usable.empty:
        return pd.DataFrame(
            columns=["n", "target", "stop", "timeout", "hit_rate", "avg_return"]
        )

    if group is None:
        usable = usable.assign(**{ALL_BARS: ALL_BARS})
        group = ALL_BARS

    keys = [group] if isinstance(group, str) else list(group)
    grouped = usable.groupby(keys, observed=True)

    counts = (
        grouped["outcome"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=["target", "stop", "timeout"], fill_value=0)
    )
    out = counts.assign(
        n=grouped.size(),
        avg_return=grouped["fwd_return_pct"].mean(),
    )
    out["hit_rate"] = out["target"] / out["n"] * 100
    out = out[["n", "target", "stop", "timeout", "hit_rate", "avg_return"]]
    return out[out["n"] >= min_n].sort_values("n", ascending=False)


def with_control(table: pd.DataFrame, control: pd.DataFrame) -> pd.DataFrame:
    """Append the all-bars row, the lift over it, and whether that lift is real.

    Lift is percentage points above the control's hit rate. Without it a rate is
    unreadable: 44% is either an edge or exactly the market, and the number alone
    cannot tell you which.

    `z` is the lift divided by its standard error -- a two-proportion test
    against the control. Below about 2 the difference is inside the noise of the
    sample, however large the sample is. It is here because "report sample sizes
    next to every rate" is easy to satisfy and still be fooled: 254,000 samples
    make a 0.2-point difference *measurable* without making it *useful*.
    """
    base = control.iloc[0]
    out = pd.concat([table, control])

    p, n = out["hit_rate"] / 100, out["n"]
    p0, n0 = base["hit_rate"] / 100, base["n"]
    se = np.sqrt(p * (1 - p) / n + p0 * (1 - p0) / n0)

    out["lift"] = out["hit_rate"] - base["hit_rate"]
    out["z"] = (out["lift"] / 100) / se.replace(0, np.nan)
    return out


# Cut points for the continuous context columns. Chosen to be readable rather
# than optimal -- an optimised cut fitted on this data is the overfitting the
# plan warns about. Stage 8 is where a surviving hypothesis gets tested on a
# period this study has not touched.
BUCKETS: dict[str, tuple[list[float], list[str]]] = {
    "rel_volume": ([0, 0.75, 1.5, 3.0, np.inf], ["<0.75x", "0.75-1.5x", "1.5-3x", ">3x"]),
    "dist_200ema_pct": (
        [-np.inf, -20, -5, 5, 20, np.inf],
        ["<-20%", "-20..-5%", "-5..+5%", "+5..+20%", ">+20%"],
    ),
    "bb_pct_b": ([-np.inf, 0, 0.2, 0.8, 1, np.inf],
                 ["below band", "0-0.2", "0.2-0.8", "0.8-1", "above band"]),
    "atr_pct": ([0, 2, 4, 6, np.inf], ["<2%", "2-4%", "4-6%", ">6%"]),
    "stoch_k": ([0, 20, 80, 100], ["<20", "20-80", ">80"]),
    "macd_hist_pct": ([-np.inf, -0.5, 0, 0.5, np.inf],
                      ["<-0.5", "-0.5..0", "0..+0.5", ">+0.5"]),
}

CATEGORICAL = ["above_200ema", "ema_stack", "rsi_zone"]
"""Already discrete -- no cutting needed."""


def bucket(frame: pd.DataFrame, column: str) -> pd.Series:
    """A readable categorical version of one context column."""
    if column in CATEGORICAL:
        return frame[column].astype("string")
    if column not in BUCKETS:
        raise KeyError(
            f"no buckets defined for {column!r}; "
            f"known: {', '.join(sorted(set(BUCKETS) | set(CATEGORICAL)))}"
        )
    edges, labels = BUCKETS[column]
    return pd.cut(frame[column], bins=edges, labels=labels, right=False)


def by_context(
    frame: pd.DataFrame,
    column: str,
    *,
    pattern: str | None = None,
    min_n: int = 200,
) -> pd.DataFrame:
    """Hit rate within each bucket of one context column, against its control.

    The control is the same bucket over *all* bars, not the overall rate -- so
    the comparison isolates the pattern rather than re-reporting that, say, low
    volatility behaves differently from high.
    """
    work = frame.assign(_bucket=bucket(frame, column))
    control = rates(work, "_bucket", min_n=min_n).add_suffix("_all")
    subset = work[work["pattern"] == pattern] if pattern else work
    table = rates(subset, "_bucket", min_n=min_n)

    joined = table.join(control, how="left")
    joined["lift"] = joined["hit_rate"] - joined["hit_rate_all"]
    p, n = joined["hit_rate"] / 100, joined["n"]
    p0, n0 = joined["hit_rate_all"] / 100, joined["n_all"]
    se = np.sqrt(p * (1 - p) / n + p0 * (1 - p0) / n0)
    joined["z"] = (joined["lift"] / 100) / se.replace(0, np.nan)
    return joined[["n", "hit_rate", "hit_rate_all", "lift", "z", "avg_return"]]


def study(
    start: date | str | None = None,
    end: date | str | None = None,
    *,
    rules: Rules | None = None,
    universe: Universe | None = None,
    which: list[str] | None = None,
    bars_dir=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load history, detect, measure. Returns (annotated bars, control table).

    The returned frame has one row per bar of every eligible symbol, carrying
    its pattern (or NA), its context columns and its forward outcome -- which is
    everything the grouping functions need.
    """
    rules = rules or Rules()
    universe = universe or Universe()

    bars = store.read(
        start=start, end=end, series=list(universe.series), bars_dir=bars_dir
    )
    if bars.empty:
        return bars, pd.DataFrame()

    counts = history_counts(
        end or max(store.available_dates(bars_dir)), universe.series, bars_dir
    )
    elig = eligibility(counts, gap_flags(bars, universe.max_overnight_gap), universe)
    # Only the history rule applies here. The gap rule is a *live* screening
    # guard; over six years almost every symbol has moved 20% at some point, and
    # excluding them all would leave nothing. Corporate actions are handled per
    # signal instead, inside forward_outcomes.
    #
    # `Universe`'s liquidity floor is deliberately off here too, by not passing
    # `liquid=` to eligibility(). Two reasons: the recorded Stage 7 numbers were
    # measured without it and must stay reproducible, and a base *rate* wants the
    # widest possible sample -- illiquidity distorts what a fill is worth, which
    # is a backtest question, not a question about how often a bar prints. Turn
    # it on and every number in the Stage 7 section of CLAUDE.md changes.
    keep = set(elig.loc[elig["enough_history"], "symbol"])
    bars = bars[bars["symbol"].isin(keep)].sort_values(["symbol", "date"])
    log.info("studying %s bars across %s symbols", f"{len(bars):,}", len(keep))

    by = bars["symbol"]
    bars = bars.join(pat.detect_by_symbol(bars, which))
    bars["pattern"] = pat.classify_by_symbol(bars, which)
    bars = bars.join(ctx.annotate_by_symbol(bars))
    bars = bars.join(forward_outcomes(bars, by=by, rules=rules))

    control = rates(bars)
    return bars, control


def main(argv: list[str] | None = None) -> int:
    """python -m nse_screener.study.base_rates"""
    import argparse

    p = argparse.ArgumentParser(prog="base_rates", description=__doc__.split("\n")[0])
    p.add_argument("--start", default="2020-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--target", type=float, default=2.0, help="target in ATR")
    p.add_argument("--stop", type=float, default=1.0, help="stop in ATR")
    p.add_argument("--horizon", type=int, default=10, help="sessions to wait")
    p.add_argument("--context", nargs="*", default=None,
                   help="context columns to break down by")
    p.add_argument("--min-n", type=int, default=500)
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    rules = Rules(target_atr=args.target, stop_atr=args.stop, horizon=args.horizon)
    bars, control = study(args.start, args.end, rules=rules)
    if bars.empty:
        print("no data in that range", flush=True)
        return 1

    pd.set_option("display.width", 140)
    print(f"\n+{rules.target_atr} ATR target / -{rules.stop_atr} ATR stop, "
          f"{rules.horizon} sessions, entry at the next open\n")
    print(with_control(rates(bars, "pattern"), control).round(2).to_string())

    for column in args.context or []:
        for pattern in sorted(bars["pattern"].dropna().unique()):
            table = by_context(bars, column, pattern=pattern, min_n=args.min_n)
            if table.empty:
                continue
            print(f"\n--- {pattern} by {column} ---")
            print(table.round(2).to_string())

    print("\nRead the lift column, not the hit rate. `z` below 2 is noise.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
