"""How ten slots get filled when five hundred signals fire.

This is the missing half of every backtest run so far. With ~480 signals a night
and 10 positions, the book is full almost always -- `skipped_no_slot` was
328,843 -- so *which* signals got traded was decided by a shuffle. Every result
in this project therefore measured an arbitrary subset of the strategy, and no
comparison between two signal families could distinguish "better signal" from
"luckier draw".

A ranker answers the question the detectors never ask. A detector is a
time-series claim: *does this bar predict this stock's own future?* A ranker is
cross-sectional: *of the five hundred names that qualify tonight, which are the
best relative to each other?* Those are different questions and they can have
different answers -- a signal with no time-series edge can still order its own
candidates usefully, and a signal with an edge can be ruined by taking the worst
ten of it every night.

Same contract as a detector, and for the same reason: a pure function, no I/O,
so the study and the backtest can both use it unchanged.

    DataFrame of candidates -> Series of scores, higher is better

Three invariants, each of which has a test:

* **`random` is a real ranker, not a fallback.** It is the same policy the
  engine used before this module existed -- uniform, no preference -- which
  makes it the control. (Not the same draw: it consumes the generator
  differently, so pre-ranking absolute figures will not reproduce to the rupee.
  Ranker-against-control on one seed, which is the comparison that matters, is
  unaffected.) A ranker that cannot beat `random` is not a ranking rule, it is
  a decoration, and that comparison is only possible because the baseline is
  named and runnable.

* **Ranking sees the signal bar and nothing after it.** Candidates are scored on
  bar `t` and filled at `t+1` open. Handing a ranker any forward column would
  manufacture an edge far larger than anything the detectors were ever measured
  at, and it would look like a discovery rather than a bug.

* **Ties break deterministically, by symbol.** Otherwise the arbitrariness this
  module exists to remove simply moves inside the tie and stops being visible.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

Scorer = Callable[[pd.DataFrame, np.random.Generator], pd.Series]


@dataclass(frozen=True)
class Ranker:
    """A named way of ordering tonight's candidates."""

    name: str
    fn: Scorer
    needs: tuple[str, ...]
    """Columns the scorer reads. Validated before it runs, so a missing context
    column is a named error rather than a silent frame of NaN scores that
    degrade quietly into the tie-break."""

    description: str

    @property
    def label(self) -> str:
        return self.name.replace("_", " ").capitalize()


_REGISTRY: dict[str, Ranker] = {}


def register(name: str, *, needs: Iterable[str] = (), description: str = ""):
    def decorate(fn: Scorer) -> Scorer:
        if name in _REGISTRY:
            raise ValueError(f"ranker {name!r} is already registered")
        _REGISTRY[name] = Ranker(name, fn, tuple(needs), description)
        return fn

    return decorate


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Ranker:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown ranker {name!r}; registered: {', '.join(names())}"
        ) from None


def rank(
    candidates: pd.DataFrame,
    by: str = "random",
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Candidates ordered best-first, with the score that ordered them.

    Returns a copy carrying `rank_score`, so a caller can report *why* a name
    took a slot rather than only that it did.

    Scores that come back NaN sort last rather than raising: a symbol missing a
    context column is a worse candidate than one that has it, but it is not a
    reason to abandon the night. Ties -- including the all-NaN case -- break by
    symbol, so the same input always produces the same order.
    """
    ranker = get(by)
    if candidates.empty:
        return candidates.assign(rank_score=pd.Series(dtype=float))

    missing = [c for c in ranker.needs if c not in candidates.columns]
    if missing:
        raise KeyError(
            f"ranker {by!r} needs column(s) {missing}, which the candidate frame "
            f"does not carry. Context columns come from "
            f"`context.annotate_by_symbol()`; the frame has: "
            f"{sorted(candidates.columns)}"
        )
    if "symbol" not in candidates.columns:
        raise KeyError("candidates must carry a 'symbol' column to break ties")

    out = candidates.copy()
    out["rank_score"] = score(candidates, by, rng=rng)
    # NaN last, then symbol ascending -- `na_position` handles the first and the
    # symbol key the second, so the order is total and reproducible.
    return out.sort_values(
        ["rank_score", "symbol"], ascending=[False, True], na_position="last"
    )


def score(
    candidates: pd.DataFrame,
    by: str = "random",
    *,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Just the scores, aligned to `candidates.index`.

    Separate from `rank()` because scoring is row-wise while ordering may not
    be: the screener ranks *within* each session, and gets that from one sort on
    `[date, rank_score]` rather than a groupby -- fewer moving parts, and it
    keeps the score comparable across the whole frame for anyone who wants it.
    """
    ranker = get(by)
    if candidates.empty:
        return pd.Series(dtype=float, index=candidates.index)
    missing = [c for c in ranker.needs if c not in candidates.columns]
    if missing:
        raise KeyError(
            f"ranker {by!r} needs column(s) {missing}, which the candidate frame "
            f"does not carry. Context columns come from "
            f"`context.annotate_by_symbol()`; the frame has: "
            f"{sorted(candidates.columns)}"
        )
    rng = rng if rng is not None else np.random.default_rng(0)
    return pd.to_numeric(ranker.fn(candidates, rng), errors="coerce").astype(float)


def top(
    candidates: pd.DataFrame,
    n: int,
    by: str = "random",
    *,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """The best `n`. `n <= 0` means no limit, matching `--limit` on the CLI."""
    ordered = rank(candidates, by, rng=rng)
    return ordered if n <= 0 else ordered.head(n)


# --------------------------------------------------------------------------
# the rankers
# --------------------------------------------------------------------------


@register(
    "random",
    description="Uniform noise. The control every other ranker is judged against.",
)
def _random(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """What the engine did before this module existed.

    Keeping it named and runnable is the whole point: "ranking by X returns
    more than ranking at random" is the only form of that claim worth making,
    and it needs the random arm to be a real arm.
    """
    return pd.Series(rng.random(len(c)), index=c.index)


@register(
    "rel_volume",
    needs=("rel_volume",),
    description="Heaviest volume relative to the symbol's own normal, first.",
)
def _rel_volume(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Conviction. The hypothesis is that a pattern printed on five times normal
    volume means something a pattern printed on a quiet Tuesday does not."""
    return c["rel_volume"]


@register(
    "trend_strength",
    needs=("dist_200ema_pct",),
    description="Furthest above the 200 EMA first. Ride what is already working.",
)
def _trend_strength(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    return c["dist_200ema_pct"]


@register(
    "low_volatility",
    needs=("atr_pct",),
    description="Calmest names first, for the best risk-adjusted use of a slot.",
)
def _low_volatility(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Negated because the contract is higher-is-better and the claim here is
    that a tighter ATR spends less of the risk budget per unit of opportunity."""
    return -c["atr_pct"]


@register(
    "liquidity",
    needs=("close", "volume"),
    description="Largest traded value first -- the fills you would actually get.",
)
def _liquidity(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """Not a prediction, a practicality. Two spectacular false positives in this
    project came from illiquid names, and while `Universe` now floors them out,
    ranking by size is the difference between a strategy that survives contact
    with a real order book and one that only survives a simulation of it."""
    return c["close"] * c["volume"]


@register(
    "signal_specificity",
    needs=("pattern",),
    description="Rarest signal first, using the pattern registry's specificity.",
)
def _signal_specificity(c: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """The registry already ranks how narrow a claim each signal makes. If that
    number means anything, the narrower signal deserves the contested slot --
    which also makes this a direct test of whether `specificity` is meaningful
    or merely tidy."""
    from . import patterns as pat

    spec = {n: pat.get(n).specificity for n in pat.names()}
    return c["pattern"].map(spec).astype(float)
