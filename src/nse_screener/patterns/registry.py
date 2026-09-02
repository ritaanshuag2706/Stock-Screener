"""Name -> detector lookup, and the thresholds each detector runs with.

A detector is a pure function: DataFrame in, boolean Series out, no I/O and no
side effects. That signature is what lets Stage 7 replay them over all history
unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import pandas as pd
import yaml

from ..paths import CONFIG_DIR

Detector = Callable[..., pd.Series]


@dataclass(frozen=True)
class Entry:
    name: str
    fn: Detector
    kind: str  # "single" | "double"
    direction: str  # "bullish" | "bearish" | "neutral"
    # How specific the pattern is, used only to pick one winner when several
    # fire on the same bar. Higher wins. Detection itself never consults it --
    # every detector answers its own question independently, and Stage 7 needs
    # them measured that way.
    specificity: int = 0

    @property
    def label(self) -> str:
        """Human-readable name, e.g. "Inverted hammer".

        `name` stays snake_case because it is an identifier -- a key in
        patterns.yaml, a column in the hits table, a CLI argument. This is the
        display form, derived so a new pattern gets one for free.
        """
        words = self.name.replace("_", " ")
        return words[:1].upper() + words[1:]


_REGISTRY: dict[str, Entry] = {}
_PARAMS: dict[str, dict] | None = None

PARAMS_PATH = CONFIG_DIR / "patterns.yaml"


def register(
    name: str, *, kind: str, direction: str, specificity: int = 0
) -> Callable[[Detector], Detector]:
    """Decorator adding a detector to the registry under `name`."""

    def wrap(fn: Detector) -> Detector:
        if name in _REGISTRY:
            raise ValueError(f"detector already registered: {name}")
        _REGISTRY[name] = Entry(
            name=name, fn=fn, kind=kind, direction=direction, specificity=specificity
        )
        return fn

    return wrap


def names() -> list[str]:
    return sorted(_REGISTRY)


def get(name: str) -> Entry:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown pattern {name!r}; known: {', '.join(names())}") from None


def load_params(*, refresh: bool = False) -> dict[str, dict]:
    """Thresholds from config/patterns.yaml, cached after first read."""
    global _PARAMS
    if _PARAMS is None or refresh:
        if PARAMS_PATH.is_file():
            _PARAMS = yaml.safe_load(PARAMS_PATH.read_text()) or {}
        else:
            _PARAMS = {}
    return _PARAMS


def params_for(name: str, **overrides) -> dict:
    """Config thresholds for `name`, with any keyword overrides applied on top."""
    get(name)  # raise early on a typo'd name
    return {**load_params().get(name, {}), **overrides}


def _reject_multi_symbol(df: pd.DataFrame, symbol_col: str = "symbol") -> None:
    """Refuse a frame holding more than one symbol.

    Detectors use .shift() for the previous bar and for the trend lookback.
    store.read() sorts by (date, symbol), so consecutive rows are usually
    *different* symbols -- shifting across that boundary compares one stock's
    bar to another's and invents patterns that never happened. Failing loudly
    is the only safe behaviour; use detect_by_symbol for multi-symbol frames.
    """
    if symbol_col in df.columns:
        found = df[symbol_col].nunique(dropna=False)
        if found > 1:
            raise ValueError(
                f"detect_all got {found} symbols in one frame. Detectors shift "
                f"across rows, so this would compare bars of different stocks. "
                f"Use detect_by_symbol(df) instead."
            )


def detect(name: str, df: pd.DataFrame, **overrides) -> pd.Series:
    """Run one detector over `df` using its configured thresholds."""
    _reject_multi_symbol(df)
    entry = get(name)
    return entry.fn(df, **params_for(name, **overrides))


def detect_all(df: pd.DataFrame, patterns: Iterable[str] | None = None) -> pd.DataFrame:
    """One boolean column per detector, indexed like `df`.

    Expects bars for a single symbol in date order. Raises on a multi-symbol
    frame -- see detect_by_symbol.
    """
    _reject_multi_symbol(df)
    wanted = list(patterns) if patterns is not None else names()
    return pd.DataFrame({name: detect(name, df) for name in wanted}, index=df.index)


def label(name: str) -> str:
    """Display form of one pattern name."""
    return get(name).label


def labels(patterns: Iterable[str] | None = None) -> dict[str, str]:
    """name -> display label, for formatting a column or a widget."""
    wanted = list(patterns) if patterns is not None else names()
    return {n: get(n).label for n in wanted}


def by_specificity(patterns: Iterable[str] | None = None) -> list[str]:
    """Pattern names, most specific first."""
    wanted = list(patterns) if patterns is not None else names()
    return sorted(wanted, key=lambda n: (-get(n).specificity, n))


def classify(
    df: pd.DataFrame, patterns: Iterable[str] | None = None
) -> pd.Series:
    """One primary label per bar, resolving overlaps by specificity.

    Detectors overlap by design -- a small-bodied bar with a long upper shadow
    that gapped down satisfies both `doji` and `inverted_hammer`, and TA-Lib
    reports both. That is right for measurement and wrong for a chart, where a
    bar wants one marker and should carry the most informative one.

    Precedence is the `specificity` on each registered pattern: the narrower
    the definition, the more it tells you, so it wins. `doji` is one condition
    and fires on roughly one bar in seven, so it loses to everything.

    Returns a string Series, NA where no pattern fired. The underlying boolean
    columns from `detect_all` are unchanged -- this only picks a winner.
    """
    hits = detect_all(df, patterns)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    # Least specific first so that more specific patterns overwrite them.
    for name in reversed(by_specificity(hits.columns)):
        out = out.mask(hits[name], name)
    return out


def detect_by_symbol(
    df: pd.DataFrame,
    patterns: Iterable[str] | None = None,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.DataFrame:
    """Run every detector per symbol, on a frame straight from store.read().

    Each symbol's bars are isolated and sorted by date before any detector
    sees them. The result is indexed like `df`, so it can be joined straight
    back onto the input.
    """
    if symbol_col not in df.columns:
        return detect_all(df, patterns)
    if date_col not in df.columns:
        raise ValueError(
            f"need a {date_col!r} column to order bars within a symbol; "
            f"got {list(df.columns)}"
        )

    wanted = list(patterns) if patterns is not None else names()
    if df.empty:
        return pd.DataFrame({n: pd.Series(dtype=bool) for n in wanted}, index=df.index)

    # One vectorised pass over every symbol at once, with `by` keeping each
    # rolling window and shift inside its own symbol.
    #
    # The obvious implementation -- loop the symbols and call detect_all on each
    # -- costs about 6ms per symbol regardless of how many bars it has, because
    # each detector fires ~50 small pandas operations and per-call overhead
    # dominates on a short series. Across 2,500 symbols that is 15 seconds.
    # Sorting once and grouping inside the arithmetic does the same work in
    # well under a second.
    ordered = df.sort_values([symbol_col, date_col])
    by = ordered[symbol_col]
    hits = {
        name: get(name).fn(ordered, by=by, **params_for(name)) for name in wanted
    }
    # reindex restores the caller's original row order
    return pd.DataFrame(hits, index=ordered.index).reindex(df.index)


def classify_by_symbol(
    df: pd.DataFrame,
    patterns: Iterable[str] | None = None,
    *,
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> pd.Series:
    """`classify` over a multi-symbol frame straight from store.read()."""
    hits = detect_by_symbol(df, patterns, symbol_col=symbol_col, date_col=date_col)
    out = pd.Series(pd.NA, index=df.index, dtype="string")
    for name in reversed(by_specificity(hits.columns)):
        out = out.mask(hits[name], name)
    return out
