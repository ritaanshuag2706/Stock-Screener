"""Pattern detectors.

Importing this package registers every detector, so `registry.names()` is
populated by the time anyone asks.
"""

from . import (  # noqa: F401  (registration)
    chart,
    divergence,
    double,
    fib,
    heikin_ashi,
    momentum,
    single,
)
from .registry import (
    Entry,
    by_specificity,
    classify,
    classify_by_symbol,
    detect,
    detect_all,
    detect_by_symbol,
    get,
    label,
    labels,
    load_params,
    names,
    params_for,
)

__all__ = [
    "Entry",
    "by_specificity",
    "classify",
    "classify_by_symbol",
    "detect",
    "detect_all",
    "detect_by_symbol",
    "get",
    "label",
    "labels",
    "load_params",
    "names",
    "params_for",
]
