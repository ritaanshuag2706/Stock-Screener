"""Pins the detectors to TA-Lib's reference implementation.

The detectors were transcribed by hand from TA-Lib's C sources. This checks the
transcription against the real thing on randomly generated candles, so a subtle
drift in a threshold or an inequality shows up here rather than as a quietly
wrong hit rate in Stage 7.

Skipped when TA-Lib is not installed -- it is a dev dependency, not a runtime
one, and nothing in src/ imports it.
"""

import numpy as np
import pandas as pd
import pytest

from nse_screener import patterns

talib = pytest.importorskip("talib", reason="TA-Lib not installed")

# TA-Lib zero-fills this many leading bars, regardless of whether the pattern
# could have been detected there. Comparisons start after the longest one.
LOOKBACK = {
    "doji": 10,
    "hammer": 11,
    "inverted_hammer": 11,
    "bullish_engulfing": 2,
}
REFERENCE = {
    "doji": "CDLDOJI",
    "hammer": "CDLHAMMER",
    "inverted_hammer": "CDLINVERTEDHAMMER",
    "bullish_engulfing": "CDLENGULFING",
}

# Settings forced back to TA-Lib's own where config/patterns.yaml deliberately
# diverges. Parity here means "the transcription is faithful", which has to be
# testable independently of which reading the project has chosen to run with.
TALIB_SETTINGS = {
    "inverted_hammer": {"rule": "talib", "require_gap_down": True},
    "hammer": {"rule": "talib", "require_near_low": True},
}


def random_candles(n=4000, seed=0):
    """Candles with enough variety in body and shadow to trigger everything."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.012, n))
    # Deliberately heavy-tailed shadows so hammers and inverted hammers appear.
    up = np.abs(rng.normal(0, 0.012, n)) * rng.choice([0.1, 1.0, 3.0], n)
    dn = np.abs(rng.normal(0, 0.012, n)) * rng.choice([0.1, 1.0, 3.0], n)
    high = np.maximum(open_, close) * (1 + up)
    low = np.minimum(open_, close) * (1 - dn)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close})


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_matches_talib(name):
    df = random_candles()
    o, h, low, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))

    reference = getattr(talib, REFERENCE[name])(o, h, low, c) > 0
    mine = patterns.detect(name, df, **TALIB_SETTINGS.get(name, {})).to_numpy()

    start = LOOKBACK[name]
    disagree = np.flatnonzero(mine[start:] != reference[start:]) + start
    assert not len(disagree), (
        f"{name}: {len(disagree)} disagreements, first at bar {disagree[:5]}"
    )


@pytest.mark.parametrize("name", sorted(REFERENCE))
def test_the_reference_actually_fires(name):
    """A parity test against two all-False arrays would pass and prove nothing."""
    df = random_candles()
    o, h, low, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    assert (getattr(talib, REFERENCE[name])(o, h, low, c) > 0).sum() > 5


def test_engulfing_is_detected_one_bar_earlier_than_talib():
    """A known, understood difference rather than a bug.

    CDLENGULFING declares a lookback of 2, so TA-Lib zero-fills bar 1 even
    though the pattern only needs one prior bar. Ours reports it.
    """
    df = pd.DataFrame(
        [(102.0, 103.0, 99.0, 100.0),    # black
         (99.5, 103.0, 99.0, 102.5)],    # white, engulfing
        columns=["open", "high", "low", "close"],
    )
    o, h, low, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    assert talib.CDLENGULFING(o, h, low, c)[1] == 0
    assert bool(patterns.detect("bullish_engulfing", df).iloc[1])
