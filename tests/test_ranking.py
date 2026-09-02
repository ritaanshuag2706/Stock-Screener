"""Ranking decides most of what gets traded, so its invariants are load-bearing.

With ~480 signals a night and 10 slots, the ranker -- not the detector -- chooses
the majority of the book. A bug here would look like a signal result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from nse_screener import ranking


def candidates(n=6, **cols) -> pd.DataFrame:
    base = {
        "symbol": [f"S{i}" for i in range(n)],
        "pattern": ["doji"] * n,
        "close": [100.0] * n,
        "volume": [1_000_000] * n,
        "rel_volume": [1.0] * n,
        "dist_200ema_pct": [5.0] * n,
        "atr_pct": [2.0] * n,
    }
    base.update(cols)
    return pd.DataFrame(base)


# --- the registry -----------------------------------------------------------


def test_every_registered_ranker_runs_and_returns_one_score_per_row():
    c = candidates()
    for name in ranking.names():
        s = ranking.score(c, name, rng=np.random.default_rng(0))
        assert len(s) == len(c), name
        assert s.index.equals(c.index), name
        assert s.dtype == float, name


def test_unknown_ranker_names_the_registered_ones():
    with pytest.raises(KeyError, match="random"):
        ranking.get("no_such_ranker")


def test_random_is_registered():
    """The control has to be a real, runnable arm. "Better than random" is the
    only useful form of a ranking claim, and it needs random to exist."""
    assert "random" in ranking.names()


# --- the ordering contract --------------------------------------------------


def test_higher_score_ranks_first():
    c = candidates(n=3, rel_volume=[1.0, 9.0, 5.0])
    out = ranking.rank(c, "rel_volume")
    assert list(out["symbol"]) == ["S1", "S2", "S0"]


def test_low_volatility_inverts_so_calmest_wins():
    """The contract is higher-is-better, so a ranker whose metric is better when
    small has to negate. Getting this backwards would silently rank the wildest
    names first and still look like a working ranker."""
    c = candidates(n=3, atr_pct=[5.0, 1.0, 3.0])
    assert list(ranking.rank(c, "low_volatility")["symbol"]) == ["S1", "S2", "S0"]


def test_ties_break_by_symbol_not_by_input_order():
    """Otherwise the arbitrariness this module removes just moves inside the tie."""
    c = candidates(n=4, symbol=["ZZZ", "AAA", "MMM", "BBB"], rel_volume=[1.0] * 4)
    out = ranking.rank(c, "rel_volume")
    assert list(out["symbol"]) == ["AAA", "BBB", "MMM", "ZZZ"]

    shuffled = c.iloc[::-1].reset_index(drop=True)
    assert list(ranking.rank(shuffled, "rel_volume")["symbol"]) == list(out["symbol"])


def test_nan_scores_sort_last_rather_than_raising():
    """A symbol missing a context column is a worse candidate, not a reason to
    abandon the night -- but it must never outrank a symbol that has one."""
    c = candidates(n=3, rel_volume=[1.0, np.nan, 5.0])
    out = ranking.rank(c, "rel_volume")
    assert list(out["symbol"]) == ["S2", "S0", "S1"]


def test_ranking_is_reproducible_for_a_given_seed():
    c = candidates(n=20)
    a = ranking.rank(c, "random", rng=np.random.default_rng(7))
    b = ranking.rank(c, "random", rng=np.random.default_rng(7))
    assert list(a["symbol"]) == list(b["symbol"])


def test_different_seeds_give_different_random_orders():
    """If they did not, `random` would be a fixed order wearing a disguise and
    every 'beats random' comparison would be against one arbitrary draw."""
    c = candidates(n=30)
    a = ranking.rank(c, "random", rng=np.random.default_rng(1))
    b = ranking.rank(c, "random", rng=np.random.default_rng(2))
    assert list(a["symbol"]) != list(b["symbol"])


def test_rank_does_not_mutate_the_input():
    c = candidates()
    before = c.copy()
    ranking.rank(c, "rel_volume")
    pd.testing.assert_frame_equal(c, before)


def test_an_empty_frame_still_carries_the_score_column():
    """Same discipline as HIT_COLUMNS: callers should never special-case empty."""
    out = ranking.rank(candidates().iloc[:0], "rel_volume")
    assert "rank_score" in out.columns
    assert out.empty


# --- the guardrails ---------------------------------------------------------


def test_a_missing_column_names_the_ranker_and_the_column():
    """The failure mode this prevents is worse than a crash: an all-NaN score
    column ranks purely on the tie-break and looks like it worked."""
    c = candidates().drop(columns=["rel_volume"])
    with pytest.raises(KeyError, match="rel_volume"):
        ranking.rank(c, "rel_volume")


def test_ranking_needs_a_symbol_column_to_break_ties():
    with pytest.raises(KeyError, match="symbol"):
        ranking.rank(candidates().drop(columns=["symbol"]), "rel_volume")


def test_no_ranker_reads_a_forward_looking_column():
    """The invariant that matters most.

    A ranker scores bar `t` and the fill happens at `t+1` open. If a ranker ever
    declared a need for an outcome column, it would manufacture an edge far
    larger than anything the detectors were measured at -- and it would present
    as a discovery rather than as a bug. This pins the declared needs against the
    names `study.base_rates` uses for forward-looking values.
    """
    forward = {
        "outcome", "hit", "r_multiple", "forward_return", "exit_price",
        "exit_reason", "exit_date", "net_pnl", "gross_pnl", "target", "stop",
    }
    for name in ranking.names():
        leaked = forward.intersection(ranking.get(name).needs)
        assert not leaked, f"ranker {name!r} reads forward data: {leaked}"


def test_top_limits_and_zero_means_no_limit():
    c = candidates(n=10, rel_volume=list(range(10)))
    assert len(ranking.top(c, 3, "rel_volume")) == 3
    assert len(ranking.top(c, 0, "rel_volume")) == 10
    assert list(ranking.top(c, 2, "rel_volume")["symbol"]) == ["S9", "S8"]


def test_specificity_ranker_agrees_with_the_pattern_registry():
    """It ranks by the registry's own specificity, so if that ordering is wrong
    this ranker is wrong in exactly the same way -- which is the point: it makes
    `specificity` a testable claim rather than a tidy-looking number."""
    from nse_screener import patterns as pat

    c = candidates(n=2, pattern=["doji", "squeeze_release"])
    out = ranking.rank(c, "signal_specificity")
    assert list(out["pattern"]) == ["squeeze_release", "doji"]
    assert pat.get("squeeze_release").specificity > pat.get("doji").specificity
