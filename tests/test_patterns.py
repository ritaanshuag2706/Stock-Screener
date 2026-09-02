"""Detector tests against TA-Lib's reference conditions.

Because the thresholds are rolling averages rather than fixed ratios, every
test builds a baseline of identical bars first so the averages are known
exactly, then appends the bar under test. With the baseline below:

    real body        = 2      -> BodyShort limit        = 1.0 x 2 = 2
    high-low range   = 4      -> BodyDoji limit         = 0.1 x 4 = 0.4
                                 ShadowVeryShort limit  = 0.1 x 4 = 0.4
                                 Near allowance         = 0.2 x 4 = 0.8
    ShadowLong limit = the test bar's own real body
"""

import pandas as pd
import pytest

from nse_screener import patterns
from nse_screener.patterns import registry

COLUMNS = ["open", "high", "low", "close"]

# body 2, range 4, white. Repeated so every rolling average is exact.
BASE_BAR = (100.0, 103.0, 99.0, 102.0)
BASE_N = 12
PREV_LOW = 99.0
PREV_BODY_BOTTOM = 100.0


def frame(rows, base_n=BASE_N, base_bar=BASE_BAR):
    return pd.DataFrame([base_bar] * base_n + list(rows), columns=COLUMNS)


def last(series):
    return bool(series.iloc[-1])


# --- the baseline must be inert ---------------------------------------------


def test_baseline_bars_trigger_nothing():
    hits = patterns.detect_all(frame([]))
    assert not hits.to_numpy().any()


def test_detectors_return_false_during_warm_up():
    """The rolling averages need `period` bars; before that there is no answer."""
    short = pd.DataFrame([BASE_BAR] * 3, columns=COLUMNS)
    hits = patterns.detect_all(short)
    assert not hits.to_numpy().any()


# --- doji -------------------------------------------------------------------


def test_doji_hit():
    # body 0.2 against a BodyDoji limit of 0.4
    assert last(patterns.detect("doji", frame([(100.0, 101.0, 99.0, 100.2)])))


def test_doji_near_miss_body_just_over_the_limit():
    # body 0.5 against a limit of 0.4
    assert not last(patterns.detect("doji", frame([(100.0, 101.0, 99.0, 100.5)])))


def test_doji_limit_scales_with_recent_range():
    """The same body is a doji in a quiet stock and not in a volatile one."""
    bar = (100.0, 101.0, 99.0, 100.5)          # body 0.5
    wide = (100.0, 106.0, 94.0, 102.0)         # range 12 -> limit 1.2
    assert last(patterns.detect("doji", frame([bar], base_bar=wide)))
    assert not last(patterns.detect("doji", frame([bar])))   # range 4 -> limit 0.4


def test_zero_range_bar_is_not_a_doji():
    assert not last(patterns.detect("doji", frame([(100.0, 100.0, 100.0, 100.0)])))


# --- hammer -----------------------------------------------------------------

# body 0.5, lower 4.0, upper 0.2, body_bottom 99.0 (<= 99.0 + 0.8)
HAMMER_BAR = (99.0, 99.7, 95.0, 99.5)


def test_hammer_hit():
    assert last(patterns.detect("hammer", frame([HAMMER_BAR])))


def test_hammer_near_miss_upper_shadow_too_long_under_the_talib_rule():
    # upper 0.5 against a ShadowVeryShort limit of 0.4
    assert not last(
        patterns.detect("hammer", frame([(99.0, 100.0, 95.0, 99.5)]), rule="talib")
    )


def test_classic_hammer_needs_the_body_high_in_its_own_range():
    """A body halfway down the range is not a hammer, however long the tail."""
    middle = (100.0, 104.0, 96.0, 100.4)   # body sits ~45% down from the high
    assert not last(patterns.detect("hammer", frame([middle])))


def test_classic_hammer_needs_a_long_lower_shadow():
    # body 0.4 sitting high, but the lower shadow is only 1.5x it
    stubby = (99.6, 100.0, 99.0, 100.0)
    assert not last(patterns.detect("hammer", frame([stubby])))


def test_classic_hammer_mirrors_the_inverted_hammer_rule():
    """Same shape flipped: body in the far third, opposite shadow >= 2x body."""
    h = frame([HAMMER_BAR])
    i = frame([INVERTED_BAR])
    assert last(patterns.detect("hammer", h))
    assert last(patterns.detect("inverted_hammer", i))
    assert not last(patterns.detect("hammer", i))
    assert not last(patterns.detect("inverted_hammer", h))


def test_hammer_near_miss_lower_shadow_not_longer_than_body():
    # body 1.0, lower 0.8 -> fails `lower > real body`, other conditions hold
    assert not last(patterns.detect("hammer", frame([(99.0, 100.1, 98.2, 99.8)])))


def test_hammer_near_miss_body_not_short_enough():
    # body 2.5 against a BodyShort limit of 2.0
    assert not last(patterns.detect("hammer", frame([(97.0, 99.6, 93.0, 99.5)])))


def test_near_low_is_off_by_default_so_a_lifted_hammer_qualifies():
    """The configured reading is pure geometry, like the inverted hammer."""
    lifted = (101.0, 101.7, 97.0, 101.5)   # body_bottom 101, prior low was 99
    assert last(patterns.detect("hammer", frame([lifted])))


def test_requiring_near_low_rejects_a_lifted_hammer():
    lifted = (101.0, 101.7, 97.0, 101.5)
    assert not last(
        patterns.detect("hammer", frame([lifted]), require_near_low=True)
    )


def test_near_low_allowance_is_a_boundary_when_switched_on():
    """body_bottom must be within Near (0.8) of the previous bar's low (99)."""
    at_edge = (99.8, 100.5, 95.0, 100.3)   # exactly 99.0 + 0.8
    beyond = (99.9, 100.6, 95.0, 100.4)    # 99.9 > 99.8
    assert last(patterns.detect("hammer", frame([at_edge]), require_near_low=True))
    assert not last(patterns.detect("hammer", frame([beyond]), require_near_low=True))


# --- inverted hammer --------------------------------------------------------

# body 0.5, upper 3.5, lower 0.2, body_top 98.5 (< previous body bottom 100)
INVERTED_BAR = (98.0, 102.0, 97.8, 98.5)


def test_inverted_hammer_hit():
    assert last(patterns.detect("inverted_hammer", frame([INVERTED_BAR])))


def test_gap_requirement_is_off_by_default_so_a_non_gapped_bar_qualifies():
    """The configured reading is the classic one: geometry, no gap test."""
    no_gap = (100.5, 104.5, 100.3, 101.0)   # body_top 101 > previous 100
    assert last(patterns.detect("inverted_hammer", frame([no_gap])))


def test_talib_reading_rejects_the_same_bar_on_the_gap():
    no_gap = (100.5, 104.5, 100.3, 101.0)
    assert not last(
        patterns.detect("inverted_hammer", frame([no_gap]), require_gap_down=True)
    )


def test_inverted_hammer_near_miss_lower_shadow_too_long_under_the_talib_rule():
    # lower 0.5 against a ShadowVeryShort limit of 0.4
    assert not last(
        patterns.detect(
            "inverted_hammer", frame([(98.0, 102.0, 97.5, 98.5)]), rule="talib"
        )
    )


def test_classic_rule_still_needs_the_body_low_in_its_own_range():
    """A body halfway up the range is not an inverted hammer, however long
    the wick above it."""
    middle = (100.0, 104.0, 96.0, 100.4)   # body sits ~55% up the range
    assert not last(patterns.detect("inverted_hammer", frame([middle])))


def test_classic_rule_still_needs_a_long_upper_shadow():
    # body 0.4 sitting low, but the upper shadow is only 1.5x it
    stubby = (98.0, 98.9, 97.9, 98.4)
    assert not last(patterns.detect("inverted_hammer", frame([stubby])))


def test_inverted_hammer_near_miss_upper_shadow_not_longer_than_body():
    # body 1.0, upper 0.8
    assert not last(patterns.detect("inverted_hammer", frame([(98.0, 99.8, 97.8, 99.0)])))


def test_hammer_and_inverted_hammer_are_distinct():
    """They were only separable by a trend proxy before; now the geometry and
    the gap condition tell them apart on their own."""
    h = frame([HAMMER_BAR])
    i = frame([INVERTED_BAR])
    assert last(patterns.detect("hammer", h))
    assert not last(patterns.detect("inverted_hammer", h))
    assert last(patterns.detect("inverted_hammer", i))
    assert not last(patterns.detect("hammer", i))


# --- bullish engulfing ------------------------------------------------------

BLACK_BAR = (102.0, 103.0, 99.0, 100.0)     # body 2, black


def test_bullish_engulfing_hit():
    df = frame([BLACK_BAR, (99.5, 103.0, 99.0, 102.5)])
    assert last(patterns.detect("bullish_engulfing", df))


def test_bullish_engulfing_needs_a_black_previous_bar():
    df = frame([BASE_BAR, (99.5, 103.0, 99.0, 102.5)])   # previous is white
    assert not last(patterns.detect("bullish_engulfing", df))


def test_bullish_engulfing_needs_a_white_current_bar():
    df = frame([BLACK_BAR, (102.5, 103.0, 99.0, 99.5)])
    assert not last(patterns.detect("bullish_engulfing", df))


def test_bullish_engulfing_near_miss_open_above_prior_close():
    df = frame([BLACK_BAR, (100.5, 103.0, 100.0, 102.5)])   # opens above 100
    assert not last(patterns.detect("bullish_engulfing", df))


def test_bullish_engulfing_near_miss_close_below_prior_open():
    df = frame([BLACK_BAR, (99.5, 103.0, 99.0, 101.5)])     # closes below 102
    assert not last(patterns.detect("bullish_engulfing", df))


def test_identical_bodies_do_not_engulf():
    """TA-Lib's paired inequalities let the bodies touch at one end only."""
    df = frame([BLACK_BAR, (100.0, 103.0, 99.0, 102.0)])
    assert not last(patterns.detect("bullish_engulfing", df))


def test_bullish_engulfing_may_touch_at_exactly_one_end():
    touch_low = frame([BLACK_BAR, (99.9, 103.0, 99.0, 102.0)])   # close == prev open
    assert last(patterns.detect("bullish_engulfing", touch_low))
    touch_high = frame([BLACK_BAR, (100.0, 103.0, 99.0, 102.1)])  # open == prev close
    assert last(patterns.detect("bullish_engulfing", touch_high))


def test_bullish_engulfing_ignores_trend_and_body_size():
    """TA-Lib applies no trend or minimum-size test; context is Stage 6's job."""
    tiny_prev = (100.02, 100.5, 99.5, 100.0)   # a near-doji black bar
    df = frame([tiny_prev, (99.9, 103.0, 99.0, 100.1)])
    assert last(patterns.detect("bullish_engulfing", df))


# --- registry ---------------------------------------------------------------


def test_the_candlestick_family_is_registered():
    """Asserts this family, not the global list -- other families are added
    alongside it and must not break these tests."""
    assert set(CANDLESTICKS) <= set(registry.names())


def test_shooting_star_is_gone():
    with pytest.raises(KeyError, match="unknown pattern"):
        registry.get("shooting_star")


def test_config_has_an_entry_for_every_pattern():
    """bullish_engulfing is legitimately empty -- TA-Lib gives it no settings."""
    config = registry.load_params()
    for name in registry.names():
        assert name in config, f"{name} missing from patterns.yaml"


def test_parameterised_patterns_have_settings():
    config = registry.load_params()
    for name in ("doji", "hammer", "inverted_hammer"):
        assert config[name], f"{name} has no settings"


def test_unknown_pattern_name_raises():
    with pytest.raises(KeyError, match="unknown pattern"):
        registry.get("marubozu")


def test_detect_all_returns_a_boolean_column_per_pattern():
    hits = patterns.detect_all(frame([HAMMER_BAR]), CANDLESTICKS)
    assert list(hits.columns) == CANDLESTICKS
    assert all(hits[c].dtype == bool for c in hits.columns)


def test_detectors_accept_uppercase_bhavcopy_headers():
    df = frame([HAMMER_BAR]).rename(columns=str.upper)
    assert last(patterns.detect("hammer", df))


def test_missing_column_raises_a_clear_error():
    df = frame([HAMMER_BAR]).drop(columns=["low"])
    with pytest.raises(ValueError, match="missing OHLC column"):
        patterns.detect("hammer", df)


def test_detectors_do_not_mutate_the_input():
    df = frame([HAMMER_BAR])
    before = df.copy()
    patterns.detect_all(df)
    pd.testing.assert_frame_equal(df, before)


def test_overrides_take_precedence_over_config():
    assert registry.params_for("doji", body_doji_factor=0.9)["body_doji_factor"] == 0.9


def test_loosening_a_factor_admits_a_near_miss():
    """The config is the only place a threshold lives."""
    bar = (100.0, 101.0, 99.0, 100.5)   # body 0.5, limit 0.4
    assert not last(patterns.detect("doji", frame([bar])))
    assert last(patterns.detect("doji", frame([bar]), body_doji_factor=0.2))


# --- classification: one label per bar --------------------------------------


def test_specificity_order_within_the_candlestick_family():
    """Narrower definition wins. doji is one condition, so it loses to all."""
    order = [n for n in registry.by_specificity() if n in CANDLESTICKS]
    assert order == ["inverted_hammer", "hammer", "bullish_engulfing", "doji"]


def test_a_bar_that_is_both_classifies_as_inverted_hammer():
    """Body 0.2 (a doji) with a long upper shadow and a gap down."""
    bar = (98.2, 102.0, 98.0, 98.0)
    df = frame([bar])
    hits = patterns.detect_all(df)
    assert last(hits["doji"]) and last(hits["inverted_hammer"])   # both fire
    assert patterns.classify(df).iloc[-1] == "inverted_hammer"    # one label


def test_classify_leaves_the_boolean_columns_untouched():
    """Stage 7 measures each pattern independently; classify is display only."""
    df = frame([(98.2, 102.0, 98.0, 98.0)])
    assert bool(patterns.detect_all(df)["doji"].iloc[-1])


def test_classify_falls_back_when_the_winner_is_excluded():
    df = frame([(98.2, 102.0, 98.0, 98.0)])
    assert patterns.classify(df).iloc[-1] == "inverted_hammer"
    assert patterns.classify(df, ["doji", "hammer"]).iloc[-1] == "doji"


def test_classify_is_na_where_nothing_fires():
    out = patterns.classify(frame([]))
    assert out.isna().all()


def test_classify_by_symbol_matches_per_symbol_classification():
    a = frame([(98.2, 102.0, 98.0, 98.0)]).assign(symbol="AAA")
    b = frame([HAMMER_BAR]).assign(symbol="BBB")
    both = pd.concat([a, b], ignore_index=True)
    both["date"] = pd.date_range("2024-01-01", periods=len(both), freq="D")
    out = patterns.classify_by_symbol(both)
    assert out[both["symbol"] == "AAA"].iloc[-1] == "inverted_hammer"
    assert out[both["symbol"] == "BBB"].iloc[-1] == "hammer"


# --- the gap-down switch ----------------------------------------------------

# RELIANCE 2026-07-15: body 1.40, upper 15.40, lower 0.00 -- a textbook
# inverted hammer that TA-Lib rejects solely because it gapped up.
REAL_GAPPED_UP_BAR = (1294.10, 1310.90, 1294.10, 1295.50)


def test_gap_up_bar_is_admitted_under_the_configured_classic_reading():
    """RELIANCE 2026-07-15. Rejected by TA-Lib on the gap alone."""
    prev = (1290.0, 1296.0, 1288.0, 1293.0)     # body bottom 1290 < this body top
    df = frame([prev, REAL_GAPPED_UP_BAR], base_bar=(1290.0, 1300.0, 1280.0, 1296.0))
    assert last(patterns.detect("inverted_hammer", df))


def test_requiring_the_gap_again_rejects_it():
    """TA-Lib's stricter reading is still one keyword away."""
    prev = (1290.0, 1296.0, 1288.0, 1293.0)
    df = frame([prev, REAL_GAPPED_UP_BAR], base_bar=(1290.0, 1300.0, 1280.0, 1296.0))
    assert not last(patterns.detect("inverted_hammer", df, require_gap_down=True))


def test_classic_rule_accepts_the_second_real_bar():
    """RELIANCE 2026-07-16: body 1.10, upper 12.80, lower 3.70 on a 17.60 range.

    TA-Lib rejects it because 3.70 exceeds 10% of the *average* range. The
    classic rule asks where the body sits in this bar's own range -- 21% to 27%
    up from the low, comfortably inside the lower third -- and accepts it.
    """
    prev = (1290.0, 1296.0, 1288.0, 1293.0)
    long_lower = (1295.5, 1309.4, 1291.8, 1296.6)
    df = frame([prev, long_lower], base_bar=(1290.0, 1300.0, 1280.0, 1296.0))
    assert last(patterns.detect("inverted_hammer", df))
    assert not last(patterns.detect("inverted_hammer", df, rule="talib"))


def test_configured_rules_are_classic_for_both_shadow_patterns():
    """Pinned so the divergence from TA-Lib stays deliberate and visible."""
    inv = registry.load_params()["inverted_hammer"]
    assert inv["rule"] == "classic"
    assert inv["require_gap_down"] is False
    ham = registry.load_params()["hammer"]
    assert ham["rule"] == "classic"
    assert ham["require_near_low"] is False


def test_unknown_hammer_rule_raises():
    with pytest.raises(ValueError, match="rule must be"):
        patterns.detect("hammer", frame([]), rule="nonsense")


def test_unknown_rule_raises():
    with pytest.raises(ValueError, match="rule must be"):
        patterns.detect("inverted_hammer", frame([]), rule="nonsense")


# --- display labels ---------------------------------------------------------


CANDLESTICKS = ["bullish_engulfing", "doji", "hammer", "inverted_hammer"]


def test_labels_are_sentence_case():
    labels = registry.labels()
    assert {k: labels[k] for k in CANDLESTICKS} == {
        "bullish_engulfing": "Bullish engulfing",
        "doji": "Doji",
        "hammer": "Hammer",
        "inverted_hammer": "Inverted hammer",
    }


def test_label_is_derived_so_a_new_pattern_gets_one_free():
    assert registry.get("inverted_hammer").label == "Inverted hammer"


def test_internal_names_stay_snake_case():
    """The name is an identifier: a patterns.yaml key, a CLI argument, a column
    in the hits table. Only the display form is prettified."""
    for n in registry.names():
        assert n == n.lower()
        assert " " not in n
    assert set(registry.load_params()) >= set(registry.names())


def test_classify_returns_identifiers_not_labels():
    """Downstream code joins on the stable key, never the display string."""
    out = patterns.classify(frame([HAMMER_BAR]))
    assert out.iloc[-1] == "hammer"
