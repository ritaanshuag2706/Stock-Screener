# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything runs through the project venv — there is no activated-shell assumption.

```bash
.venv/bin/pytest                                  # full suite (~1.5s)
.venv/bin/pytest tests/test_patterns.py           # one file
.venv/bin/pytest -k "inverted_hammer"             # one pattern's tests
.venv/bin/ruff check src tests                    # lint
.venv/bin/ruff check src tests --fix

.venv/bin/python -m nse_screener.apps.backfill --start 2026-07-01   # top up the store
.venv/bin/python -m nse_screener.apps.scan --days 3                 # tonight's hits
.venv/bin/python -m nse_screener.apps.show RELIANCE --sample 5      # bars for one symbol
.venv/bin/streamlit run src/nse_screener/apps/dashboard.py          # the two-page app
```

Python is Homebrew 3.13 (`/opt/homebrew/bin/python3.13`); the system 3.9 is EOL and
unusable here. `pytest` writes temp dirs to `.scratch/pytest` and pip caches to
`.scratch/pip` — both configured so nothing lands outside the project.

**Streamlit caches imported modules for the life of the process.** Editing anything
under `src/nse_screener/` needs a server restart, not just a browser reload. Only
edits to the page files themselves hot-reload.

## Architecture

Data flows one way, and each layer is independently testable:

```
NSE archive → bhavcopy.py → store.py → patterns/ → screener.py → apps/
              download +     parquet    detectors   eligibility   CLI + Streamlit
              parse          by year                + window
```

**`data/bhavcopy.py`** handles two incompatible NSE layouts — legacy through
2024-07-05, UDiFF from 2024-07-08 — plus a `sec_bhavdata_full` fallback for weekend
special sessions where NSE publishes that file but *not* the bhavcopy zip. Raw
downloads are kept forever under `data/raw/`, so a parsing fix costs a re-parse
rather than a re-download.

**`data/store.py`** is year-partitioned parquet, deduped on `(date, symbol)` with
last-write-wins. That is what makes the backfill safe to re-run and interrupt. It
pins price columns to float64 on write — a writer handing over whole-number prices
would otherwise create an int64 partition that disagrees with every other year when
`read()` concatenates them.

**`market_calendar.py`** wraps `pandas_market_calendars` with a YAML override file
that always wins. `reconcile()` compares the calendar against sessions actually
present in the data; disagreements mean either the calendar is wrong *or a download
was missed*. That check is how six real trading sessions were found.

**`patterns/`** — each detector is a pure `DataFrame → boolean Series`. The registry
maps names to functions and injects thresholds from `config/patterns.yaml`.

**`screener.py`** splits two jobs that fail for different reasons: eligibility (which
symbols are worth looking at) and detection (which of those printed a pattern).

**`apps/`** — `dashboard.py` is only `st.navigation`; the pages live in `apps/views/`.

## Invariants that are easy to break

**Never call `detect_all()` on a multi-symbol frame.** `store.read()` sorts by
`(date, symbol)`, so consecutive rows are *different companies*; detectors use
`.shift()` for the previous bar, and shifting across that boundary fabricates
patterns. Measured: 7 phantom hits from 10 bars on 2 symbols. `detect_all()` raises
on a multi-symbol frame; use `detect_by_symbol()`.

**`detect_by_symbol()` is vectorised, and must stay that way.** Looping symbols and
calling `detect_all()` on each costs ~6ms per symbol *regardless of bar count* —
per-call pandas overhead dominates a short series, so 2,500 symbols took 15 seconds.
One grouped pass with the `by=` key does it in 0.6s. `by` keeps every rolling window
and shift inside its own symbol; `tests/test_seam.py` pins the fast path to the
loop's answers.

**Thresholds live in `config/patterns.yaml`, never in code.** Detector defaults are
fallbacks for a missing key, not the operative values.

**`ScanResult.hits` always carries `HIT_COLUMNS`, even when empty**, so callers never
special-case "no hits" before selecting columns.

**Trading dates are IST dates.** Use `clock.ist_today()`, never `date.today()` — the
machine clock is a different calendar day from the exchange for part of every day.

## Decisions that look wrong without the reasoning

**Detectors follow TA-Lib, except where they deliberately don't.** All four were
transcribed from TA-Lib's C sources and verified against the real library on 24,495
bars with zero disagreements. `hammer` and `inverted_hammer` then moved to the
*classic* reading (`rule: classic`), because TA-Lib measures "little or no shadow"
against a fraction of the **average** range, which rejects bars whose body is plainly
at one end when that bar is wider than its neighbours. The classic rule asks where
the body sits within its **own** range. `tests/test_talib_parity.py` forces both back
to `rule: "talib"` so parity still means "the transcription is faithful",
independently of which reading the project runs.

Frequencies measured on 300 liquid symbols since 2024, if you are tempted to change
a threshold: inverted hammer 5.53% (TA-Lib) vs 4.78% (classic); hammer 1.83% vs
4.35%. **Loosening TA-Lib's shadow factor to admit a real missed bar would have taken
inverted hammer to 14% of all bars** — the classic rule accepts the same bar while
firing *less* often. Do not tune thresholds to make output look nicer.

**Patterns overlap, and that is correct.** A small-bodied bar with a long upper
shadow is both a doji and an inverted hammer; TA-Lib reports both. `detect_all()`
keeps both booleans — Stage 7 needs them measured independently — while `classify()`
picks one winner by `specificity` for display. Never collapse the overlap in the
detectors.

**The four patterns do not get four colours.** Every candidate four-hue set fails the
all-pairs colourblind gate on the app's dark surface (measured: worst OKLab ΔE
1.5–6.5 against a target of 8). So colour carries *direction* and shape carries
*identity*. The marker shapes are therefore load-bearing, not decorative — two
similar shapes is a real defect. See `apps/views/_style.py`.

**Prices are unadjusted for splits and bonuses.** 59 corporate actions are visible as
`prev_close` breaks across 2020–2026. The screener's `max_overnight_gap` rule skips a
symbol while a split is still inside the detectors' 11-bar reach, then forgives it.
This is a crude guard. Stage 7's `forward_outcomes` additionally drops any signal
whose forward window spans a gap that large, since a return measured across a split
is an artefact. Proper adjustment stays unbuilt; nothing in the base rates
implicated it.

**Turnover was removed from the screener** at the owner's request. The raw column
still exists in the store (it is NSE data, not logic) but nothing filters or displays
on it, and a test asserts it cannot creep back in.

**`Universe.min_traded_value` is not that rule sneaking back.** What was removed was
turnover as a metric shown and ranked on. What exists now is a floor below which a
simulated fill is not credible: `min_price` Rs 20 and `min_traded_value` Rs 1 crore
of median daily close x volume. Nothing displays it, and
`test_no_turnover_anywhere_in_the_rules` still asserts that.

The floor is there because its absence was measured. A breakout backtest run without
it returned **+731%**, of which sub-Rs 20 names supplied **68.8% of the P&L on 15.2%
of the trades** -- IMPEXFERRO at Rs 1.75, SPCENET at Rs 2.90, posting R multiples of
26 to 49 on fills nobody could have got. On a Rs 2 stock an ATR-sized stop is a
couple of ticks, so the R denominator collapses and the arithmetic manufactures an
edge. With the floor on, that strategy returns -82% -- **and so does its baseline**.
The floor is doing the work, not the signal.

Live effect on 2026-07-29: 1,390 eligible, 1,605 excluded as illiquid. It is *off*
inside `study/base_rates.py`, deliberately and with a comment saying so -- the Stage
7 numbers below were measured without it and have to stay reproducible, and a base
*rate* wants the widest sample. Turn it on there and every Stage 7 figure changes.

## Stage 7 answered its question, and the answer was no

The whole project pointed at one measurement. It has been made, and it should not
be quietly re-litigated by a future session that finds a promising-looking number.

Across 2.9M bars, 2,612 symbols, 2020-2026, with a +2 ATR target, -1 ATR stop and
a 10-session horizon, entry at the next open:

    pattern                  n   hit_rate   lift vs all bars
    doji               254,023     25.95%   +0.25
    inverted_hammer    197,552     25.66%   -0.05
    hammer             123,035     25.81%   +0.11
    bullish_engulfing   92,682     25.36%   -0.35
    all bars         2,876,531     25.71%       -

Every pattern sits within 0.35 percentage points of a randomly chosen bar. Average
forward return was 0.4-0.8% over ten sessions against round-trip costs of
0.25-0.5%.

**An out-of-sample split confirmed it.** Hypotheses were picked on 2020-2023 at a
strict |z|>3 and tested once on 2024-2026. Of 21 positive candidates, **3 held
up**, none above +1pp. Doji's headline +0.64pp (z=5.13) came back *negative*. The
best single finding from the full-period sweep -- hammer on >3x volume, +2.49pp,
z=3.10 -- came back at +0.53pp, z=0.47. It was noise, exactly as the multiple-
comparisons arithmetic predicted (65 of 131 comparisons cleared |z|>2 against 7
expected by chance).

Note the control itself moved: 27.56% in 2020-2023, 23.31% in 2024-2026. Much of
what looked like pattern edge was the market being kinder in the first period.
This is why a rate without its control is unreadable.

**What did replicate is negative**, at near-identical magnitudes across both
periods:

    doji, oversold RSI            -5.33  ->  -5.53   z=-9.6
    doji, below the lower band    -4.66  ->  -4.86   z=-6.1
    doji, >20% under the 200 EMA  -1.54  ->  -1.92   z=-5.2

These reversal patterns appearing in already-beaten-down conditions do measurably
worse than a random bar in the same conditions. That is the most robust finding in
the study, and it is a reason to exclude setups rather than take them.

Consequences worth holding on to:

* **Stage 8 as originally conceived has nothing to build on.** Filters that
  improve a pattern's hit rate require a hit rate to improve.
* Do not re-run the sweep and act on the best-looking cell. That is precisely
  what the split has already shown to fail.
* If a future session wants to continue, the live options are a different signal
  family, the negative filters above, or accepting that the edge is not at entry
  and moving to sizing and risk.

Reproduce with `python -m nse_screener.study.base_rates --start 2020-01-01`.
`--target`, `--stop` and `--horizon` change the outcome definition; 2:1 over ten
sessions is one choice among many and has not been varied much.

## Stage 9 put the same answer in rupees

The backtest exists and works; what it measures is a losing strategy. Traded with
Rs 10L, 10 concurrent positions, 1% risk per trade, 3:1 exits over 10 sessions and
real Indian delivery costs, 2020-2026:

    strategy           trades  win%   avg_r    return    CAGR    maxDD
    all four            2,189  32.1   -0.17    -78.5%  -20.9%   -88.5%
    hammer              2,572  31.7   -0.16    -77.9%  -20.5%   -85.6%
    inverted_hammer     1,386  32.2   -0.12    -73.8%  -18.5%   -83.3%
    doji                2,265  31.4   -0.19    -86.4%  -26.1%   -91.1%
    bullish_engulfing   1,941  31.3   -0.24    -71.3%  -17.3%   -86.5%

The decomposition is the useful part:

    gross P&L per trade   -Rs 118    <- negative BEFORE any cost
    cost per trade         Rs 200
    net per trade         -Rs 319
    breakeven target rate at 3:1   25.0%
    actual                          14.3%

So this is not a viable strategy ruined by costs. It has negative expectancy
before costs, and costs roughly triple the loss. Consistent with Stage 7 rather
than new information.

**The engine passes the plan's own verification.** `test_a_rule_with_no_edge_does_
not_show_one` buys every Monday on a random walk and must not print an edge. Run
against real data the same rule returns -25.4% with avg_r -0.07 -- its gross P&L
is *positive* (market drift) and costs make it negative. If that test ever starts
showing an edge, there is lookahead in the engine and every other number is void.

Two structural findings independent of whether the patterns work:

* **`skipped_no_slot` was 328,843.** With ~480 signals a night and 10 slots, the
  book is full almost always and *which* signals got traded was arbitrary. The
  strategy has no selection mechanism. Any future signal family needs a ranking
  rule before a backtest of it means anything.
* Costs are modelled properly in `backtest/costs.py` (STT, exchange, SEBI, stamp
  duty, GST on the taxable subset, slippage) rather than as one hand-waved
  percentage -- 0.322% round trip on Rs 1 lakh. The gap between "flat" and "bleeds
  steadily" lives entirely in that file.

`backtest/engine.py` is a plain day-by-day loop on purpose. It is not vectorised
and should not be: the value of this stage is being readable enough to believe.

## A second signal family was tried, and it fails differently

`patterns/momentum.py` -- breakouts and pullbacks rather than bar shapes. Added
after the candlesticks measured flat, to answer "is it these patterns, or is it
the whole approach". It went through Stages 6, 7 and 9 with **no changes to any
of them**, which is the architecture working as intended.

Unlike the candlesticks, the lift replicated out of sample and grew:

    signal            in-sample 2020-23    held out 2024-26
    breakout_20         +2.00pp  z=9.2       +2.51pp  z=9.5
    breakout_252        +0.51pp  z=1.5       +3.31pp  z=8.2
    squeeze_release     +1.63pp  z=4.7       +3.17pp  z=7.4
    rsi2_pullback       -1.03pp  z=-6.9      +1.15pp  z=6.3   <- sign flip

**And every variant still loses money.** Backtested at 2:1, 3:1 and 1:1, returns
run -64% to -92% with avg_r between -0.09 and -0.15.

The reason is the important part, and it generalises:

    signal            target%   stop%   timeout%
    breakout_20         28.1%   63.5%      8.4%
    all bars            25.7%   56.4%     17.9%

**Breakouts convert timeouts into resolutions, in both directions.** breakout_20
gains +2.4pp of targets and takes +10.0pp more stops. Expectancy at 2:1 is
-0.072R against a control of -0.050R -- a real, replicating hit-rate lift that is
*worse than doing nothing*.

So: never judge a signal on hit rate alone, even a hit rate that survives an
out-of-sample split. A screener reporting "breakout_20 wins 28% vs 26% baseline"
would look like a discovery. Read the full outcome distribution and the
expectancy, which `rates()` returns and `with_control()` puts beside the control.

Two bugs this exercise caught, both worth knowing about:

* **`squeeze_release` fired 28,679 times and was labelled zero times.** It is
  strictly narrower than `breakout_20` -- a breakout *plus* a compression test --
  but was registered with lower `specificity`, so `classify()` gave every hit to
  the broader signal. **When adding a signal that is another signal plus a
  condition, it must rank above it.** `tests/test_momentum.py` pins the ordering.
* Tests that asserted the exact global pattern list broke when the family was
  added. They are now scoped to `CANDLESTICKS` in `tests/test_patterns.py`. More
  families are the point of the architecture, so no test should assume the set.

## Heikin-Ashi was tried as the exit, and it changes the shape but not the answer

`patterns/heikin_ashi.py`. The premise was reasonable and different from everything
before it: the fixed ATR target had been the exit in every test so far, so maybe the
edge was being cut off by the clock rather than absent. HA smooths bars into runs,
which makes "stay in while the colour holds" a natural trailing exit.

`ha_flip_up` is registered as an entry; `flip_down` is *not* registered, because it
is an exit condition and every registered signal in this project is an entry. The
backtest takes it through `exit_signal_col`, and `ExitRules.use_target=False` turns
the fixed target off so HA alone decides when to leave.

breakout_20 entries, four exit policies, same bars, liquidity floor on every arm:

    policy                    trades   return    CAGR   maxDD  win%   avg_r
    target 3:1 (baseline)      2,117   -82.7%  -23.4%  -86.5%  27.9  -0.159
    HA flip exit, no target    2,221   -81.9%  -22.9%  -89.1%  27.8  -0.148
    HA entry + HA exit         2,118   -86.2%  -26.0%  -89.2%  27.4  -0.164
    HA exit + 3:1 target       2,564   -88.3%  -27.9%  -90.7%  29.3  -0.140

**HA does exactly what it promises and it does not help.** Timeouts fall from 286 to
107, replaced by 902 signal exits -- trades now end on a trend break instead of on
the clock, which is the whole point of the technique. The exit distribution changes
and the outcome does not. Consistent with Stage 9 rather than new information.

The useful lesson is methodological: **the exit was not what was wrong.** Three
exit policies spanning fixed-target, pure-trailing and both together land within
6 percentage points of each other on a strategy losing 82%. That is what a missing
entry edge looks like, and no exit rule repairs it.

Two bugs this caught:

* **The unfiltered run said +731%**, and it was penny stocks -- see the liquidity
  floor note above. This is the *second* time this exact artefact produced a
  spectacular false positive (the first was +418%, from `bars_held` counting only
  days a symbol traded). **Any result in this project that looks spectacular should
  be assumed to be an illiquidity artefact until the P&L concentration is checked.**
  `t.nlargest(10, "net_pnl")["net_pnl"].sum() / t["net_pnl"].sum()` takes a second.
* **`unrealised_at_end` marked open positions gross of their entry cost**, which
  equity had already been debited. Rs 1,184 short on Rs 2.13M -- small enough to
  hide behind a loose tolerance, and wrong. `reconciles()` is only evidence if it
  holds exactly.

`ha_flip_up` is registered at `specificity=15`, near the bottom. It fires on ~10% of
all bars, making it the *broadest* signal in the registry, and it was first written
at 70 where it outranked everything and quietly became the label on most flagged
bars. **Specificity ranks how narrow a signal is, and a smoothed-trend flag is not
narrow** -- the number is not a measure of how interesting the signal seems.

## The swing family reads pivots, and pivots are where lookahead hides

`patterns/_swings.py`, `patterns/fib.py`, `patterns/divergence.py`. Added because
the owner's decision sheets have a "Fib Retrace" row and a "Divergence" row, and
neither existed here.

**The pivot is the whole problem.** A swing high at bar `p` is "the highest of
the `left` bars before and the `right` bars after", and those right-hand bars
have not printed yet. Every charting package's divergence indicator repaints for
this reason: it draws the pivot back at `p` the moment `p + right` arrives, so
the line appears to have been there all along. Backtest that and the result is
not wrong by a little.

So nothing in `_swings.py` is placed at the pivot bar. Everything is placed at
the **confirmation** bar -- the first bar on which a live trader could have known
the pivot existed. There are no negative shifts in the file. The consequence is
deliberate: the most recent usable pivot is always at least `right` bars old, and
these detectors are late by construction. That lateness is the price of the
signal being real, not a lag to tune away.

`tests/test_swings.py`, `test_fib.py` and `test_divergence.py` each carry a
**prefix-invariance** test: the answer at bar `t` computed from the first `t+1`
bars must equal the answer at bar `t` computed from the whole history. That is
what "no lookahead" actually means, and it catches the property however the
arithmetic is later rearranged. If you touch this family, keep those three.

Details worth knowing:

* **Pivots are strict on the left, non-strict on the right.** With `>=` on both
  sides every bar of a flat series is simultaneously a pivot high and a pivot
  low -- which is how a swing detector ends up firing on a suspended stock.
* **`_swings._rolling` reindexes; the other rolling helpers in this project do
  not.** `groupby().rolling()` returns rows grouped rather than in input order,
  and pandas then aligns on labels -- so on a frame sorted by (date, symbol) the
  arithmetic silently compares one symbol's window to another symbol's bar.
  Everything else is called from `detect_by_symbol`, which sorts by (symbol,
  date) first and never notices. These are also called directly.
* **A divergence is an event, not a state.** It fires only on the bar confirming
  the second pivot. Held True until the next pivot it would fire on thirty
  consecutive bars, turning one observation into thirty and telling Stage 7
  something false about how often the thing happens.
* **The bearish readings are exported and deliberately not registered**, exactly
  like `heikin_ashi.flip_down`. Every registered signal here is a long entry and
  Stages 7 and 9 score a signal by whether an upside target is hit before a
  downside stop; a bearish signal measured that way reports a number that reads
  like a hit rate and is not one.
* **`fib_retracement`'s tolerance is a fraction of the leg, not of price.** A
  fixed percentage of price is a different-sized band on a 40-rupee leg than a
  400-rupee one, so one setting would be strict on wide swings and meaningless
  on narrow ones.
* Divergence takes its oscillator from `context.py` rather than reimplementing
  it, so the detector and the RSI column on the chart cannot disagree about the
  same bar. `context.macd()` was extracted for this; the import is deferred to
  the call because `context` imports `patterns._geometry`.

**A trigger and a grade are different questions, and the checklist wants both.**
The owner's SMM checklist does not use either of these as an entry. It uses them
to *grade* a setup you already have: a retracement up to 50% is healthy and
61.8% or more is watchful; a divergence pointing the way you are trading is
encouragement and one pointing the other way is a caution. So `fib.py` and
`divergence.py` each carry a second entry point -- `retracement_grade()` and
`agreement()` -- reading the same primitives the detectors read.

Two consequences worth holding on to:

* **`agreement()` is why the bearish divergences had to exist.** On a long, the
  divergence that matters to the checklist is the *bearish* one. Nothing
  registers them, and they are not dead code.
* **The checklist leaves a gap and the code does not fill it silently.** It
  names "up to 50%" and "61.8% or more" and says nothing about the band
  between. That band is labelled `deep` rather than folded into either side.
  Measured on 568,534 bars since 2025: healthy 54.6%, deep 13.2%, watchful
  32.1%. Folding `deep` into `healthy` would move a sixth of all graded bars
  across the line that decides whether a setup is worth taking, which is a
  change of method rather than a rounding choice.
* Grade thresholds live at the **top level** of `patterns.yaml`, not under a
  pattern's key. `params_for()` passes everything nested under a registered name
  straight to that detector as keyword arguments, so a threshold parked there
  arrives as an unexpected kwarg and the detector raises.

**None of this has been measured.** Stage 7 and Stage 9 have not been run on the
swing family. Given that three families in a row have measured flat -- and that
momentum's hit-rate lift replicated out of sample and *still* lost 64-92% -- the
prior on a fourth is poor, and adding a detector is not evidence about anything.
Reproduce the frequencies in `config/patterns.yaml`, then run
`study/base_rates.py` before believing a word of it.

## The chart patterns are built from published definitions, and three are too rare to measure

`patterns/chart.py`. The SMM checklist's row 4 -- head and shoulders, double
tops and bottoms, rounding turns, cup and handle, flags, failed breakouts. Six
registered long signals, five bearish mirrors exported and unregistered as usual.

**Where each rule came from is in the file, and it matters which.** Lo, Mamaysky
and Wang (2000) formalise head-and-shoulders and double tops as conditions on a
sequence of five consecutive local extrema, with real numbers -- a 1.5 percent
tolerance on matched shoulders, 22 trading days minimum between the two tops,
and a 38-day window so only patterns *completed* inside it count. Those are
theirs. Bulkowski supplies the rest, and supplies no numbers at all: "a rounded
bowl", "not sharp or pointed", "be flexible", "allow variations". Every
threshold on the rounding, cup and flag detectors is therefore a choice, and the
docstrings say which ones.

Two mistakes worth keeping, because both produced a detector that looked fine
and fired essentially never:

* **A parabola fit is not a test of roundness.** A hard V -- two straight legs
  meeting at a point, exactly what Bulkowski excludes -- fits a quadratic at
  r2 = 0.937 against a true bowl's 1.000. No r2 threshold loose enough for real
  data rejects it. What separates them is how much of the fall the *middle* of
  the window spans: bowl 0.09-0.18, V 0.31-0.37 under realistic noise. That is
  `max_base_frac`, and it is the condition doing the work.
* **Calibrating a threshold on noiseless synthetic data.** The same flatness
  test was first measured on high-low and tuned against a synthetic bowl whose
  bars had no range at all. On real bars the daily high-low counts as unevenness
  in the base, the condition passed 0.74% of bars against a median ratio of
  0.97, and `rounding_bottom` fired **once in 921,061 bars**. Fixed by measuring
  on closes -- the same series the parabola is fitted to. If a new detector here
  fires almost never, suspect a units or noise mismatch before suspecting the
  market.

Same shape of error in the flag: `flag` was set to Bulkowski's three-week
*maximum* and used as an exact length, so a short flag's range was measured over
a window that still contained pole bars. Eight bars gives 59 hits where fifteen
gives 3, on identical data.

Frequencies on 921,061 bars, 1,520 liquid symbols, 2024-01-01 -> 2026-09-01:

    signal                     hits    % bars   per symbol-year
    fake_breakdown           28,243    3.066%          7.67
    double_bottom             1,683    0.183%          0.46
    inverted_head_shoulders     970    0.105%          0.26
    flag_breakout                46    0.005%          0.01
    cup_and_handle               39    0.004%          0.01
    rounding_bottom              33    0.004%          0.01

**The bottom three cannot be measured by this project, and that is the finding.**
Detecting a 2-point lift on a 26% base rate at |z|>3 needs roughly 4,300
occurrences. Over the full 2020-2026 history `fake_breakdown` and
`double_bottom` reach that or come close; `rounding_bottom`, `cup_and_handle`
and `flag_breakout` would yield around a hundred events each. Stage 7 can return
a number for them and the number will be noise. Do not loosen their thresholds
to fix this -- that trades a signal nobody can measure for a different signal
nobody can measure, and the Stage 7 write-up above is what happens when a
promising-looking cell gets acted on.

One data observation from the live scan: ETFs sit in the EQ series and are in
the universe. `LIQGRWBEES` -- a liquid fund whose NAV only rises -- printed a
double bottom at RSI 99.99. Nothing excludes them, and a chart pattern on a
money-market NAV is not a chart pattern.

## The decision sheet is filled by `tradeplan.py`, and filling it is not evidence

`tradeplan.py` plus `apps/views/decision.py`. The screener says a pattern fired;
this says where the stop goes, what the target is, whether the setup clears the
checklist's own reward-to-risk gate, and how many shares to buy. Rows 7 to 13 of
the SMM checklist and the whole arithmetic block of the Excel sheet.

`size()` is checked against the one filled column the owner's sheet ships --
close 269, stop 266, targets 306 and 308 -- and reproduces its risk of 3, rewards
of 37 and 39, and ratios of 12.33 and 13 exactly.

Three decisions in there matter more than the arithmetic:

* **The target never comes from the stop.** Defining it as three times the risk
  would make every setup clear the sheet's "R:R above 3" gate by construction --
  a tautology wearing a filter's clothes. Both targets are confirmed swing highs,
  which is the sheet's own "Major Resistance ... previous tops or resistances".
  A signal with no structure above it gets no target and no plan: on
  2026-09-01, 67 of 548 signals, nearly all breakouts at new highs, which have
  nothing to aim at by definition.
* **The sheet sizes positions twice and reconciles neither.** Row 29 caps shares
  by the 2% risk budget, row 32 sets them from a fixed investment. On its own
  example those are 666 and 74. Both are reported, `shares` is the smaller, and
  `binding` names which one is in charge. On live data the investment limit
  binds essentially always, which means the 2% risk cap is not the constraint
  anyone thinks it is.
* **`score` is confluence, not a forecast.** Candidates are ranked by how many of
  the six checklist rows agree, because that is the sheet's own logic and it is
  reproducible. Nothing here has measured whether confluence predicts anything,
  and Stage 7 measured every individual component within 0.35 percentage points
  of a randomly chosen bar. The page says so on its face.

**A filled sheet is persuasive in a way a scan table is not, and that is the
risk.** It shows R:R of 4.83 and a 28% ROI in rupees, next to a stop and a
target that look considered. Stage 9 backtested a planned 3:1 over 2020-2026
with real costs and measured -0.17R. `rr_min` is what the trade aims at. The
distance between that and what such trades return is the entire finding of this
project, and the page carries that warning inline rather than in a footnote.

One Streamlit trap worth knowing: **magic renders any bare expression at a
page's top level**, and an attribute docstring -- the `NAME = value` followed by
`"""..."""` convention used everywhere else in this repo -- is one. Written that
way in a page file it appears as body text above the title. Use `#` comments in
`apps/views/`.

## The owner's method is documented, and it disagrees with the code in places

Two source documents define the method the screener is meant to serve: an SMM
decision checklist (13 numbered checks behind a two-step trend filter) and a
PAPA sheet (18 named trade setups, each with its trigger, stop and target). They
are third-party course material -- implement the rules, do not redistribute the
documents.

The checklist is more specific than anything previously in this repo, and four
of its statements conflict with what is built. None has been acted on, because
each is a decision about method rather than a defect:

* **It reads an inverted hammer as bearish at a top** -- what most references
  call a shooting star. `inverted_hammer` is registered here as
  `direction="bullish"`, and `test_shooting_star_is_gone` pins the removal of
  the bearish form. Stage 7 measured the bullish reading, so changing the
  direction silently would invalidate a measured result.
* **Its EMA ribbon is 5/13/26.** `context.py` uses 5/13/25.
* **Its entry gate is reward-to-risk above 3**, computed off a hand-placed stop
  and target. That is a rule, and it is codeable, but it gates on *planned* R:R
  -- Stage 9 measured planned 3:1 delivering an average of -0.17R.
* **Its first step is a higher-timeframe trend filter** (monthly or weekly
  "tide" agreeing with a weekly or daily "wave"). This is Elder's double screen,
  and it is Stage 5 -- skipped as unproven. It is now specified precisely enough
  to build.

Not built, and each a sizeable job: the chart-pattern family the checklist
leans on (double bottom, head and shoulders, flags, cup and handle), and the
PAPA setups themselves. Several PAPA rows also use terms the sheets never
define -- `BKT`, `BKP`, "Ungali Setup", "BBC", and "Weapon" -- so those setups
cannot be implemented from the documents alone.

## Project shape

Built in numbered stages from a plan the owner wrote. Stages 0-4, 6 and 7 are done:
skeleton, data layer (3.3M rows, 1,633 sessions, 2020-01-01 → present), calendar (0
reconciliation mismatches), four detectors, the screener, sixteen context columns,
and the base-rate study above. Stage 5 (timeframes in the *screener* -- the chart
page already resamples) was skipped as unproven-value. `README.md` tracks status.

Stage 9 is done (see above). Stages 8, 10, 11 and 12 are open, and the Stage 7/9
results change what they should be -- there is no hit rate for Stage 8 filters to
improve, and no strategy for Stage 12 to paper-trade.

`data/` is never committed. Override its location with `NSE_SCREENER_DATA_DIR`.

The repo is **not** under version control — `git init` was declined deliberately.
`.gitignore` exists and lists `data/` first, so initialising later stays safe.
