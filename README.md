# NSE Candlestick Screener

Evening scanner for NSE equities, built in numbered stages over 6½ years of data.

**The headline result: the four candlestick patterns have no measurable edge, and
trading them loses money.** Stage 7 measured them across 2.9M bars against a
random-bar control and confirmed on a held-out period; Stage 9 backtested them
with real costs and they lose 71–86% of capital. The scanner works and the data
is sound — what it finds is not tradeable.

## Setup

```bash
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pytest
```

## Daily use

NSE publishes the bhavcopy around 18:00 IST. After that:

```bash
.venv/bin/python -m nse_screener.apps.backfill --start 2026-07-01   # top up
.venv/bin/python -m nse_screener.apps.scan --days 1                 # tonight's hits
```

Or the two-page app — pattern scan across the whole universe, and a candlestick
chart per symbol:

```bash
.venv/bin/streamlit run src/nse_screener/apps/dashboard.py
```

Other entry points:

```bash
.venv/bin/python -m nse_screener.apps.scan --days 3 --csv hits.csv
.venv/bin/python -m nse_screener.apps.show RELIANCE --sample 5 --seed 1
.venv/bin/python -m nse_screener.apps.backfill --start 2020-01-01   # full history, ~2h
```

## What's in the store

3.3M rows · 1,633 sessions · 2020-01-01 → 2026-07-29 · ~104 MB

NSE serves two bhavcopy layouts — legacy through 2024-07-05, UDiFF from
2024-07-08 — plus `sec_bhavdata_full` for weekend special sessions, where NSE
runs the session but publishes no bhavcopy zip. All three are handled. Raw
downloads are kept under `data/raw/`, so a parsing fix never costs a
re-download.

## Layout

| Path | Contents |
| --- | --- |
| `src/nse_screener/data/` | bhavcopy download, parquet store |
| `src/nse_screener/patterns/` | registry, candlestick + momentum + Heikin-Ashi families |
| `src/nse_screener/screener.py` | universe eligibility (history, gaps, liquidity) + the scan |
| `src/nse_screener/apps/` | CLI entry points; `apps/views/` are the app pages |
| `src/nse_screener/study/` | `base_rates.py` — the Stage 7 measurement |
| `src/nse_screener/backtest/` | `engine.py` day-by-day loop, `costs.py` fee model |
| `config/` | pattern thresholds, holiday overrides |
| `data/` | **not committed** — `raw/` downloads, `bars/` parquet |

Point the data elsewhere with `NSE_SCREENER_DATA_DIR=/some/path`.

`CLAUDE.md` carries the architecture notes and the reasoning behind decisions
that look wrong without it.

## Stage progress

- [x] **0 — Skeleton.**
- [x] **1 — Data layer.** Both bhavcopy layouts, year-partitioned parquet,
  resumable backfill. Verified three ways: 19,528/19,528 `prev_close` continuity
  matches, 2,180 symbols agreeing across the format cutover, and 14 sessions of
  RELIANCE matching NSE's own JSON API exactly.
- [x] **2 — Calendar.** Overrides, `period_is_complete()`, reconciliation — 0
  mismatches over 1,633 sessions. Found 6 weekend trading sessions the
  weekday-only walk had silently missed.
- [x] **3 — Pattern detectors.** Four detectors (doji, hammer, inverted hammer,
  bullish engulfing) transcribed from TA-Lib and verified against it on 24,495
  real bars with zero disagreements. Hammer and inverted hammer then moved to the
  *classic* reading — see `CLAUDE.md` for why, and for the measured frequencies.
- [x] **4 — Screener.** Universe eligibility + the scan, ~3s over 2,600 symbols.
- [~] **5 — Timeframes.** `resample_ohlc` and `period_is_complete` exist and the
  chart page uses them; the *screener* is still daily-only. Left there
  deliberately — nothing in Stage 7 suggested the weekly view would help.
- [x] **6 — Context.** Sixteen columns in four groups — trend (200 EMA, 5/13/25
  ribbon), momentum (RSI, stochastic, MACD), volatility (ATR, Bollinger) and
  volume. Verified against TA-Lib. Columns, never filters.
- [x] **7 — Base rates.** Done. See below.
- [ ] **8 — Filters.** Nothing to build on: filters that improve a hit rate need
  a hit rate to improve.
- [x] **9 — Backtest.** Day-by-day loop, full Indian delivery cost model, trade
  table. Passes the broken-rule check. See below.
- [ ] **10 — Blind labeler** · **11 — Dashboard** · **12 — Paper log** — open,
  and there is currently no strategy for a paper log to track.

## Stage 7 — the answer

```
                         n   hit_rate   lift vs a random bar
doji               254,023     25.95%   +0.25
inverted_hammer    197,552     25.66%   -0.05
hammer             123,035     25.81%   +0.11
bullish_engulfing   92,682     25.36%   -0.35
all bars         2,876,531     25.71%       -
```

+2 ATR target, −1 ATR stop, 10 sessions, entry at the next open. Every pattern
lands within 0.35 percentage points of a randomly chosen bar. Average forward
return was 0.4–0.8% against round-trip costs of 0.25–0.5%.

**An out-of-sample split confirmed it.** Hypotheses picked on 2020–2023 at
|z|>3, tested once on 2024–2026: of 21 positive candidates, **3 held up**, none
above +1pp. Doji's headline +0.64pp (z=5.13) came back negative. The best single
finding from the full sweep — hammer on >3× volume, +2.49pp — came back at
+0.53pp, z=0.47.

**What did replicate is negative**, at near-identical size in both periods:

```
doji, oversold RSI            -5.33  ->  -5.53   z=-9.6
doji, below the lower band    -4.66  ->  -4.86   z=-6.1
doji, >20% under the 200 EMA  -1.54  ->  -1.92   z=-5.2
```

These reversal patterns appearing in already-beaten-down conditions do
measurably worse than a random bar in the same conditions. That is the most
robust finding here, and it is a reason to exclude setups rather than take them.

```bash
.venv/bin/python -m nse_screener.study.base_rates --start 2020-01-01
```

`--target`, `--stop` and `--horizon` change the outcome definition. 3:1 was also
tried and is worse: it needs a 25% hit rate to break even and delivers 14.6%.

## Stage 9 — the same answer, in rupees

₹10L capital · 10 concurrent positions · 1% risk per trade · 3:1 over 10 sessions
· real delivery costs (0.322% round trip)

```
                  trades  win%   avg_r   return    CAGR    maxDD
all four           2,189  32.1   -0.17  -78.5%  -20.9%   -88.5%
hammer             2,572  31.7   -0.16  -77.9%  -20.5%   -85.6%
inverted_hammer    1,386  32.2   -0.12  -73.8%  -18.5%   -83.3%
doji               2,265  31.4   -0.19  -86.4%  -26.1%   -91.1%
bullish_engulfing  1,941  31.3   -0.24  -71.3%  -17.3%   -86.5%
```

```
gross P&L per trade   -₹118    ← negative before any cost
cost per trade         ₹200
net per trade         -₹319
```

Not a viable strategy ruined by costs — negative expectancy before costs, which
costs then roughly triple.

**The engine passes the plan's verification:** buying every Monday shows no edge
(−25.4%, avg R −0.07). That check runs as a test, not a one-off.

One structural finding worth keeping: the book was full almost always
(`skipped_no_slot` = 328,843). With ~480 signals a night and 10 slots, *which*
signals got traded was arbitrary. Any future signal family needs a ranking rule
before a backtest of it means anything.

## Open decisions

- **Corporate actions.** Prices are unadjusted; 59 splits/bonuses show as
  `prev_close` breaks across 2020–2026. The screener's 20% overnight-gap rule is
  a crude guard, and the study drops any signal whose forward window spans one.
  Stage 7 did not look strange in a way that implicated corporate actions, so
  proper adjustment stays unbuilt.
- **Delivery %** is on the Stage 6 list but is not in the store. It needs
  `sec_bhavdata_full` fetched for every session, which is a data-layer job.
- **Five holiday override reasons** are marked `# CONFIRM` — the dates are
  evidence-backed, the labels inferred. 2026-01-15 is unidentified.
- **TradingView spot-check** — Stage 1's stated gate, not yet done. The store was
  verified three other ways, including against NSE's own JSON API.
- **A second signal family was tried.** `patterns/momentum.py` (Donchian
  breakouts, 52-week highs, RSI(2) pullbacks, squeeze releases) went through the
  same pipeline unchanged. Its hit-rate lift *did* replicate out of sample
  (+2.5 to +3.3pp, z 7–9) and it still loses 64–92% in the backtest — because
  breakouts convert timeouts into resolutions in both directions, gaining
  +2.4pp of targets while taking +10.0pp more stops. **Never judge a signal on
  hit rate alone, even one that survives a split.**
- **Heikin-Ashi was tried as the exit** (`patterns/heikin_ashi.py`), on the theory
  that a fixed ATR target was cutting winners off by the clock. It works exactly
  as advertised — timeouts fall from 286 to 107, replaced by trend-break exits —
  and it does not help: −81.9% against a −82.7% baseline. Three exit policies
  spanning fixed-target, pure-trailing and both land within 6pp of each other.
  **The exit was not what was wrong.**
- **A liquidity floor now exists** (`Universe.min_price`, `min_traded_value`).
  Without it the same backtest reported **+731%**, of which sub-₹20 names supplied
  68.8% of the P&L on 15.2% of trades. On a ₹2 stock an ATR stop is a couple of
  ticks, so the R denominator collapses and the arithmetic invents an edge. This
  artefact has now produced two spectacular false positives. **Check P&L
  concentration before believing any good result here.**
- **Stage 8 has nothing to build on.** Filters that improve a pattern's hit rate
  need a hit rate to improve. If the project continues, the live options are a
  different signal family, the negative filters above, or accepting that the edge
  is not at entry and moving to sizing and risk.
- **No selection rule.** ~480 signals a night against 10 position slots means the
  book is full almost always and the choice among signals is arbitrary. Any future
  signal needs a ranking before a backtest of it is meaningful.
- **Version control.** Not initialised, deliberately. `.gitignore` lists `data/`
  first, so `git init` stays safe to run later.
