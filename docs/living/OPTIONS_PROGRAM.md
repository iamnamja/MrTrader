# Options Strategy Program (Alpha-v5) — SSOT

**One-screen truth for the options program: architecture, confidence plan, phase status, and the per-strategy verdict table.** Append-as-you-go.

**Started:** 2026-06-09 · **Data:** Polygon Options **Developer** ($79/mo) · **Status:** OPT-0 (charter) + OPT-1a (pricing engine) shipped; OPT-1b (data layer) next.

## Why
The free-data 3rd-sleeve hunt is exhausted (reversal cost-dead, carry pre-cost-negative, estimate-revision data-blocked — see ML_EXPERIMENT_LOG Phase 4/4b/4c). Options unlock the **highest-ceiling** remaining edge: the **variance risk premium** (earnings IV-crush, cross-sectional VRP, systematic short-vol), plus options-data-as-signal and tail-hedging. Goal = a **resilient base** to explore MANY options strategies and validate each **with confidence**, honestly KEEP/KILL-ing like the prior nulls.

## Data reality (Polygon Developer — verified)
- IV/greeks/OI served **only in the CURRENT snapshot** → **we COMPUTE historical IV+greeks** (BS-European fast path + Bjerksund-Stensland for American + CRR cross-check) from EOD price + underlying + rate + dividend.
- **No historical NBBO** → backtests mark off EOD `day.close` and **model + STRESS the spread (1×/2×/3×)**; KEEP must survive **2×**. (Options spread is the dominant cost — a reversal sleeve already died purely on cost.)
- **No historical OI** → liquidity via volume/notional, never OI.
- Bulk 4yr = flat files (S3 `us_options_opra/day_aggs_v1`, OHLCV only); contract universe incl. **expired** via `/v3/reference/options/contracts?as_of=&expired=true` (survivorship cure). Earnings dates from FMP/Finnhub (BMO/AMC), not Polygon. Polygon rebranded "Massive" (docs domain only; API paths unchanged).

## Architecture — five clean layers (resilience)
```
[DATA] ⟂ [PRICING ENGINE] ⟂ [SIMULATOR + COST] ⟂ [STRATEGY SPEC] ⟂ [REUSED gates / allocator / live]
 durable      durable             durable            disposable          already built
```
A new strategy = a new scorer + a `run_*.py`; nothing below the adapter changes. If Polygon later sells historical IV, swap the data/engine internals and every strategy still runs.

**Four frozen contracts** (`app/options/contracts.py`):
1. `OptionsDataProvider` — PIT as-of (mirrors `app/data/short_interest_provider.py` knowable_date); `get_universe(include_expired)`, `get_contract_bars(as_of)`, `get_current_snapshot` (validation/live only).
2. `OptionsPricingEngine` — pure: `price`/`implied_vol`/`greeks`, `style ∈ {european, american}`.
3. `OptionsSpreadCostModel` + `OptionContractSim` — sim marks off EOD close, % -premium cost with the stress mult; emits the existing `SimResult` shape.
4. `OptionsStrategy` — duck-types EXACTLY to `scripts/walkforward/event_edge.EventEdgeStrategy` (`fetch_data`, `run_fold→FoldResult` with `daily_returns_dated`, `spy_prices`, `is_trained=False`).

**Reused below the adapter (no new code):** `run_cpcv`, all of `gates.py` (significance gate, DSR, PF, Calmar, fold-coverage), `attribution.capm_alpha`, `sequential_baseline`, `sleeve_allocator.build_book`/`vol_weights`/`DEFAULT_REGIME_TILT` + `sleeve_allocator_live`, the live-sleeve template (`trend_sleeve.py`), trackers, `agent_config` appliers, conftest env-DB isolation + CI flake8.

## Confidence plan (how we trust an options backtest)
snapshot-validated engine → PIT survivorship-safe data (expired-inclusive, knowable_date) → settlement-mark sim with stress-tested spreads (KEEP survives 2×) → the **same** `run_cpcv` + significance gate + CAPM-residual-α we already trust for equities. American exercise via BjS + tri-model cross-check; IV-crush carried by real IV data (not synthesized); rate/dividend sensitivity flagged if a verdict flips.

**Per-strategy KEEP/KILL gate (identical for all):** (1) `run_cpcv` on the adapter (full coverage via `is_trained=False`); (2) significance gate (paper tier); (3) CAPM residual-α (alpha vs beta); (4) **spread-stress — must pass at 2×**; (5) capacity (volume/notional); (6) verdict logged here + ML_EXPERIMENT_LOG + DECISIONS. A KILL is a success of the harness (cf. reversal/carry). KEEP ≠ live (needs allocator-additive + shadow reconciliation).

## OPT-0 feasibility spike result (2026-06-09) — ✅ confidence keystone PASSES
`scripts/spike_options_iv_check.py`: computed **BS-European IV vs Polygon served IV**, 15 liquid underlyings, ~4,166 contracts, current snapshot (q=0, r=4.3%).
- **Near-ATM (|delta−0.5|<0.15, n=470): median |IV err| = 0.0086 (0.86 vol-pts), mean +0.0003 (unbiased), p90 = 0.027.** → computing IV from EOD close is accurate where VRP trades.
- All contracts (n=4166): median |err| 0.011, **mean +0.035** (ITM/OTM + dividend bias from q=0/European) → OPT-1 adds BjS + real dividends/rate to tighten tails.
- **OPT-1 acceptance tolerance (set empirically):** near-ATM median |IV err| < 0.010 (target < 0.005 with BjS+div); all-contract mean bias → ~0 after dividends.

## OPT-1a engine validation result (2026-06-09) — ✅ PASSES (confidence keystone re-run with real engine)
`scripts/validate_options_engine.py`: computed **American (Bjerksund-Stensland) IV vs Polygon served IV**, 15 underlyings, current snapshot, real per-underlying dividend yield, r=4.3%, **liquidity filter (day-vol ≥ 10)** to drop stale untraded marks.
- **Near-ATM (n=231): median |IV err| = 0.0072, mean bias +0.0071, p90 = 0.024 → PASS (< 0.010).**
- **All contracts (n=1,541): mean IV bias +0.0068 → PASS (< 0.010).** (Unfiltered, the bias was +0.022, concentrated in illiquid tails — a **data-timing artifact**: the live snapshot pairs an option's last trade with the *live* spot; untraded deep contracts are stale. EOD-bar backtests pair same-day option+underlying closes, so the artifact does not occur in OPT-1b backtests.)
- **Delta cross-check (engine delta vs served delta, n=1,541): median |err| = 0.0011** — the engine greeks are essentially exact.
- Residual +0.0068 bias attributed to flat-rate (4.3% vs Polygon's curve) + crude dividend estimate; OPT-2 wires a real rate series. American + dividends + CRR-fallback removed the OPT-0 spike's +0.035 European/q=0 bias.

## Phase roadmap + status
| Phase | What | Status |
|---|---|---|
| **OPT-0** | Charter + 4 contracts + feasibility spike | ✅ shipped 2026-06-09 (spike PASS) |
| **OPT-1a** | Pricing engine (BS-European + Bjerksund-Stensland American + CRR cross-check + IV solver + greeks) + 18 unit tests + `validate_options_engine.py` vs snapshot | ✅ shipped 2026-06-09 (validation PASS) |
| **OPT-1b** | `options_provider.py` PIT parquet + `backfill_options.py` + `polygon_s3` `us_options_opra` | NEXT |
| **OPT-2** | Contract-level simulator + `OptionsSpreadCostModel` (golden-path P&L tests) | after 1 |
| **OPT-3** | Strategy adapter + **earnings IV-crush** E2E → first KEEP/KILL **[owner checkpoint]** | after 2 |
| **OPT-4** | VRP catalog: cross-sectional VRP, index short-vol, skew/term carry | after 3 |
| **OPT-5** | Data-as-signal: implied-move PEAD filter, put-skew risk-off (no options exec; needs only OPT-1) | pullable early |
| **OPT-6** | Allocator integration (re-run gate, ≥3 sleeves) | after survivors |
| **OPT-7** | Tail hedge (covers trend fast-crash gap) | after 6 |
| **OPT-8** | Live wiring (Alpaca options, shadow-first) **[owner checkpoint]** | last |

**Dependencies:** OPT-1 blocks all; OPT-2 blocks OPT-3; OPT-3 gates catalog work; OPT-5 only needs OPT-1.

## Strategy verdict table (append per validation)
| Strategy | Phase | Net Sharpe @1× | @2× spread | residual-α t | corr to book | Verdict |
|---|---|---|---|---|---|---|
| _(earnings IV-crush)_ | OPT-3 | — | — | — | — | pending |

## Risks → mitigations (top)
computed-IV inaccuracy → daily engine-vs-snapshot validation (spike already PASS) · look-ahead → knowable_date + as-of accessor · spread unrealism → settlement-mark + mandatory 2× stress · survivorship → expired-inclusive universe · American/dividend → BjS + tri-model + parity tests · capacity (no OI) → volume/notional caps · live-exec gap → shadow-first + fill reconciliation · catalog overfitting → DSR + N_TRIALS + log nulls, never filter-hunt (B5 trap).

## Pointers
Reference: `docs/reference/OPTIONS_DATA.md` (to be written OPT-1, mirrors SHORT_INTEREST_DATA.md). Plan: DECISIONS 2026-06-09. Reuse anchors: `event_edge.py`, `cpcv.py`/`gates.py`, `attribution.capm_alpha`, `short_interest_provider.py`, `polygon_s3.py`, `trend_sleeve.py`.
