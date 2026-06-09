# Options Strategy Program (Alpha-v5) — SSOT

**One-screen truth for the options program: architecture, confidence plan, phase status, and the per-strategy verdict table.** Append-as-you-go.

**Started:** 2026-06-09 · **Data:** Polygon Options **Developer** ($79/mo) · **Status:** OPT-0..OPT-4 shipped. **Two verdicts in:** earnings IV-crush = KILL (single-name, cost-killed); index short-vol = KILL standalone but **VRP real + cost-robust** (PF 2.24 @1× / 1.75 @2×) just Sharpe-weak/under-powered. **Owner checkpoint:** refine index short-vol (regime overlay + weekly cadence) vs OPT-4b cross-sectional VRP vs reassess.

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
| **OPT-1b** | `options_provider.py` PIT parquet (survivorship-from-bars, holiday-aware knowable_date) + `backfill_options.py` (S3 OPRA day files) + `polygon_s3` `us_options_opra` + `OPTIONS_DATA.md` + 22 tests | ✅ shipped 2026-06-09 (S3 path smoke-tested) |
| **OPT-2** | `OptionsSimulator` (daily MTM off real closes, intrinsic settle) + `OptionsSpreadCostModel` (1×/2×/3× stress) + 19 golden-path tests | ✅ shipped 2026-06-09 |
| **OPT-3** | Strategy adapter (`options_strategy.py`) + **earnings IV-crush** E2E → first verdict: **❌ KILL** (Opus-certified) **[owner checkpoint]** | ✅ shipped 2026-06-09 |
| **OPT-4** | Index/ETF systematic short-vol (SPY/QQQ/IWM iron condors, 1.5×-SD strikes) + `IndexShortVolStrategy` + 6 tests | ✅ shipped 2026-06-09 — **KILL standalone** (VRP real + cost-robust PF 2.24/1.75, but Sharpe-weak/under-powered) |
| **OPT-5** | Options-data-as-signal: implied-move "priced-in" filter for PEAD (`ImpliedMoveProvider` + scorer hook) on the R1K 2y backfill | ✅ shipped 2026-06-09 — **STRONGER LEAD** (improves PEAD; lift is ALPHA-like, not beta; still underpowered) |
| **OPT-4b** | Index short-vol refinement (regime/VIX overlay + weekly cadence for power) **OR** cross-sectional/relative VRP (delta-neutral long-cheap/short-rich) | deferred (owner steer) |

**OPT-5 finding (2026-06-09).** Options-data-as-SIGNAL (not execution), judged on PEAD's OWN CPCV gate — so no alpha-gate-vs-risk-premium mismatch. The implied-move "priced-in" filter (skip PEAD entries whose realized announce move was within the pre-earnings IMPLIED move, realized/implied < 1.0) **improves PEAD** on the 2y options-covered window: mean Sharpe **0.891 → 1.346** (+0.45), path-t **1.56 → 1.90**, Avg PF **2.09 → 2.52**, Calmar 5.4 → 8.6 (%pos 74.1% unchanged). This is the **opposite** of the prior *price-based* priced-in filter (which hurt PEAD) and the program's first positive options signal. **Alpha-vs-beta CONFIRMED (the key check)** — after fixing the EventEdge harness to emit daily_returns_dated (#422), residual-α now computes: baseline beta-hedged Sharpe **+0.035** (β=0.12, resid-α t +0.04 = pure beta, as known); **filtered beta-hedged Sharpe +0.587** (β=0.14 ~flat, resid-α t +0.65). So the lift is **ALPHA-like, not beta** — the filter selects trades with genuine drift. The program's first real (non-beta) edge enhancement. Still **UNDERPOWERED / do NOT deploy**: resid-α t 0.65 < 2 on the thin 2y/8-fold sample; single threshold (multiplicity); DSR saturated; neither arm clears the gate. **A materially stronger lead** — remaining confirmation: threshold robustness (0.75/1.25) + more data (4y R1K backfill, needs a partitioned-write refactor). Default OFF. Built: `app/data/options_signal.py`, PEAD scorer hook (default OFF), `run_pead_implied_filter_cpcv.py`, `--r1k` backfill (60.8M bars, 100% PEAD coverage).
| **OPT-5** | Data-as-signal: implied-move PEAD filter, put-skew risk-off (no options exec; needs only OPT-1) | pullable early |
| **OPT-6** | Allocator integration (re-run gate, ≥3 sleeves) | after survivors |
| **OPT-7** | Tail hedge (covers trend fast-crash gap) | after 6 |
| **OPT-8** | Live wiring (Alpaca options, shadow-first) **[owner checkpoint]** | last |

**Dependencies:** OPT-1 blocks all; OPT-2 blocks OPT-3; OPT-3 gates catalog work; OPT-5 only needs OPT-1.

## Strategy verdict table (append per validation)
| Strategy | Phase | Net Sharpe @1× | @2× spread | residual-α t | Avg PF @1× | Verdict |
|---|---|---|---|---|---|---|
| Earnings IV-crush — ATM iron butterfly (1-wide, strawman) | OPT-3 | −3.86 | — | −5.75 | 0.16 | ❌ KILL (tight ATM wings blown through) |
| Earnings IV-crush — OTM condor 1×EM, nearest-weekly (aggressive) | OPT-3 | −1.82 | −2.52 | −2.57 | 0.59 | ❌ KILL (mis-parameterized: ~3-DTE gamma) |
| **Earnings IV-crush — CANONICAL (≈25-DTE, 1.3×EM, no strawman)** | OPT-3 | **−1.02** | **−1.67** | **−0.24** | **1.21** | ❌ **KILL (thin / cost-killed)** |

### OPT-4 — index/ETF systematic short-vol (SPY/QQQ/IWM iron condors)
| Structure | Net Sharpe @1× | @2× spread | residual-α t | Avg PF @1× | @2× | Verdict |
|---|---|---|---|---|---|---|
| 1.0× realized-SD strikes (too tight, strawman) | −0.44 | −0.80 | −1.10 | 0.78 | 0.52 | ❌ KILL |
| **1.5× realized-SD strikes (canonical ≈16-delta)** | **+0.04** | −0.19 | −1.25 | **2.24** | **1.75** | ❌ **KILL (Sharpe-weak / under-powered)** |

**OPT-4 finding (2026-06-09, Opus-certified — PF is genuine, no look-ahead).** Systematic index short-vol (monthly ~35-DTE condors, short strikes 1.5× realized-SD ≈ 16-delta, 21-day hold) on SPY/QQQ/IWM, 4y, CPCV k=8/p=2. **The index VRP is REAL and cost-robust** — Avg PF **2.24 @1× and still 1.75 @2× spread** (index spreads are ~pennies, so the 2× stress that killed single-name earnings vol barely dents it; a clear qualitative win over OPT-3). **But it is risk-adjusted-flat** (mean Sharpe ~0, 56% positive folds, path-t ~0, residual-α t −1.25) — the crisis fat-tail eats the vol-adjusted return — and **statistically under-powered** (7-fold LOW COVERAGE on 3 ETFs / monthly cadence). So it **KILLs the significance gate** as a *naked* sleeve. Strike sizing was decisive: 1.0× realized-SD (≈32% breach) is structurally negative; 1.5× (≈16-delta) flips PF to 2.24 — the same "strikes outside the move" lesson as OPT-3. **The planned refinements** (regime/VIX de-risk overlay to cut the crisis tail → lift Sharpe; weekly cadence + more ETFs → power) are the path to potentially clearing it, but pile parameters on a thin sample (overfitting risk) — an owner checkpoint. Net: the options VRP is *real and cost-robust at the index level* but not yet a gate-clearing standalone sleeve.

---

**OPT-3 finding (2026-06-09 — authoritative, after a full Opus 4.8 deep-dive of the whole build).** Three deep-dive auditors confirmed the **harness, data, and simulator are clean** (data pristine: 0 bad closes, IV-crush visible in real marks, PIT/survivorship/OCC all verified; sim P&L sign + accounting correct, cost model mildly conservative). The fair-test auditor found the *first* parameterization was handicapped (nearest-weekly ~3-DTE expiry = max gamma/tiny vega; short strikes only 1×EM; ATM strawman on first-events), so we re-ran the **canonical** structure. Result: at realistic 1× spreads the edge is **gross-profitable but risk-adjusted-flat** — Avg PF 1.21, Calmar 0.85, **residual-α t −0.24 (≈ zero)**, but mean Sharpe −1.0 with only 33% positive folds (the classic short-vol fat left tail), and it **dies at 2× spread** (PF 0.82). **Conclusion: single-name earnings IV-crush is a real but too-thin premium, killed by options transaction costs** (cf. the equity reversal sleeve). Not a keepable standalone edge. The deep-dive *changed the reason, not the verdict* — and points the program at **index/ETF VRP**, where spreads are ~pennies and the VRP is fatter (the cost problem that kills single-name vol is minimal there). **A KILL is a success of the harness.**

## Risks → mitigations (top)
computed-IV inaccuracy → daily engine-vs-snapshot validation (spike already PASS) · look-ahead → knowable_date + as-of accessor · spread unrealism → settlement-mark + mandatory 2× stress · survivorship → expired-inclusive universe · American/dividend → BjS + tri-model + parity tests · capacity (no OI) → volume/notional caps · live-exec gap → shadow-first + fill reconciliation · catalog overfitting → DSR + N_TRIALS + log nulls, never filter-hunt (B5 trap).

## Pointers
Reference: `docs/reference/OPTIONS_DATA.md` (to be written OPT-1, mirrors SHORT_INTEREST_DATA.md). Plan: DECISIONS 2026-06-09. Reuse anchors: `event_edge.py`, `cpcv.py`/`gates.py`, `attribution.capm_alpha`, `short_interest_provider.py`, `polygon_s3.py`, `trend_sleeve.py`.
