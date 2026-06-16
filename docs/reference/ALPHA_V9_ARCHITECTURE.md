# Alpha-v9 Architecture & Implementation Plan

**Status:** ACTIVE design (companion to `ALPHA_V9_ROADMAP.md`).
**Created:** 2026-06-16 · **Author:** Opus 4.8.
**Purpose:** a detailed, future-proof, resilient design for evolving MrTrader from a *single-edge
(trend) + overlays* system into a **multi-engine premia book** that can safely add a second return
engine (VRP), a new universe (crypto), and a new asset class/broker (futures/IBKR) — **without
rewrites and without letting dead paths wake up.**

> **Design north star.** The platform's job is no longer "discover single-name equity alpha." It is to
> **operate a small book of 2–3 return engines across risk classes under a shared tail governor**, where
> every component is a uniform declaration validated by one harness, promoted as an immutable artifact,
> and executed by a deterministic, idempotent, fully-audited live engine. New asset classes and brokers
> enter through **adapters**, never parallel systems.

---

## 0. Principles (the non-negotiables this design enforces)

1. **One harness, many adapters.** All research flows through the Sleeve Lab. Crypto, futures, and
   options enter via *data/instrument adapters* + *cost-model adapters* — never a second evaluator.
   (Reviewer consensus: "add adapters, not parallel systems.")
2. **Research ≠ live.** The live runtime knows only an **immutable, versioned sleeve manifest**. It
   cannot import research scripts, and a stale config flag cannot resurrect a dead strategy.
3. **Promotion produces an artifact, not a flag flip.** Every live component is backed by a signed,
   hash-stamped promotion artifact (data version + config + gate verdict + period). Live loads artifacts.
4. **Risk budgets, not capital budgets.** The allocator sizes by risk contribution (HRP), targets a
   book-level vol, and treats idle capital as an explicit cash sleeve.
5. **Deterministic & idempotent execution.** Given a data snapshot + manifest, the executor reproduces
   the exact target orders; re-running never double-trades; every decision is in an append-only ledger.
6. **Engines diversify by skew, not by chasing corr<0.30.** Pair a positive-skew engine (trend) with a
   negative-skew engine (VRP) under the governor; that is the structural diversification.
7. **Nothing graduates on backtest alone.** Live-paper ratification + backtest-to-paper tracking error
   are first-class promotion gates (esp. for options/crypto/overnight where paper fills ≠ real fills).
8. **Fail safe, fail loud.** No `except Exception: continue` swallowing in research or live paths
   (the class of bug that produced deflationary IC≈0 results — see roadmap Ⓐ). Errors surface; windows
   that can't be built are *reported*, never silently dropped.

---

## 1. Target architecture (5 layers)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  L1  DATA LAKE & PIT LAYER                                                      │
│  immutable raw snapshots · point-in-time calendars/release stamps ·            │
│  data-version hashes stamped into every research result ·                      │
│  survivorship-free datasets flagged; survivorship-prone sources flagged unsafe │
│  adapters: yfinance · Alpaca · Polygon(frozen) · FMP · FRED · Norgate(new) ·   │
│            crypto-OHLCV(new) · FINRA short-volume(new)                          │
└───────────────┬────────────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L2  COMPONENT DECLARATIONS  (uniform unit of research)                         │
│  ReturnSleeve · Overlay · CashSleeve · HedgeSleeve · ExecutionStrategy          │
│  each declares: rationale · data deps + version · signal-timestamp convention · │
│  rebalance schedule · cost-model adapter · component_type · expected failures · │
│  acceptance criteria                                                            │
│  registries: @register_sleeve · @register_overlay  (extend, don't fork)         │
└───────────────┬────────────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L3  RESEARCH HARNESS  (Sleeve Lab — one path)                                  │
│  CPCV (honest N_eff) → Ruler-v2 (Track-A / Track-B, fixed per roadmap P0-2) →   │
│  powered stability test → adversarial review → R7 pre-registration             │
│  positive-control harness (P0-1) · premium-% option cost adapter (P2-4)         │
│  OUTPUT: immutable PromotionArtifact (verdict + data hash + config + period)    │
└───────────────┬────────────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L4  PORTFOLIO ALLOCATOR  (book construction)                                   │
│  HRP over engines · book-level vol target · explicit cash/T-bill sleeve ·       │
│  portfolio-level governor (VIX-term ⊕ credit) · tail/correlation constraints ·  │
│  Bayesian-shrunk, probationary small-size for new diversifiers                  │
│  OUTPUT: target book = {sleeve → target weight} × overlay multipliers           │
└───────────────┬────────────────────────────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│  L5  LIVE EXECUTOR  (deterministic, idempotent, audited)                        │
│  loads MANIFEST (immutable) → reconcile broker positions → compute deltas →     │
│  idempotent orders (intent IDs) → route via BrokerAdapter → reconcile again →   │
│  append decision ledger · shadow mode · tiny-live mode · replay tool            │
│  broker adapters: Alpaca(equity/ETF/options/crypto) · IBKR(futures, P4-3)       │
└──────────────────────────────────────────────────────────────────────────────┘
```

The current PM/RM/Trader agents map onto **L4 (PM = allocator) + L5 (RM = constraints, Trader =
executor)**. We keep the abstraction; we make the *state transitions* deterministic and the *runtime
surface* narrower than the research surface.

---

## 2. Layer detail & what changes vs today

### L1 — Data lake & PIT layer
- **Today:** providers in `app/data/*`, a `macro_history.parquet` cache, a frozen 4y Polygon options
  store; yfinance survivorship is a known, *un-enforced* hazard.
- **Add:**
  - **A `survivorship_safe: bool` flag per dataset.** The harness **refuses** to promote any single-name
    long/short or event sleeve beyond `exploratory` on a `survivorship_safe=False` source (enforces
    roadmap C4). yfinance-equities = False; ETF-EOD = effectively True; Norgate-equities = True.
  - **Data-version hashing.** Every `PromotionArtifact` records the hash of the exact data slice it was
    validated on, so a result is reproducible and a silent data change invalidates the artifact.
  - **New adapters (uniform interface `get_bars / get_calendar / version_hash`):** `norgate_provider`
    (futures + survivorship-free equities), `crypto_provider` (Alpaca + free exchange OHLCV fallback),
    `finra_short_volume_provider` (daily, ~2009+).
- **Resilience:** providers fail *loud* (no silent empty frames); a missing/short series raises and is
  recorded, never coerced to a no-edge zero (root-cause fix for the deflationary-bug class, roadmap Ⓐ).

### L2 — Component declarations
- **Today:** `Sleeve` + `@register_sleeve`; `Overlay` + `@register_overlay`; `component_type` partly
  used by Ruler-v2 waivers.
- **Add a single declarative schema** every component implements (extend, don't fork):
  ```
  @dataclass
  class ComponentSpec:
      name: str
      kind: Literal["return_sleeve","overlay","cash_sleeve","hedge_sleeve","execution_strategy"]
      component_type: Literal["alpha","diversifier","risk_premium","trend","cash"]
      rationale: str                     # economic mechanism (required, non-empty)
      data_deps: list[DataDep]           # each carries survivorship_safe + version_hash
      signal_timestamp: str              # PIT convention (e.g. "signal@close_t, trade@open_t+1")
      rebalance: str                     # "weekly:MON", "monthly:EOM", "daily:MOC"
      cost_model: str                    # "notional_bps" | "premium_pct_iv_aware" | "futures_tick"
      expected_failure_modes: list[str]
      acceptance: AcceptanceCriteria     # which track, which thresholds, probation eligible?
  ```
- **Why:** makes "what data, what timestamp convention, what cost model" *explicit and checkable* —
  closing the gaps reviewers flagged (wrong cost model for options; PIT timestamp leakage; survivorship).

### L3 — Research harness (the gate, fixed)
- **Keep (reviewers credited):** honest `N_eff = n_folds` (do **not** inflate paths — roadmap Ⓖ);
  Ruler-v2 Track-A/Track-B; pooled-OOS HAC; Bayesian posterior + mandatory live-paper at CAPITAL;
  R7 pre-registration; adversarial review.
- **Change (roadmap P0-2):**
  - **Powered stability test** replaces the binary both-halves guard (`scripts/walkforward/sleeves.py`):
    accept if the two half-Sharpes are **not significantly different** (bootstrap CI on the half-diff
    overlaps 0, or a Chow/homogeneity test) **and** pooled HAC-SR is significant. Episodic strategies
    (tail hedge, carry, overlay) use a **mechanism-specific** test (crisis convexity / bleed budget /
    marginal drawdown reduction) instead of sign-stability.
  - **Diversifier-aware Track-B:** drop/lower `standalone_vt_SR` for `{diversifier, risk_premium}`;
    add a **probation path** — `P(ΔSR>0) ≥ 0.75` + small size + **mandatory live-paper ratification**
    (keep the high bar for `alpha`/CAPITAL sleeves).
- **Add:**
  - **Positive-control harness (P0-1):** a first-class command that pushes a *known* anomaly (12-1
    momentum / 1-month reversal / low-vol) through the *exact* feature→label→CPCV path and asserts the
    published sign/magnitude. **This is a CI-style guard on the validator itself** — run it before
    trusting any "no edge" verdict.
  - **Premium-% IV-aware option cost adapter (P2-4):** spread as % of premium, scaled by IV — required
    for any `cost_model="premium_pct_iv_aware"` component (blocks phantom options edges, roadmap Ⓕ).
  - **`REQUIRE_TRUE_WF_FOR_PROMOTION=True`** for trained components (`app/ml/retrain_config.py`, P0-3).
- **OUTPUT:** an immutable `PromotionArtifact { spec, verdict, data_hash, oos_period, gate_version,
  created_at }` — the *only* thing L5 ever trusts.

### L4 — Portfolio allocator (the new center of gravity)
- **Today:** "trend @ 25% + cash," overlays multiply exposure, `GLOBAL_DERISK_FLOOR=0.25`.
- **Redesign to a risk-budgeted multi-engine book:**
  - **Engines:** `trend` (have), `vrp` (P3-2, when ratified), `crypto_trend` (P3-1), optional
    `risk_premia_composite` (P3-4), `futures_trend_carry` (P4, IBKR-gated).
  - **HRP allocation** across engines (robust at small N; avoids MVO blow-ups — roadmap C7/P5-3).
  - **Explicit cash/T-bill sleeve** (P1-1): all unused risk budget → SGOV/BIL, benchmarked.
  - **Portfolio-level governor:** the VIX-term ⊕ credit multipliers apply to the *book's risky engines*
    (not each sleeve in isolation), floored at `GLOBAL_DERISK_FLOOR`.
  - **Skew-pairing logic:** trend (positive-skew/long-crisis) ⊕ VRP (negative-skew/short-crisis) sized
    so trend's crisis convexity backstops VRP's short-vol tail (roadmap §5 — the core insight).
  - **Trend allocation** raised per the P1-2 owner decision (Kelly/vol-target, governor-defensible).
- **OUTPUT:** `target_book = {sleeve_name: weight}` after overlay multipliers + floor.

### L5 — Live executor (deterministic, idempotent, audited)
- **Today:** orchestrator inside the uvicorn process (`app/main.py`), `run_trend_rebalance` Mon 09:45 ET,
  `alloc *= overlay_mult`.
- **Add (reviewer-driven reliability — ChatGPT §4.1, the biggest non-research priority):**
  - **Manifest-only runtime:** L5 loads `live_manifest.json` (list of `PromotionArtifact` refs) — it
    **cannot** import research code; a dead strategy can't wake from a stale flag.
  - **Append-only decision ledger:** every rebalance writes `{ts, manifest_hash, data_snapshot_hash,
    target_book, intent_ids, orders, fills, reconcile_before/after}`.
  - **Idempotent orders with intent IDs:** re-running a rebalance with the same inputs is a no-op;
    partial fills reconcile cleanly.
  - **Broker reconciliation before & after** every rebalance.
  - **Modes:** `shadow` (compute, log, don't trade) → `tiny_live` (execution validation at $1k–5k) →
    `live`. Options/crypto/overnight **must** pass `tiny_live` before `live`.
  - **Replay tool:** "given this data snapshot + manifest, reproduce the exact target orders" — for
    audit and debugging.
  - **Backtest-to-paper tracking error** (P1-4) emitted as a first-class metric.
- **Broker adapters** behind one interface (`submit / cancel / positions / reconcile`): `AlpacaAdapter`
  (equity/ETF/options/crypto, exists), `IBKRAdapter` (futures, P4-3 — built only after P4-2 passes).

---

## 3. The skew-paired multi-engine book (the design's reason to exist)

```
                       ┌─────────── PORTFOLIO GOVERNOR (VIX-term ⊕ credit) ───────────┐
                       │                  applies to risky engines, floor 0.25         │
   ┌───────────────────┼───────────────────────────────┬──────────────────────────────┐
   ▼                   ▼                                ▼                              ▼
ENGINE 1: TREND   ENGINE 2: VRP                  ENGINE 3: (optional)            CASH SLEEVE
+ skew, long-     − skew, short-crisis           one surviving diversifier       SGOV/BIL
crisis convexity  defined-risk index structures  (risk_premia_composite          (idle risk budget,
ETF→+crypto→      (P3-2, live-paper ratified)     or probationary carry)          benchmarked)
+futures(P4)
   │                   │                                │                              │
   └─────────── HRP risk-budget allocation → book vol target ───────────────────────┘
```

- **Why this and not "a 4th uncorrelated equity sleeve":** the diversification we failed to find 4×
  *between equity sleeves* exists *between skews*. Trend profits in the sustained crises where VRP bleeds;
  VRP harvests the calm grind where trend whipsaws. The governor caps the shared left tail. This is the
  most likely path from ~0.7 to ~1.0+ book Sharpe.
- **Falsifiable design claim (carry into the build):** the trend⊕VRP blend's drawdowns are materially
  shallower than either engine alone and the blended Sharpe exceeds both. If trend and a governor-gated
  VRP book are *positively* correlated in the left tail, the pairing thesis is wrong — kill VRP, keep
  trend+cash+overlays (the honest fallback).

---

## 4. Resilience & future-proofing (how this survives contact with reality)

| Risk | Design defense |
|---|---|
| **Dead strategy resurrects from a stale flag** | Manifest-only runtime; L5 can't import research; only `PromotionArtifact` refs are live. |
| **Silent deflationary bug fakes "no edge"** (roadmap Ⓐ) | No swallowed exceptions; providers fail loud; **positive-control harness** asserts the pipeline can detect a known anomaly. |
| **Phantom options edge from wrong costs** (Ⓕ) | `premium_pct_iv_aware` cost adapter required; 4y frozen options = exploratory-only; live-paper mandatory. |
| **Survivorship-flattered single-name kills** (C4) | `survivorship_safe` flag gates promotion; single-name work capped at `exploratory` until Norgate. |
| **New asset class tempts a parallel system** | Adapter interfaces at L1 (data) + L5 (broker) + cost-model registry; Sleeve Lab unchanged. |
| **New broker (IBKR) risk** | `BrokerAdapter` interface; `tiny_live` execution-validation mode before any real futures size. |
| **Overfit gate / Type-II machine returns** | Powered stability test (not binary), diversifier probation, honest N_eff (no path inflation). |
| **Live ≠ backtest** | Tracking-error metric is a promotion gate; shadow → tiny_live → live ladder. |
| **Estimation error at small N dominates allocation** | HRP + Bayesian shrinkage + probationary small sizing; never size on point Sharpe. |

---

## 5. Implementation plan (mapped to the roadmap, in dependency order)

> Each step = one feature branch + PR (auto-merge), full pytest 0-failures, docs-no-drift, pre-register
> any confirmatory run (R7), independent adversarial review before any verdict.

### Milestone A — Validate the validator *(roadmap Phase 0; ~1–2 PRs)*
- **A1** Build the **positive-control harness** (P0-1): `scripts/walkforward/positive_control.py` runs
  12-1 momentum / 1-mo reversal / low-vol through the exact swing feature→label→CPCV path; asserts
  published sign/magnitude. Wire as a CI-style check. **Exit:** verdict on whether "IC≈0" is real.
- **A2** **Fix the gate** (P0-2): powered stability test in `scripts/walkforward/sleeves.py`;
  diversifier-aware Track-B + probation path in `app/research/ruler_v2.py`. Re-run carry F3 through it.
- **A3** Flip `REQUIRE_TRUE_WF_FOR_PROMOTION=True` for trained paths (P0-3) + pin tests.

### Milestone B — Harden & monetize *(Phase 1; mostly 👤 decisions + light code)*
- **B1** `cash_sleeve` (P1-1): SGOV/BIL sleeve + book benchmark switch. (No new data.)
- **B2** 👤 Trend-allocation analysis + DECISIONS entry (P1-2).
- **B3** 👤 Credit-overlay shadow → enable/park/kill memo (P1-3).
- **B4** Live-fill tracking-error report (P1-4): extend the live logging in `app/live_trading/`.

### Milestone C — Cost model + adapters *(Phase 2; unblocks C-engines)*
- **C1** `premium_pct_iv_aware` option cost adapter (P2-4) in the cost-model registry.
- **C2** Data adapters: `crypto_provider`, `finra_short_volume_provider` (uniform L1 interface).

### Milestone D — New return engines on Alpaca *(Phase 3)*
- **D1** Crypto trend sleeve (P3-1) — reuse TSMOM; vol-target hard; honest power floor; Track-A/B + shadow.
- **D2** VRP research track (P3-2) — defined-risk index structures, governor-gated, premium-% costs,
  exploratory backtest + **live-paper ratification before any live size.**
- **D3** Overnight/intraday decomposition (P3-3) — MOC/MOO, pre-registered net-of-cost test.
- **D4** `risk_premia_composite` (P3-4) — gate the basket through the fixed Track-B.
- **D5** FINRA daily short-volume overlay (P3-5).

### Milestone E — Strategic data bet *(Phase 4; 👤 purchase)*
- **E1** 👤 Buy Norgate; add `norgate_provider`; re-run PEAD + F2 on **survivorship-free** data (P4-1).
- **E2** Futures trend+carry research packet (P4-2) — pre-registered; Track-B vs ETF trend.
- **E3** 👤🔴 `IBKRAdapter` + tiny-live validation — **only if E2 passes** (P4-3).

### Milestone F — Allocator + modeling reframe *(Phase 4/5)*
- **F1** HRP allocator + portfolio-level governor + skew-pairing (L4 redesign; P5-3).
- **F2** Meta-labeling for sizing (P5-2) — *gated by A1 passing*.
- **F3** Vol/regime prediction targets (P5-1) — feeds D2.

---

## 6. What we explicitly are NOT building (anti-scope)

- No separate research evaluators per asset class (adapters only).
- No per-regime separate ML models; no RL/end-to-end portfolio optimizer.
- No live IBKR/futures code before futures research passes.
- No options promotion from the 4y frozen backtest alone.
- No path-inflated CPCV; no swallowed-exception "robustness."
- No new gates beyond the P0-2 fixes unless a specific, named failure mode demands one.

---

## 7. Definition of done for Alpha-v9

1. The validator is proven (A1) — we *know* whether "mined out" is the market or a bug.
2. The gate no longer kills genuine diversifiers (A2) and trained models need true WF (A3).
3. Idle capital earns the risk-free rate (B1); trend is sized by an explicit decision (B2); the credit
   overlay is resolved (B3); live ≈ backtest is measured (B4).
4. At least the **crypto engine** is live-or-shadow and the **VRP track** is in live-paper (D1, D2),
   judged by the skew-pairing claim (§3).
5. The Norgate decision is made and, if bought, PEAD/F2 have *final* honest verdicts and futures/carry
   is tested properly (E1, E2).
6. The book runs as **trend ⊕ (VRP if ratified) ⊕ (one diversifier if it survives) ⊕ cash**, allocated
   by HRP under the portfolio governor (F1) — or, if nothing survives beyond trend, the honest fallback
   (trend + overlays + cash) runs *excellently*.

Either outcome is a win, because by then we'll *know* which one is true — which is the entire point of
the discipline this platform was built around.
