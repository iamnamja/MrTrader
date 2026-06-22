# Holistic multi-strategy — current state + scope to finish "multi-strat eval" (2026-06-22)

Captures (Part 1) where "do we see things holistically across strategies?" stands today on BOTH
the live path and the research/backtest path, and (Part 2) the concrete scope to finish a unified
multi-strategy evaluation harness. Single source of truth for this question.

---

## Part 1 — Current state

### A. LIVE path (PM / RM / Trader runtime)
| Layer | State | Where |
|---|---|---|
| **Holistic measurement** — one consolidated cross-venue book with NETTED factor exposures (equity beta, rates DV01, USD, commodity, vol) | ✅ BUILT (shadow/read-only); validated on live Alpaca | R0.4 `book_state.py`, R0.4b `run_book_state_report.py` |
| **Holistic risk cap** — a whole-book gate evaluating the PROPOSED book vs risk-policy caps | ✅ BUILT, wired into the live trend rebalance in **SHADOW** (logs, blocks nothing) | R0.5 `whole_book_gate.py` |
| **Holistic enforcement** | ⏳ flips shadow→enforce after the ~1-week evidence window (~2026-06-29) | flag `pm.whole_book_gate_mode` |
| **Single chokepoint for ALL strategies** | ❌ gate currently sees the trend path only (cash is cash-equivalent → exempt); live sleeves still bypass the RiskManager | — |
| **Joint, correlation-aware SIZING** (strategies sized taking each other into account) | ❌ NOT built — this is **R2 (the unified Constructor)**, deliberately DATA-GATED (≥2 strategies × ≥6mo joint LIVE returns); inverse-vol terminal until then | Portfolio-Brain roadmap R2 |

> Note: today's "trend/cash sleeve isolation" bugfix was the *opposite* of coordination — it ensures
> the swing-review doesn't *mismanage* trend/cash (correctness), not joint sizing.

### B. RESEARCH path (backtest / walk-forward) — further along than live
| Capability | State | Where |
|---|---|---|
| **Combine N strategies into one book** — PIT **inverse-vol** (or equal / regime-tilt) weighting, weekly-rebalanced, turnover-costed; weights are DYNAMIC (rolling 60d vol), not static | ✅ BUILT | `assemble_book` → `sleeve_allocator.build_book` |
| **Marginal contribution to the existing book** (Track-B residual-alpha) — the standard keep/kill: "does this add alpha *given* what we already hold?" | ✅ BUILT + is the default discipline (carry, xsmom, credit overlay all judged this way) | `evaluate_sleeve(base_book_returns=...)`, `track_b_appraisal` |
| **Multi-strategy combination evaluated as a unit** | ✅ DONE ad-hoc (`futures_book` = carry+xsmom, Book Sharpe 0.67, residual-α t 2.29) | `sleeves.py` |
| **Cross-strategy tail / co-crash** (exceedance corr, down/up-beta asymmetry, joint crisis replay) | ✅ BUILT | GL-1 `tail_diagnostics.py` |
| **Family-level multiple-testing deflation** of the book | ✅ BUILT | P0.5 `family_registry.py` → `null_zoo` DSR |
| **Unified combined-book CPCV-as-a-unit** (the book itself run through the SAME purged folds + deflation as a single sleeve → a trustworthy book-level WF verdict, not just descriptive stats) | ❌ NOT a single harness | — |
| **Whole-book risk gate applied IN the backtest** (so backtest = live gate) | ❌ gate is live-only today | — |
| **Correlation/covariance-aware sizing** in research | ❌ inverse-vol ignores cross-correlations (ERC only ≈ for low-corr sleeves) — same data-gate as live R2 | — |
| **One entry point + report** to "evaluate this SET of strategies as a book" | ❌ pieces exist (assemble_book / Track-B / GL-1 / DSR) but not unified | — |

**Summary:** holistic *measurement* (live) and holistic *contribution + tail testing* (research) exist
and are the standard discipline. What's missing on both sides is the same thing: **joint
covariance-aware sizing** (deferred to ≥2-strat × ≥6mo live data) and, on research, a **single unified
combined-book walk-forward harness** with the live gate applied.

---

## Part 2 — Scope: what to build for "multi-strat eval done"

### Phase A — Unified combined-book WF harness  ·  FREE / no live dependency / buildable NOW
Assemble the existing pieces (assemble_book + Track-B + GL-1 + the gate + P0.5 deflation) into one
entry point + report, and add the two genuinely-missing capabilities (combined-book CPCV + gate-in-
backtest). This is the bulk of "research multi-strat eval."

- **A1 — `BookSpec` entry**: take a list of registered strategies + an allocation scheme → the
  combined daily book returns (reuse `assemble_book`/`build_book`).
- **A2 — Combined-book CPCV/WF**: run the assembled book through the SAME purged/embargoed CPCV folds
  the single-sleeve gate uses, with the family-count (P0.5) deflated-Sharpe — so the BOOK gets a
  trustworthy WF verdict, not just an in-sample Sharpe. (New: today CPCV is per-sleeve.)
- **A3 — Whole-book risk gate in the backtest**: apply the R0.5 risk-policy caps inside the book
  backtest (de-gross when a proposed book breaches), so research and live share ONE gate (replay
  parity foundation). (New: the gate is live-only today.)
- **A4 — One book report**: combined Sharpe/Calmar/maxDD + **per-sleeve attribution** + each sleeve's
  **Track-B marginal** + **cross-strategy tail** (exceedance corr / co-crash, GL-1) + the family-level
  DSR — a single artifact. (Unify; pieces exist.)
- **A5 — Ragged-history modes**: today `assemble_book` inner-joins to the common window (truncates to
  the shortest sleeve, with a loud warning). Add a "fold sleeves in as they come online" mode and
  report BOTH (common-window for clean comparison; fold-in for max data).
- **A6 — `run_multistrat_eval.py` + tests** mirroring the existing runners.

*Effort:* moderate — A1/A4/A5 are largely wiring existing components; A2/A3 are the real new work
(combined-book CPCV + porting the gate into the backtest). No external dependency.

### Phase B — Correlation-aware sizing (the joint Constructor)  ·  DATA-GATED (= R2)
- **B1 — ERC / covariance allocator** with shrinkage, as a new scheme in `sleeve_allocator`.
  **Gate:** activated ONLY once ≥2 strategies have ≥6mo of JOINT LIVE returns (can't shrink a
  covariance never observed). Inverse-vol stays terminal until then.
- **B2 — De-gross governors in the sizing**: drawdown ladder + **stress-conditional** tail-correlation
  (the GL-1 matrix, measured on stress days, NOT a 63d average), as monotonic risk reducers.
- **B3 — No Markowitz/HRP/Kelly** — ERC + dumb governors only (panel discipline).

### Phase C — Replay parity (research ↔ live)  ·  gated on R2 / IBKR
- **C1 — One allocation+gate code path** shared by the backtest harness (Phase A) AND the live
  Constructor, so a capital-eligible strategy shows matching backtest-vs-live behavior — the
  "StrategySpec replay-diff: 4 boring weekly diffs → capital-eligible" contract (MASTER_BACKLOG
  cross-cutting). This is what makes the combined-book WF *trustworthy as a live predictor*.

### Sequencing & what each unblocks
1. **Phase A now** (free) → adding a strategy becomes "register → `run_multistrat_eval` → read the
   book-level WF + marginal + tail report." Gives holistic *research* eval immediately, and builds the
   gate-in-backtest that Phase C needs.
2. **Phase B (B1)** waits on live data (R2 trigger); B2 governors can be specced alongside A.
3. **Phase C** lands with R1/R2 (IBKR + the live Constructor) to close the research↔live loop.

> Cross-refs: `PORTFOLIO_BRAIN_ROADMAP_2026-06-21.md` (R2 Constructor, data-gating), `R0_FOUNDATION_2026-06-21.md`
> (R0.4/R0.5), `GL0_GL1_FINDINGS_2026-06-21.md` (Track-B + tail), `P0_5_FAMILY_REGISTRY_2026-06-22.md`
> (deflation N), MASTER_BACKLOG (StrategySpec replay-diff). Live book UNCHANGED — this is a scope doc.
