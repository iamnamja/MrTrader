# 02 — MrTrader system design (how everything works)

*Self-contained overview. Appendices `05`–`07` have the authoritative deep detail.*

---

## 0. One-paragraph picture
MrTrader is a fully-automated systematic trading platform for US equities/ETFs (Alpaca **paper**,
$100k notional). A Python/FastAPI process runs a scheduler that, on a fixed cadence, asks three
cooperating "agents" — a **Portfolio Manager (PM)**, a **Risk Manager (RM)**, and a **Trader** — to
turn signals into orders. New signals/strategies are *not* added by intuition: each is researched in
a uniform offline harness (the **Sleeve Lab**) under walk-forward + combinatorial-purged
cross-validation (CPCV), and must clear a strict promotion gate (**Ruler-v2**) and a both-halves
stability guard before it can touch live capital. Today the only thing live is a **trend sleeve** (10-ETF
time-series momentum) at 25% allocation with a VIX-term-structure **crash governor** overlay.

---

## 1. Runtime architecture (the three agents)

The agents are not LLMs — they are deterministic Python components with clear responsibilities.
(Full runtime spec: `06_SYSTEM_BEHAVIOR.md`.)

### Portfolio Manager (PM) — *selection & sizing*
- Decides **what to hold and how much**. Reads enabled sleeves/strategies + their target weights.
- Currently: runs the **trend sleeve** rebalance (Mon 09:45 ET), producing target ETF weights from
  the TSMOM model, then applies any active **overlay multipliers** (crash governor; credit overlay is
  wired but flag-OFF) to scale gross exposure. Allocation is 25% of book to trend, remainder cash.
- Feature-flagged via a DB `agent_config` table (`pm.trend_enabled`, `pm.trend_allocation_pct`,
  `pm.crash_governor_enabled`, `pm.credit_governor_enabled`, `pm.swing_selector`, etc.).

### Risk Manager (RM) — *gates & guardrails*
- Vetoes/clamps orders for risk: position caps, gross/net exposure limits, a global de-risk floor
  (overlays can never cut exposure below 25% of target — `GLOBAL_DERISK_FLOOR=0.25`), liquidity/
  notional sanity, and kill-switch conditions.
- Enforces that nothing trades unless its strategy passed the offline gate and is flagged live.

### Trader — *execution*
- Converts approved target weights → orders, routes to Alpaca, manages order lifecycle (market/limit,
  fills, partials), and reconciles positions vs. targets. Models commissions/slippage in backtests so
  research P&L is net.

The PM/RM/Trader loop runs **inside the uvicorn/FastAPI process** (`app/main.py`) via an in-process
orchestrator/scheduler — there is no separate worker. (Restarting live logic = restart uvicorn.)

---

## 2. The trading flow (signal → fill)

```
 data refresh  →  PM builds target weights  →  overlays scale exposure
      │                   │                            │
   (EOD prices,        (sleeve model:               (crash governor /
    yields, macro)      e.g. TSMOM)                  credit overlay)
      ▼                                                    ▼
                       RM gates (caps, floors, kill-switch)
                                   │
                                   ▼
                        Trader → Alpaca orders → fills
                                   │
                                   ▼
                    reconcile positions, log, MTM equity
```

- **Cadence:** trend rebalances weekly (Mon 09:45 ET). Equity is marked-to-market daily.
- **Live now:** trend sleeve only. Swing-ML selection (`pm.swing_selector=ml_model`) and the PEAD/
  event path exist in code but are **not contributing live capital** (PEAD demoted; ML swing has no
  validated edge). Credit overlay is wired but **flag-OFF** pending owner decision.

---

## 3. Strategy research: the Sleeve Lab

All research flows through one uniform harness — `scripts/walkforward/sleeve_lab.py`. This is the key
to avoiding ad-hoc, overfit "it looked good in a notebook" decisions.

- **`Sleeve`** = a standalone return stream (a strategy producing a daily/weekly return series).
- **`evaluate_sleeve`** runs it through **CPCV → Ruler-v2** and returns a `SleeveReport`.
- **`@register_sleeve`** registry makes every sleeve runnable uniformly; **`assemble_book`** combines
  accepted sleeves into a vol-targeted book.
- **Overlay path** (added in the latest program): an **`Overlay`** is *not* a return stream — it's a
  **multiplier on an existing book's exposure** (e.g. "cut trend to 50% when the VIX term structure
  inverts"). `evaluate_overlay` / `evaluate_overlay_marginal` judge an overlay's *marginal* value on
  top of the already-overlaid book; `compose_overlays` stacks multipliers multiplicatively, clamped to
  the `GLOBAL_DERISK_FLOOR=0.25`. `@register_overlay` registry.

This sleeve-vs-overlay distinction matters: we found **additive sleeves keep hitting a correlation/IC
wall, while overlays (timing/de-risking the trend book) are where marginal value still showed up.**

---

## 4. Model training (swing ML path)

The swing-ML path (built early, now without a validated edge but still the template) works as:
- **Universe:** liquid US equities/ETFs.
- **Features:** daily technical/price-volume features (momentum, vol, mean-reversion, breadth,
  microstructure proxies), plus optional fundamentals (FMP) and macro context. (Fundamentals are
  disabled in training — `--no-fundamentals` — to prevent OOM; training uses `--workers 8`.)
- **Targets:** forward-return / forward-direction over the swing horizon, with triple-barrier-style
  labeling in places.
- **Models:** gradient-boosted trees (primary) + simpler baselines; selection via the WF/CPCV harness,
  not in-sample fit.
- **Verdict:** daily-feature **IC ≈ 0** repeatedly — no additive equity alpha from free daily data
  (see `03_RESEARCH_HISTORY.md`). The infra is sound; the *signal* isn't there.

---

## 5. Validation harness — WF + CPCV (the part to scrutinize hardest)

*(Authoritative: `05_PIPELINE_ARCHITECTURE.md`.)*

- **Walk-forward (WF):** expanding/rolling train→test splits in time; no test bar is ever used to fit
  the model that trades it.
- **CPCV (combinatorial purged cross-validation):** n_folds=8, n_paths=2; **purge** removes train
  samples whose label windows overlap the test block; **embargo** drops samples right after the test
  block. (Purge = 85 calendar days swing / 2 trading days intraday; embargo similar.) Produces many
  backtest *paths* rather than one, so the Sharpe distribution — not a single number — is judged.
- **Costs:** commissions + slippage modeled; research returns are net.
- **Sacred holdout:** a final out-of-sample period (2026-11-09 cutoff) never touched in research.
- **Simulators:** `AgentSimulator` (swing WF/CPCV, daily MTM), `IntradayAgentSimulator` (intraday
  WF/CPCV, daily equity). `StrategySimulator` is tier-2 only (not used in WF/CPCV).

### Gate thresholds (legacy gate, quick reference)
| Gate | Swing | Intraday |
|---|---|---|
| Avg Sharpe | ≥ 0.80 | ≥ 1.00 |
| Min fold Sharpe | ≥ −0.30 | ≥ −0.30 |
| DSR p-value (deflated Sharpe) | > 0.95 | > 0.95 |
| Profit factor | ≥ 1.10 | ≥ 1.10 |
| Calmar | ≥ 0.30 | ≥ 0.30 |
| N trials tested (for DSR) | 300 | 300 |

---

## 6. The promotion gate — Ruler-v2 (two-track)

*(Authoritative: `07_RULER_V2_DESIGN.md`.)* The legacy gate above was found to be a **Type-II /
false-negative machine** (too eager to kill). Ruler-v2 replaced it with a two-track acceptance:

**Track-A — standalone edge** (can this stream stand on its own?)
- *PAPER tier:* point Sharpe ≥ 0.30 + a light HAC (Newey-West) 1-sided Sharpe test p<0.05 + a regime
  waiver for genuine diversifiers / risk-premia sleeves.
- *CAPITAL tier (live-money bar):* Bayesian posterior **P(SR>0) ≥ 0.95**, a structural live-paper
  check, a multi-factor **residual-α t ≥ 2.0** (alpha after regressing out standard factors), a
  bootstrap P(SR>0) ≥ 0.95, and a **power floor** (n_obs ≥ 504, n_folds ≥ 10) — so we don't "accept"
  something we never had the statistical power to judge.

**Track-B — book-delta** (does adding it improve the *existing* live book, even if weak alone?)
- appraisal IR ≥ 0.20, P(ΔSR>0) ≥ 0.90, correlation to current book < 0.30, standalone vol-targeted
  SR > 0.20.

**Decisions** map to a research registry (kill / park / promote_paper / live / exploratory_only).

---

## 7. Research discipline (process guardrails)

- **Pre-registration (R7 registry):** confirmatory runs are registered (`preregistered_at < run_at`)
  *before* running; one shot; a re-test requires a new id + cooling-off. Kills HARKing/p-hacking.
- **Both-halves stability guard (mandatory):** split the sample at its midpoint; the edge must be
  positive in **both** halves. (This alone killed carry.)
- **Independent adversarial review:** before any verdict is recorded, an independent deep-dive
  (a fresh model instance) tries to *break* the result (look-ahead, leakage, fragility).
- **No-drift docs:** every change updates all living docs it makes stale, in the same PR.
- **Capacity & cost realism:** turnover, slippage, and notional capacity are part of the verdict.

---

## 8. Current live state (as of 2026-06-16)
- **Trend sleeve:** LIVE (paper), 25% allocation + cash, weekly rebalance.
- **Crash governor (VIX term-structure overlay):** LIVE (flag ON).
- **Credit-selective overlay:** validated as a CANDIDATE but **flag-OFF** (owner decision pending).
- **Swing ML / PEAD / intraday / options / short-interest / carry:** all **off** (killed/parked).
- **Net:** one durable edge (trend) + one or two protective overlays. The search for *additive* edges
  has repeatedly come up empty on free daily US data — which is exactly what we want your help with.
