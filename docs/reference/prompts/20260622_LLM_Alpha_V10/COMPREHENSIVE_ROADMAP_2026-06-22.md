# Comprehensive Roadmap — Alpha-v10 "while we wait for IBKR" (2026-06-22)

**Synthesis of 10 independent perspectives:** 4 internal repo-grounded Opus panelists + 1 adversarial
red-team + 5 external world-class-quant reviews (ChatGPT, Claude, DeepSeek, Gemini, Grok). Brutal-honesty
mandate; the externals were explicitly asked to *attack* the internal panel's conclusions.

> **The one-sentence verdict (UNANIMOUS, 10/10):** Stop hunting a 5th sleeve. For the next 1–3 months the
> binding constraint is **capital + a real (tiny-live) track record + not blowing up** — so make the safety
> layer you already built *load-bearing*, deploy the edge you already have, and treat every new-strategy
> idea as post-hardening, time-boxed, and pre-registered. The new-sleeve hunt is the comfortable activity;
> hardening + deploying real capital is the uncomfortable, correct one.

---

## 1. Where all 10 agreed (the mandate — not negotiable)

1. **Binding constraint is not alpha.** Every reviewer, unprompted, said the same: capital + live track
   record + operational survival dominate any 5th-sleeve EV. Time-box it (≈1–3 months), not forever.
2. **Make the safety layer load-bearing BEFORE any IBKR capital.** Reconciliation-before-trade
   (fail-closed, broker=truth), kill-switch state machine wired, whole-book gate shadow→enforce. The
   roadmap already names this the hard IBKR no-go gate; the panel proved it's not actually wired yet.
3. **The #1 futures-specific killer = contract-spec / multiplier error** (multipliers vary ~50× across
   ES/NQ/GC/CL/ZB). Verify-on-connect + a **dollar-denominated** max-notional pre-trade check that fires
   even if a multiplier is wrong. ChatGPT and Claude both rank this #1; the internal panel underweighted it.
4. **Out-of-band broker-only flatten + external dead-man watchdog are MANDATORY before futures** — not in
   the internal panel's headline; all 5 externals flagged them as non-negotiable. (Min was already bitten
   by an overnight Windows-Update restart — this is not hypothetical.)
5. **Per-order idempotency (client order IDs)** before futures, so a retry can't double-send.
6. **Do NOT turn vol-targeting up to 8%.** Universal veto. Run ~5–6% max, and only after safety is enforced.
7. **Swing equity is a sunk-cost trap** for this stack. Don't buy Norgate *for* vol-managed momentum;
   expect any single-name momentum to fail the corr<0.30 gate (like sector_rotation at 0.51).
8. **Positive-carry crash convexity is mostly a unicorn** (ChatGPT "mirage", Claude "fantasy", Grok "no free
   lunch"). The real version is *dynamic/timed* convexity (defensive trend), not a static sleeve.
9. **Classic short-horizon reversal is dead** for liquid ETFs at 2 bps.
10. **Don't over-build for a $100k book:** no unified Constructor, no covariance/ERC stack (no joint live
    history to estimate from), no HA/streaming-risk/execution-algos/FIX. "Fail safely, don't fail over."

---

## 2. The roadmap — three phases (H → D → R)

### PHASE H — HARDEN (next ~2 weeks; the hard no-go gate before IBKR capital)
*This is R0.5-enforce + the R0 wiring the roadmap already promised, PLUS the futures-specific items the
external panel added. All before one IBKR dollar.*

| # | Item | Source consensus | Effort |
|---|---|---|---|
| H1 | **Reconciliation-before-trade**, fail-closed, broker=truth, on EVERY live order path (wire the built-but-inert `reconciliation.py`) | 10/10 | M |
| H2 | **Wire the kill-switch state machine** + fix the cash-ETF mapping gap + flip `whole_book_gate_mode` shadow→enforce after a clean shadow week | 10/10 | M |
| H3 | **Pre-trade order sanity: dollar-notional cap + futures-multiplier verify-on-connect + max-order-size**, fail-closed (separate layer from reconciliation) | ChatGPT/Claude #1, DeepSeek, Grok | M |
| H4 | **Out-of-band broker-only flatten** (no DB/Redis/app dependency), **tested weekly** on the Alpaca paper path now | all 5 externals | M |
| H5 | **External dead-man watchdog** (separate process/box; heartbeat → call/SMS, optional auto-flatten) | all 5 externals | S–M |
| H6 | **Per-order idempotency** (client order IDs) + transactional order log | ChatGPT, Claude, DeepSeek, Grok | S–M |
| H7 | **Broker-side dollar limits** (IBKR precautionary settings) your code cannot override — independent backstop | Claude, ChatGPT | S |
| H8 | **Tiered alerting** (catastrophic→call/SMS, warning→push, info→daily digest) + reconciliation-break alert + "gate-didn't-run" alert | all | S |
| H9 | **Wire the drawdown de-gross ladder into the live budget** — simplified to 2–3 steps, triggered on *smoothed/broker-confirmed* equity drawdown with hysteresis | ChatGPT/DeepSeek/Grok yes; Claude "mostly forward value"; Gemini dissents | S–M |
| H10 | **Fix the doc-drift bug** (allocation 0.25→0.50 in `trend_sleeve.py`) + register all 8 `cash_sleeve.CASH_ETFS` in `instrument_master`/`book_state` | red-team / P4 | S |

> **Contested (H9 drawdown ladder):** Gemini argues a rigid ladder is "a slow stop-loss that de-grosses at
> the bottom." That objection bites a *mean-reverting* book; trend is *not* mean-reverting (it's already
> exiting risk into a downtrend), so the ladder is additive here — but build it smoothed, 2–3 steps, on
> broker-confirmed NAV, with re-risk cooldown. Its biggest value is *forward* (a non-equity sleeve breaking
> won't trip the equity-vol VIX governor).

### PHASE D — DEPLOY (start the real clock; on IBKR approval)
*Paper track record is nearly worthless for what matters — fills, slippage, operational reality, and your
own behavior under live P&L. The plan must END in tiny REAL capital, not more paper.*

| # | Item | Notes |
|---|---|---|
| D1 | **Step trend ~4.7% → ~6% vol WITH a hard leverage cap** (~1.3–1.5×) — the one "do more" return item; cap defuses the pro-cyclical objection. **Only after H is done.** | Claude's synthesis; vol target = f(diversification) |
| D2 | **Verify the cash sleeve is draining idle cash to T-bills** (free RFR, zero corr) | all |
| D3 | **Pre-register the IBKR tiny-live launch plan** (exact instruments, max contracts, margin reserve, no-trade conditions, rollback, 30-day probation) → IBKR paper (full cycle, logs reviewed) → **microscopic** live futures (carry+xsmom; micro contracts; if even micros are too chunky for a sleeve, keep it paper) | ChatGPT/Claude/Grok |
| D4 | **Sleeve combination = inverse-vol + per-sleeve cap (≤60% risk) + ISOLATED margin pools** (don't let trend PnL fund futures margin); never MVO; shrink toward covariance only as *live* joint history accrues | all; Gemini's margin-isolation point + Claude's "trust the backtest correlation, distrust the means" |
| D5 | **Bucket "robust risk premia" (trend, carry) vs "candidate alpha" (xsmom, VRP, anything new)** — smaller risk budget + higher discount for the latter until it earns live track record | DeepSeek |

### PHASE R — RESEARCH (ONLY after Phase H; time-boxed, pre-registered, ONE at a time)
*The external panel materially upgraded the research backlog. Ranked by cross-panel support.*

1. **Un-handicap trend / defensive-macro crisis overlay — HIGHEST EV, and it's barely "new."**
   (Claude #1 + ChatGPT #1, independently.) The live trend sleeve is **long-flat on an equity-heavy
   universe**, which strips trend of its convex legs (short equity, long bond/gold/USD in a flight to
   quality). *"You're running the convexity-poor leg of trend."* Once IBKR lands, run trend **long-short on
   futures** including those legs, and/or a **declared crisis-defensive overlay** (rates/gold/USD/defensive-
   FX trend, equity-index short only when equity trend is negative), gated on a pre-registered crisis score
   (VIX term + equity-below-trend + realized-vol-shock + credit proxy). **Evaluate on conditional-crisis
   behavior and marginal tail reduction, NOT standalone Sharpe.** Caveat (honest): this is *timed* convexity
   — it helps in slow de-risking/liquidity cascades, not one-day gap crashes, and the bond/gold leg failed
   in the inflationary 2022 co-crash → long-USD-trend is the most robust convex leg.

2. **Commodity calendar-spread / seasonal-storage premia — the most genuinely-distinct family you own data for.**
   (ChatGPT's desk-level idea; Grok partial.) Trade *spread* risk (CL M1–M3, RB/HO seasonal, grains
   harvest/storage), normalized vs same-calendar-month history, 2–8 wk holds, on the Norgate futures already
   owned. Different economic payer (storage/hedgers/seasonality), not equity-beta-in-disguise. A serious
   research branch (not a 2-day job); strict delivery/roll/margin-stress rules. Pre-register: post-2015
   net-of-spread-cost Sharpe > 0.40, ≥4 unrelated commodities contribute, corr-to-book < 0.25, survives
   ex-best-commodity & ex-best-decade.

3. **VIX-gated stress reversal — CONTESTED but the sharpest single new idea.**
   (Claude.) Reversal as a *liquidity-provision premium* (Nagel) that **only fires when VIX is elevated** —
   trades rarely (survives cost), pays in stress, counter-cyclical (you provide liquidity buying the crash
   → convexity-positive). The internal panel killed the *unconditional* form; this is the conditional form.
   **Dissent:** DeepSeek/Grok think even gated reversal overlaps killed value or is weak post-2015. → Pre-
   register VIX-threshold + holding period (no post-hoc tuning); hard per-name stop + strict gross cap
   (you're catching a falling knife); expect a possible kill. Worth one clean test because it's the only
   candidate that is simultaneously distinct, cost-surviving, AND convexity-positive.

4. **G10 FX value (as a hedge to FX carry) — low conviction, one clean test, expect park.**
   Broad agreement it's genuinely mean-reverting/divergent but weak post-2015. Best as the value leg paired
   with FX carry (Asness — value rallies when carry unwinds). Pre-register a single PPP/real-rate metric,
   G10-only, test the post-2015 OOS window; the overfitting tell = needing a momentum/carry overlay to make
   it work (then it's just trend/carry relabeled). Don't lead with it.

5. **Lower-priority / single-proponent (flagged, not prioritized):** VIX trend-following (DeepSeek — but
   beware the brutal negative roll carry of long VIX, which ChatGPT/Claude would veto as a bleed); curve-
   steepener trend & intraday TSMOM (Gemini — intraday avoids the overnight-gap short-convexity bleed;
   interesting, but `intraday_ml` is already a deprioritized scaffold); ETF-pair cointegration (Gemini — but
   `etf_relative_value` already KILLED).

---

## 3. The meta-critiques to internalize (the E3 answers — these matter more than any sleeve)

- **C1 — Operator-capacity & behavioral risk is the dominant UNMODELED risk (Claude — the single most
  important point in the whole review).** You deflate obsessively for measurable risk (multiple-testing) and
  *not at all* for the risk that will actually set your live result: a solo operator, running a demanding
  day job, parenting, who **will override/pause under real-P&L fear**, whose process has no ops team or
  redundancy. Backtest 0.72 → plausible solo-live **0.4–0.5** after this drag. **Mitigations:** the dead-man
  + out-of-band flatten (so you don't *have* to intervene), keep the system boring (every added sleeve
  raises operator load), pre-commit to NO discretionary override, and **assume the haircut** when sizing.
- **C2 — Even N=26 is optimistic (Claude + DeepSeek).** The true search is in the *hundreds* (your own
  appendix: "6 factors screened", "12-cell grid", "9+ ML iterations", "pre-reg FAILED → post-hoc
  selected"). The DSR cross-check vs N=26 is **under-deflated** → carry/xsmom are *less* significant than
  the deflated numbers imply. **Action:** raise/disclose the DSR N (toward the ML-path N=300 spirit), mark
  the second-engine confidence DOWN, "size modestly" is even more right.
- **C3 — You may have overfit the ERA, not the data (Gemini).** Your validation geometry is mathematically
  airtight on *data* overfitting but blind to having selected models that thrived in a 2006–2025 central-
  bank-liquidity regime. **Action:** weight the post-2015 + inflationary-2022 sub-periods heavily; treat
  premia persistence with humility; explicit regime-robustness caveats.
- **C4 — "Risk premia not alpha" is partly a euphemism (DeepSeek).** xsmom / the VIX governor / VRP are
  anomalies that may be behavioral or data-mined, not structural risk premia. **Action:** the D5 bucketing —
  robust premia (trend, carry) at full discipline; candidate anomalies (xsmom, VRP, new ideas) at a higher
  discount rate + smaller budget until live-proven.
- **C5 — The language is ahead of the executable truth (ChatGPT).** "Built", "second engine", "whole-book
  gate", "portfolio brain" describe a system the *live order path doesn't use*. **Action:** stop crediting
  shadow/research code as "live"; don't capitalize paper-futures Sharpe before real fills; update the docs
  to state the true wired-vs-built status (this synthesis already does).

---

## 4. The DON'T list (consensus — actively avoid)

- ❌ **Raise vol to 8%** (esp. before hardening) — the single most dangerous item on the table.
- ❌ **Reopen swing-equity ML / buy Norgate for vol-managed momentum** — redundant + infrastructure-gated.
- ❌ **Touch the live TSMOM signal (skip-month)** for marginal gains — it's in the noise for TS-momentum;
  if you want better crash behavior, do **asymmetric rebalance** (slow to add, fast to cut) instead.
- ❌ **Build the Constructor / covariance-ERC stack / HA / streaming-risk / execution-algos / FIX** at $100k.
- ❌ **Re-gate crypto-trend or add options dispersion as "diversifiers"** — both are short-convexity in a
  crisis (corr→1 when it matters); they re-violate "one bet in a crisis" while *feeling* diversifying.
- ❌ **Capitalize paper-futures Sharpe** before it survives IBKR paper → tiny-live → real rolls/margin/fills.
- ❌ **VIX-VRP back into the live book** while tails still concentrate (already dropped per GL-1).

---

## 5. The concrete next-two-weeks (forced ranking, IBKR pending)

1. **Make the safety layer load-bearing** — H1 (reconciliation), H2 (kill-switch + gate→enforce after the
   mapping fix), H3 (pre-trade dollar-notional + multiplier sanity). *Converts the architecture from
   aspirational to real; precondition for everything.*
2. **Build & TEST the emergency machinery on the Alpaca paper path now** — H4 (out-of-band flatten, tested
   weekly) + H5 (external dead-man). *Have it proven before futures lands.*
3. **Step trend to ~6% with a hard leverage cap** (D1) + **verify cash→T-bills** (D2). *The only "do more";
   harvests the under-deployed validated edge + free RFR, zero new alpha.*

Plus the cheap cleanups whenever (H10 doc-drift + cash mapping; H6 idempotency; H8 alerting).

**Most dangerous if done:** vol-to-8% before hardening. **Most wasteful:** the skip-month tweak and any new
single-name/sector test. **Highest-EV *research* (post-hardening):** recover trend's convexity via long-
short futures + a defensive-macro crisis overlay — not a new family, the convexity you already own.

---

## 6. How this maps to the existing Alpha-v10 / Portfolio-Brain plan
Phase H = **R0.5 enforce + the R0 reconciliation/kill-switch wiring** (already the named IBKR no-go gate),
*plus* the futures-specific H3–H7 items the external panel added (multiplier verify, out-of-band flatten,
dead-man, idempotency, broker-side limits). Phase D = **R1** (IBKR tiny-live carry+xsmom). Phase R replaces
the old "find a 5th sleeve" backlog with **"recover convexity (long-short trend / defensive overlay) →
commodity calendar spreads → VIX-gated reversal → FX value."** Phase B/C of multi-strat (covariance/ERC)
stays **R2, data-gated** — the panel emphatically agrees: don't build it before ≥2 live sleeves exist.

> Inputs: `INTERNAL_PANEL_SYNTHESIS.md`, `panelist_{1..4}.md`, `redteam.md`, `responses/*.md`.
> Next: fold this into `DECISIONS.md` + `MASTER_BACKLOG.md` + `PROJECT_STATE.md` (the Phase-H checklist
> becomes the active backlog; the DON'T list becomes a guardrail entry).
