# Internal Opus Panel — Synthesis (2026-06-22)

**Session:** "while we wait for IBKR, work with LLMs to (1) find an overlooked trading method, (2) deepen
swing, (3) better-trade what we have, (4) make the app stronger." Brutal-honesty mandate.
**Method:** 5 independent Opus-4.8 agents that READ the actual repo — 4 panelists (one per area) + 1
red-team that attacked their proposals and verified contentious claims against code. This doc is the
synthesis; raw outputs are archived alongside (`panelist_*.md`, `redteam.md`).

---

## TL;DR — the uncomfortable headline (the red-team's blind-spot finding)

**On a ~$100k paper book with IBKR pending, the binding constraint is NOT alpha. It is (a) making the
safety layer you already BUILT actually load-bearing before IBKR capital, and (b) accruing live track
record.** The panel independently converged here:

- The live book has **almost no enforced book-level risk control.** The whole-book gate, the drawdown
  de-gross ladder, book-vol targeting, and the corr-spike de-gross **all exist in code and all run in
  shadow/research only.** The *only* live tail control today is the VIX crash governor + an 80% gross
  cap. The panel spent paragraphs debating "what vol to target (6/8/12%)" — **while nothing targets any
  vol on the live path at all.** Debating the thermostat on an unplugged furnace.
- **Every "new diversifier" proposed (reversal, FX carry/value, crypto, vol-managed momentum) is a
  premia harvester that converges to the same short-liquidity bet in a crisis** — re-violating the
  program's own hard-won "effectively one bet in a crisis" lesson. No one proposed a *long-convexity*
  crash hedge (the one thing the book is actually short).
- **Testing more families is self-harming right now:** the program built a family-wise-corrected
  Track-B gate to stop p-hacking; queuing 4 more families widens the correction and lowers every
  survivor's bar — and Panelist 2's "pre-screen on biased data first" would contaminate the clean test.
- On a $100k book, 4.7% vs 8% vol / EWMA vs 60d-std / a 5th sleeve are **economically rounding-error.**
  The dollar-binding constraints are capital + live track record + not blowing up on IBKR.

**So the honest answer to "what new method should we trade?" is: not much, yet.** The highest-EV moves
are free safety + free RFR; the best *new* idea is modest and should be pre-registered carefully, not
rushed. Details below, including the one new idea actually worth running.

---

## Verified findings (code-checked by the red-team + confirmed by me)

| Claim | Verdict | Evidence |
|---|---|---|
| Live trend book runs at ~4.7% vol = ~half the 8% policy target | **TRUE — by deliberate Kelly-haircut design** | `agent_config.py:310` default 0.50; description says "0.50 targets ~4.7% standalone vol" |
| **Doc drift:** `trend_sleeve.py` docstring says allocation 0.25 | **TRUE bug** | actual default is **0.50** (raised #P1-2 2026-06-16); `trend_sleeve.py:18-19` stale → fix |
| `reconciliation.py` + `kill_switch_state.py` are built but wired to NOTHING on the live path | **TRUE** | zero live `.py` imports (only tests/scripts/`.pyc`) |
| Cash-ETF mapping gap fail-closes trend in enforce mode | **TRUE but latent** | `cash_sleeve.CASH_ETFS`=8 vs `instrument_master`=3 (SGOV/BIL/SHV); fires only if `pm.cash_universe` changed off default |
| Short-horizon reversal is a "new" idea | **DEBUNKED** | already built + shelved as `app/strategy/reversal.py` (cost-dead, `cost_bps=10`); it's the killed intraday/reversal family |
| "Docs claim R0 is a hard no-go gate that's met" | **DEBUNKED** | docs say it's the gate *before IBKR capital* and correctly NOT yet met; shadow-only is the roadmap working as designed |
| Cash sleeve "bypasses the gate" is a risk | **MOSTLY DEBUNKED** | it only buys cash-equivalents excluded from gross by construction → immaterial |

---

## The plan — tiered by EV-per-effort and risk

### TIER 0 — Do now. Free, high-EV, low-risk. "Ship the safety you already wrote."
1. **Make the safety layer load-bearing (the real #1).** Wire the *new* `reconciliation.reconcile`
   (broker-vs-DB, fail-closed) + `kill_switch_state.KillSwitch` state machine onto the live order path
   (they're built + tested, just not imported). Add an alert when reconciliation breaks and when the
   gate fails to run. **This — not a new strategy — is the hard no-go gate before any IBKR dollar.** (M)
2. **Fix the cash-ETF mapping gap, then flip `pm.whole_book_gate_mode` → enforce** (after the planned
   shadow week incl. the Mon rebalance + first cash deploy). Register all 8 `cash_sleeve.CASH_ETFS` in
   `instrument_master` + `book_state._FACTOR_MAP{}`; add a subset test. (S)
3. **Verify the cash sleeve is actually draining idle cash to T-bills** (~50% NAV idle ≈ ~$2.5k/yr of
   free RFR at ~5%, zero risk, zero correlation). The one proposal that improves the dollar outcome
   with no added tail. (S, mostly verify)
4. **Wire the drawdown de-gross ladder into the live trend budget** as another fail-safe multiplier
   alongside the VIX/credit governors (pure function of equity-from-HWM, already PIT in research). This
   gives the book mechanical crash convexity without buying tail options. (M)
5. **Fix the allocation doc drift** (`trend_sleeve.py` 0.25 → 0.50). Trivial but anyone reasoning off
   "25%" is wrong. (S)

### TIER 1 — Cheap, pre-registered research (do ONE, resist doing all)
6. **Re-gate `crypto_trend` under the v2 Track-B appraisal as a declared `diversifier`** (it was killed
   on the *old* standalone-dSR test; corr-to-trend is already a measured **0.18** — it clears the
   corr<0.30 wall the others can't). Already built + OOS clock running. Cheapest real diversifier on the
   board. **Caveat:** ~5y history = power-floor fail for *capital*; it's a PAPER diversifier only. (S)
7. **The one NEW idea worth pre-registering: vol-managed single-name cross-sectional momentum**
   (Barroso/Santa-Clara — size 12-1 momentum inverse to the momentum factor's *own* realized variance).
   It's the one equity construction never run (the ranker was always constant-gross), and survival
   doesn't need the positive IC that died 3×. **BUT** run it on **clean Norgate data with the registered
   Track-B gate** — the red-team killed Panelist 2's "pre-screen on survivorship-biased data" because
   the bias *flatters this specific strategy* (it removes the crash-blowup names the vol-management is
   meant to dodge) → a screen that can only false-positive. High redundancy risk vs the trend book
   (also momentum) = the sector_rotation failure mode. **Data-gated on the $693 Norgate buy.** (M)

### TIER 2 — Deferred / data- or IBKR-gated
- **Short-horizon mean-reversion** (the only structurally anti-correlated family) — but it's the
  cost-dead `reversal.py`/`overnight` family; only revisit with a fresh *punitive-cost* CPCV + a
  no-trade band, and check the GL-1 *exceedance* corr (it diversifies in calm, gets run over in trends).
- **G10 FX value** (mean-reverting, structurally different from the commodity-heavy carry sleeve) —
  free on owned Norgate futures, slots into the P2 IBKR pipeline; FX *carry* is likely post-2008-dead.
- **Equity dispersion / implied-correlation premium** — genuinely orthogonal but needs an options-data
  buy + a mature NBBO log; probably the killed options program in a trench coat. Not now.
- **Better-trade knobs** (EWMA λ=0.94 vol; skip-month on the live TSMOM signal; rebalance band) — real
  but second-order on a $100k book; pre-register and batch them, don't rush.

### DO NOT
- **Do NOT turn book-vol-targeting up to 8%** (Panelist 3's item 2). It levers the sole live,
  undiversified edge ~1.7× via a mechanism that's biggest right before vol spikes
  (`book_vol_max_leverage=2.0`), just as correlated IBKR futures beta is about to stack on top, with the
  ladder/de-gross still in shadow. Optimizes a statistically-irrelevant Sharpe delta while maximizing
  the exact joint-crisis-drawdown risk the philosophy says to avoid. (Both the red-team's #1 "don't.")
- **Do NOT re-animate the shelved reversal sleeve** on a calm-regime correlation that inverts in trends.
- **Do NOT pre-screen new strategies on survivorship-biased data** (contaminates the clean test;
  false-positive-only for the momentum variant).
- **Do NOT build the Constructor / ERC / covariance / tail-corr-governor stack** — no second live sleeve
  to observe a covariance from; inverse-vol is terminal until R2 (≥2 strategies × ≥6mo joint live).

---

## How this maps to the standing Alpha-v10 plan
Tier 0 = R0.5 enforce + the R0 reconciliation/kill-switch wiring that the roadmap already names as the
**hard IBKR no-go gate** — the panel just proved it's not actually wired yet. Tier 1.6 (crypto re-gate)
and Tier 1.7 (vol-managed momentum on Norgate) fold into the existing family-registry/Track-B
discipline and the parked Norgate-data accumulator. Nothing here is off-plan; the panel sharpened the
*sequencing*: **safety before strategies, RFR before Sharpe-knobs, and the next data buy (Norgate)
serves Option B + A-v2 + vol-managed-momentum together.**

> Raw panelist outputs: `panelist_1_overlooked_methods.md`, `panelist_2_swing.md`,
> `panelist_3_better_trade.md`, `panelist_4_robustness.md`, `redteam.md`. External 5-LLM pack:
> `cover.md`, `snapshot.md`, `questions.md`.
