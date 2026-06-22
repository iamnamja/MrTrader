# Red-Team — Adversarial Adjudication of the 4-Panelist Review (raw output, Opus 4.8, 2026-06-22)

> Mandate: attack the strongest proposals, verify contentious claims against code, find what all four
> missed. Brutal, adversarial. Willing to say "the panel is wrong about X."

---

# Red-Team Adjudication — MrTrader 4-Panelist Review

## Claims verified / debunked (with code evidence)

**✅ VERIFIED — P3: "trend book runs at ~4.7% vol = half its 8% target."** `agent_config.py:309-314`:
100%-gross book ann vol 9.34%, Sharpe 0.72 on 19.4y; `pm.trend_allocation_pct` default **0.50**,
described as "targets ~4.7% standalone vol — deeply Kelly-haircut." The sleeve never sets
`book_vol_target`. The number is right; the framing understates that it's an *explicit Kelly-haircut*,
not an oversight.

**⚠️ DOC DRIFT — allocation is 0.50, not 0.25.** `trend_sleeve.py:18` docstring + `PROJECT_STATE.md:101`
say 0.25; `agent_config.py:310` default is **0.50** (raised P1-2 2026-06-16, after the 0.25
reconciliation). The docstring is stale. **Anyone reasoning from "25%" is wrong** — the gap to 8% vol is
much smaller than the panel thinks.

**✅ VERIFIED — P4: reconciliation.py + kill_switch_state.py wired to nothing.** Zero live imports (only
tests/ + `scripts/run_book_state_report.py`). Live boot uses old `startup_reconciler.reconcile`
("NEVER modifies Alpaca") + old `kill_switch.KillSwitch` flag. New reconciler + 6-state machine = dead
code on the live path.

**✅ VERIFIED — P4: cash-ETF mapping gap.** `cash_sleeve.CASH_ETFS`=8; `instrument_master._CASH_ETFS` =
{SGOV,BIL,SHV}; `book_state._FACTOR_MAP` maps only those 3. In enforce, a changed `pm.cash_universe`
(e.g. USFR/BILS/TBIL) → unmapped breach → trend HOLDS. Holds (latent: default is within the mapped set).

**🔴 PARTIALLY DEBUNKED — P4: "cash sleeve bypasses the gate" as a risk.** True but mis-emphasized — the
cash sleeve only buys cash-equivalents excluded from risk gross by construction, sized off settled cash,
can't over-deploy. Harmless. Don't gate it.

**🔴 DEBUNKED — "docs claim R0 is a hard no-go gate (implying it's met)."** The GOLIVE synthesis says the
cross-venue risk surface + kill-switch is a hard no-go gate *before IBKR capital* — it does NOT claim
it's currently enforced. Shadow-only is the roadmap working as designed; IBKR is correctly still pending.

**Bonus (panel didn't state):** `risk_policy.py` v1 (drawdown ladder, 8% vol target, corr-spike
de-gross, margin ceilings) is consumed ONLY by offline `multistrat_eval.py` + the gate's static caps.
**The drawdown ladder and vol-targeting are NOT applied on any live order path.** Today the live book's
only real tail control is the VIX crash governor + the 80% gross cap. So P3 is right that
enforce/ladder buys "real tail control" — because there currently is essentially none.

## Kill shots

**1. P1's short-horizon reversal as a "new" idea — KILL the framing.** It already exists, fully built, as
`app/strategy/reversal.py` (Alpha-v4 P4, "3rd uncorrelated premium"), with the exact pitch (dollar-
neutral, skip-1, liquid top-N, PIT-mirrored), `cost_bps=10.0`, **not wired to any scheduler** = built
and shelved. Its own docstring flags "CHEAP-to-find but EXPENSIVE-to-trade... cost is make-or-break."
Same cost-death that killed the intraday/reversal family. The "mechanical negative corr" is real in calm
regimes but reversal gets **run over in trends** (short the winners when momentum extends) → diversifi-
cation evaporates in exactly the sustained-trend regime the book is most exposed to; convexity is
*short* in a melt-up. Don't re-animate without a fresh punitive-cost CPCV.

**2. Turning ON book-vol-targeting to 8% (P3) — KILL for now.** Levers the SOLE live edge ~1.7×
(4.7%→8%) right before IBKR adds correlated beta, via an overlay that can lever to
`book_vol_max_leverage=2.0` in calm regimes — maximally long at the bottom of realized vol (short gamma
on a vol spike: biggest right before the regime breaks). risk_policy itself says launch at 6%, raise
only on live evidence. P3 contradicts the program's own correct conservatism.

**3. P2's "test vol-managed momentum on survivorship-BIASED data as a one-sided kill-screen" — KILL the
logic.** Survivorship bias inflates the *level* of momentum returns, but vol-managed momentum's edge is
*timing/convexity* — and the bias removes the very crash names whose blow-ups the vol-management dodges.
So the bias **flatters the vol-managed variant disproportionately** → can manufacture a false PASS. The
bias does NOT cut both ways here; a "kill screen" that can only false-positive is not a kill screen.
Run on CLEAN Norgate data with the registered gate.

## Survivors (possibly modified)
- **P3 item 1 — drain idle cash + flip gate/ladder to enforce.** Survives; enabling the cash sleeve is
  independent and safe now; flipping the *gate* is blocked by the mapping gap.
- **P3 item 3 — EWMA λ=0.94 vol.** Survives, low priority (a tuned knob on one sleeve).
- **P4 — fix the mapping gap before any enforce.** Survives as a hard prerequisite (one-line each + test).
- **P4 — wire the new reconciler + kill-switch.** Survives as the real #1 pre-IBKR item.

## The blind spot ALL FOUR missed (highest-value output)
**The live book today has almost no enforced book-level risk control, and every panelist proposed adding
new bets / more risk without first making the existing safety layer load-bearing.**
1. Whole-book gate, drawdown ladder, vol target, corr-spike de-gross **all run shadow/research only.**
   The only live tail control is the VIX governor + 80% gross cap. They debated the thermostat on a
   furnace that isn't plugged in.
2. **Every proposed "diversifier" converges to the same short-liquidity bet in a crisis** (trend, xsmom,
   carry, even reversal=short the winners). Re-violates "effectively one bet in a crisis." No one
   proposed a *long-convexity* crash hedge (the one thing the book is short; VRP is the opposite + failed
   paper).
3. **Self-inflicted multiple-testing burden** — queuing 4 more families widens the FWER correction /
   lowers each survivor's bar; P2's biased pre-screen is an un-pre-registered peek that contaminates the
   clean test.
4. **Single-operator / single-process operational risk dwarfs the alpha questions** — in-process brain,
   one operator, notify-watcher must be running, dead-man shadow-only, no auto-flatten. About to add
   futures (gap risk, margin calls, T+1 cross-broker cash). No panelist ranked this #1.
5. **The program may be optimizing the wrong objective** — on $100k, 4.7% vs 8% vol / EWMA vs std / a 5th
   sleeve are economically rounding-error. Binding constraint = capital + live track record, not Sharpe.
   Next 2 weeks: harvest free RFR, harden safety for IBKR, stop adding strategies.

## My ranked top-3 for the next 2 weeks (IBKR pending)
1. **Make the safety layer load-bearing** — wire the new reconciler + kill-switch state machine onto the
   live path; fix the cash-ETF mapping gap; then flip `whole_book_gate_mode`→enforce. Highest value-per-
   dollar-at-risk; the program's own roadmap agrees (R0 = hard no-go gate). IBKR can't turn on without it.
2. **Enable the cash sleeve (live) to harvest RFR on idle NAV.** Built, safe, free, zero-corr. Pure win.
3. **Reconcile the allocation-doc drift; pre-register ONE next test — vol-managed momentum — on Norgate
   (clean) data with the registered Track-B gate, NOT biased data.** Lowest of the three; defer if #1 isn't done.

## The one thing to NOT do
**Do NOT turn on book-vol-targeting to 8% (P3 item 2).** Levers the sole live undiversified edge ~1.7×
via a mechanism biggest right before vol spikes, just as correlated futures beta stacks on top, with the
ladder/de-gross still in shadow. Optimizes a statistically-irrelevant Sharpe delta while maximizing the
exact joint-crisis-drawdown risk the philosophy says to avoid. Runner-up don't: re-animating the
shelved cost-dead reversal sleeve on a calm-regime correlation that inverts in trends.
