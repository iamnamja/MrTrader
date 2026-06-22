# Panelist 3 — Better-Trade What We Have (raw output, Opus 4.8, 2026-06-22)

> Mandate: sizing/timing/execution/governor/allocation improvements to the LIVE trend + paper futures
> book — squeeze more from existing edges, no new strategy. Repo-grounded. Brutal honesty.
> NOTE: several claims here were sharpened/contested by the red-team — see INTERNAL_PANEL_SYNTHESIS.md
> (esp. the allocation is **0.50** not 0.25; and "turn vol-target up to 8%" was the red-team's #1 DON'T).

---

# Better-Trade-What-We-Have: Execution & Sizing Audit

**Headline: the single biggest realized-Sharpe leak is not a parameter — it is that the only de-gross
machinery the book has (drawdown ladder, book-vol target, ERC, whole-book caps) is BUILT but SHADOW/OFF.
You wrote a risk engine and unplugged it.**

## Sizing
- **S1. Book-level vol targeting is OFF** (`TSMOMConfig.book_vol_target=None`). Per-instrument 10% vol on correlated legs → book realizes ~9.34% at 100% gross, ~4.7% at the live allocation — ~half the risk-policy launch target. Overlay code exists (`tsmom.py:197-204`, PIT, capped at `book_vol_max_leverage`). Proposed: target the policy steady value (8%), cap leverage ~1.5×. **[RED-TEAM: KILL for now — levers the sole live edge ~1.7× via a mechanism biggest right before vol spikes; risk_policy says launch at 6%, raise only on live evidence.]**
- **S2. Inverse-vol uses 60d simple rolling std** — slow, symmetric. Switch the *sizing* denominator to EWMA (RiskMetrics λ≈0.94) for faster de-/re-sizing. Low overfit risk if λ pinned to the standard. **[SURVIVES as low priority.]**
- **S3. ERC across sleeves unbuilt for the live book; allocator only knows trend+PEAD (PEAD dead).** Generalize `_load_sleeve_returns` so the hardcoded `pead` dependency doesn't block the first real 2nd sleeve. (Plumbing; correct to keep covariance sizing data-gated to R2.)

## Timing / signal
- **T1. Skip-month is MISSING from the live trend signal** (`sign(P_t/P_{t-L}-1)`, no skip) — inconsistent with the program's own futures sleeves and `etf_rotation` which use 12-1. Pre-register skip∈{0,21}; adopt only on OOS non-inferiority + lower turnover. Higher-stakes (crown-jewel edge) → "clearly better OOS or don't touch."
- **T2. Weekly rebalance is correct — do NOT speed it up.** Already pre-registered+settled (weekly +0.83 ≫ daily/band ~0.50 whipsaw).
- **T3. Sign-only ensemble is robust — leave it.** A continuous/vol-scaled signal is the classic overfit temptation; inverse-vol sizing already handles strength.

## Execution / cost
- **E1. The dust filter ($50) is the only no-trade band — add a proper rebalance band live.** Engine has `rebalance_band` (tested) but OFF live. Untested cell = weekly-recompute + band (skip small weekly nudges, keep big ones). Pre-register band∈{0,0.01,0.02} NAV-weight; accept smallest that cuts turnover ≥20% with Sharpe Δ ≥ −0.02.
- **E2. Cost assumption (2bps/side) UNVERIFIED — use `back_validation.py` (P1-4) once it matures (~15 trading days)** to replace the assumption with a measurement before any change goes to capital.
- **E3. Order timing (market at 09:45) — fine for 10 liquid ETFs; no item.**

## Governor / risk overlays
- **G1. The whole-book gate + drawdown ladder are BUILT but NOT ENFORCING — the biggest single risk leak.** `whole_book_gate_mode` default shadow; the ladder (−8→0.75/−12→0.50/−16→0.25/−20→flat) is applied only in research. The live book has no drawdown-responsive de-gross (only the VIX governor). Proposed: (a) flip to enforce on schedule (fails closed = a missed rebalance, low downside); (b) wire the ladder into the live trend budget as a fail-safe multiplier (PIT pure-function of equity-from-HWM). Gives mechanical crash convexity without buying tail options.
- **G2. VIX governor: `derisk_to=0.5` blunt step, `confirm_days=1` twitchy.** Modest pre-registered refinements: `confirm_days=2` (debounce), maybe a 2-step de-risk. Resist multi-threshold schedules (≈3 real crises = curve-fit). `confirm_days=2` is the only low-risk piece.
- **G3. PARKED credit overlay (+0.064 dSharpe PIT-confirmed; multiplicity caveat).** Keep OFF; shadow it forward and require a pre-registered forward credit-stress episode before flipping. Composes multiplicatively with the VIX governor (one-flag enable when forward evidence lands).

## Allocation
- **A1. Trend at 50% defensible; the real drag is idle cash.** ~75% of NAV sat at zero yield pre-cash-sleeve (~$3.7k/yr ≈ 3.7% NAV recoverable at ~5% RFR). Cash sleeve now enabled (first deploy Mon 2026-06-22) — verify it actually arms. Keep the 2% buffer. Fold futures in via ERC/inverse-vol only when data-gated (≥2 strategies × ≥6mo joint live) — inverse-vol, NOT covariance, until then. **[RED-TEAM: the ~$3.7k assumes ~75% idle; with allocation actually 0.50 the idle is ~50% → ~$2.5k/yr. Still free money.]**

## The 3 highest-EV changes, ranked
1. **Verify cash sleeve drains to T-bills + enforce the whole-book gate / drawdown ladder on schedule (G1+A1)** — free RFR + turning already-built de-gross from shadow into a real left-tail control. Highest EV-per-effort, lowest overfit risk. "Ship the safety you already wrote."
2. **Turn on book-level vol targeting at the 8% policy target with capped leverage (S1).** **[RED-TEAM OVERRULES: do NOT do this now — see DON'T.]**
3. **Switch inverse-vol denominator to EWMA λ=0.94 (S2).**

## Tempting changes that are probably overfitting — resist
- Speeding up rebalance (settled: weekly ≫ daily). 
- A continuous/conviction-weighted trend signal (over-engineering the crown jewel).
- Multi-threshold VIX schedules beyond `confirm_days=2` (≈3 crises = fitting noise).
- Enabling the credit overlay on its in-sample dSharpe (multiplicity).
- Covariance/ERC sizing before ≥2 live sleeves × ≥6mo (can't shrink an unobserved covariance).
- Tightening the 2% cash buffer / $50 dust filter (below-noise, real fragility).
