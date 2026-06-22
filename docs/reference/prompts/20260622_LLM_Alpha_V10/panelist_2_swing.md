# Panelist 2 — Swing Equity Deep-Dive (raw output, Opus 4.8, 2026-06-22)

> Mandate: what swing equity approach (if any) could actually clear this program's gates, post-PDT-myth.
> Repo-grounded. Brutal honesty.

---

# Swing Equity Trading for This Program — Brutally Honest Verdict

## 1. Verdict up front

**Swing equity is, for THIS program, a sunk-cost trap as a standalone alpha sleeve — but it has one narrow, genuinely-untested opening that the prior work never properly reached, and it is NOT the one you'd reflexively pick.** The program has run cross-sectional equity ML ranking to a clean IC≈0 null *three independent times* (long-only, dollar-neutral high-breadth, and the "validate-the-validator" positive control confirmed the pipeline is sound, so the null is the market, not a bug). It killed PEAD at the event level, short-interest XS, overnight, turn-of-month, and ETF relative-value. Short-term reversal — the obvious "non-ML rules-based swing premium" — was *already built and killed as cost-dead* (gross +0.40 Sharpe, −0.90 net at 10bps, 159×/yr turnover). The pattern is not "we built it wrong"; it is that **free-data, liquid-universe, US single-name directional alpha is genuinely mined out, and the program has already proven this with more rigor than most funds.** The PDT correction changes nothing here — PDT was never the binding constraint; *edge* was. The only defensible swing work left is survivorship-corrected single-name event/factor testing that **requires the Norgate US Stocks ($693) buy**, and even that should be framed as "close the question," not "find the edge." Do not build another ranker.

## 2. Post-mortem — why the prior swing attempts really failed

- **Swing ML ranker (3× null): the edge isn't there, not a build error.** v1 long-only +0.22 (t=0.17); v2 dollar-neutral/sector-neutral/high-breadth — after fixing 3 real neutrality bugs the *corrected* book printed mean Sharpe +0.136, path-t +0.18, DSR p 0.03. The long-only "edge" was confirmed market beta. The Alpha-v9 P0-1 positive control then proved the feature→label pipeline is faithful (label-fidelity +0.761) — so IC≈0 is the market/cost, definitively not a deflationary bug.
- **PEAD: real-but-beta, then event-level dead.** Drift was conditional market beta (hedged Sharpe −0.37, alpha t(HAC) −0.95); event-level inference (H1) put t=−0.77 (p=0.78) → demoted and flipped off live.
- **Short-term reversal: real signal, cost-dead.** Genuine (gross +0.40) but the most-arbitraged anomaly; 159×/yr turnover → ~16%/yr cost drag kills it at realistic 10bps. Weekly rebalance only lifts it to −0.18. Economics, not engineering.
- **Short-interest XS / overnight / turn-of-month: real-but-not-tradeable or timed-beta.** Short-interest reversed in the meme era; overnight real gross (+0.53) but cost-killed net; turn-of-month was just timed SPY beta.
- **The unifying cause:** every survivor is a **risk premium harvested across many low-cost instruments** (ETF/futures trend, futures carry, futures XS-mom). Every death is **single-name, high-turnover, or beta-in-disguise.** Swing equity sits squarely in the death zone.

## 3. Candidate swing approaches

The bar any candidate must clear: **Track-B residual-alpha IR ≥ threshold and P(ΔSR>0) ≥ 0.75 vs the live trend+futures book, deflated for the now-26-family multiple-testing burden.** Standalone Sharpe is explicitly *not* enough — sector_rotation just proved this (CPCV 0.86 standalone, Track-B FAIL at corr 0.51).

### (a) Vol-managed / trend-conditioned single-name momentum — the one real opening — Effort M, expected edge LOW-MODERATE
- **Mechanism + who pays:** Barroso–Santa-Clara vol-management of single-name cross-sectional momentum. The edge is not "momentum exists" (mined out at the index level) but that **momentum crashes are forecastable from realized momentum-portfolio volatility**, and scaling exposure inverse to that variance harvests the crash-avoidance premium. Who pays: leveraged/constrained investors who can't time the deleveraging. The *one* equity construction the program never ran (the ranker was always constant-gross).
- **Why it could survive:** the ranker died on flat IC. Vol-management doesn't need positive IC every month — it needs the *conditional variance* of the momentum factor to be predictable (robustly ~0.5 R² in the literature). A timing overlay on a known factor, same family as the program's surviving overlays.
- **Construction:** monthly-rebalanced 12-1 cross-sectional momentum decile spread on a survivorship-safe universe (Russell-1000-as-of), then scale gross by `target_vol / realized_vol_22d(momentum_factor_returns)`. Holding period weeks-to-a-month. **Norgate required.**
- **Pre-registered falsification:** vol-managed single-name momentum, residualized against SPY *and* the live ETF-trend sleeve, must show residual-alpha t(HAC) > 1.96 AND Track-B P(ΔSR>0) ≥ 0.75 on Norgate 2004→present, with the post-2015 sub-period not negative. Kill at the pre-registered sign.
- **Honest expectation:** ~40% clears Track-A standalone, ~20% clears Track-B vs the trend book (which *is itself* a momentum harvester → high redundancy risk, the sector_rotation failure mode). Best swing bet precisely because its failure would be informative and fast.

### (b) "Better engineering of the same ML ranker" — DO NOT BUILD. Sunk-cost trap.
Three nulls, a verified-faithful pipeline, a verified-faithful neutralization engine. IR≈IC·√breadth — √breadth×0=0. Adding features/boosting/breadth multiplies zero. Already correctly closed.

### (c) Swing as a TILT/overlay on the trend sleeve — Effort S, cheapest test
- **Mechanism:** use a single-name signal (52-week-high proximity; surviving short-interest conditioning) to *tilt within* the trend book's ETF holdings — overweight high-momentum constituents when SPY-trend is on. Sidesteps the standalone-SR bar; goes straight to marginal contribution.
- **Why it could survive:** the program's only live additions beyond trend are overlays (VIX governor). Overlays are scale-invariant and don't need standalone return (credit_overlay +0.064 dSharpe PIT-confirmed).
- **Falsification:** the tilt must raise the trend book's Track-A Sharpe with dSharpe CI excluding 0 under PIT, multiplicity-disclosed.
- **Honest expectation:** ~15%. Likely redundant — but an S-effort test on owned data (ETF-constituent version needs no Norgate).

### (d) Post-event drift done right (small/mid-cap PEAD) — Effort M, LOW; only if Norgate bought anyway
Drift is strongest in small/mid-caps (slow diffusion, low coverage); a survivorship-safe harness was built but never run (data-blocked). A genuine "never tested the strongest variant" gap — but large-cap PEAD died at the event level (t=−0.77), so the prior is poor and small-caps add liquidity/cost risk. Run only as a rider on a Norgate buy justified by (a).

## 4. What data would change the answer

**Yes — Norgate US Stocks ($693/yr) is the one buy that unlocks a credible swing test free data cannot.** Every prior single-name kill carries the survivorship asterisk (free/yfinance/cached-R1K has no delisted losers → *flatters* long/short and event studies). Norgate unlocks: (i) survivorship-safe single-name momentum for (a); (ii) the never-run small/mid-cap PEAD; (iii) un-biased reversal re-test. **But the EV ordering the Alpha-v10 panel set is correct: Norgate-stocks is a "close-the-question bounded audit," AFTER the free futures zoo and IBKR work, because the prior on new single-name alpha is low.** Don't buy it to "find swing alpha"; buy it to *retire the question* iff (a) looks alive on the bias-flattered data first.

## 5. The bet and the trap

**If forced to bet on ONE swing idea:** vol-managed single-name cross-sectional 12-1 momentum, sized inverse to the momentum factor's own realized variance (candidate a). The one equity construction never run; survival doesn't require the positive IC that died 3×; sits in the overlay/risk-management family that survives here.

**Test to kill it fast (cheap, before $693):** run it FIRST on the **already-owned survivorship-biased cached R1K universe** as a kill-screen. The bias *flatters* it, so it's one-sided: if it can't clear residual-alpha t(HAC) > 1.96 vs SPY+trend AND Track-B P(ΔSR>0) ≥ 0.75 on the FLATTERING data, it's dead and Norgate can't save it. Only if it passes does buying Norgate to confirm make sense. Decision in days, $0 at risk.

> [NOTE — the red-team later DEBUNKED this "one-sided kill screen" logic: survivorship bias is
> directionally FAVORABLE to the vol-managed variant specifically (it removes the crash-blowup names the
> vol-management is meant to dodge), so the screen can produce FALSE POSITIVES, not just conservative
> kills. Run the test on CLEAN Norgate data with the registered gate, not a biased pre-screen.]

**Sunk-cost traps — don't touch:** any re-engineering of the ML ranker; short-term reversal/overnight/turn-of-month in the liquid universe; standalone large-cap PEAD or options-as-signal; buying Norgate-stocks before a clean test on (a).

**Key files:** `app/strategy/reversal.py`, `app/ml/factor_scorer.py`, `app/research/family_registry.py`, `scripts/walkforward/track_b_appraisal.py`, `ML_EXPERIMENT_LOG.md` (reversal cost-death; ranker triple-null), `DATA_PROVIDERS.md` (Norgate $693 deferred).
