# 02 — Kill / Keep Ledger (everything we've tried + the verdict + why)

Brutally honest. "Verdict" is our call; challenge any of them. KEEP = trades live; CANDIDATE =
built, flag-off; PAPER = report-only OOS tracking; KILL/PARK = not pursued.

| Strategy | Asset/data | Verdict | Why |
|---|---|---|---|
| **Multi-asset TSMOM trend** | 10 ETFs | **KEEP (LIVE)** | Standalone Sharpe ~0.72 (19y), crisis-diversifying, robust across sub-periods. The one validated edge. |
| **VIX-term crash governor** | VIX/VIX3M | **KEEP (LIVE overlay)** | Cuts maxDD (−13.9%→−12.1%, COVID −10.7%→−6.5%) at ~flat Sharpe; fail-safe. |
| **Cash / T-bill sleeve** | SGOV/BIL | **KEEP (LIVE)** | Idle ~50% NAV earns RFR instead of zero. Pure carry on cash. |
| **Futures CARRY (term-structure)** | Norgate, 76 mkts | **NEW EDGE (paper-candidate)** | Sharpe 0.66 (honest ~0.55-0.60 after roll cost), positive every sub-period incl. modern (post-2015 +0.89), Track-B dSR +0.17, cost-robust. NOT yet live. |
| **Crypto trend (TSMOM)** | Alpaca, 10 pairs | **PAPER-CANDIDATE** | Sharpe 0.64, corr-to-trend 0.18 (criterion met) BUT Track-B vs trend FAILS (dSR −0.17); ~5y history → no CAPITAL. Live-paper tracking. |
| **Credit/curve de-risk overlay** | HYG/IEF | **CANDIDATE (flag-off, shadowing)** | Marginal-to-governor dSharpe +0.064, all-3-crises; small tail-insurance, post-hoc. Verdict ~mid-July. |
| **Futures TREND (TSMOM)** | Norgate, 76 mkts | **DECAYED → not standalone** | Full-sample Sharpe 0.83 is *entirely* pre-2010; post-2015 ~0.0. Redundant with our ETF trend (corr 0.44). Kept only as carry's crisis-convex partner. |
| **PEAD (post-earnings drift)** | equities/earnings | **DEMOTED** | Event-level 10d SPY-hedged mean −8.3bp, t=−0.77; market-hedged Sharpe −0.37. It was conditional beta, not alpha. |
| **Cross-sectional ML swing ranker** | equity features | **KILLED** | 3 honest purged-CV nulls; long-only "edge" was confirmed market beta; dollar-neutral = noise. |
| **Index VRP / short-vol** | options | **PARK (risk premium, not alpha)** | VRP real + cost-robust (PF 2.24/1.75) but Sharpe-weak; it's a *risk premium*, our gate is an *alpha* gate. |
| **Single-name earnings IV-crush** | options | **KILLED** | Cost-killed (premium-% spreads eat it). |
| **Options-as-signal (CPIV, 25Δ-skew, term-slope, put-O/S, IV-RV)** | options | **ALL KILLED (H4a-e)** | Decile L/S net-of-cost all significantly negative 2022-26; academic signs don't hold at equity cost. |
| **Overnight vs intraday premium** | ETFs | **KILL (clean)** | Overnight premium is REAL (gross Sharpe +0.53) but daily round-trip cost erases it (net +0.16 < floor). |
| **FINRA daily short-volume** | FINRA, 2019+ | **REAL-but-weak** | Informed-short signal confirmed, overlay beats buy-hold, but residual-α vs SPY insignificant + sub-period unstable → not standalone. → composite / post-Norgate XS. |
| **Turn-of-month / calendar** | SPY | **FAIL** | Misses significance; timed SPY beta. |
| **ETF relative-value (stat-arb)** | ETFs | **FAIL** | Orthogonal (corr −0.23) but ~0 standalone edge. |
| **Rates carry (ETF, long-flat IEF)** | ETFs | **KILLED** | Edge is a pre-2016 artifact (H1 SR +0.69 vs H2 −0.10); stability guard caught it. |
| **Aggregate short-interest (bi-monthly)** | FINRA SI | **KILLED (power)** | ~190 obs; underpowered. (Daily short-VOLUME is the powered redo — see above.) |
| **Sleeve allocator (regime/vol weighting)** | book | **OFF (scaffold)** | On 2 sleeves, equal-weight beats vol/regime; "complexity must earn it." |

## The recurring lessons (we keep relearning these)
1. **Standalone return — not orthogonality — is the binding constraint.** Many things diversify; few make money alone.
2. **Most "equity alpha" we found was market beta** (PEAD, XS-ML long-only).
3. **Cost kills high-turnover edges** (overnight, single-name options).
4. **Risk premia ≠ alpha** (VRP) — our gate kept rejecting them as if they were alpha.
5. **Sub-period stability is the sharpest overfitting guard** (killed rates carry; flipped the daily-carry "edge").
6. **Free daily US-equity *directional* alpha appears mined out** — the wins came from new *data*
   (futures carry) and new *structure* (overlays, cash), not from more equity feature mining.
