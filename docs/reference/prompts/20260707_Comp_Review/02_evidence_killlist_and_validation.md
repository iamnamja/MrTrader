# 02 — The kill-list (evidence of "empty-handed") + the validation stack

## A. The strategy kill-list — every distinct family evaluated as a deploy candidate
Sourced from the live `family_registry` + DECISIONS log. Status vocabulary: **LIVE** (trading now),
**PAPER** (passed the paper gate, not yet capital), **KILLED** (evaluated & rejected), **PARKED** (built,
held off — candidate-flag or no-benefit), **SCAFFOLD** (heavily tested, confirmed null, code dormant).

### Equity / ETF
| Family | Status | Verdict |
|---|---|---|
| ETF trend (TSMOM) | **LIVE** | the one validated free-daily edge; post-2015 +0.77 |
| PEAD earnings drift | KILLED | event-level t=−0.77 (p=0.78) → demoted, flipped off live |
| Swing cross-sectional ML ranker | SCAFFOLD | confirmed NULL (IC~0 annually); dormant |
| Intraday ML (5-min) | SCAFFOLD | cost/slippage unmodeled; never live |
| Short-interest days-to-cover (XS) | KILLED | CPCV −1.21 Sharpe; meme-era reversal → no edge |
| Options-as-signal (CPIV/skew/OI/term/IVRV) | KILLED | 5 factors, all t≪−2 → no tradeable equity edge |
| Turn-of-month | KILLED | miss HAC + zero diversification (timed SPY beta) |
| Overnight (close→open) | KILLED | gross +0.53 → net +0.16/−0.22; cost-killed |
| ETF relative-value (log-spread MR) | KILLED | orthogonal but point_SR 0.026, p 0.46 → zero edge |
| Credit-timing (HYG/IEF on SPY) | PARKED | Track-A pass but corr 0.52 to beta → not diversifying |
| Sector-ETF relative-strength rotation | PARKED | standalone CPCV SR 0.86 but **Track-B FAIL vs trend (corr 0.51)** → redundant |

### Futures / cross-asset
| Family | Status | Verdict |
|---|---|---|
| Futures trend (cross-asset TSMOM) | KILLED | real historically, **DECAYED post-2015 (+0.02)**; redundant with ETF trend |
| Futures carry (term-structure) | PAPER | roll-honest Sharpe 0.58, post-2015 +0.89; survives GL-0 |
| Futures cross-sectional momentum (12-1) | PAPER | Sharpe 0.56, corr-to-trend 0.12; max-of-6 null p=0.005 |
| **Futures book (carry + xsmom)** | PAPER | Sharpe 0.67, Track-B t 2.61 on 76 markets — **but on the 16 IBKR-tradeable markets Track-B t = −0.20 (breadth kill, 2026-07-07); ~48 markets needed to restore it → futures-live recommended SHELVE** |
| Curve-momentum / value / skewness | KILLED | killed at the pre-registered sign (no sign-flipping) |
| Basis-momentum (2nd-nearby) | KILLED | Sharpe −0.10, residual-α t 0.47; orthogonal, no edge |
| CFTC CoT hedging-pressure | KILLED | Sharpe +0.06, **perfectly orthogonal** but residual-α t 0.27 → no edge |
| Rates carry (IEF duration) | KILLED | config-robust but time-unstable (post-2016 dead) |

### Volatility / overlays / crypto / cash
| Family | Status | Verdict |
|---|---|---|
| VRP via VIX-futures curve | PAPER (DROPPED) | Sharpe 0.64, survives crashes — **dropped per GL-1 (most tail-concentrating, short-crisis)** |
| VIX-term crash governor | **LIVE** | reduce-only de-risk overlay |
| Credit / curve de-risk overlays | PARKED | credit = marginal tail-insurance (flag off); curve = no benefit |
| Crypto trend (spot TSMOM) | PAPER | paper-tracked, not live capital |
| Cash / T-bill sleeve | **LIVE** | idle-capital utilization |

**Scoreboard: ~28 families searched → 3 LIVE (trend, cash, crash-governor), ~5 PAPER (all futures/vol,
now blocked or dropped), 14+ KILLED, several PARKED/SCAFFOLD.** The recurring pattern: things are either
(a) **null** (ranker, most factors), (b) **real but redundant/non-diversifying vs trend** (sector rotation,
credit-timing, futures book on tradeable set), or (c) **cost/regime-killed** (overnight, rates carry,
futures trend post-2015). **Almost nothing is both real AND a diversifier AND tradeable at our scale.**

## B. The validation stack (critique this)
1. **Purged & embargoed CPCV** — combinatorial purged cross-validation; 85-calendar-day purge (swing) /
   2-trading-day (intraday); a "sacred holdout" tail. Averages performance over many train/test splits.
2. **Ruler-v2 gate (two-track)** — a **significance** track (path-t / HAC) AND an **economic-materiality
   floor** (avg Sharpe ≥ 0.80 swing / 1.00 intraday, profit factor ≥ 1.10, Calmar ≥ 0.30, min-fold
   Sharpe ≥ −0.30). Live-paper is a **structural** gating criterion (a strategy must survive on backtest
   AND live).
3. **Deflated Sharpe Ratio (DSR)** — deflates the observed Sharpe by the number of variants tried
   (**N_TRIALS_TESTED = 300**; family-level trial count = 25+) → guards against selection bias; p > 0.95
   to pass.
4. **Selection-aware null-zoo** — an empirical max-statistic null: permute/resample the signal, rebuild
   the book, and compute P(t_null ≥ t_obs). The futures book passed at p=0.002; xsmom cleared its
   best-of-6 selection bar at p=0.005.
5. **Track-B residual-alpha** — the **diversification gate**: regress a candidate's returns on the LIVE
   ETF-trend book, take the HAC-t of the residual alpha. This is the bar the futures book just failed on
   16 markets (t 2.61 → −0.20). It answers "does this ADD to what we already trade," not "is it good
   standalone."
6. **GL-0 / GL-1** — a null-strategy zoo (is the book real vs residue) + tail/co-crash diagnostics (are
   the premia one-bet in a crisis). These dropped VRP and confirmed the futures book "real but modest."

## C. What our own validation ADMITS it does NOT do well (hand this to the panel)
- **Regime-imbalanced folds.** CPCV averages over folds but folds are **not regime-balanced** — a
  strategy's number can hinge on which crisis (COVID/GFC/2022) lands in which fold. We do NOT decompose
  performance **regime-conditionally**.
- **One realized path.** We validate a **static** strategy on the **single** realized history — implicitly
  assuming stationarity. We do **not** systematically stress on **synthetic / bootstrapped / resampled-
  crisis** paths.
- **Unconditional-edge bias.** Every gate demands an edge that works **on average across all regimes**. A
  genuinely **regime-conditional** edge (great in ranging, flat in trending) can be *averaged to null* and
  killed — we may be **systematically rejecting conditional edges.**
- **Signal-mining, not mechanism-first.** The factor zoo screens many signals + corrects for multiplicity,
  rather than starting from a structural inefficiency + counterparty. The one non-trend survivor (PEAD)
  was the one built mechanism-first.
- **Survivorship** in yfinance equities and in the futures `liquid_universe` (selects on current liquidity
  + full history) — inflates backtest numbers, including the ones we keep.
- **DSR borderline.** Even the full futures book's DSR at the family trial count is flagged "borderline
  <0.95" → "size modestly." The surviving premia are marginal, not slam-dunks.
