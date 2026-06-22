# Panelist 1 — Overlooked Strategy Families (raw output, Opus 4.8, 2026-06-22)

> Mandate: net-new strategy families the program hasn't tried (or tried wrong) that could clear the
> Track-B gate. Repo-grounded. Brutal honesty.

---

# Brutally Honest Review — Overlooked Strategy Families for MrTrader

I read the actual system: `PROJECT_STATE.md` (Alpha-v10 narrative), the 27-entry `family_registry.py` kill list, the Track-B v2 appraisal gate (`track_b_appraisal.py`), the construction patterns (`etf_rotation.py`, `vix_vrp.py`, `futures_factors.py`), and the owned data (Norgate mirror: 105 markets incl. FX 6E/6J/6B/6A/6C/6S, rates ZT→UB, energy CL/NG/RB/HO/BRN, metals GC/SI/HG, equity ES/NQ/RTY, ags, VX; GC to 1978, CL to 1983). Alpaca = US equities/ETFs/crypto.

**The bar I'm holding everything to** (from the actual gate code): Track-B v2 PASS needs `appraisal_ir ≥ min_ir`, `P(ΔSR>0) ≥ ~0.875`, **corr-to-book < 0.30**, tail-overlap ≤ 0.30, family-wise corrected. The thing that killed sector_rotation was corr 0.51 to the trend book. That is the wall. I am NOT going to propose another flavor of trend/relative-strength dressed up.

The book's true exposure is: **long-trend (multi-asset TSMOM) + carry + xsmom**. These are all *convergent, short-volatility-of-correlation* premia — GL-1 already found post-2015 pairwise corr ~0.49 and "effectively one bet in a crisis." So the highest-value net-new families are the ones that are **structurally long convexity / mean-reverting / negatively-skewed-payer**, because those are what the existing book is *short*.

---

## 1. Time-series MEAN-REVERSION at short horizons (1–5 day) on the SAME liquid futures/ETFs

**Thesis:** The book is 100% momentum/carry (continuation). The complementary premium — short-horizon *reversal* — is the literal sign-flip of the book's bet and is paid by the same trend-followers when they overshoot.

**Economic mechanism:** Liquidity provision / overreaction reversal. After a sharp 1–3 day move with no fundamental driver, dealers and short-horizon mean-reverters get paid to take the other side of forced/momentum flow. The payer is the very momentum crowd the book belongs to (and panic de-grossers). This is a *real* documented premium (short-term reversal is one of the most robust equity anomalies), distinct from the 12-1 continuation the book harvests.

**Distinctness:** This is the only family here whose signal is *mechanically anti-correlated to the live book* — when trend is long and overshoots, the reversal sleeve fades it. Expected corr-to-book **negative or near-zero** (this is exactly what the corr<0.30 gate is built to reward). It will NOT die the way sector_rotation did, because sector_rotation was *also relative-strength momentum* (corr 0.51); this is its opposite.

**Tradeable NOW:** Yes — fully on Alpaca. Cross-sectional 1–5 day reversal on liquid ETFs (the 11 SPDR sectors + SPY/QQQ/IWM/EFA/EEM/TLT/GLD) and the most-liquid large-caps. No IBKR needed. Crypto reversal (BTC/ETH overnight reversal) is a known, fat, separate version on the same Alpaca venue.

**Data + cost realism:** Free 19y ETF history already owned; survivorship-clean for ETFs. The honest risk is **turnover** — reversal is a high-turnover signal and the program already cost-killed `overnight` (gross +0.53 → net +0.16 at 2bps). So this must be tested at *realistic* ETF cost (1–2 bps/side, which liquid ETFs genuinely have) with a no-trade band, and it lives or dies there.

**Falsification / kill criteria (pre-register):** On 2007→ ETF + crypto universe, net-of-2bps cross-sectional short-horizon reversal must show post-2015 net Sharpe > 0.35 AND **corr-to-trend-book < 0.20** AND Track-B `appraisal_ir ≥ min_ir` with `P(ΔSR>0) ≥ 0.875`. If net Sharpe < 0.30 after cost (the `overnight` outcome) → KILL.

**Effort:** S (it's `etf_rotation.py` with the sign flipped and a short lookback — the engine already exists). **Expected marginal Sharpe contribution:** modest standalone (0.3–0.6) but **high marginal value** because of negative corr.

**Brutal self-critique:** Most likely failure = **cost.** Reversal's gross edge is real but thin per-trade; if the no-trade-band version doesn't clear net 0.30 it's just `overnight` again. Second risk: short-horizon reversal *crashes in trending crises* (it fades the COVID drop and gets run over) — so it may secretly fail the tail-overlap test by being long exactly when trend is short during a melt-up reversal. Must check the GL-1 exceedance corr, not just unconditional.

---

## 2. FX carry / value as a SEPARATE cross-section (G10 currencies, not commodity-dominated)

**Thesis:** The existing `futures_carry` and `futures_xsmom` are run over a *commodity-heavy* 76-market panel and the program itself admits carry is "partly an energy/VIX bet (ex-energy ~0.54)." A clean **G10 FX-only** carry+value cross-section is a structurally different premium with a different payer.

**Economic mechanism:** FX carry = compensation for crash risk (high-yielders crash together — the "carry trade is short a put on global risk"); FX value (PPP reversion) is the *long-horizon mean-reverting* counterweight that historically rallies precisely when carry crashes (Asness/AQR "value and momentum everywhere"). The payer of carry = hedgers and the global savings glut; the payer of value = momentum/carry crowds at turning points.

**Distinctness from the existing book:** The current carry sleeve's diversification is "REAL but MARGINAL (residual-α t~1.8)" and is *contaminated by energy/equity-vol*. A pure G10 FX panel (6E/6J/6B/6A/6C/6S owned in Norgate) isolates a different risk factor. **FX value specifically is mean-reverting** — that's the orthogonality the book lacks. Expected corr-to-book < 0.25.

**Tradeable on what we have:** Needs IBKR futures (FX futures) — but it's a *zero-marginal-cost* research add on the **already-owned Norgate mirror**, and it slots into the exact P2 IBKR pipeline already being built. Could even be paper-traded on Alpaca via FX-proxy ETFs (FXE/FXY/FXB/UUP) as a cheap pre-IBKR validation.

**Data + cost realism:** Owned. Free. The honest caveat: G10 FX carry has *genuinely decayed* post-2008 (ZIRP compressed rate differentials) — so this must be judged post-2015, and it may well be a pre-2008 relic like `futures_trend` was. FX **value**, by contrast, is the part with a live pulse.

**Falsification:** Pre-register G10-only carry AND value as two factors at the economically-motivated sign (no flipping — OPT-5 discipline). Kill any factor with post-2015 Sharpe < 0.3. The *book* test: equal-weight(FX-carry, FX-value) must clear Track-B `appraisal_ir ≥ min_ir` AND corr-to-trend < 0.30. FX-value is the one I expect to survive; FX-carry I expect to be a near-miss.

**Effort:** S–M. **Expected marginal Sharpe:** small-to-moderate; FX-value as a *mean-reverting* diversifier is the real prize.

**Brutal self-critique:** Most likely failure = FX value is *too slow and too low-vol* to clear the materiality floor, and FX carry is post-2008-dead. Genuinely a "test it because it's free and structurally different," not a high-conviction bet. Risks being a re-run of `rates_carry` (KILLED: config-robust but time-unstable).

---

## 3. Equity index DISPERSION / single-stock-vs-index correlation premium

**Thesis:** A genuinely net-new *family* the program has never touched: harvest the implied-correlation risk premium — index options are systematically richer than the basket of single-name options because investors over-pay for index/portfolio crash protection.

**Economic mechanism:** Demand for index hedges (pensions, structured products) bids up index implied vol relative to single-name implied vol → realized correlation comes in below implied → being short index vol / long single-name vol (a dispersion trade) is paid. The payer is the structural hedger. The **one premium uncorrelated to direction, trend, and carry** — a bet on correlation, not level.

**Distinctness:** Totally orthogonal to trend/carry/xsmom. Expected corr-to-book near zero.

**Tradeable on what we have NOW:** **No — the honest disqualifier for the full version.** Real dispersion needs single-name option NBBO at scale, which the program does NOT have (the NBBO log is ~4 days / immature; options were already cost-killed in H4a-e and OPT-2). A poor-man's proxy is tradeable but single-name option costs are *brutal* (deep-OTM ≈ 33% half-spread).

**Data + cost realism:** Needs a real options data buy (ORATS/historical IV surface, ~$1–2k+) to even backtest honestly. Free data is structurally invalid.

**Falsification:** Can't pre-register a clean test without the data buy. Kill criterion is upstream: don't build it until the NBBO forward log matures OR a cheaper idea fails first.

**Effort:** L. **Expected marginal Sharpe:** potentially the best *orthogonality* here, but gated behind data + cost reality.

**Brutal self-critique:** Most likely the already-killed options program in a trench coat. H4a-e killed options-as-signal; OPT-2 showed single-name spreads are 33% — dispersion's P&L can be eaten by the single-name leg's cost. Not executable now and probably not worth the data buy ahead of #1.

---

## 4. Trend-following on CRYPTO + alt-asset breadth as a faster, fatter-tailed trend

**Thesis:** `crypto_trend` is already PAPER (Sharpe 0.64, corr-to-ETF-trend 0.18) but was shelved as "not a capital allocation." The overlooked move is to treat it as a **deliberate convexity sleeve** sized for its low corr, not judged standalone-vs-trend.

**Economic mechanism:** Same trend premium but in a **structurally different, retail-dominated, less-arbitraged market** with fatter tails and different macro drivers.

**Distinctness:** corr-to-ETF-trend already *measured at 0.18* — clears the corr<0.30 wall. It was killed on *standalone Track-B-vs-trend (dSR −0.17)* — the WRONG test for a diversifier; the gate code explicitly waives the standalone-SR floor for `component_type=diversifier` and judges on `appraisal_ir`. **It was judged on the old test, not the v2 appraisal gate.**

**Tradeable NOW:** Yes, fully — Alpaca spot crypto, already wired (`crypto_paper_track.py`), OOS clock running since 2026-06-16.

**Data + cost realism:** Only ~5y history, no live-paper record yet → **CAPITAL power-floor fail**, not an edge fail. 25bps cost honest. Survivorship: crypto universe selection is a real bias risk.

**Falsification:** Re-run crypto_trend through the **v2 Track-B appraisal** as a declared `diversifier`. Kill if `appraisal_ir < min_ir` OR live-paper OOS Sharpe drifts < 0. Power floor stays binding for *capital*.

**Effort:** S (already built; re-gate correctly). **Expected marginal Sharpe:** corr-0.18 means even a 0.4-standalone version contributes positive residual-α — cheapest real diversifier on the board.

**Brutal self-critique:** Most likely failure = **history/power.** 5y of crypto is ~1.5 regimes. Could be a survivorship + two-bull artifact; never clears the n_obs≥504/n_folds≥10 CAPITAL floor honestly on free data. A PAPER diversifier, not capital.

---

## The one I'd actually build first, and why

**#1 — short-horizon time-series/cross-sectional MEAN-REVERSION on liquid ETFs + crypto.** The only proposed family mechanically negatively-correlated to the entire live book; the Track-B v2 gate is built to reward exactly this; free, tradeable today, reuses `etf_rotation.py` with a sign flip + short lookback. Directly addresses the GL-1 "effectively one bet in a crisis" finding.

**Pre-registered falsification:** 2007→ liquid ETFs + Alpaca top-crypto, 5-day formation/1-day hold cross-sectional rank, inverse-vol, no-trade band, net of 2 bps/side, must clear ALL of: post-2015 net Sharpe > 0.35; corr-to-trend-book < 0.20; Track-B `appraisal_ir ≥ min_ir`, `P(ΔSR>0) ≥ 0.875`; tail-overlap ≤ 0.30 (GL-1 exceedance check). Single hard kill: net post-2015 Sharpe < 0.30 → KILL, no sign-shopping.

## Ideas I considered and rejected (respecting the kill list)
- Sector/cross-asset relative-strength rotation — PARKED 2026-06-22 (corr 0.51). Any relative-momentum variant inherits the redundancy.
- Futures curve-mom/value/skew/basis-mom/CoT — KILLED at the pre-registered sign; zoo "exhausted at carry + xsmom."
- Equity factors on US stocks — the gated Phase-4 Norgate audit; equity-ML killed (IC≈0, PEAD, options).
- Defined-risk index VRP via options — Phase 3.2, data-blocked + options-as-signal KILLED.
- Naive long-VIX/tail-hedge — fails the materiality floor (a cost, not a premium).

**Bottom line:** the continuation premia are thoroughly mined; the white space is the reversion/convexity quadrant the book is short — and of that, short-horizon mean-reversion is the one free + tradeable today.
