
**Expected Capacity:** Low-moderate. Crypto liquidity is improving but still lower than equities.

**Failure Modes:** **THIS IS VOLATILE.** You must size this ≤5% of book. Correlation to equity trend may be low (good) but crypto is a crisis-correlated risk asset (bad). The economic rationale is sound—trend works in crypto.

**Priority:** **MEDIUM.** You already have the data (Alpaca crypto) and the trend engine. This is a natural extension.

---

### 🎯 Tier 4: Long Shots / Institutional-Gated

---

#### Idea #11: Options Dealer Gamma / Flow (Event Filter)

**Economic Rationale:** Options dealer gamma positions create hedging flows that impact equity prices. Dealer gamma (net options position) can predict short-term reversals.

**Instruments / Universe:** SPY or broad equity ETFs. Short-term (1-5 day) reversal signal.

**Horizon:** Intraday to 2-day horizon.

**Data Needed:** **Institutional-level options data** (live OI, volume, IV, greeks) is expensive (>$1000/mo for CBOE data). You have a frozen 4y snapshot without OI/NBBO—insufficient.

**Signal Construction:** Complex; requires dealer gamma estimation models (which need live options surface data).

**Expected Capacity:** Unlimited.

**Failure Modes:** Data cost is prohibitive for your budget. The signal is real but institutionally gated.

**Priority:** **SKIP** until/unless you get institutional-level options data.

---

#### Idea #12: Machine Learning on Alternative Data (News, Sentiment, Satellite)

**Economic Rationale:** Alternative data (news sentiment, satellite imagery, credit card transactions) contains alpha that's not in price.

**Instruments / Universe:** Broad equity or specific sectors.

**Horizon:** Daily to weekly.

**Data Needed:** Alternative data is **expensive** ($1000-10,000+/yr). The free tier (Finnhub news, sentiment) is low-quality.

**Signal Construction:** ML model on news sentiment, social media, or earnings call transcripts.

**Expected Capacity:** Unlimited.

**Failure Modes:** Data cost is prohibitive; free versions are mined out.

**Priority:** **SKIP** unless/until you have institutional budget.

---

## 4. DATA GAPS & TOP BUYS

### What You Have (Re-evaluated)

| Data | Status | Verdict |
|------|--------|---------|
| Daily EOD equities/ETFs (yfinance) | Good for 10-ETF trend; bad for survivor-bias-free backtesting | **Adequate for current strategy** |
| 5-min bars (Polygon, frozen) | Insufficient depth for intraday ML | **Kill it; not worth the $79/mo** |
| Options OHLCV (Polygon, frozen 4y) | No OI/NBBO/IV history; impossible for signals | **Kill it; not worth the $79/mo** |
| FRED macro | Deep, free, valuable | **Keep; underutilized** |
| FMP fundamentals/earnings | Cheap ($29/mo); PEAD killed | **Keep for event panel; demote to low priority** |
| Short-interest (Polygon, 2017+) | Shallow; backfill needed | **Keep; backfill to 2005 via FINRA public data** |

### What You're Missing (Top Buys)

| Rank | Data | Cost | Why | ROI |
|------|------|------|-----|-----|
| **1** | **Clean futures data with roll schedules (Norgate)** | ~$270/yr | Unlocks genuine carry/roll-yield; the one lever you can't test on free data. | **HIGHEST** |
| **2** | **Survivorship-free equity history (Sharadar/Quandl)** | ~$100-200/yr | Your yfinance data is survivor-biased; any long/short or event study is quietly flattered. | **HIGH** |
| **3** | **FINRA short-interest backfill (2005-2017)** | Free (public data) | Gives G2 the power it needs to test 2008/2011 crises. | **HIGH (but time cost)** |
| **4** | **VIX futures term structure (CBOE)** | ~$50/mo | Enhances the VIX governor and enables VRP. | **MEDIUM** |
| **5** | **Economic calendar (FMP premium or FRED)** | Free–$30/mo | Enables FOMC/payrolls drift overlays. | **LOW** |

### The Single Highest-ROI Data Buy

**Norgate Data (~$270/yr) for clean futures + roll schedules.**

**Why:**
1. Unlocks **carry/roll-yield** (the one lever you can't test on free data)
2. Clean **continuous futures** with proper roll adjustments
3. Survivorship-free futures history (not yfinance's dirty `=F` series)
4. Doubles as a **small-cap PEAD** data door (Norgate has delisting data)
5. If the futures trend/carry sleeve passes pre-registration, you can decide whether to open a second broker (IBKR) to trade it

**But:** Don't buy Norgate *until* you have a pre-registered futures strategy that passes a free-data POC. The free-data POC should be on ETF equivalents (e.g., DBC for commodities, TLT for bonds, EFA for intl). If the ETF POC fails, futures won't save it.

---

## 5. MODELING RECOMMENDATIONS

### What You're Doing Wrong

**1. You're using the wrong target.**

Your swing ML target is "top 20% cross-sectional Sharpe-adjusted 5-day forward return." This is a *relative* ranking target. But your portfolio is *absolute* return (long-only 5 positions). You're training a ranking model and then applying it as a long-only portfolio. The disconnect is severe.

**Recommendation:** Train on the *actual portfolio objective*—either:
- **Direct portfolio optimization:** Train a model to maximize the portfolio's Sharpe directly (reinforcement learning or end-to-end portfolio optimization)
- **Factor-augmented trend:** Use the ML model as a *meta-labeler* on the trend book (long when ML says "good," flat when "bad"), not as a standalone return stream

**2. You're overfitting to the gate.**

The gate thresholds (SR ≥ 0.80, t ≥ 2.0) have become a target in themselves. Every strategy is optimized to *just* pass the gate, not to maximize economic value. This creates a culture of "gate-hacking" (testing variants until one passes).

**Recommendation:** Make the gate a *minimum bar* only. Use a *different* metric (like residual-alpha t or crisis-period Sharpe) for promotion decisions. The Bayesian posterior in Ruler-v2 is a good start.

**3. You're not using ensemble / multi-model approaches.**

Your intraday ML uses a 3-seed XGBoost ensemble (good). But your swing ML is a single XGBoost model. Ensemble methods reduce overfitting and improve OOS performance.

**Recommendation:** Add a LightGBM model to the swing ML ensemble (you already have it in `ensemble_models` but it's disabled). Blend XGBoost + LightGBM + a simple baseline (e.g., momentum).

**4. You're not using meta-labeling effectively.**

You tried meta-labeling in the intraday ML and found it added +0.000 Sharpe. But the concept is right—you just used the wrong meta-labeler.

**Recommendation:** Use the meta-labeler to *size positions*, not to *filter entries*. Train a model to predict the *confidence* of the primary model's signal, then size positions proportionally to confidence.

**5. You're not using regime-conditioned models.**

Your regime system (`coarse3`: BULL/BEAR/NEUTRAL) is used only as a gate (worst-regime floor). It should be used to *train separate models per regime*. A model trained only on BULL data will perform poorly in BEAR; a model trained only on BEAR data will perform poorly in BULL. But an *ensemble* of regime-specific models can outperform a single model.

**Recommendation:** Train separate swing ML models for each regime (BULL/BEAR/NEUTRAL). At prediction time, use the appropriate model for the current regime.

### What You Should Try (Concrete ML Approaches)

**1. Factor-augmented trend overlay (not a sleeve).**

Use an ML model to predict the *trend book's future Sharpe*. Features: macro (FRED), volatility (VIX term), credit spreads (HYG/IEF), momentum breadth. Output: a multiplier on the trend book's exposure.

**2. Position-sizing via meta-labeling.**

Train a model to predict the *probability of profit* for each position. Use that probability to size positions: `size = base_size × prob_profit`.

**3. Regime-specific ensemble.**

Train separate models for BULL/BEAR/NEUTRAL. At prediction time, weight predictions by the probability of each regime (from a regime classifier).

**4. End-to-end portfolio optimization (RL).**

This is a long shot, but: train a reinforcement learning agent to directly optimize the portfolio's Sharpe. The agent's state = features; action = portfolio weights; reward = daily return. This bypasses the "ranking model → portfolio" disconnect entirely.

**5. Causal feature selection.**

Your swing ML uses 140 features. Many are redundant or noisy. Use a causal feature selection method (e.g., PC algorithm, or a simple permutation importance + correlation filter) to select a smaller, more robust feature set. This reduces overfitting and improves OOS performance.

---

## 6. REDESIGN (If Warranted)

### The Big Problem

Your architecture is optimized for **finding additive edges** (sleeves) but not for **building a book** (overlays + sleeves combined). The 3-agent split (PM/RM/Trader) is a fine execution model, but the *research* architecture is misaligned with the *economic* reality.

### What I'd Build From Scratch

**1. Abandon the "sleeve" as the primary unit of analysis.**

Replace it with a **"factor"** —a return stream that may or may not be tradeable standalone. Factors are judged on:
- **Marginal Sharpe improvement** to the existing book (not standalone)
- **Crisis-period performance** (not average Sharpe)
- **Correlation to existing factors** (orthogonality)

**2. Build a "Factor Ensemble" allocator.**

The allocator's job is to combine factors into a book. It uses:
- **Risk budgeting** (allocate risk, not capital)
- **Regime weighting** (e.g., more trend in BULL, more carry in BEAR)
- **Crisis hedging** (overlays that cut exposure in stress)

**3. Make overlays first-class citizens.**

Overlays are not "sleeves" and should not be judged by sleeve criteria. An overlay's value is purely its marginal contribution to the book. Build a dedicated "Overlay Lab" alongside the "Sleeve Lab."

**4. Separate research from execution.**

Your current system mixes research and execution (the PM runs the model at 09:25). For a $100k paper account, this is fine. But for live capital, you want:
- **Research:** Offline, pre-registered, one-shot confirmatory runs
- **Execution:** A lightweight "executor" that loads the approved model and places orders

**5. Simplify the agent architecture.**

Collapse PM+RM into a single "Allocator" for the trend book. The Allocator:
- Reads the factor weights from the approved model
- Applies overlays (governors)
- Checks position/risk limits
- Sends orders to the Trader

### The "New" Architecture
