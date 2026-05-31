# News & LLM Integration Design Review — Prompt for External LLM Consultation

## Context: Automated Equity Trading System — News & LLM Integration Design Review

I'm building an automated swing + intraday equity trading system in Python. I want your honest architectural advice on how to best integrate news data and LLM-based sentiment analysis into the workflow. Please critique what we currently have, identify gaps, and suggest what the ideal design looks like — don't feel constrained by what we've built so far.

---

### System Overview

- **Capital**: ~$20,000 paper trading (targeting live after gate validation)
- **Strategies**:
  - **Swing**: holds 3–5 days, scans every morning ~09:45 ET, ~5–15 positions at a time, top-50 S&P500 liquid stocks
  - **Intraday**: holds 2–4 hours within a single session, scans at 3 windows (60/90/120 min post-open), ~3–5 positions
- **ML models**: XGBoost classifiers (binary: up/flat) trained on price/volume/technical features + some macro signals. Walk-forward validated. Swing Sharpe ~1.2, intraday Sharpe ~1.8.
- **Agents**: Portfolio Manager (PM) proposes trades → Risk Manager (RM) vetoes/approves → Trader executes via Alpaca
- **LLM in use**: Claude Haiku already wired in for 3 non-blocking audit tasks: (1) narrative reasoning on PM proposals, (2) plain-English RM veto explanations, (3) pre-market daily briefing summaries

---

### Current News Pipeline (what we have today)

1. **Sources**:
   - **Polygon.io** — provides pre-AI-labeled sentiment per article (positive/neutral/negative) per ticker. Used for swing training features (3d/7d rolling sentiment averages). Also used live via a background poller.
   - **Alpaca News** — just added. No pre-labeled sentiment. We apply keyword matching (e.g., "downgrade", "miss", "loss") to classify negative.

2. **Live monitor** (`NewsMonitor`):
   - Polls both sources every 5 minutes during market hours
   - Watches only symbols that PM has shortlisted for trades
   - `has_negative_news(symbol, window=30min)` → boolean
   - **Entry gate**: if True, PM blocks new entry for that symbol
   - **Exit flag**: if True for an open position, PM proposes exit
   - Fails open (never blocks trading on API outage)

3. **Training features** (swing model only):
   - `news_sentiment_3d` — avg Polygon sentiment score over 3 days
   - `news_sentiment_7d` — avg Polygon sentiment score over 7 days
   - `news_article_count_7d` — article volume (attention proxy)
   - `news_sentiment_momentum` — 3d minus 7d (recent shift)

4. **What's missing / known weaknesses**:
   - Polygon sentiment labels are generic (not ticker-specific enough — "Apple supply chain" might be labeled neutral even if negative for AAPL)
   - Keyword classifier is blunt (false positives on "tax loss harvesting", "no loss of life", etc.)
   - No LLM currently involved in news interpretation
   - No morning pre-market news digest that actually influences the model or PM decisions
   - Intraday model has no news features at all
   - No earnings calendar integration — we don't know if a symbol is reporting soon
   - No macro news (Fed, CPI, jobs reports) affecting the whole market

---

### What I'm trying to figure out

Please help me think through the following, and feel free to go beyond these questions if you see something important:

**1. Morning news digest**
   - Should there be a pre-market scan (e.g., 08:30–09:30 ET) that reviews overnight news for watchlist symbols?
   - Should this influence which symbols PM considers at all (pre-filter), or just adjust position sizing/confidence?
   - What's the right way to use an LLM here — summarize? score? rank? something else?

**2. LLM as sentiment classifier**
   - Instead of (or in addition to) Polygon's labels + keyword matching, should we pass article headline + summary to Claude/GPT and ask a focused question like: "Given we may buy {symbol} today, does this news materially increase downside risk in the next 4–8 hours?"
   - What prompt structure works best for this? What output format (score, binary, structured JSON)?
   - How do we avoid the LLM being too conservative and blocking too many trades?

**3. News as ML features vs. runtime gate**
   - Currently news only appears as training features (swing) and a runtime gate (blocking entry/exit).
   - Should news sentiment be a real-time input feature to the XGBoost model itself at inference time? If so, how do we handle the fact that Polygon labels aren't available in real-time for free-tier?
   - Is there a better architecture where LLM-derived sentiment scores become features that feed INTO the model prediction rather than just gating after the fact?

**4. Earnings and macro events**
   - How should we handle earnings risk? (Don't trade within N days of earnings?)
   - How should macro news (Fed announcements, CPI, NFP) affect the system — blanket pause? regime signal?
   - What free or low-cost data sources do you recommend for earnings calendars and macro event schedules?

**5. Free/low-cost news sources**
   - We currently have Polygon (paid) and Alpaca News (free with brokerage account).
   - What other free or near-free sources are worth integrating? (RSS feeds, SEC EDGAR, Reddit/social, etc.)
   - For each: what signal does it add that we don't already have?

**6. Feedback loop**
   - Once we have LLM-scored news, how should that score feed back into model retraining? Should news sentiment become a new training feature?
   - How do we avoid look-ahead bias when backtesting a news-aware model?

---

### Constraints

- We want to keep inference-time latency under 2 seconds per symbol (PM runs on a schedule, not HFT)
- LLM cost should stay under ~$5/day even at scale
- System must fail gracefully — news outages should never block trading
- We're using Python, FastAPI, XGBoost, Alpaca, SQLite
- Claude Haiku is already integrated; OpenAI or other providers are also options

---

### What I want from you

1. **Critique the current design** — what's architecturally wrong or naive about how we're doing this?
2. **Propose the ideal end-state workflow** — from raw news ingestion through to trade decision. Be specific about where LLMs sit in that workflow and what they're doing.
3. **Prioritize**: if we can only do 3 things to improve news integration in the next 2 weeks, what are they?
4. **Flag anything we haven't thought of** — risks, edge cases, approaches we're missing entirely.

Be direct. If our current approach is mostly fine, say so. If it's fundamentally flawed, say that too.
