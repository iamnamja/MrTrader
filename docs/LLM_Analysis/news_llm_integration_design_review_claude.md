# News & LLM Integration — Design Review & Recommendations

**Project:** Automated Equity Trading System
**Topic:** News ingestion + LLM-based sentiment integration
**Date:** April 27, 2026

---

This is a solid prompt — the design is more thought-through than most attempts I see at this stage. But there's a fundamental architectural flaw running through it, and a few gaps that are bigger than they look.

## The core problem with the current design

News is bolted on in two disconnected places, using two different representations:

- **Training features** use Polygon's pre-labeled sentiment averaged over rolling windows (numeric)
- **Runtime gate** uses keyword matching plus Polygon binary thresholds (boolean)

The model never sees at inference what it was trained on. That's classic train/serve skew. The model learned that `news_sentiment_3d` correlates weakly with returns; at decision time you bypass that learned signal and substitute a hand-rolled keyword rule. Either the model knows about news or the gate does, but having both layers disagree about what "negative news" means is worse than having neither.

A few other issues worth naming:

1. **Polygon's article-level sentiment is too generic to gate trades on.** You already flagged this. Without entity-level resolution it's noise.
2. **The keyword classifier is unreliable in both directions.** "Tax loss harvesting," "no loss of life," "downgrade their estimate" (from analyst, not on stock) — predictable false positives. This shouldn't have veto power.
3. **No earnings calendar integration is your biggest single gap.** A 3–5 day swing position holding through earnings is a binary outcome event the model wasn't designed for. Easy to fix, large risk reduction.
4. **No event taxonomy.** Macro events (Fed/CPI/NFP), idiosyncratic events (earnings, M&A, FDA), and noise events (analyst color pieces) all flow through the same pipe. They have completely different decision implications and need different routing.
5. **No replay-able event store.** If you score news with an LLM tomorrow, can you backtest a news-aware model against a year of history with the same scores? Right now, no — and that limits how seriously you can use these signals as features.

## Proposed architecture

Think of this as a layered pipeline with a single canonical event store at its core. Each layer is independently testable and swappable:

**Layer 1 — Ingestion.** Polygon + Alpaca + SEC EDGAR (free, mandatory disclosures via 8-K) + RSS for Reuters/MarketWatch + Benzinga (cheap tier, fast). Normalize everything to a common schema: `{source, ts, headline, body, primary_tickers, raw_url, hash}`. Dedupe by content hash and clustering similar headlines (same story across 10 outlets shouldn't count 10x).

**Layer 2 — Cheap deterministic classification.** Source reliability weight, regex detection of high-signal patterns (8-K filing, "halts trading," "FDA approves"), event-type taxonomy where a rule reliably catches it. Don't waste LLM calls on stuff regex handles cleanly.

**Layer 3 — LLM enrichment (the only place LLM lives in the news path).** For everything not handled at Layer 2, batch articles through Haiku with structured output — see prompt schema below. Cache by content hash so reruns are free.

**Layer 4 — Per-symbol rolling features.** Compute multi-horizon aggregates from scored events: materiality-weighted sentiment over 1d/3d/7d, event-type counts, max single-event impact in window. These become features.

**Layer 5 — Two consumption paths, kept separate:**

- **Model feature path** — features go into XGBoost the same way at train and inference. Backfill once across history so distributions match.
- **Risk overlay path** — hard gates for catastrophic events only: earnings within holding window, trading halt, M&A pending, FDA action. Binary, conservative, narrow — not a fuzzy sentiment threshold.

The key insight: the gate and the feature are doing different jobs. The gate catches discrete events that invalidate the model's assumptions. The feature gives the model a continuous signal to incorporate. Today you've collapsed them into one fuzzy mechanism.

## LLM prompt structure

Don't ask the LLM "should I block this trade" — that conflates news interpretation with trading policy. Ask it to interpret; let your code decide.

```json
{
  "is_about_symbol": true,
  "event_type": "earnings | m&a | regulatory | analyst | product | legal | macro | other",
  "directional_impact": -0.7,
  "confidence": 0.85,
  "horizon": "intraday | 1-3d | 1-2w | longer",
  "material": true,
  "rationale": "one sentence"
}
```

Tier the routing: Haiku for the volume pass, escalate to Sonnet only on low-confidence or high-materiality cases. At ~250–500 articles/day for top-50 SPX, Haiku-only spend is roughly $0.20/day, well under your $5 budget. That leaves headroom for occasional Sonnet escalation and morning digest synthesis.

## Earnings and macro

**Earnings rule:** hard veto on swing entry within N days of earnings (start with N=5). Use Polygon (you already pay) or Finnhub free tier. This is the single highest-value change you can make this week.

**Macro events:** FRED API for economic releases, ForexFactory or Investing.com for the calendar. Treat as a regime flag rather than a per-symbol signal. On Fed day: no new intraday entries 30 min before announcement, tighten swing sizing. CPI/NFP mornings: same posture.

## Source comparison

| Source | Cost | What it adds | Priority |
|---|---|---|---|
| SEC EDGAR (8-K) | Free | Real-time mandatory disclosures (M&A, exec departures, material agreements). Often before press picks it up. | High |
| Earnings calendar (Finnhub or Polygon) | Free / included | Single biggest binary risk you're not handling | High |
| FRED + macro calendar | Free | Regime signal | High |
| Benzinga Pro | ~$10–25/mo | Fast tape, analyst actions, halts | Medium |
| Reuters/MarketWatch RSS | Free | Source diversity, reliability weighting | Medium |
| StockTwits sentiment | Free tier | Crowd contrarian signal — noisy, low priority for top-50 SPX | Low |
| Reddit/Twitter | Free | Mostly noise at this universe; skip until later | Skip |

## Top 3 for the next two weeks

1. **Earnings calendar hard gate.** Highest risk reduction, smallest code footprint. Half a day of work.
2. **LLM materiality scoring with structured output, replacing keyword classifier and as primary sentiment source.** Build the prompt, JSON schema, Haiku batch call, content-hash cache, and persist scored events in SQLite so they're replay-able. This is the load-bearing change.
3. **SEC EDGAR 8-K ingestion.** Free, fast, mandatory — picks up the events that matter and rarely show up in regular news fast enough.

Defer the morning digest, news-as-feature in the model, and intraday news features until after these three. Once you have a replay-able event store and clean LLM scores, those become straightforward extensions rather than rebuilds.

## Things you haven't flagged

- **Train/serve skew on backfill.** If you start scoring with LLM today but your model was trained on Polygon labels, the inference-time signal distribution doesn't match training. You either need to backfill LLM scores across historical news (one-time cost, do it) or keep two parallel signals during transition.
- **Deduplication and source weighting.** Same story across Reuters + AP + 8 aggregators is one event, not 10. Cluster before scoring.
- **Halts and circuit breakers.** Different signal from news, hard gate, separate ingestion path.
- **Survivorship bias in news training data.** Delisted tickers' news history is often incomplete in vendor archives — affects backtest realism.
- **Silent stocks.** Zero news in a window is itself a feature; don't let it default to "neutral 0" indistinguishably from "scored 0."
- **ToS for algo consumption.** Worth verifying Polygon/Alpaca/Benzinga terms allow algorithmic use — personal-trading licenses can be more restrictive than enterprise.
