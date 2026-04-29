"""
LLM scorer for the News Intelligence Service.

Tier 1 — macro_classify(): score an economic calendar day → MacroContext
Tier 2 — stock_score():    score news for a specific symbol → NewsSignal

Design principles (from NIS spec):
- LLM is an interpreter, not a trader. Prompts never mention buy/sell/long/short.
- Deterministic Python maps LLM output to action. LLM outputs structured JSON only.
- Cache by content-hash so the same news cluster is never re-scored within TTL.
- Fail open: any LLM error returns a neutral/fallback signal, never blocks trading.
- Costs logged to llm_call_log for daily budget tracking.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

# Haiku pricing (per million tokens)
_HAIKU_INPUT_COST = 0.80 / 1_000_000
_HAIKU_OUTPUT_COST = 4.00 / 1_000_000
_MODEL = "claude-haiku-4-5-20251001"

MACRO_PROMPT_VERSION = "macro_v1"
STOCK_PROMPT_VERSION = "stock_v1"

# ── In-memory LRU cache (TTL enforced at read time) ───────────────────────────
# key → (result_dict, evaluated_at_epoch)
_cache: dict[str, tuple[dict, float]] = {}
_CACHE_TTL_FRESH = 3600       # 1 hour for same-day articles
_CACHE_TTL_OLD = 86400        # 24 hours for older articles


def _cache_key(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:32]


def _cache_get(key: str, ttl: int = _CACHE_TTL_FRESH) -> Optional[dict]:
    entry = _cache.get(key)
    if entry is None:
        return None
    result, ts = entry
    if time.time() - ts > ttl:
        _cache.pop(key, None)
        return None
    return result


def _cache_set(key: str, result: dict) -> None:
    _cache[key] = (result, time.time())


# ── Anthropic client ──────────────────────────────────────────────────────────

def _client():
    try:
        from app.config import settings
        key = settings.anthropic_api_key
        if not key:
            return None
        from anthropic import Anthropic
        return Anthropic(api_key=key)
    except Exception:
        return None


def _call_llm(prompt: str, call_type: str, symbol: Optional[str] = None) -> Optional[dict]:
    """
    Call Haiku with prompt, parse JSON response, log cost.
    Returns parsed dict or None on any failure.
    """
    client = _client()
    if client is None:
        logger.debug("Anthropic client unavailable — skipping LLM score")
        return None

    t0 = time.time()
    try:
        resp = client.messages.create(
            model=_MODEL,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        latency_ms = int((time.time() - t0) * 1000)
        raw = resp.content[0].text.strip()
        input_tok = resp.usage.input_tokens
        output_tok = resp.usage.output_tokens
        cost = input_tok * _HAIKU_INPUT_COST + output_tok * _HAIKU_OUTPUT_COST

        _log_call(call_type, symbol, input_tok, output_tok, latency_ms, cost)

        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw.strip())

    except json.JSONDecodeError as exc:
        logger.warning("LLM JSON parse failed (%s): %s", call_type, exc)
        return None
    except Exception as exc:
        logger.warning("LLM call failed (%s): %s", call_type, exc)
        return None


def _log_call(
    call_type: str,
    symbol: Optional[str],
    input_tok: int,
    output_tok: int,
    latency_ms: int,
    cost: float,
    cache_hit: bool = False,
) -> None:
    try:
        from app.database.session import get_session
        from app.database.models import LLMCallLog
        with get_session() as db:
            db.add(LLMCallLog(
                call_type=call_type,
                symbol=symbol,
                model_name=_MODEL,
                prompt_version=MACRO_PROMPT_VERSION if "macro" in call_type else STOCK_PROMPT_VERSION,
                input_tokens=input_tok,
                output_tokens=output_tok,
                latency_ms=latency_ms,
                cost_usd=cost,
                cache_hit=cache_hit,
            ))
            db.commit()
    except Exception as exc:
        logger.debug("LLM call log failed (non-fatal): %s", exc)


# ── Tier 1: Macro classifier ──────────────────────────────────────────────────

_MACRO_SYSTEM = """\
You are a financial event classification system. Your role is to assess whether
today's scheduled economic events represent meaningful uncertainty for equity markets,
or whether outcomes are already well-priced into markets.

CRITICAL RULES:
- Return STRICT JSON only. No prose before or after the JSON block.
- Do NOT recommend trades. Do NOT use words like buy, sell, long, short.
- If consensus == prior and actual is missing: likely priced in, risk is LOW unless the
  event type historically causes large surprises.
- Only output HIGH risk when there is genuine uncertainty (split consensus, surprise-prone event,
  or actual result differs significantly from estimate).
"""


def macro_classify(events: list[dict]) -> Optional[dict]:
    """
    Tier 1: Classify today's macro events into a single day-level risk signal.

    Input: list of event dicts from finnhub_source.fetch_economic_calendar()
    Returns: dict with risk_level, direction, sizing_factor, block_new_entries, rationale
             or None on failure (caller uses neutral fallback).
    """
    if not events:
        return {
            "risk_level": "LOW",
            "direction": "NEUTRAL",
            "sizing_factor": 1.0,
            "block_new_entries": False,
            "consensus_summary": "No high-impact events scheduled",
            "rationale": "No significant economic events today",
        }

    # Build cache key from event types + times + estimates
    key_parts = [MACRO_PROMPT_VERSION] + [
        f"{e['event_type']}:{e['event_time']}:{e.get('estimate')}:{e.get('actual')}"
        for e in events
    ]
    ckey = _cache_key(*key_parts)
    cached = _cache_get(ckey, ttl=_CACHE_TTL_FRESH)
    if cached:
        logger.debug("Macro classify cache hit")
        return cached

    # Build event table for prompt
    lines = []
    for e in events:
        est = e.get("estimate")
        prior = e.get("prior")
        actual = e.get("actual")
        already_released = actual is not None
        status = f"actual={actual}" if already_released else "not yet released"
        lines.append(
            f"- {e['event_type']} ({e['importance'].upper()} impact) "
            f"at {str(e['event_time'])[:16]} UTC | "
            f"prior={prior} | consensus={est} | {status}"
        )

    prompt = f"""{_MACRO_SYSTEM}

Today's scheduled economic events:
{chr(10).join(lines)}

Classify the overall day-level risk for NEW equity swing entries (3-5 day holds).

Return JSON with exactly these fields:
{{
  "risk_level": "<LOW|MEDIUM|HIGH>",
  "direction": "<BULLISH|NEUTRAL|BEARISH>",
  "sizing_factor": <float 0.5-1.0, where 1.0=no change, 0.5=half size>,
  "block_new_entries": <true only when risk_level=HIGH AND outcome is genuinely uncertain>,
  "consensus_summary": "<one sentence: what markets expect and why>",
  "rationale": "<one sentence: why this risk level was chosen>"
}}

Sizing guidance:
- LOW risk (consensus priced in, small expected move): sizing_factor=1.0
- MEDIUM risk (some uncertainty, moderate impact): sizing_factor=0.75
- HIGH risk but outcome known (actual released): sizing_factor=0.85, block=false
- HIGH risk, outcome unknown, large expected move: sizing_factor=0.5, block=true
"""

    result = _call_llm(prompt, call_type="macro_tier1")
    if result:
        _cache_set(ckey, result)
    return result


# ── Tier 2: Stock news scorer ─────────────────────────────────────────────────

_STOCK_SYSTEM = """\
You are a financial news materiality assessment system. Your role is to evaluate
whether recent news about a specific company is material to its near-term price risk.

CRITICAL RULES:
- Return STRICT JSON only. No prose before or after the JSON block.
- Do NOT recommend trades. Do NOT use words like buy, sell, long, short, enter, exit.
- Focus on: is this news new? is it already reflected in the price? how material is it?
- If there is no news or only generic market recap articles: set materiality_score=0.0,
  confidence=0.2, action_policy="ignore".
- "already_priced_in_score" should be HIGH (0.7+) if the stock has already moved
  significantly in the news direction before this assessment.
"""


def stock_score(
    symbol: str,
    articles: list[dict],
    sector: str = "Unknown",
    macro_context_summary: str = "No significant macro events today",
    lookback_hours: int = 24,
) -> Optional[dict]:
    """
    Tier 2: Score recent news for a symbol.

    Input: articles from finnhub_source.fetch_company_news()
    Returns: dict with direction_score, materiality_score, downside_risk_score,
             upside_catalyst_score, confidence, already_priced_in_score,
             action_policy, sizing_multiplier, rationale
             or None on failure.
    """
    if not articles:
        return {
            "direction_score": 0.0,
            "materiality_score": 0.0,
            "downside_risk_score": 0.0,
            "upside_catalyst_score": 0.0,
            "confidence": 0.2,
            "already_priced_in_score": 0.0,
            "action_policy": "ignore",
            "sizing_multiplier": 1.0,
            "rationale": "No recent news",
        }

    # Build cache key from headlines (not full bodies — cheap and stable)
    headlines = [a["headline"] for a in articles[:10]]
    ckey = _cache_key(STOCK_PROMPT_VERSION, symbol, *headlines)
    cached = _cache_get(ckey, ttl=_CACHE_TTL_FRESH)
    if cached:
        logger.debug("Stock score cache hit for %s", symbol)
        return cached

    # Format articles for prompt (cap at 10 to stay within token budget)
    article_lines = []
    for a in articles[:10]:
        pub = a["published_at"].strftime("%Y-%m-%d %H:%M UTC") if isinstance(
            a["published_at"], datetime) else str(a["published_at"])[:16]
        sentiment_hint = f" [pre-labeled: {a['sentiment']}]" if a.get("sentiment") else ""
        article_lines.append(
            f"- [{pub}] {a['headline']}{sentiment_hint}\n"
            f"  {a.get('summary', '')[:200]}"
        )

    prompt = f"""{_STOCK_SYSTEM}

Symbol: {symbol}
Sector: {sector}
Assessment horizon: swing trade (3-5 day hold)
News lookback: last {lookback_hours} hours
Macro context: {macro_context_summary}

Recent news articles ({len(articles)} total, showing top {min(len(articles), 10)}):
{chr(10).join(article_lines)}

Return JSON with exactly these fields:
{{
  "direction_score": <float [-1.0, 1.0], negative=bearish for {symbol}, 0=neutral>,
  "materiality_score": <float [0.0, 1.0], how relevant/important is this news>,
  "downside_risk_score": <float [0.0, 1.0], probability of meaningful downside>,
  "upside_catalyst_score": <float [0.0, 1.0], strength of positive catalyst>,
  "confidence": <float [0.0, 1.0], how confident are you in this assessment>,
  "already_priced_in_score": <float [0.0, 1.0], 1.0=fully priced in already>,
  "action_policy": "<ignore|watch|size_down_light|size_down_heavy|size_up_light|block_entry|exit_review>",
  "sizing_multiplier": <float [0.3, 1.25], 1.0=no change>,
  "rationale": "<one sentence, max 150 chars>"
}}

action_policy guidance:
- ignore: no material news
- watch: minor news worth monitoring
- size_down_light (0.75×): moderate negative uncertainty
- size_down_heavy (0.50×): significant downside risk
- size_up_light (1.15×): strong positive catalyst, not yet priced in
- block_entry: do NOT enter — material risk event imminent (e.g. earnings today, FDA ruling)
- exit_review: existing position should be reviewed for early exit
"""

    result = _call_llm(prompt, call_type="stock_tier2", symbol=symbol)
    if result:
        # Clamp values to valid ranges
        result["direction_score"] = max(-1.0, min(1.0, float(result.get("direction_score", 0))))
        result["sizing_multiplier"] = max(0.3, min(1.25, float(result.get("sizing_multiplier", 1.0))))
        for field in ("materiality_score", "downside_risk_score", "upside_catalyst_score",
                      "confidence", "already_priced_in_score"):
            result[field] = max(0.0, min(1.0, float(result.get(field, 0.0))))
        _cache_set(ckey, result)
    return result
