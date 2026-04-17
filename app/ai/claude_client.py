"""
Claude AI integration for agent decision support.

Three use-cases:
  1. PM signal review   — before proposing a trade, Claude reviews the ML signal
                          and writes a concise plain-English reasoning summary.
  2. Risk veto explain  — when Risk Manager rejects a trade, Claude explains why
                          in plain English for the audit log / dashboard.
  3. Pre-trade summary  — a daily digest of all pending proposals before the cycle
                          runs, useful for human-in-the-loop review.

All calls are optional and non-blocking: if the API key is missing or the call
fails, the caller receives None and continues normally.

Model: claude-haiku-4-5 for low-latency, low-cost inline agent decisions.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_MODEL = "claude-haiku-4-5-20251001"
_MAX_TOKENS = 300


def _client():
    """Return Anthropic client, or None if key is not configured."""
    api_key = os.environ.get("ANTHROPIC_API_KEY") or ""
    if not api_key:
        return None
    try:
        from anthropic import Anthropic
        return Anthropic(api_key=api_key)
    except Exception as exc:
        logger.debug("Anthropic client init failed: %s", exc)
        return None


def review_pm_signal(
    symbol: str,
    signal_type: str,
    confidence: float,
    reasoning: Dict[str, Any],
    regime: str = "MEDIUM",
) -> Optional[str]:
    """
    Ask Claude to review a PM trade proposal and return a plain-English summary.
    Returns None if AI is unavailable.
    """
    client = _client()
    if client is None:
        return None
    try:
        price = reasoning.get("price", "?")
        stop = reasoning.get("stop", "?")
        target = reasoning.get("target", "?")
        rsi = reasoning.get("rsi", "?")
        prompt = (
            f"You are a quant trader reviewing a trade signal. Be concise (2-3 sentences).\n\n"
            f"Symbol: {symbol}\n"
            f"Signal type: {signal_type}\n"
            f"ML confidence: {confidence:.0%}\n"
            f"Market regime: {regime}\n"
            f"Price: ${price}  Stop: ${stop}  Target: ${target}  RSI: {rsi}\n\n"
            f"Summarise why this trade looks attractive (or flag any concerns). "
            f"Focus on risk/reward and the signal strength."
        )
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        logger.debug("PM signal review failed for %s: %s", symbol, exc)
        return None


def explain_risk_veto(
    symbol: str,
    failed_rule: str,
    veto_reason: str,
    proposal: Dict[str, Any],
) -> Optional[str]:
    """
    Ask Claude to explain a risk veto in plain English for the audit log.
    Returns None if AI is unavailable.
    """
    client = _client()
    if client is None:
        return None
    try:
        prompt = (
            f"You are a risk manager. Explain this trade rejection concisely (1-2 sentences) "
            f"in plain English for a trading journal.\n\n"
            f"Symbol: {symbol}\n"
            f"Failed rule: {failed_rule}\n"
            f"Reason: {veto_reason}\n"
            f"Proposed entry: ${proposal.get('entry_price', '?')}  "
            f"Qty: {proposal.get('quantity', '?')}\n\n"
            f"Explain what risk limit was breached and why it matters."
        )
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        logger.debug("Risk veto explanation failed for %s: %s", symbol, exc)
        return None


def summarise_daily_proposals(proposals: List[Dict[str, Any]]) -> Optional[str]:
    """
    Ask Claude to write a daily pre-market briefing for all pending proposals.
    Returns None if AI is unavailable or no proposals.
    """
    if not proposals:
        return None
    client = _client()
    if client is None:
        return None
    try:
        lines = []
        for p in proposals[:10]:  # cap at 10 to stay within tokens
            lines.append(
                f"- {p.get('symbol', '?')} | {p.get('signal_type', '?')} | "
                f"conf={p.get('confidence', 0):.0%} | "
                f"entry=${p.get('entry_price', '?')} stop=${p.get('stop_price', '?')}"
            )
        prompt = (
            f"You are a trading desk analyst. Write a concise pre-market briefing "
            f"(3-5 sentences) covering these {len(proposals)} proposed trades for today:\n\n"
            + "\n".join(lines)
            + "\n\nHighlight the strongest setups, note any sector concentration, "
            "and flag any concerns about timing or market conditions."
        )
        msg = client.messages.create(
            model=_MODEL,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        logger.debug("Daily proposals summary failed: %s", exc)
        return None
