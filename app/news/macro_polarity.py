"""Macro event POLARITY — the single source of truth for "what does this surprise MEAN for equities".

A macro surprise's market impact is NOT a uniform "higher = good" (the earnings framing). For inflation
and labor-slack prints LOWER is risk-ON (good for equities); for growth/activity prints HIGHER is
risk-ON. Both the UI outcome label (Beat/Miss → Cooler/Hotter/Stronger/Weaker) and, later, the LLM
macro prompt read this map so they agree.

Polarity values:
  "lower_better"  — a print BELOW consensus is risk-ON (inflation, jobless claims, unemployment)
  "higher_better" — a print ABOVE consensus is risk-ON (growth, payrolls, activity)
  "neutral"       — direction's market meaning is ambiguous/contextual (FOMC decision, OTHER_HIGH)

event_type values are those produced by finnhub_source._classify_event.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

# Within ±this fraction of consensus, treat the print as in-line (no risk-on/off read).
IN_LINE_BAND = 0.005  # 0.5%

POLARITY: Dict[str, str] = {
    # lower-is-risk-on
    "CPI": "lower_better",
    "PPI": "lower_better",
    "PCE": "lower_better",
    "UNEMPLOYMENT": "lower_better",
    "JOBLESS_CLAIMS": "lower_better",
    # higher-is-risk-on
    "NFP": "higher_better",
    "GDP": "higher_better",
    "RETAIL_SALES": "higher_better",
    "ISM_MFG": "higher_better",
    "ISM_SVC": "higher_better",
    # ambiguous / contextual
    "FOMC": "neutral",
    "OTHER_HIGH": "neutral",
}


def polarity_for(event_type: Optional[str]) -> str:
    """Polarity for an event type; unknown types default to 'neutral' (no risk-on/off claim)."""
    return POLARITY.get(event_type or "", "neutral")


def classify_outcome(event_type: Optional[str], actual: Any, estimate: Any) -> Dict[str, Any]:
    """Map (event_type, actual, estimate) → a polarity-aware market outcome.

    Returns:
      market_outcome: 'risk_on' | 'risk_off' | 'in_line' | 'pending'
      outcome_label : human label (Cooler/Hotter/Stronger/Weaker/Above est/Below est/In-Line/Pending)
      polarity      : the event's polarity
      surprise_pct  : signed (actual-estimate)/|estimate| or None

    NEVER raises — bad/blank inputs → 'pending'. A higher market_outcome is NOT inferred for a
    neutral-polarity event (we only report the raw direction, no risk colour)."""
    pol = polarity_for(event_type)
    try:
        a = float(actual)
        e = float(estimate)
    except (TypeError, ValueError):
        return {"market_outcome": "pending", "outcome_label": "Pending",
                "polarity": pol, "surprise_pct": None}

    diff = a - e
    denom = abs(e) if e != 0 else (abs(a) if a != 0 else None)
    surprise_pct = (diff / denom) if denom else None

    # in-line band (relative if we have a denom, else exact equality)
    if (denom is not None and abs(diff) / denom < IN_LINE_BAND) or (denom is None and diff == 0):
        return {"market_outcome": "in_line", "outcome_label": "In-Line",
                "polarity": pol, "surprise_pct": surprise_pct}

    above = diff > 0
    if pol == "lower_better":
        # below consensus = risk-on (cooler), above = risk-off (hotter)
        return {"market_outcome": "risk_off" if above else "risk_on",
                "outcome_label": "Hotter" if above else "Cooler",
                "polarity": pol, "surprise_pct": surprise_pct}
    if pol == "higher_better":
        return {"market_outcome": "risk_on" if above else "risk_off",
                "outcome_label": "Stronger" if above else "Weaker",
                "polarity": pol, "surprise_pct": surprise_pct}
    # neutral: report direction only, no risk-on/off claim
    return {"market_outcome": "in_line", "outcome_label": "Above est" if above else "Below est",
            "polarity": pol, "surprise_pct": surprise_pct}


# ── Day-level deterministic floor (the LLM fail-safe + clamp) ─────────────────────
_RISK_ORDER = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def aggregate_day(events: Any) -> Dict[str, Any]:
    """Deterministic day-level macro FLOOR from the per-event polarity outcomes.

    This is the fail-safe (used directly when the LLM is unavailable) AND the safety clamp (the LLM
    result is never allowed to be LESS conservative than this). Returns:
      min_risk    : the MINIMUM acceptable risk level (LOW|MEDIUM|HIGH)
      block       : True when new entries MUST be blocked (an unreleased high-impact event)
      max_sizing  : the MAXIMUM acceptable global sizing factor
      net_lean    : BULLISH | BEARISH | NEUTRAL (net of released high-impact surprises)
      all_released: every high-impact event has an actual
      n_high      : number of high-impact events

    Logic: an unreleased high-impact event ⇒ HIGH + block (genuine uncertainty). All released with a
    material ADVERSE (risk_off) high-impact surprise ⇒ MEDIUM, size ≤0.85, no block (outcome known).
    All released and benign/in-line/risk-on ⇒ LOW, 1.0 (the step-down). No high-impact events ⇒ LOW."""
    high = [e for e in (events or []) if str(e.get("importance", "")).lower() == "high"]
    if not high:
        return {"min_risk": "LOW", "block": False, "max_sizing": 1.0,
                "net_lean": "NEUTRAL", "all_released": True, "n_high": 0}
    unreleased = [e for e in high if e.get("actual") is None]
    outcomes = [classify_outcome(e.get("event_type"), e.get("actual"), e.get("estimate")) for e in high]
    risk_on = sum(1 for o in outcomes if o["market_outcome"] == "risk_on")
    risk_off = sum(1 for o in outcomes if o["market_outcome"] == "risk_off")
    net_lean = "BULLISH" if risk_on > risk_off else "BEARISH" if risk_off > risk_on else "NEUTRAL"
    if unreleased:
        return {"min_risk": "HIGH", "block": True, "max_sizing": 0.75,
                "net_lean": net_lean, "all_released": False, "n_high": len(high)}
    if risk_off > 0:
        return {"min_risk": "MEDIUM", "block": False, "max_sizing": 0.85,
                "net_lean": net_lean, "all_released": True, "n_high": len(high)}
    return {"min_risk": "LOW", "block": False, "max_sizing": 1.0,
            "net_lean": net_lean, "all_released": True, "n_high": len(high)}


def clamp_to_floor(risk_level: str, block: bool, sizing_factor: float,
                   floor: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp an LLM macro result to be NEVER less conservative than the deterministic floor:
    risk = max(llm, floor.min_risk); block = llm OR floor.block; sizing = min(llm, floor.max_sizing).
    (The floor only ever RAISES risk/block and LOWERS sizing — the LLM still drives the step-DOWN
    to LOW when the floor permits it.)"""
    try:
        _llm_sz = float(sizing_factor)
    except (TypeError, ValueError):
        _llm_sz = floor["max_sizing"]
    final_risk = max(str(risk_level or "LOW"), floor["min_risk"],
                     key=lambda r: _RISK_ORDER.get(r, 1))
    return {
        "risk_level": final_risk,
        "block_new_entries": bool(block) or floor["block"],
        "sizing_factor": min(_llm_sz, floor["max_sizing"]),
    }
