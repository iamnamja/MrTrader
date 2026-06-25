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
