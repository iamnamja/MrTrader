"""
Post-Earnings Announcement Drift (PEAD) scorer — Phase G.

Uses FMP earnings surprise data (already fetched, PIT-safe) to identify
stocks with significant recent earnings surprises. Enters long on positive
surprise, short on negative surprise.

Typical PEAD signal:
    Long:  EPS surprise > +5%, report within last 3 trading days
    Short: EPS surprise < -5%, report within last 3 trading days
    Hold 5 days (short-term momentum after announcement)

Returns [(sym, conf, direction)] compatible with AgentSimulator.factor_scorer.

Public API:
    PEADScorer(long_threshold, short_threshold, max_days_after, max_hold_days)
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# How many calendar days after the earnings report to still enter
MAX_DAYS_AFTER_EARNINGS = 3

# EPS surprise thresholds (fraction: 0.05 = 5%)
LONG_SURPRISE_THRESHOLD  = 0.05
SHORT_SURPRISE_THRESHOLD = -0.05

# Confidence range for PEAD signals — high conviction since anomaly is well-studied
CONF_MIN = 0.65
CONF_MAX = 0.90


class PEADScorer:
    """PEAD signal generator for use as AgentSimulator.factor_scorer.

    Requires FMP earnings data to be cached (fmp_provider.get_earnings_features_at).
    Data is PIT-safe: only reports with report_date <= as_of are considered.

    Usage:
        scorer = PEADScorer()
        sim = AgentSimulator(..., factor_scorer=scorer)
    """

    def __init__(
        self,
        long_threshold: float = LONG_SURPRISE_THRESHOLD,
        short_threshold: float = SHORT_SURPRISE_THRESHOLD,
        max_days_after: int = MAX_DAYS_AFTER_EARNINGS,
        long_short: bool = True,
    ):
        self.long_threshold  = long_threshold
        self.short_threshold = short_threshold
        self.max_days_after  = max_days_after
        self.long_short      = long_short

    def __call__(
        self,
        day,
        symbols_data: dict,
        vix_history=None,
    ) -> list:
        """Score symbols on `day` using PEAD signal. Returns [(sym, conf, direction)]."""
        import pandas as pd
        from datetime import datetime, timedelta

        from app.data.fmp_provider import get_earnings_features_at

        as_of = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        cutoff = as_of - pd.Timedelta(days=self.max_days_after)

        results = []

        for sym, df in symbols_data.items():
            if df is None or df.empty:
                continue

            try:
                feats = get_earnings_features_at(sym, as_of)
            except Exception as e:
                logger.debug("PEAD: %s earnings fetch failed: %s", sym, e)
                continue

            if feats is None:
                continue

            # feats contains: fmp_surprise_1q, fmp_days_since_earnings, etc.
            surprise  = feats.get("fmp_surprise_1q")
            days_since = feats.get("fmp_days_since_earnings")

            if surprise is None or days_since is None:
                continue

            # Only act within max_days_after of the report
            if days_since > self.max_days_after:
                continue

            # Map surprise magnitude to confidence
            abs_surprise = abs(surprise)
            # Scale: 5% surprise → 0.65 conf, 20%+ surprise → 0.90 conf
            conf = min(CONF_MIN + (abs_surprise - abs(self.long_threshold)) * 2.0, CONF_MAX)
            conf = max(conf, CONF_MIN)

            if surprise >= self.long_threshold:
                results.append((sym, conf, "long"))
            elif self.long_short and surprise <= self.short_threshold:
                results.append((sym, -conf, "short"))

        # Sort by abs confidence descending
        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results
