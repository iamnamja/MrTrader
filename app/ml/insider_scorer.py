"""
Insider-Buying-Cluster scorer — candidate SECOND independent edge (orthogonal to PEAD).

Thesis (academic):
    Cohen, Malloy & Pomorski (2012, J. Finance) — "opportunistic" insiders making
    open-market PURCHASES predict positive abnormal returns over 1-6 months.
    Lakonishok & Lee (2001, RFS) — aggregate/cluster insider BUYING (not selling)
    is informative; multiple insiders and larger size are the strong signals.

Signal (long-only):
    A "cluster" is >=2 distinct insiders making open-market purchases of the same
    stock within a trailing 30-day window, OR a single large purchase (>= $1M
    notional). The signal fires on the cluster's latest Form 4 FILING date (the
    PIT-public date — NOT the transaction date, which precedes the filing by up to
    2 business days and is not yet public). Hold for the drift window (the
    AgentSimulator max_hold_bars; insider drift is months, so the default ~40-bar
    swing hold is a reasonable first pass).

    Shorting insider SELLING is deliberately NOT done — insiders sell for many
    non-informational reasons (liquidity, diversification), a known weak/false
    signal. Long-only.

Interface:
    Implements the SAME __call__(day, symbols_data, vix_history) -> [(sym, conf,
    "long")] contract PEADScorer uses, so AgentSimulator.factor_scorer works
    unchanged.

PIT-safety:
    All filings are filtered to filing_date <= as_of via
    fmp_provider.get_insider_features_at(sym, as_of). The fetch is cached
    per-symbol (full history fetched once, sliced in memory) — no per-day API
    calls, matching the get_analyst_grades_fmp caching pattern.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Confidence range. Insider clusters are a well-studied but moderate-strength
# anomaly; map size/breadth onto a confidence band similar to PEAD's.
CONF_MIN = 0.65
CONF_MAX = 0.90

# VIX regime gate (principled, same levels as PEAD — long-run ~90th pct = crisis).
VIX_BLOCK_ALL = 30.0   # crisis — block all new entries
VIX_CONF_REF = 15.0    # below: full confidence; above: linear damping


class InsiderClusterScorer:
    """Insider-buying-cluster signal for use as AgentSimulator.factor_scorer.

    Long-only. PIT-safe (get_insider_features_at windows filings to <= as_of).

    Usage:
        scorer = InsiderClusterScorer()
        sim = AgentSimulator(..., factor_scorer=scorer)
    """

    def __init__(
        self,
        max_days_after_filing: int | None = None,
        vix_block_all: float = VIX_BLOCK_ALL,
        vix_conf_ref: float = VIX_CONF_REF,
    ):
        from app.data.fmp_provider import INSIDER_MAX_DAYS_AFTER_FILING
        self.max_days_after_filing = (
            INSIDER_MAX_DAYS_AFTER_FILING if max_days_after_filing is None
            else max_days_after_filing
        )
        self.vix_block_all = vix_block_all
        self.vix_conf_ref = vix_conf_ref

    def _vix_today(self, day, symbols_data: dict):
        """PIT VIX close for `day` from symbols_data. None if unavailable."""
        import pandas as pd
        _ts = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        for key in ("^VIX", "VIX"):
            vix_df = symbols_data.get(key)
            if vix_df is not None and not vix_df.empty:
                try:
                    # strict PIT: prior day's VIX close (today's close isn't knowable at the open).
                    _sub = vix_df[vix_df.index < _ts]
                    if len(_sub):
                        return float(_sub.iloc[-1]["close"])
                except Exception:
                    pass
        return None

    def __call__(self, day, symbols_data: dict, vix_history=None) -> list:
        """Score symbols on `day`. Returns [(sym, conf, "long")] for fresh clusters."""
        import pandas as pd
        from app.data.fmp_provider import get_insider_features_at

        _ts = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        as_of = _ts.date()

        # ── Regime gate ───────────────────────────────────────────────────────
        vix = self._vix_today(day, symbols_data)
        if vix is not None and vix > self.vix_block_all:
            logger.debug("Insider: VIX=%.1f > %.0f — blocking entries (crisis)",
                         vix, self.vix_block_all)
            return []

        vix_mult = 1.0
        if vix is not None and vix > self.vix_conf_ref:
            vix_mult = max(0.3, self.vix_conf_ref / vix)

        results = []
        for sym, df in symbols_data.items():
            if sym in ("^VIX", "VIX", "SPY"):
                continue
            if df is None or df.empty:
                continue

            try:
                feats = get_insider_features_at(sym, as_of)
            except Exception as e:
                logger.debug("Insider: %s fetch failed: %s", sym, e)
                continue

            if feats is None:
                continue  # no purchase history at all

            if feats.get("insider_is_cluster", 0.0) < 1.0:
                continue  # purchases exist but no qualifying cluster in window

            # Signal freshness: act within max_days_after_filing of the latest
            # filing in the cluster (enter near the event, not weeks later).
            days_since = feats.get("insider_days_since_filing")
            if days_since is None or days_since > self.max_days_after_filing:
                continue

            # Map cluster breadth + size onto confidence.
            #   breadth: more distinct buyers → higher conviction (consensus)
            #   size:    larger total notional → higher conviction
            n_buyers = feats.get("insider_distinct_buyers", 0.0)
            max_notional = feats.get("insider_max_notional", 0.0)
            # Breadth bonus: +0.05 per buyer beyond the first (cap effect via CONF_MAX).
            breadth_bonus = max(0.0, (n_buyers - 1.0)) * 0.05
            # Size bonus: +0.05 if a single buy exceeds the large-buy threshold.
            from app.data.fmp_provider import INSIDER_LARGE_BUY_NOTIONAL
            size_bonus = 0.05 if max_notional >= INSIDER_LARGE_BUY_NOTIONAL else 0.0
            conf = min(CONF_MIN + breadth_bonus + size_bonus, CONF_MAX)
            conf = max(conf, CONF_MIN)
            conf_gated = conf * vix_mult

            results.append((sym, conf_gated, "long"))

        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results
