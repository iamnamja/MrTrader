"""
Dollar-neutral short-interest factor scorer (Alpha-v3 A2).

Economic thesis (Boehmer/Asquith/Desai short-interest anomaly): heavily-shorted,
high-days-to-cover names systematically UNDERPERFORM (short sellers are informed),
while low-short-interest names behave ~market. So a **dollar-neutral** book that is
LONG the lowest-days-to-cover decile and SHORT the highest-days-to-cover decile
should capture that spread with little market beta.

Why neutral-by-construction (the A1 lesson): A1 analyst-drift looked great long-only
(+0.894, t=2.85) but was beta/fold-skip noise — neutralizing collapsed it. This book
is dollar-neutral from day one, so its CPCV result IS the beta-isolated test; no
separate L/S re-run needed. (We still CAPM-check it.)

A single, economically-grounded factor (short interest) — NOT the dead ML
cross-sectional ranker (kitchen-sink price/fundamental features → noise).

Returns the AgentSimulator factor-scorer contract: [(sym, conf, "long"|"short")].
Signal-mode shorts are taken without enable_shorts. days_to_cover updates only
bi-monthly (knowable_date stepped), so the long/short set is stable between SI
prints and positions are held ~max_hold_bars by the harness.
"""

import logging

logger = logging.getLogger(__name__)

_SYNTHETIC = {"^VIX", "VIX", "SPY"}


class ShortInterestFactorScorer:
    def __init__(self, n_per_leg: int = 25, min_dtc_short: float = 2.0,
                 max_dtc_long: float | None = None, long_short: bool = True,
                 vix_block_all: float = 30.0):
        """
        n_per_leg:     names per side (long lowest-DTC, short highest-DTC).
        min_dtc_short: only short names with days_to_cover >= this (a real crowded
                       short; avoids shorting names with trivial SI).
        max_dtc_long:  optional cap on the long leg's DTC (default None = no cap).
        long_short:    True = dollar-neutral L/S (the thesis). False = short-only.
        """
        self.n_per_leg = n_per_leg
        self.min_dtc_short = min_dtc_short
        self.max_dtc_long = max_dtc_long
        self.long_short = long_short
        self.vix_block_all = vix_block_all

    def _vix_today(self, day, symbols_data):
        import pandas as pd
        vdf = symbols_data.get("^VIX")
        if vdf is None:
            vdf = symbols_data.get("VIX")
        if vdf is None or len(vdf) == 0:
            return None
        try:
            sub = vdf[vdf.index < pd.Timestamp(day)]   # strict PIT: prior day's VIX close
            return float(sub["close"].iloc[-1]) if len(sub) else None
        except Exception:
            return None

    def __call__(self, day, symbols_data, vix_history=None) -> list:
        import pandas as pd
        from app.data.short_interest_provider import latest_known_si

        as_of = pd.Timestamp(day).date()

        vix = self._vix_today(day, symbols_data)
        if vix is not None and vix > self.vix_block_all:
            return []  # crisis: stand down

        panel = latest_known_si(as_of)
        if panel is None or panel.empty:
            return []

        # Restrict to the tradable universe with valid, positive days-to-cover.
        avail = {s for s in symbols_data if s not in _SYNTHETIC}
        panel = panel[panel["ticker"].isin(avail)]
        panel = panel[panel["days_to_cover"].notna() & (panel["days_to_cover"] > 0)]
        if len(panel) < 2 * self.n_per_leg:
            return []

        panel = panel.sort_values("days_to_cover")

        # SHORT leg: highest days-to-cover (crowded, informed-short underperformers),
        # subject to the min-DTC floor (only short a genuinely crowded name).
        shorts = panel[panel["days_to_cover"] >= self.min_dtc_short].tail(self.n_per_leg)

        if not self.long_short:
            return [(r["ticker"], -1.0, "short") for _, r in shorts.iterrows()]

        # Dollar-neutral: LONG lowest-DTC, disjoint from shorts, BALANCED to the short
        # count. If the short leg can't populate (e.g. min-DTC floor bites), emit
        # NOTHING — never fire a long-only leg (that reintroduces the beta trap A1 hit).
        long_pool = panel[~panel["ticker"].isin(set(shorts["ticker"]))]
        if self.max_dtc_long is not None:
            long_pool = long_pool[long_pool["days_to_cover"] <= self.max_dtc_long]
        longs = long_pool.head(self.n_per_leg)

        n = min(len(longs), len(shorts))
        if n == 0:
            return []
        results = []
        for _, r in shorts.tail(n).iterrows():          # the n highest-DTC
            results.append((r["ticker"], -1.0, "short"))
        for _, r in longs.head(n).iterrows():            # the n lowest-DTC
            results.append((r["ticker"], 1.0, "long"))
        return results
