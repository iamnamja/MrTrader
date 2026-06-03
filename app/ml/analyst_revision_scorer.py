"""
Analyst up/downgrade drift scorer (Alpha-v3 A1).

Economic thesis (PEAD's close cousin, well documented): the market under-reacts to
analyst rating changes, so prices DRIFT in the direction of a recent upgrade /
downgrade for weeks. Discrete event -> measurable drift -> rules-based, F2-immune.

Event = an analyst **upgrade** or **downgrade** (action != "maintain"), from FMP
`/stable/grades` (already wired: get_analyst_grades_fmp). Same-day public, so no
dissemination lag -- filtering grade `date <= as_of` is point-in-time correct.

Signal (mirrors PEADScorer's event + sign + recency structure):
  - find the most recent upgrade/downgrade on/before `day`
  - act only within [1, max_days_after] trading-proximate days of it (enter on the
    drift, not the same-day jump)
  - confirm direction with net revision momentum (upgrades - downgrades) over a
    trailing `lookback_days` window, so a lone contrarian rating into a stream of
    opposite changes is filtered out
  - long an upgrade (net >= 0); short a downgrade (net <= 0) when long_short=True

Returns the AgentSimulator factor-scorer contract: [(symbol, conf, "long"|"short")].
Entry-at-open + hold-for-drift-window are handled by AgentSimulator (drift window =
max_hold_bars, set via EventEdgeStrategy.max_hold_bars_override).
"""

import logging
from datetime import date as _date
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

_SYNTHETIC = {"^VIX", "VIX", "SPY"}


class AnalystRevisionScorer:
    def __init__(self, lookback_days: int = 30, max_days_after: int = 5,
                 min_net_momentum: float = 1.0, long_short: bool = False,
                 vix_block_all: float = 30.0, conf_cap: float = 3.0):
        self.lookback_days = lookback_days
        self.max_days_after = max_days_after
        self.min_net_momentum = min_net_momentum
        self.long_short = long_short
        self.vix_block_all = vix_block_all
        self.conf_cap = conf_cap

    # ── VIX regime gate (mirror PEAD's crisis block) ──────────────────────────
    def _vix_today(self, day, symbols_data):
        vdf = symbols_data.get("^VIX")
        if vdf is None:
            vdf = symbols_data.get("VIX")
        if vdf is None or len(vdf) == 0:
            return None
        try:
            import pandas as pd
            ts = pd.Timestamp(day)
            sub = vdf.loc[:ts]
            if len(sub) == 0:
                return None
            return float(sub["close"].iloc[-1])
        except Exception:
            return None

    def __call__(self, day, symbols_data, vix_history=None) -> list:
        import pandas as pd
        from app.data.fmp_provider import get_analyst_grades_fmp

        as_of = pd.Timestamp(day).date()

        vix = self._vix_today(day, symbols_data)
        if vix is not None and vix > self.vix_block_all:
            return []  # crisis: block all entries

        results = []
        for sym, df in symbols_data.items():
            if sym in _SYNTHETIC or df is None or len(df) == 0:
                continue
            try:
                grades = get_analyst_grades_fmp(sym)
            except Exception as e:
                logger.debug("analyst grades fetch failed for %s: %s", sym, e)
                continue
            if not grades:
                continue

            # PIT: only grades public on/before as_of.
            events = []  # (date, action) for upgrade/downgrade
            for r in grades:
                d = r.get("date")
                act = r.get("action")
                if not d or act not in ("upgrade", "downgrade"):
                    continue
                try:
                    gd = datetime.strptime(d, "%Y-%m-%d").date()
                except (TypeError, ValueError):
                    continue
                if gd <= as_of:
                    events.append((gd, act))
            if not events:
                continue

            events.sort(key=lambda x: x[0])
            last_date, last_action = events[-1]
            days_since = (as_of - last_date).days
            # Event recency: enter on the drift, not the announcement-day jump.
            if days_since < 1 or days_since > self.max_days_after:
                continue

            # Net revision momentum over the trailing window (confirmation filter).
            cutoff = as_of - timedelta(days=self.lookback_days)
            window = [e for e in events if e[0] >= cutoff]
            up = sum(1 for _, a in window if a == "upgrade")
            down = sum(1 for _, a in window if a == "downgrade")
            net = up - down

            conf = min(1.0, (abs(net) + 1.0) / (self.conf_cap + 1.0))
            # Recent event sets direction; net momentum must CONFIRM it with
            # strength >= min_net_momentum (filters a lone contrarian rating).
            if last_action == "upgrade" and net >= self.min_net_momentum:
                results.append((sym, conf, "long"))
            elif (self.long_short and last_action == "downgrade"
                  and net <= -self.min_net_momentum):
                results.append((sym, -conf, "short"))

        return results
