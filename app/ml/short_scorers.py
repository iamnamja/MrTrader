"""
Phase H — L/S Short Selection Research scorers.

Four approaches to genuine short selection that should survive bear-market
reversals (the failure mode of bottom-N momentum shorts in Fold 2 / 2022):

  A. QualityShortScorer       — short fundamentally deteriorating names
  B. MeanReversionShortScorer — short stocks that have risen too far too fast
  C. SectorRelativeScorer     — sector-neutral top-3/bottom-3 within each GICS sector
  D. CombinedLSScorer         — PEAD + factor longs + quality shorts

All scorers conform to AgentSimulator.factor_scorer interface:
    (day, symbols_data, vix_history) -> [(sym, conf, direction)]

Conventions:
  - VIX >= 40           : extreme regime, sit in cash (return [])
  - SPY < SPY.MA200     : bear market — long leg suppressed, shorts allowed
  - conf > 0 + "long"   : long entry
  - conf < 0 + "short"  : short entry
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from app.ml.factor_scorer import (
    compute_composite_score,
    select_top_n,
)

logger = logging.getLogger(__name__)

EXTREME_VIX = 40.0
DEFAULT_VIX_REGIME = 30.0
SPY_MA_WINDOW = 200


# ── Shared regime helpers ────────────────────────────────────────────────────

def _build_closes_and_bars(day, symbols_data: dict):
    """Build aligned closes DataFrame and per-symbol PIT bars dict."""
    close_cols = {}
    for sym, df in symbols_data.items():
        if df is None or df.empty or "close" not in df.columns:
            continue
        if hasattr(df.index[0], "date"):
            mask = df.index.date < day
        else:
            mask = df.index < pd.Timestamp(day)
        past = df.loc[mask, "close"] if mask.any() else pd.Series(dtype=float)
        if len(past) >= 60:
            close_cols[sym] = past

    if not close_cols:
        return None, None, None

    closes = pd.DataFrame(close_cols)
    closes.index = pd.to_datetime(closes.index)
    as_of = closes.index[-1]

    bars_by_sym = {}
    for sym, df in symbols_data.items():
        if sym not in close_cols:
            continue
        if hasattr(df.index[0], "date"):
            bars_by_sym[sym] = df.loc[df.index.date < day]
        else:
            bars_by_sym[sym] = df.loc[df.index < pd.Timestamp(day)]

    return closes, bars_by_sym, as_of


def _vix_value(vix_history, as_of) -> Optional[float]:
    if vix_history is None:
        return None
    try:
        past = vix_history[vix_history.index <= as_of]
        if len(past) > 0:
            return float(past.iloc[-1])
    except Exception:
        return None
    return None


def _spy_bull_ok(closes: pd.DataFrame, as_of, ma_window: int = SPY_MA_WINDOW) -> bool:
    spy = closes.get("SPY", pd.Series(dtype=float))
    if spy.empty or len(spy) < ma_window:
        return True
    spy_last = float(spy.iloc[-1])
    spy_ma = float(spy.iloc[-ma_window:].mean())
    return spy_last > spy_ma


def _scale_conf(scores: pd.Series, value: float) -> float:
    s_min, s_max = float(scores.min()), float(scores.max())
    s_range = max(s_max - s_min, 1e-6)
    return 0.55 + 0.40 * ((value - s_min) / s_range)


# ── A. Quality-Short Scorer ──────────────────────────────────────────────────

class QualityShortScorer:
    """Short fundamentally deteriorating stocks; long top factor names.

    Short qualifies if >= flags_required of:
        - operating_margin <= 0
        - revenue_growth_yoy <= 0
        - debt_to_equity >= 1.5
        - fmp_surprise_1q <= -0.05 (within 90d)

    legs_mode:
        "both"        — long + short legs (default)
        "longs_only"  — emit only long picks
        "shorts_only" — emit only short picks
    """

    def __init__(self, top_n: int = 20, max_shorts: int = 15,
                 vix_threshold: float = DEFAULT_VIX_REGIME,
                 flags_required: int = 2,
                 legs_mode: str = "both"):
        self.top_n = top_n
        self.max_shorts = max_shorts
        self.vix_threshold = vix_threshold
        self.flags_required = flags_required
        self.legs_mode = legs_mode
        self._fmp_df = None

    def _fmp(self):
        if self._fmp_df is None:
            try:
                from app.data.fmp_fundamentals import load_fmp_fundamentals
                self._fmp_df = load_fmp_fundamentals()
            except Exception as exc:
                logger.warning("QualityShort: FMP load failed: %s", exc)
                self._fmp_df = pd.DataFrame()
        return self._fmp_df

    def __call__(self, day, symbols_data, vix_history=None):
        closes, bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []

        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        long_allowed = _spy_bull_ok(closes, as_of) and (vix is None or vix < self.vix_threshold)

        scores = compute_composite_score(as_of, closes, bars, use_tier2=True)
        if scores.empty:
            return []

        result = []
        long_set = set()
        if long_allowed and self.legs_mode in ("both", "longs_only"):
            longs = select_top_n(scores, n=self.top_n)
            long_set = set(longs)
            result.extend((s, _scale_conf(scores, float(scores[s])), "long") for s in longs)

        if self.legs_mode == "longs_only":
            return result

        # Build short candidates from FMP
        fmp_df = self._fmp()
        as_of_str = pd.Timestamp(as_of).strftime("%Y-%m-%d")
        candidates = []
        if fmp_df is not None and not fmp_df.empty:
            sub = fmp_df[fmp_df["as_of_date"] <= as_of_str]
            if not sub.empty:
                latest = sub.sort_values("as_of_date").groupby("symbol").tail(1)
                # Map to indexable subset (universe symbols only)
                universe = set(scores.index)
                latest = latest[latest["symbol"].isin(universe)]
                # Staleness guard: reject fundamentals older than 120 days
                _as_of_dt = pd.Timestamp(as_of_str)
                latest = latest[
                    (_as_of_dt - pd.to_datetime(latest["as_of_date"])).dt.days <= 120
                ]
                for _, row in latest.iterrows():
                    sym = row["symbol"]
                    if sym in long_set or sym == "SPY":
                        continue
                    flags = 0
                    om = row.get("operating_margin")
                    rg = row.get("revenue_growth_yoy")
                    de = row.get("debt_to_equity")
                    if pd.notna(om) and om is not None and om <= 0.0:
                        flags += 1
                    if pd.notna(rg) and rg is not None and rg <= 0.0:
                        flags += 1
                    if pd.notna(de) and de is not None and de >= 1.5:
                        flags += 1
                    # Negative earnings surprise (FMP earnings endpoint)
                    try:
                        from app.data.fmp_provider import get_earnings_features_at
                        eff = get_earnings_features_at(sym, pd.Timestamp(as_of).date())
                        if eff and eff.get("fmp_surprise_1q", 0.0) <= -0.05 \
                                and eff.get("fmp_days_since_earnings", 365.0) <= 90:
                            flags += 1
                    except Exception:
                        pass
                    if flags >= self.flags_required:
                        candidates.append((sym, float(scores[sym]), flags))

        # Rank candidates by worst composite (most negative score) — most "broken" first
        candidates.sort(key=lambda x: x[1])
        for sym, raw, _flags in candidates[: self.max_shorts]:
            conf = _scale_conf(scores, raw)
            result.append((sym, -conf, "short"))

        return result


# ── B. Mean-Reversion Short Scorer ──────────────────────────────────────────

class MeanReversionShortScorer:
    """Short overextended stocks (recently parabolic); long top factor names.

    Short qualifies if >=2 of:
        - 1-month return in top 20% of universe
        - price > 52-week high × 1.00 within last 5d (i.e. fresh new high)
        - 20-day return in top decile
    """

    def __init__(self, top_n: int = 20, max_shorts: int = 15,
                 vix_threshold: float = DEFAULT_VIX_REGIME,
                 quantile_1m: float = 0.80,
                 quantile_20d: float = 0.90,
                 legs_mode: str = "both"):
        self.top_n = top_n
        self.max_shorts = max_shorts
        self.vix_threshold = vix_threshold
        self.quantile_1m = quantile_1m
        self.quantile_20d = quantile_20d
        self.legs_mode = legs_mode

    def __call__(self, day, symbols_data, vix_history=None):
        closes, bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []

        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        long_allowed = _spy_bull_ok(closes, as_of) and (vix is None or vix < self.vix_threshold)

        scores = compute_composite_score(as_of, closes, bars, use_tier2=True)
        if scores.empty:
            return []

        # Compute overextension metrics
        idx = len(closes) - 1
        if idx < 252:
            return []

        last = closes.iloc[idx]
        prev_21 = closes.iloc[max(0, idx - 21)]
        prev_20 = closes.iloc[max(0, idx - 20)]
        ret_1m = (last / prev_21.replace(0, np.nan) - 1.0).dropna()
        ret_20d = (last / prev_20.replace(0, np.nan) - 1.0).dropna()
        win252 = closes.iloc[max(0, idx - 252): idx + 1]
        high_252 = win252.max()
        # fresh-high signal: current close >= max of last 252 days (i.e. at new high)
        at_new_high = (last >= high_252 * 0.999).reindex(closes.columns).fillna(False)

        thresh_1m = ret_1m.quantile(self.quantile_1m) if len(ret_1m) > 10 else float("inf")
        thresh_20d = ret_20d.quantile(self.quantile_20d) if len(ret_20d) > 10 else float("inf")

        result = []
        long_set = set()
        if long_allowed and self.legs_mode in ("both", "longs_only"):
            longs = select_top_n(scores, n=self.top_n)
            long_set = set(longs)
            result.extend((s, _scale_conf(scores, float(scores[s])), "long") for s in longs)

        if self.legs_mode == "longs_only":
            return result

        candidates = []
        for sym in scores.index:
            if sym in long_set or sym == "SPY":
                continue
            flags = 0
            if sym in ret_1m.index and ret_1m[sym] >= thresh_1m:
                flags += 1
            if sym in ret_20d.index and ret_20d[sym] >= thresh_20d:
                flags += 1
            if sym in at_new_high.index and bool(at_new_high[sym]):
                flags += 1
            if flags >= 2:
                # Rank: most extended = highest 1m return
                extension = float(ret_1m.get(sym, 0.0))
                candidates.append((sym, extension))

        # Rank descending by extension (most overextended first)
        candidates.sort(key=lambda x: -x[1])
        for sym, _ext in candidates[: self.max_shorts]:
            conf = _scale_conf(scores, float(scores[sym]))
            result.append((sym, -conf, "short"))

        return result


# ── C. Sector-Relative L/S Scorer ───────────────────────────────────────────

class SectorRelativeScorer:
    """Within each GICS sector: long top-3, short bottom-3 by composite score.

    Market-neutral by construction — shorts and longs hedge sector exposure.
    Bear-market resilience comes from "worst-within-sector" rather than
    "worst-in-universe" (which biases to beaten-down high-beta names).
    """

    def __init__(self, longs_per_sector: int = 3, shorts_per_sector: int = 3,
                 min_sector_size: int = 8,
                 vix_threshold: float = DEFAULT_VIX_REGIME):
        self.longs_per_sector = longs_per_sector
        self.shorts_per_sector = shorts_per_sector
        self.min_sector_size = min_sector_size
        self.vix_threshold = vix_threshold
        self._sector_map = None

    def _sectors(self):
        if self._sector_map is None:
            from app.utils.constants import SECTOR_MAP
            self._sector_map = SECTOR_MAP
        return self._sector_map

    def __call__(self, day, symbols_data, vix_history=None):
        closes, bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []

        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        long_allowed = _spy_bull_ok(closes, as_of) and (vix is None or vix < self.vix_threshold)

        scores = compute_composite_score(as_of, closes, bars, use_tier2=True)
        if scores.empty:
            return []

        sector_map = self._sectors()
        by_sector: dict[str, list[tuple[str, float]]] = {}
        for sym in scores.index:
            if sym == "SPY":
                continue
            sec = sector_map.get(sym)
            if not sec or sec == "Unknown":
                continue
            by_sector.setdefault(sec, []).append((sym, float(scores[sym])))

        result = []
        for sec, syms in by_sector.items():
            if len(syms) < self.min_sector_size:
                continue
            syms.sort(key=lambda x: -x[1])  # high to low
            if long_allowed:
                for sym, raw in syms[: self.longs_per_sector]:
                    result.append((sym, _scale_conf(scores, raw), "long"))
            # Shorts allowed in all regimes (except extreme VIX gate above)
            for sym, raw in syms[-self.shorts_per_sector:]:
                result.append((sym, -_scale_conf(scores, raw), "short"))

        return result


# ── D. Combined L/S Scorer ──────────────────────────────────────────────────

class CombinedLSScorer:
    """PEAD signals (priority) + factor longs + quality shorts."""

    def __init__(self, top_n_long: int = 15, max_shorts: int = 12,
                 vix_threshold: float = DEFAULT_VIX_REGIME):
        self.top_n_long = top_n_long
        self.max_shorts = max_shorts
        self.vix_threshold = vix_threshold
        self._pead = None
        self._quality = None

    def _pead_scorer(self):
        if self._pead is None:
            from app.ml.pead_scorer import PEADScorer
            self._pead = PEADScorer(long_short=True)
        return self._pead

    def _quality_scorer(self):
        if self._quality is None:
            self._quality = QualityShortScorer(
                top_n=self.top_n_long,
                max_shorts=self.max_shorts,
                vix_threshold=self.vix_threshold,
            )
        return self._quality

    def __call__(self, day, symbols_data, vix_history=None):
        closes, _bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []
        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        # PEAD first (priority)
        pead_picks = self._pead_scorer()(day, symbols_data, vix_history)
        pead_syms = {s for s, _, _ in pead_picks}

        # Factor + quality short
        q_picks = self._quality_scorer()(day, symbols_data, vix_history)

        # Merge: PEAD wins on collisions
        result = list(pead_picks)
        for tup in q_picks:
            if tup[0] not in pead_syms:
                result.append(tup)

        return result


# ── E. A+B Combined Short Leg (Quality OR MeanReversion) ────────────────────

class ABCombinedScorer:
    """Factor longs (top-N) + union of QualityShort and MeanReversionShort.

    A stock qualifies as a short if it satisfies the criteria of EITHER
    QualityShortScorer or MeanReversionShortScorer. Widens the short
    candidate pool while still requiring at least one disciplined signal.
    """

    def __init__(self, top_n: int = 20, max_shorts: int = 20,
                 vix_threshold: float = DEFAULT_VIX_REGIME):
        self.top_n = top_n
        self.max_shorts = max_shorts
        self.vix_threshold = vix_threshold
        self._quality = QualityShortScorer(
            top_n=top_n, max_shorts=max_shorts * 2,
            vix_threshold=vix_threshold, legs_mode="shorts_only",
        )
        self._meanrev = MeanReversionShortScorer(
            top_n=top_n, max_shorts=max_shorts * 2,
            vix_threshold=vix_threshold, legs_mode="shorts_only",
        )

    def __call__(self, day, symbols_data, vix_history=None):
        closes, bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []
        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        long_allowed = _spy_bull_ok(closes, as_of) and (vix is None or vix < self.vix_threshold)

        scores = compute_composite_score(as_of, closes, bars, use_tier2=True)
        if scores.empty:
            return []

        result = []
        long_set = set()
        if long_allowed:
            longs = select_top_n(scores, n=self.top_n)
            long_set = set(longs)
            result.extend((s, _scale_conf(scores, float(scores[s])), "long") for s in longs)

        q_shorts = self._quality(day, symbols_data, vix_history)
        m_shorts = self._meanrev(day, symbols_data, vix_history)
        seen = set()
        union = []
        for tup in list(q_shorts) + list(m_shorts):
            sym = tup[0]
            if sym in long_set or sym in seen or sym == "SPY":
                continue
            seen.add(sym)
            union.append(tup)

        union.sort(key=lambda x: float(scores.get(x[0], 0.0)))
        result.extend(union[: self.max_shorts])
        return result


# ── F. Analyst Revision Short Scorer ────────────────────────────────────────

class AnalystRevisionShortScorer:
    """Short stocks with recent net analyst downgrades; long top factor names.

    Uses FMP analyst grades (upgrade/downgrade actions).
    Short qualifies if fmp_analyst_momentum_30d <= downgrade_threshold
    (default: -2, i.e. net 2+ downgrades in the last 30 calendar days).
    """

    def __init__(self, top_n: int = 20, max_shorts: int = 15,
                 vix_threshold: float = DEFAULT_VIX_REGIME,
                 downgrade_threshold: float = -2.0,
                 legs_mode: str = "both"):
        self.top_n = top_n
        self.max_shorts = max_shorts
        self.vix_threshold = vix_threshold
        self.downgrade_threshold = downgrade_threshold
        self.legs_mode = legs_mode

    def __call__(self, day, symbols_data, vix_history=None):
        closes, bars, as_of = _build_closes_and_bars(day, symbols_data)
        if closes is None:
            return []
        vix = _vix_value(vix_history, as_of)
        if vix is not None and vix >= EXTREME_VIX:
            return []

        long_allowed = _spy_bull_ok(closes, as_of) and (vix is None or vix < self.vix_threshold)

        scores = compute_composite_score(as_of, closes, bars, use_tier2=True)
        if scores.empty:
            return []

        result = []
        long_set = set()
        if long_allowed and self.legs_mode in ("both", "longs_only"):
            longs = select_top_n(scores, n=self.top_n)
            long_set = set(longs)
            result.extend((s, _scale_conf(scores, float(scores[s])), "long") for s in longs)
        if self.legs_mode == "longs_only":
            return result

        from app.data.fmp_provider import get_analyst_features_at
        as_of_date = pd.Timestamp(as_of).date()
        candidates = []
        for sym in scores.index:
            if sym in long_set or sym == "SPY":
                continue
            try:
                af = get_analyst_features_at(sym, as_of_date)
                mom = af.get("fmp_analyst_momentum_30d", 0.0)
            except Exception:
                continue
            if mom <= self.downgrade_threshold:
                candidates.append((sym, float(scores[sym]), mom))

        candidates.sort(key=lambda x: (x[2], x[1]))
        for sym, raw, _mom in candidates[: self.max_shorts]:
            conf = _scale_conf(scores, raw)
            result.append((sym, -conf, "short"))
        return result
