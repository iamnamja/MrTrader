"""
Post-Earnings Announcement Drift (PEAD) scorer — Phase G.

Uses FMP earnings surprise data (already fetched, PIT-safe) to identify
stocks with significant recent earnings surprises. Enters long on positive
surprise, short on negative surprise.

Typical PEAD signal:
    Long:  EPS surprise > +5%, report within last 3 calendar days
    Short: EPS surprise < -5%, report within last 3 calendar days
    Hold 5 days (short-term momentum after announcement)

Returns [(sym, conf, direction)] compatible with AgentSimulator.factor_scorer.

Regime gate (Phase G+):
    VIX > VIX_BLOCK_ALL (30):  block all entries (crisis mode)
    VIX > VIX_BLOCK_SHORT (20): block short leg only (squeeze risk in elevated vol)
    VIX > VIX_CONF_REF (15): scale confidence down linearly (smooth, not cliff)

The VIX gate is principled: 20/30 are long-run ~70th/90th VIX percentiles,
not chosen from CPCV test data. Root cause: PEAD short leg inverts in
risk-off regimes (2021 meme era, Aug-2024 spike, Apr-2025 tariff shock).

Public API:
    PEADScorer(long_threshold, short_threshold, max_days_after, max_hold_days)
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# How many calendar days after the earnings report to still enter (3 cal ≈ 1-2 trading days)
MAX_DAYS_AFTER_EARNINGS = 3

# EPS surprise thresholds (fraction: 0.05 = 5%)
LONG_SURPRISE_THRESHOLD = 0.05
SHORT_SURPRISE_THRESHOLD = -0.05

# Confidence range for PEAD signals — high conviction since anomaly is well-studied
CONF_MIN = 0.65
CONF_MAX = 0.90

# VIX regime thresholds (principled, not tuned to CPCV data)
VIX_BLOCK_ALL = 30.0    # crisis — block all new PEAD entries
VIX_BLOCK_SHORT = 20.0  # elevated vol — disable short leg (squeeze risk)
VIX_CONF_REF = 15.0     # below this: full confidence; above: linear damping

# Priced-in filter: if the stock already moved this much on announcement day,
# the surprise is likely fully reflected in price and drift is exhausted.
# Academic backing: PEAD drift is strongest for moderate reactions (2-8%), not large gaps.
MAX_ANNOUNCE_DAY_MOVE = 0.08   # 8% — skip if stock already gapped >8% on report day


class PEADScorer:
    """PEAD signal generator for use as AgentSimulator.factor_scorer.

    Requires FMP earnings data to be cached (fmp_provider.get_earnings_features_at).
    Data is PIT-safe: only reports with report_date <= as_of are considered.

    VIX-based regime gate is applied when VIX data is present in symbols_data
    under key "^VIX" or "VIX". The gate runs fully PIT (only uses VIX close
    up to and including `day`).

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
        vix_block_all: float = VIX_BLOCK_ALL,
        vix_block_short: float = VIX_BLOCK_SHORT,
        vix_conf_ref: float = VIX_CONF_REF,
        max_announce_day_move: float = MAX_ANNOUNCE_DAY_MOVE,
        long_threshold_hv: float = None,  # high-vol threshold override
        vix_adaptive: float = 20.0,       # VIX level above which to use long_threshold_hv
        require_positive_revision: bool = False,  # earnings-quality gate (default OFF)
        min_analyst_momentum: float = 0.0,        # net up-down threshold (strictly >)
        # ── Alpha v2 §1.2 — regime-GENERAL risk control (default OFF) ─────────────
        # When set, REPLACES the hand-tuned VIX>30 block with a regime-general
        # de-risking control that is NOT fit to specific crisis dates. The harness
        # turns the discrete VIX block off (vix_block_all=inf) and lets this govern.
        #   regime_control="vol_target": exposure scalar = min(1, target_vol /
        #       realized_market_vol), computed PIT from SPY closes <= day.
        #   regime_control="trend": exposure scalar = 0.0 when SPY < its 200d SMA
        #       (PIT, closes <= day) else 1.0.
        # NOTE — what the scalar actually does (NOT continuous gross vol-targeting):
        # the committed PEAD book is EQUAL-WEIGHT (no pead_conviction_size), so the
        # scalar acts as
        #   (a) a vol/trend-gated ENTRY BLOCK: scalar < regime_control_floor (default
        #       0.50) -> block ALL new entries that day (the generic analogue of the
        #       discrete VIX>30 block); and
        #   (b) a coarse conviction-size tilt: the scalar multiplies signal
        #       confidence, which only feeds conviction sizing. In the equal-weight
        #       book the per-day gross is fixed and the conviction band is
        #       narrow/saturating (~[0.75-1.25x]), so above the floor the scalar
        #       barely moves realized gross — it does NOT smoothly scale total
        #       exposure down with rising vol.
        # Default None → committed +0.546 config is byte-identical.
        regime_control: str = None,
        regime_control_target_vol: float = 0.16,   # annualized portfolio/market vol target (vol_target)
        regime_control_vol_lookback: int = 20,     # trading-day lookback for realized vol
        regime_control_trend_ma: int = 200,        # SPY SMA window (trend)
        regime_control_floor: float = 0.50,        # scalar below this -> block new entries
    ):
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.max_days_after = max_days_after
        self.long_short = long_short
        self.vix_block_all = vix_block_all
        self.vix_block_short = vix_block_short
        self.vix_conf_ref = vix_conf_ref
        self.max_announce_day_move = max_announce_day_move
        self.long_threshold_hv = long_threshold_hv
        self.vix_adaptive = vix_adaptive
        self.regime_control = regime_control
        self.regime_control_target_vol = regime_control_target_vol
        self.regime_control_vol_lookback = regime_control_vol_lookback
        self.regime_control_trend_ma = regime_control_trend_ma
        self.regime_control_floor = regime_control_floor
        # Earnings-quality split: when ON, a long signal requires not just an EPS
        # beat but also POSITIVE analyst-revision momentum as-of the scoring day
        # (beat + analysts revising up = higher-conviction drift). PIT-safe: the
        # analyst feature is fetched via get_analyst_features_at(sym, as_of) which
        # only uses grade records dated <= as_of. Default OFF → committed +0.546
        # long-only config is unchanged.
        self.require_positive_revision = require_positive_revision
        self.min_analyst_momentum = min_analyst_momentum

    def _vix_today(self, day, symbols_data: dict):
        """Extract PIT VIX close for `day` from symbols_data. Returns None if unavailable."""
        import pandas as pd
        _ts = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        for key in ("^VIX", "VIX"):
            vix_df = symbols_data.get(key)
            if vix_df is not None and not vix_df.empty:
                try:
                    return float(vix_df.loc[:_ts].iloc[-1]["close"])
                except Exception:
                    pass
        return None

    def _spy_closes(self, symbols_data: dict):
        """Return the SPY close Series from symbols_data, or None."""
        for key in ("SPY", "spy"):
            df = symbols_data.get(key)
            if df is not None and not df.empty and "close" in df.columns:
                return df["close"]
        return None

    def _regime_exposure_scalar(self, day, symbols_data: dict):
        """PIT exposure scalar in [0, 1] for the regime-general control (§1.2 c).

        Uses ONLY market data with timestamp <= `day` (no look-ahead). Returns 1.0
        when regime_control is off or inputs are unavailable (fail-open to the
        committed behaviour). A future vol spike CANNOT change today's scalar — the
        SPY series is sliced with .loc[:_ts] before any computation.
        """
        if not self.regime_control:
            return 1.0
        import pandas as pd
        import numpy as np
        _ts = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        spy = self._spy_closes(symbols_data)
        if spy is None:
            return 1.0
        try:
            hist = spy.loc[:_ts]  # PIT: closes up to and including `day`
        except Exception:
            return 1.0
        if self.regime_control == "vol_target":
            lb = max(int(self.regime_control_vol_lookback), 2)
            if len(hist) < lb + 1:
                return 1.0
            rets = hist.pct_change().dropna().to_numpy(dtype=float)[-lb:]
            if len(rets) < 2:
                return 1.0
            realized = float(np.std(rets, ddof=1)) * (252.0 ** 0.5)
            if realized <= 1e-9:
                return 1.0
            return float(min(1.0, self.regime_control_target_vol / realized))
        if self.regime_control == "trend":
            ma = max(int(self.regime_control_trend_ma), 2)
            if len(hist) < ma:
                return 1.0
            sma = float(hist.iloc[-ma:].mean())
            last = float(hist.iloc[-1])
            return 0.0 if last < sma else 1.0
        return 1.0

    def __call__(
        self,
        day,
        symbols_data: dict,
        vix_history=None,
    ) -> list:
        """Score symbols on `day` using PEAD signal. Returns [(sym, conf, direction)]."""
        import pandas as pd
        from app.data.fmp_provider import get_earnings_features_at
        if self.require_positive_revision:
            from app.data.fmp_provider import get_analyst_features_at

        _ts = pd.Timestamp(day) if not isinstance(day, pd.Timestamp) else day
        as_of = _ts.date()  # get_earnings_features_at requires datetime.date

        # ── Regime gate ───────────────────────────────────────────────────────
        vix = self._vix_today(day, symbols_data)
        if vix is not None and vix > self.vix_block_all:
            logger.debug("PEAD: VIX=%.1f > %.0f — blocking all entries (crisis)", vix, self.vix_block_all)
            return []

        # §1.2 (c): regime-GENERAL control (opt-in; default OFF -> scalar 1.0).
        # Intended as a REPLACEMENT for the discrete VIX block: the harness sets
        # vix_block_all=inf so the block above never fires and this generic, non-
        # crisis-fit control governs de-risking instead. Below the floor -> block
        # all new entries (a vol/trend-gated entry block); above the floor the scalar
        # only damps signal confidence (which feeds conviction sizing). In the
        # equal-weight book this confidence damping is a coarse, saturating tilt — it
        # does NOT continuously vol-target gross.
        regime_scalar = self._regime_exposure_scalar(day, symbols_data)
        if self.regime_control and regime_scalar < self.regime_control_floor:
            logger.debug("PEAD: regime_control=%s scalar=%.2f < floor %.2f — blocking entries",
                         self.regime_control, regime_scalar, self.regime_control_floor)
            return []

        short_enabled = self.long_short and (vix is None or vix <= self.vix_block_short)

        # Adaptive threshold: use higher threshold in high-vol regimes.
        # In elevated VIX (meme era, crisis): only strongest beats survive retail noise.
        # In calm VIX (normal bull): 5% beats drift reliably; 10%+ are often priced-in.
        effective_long_threshold = self.long_threshold
        if self.long_threshold_hv is not None and vix is not None and vix > self.vix_adaptive:
            effective_long_threshold = self.long_threshold_hv

        # Confidence damping: linear from 1.0 at vix_conf_ref down to 0.3 at VIX=50
        vix_mult = 1.0
        if vix is not None and vix > self.vix_conf_ref:
            vix_mult = max(0.3, self.vix_conf_ref / vix)
        # §1.2 (c): fold the regime-general exposure scalar into the confidence
        # damping (flows to conviction sizing). At scalar=1.0 (control off or calm
        # regime) this is a no-op, preserving the committed behaviour.
        vix_mult *= regime_scalar

        results = []

        for sym, df in symbols_data.items():
            if sym in ("^VIX", "VIX", "SPY"):
                continue
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
            surprise = feats.get("fmp_surprise_1q")
            days_since = feats.get("fmp_days_since_earnings")

            if surprise is None or days_since is None:
                continue

            # Only act within max_days_after of the report
            # days_since=0 = announcement day (typically after-hours) — exclude to avoid
            # entering before the surprise is in public prices at next open.
            if days_since < 1 or days_since > self.max_days_after:
                continue

            # Priced-in filter: if the announcement-day bar moved > threshold, the
            # drift has already been captured by momentum traders. Skip to avoid
            # chasing exhausted moves (most common in 2021 meme/retail era).
            if self.max_announce_day_move < 1.0:
                try:
                    _ts_announce = _ts - pd.Timedelta(days=int(days_since))
                    _ann_bar = df.loc[:_ts_announce].iloc[-1]
                    _prev_bar = df.loc[:_ts_announce].iloc[-2] if len(df.loc[:_ts_announce]) >= 2 else None
                    if _prev_bar is not None:
                        _announce_move = abs(float(_ann_bar["close"]) / float(_prev_bar["close"]) - 1)
                        if _announce_move > self.max_announce_day_move:
                            logger.debug(
                                "PEAD priced-in skip %s: announce-day move=%.1f%% > %.0f%% threshold",
                                sym, _announce_move * 100, self.max_announce_day_move * 100,
                            )
                            continue
                except Exception:
                    pass  # if bars unavailable, proceed without filter

            # Map surprise magnitude to confidence
            abs_surprise = abs(surprise)
            # Scale: 5% surprise → 0.65 conf, 20%+ surprise → 0.90 conf
            conf = min(CONF_MIN + (abs_surprise - abs(effective_long_threshold)) * 2.0, CONF_MAX)
            conf = max(conf, CONF_MIN)
            conf_gated = conf * vix_mult

            if surprise >= effective_long_threshold:
                # Earnings-quality gate (optional): only take the long if analysts
                # are revising UP as-of the scoring day. PIT-safe — get_analyst_
                # features_at(sym, as_of) windows grade records to <= as_of only.
                if self.require_positive_revision:
                    try:
                        afeats = get_analyst_features_at(sym, as_of)
                        momentum = afeats.get("fmp_analyst_momentum_30d", 0.0)
                    except Exception as e:
                        logger.debug("PEAD: %s analyst fetch failed: %s", sym, e)
                        momentum = 0.0
                    if momentum <= self.min_analyst_momentum:
                        logger.debug(
                            "PEAD quality-gate skip %s: analyst momentum=%.1f <= %.1f",
                            sym, momentum, self.min_analyst_momentum,
                        )
                        continue
                results.append((sym, conf_gated, "long"))
            elif short_enabled and surprise <= self.short_threshold:
                results.append((sym, -conf_gated, "short"))

        # Sort by abs confidence descending
        results.sort(key=lambda x: abs(x[1]), reverse=True)
        return results
