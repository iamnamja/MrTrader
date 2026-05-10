"""
P1 — BenignGate: inference-time regime filter for ML signals.

Blocks swing/intraday ML signals when the current market regime is adverse
(composite_score < threshold). Also handles open-position stop tightening
when regime flips from favorable to adverse.

Design:
- Daily-cached: one parquet read per trading day, ~200ms cold, ~5ms warm
- Fail-closed: stale/missing data → blocks all signals (conservative)
- Persistent: every gate firing logged to regime_gate_events table
- Decoupled from RegimeDetector: uses macro_history.parquet directly
  (PIT-correct, no FRED API calls at inference time)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SCORE_CACHE: Dict[date, Tuple[float, dict]] = {}  # class-level daily cache


def _get_db():
    try:
        from app.database.session import SessionLocal
        return SessionLocal()
    except Exception as exc:
        logger.warning("BenignGate: DB unavailable — %s", exc)
        return None


class BenignGate:
    """
    Regime gate for ML signals. Call gate() before constructing any TradeProposal.

    Usage:
        gate = BenignGate()
        allowed_symbols = gate.gate(candidate_symbols, reason="swing_ml")
        # allowed_symbols is [] when regime is adverse
    """

    def __init__(self, threshold: Optional[float] = None):
        from app.ml.retrain_config import BENIGN_REGIME_THRESHOLD
        self.threshold = threshold if threshold is not None else BENIGN_REGIME_THRESHOLD

    # ------------------------------------------------------------------
    # Score computation (cached daily)
    # ------------------------------------------------------------------

    def current_score(self) -> Tuple[float, dict]:
        """Return (composite_score, components_dict) for today. Fails closed on error."""
        today = date.today()
        if today in _SCORE_CACHE:
            return _SCORE_CACHE[today]

        from app.ml.regime_score_pit import get_current_regime_score
        score, components = get_current_regime_score()
        _SCORE_CACHE[today] = (score, components)

        logger.info(
            "BenignGate: regime_score=%.3f (threshold=%.2f) components=%s",
            score, self.threshold, components,
        )

        # Persist to daily_regime_scores table (upsert)
        self._persist_daily_score(today, score, components)

        return score, components

    def is_favorable(self) -> bool:
        score, _ = self.current_score()
        return score >= self.threshold

    # ------------------------------------------------------------------
    # Gate — call before any ML signal emission
    # ------------------------------------------------------------------

    def gate(self, symbols: List[str], reason: str = "swing_ml") -> List[str]:
        """
        Filter symbols through the regime gate.
        Returns the full list if regime is favorable, [] if adverse.
        Logs every block to regime_gate_events.
        """
        score, components = self.current_score()

        if score >= self.threshold:
            return symbols  # pass — no logging needed

        # Adverse regime — block all signals
        logger.warning(
            "BenignGate FIRED: score=%.3f < threshold=%.2f — blocking %d %s signals: %s",
            score, self.threshold, len(symbols), reason,
            ", ".join(symbols[:5]) + ("..." if len(symbols) > 5 else ""),
        )

        self._log_gate_event(
            regime_score=score,
            n_blocked=len(symbols),
            blocked_symbols=symbols,
            reason=reason,
            components=components,
        )

        return []

    # ------------------------------------------------------------------
    # Open-position stop tightening (Option B from P1 spec)
    # ------------------------------------------------------------------

    def handle_regime_flip(self, prior_score: Optional[float] = None) -> int:
        """
        Called once per trading day at market open.
        If regime just flipped from favorable to adverse, tighten ATR stops
        on all open swing positions by 50%.

        Returns number of positions tightened.
        """
        score, components = self.current_score()
        favorable = score >= self.threshold

        if favorable:
            return 0  # nothing to do

        if prior_score is not None and prior_score < self.threshold:
            return 0  # already in adverse regime — don't double-tighten

        # Regime just flipped adversely — tighten stops
        n_tightened = self._tighten_open_swing_stops(score, components)
        return n_tightened

    def _tighten_open_swing_stops(self, score: float, components: dict) -> int:
        """Tighten ATR stops on all open swing positions to 50% of original."""
        db = _get_db()
        if db is None:
            return 0

        n_tightened = 0
        try:
            from app.database.models import Trade
            open_swings = (
                db.query(Trade)
                .filter(Trade.status == "ACTIVE", Trade.trade_type == "swing")
                .all()
            )

            for trade in open_swings:
                if trade.stop_price is None or trade.entry_price is None:
                    continue
                orig_distance = abs(trade.entry_price - trade.stop_price)
                new_stop = trade.entry_price - (orig_distance * 0.5)
                # Only tighten — never move stop further away
                if trade.stop_price < new_stop:
                    trade.stop_price = round(new_stop, 4)
                    n_tightened += 1

            if n_tightened > 0:
                db.commit()
                logger.warning(
                    "BenignGate: regime flip detected (score=%.3f) — "
                    "tightened stops on %d open swing positions",
                    score, n_tightened,
                )
                self._log_gate_event(
                    regime_score=score,
                    n_blocked=n_tightened,
                    blocked_symbols=[],
                    reason="stop_tighten_regime_flip",
                    components=components,
                )

        except Exception as exc:
            logger.error("BenignGate: stop tightening failed — %s", exc)
            db.rollback()
        finally:
            db.close()

        return n_tightened

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def _persist_daily_score(self, d: date, score: float, components: dict) -> None:
        db = _get_db()
        if db is None:
            return
        try:
            from app.database.models import DailyRegimeScore
            existing = db.query(DailyRegimeScore).filter_by(date=d).first()
            if existing:
                existing.composite_score = score
                existing.computed_at = datetime.utcnow()
            else:
                db.add(DailyRegimeScore(
                    date=d,
                    spy_above_ma50=components.get("spy_above_ma50", 0.0),
                    spy_above_ma200=components.get("spy_above_ma200", 0.0),
                    vix_term_ratio=components.get("vix_term_ratio", 1.0),
                    breadth_20d_change=components.get("breadth_20d_change", 0.0),
                    credit_20d_change=components.get("credit_20d_change", 0.0),
                    composite_score=score,
                ))
            db.commit()
        except Exception as exc:
            logger.debug("BenignGate: could not persist daily score — %s", exc)
            db.rollback()
        finally:
            db.close()

    def _log_gate_event(
        self,
        regime_score: float,
        n_blocked: int,
        blocked_symbols: List[str],
        reason: str,
        components: dict,
    ) -> None:
        db = _get_db()
        if db is None:
            return
        try:
            from app.database.models import RegimeGateEvent
            db.add(RegimeGateEvent(
                date=date.today(),
                regime_score=regime_score,
                threshold=self.threshold,
                n_signals_blocked=n_blocked,
                blocked_symbols=",".join(blocked_symbols[:50]),
                reason=reason,
                components=components,
            ))
            db.commit()
        except Exception as exc:
            logger.debug("BenignGate: could not log gate event — %s", exc)
            db.rollback()
        finally:
            db.close()


# ------------------------------------------------------------------
# LKG (last-known-good) helpers
# ------------------------------------------------------------------

def get_lkg_version(model_name: str) -> Optional[int]:
    """Return the last-known-good model version for model_name, or None."""
    db = _get_db()
    if db is None:
        return None
    try:
        from app.database.models import Configuration
        row = db.query(Configuration).filter_by(key=f"{model_name}.last_known_good_version").first()
        if row and row.value.isdigit():
            return int(row.value)
    except Exception:
        pass
    finally:
        db.close()
    return None


def set_lkg_version(model_name: str, version: int) -> None:
    """Persist the last-known-good version for model_name."""
    db = _get_db()
    if db is None:
        return
    try:
        from app.database.models import Configuration
        key = f"{model_name}.last_known_good_version"
        row = db.query(Configuration).filter_by(key=key).first()
        if row:
            row.value = str(version)
        else:
            db.add(Configuration(
                key=key,
                value=str(version),
                description=f"Last known good {model_name} model version (passed CPCV gate)",
            ))
        db.commit()
        logger.info("LKG: set %s = v%d", model_name, version)
    except Exception as exc:
        logger.error("LKG: failed to persist — %s", exc)
        db.rollback()
    finally:
        db.close()
