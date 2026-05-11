"""
P2 — Swing regime gate: blocks new entries on adverse-regime days.

Separates the macro/regime signal from the stock ranker (P0 pruned macro
features from the XGBoost ranker; this layer restores their gatekeeping role
at the portfolio level).

The gate evaluates raw macro values (not TS-normalized per-symbol) against
configurable thresholds. On a blocked day, no new swing positions are opened.
Existing positions are held normally.

Usage in walk-forward:
    gate = RegimeGate(threshold=0.4)
    blocked = gate.build_blocked_dates(macro_df, start, end)
    # pass blocked into AgentSimulator or check per-day

Usage for live inference (premarket):
    gate = RegimeGate.from_parquet()
    if gate.is_blocked(today):
        skip new entries
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Set

import pandas as pd

logger = logging.getLogger(__name__)

_DEFAULT_THRESHOLD = 0.4   # composite_score < this → block entries
_DEFAULT_PARQUET = Path("data/macro/macro_history.parquet")


@dataclass
class RegimeGateConfig:
    """Configuration for the regime gate."""
    threshold: float = _DEFAULT_THRESHOLD   # block if composite_score < threshold
    min_history_days: int = 200             # minimum macro rows needed (for MA200 warmup)
    fail_open: bool = True                  # if True, gate passes when data unavailable


class RegimeGate:
    """
    Point-in-time composite regime gate.

    Computes a 5-component composite score (SPY MA50/MA200, VIX term structure,
    breadth, credit spread) from macro_history and blocks new entries on days
    where the score falls below threshold.

    Thread-safe after construction (read-only lookups).
    """

    def __init__(self, config: Optional[RegimeGateConfig] = None):
        self.config = config or RegimeGateConfig()
        self._scores: Dict[date, float] = {}
        self._blocked: Set[date] = set()
        self._loaded = False

    # ── Construction ─────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        parquet: Path = _DEFAULT_PARQUET,
        config: Optional[RegimeGateConfig] = None,
    ) -> "RegimeGate":
        """Load macro history from parquet and build regime gate."""
        gate = cls(config=config)
        try:
            macro_df = pd.read_parquet(parquet)
            gate._load_from_df(macro_df)
        except FileNotFoundError:
            logger.warning("RegimeGate: %s not found — gate will fail-open", parquet)
        except Exception as exc:
            logger.warning("RegimeGate: failed to load %s — %s", parquet, exc)
        return gate

    @classmethod
    def from_df(
        cls,
        macro_df: pd.DataFrame,
        config: Optional[RegimeGateConfig] = None,
    ) -> "RegimeGate":
        """Build regime gate from an already-loaded macro DataFrame."""
        gate = cls(config=config)
        gate._load_from_df(macro_df)
        return gate

    def _load_from_df(self, macro_df: pd.DataFrame) -> None:
        from app.ml.regime_score_pit import compute_pit_regime_series
        try:
            scores_df = compute_pit_regime_series(macro_df)
            for ts, row in scores_df.iterrows():
                d = ts.date() if hasattr(ts, "date") else ts
                score = float(row["composite_score"])
                self._scores[d] = score
                if score < self.config.threshold:
                    self._blocked.add(d)
            self._loaded = True
            logger.info(
                "RegimeGate: loaded %d days, %d blocked (threshold=%.2f)",
                len(self._scores), len(self._blocked), self.config.threshold,
            )
        except Exception as exc:
            logger.warning("RegimeGate: score computation failed — %s", exc)

    # ── Query ─────────────────────────────────────────────────────────────────

    def is_blocked(self, day: date) -> bool:
        """Return True if new entries should be blocked on this day."""
        if not self._loaded:
            return not self.config.fail_open  # fail-open → not blocked
        if day not in self._scores:
            return not self.config.fail_open  # no data → fail-open
        return day in self._blocked

    def score(self, day: date) -> Optional[float]:
        """Return composite regime score for day, or None if unavailable."""
        return self._scores.get(day)

    def build_blocked_dates(
        self,
        start: date,
        end: date,
    ) -> Set[date]:
        """Return set of blocked dates in [start, end] range."""
        return {d for d in self._blocked if start <= d <= end}

    @property
    def n_loaded(self) -> int:
        return len(self._scores)

    @property
    def n_blocked(self) -> int:
        return len(self._blocked)


def build_regime_gate(
    macro_df: Optional[pd.DataFrame] = None,
    parquet: Path = _DEFAULT_PARQUET,
    threshold: float = _DEFAULT_THRESHOLD,
    fail_open: bool = True,
) -> RegimeGate:
    """
    Convenience factory: build a RegimeGate from macro DataFrame or parquet.

    Args:
        macro_df:   pre-loaded macro_history DataFrame (preferred — avoids disk IO)
        parquet:    fallback parquet path if macro_df is None
        threshold:  composite_score cutoff (< threshold → block)
        fail_open:  if True, missing data passes the gate (default True)

    Returns:
        Configured RegimeGate ready for is_blocked() queries.
    """
    config = RegimeGateConfig(threshold=threshold, fail_open=fail_open)
    if macro_df is not None and len(macro_df) > 0:
        return RegimeGate.from_df(macro_df, config=config)
    return RegimeGate.from_parquet(parquet=parquet, config=config)
