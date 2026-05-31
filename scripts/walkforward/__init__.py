"""
scripts/walkforward — Pluggable walk-forward validation engine (WF-2).

Public API (all backwards-compatible with walkforward_tier3.py):
    run_swing_walkforward(...)    — delegates to scripts.walkforward_tier3
    run_intraday_walkforward(...) — delegates to scripts.walkforward_tier3
    FoldEngine                    — strategy-agnostic engine for custom strategies
    FoldResult, WalkForwardReport — re-exported from gates.py

Extension pattern for Day Trading (WF-7):
    from scripts.walkforward import FoldEngine
    from scripts.walkforward.cost_models import SpreadCostModel

    engine = FoldEngine(
        strategy=DayTradingStrategy(...),
        purge_days=1,
        embargo_days=1,
    )
    report = engine.run(n_folds=6, total_days=365)
"""
from __future__ import annotations

# Re-export the canonical implementations from gates.py (WF-2 canonical source)
from scripts.walkforward.gates import FoldResult, WalkForwardReport
from scripts.walkforward.engine import FoldEngine
from scripts.walkforward.cost_models import FixedBpsCostModel, SpreadCostModel

__all__ = [
    "FoldEngine",
    "FoldResult",
    "WalkForwardReport",
    "FixedBpsCostModel",
    "SpreadCostModel",
    "run_swing_walkforward",
    "run_intraday_walkforward",
]


def run_swing_walkforward(*args, **kwargs):
    """Delegate to scripts.walkforward_tier3 (lazy import avoids circular dependency)."""
    from scripts.walkforward_tier3 import run_swing_walkforward as _fn
    return _fn(*args, **kwargs)


def run_intraday_walkforward(*args, **kwargs):
    """Delegate to scripts.walkforward_tier3 (lazy import avoids circular dependency)."""
    from scripts.walkforward_tier3 import run_intraday_walkforward as _fn
    return _fn(*args, **kwargs)
