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

# Public run functions — delegate to walkforward_tier3 for now.
# As WF-3 (CPCV) and beyond extend the package, these will migrate to native implementations.
from scripts.walkforward_tier3 import (
    run_swing_walkforward,
    run_intraday_walkforward,
)

__all__ = [
    "FoldEngine",
    "FoldResult",
    "WalkForwardReport",
    "FixedBpsCostModel",
    "SpreadCostModel",
    "run_swing_walkforward",
    "run_intraday_walkforward",
]
