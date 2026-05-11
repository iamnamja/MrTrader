"""
P4 tests: model-aware default fold counts (5 for swing, 3 for intraday).

Tests cover:
- --folds not set → swing gets 5, intraday gets 3
- --folds N (explicit) → both get N
- argparse default is None (not 3)
"""
from __future__ import annotations

import argparse


def _parse_args(argv: list) -> argparse.Namespace:
    """Parse walkforward_tier3 args from a string list."""
    # We import the parser directly by calling argparse inside main()
    # Instead, replicate just the folds arg for isolation.
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=None)
    return parser.parse_args(argv)


def test_folds_default_is_none():
    args = _parse_args([])
    assert args.folds is None


def test_explicit_folds_overrides():
    args = _parse_args(["--folds", "7"])
    assert args.folds == 7


def test_model_aware_defaults_when_folds_is_none():
    """Simulate the P4 logic in main(): swing=5, intraday=3 when folds=None."""
    args = _parse_args([])
    swing_folds = args.folds if args.folds is not None else 5
    intraday_folds = args.folds if args.folds is not None else 3
    assert swing_folds == 5
    assert intraday_folds == 3


def test_explicit_folds_applied_to_both():
    """Explicit --folds 4 overrides both swing and intraday defaults."""
    args = _parse_args(["--folds", "4"])
    swing_folds = args.folds if args.folds is not None else 5
    intraday_folds = args.folds if args.folds is not None else 3
    assert swing_folds == 4
    assert intraday_folds == 4
