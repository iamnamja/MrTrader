"""OPT-0: the four frozen options-program interface contracts exist + are importable.

A design-lock test — future phases implement these Protocols; this guards the seam names
+ method signatures so the layering (data ⟂ engine ⟂ sim ⟂ strategy) stays stable.
"""
from __future__ import annotations

import inspect

from app.options import contracts as C


def test_four_contracts_defined():
    for name in ("OptionsDataProvider", "OptionsPricingEngine", "OptionsSpreadCostModel",
                 "OptionContractSim", "OptionsStrategy"):
        assert hasattr(C, name), f"missing contract: {name}"


def test_pricing_engine_methods():
    for m in ("price", "implied_vol", "greeks"):
        assert m in C.OptionsPricingEngine.__dict__ or hasattr(C.OptionsPricingEngine, m)


def test_data_provider_methods():
    for m in ("get_universe", "get_contract_bars", "get_current_snapshot"):
        assert hasattr(C.OptionsDataProvider, m)


def test_strategy_adapter_matches_event_edge_duck_type():
    # The adapter must expose the names run_cpcv / FoldEngine drive.
    for m in ("fetch_data", "run_fold"):
        assert hasattr(C.OptionsStrategy, m)
    # implied_vol returns Optional[float] -> signature carries `price` + `style`
    sig = inspect.signature(C.OptionsPricingEngine.implied_vol)
    assert "price" in sig.parameters and "style" in sig.parameters


def test_kind_and_style_literals():
    assert set(getattr(C.OptionKind, "__args__", ())) == {"call", "put"}
    assert set(getattr(C.ExerciseStyle, "__args__", ())) == {"european", "american"}
