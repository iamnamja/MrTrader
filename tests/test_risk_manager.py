"""
Unit tests for Phase 2: Risk Manager validation rules.

All tests use pure-Python calls to risk_rules.py — no database, Redis, or
Alpaca connections required (those are tested via integration tests later).
"""

import pytest

from app.agents.risk_rules import (
    RiskLimits,
    calculate_dynamic_stop_loss,
    get_sector_exposure,
    validate_account_drawdown,
    validate_buying_power,
    validate_daily_loss,
    validate_open_positions,
    validate_position_size,
    validate_sector_concentration,
)


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def limits():
    return RiskLimits()


@pytest.fixture
def account_20k():
    """A $20,000 account with $15,000 buying power."""
    return {"account_value": 20_000.0, "buying_power": 15_000.0, "equity": 20_000.0}


# ─── Rule 1: Buying Power ─────────────────────────────────────────────────────

class TestBuyingPower:
    def test_sufficient(self, limits):
        ok, msg = validate_buying_power(1_000, 5_000, limits)
        assert ok is True
        assert "OK" in msg

    def test_exact_match(self, limits):
        ok, msg = validate_buying_power(5_000, 5_000, limits)
        assert ok is True

    def test_insufficient(self, limits):
        ok, msg = validate_buying_power(6_000, 5_000, limits)
        assert ok is False
        assert "Insufficient" in msg

    def test_zero_buying_power(self, limits):
        ok, msg = validate_buying_power(1, 0, limits)
        assert ok is False


# ─── Rule 2: Position Size ────────────────────────────────────────────────────

class TestPositionSize:
    def test_within_5pct(self, limits):
        # $1,000 of a $20,000 account = 5% — exactly at limit
        ok, msg = validate_position_size(1_000, 20_000, limits)
        assert ok is True

    def test_below_5pct(self, limits):
        ok, msg = validate_position_size(500, 20_000, limits)
        assert ok is True

    def test_exceeds_5pct(self, limits):
        # $2,000 of $20,000 = 10%
        ok, msg = validate_position_size(2_000, 20_000, limits)
        assert ok is False
        assert "exceeds" in msg.lower()

    def test_zero_account_value(self, limits):
        ok, msg = validate_position_size(1_000, 0, limits)
        assert ok is False

    def test_negative_account_value(self, limits):
        ok, msg = validate_position_size(1_000, -5_000, limits)
        assert ok is False

    def test_boundary_just_over(self, limits):
        # $1,001 on $20,000 = 5.005% → over limit
        ok, msg = validate_position_size(1_001, 20_000, limits)
        assert ok is False


# ─── Rule 3: Sector Concentration ─────────────────────────────────────────────

class TestSectorConcentration:
    def test_within_20pct(self, limits):
        # Adding $1,000 to an existing $3,000 in Tech on a $20k account = 20% (at limit)
        ok, msg = validate_sector_concentration(1_000, 3_000, 20_000, "Tech", limits)
        assert ok is True

    def test_exceeds_20pct(self, limits):
        # Adding $2,000 to $3,000 in Tech on $20k = 25%
        ok, msg = validate_sector_concentration(2_000, 3_000, 20_000, "Tech", limits)
        assert ok is False
        assert "Tech" in msg
        assert "exceeds" in msg.lower()

    def test_no_existing_exposure(self, limits):
        # New sector, $1,000 on $20k = 5%
        ok, msg = validate_sector_concentration(1_000, 0, 20_000, "Energy", limits)
        assert ok is True

    def test_zero_account_value(self, limits):
        ok, msg = validate_sector_concentration(1_000, 0, 0, "Tech", limits)
        assert ok is False


# ─── Rule 4: Daily Loss ───────────────────────────────────────────────────────

class TestDailyLoss:
    def test_no_loss(self, limits):
        ok, msg = validate_daily_loss(0.0, 20_000, limits)
        assert ok is True

    def test_gain_today(self, limits):
        ok, msg = validate_daily_loss(500, 20_000, limits)  # profit is fine
        assert ok is True

    def test_small_loss_ok(self, limits):
        # -$300 on $20k = 1.5% — under 2% limit
        ok, msg = validate_daily_loss(-300, 20_000, limits)
        assert ok is True

    def test_exactly_at_limit(self, limits):
        # -$400 on $20k = exactly 2% — should be blocked
        ok, msg = validate_daily_loss(-400, 20_000, limits)
        assert ok is False

    def test_exceeds_limit(self, limits):
        # -$500 on $20k = 2.5%
        ok, msg = validate_daily_loss(-500, 20_000, limits)
        assert ok is False
        assert "limit" in msg.lower()

    def test_zero_account(self, limits):
        ok, msg = validate_daily_loss(-100, 0, limits)
        assert ok is False


# ─── Rule 5: Account Drawdown ─────────────────────────────────────────────────

class TestAccountDrawdown:
    def test_no_drawdown(self, limits):
        ok, msg = validate_account_drawdown(20_000, 20_000, limits)
        assert ok is True

    def test_small_drawdown(self, limits):
        # 2.5% drawdown — under 5% limit
        ok, msg = validate_account_drawdown(19_500, 20_000, limits)
        assert ok is True

    def test_exactly_at_limit(self, limits):
        # Exactly 5% drawdown — blocked
        ok, msg = validate_account_drawdown(19_000, 20_000, limits)
        assert ok is False

    def test_exceeds_limit(self, limits):
        # 10% drawdown
        ok, msg = validate_account_drawdown(18_000, 20_000, limits)
        assert ok is False
        assert "drawdown" in msg.lower()

    def test_zero_peak(self, limits):
        ok, msg = validate_account_drawdown(10_000, 0, limits)
        assert ok is False

    def test_equity_above_peak(self, limits):
        # Current equity > peak (shouldn't happen in practice, but must not crash)
        ok, msg = validate_account_drawdown(21_000, 20_000, limits)
        assert ok is True


# ─── Rule 6: Open Position Count ─────────────────────────────────────────────

class TestOpenPositions:
    def test_zero_positions(self, limits):
        ok, msg = validate_open_positions(0, limits)
        assert ok is True

    def test_under_limit(self, limits):
        ok, msg = validate_open_positions(3, limits)
        assert ok is True

    def test_one_below_limit(self, limits):
        ok, msg = validate_open_positions(4, limits)
        assert ok is True

    def test_at_limit(self, limits):
        # 5 open positions — at limit, should block new ones
        ok, msg = validate_open_positions(5, limits)
        assert ok is False
        assert "limit" in msg.lower()

    def test_over_limit(self, limits):
        ok, msg = validate_open_positions(6, limits)
        assert ok is False

    def test_custom_limit(self):
        custom = RiskLimits()
        custom.MAX_OPEN_POSITIONS = 5
        ok, _ = validate_open_positions(5, custom)
        assert ok is False
        ok, _ = validate_open_positions(4, custom)
        assert ok is True


# ─── Rule 7: Dynamic Stop Loss ────────────────────────────────────────────────

class TestDynamicStopLoss:
    def test_no_atr_uses_base_pct(self, limits):
        # Base stop = 2% below entry
        stop = calculate_dynamic_stop_loss(100.0, atr=None, limits=limits)
        assert stop == pytest.approx(98.0, abs=0.01)

    def test_normal_atr_equals_base_pct(self, limits):
        # ATR = 2% of price → normal volatility → still 2% stop
        stop = calculate_dynamic_stop_loss(100.0, atr=2.0, limits=limits)
        assert stop == pytest.approx(98.0, abs=0.01)

    def test_high_atr_widens_stop(self, limits):
        # ATR = 4% of price (2× normal) → stop widens to 4%
        stop = calculate_dynamic_stop_loss(100.0, atr=4.0, limits=limits)
        assert stop == pytest.approx(96.0, abs=0.01)

    def test_low_atr_tightens_stop(self, limits):
        # ATR = 1% of price (0.5× normal) → stop tightens to 1%
        stop = calculate_dynamic_stop_loss(100.0, atr=1.0, limits=limits)
        assert stop == pytest.approx(99.0, abs=0.01)

    def test_stop_below_entry(self, limits):
        stop = calculate_dynamic_stop_loss(250.0, limits=limits)
        assert stop < 250.0

    def test_higher_priced_stock(self, limits):
        stop = calculate_dynamic_stop_loss(500.0, limits=limits)
        assert stop == pytest.approx(490.0, abs=0.01)


# ─── Sector Exposure Helper ───────────────────────────────────────────────────

class TestGetSectorExposure:
    def test_basic_grouping(self):
        positions = [
            {"symbol": "AAPL", "market_value": 1_000},
            {"symbol": "MSFT", "market_value": 2_000},
            {"symbol": "XOM", "market_value": 1_500},
        ]
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "XOM": "Energy"}
        exposure = get_sector_exposure(positions, sector_map)
        assert exposure["Tech"] == 3_000
        assert exposure["Energy"] == 1_500

    def test_unknown_sector_fallback(self):
        positions = [{"symbol": "AAPL", "market_value": 1_000}]
        exposure = get_sector_exposure(positions, sector_map={})
        assert exposure.get("UNKNOWN", 0) == 1_000

    def test_empty_positions(self):
        exposure = get_sector_exposure([], {})
        assert exposure == {}


# ─── RiskLimits defaults ─────────────────────────────────────────────────────

class TestRiskLimitsDefaults:
    def test_defaults(self):
        lim = RiskLimits()
        assert lim.MAX_POSITION_SIZE_PCT == 0.05
        assert lim.MAX_SECTOR_CONCENTRATION_PCT == 0.20
        assert lim.MAX_DAILY_LOSS_PCT == 0.02
        assert lim.MAX_ACCOUNT_DRAWDOWN_PCT == 0.05
        assert lim.MAX_OPEN_POSITIONS == 5
        assert lim.STOP_LOSS_BASE_PCT == 0.02


# ─── Short stop direction ─────────────────────────────────────────────────────

class TestShortStopDirection:
    """Short stops must be ABOVE entry (loss if price rises)."""

    def test_short_stop_is_above_entry(self, limits):
        from app.agents.risk_rules import calculate_dynamic_stop_loss
        stop = calculate_dynamic_stop_loss(100.0, limits=limits, direction="SELL_SHORT")
        assert stop > 100.0, "Short stop must be above entry"

    def test_long_stop_is_below_entry(self, limits):
        from app.agents.risk_rules import calculate_dynamic_stop_loss
        stop = calculate_dynamic_stop_loss(100.0, limits=limits, direction="BUY")
        assert stop < 100.0, "Long stop must be below entry"

    def test_short_stop_magnitude_matches_long(self, limits):
        """Short stop distance should equal long stop distance (same stop_pct)."""
        from app.agents.risk_rules import calculate_dynamic_stop_loss
        long_stop = calculate_dynamic_stop_loss(100.0, limits=limits, direction="BUY")
        short_stop = calculate_dynamic_stop_loss(100.0, limits=limits, direction="SELL_SHORT")
        long_dist = 100.0 - long_stop
        short_dist = short_stop - 100.0
        assert abs(long_dist - short_dist) < 0.01, "Stop distance should be symmetric"

    def test_short_stop_with_atr_still_above_entry(self, limits):
        from app.agents.risk_rules import calculate_dynamic_stop_loss
        stop = calculate_dynamic_stop_loss(50.0, atr=1.5, limits=limits, direction="SELL_SHORT")
        assert stop > 50.0


# ─── Sector concentration: direction-aware ────────────────────────────────────

class TestSectorConcentrationShort:
    """Sector concentration should apply to both long and short exposures."""

    def test_long_position_adds_to_sector(self, limits):
        from app.agents.risk_rules import validate_sector_concentration
        ok, _ = validate_sector_concentration(
            proposed_cost=4_000,
            current_sector_value=0,
            account_value=20_000,
            sector="Tech",
            limits=limits,
        )
        # 4k / 20k = 20% — exactly at limit → should pass (<=)
        assert ok

    def test_existing_exposure_plus_new_exceeds_limit(self, limits):
        from app.agents.risk_rules import validate_sector_concentration
        ok, msg = validate_sector_concentration(
            proposed_cost=2_001,
            current_sector_value=2_000,
            account_value=20_000,
            sector="Energy",
            limits=limits,
        )
        # (2001 + 2000) / 20000 = 20.005% > 20%
        assert not ok
        assert "Energy" in msg


# ─── validate_correlation: fail-open on yfinance errors ───────────────────────

class TestCorrelationFailOpen:
    """validate_correlation_risk must fail open (return True) when data is unavailable."""

    def test_fail_open_on_yfinance_exception(self, limits):
        from unittest.mock import patch
        from app.agents.risk_rules import validate_correlation_risk
        with patch("yfinance.download", side_effect=ConnectionError("network down")):
            ok, msg = validate_correlation_risk(
                symbol="AAPL",
                open_symbols=["MSFT"],
                account_value=20_000,
                position_values={"MSFT": 2_000},
                limits=limits,
            )
        assert ok, "Must fail open when yfinance throws"

    def test_no_open_positions_skips_check(self, limits):
        from app.agents.risk_rules import validate_correlation_risk
        ok, msg = validate_correlation_risk(
            symbol="AAPL",
            open_symbols=[],
            account_value=20_000,
            position_values={},
            limits=limits,
        )
        assert ok
        assert "skipped" in msg.lower()

    def test_fail_open_on_empty_dataframe(self, limits):
        """If yfinance returns empty DataFrame, must not raise and must fail open."""
        import pandas as pd
        from unittest.mock import patch
        from app.agents.risk_rules import validate_correlation_risk
        with patch("yfinance.download", return_value=pd.DataFrame()):
            ok, _ = validate_correlation_risk(
                symbol="AAPL",
                open_symbols=["MSFT"],
                account_value=20_000,
                position_values={"MSFT": 2_000},
                limits=limits,
            )
        assert ok, "Must fail open when yfinance returns empty DataFrame"


# ─── Net exposure gate ────────────────────────────────────────────────────────

class TestNetExposureGate:
    """validate_net_exposure blocks entries that push net outside target ± tolerance."""

    def test_long_entry_within_band_passes(self, limits):
        from app.agents.risk_rules import validate_net_exposure
        # long=40k, short=15k → net=25%; add 5k long → 30% → within [25%, 55%]
        ok, msg = validate_net_exposure(5_000, "BUY", 40_000, 15_000, 100_000, limits)
        assert ok

    def test_short_entry_within_band_passes(self, limits):
        from app.agents.risk_rules import validate_net_exposure
        # long=50k, short=5k → net=45%; add 5k short → 40% → within [25%, 55%]
        ok, _ = validate_net_exposure(5_000, "SELL_SHORT", 50_000, 5_000, 100_000, limits)
        assert ok

    def test_long_entry_pushes_above_hi_fails(self, limits):
        from app.agents.risk_rules import validate_net_exposure
        # long=60k, short=0 → net=60%; add 5k long → 65% → above 55%
        ok, msg = validate_net_exposure(5_000, "BUY", 60_000, 0, 100_000, limits)
        assert not ok
        assert "outside" in msg

    def test_short_entry_pushes_below_lo_fails(self, limits):
        from app.agents.risk_rules import validate_net_exposure
        # long=40k, short=15k → net=25%; add 5k short → 20% → below 25%
        ok, msg = validate_net_exposure(5_000, "SELL_SHORT", 40_000, 15_000, 100_000, limits)
        assert not ok

    def test_zero_account_value_fails_open(self, limits):
        from app.agents.risk_rules import validate_net_exposure
        ok, msg = validate_net_exposure(5_000, "BUY", 0, 0, 0, limits)
        assert ok
        assert "skipped" in msg.lower()


# ─── Short notional cap ───────────────────────────────────────────────────────

class TestShortNotionalCap:
    """validate_short_notional enforces hard cap on total short notional."""

    def test_within_cap_passes(self, limits):
        from app.agents.risk_rules import validate_short_notional
        # 10k + 60k = 70k / 100k = 70% < 75%
        ok, _ = validate_short_notional(10_000, 60_000, 100_000, limits)
        assert ok

    def test_exceeds_cap_fails(self, limits):
        from app.agents.risk_rules import validate_short_notional
        # 10k + 70k = 80k / 100k = 80% > 75%
        ok, msg = validate_short_notional(10_000, 70_000, 100_000, limits)
        assert not ok
        assert "cap" in msg.lower()

    def test_exactly_at_cap_passes(self, limits):
        from app.agents.risk_rules import validate_short_notional
        # 5k + 70k = 75k / 100k = 75% == cap
        ok, _ = validate_short_notional(5_000, 70_000, 100_000, limits)
        assert ok

    def test_zero_account_value_fails_open(self, limits):
        from app.agents.risk_rules import validate_short_notional
        ok, msg = validate_short_notional(5_000, 0, 0, limits)
        assert ok
        assert "skipped" in msg.lower()
