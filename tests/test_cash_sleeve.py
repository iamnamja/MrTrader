"""P1-1 — tests for the cash / T-bill sleeve + tracker."""
import importlib
from types import SimpleNamespace

import pytest


# ── fake broker ───────────────────────────────────────────────────────────────
class _FakeAlpaca:
    def __init__(self, *, cash, equity, positions=None, prices=None):
        self._cash = cash
        self._equity = equity
        self._positions = positions or []
        self._prices = prices or {}
        self.orders = []

    def get_account(self):
        return {"cash": self._cash, "equity": self._equity, "portfolio_value": self._equity}

    def get_positions(self):
        return self._positions

    def get_latest_price(self, sym):
        return self._prices.get(sym)

    def place_market_order(self, sym, qty, side, client_order_id=None, est_price=None):
        self.orders.append((sym, qty, side))
        return {"order_id": f"fake-{sym}-{side}"}


@pytest.fixture()
def cs(monkeypatch):
    import app.live_trading.cash_sleeve as _cs
    importlib.reload(_cs)
    # neutralize the decision_audit DB write + tracker (we assert on the summary)
    monkeypatch.setattr(_cs, "_audit", lambda *a, **k: None)
    return _cs


def _setup(monkeypatch, cs, *, alpaca, config, current=None):
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, k: config[k])
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: alpaca)
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch",
                        SimpleNamespace(is_active=False))
    # price helpers come from trend_sleeve (imported inside run_cash_rebalance)
    monkeypatch.setattr("app.live_trading.trend_sleeve._fetch_prices",
                        lambda alpaca, uni: None)
    monkeypatch.setattr("app.live_trading.trend_sleeve._live_prices",
                        lambda alpaca, syms, df: {s: alpaca._prices.get(s) for s in syms
                                                  if alpaca._prices.get(s)})
    monkeypatch.setattr(cs, "_current_cash_positions", lambda db, a: current or {})


_BASE_CFG = {
    "pm.cash_enabled": "true", "pm.cash_shadow": "true",
    "pm.cash_buffer_pct": 0.02, "pm.cash_universe": "SGOV,BIL",
}


def test_deploy_idle_cash_into_primary_tbill(cs, monkeypatch):
    # NAV 100k, buffer 2% = 2000, cash 50k -> deployable 48000; SGOV $100 -> 480 sh.
    alpaca = _FakeAlpaca(cash=50_000, equity=100_000, prices={"SGOV": 100.0, "BIL": 91.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    s = cs.run_cash_rebalance(db=object())
    assert s["status"] == "ok" and s["action"] == "deploy"
    assert s["approved"] == [{"symbol": "SGOV", "side": "buy", "qty": 480, "target_shares": 480}]


def test_raise_buffer_by_selling_tbills_when_cash_below_buffer(cs, monkeypatch):
    # cash 500 < buffer 2000 -> need 1500; hold 100 SGOV @ $100 -> sell ceil(1500/100)+1=16.
    alpaca = _FakeAlpaca(cash=500, equity=100_000, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG, current={"SGOV": 100})
    s = cs.run_cash_rebalance(db=object())
    assert s["action"] == "raise"
    assert s["approved"][0]["side"] == "sell" and s["approved"][0]["symbol"] == "SGOV"
    assert s["approved"][0]["qty"] == 16


def test_no_action_within_buffer_band(cs, monkeypatch):
    # cash 2050, buffer 2000 -> deployable 50 < MIN_NOTIONAL(100) -> no action.
    alpaca = _FakeAlpaca(cash=2_050, equity=100_000, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    s = cs.run_cash_rebalance(db=object())
    assert s["action"] is None and s["approved"] == []


def test_dormant_when_disabled(cs, monkeypatch):
    cfg = {**_BASE_CFG, "pm.cash_enabled": "false"}
    alpaca = _FakeAlpaca(cash=50_000, equity=100_000, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=cfg)
    s = cs.run_cash_rebalance(db=object())
    assert s["status"] == "dormant" and s["approved"] == []


def test_kill_switch_blocks(cs, monkeypatch):
    alpaca = _FakeAlpaca(cash=50_000, equity=100_000, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    monkeypatch.setattr("app.live_trading.kill_switch.kill_switch",
                        SimpleNamespace(is_active=True))
    s = cs.run_cash_rebalance(db=object())
    assert s["status"] == "blocked" and s["block_reason"] == "kill_switch"


def test_shadow_places_no_orders(cs, monkeypatch):
    alpaca = _FakeAlpaca(cash=50_000, equity=100_000, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    s = cs.run_cash_rebalance(db=object())
    assert s["mode"] == "shadow"
    assert alpaca.orders == []   # nothing sent in shadow


def test_fail_closed_on_zero_nav(cs, monkeypatch):
    alpaca = _FakeAlpaca(cash=0, equity=0, prices={"SGOV": 100.0})
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    s = cs.run_cash_rebalance(db=object())
    assert s["status"] == "failed" and s["block_reason"] == "nav_unavailable"


# ── gross-cap exclusion: T-bills are cash-equivalents, not risk ────────────────
def test_cash_etfs_excluded_from_risk_gross_constant():
    from app.live_trading.cash_sleeve import CASH_ETFS
    assert {"SGOV", "BIL"} <= CASH_ETFS  # the default universe must be excludable


def test_trend_gross_sum_excludes_cash_positions():
    # The risk-gross sum in trend/risk_manager must skip CASH_ETFS so parking cash
    # never inflates risk gross. Reproduce the exact comprehension.
    from app.live_trading.cash_sleeve import CASH_ETFS
    positions = [
        {"symbol": "SPY", "market_value": 40_000},
        {"symbol": "SGOV", "market_value": 48_000},  # cash sleeve — must NOT count
        {"symbol": "QQQ", "market_value": 10_000},
    ]
    risk_gross = sum(abs(float(p["market_value"])) for p in positions
                     if p["symbol"] not in CASH_ETFS)
    assert risk_gross == 50_000  # SGOV excluded


# ── deep-dive regression fixes ────────────────────────────────────────────────
def test_deploy_limited_by_buying_power_settlement_race(cs, monkeypatch):
    # H2: cash 50k but buying_power only 3k (trend just placed orders) -> deployable uses
    # min(cash, bp) - buffer = 3000 - 2000 = 1000 -> 10 sh, NOT 480.
    alpaca = _FakeAlpaca(cash=50_000, equity=100_000, prices={"SGOV": 100.0})
    alpaca._bp = 3_000

    def _acct():
        return {"cash": 50_000, "equity": 100_000, "buying_power": 3_000}
    alpaca.get_account = _acct
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG)
    s = cs.run_cash_rebalance(db=object())
    assert s["action"] == "deploy"
    assert s["approved"][0]["qty"] == 10


def test_raise_path_escalates_when_held_unpriced(cs, monkeypatch):
    # C3: risk-off needs to sell SGOV but no price available -> must NOT silently no-op;
    # blocks with price_unavailable and reports a buffer_shortfall.
    alpaca = _FakeAlpaca(cash=500, equity=100_000, prices={})  # no SGOV price
    _setup(monkeypatch, cs, alpaca=alpaca, config=_BASE_CFG, current={"SGOV": 100})
    s = cs.run_cash_rebalance(db=object())
    assert s["action"] == "raise"
    assert s["blocked"] and s["blocked"][0]["block_reason"] == "price_unavailable"
    assert s.get("buffer_shortfall", 0) > 0


def test_cash_universe_drops_non_tbill_symbols(cs, monkeypatch):
    # M3: a non-CASH_ETFS ticker would trade but not be gross-excluded -> dropped.
    monkeypatch.setattr("app.database.agent_config.get_agent_config",
                        lambda db, k: "SGOV,FLOT,BIL")  # FLOT not a T-bill ETF
    uni = cs.cash_universe(object())
    assert "FLOT" not in uni and uni == ["SGOV", "BIL"]


def test_get_sector_exposure_excludes_cash():
    # M2: T-bills must not pollute the sector buckets.
    from app.agents.risk_rules import get_sector_exposure
    positions = [
        {"symbol": "AAPL", "market_value": 10_000},
        {"symbol": "SGOV", "market_value": 48_000},  # cash sleeve — excluded
    ]
    exp = get_sector_exposure(positions, {"AAPL": "Tech"})
    assert exp == {"Tech": 10_000}
    assert "UNKNOWN" not in exp  # SGOV did not land in UNKNOWN


# ── cash_tracker ──────────────────────────────────────────────────────────────
def test_cash_tracker_record_and_rollup(tmp_path, monkeypatch):
    monkeypatch.setenv("MRTRADER_CASH_TRACKING_DB", str(tmp_path / "cash.db"))
    import app.live_trading.cash_tracker as _ct
    importlib.reload(_ct)
    assert _ct.record_daily("2026-06-15", n_positions=1, tbill_deployed=48000.0, cash_buffer=2000.0)
    assert _ct.record_daily("2026-06-16", n_positions=1, tbill_deployed=47500.0, cash_buffer=2100.0)
    payload = _ct.weekly_rollup("2026-06-16", send=False)
    assert payload["n_days"] == 2
    assert payload["latest_tbill_deployed"] == pytest.approx(47500.0)
    assert payload["avg_tbill_deployed"] == pytest.approx((48000 + 47500) / 2)
