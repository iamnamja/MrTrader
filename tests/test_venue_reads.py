"""R1.2 Phase 2 — venue-aware reads (VenueReader).

Proves: (1) the alpaca path is BYTE-IDENTICAL to the raw client (delegates, no reshaping);
(2) the ibkr path normalizes canonical CanonicalPosition/AccountState → the Alpaca dict shape the
sleeves consume; (3) the factory + resolve_venue fail-safe to alpaca.
"""
import pytest

from app.live_trading import venue_reads as vr
from app.live_trading.broker_adapter import AccountState, CanonicalPosition
from app.live_trading.execution_router import resolve_venue


# ── fakes ───────────────────────────────────────────────────────────────────────────
class _FakeAlpaca:
    def __init__(self):
        self._positions = [{"symbol": "SPY", "qty": 10, "market_value": 7350.0,
                            "current_price": 735.0}]
        self._account = {"equity": 99537.6, "cash": 2041.99, "buying_power": 281155.67,
                         "portfolio_value": 99537.6}

    def get_positions(self):
        return self._positions

    def get_account(self):
        return self._account

    def get_position(self, sym):
        return next((p for p in self._positions if p["symbol"] == sym), None)


class _FakeIBKR:
    def get_positions(self):
        return [CanonicalPosition(
            instrument_id="SPY", venue="ibkr", broker_symbol="SPY", asset_class="equity",
            quantity=10.0, price=735.0, multiplier=1.0, currency="USD",
            market_value=7350.0, notional=7350.0, mapped=True)]

    def get_account(self):
        return AccountState(venue="ibkr", nav=99537.6, cash=2041.99, buying_power=281155.67)


# ── alpaca path is byte-identical (delegates to the raw client) ──────────────────────
def test_alpaca_get_positions_is_the_raw_client_object():
    fake = _FakeAlpaca()
    reader = vr.VenueReader("alpaca", alpaca_client=fake)
    # SAME object identity — no copy, no reshape: byte-identical to pre-Phase-2.
    assert reader.get_positions() is fake._positions


def test_alpaca_get_account_is_the_raw_client_dict():
    fake = _FakeAlpaca()
    reader = vr.VenueReader("alpaca", alpaca_client=fake)
    assert reader.get_account() is fake._account


def test_alpaca_get_position_delegates():
    fake = _FakeAlpaca()
    reader = vr.VenueReader("alpaca", alpaca_client=fake)
    assert reader.get_position("SPY")["qty"] == 10
    assert reader.get_position("ZZZZ") is None


# ── ibkr path normalizes canonical → the Alpaca dict shape ───────────────────────────
def test_ibkr_positions_normalized_to_dict_shape():
    reader = vr.VenueReader("ibkr", ibkr_provider=lambda: _FakeIBKR())
    pos = reader.get_positions()
    assert pos == [{"symbol": "SPY", "qty": 10.0, "market_value": 7350.0,
                    "current_price": 735.0}]
    # the sleeves index exactly these keys
    assert pos[0]["symbol"] == "SPY" and int(pos[0]["qty"]) == 10


def test_ibkr_account_normalized_equity_maps_to_nav():
    reader = vr.VenueReader("ibkr", ibkr_provider=lambda: _FakeIBKR())
    acct = reader.get_account()
    assert acct["equity"] == 99537.6          # nav → equity (what the sleeves read)
    assert acct["portfolio_value"] == 99537.6
    assert acct["cash"] == 2041.99 and acct["buying_power"] == 281155.67


def test_ibkr_get_position_filters_book():
    reader = vr.VenueReader("ibkr", ibkr_provider=lambda: _FakeIBKR())
    assert reader.get_position("spy")["qty"] == 10.0   # case-insensitive
    assert reader.get_position("QQQ") is None


def test_ibkr_provider_only_built_lazily():
    calls = {"n": 0}

    def _provider():
        calls["n"] += 1
        return _FakeIBKR()

    reader = vr.VenueReader("ibkr", ibkr_provider=_provider)
    assert calls["n"] == 0            # not built at construction
    reader.get_positions()
    reader.get_account()
    assert calls["n"] == 1            # built once, cached


# ── factory + fail-safe venue resolution ─────────────────────────────────────────────
class _FakeDB:
    pass


def test_get_venue_reader_defaults_to_alpaca(monkeypatch):
    monkeypatch.setattr(vr, "resolve_venue", lambda db, s: "alpaca")
    fake = _FakeAlpaca()
    reader = vr.get_venue_reader(_FakeDB(), "trend", alpaca_client=fake)
    assert reader.venue == "alpaca"
    assert reader.get_positions() is fake._positions


def test_get_venue_reader_ibkr_uses_injected_provider(monkeypatch):
    monkeypatch.setattr(vr, "resolve_venue", lambda db, s: "ibkr")
    reader = vr.get_venue_reader(_FakeDB(), "trend", ibkr_provider=lambda: _FakeIBKR())
    assert reader.venue == "ibkr"
    assert reader.get_positions()[0]["symbol"] == "SPY"


def test_resolve_venue_failsafe_unknown_to_alpaca(monkeypatch):
    import app.database.agent_config as ac
    monkeypatch.setattr(ac, "get_agent_config", lambda db, k: "bogus_venue")
    assert resolve_venue(_FakeDB(), "trend") == "alpaca"


def test_ibkr_reader_without_provider_fails_closed():
    reader = vr.VenueReader("ibkr", ibkr_provider=None)
    with pytest.raises(ConnectionError):
        reader.get_positions()


def test_default_ibkr_provider_fails_closed_on_bad_connect(monkeypatch):
    """A failed/unhealthy IBKR connect must raise (fail-closed), never return a live-looking adapter
    whose empty reads could be mistaken for a flat book."""
    import app.live_trading.ibkr_adapter as ibkr_mod

    class _Health:
        connected = False

    class _Unhealthy:
        @classmethod
        def from_config(cls, db):
            return cls()

        def connect(self):
            return None

        def health(self):
            return _Health()

    monkeypatch.setattr(ibkr_mod, "IBKRReadOnlyAdapter", _Unhealthy)
    provider = vr._default_ibkr_provider(_FakeDB())
    with pytest.raises(ConnectionError):
        provider()
