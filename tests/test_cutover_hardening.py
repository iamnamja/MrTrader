"""R1.2 cutover-hardening batch (from the pre-cutover readiness deep-dive).

Covers the gateway-independent fixes:
  G2  — startup_reconciler exempts a trend/cash sleeve on a non-Alpaca venue from the ghost pass.
  G3  — IBKRReadOnlyAdapter.get_positions fails CLOSED on a connected-but-unsynced session.
  G6  — IBKRReadOnlyAdapter.get_account fails CLOSED on a missing NetLiquidation (unsynced).
  G5  — the enforce whole-book gate HOLDS on an internal eval error (not just a breach).
(G1 — per-venue reconciliation scoping — is pinned in test_reconciliation_venue.py.)
"""
from types import SimpleNamespace as NS

import pytest


# ── G2: ghost-pass venue exemption ────────────────────────────────────────────────────
def _trade(selector=None, trade_type=None):
    return NS(selector=selector, trade_type=trade_type)


def test_ghost_exempts_sleeve_cut_over_to_ibkr(monkeypatch):
    import app.live_trading.execution_router as er
    from app.startup_reconciler import _sleeve_cutover_to_non_alpaca as cut
    monkeypatch.setattr(er, "resolve_venue", lambda db, s: "ibkr" if s == "cash" else "alpaca")
    assert cut(None, _trade(selector="cash")) is True          # cash on IBKR → exempt from ghost
    assert cut(None, _trade(trade_type="cash")) is True         # via trade_type attribute too
    assert cut(None, _trade(selector="trend")) is False         # trend still Alpaca → not exempt
    assert cut(None, _trade(selector="swing")) is False         # not a venue-routable sleeve
    assert cut(None, _trade()) is False                         # no selector


def test_ghost_not_exempt_when_all_alpaca(monkeypatch):
    """Pre-cutover (every sleeve Alpaca) nothing is exempted — byte-identical ghost behavior."""
    import app.live_trading.execution_router as er
    from app.startup_reconciler import _sleeve_cutover_to_non_alpaca as cut
    monkeypatch.setattr(er, "resolve_venue", lambda db, s: "alpaca")
    assert cut(None, _trade(selector="cash")) is False
    assert cut(None, _trade(selector="trend")) is False


class _ReconAlpaca:
    """Minimal Alpaca stand-in for startup_reconciler.reconcile: the IBKR-held cash ETF is absent
    from the Alpaca snapshot (it moved venues), and the account read is healthy."""
    def get_positions(self):
        return []                                   # SGOV not on Alpaca anymore (cut over to IBKR)

    def get_account(self):
        return {"equity": 100000.0, "portfolio_value": 100000.0}


def test_ghost_pass_b_rescues_cutover_trade_not_close(monkeypatch, db_session):
    """H1: a cash trade already RECONCILE_GHOST_PENDING when its sleeve flips to IBKR must be RESCUED
    to ACTIVE by Pass B — never ghost-CLOSED (Pass A only iterates ACTIVE, so Pass B is the closer)."""
    import app.live_trading.execution_router as er
    import app.startup_reconciler as sr
    from app.database.models import Trade

    monkeypatch.setattr(er, "resolve_venue", lambda db, s: "ibkr" if s == "cash" else "alpaca")
    t = Trade(symbol="SGOV", direction="BUY", entry_price=100.0, quantity=400,
              status=sr.RECONCILE_GHOST_PENDING, selector="cash", ghost_detection_count=1)
    db_session.add(t)
    db_session.commit()
    tid = t.id

    sr.reconcile(_ReconAlpaca(), db_session)

    refreshed = db_session.query(Trade).filter_by(id=tid).one()
    assert refreshed.status == "ACTIVE"             # rescued, NOT closed
    assert refreshed.ghost_detection_count == 0


# ── G3 / G6: IBKR read fails CLOSED on an unsynced session ─────────────────────────────
class _FakeIB:
    def __init__(self, portfolio=None, acct_values=None, managed=("DU1",), connected=True):
        self._portfolio = portfolio or []
        self._acct_values = acct_values or []
        self._managed = list(managed)
        self._connected = connected

    def isConnected(self):
        return self._connected

    def portfolio(self):
        return self._portfolio

    def accountValues(self):
        return self._acct_values

    def managedAccounts(self):
        return self._managed


def _av(tag, value, account="DU1"):
    return NS(tag=tag, value=str(value), currency="USD", account=account)


def test_get_positions_fails_closed_when_unsynced():
    """Empty portfolio AND empty accountValues on a connected session = not synced → raise, never
    return [] (a false-flat book would re-buy the whole sleeve)."""
    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
    ad = IBKRReadOnlyAdapter(ib=_FakeIB(portfolio=[], acct_values=[]))
    with pytest.raises(ConnectionError):
        ad.get_positions()


def test_get_positions_flat_when_synced_but_empty():
    """accountValues present (synced) + no portfolio rows = genuinely flat → []."""
    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
    ad = IBKRReadOnlyAdapter(ib=_FakeIB(portfolio=[], acct_values=[_av("NetLiquidation", 100000)]))
    assert ad.get_positions() == []


def test_get_account_fails_closed_when_nav_missing():
    """A single managed account but no NetLiquidation row = accountValues not synced → raise
    (never report NAV 0.0, which would zero sizing / divide a gross gate)."""
    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
    ad = IBKRReadOnlyAdapter(ib=_FakeIB(acct_values=[_av("TotalCashValue", 5000)]))
    with pytest.raises(ValueError):
        ad.get_account()


def test_get_account_ok_when_synced():
    from app.live_trading.ibkr_adapter import IBKRReadOnlyAdapter
    ad = IBKRReadOnlyAdapter(ib=_FakeIB(
        acct_values=[_av("NetLiquidation", 100000), _av("TotalCashValue", 5000),
                     _av("BuyingPower", 20000)]))
    acct = ad.get_account()
    assert acct.nav == 100000.0 and acct.cash == 5000.0 and acct.buying_power == 20000.0


# ── G5: enforce whole-book gate HOLDS on an internal eval error ────────────────────────
def test_enforce_whole_book_gate_internal_error_holds(monkeypatch):
    """The gate's fail-safe returns allow=True on an internal exception; in ENFORCE the caller must
    still HOLD (the book was never actually evaluated). Pre-fix this fell OPEN and placed."""
    import tests.test_trend_sleeve as tts
    import app.live_trading.trend_sleeve as ts
    from app.live_trading import whole_book_gate as wbg

    cfg = {
        "pm.trend_enabled": "true", "pm.trend_shadow": "true",
        "pm.trend_allocation_pct": 0.40, "pm.trend_max_position_pct": 0.25,
        "pm.trend_universe": "SPY,QQQ,TLT", "pm.trend_rebalance_weekday": 0,
        "pm.whole_book_gate_mode": "enforce",   # the gate is enforcing
    }
    monkeypatch.setattr("app.database.agent_config.get_agent_config", lambda db, key: cfg.get(key))
    from app.live_trading.kill_switch import kill_switch
    monkeypatch.setattr(kill_switch, "_active", False, raising=False)
    monkeypatch.setattr("app.database.decision_audit.write_decision", lambda **kw: None)
    monkeypatch.setattr(ts, "_current_trend_positions", lambda db, source: {})
    monkeypatch.setattr(ts, "_sync_trend_trade", lambda db, s, t, p, o: None)
    monkeypatch.setattr("app.live_trading.trend_tracker.record_daily", lambda **kw: True)

    fake = tts._FakeAlpaca(tts._uptrend_prices(["SPY", "QQQ", "TLT"]))
    monkeypatch.setattr("app.integrations.get_alpaca_client", lambda: fake)

    # Gate errored internally → its fail-safe verdict (allow=True + error set).
    monkeypatch.setattr(wbg, "shadow_gate_from_intents",
                        lambda *a, **k: wbg.WholeBookGateVerdict(
                            allow=True, mode="enforce", breaches=[], details={}, error="boom"))

    summary = ts.run_trend_rebalance(db=object())

    assert summary["status"] == "blocked"
    assert summary["block_reason"] == "whole_book_gate"
    assert summary.get("gate_error") == "boom"
    assert fake.orders == []                      # nothing placed — held closed


def test_whole_book_gate_lowercase_venue_maps_etfs_no_false_breach():
    """The sleeve now passes the router venue ('alpaca', lower-case) to the gate. im.lookup is
    case-sensitive on the UPPER-case constant, so without normalization every ETF would miss → a
    false `unmapped` breach → a false HOLD in enforce on the CURRENT book. Prove lower-case behaves
    identically to im.ALPACA."""
    from app.live_trading import whole_book_gate as wbg
    from app.live_trading import instrument_master as im

    intents = [{"symbol": "SPY", "side": "buy", "qty": 10}]     # $5k of a $100k book — no breach
    prices = {"SPY": 500.0}
    v_lower = wbg.shadow_gate_from_intents([], intents, prices, 100000.0,
                                           mode="enforce", venue="alpaca")
    v_upper = wbg.shadow_gate_from_intents([], intents, prices, 100000.0,
                                           mode="enforce", venue=im.ALPACA)
    assert not any("unmapped" in b for b in v_lower.breaches)   # ETF mapped, no false breach
    assert v_lower.allow is True and v_lower.error is None
    assert list(v_lower.breaches) == list(v_upper.breaches)     # identical to the im-constant path
