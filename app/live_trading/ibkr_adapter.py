"""
ibkr_adapter.py — Alpha-v10 P2.2: the READ-ONLY IBKR (futures) broker adapter + verify-on-connect.

Mirrors `broker_adapter.AlpacaReadOnlyAdapter` for the IBKR venue: it normalizes the live broker's
truth (account, positions) into canonical objects for the consolidated cross-venue `BookState`, and
it VERIFIES the contract master (multiplier/exchange) against the live `reqContractDetails` — the
panel's #1-futures-killer mitigation (a hand-entered multiplier error can size an order 50x wrong).

Read-only by THREE layers (no capital can be placed through this):
  1. structural — this class exposes NO order method (compile-time guard at the bottom);
  2. broker-side — TWS/Gateway "Read-Only API" setting refuses any order regardless of caller;
  3. session — we connect with ib_insync `readonly=True` (no order/openOrder subscription).
Order placement + the execution path are R1, behind the whole-book risk gate + the live-paper soak.

Fail-closed: connection/health errors never raise out of `health()`; a miss in the instrument master
leaves the position `mapped=False` (the book-state/gate treat unmapped as fail-closed). `ib_insync` is
imported lazily so this module (and its unit tests, which inject a fake `ib`) load without a gateway.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from app.live_trading import instrument_master as im
from app.live_trading.broker_adapter import AccountState, BrokerHealth, CanonicalPosition

log = logging.getLogger(__name__)


def _ensure_event_loop() -> None:
    """ib_insync (via eventkit) calls asyncio.get_event_loop() at import/use; on Python 3.12 that
    RAISES `RuntimeError: no current event loop` in any thread whose loop is missing/closed (e.g. a
    scheduler worker, or a test worker a prior test left without a loop). Ensure one exists first so
    the adapter is safe to use off the main async loop. Idempotent + harmless when a loop is present."""
    import asyncio
    try:
        asyncio.get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@dataclass(frozen=True)
class ContractMismatch:
    """A verify-on-connect discrepancy between our static spec and the live contract details."""
    instrument_id: str
    field: str                  # 'multiplier' | 'exchange' | 'currency' | 'resolve'
    expected: object
    actual: object
    critical: bool              # multiplier/resolve = critical (block); exchange/currency = warn


class IBKRReadOnlyAdapter:
    """Read side of the IBKR venue (R0.3/P2.2 contract). No order methods exist (structurally
    read-only); connects `readonly=True` to a TWS/Gateway that itself has Read-Only API on."""
    venue = im.IBKR

    def __init__(self, ib=None, *, host: str = "127.0.0.1", port: int = 7497,
                 client_id: int = 1, connect_timeout: float = 20.0):
        # `ib` may be injected (a real ib_insync.IB or a duck-typed fake for tests); else created on
        # connect via a lazy import so this module imports without ib_insync present.
        self._ib = ib
        self._host = host
        self._port = int(port)
        self._client_id = int(client_id)
        self._timeout = float(connect_timeout)

    @classmethod
    def from_config(cls, db, *, ib=None) -> "IBKRReadOnlyAdapter":
        from app.database.agent_config import get_agent_config
        return cls(
            ib=ib,
            host=str(get_agent_config(db, "ibkr.host")),
            port=int(get_agent_config(db, "ibkr.port")),
            client_id=int(get_agent_config(db, "ibkr.client_id")),
        )

    # ── connection ──────────────────────────────────────────────────────────────
    def _ensure_ib(self):
        if self._ib is None:
            _ensure_event_loop()       # ib_insync/eventkit need a current loop at import/use
            from ib_insync import IB   # lazy — only when actually connecting
            self._ib = IB()
        return self._ib

    def connect(self) -> BrokerHealth:
        """Connect read-only (idempotent). Returns health; never raises on a connect failure."""
        try:
            ib = self._ensure_ib()
            if not ib.isConnected():
                ib.connect(self._host, self._port, clientId=self._client_id,
                           readonly=True, timeout=self._timeout)
                # any price pulls use FREE delayed data — never require a real-time subscription
                try:
                    ib.reqMarketDataType(3)
                except Exception:  # noqa: BLE001 — non-fatal
                    log.debug("ibkr: reqMarketDataType(delayed) failed", exc_info=True)
            return self.health()
        except Exception as e:  # noqa: BLE001 — fail-closed: surface as unhealthy, do not raise
            log.warning("ibkr: connect failed: %s", e)
            return BrokerHealth(self.venue, connected=False, clock_ok=False, detail=str(e))

    def disconnect(self) -> None:
        try:
            if self._ib is not None and self._ib.isConnected():
                self._ib.disconnect()
        except Exception:  # noqa: BLE001
            log.debug("ibkr: disconnect error", exc_info=True)

    def health(self) -> BrokerHealth:
        try:
            conn = bool(self._ib is not None and self._ib.isConnected())
            return BrokerHealth(self.venue, connected=conn, clock_ok=conn,
                                detail="ok" if conn else "not connected")
        except Exception as e:  # noqa: BLE001 — health must never raise
            return BrokerHealth(self.venue, connected=False, clock_ok=False, detail=str(e))

    def normalize_instrument(self, broker_symbol: str) -> Optional[str]:
        return im.lookup(self.venue, broker_symbol)

    def _require_connected(self) -> None:
        """Fail-CLOSED for the reads: a not-connected session would return empty positions / zero NAV
        which, when fed to the cross-venue book-state, silently UNDERSTATES gross/NAV. Raise instead so
        the caller (reconciliation/report) treats it as a hard break, never a silent-wrong-state."""
        if not (self._ib is not None and self._ib.isConnected()):
            raise ConnectionError("IBKR read-only adapter is not connected (fail-closed read)")

    # ── reads (canonical) ─────────────────────────────────────────────────────────
    def get_account(self) -> AccountState:
        """Map IBKR accountValues (USD, scoped to the single managed account) into the canonical
        AccountState. Defensive float parsing; fail-closed if not connected."""
        self._require_connected()
        accts = list(self._ib.managedAccounts() or [])
        target = accts[0] if len(accts) == 1 else None   # scope to the one account; else USD-only
        usd = {v.tag: v.value for v in self._ib.accountValues()
               if getattr(v, "currency", "") == "USD"
               and (target is None or getattr(v, "account", "") == target)}

        def f(tag: str) -> Optional[float]:
            try:
                return float(usd[tag])
            except (KeyError, TypeError, ValueError):
                return None

        maint = f("MaintMarginReq")
        return AccountState(
            venue=self.venue,
            nav=f("NetLiquidation") or 0.0,
            cash=f("TotalCashValue") or 0.0,
            buying_power=f("BuyingPower") or 0.0,
            settled_cash=f("SettledCash"),
            margin_used=maint,
            margin_available=f("AvailableFunds"),
            maintenance_margin=maint,
        )

    def get_positions(self) -> List[CanonicalPosition]:
        """IBKR portfolio() → canonical positions. Uses the BROKER's contract multiplier (reality);
        notional is recomputed as |qty|*price*mult (NOT the broker marketValue, which for futures is
        ~daily P&L and would understate gross — the trap book_state.py guards against). Fail-closed if
        not connected."""
        self._require_connected()
        out: List[CanonicalPosition] = []
        for it in self._ib.portfolio():
            c = it.contract
            bsym = c.symbol
            iid = self.normalize_instrument(bsym)
            inst = im.get(iid) if iid else None
            try:
                mult = float(c.multiplier) if c.multiplier else (inst.multiplier if inst else 1.0)
            except (TypeError, ValueError):
                mult = inst.multiplier if inst else 1.0
            asset_class = (inst.asset_class if inst
                           else (im.FUTURE if getattr(c, "secType", "") == "FUT" else im.EQUITY))
            qty = float(it.position)
            price = float(getattr(it, "marketPrice", 0.0) or 0.0)
            mv = float(getattr(it, "marketValue", 0.0) or 0.0)
            out.append(CanonicalPosition(
                instrument_id=iid or bsym, venue=self.venue, broker_symbol=bsym,
                asset_class=asset_class, quantity=qty, price=price, multiplier=mult,
                currency=getattr(c, "currency", "USD") or "USD",
                market_value=mv, notional=abs(qty) * price * mult, mapped=iid is not None))
        return out

    # ── verify-on-connect (the #1-futures-killer mitigation) ───────────────────────
    def verify_contracts(self) -> List[ContractMismatch]:
        """Resolve every futures instrument via the live reqContractDetails and compare to our static
        spec. A multiplier mismatch (or a contract that won't resolve) is CRITICAL — the caller must
        block trading that instrument. Exchange/currency differences are WARN (IBKR aliases).
        Fail-closed if not connected."""
        _ensure_event_loop()           # ib_insync/eventkit need a current loop (3.12: else RuntimeError)
        from ib_insync import Future
        self._require_connected()
        mismatches: List[ContractMismatch] = []
        for iid, inst in im.futures_instruments().items():
            req_sym = inst.broker_symbol(self.venue)   # IBKR request symbol (e.g. EUR, not 6E)
            try:
                cds = self._ib.reqContractDetails(
                    Future(req_sym, exchange=inst.exchange or "", currency=inst.currency))
            except Exception as e:  # noqa: BLE001
                mismatches.append(ContractMismatch(iid, "resolve", "ok", f"error: {e}", True))
                continue
            if not cds:
                mismatches.append(ContractMismatch(iid, "resolve", "resolvable",
                                                   "no contract details", True))
                continue
            # A symbol can return MULTIPLE products (e.g. SI -> micro SIL 1000oz + full SI 5000oz):
            # pick the contract whose tradingClass matches our canonical root, NOT cds[0] (arbitrary).
            matched = [cd.contract for cd in cds
                       if getattr(cd.contract, "tradingClass", None) == inst.root]
            if not matched:
                mismatches.append(ContractMismatch(iid, "resolve",
                                                   f"tradingClass={inst.root}",
                                                   "no contract with that tradingClass", True))
                continue
            # Check the multiplier across ALL matched expiries (not just matched[0]) — a contract
            # re-spec across months would otherwise slip past an arbitrary first-pick (false negative
            # on the #1-killer guard). Any None or any divergent multiplier is CRITICAL.
            mults = []
            for cm in matched:
                try:
                    mults.append(float(cm.multiplier) if cm.multiplier else None)
                except (TypeError, ValueError):
                    mults.append(None)
            if any(m is None for m in mults):
                mismatches.append(ContractMismatch(iid, "multiplier", inst.multiplier, None, True))
            else:
                bad = next((m for m in mults if abs(m - inst.multiplier) > 1e-6), None)
                if bad is not None:
                    mismatches.append(ContractMismatch(iid, "multiplier", inst.multiplier, bad, True))
            c = matched[0]
            if inst.exchange and c.exchange and c.exchange != inst.exchange:
                mismatches.append(ContractMismatch(iid, "exchange", inst.exchange, c.exchange, False))
            if inst.currency and c.currency and c.currency != inst.currency:
                mismatches.append(ContractMismatch(iid, "currency", inst.currency, c.currency, False))
        return mismatches


# Compile-time-ish guard: the read-only adapter must NOT expose any order method (P2.2 read-only).
for _forbidden in ("place_order", "submit_order", "placeOrder", "cancel_order", "cancelOrder",
                   "flatten_all", "liquidate_all"):
    assert not hasattr(IBKRReadOnlyAdapter, _forbidden), \
        f"IBKRReadOnlyAdapter must be read-only in P2.2 (found {_forbidden})"
