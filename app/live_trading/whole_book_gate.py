"""
whole_book_gate.py — Alpha-v10 R0.5: the WHOLE-BOOK risk gate (shadow-first).

Evaluates a PROPOSED book against the frozen risk-policy v1 hard caps (gross-ex-cash, net equity
beta, single-instrument notional, book notional) and returns an allow/block verdict. This is the
holistic gate the live sleeves currently lack — it sees the consolidated book, not one trade.

Rollout: `mode="shadow"` (default) LOGS what it would block but blocks nothing — the caller proceeds
exactly as today; `mode="enforce"` returns allow=False on a breach so the caller HOLDS (fail-closed,
a missed rebalance, never a bad trade). Controlled by the `pm.whole_book_gate_mode` config flag.

The shadow path is FAIL-SAFE: `shadow_gate_from_intents` never raises — any internal error returns an
`allow=True, error` verdict and logs, so a gate bug can never disrupt a live rebalance.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

from app.live_trading import book_state as bs, instrument_master as im
from app.live_trading.broker_adapter import AccountState, CanonicalPosition
from app.live_trading.risk_policy import RISK_POLICY_V1, RiskPolicy

log = logging.getLogger(__name__)

SHADOW = "shadow"
ENFORCE = "enforce"
OFF = "off"


@dataclass(frozen=True)
class WholeBookGateVerdict:
    allow: bool
    mode: str
    breaches: List[str] = field(default_factory=list)
    details: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate(book: bs.BookState, policy: RiskPolicy = RISK_POLICY_V1) -> WholeBookGateVerdict:
    """Pure: check a BookState against the risk-policy v1 hard caps. mode is set by the caller."""
    nav = book.total_nav if book.total_nav and book.total_nav > 0 else None
    breaches: List[str] = []
    details: Dict[str, float] = {}

    details["gross_ex_cash_frac"] = book.gross_ex_cash_frac
    if book.gross_ex_cash_frac > policy.max_gross_ex_cash + 1e-9:
        breaches.append(f"gross_ex_cash {book.gross_ex_cash_frac:.1%} > cap "
                        f"{policy.max_gross_ex_cash:.0%}")

    eq = book.factor_exposures.get(bs.EQUITY_BETA, 0.0)
    eq_frac = (eq / nav) if nav else 0.0
    details["net_equity_beta_frac"] = eq_frac
    if abs(eq_frac) > policy.max_net_equity_beta + 1e-9:
        breaches.append(f"net_equity_beta {eq_frac:+.2f}x > cap {policy.max_net_equity_beta:.2f}")

    book_frac = (book.gross_notional / nav) if nav else 0.0
    details["book_notional_frac"] = book_frac
    if nav and book_frac > policy.max_book_notional_frac + 1e-9:
        breaches.append(f"book_notional {book_frac:.2f}x > cap {policy.max_book_notional_frac:.2f}x")

    if nav:
        for p in book.positions:
            if p.is_cash_equivalent:
                continue
            # a real position with a missing/zero price reports notional 0 and would HIDE a breach
            # (false allow). Treat it as fail-closed: a breach, not a silent pass.
            if p.price <= 0 and abs(p.quantity) > 1e-9:
                breaches.append(f"missing_price {p.instrument_id} (fail-closed — cannot size)")
                continue
            frac = abs(p.notional) / nav
            if frac > policy.max_single_instrument_notional_frac + 1e-9:
                breaches.append(f"single_notional {p.instrument_id} {frac:.2f}x > cap "
                                f"{policy.max_single_instrument_notional_frac:.2f}x")

    if book.unmapped_factor_instruments:
        breaches.append(f"unmapped (no factor map -> fail-closed): "
                        f"{book.unmapped_factor_instruments}")

    return WholeBookGateVerdict(allow=not breaches, mode="", breaches=breaches, details=details)


class _StaticAdapter:
    """Feeds a fixed positions+account snapshot into build_book_state (so the gate reuses the exact
    aggregation/factor logic the live report uses)."""
    def __init__(self, venue: str, account: AccountState, positions: List[CanonicalPosition]):
        self.venue = venue
        self._a = account
        self._p = positions

    def get_account(self) -> AccountState:
        return self._a

    def get_positions(self) -> List[CanonicalPosition]:
        return self._p

    def health(self):
        from app.live_trading.broker_adapter import BrokerHealth
        return BrokerHealth(self.venue, True, True)

    def normalize_instrument(self, s):
        return im.lookup(self.venue, s)


def _canonical(symbol: str, qty: float, price: float, venue: str = im.ALPACA) -> CanonicalPosition:
    iid = im.lookup(venue, symbol)
    inst = im.get(iid) if iid else None
    mult = inst.multiplier if inst else 1.0
    asset_class = inst.asset_class if inst else im.EQUITY
    return CanonicalPosition(
        instrument_id=iid or symbol, venue=venue, broker_symbol=symbol, asset_class=asset_class,
        quantity=qty, price=price, multiplier=mult, currency="USD",
        market_value=qty * price * mult, notional=abs(qty) * price * mult, mapped=iid is not None)


def build_proposed_book(current_positions_raw: List[dict], intents: List[dict],
                        prices: Dict[str, float], nav: float,
                        *, venue: str = im.ALPACA) -> bs.BookState:
    """Apply `intents` (buy/sell deltas) to the broker's CURRENT positions to get the PROPOSED book.
    current_positions_raw: broker dicts {symbol, qty, current_price/market_value}; intents:
    {symbol, side ('buy'/'sell'), qty}; prices: live price map (for symbols not currently held)."""
    qty: Dict[str, float] = {}
    px: Dict[str, float] = dict(prices or {})
    for p in current_positions_raw or []:
        s = p.get("symbol")
        if s is None:
            continue
        qty[s] = qty.get(s, 0.0) + float(p.get("qty", 0.0))
        cp = float(p.get("current_price") or 0.0)
        if cp > 0:
            px.setdefault(s, cp)
    for it in intents or []:
        s = it["symbol"]
        d = float(it["qty"]) * (1.0 if it["side"] == "buy" else -1.0)
        qty[s] = qty.get(s, 0.0) + d
    positions = [_canonical(s, q, px.get(s, 0.0), venue) for s, q in qty.items()
                 if abs(q) > 1e-9]
    account = AccountState(venue=venue, nav=float(nav), cash=0.0, buying_power=0.0)
    return bs.build_book_state([_StaticAdapter(venue, account, positions)])


def shadow_gate_from_intents(current_positions_raw: List[dict], intents: List[dict],
                             prices: Dict[str, float], nav: float, *,
                             mode: str = SHADOW, label: str = "", venue: str = im.ALPACA,
                             policy: RiskPolicy = RISK_POLICY_V1,
                             notifier=None) -> WholeBookGateVerdict:
    """FAIL-SAFE whole-book gate for a sleeve to call before placing. Builds the proposed book,
    evaluates, logs (+ emails on a breach), and returns the verdict. NEVER raises — any error ->
    allow=True so a gate bug can't disrupt a live rebalance. The CALLER decides whether to act on
    the verdict (only in ENFORCE mode)."""
    try:
        book = build_proposed_book(current_positions_raw, intents, prices, nav, venue=venue)
        v = evaluate(book, policy)
        v = WholeBookGateVerdict(allow=v.allow, mode=mode, breaches=v.breaches,
                                 details=v.details)
        if v.breaches:
            log.warning("[whole-book-gate:%s mode=%s] WOULD-BLOCK%s: %s | details=%s",
                        label, mode, ("" if mode == ENFORCE else " (shadow: not blocking)"),
                        "; ".join(v.breaches), v.details)
            if notifier is not None:
                try:
                    notifier.enqueue("whole_book_gate_breach", {
                        "label": label, "mode": mode, "breaches": v.breaches,
                        "details": v.details})
                except Exception:
                    log.debug("whole-book-gate: notify failed", exc_info=True)
        else:
            log.info("[whole-book-gate:%s mode=%s] OK: %s", label, mode, v.details)
        return v
    except Exception as exc:  # noqa: BLE001 — the gate must NEVER break a live rebalance
        # Fail-SAFE: the rebalance proceeds (the per-order + 80% gross caps still bind). But a gate
        # that can't even EVALUATE used to be silent (debug) — H8 makes it visible so a wedged gate
        # gets fixed instead of silently not running. Alert is best-effort; never re-raises.
        log.warning("[whole-book-gate:%s] evaluation error (fail-safe allow): %s", label, exc,
                    exc_info=True)
        if notifier is not None:
            try:
                notifier.enqueue("gate_error", {
                    "gate": "whole_book", "label": label, "mode": mode, "error": str(exc)},
                    dedup_key=f"gate_error:whole_book:{label}")
            except Exception:
                log.debug("whole-book-gate: gate_error notify failed", exc_info=True)
        return WholeBookGateVerdict(allow=True, mode=mode, error=str(exc))
