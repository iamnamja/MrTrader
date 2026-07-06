"""
instrument_master.py — Alpaca-v10 R0.3: the canonical INSTRUMENT MASTER (contract master).

The single mapping between a venue-agnostic canonical instrument id and each venue's broker symbol +
the static spec (asset class, multiplier, currency, cash-equivalent flag). Every position in the
consolidated `BookState` is keyed by `instrument_id` so the brain speaks one language across Alpaca
(equities/ETFs) and IBKR (futures). PURE data + lookups; controls nothing.

R0.3 scope: the LIVE Alpaca universe (the 10 trend ETFs + the T-bill cash sleeve) is seeded with
verified static values. A small representative IBKR futures set is seeded as PLACEHOLDERS — their
multipliers/exchanges MUST be verified against `reqContractDetails` on connect in R1 (per
P2_IBKR_EXECUTION_DESIGN.md "verify-on-connect"); they are marked `verified=False`.

Fail-closed: an instrument with no master entry cannot be sized (the future risk gate treats it as
LIQUIDATION_ONLY). `lookup` returns None on a miss so callers must handle it explicitly.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

# asset-class constants (kept as plain strings for trivial JSON/DB round-tripping)
ETF = "ETF"
EQUITY = "EQUITY"
FUTURE = "FUTURE"
CASH_ETF = "CASH_ETF"          # T-bill ETF — a cash equivalent, excluded from risk gross
CRYPTO = "CRYPTO"

ALPACA = "ALPACA"
IBKR = "IBKR"


@dataclass(frozen=True)
class CanonicalInstrument:
    instrument_id: str                       # canonical, venue-agnostic (e.g. "SPY", "FUT.ES")
    asset_class: str
    currency: str = "USD"
    multiplier: float = 1.0                  # contract multiplier (1.0 for equities/ETFs)
    is_cash_equivalent: bool = False         # T-bills etc. — excluded from the risk gross cap
    venue_symbols: Dict[str, str] = field(default_factory=dict)   # {venue: broker_symbol}
    root: Optional[str] = None               # futures root (for the roll service); None for equities
    exchange: Optional[str] = None           # primary exchange (futures: CME/CBOT/NYMEX/COMEX/CFE)
    verified: bool = True                    # False = static spec must be verify-on-connect (IBKR)

    def broker_symbol(self, venue: str) -> Optional[str]:
        return self.venue_symbols.get(venue)

    @property
    def sec_type(self) -> str:
        """IBKR security type from the asset class (FUT for futures, STK for ETFs/equities)."""
        return "FUT" if self.asset_class == FUTURE else "STK"


def _etf(sym: str, cash: bool = False) -> CanonicalInstrument:
    # ETFs use the SAME ticker on both venues, so register BOTH — otherwise `lookup(IBKR, "SPY")`
    # returns None and, at the all-IBKR cutover, every IBKR-held ETF reads as `mapped=False` → a
    # fail-closed whole-book-gate breach that would HOLD the entire trend rebalance (Alpha-v10 R1, M1).
    return CanonicalInstrument(instrument_id=sym, asset_class=(CASH_ETF if cash else ETF),
                               is_cash_equivalent=cash, venue_symbols={ALPACA: sym, IBKR: sym})


def _fut(root: str, mult: float, exchange: str,
         ibkr_symbol: Optional[str] = None) -> CanonicalInstrument:
    """IBKR futures spec. `root` is the canonical / IBKR tradingClass (e.g. "6E"); `ibkr_symbol` is
    the IBKR request symbol when it differs from the root (FX/VIX: EUR/JPY/VIX). Multipliers/exchanges
    were verify-on-connect-confirmed against the live reqContractDetails (2026-06-22) but stay
    `verified=False` so the read-only adapter ALWAYS re-checks them at connect (the #1-killer guard)."""
    return CanonicalInstrument(instrument_id=f"FUT.{root}", asset_class=FUTURE, multiplier=mult,
                               venue_symbols={IBKR: ibkr_symbol or root}, root=root,
                               exchange=exchange, verified=False)


# The canonical cash-equivalent ETF universe (T-bill / ultra-short Treasury ETFs). SINGLE SOURCE OF
# TRUTH for "what counts as cash" — `cash_sleeve.CASH_ETFS` imports this, so the trading universe and
# the instrument master can NEVER drift. Why this matters (Alpha-v10 H10): the whole-book risk gate
# treats any held symbol that is NOT a registered cash-equivalent as risk gross AND, lacking a factor
# map, as `unmapped` -> a fail-closed breach. If `pm.cash_universe` were ever set to a cash ETF the
# master didn't know (the two lists had drifted), ENFORCE mode would fail-close the entire trend
# rebalance every week on a perfectly legal cash config. One list closes that landmine.
CASH_EQUIVALENT_ETFS = ("SGOV", "BIL", "SHV", "BILS", "VGSH", "GBIL", "USFR", "TBIL")

# --- the registry -------------------------------------------------------------------------------
# LIVE Alpaca universe (verified static values).
_TREND_ETFS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP"]
_CASH_ETFS = list(CASH_EQUIVALENT_ETFS)

# A small representative IBKR futures set: {root: (multiplier, exchange, ibkr_symbol|None)}.
# Values were CONFIRMED against the live reqContractDetails (TWS paper, 2026-06-22): the core book
# (ES/NQ/RTY/CL/NG/GC/SI/HG/ZN/ZB/ZF) matched; ZC/ZS were corrected 50->5000; the FX/VIX request
# symbols differ from the tradingClass root (6E->EUR, 6J->JPY, VX->VIX). `ibkr_symbol=None` => root.
_FUTURES = {"ES": (50.0, "CME", None), "NQ": (20.0, "CME", None), "RTY": (50.0, "CME", None),
            "CL": (1000.0, "NYMEX", None), "NG": (10000.0, "NYMEX", None),
            "GC": (100.0, "COMEX", None), "SI": (5000.0, "COMEX", None), "HG": (25000.0, "COMEX", None),
            "ZN": (1000.0, "CBOT", None), "ZB": (1000.0, "CBOT", None), "ZF": (1000.0, "CBOT", None),
            "ZC": (5000.0, "CBOT", None), "ZS": (5000.0, "CBOT", None),
            "6E": (125000.0, "CME", "EUR"), "6J": (12500000.0, "CME", "JPY"),
            "VX": (1000.0, "CFE", "VIX")}

_REGISTRY: Dict[str, CanonicalInstrument] = {}
for _s in _TREND_ETFS:
    _REGISTRY[_s] = _etf(_s)
for _s in _CASH_ETFS:
    _REGISTRY[_s] = _etf(_s, cash=True)
for _root, (_m, _exch, _ibsym) in _FUTURES.items():
    inst = _fut(_root, _m, _exch, _ibsym)
    _REGISTRY[inst.instrument_id] = inst

# reverse index: (venue, broker_symbol) -> instrument_id
_BY_VENUE_SYMBOL: Dict[tuple, str] = {}
for _iid, _inst in _REGISTRY.items():
    for _venue, _bsym in _inst.venue_symbols.items():
        _BY_VENUE_SYMBOL[(_venue, _bsym)] = _iid


def get(instrument_id: str) -> Optional[CanonicalInstrument]:
    """Canonical instrument by id, or None (fail-closed: callers must handle a miss)."""
    return _REGISTRY.get(instrument_id)


def lookup(venue: str, broker_symbol: str) -> Optional[str]:
    """Map a venue's broker symbol to the canonical instrument_id, or None on a miss."""
    return _BY_VENUE_SYMBOL.get((venue, broker_symbol))


def all_instruments() -> Dict[str, CanonicalInstrument]:
    return dict(_REGISTRY)


def is_cash_equivalent(instrument_id: str) -> bool:
    inst = _REGISTRY.get(instrument_id)
    return bool(inst and inst.is_cash_equivalent)


def futures_instruments() -> Dict[str, CanonicalInstrument]:
    """The IBKR futures universe (instrument_id -> spec) — the set verify-on-connect checks."""
    return {iid: inst for iid, inst in _REGISTRY.items() if inst.asset_class == FUTURE}
