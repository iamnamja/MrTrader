"""
book_state.py — Alpha-v10 R0.4: the consolidated cross-venue BOOK STATE + factor-exposure view.

Assembles one immutable view of the whole book across venues (Alpaca + IBKR-later) from the
read-only broker adapters: positions, per-venue accounts, aggregate gross/net, and the NETTED
FACTOR EXPOSURE vector (equity beta, rates DV01, USD, commodity, vol) — the thing that catches
stacked SPY-on-Alpaca + ES-on-IBKR equity beta. Shadow / report-only: it READS the adapters and
computes; it controls nothing.

"Risk globally, capital locally": exposures are aggregated ACROSS venues; cash/margin stay PER venue
(you cannot move margin between brokers). T-bill cash-equivalents are excluded from the risk gross.

The factor map is a small, STABLE set of hand-curated priors (per the roadmap: NOT re-fit weekly).
An instrument with no factor-map entry contributes 0 to factors and is flagged `unmapped_factor` —
the future risk gate treats such an instrument as fail-closed (cannot size).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.live_trading.broker_adapter import AccountState, BrokerAdapter, CanonicalPosition

# Factor keys (a small, stable set) + their UNITS. The factors live in one dict but have DIFFERENT
# units — they must NEVER be summed across keys (a consumer reads each key with its own unit).
EQUITY_BETA = "equity_beta"        # $ beta-equivalent (loading x signed $notional)
RATES_DV01 = "rates_dv01"          # $ per 1bp (sign: long bonds = positive duration)
USD = "usd"                        # $ USD exposure
COMMODITY = "commodity"            # $ commodity exposure
VOL = "vol"                        # $ vol exposure (position sign carries direction)
FACTOR_UNITS = {EQUITY_BETA: "usd_beta", RATES_DV01: "usd_per_bp", USD: "usd",
                COMMODITY: "usd", VOL: "usd_vol"}

# Hand-curated factor loadings per canonical instrument (per $ of signed notional, except DV01).
# Stable priors; deliberately coarse (catch stacked risk, do NOT overfit a factor model).
_FACTOR_MAP: Dict[str, Dict[str, float]] = {
    # equities / equity-index futures -> equity beta ~1
    "SPY": {EQUITY_BETA: 1.0}, "QQQ": {EQUITY_BETA: 1.1}, "IWM": {EQUITY_BETA: 1.1},
    "EFA": {EQUITY_BETA: 0.9}, "EEM": {EQUITY_BETA: 1.0},
    "FUT.ES": {EQUITY_BETA: 1.0}, "FUT.NQ": {EQUITY_BETA: 1.1}, "FUT.RTY": {EQUITY_BETA: 1.1},
    # rates / bond futures -> rates duration (DV01 per $notional, approx)
    "TLT": {RATES_DV01: 0.0017}, "IEF": {RATES_DV01: 0.0008},
    "FUT.ZN": {RATES_DV01: 0.0006}, "FUT.ZB": {RATES_DV01: 0.0017}, "FUT.ZF": {RATES_DV01: 0.0004},
    # USD
    "UUP": {USD: 1.0}, "FUT.6E": {USD: -1.0}, "FUT.6J": {USD: -1.0},
    # commodity
    "GLD": {COMMODITY: 1.0, USD: -0.3}, "DBC": {COMMODITY: 1.0},
    "FUT.CL": {COMMODITY: 1.0}, "FUT.GC": {COMMODITY: 1.0, USD: -0.3},
    "FUT.SI": {COMMODITY: 1.0}, "FUT.HG": {COMMODITY: 1.0}, "FUT.NG": {COMMODITY: 1.0},
    "FUT.ZC": {COMMODITY: 1.0}, "FUT.ZS": {COMMODITY: 1.0},
    # vol (short the front VIX future = negative vol exposure; the SLEEVE applies the sign)
    "FUT.VX": {VOL: 1.0},
    # T-bills: cash-equivalent, no factor exposure
    "SGOV": {}, "BIL": {}, "SHV": {},
}


@dataclass(frozen=True)
class BookState:
    as_of: Optional[str]
    positions: List[CanonicalPosition]
    accounts: Dict[str, AccountState]                  # per venue
    gross_notional: float                              # ex-cash-equivalents
    net_notional: float
    gross_ex_cash_frac: float                          # gross / total NAV
    factor_exposures: Dict[str, float]                 # netted across venues
    total_nav: float
    cash_equiv_value: float
    unmapped_factor_instruments: List[str] = field(default_factory=list)
    stale_price_instruments: List[str] = field(default_factory=list)
    reconciliation_ok: Optional[bool] = None           # set by the reconciler when run

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        d["positions"] = [p.__dict__ for p in self.positions]
        d["accounts"] = {k: v.__dict__ for k, v in self.accounts.items()}
        return d


def factor_loadings(instrument_id: str) -> Optional[Dict[str, float]]:
    """The hand-curated factor loadings for an instrument, or None if it has no entry (fail-closed
    at the risk gate). An explicit empty dict {} (e.g. T-bills) means 'mapped, zero exposure'."""
    return _FACTOR_MAP.get(instrument_id)


def build_book_state(adapters: List[BrokerAdapter], *, as_of: Optional[str] = None) -> BookState:
    """Assemble the consolidated cross-venue book state from the read-only adapters."""
    positions: List[CanonicalPosition] = []
    accounts: Dict[str, AccountState] = {}
    for ad in adapters:
        accounts[ad.venue] = ad.get_account()
        positions.extend(ad.get_positions())

    total_nav = float(sum(a.nav for a in accounts.values()))
    factors: Dict[str, float] = {}
    gross = net = cash_equiv = 0.0
    unmapped: List[str] = []
    stale_price: List[str] = []
    for p in positions:
        if p.is_cash_equivalent:
            cash_equiv += p.market_value
            continue                                   # cash-equivalents excluded from risk gross
        # FULL signed notional from (qty, price, mult) — NOT the broker market_value: for FUTURES
        # IBKR reports marketValue as daily P&L (~0 at entry), which would understate stacked beta to
        # ~0 (the exact risk this view exists to catch). qty*price*mult is full notional for futures
        # and equals market_value for equities/ETFs. gross/net/factors all use this one quantity so a
        # single bad input can't desync them.
        signed = p.quantity * p.price * p.multiplier
        if signed == 0.0 and p.market_value:           # stale/zero mark on a real position
            signed = p.market_value
            stale_price.append(p.instrument_id)
        gross += abs(signed)
        net += signed
        load = factor_loadings(p.instrument_id)
        if load is None:
            unmapped.append(p.instrument_id)
            continue
        for k, v in load.items():
            factors[k] = factors.get(k, 0.0) + signed * v

    gross_ex_cash_frac = (gross / total_nav) if total_nav > 0 else 0.0
    return BookState(
        as_of=as_of, positions=positions, accounts=accounts,
        gross_notional=gross, net_notional=net, gross_ex_cash_frac=gross_ex_cash_frac,
        factor_exposures=factors, total_nav=total_nav, cash_equiv_value=cash_equiv,
        unmapped_factor_instruments=sorted(set(unmapped)),
        stale_price_instruments=sorted(set(stale_price)))
