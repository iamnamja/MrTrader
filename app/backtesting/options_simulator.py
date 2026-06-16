"""
Contract-level options simulator + spread cost model — OPT-2.

Backtests a sequence of (defined-risk) option positions and emits a daily-MTM equity curve
in the SAME ``SimResult`` shape as the equity simulators, so every downstream WF/CPCV gate
(significance, DSR, PF, Calmar, fold-coverage, CAPM residual-α) reuses verbatim.

Design choices (and why):
  * **Mark to REAL EOD option closes** (from the OPT-1b data layer), not theoretical engine
    prices. Real closes embed the actual market IV — so IV-crush is carried by the data
    itself (a short straddle into earnings simply reprices lower the next day) with no
    synthesis. The OPT-1a engine is for greeks/analytics in strategies, not for marking.
  * **No historical NBBO**, so entry/exit cost is a MODELED spread (% of premium) with a
    mandatory stress multiplier (a KEEP must survive 2×). Held-to-expiry legs settle at
    intrinsic with NO exit cost (no trade — assignment/expiry).
  * **Defined-risk payoff caps are automatic**: every leg is marked, and at expiry each
    settles at its own intrinsic, so a vertical's payoff is naturally capped at the strike
    width minus net debit — no special-case cap logic that could drift from reality.
  * **Forward-fill** an open leg's mark on no-trade days (last close ≤ d); settle at expiry
    using the underlying close (intrinsic).

Implements the OPT-0 ``OptionsSpreadCostModel`` + ``OptionContractSim`` contracts.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from app.backtesting.strategy_simulator import SimResult, StrategySimulator
from app.backtesting.metrics import Trade
from app.data.options_provider import parse_occ
from app.options.spread_model import CalibratedSpreadModel  # noqa: F401 (cost-model type)

logger = logging.getLogger(__name__)

MULTIPLIER = 100  # shares per US equity-option contract
PROFIT_FACTOR_CAP = 99.0  # finite sentinel for an all-wins fold (inf would poison gates)


# ── Cost model (implements OptionsSpreadCostModel) ─────────────────────────────

@dataclass
class OptionsSpreadCostModel:
    """One-way cost to open/close ONE option leg (per contract). No historical NBBO, so we
    charge a modeled half-spread as a % of premium × a stress multiplier, plus a flat
    per-contract fee. `spread_mult` (1×/2×/3×) is the mandatory stress knob.

    Flat default (1% half-spread). The contract-context kwargs (moneyness/dte/contract_type)
    are accepted but IGNORED here — they let the simulator pass the same call to a
    CalibratedOptionsSpreadCostModel without branching at the call site."""
    spread_pct: float = 0.01        # half-spread as a fraction of premium (1% default)
    per_contract_fee: float = 0.65  # broker commission per contract

    def entry_exit_cost(self, premium: float, spread_mult: float = 1.0, *,
                        moneyness: Optional[float] = None, dte: Optional[float] = None,
                        contract_type: Optional[str] = None,
                        underlying: Optional[str] = None) -> float:
        """Dollar cost for ONE contract at `premium` (per-share option price). The
        simulator scales by quantity. Held-to-expiry legs are charged no exit cost."""
        return abs(premium) * self.spread_pct * spread_mult * MULTIPLIER + self.per_contract_fee


@dataclass
class CalibratedOptionsSpreadCostModel:
    """P2-4 — moneyness/DTE-aware cost model. Same premium-% × stress-mult structure as the flat
    model, but the half-spread fraction is looked up PER CONTRACT from an empirical
    `CalibratedSpreadModel` (calibrated from the live NBBO log) using the contract's underlying,
    moneyness, DTE and call/put type. ALWAYS delegates to the model — when the simulator can't
    supply context the model itself returns its CONSERVATIVE global (high), never an optimistic
    flat number. The per-contract fee is unchanged."""
    model: "CalibratedSpreadModel"
    per_contract_fee: float = 0.65

    def entry_exit_cost(self, premium: float, spread_mult: float = 1.0, *,
                        moneyness: Optional[float] = None, dte: Optional[float] = None,
                        contract_type: Optional[str] = None,
                        underlying: Optional[str] = None) -> float:
        half = self.model.half_spread_pct(moneyness, dte, contract_type, underlying)
        return abs(premium) * half * spread_mult * MULTIPLIER + self.per_contract_fee


# ── Position spec ──────────────────────────────────────────────────────────────

@dataclass
class OptionLeg:
    contract: str       # OCC ticker, e.g. O:SPY260116C00500000
    side: int           # +1 long, -1 short
    qty: int = 1        # number of contracts


@dataclass
class OptionPosition:
    legs: List[OptionLeg]
    entry_date: date
    exit_date: Optional[date] = None   # None or >= expiry -> hold to expiry (intrinsic settle)
    label: str = ""


# ── Simulator (implements OptionContractSim) ───────────────────────────────────

class OptionsSimulator:
    def __init__(self, bars: pd.DataFrame,
                 underlying_prices: Optional[Dict[str, Dict[date, float]]] = None,
                 cost_model: Optional[OptionsSpreadCostModel] = None,
                 starting_capital: float = 100_000.0):
        """bars: long OHLCV frame (app.data.options_provider.BARS_COLS — needs at least
        contract/date/close). underlying_prices: {underlying: {date: close}} for expiry
        intrinsic settlement (falls back to the option's last close if absent)."""
        self.cost_model = cost_model or OptionsSpreadCostModel()
        self.starting_capital = float(starting_capital)
        self._closes: Dict[str, List[Tuple[date, float]]] = {}
        self._meta: Dict[str, Optional[dict]] = {}
        if bars is not None and not bars.empty:
            for contract, grp in bars.groupby("contract"):
                ser = [(pd.Timestamp(d).date(), float(c))
                       for d, c in zip(grp["date"], grp["close"]) if pd.notna(c)]
                ser.sort()
                self._closes[contract] = ser
                self._meta[contract] = parse_occ(contract)
        self._und = underlying_prices or {}

    # ── marks ──────────────────────────────────────────────────────────────────

    def _close_on_or_before(self, contract: str, d: date) -> Optional[float]:
        ser = self._closes.get(contract)
        if not ser:
            return None
        last = None
        for dd, c in ser:
            if dd <= d:
                last = c
            else:
                break
        return last

    def _underlying_on_or_before(self, underlying: str, d: date) -> Optional[float]:
        m = self._und.get(underlying)
        if not m:
            return None
        best = None
        for dd in sorted(m):
            if dd <= d:
                best = m[dd]
            else:
                break
        return best

    def _intrinsic(self, contract: str, at: date) -> Optional[float]:
        meta = self._meta.get(contract)
        if not meta:
            return self._close_on_or_before(contract, at)
        S = self._underlying_on_or_before(meta["underlying"], meta["expiration"])
        if S is None:
            # No underlying price for settlement -> fall back to last option close (logged:
            # this silently changes settlement semantics, so surface it).
            logger.warning("OptionsSimulator: no underlying price to settle %s at expiry; "
                           "falling back to last option close", contract)
            return self._close_on_or_before(contract, at)
        K = meta["strike"]
        return max(0.0, S - K) if meta["contract_type"] == "call" else max(0.0, K - S)

    def _mark(self, contract: str, d: date) -> Optional[float]:
        """Per-share mark for `contract` on day `d`: real close (fwd-filled) while alive;
        intrinsic once at/after expiration."""
        meta = self._meta.get(contract)
        if meta and d >= meta["expiration"]:
            return self._intrinsic(contract, d)
        return self._close_on_or_before(contract, d)

    # ── run ──────────────────────────────────────────────────────────────────

    def run(self, positions_spec: Sequence[OptionPosition], start: date, end: date,
            spread_mult: float = 1.0) -> SimResult:
        cal = self._calendar(start, end)
        # Per-position precompute: contribution(d) and a realized Trade.
        contribs: List[Tuple[date, date, dict]] = []  # (entry, exit_eff, per-day fn data)
        trades: List[Trade] = []
        dropped = 0
        for pos in positions_spec:
            priced = self._price_position(pos, spread_mult)
            if priced is None:
                dropped += 1
                continue
            contribs.append(priced)
            trades.append(priced[2]["trade"])
        if dropped:
            logger.warning("OptionsSimulator: dropped %d/%d positions (no entry price or "
                           "unparseable contract) — NOT silent", dropped, len(positions_spec))

        equity_curve: List[Tuple[date, float]] = []
        for d in cal:
            eq = self.starting_capital
            for entry, exit_eff, info in contribs:
                if d < entry:
                    continue
                if d >= exit_eff:
                    eq += info["realized_pnl"] - info["entry_cost"] - info["exit_cost"]
                else:
                    eq += self._unrealized(info, d) - info["entry_cost"]
            equity_curve.append((d, eq))

        return self._to_simresult(equity_curve, trades, start, end, dropped)

    def _calendar(self, start: date, end: date) -> List[date]:
        days = set()
        for ser in self._closes.values():
            for dd, _ in ser:
                if start <= dd <= end:
                    days.add(dd)
        for m in self._und.values():
            for dd in m:
                if start <= dd <= end:
                    days.add(dd)
        return sorted(days)

    def _cost_context(self, contract: str, meta: Optional[dict], on_date: date
                      ) -> Tuple[Optional[float], Optional[int], Optional[str], Optional[str]]:
        """(moneyness, dte, contract_type, underlying) for the cost model — moneyness from the
        underlying close on/before `on_date`, dte in calendar days to expiry. Returns Nones if
        unknown (cost model then falls back), never raises."""
        try:
            if not meta:
                return (None, None, None, None)
            und = meta.get("underlying")
            S = self._underlying_on_or_before(und, on_date)
            strike = meta.get("strike")
            moneyness = (float(strike) / float(S)) if (S and strike) else None
            dte = (meta["expiration"] - on_date).days if meta.get("expiration") else None
            return (moneyness, dte, meta.get("contract_type"), und)
        except Exception:
            return (None, None, None, None)

    def _price_position(self, pos: OptionPosition, spread_mult: float) -> Optional[dict]:
        legs = []
        entry_cost = exit_cost = 0.0
        for leg in pos.legs:
            entry_mark = self._close_on_or_before(leg.contract, pos.entry_date)
            if entry_mark is None:
                return None  # cannot enter a position we have no entry price for (logged)
            meta = self._meta.get(leg.contract)
            if meta is None:
                # Unparseable OCC -> we can't know the expiration; drop rather than collapse
                # the position to a same-day no-op (silent wrong P&L).
                return None
            expiry = meta["expiration"]
            held_to_expiry = (pos.exit_date is None) or (pos.exit_date >= expiry)
            exit_eff = expiry if held_to_expiry else pos.exit_date
            exit_mark = self._mark(leg.contract, exit_eff)
            if exit_mark is None:
                exit_mark = entry_mark
            legs.append({"leg": leg, "entry_mark": entry_mark, "exit_mark": exit_mark,
                         "exit_eff": exit_eff})
            # Pass per-contract context (moneyness/DTE/type) so a calibrated cost model can
            # price the spread realistically; the flat model ignores it. Computed at the
            # relevant date (entry vs exit) since moneyness/DTE both move over the hold.
            em, ed, ect, eund = self._cost_context(leg.contract, meta, pos.entry_date)
            entry_cost += self.cost_model.entry_exit_cost(
                entry_mark, spread_mult, moneyness=em, dte=ed, contract_type=ect,
                underlying=eund) * leg.qty
            if not held_to_expiry:  # no exit cost when held to expiry/assignment
                xm, xd, xct, xund = self._cost_context(leg.contract, meta, exit_eff)
                exit_cost += self.cost_model.entry_exit_cost(
                    exit_mark, spread_mult, moneyness=xm, dte=xd, contract_type=xct,
                    underlying=xund) * leg.qty
        exit_eff = max(x["exit_eff"] for x in legs)
        realized = sum(x["leg"].side * (x["exit_mark"] - x["entry_mark"]) * MULTIPLIER
                       * x["leg"].qty for x in legs)
        entry_prem = sum(x["leg"].side * x["entry_mark"] * MULTIPLIER * x["leg"].qty
                         for x in legs)
        pnl = realized - entry_cost - exit_cost
        trade = Trade(
            symbol=pos.label or (self._meta.get(pos.legs[0].contract) or {}).get(
                "underlying", pos.legs[0].contract),
            entry_date=pos.entry_date, exit_date=exit_eff,
            entry_price=entry_prem, exit_price=entry_prem + realized,
            quantity=sum(abs(le.qty) for le in pos.legs), pnl=pnl,
            pnl_pct=(pnl / abs(entry_prem)) if entry_prem else 0.0,
            hold_bars=max(0, (exit_eff - pos.entry_date).days),
            exit_reason="EXPIRY" if (pos.exit_date is None or pos.exit_date >= exit_eff)
            else "CLOSE", trade_type="options")
        return (pos.entry_date, exit_eff, {
            "realized_pnl": realized, "entry_cost": entry_cost, "exit_cost": exit_cost,
            "legs": legs, "trade": trade})

    def _unrealized(self, info: dict, d: date) -> float:
        total = 0.0
        for x in info["legs"]:
            mark = self._mark(x["leg"].contract, d)
            if mark is None:
                mark = x["entry_mark"]
            total += x["leg"].side * (mark - x["entry_mark"]) * MULTIPLIER * x["leg"].qty
        return total

    # ── result assembly ────────────────────────────────────────────────────────

    def _to_simresult(self, equity_curve, trades, start, end, dropped=0) -> SimResult:
        eq_vals = [v for _, v in equity_curve]
        start_cap = self.starting_capital
        # Daily returns seeded from the capital baseline so the FIRST day's entry-cost step
        # enters Sharpe/Sortino (positions open within [start,end], so this isn't a mid-
        # position jump). One return per calendar day.
        daily_returns: List[float] = []
        dated: Dict[date, float] = {}
        prev = start_cap
        for d, v in equity_curve:
            r = (v - prev) / prev if prev else 0.0
            daily_returns.append(r)
            dated[d] = r
            prev = v

        end_cap = eq_vals[-1] if eq_vals else start_cap
        # Blow-up guard: a defined-risk book should never go <= 0; if it does (sign/qty bug
        # or an undefined-risk spec), the level metrics (calmar, max_dd) become meaningless,
        # so flag it loudly rather than reporting a benign-looking 0% Calmar.
        blown_up = any(v <= 0 for v in eq_vals)
        if blown_up:
            logger.error("OptionsSimulator: equity went <= 0 (min=%.2f) — BLOWN UP; level "
                         "metrics unreliable, treat as a hard fail", min(eq_vals))
        total_ret = (end_cap - start_cap) / start_cap if start_cap else 0.0
        n_days = max(1, (end - start).days)
        ann_ret = ((end_cap / start_cap) ** (365.0 / n_days) - 1.0) if (
            start_cap > 0 and end_cap > 0) else 0.0
        sharpe = StrategySimulator._sharpe(daily_returns)
        sortino = StrategySimulator._sortino(daily_returns)
        max_dd = StrategySimulator._max_drawdown(eq_vals)
        calmar = (ann_ret / max_dd) if max_dd > 1e-9 else 0.0

        wins = [t.pnl for t in trades if t.pnl > 0]
        losses = [t.pnl for t in trades if t.pnl < 0]
        gross_win = sum(wins)
        gross_loss = abs(sum(losses))
        profit_factor = (min(gross_win / gross_loss, PROFIT_FACTOR_CAP)
                         if gross_loss > 1e-9 else (PROFIT_FACTOR_CAP if gross_win > 0 else 0.0))
        win_rate = (len(wins) / len(trades)) if trades else 0.0
        avg_pnl_pct = (sum(t.pnl_pct for t in trades) / len(trades)) if trades else 0.0
        costs = sum(0.0 for _ in trades)  # captured inside pnl; explicit field kept 0

        res = SimResult(
            model_type="options", starting_capital=start_cap, ending_capital=end_cap,
            total_return_pct=total_ret * 100, annualized_return_pct=ann_ret * 100,
            sharpe_ratio=sharpe, sortino_ratio=sortino, max_drawdown_pct=max_dd * 100,
            calmar_ratio=calmar, total_trades=len(trades), win_rate=win_rate,
            avg_pnl_pct=avg_pnl_pct, profit_factor=profit_factor,
            transaction_costs_total=costs, equity_curve=equity_curve, trades=trades)
        # Attributes not on the SimResult dataclass (the OPT-3 adapter reads these):
        #   daily_returns_dated -> FoldResult; dropped_positions / blown_up -> health flags.
        res.daily_returns_dated = dated
        res.dropped_positions = dropped
        res.blown_up = blown_up
        return res


def daily_returns_dated(result: SimResult) -> Dict[date, float]:
    """Date->daily-return map from a SimResult's equity curve (what the OPT-3 adapter feeds
    FoldResult). Works whether or not the simulator attached it directly."""
    attached = getattr(result, "daily_returns_dated", None)
    if isinstance(attached, dict):
        return attached
    out: Dict[date, float] = {}
    ec = result.equity_curve
    for i in range(1, len(ec)):
        prev = ec[i - 1][1]
        out[ec[i][0]] = (ec[i][1] - prev) / prev if prev else 0.0
    return out
