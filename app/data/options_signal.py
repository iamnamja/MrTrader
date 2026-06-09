"""
Options-data-as-SIGNAL — OPT-5. Use the (paused-program) options data to ENHANCE the validated
equity sleeves, not to trade options. The first signal: the pre-earnings IMPLIED MOVE (the ATM
straddle priced in just before a report), used to normalize PEAD's announce-day reaction — i.e.
an options-implied "priced-in" filter (the price-only version is already in pead_scorer.py).

Judged on the HOST sleeve's existing gate (PEAD CPCV) — no options execution, so the alpha-gate
-vs-risk-premium mismatch that paused the standalone options sleeves (DECISIONS 2026-06-09) does
not apply here.

PIT: every accessor uses only option bars with knowable_date <= as_of. Efficient: bars/contracts
are read per-underlying on demand (pyarrow predicate pushdown) and cached, so a CPCV that touches
a few dozen names never loads the full ~60M-row store.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Dict, Optional

import pandas as pd

from app.data.options_provider import (
    OPTIONS_BARS_PARQUET, OPTIONS_CONTRACTS_PARQUET, parse_occ,
)

logger = logging.getLogger(__name__)


class ImpliedMoveProvider:
    """Computes the pre-earnings implied move (ATM straddle / spot) PIT, with lazy per-symbol
    loading + caching. `implied_move(symbol, as_of, spot)` returns the fractional move the option
    market priced in as of `as_of` for the front expiry, or None if the chain isn't there."""

    def __init__(self, bars_path=OPTIONS_BARS_PARQUET, contracts_path=OPTIONS_CONTRACTS_PARQUET,
                 min_dte: int = 1, max_dte: int = 60):
        self._bars_path = bars_path
        self._contracts_path = contracts_path
        self.min_dte = min_dte
        self.max_dte = max_dte
        self._bars_cache: Dict[str, pd.DataFrame] = {}
        self._meta_cache: Dict[str, pd.DataFrame] = {}

    def _sym_bars(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._bars_cache:
            try:
                df = pd.read_parquet(self._bars_path,
                                     filters=[("underlying", "==", symbol)])
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    df["knowable_date"] = pd.to_datetime(df["knowable_date"])
            except Exception:
                df = pd.DataFrame()
            self._bars_cache[symbol] = df
        return self._bars_cache[symbol]

    def _sym_meta(self, symbol: str) -> pd.DataFrame:
        """Per-contract metadata for `symbol` (contract, contract_type, strike, expiration),
        parsed from the OCC tickers in this symbol's bars (avoids loading the contracts store)."""
        if symbol not in self._meta_cache:
            bars = self._sym_bars(symbol)
            if bars.empty:
                self._meta_cache[symbol] = pd.DataFrame(
                    columns=["contract", "contract_type", "strike", "expiration"])
            else:
                rows = []
                for c in bars["contract"].unique():
                    m = parse_occ(c)
                    if m:
                        rows.append({"contract": c, "contract_type": m["contract_type"],
                                     "strike": m["strike"], "expiration": m["expiration"]})
                self._meta_cache[symbol] = pd.DataFrame(rows)
        return self._meta_cache[symbol]

    def implied_move(self, symbol: str, observation_date: date, spot: float,
                     knowable_asof: Optional[date] = None) -> Optional[float]:
        """Fractional implied move = (ATM call + ATM put) close ON `observation_date` / spot,
        front expiry in [obs+min_dte, obs+max_dte]. `knowable_asof` is the PIT cutoff (the
        decision/scoring day): the observation-day bar has knowable_date = obs+1 bday, so a caller
        deciding on `observation_date` itself could NOT see it — pass the later scoring date. When
        None, defaults to obs+5 calendar days (covers the +1 bday + weekend). None if unavailable."""
        if not spot or spot <= 0:
            return None
        bars = self._sym_bars(symbol)
        if bars.empty:
            return None
        obs_ts = pd.Timestamp(observation_date)
        cutoff = pd.Timestamp(knowable_asof) if knowable_asof else (
            obs_ts + pd.Timedelta(days=5))
        day = bars[(bars["date"] == obs_ts) & (bars["knowable_date"] <= cutoff)]
        if day.empty:
            return None
        meta = self._sym_meta(symbol)
        if meta.empty:
            return None
        day = day.merge(meta, on="contract", how="inner")
        if day.empty:
            return None
        exp_d = pd.to_datetime(day["expiration"]).dt.date
        lo = observation_date + timedelta(days=self.min_dte)
        hi = observation_date + timedelta(days=self.max_dte)
        span = day[(exp_d >= lo) & (exp_d <= hi)]
        if span.empty:
            return None
        # front expiry that spans the event
        expiry = min(pd.to_datetime(span["expiration"]).dt.date)
        leg = span[pd.to_datetime(span["expiration"]).dt.date == expiry]
        calls = leg[leg["contract_type"] == "call"]
        puts = leg[leg["contract_type"] == "put"]
        if calls.empty or puts.empty:
            return None
        # ATM = strike nearest spot (independently for call and put — usually the same strike)
        c_atm = calls.iloc[(calls["strike"] - spot).abs().argmin()]
        p_atm = puts.iloc[(puts["strike"] - spot).abs().argmin()]
        straddle = float(c_atm["close"]) + float(p_atm["close"])
        if straddle <= 0:
            return None
        return straddle / spot
