"""
sleeves.py — the SLEEVE REGISTRY (Alpha-v7 F1+): concrete premia declared for the
Sleeve Lab. Each builder fetches its (cached, deep-history) data, runs a vectorized
PIT-safe backtest, and returns a validated `Sleeve`. New premia are added HERE as a
`@register_sleeve` builder — not as a new top-level `run_*_cpcv.py` script.

CLI:  python -m scripts.walkforward.sleeves [name ...]
      (no args -> evaluate every registered sleeve through the Lab, standalone Track-A
       PAPER+CAPITAL plus Track-B vs the LIVE trend book; report-only, promotes nothing.)
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from typing import Dict, Optional

import pandas as pd

from scripts.walkforward.sleeve_lab import (
    Sleeve, register_sleeve, build_sleeve, list_sleeves,
    evaluate_sleeve, format_sleeve_report,
)

# Live trend book (the current paper book) — the Track-B base. Mirrors TSMOMConfig defaults.
LIVE_TREND_UNIVERSE = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "IEF", "GLD", "DBC", "UUP"]
DEEP_HISTORY_START = date(2007, 1, 1)


# ──────────────────────────────────────────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────────────────────────────────────────
def _today() -> date:
    return datetime.now(timezone.utc).date()


def fetch_bars(symbol: str, *, start: date = DEEP_HISTORY_START,
               end: Optional[date] = None) -> pd.DataFrame:
    """Daily OHLCV for `symbol` from the cached provider (fetches only the missing tail)."""
    from app.data.yfinance_provider import YFinanceProvider
    df = YFinanceProvider().get_daily_bars(symbol, start, end or _today())
    if df is None or df.empty:
        raise RuntimeError(f"no daily bars for {symbol}")
    df = df.copy()
    df.columns = [str(c).lower() for c in df.columns]
    return df


def fetch_universe_closes(symbols, *, start: date = DEEP_HISTORY_START,
                          end: Optional[date] = None) -> pd.DataFrame:
    """Close-price panel for `symbols` (one bulk call, cache-served where possible)."""
    from app.data.yfinance_provider import YFinanceProvider
    bulk = YFinanceProvider().get_daily_bars_bulk(list(symbols), start, end or _today())
    cols = {}
    for sym in symbols:
        df = bulk.get(sym)
        if df is not None and not df.empty:
            c = df.copy()
            c.columns = [str(x).lower() for x in c.columns]
            cols[sym] = c["close"]
    if not cols:
        raise RuntimeError("no closes fetched for the trend universe")
    return pd.DataFrame(cols).sort_index()


def live_trend_book_returns(*, start: date = DEEP_HISTORY_START,
                            end: Optional[date] = None) -> pd.Series:
    """The LIVE 10-ETF TSMOM trend book's daily net returns — the Track-B base book."""
    from app.strategy.tsmom import TSMOMConfig, tsmom_backtest
    closes = fetch_universe_closes(LIVE_TREND_UNIVERSE, start=start, end=end)
    return tsmom_backtest(closes, TSMOMConfig()).returns.dropna()


# ──────────────────────────────────────────────────────────────────────────────────
# Sleeve builders (registry)
# ──────────────────────────────────────────────────────────────────────────────────
@register_sleeve("turn_of_month")
def build_turn_of_month(*, symbol: str = "SPY", bars: Optional[pd.DataFrame] = None,
                        **cfg_kw) -> Sleeve:
    """Turn-of-month calendar premium on `symbol` (default SPY). Declared risk_premium
    (structural premium; trend-orthogonal). One pre-registered config -> n_trials=1."""
    from app.strategy.calendar_premia import TurnOfMonthConfig, turn_of_month_backtest
    bars = fetch_bars(symbol) if bars is None else bars
    res = turn_of_month_backtest(bars, TurnOfMonthConfig(symbol=symbol, **cfg_kw))
    return Sleeve(label=f"turn_of_month_{symbol}", component_type="risk_premium",
                  returns=res.returns, spy_prices=bars["close"] if symbol == "SPY" else None,
                  n_trials_registered=1, registration_id="F1-TOM",
                  notes="last 1 + first 3 trading days of month; long-flat")


@register_sleeve("overnight")
def build_overnight(*, symbol: str = "SPY", bars: Optional[pd.DataFrame] = None,
                    **cfg_kw) -> Sleeve:
    """Overnight (close->open) premium on `symbol`. Declared risk_premium; cost-sensitive
    (full round-trip charged daily). One pre-registered config -> n_trials=1."""
    from app.strategy.calendar_premia import OvernightConfig, overnight_premium_backtest
    bars = fetch_bars(symbol) if bars is None else bars
    res = overnight_premium_backtest(bars, OvernightConfig(symbol=symbol, **cfg_kw))
    return Sleeve(label=f"overnight_{symbol}", component_type="risk_premium",
                  returns=res.returns, spy_prices=bars["close"] if symbol == "SPY" else None,
                  n_trials_registered=1, registration_id="F1-OVN",
                  notes="long close->open every day; flat intraday")


# ──────────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────────
def run_sleeves(names=None, *, with_track_b: bool = True) -> Dict[str, object]:
    """Build + evaluate the named sleeves (default: all) through the Lab; print a uniform
    report each + a summary. Report-only — promotes nothing."""
    names = names or list_sleeves()
    base = live_trend_book_returns() if with_track_b else None
    if base is not None:
        print(f"[base] live trend book: {base.index[0].date()} -> {base.index[-1].date()} "
              f"({len(base)} days)")
    reports = {}
    for name in names:
        print(f"\n[build] {name} ...")
        sleeve = build_sleeve(name)
        rep = evaluate_sleeve(sleeve, base_book_returns=base)
        print(format_sleeve_report(rep))
        reports[name] = rep

    print("\n" + "=" * 78)
    print("  F1 SLEEVE SUMMARY (report-only)")
    print("=" * 78)
    print(f"  {'sleeve':28} {'pointSR':>8} {'hac_p':>7} {'A-paper':>8} "
          f"{'B-IR':>7} {'B-pass':>7}")
    for name, rep in reports.items():
        tb = rep.track_b
        print(f"  {rep.label:28} {rep.point_sr:>8.3f} {rep.hac_p_one_sided:>7.3f} "
              f"{'PASS' if rep.paper_passed else 'FAIL':>8} "
              f"{(tb.appraisal_ir if tb else float('nan')):>7.3f} "
              f"{('PASS' if tb and tb.passed else 'FAIL'):>7}")
    print("=" * 78)
    return reports


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    run_sleeves(argv or None)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from dotenv import load_dotenv
    load_dotenv()
    raise SystemExit(main())
