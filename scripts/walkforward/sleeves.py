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
    evaluate_sleeve, format_sleeve_report, register_overlay, build_overlay,
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


@register_sleeve("etf_relative_value")
def build_etf_relative_value(*, prices: Optional[pd.DataFrame] = None, **cfg_kw) -> Sleeve:
    """Slow dollar-neutral ETF relative-value (spread mean-reversion) across the
    pre-registered economically-linked pairs. Declared `diversifier` (market-neutral ->
    orthogonal to the trend book). One pre-registered config -> n_trials=1."""
    from app.strategy.etf_relative_value import (
        RelativeValueConfig, relative_value_backtest,
    )
    cfg = RelativeValueConfig(**cfg_kw)
    if prices is None:
        symbols = sorted({s for pair in cfg.pairs for s in pair} | {"SPY"})
        prices = fetch_universe_closes(symbols)
    res = relative_value_backtest(prices, cfg)
    spy = prices["SPY"] if "SPY" in prices.columns else None
    return Sleeve(label="etf_relative_value", component_type="diversifier",
                  returns=res.returns, spy_prices=spy,
                  n_trials_registered=1, registration_id="F2-RV",
                  notes=f"log-spread MR, {len(cfg.pairs)} pairs, L={cfg.lookback}, "
                        f"entry/exit z={cfg.entry_z}/{cfg.exit_z}; dollar-neutral")


def fetch_yield(ticker: str, *, start: date = date(2002, 1, 1),
                end: Optional[date] = None) -> pd.Series:
    """Deep daily yield series via yfinance (e.g. ^TNX 10y, ^IRX 13-week)."""
    import yfinance as yf
    df = yf.download(ticker, start=start.isoformat(), end=(end or _today()).isoformat(),
                     progress=False, auto_adjust=True)
    if df is None or df.empty:
        raise RuntimeError(f"no data for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [str(c).lower() for c in df.columns]
    return df["close"].dropna()


@register_sleeve("rates_carry")
def build_rates_carry(*, prices: Optional[pd.DataFrame] = None,
                      y10: Optional[pd.Series] = None, y3m: Optional[pd.Series] = None,
                      **cfg_kw) -> Sleeve:
    """Rates duration-carry: size a duration-ETF position by the 10y−3m term spread.
    Declared `risk_premium` (paid to bear duration/curve risk; crisis-correlated by
    nature). One pre-registered config -> n_trials=1."""
    from app.strategy.carry import RatesCarryConfig, rates_carry_backtest
    cfg = RatesCarryConfig(**cfg_kw)
    if prices is None:
        prices = fetch_universe_closes([cfg.duration_etf, "SPY"])
    if y10 is None:
        y10 = fetch_yield("^TNX")
    if y3m is None:
        y3m = fetch_yield("^IRX")
    res = rates_carry_backtest(prices, y10, y3m, cfg)
    spy = prices["SPY"] if "SPY" in prices.columns else None
    return Sleeve(label=res.label, component_type="risk_premium",
                  returns=res.returns, spy_prices=spy,
                  n_trials_registered=1, registration_id="F3-CARRY",
                  notes=f"duration carry: {cfg.duration_etf} sized by (10y-3m)/{cfg.scale_pct}, "
                        f"{'long-short' if cfg.long_short else 'long-flat'}")


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


# ──────────────────────────────────────────────────────────────────────────────────
# Overlays (F1b) — the VIX-term crash governor (book-modifying, not an additive sleeve)
# ──────────────────────────────────────────────────────────────────────────────────
def fetch_vix_term(*, start: date = date(2007, 1, 1),
                   end: Optional[date] = None) -> tuple:
    """Deep ^VIX / ^VIX3M daily closes via yfinance (^VIX3M ~ 2008+). Returns (vix, vix3m)
    Series. Direct fetch (not the 2018+ macro_history cache) to reach the 2008 GFC."""
    import yfinance as yf
    end = (end or _today()).isoformat()
    out = {}
    for tic, key in (("^VIX", "vix"), ("^VIX3M", "vix3m")):
        df = yf.download(tic, start=start.isoformat(), end=end, progress=False,
                         auto_adjust=True)
        if df is None or df.empty:
            raise RuntimeError(f"no data for {tic}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [str(c).lower() for c in df.columns]
        out[key] = df["close"].dropna()
    return out["vix"], out["vix3m"]


@register_overlay("vix_term_governor")
def build_vix_term_governor(*, vix=None, vix3m=None, **cfg_kw):
    """The VIX term-structure crash governor as a sleeve_lab.Overlay (de-risk on
    backwardation). Fetches ^VIX/^VIX3M unless provided (for offline tests)."""
    from scripts.walkforward.sleeve_lab import Overlay
    from app.strategy.crash_governor import VixTermGovernorConfig, vix_term_multiplier
    if vix is None or vix3m is None:
        vix, vix3m = fetch_vix_term()
    mult = vix_term_multiplier(vix, vix3m, VixTermGovernorConfig(**cfg_kw))
    return Overlay(label="vix_term_governor", multiplier=mult,
                   notes="de-risk the book to derisk_to when VIX>VIX3M (backwardation)")


def run_governor(*, derisk_to: float = 0.5) -> object:
    """Evaluate the VIX-term governor on the LIVE trend book (with vs without). Report-only."""
    from scripts.walkforward.sleeve_lab import evaluate_overlay, format_overlay_report
    base = live_trend_book_returns()
    print(f"[base] live trend book: {base.index[0].date()} -> {base.index[-1].date()} "
          f"({len(base)} days)")
    overlay = build_vix_term_governor(derisk_to=derisk_to)
    rep = evaluate_overlay(overlay, base)
    print(format_overlay_report(rep))
    return rep


# ──────────────────────────────────────────────────────────────────────────────────
# Overlays (G1) — credit / yield-curve de-risk overlays
# ──────────────────────────────────────────────────────────────────────────────────
@register_overlay("credit_governor")
def build_credit_governor(*, hyg=None, ief=None, **cfg_kw):
    """HYG/IEF credit-spread de-risk overlay (de-risk when HY underperforms = spreads widening).
    Fetches HYG/IEF deep unless provided (offline tests)."""
    from scripts.walkforward.sleeve_lab import Overlay
    from app.strategy.credit_curve_governor import CreditGovernorConfig, credit_multiplier
    if hyg is None or ief is None:
        closes = fetch_universe_closes(["HYG", "IEF"])
        hyg, ief = closes["HYG"], closes["IEF"]
    mult = credit_multiplier(hyg, ief, CreditGovernorConfig(**cfg_kw))
    return Overlay(label="credit_governor", multiplier=mult,
                   notes="de-risk when HYG/IEF below its trailing MA (credit spreads widening)")


@register_overlay("curve_governor")
def build_curve_governor(*, y10=None, y3m=None, **cfg_kw):
    """Yield-curve (10y-3m) inversion de-risk overlay. Fetches ^TNX/^IRX unless provided."""
    from scripts.walkforward.sleeve_lab import Overlay
    from app.strategy.credit_curve_governor import CurveGovernorConfig, curve_multiplier
    if y10 is None or y3m is None:
        y10, y3m = fetch_yield("^TNX"), fetch_yield("^IRX")
    mult = curve_multiplier(y10, y3m, CurveGovernorConfig(**cfg_kw))
    return Overlay(label="curve_governor", multiplier=mult,
                   notes="de-risk when 10y-3m inverted (spread < threshold)")


def run_credit_curve(*, n_boot: int = 0) -> dict:
    """G1 confirmatory eval: credit + curve overlays, STANDALONE and MARGINAL to the live VIX
    governor, on the live trend book; plus the both-halves stability guard. Report-only."""
    from scripts.walkforward.sleeve_lab import (
        evaluate_overlay, evaluate_overlay_marginal, format_overlay_report,
    )
    base = live_trend_book_returns()
    gov = build_vix_term_governor()
    print(f"[base] live trend book: {base.index[0].date()} -> {base.index[-1].date()} "
          f"({len(base)} days)")
    out = {}
    for name in ("credit_governor", "curve_governor"):
        ov = build_overlay(name)
        standalone = evaluate_overlay(ov, base)
        marginal = evaluate_overlay_marginal(ov, base, prior_overlays=[gov])
        # both-halves stability on the MARGINAL d_max_dd (split the active window at midpoint)
        mser = ov.multiplier
        mid = mser.index[len(mser) // 2]
        h1 = evaluate_overlay_marginal(
            type(ov)(ov.label, mser[mser.index < mid]), base, prior_overlays=[gov])
        h2 = evaluate_overlay_marginal(
            type(ov)(ov.label, mser[mser.index >= mid]), base, prior_overlays=[gov])
        print("\n[STANDALONE] " + name)
        print(format_overlay_report(standalone))
        print("\n[MARGINAL vs governor] " + name)
        print(format_overlay_report(marginal))
        print(f"  both-halves marginal d_max_dd: H1={h1.d_max_dd:+.4f} H2={h2.d_max_dd:+.4f} "
              f"-> both_improve={h1.d_max_dd > 0 and h2.d_max_dd > 0}")
        out[name] = {"standalone": standalone, "marginal": marginal,
                     "h1_d_max_dd": h1.d_max_dd, "h2_d_max_dd": h2.d_max_dd}
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "governor":
        run_governor()
        return 0
    if argv and argv[0] == "credit_curve":
        run_credit_curve()
        return 0
    run_sleeves(argv or None)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from dotenv import load_dotenv
    load_dotenv()
    raise SystemExit(main())
