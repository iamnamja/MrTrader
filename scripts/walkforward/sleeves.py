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
from app.data.alpaca_crypto_provider import DEFAULT_CRYPTO_UNIVERSE, CRYPTO_HISTORY_START

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


# ── Crypto trend (P3-1) — TSMOM on Alpaca spot crypto ──────────────────────────────
# ONE pre-registered config (registration_id=P3-1-CRYPTO-TREND, n_trials=1 — no sweeping):
# crypto trades 365 days/yr so ann=365 + CALENDAR-day lookbacks (~1/3/6/12mo); long-flat
# (Alpaca spot can't short); HARD book-level vol target so the wild crypto vol doesn't
# dominate; conservative 25 bps one-way cost (Alpaca crypto spread+fee >> equity ETFs).
def crypto_trend_config(**cfg_kw):
    from app.strategy.tsmom import TSMOMConfig
    base = dict(
        universe=list(DEFAULT_CRYPTO_UNIVERSE),
        lookbacks=(30, 90, 180, 365),   # calendar-day ~1/3/6/12mo (crypto 365-day year)
        vol_lookback=90, target_vol=0.10, vol_floor=0.30,   # crypto vol floor >> equity 0.03
        rebalance_days=7, max_weight=0.25, max_gross=1.0, allow_short=False,
        cost_bps=25.0, ann=365, book_vol_target=0.20, book_vol_max_leverage=2.0,
    )
    base.update(cfg_kw)
    return TSMOMConfig(**base)


def fetch_crypto_universe_closes(symbols, *, start: date = CRYPTO_HISTORY_START,
                                 end: Optional[date] = None) -> pd.DataFrame:
    """Daily crypto close panel from Alpaca (columns = symbols with data)."""
    from app.data.alpaca_crypto_provider import AlpacaCryptoProvider
    closes = AlpacaCryptoProvider().get_daily_closes(list(symbols), start=start, end=end)
    if closes is None or closes.empty:
        raise RuntimeError("no crypto closes fetched from Alpaca")
    return closes


def crypto_trend_book_returns(*, start: date = CRYPTO_HISTORY_START,
                              end: Optional[date] = None) -> pd.Series:
    """The crypto TSMOM book's daily net returns (the P3-1 sleeve return stream)."""
    from app.strategy.tsmom import tsmom_backtest
    cfg = crypto_trend_config()
    closes = fetch_crypto_universe_closes(cfg.universe, start=start, end=end)
    return tsmom_backtest(closes, cfg).returns.dropna()


# ──────────────────────────────────────────────────────────────────────────────────
# Sleeve builders (registry)
# ──────────────────────────────────────────────────────────────────────────────────
@register_sleeve("crypto_trend")
def build_crypto_trend(*, prices: Optional[pd.DataFrame] = None, **cfg_kw) -> Sleeve:
    """TSMOM on the Alpaca spot-crypto basket (BTC/ETH + liquid alts), long-flat, HARD
    book-vol-targeted, 365-day annualized. Declared `diversifier` — the P3-1 thesis is a
    new, lowly-correlated (to ETF trend) return stream. ONE pre-registered config -> n_trials=1.
    Honest power floor: Alpaca crypto history starts ~2021 (~5y), so CAPITAL is structurally
    unreachable on backtest alone (n_folds) — PAPER + live-paper ratification only."""
    from app.strategy.tsmom import tsmom_backtest
    cfg = crypto_trend_config(**cfg_kw)
    if prices is None:
        prices = fetch_crypto_universe_closes(cfg.universe)
    res = tsmom_backtest(prices, cfg)
    return Sleeve(label="crypto_trend", component_type="diversifier",
                  returns=res.returns.dropna(), spy_prices=None, periods_per_year=365,
                  n_trials_registered=1, registration_id="P3-1-CRYPTO-TREND",
                  notes=f"TSMOM on {len(cfg.universe)} Alpaca cryptos; calendar lookbacks "
                        f"{cfg.lookbacks}; long-flat; hard book_vol_target={cfg.book_vol_target}; "
                        f"ann=365; {cfg.cost_bps}bps 1-way")


# ── Futures (P4-2) — Norgate cross-asset trend + carry (reads the local parquet mirror) ──
def futures_trend_config(universe):
    """LIVE trend SIGNAL params (lookbacks/vol_lookback/target_vol/weekly), only the SIZING
    scaled for a many-market SHORTABLE book (smaller per-name cap, higher gross cap, book-vol
    target). No return-tuning — the signal is the already-validated live config."""
    from app.strategy.tsmom import TSMOMConfig
    return TSMOMConfig(universe=list(universe), lookbacks=(21, 63, 126, 252), vol_lookback=60,
                       target_vol=0.10, rebalance_days=5, max_weight=0.10, max_gross=5.0,
                       allow_short=True, vol_floor=0.03, cost_bps=3.0, ann=252,
                       book_vol_target=0.12, book_vol_max_leverage=4.0)


@register_sleeve("futures_trend")
def build_futures_trend(*, prices: Optional[pd.DataFrame] = None, **cfg_kw) -> Sleeve:
    """Cross-asset TSMOM on the liquid Norgate futures universe (~73 markets), long-short.
    HONEST: a real historical edge that has DECAYED to ~flat in the modern regime (post-2015
    Sharpe ~0.0); declared `diversifier`, ONE pre-registered config -> n_trials=1. Its value
    is as the crisis-convex partner in a trend+carry book, not standalone."""
    from app.strategy.tsmom import tsmom_backtest
    from app.research import futures_data as fd
    uni = fd.liquid_universe()
    if prices is None:
        prices = fd.synthetic_price_panel(uni)
    res = tsmom_backtest(prices, futures_trend_config(uni))
    return Sleeve(label="futures_trend", component_type="diversifier",
                  returns=res.returns.dropna(), spy_prices=None, periods_per_year=252,
                  n_trials_registered=1, registration_id="P4-2-FUT-TREND",
                  notes=f"TSMOM on {len(uni)} Norgate futures; long-short; live signal params; "
                        f"book_vol_target=0.12; difference-adj returns (ΔCCB/unadj, winsorized)")


@register_sleeve("futures_carry")
def build_futures_carry(**cfg_kw) -> Sleeve:
    """Cross-sectional CARRY on the liquid Norgate futures universe (term-structure slope).
    The robust, MODERN futures premium: standalone Sharpe ~0.67, positive in EVERY sub-period
    (post-2015 ~0.84) where trend is flat; corr-to-live-trend ~0.25; Track-B dSR +0.17.
    Declared `risk_premium`, ONE pre-registered config -> n_trials=1."""
    from app.research import futures_data as fd, futures_carry as fc, futures_roll as frl
    uni = fd.liquid_universe()
    rets = fd.returns_panel(uni)
    carry = fc.carry_panel(uni)
    # P0.2 honesty: include the roll TRANSACTION cost (3bps/side; round-trip per roll). The roll
    # YIELD is already in the back-adjusted returns -> this does not double-count the premium.
    roll_days = frl.roll_days_panel(uni, index=rets.index)
    cfg = fc.CarryConfig(roll_cost_bps=3.0, **cfg_kw)
    r = fc.carry_backtest(rets, carry, cfg, roll_days=roll_days)
    return Sleeve(label="futures_carry", component_type="risk_premium",
                  returns=r.dropna(), spy_prices=None, periods_per_year=252,
                  n_trials_registered=1, registration_id="P4-2-FUT-CARRY",
                  notes=f"cross-sectional term-structure carry on {len(uni)} Norgate futures; "
                        f"inverse-vol, book_vol_target=0.12, 3bps signal + 3bps/side roll")


@register_sleeve("futures_xsmom")
def build_futures_xsmom(**cfg_kw) -> Sleeve:
    """Cross-sectional 12-1 momentum on the liquid Norgate futures universe (long winners /
    short losers) — the P1.2 factor-zoo survivor: Sharpe ~0.56, modern-robust (post-2015 ~0.58),
    LOW corr-to-trend ~0.12 (it's RELATIVE momentum, distinct from the absolute TSMOM trend).
    Same XS engine + honest 3bps/side roll cost as carry. Declared diversifier, n_trials=1.
    (curve-momentum / value / skewness were tested + KILLED at the pre-registered sign.)"""
    from app.research import (futures_data as fd, futures_carry as fc,
                              futures_factors as ff, futures_roll as frl)
    uni = fd.liquid_universe()
    prices = fd.synthetic_price_panel(uni)
    rets = fd.returns_panel(uni)
    roll_days = frl.roll_days_panel(uni, index=rets.index)
    signal = ff.xs_momentum_signal(prices)
    cfg = fc.CarryConfig(roll_cost_bps=3.0, **cfg_kw)
    r = ff.xs_factor_backtest(rets, signal, cfg, roll_days=roll_days)
    return Sleeve(label="futures_xsmom", component_type="diversifier",
                  returns=r.dropna(), spy_prices=None, periods_per_year=252,
                  n_trials_registered=1, registration_id="P1-FUT-XSMOM",
                  notes=f"cross-sectional 12-1 momentum on {len(uni)} Norgate futures; XS engine; "
                        f"book_vol_target=0.12, 3bps signal + 3bps/side roll")


@register_sleeve("futures_book")
def build_futures_book(**cfg_kw) -> Sleeve:
    """P1.3 — the multi-factor CTA book: equal-weight(carry, XS-momentum). Both sub-sleeves are
    already book-vol-targeted (~0.12) + honestly roll-costed, so a 50/50 average is the sane-scale
    ensemble. The two individually-MARGINAL factors (residual-α t~1.7-1.8 vs trend) combine into a
    book that is a SIGNIFICANT diversifier to the live ETF-trend book (residual-α t~2.3, resid-Sharpe
    ~0.56, beta ~0.24): the composite clears a bar neither component did alone. Sharpe ~0.67
    (post-2015 ~0.83), corr(carry,xsmom) ~0.42. Declared diversifier, n_trials=1."""
    carry = build_futures_carry().returns
    xsmom = build_futures_xsmom().returns
    j = pd.concat([carry.rename("c"), xsmom.rename("x")], axis=1, join="inner").dropna()
    r = (0.5 * j["c"] + 0.5 * j["x"]).rename("futures_book")
    return Sleeve(label="futures_book", component_type="diversifier",
                  returns=r, spy_prices=None, periods_per_year=252,
                  n_trials_registered=1, registration_id="P1-3-FUT-BOOK",
                  notes="equal-weight(futures_carry, futures_xsmom); both vol-targeted + roll-costed; "
                        "significant diversifier to live trend (residual-alpha t~2.3)")


@register_sleeve("vix_vrp")
def build_vix_vrp(**cfg_kw) -> Sleeve:
    """P3.1 — variance-risk-premium via the VIX-futures curve: short the front VIX future (owned
    Norgate VX) when the curve is in CONTANGO (crash-governor gate: VIX<VIX3M), flat in
    backwardation. Reverses the earlier VRP park (it was killed on short options data + an alpha
    framing). Sharpe ~0.64, HAC-significant, positive every sub-period; SURVIVES Feb-2018 (−4.4%)
    + COVID-2020 (−4.8%) thanks to the gate. Declared risk_premium; n_trials=1. A real, crash-
    survivable premium but a MARGINAL diversifier (corr-to-trend ~0.46, residual-α t~1.5)."""
    from app.research import futures_data as fd, vix_vrp as vv
    vx = fd.true_returns("VX")
    vix, vix3m = fetch_vix_term()
    r = vv.vix_vrp_returns(vx, vix, vix3m, vv.VixVRPConfig(**cfg_kw))
    return Sleeve(label="vix_vrp", component_type="risk_premium",
                  returns=r.dropna(), spy_prices=None, periods_per_year=252,
                  n_trials_registered=1, registration_id="P3-VIX-VRP",
                  notes="short front VIX future in contango (gated by VIX<VIX3M), vol-targeted; "
                        "VRP roll-down; survives Feb-2018 + COVID via the gate")


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


@register_sleeve("credit_timing")
def build_credit_timing_sleeve(*, prices: Optional[pd.DataFrame] = None, **cfg_kw) -> Sleeve:
    """G3: ADDITIVE long-flat credit-timing equity sleeve (hold SPY when credit healthy, flat
    when HY < trailing MA). Declared `diversifier` (goes flat in stress -> meant to diverge);
    the corr<0.30 Track-B wall is the real test. One pre-registered config -> n_trials=1."""
    from app.strategy.credit_curve_governor import CreditGovernorConfig, credit_timing_returns
    cfg = CreditGovernorConfig(**cfg_kw)
    if prices is None:
        prices = fetch_universe_closes(["SPY", "HYG", "IEF"])
    res = credit_timing_returns(prices["SPY"], prices["HYG"], prices["IEF"], cfg)
    return Sleeve(label="credit_timing_SPY", component_type="diversifier",
                  returns=res, spy_prices=prices["SPY"],
                  n_trials_registered=1, registration_id="G3-CREDIT-TIMING",
                  notes="long SPY when HY>=MA(120), flat when credit-stressed (long-flat)")


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


# ──────────────────────────────────────────────────────────────────────────────────
# Overlays (G2) — aggregate short-interest de-risk overlay (slow, predictive)
# ──────────────────────────────────────────────────────────────────────────────────
@register_overlay("short_interest_governor")
def build_short_interest_governor(*, store=None, **cfg_kw):
    """Aggregate short-interest de-risk overlay (de-risk when the market-wide Short Interest
    Index is crowded). Loads the cached short-interest parquet unless `store` is provided."""
    from scripts.walkforward.sleeve_lab import Overlay
    from app.strategy.short_interest_governor import ShortInterestGovernorConfig, si_multiplier
    if store is None:
        from app.data.short_interest_provider import load_short_interest
        store = load_short_interest()
    mult = si_multiplier(store, ShortInterestGovernorConfig(**cfg_kw))
    return Overlay(label="short_interest_governor", multiplier=mult,
                   notes="de-risk when aggregate Short Interest Index (RRT) is crowded")


def run_short_interest(*, z_threshold: float = 1.0) -> dict:
    """G2 confirmatory eval: aggregate-SI overlay, MARGINAL to the VIX governor AND to the full
    {governor + credit-selective} stack, on the live trend book; + both-halves stability.
    Report-only. POWER CAVEAT: SI data starts 2017-12-29 (no GFC; ~3 in-window crises)."""
    from scripts.walkforward.sleeve_lab import evaluate_overlay_marginal, format_overlay_report
    base = live_trend_book_returns()
    gov = build_vix_term_governor()
    credit = build_overlay("credit_governor")
    si = build_overlay("short_interest_governor", z_threshold=z_threshold)
    print(f"[base] live trend book {base.index[0].date()} -> {base.index[-1].date()}")
    print(f"[si] overlay window {si.multiplier.index[0].date()} -> {si.multiplier.index[-1].date()} "
          f"derisk_days={float((si.multiplier < 1.0).mean()):.1%}")
    out = {}
    for tag, prior in (("vs governor", [gov]), ("vs governor+credit", [gov, credit])):
        rep = evaluate_overlay_marginal(si, base, prior_overlays=prior)
        print(f"\n[MARGINAL {tag}] short_interest_governor")
        print(format_overlay_report(rep))
        out[tag] = rep
    # both-halves on the marginal-vs-governor
    mser = si.multiplier
    mid = mser.index[len(mser) // 2]
    h1 = evaluate_overlay_marginal(si.__class__("si", mser[mser.index < mid]), base,
                                   prior_overlays=[gov])
    h2 = evaluate_overlay_marginal(si.__class__("si", mser[mser.index >= mid]), base,
                                   prior_overlays=[gov])
    print(f"\n  both-halves marginal d_max_dd: H1={h1.d_max_dd:+.4f} H2={h2.d_max_dd:+.4f} "
          f"-> both_improve={h1.d_max_dd > 0 and h2.d_max_dd > 0}")
    out["both_halves"] = (h1.d_max_dd, h2.d_max_dd)
    return out


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "governor":
        run_governor()
        return 0
    if argv and argv[0] == "credit_curve":
        run_credit_curve()
        return 0
    if argv and argv[0] == "short_interest":
        run_short_interest()
        return 0
    run_sleeves(argv or None)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    from dotenv import load_dotenv
    load_dotenv()
    raise SystemExit(main())
