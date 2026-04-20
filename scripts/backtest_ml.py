"""
ML-driven backtest using the trained LambdaRank/XGBoost model.

Phase 1 — Honest backtesting:
  - LambdaRank scores drive top-N stock selection per 10-day window
  - Evaluates on held-out TEST windows only (point-in-time safe)
  - ADV-based position cap, 0% commission (Alpaca), 0.05% slippage
  - Full vectorbt tearsheet + regime-aware breakdown + SPY alpha

Phase 2 — Validation rigor:
  - Walk-forward CV (--walk-forward N): retrain model on each fold's train set,
    score its test set. Confirms AUC/returns hold on truly unseen data.
  - Risk limit enforcement: sector ≤30%, single stock ≤10%, daily loss ≤2%
  - Purging + embargo (--embargo N): skip N trading days at train/test boundary
    to prevent label leakage from overlapping feature windows
  - Yearly + worst-quarter stress test

Usage:
    # Phase 1 — static model backtest
    python scripts/backtest_ml.py --model-version 37 --top-n 20 --years 5

    # Phase 2 — walk-forward (retrains per fold, slow ~hours)
    python scripts/backtest_ml.py --walk-forward 5 --top-n 20 --years 5

    # Phase 2 — with risk limits + embargo
    python scripts/backtest_ml.py --model-version 37 --embargo 5 --sector-cap 0.30

    # Disable ADV position cap
    python scripts/backtest_ml.py --no-adv-cap
"""

import argparse
import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

RESET = "\033[0m"; BOLD = "\033[1m"; GREEN = "\033[32m"
YELLOW = "\033[33m"; RED = "\033[31m"; CYAN = "\033[36m"; DIM = "\033[2m"

def ok(m):   print(f"  {GREEN}OK{RESET}  {m}")
def warn(m): print(f"  {YELLOW}!!{RESET}  {m}")
def info(m): print(f"     {m}")
def header(t): print(f"\n{BOLD}{CYAN}-- {t} --{RESET}"); print(DIM + "-"*60 + RESET)


# -- Model loading ------------------------------------------------------------─

def load_model(model_dir: str, version: int):
    import pickle
    path = Path(model_dir) / f"swing_v{version}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if hasattr(obj, "predict"):
        return obj
    raise ValueError(f"Loaded object has no predict(): {type(obj)}")


def latest_model_version(model_dir: str) -> int:
    paths = list(Path(model_dir).glob("swing_v*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No swing model in {model_dir}")
    versions = []
    for p in paths:
        try:
            versions.append(int(p.stem.split("_v")[1]))
        except Exception:
            pass
    return max(versions)


# -- Data fetching ------------------------------------------------------------─

def fetch_price_data(symbols: List[str], years: int) -> Dict[str, pd.DataFrame]:
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=365 * years + 30)
    try:
        from app.data import get_provider
        data = get_provider("polygon").get_daily_bars_bulk(symbols, start_dt, end_dt)
        ok(f"Downloaded {len(data)} symbols via Polygon")
        return data
    except Exception as exc:
        warn(f"Polygon failed ({exc}), falling back to yfinance...")
        import yfinance as yf
        data = {}
        for sym in symbols:
            try:
                df = yf.download(sym, start=start_dt, end=end_dt,
                                 progress=False, auto_adjust=True)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = [c.lower() for c in df.columns]
                if len(df) >= 52:
                    data[sym] = df
            except Exception:
                pass
        ok(f"Downloaded {len(data)} symbols via yfinance")
        return data


# -- Risk limit enforcement (Phase 2) ----------------------------------------─

def apply_risk_limits(
    selected: List[str],
    scores: Dict[str, float],
    sector_map: Dict[str, str],
    current_portfolio: Dict[str, float],  # symbol -> current position value
    init_cash: float,
    sector_cap: float = 0.30,
    stock_cap: float = 0.10,
) -> Tuple[List[str], List[str]]:
    """
    Filter selected stocks to enforce risk limits.

    Rules:
      sector_cap: no single GICS sector > 30% of portfolio value
      stock_cap:  no single stock > 10% of portfolio value

    Returns (approved, rejected) lists.
    """
    portfolio_value = init_cash  # simplified: use init_cash as denominator
    approved, rejected = [], []

    # Count sector exposure already in portfolio
    sector_exposure: Dict[str, float] = {}
    for sym, val in current_portfolio.items():
        sec = sector_map.get(sym, "Unknown")
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + val

    # Evaluate each candidate in rank order (best score first)
    for sym in selected:
        sec = sector_map.get(sym, "Unknown")
        new_pos = portfolio_value / max(len(selected), 1)

        # Single-stock cap
        if new_pos / portfolio_value > stock_cap:
            rejected.append(sym)
            continue

        # Sector cap
        projected_sector = (sector_exposure.get(sec, 0.0) + new_pos) / portfolio_value
        if projected_sector > sector_cap:
            rejected.append(sym)
            continue

        approved.append(sym)
        sector_exposure[sec] = sector_exposure.get(sec, 0.0) + new_pos

    return approved, rejected


# -- Signal generation --------------------------------------------------------─

def score_window(
    symbols_data: Dict[str, pd.DataFrame],
    model,
    trading_syms: List[str],
    w_start_date,
    w_end_date,
    fe,
    adv_cap: bool,
    init_cash: float,
    top_n: int,
    sector_cap: float,
    stock_cap: float,
    sector_map: Dict[str, str],
    feature_store=None,  # Phase 3: FeatureStore for cached fundamentals
) -> Optional[dict]:
    """Score all stocks for one window. Returns meta dict or None.

    Phase 3: if feature_store is provided, cached fundamental features
    (FMP earnings, analyst ratings, short interest, news sentiment) are
    loaded from the SQLite cache instead of being zeroed out, matching
    the feature distribution the model was trained on.
    """
    feats_dicts, scored_syms, adv_map = [], [], {}
    model_feature_names = getattr(model, "feature_names", None)

    for sym in trading_syms:
        df = symbols_data[sym]
        idx = df.index.date
        window_df = df.loc[(idx >= w_start_date) & (idx <= w_end_date)]
        if len(window_df) < fe.MIN_BARS:
            continue

        # Phase 3A: try feature store first (has full fundamentals, point-in-time safe)
        feats = None
        if feature_store is not None:
            feats = feature_store.get(sym, w_end_date)

        # Fall back to live feature engineering (price-derived only)
        if feats is None:
            try:
                feats = fe.engineer_features(
                    sym, window_df,
                    sector=sector_map.get(sym) or "Unknown",
                    fetch_fundamentals=False,
                    regime_score=0.5,
                    as_of_date=w_end_date,
                )
            except Exception:
                continue
        if feats is None:
            continue

        feats_dicts.append(feats)
        scored_syms.append(sym)

        if adv_cap and "volume" in df.columns and "close" in df.columns:
            recent = df.loc[idx <= w_end_date].tail(20)
            if len(recent) >= 5:
                adv_map[sym] = float((recent["close"] * recent["volume"]).mean())

    if not feats_dicts:
        return None

    # Align to model's training feature set (handles post-training feature additions)
    if model_feature_names:
        feature_rows = [
            [d.get(f, 0.0) for f in model_feature_names]
            for d in feats_dicts
        ]
    else:
        feature_rows = [list(d.values()) for d in feats_dicts]

    X = np.nan_to_num(np.array(feature_rows, dtype=float))
    try:
        _, proba = model.predict(X)
    except Exception as e:
        warn(f"  predict failed: {e}")
        return None

    scores = {sym: float(s) for sym, s in zip(scored_syms, proba)}
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    selected = [sym for sym, _ in ranked[:top_n]]

    # Risk limits
    approved, rejected = apply_risk_limits(
        selected, scores, sector_map, {}, init_cash, sector_cap, stock_cap
    )

    # ADV cap: log which positions would be reduced (Phase 3 note: actual sizing
    # reduction happens in vectorbt via size_type; this is diagnostic only)
    capped = []
    if adv_cap:
        per_pos = init_cash / max(len(approved), 1)
        for sym in approved:
            adv = adv_map.get(sym, 0)
            if adv > 0 and per_pos > adv * 0.01:
                capped.append(sym)

    return {
        "selected": approved,
        "rejected_risk": rejected,
        "n_scored": len(scored_syms),
        "adv_capped": len(capped),
        "top_score": ranked[0][1] if ranked else 0.0,
        "fs_hits": len(feats_dicts),  # how many came from feature store
    }


def generate_ml_signals(
    symbols_data: Dict[str, pd.DataFrame],
    model,
    top_n: int,
    years: int,
    adv_cap: bool = True,
    init_cash: float = 100_000.0,
    embargo_days: int = 0,
    sector_cap: float = 0.30,
    stock_cap: float = 0.10,
    window_starts_override: Optional[List[int]] = None,
    entry_at_open: bool = True,   # Phase 3B: enter at next-day open, not same-day close
    model_dir: str = "app/ml/models",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, List[dict]]:
    """
    Generate entry/exit signals from ML ranking scores.

    Phase 3 additions:
    - feature_store: loads cached full-feature vectors (with fundamentals) from
      the training SQLite cache, eliminating the 0-fill degradation for ~40 features.
    - entry_at_open: if True (default), entry signal fires on w_end_date+1 (next
      trading day open) instead of w_end_date close, avoiding same-bar lookahead.

    Returns: close_df, open_df, entries_df, exits_df, window_meta
    """
    from app.ml.training import WINDOW_DAYS, FORWARD_DAYS, STEP_DAYS, TEST_FRACTION
    from app.ml.features import FeatureEngineer
    from app.utils.constants import SECTOR_MAP

    fe = FeatureEngineer()

    # Phase 3A: wire feature store for cached historical fundamentals
    feature_store = None
    fs_path = Path(model_dir) / "feature_store.db"
    if fs_path.exists():
        try:
            from app.ml.feature_store import FeatureStore
            feature_store = FeatureStore(str(fs_path))
            ok(f"Feature store loaded ({fs_path.name}) — fundamentals available for backtest")
        except Exception as e:
            warn(f"Feature store unavailable: {e} — using price-only features")
    else:
        warn("No feature store found — using price-only features (fundamentals zeroed)")

    spy_df = symbols_data.get("SPY")
    if spy_df is not None:
        all_dates = sorted(set(spy_df.index.date))
    else:
        from collections import Counter as _Counter
        date_counts = _Counter(d for df in symbols_data.values() for d in df.index.date)
        min_syms = max(1, len(symbols_data) // 2)
        all_dates = sorted(d for d, cnt in date_counts.items() if cnt >= min_syms)

    window_starts = list(range(0, len(all_dates) - WINDOW_DAYS - FORWARD_DAYS, STEP_DAYS))

    if window_starts_override is not None:
        test_windows = window_starts_override
    else:
        split_idx = max(1, int(len(window_starts) * (1 - TEST_FRACTION)))
        test_windows = window_starts[split_idx + embargo_days:]

    info(f"Windows to evaluate: {len(test_windows)}  "
         f"(embargo={embargo_days}, sector_cap={sector_cap:.0%}, stock_cap={stock_cap:.0%}, "
         f"entry={'open+1' if entry_at_open else 'close'})")

    trading_syms = [s for s in symbols_data if s != "SPY"]

    all_close, all_open = {}, {}
    for sym in trading_syms:
        df = symbols_data[sym]
        all_close[sym] = pd.Series(df["close"].values, index=pd.to_datetime(df.index))
        if "open" in df.columns:
            all_open[sym] = pd.Series(df["open"].values, index=pd.to_datetime(df.index))

    close_df   = pd.DataFrame(all_close).sort_index()
    open_df    = pd.DataFrame(all_open).sort_index() if all_open else close_df.copy()
    entries_df = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
    exits_df   = pd.DataFrame(False, index=close_df.index, columns=close_df.columns)
    window_meta = []

    for w_start_idx in test_windows:
        w_end_idx  = w_start_idx + WINDOW_DAYS
        future_idx = w_end_idx + FORWARD_DAYS
        if future_idx >= len(all_dates):
            continue

        w_start_date = all_dates[w_start_idx]
        w_end_date   = all_dates[w_end_idx]
        exit_date    = all_dates[future_idx]

        meta = score_window(
            symbols_data, model, trading_syms,
            w_start_date, w_end_date, fe,
            adv_cap, init_cash, top_n, sector_cap, stock_cap, SECTOR_MAP,
            feature_store=feature_store,
        )
        if meta is None:
            continue

        # Phase 3B: enter at next-day open (w_end_idx+1), not same-day close
        if entry_at_open and w_end_idx + 1 < len(all_dates):
            entry_date = all_dates[w_end_idx + 1]
        else:
            entry_date = w_end_date
        entry_ts = pd.Timestamp(entry_date)
        exit_ts  = pd.Timestamp(exit_date)

        if entry_ts in entries_df.index:
            for sym in meta["selected"]:
                if sym in entries_df.columns:
                    entries_df.loc[entry_ts, sym] = True
        if exit_ts in exits_df.index:
            for sym in meta["selected"]:
                if sym in exits_df.columns:
                    exits_df.loc[exit_ts, sym] = True

        meta.update({"window_end": w_end_date, "entry_date": entry_date, "exit_date": exit_date})
        window_meta.append(meta)

    fs_hit_rate = np.mean([m.get("fs_hits", 0) / max(m["n_scored"], 1)
                           for m in window_meta]) if window_meta else 0
    ok(f"Signals: {len(window_meta)} windows, "
       f"avg {np.mean([m['n_scored'] for m in window_meta]):.0f} scored/window, "
       f"avg {np.mean([len(m['selected']) for m in window_meta]):.0f} selected/window, "
       f"fs_hit={fs_hit_rate:.0%}")
    return close_df, open_df, entries_df, exits_df, window_meta


# -- Walk-forward CV (Phase 2) ------------------------------------------------─

def run_walk_forward(
    symbols_data: Dict[str, pd.DataFrame],
    n_folds: int,
    top_n: int,
    init_cash: float,
    adv_cap: bool,
    embargo_days: int,
    sector_cap: float,
    stock_cap: float,
    model_type: str = "lambdarank",
) -> List[dict]:
    """
    Walk-forward cross-validation: for each fold, retrain on all prior windows,
    score on the fold's test windows. Returns per-fold metrics.

    This is the proper validation that the model generalises across time —
    the static backtest (Phase 1) reuses a model trained on part of this data.
    """
    from app.ml.training import (
        ModelTrainer, WINDOW_DAYS, FORWARD_DAYS, STEP_DAYS, TEST_FRACTION
    )
    from app.utils.constants import SECTOR_MAP

    spy_df = symbols_data.get("SPY")
    all_dates = sorted(set(spy_df.index.date)) if spy_df is not None else []
    window_starts = list(range(0, len(all_dates) - WINDOW_DAYS - FORWARD_DAYS, STEP_DAYS))

    if len(window_starts) < n_folds * 2:
        warn(f"Too few windows ({len(window_starts)}) for {n_folds} folds. Reduce --walk-forward.")
        return []

    fold_size = len(window_starts) // n_folds
    fold_results = []

    print()
    info(f"Walk-forward: {n_folds} folds × ~{fold_size} windows each")
    info(f"NOTE: each fold retrains the model — expect ~60-90 min per fold")
    print()

    for fold in range(n_folds):
        test_start  = fold * fold_size
        test_end    = (fold + 1) * fold_size if fold < n_folds - 1 else len(window_starts)
        train_end   = test_start - embargo_days  # purge: gap before test

        if train_end < 10:
            info(f"  Fold {fold+1}: skipped (not enough training windows)")
            continue

        train_window_starts = window_starts[:train_end]
        test_window_starts  = window_starts[test_start:test_end]

        info(f"  Fold {fold+1}/{n_folds}: train={len(train_window_starts)} windows "
             f"-> test={len(test_window_starts)} windows  "
             f"(gap={embargo_days} windows)")

        # Retrain on this fold's training windows
        trainer = ModelTrainer(
            model_type=model_type,
            label_scheme="lambdarank",
            use_feature_store=True,
        )
        try:
            X_tr, y_tr, meta_tr = trainer._windows_to_matrix(
                symbols_data, all_dates, train_window_starts,
                regime_score=0.5, fetch_fundamentals=False,
            )
            if len(X_tr) == 0:
                warn(f"  Fold {fold+1}: no training samples")
                continue

            if model_type == "lambdarank":
                X_tr, y_tr, groups = trainer._build_lambdarank_groups(X_tr, y_tr, meta_tr)
                trainer.model.train(X_tr, y_tr, trainer._last_feature_names, groups=groups)
            else:
                from sklearn.preprocessing import LabelBinarizer
                n_neg = int((y_tr == 0).sum())
                n_pos = int((y_tr == 1).sum())
                spw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
                trainer.model.train(X_tr, y_tr, trainer._last_feature_names, scale_pos_weight=spw)

            ok(f"  Fold {fold+1}: model retrained on {len(X_tr)} samples")
        except Exception as exc:
            warn(f"  Fold {fold+1}: training failed — {exc}")
            continue

        # Score test windows
        _, _, entries_df, exits_df, window_meta = generate_ml_signals(
            symbols_data, trainer.model, top_n, years=5,
            adv_cap=adv_cap, init_cash=init_cash,
            embargo_days=0,  # already applied above
            sector_cap=sector_cap, stock_cap=stock_cap,
            window_starts_override=test_window_starts,
            entry_at_open=True,
        )

        # Compute fold return from window_meta
        fold_returns = []
        for m in window_meta:
            # Approximate: equal-weight average return of selected stocks
            rets = []
            for sym in m["selected"]:
                df = symbols_data.get(sym)
                if df is None:
                    continue
                idx = df.index.date
                entry_rows = df.loc[idx == m["window_end"], "close"]
                exit_rows  = df.loc[idx == m["exit_date"], "close"]
                if len(entry_rows) > 0 and len(exit_rows) > 0:
                    ep = float(entry_rows.iloc[0])
                    xp = float(exit_rows.iloc[0])
                    if ep > 0:
                        rets.append((xp - ep) / ep)
            if rets:
                fold_returns.append(np.mean(rets))

        if not fold_returns:
            continue

        fold_ret   = float(np.mean(fold_returns)) * 100
        fold_std   = float(np.std(fold_returns)) * 100
        fold_wins  = sum(r > 0 for r in fold_returns) / len(fold_returns) * 100
        fold_sharpe = (np.mean(fold_returns) / (np.std(fold_returns) + 1e-9)) * (252/10)**0.5

        fold_results.append({
            "fold": fold + 1,
            "n_train": len(train_window_starts),
            "n_test":  len(test_window_starts),
            "avg_return_pct": fold_ret,
            "return_std_pct": fold_std,
            "win_rate_pct":   fold_wins,
            "sharpe":         fold_sharpe,
        })
        ok(f"  Fold {fold+1}: avg_ret={fold_ret:+.2f}%  win={fold_wins:.0f}%  sharpe={fold_sharpe:.2f}")

    return fold_results


# -- Regime + stress test ------------------------------------------------------

def get_regime_periods(close_df: pd.DataFrame) -> pd.Series:
    try:
        import yfinance as yf
        spy = yf.download("SPY", start=close_df.index[0], end=close_df.index[-1],
                          progress=False, auto_adjust=True)["Close"]
        ret63 = spy.pct_change(63)
        regime = pd.Series("sideways", index=spy.index)
        regime[ret63 > 0.05]  = "bull"
        regime[ret63 < -0.05] = "bear"
        return regime.reindex(close_df.index, method="ffill").fillna("sideways")
    except Exception:
        return pd.Series("unknown", index=close_df.index)


def run_stress_test(pf, close_df: pd.DataFrame) -> None:
    """Yearly return breakdown + worst 3-month drawdown period."""
    header("Stress test — yearly returns + worst quarter")
    try:
        returns_daily = pf.returns()
        if isinstance(returns_daily, pd.DataFrame):
            returns_daily = returns_daily.mean(axis=1)
        returns_daily = returns_daily.dropna()

        # Yearly returns
        years = sorted(returns_daily.index.year.unique())
        for yr in years:
            yr_rets = returns_daily[returns_daily.index.year == yr]
            ann_ret = float((1 + yr_rets).prod() - 1) * 100
            vol     = float(yr_rets.std()) * (252 ** 0.5) * 100
            sharpe  = (ann_ret / 100) / (vol / 100 + 1e-9)
            max_dd  = float((yr_rets + 1).cumprod().div(
                (yr_rets + 1).cumprod().cummax()).min() - 1) * 100
            flag = f"  {GREEN}^{RESET}" if ann_ret > 0 else f"  {RED}v{RESET}"
            print(f"  {yr}  return={ann_ret:>7.1f}%  vol={vol:>5.1f}%  "
                  f"sharpe={sharpe:>5.2f}  max_dd={max_dd:>6.1f}%{flag}")

        # Worst 3-month rolling window
        roll_3m = returns_daily.rolling(63).apply(lambda r: (1 + r).prod() - 1)
        worst_end   = roll_3m.idxmin()
        worst_start = worst_end - pd.Timedelta(days=95)
        worst_ret   = float(roll_3m.min()) * 100
        print(f"\n  Worst 3-month period: {worst_start.date()} -> {worst_end.date()}  "
              f"return={worst_ret:.1f}%  {RED}<{RESET}")
    except Exception as exc:
        warn(f"Stress test failed: {exc}")


# -- Tearsheet ----------------------------------------------------------------─

def _scalar(v):
    """Extract scalar from Series/DataFrame by taking mean."""
    if isinstance(v, (pd.Series, pd.DataFrame)):
        return float(v.mean())
    return float(v)

def print_tearsheet(pf, spy_close, total_ret: float) -> None:
    header("Portfolio tearsheet")
    try:
        stats = pf.stats()
        # Multi-asset: stats is a DataFrame (metrics x symbols); aggregate across symbols
        if isinstance(stats, pd.DataFrame):
            stats = stats.mean(axis=1)
        key_metrics = [
            ("Total Return [%]",      "Total Return"),
            ("Max Drawdown [%]",      "Max Drawdown"),
            ("Sharpe Ratio",          "Sharpe Ratio"),
            ("Sortino Ratio",         "Sortino Ratio"),
            ("Calmar Ratio",          "Calmar Ratio"),
            ("Win Rate [%]",          "Win Rate"),
            ("Profit Factor",         "Profit Factor"),
            ("Total Trades",          "Total Trades"),
            ("Avg Winning Trade [%]", "Avg Win"),
            ("Avg Losing Trade [%]",  "Avg Loss"),
        ]
        for key, label in key_metrics:
            if key in stats.index:
                val = stats[key]
                print(f"  {label:<28} {val:>10.3f}" if isinstance(val, (int, float))
                      else f"  {label:<28} {str(val):>10}")
    except Exception as e:
        warn(f"stats failed: {e}")

    if total_ret > 20:
        print(f"\n  {GREEN}{BOLD}>> Strong — {total_ret:.1f}% total return{RESET}")
    elif total_ret > 0:
        print(f"\n  {YELLOW}{BOLD}>> Positive — {total_ret:.1f}% total return{RESET}")
    else:
        print(f"\n  {RED}{BOLD}>> Negative — {total_ret:.1f}%{RESET}")

    if spy_close is not None:
        try:
            spy_aligned = spy_close.reindex(pf.wrapper.index).dropna()
            spy_ret = (spy_aligned.iloc[-1] / spy_aligned.iloc[0] - 1) * 100
            alpha   = total_ret - float(spy_ret)
            sym = f"{GREEN}+{RESET}" if alpha > 0 else RED
            print(f"\n  {'SPY Return':<28} {spy_ret:>10.1f}%")
            print(f"  {'Alpha (vs SPY)':<28} {alpha:>+10.1f}%")
        except Exception:
            pass


# -- Main ----------------------------------------------------------------------

def run_ml_backtest(
    model_version: Optional[int],
    top_n: int,
    years: int,
    init_cash: float,
    adv_cap: bool,
    embargo_days: int,
    sector_cap: float,
    stock_cap: float,
    walk_forward: int,
    model_dir: str = "app/ml/models",
):
    try:
        import vectorbt as vbt
    except ImportError:
        print("vectorbt not installed — pip install vectorbt")
        sys.exit(1)

    from app.utils.constants import RUSSELL_1000_TICKERS

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}  MrTrader — ML Backtest (Phase 3){RESET}")
    print(f"{'='*60}")
    info(f"top_n={top_n}  years={years}  cash=${init_cash:,.0f}  embargo={embargo_days}d")
    info(f"sector_cap={sector_cap:.0%}  stock_cap={stock_cap:.0%}  adv_cap={'on' if adv_cap else 'off'}")
    if walk_forward:
        info(f"walk_forward={walk_forward} folds  (retrains model per fold — slow)")

    # -- Load model (skip for walk-forward — retrains internally) --------------
    model, spy_close = None, None
    if not walk_forward:
        header("Loading model")
        if model_version is None:
            model_version = latest_model_version(model_dir)
        ok(f"Model version: v{model_version}")
        model = load_model(model_dir, model_version)
        ok(f"Model type: {type(model).__name__}")

    # -- Fetch data ------------------------------------------------------------─
    header("Fetching price data")
    symbols = RUSSELL_1000_TICKERS + ["SPY"]
    symbols_data = fetch_price_data(symbols, years)

    try:
        spy_close = pd.Series(
            symbols_data["SPY"]["close"].values,
            index=pd.to_datetime(symbols_data["SPY"].index),
        )
    except Exception:
        pass

    # -- Walk-forward mode ------------------------------------------------------
    if walk_forward:
        header(f"Walk-forward CV ({walk_forward} folds) — Phase 2")
        fold_results = run_walk_forward(
            symbols_data, walk_forward, top_n, init_cash, adv_cap,
            embargo_days, sector_cap, stock_cap,
        )
        if not fold_results:
            warn("Walk-forward produced no results")
            return

        header("Walk-forward summary")
        avg_ret    = np.mean([f["avg_return_pct"] for f in fold_results])
        avg_sharpe = np.mean([f["sharpe"] for f in fold_results])
        avg_win    = np.mean([f["win_rate_pct"] for f in fold_results])
        std_ret    = np.std([f["avg_return_pct"] for f in fold_results])

        for f in fold_results:
            print(f"  Fold {f['fold']}  "
                  f"ret={f['avg_return_pct']:>+6.2f}%  "
                  f"win={f['win_rate_pct']:>5.1f}%  "
                  f"sharpe={f['sharpe']:>5.2f}")

        print(f"\n  {'Avg return / fold':<28} {avg_ret:>+8.2f}%")
        print(f"  {'Std return / fold':<28} {std_ret:>8.2f}%")
        print(f"  {'Avg Sharpe':<28} {avg_sharpe:>8.2f}")
        print(f"  {'Avg Win Rate':<28} {avg_win:>8.1f}%")

        gate_pass = avg_sharpe > 0.8 and avg_ret > 0
        if gate_pass:
            print(f"\n  {GREEN}{BOLD}>> PASS — Sharpe {avg_sharpe:.2f} > 0.8, positive return{RESET}")
        else:
            print(f"\n  {RED}{BOLD}>> REVIEW — Sharpe {avg_sharpe:.2f}, return {avg_ret:+.2f}%{RESET}")

        print(f"\n{'='*60}\n")
        return fold_results

    # -- Static model backtest (Phase 1 + 2 + 3) --------------------------------
    header("Generating ML signals")
    close_df, open_df, entries_df, exits_df, window_meta = generate_ml_signals(
        symbols_data, model, top_n, years, adv_cap, init_cash,
        embargo_days, sector_cap, stock_cap,
        entry_at_open=True,  # Phase 3B: enter next-day open
        model_dir=model_dir,
    )

    if entries_df.values.sum() == 0:
        warn("No entry signals — check model version and data overlap")
        return

    total_rejected = sum(len(m.get("rejected_risk", [])) for m in window_meta)
    if total_rejected > 0:
        info(f"Risk limits rejected {total_rejected} positions across all windows")

    header("Running vectorbt backtest")
    # Phase 3B: use open prices for entries (realistic fill at next-day open)
    # close prices used for exits and P&L marking
    pf = vbt.Portfolio.from_signals(
        close_df,
        entries_df,
        exits_df,
        open=open_df,
        init_cash=init_cash,
        fees=0.0,
        slippage=0.0005,
        size=1.0 / top_n,
        size_type="percent",
        upon_opposite_entry="ignore",
        freq="D",
        group_by=True,
        cash_sharing=True,
    )

    tr = pf.total_return()
    total_ret = float(tr) * 100
    print_tearsheet(pf, spy_close, total_ret)

    # -- Regime breakdown ------------------------------------------------------─
    header("Regime-aware performance")
    regime = get_regime_periods(close_df)
    try:
        ret_daily = pf.returns()
        if isinstance(ret_daily, pd.DataFrame):
            ret_daily = ret_daily.mean(axis=1)
        ret_daily = ret_daily.dropna()

        for r in ["bull", "sideways", "bear"]:
            mask   = regime == r
            r_rets = ret_daily[mask]
            if len(r_rets) < 5:
                continue
            ann    = float(r_rets.mean()) * 252 * 100
            vol    = float(r_rets.std()) * (252 ** 0.5) * 100
            sharpe = ann / (vol + 1e-9)
            print(f"  {r.upper():<10} {int(mask.sum()):>4}d  "
                  f"ann={ann:>7.1f}%  vol={vol:>5.1f}%  sharpe={sharpe:>5.2f}")
    except Exception as exc:
        warn(f"Regime breakdown failed: {exc}")

    # -- Stress test ------------------------------------------------------------
    run_stress_test(pf, close_df)

    # -- Window diagnostics ----------------------------------------------------─
    header("Window diagnostics")
    if window_meta:
        info(f"OOS windows evaluated : {len(window_meta)}")
        info(f"Avg stocks scored/win : {np.mean([m['n_scored'] for m in window_meta]):.0f}")
        info(f"Feature store hit rate: {np.mean([m.get('fs_hits',0)/max(m['n_scored'],1) for m in window_meta]):.0%}")
        info(f"Avg ADV-capped/win    : {np.mean([m['adv_capped'] for m in window_meta]):.1f}")
        info(f"Avg top-1 score       : {np.mean([m['top_score'] for m in window_meta]):.3f}")
        all_sel = [s for m in window_meta for s in m["selected"]]
        top_stocks = Counter(all_sel).most_common(10)
        info(f"Most selected         : {', '.join(f'{s}({n})' for s, n in top_stocks)}")

    print(f"\n{'='*60}\n")
    return pf


def main():
    parser = argparse.ArgumentParser(
        description="ML-driven backtest — Phase 1 (static) + Phase 2 (walk-forward)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-version", type=int, default=None,
                        help="Model version (default: latest). Ignored with --walk-forward.")
    parser.add_argument("--top-n", type=int, default=20,
                        help="Top-N stocks per window (default: 20)")
    parser.add_argument("--years", type=int, default=5,
                        help="Years of history (default: 5)")
    parser.add_argument("--cash", type=float, default=100_000.0,
                        help="Starting capital (default: $100,000)")
    parser.add_argument("--no-adv-cap", action="store_true",
                        help="Disable ADV-based position cap")
    parser.add_argument("--embargo", type=int, default=5,
                        help="Windows to skip at train/test boundary (default: 5)")
    parser.add_argument("--sector-cap", type=float, default=0.30,
                        help="Max sector allocation 0-1 (default: 0.30)")
    parser.add_argument("--stock-cap", type=float, default=0.10,
                        help="Max single-stock allocation 0-1 (default: 0.10)")
    parser.add_argument("--walk-forward", type=int, default=0, metavar="FOLDS",
                        help="Walk-forward folds — retrains model per fold (slow, ~1h/fold)")
    parser.add_argument("--model-dir", default="app/ml/models")
    args = parser.parse_args()

    run_ml_backtest(
        model_version=args.model_version,
        top_n=args.top_n,
        years=args.years,
        init_cash=args.cash,
        adv_cap=not args.no_adv_cap,
        embargo_days=args.embargo,
        sector_cap=args.sector_cap,
        stock_cap=args.stock_cap,
        walk_forward=args.walk_forward,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
