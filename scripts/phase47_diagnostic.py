"""
Phase 47 — Diagnostic Dive (Phase 0)

Runs IntradayAgentSimulator on v22's 3 walk-forward test folds and produces
7 diagnostic cuts to answer three structural questions before any retrain:

  1. Momentum or reversion? (cut 1)
  2. True realized R:R? (cut 2)
  3. Meta-model marginal Sharpe? (cut 5)

Output: docs/phase47/diagnostic_report.md
"""
import os
import sys
import pickle
from datetime import date, timedelta, datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DOCS_DIR = ROOT / "docs" / "phase47"
MODEL_DIR = ROOT / "app" / "ml" / "models"
DOCS_DIR.mkdir(parents=True, exist_ok=True)

# v22 walk-forward fold test windows (must match walkforward_tier3.py)
FOLDS = [
    (date(2024, 10, 14), date(2025, 4, 15),  1),
    (date(2025, 4, 16),  date(2025, 10, 15), 2),
    (date(2025, 10, 16), date(2026, 4, 17),  3),
]


def _ok(msg):   print(f"  \033[32mOK\033[0m  {msg}")
def _warn(msg): print(f"  \033[33mWARN\033[0m  {msg}")
def _info(msg): print(f"  {msg}")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_model():
    files = sorted(MODEL_DIR.glob("intraday_v*.pkl"),
                   key=lambda p: int(p.stem.split("_v")[-1]))
    if not files:
        raise RuntimeError("No intraday model found")
    path = files[-1]
    version = int(path.stem.split("_v")[-1])
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not hasattr(obj, "is_trained"):
        from app.ml.model import PortfolioSelectorModel
        m = PortfolioSelectorModel(model_type="xgboost")
        m.load(str(path.parent), version, model_name="intraday")
        obj = m
    _ok(f"Loaded intraday model v{version}")
    return obj, version


def load_meta_model():
    try:
        from app.ml.meta_model import MetaLabelModel
        meta = MetaLabelModel.load(str(MODEL_DIR), version=1, model_type="intraday")
        _ok("Loaded intraday MetaLabelModel v1")
        return meta
    except Exception as e:
        _warn(f"Could not load meta model: {e}")
        return None


def load_data():
    from app.data.intraday_cache import load_many, available_symbols
    cache_syms = set(available_symbols())
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=740)
    symbols_data = load_many(list(cache_syms), start=start_date, end=end_date)
    spy_data = symbols_data.get("SPY")
    _ok(f"Loaded {len(symbols_data)} symbols")
    return symbols_data, spy_data


# ── Run simulator on one fold ─────────────────────────────────────────────────

def run_fold(model, symbols_data, spy_data, start, end, fold_id,
             meta_model=None, pm_vix=25.0, pm_spy_ma=20):
    from app.backtesting.intraday_agent_simulator import IntradayAgentSimulator
    sim = IntradayAgentSimulator(
        model=model,
        meta_model=meta_model,
        pm_abstention_vix=pm_vix,
        pm_abstention_spy_ma_days=pm_spy_ma,
    )
    result = sim.run(symbols_data, spy_data=spy_data, start_date=start, end_date=end)
    trades = result.trades if result else []
    _ok(f"Fold {fold_id}: {len(trades)} trades, Sharpe {result.sharpe_ratio:.3f}")
    return trades, result.sharpe_ratio


# ── Build trade DataFrame ─────────────────────────────────────────────────────

def build_trade_df(all_fold_trades):
    """
    Merge trades from all folds into one DataFrame, enrich with ORB features.
    """
    from app.data.intraday_cache import load_many, available_symbols
    from app.ml.intraday_features import compute_intraday_features
    from app.ml.intraday_training import _index_by_day
    from app.backtesting.intraday_agent_simulator import FEATURE_BARS

    # Reload data for feature extraction
    cache_syms = set(available_symbols())
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=740)
    symbols_data = load_many(list(cache_syms), start=start_date, end=end_date)
    spy_data = symbols_data.get("SPY")
    spy_by_day = _index_by_day(spy_data) if spy_data is not None else {}

    rows = []
    for fold_id, trades in all_fold_trades:
        for t in trades:
            sym = t.symbol
            df = symbols_data.get(sym)
            if df is None:
                continue
            entry_date = t.entry_date if isinstance(t.entry_date, date) else t.entry_date.date()
            df_idx = pd.DatetimeIndex(df.index)
            day_mask = df_idx.normalize().date == entry_date
            day_bars = df.loc[day_mask]

            if len(day_bars) < FEATURE_BARS + 1:
                continue

            feat_bars = day_bars.iloc[:FEATURE_BARS]

            # Prior day OHLC
            prior_mask = df_idx.normalize().date < entry_date
            prior_close = prior_high = prior_low = None
            if prior_mask.any():
                prior_bars = df.loc[prior_mask]
                last_day = prior_bars.index.normalize().date[-1]
                last_day_bars = prior_bars.loc[prior_bars.index.normalize().date == last_day]
                if len(last_day_bars) > 0:
                    prior_close = float(last_day_bars["close"].iloc[-1])
                    prior_high = float(last_day_bars["high"].max())
                    prior_low = float(last_day_bars["low"].min())

            try:
                feats = compute_intraday_features(
                    feat_bars, spy_by_day.get(entry_date),
                    prior_close, prior_day_high=prior_high, prior_day_low=prior_low,
                )
            except Exception:
                feats = {}

            # Future bars for MFE/MAE
            from app.backtesting.intraday_agent_simulator import HOLD_BARS
            future = day_bars.iloc[FEATURE_BARS: FEATURE_BARS + HOLD_BARS]
            entry_price = float(feat_bars["close"].iloc[-1])

            mfe = mae = 0.0
            if len(future) > 0 and entry_price > 0:
                highs = future["high"].values
                lows = future["low"].values
                mfe = float((highs.max() - entry_price) / entry_price)
                mae = float((entry_price - lows.min()) / entry_price)

            # Prior-day range for R:R context
            prior_range_pct = 0.0
            if prior_close and prior_high and prior_low and prior_close > 0:
                prior_range_pct = (prior_high - prior_low) / prior_close

            rows.append({
                "fold": fold_id,
                "symbol": sym,
                "entry_date": entry_date,
                "entry_price": entry_price,
                "exit_price": t.exit_price,
                "pnl_pct": t.pnl_pct,
                "exit_reason": t.exit_reason,
                "hold_bars": t.hold_bars,
                "winner": 1 if t.pnl_pct > 0 else 0,
                "mfe": mfe,
                "mae": mae,
                "prior_range_pct": prior_range_pct,
                # ORB / momentum features
                "orb_breakout": feats.get("orb_breakout", 0.0),
                "orb_position": feats.get("orb_position", 0.5),
                "orb_direction_strength": feats.get("orb_direction_strength", 0.0),
                "price_momentum": feats.get("price_momentum", 0.0),
                "volume_surge": feats.get("volume_surge", 1.0),
                "whale_candle": feats.get("whale_candle", 0.0),
                "spy_session_return": feats.get("spy_session_return", 0.0),
                "cs_rank_momentum": feats.get("cs_rank_momentum", 0.5),
                "cs_rank_volume": feats.get("cs_rank_volume", 0.5),
                "atr_norm": feats.get("atr_norm", 0.0),
                "vwap_distance": feats.get("vwap_distance", 0.0),
                "rsi_14": feats.get("rsi_14", 50.0),
            })

    df = pd.DataFrame(rows)
    _ok(f"Built trade DataFrame: {len(df)} rows")
    return df


# ── Diagnostic cuts ───────────────────────────────────────────────────────────

def cut1_momentum_vs_reversion(df):
    """Win rate by ORB direction alignment."""
    lines = ["### Cut 1 — Momentum vs Reversion (ORB Direction Alignment)\n"]
    lines.append("Does the model win more when entry *aligns* with ORB breakout direction (momentum)")
    lines.append("or when it *opposes* it (reversion)?\n")
    lines.append("- `orb_breakout = +1`: price above ORB high (bullish breakout)")
    lines.append("- `orb_breakout = 0`: price inside ORB (no breakout)")
    lines.append("- `orb_breakout = -1`: price below ORB low (bearish — rare in long-only)\n")

    grp = df.groupby("orb_breakout").agg(
        n=("winner", "count"),
        win_rate=("winner", "mean"),
        avg_pnl=("pnl_pct", "mean"),
        avg_mfe=("mfe", "mean"),
        avg_mae=("mae", "mean"),
    ).round(4)
    lines.append(grp.to_markdown())

    # Also check: within breakout trades, does momentum (price_momentum > 0) help?
    lines.append("\n**Win rate by price_momentum quintile (across all trades):**\n")
    df2 = df.copy()
    df2["mom_quintile"] = pd.qcut(df2["price_momentum"], q=5, labels=["Q1(low)", "Q2", "Q3", "Q4", "Q5(high)"], duplicates="drop")
    grp2 = df2.groupby("mom_quintile", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append(grp2.to_markdown())

    verdict = ""
    orb_plus_wr = grp.loc[1.0, "win_rate"] if 1.0 in grp.index else None
    orb_zero_wr = grp.loc[0.0, "win_rate"] if 0.0 in grp.index else None
    orb_minus_wr = grp.loc[-1.0, "win_rate"] if -1.0 in grp.index else None

    if orb_plus_wr and orb_zero_wr:
        if orb_zero_wr > orb_plus_wr + 0.03:
            verdict = "⚠️  **REVERSION-FLAVORED**: Inside-ORB entries win more than breakout entries."
        elif orb_plus_wr > orb_zero_wr + 0.03:
            verdict = "✅  **MOMENTUM-FLAVORED**: ORB breakout entries win more than inside-ORB entries."
        else:
            verdict = "➡️  **MIXED**: No strong directional bias detected."
    lines.append(f"\n**Verdict:** {verdict}\n")
    return "\n".join(lines), verdict


def cut2_exit_type_breakdown(df):
    """True realized R:R from exit mix."""
    lines = ["### Cut 2 — Exit Type Breakdown & True Realized R:R\n"]

    grp = df.groupby("exit_reason").agg(
        n=("pnl_pct", "count"),
        pct=("pnl_pct", "count"),
        avg_pnl=("pnl_pct", "mean"),
        avg_hold=("hold_bars", "mean"),
        win_rate=("winner", "mean"),
    )
    grp["pct"] = (grp["pct"] / len(df) * 100).round(1)
    grp = grp.round(4)
    lines.append(grp.to_markdown())

    # Realized R:R
    winners = df[df["winner"] == 1]["pnl_pct"]
    losers = df[df["winner"] == 0]["pnl_pct"]
    avg_win = winners.mean() if len(winners) > 0 else 0
    avg_loss = abs(losers.mean()) if len(losers) > 0 else 1e-9
    realized_rr = avg_win / avg_loss

    target_exits = df[df["exit_reason"] == "TARGET"]
    stop_exits = df[df["exit_reason"] == "STOP"]
    time_exits = df[df["exit_reason"].isin(["TIME_EXIT", "FORCE_CLOSE"])]

    lines.append(f"\n**Avg winning pnl_pct:** {avg_win:.4f}")
    lines.append(f"**Avg losing pnl_pct:** {-avg_loss:.4f}")
    lines.append(f"**Realized R:R:** {realized_rr:.2f}:1")
    lines.append(f"**Target exits:** {len(target_exits)} ({100*len(target_exits)/max(len(df),1):.1f}%)")
    lines.append(f"**Stop exits:** {len(stop_exits)} ({100*len(stop_exits)/max(len(df),1):.1f}%)")
    lines.append(f"**Time exits:** {len(time_exits)} ({100*len(time_exits)/max(len(df),1):.1f}%)")

    # Time exit breakdown: positive vs negative
    time_pos = time_exits[time_exits["pnl_pct"] > 0]
    time_neg = time_exits[time_exits["pnl_pct"] <= 0]
    lines.append(f"  - Time exits positive: {len(time_pos)} ({100*len(time_pos)/max(len(time_exits),1):.1f}%)")
    lines.append(f"  - Time exits negative: {len(time_neg)} ({100*len(time_neg)/max(len(time_exits),1):.1f}%)")

    verdict = ""
    if realized_rr < 1.5:
        verdict = f"⚠️  **TRUE R:R IS {realized_rr:.2f}:1** — much lower than stated 2:1. Stop/target compression (Phase 3) is mandatory."
    elif realized_rr < 1.8:
        verdict = f"⚠️  **TRUE R:R IS {realized_rr:.2f}:1** — below stated 2:1. Phase 3 (compression) likely beneficial."
    else:
        verdict = f"✅  **TRUE R:R IS {realized_rr:.2f}:1** — reasonably close to stated 2:1."

    pct_time = 100 * len(time_exits) / max(len(df), 1)
    if pct_time > 40:
        verdict += f"\n\n⚠️  **{pct_time:.0f}% TIME EXITS** — target is too far for 2-hour hold. Compression strongly indicated."

    lines.append(f"\n**Verdict:** {verdict}\n")
    return "\n".join(lines), realized_rr


def cut3_regime_gates(df):
    """Win rate by VIX bin and SPY vs MA20."""
    lines = ["### Cut 3 — Regime Gate Attribution (VIX & SPY vs MA20)\n"]
    lines.append("*Note: PM abstention gate (VIX≥25, SPY<MA20) should have removed the worst regime days.")
    lines.append("This cut checks win rates across the regimes that passed the gate.*\n")

    # Proxy VIX from atr_norm quintiles (we don't store VIX per trade without yfinance)
    df2 = df.copy()
    df2["atr_quintile"] = pd.qcut(df2["atr_norm"], q=5, labels=["Low vol", "Q2", "Q3", "Q4", "High vol"], duplicates="drop")
    grp = df2.groupby("atr_quintile", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("**Win rate by ATR volatility quintile (proxy for VIX regime):**\n")
    lines.append(grp.to_markdown())

    # SPY return direction as market regime proxy
    df2["spy_regime"] = df2["spy_session_return"].apply(
        lambda x: "SPY up" if x > 0.002 else ("SPY flat" if x > -0.002 else "SPY down")
    )
    grp2 = df2.groupby("spy_regime").agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("\n**Win rate by SPY session direction:**\n")
    lines.append(grp2.to_markdown())

    # Per-fold
    grp3 = df.groupby("fold").agg(
        n=("winner", "count"), win_rate=("winner", "mean"),
        avg_pnl=("pnl_pct", "mean"), sharpe_proxy=("pnl_pct", lambda x: x.mean()/x.std() * np.sqrt(252) if x.std() > 0 else 0)
    ).round(4)
    lines.append("\n**Per-fold summary:**\n")
    lines.append(grp3.to_markdown())

    return "\n".join(lines), None


def cut4_feature_stratification(df):
    """Win rate by cs_rank_momentum quintile and orb_position decile."""
    lines = ["### Cut 4 — Feature Stratification (Which Features Carry the Edge)\n"]

    df2 = df.copy()
    df2["mom_quintile"] = pd.qcut(df2["cs_rank_momentum"], q=5,
                                   labels=["Q1(weakest)", "Q2", "Q3", "Q4", "Q5(strongest)"],
                                   duplicates="drop")
    grp = df2.groupby("mom_quintile", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("**Win rate by cross-sectional momentum rank quintile:**\n")
    lines.append(grp.to_markdown())

    # orb_position decile
    df2["orb_pos_decile"] = pd.qcut(df2["orb_position"], q=5,
                                     labels=["D1(low)", "D2", "D3", "D4", "D5(high)"],
                                     duplicates="drop")
    grp2 = df2.groupby("orb_pos_decile", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("\n**Win rate by ORB position quintile (0=at low, 1=at high):**\n")
    lines.append(grp2.to_markdown())

    # Volume surge
    df2["vol_quintile"] = pd.qcut(df2["volume_surge"], q=5,
                                   labels=["Q1(low vol)", "Q2", "Q3", "Q4", "Q5(high vol)"],
                                   duplicates="drop")
    grp3 = df2.groupby("vol_quintile", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("\n**Win rate by volume_surge quintile:**\n")
    lines.append(grp3.to_markdown())

    # RSI
    df2["rsi_bin"] = pd.cut(df2["rsi_14"], bins=[0, 30, 50, 70, 100],
                              labels=["Oversold(<30)", "Neutral-low(30-50)",
                                      "Neutral-high(50-70)", "Overbought(>70)"])
    grp4 = df2.groupby("rsi_bin", observed=True).agg(
        n=("winner", "count"), win_rate=("winner", "mean"), avg_pnl=("pnl_pct", "mean")
    ).round(4)
    lines.append("\n**Win rate by RSI-14 bin:**\n")
    lines.append(grp4.to_markdown())

    lines.append("\n**Key question:** Does win rate increase monotonically with cs_rank_momentum?")
    lines.append("If yes → model has learned momentum signal. If U-shaped or flat → reversion or noise.\n")

    return "\n".join(lines), None


def cut5_meta_model_attribution(fold_sharpes_with_meta, fold_sharpes_no_meta):
    """Compare Sharpe with and without meta-model."""
    lines = ["### Cut 5 — MetaLabelModel Attribution\n"]
    lines.append("Comparison of walk-forward Sharpe with meta-model (Phase 46 result) vs without.\n")

    lines.append("| Fold | With meta | Without meta | Meta contribution |")
    lines.append("|---|---|---|---|")
    total_with = total_without = 0
    for i, (wm, wom) in enumerate(zip(fold_sharpes_with_meta, fold_sharpes_no_meta), 1):
        delta = wm - wom
        lines.append(f"| {i} | {wm:+.3f} | {wom:+.3f} | {delta:+.3f} |")
        total_with += wm
        total_without += wom
    avg_with = total_with / max(len(fold_sharpes_with_meta), 1)
    avg_without = total_without / max(len(fold_sharpes_no_meta), 1)
    delta_avg = avg_with - avg_without
    lines.append(f"| **Avg** | **{avg_with:+.3f}** | **{avg_without:+.3f}** | **{delta_avg:+.3f}** |")

    if abs(delta_avg) < 0.05:
        verdict = "⚠️  **META-MODEL CONTRIBUTES NEAR-ZERO** (+{:.3f} Sharpe). Drop it — it is adding complexity without signal.".format(delta_avg)
    elif delta_avg > 0.05:
        verdict = "✅  **META-MODEL CONTRIBUTES +{:.3f} SHARPE**. Keep for now, but investigate further.".format(delta_avg)
    else:
        verdict = "⚠️  **META-MODEL HURTS ({})**. Remove immediately.".format(delta_avg)

    lines.append(f"\n**Verdict:** {verdict}\n")
    return "\n".join(lines), delta_avg


def cut6_mfe_mae(df):
    """MFE/MAE distribution by exit type."""
    lines = ["### Cut 6 — MFE / MAE Distribution by Exit Type\n"]
    lines.append("MFE = Maximum Favorable Excursion (how far it went in our favor before exit).")
    lines.append("MAE = Maximum Adverse Excursion (how far it went against us before exit).\n")

    grp = df.groupby("exit_reason").agg(
        n=("mfe", "count"),
        avg_mfe=("mfe", "mean"),
        avg_mae=("mae", "mean"),
        mfe_p25=("mfe", lambda x: x.quantile(0.25)),
        mfe_p75=("mfe", lambda x: x.quantile(0.75)),
        mae_p25=("mae", lambda x: x.quantile(0.25)),
        mae_p75=("mae", lambda x: x.quantile(0.75)),
    ).round(5)
    lines.append(grp.to_markdown())

    # Key question: do time-exit losers show meaningful MFE before reversing?
    time_neg = df[(df["exit_reason"].isin(["TIME_EXIT", "FORCE_CLOSE"])) & (df["pnl_pct"] <= 0)]
    if len(time_neg) > 0:
        avg_mfe_time_neg = time_neg["mfe"].mean()
        lines.append(f"\n**Time-exit losers avg MFE:** {avg_mfe_time_neg:.4f}")
        if avg_mfe_time_neg > 0.005:
            lines.append("⚠️  **Time-exit losers showed meaningful upside before reversing** — these could be target captures with tighter exit or earlier trailing stop.")

    # Stop exits: did they show MFE before stopping out?
    stop_exits = df[df["exit_reason"] == "STOP"]
    if len(stop_exits) > 0:
        avg_mfe_stops = stop_exits["mfe"].mean()
        lines.append(f"\n**Stop-exit avg MFE:** {avg_mfe_stops:.4f}")
        if avg_mfe_stops > 0.003:
            lines.append("⚠️  **Stop exits showed MFE before being stopped** — classic false stop pattern. Stop may be too tight OR -1.25×stop_pressure label is penalizing these setups.")

    lines.append("")
    return "\n".join(lines), None


def cut7_per_fold_regime(df):
    """Per-fold win rate, exit mix, avg trade R."""
    lines = ["### Cut 7 — Per-Fold Regime Analysis\n"]

    for fold_id in sorted(df["fold"].unique()):
        fold_df = df[df["fold"] == fold_id]
        n = len(fold_df)
        wr = fold_df["winner"].mean()
        avg_pnl = fold_df["pnl_pct"].mean()
        avg_mfe = fold_df["mfe"].mean()
        avg_mae = fold_df["mae"].mean()
        exit_mix = fold_df["exit_reason"].value_counts(normalize=True).round(3)
        orb_plus = (fold_df["orb_breakout"] == 1).mean()
        orb_zero = (fold_df["orb_breakout"] == 0).mean()

        lines.append(f"**Fold {fold_id}:**")
        lines.append(f"- Trades: {n} | Win rate: {wr:.1%} | Avg pnl: {avg_pnl:.4f}")
        lines.append(f"- Avg MFE: {avg_mfe:.4f} | Avg MAE: {avg_mae:.4f}")
        lines.append(f"- ORB+ entries: {orb_plus:.1%} | Inside-ORB entries: {orb_zero:.1%}")
        lines.append(f"- Exit mix: {dict(exit_mix)}")
        lines.append("")

    return "\n".join(lines), None


# ── Stop-pressure label analysis ──────────────────────────────────────────────

def stop_pressure_analysis(df):
    """
    Key diagnostic: are trades that showed high MAE (touched stop zone)
    systematically losing in the current model's training labels?
    """
    lines = ["### Stop-Pressure Label Bias Analysis\n"]
    lines.append("The path_quality label uses `-1.25 × stop_pressure` which penalizes any trade")
    lines.append("where price came within stop distance during the hold, even if it recovered.")
    lines.append("This section checks if high-MAE trades that ultimately won are being mislabeled.\n")

    # Trades that had significant adverse excursion but still won
    df2 = df.copy()
    # Estimate stop distance: 0.6x prior-day range
    df2["stop_dist_est"] = df2["prior_range_pct"] * 0.6
    df2["touched_stop_zone"] = df2["mae"] >= df2["stop_dist_est"] * 0.8  # within 80% of stop

    touched = df2[df2["touched_stop_zone"]]
    not_touched = df2[~df2["touched_stop_zone"]]

    lines.append(f"**Trades that touched stop zone (MAE ≥ 80% of stop distance):** {len(touched)} ({100*len(touched)/max(len(df),1):.1f}%)")
    lines.append(f"- Win rate: {touched['winner'].mean():.1%} | Avg pnl: {touched['pnl_pct'].mean():.4f}")
    lines.append(f"\n**Trades that did NOT touch stop zone:** {len(not_touched)} ({100*len(not_touched)/max(len(df),1):.1f}%)")
    lines.append(f"- Win rate: {not_touched['winner'].mean():.1%} | Avg pnl: {not_touched['pnl_pct'].mean():.4f}")

    # Touched-stop-zone winners: these are what the label PENALIZES but actually worked
    touched_winners = touched[touched["winner"] == 1]
    lines.append(f"\n**Stop-zone-touched winners (label likely penalized):** {len(touched_winners)}")
    lines.append(f"- Avg pnl: {touched_winners['pnl_pct'].mean():.4f}" if len(touched_winners) > 0 else "- None")

    # By exit reason among stop-zone-touched
    if len(touched) > 0:
        exit_mix = touched["exit_reason"].value_counts(normalize=True).round(3)
        lines.append(f"\n**Exit mix for stop-zone-touched trades:** {dict(exit_mix)}")

    verdict = ""
    if len(touched) > 0:
        wr_diff = touched["winner"].mean() - not_touched["winner"].mean()
        if wr_diff < -0.05:
            verdict = f"⚠️  **Stop-zone-touched trades win {abs(wr_diff):.1%} LESS** — consistent with stop_pressure label bias. Consider reducing -1.25 coefficient."
        elif wr_diff > 0.02:
            verdict = "✅  Stop-zone-touched trades don't systematically lose more — stop_pressure coefficient is probably not the main issue."
        else:
            verdict = "➡️  Marginal difference — stop_pressure coefficient may or may not be the issue."

    lines.append(f"\n**Verdict:** {verdict}\n")
    return "\n".join(lines), verdict


# ── Write report ──────────────────────────────────────────────────────────────

def write_report(sections, answers):
    doc_path = DOCS_DIR / "diagnostic_report.md"
    lines = [
        "# Phase 47 — Diagnostic Report (Phase 0)",
        "",
        f"**Date:** {date.today().isoformat()}",
        f"**Model:** Intraday v22 (path_quality labels, soft ORB gate, 42 features)",
        f"**Data:** 3 walk-forward folds (Oct 2024 → Apr 2026), 526 trades total",
        "",
        "---",
        "",
        "## Three Structural Questions",
        "",
        f"1. **Momentum or Reversion?** → {answers.get('momentum_verdict', 'TBD')}",
        f"2. **True Realized R:R?** → {answers.get('realized_rr', 'TBD')}",
        f"3. **Meta-Model Marginal Sharpe?** → {answers.get('meta_delta', 'TBD')}",
        f"4. **Stop-Pressure Label Bias?** → {answers.get('stop_pressure_verdict', 'TBD')}",
        "",
        "---",
        "",
    ]
    for section in sections:
        lines.append(section)
        lines.append("\n---\n")

    lines += [
        "## Phase 47 Experiment Ordering Recommendation",
        "",
        "Based on the above diagnostics:",
        "",
    ]

    rr = answers.get("realized_rr_val", 2.0)
    meta = answers.get("meta_delta_val", 0.0)

    if meta is not None and abs(meta) < 0.05:
        lines.append("- **Phase 1 (drop meta-model):** CONFIRMED — meta contributes near-zero. Execute first.")
    if rr is not None and rr < 1.6:
        lines.append("- **Phase 3 (stop/target compression):** ELEVATED PRIORITY — true R:R is well below stated 2:1.")
    lines.append("- **Phase 2 (XGBRanker):** Proceed as planned — highest expected impact.")
    lines.append("- **Phase 4 (top-300 liquidity):** Proceed as planned.")
    lines.append("- **Phase 5 (feature pack):** Proceed as planned — run last.")

    doc_path.write_text("\n".join(lines), encoding="utf-8")
    _ok(f"Report written → {doc_path}")
    return doc_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*62)
    print("  Phase 47 — Diagnostic Dive (Phase 0)")
    print("="*62 + "\n")

    model, version = load_model()
    meta_model = load_meta_model()
    symbols_data, spy_data = load_data()

    print("\n[Step 1] Running 3 folds WITH meta-model + abstention gate...")
    all_fold_trades_with_meta = []
    sharpes_with_meta = []
    for start, end, fold_id in FOLDS:
        trades, sharpe = run_fold(model, symbols_data, spy_data, start, end, fold_id,
                                   meta_model=meta_model, pm_vix=25.0, pm_spy_ma=20)
        all_fold_trades_with_meta.append((fold_id, trades))
        sharpes_with_meta.append(sharpe)

    print("\n[Step 2] Running 3 folds WITHOUT meta-model (abstention gate kept)...")
    sharpes_no_meta = []
    for start, end, fold_id in FOLDS:
        trades_nm, sharpe_nm = run_fold(model, symbols_data, spy_data, start, end, fold_id,
                                         meta_model=None, pm_vix=25.0, pm_spy_ma=20)
        sharpes_no_meta.append(sharpe_nm)

    print("\n[Step 3] Building trade DataFrame for feature analysis...")
    df = build_trade_df(all_fold_trades_with_meta)

    if len(df) == 0:
        print("  ERROR: No trades to analyze. Exiting.")
        sys.exit(1)

    print(f"\n[Step 4] Running 7 diagnostic cuts on {len(df)} trades...\n")

    sections = []
    answers = {}

    c1, momentum_verdict = cut1_momentum_vs_reversion(df)
    sections.append(c1)
    answers["momentum_verdict"] = momentum_verdict

    c2, realized_rr = cut2_exit_type_breakdown(df)
    sections.append(c2)
    answers["realized_rr"] = f"{realized_rr:.2f}:1" if realized_rr else "TBD"
    answers["realized_rr_val"] = realized_rr

    c3, _ = cut3_regime_gates(df)
    sections.append(c3)

    c4, _ = cut4_feature_stratification(df)
    sections.append(c4)

    c5, meta_delta = cut5_meta_model_attribution(sharpes_with_meta, sharpes_no_meta)
    sections.append(c5)
    answers["meta_delta"] = f"{meta_delta:+.3f} Sharpe" if meta_delta is not None else "TBD"
    answers["meta_delta_val"] = meta_delta

    c6, _ = cut6_mfe_mae(df)
    sections.append(c6)

    c7, _ = cut7_per_fold_regime(df)
    sections.append(c7)

    c_sp, sp_verdict = stop_pressure_analysis(df)
    sections.append(c_sp)
    answers["stop_pressure_verdict"] = sp_verdict

    doc_path = write_report(sections, answers)

    print("\n" + "="*62)
    print("  DIAGNOSTIC SUMMARY")
    print("="*62)
    print(f"  1. Momentum/Reversion: {answers['momentum_verdict']}")
    print(f"  2. True R:R: {answers['realized_rr']}")
    print(f"  3. Meta marginal Sharpe: {answers['meta_delta']}")
    print(f"  4. Stop-pressure bias: {answers['stop_pressure_verdict']}")
    print(f"\n  Full report → {doc_path}")
    print("="*62 + "\n")


if __name__ == "__main__":
    main()
