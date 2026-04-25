"""
Phase 45 Phase 2: Train v120 with path_quality regression label.

Path quality label (Config B: stop=0.5x ATR, target=1.0x ATR):
  score = 1.0*upside_capture - 1.25*stop_pressure + 0.25*close_strength

Trains XGBRegressor (auto-detected from float labels in model.py).
Runs Tier 3 walk-forward after training and writes results to docs/phase45/.
"""
import sys
import os

# Windows: fix OMP deadlock before any numpy/xgboost imports
os.environ.setdefault("OMP_NUM_THREADS", "1")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import subprocess
import json
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results" / "phase45"
DOCS_DIR = ROOT / "docs" / "phase45"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def run_training():
    print("[Phase 45 P2] Starting v120 path_quality training...")
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "train_model.py"),
        "--label-scheme", "path_quality",
        "--model-type", "xgboost",
        "--years", "5",
        "--workers", "8",
        "--no-fundamentals",
    ]
    log_path = ROOT / "results" / "train_v120_log.txt"
    print(f"[Phase 45 P2] Logging to {log_path}")
    with open(log_path, "w", encoding="utf-8") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[Phase 45 P2] TRAINING FAILED (exit {proc.returncode}). Check {log_path}")
        sys.exit(1)
    print("[Phase 45 P2] Training complete.")
    return log_path


def run_walkforward(stop_mult: float = 0.5, target_mult: float = 1.0):
    print(f"[Phase 45 P2] Running Tier 3 walk-forward (stop={stop_mult}x, target={target_mult}x)...")
    out_path = RESULTS_DIR / f"p2_walkforward_{date.today().isoformat()}.json"
    cmd = [
        sys.executable, "-u",
        str(ROOT / "scripts" / "walkforward_tier3.py"),
        "--stop-mult", str(stop_mult),
        "--target-mult", str(target_mult),
        "--output", str(out_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode != 0:
        print(f"[Phase 45 P2] Walk-forward FAILED:\n{proc.stderr}")
        sys.exit(1)
    print(f"[Phase 45 P2] Walk-forward complete -> {out_path}")
    return out_path


def write_results_doc(wf_path: Path):
    try:
        with open(wf_path) as f:
            data = json.load(f)
    except Exception as e:
        print(f"[Phase 45 P2] Could not load walk-forward JSON: {e}")
        return

    folds = data.get("folds", [])
    if not folds:
        print("[Phase 45 P2] No folds in walk-forward output.")
        return

    sharpes = [fd.get("sharpe", 0) for fd in folds]
    avg_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0
    min_sharpe = min(sharpes) if sharpes else 0

    gate = avg_sharpe > 0.8 and min_sharpe > -0.40
    gate_str = "PASSED" if gate else "FAILED"

    lines = [
        "# Phase 45 - Phase 2 (v120): Path Quality Regression Label",
        "",
        f"**Date:** {date.today().isoformat()}",
        "**Label:** path_quality (regression, Config B: stop=0.5x ATR, target=1.0x ATR)",
        "**Model:** v120 (XGBRegressor, 84 OHLCV features)",
        "",
        "## Walk-Forward Results",
        "",
        "| Fold | OOS Period | Trades | Win% | Sharpe | PF | Stop% |",
        "|---|---|---|---|---|---|---|",
    ]
    for fd in folds:
        lines.append(
            f"| {fd.get('fold', '?')} | {fd.get('test_start', '?')} -> {fd.get('test_end', '?')} "
            f"| {fd.get('trades', '?')} | {fd.get('win_rate', 0)*100:.1f}% "
            f"| {fd.get('sharpe', 0):+.3f} | {fd.get('profit_factor', 0):.3f} "
            f"| {fd.get('stop_exits_pct', 0)*100:.1f}% |"
        )

    lines += [
        "",
        "## Gate Assessment: " + gate_str,
        "",
        f"- Avg Sharpe: {avg_sharpe:+.3f} (gate: > +0.80)",
        f"- Min Fold: {min_sharpe:+.3f} (gate: > -0.40)",
        "",
        "## vs Baselines",
        "",
        "| Model | Label | Avg Sharpe | Notes |",
        "|---|---|---|---|",
        "| v110 (baseline) | cross_sectional binary | +0.34 | Gate not met (walk-forward avg -0.73) |",
        f"| v120 (Phase 2) | path_quality regression | {avg_sharpe:+.3f} | {'Gate MET' if gate else 'Gate NOT met'} |",
    ]

    doc_path = DOCS_DIR / "v120_results.md"
    doc_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Phase 45 P2] Results written -> {doc_path}")
    print(f"[Phase 45 P2] Gate: {gate_str}  Avg Sharpe: {avg_sharpe:+.3f}  Min Fold: {min_sharpe:+.3f}")


if __name__ == "__main__":
    run_training()
    wf_path = run_walkforward(stop_mult=0.5, target_mult=1.0)
    write_results_doc(wf_path)
    print("[Phase 45 P2] Done.")
