"""Gate ablation matrix for v186 (R2).

Runs 6 walk-forward configurations on the already-trained v186 model,
varying which default-on gates are active. Identifies which gates improve
Sharpe vs. suppress trades without benefit.

Usage:
    python scripts/gate_ablation_v186.py
    python scripts/gate_ablation_v186.py --max-symbols 300 --folds 5 --dsr-n 200
    python scripts/gate_ablation_v186.py --dry-run   # print commands only

Output:
    logs/gate_ablation_v186.json  — machine-readable results
    Markdown table printed to stdout (paste into ML_EXPERIMENT_LOG.md)
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Note: dispersion-gate is intraday-only — not included in swing ablation.
CONFIGS = [
    ("A_all_on",       [],                                                       "All gates ON (baseline)"),
    ("B_opp_only",     ["--no-earnings-blackout", "--no-macro-gate"],            "Opportunity score only"),
    ("C_earnings_only",["--no-pm-opportunity-score", "--no-macro-gate"],         "Earnings blackout only"),
    ("D_macro_only",   ["--no-pm-opportunity-score", "--no-earnings-blackout"],  "Macro gate only"),
    ("E_regime_only",  ["--no-pm-opportunity-score", "--no-earnings-blackout",
                        "--no-macro-gate", "--benign-gate"],                     "Regime/benign gate only"),
    ("F_all_off",      ["--no-pm-opportunity-score", "--no-earnings-blackout",
                        "--no-macro-gate"],                                       "All gates OFF"),
]


def _parse_sharpe(text: str) -> float | None:
    """Parse 'Avg Sharpe: +0.644' from WF stdout."""
    m = re.search(r"Avg Sharpe[:\s]+([\+\-]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def _parse_min_fold_sharpe(text: str) -> float | None:
    m = re.search(r"Min fold Sharpe[:\s]+([\+\-]?\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def _parse_total_trades(text: str) -> int | None:
    m = re.search(r"Total trades[:\s]+(\d+)", text)
    return int(m.group(1)) if m else None


def _parse_fold_sharpes(text: str) -> list[float]:
    """Parse per-fold Sharpe values from WF stdout."""
    return [float(x) for x in re.findall(r"Sharpe=([\+\-]?\d+\.\d+)", text)]


def run_config(
    name: str,
    extra_flags: list[str],
    description: str,
    folds: int,
    max_symbols: int,
    years: int,
    dsr_n: int,
    dry_run: bool,
) -> dict:
    cmd = [
        sys.executable, "scripts/walkforward_tier3.py",
        "--model", "swing",
        "--swing-model-version", "186",
        "--folds", str(folds),
        "--years", str(years),
        "--dsr-n", str(dsr_n),
        "--wf-max-symbols", str(max_symbols),
        *extra_flags,
    ]
    print(f"\n{'='*60}")
    print(f"  Config {name}: {description}")
    print(f"  Flags: {' '.join(extra_flags) if extra_flags else '(none)'}")
    print(f"  Command: {' '.join(cmd)}")

    if dry_run:
        return {"name": name, "description": description, "dry_run": True}

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    elapsed = time.time() - t0
    stdout = result.stdout + result.stderr

    avg_sharpe = _parse_sharpe(stdout)
    min_sharpe = _parse_min_fold_sharpe(stdout)
    total_trades = _parse_total_trades(stdout)
    fold_sharpes = _parse_fold_sharpes(stdout)

    print(f"  → elapsed: {elapsed:.0f}s  exit: {result.returncode}")
    print(f"  → avg_sharpe={avg_sharpe}  min_sharpe={min_sharpe}  trades={total_trades}")
    print(f"  → fold_sharpes={fold_sharpes}")

    return {
        "name": name,
        "description": description,
        "flags": extra_flags,
        "avg_sharpe": avg_sharpe,
        "min_fold_sharpe": min_sharpe,
        "total_trades": total_trades,
        "fold_sharpes": fold_sharpes,
        "elapsed_s": round(elapsed),
        "exit_code": result.returncode,
    }


def _markdown_table(results: list[dict]) -> str:
    all_off = next((r for r in results if r["name"] == "F_all_off"), {})
    all_off_trades = all_off.get("total_trades") or 1
    all_off_sharpe = all_off.get("avg_sharpe") or 0.0

    rows = []
    for r in results:
        if r.get("dry_run"):
            continue
        sharpe = r.get("avg_sharpe")
        min_s = r.get("min_fold_sharpe")
        trades = r.get("total_trades")
        trade_pct = f"{trades/all_off_trades:.0%}" if trades and all_off_trades else "—"
        sharpe_delta = f"{sharpe - all_off_sharpe:+.3f}" if sharpe is not None else "—"
        gate_verdict = ""
        if sharpe is not None and trades is not None:
            no_lift = sharpe <= all_off_sharpe + 0.05
            suppresses = trades < 0.85 * all_off_trades
            if no_lift and suppresses:
                gate_verdict = "⚠ suppress-only"
            elif sharpe > all_off_sharpe + 0.10:
                gate_verdict = "✓ load-bearing"
        rows.append(
            f"| {r['name']} | {r['description']} | "
            f"{sharpe:+.3f} | {min_s:+.3f} | {trades} ({trade_pct}) | "
            f"{sharpe_delta} | {gate_verdict} |"
            if sharpe is not None and min_s is not None and trades is not None
            else f"| {r['name']} | {r['description']} | — | — | — | — | — |"
        )

    header = (
        "| Config | Description | Avg Sharpe | Min Fold Sharpe | "
        "Trades (vs all-off) | Δ vs all-off | Verdict |\n"
        "|---|---|---|---|---|---|---|"
    )
    return header + "\n" + "\n".join(rows)


def main():
    parser = argparse.ArgumentParser(description="Gate ablation matrix on v186 (R2)")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--max-symbols", type=int, default=300,
                        help="Cap universe for time budget (default: 300)")
    parser.add_argument("--dsr-n", type=int, default=200,
                        help="DSR n_trials (default: 200, post-R1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands only, do not execute")
    parser.add_argument("--configs", nargs="+", default=None,
                        help="Run only specific config names (e.g. A_all_on F_all_off)")
    args = parser.parse_args()

    configs_to_run = [
        (name, flags, desc) for name, flags, desc in CONFIGS
        if args.configs is None or name in args.configs
    ]

    print(f"\n{'='*60}")
    print(f"  Gate Ablation — v186 (R2)")
    print(f"  folds={args.folds}  years={args.years}  max_symbols={args.max_symbols}")
    print(f"  dsr_n={args.dsr_n}  configs={[c[0] for c in configs_to_run]}")
    print(f"{'='*60}")

    results = []
    for name, flags, desc in configs_to_run:
        try:
            r = run_config(
                name=name, extra_flags=flags, description=desc,
                folds=args.folds, max_symbols=args.max_symbols,
                years=args.years, dsr_n=args.dsr_n, dry_run=args.dry_run,
            )
            results.append(r)
        except subprocess.TimeoutExpired:
            print(f"  ✗ Config {name} timed out (>7200s)")
            results.append({"name": name, "description": desc, "error": "timeout"})
        except Exception as exc:
            print(f"  ✗ Config {name} failed: {exc}")
            results.append({"name": name, "description": desc, "error": str(exc)})

    # Save JSON
    out_path = Path("logs/gate_ablation_v186.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_at": datetime.utcnow().isoformat(),
        "params": {
            "model_version": 186,
            "folds": args.folds,
            "years": args.years,
            "max_symbols": args.max_symbols,
            "dsr_n": args.dsr_n,
        },
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Print markdown table
    print("\n\n## Gate Ablation v186 Results\n")
    print(_markdown_table(results))
    print("\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
