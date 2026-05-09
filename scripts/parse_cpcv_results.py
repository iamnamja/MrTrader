"""
parse_cpcv_results.py — extract key CPCV metrics from a walkforward_tier3 log.

Usage:
    python scripts/parse_cpcv_results.py logs/p0_v171_cpcv_baseline.log
    python scripts/parse_cpcv_results.py logs/p0_v51_cpcv_baseline.log --json

Parses the CPCV report block printed by scripts/walkforward/cpcv.py::CPCVResult.print
and emits the headline P0 numbers:
  * mean Sharpe (point estimate across paths)
  * P5 Sharpe (5th percentile — the floor we care about)
  * P95 Sharpe
  * % positive paths
  * DSR p-value (Deflated Sharpe Ratio significance)
  * avg profit factor / avg Calmar (when present)

Designed for a log file containing exactly one CPCV report. If multiple CPCV
sections are present (e.g. swing + intraday in one log), the LAST one wins —
pass --section swing|intraday to disambiguate.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional


# Regexes anchored to the exact format CPCVResult.print emits.
_PATTERNS = {
    "mean_sharpe":        r"Mean Sharpe:\s+([+\-]?\d+\.\d+)",
    "std_sharpe":         r"Std Sharpe:\s+([+\-]?\d+\.\d+)",
    "p5_sharpe":          r"P5 Sharpe:\s+([+\-]?\d+\.\d+)",
    "p95_sharpe":         r"P95 Sharpe:\s+([+\-]?\d+\.\d+)",
    "pct_positive":       r"% positive:\s+([\d.]+)%",
    "dsr_p":              r"DSR p:\s+([\d.]+)",
    "avg_profit_factor":  r"Avg PF:\s+([\d.]+)",
    "avg_calmar":         r"Avg Calmar:\s+([+\-]?\d+\.\d+)",
}

_HEADER_RE = re.compile(r"CPCV Report\s+-\s+(\w+)\s+C\((\d+),(\d+)\)=(\d+) paths")


def parse_cpcv_log(path: Path, section: Optional[str] = None) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")

    # Split into sections by CPCV header
    sections = []
    matches = list(_HEADER_RE.finditer(text))
    if not matches:
        raise ValueError(f"No CPCV report header found in {path}")
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append({
            "model_type": m.group(1).lower(),
            "n_folds": int(m.group(2)),
            "n_paths": int(m.group(3)),
            "n_combinations": int(m.group(4)),
            "body": text[start:end],
        })

    if section:
        sections = [s for s in sections if s["model_type"] == section.lower()]
        if not sections:
            raise ValueError(f"No CPCV section for model_type={section!r} in {path}")

    target = sections[-1]  # last (most recent) report wins
    body = target["body"]
    metrics: dict = {
        "log_file": str(path),
        "model_type": target["model_type"],
        "n_folds": target["n_folds"],
        "n_paths": target["n_paths"],
        "n_combinations": target["n_combinations"],
    }
    for key, pat in _PATTERNS.items():
        m = re.search(pat, body)
        metrics[key] = float(m.group(1)) if m else None

    # Convert percent
    if metrics.get("pct_positive") is not None:
        metrics["pct_positive"] = metrics["pct_positive"] / 100.0

    # Gate verdict (parsed from text — easier than recomputing)
    if "CPCV GATE PASSED" in body:
        metrics["gate_passed"] = True
    elif "CPCV GATE NOT MET" in body:
        metrics["gate_passed"] = False
    else:
        metrics["gate_passed"] = None
    return metrics


def format_human(metrics: dict) -> str:
    def _fmt(v, fmt):
        return ("n/a" if v is None else format(v, fmt))
    pp = metrics["pct_positive"]
    pp_s = "n/a" if pp is None else f"{pp:.1%}"
    lines = [
        f"CPCV log         : {metrics['log_file']}",
        f"Model            : {metrics['model_type']}  "
        f"(C({metrics['n_folds']},{metrics['n_paths']})={metrics['n_combinations']} paths)",
        f"Median Sharpe    : {_fmt(metrics['mean_sharpe'], '+.3f')}",
        f"Std Sharpe       : {_fmt(metrics['std_sharpe'], '.3f')}",
        f"P5  Sharpe (5th) : {_fmt(metrics['p5_sharpe'], '+.3f')}",
        f"P95 Sharpe (95th): {_fmt(metrics['p95_sharpe'], '+.3f')}",
        f"% positive paths : {pp_s}",
        f"DSR p-value      : {_fmt(metrics['dsr_p'], '.3f')}",
        f"Avg profit factor: {_fmt(metrics['avg_profit_factor'], '.3f')}",
        f"Avg Calmar       : {_fmt(metrics['avg_calmar'], '.3f')}",
        f"Gate passed      : {metrics['gate_passed']}",
    ]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_file", type=Path, help="Path to a walkforward_tier3 log containing CPCV output")
    ap.add_argument("--section", choices=["swing", "intraday"], default=None,
                    help="Pick the section by model_type if log contains multiple")
    ap.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = ap.parse_args()

    if not args.log_file.exists():
        print(f"ERROR: log file not found: {args.log_file}", file=sys.stderr)
        return 2

    metrics = parse_cpcv_log(args.log_file, section=args.section)
    if args.json:
        print(json.dumps(metrics, indent=2))
    else:
        print(format_human(metrics))
    return 0


if __name__ == "__main__":
    sys.exit(main())
