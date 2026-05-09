"""
Session context injector for Claude Code UserPromptSubmit hook.

Outputs a <session-context> block to stdout. Claude Code injects this
into the model's context automatically on every prompt.

Hard constraints:
- Must complete in <500ms (caches aggressively)
- Must never raise an uncaught exception (swallowed in main)
- Must work on Windows with forward-slash paths
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

ROOT = Path("c:/Projects/MrTrader")
MODELS_DIR = ROOT / "app/ml/models"
LOGS_DIR = ROOT / "logs"
DB_PATH = ROOT / "data/mrtrader.db"
RETRAIN_CFG = ROOT / "app/ml/retrain_config.py"
EXPERIMENT_LOG = ROOT / "docs/ML_EXPERIMENT_LOG.md"
CACHE_FILE = ROOT / ".claude/state/model_meta_cache.json"
OVERNIGHT_JOBS = ROOT / ".claude/state/overnight_jobs.json"


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def _load_cache() -> dict:
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    try:
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Model versions
# ---------------------------------------------------------------------------

def _get_active_models(cache: dict) -> list[dict]:
    """Scan models dir, pick highest version per family, pull AUC from pkl or cache."""
    if not MODELS_DIR.exists():
        return []

    pattern = re.compile(r"^(swing|intraday_meta|regime_model)_v(\d+)\.pkl$")
    families: dict[str, tuple[int, Path]] = {}

    for f in MODELS_DIR.iterdir():
        m = pattern.match(f.name)
        if not m:
            continue
        family, ver = m.group(1), int(m.group(2))
        if family not in families or ver > families[family][0]:
            families[family] = (ver, f)

    results = []
    for family, (ver, path) in sorted(families.items()):
        mtime = path.stat().st_mtime
        cache_key = f"{path.name}:{mtime}"
        auc = cache.get(cache_key)

        if auc is None:
            auc = _extract_auc(path)
            cache[cache_key] = auc

        age_days = int((time.time() - mtime) / 86400)
        results.append({
            "family": family,
            "version": ver,
            "auc": auc,
            "age_days": age_days,
        })

    return results


def _extract_auc(pkl_path: Path) -> float | None:
    """Extract AUC from the most recent matching retrain log, or the experiment log."""
    family = pkl_path.stem.split("_v")[0]  # e.g. "swing", "intraday_meta"
    version_str = pkl_path.stem.split("_v")[-1]  # e.g. "181"

    # Try retrain logs first
    if LOGS_DIR.exists():
        candidates = sorted(
            LOGS_DIR.glob(f"retrain_{family}*.log"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        for log in candidates[:3]:
            try:
                text = log.read_text(encoding="utf-8", errors="ignore")
                if f"v{version_str}" not in text and f"_v{version_str}.pkl" not in text:
                    continue
                m = re.search(r"ROC-AUC\s*:\s*([0-9.]+)", text)
                if m:
                    return round(float(m.group(1)), 3)
            except Exception:
                pass

    # Fall back to experiment log
    try:
        text = EXPERIMENT_LOG.read_text(encoding="utf-8", errors="ignore")
        # Find the section for this version
        section_m = re.search(
            rf"## (?:Swing|Intraday)[^\n]*v{version_str}.*?(?=^## |\Z)",
            text, re.MULTILINE | re.DOTALL
        )
        if section_m:
            auc_m = re.search(r"AUC[|\s:]+([0-9.]+)", section_m.group(0))
            if auc_m:
                return round(float(auc_m.group(1)), 3)
    except Exception:
        pass

    return None


# ---------------------------------------------------------------------------
# Sacred holdout
# ---------------------------------------------------------------------------

def _get_sacred_holdout() -> str | None:
    try:
        text = RETRAIN_CFG.read_text(encoding="utf-8")
        m = re.search(r'SACRED_HOLDOUT_START\s*=\s*["\'](\d{4}-\d{2}-\d{2})["\']', text)
        if m:
            return m.group(1)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Regime score
# ---------------------------------------------------------------------------

def _get_regime() -> dict | None:
    try:
        conn = sqlite3.connect(str(DB_PATH), timeout=1)
        row = conn.execute(
            "SELECT score, regime, captured_at FROM regime_snapshots "
            "ORDER BY captured_at DESC LIMIT 1"
        ).fetchone()
        conn.close()
        if row:
            return {"score": round(row[0], 3), "regime": row[1], "captured_at": row[2][:10]}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Git state
# ---------------------------------------------------------------------------

def _get_git_state() -> dict:
    def run(cmd):
        try:
            return subprocess.check_output(
                cmd, shell=True, cwd=str(ROOT),
                stderr=subprocess.DEVNULL, timeout=3,
            ).decode().strip()
        except Exception:
            return ""

    branch = run("git rev-parse --abbrev-ref HEAD") or "<unknown>"
    dirty_count = len([l for l in run("git status --porcelain").splitlines() if l.strip()])
    ahead_behind = run("git rev-list --left-right --count origin/main...HEAD")
    behind, ahead = 0, 0
    if "\t" in ahead_behind:
        parts = ahead_behind.split("\t")
        try:
            behind, ahead = int(parts[0]), int(parts[1])
        except ValueError:
            pass

    return {
        "branch": branch,
        "dirty": dirty_count,
        "ahead": ahead,
        "behind": behind,
    }


# ---------------------------------------------------------------------------
# Latest CPCV result
# ---------------------------------------------------------------------------

def _get_latest_cpcv() -> str | None:
    """Pull the latest logged CPCV headline from ML_EXPERIMENT_LOG.md."""
    try:
        text = EXPERIMENT_LOG.read_text(encoding="utf-8")
        # Look for CPCV result lines like "mean_sharpe: X.XX"
        m = re.search(r"mean_sharpe[:\s]+([0-9.\-]+).*?DSR[:\s]+p=([0-9.]+)", text, re.DOTALL)
        if m:
            return f"mean_sharpe={m.group(1)} DSR_p={m.group(2)}"
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Active background jobs
# ---------------------------------------------------------------------------

def _get_active_jobs() -> list[str]:
    """Detect logs written in the last 90 seconds — heuristic for running jobs."""
    active = []
    now = time.time()

    # Check overnight_jobs.json first (authoritative)
    try:
        jobs = json.loads(OVERNIGHT_JOBS.read_text(encoding="utf-8"))
        for job in jobs:
            log_path = Path(job.get("log", ""))
            if log_path.exists() and (now - log_path.stat().st_mtime) < 90:
                active.append(f"{log_path.name} (launched {job.get('launched_at', '?')})")
    except Exception:
        pass

    # Fallback: any log file modified in last 90s
    if not active and LOGS_DIR.exists():
        for f in LOGS_DIR.glob("*.log"):
            try:
                if (now - f.stat().st_mtime) < 90:
                    size_mb = f.stat().st_size / 1_048_576
                    active.append(f"{f.name} ({size_mb:.1f} MB, active)")
            except Exception:
                pass

    return active


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    cache = _load_cache()
    models = _get_active_models(cache)
    _save_cache(cache)

    holdout = _get_sacred_holdout()
    regime = _get_regime()
    git = _get_git_state()
    active_jobs = _get_active_jobs()
    cpcv_headline = _get_latest_cpcv()

    lines = ["<session-context>"]
    lines.append(f"Date: {date.today().isoformat()} | Branch: {git['branch']}", )

    # Branch status
    status_parts = []
    if git["dirty"]:
        status_parts.append(f"{git['dirty']} uncommitted changes")
    if git["behind"]:
        status_parts.append(f"{git['behind']} behind origin/main — merge before pushing")
    if git["ahead"]:
        status_parts.append(f"{git['ahead']} ahead of origin/main")
    if not status_parts:
        status_parts.append("clean")
    lines[-1] += f" ({', '.join(status_parts)})"

    # Models
    if models:
        lines.append("Models:")
        for m in models:
            auc_str = f"AUC {m['auc']}" if m["auc"] is not None else "AUC unknown"
            lines.append(f"  {m['family']}_v{m['version']}  {auc_str}  age {m['age_days']}d")
    else:
        lines.append("Models: none found")

    # Regime
    if regime:
        lines.append(
            f"Regime: score={regime['score']} ({regime['regime']}) "
            f"as of {regime['captured_at']}"
        )

    # Sacred holdout
    if holdout:
        holdout_date = date.fromisoformat(holdout)
        days = (holdout_date - date.today()).days
        suffix = f"({abs(days)}d ago — fully reserved)" if days < 0 else f"({days}d from now)"
        lines.append(f"Sacred holdout: {holdout} {suffix}")

    # Latest CPCV
    if cpcv_headline:
        lines.append(f"Last CPCV: {cpcv_headline}")

    # Active jobs
    if active_jobs:
        lines.append("Active background jobs:")
        for j in active_jobs:
            lines.append(f"  {j}")

    lines.append("</session-context>")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        pass  # never block the session
