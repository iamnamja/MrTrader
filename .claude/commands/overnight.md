---
description: List jobs appropriate to run overnight; launch the ones you pick
---

## Step 1 — Read current project state
Gather:
- Active model versions + AUC (from `app/ml/models/` newest pkl per family)
- Last retrain date (mtime of newest swing/intraday pkl)
- Last WF/CPCV date (mtime of newest wf_*.log in logs/)
- Current branch and git cleanliness
- Any jobs already running (mtime of log files < 90s)
- Active phase from `docs/MASTER_BACKLOG.md` (first 50 lines)

## Step 2 — Build candidate job list
Evaluate each candidate against pre-conditions:

| # | Job | ETA | Pre-conditions | Command |
|---|-----|-----|----------------|---------|
| 1 | CPCV swing v{N} | ~4h | swing model exists, no CPCV ran in last 7d | see below |
| 2 | CPCV intraday v{N} | ~4h | intraday model exists, no CPCV ran in last 7d | see below |
| 3 | Swing retrain | ~3h | NOT within 3d of last retrain | see below |
| 4 | Intraday retrain | ~2h | NOT within 3d of last retrain | see below |
| 5 | FMP fundamentals update | ~5min | fmp_fundamentals_history.parquet exists | see below |
| 6 | yfinance incremental update | ~10min | price_cache/ exists | see below |
| 7 | Macro history update | ~2min | macro_history.parquet exists | see below |

**CPCV swing command:**
```
python c:/Projects/MrTrader/scripts/walkforward_tier3.py --model swing --swing-model-version {N} --cpcv --cpcv-k 6 --cpcv-paths 2 --allow-sacred-holdout 2>&1 | tee c:/Projects/MrTrader/logs/p0_swing_v{N}_cpcv.log
```

**CPCV intraday command:**
```
python c:/Projects/MrTrader/scripts/walkforward_tier3.py --model intraday --intraday-model-version {N} --cpcv --cpcv-k 6 --cpcv-paths 2 --allow-sacred-holdout 2>&1 | tee c:/Projects/MrTrader/logs/p0_intraday_v{N}_cpcv.log
```

**Swing retrain command:**
```
python c:/Projects/MrTrader/scripts/train_model.py --no-fundamentals --workers 8 --allow-sacred-holdout 2>&1 | tee c:/Projects/MrTrader/logs/retrain_swing.log
```

**Intraday retrain command:**
```
python c:/Projects/MrTrader/scripts/retrain_intraday.py --workers 8 --allow-sacred-holdout 2>&1 | tee c:/Projects/MrTrader/logs/retrain_intraday.log
```

**FMP update command:**
```
python c:/Projects/MrTrader/scripts/backfill_fmp_fundamentals.py --incremental --workers 4 2>&1 | tee c:/Projects/MrTrader/logs/backfill_fmp.log
```

## Step 3 — Show the menu
Display only candidates with pre-conditions met (skip already-running jobs).
Mark any that conflict with each other (e.g., two retrains = OOM risk).

⚠️ NEVER include a command with `fetch_fundamentals=True` — OOM risk (per feedback_training_fundamentals.md).
⚠️ NEVER include `--allow-sacred-holdout` on a command the user hasn't explicitly requested for final promotion.
⚠️ Running >1 XGBoost training job in parallel risks OOM — warn if user picks multiple.

## Step 4 — Launch selected jobs
For each job the user picks:
1. Launch using the Bash tool with `run_in_background: true`.
2. Record to `.claude/state/overnight_jobs.json`:
   ```json
   [{"job": "CPCV swing v181", "log": "logs/p0_swing_v181_cpcv.log", "launched_at": "2026-05-09 22:30"}]
   ```
3. Confirm each launch with the exact log file path.

## Step 5 — Important Windows caveat
⚠️ Background jobs launched via Claude Code's Bash tool may not survive if the Claude Code session or terminal window closes. For jobs that must run unattended overnight:
- Keep the Claude Code session open, OR
- Use Windows Task Scheduler, OR
- Use `start /B python ...` in a separate terminal window that stays open.

If you close Claude Code, check the log files directly in the morning with `/morning`.
