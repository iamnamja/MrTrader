---
description: Parse a walk-forward or CPCV log and append entry to ML_EXPERIMENT_LOG.md
argument-hint: "[path-to-log] [swing|intraday]"
---

You are logging the result of a walk-forward or CPCV run. Follow exactly:

## Step 1 — Find the log file
- If `$ARGUMENTS` includes a path, use it.
- Otherwise: list `c:/Projects/MrTrader/logs/wf_*.log` and `c:/Projects/MrTrader/logs/p0_*_cpcv*.log` sorted by mtime. Show options and ask user to confirm.

## Step 2 — Parse with the existing parser
Run:
```
c:/Projects/MrTrader/venv/Scripts/python.exe c:/Projects/MrTrader/scripts/parse_cpcv_results.py <log> --json
```
If the log contains both swing and intraday sections, also try:
```
... --section swing
... --section intraday
```

If `parse_cpcv_results.py` errors with "No CPCV report header found", fall back to manual parsing:
- Grep for "Fold X" lines with Sharpe values
- Compute avg and min-fold Sharpe manually
- Note that DSR is not available and set dsr_p to "N/A"

Show the parsed JSON or extracted numbers to the user for confirmation before continuing.

## Step 3 — Determine gate verdict
Read gates from `c:/Projects/MrTrader/app/ml/retrain_config.py` (SWING_GATE / INTRADAY_GATE).

CPCV gate (from ML_ARCHITECTURE_ROADMAP.md §P0):
- mean_sharpe ≥ GATE.min_avg_sharpe
- p5_sharpe ≥ -0.30
- pct_positive ≥ 75%
- dsr_p > 0.95
- avg_profit_factor > 1.0 (if available)
- avg_calmar > 0.30 (if available)

All must pass for ✅. List each metric with Pass/Fail.

## Step 4 — Read experiment log format
Same as /log-model Step 5. Mirror the last entry's format exactly.

Standard CPCV entry format:
```
## {Model} v{N} — CPCV Baseline — {YYYY-MM-DD}
**Run type:** CPCV C(6,2)=15 paths / 5-fold standard WF
**Model:** swing_v{N} / intraday_meta_v{N}
**Results:**
| Metric | Value | Gate | Pass? |
|--------|-------|------|-------|
| Mean Sharpe | X.XX | ≥ Y.YY | ✅/❌ |
| P5 Sharpe | X.XX | ≥ -0.30 | ✅/❌ |
| % Positive paths | XX% | ≥ 75% | ✅/❌ |
| DSR p-value | X.XX | > 0.95 | ✅/❌ |
| Avg Calmar | X.XX | > 0.30 | ✅/❌ |
**Verdict:** ✅ Promoted / ❌ Gate not met / 🔄 Pending
**Notes:** ...
---
```

## Step 5 — Draft, confirm, append
Same as /log-model Steps 6-7. Show draft, wait for approval, then append.

## Step 6 — If promoted (✅)
Update the "Current Champion Models" table at the top of the log AND note in the entry that this model is cleared for paper trading.

## Step 7 — Do NOT commit
Remind user to run `/pr-ready` before merging.
