---
description: Parse a retrain log and append a formatted entry to ML_EXPERIMENT_LOG.md
argument-hint: "[path-to-log]  (default: newest retrain_*.log in logs/)"
---

You are logging the result of a model retrain for the user. Follow these steps exactly — show your work at each step before proceeding.

## Step 1 — Find the log file
- If `$ARGUMENTS` is given, use that path.
- Otherwise: list `c:/Projects/MrTrader/logs/retrain_*.log` and `c:/Projects/MrTrader/logs/train_*.log` sorted by mtime (newest first). Also check for `retrain_stdout.log` in the repo root.
- Show the chosen file path and ask the user to confirm before parsing.

## Step 2 — Extract metrics from the log
Read the log (tail 600 lines is usually enough — summary lives at the end). Extract:
- Model family (swing / intraday) and version N — look for `swing_v{N}.pkl` or `intraday_meta_v{N}.pkl` in a "Saved" or "Model saved" line.
- Total features count.
- Per-fold OOS AUC and Sharpe (look for "Fold X" blocks with AUC/Sharpe values).
- Avg AUC, avg Sharpe, min-fold Sharpe across all folds.
- Training duration (seconds).
- Any notable warnings (MODEL DRIFT ALERT, regime score, etc).

Show the extracted numbers to the user and ask for confirmation before writing anything.

## Step 3 — Determine the gate verdict
Read `c:/Projects/MrTrader/app/ml/retrain_config.py` and extract `SWING_GATE` and `INTRADAY_GATE` thresholds (do NOT trust the experiment log header — use the code).

Gate logic:
- ✅ Keep: avg Sharpe ≥ min_avg_sharpe AND min-fold Sharpe ≥ min_fold_sharpe
- ❌ Revert: either threshold not met
- 🔄 Pending: user explicitly says results are pending further testing

## Step 4 — Verify the pkl exists
Confirm `c:/Projects/MrTrader/app/ml/models/{family}_v{N}.pkl` exists. If not, warn the user before continuing.

## Step 5 — Read the current experiment log format
Read `c:/Projects/MrTrader/docs/ML_EXPERIMENT_LOG.md`:
- Find the last `## ` heading (use Grep `^## ` to list all headings, take the last).
- Mirror that entry's exact format for the new entry.
- The standard format is:
  ```
  ## {Model} v{N} — {one-line description} — {YYYY-MM-DD}
  **Goal:** ...
  **Hypothesis:** ...
  **What was built:** ...
  **Results:**
  | Metric | Value | Gate | Pass? |
  |--------|-------|------|-------|
  | Avg Sharpe | X.XX | ≥ Y.YY | ✅/❌ |
  | Min-fold Sharpe | X.XX | ≥ -0.30 | ✅/❌ |
  | OOS AUC | X.XXX | — | — |
  | Training time | Xs | — | — |
  **Top features:** (list top 10 from log)
  **Verdict:** ✅ Keep / ❌ Revert / 🔄 Pending
  **Notes:** ...
  ---
  ```

## Step 6 — Draft the new entry
Draft the full entry. Show it to the user. Wait for explicit approval ("looks good", "yes", etc.) before writing.

## Step 7 — Append to the log
Use the Edit tool to append after the last `---` separator in the file. Do NOT overwrite anything.

## Step 8 — Update champion table (if ✅)
If verdict is ✅, find the "Current Champion Models" or "Active Models" table near the top of the file and update the row for this model family with the new version, AUC, and date.

## Step 9 — Do NOT commit
Leave staging to the user. Remind them to run `/pr-ready` before merging.

## Failure conditions — stop and ask the user:
- Log file not found
- Version number not inferable from log
- pkl file missing
- Extracted numbers seem implausible (e.g. AUC > 1.0 or Sharpe > 20)
