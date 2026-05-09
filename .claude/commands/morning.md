---
description: Summarize overnight job outputs and recommend next action for the day
---

Good morning. Run this debrief to start the day.

## Step 1 — Find overnight logs
Find log files modified since yesterday 18:00:

```
c:/Projects/MrTrader/venv/Scripts/python.exe -c "
import os, time
from pathlib import Path
cutoff = time.time() - 18 * 3600  # last 18 hours
logs_dir = Path('c:/Projects/MrTrader/logs')
root_logs = list(Path('c:/Projects/MrTrader').glob('*.log'))
all_logs = list(logs_dir.glob('*.log')) + root_logs if logs_dir.exists() else root_logs
recent = [(f, f.stat().st_mtime) for f in all_logs if f.stat().st_mtime > cutoff]
for f, mt in sorted(recent, key=lambda x: x[1]):
    age_min = int((time.time() - mt) / 60)
    still_running = age_min < 2
    size_mb = f.stat().st_size / 1_048_576
    print(f'{'RUNNING' if still_running else 'DONE':8s}  {age_min:5d}m ago  {size_mb:6.1f}MB  {f.name}')
"
```

Also check `.claude/state/overnight_jobs.json` for jobs launched via `/overnight`.

## Step 2 — Classify and summarize each log
For each completed log, determine job type from filename:

**Retrain logs** (`retrain_*.log`, `train_*.log`):
- Tail last 100 lines. Extract: version saved, AUC, fold Sharpes, gate verdict.
- Check if `app/ml/models/swing_v{N}.pkl` or `intraday_meta_v{N}.pkl` was actually saved.
- Check if ML_EXPERIMENT_LOG.md has an entry for this version.

**Walk-forward / CPCV logs** (`wf_*.log`, `p0_*cpcv*.log`):
- Run: `c:/Projects/MrTrader/venv/Scripts/python.exe c:/Projects/MrTrader/scripts/parse_cpcv_results.py <log> --json`
- Extract: mean_sharpe, p5_sharpe, pct_positive, dsr_p, verdict.

**Backfill logs** (`backfill_*.log`):
- Tail last 20 lines. Extract: symbols processed, errors, completion status.

**Other logs**: tail last 30 lines and summarize.

## Step 3 — Check documentation gaps
For each completed training/WF job:
- Is there a corresponding ML_EXPERIMENT_LOG.md entry? (grep for the version number)
- If not → flag it: "⚠️ v{N} has no experiment log entry — run /log-model or /log-wf"

## Step 4 — Recommend next steps
Based on results, give a numbered list:
1. If gate passed → recommend CPCV if not run, or paper trading setup
2. If gate failed → recommend what to try next per ML_ARCHITECTURE_ROADMAP.md
3. If experiment log is missing entries → run /log-model or /log-wf
4. If nothing ran overnight → suggest /overnight to schedule something
5. If CPCV still running → estimated ETA from progress lines

## Step 5 — Branch hygiene
Quick check: `git -C c:/Projects/MrTrader branch -a | grep -v "HEAD\|main\|master"`
Flag any branches with no open PR. Suggest deletion for any that are merged.

## Output format
```
=== Morning Debrief — {DATE} ===

OVERNIGHT ACTIVITY:
  ✅/❌/🔄  {job name}  →  {one-line result}
  ...

DOCUMENTATION GAPS:
  ⚠️  {version} missing experiment log entry

RECOMMENDED NEXT STEPS:
  1. ...
  2. ...

BRANCH STATUS:
  {branch count, any stale ones}
```
