# MrTrader — Claude Code Instructions

## Architecture Documentation (MANDATORY)

**`docs/living/PIPELINE_ARCHITECTURE.md` is the single source of truth for the WF/CPCV pipeline.**

### Update rule
Any PR that touches files in any of these directories/files **must** update `docs/living/PIPELINE_ARCHITECTURE.md` in the same commit:

- `scripts/walkforward/` (any file)
- `scripts/walkforward_tier3.py`
- `app/backtesting/agent_simulator.py`
- `app/backtesting/intraday_agent_simulator.py`
- `app/backtesting/strategy_simulator.py`
- `app/ml/retrain_config.py` (when changing gate thresholds or feature flags)
- `scripts/retrain_cron.py`
- `scripts/retrain_intraday.py`

**What to update:**
- Changelog table (date, PR, change summary, files changed)
- Gate inventory if any gate threshold, behavior, or activation status changed
- Known Limitations if a KL was resolved or a new one discovered
- Feature Flags table if a new flag was added or default changed
- "Last full verification" date at the top when doing a comprehensive review

---

## LLM Review Briefing Kit

When briefing any LLM for pipeline code review, paste **exactly these** in order:

1. `docs/living/PIPELINE_ARCHITECTURE.md` — SSOT for pipeline, simulators, gate inventory, known limitations
2. `docs/living/MODEL_STATUS.md` — what's live now and its gate results
3. `docs/living/ML_EXPERIMENT_LOG.md` — active experiment campaign (current slice only)
4. `docs/living/MASTER_BACKLOG.md` — what's planned / in flight
5. `docs/reference/SYSTEM_BEHAVIOR.md` — runtime PM/RM/Trader agent behavior
6. `docs/living/DECISIONS.md` — why things are the way they are
7. `docs/reference/prompts/WF_LLM_REVIEW_PROMPT.md` — review framing template
8. The specific source files under review (`gates.py`, `cpcv.py`, the simulator)

**This kit only references `living/` and `reference/` — structurally cannot include stale content.**

### Critical context to paste for simulator questions
From `PIPELINE_ARCHITECTURE.md` Section 2 — always include when touching backtesting code:
- `AgentSimulator` (app/backtesting/agent_simulator.py) — swing WF/CPCV, DAILY MTM equity
- `IntradayAgentSimulator` (app/backtesting/intraday_agent_simulator.py) — intraday WF/CPCV, DAILY equity
- `StrategySimulator` (app/backtesting/strategy_simulator.py) — TIER-2 ONLY, not used in WF/CPCV

---

## Docs Structure

```
docs/
├── living/       ← current truth (6 files; continuously updated)
├── reference/    ← stable reference (correct-when-written; rarely edited)
│   └── prompts/
└── archive/      ← frozen history (never navigate daily)
    ├── experiment-log/
    ├── ml-history/
    ├── phase-specs/
    ├── llm-reviews/  (date-foldered)
    ├── audits/
    └── data/
```

**Living docs** (only these 6):
- `MASTER_BACKLOG.md` — backlog + phase roadmap
- `ML_EXPERIMENT_LOG.md` — append-only experiment journal
- `PIPELINE_ARCHITECTURE.md` — WF/CPCV SSOT
- `MODEL_STATUS.md` — active model versions + gate results
- `DECISIONS.md` — append-only architectural decisions
- `PROJECT_STATE.md` — one-screen "what's happening now"

---

## Development Practices

### Branches & PRs
- Always use a feature branch + PR; never push directly to main
- Enable auto-merge with `--auto` flag on PRs
- Delete branch after merge (`--delete-branch`)
- Merge PRs immediately when CI passes — do not leave open

### Merging
- Full pytest suite must pass (0 failures) before any merge
- `git fetch origin main && git merge origin/main` before pushing feature branches
- Merge open PRs on gates.py/cpcv.py before starting new fix branches

### Training
- Always use `--no-fundamentals --workers 8` for swing training (prevents OOM)
- Never use `fetch_fundamentals=True` in overnight job commands

### Documentation
- Update `docs/living/ML_EXPERIMENT_LOG.md` for every retrain/WF/CPCV run
- Update `docs/living/PIPELINE_ARCHITECTURE.md` per the rule above
- Update `docs/living/MODEL_STATUS.md` after every model promotion/retrain
- Update `docs/living/PROJECT_STATE.md` at session start/end when focus changes

### Notifications
- Use `notifier.enqueue("phase_complete", {...})` to email kimminjae@gmail.com on phase completion
- `notify_watcher.py` must be running to drain the queue

---

## Gate Thresholds (quick reference — authoritative source: app/ml/retrain_config.py)

| Gate | Swing | Intraday |
|---|---|---|
| Avg Sharpe | ≥ 0.80 | ≥ 1.00 |
| Min fold Sharpe | ≥ -0.30 | ≥ -0.30 |
| DSR p-value | > 0.95 | > 0.95 |
| Profit factor | ≥ 1.10 | ≥ 1.10 |
| Calmar ratio | ≥ 0.30 | ≥ 0.30 |
| Purge days | 85 calendar | 2 trading days |
| Sacred holdout | 2026-11-09 | 2026-11-09 |
| N_TRIALS_TESTED (DSR) | 250 | 250 |

---

## Monitoring

- Active job logs: `logs/mrtrader_YYYY-MM-DD.log`
- WF/CPCV logs: `logs/p0_*.log`, `logs/wf_*.log`
- Never use the Monitor tool; use ScheduleWakeup or background Bash until-loops
