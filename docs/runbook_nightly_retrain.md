# Nightly Retrain Runbook

Run this after market close whenever a new phase has been merged that changes features or training logic.

**Tonight's trigger:** Phases 86b + 88 + 89a + 89b merged. Feature schema changed (3 fundamentals un-pruned). Must delete feature_store.db.

---

## Step 1 — Merge pending PRs

Check that all today's PRs have auto-merged:

```
gh pr list --state open
```

Expected: #129, #130, #131, #132 all MERGED. If any are still open, merge manually:

```
gh pr merge <number> --squash
```

---

## Step 2 — Pull latest main

```
git checkout main
git pull
```

---

## Step 3 — Run fundamentals backfill (Phase 89a)

Fetches 5+ years of 10-K EDGAR data for all S&P 500 symbols. Takes 30–60 min.

```
python scripts/backfill_fundamentals_history.py --workers 4
```

Expected output: `Written: data/fundamentals/fundamentals_history.parquet (NNNN rows, NNN symbols)`

If it fails partway, re-run — it will overwrite and redo all symbols (no resume). Can reduce workers if hitting EDGAR rate limits.

---

## Step 4 — Run sector ETF backfill (Phase 89b)

Fetches 5yr daily bars for all 11 sector ETFs. Takes < 1 min.

```
python scripts/backfill_sector_etf_history.py
```

Expected output: `Written: data/sector_etf/sector_etf_history.parquet (NNNN rows, 11 ETFs)`

---

## Step 5 — Delete stale feature store ⚠️ CRITICAL

Feature schema changed (profit_margin / revenue_growth / debt_to_equity are now active features). Cached rows from old training have the wrong schema — keeping the DB causes inhomogeneous filter → 0 training rows → model fails silently.

```
del app\ml\models\feature_store.db
```

Or on bash:
```
rm app/ml/models/feature_store.db
```

Confirm it's gone before proceeding.

---

## Step 6 — Retrain swing model

```
python scripts/retrain_cron.py --swing-only --no-fundamentals --workers 8
```

- `--no-fundamentals`: skip live EDGAR API calls (we use PIT parquet store instead)
- `--workers 8`: cap at 8 to avoid OOM

Watch for gate result in output. Pass = Tier 3 Sharpe > 0.8. If it fails:
- Try `--training-years 3` to exclude 2023 regime collapse
- Check ML_EXPERIMENT_LOG.md for prior context

Log the result in `docs/ML_EXPERIMENT_LOG.md` before moving on.

---

## Step 7 — Retrain intraday model

```
python scripts/retrain_cron.py --intraday-only --workers 8
```

Pass = Tier 3 Sharpe > 1.5. If it fails, check the log — v29 (+1.830) stays active as fallback.

Log the result in `docs/ML_EXPERIMENT_LOG.md`.

---

## Step 8 — Restart uvicorn

```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Check startup logs for:
- `Intraday model loaded: v<N>` — confirms new model is active
- `Swing model loaded: v<N>` — confirms new swing model is active
- No `ERROR` lines in the first 30 seconds

---

## Step 9 — Smoke check

```
python -m pytest -q tests/test_backtest_tier1_intraday.py tests/test_gate_enforcement.py
```

Both should pass in < 30 seconds.

---

## Checklist

- [ ] All PRs merged (#129 #130 #131 #132)
- [ ] `git pull` on main
- [ ] Fundamentals backfill done (`fundamentals_history.parquet` exists)
- [ ] Sector ETF backfill done (`sector_etf_history.parquet` exists)
- [ ] `feature_store.db` deleted ← most common mistake
- [ ] Swing retrain: gate passed (Sharpe > 0.8), logged in ML_EXPERIMENT_LOG.md
- [ ] Intraday retrain: gate passed (Sharpe > 1.5), logged in ML_EXPERIMENT_LOG.md
- [ ] uvicorn restarted, both models loaded in logs
- [ ] Smoke tests pass

---

## If a model fails its gate

**Swing fails:** Don't deploy. Check fold Sharpes — if fold 3 (2023) is dragging it down, try `--training-years 3`. If still fails, keep v142 active and note in log.

**Intraday fails:** Don't deploy. v29 (+1.830) stays loaded — it's the in-memory fallback. Note in log and investigate tomorrow.

**Never deploy a model that failed its Tier 3 gate.**
