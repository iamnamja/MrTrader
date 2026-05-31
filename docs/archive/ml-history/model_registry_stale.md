# Model Registry

## Active Models

| Model | Version | Status | WF Sharpe | Notes |
|-------|---------|--------|-----------|-------|
| swing | v215 | ACTIVE (paper) | -0.571 (post-bug-fix rerun) | Restored after v216 gate fail |
| intraday_meta | v63 | ACTIVE (paper) | — | No recent WF |
| regime | v5 | ACTIVE | — | Regime classification |

## Gate-Failed Models

| Model | Version | Avg WF Sharpe | Gate Result | Reason |
|-------|---------|---------------|-------------|--------|
| swing | v216 | -0.91 | FAILED | LambdaRank, 18 features, 20d. PF=0.00 every fold. |

## Walk-Forward Gate Criteria
- Swing: Sharpe ≥ 0.80, no single fold < 0.50, beats null 2σ
- Intraday: Sharpe ≥ 1.50

## Version History (swing)

| Version | Date | Features | Label | WF Sharpe | Notes |
|---------|------|----------|-------|-----------|-------|
| v215 | 2026-04 | 16 | 10d return | -0.571 | Post bug-fix rerun. Previous sim was unreliable. |
| v216 | 2026-05 | 18 | 20d rank | -0.91 | LambdaRank. Gate failed. PF=0.00 every fold. |

See `docs/ML_EXPERIMENT_LOG.md` for detailed fold results.
