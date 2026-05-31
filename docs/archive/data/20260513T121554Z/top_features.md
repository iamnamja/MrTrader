# Phase A1 — Feature IC Diagnostic Report

**Generated:** 2026-05-13 12:34 UTC

## Kill Criteria

- max |IC_mean| >= 0.02 (h=5d): **FAIL**
- Features passing all thresholds: **0** (need >= 3 to confirm edge exists)

## Interpretation

- **|IC_mean| >= 0.02**: minimum signal floor for a 5bps-cost strategy
- **|IC_IR| >= 0.5**: annualised risk-adjusted IC persistence
- **hit_rate >= 0.53**: IC is consistently in the right direction

If < 3 features pass all thresholds -> feature set has insufficient signal.
Go to Phase C (re-architect: label, model, or strategy change).
