# MrTrader — Project State

**One-screen view of what's happening right now. Updated at session start/end when focus changes.**

> **Update rule:** Human updates this at session boundaries. Keep it to one screen. This is NOT a planning doc (that's MASTER_BACKLOG.md) and NOT a history doc (that's ML_EXPERIMENT_LOG.md). It answers: "If I open the laptop cold, what do I need to know in 30 seconds?"

**Last updated:** 2026-05-31

---

## Active Phase
**Docs restructure + Gate Integrity fixes (Phase 1 of 3-phase pipeline hardening)**

Phase 1 scope: CRITICAL-1 (DSR ceiling), CRITICAL-2 (deployment-adjusted Sharpe), MEDIUM-1 (Calmar vol-floor), MEDIUM-3 (data span gate). Branch: `fix/gate-integrity-phase1` (pending creation after docs merge).

---

## Model Status
→ See `docs/living/MODEL_STATUS.md` for full details.

- **swing v224** — trained 2026-05-29, age 1d, no post-audit WF yet
- **intraday_meta v63** — CPCV PASSED 2026-05-30 (mean +5.143), but deployment-adjusted Sharpe not yet computed
- **regime v5** — active

---

## Immediate Next Actions (top 3)
1. Merge docs restructure PR (this branch)
2. Implement Phase 1 gate integrity fixes (CRITICAL-1/2, MEDIUM-1/3)
3. Run swing v224 CPCV after Phase 1 gates are in (first honest post-audit swing validation)

---

## Active Background Jobs
- `mrtrader_2026-05-31.log` — paper trading live

---

## Blockers / Risks
- Phase 1 fixes required before any CPCV result can be trusted at face value (KL-1 through KL-5 in PIPELINE_ARCHITECTURE.md)
- Swing v224 has never been CPCV-validated on the corrected pipeline

---

## Recently Completed
- 2026-05-31: 13-round adversarial audit complete (PRs #323–327), pipeline clean
- 2026-05-31: PIPELINE_ARCHITECTURE.md created, docs restructured
- 2026-05-30: intraday v63 CPCV PASSED on corrected pipeline
