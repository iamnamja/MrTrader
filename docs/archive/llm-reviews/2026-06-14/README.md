# 20260614 LLM Research Pack — find-the-next-alpha

External-LLM research request: after the Ruler-v2 gate rebuild + an honest candidate sweep
(only trend survived), get world-class-quant ideas for the best next research steps to find
alpha, then consolidate into a plan.

## How to use
1. Paste **`00_COVER_NOTE.md`** first (short guidance — some LLMs need lead-in text).
2. Then **`01_PROMPT.md`** (the full ask + output format).
3. Then attach **`02_STATE_SNAPSHOT.md`** (self-contained context — the key file).
4. For deeper digs, attach from **`files/`** as the LLM's context budget allows
   (minimum useful add: `DATA_PROVIDERS.md` + `RULER_V2_DESIGN.md`).

## Contents
| File | What | Size |
|---|---|---|
| `00_COVER_NOTE.md` | paste-first guidance + the ask in brief | small |
| `01_PROMPT.md` | the detailed prompt + output format | small |
| `02_STATE_SNAPSHOT.md` | **self-contained**: data we have, what we tried + why it died, what's live, the gate, constraints, the open question | ~16K |
| `files/DATA_PROVIDERS.md` | the data envelope (free/paid/cancelled) | 12K |
| `files/RULER_V2_DESIGN.md` | the live promotion gate any idea must pass | 16K |
| `files/ALPHA_V7_SYNTHESIS_AND_PLAN.md` | current direction (risk-premia book) — pressure-test it | 16K |
| `files/NEXT_PHASE_BLUEPRINT_2026-06.md` | prior 5-LLM synthesis / 7-phase plan | 56K |
| `files/MODEL_STATUS.md` | what's live now | 20K |
| `files/MASTER_BACKLOG.md` | what's planned | 56K |
| `files/SYSTEM_BEHAVIOR.md` | runtime PM/RM/Trader behavior | 24K |
| `files/ML_EXPERIMENT_LOG.md` | full kill ledger (LARGE — skim) | 560K |
| `files/DECISIONS.md` | architectural/strategic decision log (LARGE — skim) | 132K |
| `files/PIPELINE_ARCHITECTURE.md` | WF/CPCV harness + gate inventory (LARGE — reference) | 148K |

## After responses come back
Drop each LLM's reply in a `responses/` subfolder (e.g. `responses/01_<model>.md`), then
synthesize into a consolidated plan (same flow as the prior `20260612_Alpha_v7_Review` pack).
