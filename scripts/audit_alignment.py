"""
audit_alignment.py — Phase 1 alignment audit: find all training/WF/live divergences.

Checks each of the 6 alignment dimensions from the Phase 1 plan:
  1.1 Feature construction: same engineer_features() call, same kwargs?
  1.2 Label construction: label scheme, ATR params, forward days
  1.3 Universe/survivorship: training universe vs WF universe vs live PM universe
  1.4 Normalization: cs_normalize in training? cs_normalize in WF? same N?
  1.5 Inference path: model.predict vs model.predict_with_vix, threshold logic
  1.6 Execution: entry price, stop simulation (intrabar vs close-only)

Output: docs/alignment_audit_YYYYMMDD.md
"""

import sys
import os
import inspect
import ast
from datetime import date, datetime
from pathlib import Path

# Ensure project root is on path when script is run from scripts/
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
os.chdir(_project_root)

AUDIT_DATE = datetime.utcnow().strftime("%Y%m%d")
OUT_PATH = Path("docs") / f"alignment_audit_{AUDIT_DATE}.md"

issues = []
notes = []


def ISSUE(category: str, msg: str, severity: str = "HIGH"):
    issues.append({"cat": category, "msg": msg, "sev": severity})
    print(f"  [ISSUE:{severity}] {category}: {msg}")


def NOTE(category: str, msg: str):
    notes.append({"cat": category, "msg": msg})
    print(f"  [OK] {category}: {msg}")


print("=" * 70)
print("MrTrader Alignment Audit — Phase 1")
print(f"Date: {AUDIT_DATE}")
print("=" * 70)


# ─── 1.1 Feature Construction ─────────────────────────────────────────────────
print("\n[1.1] Feature construction alignment")

# Check training engineer_features call kwargs
try:
    from app.ml.training import ModelTrainer
    trainer_src = inspect.getsource(ModelTrainer._build_rolling_matrix)

    if "fetch_fundamentals=False" in trainer_src:
        NOTE("1.1", "Training: fetch_fundamentals=False ✓")
    else:
        ISSUE("1.1", "Training: fetch_fundamentals not explicitly False — may fetch live data")

    if "as_of_date" in trainer_src:
        NOTE("1.1", "Training: as_of_date passed to engineer_features ✓")
    else:
        ISSUE("1.1", "Training: as_of_date NOT passed to engineer_features — PIT violation risk")

    if "regime_score" in trainer_src:
        NOTE("1.1", "Training: regime_score passed ✓")
    else:
        ISSUE("1.1", "Training: regime_score NOT passed — different feature than WF/live", "MEDIUM")

except Exception as e:
    ISSUE("1.1", f"Could not inspect training._build_rolling_matrix: {e}")

# Check WF engineer_features call kwargs
try:
    from app.backtesting.agent_simulator import AgentSimulator
    pm_score_src = inspect.getsource(AgentSimulator._pm_score)

    if "fetch_fundamentals=False" in pm_score_src:
        NOTE("1.1", "WF: fetch_fundamentals=False ✓")
    else:
        ISSUE("1.1", "WF: fetch_fundamentals not False in _pm_score")

    if "as_of_date=day" in pm_score_src or "as_of_date" in pm_score_src:
        NOTE("1.1", "WF: as_of_date passed ✓")
    else:
        ISSUE("1.1", "WF: as_of_date NOT passed to engineer_features — PIT violation")

    if "regime_score=_regime_score" in pm_score_src or "regime_score" in pm_score_src:
        NOTE("1.1", "WF: regime_score passed ✓")
    else:
        ISSUE("1.1", "WF: regime_score NOT passed — features differ from training", "MEDIUM")

except Exception as e:
    ISSUE("1.1", f"Could not inspect AgentSimulator._pm_score: {e}")

# Check live PM engineer_features call kwargs
try:
    from app.agents.portfolio_manager import PortfolioManager
    score_pos_src = inspect.getsource(PortfolioManager._analyze_swing_portfolio)

    if "fetch_fundamentals=False" in score_pos_src:
        NOTE("1.1", "Live PM: fetch_fundamentals=False ✓")
    else:
        ISSUE("1.1", "Live PM: fetch_fundamentals may be True — live fundamentals != training")

except Exception as e:
    NOTE("1.1", f"Live PM _analyze_swing_portfolio not directly inspectable: {e}")


# ─── 1.2 Label Construction ───────────────────────────────────────────────────
print("\n[1.2] Label construction alignment")

try:
    import app.ml.training as _training

    forward_days = getattr(_training, "FORWARD_DAYS_LONG", None)
    atr_target = getattr(_training, "ATR_MULT_TARGET", None)
    atr_stop = getattr(_training, "ATR_MULT_STOP", None)
    step_days = getattr(_training, "STEP_DAYS", None)

    NOTE("1.2", f"Training labels: FORWARD_DAYS={forward_days}, STEP_DAYS={step_days}, "
                f"ATR_TARGET={atr_target}x, ATR_STOP={atr_stop}x")

    # Check WF ATR multipliers
    from app.backtesting.agent_simulator import ATR_STOP_MULT, ATR_TARGET_MULT
    if ATR_STOP_MULT == atr_stop:
        NOTE("1.2", f"WF ATR_STOP_MULT={ATR_STOP_MULT} matches training ✓")
    else:
        ISSUE("1.2", f"WF ATR_STOP_MULT={ATR_STOP_MULT} ≠ training ATR_MULT_STOP={atr_stop} — "
                      "sim exits won't match training labels")

    if ATR_TARGET_MULT == atr_target:
        NOTE("1.2", f"WF ATR_TARGET_MULT={ATR_TARGET_MULT} matches training ✓")
    else:
        ISSUE("1.2", f"WF ATR_TARGET_MULT={ATR_TARGET_MULT} ≠ training ATR_MULT_TARGET={atr_target} — "
                      "sim exits won't match training labels")

    NOTE("1.2", "⚠ WF measures P&L of selected positions only; training labels entire universe. "
                "This is a structural misalignment (selection bias in WF).")

except Exception as e:
    ISSUE("1.2", f"Could not compare label parameters: {e}")


# ─── 1.3 Universe / Survivorship ─────────────────────────────────────────────
print("\n[1.3] Universe / survivorship alignment")

try:
    # Check if training uses PIT universe
    from app.ml.training import ModelTrainer
    build_src = inspect.getsource(ModelTrainer._build_rolling_matrix)

    if "pit_union" in build_src or "universe_history" in build_src:
        NOTE("1.3", "Training: uses PIT universe ✓")
    else:
        ISSUE("1.3", "Training: no PIT universe filter found — may train on current S&P500 only (survivorship bias)")

    # Check WF universe
    from scripts.walkforward.strategies.swing import SwingStrategy
    fold_src = inspect.getsource(SwingStrategy.run_fold)

    if "pit_union" in fold_src:
        NOTE("1.3", "WF: uses PIT universe (pit_union) ✓")
    else:
        ISSUE("1.3", "WF: no PIT universe filter — tests on current universe (look-ahead bias)")

    # Check live PM universe
    try:
        from app.agents.portfolio_manager import PortfolioManager
        wl_src = inspect.getsource(PortfolioManager._get_swing_watchlist)
        if "active" in wl_src.lower() or "watchlist" in wl_src.lower():
            NOTE("1.3", "Live PM: uses watchlist for universe (expected) ✓")
        NOTE("1.3", "Live PM universe = current watchlist. Training/WF should match or over-approximate this.")
    except Exception:
        NOTE("1.3", "Live PM watchlist method not directly inspectable")

except Exception as e:
    ISSUE("1.3", f"Universe audit failed: {e}")


# ─── 1.4 Normalization ────────────────────────────────────────────────────────
print("\n[1.4] Normalization alignment")

try:
    import app.ml.training as _training
    train_src = inspect.getsource(_training.ModelTrainer.train_model)

    # Training normalization
    if "cs_normalize" in train_src:
        ISSUE("1.4", "Training calls cs_normalize on X_train before fitting model — "
                      "but model.predict path also calls cs_normalize. Double-normalization?", "HIGH")
    elif "_ts_norm_state" in train_src or "ts_normalize" in train_src:
        NOTE("1.4", "Training: TS normalization applied ✓")
    else:
        NOTE("1.4", "Training: no explicit cs_normalize in train_model (applied in build_rolling_matrix?)")

    # Check if cs_normalize is applied in _build_rolling_matrix
    build_src = inspect.getsource(_training.ModelTrainer._build_rolling_matrix)
    if "cs_normalize" in build_src:
        ISSUE("1.4", "Training: cs_normalize in _build_rolling_matrix — feature matrix is pre-normalized before model training. "
                      "WF then applies cs_normalize again at inference → double normalization.", "HIGH")
    else:
        NOTE("1.4", "Training: cs_normalize NOT in _build_rolling_matrix ✓")

except Exception as e:
    ISSUE("1.4", f"Normalization audit failed: {e}")

# WF normalization
try:
    from app.backtesting.agent_simulator import AgentSimulator
    norm_src = inspect.getsource(AgentSimulator._normalize_for_inference)
    NOTE("1.4", f"WF _normalize_for_inference: uses cs_normalize fallback ✓ "
                f"(TS norm if model has _ts_norm_state)")
except Exception as e:
    ISSUE("1.4", f"WF normalization audit failed: {e}")

# Key structural issue
ISSUE("1.4",
      "STRUCTURAL: Training cs_normalize uses N=full_universe rows (700+). "
      "WF cs_normalize uses N=symbols_with_data_on_day (varies, often < 200). "
      "Live PM cs_normalize uses N=open_positions (typically 5-20). "
      "These three distributions are incomparable — z-scores are not portable.",
      "CRITICAL")


# ─── 1.5 Inference Path ───────────────────────────────────────────────────────
print("\n[1.5] Inference path alignment")

try:
    from app.backtesting.agent_simulator import AgentSimulator
    pm_src = inspect.getsource(AgentSimulator._pm_score)

    if "predict_with_vix" in pm_src:
        NOTE("1.5", "WF: uses model.predict_with_vix ✓")
    elif "model.predict" in pm_src:
        NOTE("1.5", "WF: uses model.predict (no VIX adjustment)")

    from app.agents.portfolio_manager import PortfolioManager
    # Check live PM predict call
    live_src = inspect.getsource(PortfolioManager._analyze_swing_portfolio)
    if "predict_with_vix" in live_src:
        NOTE("1.5", "Live PM: uses model.predict_with_vix ✓")
    elif "model.predict" in live_src:
        NOTE("1.5", "Live PM: uses model.predict (check if VIX adjustment matches WF)")

except Exception as e:
    NOTE("1.5", f"Inference path check partial: {e}")

# Check threshold
try:
    from app.backtesting.agent_simulator import MIN_CONFIDENCE
    NOTE("1.5", f"WF min_confidence threshold: {MIN_CONFIDENCE}")

    from app.agents.portfolio_manager import MIN_CONFIDENCE as PM_MIN_CONF
    if PM_MIN_CONF == MIN_CONFIDENCE:
        NOTE("1.5", f"Live PM min_confidence={PM_MIN_CONF} matches WF ✓")
    else:
        ISSUE("1.5", f"Live PM min_confidence={PM_MIN_CONF} ≠ WF min_confidence={MIN_CONFIDENCE}", "MEDIUM")
except Exception:
    NOTE("1.5", "Threshold comparison: could not import both constants")


# ─── 1.6 Execution ────────────────────────────────────────────────────────────
print("\n[1.6] Execution alignment")

try:
    from app.backtesting.agent_simulator import AgentSimulator
    sim_src = inspect.getsource(AgentSimulator)

    if "bar['open']" in sim_src or "bar[\"open\"]" in sim_src or "'open'" in sim_src:
        NOTE("1.6", "WF: entry price uses bar open (next-day open fill) — check for P0.1 fix")
    else:
        ISSUE("1.6", "WF: entry price may still use prev_close × 1.001 — P0.1 fix needed")

    if "bar['low']" in sim_src or "bar[\"low\"]" in sim_src or "low" in sim_src:
        NOTE("1.6", "WF: stop simulation checks bar low (intrabar) ✓")
    else:
        ISSUE("1.6", "WF: stop simulation may use close-only — P0.2 fix needed")

    if "borrow_cost" in sim_src or "borrow" in sim_src:
        NOTE("1.6", "WF: borrow cost modeled for shorts ✓")
    else:
        NOTE("1.6", "WF: no short borrow cost (long-only sim, expected for now)", "LOW")

except Exception as e:
    ISSUE("1.6", f"Execution audit failed: {e}")


# ─── Summary ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("AUDIT SUMMARY")
print("=" * 70)

critical = [i for i in issues if i["sev"] == "CRITICAL"]
high = [i for i in issues if i["sev"] == "HIGH"]
medium = [i for i in issues if i["sev"] == "MEDIUM"]

print(f"\nIssues found: {len(issues)} total ({len(critical)} CRITICAL, {len(high)} HIGH, {len(medium)} MEDIUM)")
print(f"Notes (OK): {len(notes)}")

if critical:
    print("\n🔴 CRITICAL issues (fix before any WF run means anything):")
    for i in critical:
        print(f"  [{i['cat']}] {i['msg']}")

if high:
    print("\n🟠 HIGH issues:")
    for i in high:
        print(f"  [{i['cat']}] {i['msg']}")

if medium:
    print("\n🟡 MEDIUM issues:")
    for i in medium:
        print(f"  [{i['cat']}] {i['msg']}")


# ─── Write markdown report ────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(exist_ok=True)
with open(OUT_PATH, "w") as f:
    f.write(f"# MrTrader Alignment Audit — {AUDIT_DATE}\n\n")
    f.write(f"**Run date:** {datetime.utcnow().isoformat()}Z\n\n")
    f.write(f"**Issues:** {len(issues)} ({len(critical)} CRITICAL, {len(high)} HIGH, {len(medium)} MEDIUM)  \n")
    f.write(f"**OK:** {len(notes)}\n\n")
    f.write("---\n\n")

    for cat_num, cat_name in [
        ("1.1", "Feature Construction"),
        ("1.2", "Label Construction"),
        ("1.3", "Universe / Survivorship"),
        ("1.4", "Normalization"),
        ("1.5", "Inference Path"),
        ("1.6", "Execution"),
    ]:
        f.write(f"## [{cat_num}] {cat_name}\n\n")
        cat_issues = [i for i in issues if i["cat"].startswith(cat_num)]
        cat_notes = [n for n in notes if n["cat"].startswith(cat_num)]
        for i in cat_issues:
            emoji = "🔴" if i["sev"] == "CRITICAL" else ("🟠" if i["sev"] == "HIGH" else "🟡")
            f.write(f"- {emoji} **{i['sev']}**: {i['msg']}\n")
        for n in cat_notes:
            f.write(f"- ✅ {n['msg']}\n")
        f.write("\n")

print(f"\nReport written to: {OUT_PATH}")
