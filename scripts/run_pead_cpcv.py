"""
Phase G: CPCV validation of the PEAD scorer.

Combinatorial Purged Cross-Validation (k=8, paths=2). Higher statistical power
than standard 5-fold WF; catches overfitting to specific fold boundaries.

PEADStrategy is now a thin subclass of the reusable EventEdgeStrategy (Alpha-v3
A0, scripts/walkforward/event_edge.py) — it inherits the generic fetch_data /
run_fold and overrides only the PEAD-specific AgentSimulator kwargs, so this run
stays byte-identical to the committed long-only +0.546 config.

Usage:
    python scripts/run_pead_cpcv.py
"""
import logging
import os
import sys
from datetime import datetime, timedelta

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

from app.ml.retrain_config import MAX_THREADS as _max_threads
os.environ.setdefault("OMP_NUM_THREADS", str(_max_threads))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(_max_threads))

from scripts.walkforward.event_edge import EventEdgeStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [pead_cpcv] %(message)s",
)
logger = logging.getLogger(__name__)

CPCV_K = 8       # number of fold groups
CPCV_PATHS = 2   # test groups per combination → C(8,2)=28 paths
TOTAL_YEARS = 6


class PEADStrategy(EventEdgeStrategy):
    """PEAD adapter: EventEdgeStrategy with the PEAD-specific AgentSimulator knobs.

    Inherits the generic fetch_data / run_fold. Overrides only `_fold_sim_kwargs`
    to reproduce the committed run exactly: no_prefilters + env-driven hold cap
    (PEAD_MAX_HOLD_BARS) + conviction sizing (PEAD_CONVICTION_SIZE) + optional
    §1.1 slippage overrides. Constructor signature is unchanged (tests rely on it).
    """

    model_type = "pead"
    pit_index = "russell1000"
    pit_trade_type = "swing"

    def __init__(self, scorer, symbols, transaction_cost_pct=0.0005,
                 entry_slippage_pct=None, stop_slippage_pct=None):
        super().__init__(
            scorer, symbols,
            transaction_cost_pct=transaction_cost_pct,
            entry_slippage_pct=entry_slippage_pct,
            stop_slippage_pct=stop_slippage_pct,
            no_prefilters=True,
        )

    def _fold_sim_kwargs(self, tr_start, te_start) -> dict:
        # Hold-period lever: PEAD positions exit via max_hold_bars (default 40 ≈ 8wk,
        # which over-stays and gives back drift). PEAD_MAX_HOLD_BARS env caps it
        # (e.g. 15 ≈ 3wk). Unset → AgentSimulator default (None == 40).
        _hold = (int(os.environ["PEAD_MAX_HOLD_BARS"])
                 if os.environ.get("PEAD_MAX_HOLD_BARS") else None)
        # PEAD_CONVICTION_SIZE=1 → size new long entries by clip(SUE_z,0,3)/realized_vol,
        # gross-normalized to the equal-weight book (same names/gross, only weights).
        # Default OFF → committed equal-weight +0.546 config is byte-identical.
        _conviction = os.environ.get("PEAD_CONVICTION_SIZE") == "1"
        kw = {
            "no_prefilters": True,
            "max_hold_bars_override": _hold,
            "pead_conviction_size": _conviction,
        }
        # §1.1: only pass slippage when explicitly overridden, so the default run
        # uses AgentSimulator's module-constant defaults (byte-identical +0.546).
        if self.entry_slippage_pct is not None:
            kw["entry_slippage_pct"] = self.entry_slippage_pct
        if self.stop_slippage_pct is not None:
            kw["stop_slippage_pct"] = self.stop_slippage_pct
        return kw


def build_pead_scorer():
    """Construct the validated PEAD scorer (the committed +0.546 long-only config).

    Extracted so the §1.1 cost-sensitivity sweep reuses the EXACT same scorer
    config — the only thing the sweep varies is transaction cost / slippage.
    Honors the same PEAD_LONG_SHORT / PEAD_QUALITY_GATE env switches as main().
    """
    from app.ml.pead_scorer import PEADScorer

    # Best config found: long-only, no VIX gate, no priced-in filter.
    # CPCV campaign (4 runs): Run 3 (this config) achieved mean=0.349 — the best result.
    # Priced-in filter (Run 4) hurt: large announce-day gaps have strongest drift continuation.
    # Default = best long-only config (+0.546 honest CPCV). PEAD_LONG_SHORT=1 enables
    # the short leg (long positive-surprise + short negative-surprise) with a VIX>20
    # squeeze-guard on shorts — the untried dollar-neutral test.
    _ls = os.environ.get("PEAD_LONG_SHORT") == "1"
    # Earnings-quality split (last high-EV lever): PEAD_QUALITY_GATE=1 requires a
    # long signal to be "EPS beat + analysts revising up" (positive analyst-revision
    # momentum as-of the scoring day) rather than a bare beat. Higher-conviction
    # drift, fewer trades. PIT-safe (analyst feature windowed to <= scoring day).
    # Default OFF → committed long-only +0.546 config is unchanged.
    _quality_gate = os.environ.get("PEAD_QUALITY_GATE") == "1"
    # B5: regime-general trend control as a REPLACEMENT for the discrete VIX>30 block.
    # PEAD_REGIME_CONTROL=trend (or vol_target) turns the VIX block off (inf) and lets
    # the regime control govern de-risking instead — the validation harness for B5.
    # Default unset → committed +0.546 VIX-block config is byte-identical.
    _regime = os.environ.get("PEAD_REGIME_CONTROL") or None
    _vix_block = float("inf") if _regime else 30.0
    scorer = PEADScorer(
        long_threshold=0.05,
        short_threshold=-0.05,
        long_short=_ls,
        vix_block_all=_vix_block,
        vix_block_short=(20.0 if _ls else 100.0),  # squeeze guard when short leg active
        vix_conf_ref=100.0,
        max_announce_day_move=1.0,  # disabled — large gaps retain drift signal
        require_positive_revision=_quality_gate,
        min_analyst_momentum=0.0,
        regime_control=_regime,
        regime_control_trend_ma=int(os.environ.get("PEAD_TREND_MA", "200")),
        regime_control_floor=float(os.environ.get("PEAD_REGIME_FLOOR", "0.5")),
    )
    if _regime:
        logger.info("PEAD_REGIME_CONTROL=%s (VIX block OFF, trend_ma=%s, floor=%s) - B5 validation",
                    _regime, os.environ.get("PEAD_TREND_MA", "200"),
                    os.environ.get("PEAD_REGIME_FLOOR", "0.5"))
    if _quality_gate:
        logger.info("PEAD_QUALITY_GATE=1 - long signals require positive analyst revision")
    if os.environ.get("PEAD_CONVICTION_SIZE") == "1":
        logger.info(
            "PEAD_CONVICTION_SIZE=1 - long entries weighted by clip(SUE_z,0,3)/realized_vol, "
            "gross-normalized to the equal-weight book (same entry set, no leverage confound)"
        )
    return scorer


def main() -> int:
    from app.utils.constants import RUSSELL_1000_TICKERS
    from scripts.walkforward.cpcv import run_cpcv

    scorer = build_pead_scorer()
    strategy = PEADStrategy(
        scorer=scorer,
        symbols=list(RUSSELL_1000_TICKERS),
        transaction_cost_pct=0.0005,
    )

    end_all = datetime.now()
    start_all = end_all - timedelta(days=TOTAL_YEARS * 365 + 30)
    strategy.fetch_data(start_all, end_all)

    logger.info("Running PEAD CPCV: k=%d paths=%d -> C(%d,%d)=%d test paths",
                CPCV_K, CPCV_PATHS, CPCV_K, CPCV_PATHS,
                len(list(__import__("itertools").combinations(range(CPCV_K), CPCV_PATHS))))

    result = run_cpcv(
        strategy=strategy,
        purge_days=10,
        embargo_days=10,
        n_folds=CPCV_K,
        n_paths=CPCV_PATHS,
        total_years=TOTAL_YEARS,
    )

    result.print()

    gate_detail = result.gate_detail()
    gate_ok = all(v for _, v in gate_detail.values())
    verdict = "CPCV GATE PASSED" if gate_ok else "CPCV GATE FAILED"
    logger.info(
        "PEAD CPCV %s - mean_sharpe=%.3f  p5=%.3f  p95=%.3f",
        verdict, result.mean_sharpe, result.p5_sharpe, result.p95_sharpe,
    )

    if os.environ.get("PEAD_NO_EMAIL") == "1":
        logger.info("PEAD_NO_EMAIL=1 - skipping email (validation run)")
        return 0 if gate_ok else 2

    try:
        from app.notifications.notifier import _smtp_send
        _smtp_send(
            subject=f"MrTrader PEAD CPCV: {verdict} (mean={result.mean_sharpe:.3f})",
            html_body=f"""
<h2>PEAD CPCV Result</h2>
<p><b>{verdict}</b></p>
<ul>
  <li>Mean Sharpe: {result.mean_sharpe:.3f} (gate: ≥0.80)</li>
  <li>P5 Sharpe: {result.p5_sharpe:.3f} (gate: ≥-0.30)</li>
  <li>P95 Sharpe: {result.p95_sharpe:.3f}</li>
  <li>Gate detail: {gate_detail}</li>
  <li>N paths: {len(result.path_sharpes)}</li>
</ul>
""",
        )
    except Exception as _e:
        logger.warning("Email notification failed: %s", _e)

    return 0 if gate_ok else 2


if __name__ == "__main__":
    sys.exit(main())
