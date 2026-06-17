"""P3-1 — crypto trend LIVE-PAPER tracker (report-only; NO capital, NO orders).

The P3-1 crypto-trend sleeve cleared its falsifiable criterion (Sharpe 0.64 / corr-to-trend
0.18) but is a PAPER-CANDIDATE, not a capital allocation (Track-B vs the trend book FAILs;
CAPITAL is power-floored by ~5y of history). The honest next step is to accrue an
OUT-OF-SAMPLE live-paper record so a future capital decision rests on forward data, not the
short backtest.

Because the sleeve is RULES-BASED and PIT (position[t] depends only on data <= t-1, applied
as held.shift(1)*rets), re-running its backtest on the latest live Alpaca closes and slicing
from an "inception" date forward yields a genuine forward OOS return stream — no execution,
no orders, no risk-cap interaction. This tracker freezes that OOS slice and reports its
Sharpe-to-date vs the 0.64 backtest expectation.

Enabled via pm.crypto_paper_enabled (default true). Runs weekly from the orchestrator.
Touches NO live trading code: it only reads market data and writes a parquet + a
decision_audit breadcrumb. Crypto trades 24/7 so there is no market-open gate.
"""
from __future__ import annotations

import logging
import os
from datetime import date as _date
from typing import Any, Dict, Optional

log = logging.getLogger(__name__)

CRYPTO_PAPER_PATH = os.path.join("data", "crypto_paper_track.parquet")
BACKTEST_SHARPE = 0.64          # the P3-1 standalone backtest expectation (365-ann)
ANN_CRYPTO = 365               # crypto trades 365 days/yr


def _truthy(val: Any) -> bool:
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _metrics(returns) -> Dict[str, float]:
    import numpy as np
    r = returns.dropna()
    if len(r) < 2:   # need >=2 points for a std; a 1-day OOS has no Sharpe/vol yet
        cum = float((1.0 + r).prod() - 1.0) if not r.empty else 0.0
        return {"sharpe": 0.0, "cum": cum, "ann_vol": 0.0, "n_days": int(len(r))}
    mu, sd = float(r.mean()), float(r.std())
    sharpe = float(mu / sd * np.sqrt(ANN_CRYPTO)) if sd > 0 else 0.0
    return {"sharpe": sharpe, "cum": float((1.0 + r).prod() - 1.0),
            "ann_vol": float(sd * np.sqrt(ANN_CRYPTO)), "n_days": int(len(r))}


def _load(path: str):
    import pandas as pd
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
        if "ret" not in df.columns or df.empty:
            return None
        s = df.set_index("date")["ret"] if "date" in df.columns else df["ret"]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception as exc:
        log.warning("crypto_paper: failed to load %s: %s", path, exc)
        return None


def _save(path: str, oos) -> None:
    import pandas as pd
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = pd.DataFrame({"date": pd.to_datetime(oos.index), "ret": oos.to_numpy()})
    out.to_parquet(path, index=False)


def _audit(n_days: int, sharpe: float) -> None:
    try:
        from app.database.decision_audit import write_decision
        write_decision(symbol="CRYPTO_BOOK", strategy="crypto_paper",
                       final_decision="track", price_at_decision=0.0,
                       block_reason=f"oos_n={n_days};oos_sharpe={sharpe:.3f}")
    except Exception:
        log.debug("crypto_paper: decision_audit write failed (swallowed)", exc_info=True)


def run_crypto_paper_track(db=None, *, force: bool = False) -> Dict[str, Any]:
    """Recompute the rules-based crypto-trend book on live Alpaca data and freeze the
    forward OOS slice (from inception). Report-only; never raises. Returns a summary dict."""
    summary: Dict[str, Any] = {"status": "ok", "enabled": None, "inception": None,
                               "n_oos_days": 0, "oos_sharpe": None, "oos_cum": None,
                               "backtest_sharpe": BACKTEST_SHARPE}
    own_db = db is None
    if own_db:
        from app.database.session import get_session
        db = get_session()
    try:
        from app.database.agent_config import get_agent_config
        enabled = _truthy(get_agent_config(db, "pm.crypto_paper_enabled"))
        summary["enabled"] = enabled
        if not enabled and not force:
            log.info("crypto_paper: DORMANT (pm.crypto_paper_enabled=false)")
            summary["status"] = "dormant"
            return summary

        # The rules-based crypto-trend book's daily NET returns on live Alpaca closes.
        # Imported lazily (app->scripts is used elsewhere, e.g. orchestrator NBBO logger).
        try:
            from scripts.walkforward.sleeves import crypto_trend_book_returns
            returns = crypto_trend_book_returns()
        except Exception as exc:
            log.warning("crypto_paper: could not compute crypto-trend returns: %s", exc)
            summary["status"] = "failed"
            summary["block_reason"] = "returns_unavailable"
            return summary
        if returns is None or returns.empty:
            summary["status"] = "failed"
            summary["block_reason"] = "no_returns"
            return summary

        # Inception = when we STARTED tracking (first run) — preserved across runs as the
        # min date of the stored OOS slice. We never back-date the OOS record into the
        # pre-tracking backtest; the clock starts the first time this runs.
        prior = _load(CRYPTO_PAPER_PATH)
        if prior is not None and not prior.empty:
            inception = prior.index.min()
        else:
            inception = returns.index.max()    # start the OOS clock now

        oos = returns[returns.index >= inception].dropna()
        _save(CRYPTO_PAPER_PATH, oos)
        m = _metrics(oos)
        summary.update(inception=str(inception.date()), n_oos_days=m["n_days"],
                       oos_sharpe=m["sharpe"], oos_cum=m["cum"], oos_ann_vol=m["ann_vol"])
        _audit(m["n_days"], m["sharpe"])
        log.info("crypto_paper: OOS n=%d sharpe=%.3f (backtest %.2f) since %s",
                 m["n_days"], m["sharpe"], BACKTEST_SHARPE, summary["inception"])
        return summary
    except Exception:
        log.exception("crypto_paper: run_crypto_paper_track failed (swallowed)")
        summary["status"] = "failed"
        return summary
    finally:
        if own_db:
            try:
                db.close()
            except Exception:
                pass


def weekly_email(summary: Optional[Dict[str, Any]] = None) -> None:
    """Best-effort weekly crypto live-paper email (event 'crypto_weekly')."""
    try:
        if summary is None:
            summary = run_crypto_paper_track()
        from app.notifications import notifier
        notifier.enqueue("crypto_weekly", {
            "week_ending": _date.today().strftime("%Y-%m-%d"),
            "inception": summary.get("inception"),
            "n_oos_days": summary.get("n_oos_days"),
            "oos_sharpe": summary.get("oos_sharpe"),
            "backtest_sharpe": summary.get("backtest_sharpe"),
            "oos_cum": summary.get("oos_cum"),
        }, dedup_key=f"crypto_weekly_{_date.today().strftime('%Y%W')}")
    except Exception:
        log.debug("crypto_paper: weekly email failed (swallowed)", exc_info=True)
