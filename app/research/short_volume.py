"""
short_volume.py — Alpha-v9 P3-5: aggregate short-volume market-timing signal.

Tests whether the FINRA daily AGGREGATE market short-volume ratio (sum ShortVolume /
sum TotalVolume across all NMS names) carries timing information for the broad market.
The economic prior (Boehmer-Jones-Zhang 2008; Diether-Lee-Werner 2009) is that short
flow is INFORMED -> elevated short selling predicts NEGATIVE forward returns. So the
PRE-REGISTERED direction is "de-risk when short-volume is elevated" (risk_off_when_high).

Why aggregate (not cross-sectional): the cross-sectional informed-short result is stronger,
but a per-name long/short test on free data is survivorship-biased (dead names dropped) —
exactly the contamination Norgate (P4-1) fixes. The AGGREGATE ratio is a market-wide sum
with no survivorship bias, so it is the clean free-data test; XS is deferred to post-Norgate.

The ratio has a strong secular uptrend (HFT/market-maker hedging is bona-fide "short
volume"), so the signal is a TRAILING Z-SCORE (detrended), never the raw level.

Frozen pre-registration (registration_id P3-5-SHORTVOL-AGG) — no sweeping, no direction-
flipping (the OPT-5 trap):
  * window = 63 trading days (~1 quarter), z_threshold = 1.0, cost = 1.0 bps/side
  * direction = risk_off_when_high (the literature's informed-short prior)
  * PASS iff the long-flat SPY overlay, net of cost, has annualized Sharpe >= 0.30 AND
    HAC one-sided p < 0.05 AND beats SPY buy-and-hold Sharpe; else KILL.
The opposite direction + forward-return-by-bucket are reported as DIAGNOSTICS only.

PIT: z_t uses the ratio through day t (knowable after t's close); the position it implies
is applied to day t+1's SPY return (position = raw.shift(1)).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

ANN = 252
REGISTRATION_ID = "P3-5-SHORTVOL-AGG"
Z_WINDOW = 63
Z_THRESHOLD = 1.0
COST_BPS = 1.0
PAPER_SR_FLOOR = 0.30
HAC_P_MAX = 0.05
# A STANDALONE edge (beyond timed SPY beta) additionally needs significant residual alpha
# vs SPY AND sub-period stability — the project's own rigor (Ruler-v2 residual-alpha + the
# F3 carry sub-period guard). The pre-registered PASS/KILL is NOT changed by these; they
# only decide the honest `standalone_edge` flag (PASS-but-timed-beta vs a real standalone).
INCR_ALPHA_T_MIN = 1.65            # ~one-sided 5% on the HAC alpha t


@dataclass(frozen=True)
class ShortVolumeVerdict:
    verdict: str               # "PASS" | "KILL" — the FROZEN pre-registered criterion
    reason: str
    registration_id: str
    n_days: int
    overlay_sharpe: float
    overlay_hac_p: float
    buyhold_sharpe: float
    opposite_overlay_sharpe: float       # diagnostic only (NOT used for the verdict)
    fwd_ret_by_z_tercile: Dict[str, float]   # diagnostic: mean next-day SPY ret by z bucket
    # Robustness layer (does the PASS reflect a STANDALONE edge or just timed SPY beta?):
    incr_alpha_ann: float = 0.0          # residual alpha vs SPY (annualized)
    incr_alpha_t: float = 0.0            # HAC t of that alpha
    beta_spy: float = 0.0
    h1_delta: float = 0.0                # overlay-minus-buyhold Sharpe, first half
    h2_delta: float = 0.0                # overlay-minus-buyhold Sharpe, second half
    standalone_edge: bool = False        # PASS AND significant alpha AND sub-period stable
    params: Dict[str, float] = field(default_factory=dict)


def trailing_zscore(s: pd.Series, window: int) -> pd.Series:
    mu = s.rolling(window, min_periods=window).mean()
    sd = s.rolling(window, min_periods=window).std()
    z = (s - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


def _sharpe(r: pd.Series, ann: int = ANN) -> float:
    r = r.dropna()
    if len(r) < 2:
        return 0.0
    sd = float(r.std())
    return float(r.mean() / sd * np.sqrt(ann)) if sd > 0 else 0.0


def overlay_returns(ratio: pd.Series, spy_close: pd.Series, *, window: int = Z_WINDOW,
                    z_threshold: float = Z_THRESHOLD, cost_bps: float = COST_BPS,
                    direction: str = "risk_off_when_high") -> pd.DataFrame:
    """Long-flat SPY overlay driven by the short-volume z-score. Returns a frame with
    columns [spy_ret, position, net] (net = position.shift(1)*spy_ret - turnover cost)."""
    df = pd.DataFrame({"ratio": ratio.astype(float), "close": spy_close.astype(float)})
    df = df.dropna().sort_index()
    z = trailing_zscore(df["ratio"], window)
    spy_ret = df["close"].pct_change()
    if direction == "risk_off_when_high":
        raw = (z <= z_threshold).astype(float)        # flat when shorting is elevated
    elif direction == "risk_on_when_high":
        raw = (z >= z_threshold).astype(float)         # the opposite (diagnostic)
    else:
        raise ValueError(f"unknown direction: {direction}")
    position = raw.shift(1)                             # signal known at t -> trade t+1
    gross = position * spy_ret
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cost_bps / 1e4)
    net = (gross - cost).rename("net")
    out = pd.DataFrame({"spy_ret": spy_ret, "position": position, "net": net})
    return out.dropna(subset=["net"])


def _fwd_return_by_tercile(ratio: pd.Series, spy_close: pd.Series,
                           window: int = Z_WINDOW) -> Dict[str, float]:
    """Diagnostic: mean NEXT-DAY SPY return conditional on the short-vol z tercile.
    The informed-short prior predicts high-z -> lower forward return."""
    df = pd.DataFrame({"ratio": ratio.astype(float), "close": spy_close.astype(float)})
    df = df.dropna().sort_index()
    z = trailing_zscore(df["ratio"], window)
    fwd = df["close"].pct_change().shift(-1)            # return from t -> t+1
    d = pd.DataFrame({"z": z, "fwd": fwd}).dropna()
    if len(d) < 30:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    try:
        d["bucket"] = pd.qcut(d["z"], 3, labels=["low", "mid", "high"])
    except ValueError:
        return {"low": float("nan"), "mid": float("nan"), "high": float("nan")}
    means = d.groupby("bucket", observed=True)["fwd"].mean()
    return {k: float(means.get(k, float("nan"))) for k in ("low", "mid", "high")}


def short_volume_verdict(ratio: pd.Series, spy_close: pd.Series, *, window: int = Z_WINDOW,
                         z_threshold: float = Z_THRESHOLD, cost_bps: float = COST_BPS,
                         paper_sr_floor: float = PAPER_SR_FLOOR,
                         hac_p_max: float = HAC_P_MAX, ann: int = ANN) -> ShortVolumeVerdict:
    """Run the frozen pre-registered aggregate-short-volume timing test → PASS/KILL."""
    from app.research.inference import hac_sharpe, multifactor_alpha

    main = overlay_returns(ratio, spy_close, window=window, z_threshold=z_threshold,
                           cost_bps=cost_bps, direction="risk_off_when_high")
    opp = overlay_returns(ratio, spy_close, window=window, z_threshold=z_threshold,
                          cost_bps=cost_bps, direction="risk_on_when_high")
    n = int(len(main))
    net, spyr = main["net"], main["spy_ret"]
    overlay_sr = _sharpe(net, ann)
    buyhold_sr = _sharpe(spyr, ann)
    opp_sr = _sharpe(opp["net"], ann)
    hac = hac_sharpe(net.to_numpy(), annualize=ann)
    fwd = _fwd_return_by_tercile(ratio, spy_close, window)

    # ── frozen pre-registered PASS/KILL (unchanged) ──
    c1 = overlay_sr >= paper_sr_floor
    c2 = hac.p_one_sided < hac_p_max
    c3 = overlay_sr > buyhold_sr
    pre_pass = c1 and c2 and c3

    # ── robustness layer: standalone edge vs timed SPY beta ──
    a = multifactor_alpha(net, pd.DataFrame({"SPY": spyr}))
    incr_alpha_ann = float(a.get("alpha_ann", 0.0) or 0.0)
    incr_alpha_t = float(a.get("t_alpha_hac", 0.0) or 0.0)
    beta_spy = float((a.get("betas") or {}).get("SPY", 0.0) or 0.0)
    if n >= 4:
        mid = net.index[n // 2]
        h1 = net.index <= mid
        h2 = net.index > mid
        h1_delta = _sharpe(net[h1], ann) - _sharpe(spyr[h1], ann)
        h2_delta = _sharpe(net[h2], ann) - _sharpe(spyr[h2], ann)
    else:
        h1_delta = h2_delta = 0.0
    subperiod_stable = (h1_delta > 0) and (h2_delta > 0)
    standalone_edge = bool(pre_pass and incr_alpha_t >= INCR_ALPHA_T_MIN and subperiod_stable)

    if not pre_pass:
        verdict = "KILL"
        fails = []
        if not c1:
            fails.append(f"Sharpe {overlay_sr:.2f} < {paper_sr_floor:.2f}")
        if not c2:
            fails.append(f"HAC p {hac.p_one_sided:.3f} >= {hac_p_max} (not significant)")
        if not c3:
            fails.append(f"does not beat buy-hold ({overlay_sr:.2f} <= {buyhold_sr:.2f})")
        reason = "aggregate short-vol timing fails the pre-registered test: " + "; ".join(fails)
    elif standalone_edge:
        verdict = "PASS"
        reason = (f"PASS + STANDALONE: overlay Sharpe {overlay_sr:.2f} beats buy-hold "
                  f"{buyhold_sr:.2f}; residual alpha {incr_alpha_ann:+.2%} (t {incr_alpha_t:.2f}) "
                  f"significant; sub-period stable (H1 {h1_delta:+.2f}, H2 {h2_delta:+.2f})")
    else:
        # Pre-registered test passes, but it is NOT a robust standalone edge — co-headline.
        verdict = "PASS"
        why = []
        if incr_alpha_t < INCR_ALPHA_T_MIN:
            why.append(f"residual alpha vs SPY insignificant (t {incr_alpha_t:.2f}); "
                       f"mostly timed beta (beta {beta_spy:.2f})")
        if not subperiod_stable:
            why.append(f"not sub-period stable (H1 {h1_delta:+.2f}, H2 {h2_delta:+.2f})")
        reason = ("PASS on the pre-registered test (Sharpe {:.2f} > buy-hold {:.2f}, HAC p {:.3f}) "
                  "but REAL-BUT-WEAK, not a standalone edge: ".format(
                      overlay_sr, buyhold_sr, hac.p_one_sided) + "; ".join(why)
                  + " -> route to the P3-4 composite + cross-sectional post-Norgate.")

    return ShortVolumeVerdict(
        verdict=verdict, reason=reason, registration_id=REGISTRATION_ID, n_days=n,
        overlay_sharpe=overlay_sr, overlay_hac_p=float(hac.p_one_sided),
        buyhold_sharpe=buyhold_sr, opposite_overlay_sharpe=opp_sr,
        fwd_ret_by_z_tercile=fwd, incr_alpha_ann=incr_alpha_ann, incr_alpha_t=incr_alpha_t,
        beta_spy=beta_spy, h1_delta=h1_delta, h2_delta=h2_delta,
        standalone_edge=standalone_edge,
        params={"window": float(window), "z_threshold": float(z_threshold),
                "cost_bps": float(cost_bps)})
