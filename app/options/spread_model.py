"""P2-4 — calibrated, MONEYNESS/DTE-aware option spread cost model.

The OPT-2 simulator charges option transaction cost as a HALF-SPREAD fraction of PREMIUM
(`OptionsSpreadCostModel`, default a flat 1%). Because cost is a fraction of premium it is
already IV-SENSITIVE through the premium base; what flat-1% misses is the SPREAD STRUCTURE: a
liquid SPY ATM weekly trades ~1-2 bps of premium while a deep-OTM single-name 60-DTE put can
trade tens of percent. P2-4 replaces the flat number with an EMPIRICAL spread surface
calibrated from the live NBBO observation log (`data/options_spread_obs.parquet`, P1c fuse).
(Note: the surface is conditioned on underlying/moneyness/DTE — NOT on IV directly; the feed's
IV is ~30% NaN, so an explicit IV dimension is deferred. "IV-aware" in the roadmap is satisfied
by the premium-% base, not by IV bucketing.)

Approach (deliberately non-parametric, robust to thin data):
  * Bucket observed RELATIVE spread = (ask-bid)/mid by (underlying, contract_type, moneyness,
    DTE) and take the MEDIAN per bucket (median, not mean — spreads are heavy-tailed).
  * HIERARCHICAL FALLBACK for sparse buckets: per-underlying -> panel -> moneyness-marginal ->
    DTE-marginal -> conservative global, requiring MIN_OBS samples at each level. Crucially the
    last-resort global is a CONSERVATIVE percentile (not the median): a contract that falls all
    the way through is one the surface has no real coverage for, so it should be costed HIGH
    (erring toward killing a marginal edge, never inventing a phantom one).
  * PIT calibration: `calibrate(..., as_of=d)` uses only obs with obs_date <= d.

IMPORTANT — half-spread convention: the observation `spread_pct` is the FULL relative spread
(ask-bid)/mid. Crossing from mid costs HALF of that per side, matching the simulator's
`spread_pct`. So `half_spread_pct()` returns predicted_full / 2.

⚠️ ANACHRONISM — READ BEFORE USING FOR ANY VERDICT. The NBBO log currently spans only days
(2026-06, calm-tape, indicative feed). The surface is a LEVEL snapshot, NOT regime-conditioned:
applying it to multi-year backtest history understates cost in stressed periods (exactly when a
short-vol/VRP book loses), biasing a backtest CHEAPER than reality. `covers_date()` reports
whether a date is inside the calibration window; callers must guard. This PR ships the
FRAMEWORK + a PRELIMINARY calibration and is NOT wired into any live VRP go/no-go — re-run
`scripts/calibrate_option_spreads.py` as data accrues; a VRP verdict needs a mature surface
spanning the test window + live-paper validation. The 1x/2x/3x stress multiplier still applies.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Bucket edges. Moneyness = strike / underlying_price (bucketed per contract_type so call vs
# put OTM/ITM asymmetry is preserved). DTE in calendar days (obs range observed 0-70).
MONEYNESS_EDGES: Tuple[float, ...] = (0.80, 0.90, 0.95, 0.975, 1.025, 1.05, 1.10, 1.20)
DTE_EDGES: Tuple[int, ...] = (7, 21, 45, 90)
MIN_OBS = 30                 # min samples for a bucket/marginal to be trusted before fallback
SPREAD_PCT_CAP = 2.0         # clip absurd relative spreads (illiquid quotes) before median
DEFAULT_FULL_SPREAD = 0.04   # last-resort full relative spread if calibration is empty (=2% half)
# distinct trading days before the surface is trustworthy for a verdict (~4 weeks; below this
# the calibration is PRELIMINARY / anachronistic).
MATURE_MIN_DAYS = 20


def _bin_index(value: float, edges: Tuple) -> int:
    """Half-open bin index: edges define len(edges)+1 buckets [.. e0)|[e0,e1)|..|[e_last,..)."""
    i = 0
    for e in edges:
        if value < e:
            return i
        i += 1
    return i


def moneyness_bin(moneyness: float) -> int:
    return _bin_index(float(moneyness), MONEYNESS_EDGES)


def dte_bin(dte: float) -> int:
    return _bin_index(float(dte), DTE_EDGES)


@dataclass
class CalibratedSpreadModel:
    """Empirical relative-spread surface with hierarchical fallback. `predict_full_spread_pct`
    returns the FULL relative spread (ask-bid)/mid; `half_spread_pct` returns half of it (what
    the cost model charges per side)."""
    # Per-UNDERLYING fine surface "UND|ctype|mbin|dbin" -> median full spread (so a liquid
    # SPY ATM is not charged single-name-illiquid spreads — load-bearing for VRP on SPY/QQQ/IWM).
    und_buckets: Dict[str, float] = field(default_factory=dict)
    und_bucket_n: Dict[str, int] = field(default_factory=dict)
    # Panel-wide (all underlyings pooled) "ctype|mbin|dbin" -> median full spread; + marginals.
    buckets: Dict[str, float] = field(default_factory=dict)
    bucket_n: Dict[str, int] = field(default_factory=dict)
    moneyness_marginal: Dict[str, float] = field(default_factory=dict)   # "ctype|mbin" -> median
    moneyness_marginal_n: Dict[str, int] = field(default_factory=dict)
    dte_marginal: Dict[str, float] = field(default_factory=dict)         # "ctype|dbin" -> median
    dte_marginal_n: Dict[str, int] = field(default_factory=dict)
    global_median: float = DEFAULT_FULL_SPREAD       # pooled median (reporting / context-present)
    conservative_global: float = DEFAULT_FULL_SPREAD  # pooled p75 — the "no coverage" last resort
    calibrated_from: Optional[str] = None            # min obs_date (anachronism guard)
    calibrated_through: Optional[str] = None          # max obs_date
    n_obs: int = 0
    n_days: int = 0                                   # distinct trading days in the obs window

    @property
    def is_mature(self) -> bool:
        """True once the obs window spans >= MATURE_MIN_DAYS distinct trading days — below
        this the surface is PRELIMINARY and must not drive a VRP verdict (see module docstring)."""
        return self.n_days >= MATURE_MIN_DAYS

    def covers_date(self, d) -> bool:
        """True iff `d` is inside the calibration window [calibrated_from, calibrated_through].
        Callers MUST guard backtest trades: applying this surface outside the window is
        anachronistic (see module docstring)."""
        if not self.calibrated_from or not self.calibrated_through or self.n_obs == 0:
            return False
        ds = (d.isoformat() if isinstance(d, date) else str(d))[:10]
        return self.calibrated_from <= ds <= self.calibrated_through

    def predict_full_spread_pct(self, moneyness: Optional[float], dte: Optional[float],
                                contract_type: Optional[str],
                                underlying: Optional[str] = None) -> float:
        """Best available FULL relative spread for the contract, falling back coarser:
        per-underlying bucket -> panel bucket -> moneyness-marginal -> DTE-marginal -> global.
        Returns the global/default if context is missing."""
        if (moneyness is None or dte is None
                or not (moneyness == moneyness) or not (dte == dte)):  # None / NaN (either)
            return self.conservative_global  # missing context -> cost HIGH, never optimistic
        ct = (contract_type or "any").lower()
        mb, db = moneyness_bin(moneyness), dte_bin(dte)
        # 1. per-underlying fine bucket (the liquid-name accuracy that matters for VRP)
        if underlying:
            uk = f"{str(underlying).upper()}|{ct}|{mb}|{db}"
            if self.und_bucket_n.get(uk, 0) >= MIN_OBS:
                return self.und_buckets[uk]
        # 2. panel-wide bucket (all underlyings pooled at this moneyness/DTE)
        k = f"{ct}|{mb}|{db}"
        if self.bucket_n.get(k, 0) >= MIN_OBS:
            return self.buckets[k]
        # 3. moneyness marginal (collapse DTE)
        mk = f"{ct}|{mb}"
        if self.moneyness_marginal_n.get(mk, 0) >= MIN_OBS:
            return self.moneyness_marginal[mk]
        # 4. DTE marginal (collapse moneyness)
        dk = f"{ct}|{db}"
        if self.dte_marginal_n.get(dk, 0) >= MIN_OBS:
            return self.dte_marginal[dk]
        # 5. no coverage at any level -> CONSERVATIVE global (p75), not the median: the
        # contracts that reach here are the ones the surface can't price, so cost them high.
        return self.conservative_global

    def half_spread_pct(self, moneyness: Optional[float], dte: Optional[float],
                        contract_type: Optional[str], underlying: Optional[str] = None) -> float:
        """Half of the full relative spread — the fraction of premium crossed per side."""
        return self.predict_full_spread_pct(moneyness, dte, contract_type, underlying) / 2.0

    # ── persistence ──
    def to_dict(self) -> Dict[str, Any]:
        return {
            "und_buckets": self.und_buckets, "und_bucket_n": self.und_bucket_n,
            "buckets": self.buckets, "bucket_n": self.bucket_n,
            "moneyness_marginal": self.moneyness_marginal,
            "moneyness_marginal_n": self.moneyness_marginal_n,
            "dte_marginal": self.dte_marginal, "dte_marginal_n": self.dte_marginal_n,
            "global_median": self.global_median, "conservative_global": self.conservative_global,
            "calibrated_from": self.calibrated_from, "calibrated_through": self.calibrated_through,
            "n_obs": self.n_obs, "n_days": self.n_days,
            "moneyness_edges": list(MONEYNESS_EDGES), "dte_edges": list(DTE_EDGES),
            "min_obs": MIN_OBS, "schema": "p2-4-spread-v1",
        }

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2, default=str))

    @classmethod
    def load(cls, path: str | Path) -> "CalibratedSpreadModel":
        d = json.loads(Path(path).read_text())
        # Schema/edge guard (M3): the saved bin INDICES are meaningless if the edges changed
        # in code since the artifact was written — refuse rather than silently mis-look-up.
        if (d.get("schema") != "p2-4-spread-v1"
                or d.get("moneyness_edges") != list(MONEYNESS_EDGES)
                or d.get("dte_edges") != list(DTE_EDGES)
                or d.get("min_obs") != MIN_OBS):
            raise ValueError(
                "spread model artifact was built with different bin edges/schema "
                f"({path}); re-run scripts/calibrate_option_spreads.py")
        return cls(
            und_buckets={k: float(v) for k, v in d.get("und_buckets", {}).items()},
            und_bucket_n={k: int(v) for k, v in d.get("und_bucket_n", {}).items()},
            buckets={k: float(v) for k, v in d.get("buckets", {}).items()},
            bucket_n={k: int(v) for k, v in d.get("bucket_n", {}).items()},
            moneyness_marginal={k: float(v) for k, v in d.get("moneyness_marginal", {}).items()},
            moneyness_marginal_n={k: int(v) for k, v in d.get("moneyness_marginal_n", {}).items()},
            dte_marginal={k: float(v) for k, v in d.get("dte_marginal", {}).items()},
            dte_marginal_n={k: int(v) for k, v in d.get("dte_marginal_n", {}).items()},
            global_median=float(d.get("global_median", DEFAULT_FULL_SPREAD)),
            conservative_global=float(d.get("conservative_global",
                                            d.get("global_median", DEFAULT_FULL_SPREAD))),
            calibrated_from=d.get("calibrated_from"),
            calibrated_through=d.get("calibrated_through"),
            n_obs=int(d.get("n_obs", 0)),
            n_days=int(d.get("n_days", 0)),
        )

    def summary(self) -> str:
        lines = [
            f"CalibratedSpreadModel  n_obs={self.n_obs}  through={self.calibrated_through}",
            f"  global median full spread = {self.global_median:.4f} "
            f"({self.global_median/2*100:.2f}% half / premium)",
            f"  per-underlying buckets={len(self.und_buckets)}  panel buckets={len(self.buckets)}"
            f"  moneyness-marginals={len(self.moneyness_marginal)} dte-marginals={len(self.dte_marginal)}",
        ]
        und_trusted = sorted((k for k in self.und_buckets if self.und_bucket_n.get(k, 0) >= MIN_OBS),
                             key=lambda x: self.und_bucket_n[x], reverse=True)[:8]
        if und_trusted:
            lines.append("  top per-underlying buckets:")
            for k in und_trusted:
                lines.append(f"    {k:24} full={self.und_buckets[k]:.4f}  n={self.und_bucket_n[k]}")
        trusted = {k: v for k, v in self.buckets.items() if self.bucket_n.get(k, 0) >= MIN_OBS}
        for k in sorted(trusted, key=lambda x: self.bucket_n[x], reverse=True)[:12]:
            lines.append(f"    {k:18} full={trusted[k]:.4f}  n={self.bucket_n[k]}")
        return "\n".join(lines)


def _median(values: List[float]) -> float:
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _percentile(values: List[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]). Used for the conservative global fallback."""
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, n - 1)
    return s[lo] + (s[hi] - s[lo]) * frac


def calibrate(obs_df, *, as_of: Optional[date] = None) -> CalibratedSpreadModel:
    """Build a CalibratedSpreadModel from the NBBO observation frame. Pure (no I/O).

    Requires columns: contract_type, moneyness, dte, spread_pct (and bid/ask/mid if present
    for sanity filtering). PIT: keep only obs_date <= as_of when as_of is given.
    """
    import math

    df = obs_df
    if as_of is not None and "obs_date" in df.columns:
        cutoff = as_of.isoformat() if isinstance(as_of, date) else str(as_of)
        df = df[df["obs_date"].astype(str).str[:10] <= cutoff]

    # Collect cleaned (underlying, ctype, mbin, dbin, spread_full) rows.
    full: List[Tuple[str, str, int, int, float]] = []
    cols = df.columns
    has_ba = "bid" in cols and "ask" in cols and "mid" in cols
    for row in df.itertuples(index=False):
        d = row._asdict() if hasattr(row, "_asdict") else dict(zip(cols, row))
        sp = d.get("spread_pct")
        m = d.get("moneyness")
        dte = d.get("dte")
        ct = str(d.get("contract_type") or "any").lower()
        und = str(d.get("underlying") or "").upper()
        try:
            sp = float(sp)
            m = float(m)
            dte = float(dte)
        except (TypeError, ValueError):
            continue
        if not (math.isfinite(sp) and math.isfinite(m) and math.isfinite(dte)):
            continue
        # Keep locked markets (sp == 0, ask==bid) — they are real tight quotes on the most
        # liquid names; dropping them biases the liquid-bucket median UP. Only drop crossed.
        if sp < 0.0 or m <= 0.0 or dte < 0:
            continue
        if has_ba:  # require a real two-sided quote
            try:
                b, a, mid = float(d["bid"]), float(d["ask"]), float(d["mid"])
            except (TypeError, ValueError, KeyError):
                b = a = mid = 0.0
            if not (b > 0 and a >= b and mid > 0):
                continue
        sp = min(sp, SPREAD_PCT_CAP)
        full.append((und, ct, moneyness_bin(m), dte_bin(dte), sp))

    if not full:
        return CalibratedSpreadModel(
            calibrated_through=(as_of.isoformat() if isinstance(as_of, date)
                                else (str(as_of) if as_of else None)),
            n_obs=0)

    by_und: Dict[str, List[float]] = {}
    by_bucket: Dict[str, List[float]] = {}
    by_money: Dict[str, List[float]] = {}
    by_dte: Dict[str, List[float]] = {}
    all_sp: List[float] = []
    for und, ct, mb, db, sp in full:
        if und:
            by_und.setdefault(f"{und}|{ct}|{mb}|{db}", []).append(sp)
        by_bucket.setdefault(f"{ct}|{mb}|{db}", []).append(sp)
        by_money.setdefault(f"{ct}|{mb}", []).append(sp)
        by_dte.setdefault(f"{ct}|{db}", []).append(sp)
        all_sp.append(sp)

    model = CalibratedSpreadModel(
        und_buckets={k: _median(v) for k, v in by_und.items()},
        und_bucket_n={k: len(v) for k, v in by_und.items()},
        buckets={k: _median(v) for k, v in by_bucket.items()},
        bucket_n={k: len(v) for k, v in by_bucket.items()},
        moneyness_marginal={k: _median(v) for k, v in by_money.items()},
        moneyness_marginal_n={k: len(v) for k, v in by_money.items()},
        dte_marginal={k: _median(v) for k, v in by_dte.items()},
        dte_marginal_n={k: len(v) for k, v in by_dte.items()},
        global_median=_median(all_sp),
        conservative_global=_percentile(all_sp, 0.75),  # "no coverage" last resort (high)
        calibrated_from=_min_obs_date(df),
        calibrated_through=(as_of.isoformat() if isinstance(as_of, date)
                            else (str(as_of) if as_of else _max_obs_date(df))),
        n_obs=len(full),
        n_days=_n_distinct_days(df),
    )
    return model


def _max_obs_date(df) -> Optional[str]:
    try:
        return str(df["obs_date"].astype(str).str[:10].max())
    except Exception:
        return None


def _min_obs_date(df) -> Optional[str]:
    try:
        return str(df["obs_date"].astype(str).str[:10].min())
    except Exception:
        return None


def _n_distinct_days(df) -> int:
    try:
        return int(df["obs_date"].astype(str).str[:10].nunique())
    except Exception:
        return 0


def calibrate_from_parquet(path: str | Path, *, as_of: Optional[date] = None
                           ) -> CalibratedSpreadModel:
    """Load the NBBO obs parquet and calibrate. Returns an empty (global-default) model if the
    file is missing/unreadable (never raises)."""
    import pandas as pd
    p = Path(path)
    if not p.exists():
        log.warning("spread_model: obs parquet missing at %s — empty model", p)
        return CalibratedSpreadModel(n_obs=0)
    try:
        df = pd.read_parquet(p)
    except Exception:
        log.exception("spread_model: failed to read %s — empty model", p)
        return CalibratedSpreadModel(n_obs=0)
    return calibrate(df, as_of=as_of)


# Default on-disk locations (the obs log + the calibrated artifact).
OBS_PARQUET = "data/options_spread_obs.parquet"
MODEL_PATH = "data/options_spread_model.json"
