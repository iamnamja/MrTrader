"""
Tests for scripts/run_event_panel_inference.py — the H1 confirmatory runner.

Coverage:
  - verdict mapping at the frozen boundaries (p=0.049/0.051/0.149/0.151);
  - concentration-caps math (quarter<=40%, name<=15%, non-positive total);
  - gapper-stress identity (coef on y-50bps == coef - 50bps, same SEs);
  - population selection (qualified + clean quality bits + primary y present);
  - registry enforcement end-to-end on a TEMP registry
    (MRTRADER_RESEARCH_REGISTRY_DB — the conftest isolates every test from
    data/research_registry.db): unregistered id fails fast BEFORE the panel
    loads; a registered+preregistered id records its result with
    decision=None; --exploratory writes nothing;
  - the REAL one-shot id H1-PEAD-EVENTLEVEL-20260611 is NEVER consumed: it is
    unregistered in the temp registry, so begin_run refuses it (and the real
    registry file is never opened);
  - the CONFIRMATORY COVERAGE GATE (one-shot protection): a hypothesis-id run
    refuses a non-default panel path (unless --allow-nonstandard-panel), a
    smoke-sized/thinned panel, a short year span, or a year with zero
    qualified events; exploratory runs skip the gate; the recorded result
    pins the panel's SHA256 + coverage summary;
  - --exploratory combined with a real --hypothesis-id fails fast (F3).

The synthetic panel is generated, not loaded — no network, no FMP.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.run_event_panel_inference import (
    DEFAULT_PANEL,
    GAPPER_STRESS,
    MIN_PANEL_EVENTS,
    MIN_POPULATION,
    PRIMARY_Y,
    concentration_caps,
    enforce_confirmatory_coverage,
    main,
    panel_coverage,
    run_inference,
    select_population,
    verdict_from_p,
)
from app.research.event_panel import QF_INCOMPLETE_FWD20, QF_NO_SPY_HEDGE


# ─────────────────────────────────────────────── verdict mapping (frozen rule)

@pytest.mark.parametrize("p,expected", [
    (0.049, "GRADUATE"),
    (0.051, "INCONCLUSIVE"),
    (0.149, "INCONCLUSIVE"),
    (0.151, "DEMOTE"),
    (0.05, "INCONCLUSIVE"),   # boundaries: rule is STRICT < / >
    (0.15, "INCONCLUSIVE"),
])
def test_verdict_mapping(p, expected):
    assert verdict_from_p(p) == expected


# ──────────────────────────────────────────────────────────── synthetic panel

def _synthetic_panel(n_quarters: int = 8, names_per_q: int = 18,
                     mean: float = 0.004, seed: int = 9,
                     start: date = date(2023, 2, 1)) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    d0 = start
    for q in range(n_quarters):
        q_start = d0 + timedelta(days=91 * q)
        for i in range(names_per_q):
            ann = q_start + timedelta(days=int(rng.integers(0, 25)))
            y10 = rng.normal(mean, 0.03)
            rows.append({
                "event_id": f"S{i:02d}|{ann.isoformat()}",
                "symbol": f"S{i:02d}",
                "announce_date": ann,
                "sector": ["Technology", "Energy", "Healthcare"][i % 3],
                "sue": float(rng.uniform(0.05, 0.5)),
                "pead_score_v1": float(rng.normal(0, 1)),
                "pead_qualified": True,
                "qual_reason": "ok",
                "quality_flags": 0,
                "spy_below_200d": bool(q == 2),   # one risk-off quarter
                "beta_60d": float(rng.normal(1.0, 0.2)),
                "fwd_ret_5_spyhedged": y10 * 0.6,
                "fwd_ret_10_spyhedged": y10,
                "fwd_ret_20_spyhedged": y10 * 1.2,
                "fwd_ret_10_raw": y10 + 0.002,
                "entry_open_next": 100.0,
                "entry_open_next2": 100.0 + float(rng.normal(0.3, 0.5)),
            })
    df = pd.DataFrame(rows)
    df["spy_below_200d"] = df["spy_below_200d"].astype("boolean")
    return df


def _covered_panel(mean: float = 0.012) -> pd.DataFrame:
    """A panel that SATISFIES the confirmatory coverage gate: 30 quarters from
    2019 into 2026, 3300 events (>= MIN_PANEL_EVENTS), all qualified/clean
    (population >= MIN_POPULATION), every year 2019..2025 qualified."""
    return _synthetic_panel(n_quarters=30, names_per_q=110, mean=mean,
                            start=date(2019, 1, 15))


# ─────────────────────────────────────────────────────────────────── caps math

def test_concentration_caps_known_shares():
    panel = pd.DataFrame({
        "announce_date": [date(2024, 2, 1)] * 2 + [date(2024, 5, 1)] * 2,
        "symbol": ["A", "B", "A", "C"],
        PRIMARY_Y: [0.30, 0.10, 0.40, 0.20],   # total = 1.0
    })
    caps = concentration_caps(panel, PRIMARY_Y)
    assert caps["total_pnl"] == pytest.approx(1.0)
    # Quarters: 2024Q1 = 0.40, 2024Q2 = 0.60 -> max 0.60 breaches the 40% cap.
    assert caps["max_quarter_share"] == pytest.approx(0.60)
    assert caps["max_quarter"] == "2024Q2"
    assert caps["quarter_cap_ok"] is False
    # Names: A = 0.70 breaches the 15% cap.
    assert caps["max_name_share"] == pytest.approx(0.70)
    assert caps["max_name"] == "A"
    assert caps["name_cap_ok"] is False


def test_concentration_caps_pass_when_diversified():
    panel = _synthetic_panel(mean=0.004)
    caps = concentration_caps(panel, PRIMARY_Y)
    assert caps["total_pnl"] > 0
    assert caps["max_quarter_share"] is not None


def test_concentration_caps_non_positive_total_fails():
    panel = _synthetic_panel(mean=-0.02)
    caps = concentration_caps(panel, PRIMARY_Y)
    assert caps["total_pnl"] <= 0
    assert caps["quarter_cap_ok"] is False and caps["name_cap_ok"] is False
    assert caps["max_quarter_share"] is None


# ───────────────────────────────────────────────────────── stress + population

def test_gapper_stress_shifts_mean_not_se():
    panel = _synthetic_panel()
    out = run_inference(panel, n_resamples=50)
    base = out["horizons"]["10d"]
    stress = out["gapper_stress_50bps"]
    assert stress["coef"][0] == pytest.approx(base["coef"][0] - GAPPER_STRESS,
                                              abs=1e-12)
    assert stress["se"][0] == pytest.approx(base["se"][0], rel=1e-9)
    assert stress["stress_bps"] == pytest.approx(50.0)


def test_select_population_filters():
    panel = _synthetic_panel()
    panel.loc[0, "pead_qualified"] = False
    panel.loc[1, "quality_flags"] = QF_INCOMPLETE_FWD20
    panel.loc[2, "quality_flags"] = QF_NO_SPY_HEDGE
    panel.loc[3, PRIMARY_Y] = np.nan
    pop, acc = select_population(panel)
    assert acc["panel_events"] == len(panel)
    assert acc["qualified"] == len(panel) - 1
    assert acc["excluded_quality"] == 2
    assert acc["excluded_missing_primary_y"] == 1
    assert acc["population"] == len(panel) - 4
    assert len(pop) == acc["population"]


def test_run_inference_structure_and_reported_cuts():
    panel = _synthetic_panel(mean=0.012)  # strong edge -> graduate
    out = run_inference(panel, n_resamples=100)
    assert out["verdict"] == "GRADUATE"
    assert out["primary_p_one_sided"] < 0.05
    assert set(out["horizons"]) == {"5d", "10d", "20d"}
    assert out["quarter_bootstrap_conservative"]["n_clusters"] == 8
    rep = out["reported_not_deciding"]
    assert rep["beta_adjusted_10d"] is not None
    assert rep["trend_gated_slice_10d"]["n_blocked_by_trend_gate"] == 18
    assert len(rep["loco"]["quarter"]) == 8
    assert len(rep["loco"]["sector"]) == 3
    assert rep["loco"]["top10"][0]["n_obs"] == len(panel) - 10
    assert "is_monotone" in rep["deciles_pead_score_v1"]
    assert out["n_obs"] == len(panel)


def test_run_inference_rejects_holdout_panel():
    panel = _synthetic_panel()
    panel.loc[0, "announce_date"] = date(2026, 11, 10)
    with pytest.raises(AssertionError, match="sacred holdout"):
        run_inference(panel, n_resamples=10)


# ───────────────────────────── confirmatory coverage gate (one-shot protection)

def test_coverage_gate_passes_adequate_panel_on_default_path():
    cov = enforce_confirmatory_coverage(_covered_panel(), DEFAULT_PANEL)
    assert cov["panel_events"] >= MIN_PANEL_EVENTS
    assert cov["population"] >= MIN_POPULATION
    assert cov["min_year"] <= 2019 and cov["max_year"] >= 2026
    assert all(cov["per_year_qualified"][y] > 0 for y in range(2019, 2026))


def test_coverage_gate_rejects_nonstandard_path(tmp_path):
    panel = _covered_panel()
    with pytest.raises(ValueError, match="default full panel"):
        enforce_confirmatory_coverage(panel, tmp_path / "smoke.parquet")
    # The explicit override allows an alternate PATH — floors still apply.
    cov = enforce_confirmatory_coverage(panel, tmp_path / "smoke.parquet",
                                        allow_nonstandard_panel=True)
    assert cov["population"] >= MIN_POPULATION


def test_coverage_gate_rejects_smoke_sized_panel():
    # The default synthetic (144 events, 2023->2025) is the smoke-panel shape.
    with pytest.raises(ValueError, match="coverage gate REFUSED"):
        enforce_confirmatory_coverage(_synthetic_panel(), DEFAULT_PANEL)


def test_coverage_gate_rejects_missing_year_qualified():
    panel = _covered_panel()
    years = pd.to_datetime(panel["announce_date"]).dt.year
    panel.loc[years == 2021, "pead_qualified"] = False   # silently thinned year
    with pytest.raises(ValueError, match=r"year\(s\) \[2021\]"):
        enforce_confirmatory_coverage(panel, DEFAULT_PANEL)


def test_coverage_gate_rejects_short_year_span():
    panel = _covered_panel()
    years = pd.to_datetime(panel["announce_date"]).dt.year
    with pytest.raises(ValueError, match="does not cover"):
        enforce_confirmatory_coverage(panel[years <= 2025], DEFAULT_PANEL)


def test_panel_coverage_summary_shape():
    cov = panel_coverage(_synthetic_panel())
    assert set(cov) == {"panel_events", "population", "min_year", "max_year",
                        "per_year_qualified"}
    assert cov["panel_events"] == 144
    assert cov["min_year"] == 2023


# ─────────────────────────────────────── registry enforcement (temp registry)

REAL_H1_ID = "H1-PEAD-EVENTLEVEL-20260611"


@pytest.fixture()
def temp_registry(tmp_path, monkeypatch):
    """Explicitly isolated registry (defense-in-depth on top of conftest's
    per-worker isolation): the REAL data/research_registry.db is untouched."""
    db = tmp_path / "registry.db"
    monkeypatch.setenv("MRTRADER_RESEARCH_REGISTRY_DB", str(db))
    from app.research.registry import ResearchRegistry
    reg = ResearchRegistry()
    assert str(reg.db_path) == str(db)
    return reg


@pytest.fixture()
def panel_file(tmp_path) -> Path:
    p = tmp_path / "panel.parquet"
    _synthetic_panel(mean=0.012).to_parquet(p, index=False)
    return p


@pytest.fixture()
def covered_panel_file(tmp_path) -> Path:
    """A coverage-gate-passing panel on a NON-default path (confirmatory main()
    tests pass --allow-nonstandard-panel; the floors still apply)."""
    p = tmp_path / "panel_full.parquet"
    _covered_panel().to_parquet(p, index=False)
    return p


def test_main_fails_fast_on_unregistered_id(temp_registry, panel_file):
    from scripts.walkforward.registry_enforcement import RegistryEnforcementError
    with pytest.raises(RegistryEnforcementError, match="not registered"):
        main(["--hypothesis-id", "H1-GHOST", "--panel", str(panel_file),
              "--n-resamples", "10"])


def test_main_never_consumes_the_real_h1_id(temp_registry, panel_file):
    """The real one-shot id does not exist in the temp registry -> begin_run
    refuses BEFORE any panel load; the real registry file is never written."""
    from scripts.walkforward.registry_enforcement import RegistryEnforcementError
    with pytest.raises(RegistryEnforcementError, match="not registered"):
        main(["--hypothesis-id", REAL_H1_ID, "--panel", str(panel_file),
              "--n-resamples", "10"])
    assert temp_registry.get(REAL_H1_ID) is None


def _register_confirmatory(reg, hid="H1-TEST"):
    reg.register(hid, label="confirmatory", family="pead")
    reg.preregister(
        hid,
        acceptance_criteria={"primary": "10d spyhedged one-sided p<0.05"},
        preregistered_at="2026-06-11T00:00:00+00:00",
    )
    return hid


def test_main_records_result_against_temp_hypothesis(temp_registry,
                                                     covered_panel_file):
    _register_confirmatory(temp_registry)
    rc = main(["--hypothesis-id", "H1-TEST", "--panel", str(covered_panel_file),
               "--allow-nonstandard-panel", "--n-resamples", "50"])
    assert rc == 0  # strong synthetic edge -> GRADUATE
    row = temp_registry.get("H1-TEST")
    assert row["run_at"] is not None
    assert row["decision"] is None          # promotion stays owner-gated
    assert row["result_json"]["verdict"] == "GRADUATE"
    assert row["result_json"]["primary_p_one_sided"] < 0.05
    # F2: the recorded result pins the EXACT panel the verdict came from.
    import hashlib
    expected_sha = hashlib.sha256(covered_panel_file.read_bytes()).hexdigest()
    assert row["result_json"]["panel_sha256"] == expected_sha
    cov = row["result_json"]["panel_coverage"]   # JSON round-trip: str keys
    assert cov["panel_events"] >= MIN_PANEL_EVENTS
    assert cov["population"] >= MIN_POPULATION
    assert cov["per_year_qualified"]["2019"] > 0


def test_main_confirmatory_refuses_smoke_sized_panel(temp_registry, panel_file):
    """The one-shot protection end-to-end: a confirmatory id pointed at a
    smoke-sized panel refuses BEFORE inference; the hypothesis stays unburned
    (re-runnable) — even with the path check explicitly overridden."""
    _register_confirmatory(temp_registry, "H1-COV")
    rc = main(["--hypothesis-id", "H1-COV", "--panel", str(panel_file),
               "--allow-nonstandard-panel", "--n-resamples", "10"])
    assert rc == 1
    row = temp_registry.get("H1-COV")
    assert row["run_at"] is None and row["result_json"] is None


def test_main_confirmatory_refuses_nonstandard_path_without_override(
        temp_registry, covered_panel_file):
    _register_confirmatory(temp_registry, "H1-PATH")
    rc = main(["--hypothesis-id", "H1-PATH", "--panel", str(covered_panel_file),
               "--n-resamples", "10"])   # no --allow-nonstandard-panel
    assert rc == 1
    assert temp_registry.get("H1-PATH")["result_json"] is None


def test_main_exploratory_with_real_id_fails_fast(temp_registry, panel_file):
    """F3 end-to-end: `--exploratory --hypothesis-id H1-...` must refuse
    outright (begin_run contradiction guard) — never record under the id."""
    from scripts.walkforward.registry_enforcement import RegistryEnforcementError
    with pytest.raises(RegistryEnforcementError, match="cannot be combined"):
        main(["--exploratory", "--hypothesis-id", REAL_H1_ID,
              "--panel", str(panel_file), "--n-resamples", "10"])
    assert temp_registry.get(REAL_H1_ID) is None


def test_main_exploratory_records_nothing(temp_registry, panel_file):
    rc = main(["--exploratory", "--panel", str(panel_file),
               "--n-resamples", "50"])
    assert rc in (0, 2, 3)
    assert temp_registry.summary()["total"] == 0  # nothing written


def test_main_missing_panel_returns_error(temp_registry, tmp_path):
    rc = main(["--exploratory", "--panel", str(tmp_path / "nope.parquet")])
    assert rc == 1
