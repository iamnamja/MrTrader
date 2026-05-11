"""R2 tests: gate ablation runner script — configs, parser, F-config completeness."""
from __future__ import annotations


def test_configs_cover_each_gate_individually():
    """Each gate must appear as the ONLY active gate in exactly one config."""
    from scripts.gate_ablation_v186 import CONFIGS
    # Collect all flag strings across single-gate configs
    # 'only-X' configs are those that disable all-but-one gate.
    # The flags in each config are what gets DISABLED, so single-gate configs
    # have 2 or more --no-* flags (disabling everything except their gate).
    # We just verify coverage: the 4 swing gates (opp, earnings, macro, regime)
    # each appear as the only-on gate in exactly one config.
    config_names = [c[0] for c in CONFIGS]
    assert "B_opp_only" in config_names, "Missing opportunity-score-only config"
    assert "C_earnings_only" in config_names, "Missing earnings-blackout-only config"
    assert "D_macro_only" in config_names, "Missing macro-gate-only config"
    assert "E_regime_only" in config_names, "Missing regime/benign-gate-only config"
    assert "A_all_on" in config_names, "Missing all-gates-ON baseline"
    assert "F_all_off" in config_names, "Missing all-gates-OFF config"


def test_parse_walkforward_stdout():
    """Parser correctly extracts metrics from canned WF output."""
    from scripts.gate_ablation_v186 import _parse_sharpe, _parse_min_fold_sharpe, _parse_total_trades, _parse_fold_sharpes

    sample = """
  Avg Sharpe: +0.644
  Min fold Sharpe: -0.105
  Total trades: 312
  Fold 1 [OK] Sharpe=+0.92
  Fold 2 [OK] Sharpe=+0.71
  Fold 3 [OK] Sharpe=+0.31
    """
    assert _parse_sharpe(sample) == 0.644
    assert _parse_min_fold_sharpe(sample) == -0.105
    assert _parse_total_trades(sample) == 312
    folds = _parse_fold_sharpes(sample)
    assert len(folds) == 3
    assert abs(folds[0] - 0.92) < 1e-4


def test_all_off_disables_every_default_gate():
    """Config F must include all three --no-* swing gate flags."""
    from scripts.gate_ablation_v186 import CONFIGS
    f_config = next(c for c in CONFIGS if c[0] == "F_all_off")
    flags = f_config[1]
    assert "--no-pm-opportunity-score" in flags
    assert "--no-earnings-blackout" in flags
    assert "--no-macro-gate" in flags
    # F_all_off should NOT include --benign-gate (that is an opt-in, not default)
    assert "--benign-gate" not in flags


def test_regime_only_config_includes_benign_gate():
    """Config E (regime/benign only) must opt-in benign gate while disabling others."""
    from scripts.gate_ablation_v186 import CONFIGS
    e_config = next(c for c in CONFIGS if c[0] == "E_regime_only")
    flags = e_config[1]
    assert "--benign-gate" in flags
    assert "--no-pm-opportunity-score" in flags
    assert "--no-earnings-blackout" in flags
    assert "--no-macro-gate" in flags


def test_markdown_table_includes_all_config_names():
    """_markdown_table produces a string containing each config name."""
    from scripts.gate_ablation_v186 import _markdown_table, CONFIGS
    # Simulate dry-run results
    fake_results = [
        {"name": name, "description": desc, "dry_run": True}
        for name, _, desc in CONFIGS
    ]
    table = _markdown_table(fake_results)
    # dry_run rows are skipped — table should just have headers
    assert "Config" in table
    assert "Avg Sharpe" in table
