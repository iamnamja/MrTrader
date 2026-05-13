"""
app/ml/diagnostics — Reusable diagnostic infrastructure for ML model monitoring.

Modules:
    ic.py     — Cross-sectional information coefficient (Spearman IC) computation.
    metrics.py — Shared performance metrics (importable by diag scripts + gates).

Design principles:
    - All functions are pure (no database calls, no file I/O).
    - CLI wrappers live in scripts/diag_*.py; this package is the library.
    - Respect MAX_WORKERS from retrain_config for parallelism.
    - Never bypass SACRED_HOLDOUT guards; callers pass pre-filtered data.
"""
