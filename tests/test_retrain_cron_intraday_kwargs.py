"""Regression guard for the nightly intraday retrain crash.

INTRADAY_RETRAIN carries config-level keys (wf_folds, n_workers) that
IntradayModelTrainer.train_model() does NOT accept. retrain_cron splat-forwards
the dict, so any such orphan key raises TypeError and crashes the nightly intraday
retrain (champion silently retained). This test asserts every forwarded kwarg is a
real train_model parameter, so a future orphan key fails CI, not production.
"""

import inspect

from app.ml.retrain_config import INTRADAY_RETRAIN
from app.ml.intraday_training import IntradayModelTrainer

# Must mirror retrain_cron.run_intraday's exclusion set.
_NON_TRAIN_KEYS = {"wf_folds", "n_workers"}


def _train_model_params():
    sig = inspect.signature(IntradayModelTrainer.train_model)
    return {p for p in sig.parameters if p != "self"}


def test_intraday_train_kwargs_are_all_valid_params():
    params = _train_model_params()
    forwarded = (set(INTRADAY_RETRAIN) - _NON_TRAIN_KEYS) | {"promote_to_active"}
    orphans = forwarded - params
    assert not orphans, (
        f"INTRADAY_RETRAIN forwards kwargs train_model() does not accept: {orphans}. "
        "Add them to _NON_TRAIN_KEYS in retrain_cron.run_intraday or to train_model."
    )


def test_n_workers_is_excluded():
    # n_workers IS in the config (parallelism knob) but must NOT reach train_model.
    assert "n_workers" in INTRADAY_RETRAIN
    assert "n_workers" not in (set(INTRADAY_RETRAIN) - _NON_TRAIN_KEYS)


def test_promote_to_active_is_a_real_param():
    assert "promote_to_active" in _train_model_params()
