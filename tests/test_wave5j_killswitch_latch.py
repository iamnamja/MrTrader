"""Alpha-v10 audit Wave 5j — kill-switch must not UN-ARM itself on a stale DB read.

The daemon's 3s state-sync calls load_state(). Previously load_state set `_active = db_value` for any
bool, so if activate() set _active=True but the persist FAILED (DB still False), the next sync would
silently un-arm the switch within ~3s. Fix: a monotonic seq distinguishes an AUTHORITATIVE change (a
real cross-process reset, higher seq) from a STALE value (same/older seq, our failed persist).
Asymmetric apply: upgrades to ACTIVE always honored (fail-safe); downgrades to INACTIVE only on a
strictly newer seq.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.live_trading.kill_switch import KillSwitch, _CFG_KS_ACTIVE, _CFG_KS_SEQ


def _load(ks: KillSwitch, *, active, seq):
    """Run ks.load_state() with the config store returning {active, seq} per key."""
    values = {_CFG_KS_ACTIVE: active, _CFG_KS_SEQ: seq}
    db = MagicMock()
    with patch("app.live_trading.kill_switch.get_session", return_value=db), \
         patch("app.database.config_store.get_config", side_effect=lambda _db, k: values.get(k)):
        return ks.load_state()


def test_refuses_to_unarm_on_stale_db():
    # THE fix: in-memory ACTIVE (seq 5), DB says inactive at the SAME seq (our activate persist
    # failed, DB never advanced) -> must STAY active.
    ks = KillSwitch()
    ks._active = True
    ks._seq = 5
    _load(ks, active=False, seq=5)
    assert ks.is_active is True            # refused to un-arm on a stale read


def test_refuses_to_unarm_on_older_seq():
    ks = KillSwitch()
    ks._active = True
    ks._seq = 7
    _load(ks, active=False, seq=3)        # older snapshot
    assert ks.is_active is True


def test_honors_authoritative_reset_newer_seq():
    # a genuine cross-process reset bumps the seq -> the daemon MUST un-arm.
    ks = KillSwitch()
    ks._active = True
    ks._seq = 5
    _load(ks, active=False, seq=6)        # newer authoritative reset
    assert ks.is_active is False
    assert ks._seq == 6                   # watermark advanced


def test_upgrade_to_active_always_honored_regardless_of_seq():
    # fail-safe: a persisted halt binds even with an older/seq-less row.
    ks = KillSwitch()
    ks._active = False
    ks._seq = 10
    _load(ks, active=True, seq=1)         # older seq, but it's an UPGRADE
    assert ks.is_active is True


def test_upgrade_to_active_with_no_seq_row():
    # legacy DB without a seq row (None) -> still restore a persisted halt.
    ks = KillSwitch()
    _load(ks, active=True, seq=None)
    assert ks.is_active is True


def test_malformed_value_leaves_state_unchanged():
    # non-bool (legacy string / MagicMock) must neither halt nor un-arm.
    ks = KillSwitch()
    ks._active = True
    ks._seq = 4
    _load(ks, active="false", seq=9)
    assert ks.is_active is True           # not un-armed by a malformed value (even at a newer seq)

    ks2 = KillSwitch()                    # fresh (inactive) -> stays inactive on garbage
    _load(ks2, active=MagicMock(), seq=9)
    assert ks2.is_active is False


def test_reset_clears_in_memory_directly():
    # reset() un-arms the OWNING process directly (not via load_state) — the seq path is only for
    # propagation to OTHER processes.
    ks = KillSwitch()
    ks._active = True
    with patch.object(ks, "_persist_state", return_value=True):
        ks.reset("test")
    assert ks.is_active is False
    assert ks._flattened is False
