"""Standing accounting + control-liveness invariants (2026-09-02).

Each test below reconstructs an ACTUAL defect from the 2026-08/09 review and asserts the
corresponding invariant would have caught it. None are hypothetical, and the point of the module
is the elapsed time in the right-hand column:

    regime scorer pinned to v9 while v40 existed      ~7 weeks
    crash governor unable to compute (VIX3M gone)     ~5 weeks
    order-status enum leak disabling three features   months
    DBC double-buy, 11.5% of equity                   until someone looked
    P&L scorecard writing NULLs                       3 months

Not one raised an error. Every one leaves a number that does not add up.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.live_trading import invariants as inv


def _acct(equity):
    return {"equity": str(equity), "portfolio_value": str(equity)}


def _pos(symbol="SPY", qty=10, unreal=100.0):
    return {"symbol": symbol, "qty": str(qty), "unrealized_pl": str(unreal),
            "current_price": "100", "market_value": str(qty * 100), "avg_entry_price": "90"}


class TestCheckSemantics:
    def test_breached_treats_none_as_breach(self):
        """'Could not evaluate' is the state the crash governor sat in for five weeks."""
        assert inv.Check("x", None).breached is True
        assert inv.Check("x", False).breached is True
        assert inv.Check("x", True).breached is False

    def test_report_ok_only_when_every_check_passes(self):
        r = inv.InvariantReport([inv.Check("a", True), inv.Check("b", True)])
        assert r.ok
        r.checks.append(inv.Check("c", None, "unknown"))
        assert not r.ok and len(r.breaches) == 1


class TestAccountIdentity:
    """equity == deposits + realized + unrealized + fees."""

    def _run(self, equity, realized, unreal, deposits=100000.0, fees=-4.0):
        alpaca = MagicMock()
        alpaca.get_all_orders.return_value = []
        alpaca.get_positions.return_value = [_pos(unreal=unreal)]
        alpaca.get_account.return_value = _acct(equity)
        with patch.object(inv, "_deposits", return_value=deposits), \
             patch.object(inv, "_fees", return_value=fees), \
             patch("app.analytics.execution_pnl.compute_realized_pnl",
                   return_value={"o1": {"realized_pnl": realized}}):
            return inv.check_account_identity(alpaca)

    def test_balanced_book_passes(self):
        c = self._run(equity=100_000 + 500 + 100 - 4, realized=500.0, unreal=100.0)
        assert c.ok is True

    def test_phantom_gain_is_caught(self):
        """Equity higher than the fills justify — the shape of a double-counted position."""
        c = self._run(equity=100_000 + 500 + 100 - 4 + 5_000, realized=500.0, unreal=100.0)
        assert c.ok is False and c.value == pytest.approx(5_000, abs=1)

    def test_small_residual_within_tolerance(self):
        c = self._run(equity=100_000 + 500 + 100 - 4 + 3.0, realized=500.0, unreal=100.0)
        assert c.ok is True, "fees/rounding must not trip a false breach"


class TestPositionReconcile:
    def test_dbc_double_buy_is_caught(self):
        """The real 2026-08-24 break: DB said 214, the broker held 442."""
        brk = MagicMock(instrument_id="DBC", expected_qty=214.0, actual_qty=442.0)
        res = MagicMock(status="FAIL_CLOSED", position_breaks=[brk])
        with patch("app.live_trading.reconciliation.reconcile", return_value=res), \
             patch("app.live_trading.reconciliation.db_expected_positions", return_value={}), \
             patch("app.live_trading.reconciliation.db_pending_positions", return_value={}), \
             patch("app.live_trading.reconciliation.alpaca_actual_positions", return_value=[]), \
             patch("app.integrations.alpaca.AlpacaClient"), \
             patch("app.database.SessionLocal"):
            c = inv.check_position_reconcile()
        assert c.ok is False
        assert "DBC" in c.detail and "442" in c.detail

    def test_match_passes(self):
        res = MagicMock(status="MATCH", position_breaks=[])
        with patch("app.live_trading.reconciliation.reconcile", return_value=res), \
             patch("app.live_trading.reconciliation.db_expected_positions", return_value={}), \
             patch("app.live_trading.reconciliation.db_pending_positions", return_value={}), \
             patch("app.live_trading.reconciliation.alpaca_actual_positions", return_value=[]), \
             patch("app.integrations.alpaca.AlpacaClient"), \
             patch("app.database.SessionLocal"):
            assert inv.check_position_reconcile().ok is True


class TestCrashGovernorLiveness:
    """The five-week silent failure: fail-safe made 'cannot compute' look like 'no de-risk'."""

    def _mh(self, n_paired, end=None):
        # Dates must END AT TODAY: the check also enforces a 7-day freshness bound, so a fixture
        # anchored to a fixed past date fails for the wrong reason (and would rot over time).
        end = end or pd.Timestamp.today().normalize()
        dates = pd.date_range(end=end, periods=6).strftime("%Y-%m-%d")
        vix = [15.0] * 6
        v3m = [18.0 if i >= 6 - n_paired else float("nan") for i in range(6)]
        return pd.DataFrame({"date": dates, "vix": vix, "vix3m": v3m})

    def _run(self, mh, enabled="true"):
        with patch("app.data.macro_history.load_macro_history", return_value=mh), \
             patch("app.database.agent_config.get_agent_config", return_value=enabled), \
             patch("app.database.SessionLocal"):
            return inv.check_crash_governor_live()

    def test_vix3m_outage_is_caught(self):
        """Exactly the 2026-07/08 condition: VIX present, VIX3M gone."""
        assert self._run(self._mh(n_paired=0)).ok is False

    def test_healthy_pairs_pass(self):
        c = self._run(self._mh(n_paired=6))
        assert c.ok is True and c.value == 6

    def test_missing_column_is_caught(self):
        assert self._run(pd.DataFrame({"date": ["2026-09-01"], "vix": [15.0]})).ok is False

    def test_stale_but_paired_data_is_caught(self):
        """A wedged feed still has pairs — they are just old. Freshness is its own condition:
        de-risking off month-old VIX would be worse than not de-risking at all."""
        old = pd.Timestamp.today().normalize() - pd.Timedelta(days=30)
        c = self._run(self._mh(n_paired=6, end=old))
        assert c.ok is False and "30d old" in c.detail

    def test_disabled_is_not_a_breach(self):
        """A control switched off deliberately is a choice, not a fault."""
        assert self._run(self._mh(0), enabled="false").ok is True


class TestScorecardRecording:
    """Self-referential: three months of NULL P&L went unnoticed because nothing asserted
    that recording was happening."""

    def test_all_null_pnl_is_caught(self, tmp_path):
        import sqlite3
        db = tmp_path / "t.db"
        c = sqlite3.connect(db)
        c.execute("CREATE TABLE trend_daily (trade_date TEXT, cumulative_pnl REAL)")
        c.execute("INSERT INTO trend_daily VALUES ('2026-09-01', NULL)")   # the real symptom
        c.commit()
        with patch("app.live_trading.trend_tracker.DB_PATH", db), \
             patch("app.live_trading.cash_tracker.DB_PATH", db):
            check = inv.check_scorecard_recording()
        assert check.ok is False and "no P&L rows" in check.detail

    def test_stale_series_is_caught(self, tmp_path):
        import sqlite3
        from datetime import date, timedelta
        old = (date.today() - timedelta(days=30)).isoformat()
        db = tmp_path / "t.db"
        c = sqlite3.connect(db)
        for t in ("trend_daily", "cash_daily"):
            c.execute(f"CREATE TABLE {t} (trade_date TEXT, cumulative_pnl REAL)")
            c.execute(f"INSERT INTO {t} VALUES (?, 1.0)", (old,))
        c.commit()
        with patch("app.live_trading.trend_tracker.DB_PATH", db), \
             patch("app.live_trading.cash_tracker.DB_PATH", db):
            check = inv.check_scorecard_recording()
        assert check.ok is False and "30d ago" in check.detail

    def test_current_series_passes(self, tmp_path):
        import sqlite3
        from datetime import date
        db = tmp_path / "t.db"
        c = sqlite3.connect(db)
        for t in ("trend_daily", "cash_daily"):
            c.execute(f"CREATE TABLE {t} (trade_date TEXT, cumulative_pnl REAL)")
            c.execute(f"INSERT INTO {t} VALUES (?, 1.0)", (date.today().isoformat(),))
        c.commit()
        with patch("app.live_trading.trend_tracker.DB_PATH", db), \
             patch("app.live_trading.cash_tracker.DB_PATH", db):
            assert inv.check_scorecard_recording().ok is True


class TestRunAll:
    def test_a_raising_check_is_a_breach_not_a_crash(self):
        """A monitor its own errors can silence is worse than no monitor."""
        boom = MagicMock(side_effect=RuntimeError("broker down"))
        boom.__name__ = "check_boom"
        with patch.object(inv, "_CHECKS", (boom,)):
            rep = inv.run_all()
        assert not rep.ok
        assert rep.checks[0].ok is None and "RuntimeError" in rep.checks[0].detail

    def test_one_failure_does_not_suppress_the_others(self):
        boom = MagicMock(side_effect=RuntimeError("x")); boom.__name__ = "check_boom"
        good = MagicMock(return_value=inv.Check("good", True)); good.__name__ = "check_good"
        with patch.object(inv, "_CHECKS", (boom, good)):
            rep = inv.run_all()
        assert len(rep.checks) == 2
        assert any(c.name == "good" and c.ok for c in rep.checks)

    def test_summary_names_the_breaches(self):
        rep = inv.InvariantReport([inv.Check("a", True), inv.Check("b", False, "off by $5000")])
        assert "b" in rep.summary() and "5000" in rep.summary()
