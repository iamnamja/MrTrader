"""Tests for LKG (last-known-good) rollback mechanism."""
import pytest
from unittest.mock import MagicMock, patch


class TestLKGPersistence:
    def _mock_db_with_config(self, existing_value=None):
        db = MagicMock()
        query = MagicMock()
        db.query.return_value = query
        filter_result = MagicMock()
        query.filter_by.return_value = filter_result

        if existing_value is not None:
            row = MagicMock()
            row.value = str(existing_value)
            filter_result.first.return_value = row
        else:
            filter_result.first.return_value = None

        return db

    def test_set_lkg_creates_new_entry(self):
        from app.strategy.benign_gate import set_lkg_version
        db = self._mock_db_with_config(existing_value=None)
        with patch("app.strategy.benign_gate._get_db", return_value=db):
            set_lkg_version("swing", 182)
        db.add.assert_called_once()
        db.commit.assert_called_once()

    def test_set_lkg_updates_existing_entry(self):
        from app.strategy.benign_gate import set_lkg_version
        db = self._mock_db_with_config(existing_value=181)
        with patch("app.strategy.benign_gate._get_db", return_value=db):
            set_lkg_version("swing", 182)
        # Should update existing row, not add
        db.add.assert_not_called()
        db.commit.assert_called_once()

    def test_get_lkg_returns_version(self):
        from app.strategy.benign_gate import get_lkg_version
        db = self._mock_db_with_config(existing_value=181)
        with patch("app.strategy.benign_gate._get_db", return_value=db):
            result = get_lkg_version("swing")
        assert result == 181

    def test_get_lkg_returns_none_when_no_entry(self):
        from app.strategy.benign_gate import get_lkg_version
        db = self._mock_db_with_config(existing_value=None)
        with patch("app.strategy.benign_gate._get_db", return_value=db):
            result = get_lkg_version("swing")
        assert result is None

    def test_get_lkg_returns_none_on_db_failure(self):
        from app.strategy.benign_gate import get_lkg_version
        with patch("app.strategy.benign_gate._get_db", return_value=None):
            result = get_lkg_version("swing")
        assert result is None


class TestRestoreLKG:
    def test_restore_lkg_promotes_correct_version(self):
        from scripts.promote_lkg import restore_lkg

        lkg_row = MagicMock()
        lkg_row.version = 181
        lkg_row.status = "RETIRED"

        active_row = MagicMock()
        active_row.version = 182
        active_row.status = "ACTIVE"

        db = MagicMock()
        # filter_by for lkg version lookup returns lkg_row
        db.query.return_value.filter_by.return_value.first.return_value = lkg_row
        # filter_by for current ACTIVE returns list
        db.query.return_value.filter_by.return_value.all.return_value = [active_row]

        with patch("app.strategy.benign_gate.get_lkg_version", return_value=181), \
             patch("app.database.session.SessionLocal", return_value=db):
            result = restore_lkg("swing")

        assert result is True

    def test_restore_lkg_no_op_when_already_active(self):
        from scripts.promote_lkg import restore_lkg

        lkg_row = MagicMock()
        lkg_row.version = 181
        lkg_row.status = "ACTIVE"

        db = MagicMock()
        db.query.return_value.filter_by.return_value.first.return_value = lkg_row

        with patch("app.strategy.benign_gate.get_lkg_version", return_value=181), \
             patch("app.database.session.SessionLocal", return_value=db):
            result = restore_lkg("swing")

        assert result is False

    def test_restore_lkg_returns_false_when_no_lkg(self):
        from scripts.promote_lkg import restore_lkg
        with patch("app.strategy.benign_gate.get_lkg_version", return_value=None):
            result = restore_lkg("swing")
        assert result is False
