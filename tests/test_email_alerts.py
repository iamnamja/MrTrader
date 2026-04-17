"""
Tests for the email alert module.
All tests use mocking — no real SMTP connections.
"""
import pytest
from unittest.mock import patch, MagicMock
from app.alerts.email_sender import send_email_alert, build_alert_html


class TestBuildAlertHtml:
    def test_contains_severity(self):
        html = build_alert_html("CRITICAL", "Test Title", "Test message")
        assert "CRITICAL" in html
        assert "Test Title" in html
        assert "Test message" in html

    def test_critical_uses_red(self):
        assert "#ef4444" in build_alert_html("CRITICAL", "T", "M")

    def test_warning_uses_yellow(self):
        assert "#eab308" in build_alert_html("WARNING", "T", "M")

    def test_info_uses_green(self):
        assert "#22c55e" in build_alert_html("INFO", "T", "M")


def _mock_settings(**overrides):
    defaults = dict(
        smtp_host="smtp.example.com",
        smtp_port=587,
        smtp_user="sender@example.com",
        smtp_password="secret",
        alert_email="recipient@example.com",
    )
    defaults.update(overrides)
    s = MagicMock()
    for k, v in defaults.items():
        setattr(s, k, v)
    return s


class TestSendEmailAlert:
    def test_sends_when_fully_configured(self):
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)

        with patch("app.alerts.email_sender.smtplib.SMTP", return_value=mock_smtp) as smtp_cls, \
             patch("app.alerts.email_sender.settings", _mock_settings()):
            result = send_email_alert("Test Subject", "Body text", "<b>HTML</b>")

        assert result is True
        smtp_cls.assert_called_once_with("smtp.example.com", 587, timeout=10)
        mock_smtp.starttls.assert_called_once()
        mock_smtp.login.assert_called_once_with("sender@example.com", "secret")
        mock_smtp.sendmail.assert_called_once()

    def test_skipped_when_no_smtp_host(self):
        with patch("app.alerts.email_sender.settings", _mock_settings(smtp_host=None)):
            assert send_email_alert("Subject", "Body") is False

    def test_skipped_when_no_recipient(self):
        with patch("app.alerts.email_sender.settings", _mock_settings(alert_email=None)):
            assert send_email_alert("Subject", "Body") is False

    def test_returns_false_on_smtp_error(self):
        with patch("app.alerts.email_sender.smtplib.SMTP", side_effect=ConnectionRefusedError("refused")), \
             patch("app.alerts.email_sender.settings", _mock_settings()):
            assert send_email_alert("Subject", "Body") is False

    def test_subject_prefixed(self):
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = MagicMock(return_value=mock_smtp)
        mock_smtp.__exit__ = MagicMock(return_value=False)
        captured: dict = {}

        def capture_sendmail(from_addr, to_addr, msg_str):
            captured["msg"] = msg_str

        mock_smtp.sendmail.side_effect = capture_sendmail

        with patch("app.alerts.email_sender.smtplib.SMTP", return_value=mock_smtp), \
             patch("app.alerts.email_sender.settings", _mock_settings()):
            send_email_alert("My Alert", "body")

        assert "[MrTrader] My Alert" in captured["msg"]

    def test_skipped_when_no_password(self):
        with patch("app.alerts.email_sender.settings", _mock_settings(smtp_password=None)):
            assert send_email_alert("Subject", "Body") is False
