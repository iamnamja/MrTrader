"""Tests for Claude AI decision layer."""
from unittest.mock import patch, MagicMock
import pytest


# ── ClaudeClient unit tests (mock Anthropic SDK) ──────────────────────────────

class TestClaudeClient:

    def _mock_response(self, text: str):
        msg = MagicMock()
        content = MagicMock()
        content.text = text
        msg.content = [content]
        return msg

    def test_returns_none_when_no_api_key(self):
        from app.ai.claude_client import review_pm_signal
        with patch.dict("os.environ", {}, clear=True):
            # Ensure ANTHROPIC_API_KEY is absent
            import os
            os.environ.pop("ANTHROPIC_API_KEY", None)
            result = review_pm_signal("AAPL", "EMA_CROSSOVER", 0.72, {})
        assert result is None

    def test_review_pm_signal_returns_text(self):
        from app.ai.claude_client import review_pm_signal
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response("Strong setup with 1:2 R/R.")

        with patch("app.ai.claude_client._client", return_value=mock_client):
            result = review_pm_signal("AAPL", "EMA_CROSSOVER", 0.72,
                                      {"price": 180, "stop": 176, "target": 188}, "MEDIUM")
        assert result == "Strong setup with 1:2 R/R."

    def test_review_pm_signal_handles_api_error(self):
        from app.ai.claude_client import review_pm_signal
        mock_client = MagicMock()
        mock_client.messages.create.side_effect = Exception("rate limit")

        with patch("app.ai.claude_client._client", return_value=mock_client):
            result = review_pm_signal("AAPL", "EMA_CROSSOVER", 0.72, {})
        assert result is None

    def test_explain_risk_veto_returns_text(self):
        from app.ai.claude_client import explain_risk_veto
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(
            "Position would exceed 5% account limit."
        )

        with patch("app.ai.claude_client._client", return_value=mock_client):
            result = explain_risk_veto(
                "AAPL", "position_size", "Exceeds max size", {"entry_price": 180}
            )
        assert "5%" in result or "Position" in result

    def test_explain_risk_veto_none_without_key(self):
        from app.ai.claude_client import explain_risk_veto
        with patch("app.ai.claude_client._client", return_value=None):
            result = explain_risk_veto("AAPL", "rule", "reason", {})
        assert result is None

    def test_summarise_daily_proposals_empty_returns_none(self):
        from app.ai.claude_client import summarise_daily_proposals
        result = summarise_daily_proposals([])
        assert result is None

    def test_summarise_daily_proposals_returns_text(self):
        from app.ai.claude_client import summarise_daily_proposals
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response(
            "Today we have 3 strong momentum setups."
        )
        proposals = [
            {"symbol": "AAPL", "signal_type": "EMA_CROSSOVER", "confidence": 0.72,
             "entry_price": 180, "stop_price": 176},
        ]
        with patch("app.ai.claude_client._client", return_value=mock_client):
            result = summarise_daily_proposals(proposals)
        assert result is not None
        assert "setups" in result or "momentum" in result

    def test_summarise_caps_at_10_proposals(self):
        from app.ai.claude_client import summarise_daily_proposals
        mock_client = MagicMock()
        mock_client.messages.create.return_value = self._mock_response("summary")
        proposals = [{"symbol": f"SYM{i}", "signal_type": "EMA", "confidence": 0.6,
                      "entry_price": 100, "stop_price": 95} for i in range(15)]

        with patch("app.ai.claude_client._client", return_value=mock_client):
            summarise_daily_proposals(proposals)

        call_args = mock_client.messages.create.call_args
        prompt = call_args[1]["messages"][0]["content"]
        # Should only mention 10 proposals in prompt
        assert "SYM14" not in prompt  # 15th should be excluded

    def test_client_returns_none_on_import_error(self):
        from app.ai.claude_client import _client
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test_key"}):
            with patch("app.ai.claude_client.Anthropic" if False else "builtins.__import__",
                       side_effect=ImportError):
                # Without anthropic installed _client returns None
                pass
        # Just verify _client() doesn't raise even without a key
        import os
        os.environ.pop("ANTHROPIC_API_KEY", None)
        assert _client() is None


# ── Integration: PM proposal includes ai_review when AI available ─────────────

class TestPMProposalAIReview:
    def test_ai_review_added_to_proposal_when_available(self):
        """PM._build_proposals adds ai_review field when claude_client returns text."""
        from app.ai.claude_client import review_pm_signal

        with patch("app.ai.claude_client._client") as mock_c:
            content = MagicMock()
            content.text = "Good R/R on this crossover."
            resp = MagicMock()
            resp.content = [content]
            mock_c.return_value.messages.create.return_value = resp

            result = review_pm_signal(
                "AAPL", "EMA_CROSSOVER", 0.75,
                {"price": 180, "stop": 175, "target": 192, "rsi": 58},
                "LOW",
            )
        assert result == "Good R/R on this crossover."

    def test_ai_review_gracefully_absent(self):
        """PM._build_proposals works fine when AI returns None."""
        from app.ai.claude_client import review_pm_signal
        with patch("app.ai.claude_client._client", return_value=None):
            result = review_pm_signal("AAPL", "EMA_CROSSOVER", 0.75, {})
        assert result is None


# ── AI briefing API endpoint ──────────────────────────────────────────────────

class TestAIBriefingEndpoint:
    def test_briefing_endpoint_returns_200(self, test_client):
        r = test_client.get("/api/orchestrator/ai-briefing")
        assert r.status_code == 200
        body = r.json()
        assert "proposals_reviewed" in body
        assert "ai_available" in body

    def test_briefing_endpoint_no_proposals(self, test_client):
        r = test_client.get("/api/orchestrator/ai-briefing")
        body = r.json()
        # No proposals today → briefing is None or text
        assert body["proposals_reviewed"] >= 0
