"""
Tests for llm_client.py improvements:
- Improvement 3: Retry with exponential backoff
- Improvement 4: Single-pass extraction
- Improvement 9: Robust JSON parsing
- Improvement 12: Confidence scoring
"""

import json
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from src.extraction.llm_client import (
    _retry_with_backoff,
    _parse_json_response,
    compute_confidence,
    MockLLMClient,
    AdExtractor,
)
from src.extraction.prompts import FrameForPrompt


# ============================================================================
# Improvement 3: Retry Logic
# ============================================================================

class TestRetryWithBackoff:
    """Test retry mechanism."""

    def test_succeeds_first_try(self):
        """Should return immediately on first success."""
        result = _retry_with_backoff(lambda: "ok", max_retries=3)
        assert result == "ok"

    def test_retries_on_transient_error(self):
        """Should retry on ConnectionError."""
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ConnectionError("Connection reset")
            return "ok"

        result = _retry_with_backoff(flaky, max_retries=3, base_delay=0.01)
        assert result == "ok"
        assert call_count[0] == 3

    def test_exhausts_retries(self):
        """Should raise after all retries exhausted."""
        def always_fail():
            raise ConnectionError("Always fails")

        with pytest.raises(ConnectionError):
            _retry_with_backoff(always_fail, max_retries=2, base_delay=0.01)

    def test_non_retryable_error_raised_immediately(self):
        """Non-retryable errors should not be retried."""
        call_count = [0]

        def bad_code():
            call_count[0] += 1
            raise ValueError("Bad argument")

        with pytest.raises(ValueError):
            _retry_with_backoff(bad_code, max_retries=3, base_delay=0.01)

        assert call_count[0] == 1, "Should not retry ValueError"

    def test_rate_limit_error_retried(self):
        """Rate limit errors (keyword match) should be retried."""
        call_count = [0]

        class RateLimitError(Exception):
            pass

        def rate_limited():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RateLimitError("rate limit exceeded")
            return "ok"

        result = _retry_with_backoff(rate_limited, max_retries=3, base_delay=0.01)
        assert result == "ok"


# ============================================================================
# Improvement 9: Robust JSON Parsing
# ============================================================================

class TestRobustJsonParsing:
    """Test JSON extraction from various LLM response formats."""

    def test_clean_json(self):
        """Parse clean JSON."""
        result = _parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_markdown_code_block(self):
        """Parse JSON inside markdown code block."""
        response = '```json\n{"brand": "Test"}\n```'
        result = _parse_json_response(response)
        assert result == {"brand": "Test"}

    def test_code_block_without_language(self):
        """Parse JSON inside code block without 'json' label."""
        response = '```\n{"brand": "Test"}\n```'
        result = _parse_json_response(response)
        assert result == {"brand": "Test"}

    def test_json_with_surrounding_text(self):
        """Extract JSON from response with explanatory text."""
        response = 'Here is the extraction:\n{"brand": "Test"}\nDone!'
        result = _parse_json_response(response)
        assert result == {"brand": "Test"}

    def test_trailing_comma_fixed(self):
        """Handle trailing commas in JSON."""
        response = '{"a": 1, "b": 2,}'
        result = _parse_json_response(response)
        assert result == {"a": 1, "b": 2}

    def test_nested_json(self):
        """Parse nested JSON objects."""
        data = {"brand": {"name": "Test", "logo": True}, "score": 4.5}
        response = json.dumps(data)
        result = _parse_json_response(response)
        assert result == data

    def test_invalid_json_raises(self):
        """Should raise JSONDecodeError for truly invalid input."""
        with pytest.raises(json.JSONDecodeError):
            _parse_json_response("This is not JSON at all")


# ============================================================================
# Improvement 4: Single-Pass Extraction
# ============================================================================

class TestSinglePassExtraction:
    """Test merged type detection + extraction."""

    def test_mock_single_pass_includes_ad_type(self):
        """Mock client response should include ad_type in single-pass mode."""
        extractor = AdExtractor(
            provider="mock",
            single_pass=True,
            max_retries=0,
        )

        frames = [(0.5, np.zeros((100, 100, 3), dtype=np.uint8))]
        result = extractor.extract(frames, 10.0)

        assert "ad_type" in result, "Single-pass should include ad_type"
        assert "error" not in result
        assert result["_metadata"]["single_pass"] is True

    def test_single_pass_one_api_call(self):
        """Single-pass should only make one API call (not two)."""
        mock_client = MagicMock()
        mock_client.extract.return_value = json.dumps({
            "ad_type": "product_demo",
            "brand": {"brand_name_text": "Test"},
        })

        extractor = AdExtractor(provider="mock", single_pass=True)
        extractor.client = mock_client

        frames = [(0.5, np.zeros((100, 100, 3), dtype=np.uint8))]
        extractor.extract(frames, 10.0)

        # Should be called once (not twice for type detect + extract)
        assert mock_client.extract.call_count == 1


# ============================================================================
# Improvement 12: Confidence Scoring
# ============================================================================

class TestConfidenceScoring:
    """Test extraction confidence scoring."""

    def test_error_result_zero_confidence(self):
        """Error results should have 0.0 confidence."""
        assert compute_confidence({"error": "failed"}) == 0.0

    def test_complete_result_high_confidence(self):
        """Complete results with audio should have high confidence."""
        result = {
            "brand": {"name": "Test", "logo": True},
            "product": {"name": "Widget"},
            "promo": {"text": "50% off"},
        }
        audio_context = {
            "has_speech": True,
            "transcription": [{"text": "hello"}],
            "key_phrases": [{"text": "off"}],
        }

        score = compute_confidence(result, audio_context=audio_context, num_frames=5)
        assert score > 0.5, f"Complete result should have high confidence, got {score}"

    def test_empty_result_low_confidence(self):
        """Sparse results should have lower confidence."""
        result = {"brand": {"name": None}, "product": {"name": None}}
        score = compute_confidence(result, num_frames=1)
        assert score < 0.5, f"Empty result should have low confidence, got {score}"

    def test_confidence_bounded_zero_one(self):
        """Confidence should be in [0, 1]."""
        result = {"a": "b", "c": {"d": "e", "f": "g", "h": "i"}}
        audio = {"has_speech": True, "transcription": [{}], "key_phrases": [{}]}
        score = compute_confidence(result, audio_context=audio, num_frames=10)
        assert 0.0 <= score <= 1.0

    def test_metadata_includes_confidence(self):
        """extract() should include confidence in _metadata."""
        extractor = AdExtractor(provider="mock", single_pass=True, max_retries=0)
        frames = [(0.5, np.zeros((100, 100, 3), dtype=np.uint8))]
        result = extractor.extract(frames, 10.0)

        assert "confidence" in result.get("_metadata", {}), "Missing confidence in metadata"
        assert 0.0 <= result["_metadata"]["confidence"] <= 1.0


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
