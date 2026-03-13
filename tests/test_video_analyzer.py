"""
Consolidated test suite for KriNein.
"""
import pytest
import numpy as np
from pathlib import Path


class TestImports:
    """Test that all core modules can be imported."""

    def test_pipeline_import(self):
        from video_analyzer import AdVideoPipeline, ParallelVideoPipeline
        assert AdVideoPipeline is not None
        assert ParallelVideoPipeline is not None

    def test_utils_imports(self):
        from video_analyzer.utils.logging import setup_logging
        from video_analyzer.utils.frame_store import FrameStore, LazyFrame
        assert setup_logging is not None
        assert FrameStore is not None

    def test_deduplication_imports(self):
        from video_analyzer.deduplication.base import BaseDeduplicator
        from video_analyzer.deduplication.hierarchical import HierarchicalDeduplicator
        assert BaseDeduplicator is not None

    def test_selection_imports(self):
        from video_analyzer.selection.representative import FrameSelector
        from video_analyzer.selection.clustering import TemporalClusterer
        assert FrameSelector is not None

    def test_extraction_imports(self):
        from video_analyzer.extraction.llm_client import AdExtractor
        from video_analyzer.extraction.schema import FrameForPrompt
        assert AdExtractor is not None
        assert FrameForPrompt is not None

    def test_detection_imports(self):
        from video_analyzer.detection.scene_detector import SceneDetector
        from video_analyzer.detection.ocr_extractor import OCRExtractor
        from video_analyzer.detection.visual_features import VisualFeatureDetector
        assert SceneDetector is not None
        assert OCRExtractor is not None
        assert VisualFeatureDetector is not None

    def test_ingestion_imports(self):
        from video_analyzer.ingestion.video_loader import VideoLoader
        from video_analyzer.ingestion.audio_extractor import AudioExtractor
        assert VideoLoader is not None
        assert AudioExtractor is not None


class TestJSONParsing:
    """Test JSON parsing utilities."""

    def test_parse_clean_json(self):
        from video_analyzer.extraction.llm_client import _parse_json_response
        result = _parse_json_response('{"a": 1}')
        assert result == {"a": 1}

    def test_parse_markdown_json(self):
        from video_analyzer.extraction.llm_client import _parse_json_response
        result = _parse_json_response('```json\n{"b": 2}\n```')
        assert result == {"b": 2}

    def test_parse_surrounded_json(self):
        from video_analyzer.extraction.llm_client import _parse_json_response
        result = _parse_json_response('Here is result: {"c": 3} done')
        assert result == {"c": 3}

    def test_parse_trailing_comma(self):
        from video_analyzer.extraction.llm_client import _parse_json_response
        result = _parse_json_response('{"a": 1, "b": 2,}')
        assert result == {"a": 1, "b": 2}


class TestRetryLogic:
    """Test retry with backoff."""

    def test_immediate_success(self):
        from video_analyzer.extraction.llm_client import _retry_with_backoff
        result = _retry_with_backoff(lambda: "ok", max_retries=1)
        assert result == "ok"


class TestConfidenceScoring:
    """Test confidence computation."""

    def test_error_score(self):
        from video_analyzer.extraction.llm_client import compute_confidence
        score = compute_confidence({"error": "test"})
        assert score == 0.0

    def test_bounded_score(self):
        from video_analyzer.extraction.llm_client import compute_confidence
        score = compute_confidence({"a": "b", "c": {"d": "e"}}, num_frames=5)
        assert 0.0 <= score <= 1.0


class TestMockLLMClient:
    """Test mock LLM client."""

    def test_mock_extraction(self):
        import json
        from video_analyzer.extraction.llm_client import MockLLMClient
        from video_analyzer.extraction.schema import FrameForPrompt

        mock = MockLLMClient()
        frames = [FrameForPrompt(timestamp=1.0, base64_image="abc")]
        result = json.loads(mock.extract(frames, "test"))
        assert "ad_type" in result


class TestVisualFeatureDetector:
    """Test visual feature detection."""

    def test_detect_all(self):
        from video_analyzer.detection.visual_features import VisualFeatureDetector

        detector = VisualFeatureDetector()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = detector.detect_all(frame)

        assert "has_text" in result
        assert "has_face" in result
        assert "text_density" in result


class TestFrameStore:
    """Test frame storage utilities."""

    def test_save_load_roundtrip(self):
        from video_analyzer.utils.frame_store import FrameStore

        store = FrameStore()
        frame = np.zeros((50, 50, 3), dtype=np.uint8)
        store.save(1.5, frame)
        loaded = store.load(1.5)

        assert loaded is not None
        assert loaded.shape == (50, 50, 3)
        store.cleanup()


class TestFrameSelector:
    """Test frame selection logic."""

    def test_compute_frame_budget(self):
        from video_analyzer.selection.representative import FrameSelector

        selector = FrameSelector(global_max_frames=25, use_visual_features=False)
        b1 = selector._compute_frame_budget(2.0)
        b2 = selector._compute_frame_budget(200.0)

        assert b1 == 5  # min
        assert b2 == 25  # max cap


class TestPromptBuilding:
    """Test prompt building utilities."""

    def test_single_pass_prompt(self):
        from video_analyzer.extraction.prompts import build_single_pass_prompt
        from video_analyzer.extraction.schema import FrameForPrompt

        prompt = build_single_pass_prompt(
            [FrameForPrompt(timestamp=1.0, base64_image="abc")],
            10.0,
            {"brand": "string"},
        )
        assert "ad_type" in prompt

    def test_segmented_prompt(self):
        from video_analyzer.extraction.prompts import build_segmented_prompt
        from video_analyzer.extraction.schema import FrameForPrompt

        frames = [
            FrameForPrompt(timestamp=0.5, base64_image="abc", position_label="OPENING"),
            FrameForPrompt(timestamp=3.0, base64_image="def"),
        ]
        prompt = build_segmented_prompt(
            frames, 10.0, {"brand": "string"}, [(0.0, 2.0), (2.0, 10.0)]
        )

        assert "SCENE 1" in prompt
        assert "SCENE 2" in prompt


class TestLoggingSetup:
    """Test logging configuration."""

    def test_setup_logging(self):
        from video_analyzer.utils.logging import setup_logging
        import logging

        # Should not raise
        setup_logging(level="INFO")
        setup_logging(level="DEBUG")

        # Verify logging is configured
        assert logging.getLogger().level <= logging.DEBUG
