"""
Tests for audio_extractor.py improvements:
- Improvement 1: Audio loaded once per extract_full_context()
- Improvement 2: Whisper model cached across calls
- Improvement 8: ML mood classification
- Improvement 11: Pre-detected speech segments accepted
"""

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from src.ingestion.audio_extractor import AudioExtractor


# ============================================================================
# Improvement 1: Audio Reload Elimination
# ============================================================================

class TestAudioReloadElimination:
    """Verify audio is loaded ONCE per extract_full_context()."""

    @patch.object(AudioExtractor, 'load_audio')
    @patch.object(AudioExtractor, 'transcribe_audio')
    def test_single_load_audio_call(self, mock_transcribe, mock_load):
        """extract_full_context should call load_audio exactly once."""
        # Setup mock
        mock_audio = (np.random.randn(16000 * 5), 16000)  # 5s of audio
        mock_load.return_value = mock_audio
        mock_transcribe.return_value = []

        extractor = AudioExtractor()
        extractor.extract_full_context("/fake/audio.wav", transcribe=False)

        # Should load once, not 5+ times
        assert mock_load.call_count == 1, (
            f"load_audio called {mock_load.call_count} times, expected 1"
        )

    @patch.object(AudioExtractor, 'load_audio')
    def test_preloaded_audio_passed_to_submethods(self, mock_load):
        """Sub-methods should receive preloaded audio, not call load_audio again."""
        mock_audio = (np.random.randn(16000 * 2), 16000)
        mock_load.return_value = mock_audio

        extractor = AudioExtractor()
        # Call with preloaded audio
        peaks = extractor.extract_energy_peaks("/fake/audio.wav", preloaded_audio=mock_audio)

        # load_audio should NOT be called when preloaded_audio is provided
        assert mock_load.call_count == 0, "load_audio should not be called when preloaded_audio is provided"

    @patch.object(AudioExtractor, 'load_audio')
    def test_silence_detection_with_preloaded(self, mock_load):
        """detect_silence should use preloaded audio."""
        mock_audio = (np.random.randn(16000 * 2), 16000)
        mock_load.return_value = mock_audio

        extractor = AudioExtractor()
        extractor.detect_silence("/fake/audio.wav", preloaded_audio=mock_audio)
        assert mock_load.call_count == 0


# ============================================================================
# Improvement 2: Whisper Model Caching
# ============================================================================

class TestWhisperModelCaching:
    """Verify Whisper model is loaded only once."""

    @patch('whisper.load_model')
    def test_model_loaded_once_across_calls(self, mock_load_model):
        """Whisper model should be loaded once and cached."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"segments": []}
        mock_load_model.return_value = mock_model

        extractor = AudioExtractor()
        extractor._whisper = MagicMock()  # Bypass lazy import

        # Get model multiple times
        model1 = extractor._get_whisper_model("base")
        model2 = extractor._get_whisper_model("base")

        assert mock_load_model.call_count == 1, "Model should be loaded once"
        assert model1 is model2, "Should return same cached model instance"

    @patch('whisper.load_model')
    def test_different_sizes_loaded_separately(self, mock_load_model):
        """Different model sizes get separate cache entries."""
        mock_load_model.return_value = MagicMock()

        extractor = AudioExtractor()
        extractor._whisper = MagicMock()

        extractor._get_whisper_model("base")
        extractor._get_whisper_model("small")

        assert mock_load_model.call_count == 2, "Each size should trigger one load"


# ============================================================================
# Improvement 8: ML Mood Classification
# ============================================================================

class TestMLMoodClassification:
    """Test ML-based mood classification with fallback."""

    def test_heuristic_mood_returns_valid_mood(self):
        """Heuristic mood should return a valid category."""
        extractor = AudioExtractor()
        audio = (np.random.randn(16000 * 5), 16000)

        mood = extractor._classify_mood_heuristic("/fake.wav", preloaded_audio=audio)
        valid_moods = {"energetic", "upbeat", "calm", "dramatic", "melancholic", "neutral"}
        assert mood in valid_moods, f"Invalid mood: {mood}"

    def test_ml_mood_falls_back_to_heuristic(self):
        """ML mood should fall back to heuristic if transformers unavailable."""
        extractor = AudioExtractor()
        audio = (np.random.randn(16000 * 5), 16000)

        # This should fall back gracefully since we may not have transformers
        mood = extractor.classify_mood("/fake.wav", use_ml=True, preloaded_audio=audio)
        assert isinstance(mood, str), "Should return a string mood"
        assert len(mood) > 0, "Should return non-empty mood"


# ============================================================================
# Improvement 11: Pre-detected Speech Segments
# ============================================================================

class TestPreDetectedSpeech:
    """Test passing pre-detected speech to avoid redundant detection."""

    @patch.object(AudioExtractor, 'load_audio')
    @patch.object(AudioExtractor, 'detect_speech_segments')
    def test_pre_detected_speech_bypasses_detection(self, mock_detect, mock_load):
        """extract_full_context should use pre_detected_speech without re-detecting."""
        mock_load.return_value = (np.random.randn(16000 * 2), 16000)

        extractor = AudioExtractor()
        pre_detected = [(0.5, 1.0), (2.0, 3.0)]

        context = extractor.extract_full_context(
            "/fake/audio.wav",
            transcribe=False,
            pre_detected_speech=pre_detected,
        )

        # Should NOT call detect_speech_segments since we pre-detected
        mock_detect.assert_not_called()
        assert context["speech_segments"] == pre_detected
        assert context["has_speech"] is True

    @patch.object(AudioExtractor, 'load_audio')
    @patch.object(AudioExtractor, 'detect_speech_segments')
    def test_empty_predetected_speech(self, mock_detect, mock_load):
        """Pre-detected empty list should be accepted (no speech)."""
        mock_load.return_value = (np.random.randn(16000 * 2), 16000)

        extractor = AudioExtractor()
        context = extractor.extract_full_context(
            "/fake/audio.wav",
            transcribe=False,
            pre_detected_speech=[],
        )

        mock_detect.assert_not_called()
        assert context["has_speech"] is False


# ============================================================================
# Run
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
