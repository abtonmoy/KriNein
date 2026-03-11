# Ingestion Module

Audio and video ingestion components for the ad analysis pipeline.

## Modules

### `video_loader.py`
Video file loading and frame extraction using OpenCV. Handles format detection, resolution, FPS, and frame-level access.

### `audio_extractor.py`
Comprehensive audio feature extraction for video advertisements.

**Capabilities:**
- **Audio loading** via `librosa` with configurable sample rate
- **Speech detection** via WebRTC VAD (Voice Activity Detection)
- **Transcription** via OpenAI Whisper (with model caching across calls)
- **Energy analysis** — peak detection, silence segmentation
- **Mood classification** — ML-based (HuggingFace) with heuristic fallback
- **Key phrase detection** — identifies promotional keywords in transcriptions

**Key API:**
```python
extractor = AudioExtractor()

# Full context extraction (audio loaded once, reused across all sub-methods)
context = extractor.extract_full_context(
    audio_path,
    transcribe=True,
    model_size="base",
    pre_detected_speech=None,  # Optional: reuse upstream VAD results
)

# Individual methods (all accept preloaded_audio=(y, sr) to skip reloading)
peaks = extractor.extract_energy_peaks(audio_path, preloaded_audio=(y, sr))
silence = extractor.detect_silence(audio_path, preloaded_audio=(y, sr))
mood = extractor.classify_mood(audio_path, use_ml=True, preloaded_audio=(y, sr))
```

**Performance Notes:**
- Audio data loaded once per `extract_full_context()` call, not per sub-method
- Whisper models cached across calls via `_get_whisper_model(size)`
- Pre-detected speech segments can be passed to avoid redundant VAD
