# Ingestion Module — Changelog

## Audio Extractor Improvements

### Performance: Single-Load Audio Pipeline
Audio data (`y`, `sr`) is now loaded **once** per `extract_full_context()` call and passed to all sub-methods via a `preloaded_audio` parameter. Previously, each sub-method called `load_audio()` independently, causing the same file to be decoded 5+ times per video.

### Performance: Whisper Model Caching
Whisper models are cached in `_whisper_models` dict by size key. Calling `_get_whisper_model("base")` multiple times returns the same model object — no redundant loading between videos processed by the same `AudioExtractor` instance.

### Accuracy: ML-Based Mood Classification
Added `classify_mood()` with optional ML classification via HuggingFace `transformers` audio classifier. Falls back to a refined heuristic analysis (RMS energy, spectral centroid, tempo) if the ML model is unavailable. The heuristic has 6 categories: energetic, upbeat, calm, dramatic, melancholic, neutral.

### Performance: Pre-Detected Speech Segments
`extract_full_context()` accepts an optional `pre_detected_speech` parameter. When the pipeline has already run VAD upstream (e.g., to decide whether to skip transcription), those segments are reused instead of re-running detection.
