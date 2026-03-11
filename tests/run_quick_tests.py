"""Quick test runner to verify imports and basic test execution."""
import sys
sys.path.insert(0, '.')

# Test 1: llm_client imports
try:
    from src.extraction.llm_client import (
        _retry_with_backoff,
        _parse_json_response,
        compute_confidence,
        MockLLMClient,
        AdExtractor,
    )
    print("[PASS] llm_client imports OK")
except Exception as e:
    print(f"[FAIL] llm_client imports: {e}")

# Test 2: prompts imports
try:
    from src.extraction.prompts import (
        build_single_pass_prompt,
        build_segmented_prompt,
        FrameForPrompt,
    )
    print("[PASS] prompts imports OK")
except Exception as e:
    print(f"[FAIL] prompts imports: {e}")

# Test 3: visual features
try:
    from src.detection.visual_features import VisualFeatureDetector
    print("[PASS] visual_features imports OK")
except Exception as e:
    print(f"[FAIL] visual_features imports: {e}")

# Test 4: ocr extractor
try:
    from src.detection.ocr_extractor import OCRExtractor
    print("[PASS] ocr_extractor imports OK")
except Exception as e:
    print(f"[FAIL] ocr_extractor imports: {e}")

# Test 5: frame store
try:
    from src.utils.frame_store import FrameStore, LazyFrame
    print("[PASS] frame_store imports OK")
except Exception as e:
    print(f"[FAIL] frame_store imports: {e}")

# Test 6: audio extractor
try:
    from src.ingestion.audio_extractor import AudioExtractor
    print("[PASS] audio_extractor imports OK")
except Exception as e:
    print(f"[FAIL] audio_extractor imports: {e}")

# Test 7: representative
try:
    from src.selection.representative import FrameSelector, create_selector
    print("[PASS] representative imports OK")
except Exception as e:
    print(f"[FAIL] representative imports: {e}")

print("\n--- Running quick functional tests ---")

# Quick test: JSON parsing
try:
    result = _parse_json_response('{"a": 1}')
    assert result == {"a": 1}
    print("[PASS] _parse_json_response: clean JSON")
except Exception as e:
    print(f"[FAIL] _parse_json_response: {e}")

try:
    result = _parse_json_response('```json\n{"b": 2}\n```')
    assert result == {"b": 2}
    print("[PASS] _parse_json_response: markdown block")
except Exception as e:
    print(f"[FAIL] _parse_json_response: {e}")

try:
    result = _parse_json_response('Here is result: {"c": 3} done')
    assert result == {"c": 3}
    print("[PASS] _parse_json_response: surrounded text")
except Exception as e:
    print(f"[FAIL] _parse_json_response: {e}")

try:
    result = _parse_json_response('{"a": 1, "b": 2,}')
    assert result == {"a": 1, "b": 2}
    print("[PASS] _parse_json_response: trailing comma")
except Exception as e:
    print(f"[FAIL] _parse_json_response: {e}")

# Quick test: retry
try:
    result = _retry_with_backoff(lambda: "ok", max_retries=1)
    assert result == "ok"
    print("[PASS] _retry_with_backoff: immediate success")
except Exception as e:
    print(f"[FAIL] _retry_with_backoff: {e}")

# Quick test: confidence
try:
    score = compute_confidence({"error": "test"})
    assert score == 0.0
    print("[PASS] compute_confidence: error => 0.0")
except Exception as e:
    print(f"[FAIL] compute_confidence: {e}")

try:
    score = compute_confidence({"a": "b", "c": {"d": "e"}}, num_frames=5)
    assert 0.0 <= score <= 1.0
    print(f"[PASS] compute_confidence: bounded ({score:.3f})")
except Exception as e:
    print(f"[FAIL] compute_confidence: {e}")

# Quick test: MockLLMClient
try:
    import json
    import numpy as np
    mock = MockLLMClient()
    frames = [FrameForPrompt(timestamp=1.0, base64_image="abc")]
    result = json.loads(mock.extract(frames, "test"))
    assert "ad_type" in result
    print("[PASS] MockLLMClient: includes ad_type")
except Exception as e:
    print(f"[FAIL] MockLLMClient: {e}")

# Quick test: VisualFeatureDetector
try:
    import numpy as np
    detector = VisualFeatureDetector()
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = detector.detect_all(frame)
    assert "has_text" in result
    assert "has_face" in result
    assert "text_density" in result
    print("[PASS] VisualFeatureDetector: detect_all")
except Exception as e:
    print(f"[FAIL] VisualFeatureDetector: {e}")

# Quick test: FrameStore
try:
    import numpy as np
    store = FrameStore()
    frame = np.zeros((50, 50, 3), dtype=np.uint8)
    store.save(1.5, frame)
    loaded = store.load(1.5)
    assert loaded is not None
    assert loaded.shape == (50, 50, 3)
    store.cleanup()
    print("[PASS] FrameStore: save/load round-trip")
except Exception as e:
    print(f"[FAIL] FrameStore: {e}")

# Quick test: Smart frame budget
try:
    selector = FrameSelector(global_max_frames=25, use_visual_features=False)
    b1 = selector._compute_frame_budget(2.0)
    b2 = selector._compute_frame_budget(200.0)
    assert b1 == 5  # min
    assert b2 == 25  # max cap
    print(f"[PASS] SmartFrameBudget: short={b1}, long={b2}")
except Exception as e:
    print(f"[FAIL] SmartFrameBudget: {e}")

# Quick test: Single-pass prompt
try:
    prompt = build_single_pass_prompt(
        [FrameForPrompt(timestamp=1.0, base64_image="abc")],
        10.0,
        {"brand": "string"},
    )
    assert "ad_type" in prompt
    print("[PASS] build_single_pass_prompt: contains ad_type")
except Exception as e:
    print(f"[FAIL] build_single_pass_prompt: {e}")

# Quick test: Segmented prompt
try:
    frames = [
        FrameForPrompt(timestamp=0.5, base64_image="abc", position_label="OPENING"),
        FrameForPrompt(timestamp=3.0, base64_image="def"),
    ]
    prompt = build_segmented_prompt(frames, 10.0, {"brand": "string"}, [(0.0, 2.0), (2.0, 10.0)])
    assert "SCENE 1" in prompt
    assert "SCENE 2" in prompt
    print("[PASS] build_segmented_prompt: scene grouping")
except Exception as e:
    print(f"[FAIL] build_segmented_prompt: {e}")

print("\n--- All quick tests complete ---")
