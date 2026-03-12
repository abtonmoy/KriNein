# Detection Module

Frame change detection, scene detection, and visual feature analysis.

## Modules

### `change_detector.py`
Adaptive change detection for initial frame candidate extraction. Methods: frame difference, histogram comparison, edge detection.

### `scene_detector.py`
Scene boundary detection using PySceneDetect. Identifies cuts and transitions for scene-level analysis.

### `visual_features.py`
Lightweight visual feature detector using OpenCV (no ML dependencies required).

**Detectors:**
- **Text** — Sobel + morphological closing to find text-like horizontal clusters
- **Faces** — Haar cascade (`haarcascade_frontalface_default.xml`)
- **Text density** — Canny edge ratio (low/medium/high)

```python
detector = VisualFeatureDetector()

# Single frame
result = detector.detect_all(frame)
# {"has_text": True, "has_face": False, "text_density": "medium"}

# Batch processing
results = detector.detect_batch([(0.5, frame1), (1.0, frame2)])
# {0.5: {...}, 1.0: {...}}
```

### `ocr_extractor.py`
MSER-based text region detection for pre-processing before LLM extraction.

```python
ocr = OCRExtractor()

# Per-frame analysis
context = ocr.extract_text_context(frame, timestamp=1.5)
# {"timestamp": 1.5, "has_text": True, "text_region_count": 3, "text_coverage": 0.12}

# Generate LLM prompt context
prompt_context = ocr.build_ocr_context_for_prompt(frames)
```
