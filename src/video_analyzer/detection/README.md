# Detection Module

The `detection` module handles scene detection, change detection, OCR, and visual feature extraction.

## Overview

This module provides tools for analyzing video content to identify:
- Scene boundaries (where shots change)
- Frame-level changes (histogram differences)
- Text regions (OCR)
- Visual features (faces, text density, etc.)

## Components

### SceneDetector

Detects scene boundaries using PySceneDetect.

```python
from video_analyzer.detection import SceneDetector

detector = SceneDetector(
    method="content",           # "content" or "threshold"
    threshold=27.0,             # Sensitivity threshold
    min_scene_length_s=0.5      # Minimum scene duration
)

scenes = detector.detect_scenes("path/to/video.mp4")
# Returns: [(0.0, 2.5), (2.5, 5.8), ...]
```

**Methods:**
- `content`: Content-aware detection (recommended)
- `threshold`: Simple threshold-based detection

### ChangeDetector

Detects frame-level changes using histogram comparison.

```python
from video_analyzer.detection import ChangeDetector

detector = ChangeDetector(
    method="histogram",
    threshold=0.15,
    min_interval_ms=100
)

changes = detector.detect_changes(frame1, frame2)
```

### OCRExtractor

Extracts text from frames using Tesseract OCR.

```python
from video_analyzer.detection.ocr_extractor import OCRExtractor

ocr = OCRExtractor()
text_regions = ocr.extract(frame)
# Returns: [(text, confidence, bbox), ...]
```

### VisualFeatureDetector

Detects visual features like text density, faces, and visual complexity.

```python
from video_analyzer.detection import VisualFeatureDetector

detector = VisualFeatureDetector()
features = detector.detect_all(frame)

# Returns dict with:
# - has_text: bool
# - has_face: bool
# - text_density: float
# - visual_complexity: float
```

## Configuration

```yaml
scene_detection:
  method: content
  threshold: 27.0
  min_scene_length: 0.5

change_detection:
  method: histogram
  threshold: 0.15
  min_interval_ms: 100

ocr:
  enabled: true
  lang: eng
```

## Dependencies

- **PySceneDetect**: Scene boundary detection
- **OpenCV**: Image processing
- **Tesseract**: OCR text extraction
- **NumPy**: Array operations

## Output Format

**Scene boundaries**: List of tuples `[(start_time, end_time), ...]` in seconds

**Visual features**: Dictionary with boolean and float values for each detected feature
