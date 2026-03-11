# Detection Module — Changelog

## New: Visual Feature Detection (`visual_features.py`)
OpenCV-based detectors for frame importance scoring:
- **Text detection** — morphological operations to find text-like horizontal regions
- **Face detection** — Haar cascade classifier
- **Text density estimation** — edge ratio analysis (low/medium/high)

Integrated into `FrameSelector` — automatically runs on candidate frames to boost importance scores for frames with text overlays and faces.

## New: OCR Extraction (`ocr_extractor.py`)
MSER-based text region detection:
- Detects text regions with NMS-style merging
- Computes per-frame text coverage
- Generates context strings for LLM prompt injection
- Improves extraction of prices, promo codes, and URLs
