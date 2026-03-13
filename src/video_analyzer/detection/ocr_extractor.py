# src\detection\ocr_extractor.py
"""
Lightweight OCR / text region detection for video frames.

Detects text regions and optionally extracts text using OpenCV-based
morphological approach. Extracted text is injected into the LLM prompt
as additional context to improve price/promo/URL extraction accuracy.

No new dependencies required — uses OpenCV (already installed).
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class OCRExtractor:
    """
    Lightweight text detection and extraction using OpenCV.

    Uses MSER (Maximally Stable Extremal Regions) for text region detection,
    which is effective for high-contrast text overlays common in advertisements.
    """

    def __init__(self, min_area: int = 100, max_area: int = 50000):
        """
        Args:
            min_area: Minimum text region area in pixels
            max_area: Maximum text region area in pixels
        """
        self.min_area = min_area
        self.max_area = max_area

    def detect_text_regions(
        self, frame: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect text regions in a frame.

        Uses MSER + morphological grouping to find text-like regions.

        Args:
            frame: BGR numpy array

        Returns:
            List of (x, y, w, h) bounding boxes for detected text regions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # MSER for stable text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Filter and group regions
        text_boxes = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            area = w * h

            if area < self.min_area or area > self.max_area:
                continue

            # Text heuristics: moderate aspect ratio
            aspect_ratio = w / max(h, 1)
            if 0.1 < aspect_ratio < 15 and h > 5:
                text_boxes.append((x, y, w, h))

        # Group overlapping boxes
        if text_boxes:
            text_boxes = self._merge_overlapping(text_boxes)

        return text_boxes

    def _merge_overlapping(
        self, boxes: List[Tuple[int, int, int, int]], overlap_threshold: float = 0.3
    ) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes using NMS-like approach."""
        if not boxes:
            return []

        # Convert to numpy for NMS
        np_boxes = np.array(boxes)
        x1 = np_boxes[:, 0]
        y1 = np_boxes[:, 1]
        x2 = x1 + np_boxes[:, 2]
        y2 = y1 + np_boxes[:, 3]
        areas = np_boxes[:, 2] * np_boxes[:, 3]

        # Sort by area (largest first)
        order = areas.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            overlap = (w * h) / areas[order[1:]]
            inds = np.where(overlap <= overlap_threshold)[0]
            order = order[inds + 1]

        return [boxes[i] for i in keep]

    def extract_text_context(
        self, frame: np.ndarray, timestamp: float
    ) -> Dict[str, Any]:
        """
        Extract text context from a single frame.

        Args:
            frame: BGR numpy array
            timestamp: Frame timestamp in seconds

        Returns:
            Dictionary with text context:
            {
                "timestamp": float,
                "has_text": bool,
                "text_region_count": int,
                "text_coverage": float (0-1, fraction of frame covered by text),
            }
        """
        regions = self.detect_text_regions(frame)
        frame_area = frame.shape[0] * frame.shape[1]

        text_area = sum(w * h for _, _, w, h in regions)
        coverage = text_area / frame_area if frame_area > 0 else 0.0

        return {
            "timestamp": timestamp,
            "has_text": len(regions) > 0,
            "text_region_count": len(regions),
            "text_coverage": round(coverage, 4),
        }

    def extract_batch(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> List[Dict[str, Any]]:
        """
        Extract text context from a batch of frames.

        Args:
            frames: List of (timestamp, frame) tuples

        Returns:
            List of text context dicts
        """
        results = []
        for ts, frame in frames:
            try:
                results.append(self.extract_text_context(frame, ts))
            except Exception as e:
                logger.warning(f"OCR extraction failed for frame at {ts:.1f}s: {e}")
                results.append({
                    "timestamp": ts,
                    "has_text": False,
                    "text_region_count": 0,
                    "text_coverage": 0.0,
                })
        return results

    def build_ocr_context_for_prompt(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> str:
        """
        Build OCR context string for injection into LLM prompt.

        Args:
            frames: List of (timestamp, frame) tuples

        Returns:
            Formatted string with text detection summary
        """
        results = self.extract_batch(frames)

        text_frames = [r for r in results if r["has_text"]]
        if not text_frames:
            return ""

        context = "TEXT DETECTION (OCR Pre-Processing):\n"
        for r in text_frames:
            context += (
                f"  Frame @ {r['timestamp']:.1f}s: "
                f"{r['text_region_count']} text regions, "
                f"{r['text_coverage']:.1%} coverage\n"
            )

        return context
