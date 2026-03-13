# src\detection\visual_features.py
"""
Visual feature detection for frame importance scoring.

Detects:
- Text regions (OpenCV morphological approach)
- Faces (Haar cascade)
- Text density estimation

All detectors use OpenCV — no additional dependencies.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class VisualFeatureDetector:
    """
    Lightweight visual feature detector using OpenCV.

    Detects text, faces, and estimates text density in video frames.
    Used by ImportanceScorer to prioritize frames with important content.
    """

    def __init__(self):
        self._face_cascade = None

    def _get_face_cascade(self):
        """Lazily load Haar cascade for face detection."""
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
        return self._face_cascade

    def detect_text(self, frame: np.ndarray) -> bool:
        """
        Detect presence of text in a frame using morphological operations.

        Uses edge detection + morphological closing to find text-like regions
        (dense horizontal clusters of edges).

        Args:
            frame: BGR numpy array

        Returns:
            True if text-like regions detected
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Sobel for horizontal edges (text tends to be horizontal)
        sobel_x = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)

        # Threshold
        _, binary = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological closing to connect text characters
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        text_regions = 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            # Text region heuristics: reasonable aspect ratio and minimum size
            aspect_ratio = w / max(h, 1)
            relative_area = area / frame_area

            if (
                aspect_ratio > 1.5
                and 0.001 < relative_area < 0.3
                and h > 8
                and w > 20
            ):
                text_regions += 1

        return text_regions >= 1

    def detect_faces(self, frame: np.ndarray) -> bool:
        """
        Detect faces using Haar cascade.

        Args:
            frame: BGR numpy array

        Returns:
            True if any faces detected
        """
        cascade = self._get_face_cascade()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        return len(faces) > 0

    def estimate_text_density(self, frame: np.ndarray) -> str:
        """
        Estimate text density level.

        Args:
            frame: BGR numpy array

        Returns:
            "low", "medium", or "high"
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Edge detection for text-like regions
        edges = cv2.Canny(gray, 50, 150)

        # Count edge pixels
        edge_ratio = np.count_nonzero(edges) / edges.size

        if edge_ratio > 0.15:
            return "high"
        elif edge_ratio > 0.05:
            return "medium"
        else:
            return "low"

    def detect_all(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run all detectors on a single frame.

        Args:
            frame: BGR numpy array

        Returns:
            Dictionary with detection results:
            {
                "has_text": bool,
                "has_face": bool,
                "text_density": str ("low"/"medium"/"high"),
            }
        """
        return {
            "has_text": self.detect_text(frame),
            "has_face": self.detect_faces(frame),
            "text_density": self.estimate_text_density(frame),
        }

    def detect_batch(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> Dict[float, Dict[str, Any]]:
        """
        Run all detectors on a batch of frames.

        Args:
            frames: List of (timestamp, frame) tuples

        Returns:
            Dictionary mapping timestamp -> detection results
        """
        results = {}
        for ts, frame in frames:
            try:
                results[ts] = self.detect_all(frame)
            except Exception as e:
                logger.warning(f"Visual feature detection failed for frame at {ts:.1f}s: {e}")
                results[ts] = {
                    "has_text": False,
                    "has_face": False,
                    "text_density": "low",
                }
        return results
