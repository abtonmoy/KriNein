"""
Test 3: Color Histogram Correlation deduplication.

Converts frames to HSV, computes 3D histograms, and keeps frames
whose correlation with the last kept frame falls below threshold.
"""

from typing import Any, List, Tuple
import cv2
import numpy as np

from benchmarks.base import BaselineMethod, _maybe_resize


class HistogramDedup(BaselineMethod):

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    @property
    def name(self) -> str:
        return "histogram"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        threshold = kwargs.get("histogram_threshold", self.threshold)
        max_res = kwargs.get("max_resolution", 720)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        # Sample at ~100ms intervals to match pipeline candidate extraction rate
        step = max(1, int(round(fps * 0.1)))

        selected: List[Tuple[float, np.ndarray]] = []
        prev_hist = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = _maybe_resize(frame, max_res)
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist(
                    [hsv], [0, 1, 2], None,
                    [16, 16, 16], [0, 180, 0, 256, 0, 256]
                )
                cv2.normalize(hist, hist)
                hist = hist.flatten()

                if prev_hist is None:
                    selected.append((idx / fps, frame))
                    prev_hist = hist
                else:
                    corr = cv2.compareHist(
                        prev_hist, hist, cv2.HISTCMP_CORREL
                    )
                    if corr < threshold:
                        selected.append((idx / fps, frame))
                        prev_hist = hist
            idx += 1

        cap.release()
        return selected