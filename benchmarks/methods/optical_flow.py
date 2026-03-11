"""
Test 5: Optical Flow (Motion Peak) selection.

Two-pass approach:
  1. Compute Farneback flow magnitudes for all sampled frames.
  2. Select frames where motion is above the Nth percentile.
Always includes first and last frame.
"""

from typing import Any, List, Tuple
import cv2
import numpy as np

from benchmarks.base import BaselineMethod, _maybe_resize


class OpticalFlowPeaks(BaselineMethod):

    def __init__(self, percentile: float = 85.0):
        self.percentile = percentile

    @property
    def name(self) -> str:
        return "optical_flow"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        pct = kwargs.get("optical_flow_percentile", self.percentile)
        max_res = kwargs.get("max_resolution", 720)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * 0.1)))

        # --- Pass 1: compute magnitudes ---
        frame_data: List[Tuple[float, np.ndarray]] = []
        magnitudes: List[float] = []
        prev_gray = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = _maybe_resize(frame, max_res)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_data.append((idx / fps, frame))

                if prev_gray is None:
                    magnitudes.append(0.0)
                else:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None,
                        pyr_scale=0.5, levels=3, winsize=15,
                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
                    )
                    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    magnitudes.append(float(mag.mean()))

                prev_gray = gray
            idx += 1

        cap.release()

        if not frame_data:
            return []

        # --- Pass 2: select peaks ---
        thresh = np.percentile(magnitudes, pct)
        selected_idx = set()

        for i, m in enumerate(magnitudes):
            if m >= thresh:
                selected_idx.add(i)

        # Always include first and last
        selected_idx.add(0)
        selected_idx.add(len(frame_data) - 1)

        return [frame_data[i] for i in sorted(selected_idx)]