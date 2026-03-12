"""
Test 1: Uniform Sampling at a fixed FPS (default 1 FPS).
"""

from typing import Any, List, Tuple
import cv2
import numpy as np

from benchmarks.base import BaselineMethod, _maybe_resize


class UniformSampling(BaselineMethod):
    """Extract frames at fixed time intervals."""

    def __init__(self, target_fps: float = 1.0):
        self.target_fps = target_fps

    @property
    def name(self) -> str:
        return f"uniform_{self.target_fps:.0f}fps"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        max_res = kwargs.get("max_resolution", 720)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps / self.target_fps)))

        frames: List[Tuple[float, np.ndarray]] = []
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = _maybe_resize(frame, max_res)
                frames.append((idx / fps, frame))
            idx += 1

        cap.release()
        return frames