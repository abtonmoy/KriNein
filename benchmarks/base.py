"""
Abstract base class for all baseline frame-selection methods.

Every baseline must return List[Tuple[float, np.ndarray]] — the exact
format that AdExtractor.extract() expects as its `frames` parameter.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class BaselineMethod(ABC):
    """Interface that all benchmark baselines implement."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier: 'uniform_1fps', 'histogram', etc."""
        ...

    @property
    def requires_gpu(self) -> bool:
        """Whether this method needs CLIP / GPU resources."""
        return False

    @abstractmethod
    def select_frames(
        self,
        video_path: str,
        **kwargs: Any,
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Select keyframes from a raw video file.

        Keyword args may include shared resources injected by the runner:
            target_k          – desired frame count (for random sampling)
            all_frames         – pre-decoded List[Tuple[float, np.ndarray]]
            clip_embeddings    – np.ndarray (N, 512) pre-computed CLIP vectors
            max_resolution     – int, max pixel dim for resizing (default 720)

        Returns:
            Sorted list of (timestamp_seconds, bgr_frame) tuples.
        """
        ...

    def run_timed(
        self, video_path: str, **kwargs: Any
    ) -> Tuple[List[Tuple[float, np.ndarray]], float]:
        """Run select_frames and return (frames, wall_clock_seconds)."""
        t0 = time.perf_counter()
        frames = self.select_frames(video_path, **kwargs)
        elapsed = time.perf_counter() - t0
        logger.info(f"[{self.name}] Selected {len(frames)} frames in {elapsed:.2f}s")
        return frames, elapsed


# ---------------------------------------------------------------------------
# Shared video-decoding helpers used by CPU-bound baselines
# ---------------------------------------------------------------------------

def decode_frames_at_interval(
    video_path: str,
    interval_ms: float = 100.0,
    max_resolution: int = 720,
) -> List[Tuple[float, np.ndarray]]:
    """
    Decode video frames at a fixed time interval.

    Returns list of (timestamp_s, bgr_frame) sorted by time.
    Used by baselines that need access to regularly-sampled frames
    (random, histogram, orb, optical_flow).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps

    step_frames = max(1, int(round(fps * interval_ms / 1000.0)))
    frames: List[Tuple[float, np.ndarray]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step_frames == 0:
            frame = _maybe_resize(frame, max_resolution)
            ts = idx / fps
            frames.append((ts, frame))
        idx += 1

    cap.release()
    return frames


def decode_all_frames(
    video_path: str,
    max_resolution: int = 720,
) -> Tuple[List[Tuple[float, np.ndarray]], float, int]:
    """
    Decode every frame. Returns (frames, fps, total_count).
    Used sparingly — mainly for uniform sampling where we need exact frame indices.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: List[Tuple[float, np.ndarray]] = []
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = _maybe_resize(frame, max_resolution)
        frames.append((idx / fps, frame))
        idx += 1

    cap.release()
    return frames, fps, idx


def get_video_info(video_path: str) -> Tuple[int, float, float]:
    """Return (total_frames, fps, duration_s)."""
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return n, fps, n / fps


def _maybe_resize(frame: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame = cv2.resize(
            frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return frame