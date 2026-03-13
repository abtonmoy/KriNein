# src\utils\frame_store.py
"""
Disk-based frame storage for memory-efficient processing.

For long or high-resolution videos, keeping all candidate frames in RAM
as numpy arrays can consume several GB. This module provides disk-backed
frame storage using JPEG compression, loading frames on demand.

Usage:
    store = FrameStore(base_dir="/tmp/video_frames")
    store.save(1.5, frame_array)
    frame = store.load(1.5)
    store.cleanup()
"""

import os
import shutil
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FrameStore:
    """
    Disk-backed frame storage with JPEG compression.

    Saves frames as JPEG files in a temporary directory.
    Reduces memory pressure for pipelines processing many frames.
    """

    def __init__(
        self,
        base_dir: Optional[str] = None,
        quality: int = 95,
        prefix: str = "pipeline_frames_",
    ):
        """
        Args:
            base_dir: Base directory for frame storage (uses tempdir if None)
            quality: JPEG compression quality (1-100)
            prefix: Prefix for temp directory name
        """
        if base_dir:
            self.frame_dir = Path(base_dir)
            self.frame_dir.mkdir(parents=True, exist_ok=True)
            self._is_temp = False
        else:
            self.frame_dir = Path(tempfile.mkdtemp(prefix=prefix))
            self._is_temp = True

        self.quality = quality
        self._frame_index: Dict[float, str] = {}  # timestamp -> filename
        logger.info(f"FrameStore initialized at: {self.frame_dir}")

    def save(self, timestamp: float, frame: np.ndarray) -> str:
        """
        Save a frame to disk.

        Args:
            timestamp: Frame timestamp in seconds
            frame: BGR numpy array

        Returns:
            Path to saved frame file
        """
        filename = f"frame_{timestamp:.4f}.jpg"
        filepath = self.frame_dir / filename

        cv2.imwrite(str(filepath), frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        self._frame_index[timestamp] = str(filepath)

        return str(filepath)

    def load(self, timestamp: float) -> Optional[np.ndarray]:
        """
        Load a frame from disk.

        Args:
            timestamp: Frame timestamp

        Returns:
            BGR numpy array, or None if not found
        """
        filepath = self._frame_index.get(timestamp)
        if filepath and os.path.exists(filepath):
            return cv2.imread(filepath)
        return None

    def save_batch(
        self, frames: List[Tuple[float, np.ndarray]]
    ) -> List[Tuple[float, str]]:
        """
        Save a batch of frames to disk.

        Args:
            frames: List of (timestamp, frame) tuples

        Returns:
            List of (timestamp, filepath) tuples
        """
        results = []
        for ts, frame in frames:
            path = self.save(ts, frame)
            results.append((ts, path))
        return results

    def load_batch(
        self, timestamps: Optional[List[float]] = None
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Load frames from disk.

        Args:
            timestamps: Specific timestamps to load (None = all)

        Returns:
            List of (timestamp, frame) tuples
        """
        if timestamps is None:
            timestamps = sorted(self._frame_index.keys())

        results = []
        for ts in timestamps:
            frame = self.load(ts)
            if frame is not None:
                results.append((ts, frame))

        return results

    def get_timestamps(self) -> List[float]:
        """Get all stored timestamps, sorted."""
        return sorted(self._frame_index.keys())

    def __len__(self) -> int:
        """Number of stored frames."""
        return len(self._frame_index)

    def __contains__(self, timestamp: float) -> bool:
        """Check if a timestamp has a stored frame."""
        return timestamp in self._frame_index

    def cleanup(self):
        """Remove all stored frames and temp directory."""
        if self._is_temp and self.frame_dir.exists():
            shutil.rmtree(self.frame_dir, ignore_errors=True)
            logger.info(f"FrameStore cleaned up: {self.frame_dir}")
        self._frame_index.clear()

    def __del__(self):
        """Auto-cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass


class LazyFrame:
    """
    Lazy-loading wrapper around a disk-stored frame.

    Loads the frame from disk only when accessed, keeping memory usage low.
    """

    def __init__(self, timestamp: float, filepath: str):
        self.timestamp = timestamp
        self.filepath = filepath
        self._frame = None

    @property
    def frame(self) -> Optional[np.ndarray]:
        """Load and cache frame on first access."""
        if self._frame is None:
            if os.path.exists(self.filepath):
                self._frame = cv2.imread(self.filepath)
        return self._frame

    def release(self):
        """Release cached frame from memory."""
        self._frame = None

    @property
    def shape(self) -> Optional[tuple]:
        """Get frame shape without loading full frame (reads header only)."""
        if self._frame is not None:
            return self._frame.shape
        if os.path.exists(self.filepath):
            f = self.frame
            return f.shape if f is not None else None
        return None
