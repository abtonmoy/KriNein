"""
Test 2: Random Sampling — pick k frames at random.

k is set to match the pipeline's selected frame count for a fair comparison.
Falls back to ~1 FPS equivalent if no target_k is provided.
"""

import random
from typing import Any, List, Tuple
import numpy as np

from benchmarks.base import BaselineMethod, decode_frames_at_interval, get_video_info


class RandomSampling(BaselineMethod):

    @property
    def name(self) -> str:
        return "random"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        max_res = kwargs.get("max_resolution", 720)

        # Use pre-decoded frames if available, otherwise decode at 100ms
        all_frames = kwargs.get("all_frames")
        if all_frames is None:
            all_frames = decode_frames_at_interval(
                video_path, interval_ms=100.0, max_resolution=max_res
            )

        # Determine k
        target_k = kwargs.get("target_k")
        if target_k is None:
            _, _, dur = get_video_info(video_path)
            target_k = max(1, int(dur))  # ~1 FPS equivalent

        k = min(target_k, len(all_frames))
        selected = sorted(random.sample(all_frames, k), key=lambda x: x[0])
        return selected