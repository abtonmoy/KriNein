"""
Test 4: ORB Feature Matching deduplication.

Detects ORB keypoints, matches via BFMatcher with Hamming distance.
Keeps frame when too few good matches indicate a visual change.
"""

from typing import Any, List, Tuple
import cv2
import numpy as np

from benchmarks.base import BaselineMethod, _maybe_resize


class ORBDedup(BaselineMethod):

    def __init__(
        self,
        match_threshold: int = 40,
        distance_threshold: int = 50,
        n_features: int = 500,
    ):
        self.match_threshold = match_threshold
        self.distance_threshold = distance_threshold
        self.n_features = n_features

    @property
    def name(self) -> str:
        return "orb"

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        match_thresh = kwargs.get("orb_good_matches", self.match_threshold)
        dist_thresh = kwargs.get("orb_match_distance", self.distance_threshold)
        max_res = kwargs.get("max_resolution", 720)

        orb = cv2.ORB_create(nfeatures=self.n_features)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(1, int(round(fps * 0.1)))

        selected: List[Tuple[float, np.ndarray]] = []
        prev_des = None
        idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % step == 0:
                frame = _maybe_resize(frame, max_res)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, des = orb.detectAndCompute(gray, None)

                if des is None:
                    # Featureless frame — keep it (likely a title card / solid color)
                    selected.append((idx / fps, frame))
                    prev_des = None
                elif prev_des is None:
                    selected.append((idx / fps, frame))
                    prev_des = des
                else:
                    matches = bf.match(prev_des, des)
                    good = [m for m in matches if m.distance < dist_thresh]
                    if len(good) < match_thresh:
                        selected.append((idx / fps, frame))
                        prev_des = des
            idx += 1

        cap.release()
        return selected