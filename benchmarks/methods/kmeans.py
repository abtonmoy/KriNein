"""
Test 7: K-Means Clustering on CLIP embeddings.

Clusters all frame embeddings, then picks the frame nearest each
cluster centroid as the representative.

k is adaptive: ~1 cluster per 3 seconds of video, clamped [5, 20].
Expects pre-computed embeddings and all_frames from the runner.
"""

import logging
from typing import Any, List, Tuple
import numpy as np
from sklearn.cluster import KMeans

from benchmarks.base import BaselineMethod, get_video_info

logger = logging.getLogger(__name__)


class KMeansClustering(BaselineMethod):

    def __init__(self, seconds_per_cluster: float = 3.0):
        self.seconds_per_cluster = seconds_per_cluster

    @property
    def name(self) -> str:
        return "kmeans"

    @property
    def requires_gpu(self) -> bool:
        return True

    def select_frames(
        self, video_path: str, **kwargs: Any
    ) -> List[Tuple[float, np.ndarray]]:
        all_frames: List[Tuple[float, np.ndarray]] = kwargs.get("all_frames")
        embeddings: np.ndarray = kwargs.get("clip_embeddings")

        if all_frames is None or embeddings is None:
            raise ValueError(
                "KMeansClustering requires 'all_frames' and 'clip_embeddings' in kwargs."
            )

        spc = kwargs.get("kmeans_seconds_per_cluster", self.seconds_per_cluster)
        _, _, duration = get_video_info(video_path)

        # Adaptive k
        k = max(5, min(20, int(duration / spc)))
        k = min(k, len(all_frames))  # can't have more clusters than frames

        logger.info(f"K-Means: k={k} clusters for {duration:.1f}s video ({len(all_frames)} frames)")

        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)

        # Select frame nearest each centroid
        selected_indices = []
        for c in range(k):
            cluster_mask = np.where(labels == c)[0]
            if len(cluster_mask) == 0:
                continue
            dists = np.linalg.norm(
                embeddings[cluster_mask] - km.cluster_centers_[c], axis=1
            )
            best = cluster_mask[np.argmin(dists)]
            selected_indices.append(int(best))

        # Sort by timestamp for chronological order
        selected_indices.sort(key=lambda i: all_frames[i][0])

        return [all_frames[i] for i in selected_indices]