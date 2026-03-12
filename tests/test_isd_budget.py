import numpy as np
import pytest
from unittest.mock import MagicMock
from src.selection.representative import FrameSelector
from src.selection.clustering import FrameCandidate
import random

def test_isd_dynamic_budget_redundant():
    """
    Test that a video with highly redundant semantic frames (e.g. constant talking head)
    produces a very low Intrinsic Semantic Dimensionality (ISD) and therefore caps the budget low.
    """
    selector = FrameSelector(use_hib_budget=True, global_max_frames=100)
    video_duration = 30.0
    scene_boundaries = [(0.0, 15.0), (15.0, 30.0)]
    density = 1.0

    # 30 frames of the exact same semantic meaning (plus tiny noise)
    base_emb = np.ones(512)
    candidates = []
    
    np.random.seed(42)  # For deterministic SVD
    for i in range(30):
        # Very tiny noise in one dimension to simulate almost identical frames
        emb = base_emb.copy()
        emb[0] += np.random.normal(0, 0.001)
        cand = FrameCandidate(
            timestamp=float(i),
            frame=np.zeros((10,10,3)),
            embedding=emb, 
            importance_score=1.0, 
            scene_id=i//15
        )
        candidates.append(cand)

    budget = selector._compute_frame_budget(video_duration, candidates, scene_boundaries, density)
    
    # Assert the budget is extremely constrained (bounded heavily by the low ISD)
    # With base_budget ~ 5, and ISD likely 1 or 2, max cap should be around 5 to 8
    assert budget <= 8, f"Expected budget to hit low ISD floor (<=8), got {budget}"


def test_isd_dynamic_budget_chaotic():
    """
    Test that a highly chaotic, fast-cut video (e.g. action movie trailer)
    produces a very high ISD and naturally expands the budget constraint appropriately,
    ignoring the global_max_frames logic.
    """
    selector = FrameSelector(use_hib_budget=True, global_max_frames=100)
    video_duration = 30.0
    scene_boundaries = [(0.0, 5.0), (5.0, 10.0), (10.0, 15.0), (15.0, 20.0), (20.0, 30.0)]
    density = 1.0

    np.random.seed(42)
    candidates = []
    
    # 30 frames of completely orthogonal/random semantic meaning
    for i in range(30):
        emb = np.random.randn(512) # Uniformly random semantic vectors
        cand = FrameCandidate(
            timestamp=float(i),
            frame=np.zeros((10,10,3)),
            embedding=emb, 
            importance_score=2.0, # High importance (mimics text/face hits)
            scene_id=i//6
        )
        candidates.append(cand)

    budget = selector._compute_frame_budget(video_duration, candidates, scene_boundaries, density)
    
    # The high variance and importance scores will drive raw adaptive budget high
    # The pure ISD ceiling should naturally scale to > 25 for 30 distinct 512d concepts.
    # 5 base + (ISD ~ 30 * 1.5) = Max Cap ~ 50.
    assert budget >= 25, f"Expected uncapped budget to expand significantly for high variance, got {budget}"

def test_legacy_budget_strict_ceiling():
    """
    Ensure the legacy pipeline still perfectly respects the hard-coded global_max_frames.
    """
    selector = FrameSelector(use_hib_budget=False, global_max_frames=15)
    video_duration = 100.0 # Huge video
    density = 1.0 # Wants 100 frames
    
    # Mock some basic candidates
    candidates = [FrameCandidate(timestamp=float(i), frame=np.zeros((10,10,3)), embedding=np.zeros(512), importance_score=1.0, scene_id=0) for i in range(50)]
    budget = selector._compute_frame_budget(video_duration, candidates, [(0.0, 100.0)], density)
    
    assert budget == 15, "Legacy budget must strictly stop at global_max_frames"

if __name__ == "__main__":
    pytest.main([__file__])
