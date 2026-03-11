from typing import Any, List, Tuple
import numpy as np
from pathlib import Path
import sys

from benchmarks.base import BaselineMethod

class HIBPipelineBaseline(BaselineMethod):
    """
    Runs the full AdVideoPipeline with the novel Hamiltonian Information Budget (HIB) enabled.
    This acts as a baseline wrapper to compare HIB directly against other heuristics like Uniform or K-Means.
    """
    @property
    def name(self) -> str:
        return "hib_pipeline"

    @property
    def requires_gpu(self) -> bool:
        return True # Uses CLIP underneath

    def select_frames(self, video_path: str, **kwargs: Any) -> List[Tuple[float, np.ndarray]]:
        # Import pipeline locally to avoid circular dependencies
        from src.pipeline import AdVideoPipeline
        
        # Override config to ensure HIB is enabled
        overrides = {
            "selection": {
                "use_hib_budget": True,
                "global_max_frames": 25
            }
        }
        
        pipeline = AdVideoPipeline(overrides=overrides)
        # Skip extraction because BenchmarkRunner handles LLM calls separately
        result = pipeline.process(video_path, skip_extraction=True)
        
        # result.selected_frames is typically a list of FrameInfo objects.
        # We need to return List[Tuple[float, np.ndarray]]
        out_frames = []
        if getattr(result, "selected_frames", None):
            for cand in result.selected_frames:
                out_frames.append((cand.timestamp, cand.frame))
                
        return out_frames


class StaticPipelineBaseline(BaselineMethod):
    """
    Runs the full AdVideoPipeline but strictly uses the old linear static budget (duration * density).
    """
    @property
    def name(self) -> str:
        return "static_pipeline"

    @property
    def requires_gpu(self) -> bool:
        return True

    def select_frames(self, video_path: str, **kwargs: Any) -> List[Tuple[float, np.ndarray]]:
        from src.pipeline import AdVideoPipeline
        
        # Override config to disable HIB and use static budget
        overrides = {
            "selection": {
                "use_hib_budget": False,
                "global_max_frames": 25
            }
        }
        
        pipeline = AdVideoPipeline(overrides=overrides)
        result = pipeline.process(video_path, skip_extraction=True)
        
        out_frames = []
        if getattr(result, "selected_frames", None):
            for cand in result.selected_frames:
                out_frames.append((cand.timestamp, cand.frame))
                
        return out_frames
