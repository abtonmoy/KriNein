# Complete Pipeline Extract: Stages 1-4 with Timing
# Stage 1: Video Ingestion → Stage 2: Scene Detection → Stage 3: Candidate Extraction → Stage 4: Deduplication

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import logging
import numpy as np
import yaml
import time
import sys
from dataclasses import dataclass
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# Import existing components
from src.ingestion.video_loader import VideoLoader
from src.detection.change_detector import get_change_detector
from src.detection.scene_detector import SceneDetector, CandidateFrameExtractor
from src.deduplication.hierarchical import create_deduplicator
from src.utils.config import get_device

logger = logging.getLogger(__name__)


@dataclass
class StageTimings:
    """Container for stage timing information."""
    stage_1_ingestion: float = 0.0
    stage_2_scene_detection: float = 0.0
    stage_3_candidate_extraction: float = 0.0
    stage_4_hash_voting: float = 0.0
    stage_4_ssim: float = 0.0
    stage_4_lpips: float = 0.0
    stage_4_clip: float = 0.0
    total_stage_4: float = 0.0
    total_pipeline: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "stage_1_ingestion_s": self.stage_1_ingestion,
            "stage_2_scene_detection_s": self.stage_2_scene_detection,
            "stage_3_candidate_extraction_s": self.stage_3_candidate_extraction,
            "stage_4_deduplication": {
                "hash_voting_s": self.stage_4_hash_voting,
                "ssim_s": self.stage_4_ssim,
                "lpips_s": self.stage_4_lpips,
                "clip_s": self.stage_4_clip,
                "total_s": self.total_stage_4
            },
            "total_pipeline_s": self.total_pipeline
        }
    
    def print_summary(self):
        """Print formatted timing summary."""
        print("\n" + "="*70)
        print("PIPELINE TIMING BREAKDOWN")
        print("="*70)
        print(f"Stage 1: Video Ingestion          {self.stage_1_ingestion:8.2f}s  ({self._percent(self.stage_1_ingestion):5.1f}%)")
        print(f"Stage 2: Scene Detection          {self.stage_2_scene_detection:8.2f}s  ({self._percent(self.stage_2_scene_detection):5.1f}%)")
        print(f"Stage 3: Candidate Extraction     {self.stage_3_candidate_extraction:8.2f}s  ({self._percent(self.stage_3_candidate_extraction):5.1f}%)")
        print(f"Stage 4: Deduplication (total)    {self.total_stage_4:8.2f}s  ({self._percent(self.total_stage_4):5.1f}%)")
        if self.stage_4_hash_voting > 0:
            print(f"  ├─ Hash Voting                  {self.stage_4_hash_voting:8.2f}s")
        if self.stage_4_ssim > 0:
            print(f"  ├─ SSIM                         {self.stage_4_ssim:8.2f}s")
        if self.stage_4_lpips > 0:
            print(f"  ├─ LPIPS                        {self.stage_4_lpips:8.2f}s")
        if self.stage_4_clip > 0:
            print(f"  └─ CLIP                         {self.stage_4_clip:8.2f}s")
        print("-"*70)
        print(f"TOTAL PIPELINE TIME               {self.total_pipeline:8.2f}s  (100.0%)")
        print("="*70 + "\n")
    
    def _percent(self, stage_time: float) -> float:
        """Calculate percentage of total time."""
        if self.total_pipeline == 0:
            return 0.0
        return (stage_time / self.total_pipeline) * 100


# Load configuration from YAML
def load_config(config_path: str = "config/default.yaml") -> Dict:
    """
    Load pipeline configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"Loaded configuration from: {config_path}")
    return config


# Stage 1: Video Ingestion & Metadata Extraction
def load_video_and_metadata(
    video_path: str,
    ingestion_config: Dict
) -> Tuple[Any, Optional[str], float]:
    """
    Load video and extract metadata + audio.
    
    Args:
        video_path: Path to video file
        ingestion_config: Ingestion configuration dict
        
    Returns:
        Tuple of (metadata, audio_path, elapsed_time)
    """
    start_time = time.time()
    logger.info("Stage 1: Loading video and extracting metadata...")
    
    # Initialize video loader
    loader = VideoLoader(
        max_resolution=ingestion_config.get("max_resolution", 720),
        extract_audio=ingestion_config.get("extract_audio", True)
    )
    
    # Load video and extract audio
    metadata, audio_path = loader.load(video_path)
    
    elapsed = time.time() - start_time
    
    logger.info(f"Video loaded: {metadata.width}x{metadata.height}, "
               f"{metadata.fps:.2f} fps, {metadata.duration:.2f}s, "
               f"{metadata.frame_count} frames (took {elapsed:.2f}s)")
    
    if audio_path:
        logger.info(f"Audio extracted to: {audio_path}")
    
    return metadata, audio_path, elapsed


# Stage 2: Scene Detection with Fallback
def detect_scenes_with_fallback(
    video_path: str,
    metadata: Any,
    scene_config: Dict
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Detect scene boundaries with automatic fallback strategy.
    
    Args:
        video_path: Path to video file
        metadata: Video metadata object
        scene_config: Scene detection configuration
        
    Returns:
        Tuple of (scene_boundaries, elapsed_time)
    """
    start_time = time.time()
    logger.info("Stage 2: Detecting scene boundaries...")
    
    # Initialize primary scene detector
    scene_detector = SceneDetector(
        method=scene_config.get("method", "content"),
        threshold=scene_config.get("threshold", 27.0),
        min_scene_length_s=scene_config.get("min_scene_length_s", 0.5)
    )
    
    # Try primary detection
    scene_boundaries = scene_detector.detect_scenes(video_path)
    
    if scene_boundaries:
        elapsed = time.time() - start_time
        logger.info(f"Detected {len(scene_boundaries)} scenes with threshold={scene_config.get('threshold', 27.0)} (took {elapsed:.2f}s)")
        return scene_boundaries, elapsed
    
    # Fallback configuration
    fallback_config = scene_config.get("fallback", {})
    
    if not fallback_config.get("enabled", True):
        elapsed = time.time() - start_time
        logger.warning("Scene detection failed, fallback disabled. Using entire video as one scene.")
        return [(0.0, metadata.duration)], elapsed
    
    # Fallback 1: Lower threshold
    logger.warning("No scenes detected, retrying with lower threshold...")
    fallback_threshold = fallback_config.get("threshold", 15.0)
    
    fallback_detector = SceneDetector(
        method="content",
        threshold=fallback_threshold,
        min_scene_length_s=scene_config.get("min_scene_length_s", 0.5)
    )
    
    scene_boundaries = fallback_detector.detect_scenes(video_path)
    
    if scene_boundaries:
        elapsed = time.time() - start_time
        logger.info(f"Fallback detected {len(scene_boundaries)} scenes with threshold={fallback_threshold} (took {elapsed:.2f}s)")
        return scene_boundaries, elapsed
    
    # Fallback 2: Artificial chunks
    if fallback_config.get("artificial_chunks", True):
        logger.warning("Creating artificial scene chunks...")
        chunk_size = fallback_config.get("chunk_size_s", 10.0)
        
        scene_boundaries = []
        current_time = 0.0
        
        while current_time < metadata.duration:
            end_time = min(current_time + chunk_size, metadata.duration)
            scene_boundaries.append((current_time, end_time))
            current_time = end_time
        
        elapsed = time.time() - start_time
        logger.info(f"Created {len(scene_boundaries)} artificial chunks of {chunk_size}s each (took {elapsed:.2f}s)")
        return scene_boundaries, elapsed
    
    # Last resort: single scene
    elapsed = time.time() - start_time
    logger.warning("All fallbacks failed, treating entire video as one scene")
    return [(0.0, metadata.duration)], elapsed


# Stage 3: Extract Candidate Frames (Change Detection)
def extract_candidate_frames(
    video_path: str,
    change_config: Dict,
    max_resolution: int = 720
) -> Tuple[List, float]:
    """
    Extract candidate frames based on visual change detection.
    
    Args:
        video_path: Path to video file
        change_config: Change detection configuration
        max_resolution: Maximum frame resolution
        
    Returns:
        Tuple of (candidate_frames, elapsed_time)
    """
    start_time = time.time()
    logger.info("Stage 3: Extracting candidate frames...")
    
    # Initialize change detector
    method = change_config.get("method", "histogram")
    threshold = change_config.get("threshold", 0.15)
    min_interval_ms = change_config.get("min_interval_ms", 100)
    
    change_detector = get_change_detector(method)
    
    # Create candidate extractor
    candidate_extractor = CandidateFrameExtractor(
        change_detector=change_detector,
        threshold=threshold,
        min_interval_ms=min_interval_ms
    )
    
    # Extract candidates at change points
    candidates = candidate_extractor.extract_candidates(
        video_path,
        max_resolution=max_resolution
    )
    
    elapsed = time.time() - start_time
    
    logger.info(f"Extracted {len(candidates)} candidate frames "
               f"(method={method}, threshold={threshold}, min_interval={min_interval_ms}ms) "
               f"(took {elapsed:.2f}s)")
    
    return candidates, elapsed


# Stage 4: Hierarchical Deduplication using existing hierarchical.py
def deduplicate_frames_with_timing(
    candidates: List,
    config: Dict
) -> Tuple[List, np.ndarray, Dict, Dict[str, float]]:
    """
    Hierarchical deduplication pipeline with timing using existing hierarchical.py.
    
    Args:
        candidates: List of candidate frames
        config: Full pipeline configuration dict
        
    Returns:
        Tuple of (deduplicated_frames, clip_embeddings, dedup_stats, stage_timings)
    """
    logger.info("Stage 4: Hierarchical deduplication...")
    stage_4_start = time.time()
    
    # Create deduplicator using existing factory
    deduplicator = create_deduplicator(config)
    
    # Run deduplication (this handles all sub-stages internally)
    deduped_frames, embeddings, dedup_stats = deduplicator.deduplicate(candidates)
    
    # Total stage 4 time
    total_time = time.time() - stage_4_start
    
    # Extract sub-stage timings if available from dedup_stats
    timings = {
        "hash_voting": dedup_stats.get("hash_voting_time", 0.0),
        "ssim": dedup_stats.get("ssim_time", 0.0),
        "lpips": dedup_stats.get("lpips_time", 0.0),
        "clip": dedup_stats.get("clip_time", 0.0),
        "total": total_time
    }
    
    # Summary
    initial = dedup_stats.get("initial_candidates", len(candidates))
    total_removed = initial - len(deduped_frames)
    reduction_rate = total_removed / initial if initial > 0 else 0
    
    logger.info(f"Deduplication complete: {initial} → {len(deduped_frames)} frames "
               f"({reduction_rate:.1%} reduction) (total time: {total_time:.2f}s)")
    logger.info(f"Pipeline breakdown: Hash={dedup_stats.get('after_hash_voting', 0)}, "
               f"SSIM={dedup_stats.get('after_ssim', 'N/A')}, "
               f"LPIPS={dedup_stats.get('after_lpips', 'N/A')}, "
               f"CLIP={dedup_stats.get('after_clip', len(deduped_frames))}")
    
    return deduped_frames, embeddings, dedup_stats, timings


# Complete Stages 1-4 Pipeline with Timing
def process_video_stages_1_to_4(
    video_path: str,
    config_path: str = "config/default.yaml"
) -> Tuple[Any, Optional[str], List[Tuple[float, float]], List, np.ndarray, Dict, StageTimings]:
    """
    Execute complete pipeline stages 1-4 with detailed timing.
    
    Args:
        video_path: Path to video file
        config_path: Path to YAML configuration file
        
    Returns:
        Tuple of (metadata, audio_path, scene_boundaries, deduped_frames, embeddings, stats, timings)
    """
    pipeline_start = time.time()
    logger.info(f"Starting pipeline stages 1-4 for: {video_path}")
    
    # Load configuration
    config = load_config(config_path)
    
    # Initialize timings
    timings = StageTimings()
    
    # Stage 1: Video ingestion
    ingestion_config = config.get("ingestion", {})
    metadata, audio_path, t1 = load_video_and_metadata(video_path, ingestion_config)
    timings.stage_1_ingestion = t1
    
    # Stage 2: Scene detection
    scene_config = config.get("scene_detection", {})
    scene_boundaries, t2 = detect_scenes_with_fallback(video_path, metadata, scene_config)
    timings.stage_2_scene_detection = t2
    
    # Stage 3: Candidate extraction
    change_config = config.get("change_detection", {})
    max_resolution = ingestion_config.get("max_resolution", 720)
    candidates, t3 = extract_candidate_frames(video_path, change_config, max_resolution)
    timings.stage_3_candidate_extraction = t3
    
    # Stage 4: Deduplication using existing hierarchical.py
    deduped_frames, embeddings, dedup_stats, stage_4_timings = deduplicate_frames_with_timing(
        candidates, 
        config
    )
    timings.stage_4_hash_voting = stage_4_timings.get("hash_voting", 0.0)
    timings.stage_4_ssim = stage_4_timings.get("ssim", 0.0)
    timings.stage_4_lpips = stage_4_timings.get("lpips", 0.0)
    timings.stage_4_clip = stage_4_timings.get("clip", 0.0)
    timings.total_stage_4 = stage_4_timings.get("total", 0.0)
    
    # Total pipeline time
    timings.total_pipeline = time.time() - pipeline_start
    
    # Print timing summary
    timings.print_summary()
    
    # Summary statistics
    logger.info("="*60)
    logger.info(f"Stages 1-4 Complete:")
    logger.info(f"  Video: {metadata.duration:.2f}s, {len(scene_boundaries)} scenes")
    logger.info(f"  Candidates extracted: {len(candidates)}")
    logger.info(f"  After deduplication: {len(deduped_frames)}")
    logger.info(f"  Reduction rate: {(1 - len(deduped_frames)/len(candidates)):.1%}")
    logger.info(f"  Audio extracted: {'Yes' if audio_path else 'No'}")
    logger.info(f"  Total processing time: {timings.total_pipeline:.2f}s")
    logger.info("="*60)
    
    return metadata, audio_path, scene_boundaries, deduped_frames, embeddings, dedup_stats, timings


# Example usage
if __name__ == "__main__":
    import sys
    from src.utils.logging import setup_logging
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Process video
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "data/hussain_videos/-1yXdOufzKE.mp4"
    
    try:
        results = process_video_stages_1_to_4(video_path, config_path="config/default.yaml")
        metadata, audio_path, scene_boundaries, deduped_frames, embeddings, stats, timings = results
        
        print("\nPipeline Results:")
        print(f"- Scenes detected: {len(scene_boundaries)}")
        print(f"- Frames after deduplication: {len(deduped_frames)}")
        print(f"- Deduplication stats: {stats}")
        
        # Save timing report
        import json
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timing_report = timings.to_dict()
        with open(output_dir / "timing_report.json", "w") as f:
            json.dump(timing_report, f, indent=2)
        print(f"\nTiming report saved to: {output_dir / 'timing_report.json'}")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        sys.exit(1)