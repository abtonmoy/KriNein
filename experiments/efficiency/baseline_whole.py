# experiments\efficiency\baseline_whole.py
"""
Ablation Study: Baseline Frame Sampling Methods with LLM Extraction
Compare against the full HMMD pipeline for ICMR 2026 submission
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
import os
import sys
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Find project root and add to path
current_file = Path(__file__).resolve()
project_root = current_file.parent
while project_root != project_root.parent:
    if (project_root / "src").exists():
        break
    project_root = project_root.parent

src_path = project_root / "src"
config_path = project_root / "config"

if src_path.exists() and str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"✓ Added to path: {src_path}")

# Load environment variables
from dotenv import load_dotenv
env_path = src_path / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)
    print(f"✓ Loaded environment from: {env_path}")
else:
    load_dotenv(override=True)
    print("✓ Loaded environment from current directory")

# Verify API key
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if api_key:
    print(f"✓ API key found: {api_key[:10]}...{api_key[-4:]}")
else:
    print("⚠ WARNING: No GOOGLE_API_KEY or GEMINI_API_KEY found in environment!")

# Optional imports
try:
    from scenedetect import detect, ContentDetector
    SCENEDETECT_AVAILABLE = True
except ImportError:
    SCENEDETECT_AVAILABLE = False
    print("Warning: scenedetect not available, using fallback scene detection")

try:
    import torch
    import open_clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    print("Warning: CLIP not available, semantic sampling disabled")


def load_default_config() -> Dict[str, Any]:
    """Load default configuration from config/default.yaml"""
    default_yaml = config_path / "default.yaml"
    if default_yaml.exists():
        with open(default_yaml, 'r') as f:
            config = yaml.safe_load(f)
            print(f"✓ Loaded config from: {default_yaml}")
            return config
    else:
        print(f"⚠ Config not found at {default_yaml}, using hardcoded defaults")
        return get_hardcoded_default_config()


def get_hardcoded_default_config() -> Dict[str, Any]:
    """Fallback hardcoded config matching default.yaml structure"""
    return {
        "extraction": {
            "provider": "gemini",
            "model": "gemini-3-flash-preview",
            "max_tokens": 4000,
            "temperature": 0.0,
            "audio_context": {
                "enabled": True,
                "include_transcription": True,
                "include_key_phrases": True,
                "include_mood": True,
                "max_transcription_segments": 10
            },
            "temporal_context": {
                "enabled": True,
                "include_timestamps": True,
                "include_time_deltas": True,
                "include_position_labels": True,
                "include_narrative_instructions": True
            },
            "schema": {
                "mode": "adaptive"
            }
        }
    }


@dataclass
class SampledFrame:
    """Container for a sampled frame with metadata"""
    frame: np.ndarray
    timestamp: float  # seconds
    frame_index: int
    scene_id: Optional[int] = None
    importance_score: float = 1.0
    method: str = ""


@dataclass 
class SamplingResult:
    """Result from a sampling method"""
    frames: List[SampledFrame]
    method_name: str
    video_path: str
    video_duration: float
    total_video_frames: int
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    extraction_result: Optional[Dict[str, Any]] = None
    extraction_time: float = 0.0
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def compression_ratio(self) -> float:
        if self.total_video_frames == 0:
            return 0.0
        return 1 - (self.num_frames / self.total_video_frames)
    
    @property
    def total_time(self) -> float:
        return self.processing_time + self.extraction_time
    
    def get_frames_with_timestamps(self) -> List[Tuple[float, np.ndarray]]:
        return [(f.timestamp, f.frame) for f in self.frames]


class BaseSampler(ABC):
    """Abstract base class for frame sampling methods"""
    
    def __init__(self, name: str):
        self.name = name
        self._extractor = None
        self._default_config = None
    
    @abstractmethod
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        pass
    
    def _load_video_metadata(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        metadata = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
        }
        return cap, metadata
    
    def _extract_frame_at_index(self, cap: cv2.VideoCapture, 
                                 frame_idx: int, 
                                 fps: float) -> Optional[SampledFrame]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            timestamp = frame_idx / fps
            return SampledFrame(
                frame=frame,
                timestamp=timestamp,
                frame_index=frame_idx,
                method=self.name
            )
        return None
    
    def _extract_frame_at_time(self, cap: cv2.VideoCapture,
                                timestamp: float,
                                fps: float) -> Optional[SampledFrame]:
        frame_idx = int(timestamp * fps)
        return self._extract_frame_at_index(cap, frame_idx, fps)
    
    def _get_default_config(self) -> Dict:
        """Get default config, loading from YAML if not cached"""
        if self._default_config is None:
            self._default_config = load_default_config()
        return self._default_config
    
    def _get_extractor(self, config: Optional[Dict] = None):
        """Lazy-load LLM extractor using create_extractor function"""
        if self._extractor is None:
            from src.extraction.llm_client import create_extractor
            
            # Use provided config or load default
            if config is None:
                config = self._get_default_config()
            
            # Ensure config has the right structure for create_extractor
            if "extraction" not in config:
                config = {"extraction": config}
            
            self._extractor = create_extractor(config)
        return self._extractor
    
    def extract_with_llm(
        self, 
        result: SamplingResult,
        config: Optional[Dict] = None,
        audio_context: Optional[Dict] = None
    ) -> SamplingResult:
        """Run LLM extraction on sampled frames using AdExtractor."""
        import time
        
        if not result.frames:
            print(f"Warning: No frames to extract from for {result.method_name}")
            return result
        
        try:
            extractor = self._get_extractor(config)
            
            print(f"  Running LLM extraction on {result.num_frames} frames...", end="", flush=True)
            start_time = time.time()
            
            frames_for_llm = result.get_frames_with_timestamps()
            
            extraction_result = extractor.extract(
                frames=frames_for_llm,
                video_duration=result.video_duration,
                audio_context=audio_context
            )
            
            extraction_time = time.time() - start_time
            
            result.extraction_result = extraction_result
            result.extraction_time = extraction_time
            
            print(f" Done! ({extraction_time:.2f}s)")
            
        except Exception as e:
            print(f" Failed! Error: {e}")
            import traceback
            traceback.print_exc()
            result.extraction_result = {"error": str(e)}
            result.extraction_time = 0.0
        
        return result


class UniformSampler(BaseSampler):
    """Uniform temporal sampling - extract frames at fixed intervals"""
    
    def __init__(self, sample_fps: float = 1.0):
        super().__init__(name=f"uniform_{sample_fps}fps")
        self.sample_fps = sample_fps
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        frame_interval = int(fps / self.sample_fps)
        frame_interval = max(1, frame_interval)
        
        frames = []
        frame_idx = 0
        
        while frame_idx < total_frames:
            sampled = self._extract_frame_at_index(cap, frame_idx, fps)
            if sampled:
                frames.append(sampled)
            frame_idx += frame_interval
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"sample_fps": self.sample_fps, "frame_interval": frame_interval}
        )


class UniformCountSampler(BaseSampler):
    """Uniform sampling with fixed frame count"""
    
    def __init__(self, num_frames: int = 8):
        super().__init__(name=f"uniform_{num_frames}frames")
        self.num_frames = num_frames
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in indices:
            sampled = self._extract_frame_at_index(cap, int(idx), fps)
            if sampled:
                frames.append(sampled)
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"target_frames": self.num_frames, "actual_frames": len(frames)}
        )


class RandomSampler(BaseSampler):
    """Random frame sampling"""
    
    def __init__(self, num_frames: int = 8, seed: Optional[int] = None):
        super().__init__(name=f"random_{num_frames}frames")
        self.num_frames = num_frames
        self.seed = seed
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        k = min(self.num_frames, total_frames)
        indices = sorted(random.sample(range(total_frames), k))
        
        frames = []
        for idx in indices:
            sampled = self._extract_frame_at_index(cap, idx, fps)
            if sampled:
                frames.append(sampled)
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"seed": self.seed, "requested_frames": self.num_frames}
        )


class FirstLastMidSampler(BaseSampler):
    """Simple heuristic: First frame, last frame, and middle frame(s)"""
    
    def __init__(self, include_quartiles: bool = False):
        name = "first_last_mid_quartiles" if include_quartiles else "first_last_mid"
        super().__init__(name=name)
        self.include_quartiles = include_quartiles
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        if self.include_quartiles:
            positions = [0, 0.25, 0.5, 0.75, 1.0]
        else:
            positions = [0, 0.5, 1.0]
        
        indices = [int(p * (total_frames - 1)) for p in positions]
        indices = list(dict.fromkeys(indices))
        
        frames = []
        for idx in indices:
            sampled = self._extract_frame_at_index(cap, idx, fps)
            if sampled:
                frames.append(sampled)
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"positions": positions}
        )


class SceneMidpointSampler(BaseSampler):
    """Scene-aware sampling: One frame from midpoint of each scene"""
    
    def __init__(self, 
                 detection_threshold: float = 27.0,
                 min_scene_length: float = 0.5,
                 fallback_chunk_size: float = 5.0):
        super().__init__(name="scene_midpoint")
        self.detection_threshold = detection_threshold
        self.min_scene_length = min_scene_length
        self.fallback_chunk_size = fallback_chunk_size
    
    def _detect_scenes(self, video_path: str, fps: float, total_frames: int) -> List[Tuple[float, float]]:
        if SCENEDETECT_AVAILABLE:
            try:
                scene_list = detect(
                    video_path, 
                    ContentDetector(threshold=self.detection_threshold,
                                   min_scene_len=int(self.min_scene_length * fps))
                )
                
                if scene_list:
                    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
            except Exception as e:
                print(f"  SceneDetect failed: {e}, using fallback")
        
        duration = total_frames / fps
        scenes = []
        current = 0.0
        while current < duration:
            end = min(current + self.fallback_chunk_size, duration)
            scenes.append((current, end))
            current = end
        
        return scenes
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        scenes = self._detect_scenes(video_path, fps, total_frames)
        
        frames = []
        for scene_id, (scene_start, scene_end) in enumerate(scenes):
            mid_time = (scene_start + scene_end) / 2
            sampled = self._extract_frame_at_time(cap, mid_time, fps)
            if sampled:
                sampled.scene_id = scene_id
                frames.append(sampled)
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"num_scenes": len(scenes), "scenes": scenes}
        )


class HistogramChangeSampler(BaseSampler):
    """Change detection sampling based on color histogram differences"""
    
    def __init__(self, 
                 threshold: float = 0.3,
                 min_interval_frames: int = 15,
                 max_frames: int = 20):
        super().__init__(name="histogram_change")
        self.threshold = threshold
        self.min_interval_frames = min_interval_frames
        self.max_frames = max_frames
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        return hist.flatten()
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        frames = []
        prev_hist = None
        last_sampled_idx = -self.min_interval_frames
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx == 0:
                sampled = SampledFrame(
                    frame=frame.copy(),
                    timestamp=0.0,
                    frame_index=0,
                    method=self.name
                )
                frames.append(sampled)
                prev_hist = self._compute_histogram(frame)
                last_sampled_idx = 0
            else:
                curr_hist = self._compute_histogram(frame)
                correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                change_score = 1 - correlation
                
                if (change_score > self.threshold and 
                    frame_idx - last_sampled_idx >= self.min_interval_frames):
                    
                    sampled = SampledFrame(
                        frame=frame.copy(),
                        timestamp=frame_idx / fps,
                        frame_index=frame_idx,
                        importance_score=change_score,
                        method=self.name
                    )
                    frames.append(sampled)
                    last_sampled_idx = frame_idx
                    
                    if len(frames) >= self.max_frames:
                        break
                
                prev_hist = curr_hist
            
            frame_idx += 1
        
        cap.release()
        
        if frames and frames[-1].frame_index < total_frames - 1:
            cap = cv2.VideoCapture(video_path)
            last_sampled = self._extract_frame_at_index(cap, total_frames - 1, fps)
            if last_sampled:
                frames.append(last_sampled)
            cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={"threshold": self.threshold, "min_interval": self.min_interval_frames}
        )


def run_ablation_study(
    video_paths: List[str], 
    output_dir: str = "ablation_results",
    run_llm_extraction: bool = True,
    extraction_config: Optional[Dict] = None
):
    """Run all baselines on videos with optional LLM extraction"""
    import json
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load default config if not provided
    if extraction_config is None:
        extraction_config = load_default_config()
    
    # Define baselines
    baselines = [
        UniformSampler(sample_fps=1.0),
        UniformCountSampler(num_frames=10),
        FirstLastMidSampler(include_quartiles=True),
        HistogramChangeSampler(threshold=0.3),
        SceneMidpointSampler(),
    ]
    
    results_summary = []
    
    print(f"Starting Ablation Study on {len(video_paths)} videos...")
    print(f"Comparing {len(baselines)} baseline methods")
    print(f"LLM Extraction: {'ENABLED' if run_llm_extraction else 'DISABLED'}")
    
    # Print extraction config summary
    ext_cfg = extraction_config.get("extraction", extraction_config)
    print(f"Provider: {ext_cfg.get('provider', 'N/A')}, Model: {ext_cfg.get('model', 'N/A')}")
    print(f"Schema Mode: {ext_cfg.get('schema', {}).get('mode', 'adaptive')}")
    
    total_start_time = time.time()
    
    for vid_idx, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        print(f"\n{'='*80}")
        print(f"Processing [{vid_idx+1}/{len(video_paths)}]: {video_name}")
        print(f"{'='*80}")
        
        for sampler in baselines:
            try:
                print(f"\n--- Method: {sampler.name} ---")
                print(f"  Sampling frames...", end="", flush=True)
                
                result = sampler.sample(video_path)
                print(f" Done! ({result.num_frames} frames, {result.processing_time:.2f}s)")
                
                if run_llm_extraction:
                    result = sampler.extract_with_llm(result, config=extraction_config)
                
                save_contact_sheet(result, output_dir)
                
                result_dict = {
                    "video": video_name,
                    "method": sampler.name,
                    "num_frames": result.num_frames,
                    "compression_ratio": result.compression_ratio,
                    "sampling_time_s": result.processing_time,
                    "extraction_time_s": result.extraction_time,
                    "total_time_s": result.total_time,
                    "fps_equivalent": result.num_frames / result.video_duration if result.video_duration > 0 else 0,
                }
                
                if result.extraction_result:
                    if "error" in result.extraction_result:
                        result_dict["extraction_error"] = result.extraction_result["error"]
                    else:
                        result_dict["extraction_success"] = True
                        result_dict["extracted_fields"] = list(result.extraction_result.keys())
                        
                        # Store full extraction result for schema validation
                        result_dict["extraction_data"] = result.extraction_result
                
                results_summary.append(result_dict)
                
            except Exception as e:
                print(f" Failed! Error: {e}")
                import traceback
                traceback.print_exc()
                results_summary.append({
                    "video": video_name,
                    "method": sampler.name,
                    "error": str(e)
                })
    
    total_time = time.time() - total_start_time
    
    # Save results
    json_path = os.path.join(output_dir, "ablation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*80}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*80}")
    print(f"Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"Results saved to: {json_path}")
    
    # Compute statistics
    methods = {}
    for result in results_summary:
        if "error" in result:
            continue
        method = result["method"]
        if method not in methods:
            methods[method] = {
                "frame_counts": [],
                "sampling_times": [],
                "extraction_times": [],
                "total_times": []
            }
        methods[method]["frame_counts"].append(result["num_frames"])
        methods[method]["sampling_times"].append(result["sampling_time_s"])
        if "extraction_time_s" in result:
            methods[method]["extraction_times"].append(result["extraction_time_s"])
            methods[method]["total_times"].append(result["total_time_s"])
    
    print("\n" + "="*100)
    print("SUMMARY STATISTICS")
    print("="*100)
    print(f"{'Method':<25} {'Frames':<10} {'Sample(s)':<12} {'Extract(s)':<12} {'Total(s)':<12}")
    print("-" * 100)
    for method, stats in sorted(methods.items()):
        avg_frames = np.mean(stats["frame_counts"])
        avg_sampling = np.mean(stats["sampling_times"])
        avg_extract = np.mean(stats["extraction_times"]) if stats["extraction_times"] else 0
        avg_total = np.mean(stats["total_times"]) if stats["total_times"] else avg_sampling
        print(f"{method:<25} {avg_frames:<10.1f} {avg_sampling:<12.2f} {avg_extract:<12.2f} {avg_total:<12.2f}")
    
    return results_summary


def save_contact_sheet(result: SamplingResult, output_dir: str):
    """Helper to visualize sampled frames"""
    if not result.frames:
        return
    
    frames = [f.frame for f in result.frames]
    count = len(frames)
    cols = 5
    rows = (count + cols - 1) // cols
    
    thumb_w, thumb_h = 320, 180
    thumbs = [cv2.resize(f, (thumb_w, thumb_h)) for f in frames]
    
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb
        
        cv2.putText(sheet, f"{result.frames[i].timestamp:.1f}s", 
                   (c*thumb_w + 10, r*thumb_h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    filename = f"{Path(result.video_path).stem}_{result.method_name}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), sheet)


def run_batch_ablation(
    video_directory: str,
    output_dir: str = "ablation_results_batch",
    run_llm_extraction: bool = True,
    extraction_config: Optional[Dict] = None,
    video_extensions: List[str] = None,
    max_videos: Optional[int] = None
) -> List[Dict]:
    """Run ablation study on all videos in a directory"""
    if video_extensions is None:
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.m4v']
    
    video_dir = Path(video_directory)
    if not video_dir.exists():
        raise ValueError(f"Directory does not exist: {video_directory}")
    
    video_paths = []
    for ext in video_extensions:
        video_paths.extend(video_dir.glob(f"**/*{ext}"))
        video_paths.extend(video_dir.glob(f"**/*{ext.upper()}"))
    
    video_paths = sorted(set(str(p) for p in video_paths))
    
    if not video_paths:
        print(f"No videos found in {video_directory}")
        return []
    
    print(f"Found {len(video_paths)} videos")
    
    if max_videos is not None and len(video_paths) > max_videos:
        print(f"Limiting to first {max_videos} videos")
        video_paths = video_paths[:max_videos]
    
    return run_ablation_study(
        video_paths=video_paths,
        output_dir=output_dir,
        run_llm_extraction=run_llm_extraction,
        extraction_config=extraction_config
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ablation study on video frame sampling methods")
    parser.add_argument("--videos", nargs="+", help="List of video paths")
    parser.add_argument("--video-dir", type=str, help="Directory containing videos")
    parser.add_argument("--output-dir", type=str, default="ablation_results", help="Output directory")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM extraction")
    parser.add_argument("--max-videos", type=int, help="Maximum number of videos")
    parser.add_argument("--config", type=str, help="Path to custom config YAML file")
    parser.add_argument("--provider", type=str, choices=["gemini", "anthropic", "openai"], help="Override provider")
    parser.add_argument("--model", type=str, help="Override model name")
    
    args = parser.parse_args()
    
    # Load config: custom file > default.yaml > hardcoded
    if args.config:
        with open(args.config, 'r') as f:
            extraction_config = yaml.safe_load(f)
        print(f"✓ Loaded custom config from: {args.config}")
    else:
        extraction_config = load_default_config()
    
    # Apply CLI overrides
    if args.provider or args.model:
        if "extraction" not in extraction_config:
            extraction_config["extraction"] = {}
        if args.provider:
            extraction_config["extraction"]["provider"] = args.provider
        if args.model:
            extraction_config["extraction"]["model"] = args.model
    
    # Run based on input mode
    if args.video_dir:
        print(f"Running batch ablation on directory: {args.video_dir}")
        results = run_batch_ablation(
            video_directory=args.video_dir,
            output_dir=args.output_dir,
            run_llm_extraction=not args.no_llm,
            extraction_config=extraction_config,
            max_videos=args.max_videos
        )
    elif args.videos:
        print(f"Running ablation on {len(args.videos)} specific videos")
        results = run_ablation_study(
            video_paths=args.videos,
            output_dir=args.output_dir,
            run_llm_extraction=not args.no_llm,
            extraction_config=extraction_config
        )
    else:
        # Default: run on test videos if they exist
        print("No videos specified, running on default test videos...")
        test_videos = [
            "data/hussain_videos/-1yXdOufzKE.mp4"
            # "data/hussain_videos/-HWd3Nqu8_0.mp4"
        ]
        
        valid_videos = [v for v in test_videos if os.path.exists(v)]
        
        if valid_videos:
            print(f"Found {len(valid_videos)} test videos")
            results = run_ablation_study(
                valid_videos,
                run_llm_extraction=not args.no_llm,
                output_dir=args.output_dir,
                extraction_config=extraction_config
            )
        else:
            print("No test videos found. Please specify --videos or --video-dir")
            print("\nUsage examples:")
            print("  python baseline_whole.py --videos video1.mp4 video2.mp4")
            print("  python baseline_whole.py --video-dir data/videos --max-videos 10")
            print("  python baseline_whole.py --video-dir data/videos --no-llm")
            print("  python baseline_whole.py --config config/custom.yaml")