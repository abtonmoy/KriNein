"""
Ablation Study: Baseline Frame Sampling Methods
Compare against the full HMMD pipeline for ICMR 2026 submission
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import random
from collections import defaultdict

# Optional imports - graceful fallback
try:
    from scenedetect import detect, ContentDetector, ThresholdDetector
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
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def compression_ratio(self) -> float:
        if self.total_video_frames == 0:
            return 0.0
        return 1 - (self.num_frames / self.total_video_frames)
    
    def get_frame_arrays(self) -> List[np.ndarray]:
        """Extract just the frame arrays for LLM processing"""
        return [f.frame for f in self.frames]
    
    def get_timestamps(self) -> List[float]:
        """Extract timestamps"""
        return [f.timestamp for f in self.frames]


class BaseSampler(ABC):
    """Abstract base class for frame sampling methods"""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        """Sample frames from video"""
        pass
    
    def _load_video_metadata(self, video_path: str) -> Tuple[cv2.VideoCapture, Dict]:
        """Load video and extract metadata"""
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
        """Extract a single frame at given index"""
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
        """Extract frame at specific timestamp"""
        frame_idx = int(timestamp * fps)
        return self._extract_frame_at_index(cap, frame_idx, fps)


class UniformSampler(BaseSampler):
    """
    Uniform temporal sampling - extract frames at fixed intervals
    
    This is the most common naive baseline. Extracts 1 frame every N seconds
    or at a specified frames-per-second rate.
    """
    
    def __init__(self, sample_fps: float = 1.0):
        """
        Args:
            sample_fps: Frames per second to sample (e.g., 1.0 = 1 frame/second)
        """
        super().__init__(name=f"uniform_{sample_fps}fps")
        self.sample_fps = sample_fps
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Calculate frame interval
        frame_interval = int(fps / self.sample_fps)
        frame_interval = max(1, frame_interval)  # At least 1
        
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
    """
    Uniform sampling with fixed frame count
    
    Always extracts exactly K frames, uniformly distributed across video.
    Useful for fair comparison when you want same number of frames across methods.
    """
    
    def __init__(self, num_frames: int = 8):
        """
        Args:
            num_frames: Exact number of frames to extract
        """
        super().__init__(name=f"uniform_{num_frames}frames")
        self.num_frames = num_frames
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Calculate uniform indices
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
    """
    Random frame sampling
    
    Randomly selects K frames from the video. 
    Useful as a stochastic baseline to show that intelligent selection matters.
    """
    
    def __init__(self, num_frames: int = 8, seed: Optional[int] = None):
        """
        Args:
            num_frames: Number of frames to randomly sample
            seed: Random seed for reproducibility
        """
        super().__init__(name=f"random_{num_frames}frames")
        self.num_frames = num_frames
        self.seed = seed
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        # Set seed for reproducibility
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Random selection
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
    """
    Simple heuristic: First frame, last frame, and middle frame(s)
    
    A minimal baseline that captures video boundaries.
    """
    
    def __init__(self, include_quartiles: bool = False):
        """
        Args:
            include_quartiles: If True, also include 25% and 75% points
        """
        name = "first_last_mid_quartiles" if include_quartiles else "first_last_mid"
        super().__init__(name=name)
        self.include_quartiles = include_quartiles
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Define positions to sample
        if self.include_quartiles:
            positions = [0, 0.25, 0.5, 0.75, 1.0]
        else:
            positions = [0, 0.5, 1.0]
        
        indices = [int(p * (total_frames - 1)) for p in positions]
        indices = list(dict.fromkeys(indices))  # Remove duplicates, preserve order
        
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
    """
    Scene-aware sampling: One frame from the midpoint of each scene
    
    Uses PySceneDetect for scene boundary detection, then extracts
    the middle frame from each detected scene.
    """
    
    def __init__(self, 
                 detection_threshold: float = 27.0,
                 min_scene_length: float = 0.5,
                 fallback_chunk_size: float = 5.0):
        """
        Args:
            detection_threshold: ContentDetector threshold (lower = more scenes)
            min_scene_length: Minimum scene duration in seconds
            fallback_chunk_size: If no scenes detected, chunk video into this size
        """
        super().__init__(name="scene_midpoint")
        self.detection_threshold = detection_threshold
        self.min_scene_length = min_scene_length
        self.fallback_chunk_size = fallback_chunk_size
    
    def _detect_scenes(self, video_path: str, fps: float, total_frames: int) -> List[Tuple[float, float]]:
        """Detect scene boundaries, return list of (start_time, end_time) tuples"""
        
        if SCENEDETECT_AVAILABLE:
            try:
                scene_list = detect(
                    video_path, 
                    ContentDetector(threshold=self.detection_threshold,
                                   min_scene_len=int(self.min_scene_length * fps))
                )
                
                if scene_list:
                    scenes = []
                    for scene in scene_list:
                        start_time = scene[0].get_seconds()
                        end_time = scene[1].get_seconds()
                        scenes.append((start_time, end_time))
                    return scenes
            except Exception as e:
                print(f"SceneDetect failed: {e}, using fallback")
        
        # Fallback: chunk video into fixed-size segments
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
        
        # Detect scenes
        scenes = self._detect_scenes(video_path, fps, total_frames)
        
        frames = []
        for scene_id, (scene_start, scene_end) in enumerate(scenes):
            # Get midpoint timestamp
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
            metadata={
                "num_scenes": len(scenes),
                "scenes": scenes,
                "detection_threshold": self.detection_threshold
            }
        )


class SceneMultiSampler(BaseSampler):
    """
    Scene-aware sampling: Multiple frames per scene (start, middle, end)
    
    Extracts frames at the beginning, middle, and end of each scene
    to capture scene transitions and key moments.
    """
    
    def __init__(self,
                 frames_per_scene: int = 3,
                 detection_threshold: float = 27.0,
                 min_scene_length: float = 0.5):
        """
        Args:
            frames_per_scene: Number of frames to extract per scene (1-5)
            detection_threshold: ContentDetector threshold
            min_scene_length: Minimum scene duration in seconds
        """
        super().__init__(name=f"scene_multi_{frames_per_scene}per")
        self.frames_per_scene = min(max(1, frames_per_scene), 5)
        self.detection_threshold = detection_threshold
        self.min_scene_length = min_scene_length
    
    def _get_scene_positions(self) -> List[float]:
        """Get relative positions within scene based on frames_per_scene"""
        if self.frames_per_scene == 1:
            return [0.5]  # Just middle
        elif self.frames_per_scene == 2:
            return [0.25, 0.75]
        elif self.frames_per_scene == 3:
            return [0.1, 0.5, 0.9]  # Start, middle, end (with margin)
        elif self.frames_per_scene == 4:
            return [0.1, 0.4, 0.6, 0.9]
        else:  # 5
            return [0.1, 0.3, 0.5, 0.7, 0.9]
    
    def _detect_scenes(self, video_path: str, fps: float, total_frames: int) -> List[Tuple[float, float]]:
        """Detect scene boundaries"""
        if SCENEDETECT_AVAILABLE:
            try:
                scene_list = detect(
                    video_path,
                    ContentDetector(threshold=self.detection_threshold,
                                   min_scene_len=int(self.min_scene_length * fps))
                )
                if scene_list:
                    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
            except:
                pass
        
        # Fallback
        duration = total_frames / fps
        chunk_size = 5.0
        scenes = []
        current = 0.0
        while current < duration:
            scenes.append((current, min(current + chunk_size, duration)))
            current += chunk_size
        return scenes
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        scenes = self._detect_scenes(video_path, fps, total_frames)
        positions = self._get_scene_positions()
        
        frames = []
        for scene_id, (scene_start, scene_end) in enumerate(scenes):
            scene_duration = scene_end - scene_start
            
            for pos in positions:
                timestamp = scene_start + (scene_duration * pos)
                sampled = self._extract_frame_at_time(cap, timestamp, fps)
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
            metadata={
                "num_scenes": len(scenes),
                "frames_per_scene": self.frames_per_scene,
                "positions": positions
            }
        )


class HistogramChangeSampler(BaseSampler):
    """
    Change detection sampling based on color histogram differences
    
    Extracts frames where significant visual change occurs.
    Simple but effective for detecting scene changes and key moments.
    """
    
    def __init__(self, 
                 threshold: float = 0.3,
                 min_interval_frames: int = 15,
                 max_frames: int = 20):
        """
        Args:
            threshold: Histogram difference threshold (0-1, higher = less sensitive)
            min_interval_frames: Minimum frames between detections
            max_frames: Maximum number of frames to return
        """
        super().__init__(name="histogram_change")
        self.threshold = threshold
        self.min_interval_frames = min_interval_frames
        self.max_frames = max_frames
    
    def _compute_histogram(self, frame: np.ndarray) -> np.ndarray:
        """Compute normalized color histogram"""
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
            
            # Always include first frame
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
                
                # Compute histogram correlation (1 = identical, 0 = different)
                correlation = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)
                change_score = 1 - correlation
                
                # Check if significant change and minimum interval passed
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
        
        # Always include last frame if not already included
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
            metadata={
                "threshold": self.threshold,
                "min_interval": self.min_interval_frames
            }
        )


class MotionBasedSampler(BaseSampler):
    """
    Motion-based sampling using optical flow or frame differencing
    
    Extracts frames with high motion (action scenes) and low motion 
    (static scenes / text displays).
    """
    
    def __init__(self,
                 high_motion_threshold: float = 30.0,
                 low_motion_threshold: float = 5.0,
                 sample_interval: int = 5,
                 max_frames: int = 15):
        """
        Args:
            high_motion_threshold: Motion score above this = high action
            low_motion_threshold: Motion score below this = static/text
            sample_interval: Check every N frames for efficiency
            max_frames: Maximum frames to return
        """
        super().__init__(name="motion_based")
        self.high_motion_threshold = high_motion_threshold
        self.low_motion_threshold = low_motion_threshold
        self.sample_interval = sample_interval
        self.max_frames = max_frames
    
    def _compute_motion_score(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Compute motion score using frame differencing"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # First pass: compute motion scores
        motion_scores = []
        prev_frame = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % self.sample_interval == 0:
                if prev_frame is not None:
                    score = self._compute_motion_score(prev_frame, frame)
                    motion_scores.append((frame_idx, score))
                prev_frame = frame.copy()
            
            frame_idx += 1
        
        # Select frames: high motion peaks + low motion (static)
        high_motion = [(idx, s) for idx, s in motion_scores if s > self.high_motion_threshold]
        low_motion = [(idx, s) for idx, s in motion_scores if s < self.low_motion_threshold]
        
        # Sort by score (most extreme first)
        high_motion.sort(key=lambda x: x[1], reverse=True)
        low_motion.sort(key=lambda x: x[1])
        
        # Combine and limit
        selected_indices = []
        selected_indices.extend([idx for idx, _ in high_motion[:self.max_frames // 2]])
        selected_indices.extend([idx for idx, _ in low_motion[:self.max_frames // 2]])
        selected_indices = sorted(set(selected_indices))[:self.max_frames]
        
        # Extract frames
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frames = []
        for idx in selected_indices:
            sampled = self._extract_frame_at_index(cap, idx, fps)
            if sampled:
                # Find motion score for this frame
                score = next((s for i, s in motion_scores if i == idx), 0)
                sampled.importance_score = score
                frames.append(sampled)
        
        cap.release()
        
        return SamplingResult(
            frames=frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={
                "high_motion_count": len(high_motion),
                "low_motion_count": len(low_motion),
                "total_analyzed": len(motion_scores)
            }
        )


class CLIPDiversitySampler(BaseSampler):
    """
    CLIP-based semantic diversity sampling
    
    Uses CLIP embeddings to select maximally diverse frames.
    This is closer to your full pipeline but without the hierarchical
    deduplication stages.
    """
    
    def __init__(self,
                 num_frames: int = 8,
                 model_name: str = "ViT-B-32",
                 candidate_fps: float = 2.0,
                 device: str = "auto"):
        """
        Args:
            num_frames: Target number of diverse frames
            model_name: CLIP model to use
            candidate_fps: FPS for candidate extraction
            device: 'cuda', 'cpu', or 'auto'
        """
        super().__init__(name=f"clip_diversity_{num_frames}frames")
        self.num_frames = num_frames
        self.model_name = model_name
        self.candidate_fps = candidate_fps
        
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None
        self.preprocess = None
    
    def _load_model(self):
        """Lazy load CLIP model"""
        if self.model is None and CLIP_AVAILABLE:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                self.model_name, pretrained='openai'
            )
            self.model = self.model.to(self.device)
            self.model.eval()
    
    def _compute_embeddings(self, frames: List[np.ndarray]) -> np.ndarray:
        """Compute CLIP embeddings for frames"""
        from PIL import Image
        
        self._load_model()
        
        embeddings = []
        with torch.no_grad():
            for frame in frames:
                # Convert BGR to RGB PIL Image
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                
                # Preprocess and encode
                img_tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
                embedding = self.model.encode_image(img_tensor)
                embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy().flatten())
        
        return np.array(embeddings)
    
    def _max_diversity_selection(self, embeddings: np.ndarray, k: int) -> List[int]:
        """Select k most diverse frames using greedy max-min diversity"""
        n = len(embeddings)
        if n <= k:
            return list(range(n))
        
        # Compute pairwise distances (1 - cosine similarity)
        similarities = embeddings @ embeddings.T
        distances = 1 - similarities
        
        selected = [0]  # Start with first frame
        
        while len(selected) < k:
            # Find frame with maximum minimum distance to selected
            max_min_dist = -1
            best_idx = -1
            
            for i in range(n):
                if i in selected:
                    continue
                
                min_dist = min(distances[i, j] for j in selected)
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    best_idx = i
            
            if best_idx >= 0:
                selected.append(best_idx)
            else:
                break
        
        return sorted(selected)
    
    def sample(self, video_path: str, **kwargs) -> SamplingResult:
        import time
        start_time = time.time()
        
        if not CLIP_AVAILABLE:
            raise RuntimeError("CLIP not available. Install with: pip install open-clip-torch")
        
        cap, metadata = self._load_video_metadata(video_path)
        fps = metadata["fps"]
        total_frames = metadata["total_frames"]
        
        # Extract candidates at specified FPS
        frame_interval = int(fps / self.candidate_fps)
        candidates = []
        candidate_indices = []
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                candidates.append(frame)
                candidate_indices.append(frame_idx)
            
            frame_idx += 1
        
        cap.release()
        
        if not candidates:
            return SamplingResult([], self.name, video_path, 0, 0)

        # Compute embeddings for all candidates
        embeddings = self._compute_embeddings(candidates)
        
        # Select diverse frames
        selected_indices_in_candidates = self._max_diversity_selection(embeddings, self.num_frames)
        
        # Map back to original video frames
        final_frames = []
        for idx in selected_indices_in_candidates:
            frame_array = candidates[idx]
            original_idx = candidate_indices[idx]
            timestamp = original_idx / fps
            
            sampled = SampledFrame(
                frame=frame_array,
                timestamp=timestamp,
                frame_index=original_idx,
                method=self.name
            )
            final_frames.append(sampled)
        
        return SamplingResult(
            frames=final_frames,
            method_name=self.name,
            video_path=video_path,
            video_duration=metadata["duration"],
            total_video_frames=total_frames,
            processing_time=time.time() - start_time,
            metadata={
                "candidate_count": len(candidates),
                "embedding_dim": embeddings.shape[1] if len(embeddings) > 0 else 0
            }
        )


def run_ablation_study(video_paths: List[str], output_dir: str = "ablation_results"):
    """
    Run all baselines on a set of videos and save results for comparison.
    """
    import os
    import json
    import time
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Define baselines to compare
    baselines = [
        UniformSampler(sample_fps=1.0),                  # Standard industry baseline
        UniformCountSampler(num_frames=10),              # Fixed budget constraint
        FirstLastMidSampler(include_quartiles=True),     # Minimal heuristic
        HistogramChangeSampler(threshold=0.3),           # Visual change baseline
        SceneMidpointSampler(),                          # Structural baseline
    ]
    
    # Add heavy baselines if available
    if SCENEDETECT_AVAILABLE:
        baselines.append(SceneMultiSampler(frames_per_scene=3))
    
    if CLIP_AVAILABLE:
        baselines.append(CLIPDiversitySampler(num_frames=10))

    results_summary = []

    print(f"Starting Ablation Study on {len(video_paths)} videos...")
    print(f" comparing {len(baselines)} baseline methods.")

    for vid_idx, video_path in enumerate(video_paths):
        video_name = os.path.basename(video_path)
        print(f"\nProcessing [{vid_idx+1}/{len(video_paths)}]: {video_name}")
        
        for sampler in baselines:
            try:
                print(f"  - Running {sampler.name}...", end="", flush=True)
                
                # Run sampling
                result = sampler.sample(video_path)
                
                # Save preview (optional, helpful for visual verification)
                # save_contact_sheet(result, output_dir) 
                
                print(f" Done. ({result.num_frames} frames, {result.processing_time:.2f}s)")
                
                # Record metrics
                results_summary.append({
                    "video": video_name,
                    "method": sampler.name,
                    "num_frames": result.num_frames,
                    "compression_ratio": result.compression_ratio,
                    "processing_time": result.processing_time,
                    "fps_equivalent": result.num_frames / result.video_duration if result.video_duration > 0 else 0
                })
                
            except Exception as e:
                print(f" Failed! Error: {e}")
                results_summary.append({
                    "video": video_name,
                    "method": sampler.name,
                    "error": str(e)
                })

    # Save aggregate results
    json_path = os.path.join(output_dir, "ablation_metrics.json")
    with open(json_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print(f"\nAblation study complete. Metrics saved to {json_path}")
    return results_summary

def save_contact_sheet(result: SamplingResult, output_dir: str):
    """Helper to visualize the sampled frames"""
    if not result.frames:
        return

    # Create a simple grid
    frames = [f.frame for f in result.frames]
    count = len(frames)
    cols = 5
    rows = (count + cols - 1) // cols
    
    # Resize for contact sheet
    thumb_w, thumb_h = 320, 180
    thumbs = [cv2.resize(f, (thumb_w, thumb_h)) for f in frames]
    
    # Create blank canvas
    sheet = np.zeros((rows * thumb_h, cols * thumb_w, 3), dtype=np.uint8)
    
    for i, thumb in enumerate(thumbs):
        r = i // cols
        c = i % cols
        sheet[r*thumb_h:(r+1)*thumb_h, c*thumb_w:(c+1)*thumb_w] = thumb
        
        # Add timestamp text
        cv2.putText(sheet, f"{result.frames[i].timestamp:.1f}s", 
                   (c*thumb_w + 10, r*thumb_h + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    filename = f"{os.path.basename(result.video_path)}_{result.method_name}.jpg"
    cv2.imwrite(os.path.join(output_dir, filename), sheet)

if __name__ == "__main__":
    # Example usage
    # Replace with your actual video paths from the 'hard' dataset subset
    test_videos = [
        "data/hussain_videos/-1yXdOufzKE.mp4", 
        "data/hussain_videos/-HWd3Nqu8_0.mp4"
    ]
    
    # Filter only existing files
    import os
    valid_videos = [v for v in test_videos if os.path.exists(v)]
    
    if valid_videos:
        run_ablation_study(valid_videos)
    else:
        print("No valid videos found. Please update test_videos list.")