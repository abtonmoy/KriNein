# Adaptive Video Advertisement Analysis Pipeline

## Project Overview

This research project develops an **efficient, hierarchical frame extraction pipeline** for analyzing video advertisements using Large Language Models (LLMs). The core innovation is reducing computational cost and API usage while maintaining extraction quality through intelligent, multi-stage frame selection.

### Research Question

> Can we significantly reduce the number of frames sent to vision-language models for advertisement analysis while preserving (or improving) the quality of extracted insights?

### Target Publication Venues

- **Primary:** ICME, MMM, WACV, ACM Multimedia workshops
- **Reach:** ACM MM, ECCV workshops

---

## Problem Statement

### Current Approach (Naive Pipeline)

```
Video → Fixed-interval sampling (e.g., 1 frame/0.3s) → CLIP embedding →
Cosine similarity deduplication → LLM Vision API → Insights
```

**Problems:**

1. **Wasteful extraction:** Many frames extracted only to be discarded
2. **Arbitrary thresholds:** 0.3s interval is content-agnostic
3. **Expensive deduplication:** CLIP embeddings computed for all frames before filtering
4. **No temporal awareness:** Similarity computed pairwise without scene context
5. **High API costs:** More frames = more tokens = higher costs

### Proposed Approach (Hierarchical Pipeline)

```
Video → Lightweight change detection → Adaptive sampling →
Hierarchical deduplication (pHash → Scene → CLIP) →
Temporal clustering → LLM Vision API → Insights
```

**Improvements:**

1. **Smart extraction:** Only extract frames when content changes significantly
2. **Content-adaptive:** Sampling rate responds to video dynamics
3. **Cheap-to-expensive filtering:** Use fast methods first, expensive methods last
4. **Scene-aware:** Respect narrative structure of advertisements
5. **Cost-efficient:** Fewer, more informative frames sent to LLM

---

## Key Features Summary

| Feature                        | Status         | Description                                             |
| ------------------------------ | -------------- | ------------------------------------------------------- |
| **Hierarchical Deduplication** | ✅ Core        | pHash → SSIM → CLIP (cheap to expensive)                |
| **Scene-Aware Selection**      | ✅ Core        | Respects narrative structure via scene detection        |
| **Batch Processing**           | ✅ Implemented | Parallel video processing + GPU-batched CLIP            |
| **Temporal Reasoning**         | ✅ Implemented | Multi-frame prompts with timestamps & narrative context |
| **Adaptive Schema**            | ✅ Implemented | Two-pass extraction with type-specific schemas          |
| **Multilingual Support**       | 🟡 Planned     | Swap to multilingual OCR/ASR models                     |
| **Audio-Visual Fusion**        | 🟡 Planned     | Boost frames near audio events                          |
| **Streaming Processing**       | 🔴 Future      | Requires architecture redesign                          |
| **Learned Thresholds**         | 🔴 Future      | Requires meta-learning research                         |

---

## Technical Architecture

### Stage 1: Video Ingestion & Metadata Extraction

**Input:** Video file (MP4, MOV, etc.)

**Operations:**

- Extract video metadata (duration, fps, resolution, codec)
- Extract audio track for parallel processing
- Compute video-level statistics (average brightness, motion intensity)

**Output:**

- Video metadata JSON
- Separated audio file
- Initial video statistics

**Tools:** FFmpeg, OpenCV

### Stage 2: Lightweight Change Detection

**Purpose:** Identify candidate moments where frame extraction is worthwhile

**Methods (in order of computational cost):**

| Method                      | Cost      | What It Detects           |
| --------------------------- | --------- | ------------------------- |
| Frame difference (L1/L2)    | Very Low  | Any pixel changes         |
| Histogram difference        | Low       | Color distribution shifts |
| Motion vectors (from codec) | Near Zero | Movement between frames   |
| Edge change ratio           | Low       | Structural changes        |

**Implementation:**

```python
def compute_frame_difference(frame1, frame2):
    """L1 norm of grayscale difference, normalized by frame size."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    diff = np.abs(gray1.astype(float) - gray2.astype(float))
    return np.mean(diff) / 255.0

def compute_histogram_difference(frame1, frame2):
    """Chi-square distance between color histograms."""
    hist1 = cv2.calcHist([frame1], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    hist2 = cv2.calcHist([frame2], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
```

**Output:** List of timestamps where significant change detected

**Adaptive Threshold Logic:**

- Fast-paced ads (many scene cuts): Lower threshold, more candidates
- Slow-paced ads (few cuts): Higher threshold, fewer candidates
- Threshold adjusts based on running statistics of the video

### Stage 3: Scene Boundary Detection

**Purpose:** Segment video into coherent scenes/shots

**Methods:**

| Method                            | Description                    | When to Use          |
| --------------------------------- | ------------------------------ | -------------------- |
| PySceneDetect (ContentDetector)   | Detects content changes        | General purpose      |
| PySceneDetect (ThresholdDetector) | Detects fade-to-black          | TV commercials       |
| TransNetV2                        | Neural shot boundary detection | High accuracy needed |

**Implementation:**

```python
from scenedetect import detect, ContentDetector, ThresholdDetector

def detect_scenes(video_path, method='content'):
    """Detect scene boundaries in video."""
    if method == 'content':
        detector = ContentDetector(threshold=27.0)
    elif method == 'threshold':
        detector = ThresholdDetector(threshold=12)

    scene_list = detect(video_path, detector)
    return [(s[0].get_seconds(), s[1].get_seconds()) for s in scene_list]
```

**Output:** List of (start_time, end_time) tuples for each scene

### Stage 4: Hierarchical Frame Deduplication

**Purpose:** Remove redundant frames using progressively expensive methods

**Layer 1: Perceptual Hashing (pHash)**

- **Cost:** ~0.1ms per frame
- **What it catches:** Near-identical frames, minor compression artifacts
- **Threshold:** Hamming distance < 8

```python
import imagehash
from PIL import Image

def compute_phash(frame):
    """Compute perceptual hash of frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    return imagehash.phash(pil_image)

def phash_similar(hash1, hash2, threshold=8):
    """Check if two frames are similar via pHash."""
    return hash1 - hash2 < threshold
```

**Layer 2: Structural Similarity (SSIM)**

- **Cost:** ~5ms per frame pair
- **What it catches:** Frames with same structure but different details
- **Threshold:** SSIM > 0.92

```python
from skimage.metrics import structural_similarity as ssim

def ssim_similar(frame1, frame2, threshold=0.92):
    """Check structural similarity between frames."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score = ssim(gray1, gray2)
    return score > threshold
```

**Layer 3: CLIP Embedding Similarity**

- **Cost:** ~50ms per frame (GPU), ~500ms (CPU)
- **What it catches:** Semantically similar frames with visual differences
- **Threshold:** Cosine similarity > 0.90

```python
import torch
import clip

class CLIPEmbedder:
    def __init__(self, device='cuda'):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)

    def embed(self, frame):
        """Compute CLIP embedding for frame."""
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image_input = self.preprocess(pil_image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model.encode_image(image_input)
        return embedding.cpu().numpy().flatten()

    def similarity(self, emb1, emb2):
        """Compute cosine similarity between embeddings."""
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
```

**Hierarchical Flow:**

```
All candidate frames
    │
    ▼ pHash filtering (removes ~40-60%)
Frames passing pHash
    │
    ▼ SSIM filtering (removes ~20-30% more)
Frames passing SSIM
    │
    ▼ CLIP filtering (removes ~10-20% more)
Final keyframes
```

### Stage 5: Temporal Clustering & Representative Selection

**Purpose:** Group remaining frames by scene and select best representatives

**Algorithm:**

1. Group frames by scene boundaries (from Stage 3)
2. Within each scene, cluster by CLIP embeddings
3. Select frame closest to cluster centroid as representative
4. Ensure temporal spread (don't select adjacent frames)

```python
from sklearn.cluster import KMeans

def select_representatives(frames, embeddings, scene_boundaries, max_per_scene=3):
    """Select representative frames from each scene."""
    representatives = []

    for start, end in scene_boundaries:
        # Get frames in this scene
        scene_frames = [(f, e) for f, e in zip(frames, embeddings)
                        if start <= f['timestamp'] < end]

        if len(scene_frames) <= max_per_scene:
            representatives.extend([f for f, e in scene_frames])
            continue

        # Cluster and select centroids
        scene_embeddings = np.array([e for f, e in scene_frames])
        n_clusters = min(max_per_scene, len(scene_frames))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(scene_embeddings)

        # Select frame closest to each centroid
        for i in range(n_clusters):
            cluster_frames = [f for (f, e), label in zip(scene_frames, kmeans.labels_)
                             if label == i]
            cluster_embeddings = [e for (f, e), label in zip(scene_frames, kmeans.labels_)
                                  if label == i]
            centroid = kmeans.cluster_centers_[i]
            distances = [np.linalg.norm(e - centroid) for e in cluster_embeddings]
            best_idx = np.argmin(distances)
            representatives.append(cluster_frames[best_idx])

    return representatives
```

### Stage 6: Audio-Visual Alignment (Optional Enhancement)

**Purpose:** Align frame selection with audio events for better context

**Audio Features to Extract:**

- Speech boundaries (using VAD - Voice Activity Detection)
- Music/jingle detection
- Audio energy peaks
- Silence detection (often indicates scene transitions)

```python
import librosa

def extract_audio_events(audio_path):
    """Extract significant audio events."""
    y, sr = librosa.load(audio_path)

    # Energy-based event detection
    rms = librosa.feature.rms(y=y)[0]
    times = librosa.times_like(rms, sr=sr)

    # Find peaks in audio energy
    peaks = librosa.util.peak_pick(rms, pre_max=3, post_max=3,
                                    pre_avg=3, post_avg=5, delta=0.1, wait=10)

    return times[peaks].tolist()
```

**Integration:** Boost importance of frames near audio events

### Stage 7: LLM Vision API Integration

**Purpose:** Extract structured insights from selected keyframes

**Supported APIs:**

- Claude Vision (Anthropic)
- GPT-4V (OpenAI)
- Gemini Pro Vision (Google)

**Temporal-Aware Prompt Construction:**

The LLM receives all keyframes together with temporal context, enabling narrative understanding:

```python
def build_temporal_prompt(frames_with_timestamps, schema):
    """
    Build a prompt that gives LLM temporal context for narrative understanding.

    Args:
        frames_with_timestamps: List of (frame, timestamp) tuples
        schema: JSON schema for extraction

    Returns:
        Formatted prompt string
    """
    video_duration = frames_with_timestamps[-1][1]
    num_frames = len(frames_with_timestamps)

    prompt = f"""You are analyzing a {video_duration:.1f}-second video advertisement through {num_frames} keyframes.

The frames are in CHRONOLOGICAL ORDER with timestamps. Analyze both individual frames AND the narrative progression.

TEMPORAL CONTEXT:
"""

    for i, (frame, ts) in enumerate(frames_with_timestamps):
        prompt += f"\n[Frame {i+1} @ {ts:.1f}s]"
        if i > 0:
            time_gap = ts - frames_with_timestamps[i-1][1]
            prompt += f" (Δ {time_gap:.1f}s from previous)"

        # Add position context
        position = ts / video_duration
        if position < 0.2:
            prompt += " [OPENING]"
        elif position > 0.8:
            prompt += " [CLOSING]"

    prompt += f"""

ANALYSIS INSTRUCTIONS:
1. Identify what CHANGES between frames (scene transitions, new elements, text changes)
2. Track the NARRATIVE ARC (setup → development → conclusion/CTA)
3. Note any RECURRING ELEMENTS (logo appearances, product shots, faces)

Extract the following information in JSON format:
{json.dumps(schema, indent=2)}

Respond ONLY with valid JSON."""

    return prompt
```

**Adaptive Schema Selection:**

The pipeline automatically selects the appropriate extraction schema based on detected ad type:

```python
class AdaptiveSchemaSelector:
    """Select extraction schema based on ad type detection."""

    # Base schema (always extracted)
    BASE_SCHEMA = {
        "brand": {
            "name": "string",
            "logo_visible": "boolean",
            "logo_timestamps": ["float"]
        },
        "message": {
            "primary_message": "string",
            "call_to_action": "string | null",
            "tagline": "string | null"
        },
        "creative_elements": {
            "dominant_colors": ["string"],
            "text_overlays": ["string"],
            "music_mood": "string | null"
        }
    }

    # Type-specific schema extensions
    SCHEMA_EXTENSIONS = {
        "product_demo": {
            "product": {
                "name": "string",
                "category": "string",
                "features_demonstrated": ["string"],
                "price_shown": "string | null"
            },
            "demo_steps": ["string"]
        },
        "testimonial": {
            "testimonial": {
                "speaker_name": "string | null",
                "speaker_role": "string | null",
                "key_quotes": ["string"],
                "credibility_markers": ["string"]
            }
        },
        "brand_awareness": {
            "emotional_appeal": {
                "primary_emotion": "string",
                "storytelling_elements": ["string"],
                "brand_values_conveyed": ["string"]
            }
        },
        "tutorial": {
            "tutorial": {
                "skill_taught": "string",
                "steps": ["string"],
                "tools_shown": ["string"]
            }
        },
        "entertainment": {
            "entertainment": {
                "humor_type": "string | null",
                "celebrity_featured": "string | null",
                "viral_elements": ["string"]
            }
        }
    }

    def detect_ad_type(self, frames, llm_client):
        """First pass: detect ad type from frames."""
        detection_prompt = """
        Classify this advertisement into exactly ONE category:
        - product_demo (shows product features/usage)
        - testimonial (features customer/expert reviews)
        - brand_awareness (emotional storytelling, no specific product)
        - tutorial (teaches how to do something)
        - entertainment (comedy, celebrity, viral content)

        Respond with ONLY the category name, nothing else.
        """
        ad_type = llm_client.extract(frames, detection_prompt).strip().lower()
        return ad_type if ad_type in self.SCHEMA_EXTENSIONS else "brand_awareness"

    def get_schema(self, ad_type):
        """Get combined schema for ad type."""
        schema = self.BASE_SCHEMA.copy()
        if ad_type in self.SCHEMA_EXTENSIONS:
            schema.update(self.SCHEMA_EXTENSIONS[ad_type])
        return schema

    def extract_adaptive(self, frames, llm_client):
        """Two-pass extraction with adaptive schema."""
        # Pass 1: Detect type
        ad_type = self.detect_ad_type(frames, llm_client)

        # Pass 2: Extract with type-specific schema
        schema = self.get_schema(ad_type)
        result = llm_client.extract(frames, schema)
        result["_detected_ad_type"] = ad_type

        return result
```

**Single-Pass Flexible Schema (Alternative):**

For simpler use cases, use a universal flexible schema:

```python
FLEXIBLE_SCHEMA = {
    "brand": {
        "name": "string",
        "logo_visible": "boolean"
    },
    "ad_type": "string (product_demo | testimonial | brand_awareness | tutorial | entertainment)",
    "message": {
        "primary_message": "string",
        "call_to_action": "string | null"
    },
    "narrative": {
        "opening_hook": "string",
        "middle_development": "string",
        "closing_resolution": "string"
    },

    # Conditional fields - LLM fills only if applicable
    "product": "object | null (include if product is shown)",
    "testimonial": "object | null (include if testimonial present)",
    "emotional_appeal": "object | null (include if emotion-focused)",

    "persuasion_techniques": ["string"],
    "target_audience": {
        "age_group": "string",
        "interests": ["string"]
    }
}
```

---

## Evaluation Framework

### Datasets

| Dataset                        | Size             | Annotations                     | Use Case               |
| ------------------------------ | ---------------- | ------------------------------- | ---------------------- |
| **Hussain et al. (CVPR 2017)** | 3,477 video ads  | Topic, sentiment, action-reason | Primary benchmark      |
| **LAMBDA**                     | 2,205 ads        | Memorability scores             | Secondary validation   |
| **Custom sponsored content**   | 42 videos/images | Full extraction ground truth    | LLM quality evaluation |

### Metrics

#### Efficiency Metrics

| Metric               | Formula                              | Target              |
| -------------------- | ------------------------------------ | ------------------- |
| Frame Reduction Rate | 1 - (selected_frames / total_frames) | > 70%               |
| API Cost Reduction   | 1 - (our_tokens / baseline_tokens)   | > 60%               |
| Processing Time      | Total pipeline time in seconds       | < 2x video duration |
| Memory Usage         | Peak RAM usage in MB                 | < 4GB               |

#### Quality Metrics

| Metric                  | Description                                             | How to Compute            |
| ----------------------- | ------------------------------------------------------- | ------------------------- |
| Extraction Accuracy     | Match between extracted and ground truth fields         | Field-by-field comparison |
| Extraction Completeness | Percentage of ground truth fields recovered             | Recall of fields          |
| Semantic Similarity     | Embedding similarity of extracted vs. ground truth text | SBERT similarity          |
| Human Evaluation        | Blind comparison of outputs                             | A/B preference study      |

### Baselines to Compare Against

| Baseline         | Description                                                |
| ---------------- | ---------------------------------------------------------- |
| **Uniform-0.3s** | Extract frame every 0.3 seconds                            |
| **Uniform-1.0s** | Extract frame every 1.0 seconds                            |
| **CLIP-Only**    | Uniform extraction + CLIP deduplication (current approach) |
| **Scene-Only**   | Extract first frame of each scene                          |
| **LMSKE-style**  | TransNetV2 + CLIP + adaptive clustering                    |
| **Random**       | Random frame selection (negative baseline)                 |

### Ablation Studies

| Experiment             | What We Vary                   | What We Measure                     |
| ---------------------- | ------------------------------ | ----------------------------------- |
| pHash threshold        | Hamming distance: 4, 8, 12, 16 | Frames retained, downstream quality |
| CLIP threshold         | Cosine sim: 0.85, 0.90, 0.95   | Frames retained, downstream quality |
| Hierarchical vs. flat  | With/without pHash/SSIM layers | Processing time, quality            |
| Scene detection method | PySceneDetect vs. TransNetV2   | Accuracy, speed tradeoff            |
| Frames per scene       | 1, 2, 3, 5 representatives     | Coverage vs. efficiency             |

---

## Project Structure

```
ad-video-pipeline/
├── README.md
├── requirements.txt
├── setup.py
│
├── configs/
│   ├── default.yaml           # Default pipeline configuration
│   ├── fast.yaml              # Speed-optimized configuration
│   └── quality.yaml           # Quality-optimized configuration
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # Main pipeline orchestrator
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── video_loader.py    # Video loading and metadata
│   │   └── audio_extractor.py # Audio track extraction
│   │
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── change_detector.py # Lightweight change detection
│   │   ├── scene_detector.py  # Scene boundary detection
│   │   └── audio_events.py    # Audio event detection
│   │
│   ├── deduplication/
│   │   ├── __init__.py
│   │   ├── phash.py           # Perceptual hashing
│   │   ├── ssim.py            # Structural similarity
│   │   ├── clip_embed.py      # CLIP embeddings
│   │   └── hierarchical.py    # Hierarchical dedup orchestrator
│   │
│   ├── selection/
│   │   ├── __init__.py
│   │   ├── clustering.py      # Temporal clustering
│   │   └── representative.py  # Representative frame selection
│   │
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── llm_client.py      # LLM API wrapper
│   │   ├── prompts.py         # Prompt templates
│   │   └── schema.py          # Extraction schema definitions
│   │
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py     # Video I/O utilities
│       ├── metrics.py         # Evaluation metrics
│       └── visualization.py   # Result visualization
│
├── data/
│   ├── raw/                   # Original video files
│   ├── processed/             # Extracted frames and features
│   ├── annotations/           # Ground truth annotations
│   └── results/               # Extraction results
│
├── experiments/
│   ├── run_baseline.py        # Run baseline methods
│   ├── run_ablation.py        # Run ablation studies
│   ├── evaluate.py            # Compute metrics
│   └── visualize_results.py   # Generate figures
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_pipeline_demo.ipynb
│   ├── 03_results_analysis.ipynb
│   └── 04_paper_figures.ipynb
│
└── tests/
    ├── test_detection.py
    ├── test_deduplication.py
    └── test_extraction.py
```

---

## Configuration

### Default Configuration (configs/default.yaml)

```yaml
# Pipeline configuration
pipeline:
  name: "adaptive-ad-pipeline"
  version: "1.0.0"

# Video ingestion
ingestion:
  max_resolution: 720 # Downscale if larger
  extract_audio: true

# Change detection
change_detection:
  method: "histogram" # Options: frame_diff, histogram, edge
  threshold: 0.15
  min_interval_ms: 100 # Minimum time between candidates

# Scene detection
scene_detection:
  method: "content" # Options: content, threshold, transnet
  threshold: 27.0
  min_scene_length_s: 0.5

# Hierarchical deduplication
deduplication:
  phash:
    enabled: true
    threshold: 8 # Hamming distance
  ssim:
    enabled: true
    threshold: 0.92
  clip:
    enabled: true
    model: "ViT-B/32"
    threshold: 0.90
    device: "cuda" # Options: cuda, cpu
    batch_size: 32 # GPU batching for efficiency

# Representative selection
selection:
  method: "clustering" # Options: clustering, uniform, first
  max_frames_per_scene: 3
  min_temporal_gap_s: 0.5

# LLM extraction
extraction:
  provider: "anthropic" # Options: anthropic, openai, google
  model: "claude-sonnet-4-20250514"
  max_tokens: 2000
  temperature: 0.0

  # Temporal reasoning
  temporal_context:
    enabled: true
    include_timestamps: true
    include_time_deltas: true
    include_position_labels: true # [OPENING], [CLOSING]
    include_narrative_instructions: true

  # Adaptive schema
  schema:
    mode: "adaptive" # Options: adaptive, fixed, flexible
    # For adaptive mode: two-pass (detect type, then extract)
    # For fixed mode: use schema_name
    # For flexible mode: universal schema with optional fields
    schema_name: "full" # Used when mode=fixed
    confidence_sampling:
      enabled: false
      n_samples: 3
      temperature: 0.3

# Batch processing
batch:
  enabled: true
  max_workers: 4 # Number of parallel video processors
  gpu_batch_size: 32 # CLIP embedding batch size

# Evaluation
evaluation:
  metrics:
    - frame_reduction_rate
    - api_cost_reduction
    - processing_time
    - extraction_accuracy
  save_intermediate: true
```

### Speed-Optimized Configuration (configs/fast.yaml)

```yaml
# Inherit from default
_extends: default.yaml

# Override for speed
ingestion:
  max_resolution: 480 # Lower resolution

deduplication:
  ssim:
    enabled: false # Skip SSIM layer
  clip:
    model: "ViT-B/16" # Faster CLIP model
    threshold: 0.85 # More aggressive dedup
    batch_size: 64

selection:
  max_frames_per_scene: 2 # Fewer frames

extraction:
  schema:
    mode: "fixed"
    schema_name: "minimal"
```

### Quality-Optimized Configuration (configs/quality.yaml)

```yaml
# Inherit from default
_extends: default.yaml

# Override for quality
scene_detection:
  method: "transnet" # Neural scene detection

deduplication:
  phash:
    threshold: 12 # Less aggressive
  clip:
    model: "ViT-L/14" # Larger CLIP model
    threshold: 0.95 # Keep more frames

selection:
  max_frames_per_scene: 5 # More frames per scene

extraction:
  schema:
    mode: "adaptive"
    confidence_sampling:
      enabled: true
      n_samples: 3
```

---

## API Usage

### Basic Pipeline Usage

```python
from src.pipeline import AdVideoPipeline

# Initialize pipeline
pipeline = AdVideoPipeline(config_path="configs/default.yaml")

# Process single video
result = pipeline.process("path/to/video.mp4")

# Access results
print(f"Frames extracted: {result.num_frames}")
print(f"Frames selected: {result.num_selected}")
print(f"Reduction rate: {result.reduction_rate:.2%}")
print(f"Extracted insights: {result.insights}")

# Process batch (parallelized)
results = pipeline.process_batch("path/to/video/directory/", max_workers=4)
```

### Batch Processing (Parallelized)

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from src.pipeline import AdVideoPipeline

class BatchProcessor:
    def __init__(self, config_path="configs/default.yaml"):
        self.config_path = config_path

    def process_batch(self, video_paths, max_workers=4):
        """Process multiple videos in parallel using multiprocessing."""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self._process_single, video_paths))
        return results

    def _process_single(self, video_path):
        """Process a single video (each worker gets its own pipeline instance)."""
        pipeline = AdVideoPipeline(config_path=self.config_path)
        return pipeline.process(video_path)

# Usage
processor = BatchProcessor()
video_paths = ["ad1.mp4", "ad2.mp4", "ad3.mp4", "ad4.mp4"]
results = processor.process_batch(video_paths, max_workers=4)
```

### GPU-Batched CLIP Embeddings

```python
import torch
import numpy as np

class BatchedCLIPEmbedder:
    """Efficient batched CLIP inference for multiple frames."""

    def __init__(self, model_name="ViT-B/32", device="cuda", batch_size=32):
        import clip
        self.device = device
        self.batch_size = batch_size
        self.model, self.preprocess = clip.load(model_name, device=device)

    def embed_batch(self, frames):
        """
        Embed multiple frames efficiently using GPU batching.

        Args:
            frames: List of PIL Images or numpy arrays

        Returns:
            numpy array of shape (num_frames, embedding_dim)
        """
        embeddings = []

        for i in range(0, len(frames), self.batch_size):
            batch = frames[i:i + self.batch_size]

            # Preprocess batch
            batch_tensor = torch.stack([
                self.preprocess(self._to_pil(f)) for f in batch
            ]).to(self.device)

            # Batch inference
            with torch.no_grad():
                batch_embeddings = self.model.encode_image(batch_tensor)

            embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _to_pil(self, frame):
        """Convert numpy array to PIL Image if needed."""
        from PIL import Image
        if isinstance(frame, np.ndarray):
            return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return frame

# Usage
embedder = BatchedCLIPEmbedder(batch_size=32, device="cuda")
frames = [frame1, frame2, frame3, ...]  # List of 100+ frames
embeddings = embedder.embed_batch(frames)  # Much faster than one-by-one
```

````

### Custom Configuration

```python
from src.pipeline import AdVideoPipeline

# Override specific settings
custom_config = {
    "deduplication": {
        "clip": {
            "threshold": 0.85  # More aggressive deduplication
        }
    },
    "selection": {
        "max_frames_per_scene": 2  # Fewer frames
    }
}

pipeline = AdVideoPipeline(
    config_path="configs/default.yaml",
    overrides=custom_config
)
````

### Evaluation

```python
from experiments.evaluate import Evaluator

evaluator = Evaluator(
    ground_truth_path="data/annotations/",
    results_path="data/results/"
)

# Compute all metrics
metrics = evaluator.compute_all()
print(metrics.to_dataframe())

# Compare against baselines
comparison = evaluator.compare_baselines([
    "uniform_0.3s",
    "clip_only",
    "ours"
])
comparison.plot()
```

---

## Dependencies

### Core Dependencies

```
# requirements.txt

# Video processing
opencv-python>=4.8.0
ffmpeg-python>=0.2.0
scenedetect>=0.6.0

# Image processing
Pillow>=10.0.0
imagehash>=4.3.0
scikit-image>=0.21.0

# Deep learning
torch>=2.0.0
clip @ git+https://github.com/openai/CLIP.git
transformers>=4.30.0

# Audio processing
librosa>=0.10.0
soundfile>=0.12.0

# ML utilities
numpy>=1.24.0
scikit-learn>=1.3.0
scipy>=1.11.0

# LLM APIs
anthropic>=0.18.0
openai>=1.0.0
google-generativeai>=0.3.0

# Utilities
pyyaml>=6.0
tqdm>=4.65.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Evaluation
sentence-transformers>=2.2.0  # For semantic similarity
```

### Optional Dependencies

```
# For TransNetV2 scene detection
transnetv2 @ git+https://github.com/soCzech/TransNetV2.git

# For GPU acceleration
cupy-cuda12x>=12.0.0

# For experiment tracking
wandb>=0.15.0
mlflow>=2.5.0
```

---

## Timeline & Milestones

### Phase 1: Foundation (Weeks 1-2)

- [ ] Set up project structure
- [ ] Implement video ingestion module
- [ ] Implement change detection methods
- [ ] Basic scene detection integration

### Phase 2: Core Pipeline (Weeks 3-4)

- [ ] Implement hierarchical deduplication
- [ ] Implement temporal clustering
- [ ] Integrate LLM extraction
- [ ] End-to-end pipeline working

### Phase 3: Evaluation (Weeks 5-6)

- [ ] Download and preprocess Hussain et al. dataset
- [ ] Implement baseline methods
- [ ] Create ground truth annotations for subset
- [ ] Run full evaluation suite

### Phase 4: Analysis & Writing (Weeks 7-8)

- [ ] Ablation studies
- [ ] Generate paper figures
- [ ] Write paper draft
- [ ] Code cleanup and documentation

---

## Known Limitations & Future Work

### Implemented Features (Previously Listed as Limitations)

These items were initially considered limitations but are now implemented in the pipeline:

| Feature               | Implementation                                          | Complexity |
| --------------------- | ------------------------------------------------------- | ---------- |
| ✅ Batch processing   | `ProcessPoolExecutor` + GPU-batched CLIP                | Easy       |
| ✅ Temporal reasoning | Multi-frame prompts with timestamps & narrative context | Easy       |
| ✅ Adaptive schema    | Two-pass detection + type-specific schemas              | Easy       |

### Medium-Difficulty Enhancements (Planned)

These require additional work but are achievable with existing tools:

#### 1. Multilingual Support

**Challenge:** OCR and ASR currently assume English content.

**Solution Path:**

```python
# Use multilingual models
from transformers import pipeline

# Multilingual OCR
ocr = pipeline("image-to-text", model="microsoft/trocr-large-printed")

# Multilingual ASR
asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3")

# Language detection
from langdetect import detect
detected_lang = detect(extracted_text)

# Multilingual embeddings for text
from sentence_transformers import SentenceTransformer
multilingual_encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
```

**Status:** Requires model swaps and testing, ~2-3 days of work

#### 2. Audio-Visual Fusion

**Challenge:** Currently audio and visual tracks are processed independently.

**Solution Path:**

```python
class AudioVisualAligner:
    """Align audio events with visual keyframes."""

    def __init__(self):
        self.vad = load_vad_model()  # Voice Activity Detection
        self.audio_classifier = load_audio_classifier()

    def extract_audio_events(self, audio_path):
        """Extract speech, music, and silence boundaries."""
        y, sr = librosa.load(audio_path)

        events = {
            "speech_segments": self.vad.detect(y, sr),
            "music_segments": self.detect_music(y, sr),
            "silence_segments": self.detect_silence(y, sr),
            "energy_peaks": self.detect_peaks(y, sr)
        }
        return events

    def boost_frame_importance(self, frames, audio_events):
        """Increase importance of frames near audio events."""
        for frame in frames:
            ts = frame["timestamp"]

            # Boost if frame is at speech onset
            if self.near_event(ts, audio_events["speech_segments"], "start"):
                frame["importance"] *= 1.5

            # Boost if frame is at music change
            if self.near_event(ts, audio_events["music_segments"], "boundary"):
                frame["importance"] *= 1.3

            # Boost if frame is after silence (attention reset)
            if self.near_event(ts, audio_events["silence_segments"], "end"):
                frame["importance"] *= 1.4

        return frames
```

**Status:** Requires audio ML integration, ~1 week of work

#### 3. Confidence-Based Extraction

**Challenge:** LLM extractions don't include confidence scores.

**Solution Path:**

```python
def extract_with_confidence(frames, schema, llm_client, n_samples=3):
    """
    Extract multiple times and compute confidence via agreement.
    """
    extractions = []
    for i in range(n_samples):
        result = llm_client.extract(frames, schema, temperature=0.3)
        extractions.append(result)

    # Compute field-level confidence based on agreement
    final_result = {}
    confidences = {}

    for field in schema.keys():
        values = [e.get(field) for e in extractions]
        most_common = Counter(values).most_common(1)[0]
        final_result[field] = most_common[0]
        confidences[field] = most_common[1] / n_samples

    return final_result, confidences
```

**Status:** Simple to implement, adds API cost, ~1 day of work

### Hard Limitations (Future Research Directions)

These require significant research effort and are beyond the scope of the current project:

#### 1. Real-Time Streaming Processing

**Challenge:** Current pipeline processes complete videos. Streaming requires frame-by-frame decisions without future context.

**Why It's Hard:**

- Cannot use scene detection (needs full video)
- Cannot cluster without all frames
- Must make instant keep/discard decisions
- Requires online learning algorithms

**Research Directions:**

- Reinforcement learning for frame selection policy
- Recurrent models that maintain state
- Predictive coding to anticipate scene changes

**Estimated Effort:** 3-6 months research project

#### 2. Learning Optimal Thresholds

**Challenge:** Current thresholds (pHash distance=8, CLIP similarity=0.90) are hand-tuned. Optimal thresholds vary by ad type, video quality, and downstream task.

**Why It's Hard:**

- Requires large annotated dataset of "ideal" keyframes
- Thresholds interact non-linearly
- Different downstream tasks need different thresholds
- Meta-learning or AutoML approaches needed

**Research Directions:**

- Bayesian optimization over threshold space
- Meta-learning across ad categories
- Reinforcement learning with extraction quality as reward

**Estimated Effort:** 2-4 months research project

#### 3. Cross-Video Transfer Learning

**Challenge:** Learning from one set of videos to improve processing of new, unseen videos.

**Why It's Hard:**

- Ads vary dramatically in style, pacing, content
- Domain shift between training and test videos
- Few-shot adaptation needed for new ad categories
- Need to learn "what makes a good keyframe" abstractly

**Research Directions:**

- Domain adaptation techniques
- Self-supervised pre-training on unlabeled ads
- Prototype-based learning for ad categories

**Estimated Effort:** 6+ months research project

#### 4. Causal Understanding of Ad Effectiveness

**Challenge:** Understanding not just what's in an ad, but why it works (or doesn't).

**Why It's Hard:**

- Requires linking visual content to behavioral outcomes (clicks, purchases)
- Confounding factors (audience, placement, timing)
- Causal inference from observational data
- Need access to performance metrics (proprietary data)

**Research Directions:**

- Causal discovery from ad A/B test data
- Counterfactual reasoning with generative models
- Integration with marketing attribution models

**Estimated Effort:** Full PhD thesis territory

#### 5. Temporal Grounding with Precise Localization

**Challenge:** Precisely locating when specific events occur (e.g., "the exact frame where the product is first shown").

**Why It's Hard:**

- Requires frame-level annotations (expensive)
- Ambiguous boundaries (gradual transitions)
- Long-tail of event types
- Temporal reasoning over variable-length sequences

**Research Directions:**

- Video temporal grounding models (e.g., Moment-DETR)
- Dense video captioning with timestamps
- Weakly-supervised temporal localization

**Estimated Effort:** 3-6 months research project

---

## Difficulty Summary

| Category           | Items                                                                                         | Effort          |
| ------------------ | --------------------------------------------------------------------------------------------- | --------------- |
| ✅ **Implemented** | Batch processing, temporal reasoning, adaptive schema                                         | Done            |
| 🟡 **Medium**      | Multilingual, audio-visual fusion, confidence scores                                          | 1-2 weeks each  |
| 🔴 **Hard**        | Streaming, learned thresholds, cross-video transfer, causal understanding, temporal grounding | Months to years |

The current pipeline addresses the **easy** items and is scoped appropriately for a publishable paper. The **medium** items are good follow-up work or paper extensions. The **hard** items represent future research directions that could each be separate papers.

---

## References

### Key Papers

1. Hussain, Z., et al. "Automatic Understanding of Image and Video Advertisements." CVPR 2017.
2. Tan, K., et al. "Large Model based Sequential Keyframe Extraction for Video Summarization." CMLDS 2024.
3. Hu, W., et al. "M-LLM Based Video Frame Selection for Efficient Video Understanding." CVPR 2025.
4. TriPSS: "A Tri-Modal Keyframe Extraction Framework." ACM MM Workshop 2025.
5. EVS: "Efficient Video Sampling: Pruning Temporally Redundant Tokens." arXiv 2025.

### Tools & Libraries

- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [TransNetV2](https://github.com/soCzech/TransNetV2)
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [ImageHash](https://github.com/JohannesBuchner/imagehash)

---

## Contact & Contribution

**Author:** Abdul Basit Tonmoy  
**Email:** abdulbasittonmoy@gmail.com  
**GitHub:** github.com/abtonmoy

For questions, issues, or contributions, please open an issue on the GitHub repository.
