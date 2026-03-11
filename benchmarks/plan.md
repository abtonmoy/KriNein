# Benchmark Integration Plan

## Task

Integrate a multi-baseline benchmarking suite into the existing video advertisement analysis pipeline. The benchmark compares 7 frame-selection baselines against the existing cascaded deduplication pipeline, using the same LLM extraction engine (Stage 7) for apples-to-apples comparison.

---

## Phase 1: Research (Codebase Understanding)

### 1.1 Existing Pipeline Architecture

The pipeline is an 8-stage system in `src/pipeline.py` → `AdVideoPipeline`:

```
Stage 1: VideoLoader.load() → metadata, audio_path
Stage 2: SceneDetector.detect_scenes() → scene_boundaries
Stage 3: CandidateFrameExtractor.extract_candidates() → candidate frames
Stage 4: HierarchicalDeduplicator.deduplicate() → deduped_frames, embeddings, stats
Stage 5: AudioExtractor.extract_full_context() → audio_context
Stage 6: Selector.select() → selected_candidates
Stage 7: AdExtractor.extract() → structured JSON
```

Stages 2-4 (frames) and Stage 5 (audio) run in parallel via `ThreadPoolExecutor`.

### 1.2 Key Interfaces We Must Match

**Frame format consumed by Stage 7:**

```python
# AdExtractor.extract() signature (src/extraction/llm_client.py:L180)
def extract(
    self,
    frames: List[Tuple[float, np.ndarray]],  # (timestamp, BGR frame)
    video_duration: float,
    audio_context: Optional[Dict] = None
) -> Dict[str, Any]
```

**AdExtractor constructor controls (src/extraction/llm_client.py:L140-L165):**

```python
AdExtractor(
    provider, model, max_tokens, temperature,
    schema_mode="adaptive"|"fixed"|"flexible",  # "fixed" skips type detection pass
    temporal_context=True|False,                  # master switch
    include_timestamps=True|False,
    include_time_deltas=True|False,
    include_position_labels=True|False,
    include_narrative_instructions=True|False,
)
```

Setting `temporal_context=False` + all `include_*=False` strips all temporal formatting from `build_temporal_prompt()`. Setting `schema_mode="fixed"` skips the type-detection LLM call. Passing `audio_context=None` removes audio section from prompt.

**Factory function (src/extraction/llm_client.py:L240):**

```python
create_extractor(config: Dict) → AdExtractor  # reads from config["extraction"]
```

**CLIP embeddings (src/deduplication/clip_embed.py):**

```python
CLIPDeduplicator(model_name="ViT-B-32", pretrained="openai", threshold=0.90, device="auto")
    .compute_signatures_batch(frames: List[np.ndarray]) → np.ndarray  # (N, 512)
    .deduplicate(frames: List[Tuple[float, np.ndarray]]) → (kept_frames, kept_embeddings)
```

**VideoLoader (src/ingestion/video_loader.py):**

```python
VideoLoader(max_resolution=720, extract_audio=True)
    .load(video_path) → (metadata, audio_path)
    .get_frame_iterator(video_path, interval_ms=100) → context manager yielding (timestamp, frame)
```

**AudioExtractor (src/ingestion/audio_extractor.py):**

```python
AudioExtractor()
    .extract_full_context(audio_path, transcribe=True, model_size="base") → Dict
```

**Config loader (src/utils/config.py):**

```python
load_config(config_path, overrides=None) → Dict
```

### 1.3 Existing Pipeline Results Format

From `src/parallel_pipeline.py:_result_to_dict()` and pipeline README, results are structured as:

```json
{
  "results": [
    {
      "status": "success",
      "video_path": "...",
      "video_name": "...",
      "pipeline_stats": {
        "total_frames_sampled": int,
        "final_frame_count": int,     // ← used as k for random sampling
        "reduction_rate": float
      },
      "extraction": { ... }           // ← the Stage 7 output
    }
  ]
}
```

### 1.4 Dependencies Already Available

| Need                             | Available In                                           |
| -------------------------------- | ------------------------------------------------------ |
| Video decoding + metadata        | `src/ingestion/video_loader.py` → `VideoLoader`        |
| Audio extraction + transcription | `src/ingestion/audio_extractor.py` → `AudioExtractor`  |
| CLIP embeddings (ViT-B/32)       | `src/deduplication/clip_embed.py` → `CLIPDeduplicator` |
| LLM extraction (multi-provider)  | `src/extraction/llm_client.py` → `AdExtractor`         |
| Prompt building                  | `src/extraction/prompts.py`                            |
| Schema definitions               | `src/extraction/schema.py`                             |
| Config loading                   | `src/utils/config.py` → `load_config`                  |

### 1.5 What Does NOT Exist Yet

- Baseline frame-selection methods (uniform, random, histogram, ORB, optical flow, CLIP-only, K-means)
- Extraction wrapper for bare vs. full extraction modes
- Info density metric computation
- Benchmark runner orchestrator
- Benchmark config file
- CLI entry points

---

## Phase 2: Plan

### 2.1 File Structure

```
benchmarks/                          # NEW — top-level, outside src/
├── __init__.py
├── base.py                          # BaselineMethod ABC
├── methods/
│   ├── __init__.py
│   ├── uniform.py                   # Test 1: Uniform 1 FPS
│   ├── random_sampling.py           # Test 2: Random (k from pipeline results)
│   ├── histogram.py                 # Test 3: HSV histogram correlation
│   ├── orb.py                       # Test 4: ORB feature matching
│   ├── optical_flow.py              # Test 5: Farneback motion peaks
│   ├── clip_dedup.py                # Test 6: CLIP-only sequential dedup
│   └── kmeans.py                    # Test 7: K-means clustering
├── extraction_wrapper.py            # Bare + Full AdExtractor wrapper
├── metrics.py                       # Info density, VLM cost, comparison
└── runner.py                        # BenchmarkRunner orchestrator

config/
└── benchmark.yaml                   # NEW — benchmark-specific config

experiments/
├── run_benchmark.py                 # NEW — main CLI
└── run_30fps.py                     # NEW — separate expensive 30 FPS script
```

**No existing files are modified.**

### 2.2 Implementation Phases

---

#### Phase 2.2.1: `benchmarks/base.py` — Abstract Base

**Purpose:** Define the interface all baselines implement.

```python
from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np

class BaselineMethod(ABC):
    """All baselines must return frames in AdExtractor's expected format."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier: 'uniform_1fps', 'histogram', etc."""
        ...

    @property
    def requires_gpu(self) -> bool:
        return False

    @abstractmethod
    def select_frames(
        self,
        video_path: str,
        **kwargs  # target_k, clip_embeddings, all_frames, etc.
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Returns list of (timestamp_seconds, bgr_frame) tuples.
        This is EXACTLY the format AdExtractor.extract() expects.
        """
        ...
```

**Key design decision:** `kwargs` allows the runner to pass shared resources (CLIP embeddings, pre-decoded frames, target frame count) without bloating the base interface.

**Verification:** Import and confirm ABC cannot be instantiated directly.

---

#### Phase 2.2.2: `benchmarks/methods/` — 7 Baseline Implementations

Each file implements one `BaselineMethod` subclass. All use OpenCV for video decoding (consistent with the existing pipeline's approach).

**Shared video decoding pattern** (used by Tests 1-5):

```python
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# ... iterate, resize to max 720p (matching pipeline's ingestion.max_resolution)
cap.release()
```

**Test 1: `uniform.py` — UniformSampling**

- Decode every `round(fps / target_fps)`-th frame (default target_fps=1.0)
- Return `(frame_index / fps, frame)` tuples
- ~15 lines of core logic

**Test 2: `random_sampling.py` — RandomSampling**

- Accept `target_k` via kwargs (from pipeline results' `final_frame_count`)
- Decode all frames at 100ms interval (matching pipeline's `change_detection.min_interval_ms`)
- `random.sample(all_frames, k)`, sorted by timestamp
- Fallback if no `target_k`: use `duration // 1` (1 FPS equivalent)

**Test 3: `histogram.py` — HistogramDedup**

- Sequential scan, convert each frame to HSV
- `cv2.calcHist([hsv], [0,1,2], None, [16,16,16], [0,180,0,256,0,256])`
- `cv2.compareHist(prev, curr, cv2.HISTCMP_CORREL)` — keep if correlation < 0.95
- Compare against last KEPT frame (not just previous frame)

**Test 4: `orb.py` — ORBDedup**

- `cv2.ORB_create(nfeatures=500)` + `cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)`
- Keep frame if fewer than 40 good matches (distance < 50) vs. last kept frame
- Handle `des is None` (featureless frames) by auto-keeping

**Test 5: `optical_flow.py` — OpticalFlowPeaks**

- Two-pass: first compute Farneback flow magnitudes for all frames
- `cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)`
- Select frames where `mean(magnitude) >= np.percentile(all_magnitudes, 85)`
- Always include first and last frame

**Test 6: `clip_dedup.py` — CLIPOnlyDedup**

- Accepts pre-computed `clip_embeddings` and `all_frames` via kwargs
- Sequential dedup: keep frame if cosine similarity < 0.92 vs. last kept
- Uses existing `CLIPDeduplicator` from `src/deduplication/clip_embed.py` for embeddings
- `requires_gpu = True`

**Test 7: `kmeans.py` — KMeansClustering**

- Accepts pre-computed `clip_embeddings` and `all_frames` via kwargs
- `sklearn.cluster.KMeans(n_clusters=k)` where k = adaptive based on duration
  - k = `max(5, min(20, int(duration / 3)))` — roughly 1 cluster per 3 seconds
- Select frame nearest each centroid: `argmin ||e_i - center_j||`
- `requires_gpu = True` (for CLIP embedding computation)

**`methods/__init__.py`** — Registry:

```python
from .uniform import UniformSampling
from .random_sampling import RandomSampling
from .histogram import HistogramDedup
from .orb import ORBDedup
from .optical_flow import OpticalFlowPeaks
from .clip_dedup import CLIPOnlyDedup
from .kmeans import KMeansClustering

ALL_METHODS = {
    "uniform_1fps": UniformSampling,
    "random": RandomSampling,
    "histogram": HistogramDedup,
    "orb": ORBDedup,
    "optical_flow": OpticalFlowPeaks,
    "clip_only": CLIPOnlyDedup,
    "kmeans": KMeansClustering,
}
```

**Verification:** For each method, run on a single short test video (~10s), confirm it returns `List[Tuple[float, np.ndarray]]` with valid timestamps and BGR frames.

---

#### Phase 2.2.3: `benchmarks/extraction_wrapper.py` — Dual-Mode Extraction

**Purpose:** Wrap the existing `AdExtractor` for Option C (bare + full extraction).

```python
class ExtractionWrapper:
    def __init__(self, config: Dict):
        ext_config = config.get("extraction", {})

        # BARE: no temporal context, no audio, fixed schema (1 LLM call)
        self.bare = AdExtractor(
            provider=ext_config.get("provider", "gemini"),
            model=ext_config.get("model", "gemini-2.0-flash-exp"),
            max_tokens=ext_config.get("max_tokens", 4000),
            temperature=0.0,
            schema_mode="fixed",
            temporal_context=False,
            include_timestamps=False,
            include_time_deltas=False,
            include_position_labels=False,
            include_narrative_instructions=False,
        )

        # FULL: your complete Stage 7 (2 LLM calls for adaptive)
        self.full = create_extractor(config)

    def extract_bare(self, frames, duration):
        """Fair comparison — same minimal prompt for all methods."""
        return self.bare.extract(frames, duration, audio_context=None)

    def extract_full(self, frames, duration, audio_context):
        """System comparison — full Stage 7 treatment."""
        return self.full.extract(frames, duration, audio_context=audio_context)
```

**Key insight:** `schema_mode="fixed"` uses the comprehensive baseline schema (brand, product, promotion, CTA, topic, sentiment, engagement) WITHOUT the type-detection first pass. This means bare extraction costs 1 LLM call vs. full's 2 calls.

**Verification:** Instantiate with benchmark config, call both methods on dummy frames, confirm valid JSON returned.

---

#### Phase 2.2.4: `benchmarks/metrics.py` — Evaluation Metrics

**Purpose:** Compute frame-selection and extraction quality metrics.

**Frame Selection Metrics:**

```python
def compute_selection_metrics(
    selected_frames: List[Tuple[float, np.ndarray]],
    total_frames: int,
    selection_latency: float,
    clip_deduplicator: CLIPDeduplicator  # reuse existing
) -> Dict:
    count = len(selected_frames)
    return {
        "selected_count": count,
        "compression_ratio": round(total_frames / count, 2) if count > 0 else float("inf"),
        "latency_s": round(selection_latency, 3),
        "info_density": compute_info_density(selected_frames, clip_deduplicator),
        "vlm_cost_usd": round((count * 765 / 1000) * 0.015, 4),
    }
```

**Info Density** (mean pairwise CLIP cosine distance — higher = more diverse):

```python
def compute_info_density(
    frames: List[Tuple[float, np.ndarray]],
    clip_dedup: CLIPDeduplicator
) -> float:
    if len(frames) < 2:
        return 0.0
    frame_arrays = [f for _, f in frames]
    embeddings = clip_dedup.compute_signatures_batch(frame_arrays)
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    return round(float(1.0 - upper_tri.mean()), 5)
```

**Extraction Comparison Metrics** (compare baseline extraction vs. pipeline reference):

```python
def compare_extractions(baseline_result: Dict, reference_result: Dict) -> Dict:
    """Compare extracted fields between baseline and reference pipeline result."""
    return {
        "brand_match": _field_match(baseline_result, reference_result, ["brand", "brand_name_text"]),
        "promo_detected": _bool_match(baseline_result, ["promotion", "promo_present"]),
        "cta_detected": _bool_match(baseline_result, ["call_to_action", "cta_present"]),
        "topic_match": _field_match(baseline_result, reference_result, ["topic", "topic_id"]),
        "effectiveness": _safe_get(baseline_result, ["engagement_metrics", "effectiveness_score"]),
    }
```

**Verification:** Run info_density on a known set of diverse vs. similar frames, confirm diverse > similar.

---

#### Phase 2.2.5: `benchmarks/runner.py` — BenchmarkRunner Orchestrator

**Purpose:** The main orchestration loop. This is the most complex file.

**Class outline:**

```python
class BenchmarkRunner:
    def __init__(self, config_path: str, pipeline_results_path: str, output_dir: str):
        self.config = load_config(config_path)
        self.pipeline_results = self._load_pipeline_results(pipeline_results_path)
        self.output_dir = Path(output_dir)

        # Shared infrastructure (initialized once)
        self.loader = VideoLoader(max_resolution=720, extract_audio=True)
        self.audio_extractor = AudioExtractor()
        self.clip_dedup = CLIPDeduplicator(model_name="ViT-B-32", device="auto")
        self.extraction_wrapper = ExtractionWrapper(self.config)

        # Method registry
        self.methods = self._init_methods()
```

**Main loop (`run` method):**

```
For each video in video_dir:
  1. loader.load(video_path) → metadata, audio_path

  2. audio_extractor.extract_full_context(audio_path) → audio_context
     (extracted ONCE, shared across all baselines for full extraction)

  3. IF any GPU methods enabled:
     - Decode all frames at 100ms interval → all_frames
     - clip_dedup.compute_signatures_batch() → clip_embeddings
     (computed ONCE, shared by clip_only + kmeans + info_density)

  4. Load pipeline reference for this video:
     - pipeline_k = pipeline_results[video_name]["pipeline_stats"]["final_frame_count"]
     - pipeline_extraction = pipeline_results[video_name]["extraction"]

  5. FOR each enabled baseline method:
     a. START timer
     b. method.select_frames(video_path,
            target_k=pipeline_k,              # for random sampling
            clip_embeddings=clip_embeddings,   # for clip_only, kmeans
            all_frames=all_frames,             # for clip_only, kmeans, random
        )
     c. STOP timer → selection_latency

     d. compute_selection_metrics(selected_frames, total_frames, latency, clip_dedup)

     e. IF not selection_only:
        - extraction_wrapper.extract_bare(frames, duration) → bare_result
        - extraction_wrapper.extract_full(frames, duration, audio_context) → full_result
        - compare_extractions(bare_result, pipeline_extraction) → bare_comparison
        - compare_extractions(full_result, pipeline_extraction) → full_comparison

     f. Store results for this (video, method) pair

  6. Write per-video results to benchmark_results.json
  7. Append rows to benchmark_results.csv

AFTER all videos:
  - Write final aggregated summary
```

**CLI flags handled by runner:**

| Flag               | Effect                                              |
| ------------------ | --------------------------------------------------- |
| `--selection_only` | Skip all LLM calls, compute only frame metrics      |
| `--bare_only`      | Run only bare extraction (1 LLM call per baseline)  |
| `--full_only`      | Run only full extraction (2 LLM calls per baseline) |
| `--methods X Y Z`  | Run only specified baselines                        |
| `--skip_gpu`       | Skip clip_only + kmeans                             |

**Verification:** Run on 1 video with `--selection_only` and 2 baselines. Confirm JSON + CSV output structure.

---

#### Phase 2.2.6: `config/benchmark.yaml`

```yaml
benchmark:
  # Which baselines to run (comment out to disable)
  methods:
    - uniform_1fps
    - random
    - histogram
    - orb
    - optical_flow
    - clip_only
    - kmeans

  # Baseline-specific thresholds
  thresholds:
    histogram_correlation: 0.95
    orb_good_matches: 40
    orb_match_distance: 50
    optical_flow_percentile: 85
    clip_cosine: 0.92 # intentionally different from pipeline's 0.90
    kmeans_seconds_per_cluster: 3

  # Shared infrastructure
  clip:
    model: "ViT-B-32"
    device: "auto"
    batch_size: 32

  # Frame decoding
  video:
    max_resolution: 720
    sample_interval_ms: 100 # for pre-decoding all frames

  # Extraction (inherits from main config, can override)
  extraction:
    provider: "gemini"
    model: "gemini-2.0-flash-exp"
    max_tokens: 4000
    temperature: 0.0

  # Output
  output:
    save_selected_frames: false
    save_per_video_json: true
```

---

#### Phase 2.2.7: `experiments/run_benchmark.py` — CLI Entry Point

```python
"""
CLI entry point for benchmarking.

Usage:
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --output_dir results/benchmark

    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --methods uniform_1fps histogram clip_only \
        --selection_only

    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --bare_only
"""
```

Argparse with:

- `--video_dir` (required)
- `--pipeline_results` (required — path to your existing results JSON)
- `--output_dir` (default: `results/benchmark`)
- `--config` (default: `config/benchmark.yaml`)
- `--methods` (optional, space-separated list)
- `--selection_only` / `--bare_only` / `--full_only` (mutually exclusive group)
- `--skip_gpu`

---

#### Phase 2.2.8: `experiments/run_30fps.py` — Separate Expensive Script

Standalone script for Uniform 30 FPS baseline. Kept separate because:

- Sends potentially hundreds of frames per video to LLM
- Extremely expensive ($4+ per video)
- Only needed as theoretical upper bound

Computes frame-selection metrics (free), optionally runs extraction with `--run_extraction` flag.

---

### 2.3 Output Artifacts

**`benchmark_results.json`** — Detailed per-video, per-method results:

```json
{
  "metadata": {
    "timestamp": "2026-02-06T...",
    "config": "config/benchmark.yaml",
    "videos_processed": 23,
    "baselines_run": ["uniform_1fps", "histogram", ...],
    "extraction_modes": ["bare", "full"]
  },
  "per_video": {
    "unicef_tap.mp4": {
      "video_metadata": { "duration": 31.96, "fps": 24.0, "total_frames": 767 },
      "audio_context_available": true,
      "pipeline_reference": {
        "final_frame_count": 20,
        "reduction_rate": 0.80,
        "extraction": { "brand": {...}, ... }
      },
      "baselines": {
        "uniform_1fps": {
          "selection": {
            "selected_count": 32,
            "compression_ratio": 23.97,
            "latency_s": 0.02,
            "info_density": 0.31,
            "vlm_cost_usd": 0.37
          },
          "bare_extraction": { ... },
          "full_extraction": { ... },
          "bare_vs_pipeline": { "brand_match": true, "promo_detected": true, ... },
          "full_vs_pipeline": { "brand_match": true, "promo_detected": true, ... }
        }
      }
    }
  }
}
```

**`benchmark_results.csv`** — One row per (video, method), for easy analysis:

```
video,method,selected_count,compression_ratio,latency_s,info_density,vlm_cost_usd,
bare_brand_match,bare_promo,bare_cta,bare_topic_id,bare_effectiveness,
full_brand_match,full_promo,full_cta,full_topic_id,full_effectiveness
```

### 2.4 Cost Estimate

| Scenario                    | LLM calls per video     | Cost per video | 23 videos |
| --------------------------- | ----------------------- | -------------- | --------- |
| `--selection_only`          | 0                       | $0             | $0        |
| `--bare_only` (7 baselines) | 7                       | ~$0.35         | ~$8       |
| `--full_only` (7 baselines) | 14 (adaptive = 2 calls) | ~$0.56         | ~$13      |
| Full Option C (bare + full) | 21                      | ~$0.91         | ~$21      |

---

## Phase 3: Implementation Order

Execute phases sequentially. After each phase, verify before proceeding.

| Step | Files                                                | Verification                                           | Context Risk                                       |
| ---- | ---------------------------------------------------- | ------------------------------------------------------ | -------------------------------------------------- |
| 3.1  | `benchmarks/__init__.py`, `benchmarks/base.py`       | Import, confirm ABC                                    | Low                                                |
| 3.2  | `benchmarks/methods/__init__.py`, all 7 method files | Run each on test video, check output format            | Medium — 7 files, but each is small (~40-80 lines) |
| 3.3  | `benchmarks/extraction_wrapper.py`                   | Instantiate, call both modes on dummy data             | Low                                                |
| 3.4  | `benchmarks/metrics.py`                              | Run info_density on diverse vs. similar frames         | Low                                                |
| 3.5  | `config/benchmark.yaml`                              | Load with `load_config`, confirm all keys              | Low                                                |
| 3.6  | `benchmarks/runner.py`                               | Run on 1 video, `--selection_only`, confirm JSON + CSV | High — largest file, most integration              |
| 3.7  | `experiments/run_benchmark.py`                       | End-to-end: 1 video, 2 baselines, bare_only            | Medium                                             |
| 3.8  | `experiments/run_30fps.py`                           | Run on 1 video, selection_only                         | Low                                                |

**Total estimated implementation: ~800-1000 lines across 16 files.**

### Compaction Note

After each implementation step, the status should be updated here. If context grows large during implementation, compact by summarizing completed phases and focusing on the current step.

```
[ ] Phase 3.1 — base.py
[ ] Phase 3.2 — 7 methods
[ ] Phase 3.3 — extraction_wrapper.py
[ ] Phase 3.4 — metrics.py
[ ] Phase 3.5 — benchmark.yaml
[ ] Phase 3.6 — runner.py
[ ] Phase 3.7 — run_benchmark.py CLI
[ ] Phase 3.8 — run_30fps.py
```
