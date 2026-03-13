# Selection Module

The `selection` module handles intelligent frame selection from deduplicated candidates, ensuring temporal diversity and semantic representativeness.

## Overview

After deduplication, you may still have many candidate frames. This module selects the most representative frames using clustering and importance scoring.

## Components

### TemporalClusterer

Groups frames by visual similarity and temporal position using K-means clustering.

```python
from video_analyzer.selection import TemporalClusterer

clusterer = TemporalClusterer(
    max_clusters=10,      # Target number of clusters
    temporal_weight=0.3   # Weight for temporal proximity
)

clusters = clusterer.cluster(frames, embeddings)
```

**How it works:**
1. Maps frames to CLIP embedding space
2. Applies K-means clustering
3. Selects representative closest to each centroid
4. Enforces temporal gap constraints

### NMSSelector

Non-Maximum Suppression selector with temporal-aware thresholds.

```python
from video_analyzer.selection import NMSSelector

selector = NMSSelector(
    temporal_threshold_s=0.5,   # Minimum gap between frames
    semantic_threshold=0.88,    # Similarity threshold
    importance_weight=1.0,      # Weight for importance scores
    diversity_bonus=0.1         # Bonus for diverse frames
)

selected = selector.select(candidates, max_frames=10)
```

**Features:**
- Temporal-aware threshold relaxation (allows repeated logos/CTAs)
- Importance score integration
- Scene boundary awareness
- Diversity bonus for semantic coverage

### FrameSelector

Main interface combining multiple selection strategies.

```python
from video_analyzer.selection import FrameSelector

selector = FrameSelector(
    method="clustering",      # "clustering", "nms", or "uniform"
    max_frames=10,
    min_gap_s=0.5,
    use_importance=True
)

selected_frames = selector.select(frames, embeddings, scene_boundaries)
```

**Selection methods:**
- `clustering`: K-means based semantic clustering
- `nms`: Non-maximum suppression with importance scoring
- `uniform`: Even temporal sampling (fallback)

### ImportanceScorer

Scores frames based on multiple factors.

```python
from video_analyzer.selection.representative import ImportanceScorer

scorer = ImportanceScorer(
    position_weight=1.5,      # Boost opening/closing frames
    audio_weight=1.3,         # Boost near audio events
    scene_boundary_weight=1.4 # Boost scene transitions
)

scores = scorer.score(frames, audio_events, scene_boundaries)
```

## Configuration

```yaml
selection:
  method: clustering
  max_frames: 10
  min_gap_s: 0.5
  use_importance: true

  clustering:
    temporal_weight: 0.3

  nms:
    semantic_threshold: 0.88
    temporal_aware: true
```

## Adaptive Frame Budget

The module supports adaptive frame budgets based on video content:

- **Short videos** (< 10s): 3-5 frames
- **Medium videos** (10-60s): 5-15 frames
- **Long videos** (> 60s): 10-25 frames (or ISD-based)

## ISD (Intrinsic Semantic Dimensionality)

Advanced frame budgeting using SVD on CLIP embeddings:

```python
from video_analyzer.selection import compute_isd_budget

# Compute ISD: number of components explaining 90% variance
isd = compute_isd_budget(embeddings, variance_threshold=0.90)

# Dynamic frame cap: higher ISD = more unique content
frame_budget = min(25, max(5, isd * 2))
```

## Dependencies

- **scikit-learn**: K-means clustering
- **NumPy**: Numerical operations
- **SciPy**: SVD computation

## Output Format

**Selected frames**: List of frame indices or timestamps

**FrameCandidate**: Data class with:
- `timestamp`: Frame timestamp in seconds
- `frame`: Frame array (numpy)
- `embedding`: CLIP embedding vector
- `scene_id`: Assigned scene ID
- `importance_score`: Computed importance
- `is_representative`: Whether selected as representative
