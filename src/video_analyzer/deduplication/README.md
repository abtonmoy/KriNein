# Deduplication Module

The `deduplication` module provides multiple strategies for removing duplicate or near-duplicate frames from video processing pipelines.

## Overview

Video frames often contain redundant content - static scenes, slow pans, or repeated shots. This module implements a hierarchical deduplication cascade that progresses from fast perceptual hashing to slower but more accurate semantic analysis.

## Deduplication Methods

### Hierarchical (Recommended)

A three-tier cascade that combines multiple methods for optimal speed and accuracy:

```python
from video_analyzer.deduplication import HierarchicalDeduplicator

dedup = HierarchicalDeduplicator(
    phash_threshold=8,      # Hamming distance threshold
    ssim_threshold=0.92,    # Structural similarity threshold
    clip_threshold=0.90     # Cosine similarity threshold
)

unique_frames = dedup.deduplicate(frames)
```

**Cascade stages:**
1. **pHash** (Perceptual Hash): Fast frequency-domain comparison (~2ms/frame)
2. **SSIM** (Structural Similarity): Structural comparison (~50ms/frame)
3. **CLIP** (Semantic Embedding): Semantic similarity via deep learning

### Individual Methods

#### PHashDeduplicator

Perceptual hash using DCT (Discrete Cosine Transform).

```python
from video_analyzer.deduplication import PHashDeduplicator

dedup = PHashDeduplicator(threshold=8)  # Lower = stricter
```

**Best for**: Detecting scaling artifacts, compression artifacts

#### DHashDeduplicator

Difference hash using gradient comparison.

```python
from video_analyzer.deduplication import DHashDeduplicator

dedup = DHashDeduplicator(threshold=8)
```

**Best for**: Brightness and contrast changes

#### WHashDeduplicator

Wavelet hash for noise-robust comparison.

```python
from video_analyzer.deduplication import WHashDeduplicator

dedup = WHashDeduplicator(threshold=8)
```

**Best for**: Small perturbations, noise

#### SSIMDeduplicator

Structural Similarity Index comparison.

```python
from video_analyzer.deduplication import SSIMDeduplicator

dedup = SSIMDeduplicator(threshold=0.92)
```

**Best for**: Structural changes, object positioning

#### LPIPSDeduplicator

Learned Perceptual Image Patch Similarity using deep neural networks.

```python
from video_analyzer.deduplication import LPIPSDeduplicator

dedup = LPIPSDeduplicator(threshold=0.5)
```

**Best for**: Perceptual similarity (GPU recommended)

#### CLIPDeduplicator

Semantic deduplication using CLIP embeddings.

```python
from video_analyzer.deduplication import CLIPDeduplicator

dedup = CLIPDeduplicator(threshold=0.90)
```

**Best for**: Semantic similarity detection

## Hash Voting System

For robust detection, use the multi-hash voting system:

```python
from video_analyzer.deduplication import HashVotingDeduplicator

dedup = HashVotingDeduplicator(
    phash_threshold=8,
    dhash_threshold=8,
    whash_threshold=8,
    min_votes=2  # At least 2 of 3 hash types must agree
)
```

## Configuration

```yaml
deduplication:
  method: hierarchical
  threshold: 0.95
  hash_voting:
    min_votes: 2
```

## Performance Comparison

| Method      | Speed      | Accuracy | GPU Required |
|-------------|------------|----------|--------------|
| pHash       | Very Fast  | Medium   | No           |
| dHash       | Very Fast  | Medium   | No           |
| wHash       | Fast       | Medium   | No           |
| SSIM        | Medium     | High     | No           |
| LPIPS       | Slow       | Very High| Yes (recommended) |
| CLIP        | Slow       | Very High| Yes (recommended) |
| Hierarchical| Balanced   | Very High| Optional     |

## Dependencies

- **ImageHash**: Perceptual hashing algorithms
- **scikit-image**: SSIM implementation
- **PyTorch**: LPIPS and CLIP models
- **OpenCV**: Image preprocessing
- **CLIP**: Semantic embeddings (OpenAI)
