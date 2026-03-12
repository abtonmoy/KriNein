# Selection Module — Changelog

## Smart Global Frame Budget
`FrameSelector` now has a `global_max_frames` parameter (default 25). After NMS, if too many frames remain, the top-scoring frames by importance are kept. Uses diminishing returns: short videos (< 20s) get at least 5 frames, long videos (> 100s) are capped.

## Integrated Visual Feature Detection
`FrameSelector.select()` now auto-detects text and face presence in candidate frames using `VisualFeatureDetector`. Frames with text overlays get a 1.3x importance boost; frames with faces get 1.2x. This is enabled by default and runs automatically when no `visual_features` dict is provided.

## Configuration
New config options in `selection:`:
```yaml
selection:
  global_max_frames: 25     # Max total frames sent to LLM (for legacy mode)
  use_hib_budget: true       # Enables ISD scaling instead of legacy mode
  use_visual_features: true  # Auto-detect text/faces for scoring
```

## Intrinsic Semantic Dimensionality (ISD)
The `global_max_frames` cap is no longer a strict absolute limit when `use_hib_budget` is enabled. Instead, the dynamic maximum cap is mathematically determined via the Singular Value Decomposition (SVD) of the visual embeddings.
- **Boring Videos:** SVD finds 1 or 2 principal components; budget is capped heavily (e.g. 5-8 frames)
- **Chaotic Videos:** SVD determines high semantic cardinality; budget dynamically expands (e.g. 30-50 frames)
This prevents the pipeline from overflowing token costs on repetitive videos, whilst allowing chaotic clips to provide full semantic context without token truncation.
