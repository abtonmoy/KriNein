# Selection Module

Representative frame selection with importance scoring and temporal-aware NMS.

## Modules

### `clustering.py`
Temporal clustering and NMS (Non-Maximum Suppression) for frame candidates. Supports multiple selection methods: NMS, K-means, uniform, and hybrid.

### `representative.py`
Main frame selection pipeline combining clustering and importance scoring.

**Importance Signals:**
| Signal | Weight | Description |
|--------|--------|-------------|
| Video position | 1.5x opening, 1.4x closing | Brand reveal / CTA moments |
| Scene position | 1.4x start, 1.2x end | Transition points |
| Audio events | 1.3x peaks, 1.4x post-silence | Attention-grabbing moments |
| Key phrases | 1.5x | Promotional keyword proximity |
| Text presence | 1.3x | Frames with text overlays |
| Face presence | 1.2x | Testimonials, presenters |

**Smart Frame Budget (Intrinsic Semantic Dimensionality):**
By default, limits the total frames sent to the LLM mathematically using SVD on the visual CLIP embeddings:
- Computes how many principal components are required to explain 90% of visual variance (ISD).
- Ceiling floats automatically: `Base Budget + (ISD * 1.5)`.
- highly redundant videos are clamped heavily to save tokens, chaotic videos dynamically expand to preserve narrative entropy.
- Can be disabled to fallback to a strict static `global_max_frames` cap.

```python
selector = create_selector(config)
selected = selector.select(
    frames=candidate_frames,
    embeddings=clip_embeddings,
    scene_boundaries=scenes,
    video_duration=30.0,
    audio_events=audio_context,
)
```
