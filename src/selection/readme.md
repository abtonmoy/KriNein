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

**Smart Frame Budget:**
Limits total frames sent to LLM based on video duration with diminishing returns:
- 2s video → 5 frames (minimum)
- 40s video → 10 frames
- 200s video → 25 frames (capped at `global_max_frames`)

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
