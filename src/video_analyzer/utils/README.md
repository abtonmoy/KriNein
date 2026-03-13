# Utils Module

The `utils` module provides shared utilities, configuration management, logging, metrics, and helper functions used across the video analysis pipeline.

## Overview

This module contains foundational utilities that support all other modules in the pipeline.

## Components

### Config (config.py)

Configuration loading and management.

```python
from video_analyzer.utils import load_config, deep_merge, get_device

# Load YAML configuration
config = load_config("config.yaml")

# Merge with overrides
config = deep_merge(config, {"scene_detection": {"threshold": 30.0}})

# Get best available device
device = get_device(preference="auto")  # "cuda" if available, else "cpu"
```

**Configuration structure:**
```yaml
ingestion:
  max_resolution: 720
  extract_audio: true

scene_detection:
  threshold: 27.0
  min_scene_length: 0.5

deduplication:
  method: hierarchical
  threshold: 0.95

selection:
  max_frames: 10

extraction:
  llm_provider: anthropic
  model: claude-sonnet-4-6
```

### Logging (logging.py)

Logging setup and configuration.

```python
from video_analyzer.utils import setup_logging

setup_logging(
    level="INFO",
    log_file="pipeline.log"
)
```

### Metrics (metrics.py)

Pipeline metrics and result tracking.

```python
from video_analyzer.utils.metrics import (
    PipelineResult,
    FrameInfo,
    SceneInfo
)

result = PipelineResult(
    frames=selected_frames,
    extraction=output,
    processing_time=45.2,
    frame_reduction=0.85
)

# Save results
result.save("output.json")
```

### Frame Store (frame_store.py)

Disk-backed frame storage to prevent OOM errors.

```python
from video_analyzer.utils import FrameStore, LazyFrame

store = FrameStore(base_dir="/tmp/frames")

# Save frame to disk
store.save(timestamp=2.5, frame=np_array)

# Load on demand
frame = store.load(timestamp=2.5)

# Cleanup when done
store.cleanup()
```

### Video Utils (video_utils.py)

Video processing utilities.

```python
from video_analyzer.utils import (
    get_video_metadata,
    VideoFrameIterator,
    VideoMetadata
)

# Get metadata without loading full video
metadata = get_video_metadata("video.mp4")
print(f"{metadata.duration}s, {metadata.fps}fps")

# Iterate frames efficiently
for frame_num, frame, timestamp in VideoFrameIterator("video.mp4"):
    process(frame)
```

### Visualization (visualization.py)

Frame and result visualization utilities.

```python
from video_analyzer.utils import visualize_frames, plot_timeline

# Create contact sheet
visualize_frames(frames, output="contact_sheet.jpg")

# Plot scene timeline
plot_timeline(scene_boundaries, selected_frames, output="timeline.png")
```

## Configuration

```yaml
logging:
  level: INFO
  log_file: pipeline.log

utils:
  frame_store_dir: /tmp/frames
  cache_enabled: true
```

## Dependencies

- **PyYAML**: Configuration parsing
- **NumPy**: Array operations
- **Pillow**: Image handling
- **OpenCV**: Video utilities
- **Matplotlib**: Visualization
- **Torch**: Device detection

## Frame Store Details

The `FrameStore` class provides:
- Automatic disk spillover when memory is constrained
- Lazy loading with `LazyFrame` wrapper
- Automatic cleanup on process exit
- Configurable base directory

```python
# High-volume processing
store = FrameStore(max_memory_frames=100)  # Keep 100 in RAM, rest on disk

for frame in many_frames:
    store.save(frame.timestamp, frame.array)

# Later access loads from disk automatically
frame = store.load(target_timestamp)
```

## Output Format

**PipelineResult** fields:
- `frames`: List of selected frame timestamps
- `extraction`: LLM output JSON
- `metadata`: Processing statistics
- `processing_time`: Total time in seconds
- `frame_reduction`: Reduction ratio (0.0-1.0)

## Error Handling

All utilities include comprehensive error handling:
- Configuration validation
- Graceful fallbacks
- Detailed error messages
- Logging of failures
```

