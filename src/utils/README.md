# Utils Module

Shared utilities for the video analysis pipeline.

## Modules

### `config.py`
YAML configuration loading and validation. Reads `config/default.yaml` and supports overrides.

### `logging.py`
Structured logging setup with configurable verbosity levels.

### `metrics.py`
Processing metrics collection (timing, frame counts, dedup ratios).

### `video_utils.py`
Video file utilities — format detection, audio extraction via FFmpeg, metadata reading.

### `frame_store.py`
Disk-backed frame storage for memory-efficient processing.

```python
store = FrameStore(quality=95)  # JPEG compression quality

# Save frames to disk
store.save(1.5, frame_array)        # Single
store.save_batch([(0.5, f1), (1.0, f2)])  # Batch

# Load on demand
frame = store.load(1.5)             # Single
frames = store.load_batch([0.5, 1.0])     # Batch

# LazyFrame — loads only when accessed
lazy = LazyFrame(1.5, "/path/to/frame.jpg")
data = lazy.frame   # Loaded now
lazy.release()      # Release from memory

# Cleanup temp directory
store.cleanup()
```
