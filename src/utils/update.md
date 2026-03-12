# Utils Module — Changelog

## New: Disk-Based Frame Storage (`frame_store.py`)
`FrameStore` provides JPEG-compressed disk storage for video frames. For long or high-resolution videos, keeping all candidate frames as numpy arrays in RAM can consume several GB.

- `FrameStore.save(timestamp, frame)` → saves as JPEG to temp directory
- `FrameStore.load(timestamp)` → loads on demand
- `FrameStore.cleanup()` → removes temp directory
- `LazyFrame` wrapper loads frame from disk only when `.frame` property is accessed
