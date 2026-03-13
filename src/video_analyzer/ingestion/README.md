# Ingestion Module

The `ingestion` module handles loading video files and extracting audio tracks.

## Overview

This module provides two main classes:

- **VideoLoader**: Validates and loads video files, extracts metadata
- **AudioExtractor**: Extracts audio from video files for transcription and analysis

## Components

### VideoLoader

Handles video file validation, loading, and metadata extraction.

```python
from video_analyzer.ingestion import VideoLoader

loader = VideoLoader(
    max_resolution=720,  # Downscale videos larger than this
    extract_audio=True   # Whether to extract audio track
)

metadata, audio_path = loader.load("path/to/video.mp4")
print(f"Duration: {metadata.duration}s")
print(f"Resolution: {metadata.width}x{metadata.height}")
print(f"FPS: {metadata.fps}")
```

**Supported formats**: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`, `.m4v`

### AudioExtractor

Extracts audio from video files using FFmpeg.

```python
from video_analyzer.ingestion import AudioExtractor

extractor = AudioExtractor(
    sample_rate=16000,  # Output sample rate in Hz
    channels=1          # Mono audio
)

audio_path = extractor.extract("path/to/video.mp4", output_dir="outputs/audio")
```

## Dependencies

- **FFmpeg**: Required for audio extraction
- **OpenCV**: Used for video metadata extraction
- **NumPy**: Frame array handling

## Configuration

The ingestion behavior can be configured via YAML:

```yaml
ingestion:
  max_resolution: 720
  extract_audio: true
  supported_formats:
    - .mp4
    - .mov
    - .avi
```

## Error Handling

The module raises specific exceptions:

- `FileNotFoundError`: Video file doesn't exist
- `ValueError`: Unsupported video format
- `RuntimeError`: FFmpeg extraction failed

## Output

- **VideoMetadata**: Named tuple with duration, resolution, FPS, codec info
- **Audio file**: WAV format, 16kHz mono by default
