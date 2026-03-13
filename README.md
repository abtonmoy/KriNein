# KriNein

A clean, efficient Python library for video content analysis and frame extraction using Vision-Language Models (VLMs).

## Features

- **Scene Detection**: Automatic scene boundary detection using PySceneDetect
- **Hierarchical Deduplication**: Three-tier cascade (pHash → SSIM → CLIP) for efficient frame filtering
- **Adaptive Frame Selection**: Content-aware sampling based on visual complexity
- **Temporal Clustering**: K-means clustering to preserve semantic diversity
- **Multimodal Audio Analysis**: Speech transcription, keyword detection, mood classification
- **LLM-Based Extraction**: Structured content extraction with Claude, GPT, or Gemini

## Installation

```bash
# Using pip
pip install KriNein

# Or from source
git clone https://github.com/your-org/KriNein
cd KriNein
pip install -e .
```

### Requirements

- Python 3.11+
- FFmpeg
- CUDA-capable GPU (optional, for faster processing)

## Quick Start

### Single Video Processing

```python
from video_analyzer import AdVideoPipeline

# Initialize pipeline
pipeline = AdVideoPipeline()

# Process a video
result = pipeline.process_video("path/to/video.mp4")

# Access results
print(f"Selected {len(result.frames)} frames")
print(f"Extraction: {result.extraction}")
```

### Batch Processing

```python
from pathlib import Path
from video_analyzer import AdVideoPipeline

pipeline = AdVideoPipeline()

video_dir = Path("videos")
for video_path in video_dir.glob("*.mp4"):
    result = pipeline.process_video(str(video_path))
    result.save(f"results/{video_path.stem}.json")
```

### CLI Usage

```bash
# Process a single video
video-analyzer --video path/to/video.mp4 --output-dir results

# Batch processing
video-analyzer --batch --input-dir videos/ --output-dir results

# With custom configuration
video-analyzer -v video.mp4 --max-frames 15 --dedup-method hierarchical
```

## Configuration

Create a YAML configuration file:

```yaml
scene_detection:
  threshold: 27.0
  min_scene_length: 0.5

deduplication:
  method: hierarchical
  threshold: 0.95

selection:
  max_frames: 10
  method: clustering

extraction:
  llm_provider: anthropic
  model: claude-sonnet-4-6
```

Use with CLI:
```bash
video-analyzer -v video.mp4 --config config.yaml
```

Or programmatically:
```python
config = {
    "deduplication": {"method": "hierarchical", "threshold": 0.95},
    "selection": {"max_frames": 10},
}
pipeline = AdVideoPipeline(config=config)
```

## Pipeline Stages

1. **Video Ingestion**: Load video, extract audio (16kHz mono WAV)
2. **Scene Detection**: Detect scene boundaries using content change analysis
3. **Candidate Extraction**: Sample frames at 50ms intervals with change detection
4. **Deduplication**: Three-tier cascade filtering
   - pHash: Perceptual hashing (Hamming distance ≤ 8)
   - SSIM: Structural similarity (threshold: 0.92)
   - CLIP: Semantic embeddings (cosine similarity: 0.90)
5. **Audio Analysis**: Whisper transcription, keyword detection, mood classification
6. **Frame Selection**: Adaptive density clustering with importance scoring
7. **LLM Extraction**: Single-pass structured data extraction

## Examples

See the `examples/` directory for:
- `basic_usage.py` - Simple single video processing
- `batch_processing.py` - Batch processing with error handling
- `custom_config.py` - Advanced configuration options

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=video_analyzer tests/
```

## Project Structure

```
KriNein/
├── src/video_analyzer/    # Main package
│   ├── __init__.py
│   ├── pipeline.py        # AdVideoPipeline class
│   ├── parallel.py        # Parallel processing
│   ├── cli.py             # Command-line interface
│   ├── ingestion/         # Video/audio loading
│   ├── detection/         # Scene/change detection
│   ├── deduplication/     # Frame deduplication
│   ├── selection/         # Frame selection
│   ├── extraction/        # LLM extraction
│   └── utils/             # Utilities
├── examples/              # Usage examples
├── tests/                 # Test suite
├── benchmarks/            # Benchmark suite
└── pyproject.toml
```

## License

MIT License - see LICENSE file for details.

## Citation

```bibtex
@software{KriNein,
  title = {KriNein: Video Content Analysis Library},
  author = {KriNein Team},
  year = {2026},
  version = {1.0.0},
}
```
