```
 _   __     _        _   _      _      
| | / /    (_)      | \ | |    (_)     
| |/ / _ __ _       |  \| | ___ _ _ __ 
|    \| '__| |      | . ` |/ _ \ | '_ \
| |\  \ |  | |      | |\  |  __/ | | | |
\_| \_/_|  |_|      \_| \_/\___|_|_| |_|

     Intelligent Video Content Analysis
```

[![PyPI](https://img.shields.io/pypi/v/KriNein.svg)](https://pypi.org/project/KriNein/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/abtonmoy/KriNein/actions/workflows/tests.yml/badge.svg)](https://github.com/abtonmoy/KriNein/actions)

**KriNein** is a production-ready Python library for intelligent video content analysis and frame extraction using Vision-Language Models (VLMs). It transforms raw video into structured, actionable insights through a sophisticated multi-stage pipeline.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [CLI Usage](#cli-usage)
- [Examples](#examples)
- [Development](#development)
- [License](#license)

---

## Features

| Feature | Description |
|---------|-------------|
| **Scene Detection** | Automatic scene boundary detection using PySceneDetect with content-aware thresholds |
| **Hierarchical Deduplication** | Three-tier cascade (pHash → SSIM → CLIP) achieving 80-95% frame reduction |
| **Adaptive Frame Selection** | Content-aware sampling based on visual complexity (ISD-based budget) |
| **Temporal Clustering** | K-means clustering with scene-aware importance scoring |
| **Multimodal Audio Analysis** | Whisper transcription, keyword detection, mood classification |
| **Multi-LLM Support** | Structured extraction with Claude, GPT, or Gemini models |
| **Parallel Processing** | Batch processing with configurable concurrency |

---

## Installation

### From PyPI (Recommended)

```bash
pip install KriNein
```

### From Source

```bash
git clone https://github.com/abtonmoy/KriNein.git
cd KriNein
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### System Requirements

- **Python**: 3.11 or higher
- **FFmpeg**: Required for video processing
- **GPU**: CUDA-capable GPU (optional, recommended for faster processing)

```bash
# Install FFmpeg
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg

# Windows (using winget)
winget install FFmpeg.FFmpeg
```

---

## Quick Start

### Single Video Analysis

```python
from video_analyzer import AdVideoPipeline

# Initialize the pipeline
pipeline = AdVideoPipeline()

# Process a video file
result = pipeline.process_video("path/to/video.mp4")

# Access results
print(f"Frames selected: {len(result.frames)}")
print(f"Extraction result: {result.extraction}")
```

### Batch Processing

```python
from pathlib import Path
from video_analyzer import AdVideoPipeline

pipeline = AdVideoPipeline()

video_dir = Path("videos")
output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

for video_path in video_dir.glob("*.mp4"):
    result = pipeline.process_video(str(video_path))
    result.save(output_dir / f"{video_path.stem}_result.json")
```

### With Custom Configuration

```python
from video_analyzer import AdVideoPipeline

config = {
    "deduplication": {
        "method": "hierarchical",
        "threshold": 0.95,
    },
    "selection": {
        "max_frames": 15,
    },
    "extraction": {
        "llm_provider": "anthropic",
        "model": "claude-sonnet-4-6",
    }
}

pipeline = AdVideoPipeline(config=config)
result = pipeline.process_video("video.mp4")
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            KriNein Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────┐    │
│  │   Video      │   │    Scene     │   │      Candidate Frame         │    │
│  │   Ingestion  │──▶│   Detection  │──▶│         Extraction           │    │
│  │              │   │              │   │                              │    │
│  │  • Load MP4  │   │  • PyScene   │   │  • 50ms interval sampling    │    │
│  │  • Extract   │   │  • Threshold │   │  • Change detection          │    │
│  │    Audio     │   │    detection │   │  • OCR extraction            │    │
│  └──────────────┘   └──────────────┘   └──────────────┬───────────────┘    │
│                                                       │                      │
│                                                       ▼                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────────────────────┐    │
│  │    LLM       │   │    Frame     │   │      Hierarchical            │    │
│  │  Extraction  │◀──│  Selection   │◀──│       Deduplication          │    │
│  │              │   │              │   │                              │    │
│  │  • Claude    │   │  • K-means   │   │  • pHash (fast)              │    │
│  │  • GPT       │   │  • Temporal  │   │  • SSIM (medium)             │    │
│  │  • Gemini    │   │  • Scoring   │   │  • CLIP (semantic)           │    │
│  └──────────────┘   └──────────────┘   └──────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Processing Stages

| Stage | Method | Purpose |
|-------|--------|---------|
| **1. Ingestion** | FFmpeg | Load video, extract 16kHz mono WAV audio |
| **2. Scene Detection** | PySceneDetect | Detect scene boundaries via content change |
| **3. Candidate Extraction** | Frame sampling | Sample at 50ms intervals + change detection |
| **4. Deduplication** | pHash → SSIM → CLIP | Three-tier cascade filtering |
| **5. Audio Analysis** | Whisper | Transcription, keywords, mood |
| **6. Frame Selection** | Adaptive clustering | Density-based selection with ISD budget |
| **7. LLM Extraction** | Multi-provider | Single-pass structured data extraction |

---

## Configuration

### YAML Configuration File

```yaml
# config.yaml
scene_detection:
  threshold: 27.0          # Scene change sensitivity
  min_scene_length: 0.5    # Minimum scene duration (seconds)

deduplication:
  method: hierarchical     # hierarchical, hash, ssim, clip
  threshold: 0.95          # Similarity threshold

selection:
  max_frames: 10           # Maximum frames to select
  method: clustering       # clustering, importance

extraction:
  llm_provider: anthropic  # anthropic, openai, google
  model: claude-sonnet-4-6
  schema_fields:
    - ad_type
    - brand
    - product_category
```

### Programmatic Configuration

```python
config = {
    "scene_detection": {"threshold": 27.0},
    "deduplication": {"method": "hierarchical", "threshold": 0.95},
    "selection": {"max_frames": 10, "method": "clustering"},
    "extraction": {"llm_provider": "anthropic", "model": "claude-sonnet-4-6"},
}

pipeline = AdVideoPipeline(config=config)
```

### Environment Variables

```bash
# API Keys (required for LLM extraction)
export ANTHROPIC_API_KEY=your_key_here
export OPENAI_API_KEY=your_key_here
export GOOGLE_API_KEY=your_key_here
```

---

## API Reference

### AdVideoPipeline

```python
from video_analyzer import AdVideoPipeline

pipeline = AdVideoPipeline(
    config: dict = None,      # Optional configuration dict
    verbose: bool = False,    # Enable verbose logging
)

result = pipeline.process_video(
    video_path: str,          # Path to video file
    run_extraction: bool = True,  # Run LLM extraction
)
```

### PipelineResult

```python
# Result attributes
result.frames           # List of selected frames
result.extraction       # LLM extraction result (JSON)
result.audio_analysis   # Audio transcription and analysis
result.metrics          # Processing metrics

# Save/load results
result.save("output.json")
result.save_frames("output_dir/")
```

### ParallelVideoPipeline

```python
from video_analyzer import ParallelVideoPipeline

pipeline = ParallelVideoPipeline(
    max_workers: int = 4,   # Number of parallel workers
    config: dict = None,    # Shared configuration
)

results = pipeline.process_batch(
    video_paths: list[str],  # List of video paths
)
```

---

## CLI Usage

```bash
# Single video
krainein --video path/to/video.mp4 --output-dir results/

# Batch processing
krainein --batch --input-dir videos/ --output-dir results/

# With configuration
krainein -v video.mp4 --config config.yaml

# Custom options
krainein -v video.mp4 \
    --max-frames 15 \
    --dedup-method hierarchical \
    --llm-provider anthropic \
    --model claude-sonnet-4-6 \
    --verbose
```

### CLI Options

| Option | Short | Description | Default |
|--------|-------|-------------|---------|
| `--video` | `-v` | Path to single video file | - |
| `--batch` | `-b` | Enable batch processing mode | - |
| `--input-dir` | `-i` | Directory with videos (batch mode) | - |
| `--output-dir` | `-o` | Output directory for results | `results/` |
| `--config` | `-c` | Path to YAML config file | - |
| `--max-frames` | - | Maximum frames to select | 10 |
| `--dedup-method` | - | Deduplication method | hierarchical |
| `--llm-provider` | - | LLM provider | anthropic |
| `--model` | - | LLM model name | claude-sonnet-4-6 |
| `--skip-extraction` | - | Skip LLM extraction | - |
| `--verbose` | - | Enable verbose logging | - |

---

## Examples

The `examples/` directory contains working code:

| Example | Description |
|---------|-------------|
| `basic_usage.py` | Simple single video processing |
| `batch_processing.py` | Batch processing with error handling |
| `custom_config.py` | Advanced configuration options |

### Jupyter Notebooks

- `examples/notebooks/data_exploration.ipynb` - Data exploration
- `examples/notebooks/evaluation.ipynb` - Evaluation workflows
- `examples/notebooks/result_analysis.ipynb` - Result analysis

---

## Development

### Setting Up Development Environment

```bash
# Clone repository
git clone https://github.com/abtonmoy/KriNein.git
cd KriNein

# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=video_analyzer tests/
```

### Project Structure

```
KriNein/
├── src/video_analyzer/       # Main package
│   ├── __init__.py           # Package exports
│   ├── pipeline.py           # AdVideoPipeline
│   ├── parallel.py           # ParallelVideoPipeline
│   ├── cli.py                # CLI entry point
│   ├── ingestion/            # Video/audio loading
│   ├── detection/            # Scene/change detection
│   ├── deduplication/        # Frame deduplication
│   ├── selection/            # Frame selection
│   ├── extraction/           # LLM extraction
│   └── utils/                # Utilities
├── examples/                 # Usage examples
│   ├── basic_usage.py
│   ├── batch_processing.py
│   ├── custom_config.py
│   ├── figures/
│   └── notebooks/
├── tests/                    # Test suite
├── pyproject.toml            # Package configuration
├── LICENSE                   # MIT License
└── README.md                 # This file
```

---

## License

```
MIT License

Copyright (c) 2026 KriNein Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## Citation

```bibtex
@software{KriNein,
  author = {Abdul Basit Tonmoy},
  title = {KriNein: Intelligent Video Content Analysis Library},
  year = {2026},
  version = {1.0.0},
  url = {https://github.com/abtonmoy/KriNein},
}
```
