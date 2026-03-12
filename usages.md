# Video Analysis Pipeline: Usage Guide

This document outlines the terminal commands to run the Ad Video Pipeline, Benchmarking Suite, and other tools. The project uses `uv` for environment management, so it is recommended to prefix commands with `uv run` if you are executing from outside the activated environment.

## 1. Main Pipeline Execution (`main.py`)

The main entry point for batch processing video advertisements. It supports parallel processing, incremental saving, and resume capability.

### Basic Workflow
Process all videos in the default directory (`data/hussain_videos`):
```bash
python main.py
```

### Specify Input Directory
Process videos from a specific directory:
```bash
python main.py --batch data/ads/ads/videos
```

### Performance & Parallelism
Adjust the number of parallel workers (default is 4):
```bash
python main.py --workers 8
```

### Pipeline Modes
Run only the frame selection and audio processing (Stages 1-6), skipping the LLM extraction (Stage 7):
```bash
python main.py --skip-extraction
```

### Utility Commands
Reset all previous progress and start from scratch:
```bash
python main.py --reset
```

Enable verbose debugging logs:
```bash
python main.py --verbose
```

---

## 2. Benchmarking Suite (`experiments/run_benchmark.py`)

The benchmarking suite allows you to compare different frame selection heuristics (e.g., HIB vs. Static, Uniform, K-Means) head-to-head. It uses the configuration defined in `config/benchmark.yaml`.

### Basic Benchmark Run
Run all baselines configured in `benchmark.yaml`:
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --output_dir results/benchmark
```

### Specific Baselines
Run only specific algorithms (e.g., testing the new HIB approach against the legacy static one):
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --methods static_pipeline hib_pipeline uniform_1fps
```

### Costly vs. Free Extraction Modes
Only calculate frame-selection mathematical metrics (No LLM API calls, completely free):
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --selection_only
```

Run only bare extraction (1 LLM call per baseline):
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --bare_only
```

Run full Stage 7 extraction (includes audio/temporal context):
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --full_only
```

### Hardware Flags
Skip methods that require a GPU (like `clip_only`, `kmeans`, `hib_pipeline`, `static_pipeline`):
```bash
uv run python -m experiments.run_benchmark --video_dir data/ads/ads/videos --pipeline_results results/analysis.json --skip_gpu
```

---

## 3. Comparing Results (`compare/compare_results.py`)

Generate a comparison report from JSON files produced by the benchmarking suite.

```bash
python compare/compare_results.py --dir results/benchmark
```

This will output a comparative analysis of processing times, frame reduction rates, and side-by-side methodology differences.

---

## 4. Testing

Run the full testing suite:
```bash
pytest tests/ -v
```

Test specific improvements (e.g., the HIB Frame Budget):
```bash
pytest tests/test_improvements.py -k "test_frame_budget" -v
```
