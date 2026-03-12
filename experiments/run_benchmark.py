"""
CLI entry point for the benchmarking suite.

Compares 7 baseline frame-selection methods against the existing
cascaded deduplication pipeline using the same LLM extraction engine.

Usage:
    # Run all baselines, both extraction modes (Option C)
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --output_dir results/benchmark

    # Frame selection metrics only (no LLM calls — free)
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --selection_only

    # Bare extraction only (1 LLM call per baseline — cheaper)
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --bare_only

    # Full extraction only (2 LLM calls per baseline)
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --full_only

    # Specific baselines only
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --methods uniform_1fps histogram clip_only

    # Skip GPU-dependent methods
    python -m experiments.run_benchmark \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --skip_gpu
"""

import argparse
import logging
import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.config import load_config
from benchmarks.runner import BenchmarkRunner


def find_videos(video_dir: str) -> list:
    """Find all video files in directory."""
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
    vdir = Path(video_dir)
    if not vdir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    videos = sorted(
        str(p) for p in vdir.iterdir() if p.suffix.lower() in exts
    )
    if not videos:
        raise FileNotFoundError(f"No video files found in {video_dir}")
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark frame-selection baselines against pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--video_dir", type=str, required=True,
        help="Directory containing input videos",
    )
    parser.add_argument(
        "--pipeline_results", type=str, required=True,
        help="Path to existing pipeline results JSON",
    )
    parser.add_argument(
        "--output_dir", type=str, default="results/benchmark",
        help="Output directory (default: results/benchmark)",
    )
    parser.add_argument(
        "--config", type=str, default="config/benchmark.yaml",
        help="Benchmark config file (default: config/benchmark.yaml)",
    )
    parser.add_argument(
        "--methods", type=str, nargs="*", default=None,
        help="Specific baselines to run (e.g., uniform_1fps histogram clip_only)",
    )

    # Extraction mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--selection_only", action="store_true",
        help="Only compute frame-selection metrics, no LLM calls",
    )
    mode_group.add_argument(
        "--bare_only", action="store_true",
        help="Only run bare extraction (no temporal/audio context)",
    )
    mode_group.add_argument(
        "--full_only", action="store_true",
        help="Only run full Stage 7 extraction",
    )

    parser.add_argument(
        "--skip_gpu", action="store_true",
        help="Skip GPU-dependent methods (clip_only, kmeans)",
    )
    parser.add_argument(
        "--log_level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Find videos
    videos = find_videos(args.video_dir)
    logger.info(f"Found {len(videos)} videos in {args.video_dir}")

    # Load default config and merge benchmark overrides
    from src.utils.config import load_config, deep_merge
    config = {}
    default_path = Path("config/default.yaml")
    if default_path.exists():
        config = load_config(str(default_path))
        
    config_path = Path(args.config)
    if config_path.exists():
        bench_config = load_config(str(config_path))
        config = deep_merge(config, bench_config)
        
        # Manually promote benchmark extraction overrides to top level 
        # so create_extractor() uses them instead of falling back to default.yaml
        if "benchmark" in bench_config and "extraction" in bench_config["benchmark"]:
            if "extraction" not in config:
                config["extraction"] = {}
            for k, v in bench_config["benchmark"]["extraction"].items():
                config["extraction"][k] = v
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")

    # Create and run benchmark
    runner = BenchmarkRunner(
        config=config,
        pipeline_results_path=args.pipeline_results,
        output_dir=args.output_dir,
        methods=args.methods,
        skip_gpu=args.skip_gpu,
        selection_only=args.selection_only,
        bare_only=args.bare_only,
        full_only=args.full_only,
    )

    logger.info(
        f"Starting benchmark: {len(runner.methods)} methods × {len(videos)} videos"
    )
    if args.selection_only:
        logger.info("Mode: selection_only (no LLM calls)")
    elif args.bare_only:
        logger.info("Mode: bare_only (1 LLM call per baseline per video)")
    elif args.full_only:
        logger.info("Mode: full_only (2 LLM calls per baseline per video)")
    else:
        logger.info("Mode: full Option C (bare + full extraction)")

    results = runner.run(videos)

    # Summary
    n_videos = results["metadata"]["videos_processed"]
    n_calls = results["metadata"]["total_llm_calls"]
    n_methods = len(runner.methods)

    logger.info(f"\n{'='*60}")
    logger.info(f"Benchmark complete!")
    logger.info(f"  Videos processed: {n_videos}")
    logger.info(f"  Baselines run:    {n_methods}")
    logger.info(f"  Total LLM calls:  {n_calls}")
    logger.info(f"  Results:          {args.output_dir}/benchmark_results.json")
    logger.info(f"  Summary CSV:      {args.output_dir}/benchmark_results.csv")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()