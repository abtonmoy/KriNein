#!/usr/bin/env python3
"""
Command-line interface for KriNein.

Usage:
    python -m video_analyzer.cli --video <path> [options]
    python -m video_analyzer.cli --batch --input-dir <dir> [options]
"""

import argparse
import json
import sys
from pathlib import Path

from video_analyzer import AdVideoPipeline
from video_analyzer.utils.logging import setup_logging


def parse_args():
    parser = argparse.ArgumentParser(
        description="KriNein - Extract key frames and content from videos"
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video", "-v",
        type=str,
        help="Path to a single video file"
    )
    input_group.add_argument(
        "--batch", "-b",
        action="store_true",
        help="Process all videos in a directory"
    )

    # Directory for batch processing
    parser.add_argument(
        "--input-dir", "-i",
        type=str,
        help="Directory containing videos (required for --batch)"
    )

    # Output options
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )

    # Configuration options
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (YAML)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=10,
        help="Maximum number of frames to select (default: 10)"
    )
    parser.add_argument(
        "--dedup-method",
        type=str,
        choices=["hierarchical", "hash", "ssim", "clip"],
        default="hierarchical",
        help="Deduplication method (default: hierarchical)"
    )
    parser.add_argument(
        "--dedup-threshold",
        type=float,
        default=0.95,
        help="Deduplication threshold (default: 0.95)"
    )

    # LLM options
    parser.add_argument(
        "--llm-provider",
        type=str,
        choices=["anthropic", "openai", "google"],
        default="anthropic",
        help="LLM provider for content extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-sonnet-4-6",
        help="LLM model name"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip LLM content extraction (frames only)"
    )

    # Logging
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file"
    )

    return parser.parse_args()


def process_single(pipeline, video_path, output_dir, skip_extraction):
    """Process a single video file."""
    print(f"Processing: {video_path}")

    try:
        result = pipeline.process_video(
            str(video_path),
            run_extraction=not skip_extraction
        )

        # Save result
        video_name = video_path.stem
        output_file = Path(output_dir) / f"{video_name}_result.json"

        result.save(str(output_file))
        print(f"  Saved: {output_file}")
        print(f"  Selected {len(result.frames)} frames")

        return True

    except Exception as e:
        print(f"  Error: {e}", file=sys.stderr)
        return False


def main():
    args = parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(level=log_level, log_file=args.log_file)

    # Validate batch mode
    if args.batch and not args.input_dir:
        print("Error: --input-dir is required for --batch mode", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build configuration
    config = {
        "deduplication": {
            "method": args.dedup_method,
            "threshold": args.dedup_threshold,
        },
        "selection": {
            "max_frames": args.max_frames,
        },
        "extraction": {
            "llm_provider": args.llm_provider,
            "model": args.model,
        }
    }

    # Load external config if provided
    if args.config:
        import yaml
        with open(args.config, "r") as f:
            ext_config = yaml.safe_load(f)
            # Deep merge
            for key, value in ext_config.items():
                if isinstance(value, dict) and key in config:
                    config[key].update(value)
                else:
                    config[key] = value

    # Initialize pipeline
    pipeline = AdVideoPipeline(config=config)

    # Process videos
    if args.video:
        # Single video mode
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video not found: {video_path}", file=sys.stderr)
            sys.exit(1)

        success = process_single(pipeline, video_path, output_dir, args.skip_extraction)
        sys.exit(0 if success else 1)

    elif args.batch:
        # Batch mode
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Directory not found: {input_dir}", file=sys.stderr)
            sys.exit(1)

        video_files = list(input_dir.glob("*.mp4")) + \
                      list(input_dir.glob("*.avi")) + \
                      list(input_dir.glob("*.mov")) + \
                      list(input_dir.glob("*.mkv"))

        if not video_files:
            print(f"No video files found in {input_dir}")
            sys.exit(0)

        print(f"Found {len(video_files)} videos")

        results = {"success": [], "failed": []}
        for video_path in video_files:
            success = process_single(pipeline, video_path, output_dir, args.skip_extraction)
            if success:
                results["success"].append(str(video_path))
            else:
                results["failed"].append(str(video_path))

        # Save summary
        summary_file = Path(output_dir) / "batch_summary.json"
        with open(summary_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nBatch complete: {len(results['success'])} succeeded, {len(results['failed'])} failed")

        if results["failed"]:
            print("Failed videos:")
            for v in results["failed"]:
                print(f"  - {v}")


if __name__ == "__main__":
    main()
