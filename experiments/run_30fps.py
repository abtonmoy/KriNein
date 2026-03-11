"""
Separate script for Uniform 30 FPS baseline (theoretical upper bound).

This is kept separate because it is EXTREMELY expensive:
- A 60s video at 30 FPS = 1800 frames → ~$13.50 per video in LLM costs
- Only run this if you can afford it

By default, only computes frame-selection metrics (free).
Pass --run_extraction to actually send frames to the LLM.

Usage:
    # Metrics only (free)
    python -m experiments.run_30fps \
        --video_dir data/ads \
        --output_dir results/benchmark_30fps

    # With LLM extraction ($$$$)
    python -m experiments.run_30fps \
        --video_dir data/ads \
        --pipeline_results results/analysis.json \
        --output_dir results/benchmark_30fps \
        --run_extraction \
        --max_frames 100  # safety cap
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
from benchmarks.base import get_video_info
from benchmarks.methods.uniform import UniformSampling
from benchmarks.metrics import compute_selection_metrics

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="30 FPS baseline (expensive)")
    parser.add_argument("--video_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results/benchmark_30fps")
    parser.add_argument("--pipeline_results", type=str, default=None)
    parser.add_argument("--run_extraction", action="store_true",
                        help="Actually send frames to LLM (expensive!)")
    parser.add_argument("--max_frames", type=int, default=200,
                        help="Safety cap on frames sent to LLM (default 200)")
    parser.add_argument("--config", type=str, default="config/benchmark.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".m4v"}
    videos = sorted(
        str(p) for p in Path(args.video_dir).iterdir() if p.suffix.lower() in exts
    )
    logger.info(f"Found {len(videos)} videos")

    method = UniformSampling(target_fps=30.0)
    rows = []

    for vpath in videos:
        vname = Path(vpath).name
        total_frames, fps, duration = get_video_info(vpath)
        logger.info(f"\n{vname}: {total_frames} frames, {duration:.1f}s")

        frames, latency = method.run_timed(vpath)
        metrics = compute_selection_metrics(
            frames, total_frames, latency, clip_deduplicator=None
        )

        row = {"video": vname, "method": "uniform_30fps", **metrics}

        if args.run_extraction and frames:
            if len(frames) > args.max_frames:
                logger.warning(
                    f"  Capping {len(frames)} frames to {args.max_frames} for LLM"
                )
                # Subsample evenly
                step = len(frames) / args.max_frames
                indices = [int(i * step) for i in range(args.max_frames)]
                frames = [frames[i] for i in indices]

            from src.utils.config import load_config
            from benchmarks.extraction_wrapper import ExtractionWrapper

            config = load_config(args.config) if Path(args.config).exists() else {}
            wrapper = ExtractionWrapper(config)

            logger.info(f"  Bare extraction ({len(frames)} frames)...")
            bare = wrapper.extract_bare(frames, duration)
            row["bare_extraction_success"] = "error" not in bare

            logger.info(f"  Full extraction ({len(frames)} frames)...")
            full = wrapper.extract_full(frames, duration)
            row["full_extraction_success"] = "error" not in full

        rows.append(row)
        logger.info(f"  → {metrics['selected_count']} frames, "
                     f"cost=${metrics['vlm_cost_usd']:.2f}")

    # Write results
    df = pd.DataFrame(rows)
    csv_path = out_dir / "30fps_results.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"\nResults saved to {csv_path}")
    print(df.to_string(index=False))

    # Summary
    total_cost = df["vlm_cost_usd"].sum()
    logger.info(f"\nTotal estimated VLM cost (if extracted): ${total_cost:.2f}")


if __name__ == "__main__":
    main()