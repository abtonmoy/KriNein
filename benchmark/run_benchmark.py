import os
import argparse
import json
import logging
import time
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import AdVideoPipeline, process_directory

logger = logging.getLogger(__name__)

def run_benchmark(dataset_dir: str, output_dir: str, methods: list, max_workers: int = 1):
    """
    Run pipeline on a dataset with multiple configurations.
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        logger.error(f"Dataset directory {dataset_dir} does not exist.")
        return

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    results_summary = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Running benchmark with method: {method.upper()}")
        print(f"{'='*50}")
        
        # Configure the method
        use_hib = (method.lower() == 'hib')
        
        # We skip actual LLM extraction for pure frame-budget benchmarking to save API costs
        # (Unless explicitly configured otherwise in the future).
        config_overrides = {
            "selection": {
                "use_hib_budget": use_hib,
                "global_max_frames": 25
            }
        }
        
        pipeline = AdVideoPipeline(overrides=config_overrides)
        
        # Find all videos
        extensions = [".mp4", ".mov", ".avi", ".mkv", ".webm"]
        video_paths = []
        for ext in extensions:
            video_paths.extend(dataset_path.glob(f"*{ext}"))
            video_paths.extend(dataset_path.glob(f"*{ext.upper()}"))
            
        video_paths = [str(p) for p in sorted(set(video_paths))]
        print(f"Found {len(video_paths)} videos in {dataset_dir}")
        
        if not video_paths:
            continue
            
        # Process the batch using the pipeline's built-in parallelization
        start_time = time.time()
        results = pipeline.process_batch(
            video_paths, 
            max_workers=max_workers, 
            skip_extraction=False # Skip LLM to save time/cost during selection bench
        )
        total_time = time.time() - start_time
        
        # Aggregate metrics
        method_metrics = []
        for res in results:
            if res is None:
                continue
            metrics = res.get_metrics()
            method_metrics.append({
                "video_file": Path(res.video_path).name,
                "duration": res.metadata.duration,
                "original_frames": res.metadata.frame_count,
                "total_frames_sampled": res.total_frames_sampled,
                "final_frame_count": res.final_frame_count,
                "processing_time_s": res.processing_time_s
            })
            
        # Save exact results for this method
        method_out_file = out_path / f"benchmark_{method}_results.json"
        with open(method_out_file, "w") as f:
            json.dump({
                "method": method,
                "config": config_overrides,
                "total_processing_time": total_time,
                "videos": method_metrics
            }, f, indent=2)
            
        print(f"Saved {method} benchmark results to {method_out_file}")
        results_summary[method] = method_out_file

    print("\nBenchmark complete! Run compare/compare_results.py to see differences.")
    return results_summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Video Pipeline Benchmarks")
    parser.add_argument("--dataset", type=str, default=r"data\ads\ads\videos", help="Path to video dataset")
    parser.add_argument("--output", type=str, default="benchmark/results", help="Output directory for results")
    parser.add_argument("--methods", nargs='+', default=["static", "hib"], help="Methods to benchmark ('static', 'hib')")
    parser.add_argument("--workers", type=int, default=1, help="Max parallel workers")
    
    args = parser.parse_args()
    
    # Setup basic console logging
    logging.basicConfig(level=logging.WARNING) 
    
    run_benchmark(args.dataset, args.output, args.methods, args.workers)
