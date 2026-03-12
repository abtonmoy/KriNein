import argparse
import json
import logging
from pathlib import Path

def compare_results(results_dir: str):
    """
    Compare the JSON results produced by run_benchmark.py
    """
    p = Path(results_dir)
    if not p.exists():
        print(f"Directory {results_dir} not found.")
        return
        
    result_files = list(p.glob("benchmark_*_results.json"))
    if len(result_files) < 2:
        print("Need at least 2 benchmark result files to compare.")
        return
        
    print(f"Found {len(result_files)} result files. Loading...")
    
    datasets = {}
    for f in result_files:
        with open(f, "r") as json_file:
            data = json.load(json_file)
            method = data.get("method", f.stem)
            datasets[method] = data
            
    # Assuming we want to compare 'static' vs 'hib'
    methods = list(datasets.keys())
    
    print("\n" + "="*70)
    print(f"{'Benchmarking Comparison Report':^70}")
    print("="*70)
    
    for method, data in datasets.items():
        videos = data.get("videos", [])
        if not videos:
            continue
            
        total_videos = len(videos)
        avg_time = sum(v["processing_time_s"] for v in videos) / total_videos
        
        # Calculate Frame Reductions
        total_original = sum(v["original_frames"] for v in videos)
        total_sampled = sum(v["total_frames_sampled"] for v in videos)
        total_final = sum(v["final_frame_count"] for v in videos)
        
        avg_final = total_final / total_videos
        max_final = max(v["final_frame_count"] for v in videos)
        min_final = min(v["final_frame_count"] for v in videos)
        
        compression_ratio = total_original / total_final if total_final > 0 else 0
        
        # Print Summary
        print(f"\nMethod: {method.upper()}")
        print(f"-> Total Videos Processed: {total_videos}")
        print(f"-> Avg Processing Time:    {avg_time:.2f}s per video")
        print(f"-> Frame Compression:      {compression_ratio:.1f}x (from {total_original} to {total_final})")
        print(f"-> Final Frames per Video: {avg_final:.1f} avg [Min: {min_final}, Max: {max_final}]")
        
    # Per-video head-to-head
    if len(methods) >= 2:
        m1, m2 = methods[0], methods[1]
        print(f"\n{'-'*70}")
        print(f"{'Head-to-Head Frame Count Comparison ('+m1+' vs '+m2+')':^70}")
        print(f"{'-'*70}")
        
        v1_dict = {v["video_file"]: v["final_frame_count"] for v in datasets[m1].get("videos", [])}
        v2_dict = {v["video_file"]: v["final_frame_count"] for v in datasets[m2].get("videos", [])}
        
        common_videos = set(v1_dict.keys()).intersection(set(v2_dict.keys()))
        
        print(f"{'Video File':<30} | {m1:>10} | {m2:>10} | Delta")
        print("-" * 70)
        
        for vid in sorted(common_videos):
            c1 = v1_dict[vid]
            c2 = v2_dict[vid]
            delta = c2 - c1
            delta_str = f"+{delta}" if delta > 0 else str(delta)
            if len(vid) > 28:
                display_vid = vid[:25] + "..."
            else:
                display_vid = vid
            print(f"{display_vid:<30} | {c1:10} | {c2:10} | {delta_str}")
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Benchmark Results")
    parser.add_argument("--dir", type=str, default="benchmark/results", help="Directory containing JSON results")
    args = parser.parse_args()
    
    compare_results(args.dir)
