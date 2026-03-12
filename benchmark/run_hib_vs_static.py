import subprocess
import sys
from pathlib import Path
import platform

def main():
    root_dir = Path(__file__).parent.parent
    
    # Run the benchmark
    print("==================================================")
    print("1. RUNNING BENCHMARK: HIB vs STATIC")
    print("==================================================")
    
    # The dataset defined by the user
    dataset = r"data\ads\ads\videos"
    
    # We use the built-in experiments module with selection_only (cheaper, exact metrics)
    benchmark_cmd = [
        "uv", "run", "python", "-m", "experiments.run_benchmark",
        "--video_dir", dataset,
        "--pipeline_results", "results/analysis.json",
        "--methods", "static_pipeline", "hib_pipeline",
        "--selection_only"
    ]
    
    print(f"Executing: {' '.join(benchmark_cmd)}")
    try:
        if platform.system() == "Windows":
            subprocess.run(benchmark_cmd, cwd=root_dir, check=True, shell=True)
        else:
            subprocess.run(benchmark_cmd, cwd=root_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Benchmark failed with exit code: {e.returncode}")
        sys.exit(1)
        
    # Note: The BenchmarkRunner natively outputs compare CSVs, so a separate compare logic isn't strictly necessary.
    # The output is placed automatically in results/benchmark/benchmark_results.csv
    print("\n✅ Benchmark Complete. See results/benchmark/benchmark_results.csv for HIB vs Static numerical performance.")

if __name__ == "__main__":
    main()
