"""
Batch processing example for KriNein.

This example demonstrates how to process multiple videos efficiently.
"""

from pathlib import Path
from video_analyzer import AdVideoPipeline
from video_analyzer.utils.logging import setup_logging

def main():
    # Setup logging
    setup_logging(level="INFO")

    # Initialize the pipeline
    pipeline = AdVideoPipeline()

    # Define video directory
    video_dir = Path("path/to/videos")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Process all videos in directory
    video_files = list(video_dir.glob("*.mp4"))

    for video_path in video_files:
        print(f"Processing: {video_path}")
        try:
            result = pipeline.process_video(str(video_path))

            # Save results
            output_file = output_dir / f"{video_path.stem}_result.json"
            result.save(str(output_file))

        except Exception as e:
            print(f"Error processing {video_path}: {e}")

if __name__ == "__main__":
    main()
