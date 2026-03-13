"""
Basic usage example for KriNein.

This example demonstrates how to analyze a single video file.
"""

from video_analyzer import AdVideoPipeline

def main():
    # Initialize the pipeline
    pipeline = AdVideoPipeline()

    # Process a single video
    video_path = "path/to/your/video.mp4"
    result = pipeline.process_video(video_path)

    # Print results
    print(f"Processed {len(result.frames)} frames")
    print(f"Extracted content: {result.extraction}")

if __name__ == "__main__":
    main()
