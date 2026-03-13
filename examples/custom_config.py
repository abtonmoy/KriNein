"""
Custom configuration example for KriNein.

This example demonstrates how to customize pipeline settings.
"""

from video_analyzer import AdVideoPipeline

def main():
    # Custom configuration
    config = {
        "scene_detection": {
            "threshold": 27.0,  # Scene change threshold
        },
        "deduplication": {
            "method": "hierarchical",  # or "hash", "ssim", "clip"
            "threshold": 0.95,
        },
        "selection": {
            "max_frames": 10,
            "method": "clustering",
        },
        "extraction": {
            "llm_provider": "anthropic",  # or "openai", "google"
            "model": "claude-sonnet-4-6",
        }
    }

    # Initialize pipeline with custom config
    pipeline = AdVideoPipeline(config=config)

    # Process video
    video_path = "path/to/your/video.mp4"
    result = pipeline.process_video(video_path)

    print(f"Selected {len(result.frames)} frames")
    print(f"Extraction: {result.extraction}")

if __name__ == "__main__":
    main()
