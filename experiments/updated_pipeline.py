# experiments/updated_pipeline.py
"""
Updated pipeline experiment showcasing ALL 15 improvements.

Usage:
    uv run python -m experiments.updated_pipeline

Improvements exercised:
  1. Single audio load per video
  2. Whisper model caching across videos
  3. LLM retry with exponential backoff
  4. Single-pass extraction (type + content in one call)
  5. Visual feature scoring (text/face detection)
  6. Parallel batch processing (ThreadPoolExecutor)
  7. Gemini native video mode (optional)
  8. ML mood classification (with fallback)
  9. Robust JSON parsing
  10. Smart frame budget (diminishing returns)
  11. Pre-detected speech reuse
  12. Confidence scoring
  13. Segment-level prompting
  14. OCR pre-processing for text-heavy frames
  15. Disk-based frame storage
"""

from dotenv import load_dotenv
load_dotenv()

import os
import sys
import json
import time
import logging
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from src.pipeline import AdVideoPipeline
from src.detection.ocr_extractor import OCRExtractor
from src.detection.visual_features import VisualFeatureDetector
from src.utils.frame_store import FrameStore

logger = logging.getLogger(__name__)

# -- 10 test videos from data/hussain_videos --
VIDEOS = [
    r"data/hussain_videos/_NJKQkQu3NI.mp4",
    r"data/hussain_videos/0gYBy1rECII.mp4",
    r"data/hussain_videos/0tteHhYh9rU.mp4",
    r"data/hussain_videos/0rFt8QE4lkQ.mp4",
    r"data/hussain_videos/2uT9fjUmm-M.mp4",
    r"data/hussain_videos/3dnW8QJxHD0.mp4",
    r"data/hussain_videos/0A43tb18T8Y.mp4",
    r"data/hussain_videos/1dc1eKHgy_o.mp4",
    r"data/hussain_videos/1AIDh3-sGW0.mp4",
    r"data/hussain_videos/14X98i8psts.mp4",
]


def run_single_video_demo():
    """
    Process a single video showing all improvement outputs.
    Demonstrates: improvements 1-5, 8-15
    """
    print("=" * 70)
    print("  DEMO 1: Single Video -- Full Feature Showcase")
    print("=" * 70)

    pipeline = AdVideoPipeline(
        config_path="config/default.yaml",
        overrides={
            "extraction": {"single_pass": True},
            "audio_analysis": {
                "mood_classification": {"enabled": True, "use_ml": True},
            },
            "selection": {"global_max_frames": 15, "use_visual_features": True},
        },
    )

    video = VIDEOS[0]
    print(f"\n  [>] Processing: {video}")
    start = time.time()
    result = pipeline.process(video, skip_extraction=False)
    elapsed = time.time() - start

    # Core pipeline results
    print(f"\n{'-'*50}")
    print(f"  Video:      {result.video_path}")
    print(f"  Duration:   {result.metadata.duration:.1f}s")
    print(f"  Scenes:     {len(result.scenes)}")
    print(f"  Candidates: {result.total_frames_sampled}")
    print(f"  After Hash: {result.frames_after_phash}")
    print(f"  After LPIPS:{result.frames_after_ssim}")
    print(f"  After CLIP: {result.frames_after_clip}")
    print(f"  Final:      {result.final_frame_count}")
    print(f"  Reduction:  {result.reduction_rate:.1%}")
    print(f"  Time:       {elapsed:.1f}s")

    # Improvement 12: Confidence scoring
    if result.extraction_result:
        meta = result.extraction_result.get("_metadata", {})
        confidence = meta.get("confidence", "N/A")
        single_pass = meta.get("single_pass", False)
        print(f"\n  * Confidence:   {confidence}")
        print(f"  * Single-pass:  {single_pass}")
        print(f"  * Ad type:      {result.extraction_result.get('ad_type', 'N/A')}")
        print(f"  * Schema mode:  {meta.get('schema_mode', 'N/A')}")

    # Improvement 14: OCR pre-processing
    print(f"\n{'-'*50}")
    print("  OCR Pre-Processing (Improvement #14)")
    print(f"{'-'*50}")

    ocr = OCRExtractor()

    # Extract a few frames directly using OpenCV for the demo
    import cv2
    cap = cv2.VideoCapture(video)
    demo_frames = []
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Sample 3 evenly spaced frames
    for idx in [int(total * 0.1), int(total * 0.5), int(total * 0.9)]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            ts = idx / fps
            demo_frames.append((ts, frame))
    cap.release()

    if demo_frames:
        ocr_results = ocr.extract_batch(demo_frames)
        for ctx in ocr_results:
            print(
                f"    t={ctx['timestamp']:.2f}s  |  "
                f"text={ctx['has_text']}  regions={ctx['text_region_count']}  "
                f"coverage={ctx['text_coverage']:.3f}"
            )

        prompt_context = ocr.build_ocr_context_for_prompt(demo_frames)
        if len(prompt_context) > 120:
            print(f"    Prompt context: {prompt_context[:120]}...")
        else:
            print(f"    Prompt context: {prompt_context}")

        # Improvement 15: Frame store demo
        print(f"\n{'-'*50}")
        print("  Frame Store (Improvement #15)")
        print(f"{'-'*50}")

        store = FrameStore(quality=90)
        saved = store.save_batch(demo_frames)
        print(f"    Saved {len(saved)} frames to disk: {store.frame_dir}")
        for ts, path in saved:
            print(f"      t={ts:.2f}s -> {Path(path).name}")
        loaded = store.load_batch()
        print(f"    Loaded back {len(loaded)} frames from disk")
        store.cleanup()
        print(f"    Cleaned up temp storage")
    else:
        print("    Could not extract demo frames")

    # Full extraction result
    if result.extraction_result:
        print(f"\n{'-'*50}")
        print("  Extracted Ad Data")
        print(f"{'-'*50}")
        print(json.dumps(result.extraction_result, indent=2))

    return result


def run_batch_demo():
    """
    Process 10 videos in parallel with batch processing.
    Demonstrates: improvement 6, 2, 12, 10
    """
    print("\n" + "=" * 70)
    print("  DEMO 2: Batch Processing -- 10 Videos (Improvement #6)")
    print("=" * 70)

    existing = [v for v in VIDEOS if os.path.exists(v)]
    missing = [v for v in VIDEOS if not os.path.exists(v)]
    if missing:
        print(f"\n  [!] {len(missing)} videos not found, skipping them:")
        for v in missing:
            print(f"    x {v}")
    print(f"\n  Processing {len(existing)} videos...")

    pipeline = AdVideoPipeline(
        config_path="config/default.yaml",
        overrides={
            "extraction": {"single_pass": True},
            "selection": {"global_max_frames": 15, "use_visual_features": True},
        },
    )

    # Improvement 6: True parallel batch processing with max_workers=2
    start = time.time()
    results = pipeline.process_batch(
        existing,
        max_workers=2,
        skip_extraction=True,  # Skip LLM to save API cost during batch test
    )
    batch_elapsed = time.time() - start

    # Summary table
    print(f"\n{'-'*90}")
    print(
        f"  {'Video':<30} {'Duration':>8} {'Scenes':>7} {'Cands':>7} "
        f"{'Final':>7} {'Reduc':>8} {'Time':>8}"
    )
    print(f"{'-'*90}")

    total_candidates = 0
    total_final = 0

    for i, r in enumerate(results):
        if r is None:
            print(f"  {Path(existing[i]).stem:<30} {'FAILED':>8}")
            continue

        total_candidates += r.total_frames_sampled
        total_final += r.final_frame_count

        print(
            f"  {Path(r.video_path).stem:<30} "
            f"{r.metadata.duration:>7.1f}s "
            f"{len(r.scenes):>7} "
            f"{r.total_frames_sampled:>7} "
            f"{r.final_frame_count:>7} "
            f"{r.reduction_rate:>7.1%} "
            f"{r.processing_time_s:>7.1f}s"
        )

    print(f"{'-'*90}")
    successful = sum(1 for r in results if r is not None)
    avg_reduction = (1 - total_final / total_candidates) if total_candidates > 0 else 0
    print(
        f"  Total: {successful}/{len(existing)} succeeded  |  "
        f"{total_candidates} -> {total_final} frames  |  "
        f"{avg_reduction:.1%} avg reduction  |  "
        f"{batch_elapsed:.1f}s total ({batch_elapsed / max(successful, 1):.1f}s/video)"
    )

    return results


def run_visual_features_demo():
    """
    Show visual feature detection across multiple videos.
    Demonstrates: improvement 5 (visual features), 14 (OCR)
    """
    print("\n" + "=" * 70)
    print("  DEMO 3: Visual Feature Detection (Improvement #5)")
    print("=" * 70)

    detector = VisualFeatureDetector()

    test_videos = [v for v in VIDEOS[:3] if os.path.exists(v)]
    for video in test_videos:
        print(f"\n  [>] {Path(video).stem}")

        import cv2
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        for idx in [int(total * f) for f in [0.1, 0.3, 0.5, 0.7, 0.9]]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append((idx / fps, frame))
        cap.release()

        results = detector.detect_batch(frames)
        for ts, features in sorted(results.items()):
            text_icon = "[T]" if features["has_text"] else "   "
            face_icon = "[F]" if features["has_face"] else "   "
            print(
                f"    t={ts:>6.2f}s  {text_icon} text  {face_icon} face  "
                f"density={features['text_density']}"
            )


def run_confidence_comparison():
    """
    Compare confidence scores across videos.
    Demonstrates: improvement 12 (confidence scoring)
    """
    print("\n" + "=" * 70)
    print("  DEMO 4: Confidence Scoring (Improvement #12)")
    print("=" * 70)

    pipeline = AdVideoPipeline(
        config_path="config/default.yaml",
        overrides={
            "extraction": {"single_pass": True},
            "selection": {"global_max_frames": 10, "use_visual_features": True},
        },
    )

    test_videos = [v for v in VIDEOS[:3] if os.path.exists(v)]
    for video in test_videos:
        print(f"\n  [>] {Path(video).stem}")
        result = pipeline.process(video, skip_extraction=False)

        if result.extraction_result:
            meta = result.extraction_result.get("_metadata", {})
            conf = meta.get("confidence", 0)
            bar = "#" * int(conf * 20) + "." * (20 - int(conf * 20))
            ad_type = result.extraction_result.get("ad_type", "unknown")
            brand = result.extraction_result.get("brand", {})
            brand_name = (
                brand.get("brand_name_text", "N/A") if isinstance(brand, dict) else "N/A"
            )

            print(f"    Confidence: [{bar}] {conf:.2f}")
            print(f"    Ad type:    {ad_type}")
            print(f"    Brand:      {brand_name}")
            print(f"    Frames:     {result.final_frame_count}")
            print(f"    Single-pass:{meta.get('single_pass', False)}")
        else:
            print(f"    x Extraction failed")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print("  VIDEO ANALYSIS PIPELINE -- ALL 15 IMPROVEMENTS")
    print("=" * 70)

    print(
        """
    Improvements being tested:
     [1]  Single audio load          [9]  Robust JSON parsing
     [2]  Whisper model caching      [10] Smart frame budget
     [3]  LLM retry + backoff        [11] Pre-detected speech reuse
     [4]  Single-pass extraction     [12] Confidence scoring
     [5]  Visual feature scoring     [13] Segment-level prompting
     [6]  Parallel batch processing  [14] OCR pre-processing
     [7]  Gemini native video mode   [15] Disk-based frame storage
     [8]  ML mood classification
    """
    )

    # Demo 1: Single video, full feature showcase
    run_single_video_demo()

    # Demo 2: Batch processing with 10 videos
    run_batch_demo()

    # Demo 3: Visual feature detection
    run_visual_features_demo()

    # Demo 4: Confidence scoring comparison
    run_confidence_comparison()

    print("\n" + "=" * 70)
    print("  ALL DEMOS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
