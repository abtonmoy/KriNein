# Main Pipeline & Configuration Updates

This document summarizes the changes made to the core pipeline orchestrator (`src/pipeline.py`) and the main configuration file (`config/default.yaml`) to integrate all recent subsystem improvements.

## 1. Config Unification (`config/default.yaml`)

We have explicitly exposed the configuration keys for several newly implemented improvements so they can be easily toggled without modifying code:

*   **Smart Global Frame Budget (Intrinsic Semantic Dimensionality)**: Converted the legacy frame capper into an adaptive mathematical SVD algorithm (`selection.use_hib_budget: true`). This computes the Intrinsic Semantic Dimensionality (ISD) of the visual embeddings and automatically clamps repetitive videos to save tokens, while dynamically expanding the limit on chaotic videos to preserve entropy. Can be reverted to a strict cap (`selection.global_max_frames: 25`).
*   **Visual Feature Scoring**: Exposed `selection.use_visual_features: true`. This enables OpenCV-based text, face, and logo detection to boost the importance score of frames containing human-readable or brand-heavy content.
*   **Single-Pass Extraction**: Exposed `extraction.single_pass: true`. This merges the ad-type classification pass and the data extraction pass into a single LLM API call, saving ~50% in token costs.
*   **OCR Pre-processing**: Added `extraction.ocr_context.enabled: true`.
*   **Segment Prompting (Optional)**: Exposed `extraction.segment_prompting: false`. Can be enabled for better temporal understanding on complex narratives, guiding the LLM scene-by-scene.

## 2. Pipeline Orchestration (`src/pipeline.py`)

The main `AdVideoPipeline.process()` method has been deeply integrated with the new extraction and selection sub-systems:

*   **OCR Context Injection**: Added an `_ocr_extractor` lazy property. During Stage 7, if OCR is enabled in config, the pipeline now runs `detect_text_regions` on the mathematically selected frames. The returned string (e.g. `"Frame @ 2.5s: 3 text regions, 15.2% coverage"`) is injected directly into `audio_context["ocr_context"]` so the LLM prompt builder includes it as mathematical evidence of text.
*   **Scene Boundary Propagation**: The extraction stage now receives `scene_boundaries` from Stage 2. This allows `build_segmented_prompt` to group frames chronologically by the scene they belong to, providing the LLM with a structural map of the video rather than just a flat list of frames.
*   **Non-deterministic Array Parsing Fix**: Modified `_parse_json_response` and the extraction flow in `llm_client.py` to gracefully handle cases where the LLM occasionally returns a JSON array (e.g., `[{...}]`) instead of a JSON object (`{...}`). This prevents pipeline crashes (`list indices must be integers or slices`) during the metadata-appending phase.

## Usage Note

You can now run the complete dataset batch pipeline using:
```bash
uv run python main.py
```
This will automatically utilize all 15 performance, cost, and accuracy improvements simultaneously via the `ParallelPipeline` wrapper.
