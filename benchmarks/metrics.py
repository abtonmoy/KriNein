"""
Evaluation metrics for the benchmarking suite.

- Frame selection metrics: count, compression, latency, info density, VLM cost
- Extraction comparison: field-level match against pipeline reference
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Constants matching the pipeline spec
TOKENS_PER_FRAME = 765
VLM_COST_PER_1K_TOKENS = 0.015


# ============================================================================
# Frame Selection Metrics
# ============================================================================

def compute_selection_metrics(
    selected_frames: List[Tuple[float, np.ndarray]],
    total_frames: int,
    selection_latency: float,
    clip_deduplicator=None,
) -> Dict[str, Any]:
    """
    Compute frame-selection quality metrics.

    Args:
        selected_frames: Baseline's output frames
        total_frames: Total frames in video (or total candidates)
        selection_latency: Wall-clock seconds for frame selection
        clip_deduplicator: Optional CLIPDeduplicator instance for info density

    Returns:
        Dict with selection metrics
    """
    count = len(selected_frames)
    compression = round(total_frames / count, 2) if count > 0 else float("inf")
    cost = round((count * TOKENS_PER_FRAME / 1000.0) * VLM_COST_PER_1K_TOKENS, 4)

    density = 0.0
    if clip_deduplicator is not None and count >= 2:
        density = compute_info_density(selected_frames, clip_deduplicator)

    return {
        "selected_count": count,
        "compression_ratio": compression,
        "latency_s": round(selection_latency, 3),
        "info_density": density,
        "vlm_cost_usd": cost,
    }


def compute_info_density(
    frames: List[Tuple[float, np.ndarray]],
    clip_deduplicator,
) -> float:
    """
    Mean pairwise CLIP cosine distance among selected frames.
    Higher = more diverse / more unique information.

    Uses the pipeline's own CLIPDeduplicator for consistent embeddings.
    """
    if len(frames) < 2:
        return 0.0

    frame_arrays = [f for _, f in frames]

    try:
        embeddings = clip_deduplicator.compute_signatures_batch(frame_arrays)
    except Exception as e:
        logger.warning(f"Info density computation failed: {e}")
        return 0.0

    # embeddings are already L2-normalized by CLIPDeduplicator
    sim_matrix = embeddings @ embeddings.T
    n = len(embeddings)
    upper_tri = sim_matrix[np.triu_indices(n, k=1)]
    mean_distance = float(1.0 - upper_tri.mean())

    return round(mean_distance, 5)


# ============================================================================
# Extraction Comparison Metrics
# ============================================================================

def compare_extractions(
    baseline_result: Dict[str, Any],
    reference_result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compare a baseline's extraction against the pipeline reference.

    Checks key fields: brand, promotion, CTA, topic, effectiveness.
    Returns a dict of match/value indicators.
    """
    if not baseline_result or "error" in baseline_result:
        return {
            "brand_match": None,
            "promo_detected": None,
            "cta_detected": None,
            "topic_match": None,
            "effectiveness": None,
            "error": baseline_result.get("error", "unknown"),
        }

    return {
        "brand_match": _brand_match(baseline_result, reference_result),
        "promo_detected": _safe_get_bool(baseline_result, ["promotion", "promo_present"]),
        "cta_detected": _safe_get_bool(baseline_result, ["call_to_action", "cta_present"]),
        "topic_match": _topic_match(baseline_result, reference_result),
        "effectiveness": _safe_get(
            baseline_result, ["engagement_metrics", "effectiveness_score"]
        ),
    }


def _brand_match(baseline: Dict, reference: Dict) -> Optional[bool]:
    """Check if brand names match (case-insensitive, partial match)."""
    b_brand = _safe_get(baseline, ["brand", "brand_name_text"])
    r_brand = _safe_get(reference, ["brand", "brand_name_text"])

    if b_brand is None or r_brand is None:
        return None

    b_lower = str(b_brand).lower().strip()
    r_lower = str(r_brand).lower().strip()

    # Exact or containment match
    return b_lower == r_lower or b_lower in r_lower or r_lower in b_lower


def _topic_match(baseline: Dict, reference: Dict) -> Optional[bool]:
    """Check if topic IDs match."""
    b_topic = _safe_get(baseline, ["topic", "topic_id"])
    r_topic = _safe_get(reference, ["topic", "topic_id"])

    if b_topic is None or r_topic is None:
        return None

    try:
        return int(b_topic) == int(r_topic)
    except (ValueError, TypeError):
        return None


def _safe_get(d: Dict, keys: List[str]) -> Any:
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


def _safe_get_bool(d: Dict, keys: List[str]) -> Optional[bool]:
    """Get a boolean value from nested dict."""
    val = _safe_get(d, keys)
    if val is None:
        return None
    return bool(val)