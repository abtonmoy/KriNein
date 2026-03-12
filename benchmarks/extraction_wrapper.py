"""
Dual-mode extraction wrapper for Option C benchmarking.

Wraps the existing AdExtractor to provide two extraction modes:
  - BARE:  No temporal context, no audio, fixed schema (1 LLM call)
           → fair frame-selection comparison
  - FULL:  Complete Stage 7 treatment (2 LLM calls for adaptive schema)
           → system-level comparison

Both modes use the exact same LLM provider/model so the only variable
is the quality of the selected frames + context richness.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from src.extraction.llm_client import AdExtractor, create_extractor

logger = logging.getLogger(__name__)


class ExtractionWrapper:
    """
    Provides bare and full extraction using the pipeline's own AdExtractor.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Args:
            config: Full pipeline config dict (same format as default.yaml).
                    The extraction section is used for provider/model settings.
        """
        ext = config.get("extraction", {})
        # Allow benchmark config to override provider/model
        bench_ext = config.get("benchmark", {}).get("extraction", {})
        provider = bench_ext.get("provider", ext.get("provider", "gemini"))
        model = bench_ext.get("model", ext.get("model", "gemini-2.0-flash-exp"))
        max_tokens = bench_ext.get("max_tokens", ext.get("max_tokens", 4000))

        # ------------------------------------------------------------------
        # BARE extractor: no temporal context, no audio, fixed schema
        # This gives every baseline the same minimal prompt so the only
        # variable is which frames were selected.
        # ------------------------------------------------------------------
        self.bare = AdExtractor(
            provider=provider,
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            schema_mode="fixed",
            temporal_context=False,
            include_timestamps=False,
            include_time_deltas=False,
            include_position_labels=False,
            include_narrative_instructions=False,
        )

        # ------------------------------------------------------------------
        # FULL extractor: complete Stage 7 (temporal + audio + adaptive)
        # Uses create_extractor() which reads all settings from config.
        # ------------------------------------------------------------------
        self.full = create_extractor(config)

        logger.info(
            f"ExtractionWrapper initialized: provider={provider}, model={model}"
        )

    def extract_bare(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
    ) -> Dict[str, Any]:
        """
        Fair comparison extraction — same minimal prompt for all methods.

        No temporal context, no audio, no adaptive schema.
        Uses fixed schema (1 LLM call).
        """
        if not frames:
            return {"error": "No frames provided"}

        try:
            return self.bare.extract(frames, video_duration, audio_context=None)
        except Exception as e:
            logger.error(f"Bare extraction failed: {e}")
            return {"error": str(e)}

    def extract_full(
        self,
        frames: List[Tuple[float, np.ndarray]],
        video_duration: float,
        audio_context: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        System comparison extraction — full Stage 7 treatment.

        Includes temporal context, audio context, adaptive schema with
        type detection (2 LLM calls).
        """
        if not frames:
            return {"error": "No frames provided"}

        try:
            return self.full.extract(
                frames, video_duration, audio_context=audio_context
            )
        except Exception as e:
            logger.error(f"Full extraction failed: {e}")
            return {"error": str(e)}