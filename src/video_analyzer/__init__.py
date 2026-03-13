"""
KriNein - A clean, efficient library for video content analysis.

This package provides tools for:
- Video ingestion and audio extraction
- Scene and change detection
- Frame deduplication using various methods (hash-based, SSIM, LPIPS, CLIP)
- Key frame selection via clustering
- LLM-based content extraction
"""

__version__ = "1.0.0"

from .pipeline import AdVideoPipeline
from .parallel import ParallelVideoPipeline
from .utils.logging import setup_logging

__all__ = [
    "AdVideoPipeline",
    "ParallelVideoPipeline",
    "setup_logging",
    "__version__",
]
