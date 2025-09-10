"""Integration interfaces between the pipeline and PHI analyzers.

This module defines the function signature and data contracts used by the
ClinicalImageMaskingPipeline when calling out to analyzers (Presidio or
custom). Keep the implementation minimal and import-safe; callers in the
pipeline should perform lazy imports and fall back when an analyzer isn't
available.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def analyze_text_for_phi(
    text: str,
    analyzer: Optional[object] = None,
    entities: Optional[List[str]] = None,
    metadata: Optional[Dict] = None,
) -> List[Dict[str, object]]:
    """Analyze `text` and return a list of PHI detections.

    Signature contract:
      - text: raw string to analyze
      - analyzer: optional analyzer object (for example, a Presidio AnalyzerEngine
        instance or a wrapper with method analyze(text, language, entities))
      - entities: optional list of entity names to limit detection to (improves perf)
      - metadata: optional dict with contextual info (source, page, bbox)

    Returns:
      A list of dicts, each with keys: 'start'(int), 'end'(int), 'entity_type'(str),
      'score'(float), 'text'(str), and optional 'meta'.

    Implementation note: This function is intentionally a thin adapter. The
    pipeline should prefer calling a centralized wrapper (for example,
    src.integration.presidio_wrapper.get_analyzer()) and pass `entities` to
    it; when analyzer is None the implementation should gracefully return
    an empty list.
    """
    # ...existing code...
    # The real implementation will: (1) call analyzer.analyze(text, language='en', entities=entities)
    # (2) map results to the expected dict shape and attach metadata
    # (3) filter by configured thresholds if analyzer does not already do so
    # For now return empty list as a safe no-op.
    return []

__all__ = ["analyze_text_for_phi"]
