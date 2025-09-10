"""Active learning helpers for selecting items to send for human review.

The primary function `prioritize_for_review` implements a simple uncertainty
sampling strategy: pick detections whose confidence lies in an uncertain
range (e.g. 0.4-0.7) and randomly sample a fixed-size subset for annotation.

This module is intentionally small and dependency-free so it can be used in
unit tests and CI without heavy runtime deps.
"""
from __future__ import annotations

import random
from typing import Iterable, List, Mapping


def prioritize_for_review(
    detections: Iterable[Mapping[str, object]],
    low: float = 0.4,
    high: float = 0.7,
    sample_size: int = 50,
    seed: int | None = None,
) -> List[str]:
    """Select document_ids for human annotation using uncertainty sampling.

    Parameters
    - detections: iterable of mappings containing at least 'document_id' and 'confidence'.
      Example: {'document_id': 'doc-123', 'pii_text': '...', 'confidence': 0.56}
    - low/high: inclusive confidence window treated as "uncertain" (0.0-1.0).
    - sample_size: maximum number of document_ids to return.
    - seed: optional random seed for reproducible sampling.

    Returns a list of selected `document_id` strings.

    Why this approach:
    - Uncertainty sampling focuses human labeling effort on examples where the model
      is least confident. These examples are typically the most informative for
      updating model decision boundaries and improving calibration.
    - Compared to random sampling, uncertainty sampling yields higher value labels
      per annotated example because it targets the model's blind spots instead of
      wasting effort on examples the model already handles well.
    - The confidence window (low..high) avoids asking humans to review very low-
      confidence items that are likely non-informative (e.g., noise) or very high-
      confidence items that are almost certainly correct.
    """
    if low < 0.0 or high > 1.0 or low >= high:
        raise ValueError("Invalid low/high bounds: must satisfy 0.0 <= low < high <= 1.0")

    # Collect uncertain detections
    uncertain_ids: List[str] = []
    for d in detections:
        try:
            conf = float(d.get("confidence", d.get("score", None)))  # support 'score' alias
        except Exception:
            # Skip entries with non-numeric confidence
            continue
        if conf is None:
            continue
        if low <= conf <= high:
            doc_id = d.get("document_id") or d.get("doc_id") or d.get("id")
            if doc_id:
                uncertain_ids.append(str(doc_id))

    # If we have fewer than sample_size, shuffle and return all (to avoid selection bias)
    rnd = random.Random(seed)
    if len(uncertain_ids) <= sample_size:
        rnd.shuffle(uncertain_ids)
        return uncertain_ids

    # Otherwise randomly sample without replacement
    return rnd.sample(uncertain_ids, sample_size)
