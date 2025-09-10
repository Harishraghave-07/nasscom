"""Utilities to build Presidio entity lists from application config.

Provides a small helper to read entity groupings from the application's
configuration and return a deduplicated, flat list suitable for passing to
Presidio's Analyzer/Recognizer APIs.

Batching entities into groups (instead of one-by-one calls) improves
performance because the Presidio Analyzer can analyze the text once and
return all matching entities in a single pass. This reduces repeated
tokenization, model invocations, and inter-process or remote calls. It
also allows the analyzer to resolve overlapping spans and prefer higher-
confidence labels centrally.
"""

from __future__ import annotations

from typing import Iterable, List

def load_presidio_entities(config) -> List[str]:
    """Read entity lists from `config` and return a flattened deduplicated list.

    Expected config shape (flexible):
      - config.critical_entities: Iterable[str]
      - config.high_priority_entities: Iterable[str]
      - config.contextual_entities: Iterable[str]
      - config.emerging_entities: Iterable[str]

    The function tolerates missing attributes and accepts either lists or
    comma-separated strings. Order is preserved by priority (critical first).

    Returns:
        A list[str] containing unique entity names suitable for passing to
        Presidio AnalyzerEngine.analyze(..., entities=entities).
    """
    groups = []

    def _ensure_iter(x) -> Iterable[str]:
        if x is None:
            return []
        if isinstance(x, str):
            return [s.strip() for s in x.split(",") if s.strip()]
        try:
            # treat as iterable
            return list(x)
        except Exception:
            return []

    try:
        groups.extend(_ensure_iter(getattr(config, "critical_entities", None)))
        groups.extend(_ensure_iter(getattr(config, "high_priority_entities", None)))
        groups.extend(_ensure_iter(getattr(config, "contextual_entities", None)))
        groups.extend(_ensure_iter(getattr(config, "emerging_entities", None)))
    except Exception:
        # defensive: if config is a dict-like
        try:
            groups.extend(_ensure_iter(config.get("critical_entities")))
            groups.extend(_ensure_iter(config.get("high_priority_entities")))
            groups.extend(_ensure_iter(config.get("contextual_entities")))
            groups.extend(_ensure_iter(config.get("emerging_entities")))
        except Exception:
            return []

    # deduplicate while preserving order
    seen = set()
    out: List[str] = []
    for e in groups:
        if not e:
            continue
        key = e.strip()
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out

__all__ = ["load_presidio_entities"]
