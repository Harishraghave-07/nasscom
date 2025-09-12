"""Presidio integration wrapper with safe lazy imports and fallbacks.

Provides PresidioAnalyzer.detect_phi_in_text(text, meta) which returns a
list of detection dicts. If the official Presidio Analyzer is available it
will be used. Otherwise we fall back to a lightweight regex-based detector
using patterns from AppConfig and an optional spaCy NER pass if spaCy is
installed.

This module is intentionally dependency-tolerant so the repo can run in
CI/dev environments that lack presidio or heavy ML libs.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from src.core.config import SETTINGS
except Exception:  # pragma: no cover - defensive
    SETTINGS = None


class PresidioAnalyzer:
    def __init__(self):
        self._engine = None
        # lazy attempt to import presidio AnalyzerEngine
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore

            self._engine = AnalyzerEngine()
            logger.info("Presidio Analyzer detected; using it for PHI detection")
        except Exception:
            logger.info("Presidio Analyzer not available; falling back to regex/spacy detector")
            self._engine = None

        # compile regexes from SETTINGS if available
        self._regexes = []
        try:
            if SETTINGS is not None:
                patterns = SETTINGS.get_phi_patterns()
                for name, pat in (patterns or {}).items():
                    try:
                        self._regexes.append((name, re.compile(pat, flags=re.IGNORECASE)))
                    except re.error:
                        logger.debug("Skipping invalid regex %s", name)
        except Exception:
            # ignore config parse issues
            pass

        # lazy spaCy model holder
        self._spacy_nlp = None

    def _ensure_spacy(self):
        if self._spacy_nlp is not None:
            return
        try:
            import spacy

            model = None
            try:
                model = SETTINGS.phi.spacy_model_name if SETTINGS is not None else "en_core_web_sm"
            except Exception:
                model = "en_core_web_sm"
            try:
                self._spacy_nlp = spacy.load(model)
                logger.info("Loaded spaCy model %s for fallback NER", model)
            except Exception:
                # try loading the small default without download
                try:
                    self._spacy_nlp = spacy.load("en_core_web_sm")
                except Exception:
                    logger.info("spaCy model not available; skipping spaCy fallback")
                    self._spacy_nlp = None
        except Exception:
            self._spacy_nlp = None

    def detect_phi_in_text(self, text: str, meta: Optional[Dict] = None) -> List[Dict]:
        meta = meta or {}
        detections: List[Dict] = []

        # If presidio is available prefer it
        if self._engine is not None:
            try:
                results = self._engine.analyze(text=text, language="en")
                # DEBUG: log presidio entities tuples (type, start, end, snippet)
                try:
                    logger.warning("DEBUG Presidio entities: %s",
                                   [(getattr(e, 'entity_type', None) or getattr(e, 'label', None), int(getattr(e, 'start', None) or e.start), int(getattr(e, 'end', None) or e.end), text[int(getattr(e, 'start', 0)):int(getattr(e, 'end', 0))]) for e in results])
                except Exception:
                    logger.warning("DEBUG Presidio entities: (failed to render entity tuples)")

                for r in results:
                    detections.append({
                        "start": int(getattr(r, "start", None) or r.start),
                        "end": int(getattr(r, "end", None) or r.end),
                        "confidence": float(getattr(r, "score", None) or getattr(r, "score", 0.0) or getattr(r, "confidence", 0.0)),
                        "entity_type": getattr(r, "entity_type", None) or getattr(r, "label", None),
                        "text": text[int(getattr(r, "start", 0)) : int(getattr(r, "end", 0))],
                        "meta": meta,
                    })
                return detections
            except Exception:
                logger.exception("Presidio Analyzer failed; falling back to regex/spacy")

        # regex-based detections
        for name, cre in self._regexes:
            for m in cre.finditer(text or ""):
                detections.append({"start": m.start(), "end": m.end(), "confidence": 0.9, "entity_type": name, "text": m.group(0), "meta": meta})

        # optional spaCy NER fallback
        try:
            self._ensure_spacy()
            if self._spacy_nlp is not None:
                doc = self._spacy_nlp(text or "")
                for ent in doc.ents:
                    detections.append({"start": ent.start_char, "end": ent.end_char, "confidence": 0.85, "entity_type": ent.label_, "text": ent.text, "meta": meta})
        except Exception:
            logger.debug("spaCy fallback NER not applied")

        return detections


_GLOBAL = None


def get_analyzer() -> PresidioAnalyzer:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = PresidioAnalyzer()
    return _GLOBAL


__all__ = ["PresidioAnalyzer", "get_analyzer"]
