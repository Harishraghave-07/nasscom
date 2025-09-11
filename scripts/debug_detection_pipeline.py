#!/usr/bin/env python3
"""Trace the detection pipeline: config, presidio/spaCy availability, detections, and threshold filtering.

Usage:
  python3 scripts/debug_detection_pipeline.py

This script is defensive: it attempts imports and reports availability rather
than failing hard. It uses a sample text block taken from the Emily Dawson PDF
for testing.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import sys
from typing import Any, Dict, List, Tuple
from pathlib import Path

SAMPLE_TEXT = """
Highlighted PHI Details for Medical Record - Emily R. Dawson
Patient Information
Name: Emily R. Dawson
DOB: March 12, 1975
Address: 1284 Pine Ridge Drive, Aurora, CO 80014
Phone: (303) 555-7294
Email: emily.dawson75@example.com
SSN: 527-84-1937
Oncologist: Dr. Marissa Greene, MD
"""

# Simple regex patterns for verification
REGEX_PATTERNS: Dict[str, str] = {
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "PHONE": r"\(\d{3}\)\s*\d{3}[- ]?\d{4}|\b\d{3}-\d{3}-\d{4}\b",
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    "NAME_SIMPLE": r"\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*[A-Z]\.)?\b",
}

LOG = logging.getLogger("debug_detection")


def print_heading(title: str) -> None:
    print("\n" + "#" * 3 + f" {title} " + "#" * 3 + "\n")


def load_settings() -> Any:
    try:
        from src.core.config import SETTINGS

        return SETTINGS
    except Exception as e:
        print(f"Failed to import SETTINGS: {e}")
        return None


def check_presidio() -> Dict[str, Any]:
    info: Dict[str, Any] = {"presidio_available": False, "analyzer_ok": False, "anonymizer_ok": False}
    try:
        import presidio_analyzer
    except Exception as e:  # pragma: no cover - defensive
        info["presidio_available"] = False
        info["presidio_analyzer_import_error"] = str(e)
        return info

    # presidio_analyzer import succeeded
    info["presidio_available"] = True
    try:
        from presidio_analyzer import AnalyzerEngine

        ae = AnalyzerEngine()
        info["analyzer_ok"] = True
        # quick analyze smoke test
        try:
            res = ae.analyze(text="John 123-45-6789 john@example.com", language="en")
            info["analyzer_test_count"] = len(res)
        except Exception as e:
            info["analyzer_test_error"] = str(e)
    except Exception as e:
        info["analyzer_ok_exception"] = str(e)

    try:
        from presidio_anonymizer import AnonymizerEngine

        _ = AnonymizerEngine()
        info["anonymizer_ok"] = True
    except Exception as e:
        info["anonymizer_ok_exception"] = str(e)

    return info


def check_spacy_model(settings: Any) -> Dict[str, Any]:
    info: Dict[str, Any] = {"spacy_installed": False, "model_loaded": False, "model_name": None}
    try:
        import spacy

        info["spacy_installed"] = True
        model_name = None
        try:
            model_name = settings.phi.spacy_model_name if settings is not None else "en_core_web_sm"
        except Exception:
            model_name = "en_core_web_sm"
        info["model_name"] = model_name
        try:
            nlp = spacy.load(model_name)
            info["model_loaded"] = True
        except Exception as e:
            info["model_load_error"] = str(e)
    except Exception as e:
        info["spacy_import_error"] = str(e)
    return info


def presidio_direct_analyze(text: str) -> List[Dict[str, Any]]:
    try:
        from presidio_analyzer import AnalyzerEngine

        ae = AnalyzerEngine()
        results = ae.analyze(text=text, language="en")
        # normalize results
        out = []
        for r in results:
            out.append({"entity_type": r.entity_type, "start": r.start, "end": r.end, "score": float(r.score), "text": text[r.start:r.end]})
        return out
    except Exception as e:
        print(f"Presidio analyze failed: {e}")
        return []


def presidio_wrapper_detect(text: str) -> List[Dict[str, Any]]:
    try:
        from src.integration.presidio_wrapper import get_analyzer

        an = get_analyzer()
        dets = an.detect_phi_in_text(text)
        # wrapper returns list of dicts with start,end,score,entity_type,text
        return dets
    except Exception as e:
        print(f"Presidio wrapper call failed: {e}")
        return []


def spacy_detect(text: str, model_name: str = "en_core_web_sm") -> List[Dict[str, Any]]:
    try:
        import spacy

        nlp = spacy.load(model_name)
        doc = nlp(text)
        out = []
        for ent in getattr(doc, "ents", []):
            out.append({"label": ent.label_, "text": ent.text, "start": ent.start_char, "end": ent.end_char})
        return out
    except Exception as e:
        print(f"spaCy detect failed: {e}")
        return []


def regex_detect(text: str) -> List[Dict[str, Any]]:
    out = []
    for name, pat in REGEX_PATTERNS.items():
        for m in re.finditer(pat, text):
            out.append({"pattern": name, "text": m.group(0), "start": m.start(), "end": m.end()})
    return out


def apply_thresholds(detections: List[Dict[str, Any]], settings: Any) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    kept = []
    rejected = []
    # default threshold
    default_thresh = None
    try:
        default_thresh = float(settings.phi.phi_confidence_threshold)
    except Exception:
        default_thresh = 0.8

    for d in detections:
        score = None
        # presidio result might be object or dict
        if isinstance(d, dict):
            score = float(d.get("score") or d.get("confidence") or 0.0)
        else:
            # fallback: try attributes
            try:
                score = float(getattr(d, "score", 0.0))
            except Exception:
                score = 0.0

        if score >= default_thresh:
            kept.append({**d, "score": score})
        else:
            rejected.append({**d, "score": score})
    return kept, rejected


def main() -> int:
    settings = load_settings()

    print_heading("CONFIGURATION")
    if settings is None:
        print("SETTINGS not available (import failed)")
    else:
        cfg = {
            "use_presidio": getattr(settings, "use_presidio", None),
            "presidio_canary_percentage": getattr(settings, "presidio_canary_percentage", None),
            "shadow_mode_enabled": getattr(settings, "shadow_mode_enabled", None),
            "phi_confidence_threshold": getattr(settings.phi, "phi_confidence_threshold", getattr(settings.phi, "phi_confidence_threshold", None) if hasattr(settings, "phi") else None),
            "phi_default_threshold": getattr(settings.phi, "phi_confidence_threshold", None),
        }
        print(json.dumps(cfg, indent=2))

    print_heading("PRESIDIO AVAILABILITY")
    pres_info = check_presidio()
    print(json.dumps(pres_info, indent=2))

    print_heading("spaCy MODEL CHECK")
    spacy_info = check_spacy_model(settings)
    print(json.dumps(spacy_info, indent=2))

    print_heading("DETECTION TRACES ON SAMPLE TEXT")
    print("Sample text:\n")
    print(SAMPLE_TEXT)

    print_heading("Presidio direct Analyzer (if available)")
    pres_direct = presidio_direct_analyze(SAMPLE_TEXT)
    print(f"Presidio direct found: {len(pres_direct)}")
    for d in pres_direct:
        print(f"  {d['entity_type']}: '{d['text']}' score={d['score']:.3f} span={d['start']}-{d['end']}")

    print_heading("Presidio wrapper (regex+spaCy fallback)")
    pres_wrap = presidio_wrapper_detect(SAMPLE_TEXT)
    print(f"Wrapper detections: {len(pres_wrap)}")
    for d in pres_wrap:
        # normalize keys
        print(f"  {d.get('entity_type') or d.get('pattern')}: '{d.get('text')}' score={d.get('score', d.get('confidence', 0.0))} span={d.get('start')}-{d.get('end')}")

    print_heading("spaCy NER")
    model_name = (settings.phi.spacy_model_name if settings is not None else "en_core_web_sm")
    spacy_out = spacy_detect(SAMPLE_TEXT, model_name=model_name)
    print(f"spaCy ents: {len(spacy_out)}")
    for e in spacy_out:
        print(f"  {e['label']}: '{e['text']}' span={e['start']}-{e['end']}")

    print_heading("Regex detection")
    rx = regex_detect(SAMPLE_TEXT)
    print(f"Regex matches: {len(rx)}")
    for m in rx:
        print(f"  {m['pattern']}: '{m['text']}' span={m['start']}-{m['end']}")

    print_heading("THRESHOLD FILTER SIMULATION")
    # combine detections from wrapper and presidio direct for thresholding demo
    combined = []
    for d in pres_direct:
        combined.append({"text": d.get("text"), "score": d.get("score", 0.0), "source": "presidio_direct"})
    for d in pres_wrap:
        combined.append({"text": d.get("text"), "score": float(d.get("score", d.get("confidence", 0.0))), "source": "presidio_wrapper"})
    for m in rx:
        combined.append({"text": m.get("text"), "score": 0.85, "source": "regex"})

    kept, rejected = apply_thresholds(combined, settings)
    print(f"Kept ({len(kept)}):")
    for k in kept:
        print(f"  [{k.get('source')}] '{k.get('text')}' score={k.get('score')}")
    print(f"Rejected ({len(rejected)}):")
    for r in rejected:
        print(f"  [{r.get('source')}] '{r.get('text')}' score={r.get('score')}")

    print_heading("SUMMARY")
    print("Finished detection pipeline diagnostics. Review above to see where detections are missing or filtered.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
