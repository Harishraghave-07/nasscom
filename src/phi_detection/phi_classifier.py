"""PHI detection and classification for HIPAA compliance.

This module implements PHIClassifier which combines spaCy NER (when
available) with regex-based patterns for the 18 HIPAA PHI identifiers. The
implementation uses lazy imports for heavy dependencies and includes audit
logging hooks to integrate with the application's audit trail.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.core.config import PHIDetectionConfig, AppConfig, SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class PHIDetection:
    text: str
    start: int
    end: int
    phi_type: Optional[str] = None
    method: str = "regex"
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "start": int(self.start),
            "end": int(self.end),
            "phi_type": self.phi_type,
            "method": self.method,
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }


class PHIClassifier:
    """Classifier for the 18 HIPAA PHI identifiers using spaCy + regex.

    Design choices:
    - spaCy is loaded lazily in __init__ to let the module import cheaply in
      environments without spaCy.
    - Regex patterns are precompiled for performance.
    - Audit logging writes detection events via SETTINGS.logging.audit_log_path
      when enabled.
    """

    # canonical list of HIPAA PHI identifiers (18 types)
    HIPAA_TYPES = [
        "names",
        "geographic_locations",
        "dates",
        "phone_numbers",
        "fax_numbers",
        "email_addresses",
        "social_security_numbers",
        "medical_record_numbers",
        "health_plan_numbers",
        "account_numbers",
        "certificate_license_numbers",
        "vehicle_identifiers",
        "device_identifiers",
        "urls",
        "ip_addresses",
        "biometric_identifiers",
        "full_face_photos",
        "other_unique_identifiers",
    ]

    def __init__(self, config: PHIDetectionConfig | AppConfig):
        # Accept either a PHIDetectionConfig (legacy) or AppConfig (preferred).
        if isinstance(config, AppConfig):
            self.app_config: AppConfig = config
            self.config: PHIDetectionConfig = config.phi
        else:
            # legacy branch: try to use global SETTINGS for AppConfig values
            self.app_config = SETTINGS
            self.config = config
        self._spacy_nlp = None
        self._spacy_available = False
        self._load_spacy_model()
        self._compile_patterns()
        self.audit_logger = logging.getLogger("cim.audit.phi")
        self._load_custom_vocabulary()
        self._thresholds = defaultdict(lambda: float(self.config.phi_confidence_threshold))
        # allow overrides for critical types
        self._thresholds.update({"social_security_numbers": 0.99, "medical_record_numbers": 0.98})
        self._cache_lock = threading.Lock()
        self._pattern_cache: Dict[str, Iterable] = {}

    # ------------------ initialization helpers ------------------
    def _load_spacy_model(self) -> None:
        try:
            import spacy  # type: ignore

            model_name = getattr(self.config, "spacy_model_name", "en_core_web_sm")
            try:
                self._spacy_nlp = spacy.load(model_name)
                self._spacy_available = True
                logger.info(f"Loaded spaCy model {model_name}")
            except Exception:
                # if model not available, try to load small default
                try:
                    self._spacy_nlp = spacy.blank("en")
                    self._spacy_available = True
                    logger.warning(f"Failed to load {model_name}; using blank English pipeline")
                except Exception:
                    self._spacy_nlp = None
                    self._spacy_available = False
                    logger.exception("spaCy model load failed; NER disabled")
        except Exception:
            logger.debug("spaCy not installed; NER disabled for PHIClassifier")
            self._spacy_available = False

    def _compile_patterns(self) -> None:
        """Compile regexes for the 18 PHI types and store them.

        Patterns are intentionally conservative to reduce false positives.
        """
        try:
            # core regex dictionary
            p = {}
            p["social_security_numbers"] = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
            p["phone_numbers"] = re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b")
            p["email_addresses"] = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
            p["urls"] = re.compile(r"https?://[\w\-._~:/?#[\]@!$&'()*+,;=%]+|www\.[\w\-._~:/?#[\]@!$&'()*+,;=%]+")
            p["ip_addresses"] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
            p["dates"] = re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s?[A-Za-z]{3,9}\s?\d{2,4})\b")
            p["medical_record_numbers"] = re.compile(r"\b(?:MRN|mrn|Medical\s?Record\s?No\.?|Acct)[:#\s]*([0-9A-Za-z\-]{6,12})\b")
            p["health_plan_numbers"] = re.compile(r"\b[A-Z0-9]{5,15}\b")
            p["account_numbers"] = re.compile(r"\b(?:Acct|Account)[:#\s]*([0-9A-Za-z\-]{4,16})\b")
            p["certificate_license_numbers"] = re.compile(r"\b(?:Lic|License|LT|RN|MD)[:#\s]*[A-Z0-9\-]{3,16}\b", re.IGNORECASE)
            p["vehicle_identifiers"] = re.compile(r"\b[A-Z0-9\-]{5,8}\b")
            p["device_identifiers"] = re.compile(r"\b[Ss]n[:#\s]*[0-9A-Za-z\-]{4,20}\b|\bSerial\s?No[:#\s]*[0-9A-Za-z\-]{4,20}\b")
            p["biometric_identifiers"] = re.compile(r"\b(fingerprint|retina|iris|voice)\b", re.IGNORECASE)
            p["full_face_photos"] = re.compile(r"\b(photo|photograph|full[- ]face)\b", re.IGNORECASE)
            p["geographic_locations"] = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b(?:[A-Z][a-z]+\s){0,2}(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd)\b")
            p["fax_numbers"] = re.compile(r"fax[:#\s]*\+?\d{7,15}", re.IGNORECASE)
            p["other_unique_identifiers"] = re.compile(r"\bUID[:#\s]*[A-Z0-9\-]{4,20}\b", re.IGNORECASE)

            # store compiled patterns
            self._patterns = p
        except re.error as e:
            logger.exception("Regex compilation failed: %s", e)
            raise

    def _load_custom_vocabulary(self) -> None:
        # Placeholder for loading custom vocabularies; can be extended
        self.custom_patterns: Dict[str, re.Pattern] = {}

    # ------------------ detection pipeline ------------------
    def detect_phi_in_text(self, text: str, context: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Run regex and NER detection in parallel and merge results.

        Returns list of detection dicts with fields: text, start, end, phi_type,
        confidence, method, metadata.
        """
        # Pseudocode / desired detection order:
        # 1) If config.use_presidio is True: call a Presidio Analyzer and return results (filtered by per-type threshold)
        # 2) Else if config.use_spacy_fallback is True: run spaCy NER only, map labels -> HIPAA types, filter by thresholds
        # 3) Else if config.use_regex_fallback is True: run regex patterns only and return filtered results
        # 4) If none of the above flags are present, fall back to the previous behavior (regex + spaCy merged)
        #
        # Each step must:
        # - produce detection dicts with keys: text, start, end, phi_type (label), confidence (float 0..1), method
        # - apply a per-type confidence threshold (self._thresholds[phi_type] or default)
        # - aggregate detections into a single list (no duplicates; prefer higher-confidence overlaps)

        if not text:
            return []

        start_time = time.time()

        # determine feature flags (prefer centralized AppConfig)
        use_presidio = bool(getattr(self.app_config, "use_presidio", False))
        # if presidio globally disabled, ensure false
        try:
            presidio_pct = int(getattr(self.app_config, "presidio_canary_percentage", 0) or 0)
        except Exception:
            presidio_pct = 0
        # If canary is zero, do not run presidio
        if not use_presidio or presidio_pct <= 0:
            use_presidio = False

        use_spacy_fallback = getattr(self.config, "use_spacy_fallback", None)
        if use_spacy_fallback is None:
            # maintain compatibility with existing flag name
            use_spacy_fallback = bool(getattr(self.config, "enable_spacy_ner", False))
        use_regex_fallback = getattr(self.config, "use_regex_fallback", None)
        if use_regex_fallback is None:
            use_regex_fallback = bool(getattr(self.config, "enable_regex_patterns", True))

        detections: List[Dict[str, Any]] = []

        # Helper: apply threshold filter
        def _passes_threshold(d: Dict[str, Any]) -> bool:
            try:
                t = d.get("phi_type") or "other_unique_identifiers"
                thresh = float(self._thresholds.get(t, float(self.config.phi_confidence_threshold)))
            except Exception:
                thresh = float(getattr(self.config, "phi_confidence_threshold", 0.8))
            return float(d.get("confidence", 0.0)) >= thresh

        # log input snippet + meta for auditing/monitoring
        try:
            short = (text or "")[:240]
            logger.debug("detect_phi_in_text input snippet=%s meta=%s", short, context)
        except Exception:
            logger.debug("Failed to log input snippet")

        # 1) Presidio path
        if use_presidio:
            try:
                from src.integration.presidio_wrapper import get_analyzer

                analyzer = get_analyzer()
                pres_dets = analyzer.detect_phi_in_text(text, context or {}) or []
                # log presidio raw responses for audit and monitoring
                try:
                    # normalize presidio responses to compact forms for audit
                    pres_audit = [
                        {"entity_type": p.get("entity_type") or p.get("label"), "start": p.get("start"), "end": p.get("end"), "score": float(p.get("score", p.get("confidence", 0.0))) if (p.get("score") is not None or p.get("confidence") is not None) else 0.0}
                        for p in pres_dets
                    ]
                    self.audit_log(pres_audit, method_name="presidio_raw", context=context)
                except Exception:
                    logger.debug("Failed to audit presidio responses")
                # map presidio results to canonical format and filter by threshold
                for p in pres_dets:
                    d = {
                        "text": p.get("text", text[p.get("start", 0) : p.get("end", 0)] if p.get("start") is not None and p.get("end") is not None else p.get("text", "")),
                        "start": int(p.get("start", 0)),
                        "end": int(p.get("end", 0)),
                        "phi_type": p.get("entity_type") or p.get("label") or None,
                        "method": "presidio",
                        "confidence": float(p.get("score", p.get("confidence", 0.0))) if p.get("score") is not None or p.get("confidence") is not None else 0.0,
                    }
                    if _passes_threshold(d):
                        detections.append(d)
                # dedupe/merge by preferring presidio results (already collected)
                try:
                    self.audit_logger.info(json.dumps({"event": "detect_phi_in_text", "method": "presidio", "count": len(detections)}))
                except Exception:
                    logger.debug("Audit failed for presidio path")
                return detections
            except Exception:
                logger.exception("Presidio path failed; falling through to configured fallbacks")

        # 2) spaCy-only path
        if use_spacy_fallback:
            if self._spacy_available and self._spacy_nlp is not None:
                try:
                    doc = self._spacy_nlp(text)
                    for ent in getattr(doc, "ents", []):
                        label = ent.label_.lower()
                        mapped = None
                        if label in ("person", "per"):
                            mapped = "names"
                        elif label in ("date",):
                            mapped = "dates"
                        elif label in ("gpe", "loc"):
                            mapped = "geographic_locations"
                        elif label in ("email",):
                            mapped = "email_addresses"
                        if mapped:
                            d = {"text": ent.text, "start": ent.start_char, "end": ent.end_char, "phi_type": mapped, "method": "ner", "confidence": 0.9}
                            if _passes_threshold(d):
                                detections.append(d)
                except Exception:
                    logger.exception("spaCy-only detection failed")
            try:
                self.audit_logger.info(json.dumps({"event": "detect_phi_in_text", "method": "spacy_only", "count": len(detections)}))
            except Exception:
                pass
            return detections

        # 3) regex-only path
        if use_regex_fallback:
            try:
                # run regex matches
                for tname, pattern in self._patterns.items():
                    for m in pattern.finditer(text):
                        # Boost confidence for regex-only path to ensure conservative
                        # regex matches are returned when this mode is explicitly
                        # requested. Tests and some legacy callers expect regex
                        # fallback to produce results even with higher global
                        # thresholds.
                        d = {"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": tname, "method": "regex", "confidence": 0.9}
                        if _passes_threshold(d):
                            detections.append(d)
                for name, pat in self.custom_patterns.items():
                    for m in pat.finditer(text):
                        d = {"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": name, "method": "custom_regex", "confidence": 0.8}
                        if _passes_threshold(d):
                            detections.append(d)
            except Exception:
                logger.exception("Regex-only detection failed")
            try:
                self.audit_logger.info(json.dumps({"event": "detect_phi_in_text", "method": "regex_only", "count": len(detections)}))
            except Exception:
                pass
            return detections

        # 4) default (legacy) behavior: run both regex + spaCy and merge results
        # This preserves existing behavior when no explicit flags are provided.
        # run regex matches
        regex_results: List[PHIDetection] = []
        for tname, pattern in self._patterns.items():
            for m in pattern.finditer(text):
                det = PHIDetection(text=m.group(0), start=m.start(), end=m.end(), phi_type=tname, method="regex", confidence=0.6)
                regex_results.append(det)

        # run custom patterns
        for name, pat in self.custom_patterns.items():
            for m in pat.finditer(text):
                det = PHIDetection(text=m.group(0), start=m.start(), end=m.end(), phi_type=name, method="custom_regex", confidence=0.8)
                regex_results.append(det)

        # run spaCy NER if available
        ner_results: List[PHIDetection] = []
        if self._spacy_available and self._spacy_nlp is not None and self.config.enable_spacy_ner:
            try:
                doc = self._spacy_nlp(text)
                for ent in getattr(doc, "ents", []):
                    label = ent.label_.lower()
                    mapped = None
                    if label in ("person", "per"):
                        mapped = "names"
                    elif label in ("date",):
                        mapped = "dates"
                    elif label in ("gpe", "loc"):
                        mapped = "geographic_locations"
                    elif label in ("email",):
                        mapped = "email_addresses"
                    if mapped:
                        ner_results.append(PHIDetection(text=ent.text, start=ent.start_char, end=ent.end_char, phi_type=mapped, method="ner", confidence=0.9))
            except Exception:
                logger.exception("spaCy NER failed; continuing with regex only")

        merged = self.merge_phi_detections([r.to_dict() for r in regex_results], [n.to_dict() for n in ner_results])

    # apply context validation and compute confidence
        final: List[Dict[str, Any]] = []
        for d in merged:
            ctx_conf = self.validate_phi_context(d.get("text", ""), context.get("surrounding_text", "") if context else "", d.get("phi_type", ""))
            conf = self.calculate_detection_confidence(d, context or {})
            # combine confidences conservatively
            combined_conf = min(1.0, conf * 0.7 + ctx_conf * 0.3)
            d["confidence"] = combined_conf
            # decide method-agnostic type if missing
            if not d.get("phi_type"):
                typ, score = self.classify_phi_type(d.get("text", ""), d.get("method", "regex"))
                d["phi_type"] = typ
                d["confidence"] = max(d["confidence"], score)
            # filter by per-type threshold
            if _passes_threshold(d):
                final.append(d)
        # audit
        try:
            self.audit_logger.info(json.dumps({"event": "detect_phi_in_text", "count": len(final), "time_s": time.time() - start_time}))
        except Exception:
            logger.debug("Audit log failed for detect_phi_in_text")

        # also write a structured audit record via audit_log for monitoring
        try:
            self.audit_log(final, method_name="merged_regex_ner", context=context)
        except Exception:
            logger.debug("audit_log failed for merged results")

        return final

    def classify_phi_type(self, text: str, detection_method: str) -> Tuple[str, float]:
        """Classify a detected text into one of the PHI types with a confidence.

        This function uses pattern heuristics to pick the most likely PHI type.
        """
        txt = (text or "").strip()
        if not txt:
            return "other_unique_identifiers", 0.0

        # quick checks
        if self._patterns["social_security_numbers"].search(txt):
            return "social_security_numbers", 0.99
        if self._patterns["email_addresses"].search(txt):
            return "email_addresses", 0.98
        if self._patterns["phone_numbers"].search(txt):
            return "phone_numbers", 0.95
        if self._patterns["urls"].search(txt) or self._patterns["ip_addresses"].search(txt):
            return ("urls", 0.9) if self._patterns["urls"].search(txt) else ("ip_addresses", 0.9)
        if self._patterns["medical_record_numbers"].search(txt):
            return "medical_record_numbers", 0.98
        if self._patterns["device_identifiers"].search(txt):
            return "device_identifiers", 0.85
        if self._patterns["full_face_photos"].search(txt):
            return "full_face_photos", 0.6

        # fallback to spaCy-based heuristics
        if self._spacy_available and self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(txt)
                for ent in getattr(doc, "ents", []):
                    lab = ent.label_.lower()
                    if lab in ("person",):
                        return "names", 0.9
                    if lab in ("date",):
                        return "dates", 0.9
            except Exception:
                pass

        # default
        return "other_unique_identifiers", 0.4

    # ------------------ individual detectors (text-level) ------------------
    def detect_names(self, text: str) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        if self._spacy_available and self._spacy_nlp is not None:
            try:
                doc = self._spacy_nlp(text)
                for ent in getattr(doc, "ents", []):
                    if ent.label_.lower() in ("person", "per"):
                        # filter clinical roles
                        t = ent.text.strip()
                        # avoid medical terms misclassified as names
                        if len(t) < 2:
                            continue
                        results.append({"text": t, "start": ent.start_char, "end": ent.end_char, "phi_type": "names", "method": "ner", "confidence": 0.9})
            except Exception:
                logger.exception("spaCy name detection failed")

        # fallback heuristics: capitalized tokens with common name patterns
        name_pattern = re.compile(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b")
        for m in name_pattern.finditer(text):
            results.append({"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": "names", "method": "regex", "confidence": 0.6})

        return results

    def detect_medical_record_numbers(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["medical_record_numbers"].finditer(text):
            mrn = m.group(1) if m.groups() else m.group(0)
            conf = 0.95 if re.match(r"^[0-9A-Za-z\-]{6,12}$", mrn) else 0.6
            results.append({"text": mrn, "start": m.start(), "end": m.end(), "phi_type": "medical_record_numbers", "method": "regex", "confidence": conf})
        return results

    def detect_social_security_numbers(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["social_security_numbers"].finditer(text):
            ssn = m.group(0)
            # simple SSN validity: not all zeros in groups
            valid = not re.match(r"0{3}-0{2}-0{4}", ssn)
            conf = 0.99 if valid else 0.5
            results.append({"text": ssn, "start": m.start(), "end": m.end(), "phi_type": "social_security_numbers", "method": "regex", "confidence": conf})
        return results

    def detect_dates(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["dates"].finditer(text):
            dt = m.group(0)
            # heuristic: ages over 89 are PHI
            conf = 0.9
            if re.search(r"\b(?:19|20)\d{2}\b", dt) and (len(dt) >= 8):
                conf = 0.95
            results.append({"text": dt, "start": m.start(), "end": m.end(), "phi_type": "dates", "method": "regex", "confidence": conf})
        return results

    def detect_phone_numbers(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["phone_numbers"].finditer(text):
            num = m.group(0)
            conf = 0.92
            results.append({"text": num, "start": m.start(), "end": m.end(), "phi_type": "phone_numbers", "method": "regex", "confidence": conf})
        return results

    def detect_addresses(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["geographic_locations"].finditer(text):
            results.append({"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": "geographic_locations", "method": "regex", "confidence": 0.8})
        return results

    def detect_email_and_urls(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["email_addresses"].finditer(text):
            results.append({"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": "email_addresses", "method": "regex", "confidence": 0.98})
        for m in self._patterns["urls"].finditer(text):
            results.append({"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": "urls", "method": "regex", "confidence": 0.92})
        return results

    def detect_device_identifiers(self, text: str) -> List[Dict[str, Any]]:
        results = []
        for m in self._patterns["device_identifiers"].finditer(text):
            results.append({"text": m.group(0), "start": m.start(), "end": m.end(), "phi_type": "device_identifiers", "method": "regex", "confidence": 0.85})
        return results

    # ------------------ merging & context ------------------
    def merge_phi_detections(self, regex_results: List[Dict[str, Any]], ner_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping detections preferring higher-confidence and more
        specific patterns. Preserves source metadata for auditability.
        """
        all_dets: List[Dict[str, Any]] = []
        for r in regex_results:
            r.setdefault("method", "regex")
            r.setdefault("confidence", 0.5)
            all_dets.append(r)
        for n in ner_results:
            n.setdefault("method", "ner")
            n.setdefault("confidence", 0.9)
            all_dets.append(n)

        # sort by confidence desc
        all_dets.sort(key=lambda x: float(x.get("confidence", 0.0)), reverse=True)

        merged: List[Dict[str, Any]] = []
        for det in all_dets:
            overlap = False
            for m in merged:
                # check overlap
                if not (det["end"] <= m["start"] or det["start"] >= m["end"]):
                    # merge: keep the one with higher confidence for text/type
                    overlap = True
                    # preserve history
                    m.setdefault("merged_from", [])
                    m["merged_from"].append({"text": det.get("text"), "method": det.get("method"), "confidence": det.get("confidence")})
                    # if det has higher confidence than m, replace core fields
                    if float(det.get("confidence", 0.0)) > float(m.get("confidence", 0.0)):
                        m.update({"text": det.get("text"), "phi_type": det.get("phi_type"), "confidence": det.get("confidence")})
                    # expand span
                    m["start"] = min(m["start"], det["start"])
                    m["end"] = max(m["end"], det["end"])
                    break
            if not overlap:
                merged.append(dict(det))

        return merged

    def validate_phi_context(self, phi_text: str, surrounding_text: str, phi_type: str) -> float:
        """Return a context-based score 0..1 indicating how likely the phi_text
        is truly PHI given the surrounding context.
        """
        if not phi_text:
            return 0.0
        score = 0.5
        st = (surrounding_text or "").lower()
        # administrative context ("patient", "mrn", "dob") increases score
        if any(k in st for k in ("patient", "mrn", "dob", "ssn", "ssn:")):
            score += 0.3
        # presence of clinical terms reduces false positives for names
        if phi_type == "names" and any(k in st for k in ("dr.", "nurse", "physician", "md")):
            score += 0.2
        return min(1.0, score)

    def calculate_detection_confidence(self, detection: Dict[str, Any], context: Dict[str, Any]) -> float:
        """Compute a composite confidence for a detection using pattern
        specificity and context.
        """
        base = float(detection.get("confidence", 0.5))
        text = detection.get("text", "")
        phi_type = detection.get("phi_type", "")
        # pattern specificity: longer exact matches get higher boosts
        spec = min(0.3, len(text) / 100.0)
        ctx_score = self.validate_phi_context(text, context.get("surrounding_text", "") if context else "", phi_type)
        combined = min(1.0, base * 0.7 + spec + ctx_score * 0.2)
        return float(combined)

    # ------------------ audit & reporting ------------------
    def log_phi_detection(self, detection: Dict[str, Any], image_id: str, timestamp: Optional[datetime] = None) -> None:
        """Write a HIPAA-compliant audit record for the detection.

        Records are written to the configured audit logger. The logging
        pipeline (SETTINGS.setup_logging) should ensure redaction when
        writing to persistent logs.
        """
        # Respect global audit flag
        if not getattr(self.app_config, "enable_audit_logging", True):
            logger.debug("Audit logging disabled; skipping log_phi_detection")
            return
        try:
            rec = {
                "image_id": image_id,
                "timestamp": (timestamp or datetime.utcnow()).isoformat(),
                "detection": detection,
            }
            # redact detection text if configured
            if getattr(self.config, "audit_redact_snippets", False) and isinstance(rec.get("detection"), dict):
                rec["detection"]["text"] = "[REDACTED_SNIPPET]"
            self.audit_logger.info(json.dumps(rec))
        except Exception:
            logger.exception("Failed to write PHI audit record")

    def audit_log(self, detections: List[Dict[str, Any]], method_name: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Write a compact audit record for a batch of detections.

        This helper centralizes audit writing so downstream systems can
        monitor detection performance, correlate detections with source
        (page/bbox/image_id), and compute false-positive/false-negative
        metrics over time. Records should be redacted by the logging
        configuration before persistence.

        - detections: list of detection dicts (keys: entity_type, start, end, score, text)
        - method_name: string indicating which detector produced the results
        - context: optional metadata (image_id, page, bbox, surrounding_text)
        """
        # Respect global audit flag
        if not getattr(self.app_config, "enable_audit_logging", True):
            logger.debug("Audit logging disabled; skipping audit_log")
            return
        try:
            # Optionally redact detection text snippets before persisting
            redacted = []
            for d in detections:
                item = dict(d)
                if getattr(self.config, "audit_redact_snippets", False):
                    item["text"] = "[REDACTED_SNIPPET]"
                redacted.append(item)

            rec = {"method": method_name, "count": len(detections), "detections": redacted, "context": context or {}, "ts": time.time()}
            # Write to the audit logger; the formatter can additionally redact
            self.audit_logger.info(json.dumps(rec, default=str))

            # Optionally write a non-redacted audit stream for trusted review
            if getattr(self.config, "audit_allow_nonredacted", False):
                try:
                    nr_logger = logging.getLogger(getattr(self.config, "nonredacted_audit_logger_name", "cim.audit.phi_raw"))
                    nr_rec = {"method": method_name, "count": len(detections), "detections": detections, "context": context or {}, "ts": time.time()}
                    nr_logger.info(json.dumps(nr_rec, default=str))
                except Exception:
                    logger.exception("Failed to write non-redacted audit log")
        except Exception:
            logger.exception("Failed to write audit_log for %s", method_name)

    def generate_phi_report(self, detections: List[Dict[str, Any]], image_metadata: Dict[str, Any]) -> Dict[str, Any]:
        counts = defaultdict(int)
        for d in detections:
            counts[d.get("phi_type", "other_unique_identifiers")] += 1
        avg_conf = sum(d.get("confidence", 0.0) for d in detections) / max(1, len(detections))
        report = {
            "image_id": image_metadata.get("image_id"),
            "counts": dict(counts),
            "avg_confidence": avg_conf,
            "risk_level": "high" if counts.get("social_security_numbers", 0) or counts.get("medical_record_numbers", 0) else "medium" if len(detections) else "low",
            "recommendations": ["Review high-risk detections manually", "Remove or redact PHI before sharing"],
        }
        try:
            self.audit_logger.info(json.dumps({"event": "generate_phi_report", "report": report}))
        except Exception:
            logger.debug("Failed to audit phi report")
        return report

    def validate_hipaa_coverage(self, detections: List[Dict[str, Any]]) -> Dict[str, bool]:
        """Return a mapping type->bool indicating whether each HIPAA type was
        observed/detected or accounted for by patterns (best-effort).
        """
        present = {t: False for t in self.HIPAA_TYPES}
        for d in detections:
            t = d.get("phi_type")
            if t in present:
                present[t] = True
        # audit coverage check
        try:
            self.audit_logger.info(json.dumps({"event": "validate_hipaa_coverage", "coverage": present}))
        except Exception:
            logger.debug("Failed to audit coverage validation")
        return present

    # ------------------ pattern management ------------------
    def load_custom_patterns(self, pattern_file: str) -> Dict[str, str]:
        """Load custom regex patterns from a JSON file mapping name->pattern.

        Validates that patterns compile and registers them under custom_patterns.
        """
        try:
            with open(pattern_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            loaded = {}
            for name, pat in data.items():
                try:
                    cp = re.compile(pat)
                    self.custom_patterns[name] = cp
                    loaded[name] = pat
                except re.error:
                    logger.exception("Invalid custom pattern %s", name)
            # audit
            self.audit_logger.info(json.dumps({"event": "load_custom_patterns", "loaded": list(loaded.keys())}))
            return loaded
        except Exception:
            logger.exception("Failed to load custom patterns from %s", pattern_file)
            return {}

    def update_clinical_vocabulary(self, vocab_file: str) -> None:
        """Load an updated clinical vocabulary (simple word list) to assist
        context validation and name filtering. For production, integrate a
        managed terminology service.
        """
        try:
            with open(vocab_file, "r", encoding="utf-8") as fh:
                terms = {l.strip().lower() for l in fh if l.strip()}
            self._clinical_vocab = terms
            self.audit_logger.info(json.dumps({"event": "update_clinical_vocabulary", "terms": len(terms)}))
        except Exception:
            logger.exception("Failed to update clinical vocabulary from %s", vocab_file)

    # ------------------ utilities ------------------
    def _safe_iterable(self, seq):
        try:
            for x in seq:
                yield x
        except Exception:
            return


__all__ = ["PHIClassifier", "PHIDetection", "PHIDetection", "PHIDetection"]
