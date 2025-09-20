from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field

from jose import JWTError, jwt

from src.core.config import AppConfig, SETTINGS
from src.integration.presidio_wrapper import get_analyzer
from src.phi_detection.phi_classifier import PHIClassifier
from src.integration.policy_matrix import POLICY_MATRIX

# Optional DB support (Postgres) via SQLAlchemy
_DB_ENGINE = None
_DB_CONN = None
_DB_AVAILABLE = False
try:
    from sqlalchemy import Table, Column, Integer, String, MetaData, DateTime, create_engine
    from sqlalchemy.exc import SQLAlchemyError
    from datetime import datetime

    DATABASE_URL = os.environ.get("DETECTOR_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if DATABASE_URL:
        _DB_ENGINE = create_engine(DATABASE_URL)
        _DB_CONN = _DB_ENGINE.connect()
        metadata = MetaData()
        mappings = Table(
            "anonymization_mappings",
            metadata,
            Column("id", Integer, primary_key=True),
            Column("original_text", String),
            Column("anonymized_text", String),
            Column("entity_type", String),
            Column("policy", String),
            Column("created_at", DateTime),
        )
        try:
            metadata.create_all(_DB_ENGINE)
            _DB_AVAILABLE = True
        except Exception:
            _DB_AVAILABLE = False
except Exception:
    _DB_AVAILABLE = False

logger = logging.getLogger("cim.api.detector")

security = HTTPBearer()

# JWT settings
JWT_SECRET = os.environ.get("DETECTOR_JWT_SECRET", os.environ.get("GATEWAY_JWT_SECRET", "changeme"))
JWT_ALGORITHM = os.environ.get("DETECTOR_JWT_ALGORITHM", "HS256")


class DetectRequest(BaseModel):
    text: str = Field(..., description="Input text to analyze for PHI")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    use_presidio: Optional[bool] = Field(None, description="Override AppConfig to run Presidio")


class DetectionResult(BaseModel):
    text: str
    start: int
    end: int
    phi_type: Optional[str]
    method: str
    confidence: float


class DetectResponse(BaseModel):
    detections: List[DetectionResult]
    count: int


class AnonymizeRequest(BaseModel):
    text: str = Field(...)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    policy_overrides: Optional[Dict[str, Dict[str, Any]]] = Field(default_factory=dict)


class AnonymizeMapping(BaseModel):
    original: str
    anonymized: str
    start: int
    end: int
    anonymized_start: Optional[int]
    anonymized_end: Optional[int]
    phi_type: Optional[str]
    policy: Optional[str]


class AnonymizeResponse(BaseModel):
    original_text: str
    anonymized_text: str
    mappings: List[AnonymizeMapping]



app = FastAPI(title="CIM PHI Detector")


def verify_jwt(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning("JWT validation failed: %s", str(e))
        raise HTTPException(status_code=401, detail="Invalid authentication token")


def require_roles(*allowed_roles: str):
    def inner(credentials: HTTPAuthorizationCredentials = Depends(security)):
        payload = verify_jwt(credentials)
        roles = payload.get("roles", []) if isinstance(payload, dict) else []
        if not any(r in roles for r in allowed_roles):
            raise HTTPException(status_code=403, detail="Insufficient role")
        return payload

    return inner


def _redact_for_log(s: str) -> str:
    # Keep only short non-sensitive snippet for logs
    if not s:
        return ""
    s = s.strip()
    return (s[:120] + "...") if len(s) > 120 else s


def _compute_orig_to_new_index_map(orig: str, new: str):
    from difflib import SequenceMatcher

    sm = SequenceMatcher(None, orig or "", new or "")
    orig_to_new = [None] * (len(orig) + 1)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for offset in range(i2 - i1):
                oi = i1 + offset
                orig_to_new[oi] = j1 + offset
            orig_to_new[i2] = j2
        elif tag == "replace":
            for oi in range(i1, i2 + 1):
                orig_to_new[oi] = j1
        elif tag == "delete":
            for oi in range(i1, i2 + 1):
                orig_to_new[oi] = j1
        elif tag == "insert":
            if i1 >= 0 and i1 <= len(orig):
                orig_to_new[i1] = j1

    last_known = 0
    for i in range(len(orig) + 1):
        if orig_to_new[i] is None:
            orig_to_new[i] = last_known
        else:
            last_known = orig_to_new[i]
    return orig_to_new


def _remap_offsets_after_anonymize(page_text: str, anonymized_text: str, regions: List[Dict], detections: List[Dict]) -> List[Dict]:
    try:
        if not isinstance(page_text, str) or not isinstance(anonymized_text, str):
            return detections

        # build page-level word_boxes with absolute char offsets
        page_word_boxes = []
        cursor = 0
        for r in regions:
            rtext = (r.get("text") or "")
            if cursor != 0 and rtext:
                cursor += 1
            region_start = cursor
            word_boxes = r.get("word_boxes") or []
            for w in word_boxes:
                try:
                    ws = int(w.get("start_char", 0))
                    we = int(w.get("end_char", 0))
                    abs_start = region_start + ws
                    abs_end = region_start + we
                    pb = dict(w)
                    pb["start_char"] = abs_start
                    pb["end_char"] = abs_end
                    page_word_boxes.append(pb)
                except Exception:
                    continue
            cursor = region_start + len(rtext)

        index_map = _compute_orig_to_new_index_map(page_text or "", anonymized_text or "")

        def map_span(s: int, e: int):
            s = max(0, min(len(page_text), int(s or 0)))
            e = max(0, min(len(page_text), int(e or 0)))
            new_s = int(index_map[s])
            if e > 0:
                new_e = int(index_map[max(0, e - 1)]) + 1
            else:
                new_e = int(index_map[0])
            return new_s, new_e

        # attempt to re-map bbox using mapping util if available
        try:
            from src.ocr.mapping import map_text_spans_to_image_bboxes
        except Exception:
            map_text_spans_to_image_bboxes = None

        for d in detections:
            try:
                orig_start = d.get("start") if d.get("start") is not None else d.get("start_char")
                orig_end = d.get("end") if d.get("end") is not None else d.get("end_char")
                orig_start = int(orig_start) if orig_start is not None else None
                orig_end = int(orig_end) if orig_end is not None else None
            except Exception:
                orig_start, orig_end = None, None

            if orig_start is not None and orig_end is not None:
                new_s, new_e = map_span(orig_start, orig_end)
                d["anonymized_start"] = new_s
                d["anonymized_end"] = new_e
                d["start"] = new_s
                d["end"] = new_e
            else:
                txt = (d.get("text") or "").strip()
                if txt:
                    pos = anonymized_text.find(txt)
                    if pos >= 0:
                        d["start"] = pos
                        d["end"] = pos + len(txt)
                        d["anonymized_start"] = d["start"]
                        d["anonymized_end"] = d["end"]

            try:
                if map_text_spans_to_image_bboxes is not None:
                    mapped = map_text_spans_to_image_bboxes([
                        {"text": d.get("text", ""), "start": int(d.get("start", 0)), "end": int(d.get("end", 0))}
                    ], page_word_boxes, None, page_number=1)
                    if mapped and mapped[0].get("bbox"):
                        m = mapped[0]
                        d["bbox"] = m.get("bbox")
                        d["page_number"] = m.get("page_number", 1)
            except Exception:
                logger.exception("map_text_spans_to_image_bboxes failed during remap; leaving bbox unchanged for detection %s", d.get("text"))

        return detections
    except Exception as e:
        logger.exception("Failed to remap offsets after anonymization: %s", e)
        return detections


def _store_mapping_db(original: str, anonymized: str, entity_type: str, policy: str):
    if not _DB_AVAILABLE:
        return
    try:
        ins = mappings.insert().values(original_text=original, anonymized_text=anonymized, entity_type=entity_type, policy=policy, created_at=datetime.utcnow())
        _DB_CONN.execute(ins)
    except Exception:
        logger.exception("Failed to store mapping in DB")


def _apply_fallback_anonymizer(text: str, detections: List[Dict], policy_matrix: Dict) -> tuple[str, List[Dict]]:
    # Build deterministic mapping per original value + type to preserve referential integrity
    replacement_map: Dict[str, str] = {}
    out = []
    cursor = 0
    parts = []
    last_i = 0
    # Sort detections by start to build replaced string
    dets = sorted([d for d in detections if d.get("start") is not None and d.get("end") is not None], key=lambda x: int(x.get("start", 0)))
    for d in dets:
        s = int(d.get("start", 0))
        e = int(d.get("end", 0))
        orig = text[s:e]
        ent = (d.get("phi_type") or d.get("entity_type") or "OTHER_UNIQUE_IDENTIFIER").upper()
        pm = policy_matrix.get(ent) or policy_matrix.get(ent.upper()) or None
        op = (pm.get("operator") if pm else "replace")
        params = pm.get("params", {}) if pm else {}

        # Determine replacement value deterministically
        key = f"{ent}||{orig}"
        if key in replacement_map:
            repl = replacement_map[key]
        else:
            if op == "replace":
                repl = params.get("new_value") or f"[REDACTED_{ent}]"
            elif op == "mask":
                masking_char = params.get("masking_char", "X")
                chars_to_mask = int(params.get("chars_to_mask", max(1, len(orig) - 4)))
                from_end = bool(params.get("from_end", True))
                if any(c.isdigit() for c in orig):
                    # mask digits only preserving separators
                    digits = [c for c in orig if c.isdigit()]
                    to_mask = min(len(digits), chars_to_mask)
                    # build masked digits preserving non-digits positions
                    masked = []
                    masked_count = 0
                    if from_end:
                        # mask from end: replace last to_mask digits
                        rev_digits = digits[::-1]
                        masked_digits = [masking_char if i < to_mask else d for i, d in enumerate(rev_digits)]
                        masked_digits = masked_digits[::-1]
                        di = 0
                        for ch in orig:
                            if ch.isdigit():
                                masked.append(masked_digits[di])
                                di += 1
                            else:
                                masked.append(ch)
                        repl = "".join(masked)
                    else:
                        # mask from start
                        masked_digits = [masking_char if i < to_mask else d for i, d in enumerate(digits)]
                        di = 0
                        for ch in orig:
                            if ch.isdigit():
                                masked.append(masked_digits[di])
                                di += 1
                            else:
                                masked.append(ch)
                        repl = "".join(masked)
                else:
                    # non-digit: simple char mask preserving length
                    repl = (masking_char * min(len(orig), chars_to_mask)) + orig[min(len(orig), chars_to_mask):]
            else:
                repl = params.get("new_value") or f"[REDACTED_{ent}]"
            replacement_map[key] = repl

        # append untouched segment
        parts.append(text[last_i:s])
        parts.append(repl)
        # push mapping info
        out.append({"original": orig, "anonymized": repl, "start": s, "end": e, "phi_type": ent, "policy": op})
        last_i = e

    parts.append(text[last_i:])
    anonymized_text = "".join(parts)

    # remap offsets
    remapped = _remap_offsets_after_anonymize(text, anonymized_text, [], dets)
    # merge anonymized positions into out list
    for info, dnew in zip(out, remapped):
        info.update({"anonymized_start": dnew.get("start"), "anonymized_end": dnew.get("end")})
        # persist mapping to DB if available
        try:
            _store_mapping_db(info.get("original"), info.get("anonymized"), info.get("phi_type"), info.get("policy"))
        except Exception:
            pass

    return anonymized_text, out


@app.post("/api/v1/detect", response_model=DetectResponse)
async def detect_phi(req: DetectRequest, request: Request, user=Depends(require_roles("admin", "service_account"))):
    # Use AppConfig from codebase to configure detectors
    cfg: AppConfig = SETTINGS if isinstance(SETTINGS, AppConfig) else AppConfig()

    # Respect per-request override for presidio
    if req.use_presidio is not None:
        # temporary override on the config object attribute used by PHIClassifier
        try:
            cfg.use_presidio = bool(req.use_presidio)
        except Exception:
            pass

    # instantiate detectors
    classifier = PHIClassifier(cfg)
    analyzer = get_analyzer()

    text = (req.text or "")
    if not text.strip():
        raise HTTPException(status_code=400, detail="Text must not be empty")

    # Log safe snippet only
    logger.info("detect request user=%s snippet=%s", user.get("sub", "unknown"), _redact_for_log(text))

    try:
        # If the app config indicates Presidio should be used, prefer it
        use_presidio = bool(getattr(cfg, "use_presidio", False))
        # allow request to force presidio if use_presidio True or override set above
        if use_presidio:
            pres_results = analyzer.detect_phi_in_text(text, req.context or {}) or []
            # Normalize presidio output to expected detection format
            detections = []
            for p in pres_results:
                det = {
                    "text": p.get("text", text[p.get("start", 0) : p.get("end", 0)]) if p.get("text") is not None else text[p.get("start", 0) : p.get("end", 0)],
                    "start": int(p.get("start", 0)),
                    "end": int(p.get("end", 0)),
                    "phi_type": p.get("entity_type") or p.get("label"),
                    "method": "presidio",
                    "confidence": float(p.get("confidence", p.get("score", 0.0))) if (p.get("confidence") is not None or p.get("score") is not None) else 0.0,
                }
                detections.append(det)
        else:
            # legacy classifier path
            detections = classifier.detect_phi_in_text(text, req.context or {})

        # sanitize detections for response (no internal metadata)
        resp_dets = []
        for d in detections:
            resp_dets.append({
                "text": d.get("text", ""),
                "start": int(d.get("start", 0)),
                "end": int(d.get("end", 0)),
                "phi_type": d.get("phi_type"),
                "method": d.get("method", ""),
                "confidence": float(d.get("confidence", 0.0)),
            })

        # Audit via classifier audit_log helper but ensure audit config will redact if needed
        try:
            classifier.audit_log(resp_dets, method_name=("presidio" if use_presidio else "legacy"), context=req.context or {})
        except Exception:
            logger.debug("audit_log failed")

        return DetectResponse(detections=resp_dets, count=len(resp_dets))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("detect_phi failed: %s", str(e))
        raise HTTPException(status_code=500, detail="Internal detection error")


    @app.post("/api/v1/anonymize", response_model=AnonymizeResponse)
    async def anonymize(req: AnonymizeRequest, request: Request, user=Depends(require_roles("admin", "service_account"))):
        cfg: AppConfig = SETTINGS if isinstance(SETTINGS, AppConfig) else AppConfig()
        text = (req.text or "")
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text must not be empty")

            # Respect feature flag: do not perform dummy data replacement when disabled
            if not getattr(cfg, "enable_dummy_data_replacement", False):
                logger.info("Dummy data replacement disabled via feature flag; returning original text")
                return AnonymizeResponse(anonymized_text=text, mappings=[])

        logger.info("anonymize request user=%s snippet=%s", user.get("sub", "unknown"), _redact_for_log(text))

        # run detections (prefer presidio analyzer for entities)
        analyzer = get_analyzer()
        try:
            detections = analyzer.detect_phi_in_text(text, req.context or {}) or []
        except Exception:
            logger.exception("Presidio analyzer failed during anonymize; falling back to classifier")
            classifier = PHIClassifier(cfg)
            detections = classifier.detect_phi_in_text(text, req.context or {})

        # Build operators using POLICY_MATRIX and any overrides
        operators = {}
        for d in detections:
            ent = (d.get("entity_type") or d.get("phi_type") or "OTHER_UNIQUE_IDENTIFIER").upper()
            pm = req.policy_overrides.get(ent) if req.policy_overrides and ent in req.policy_overrides else POLICY_MATRIX.get(ent) or POLICY_MATRIX.get(ent.upper())
            if pm is None:
                # default to replace
                operators[ent] = {"operator": "replace", "params": {"new_value": f"[REDACTED_{ent}]"}}
            else:
                operators[ent] = pm

        # Try to use Presidio AnonymizerEngine if available
        anonymized_text = None
        mappings = []
        try:
            from presidio_anonymizer import AnonymizerEngine, OperatorConfig  # type: ignore

            # Build operator config mapping expected by engine
            ops_for_presidio = {}
            for ent, pm in operators.items():
                op_name = pm.get("operator")
                params = pm.get("params", {})
                try:
                    ops_for_presidio[ent] = [OperatorConfig(op_name, params)]
                except Exception:
                    ops_for_presidio[ent] = [{"type": op_name, "params": params}]

            engine = AnonymizerEngine()
            anonymized = engine.anonymize(text=text, analyzer_results=detections, operators=ops_for_presidio)
            anonymized_text = getattr(anonymized, "text", None) or (anonymized.get("text") if isinstance(anonymized, dict) else None)
            # attempt to map spans using pipeline remap util
            if anonymized_text and anonymized_text != text:
                remapped = _remap_offsets_after_anonymize(text, anonymized_text, [], detections)
                for d in remapped:
                    mappings.append({
                        "original": d.get("text"),
                        "anonymized": anonymized_text[d.get("start", 0) : d.get("end", 0)] if anonymized_text else None,
                        "start": int(d.get("start", 0)),
                        "end": int(d.get("end", 0)),
                        "anonymized_start": int(d.get("anonymized_start", d.get("start", 0))),
                        "anonymized_end": int(d.get("anonymized_end", d.get("end", 0))),
                        "phi_type": d.get("phi_type") or d.get("entity_type"),
                        "policy": operators.get((d.get("phi_type") or d.get("entity_type") or "").upper(), {}).get("operator"),
                    })
            else:
                anonymized_text = text
                for d in detections:
                    mappings.append({"original": d.get("text"), "anonymized": d.get("text"), "start": int(d.get("start", 0)), "end": int(d.get("end", 0)), "anonymized_start": int(d.get("start", 0)), "anonymized_end": int(d.get("end", 0)), "phi_type": d.get("entity_type") or d.get("phi_type"), "policy": operators.get((d.get("entity_type") or d.get("phi_type") or "").upper(), {}).get("operator")})
        except Exception:
            # Presidio anonymizer unavailable or failed; apply fallback replacer
            try:
                anonymized_text, mapping_infos = _apply_fallback_anonymizer(text, detections, POLICY_MATRIX)
                for mi in mapping_infos:
                    mappings.append(mi)
            except Exception:
                logger.exception("Fallback anonymizer failed")
                raise HTTPException(status_code=500, detail="Anonymization failed")

        # Ensure referential integrity: deterministic replacements already used in fallback; for presidio the engine should preserve repeated values depending on operator
        # Persist audit log
        try:
            audit_logger = logging.getLogger("cim.audit.anonymize")
            audit_logger.info({"event": "anonymize", "user": user.get("sub"), "count": len(mappings)})
        except Exception:
            logger.debug("Failed to write anonymization audit log")

        # return structured mapping objects
        resp_maps = []
        for m in mappings:
            resp_maps.append(AnonymizeMapping(
                original=m.get("original"),
                anonymized=m.get("anonymized"),
                start=int(m.get("start", 0) or 0),
                end=int(m.get("end", 0) or 0),
                anonymized_start=int(m.get("anonymized_start")) if m.get("anonymized_start") is not None else None,
                anonymized_end=int(m.get("anonymized_end")) if m.get("anonymized_end") is not None else None,
                phi_type=m.get("phi_type"),
                policy=m.get("policy"),
            ))

        return AnonymizeResponse(original_text=text, anonymized_text=anonymized_text or text, mappings=resp_maps)
