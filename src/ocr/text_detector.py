"""Clinical-grade OCR text detector and recognizer using EasyOCR.

This module provides a TextDetector that wraps EasyOCR where available and
includes clinical-focused post-processing, auditing, and performance
monitoring suitable for HIPAA-aware workflows.

Design notes:
- EasyOCR is imported lazily inside TextDetector to avoid hard runtime
  dependencies during unit tests. If EasyOCR is not installed, the detector
  will still load but operations will raise informative errors.
- All text detections are audit-logged using the application `SETTINGS`
  logging configuration and the audit log path.
"""

from __future__ import annotations

import codecs
import json
import logging
import math
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import re

try:
    import cv2
except Exception:
    cv2 = None  # optional; some environments may not have OpenCV

from src.core.config import OCRConfig, SETTINGS
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


@dataclass
class TextRegion:
    bbox: Tuple[int, int, int, int]
    text: str = ""
    confidence: float = 0.0
    region_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def expand_bbox(self, pixels: int, image_shape: Optional[Tuple[int, int]] = None) -> None:
        x1, y1, x2, y2 = self.bbox
        x1 -= pixels
        y1 -= pixels
        x2 += pixels
        y2 += pixels
        if image_shape is not None:
            h, w = image_shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
        self.bbox = (int(x1), int(y1), int(x2), int(y2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bbox": self.bbox,
            "text": self.text,
            "confidence": float(self.confidence),
            "region_type": self.region_type,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextRegion":
        return cls(
            bbox=tuple(data.get("bbox", (0, 0, 0, 0))),
            text=data.get("text", ""),
            confidence=float(data.get("confidence", 0.0)),
            region_type=data.get("region_type", "text"),
            metadata=data.get("metadata", {}),
        )

    def __post_init__(self):
        # validation
        x1, y1, x2, y2 = self.bbox
        if x1 < 0 or y1 < 0 or x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox coordinates: {self.bbox}")
        # normalize text encoding
        if isinstance(self.text, bytes):
            self.text = codecs.decode(self.text, "utf-8", errors="replace")

    def __str__(self) -> str:  # useful for debugging/audit logs
        return f"TextRegion(bbox={self.bbox}, conf={self.confidence:.3f}, text={self.text!r})"


@dataclass
class OCRResult:
    image_id: str
    regions: List[TextRegion]
    processing_time: float = 0.0
    quality_metrics: Dict[str, float] = field(default_factory=dict)

    def get_high_confidence_regions(self, threshold: float = 0.8) -> List[TextRegion]:
        return [r for r in self.regions if r.confidence >= threshold]

    def export_to_json(self) -> str:
        return json.dumps(
            {
                "image_id": self.image_id,
                "processing_time": self.processing_time,
                "quality_metrics": self.quality_metrics,
                "regions": [r.to_dict() for r in self.regions],
            }
        )

    def get_statistics(self) -> Dict[str, Any]:
        confs = [r.confidence for r in self.regions] if self.regions else [0.0]
        areas = [r.get_area() for r in self.regions] if self.regions else [0]
        return {
            "region_count": len(self.regions),
            "avg_confidence": float(sum(confs) / len(confs)) if confs else 0.0,
            "avg_area": float(sum(areas) / len(areas)) if areas else 0.0,
        }


def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    union = boxAArea + boxBArea - interArea
    if union <= 0:
        return 0.0
    return interArea / union


class TextDetector:
    """Clinical Text Detector using EasyOCR with clinical post-processing.

    The detector is robust to environments where EasyOCR is not installed; in
    that case operations raise informative RuntimeErrors so callers can handle
    degraded operation (for example, in unit tests or CI without heavy DL
    dependencies).
    """

    def __init__(self, config: OCRConfig):
        self.config = config
        self.reader = None
        self._easyocr_available = False
        self._init_time = time.time()
        self.performance = {
            "images_processed": 0,
            "total_time_s": 0.0,
            "last_image_time_s": 0.0,
        }

        # Audit logger
        self.audit_logger = logging.getLogger("cim.audit.ocr")
        if SETTINGS and SETTINGS.logging and SETTINGS.logging.audit_log_path:
            # Audit path config is handled by SETTINGS.setup_logging(); just log
            self.audit_logger.setLevel(logging.INFO)

        # Lazy import EasyOCR to avoid hard dependency at module import time
        try:
            import easyocr  # type: ignore

            # instantiate reader with configured languages and GPU preference
            try:
                self.reader = easyocr.Reader(
                    lang_list=self.config.supported_languages,
                    gpu=bool(self.config.gpu_enabled),
                )
                self._easyocr_available = True
            except Exception:
                # If easyocr fails to initialize (missing torch or GPU mismatch),
                # fall back to not available and record debug info.
                logger.exception("EasyOCR initialization failed")
                self.reader = None
                self._easyocr_available = False
        except Exception:
            # EasyOCR not installed; leave reader as None and defer raising
            # until detection is attempted.
            logger.debug("EasyOCR not available in environment; TextDetector will not perform OCR until installed")

        # Set clinical thresholds
        self.confidence_threshold = float(config.confidence_threshold)
        self.min_text_area = int(config.min_text_area)

        # Lock for updating performance counters
        self._perf_lock = threading.Lock()

    def filter_by_confidence(self, detections: List[Dict[str, Any]], threshold: float, slack: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split detections into accepted and rejected using thresholding.

        Uses adaptive thresholding heuristics: short text receives higher min
        confidence requirements because of increased ambiguity.
        """
        accepted = []
        rejected = []
        # Determine slack: prefer explicit arg, otherwise use config default
        if slack is None:
            try:
                slack = float(getattr(self.config, "confidence_slack", 0.1))
            except Exception:
                slack = 0.1
        for d in detections:
            txt = d.get("text", "")
            conf = float(d.get("confidence", 0.0))
            # adaptive threshold: shorter texts need higher conf
            length = len(txt.strip())
            adapt = 0.0
            if length <= 3:
                adapt = 0.15
            elif length <= 6:
                adapt = 0.08
            adj_threshold = min(0.99, threshold + adapt)
            # allow slack below adjusted threshold
            effective_thresh = max(0.0, adj_threshold - float(slack))
            if conf >= effective_thresh:
                accepted.append(d)
            else:
                d["rejection_reason"] = f"confidence {conf:.3f} < {effective_thresh:.3f} (adj {adj_threshold:.3f}, slack {slack:.3f})"
                rejected.append(d)

        # Audit filter decisions
        try:
            self.audit_logger.info(json.dumps({"event": "filter_by_confidence", "accepted": len(accepted), "rejected": len(rejected)}))
        except Exception:
            logger.debug("Failed to write audit log for filter_by_confidence")

        # If no regions accepted, keep the highest-confidence region as a fallback
        if not accepted and detections:
            best = max(detections, key=lambda x: float(x.get("confidence", 0.0)))
            # remove best from rejected if present
            rejected = [r for r in rejected if r is not best]
            if best not in accepted:
                accepted.append(best)

        return accepted, rejected

    def merge_overlapping_regions(self, regions: List[Dict[str, Any]], overlap_thresh: float = 0.3) -> List[Dict[str, Any]]:
        """Merge overlapping bounding boxes preserving the highest-confidence text.

        This implementation builds an adjacency via intersection-over-minimum-area
        (IoM) and merges any connected components of boxes where a pairwise
        IoM >= overlap_thresh. The merged bbox is the union of member boxes and
        the canonical text/confidence is taken from the highest-confidence
        member. Returned dicts include an optional "merged_from" list.
        """
        if not regions:
            return []

        def area(b):
            return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

        def iom(boxA, boxB):
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            minArea = min(area(boxA), area(boxB))
            if minArea <= 0:
                return 0.0
            return float(interArea) / float(minArea)

        # normalize boxes and scores
        boxes: List[Tuple[int, int, int, int]] = []
        scores: List[float] = []
        low_flags: List[bool] = []
        for r in regions:
            try:
                b = tuple(int(x) for x in r.get("bbox", (0, 0, 0, 0)))
            except Exception:
                b = (0, 0, 0, 0)
            boxes.append(b)
            scores.append(float(r.get("confidence", 0.0)))
            low_flags.append(bool(r.get("low_confidence", False)))

        n = len(boxes)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra = find(a)
            rb = find(b)
            if ra == rb:
                return
            parent[rb] = ra

        # pairwise union where IoM >= threshold
        for i in range(n):
            for j in range(i + 1, n):
                try:
                    if iom(boxes[i], boxes[j]) >= overlap_thresh:
                        union(i, j)
                except Exception:
                    continue

        # collect groups
        groups: Dict[int, List[int]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        merged_results: List[Dict[str, Any]] = []
        for grp in groups.values():
            if not grp:
                continue
            if len(grp) == 1:
                single = dict(regions[grp[0]])
                try:
                    single_bbox = tuple(int(x) for x in single.get("bbox", (0, 0, 0, 0)))
                except Exception:
                    single_bbox = tuple(single.get("bbox", (0, 0, 0, 0)))
                single["bbox"] = single_bbox
                merged_results.append(single)
                continue
            # choose highest-confidence member as canonical
            best_idx = max(grp, key=lambda ii: scores[ii])
            xs1 = [boxes[ii][0] for ii in grp]
            ys1 = [boxes[ii][1] for ii in grp]
            xs2 = [boxes[ii][2] for ii in grp]
            ys2 = [boxes[ii][3] for ii in grp]
            union_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
            merged = dict(regions[best_idx])
            merged["bbox"] = union_bbox
            merged["merged_from"] = [regions[ii].get("bbox") for ii in grp]
            # low_confidence flag: True if all members were low confidence
            merged["low_confidence"] = all(low_flags[ii] for ii in grp)
            merged_results.append(merged)

        return merged_results

    # ------------------------- detection pipeline -------------------------
    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Detect text regions using EasyOCR and return raw detections.

        Returns a list of dicts: {bbox, confidence, box_points}
        """
        if self.reader is None:
            raise RuntimeError("EasyOCR reader not initialized. Install easyocr and torch to enable OCR.")

        start = time.time()
        # EasyOCR expects RGB images
        img = image
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) if cv2 is not None else np.stack([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        # run detection
        try:
            detection_results = self.reader.detect(img)
        except Exception:
            logger.exception("EasyOCR detection error")
            raise

        regions: List[Dict[str, Any]] = []
        for box, score in zip(detection_results[0], detection_results[1]):
            # EasyOCR may return score as a float or a list/array of floats
            try:
                if isinstance(score, (list, tuple)):
                    # take the max confidence if multiple are provided
                    s = float(max(score)) if len(score) > 0 else 0.0
                else:
                    # numpy types may require conversion
                    s = float(score)
            except Exception:
                try:
                    import numpy as _np

                    if isinstance(score, _np.ndarray):
                        s = float(score.max())
                    else:
                        s = float(score)
                except Exception:
                    s = 0.0
            # box is 4 points; convert to bbox
            xs = [int(p[0]) for p in box]
            ys = [int(p[1]) for p in box]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            bbox = (x1, y1, x2, y2)
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_text_area:
                continue
            if s < self.confidence_threshold:
                # still record low-confidence for audit but skip output
                regions.append({"bbox": bbox, "confidence": float(s), "low_confidence": True, "box_points": box})
            else:
                regions.append({"bbox": bbox, "confidence": float(s), "low_confidence": False, "box_points": box})

        # Merge overlapping detections (robust; fall back to raw regions on failure)
        try:
            # Merge all regions first so grouping decisions consider adjacency
            # across confidence bands. Later we will re-attach low-confidence
            # regions as needed.
            merged_all = self.merge_overlapping_regions(regions, overlap_thresh=0.5)
            # preserve low_confidence markers (some implementations may
            # have moved them into merged entries); ensure any original
            # low_confidence entries are not silently dropped: if a merged
            # entry contains only low_confidence members, keep its flag.
            merged = merged_all
        except Exception:
            logger.exception("merge_overlapping_regions failed")
            # If configured to fail on merge errors, re-raise to surface the
            # problem to callers (fail-fast). Otherwise fall back to raw regions.
            try:
                if getattr(self.config, "fail_on_merge_error", False):
                    raise
            except Exception:
                # avoid masking original exception path if getattr itself fails
                raise
            merged = regions

        # Debug export: write raw and merged region JSON for inspection.
        # Some OCR return objects may contain numpy types or other non-JSON
        # serializables; sanitize recursively and write atomically to avoid
        # partial/truncated output files.
        try:
            # derive a simple page index based on images_processed counter
            page_idx = int(self.performance.get("images_processed", 0)) + 1
            outdir = "debug_output"
            import os, tempfile

            os.makedirs(outdir, exist_ok=True)

            def _sanitize(obj):
                # primitives
                if obj is None:
                    return None
                if isinstance(obj, (str, bool)):
                    return obj
                if isinstance(obj, (int, float)):
                    # convert numpy scalars to native
                    try:
                        return obj.item()  # type: ignore
                    except Exception:
                        return obj
                # numpy arrays
                try:
                    import numpy as _np

                    if isinstance(obj, _np.ndarray):
                        return _np.asarray(obj).tolist()
                except Exception:
                    pass
                # list/tuple
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                # dict
                if isinstance(obj, dict):
                    return {str(k): _sanitize(v) for k, v in obj.items()}
                # objects with __dict__
                if hasattr(obj, "__dict__"):
                    try:
                        return _sanitize(vars(obj))
                    except Exception:
                        return repr(obj)
                # fallback to string representation
                try:
                    return str(obj)
                except Exception:
                    return repr(obj)

            raw_s = {"regions": [_sanitize(r) for r in regions]}
            merged_s = {"merged": [_sanitize(r) for r in merged]}

            raw_path = os.path.join(outdir, f"ocr_regions_page_{page_idx}.json")
            merged_path = os.path.join(outdir, f"merged_regions_page_{page_idx}.json")

            # write atomically via tempfile then replace
            for data, path in ((raw_s, raw_path), (merged_s, merged_path)):
                fd, tmp = tempfile.mkstemp(prefix="tmpocr", dir=outdir)
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2, ensure_ascii=False)
                    os.replace(tmp, path)
                finally:
                    if os.path.exists(tmp):
                        try:
                            os.remove(tmp)
                        except Exception:
                            pass

            logger.debug("Wrote OCR debug files %s and %s", raw_path, merged_path)
        except Exception:
            logger.exception("Failed to write OCR debug artifacts")

        # Add low-confidence entries back as separate entries (they're flagged)
        low_conf = [r for r in regions if r.get("low_confidence")]
        merged.extend(low_conf)

        elapsed = time.time() - start
        with self._perf_lock:
            self.performance["images_processed"] += 1
            self.performance["total_time_s"] += elapsed
            self.performance["last_image_time_s"] = elapsed

        # Audit log
        try:
            self.audit_logger.info(json.dumps({"event": "detect_text_regions", "count": len(merged), "time_s": elapsed}))
        except Exception:
            logger.debug("Failed to write audit log for detect_text_regions")

        return merged

    def extract_text_content(self, image: np.ndarray, regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Run OCR on each region and enrich with text and per-character confidence.

        Regions are expected to have 'bbox' keys.
        """
        if self.reader is None:
            raise RuntimeError("EasyOCR reader not initialized. Install easyocr and torch to enable OCR.")

        results: List[Dict[str, Any]] = []
        for r in regions:
            try:
                x1, y1, x2, y2 = r["bbox"]
                crop = image[y1:y2, x1:x2]
                # EasyOCR recognizes on the full crop
                # Request word-level (non-paragraph) output so we get one bbox per
                # word/small phrase instead of grouped paragraph boxes. This uses
                # EasyOCR's `paragraph=False` flag to preserve granular boxes.
                # Assumption: EasyOCR version in use supports `paragraph` kwarg.
                try:
                    ocr_out = self.reader.readtext(crop, detail=1, paragraph=False)
                except TypeError:
                    # Older easyocr versions may not accept `paragraph`; fall back
                    # to default behavior.
                    ocr_out = self.reader.readtext(crop, detail=1)
                # ocr_out: list of (bbox, text, confidence) where bbox coords are
                # relative to the cropped image. Convert each word bbox to full-image
                # coordinates so downstream mapping/masking can operate precisely.
                texts = []
                confs = []
                word_boxes: List[Dict[str, Any]] = []
                char_offset = 0
                # When EasyOCR returns word-level items for the crop, we need
                # to record approximate character offsets relative to the
                # combined region text so downstream mapping can align spans
                # to word boxes. We conservatively compute offsets by joining
                # words with single spaces.
                for item in ocr_out:
                    bbox_pts, txt, conf = item
                    # normalize whitespace
                    txt = " ".join(str(txt).split())
                    texts.append(txt)
                    confs.append(float(conf))

                    # compute bbox from bbox_pts (list of 4 points) relative to crop
                    try:
                        xs = [int(p[0]) for p in bbox_pts]
                        ys = [int(p[1]) for p in bbox_pts]
                        wx1, wx2 = min(xs), max(xs)
                        wy1, wy2 = min(ys), max(ys)
                        # map to full-image coordinates by offsetting with region origin
                        abs_box = (int(x1 + wx1), int(y1 + wy1), int(x1 + wx2), int(y1 + wy2))
                    except Exception:
                        abs_box = (int(x1), int(y1), int(x2), int(y2))
                    # estimate start/end char offsets within combined_text
                    start_ch = char_offset
                    token = str(txt or "")
                    token_len = len(token)
                    end_ch = start_ch + token_len
                    word_boxes.append({"text": token, "bbox": abs_box, "confidence": float(conf), "start_char": start_ch, "end_char": end_ch})
                    # advance offset (account for a space separator)
                    char_offset = end_ch + 1

                # Use a canonical single-space join for downstream offset mapping.
                # Earlier implementations used newlines which led to inconsistencies
                # when pipeline code assumed single-space joining. Keep token
                # offsets consistent with this single-space representation.
                combined_text = " ".join(texts)
                avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

                r_out = dict(r)
                r_out.update({
                    "text": combined_text,
                    "confidence": avg_conf,
                    "ocr_lines": texts,
                    "word_boxes": word_boxes,
                })
                results.append(r_out)
            except Exception:
                logger.exception("Failed to OCR region")
                r_out = dict(r)
                r_out.update({"text": "", "confidence": 0.0, "ocr_lines": []})
                results.append(r_out)

        # Audit
        try:
            self.audit_logger.info(json.dumps({"event": "extract_text_content", "num_regions": len(results)}))
        except Exception:
            logger.debug("Failed to write audit log for extract_text_content")

        return results

    def process_clinical_text(self, text_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply clinical normalization, abbreviation expansion, error correction.

        This implements conservative heuristics rather than clinical-grade
        dictionary replacements. For production, connect to an authoritative
        medical vocabulary and abbreviation map.
        """
        processed = []
        for r in text_results:
            txt = r.get("text", "")
            original = txt
            # common character substitutions
            txt = txt.replace("0", "O") if any(c.isalpha() for c in txt) else txt
            txt = txt.replace("1", "I")
            txt = txt.replace("\u2019", "'")
            # normalize whitespace and encoding
            txt = " ".join(txt.split())

            # simple abbreviation expansion (example mapping)
            abbr = {"BP": "blood pressure", "HR": "heart rate", "DOB": "date of birth"}
            tokens = txt.split()
            expanded_tokens = [abbr.get(t.upper(), t) for t in tokens]
            txt_exp = " ".join(expanded_tokens)

            # medical vocabulary validation (best-effort)
            is_medical, med_score = validate_medical_terminology(txt_exp)

            # boost confidence slightly if medical context found
            conf = float(r.get("confidence", 0.0))
            if is_medical:
                conf = min(1.0, conf + 0.05 * med_score)

            r_out = dict(r)
            r_out.update({"text": txt_exp, "confidence": conf, "_original_text": original, "medical_score": med_score})
            # flag for manual review if medical and low confidence
            r_out["needs_manual_review"] = is_medical and conf < 0.85
            processed.append(r_out)

        # Audit summary
        try:
            self.audit_logger.info(json.dumps({"event": "process_clinical_text", "processed": len(processed)}))
        except Exception:
            logger.debug("Failed to write audit log for process_clinical_text")

        return processed


def split_regions_by_words(regions: List[Dict[str, Any]], max_words_per_region: int = 3) -> List[Dict[str, Any]]:
    """Split large text regions into smaller line- or word-level regions.

    Strategy:
    - If a region already contains `word_boxes`, prefer and return those (annotated).
    - Otherwise split by lines. If a line has more than `max_words_per_region`
      words, split that line into individual word regions by approximating
      horizontal slices (uniform width per word). Vertical positions are
      approximated by equally dividing the region height across lines.

    This is a heuristic helper for cases where OCR returns a large paragraph
    bbox but downstream mapping/masking expects tighter boxes.
    """
    out: List[Dict[str, Any]] = []
    for r in regions:
        text = (r.get("text") or "").strip()
        if not text:
            # keep region as-is if no text
            out.append(r)
            continue

        # If accurate per-word boxes already exist, prefer them
        word_boxes = r.get("word_boxes") or []
        if word_boxes:
            for w in word_boxes:
                try:
                    bbox = [int(x) for x in w.get("bbox", r.get("bbox", [0, 0, 0, 0]))]
                except Exception:
                    bbox = list(r.get("bbox", [0, 0, 0, 0]))
                out.append({
                    "text": w.get("text", ""),
                    "bbox": bbox,
                    "confidence": float(w.get("confidence", r.get("confidence", 0.0))),
                    "region_type": r.get("region_type", "text"),
                    "split_from": text,
                })
            continue

        # fallback heuristic: split by lines first
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            lines = [text]

        x1, y1, x2, y2 = r.get("bbox", (0, 0, 0, 0))
        try:
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        except Exception:
            x1, y1, x2, y2 = 0, 0, 0, 0

        region_width = max(1, x2 - x1)
        region_height = max(1, y2 - y1)

        for li, line in enumerate(lines):
            words = [w for w in line.split() if w]
            # compute vertical slice for this line
            line_h = int(region_height / len(lines))
            ly1 = y1 + li * line_h
            ly2 = y1 + (li + 1) * line_h if li < len(lines) - 1 else y2

            if len(words) == 0:
                continue

            if len(words) <= max_words_per_region:
                # keep the whole line as one region, use full width
                out.append({
                    "text": line,
                    "bbox": [x1, ly1, x2, ly2],
                    "confidence": float(r.get("confidence", 0.0)),
                    "region_type": r.get("region_type", "text"),
                    "split_from": text,
                })
            else:
                # split into word boxes by uniformly slicing the width
                word_width = region_width / len(words)
                for i, wtext in enumerate(words):
                    wx1 = int(x1 + i * word_width)
                    wx2 = int(x1 + (i + 1) * word_width) if i < len(words) - 1 else x2
                    out.append({
                        "text": wtext,
                        "bbox": [wx1, ly1, wx2, ly2],
                        "confidence": float(r.get("confidence", 0.0)),
                        "region_type": r.get("region_type", "text"),
                        "split_from": text,
                    })

    return out


def _fuzzy_ratio(a: str, b: str) -> int:
    """Return a 0-100 similarity ratio between two strings.

    Prefer fuzzywuzzy if available; otherwise use difflib.SequenceMatcher.
    """
    try:
        # lazy import fuzzywuzzy if available
        from fuzzywuzzy import fuzz  # type: ignore

        return int(fuzz.token_sort_ratio(a, b))
    except Exception:
        # fallback to difflib ratio scaled to 0-100
        if not a and not b:
            return 100
        try:
            return int(SequenceMatcher(None, a, b).ratio() * 100)
        except Exception:
            return 0


def merge_adjacent_regions(regions: List[Dict[str, Any]], phi_texts: List[str], fuzz_threshold: int = 80, horizontal_tolerance: int = 10) -> List[Dict[str, Any]]:
    """Merge adjacent word-level OCR regions when their concatenation matches any PHI text.

    Args:
        regions: list of word-level regions (dicts with 'text' and 'bbox')
        phi_texts: list of target PHI phrases to match against
        fuzz_threshold: integer 0-100 controlling fuzzy match acceptance
        horizontal_tolerance: pixels tolerance to consider words adjacent on same line

    Returns:
        New list of regions where matched adjacent sequences are merged into single region dicts.
    """
    if not regions or not phi_texts:
        return regions

    # normalize phi texts
    norm_phis = [re.sub(r"\s+", " ", (p or "").strip()).lower() for p in phi_texts if p]

    # sort regions by top-to-bottom then left-to-right
    def _sort_key(r: Dict[str, Any]):
        try:
            x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
        except Exception:
            x1, y1 = 0, 0
        return (y1, x1)

    sorted_regs = sorted(regions, key=_sort_key)
    used = [False] * len(sorted_regs)
    out: List[Dict[str, Any]] = []

    for i, r in enumerate(sorted_regs):
        if used[i]:
            continue
        # begin candidate sequence
        seq_indices = [i]
        seq_texts = [ (r.get("text") or "").strip() ]
        seq_boxes = [ list(map(int, r.get("bbox", [0,0,0,0]))) ]
        used[i] = True

        # attempt to grow to the right across adjacent tokens on same line
        for j in range(i+1, len(sorted_regs)):
            if used[j]:
                continue
            rj = sorted_regs[j]
            try:
                x1a, y1a, x2a, y2a = map(int, seq_boxes[-1])
                x1b, y1b, x2b, y2b = map(int, rj.get("bbox", (0,0,0,0)))
            except Exception:
                break
            # check vertical alignment (simple overlap / same line)
            vert_overlap = min(y2a, y2b) - max(y1a, y1b)
            min_h = min(max(1, y2a - y1a), max(1, y2b - y1b))
            if vert_overlap < -horizontal_tolerance:
                # too far vertically; likely next line
                break
            # check horizontal adjacency: next box should start near or after previous end
            if x1b < x2a - horizontal_tolerance:
                # overlapping backwards; skip as not a forward neighbor
                continue
            # candidate concatenation: current sequence + this candidate token
            candidate_text = " ".join(seq_texts + [ (rj.get("text") or "").strip() ])
            # normalize
            cand_norm = re.sub(r"\s+", " ", candidate_text.strip()).lower()
            # test against all phi texts using fuzzy match
            matched = False
            for phi in norm_phis:
                score = _fuzzy_ratio(cand_norm, phi)
                if score >= fuzz_threshold:
                    # accept merge: extend sequences
                    seq_indices.append(j)
                    seq_texts.append((rj.get("text") or "").strip())
                    seq_boxes.append(list(map(int, rj.get("bbox", [0,0,0,0]))))
                    used[j] = True
                    matched = True
                    break
            if not matched:
                # not matched for this extension; do not include rj; stop growing
                break
            else:
                # we matched a PHI phrase with current concatenation; stop growing further
                break

        # build merged region from seq_texts and seq_boxes
        if len(seq_boxes) == 1:
            out.append(r)
        else:
            xs1 = [b[0] for b in seq_boxes]
            ys1 = [b[1] for b in seq_boxes]
            xs2 = [b[2] for b in seq_boxes]
            ys2 = [b[3] for b in seq_boxes]
            merged_bbox = [min(xs1), min(ys1), max(xs2), max(ys2)]
            merged_text = re.sub(r"\s+", " ", " ".join([t for t in seq_texts if t]).strip())
            # compute average confidence across merged indices
            try:
                confs = [float(sorted_regs[idx].get("confidence", 0.0)) for idx in seq_indices]
                merged_conf = float(sum(confs) / max(1, len(confs)))
            except Exception:
                merged_conf = float(r.get("confidence", 0.0))
            out.append({
                "text": merged_text,
                "bbox": merged_bbox,
                "confidence": merged_conf,
                "region_type": r.get("region_type", "text"),
                "merged_from": [list(b) for b in seq_boxes],
            })

    return out


def map_phi_to_exact_regions(phi_detections: List[Dict[str, Any]], text_regions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize PHI detections to the canonical PHI detection contract.

    Each returned detection has keys:
        - entity_type (str)
        - text (str)
        - start_char (int)
        - end_char (int)
        - page_number (int)
        - bbox ([x1,y1,x2,y2])
        - confidence (float 0..1)

    This function attempts to map analyzer detections (which may only have
    character offsets or only text) to OCR-derived `text_regions` and their
    `word_boxes`. It uses the centralized `map_text_spans_to_image_bboxes`
    utility when available.
    """
    from src.ocr.mapping import map_text_spans_to_image_bboxes

    canonical: List[Dict[str, Any]] = []

    # Build an index of regions: we expect text_regions to have keys
    # 'text', 'bbox', 'word_boxes' and optionally 'page_number'
    for region in text_regions or []:
        region_text = (region.get("text") or "").strip()
        region_bbox = region.get("bbox")
        page = int(region.get("page_number", region.get("page", 1) or 1))
        word_boxes = region.get("word_boxes") or []

        # If word_boxes exist but lack char offsets, try to augment them by
        # distributing offsets across the region text (best-effort).
        # Note: extract_text_content now attempts to provide start_char/end_char
        # for each word when possible.

        # For each phi detection, if it specifies start/end offsets we prefer
        # to map by that; otherwise we attempt text matching.
        for det in phi_detections or []:
            ent = det.get("entity_type") or det.get("phi_type") or det.get("label") or det.get("type")
            text = det.get("text") or ""
            # prefer detection offsets when present
            start = det.get("start") if det.get("start") is not None else det.get("start_char")
            end = det.get("end") if det.get("end") is not None else det.get("end_char")
            conf = float(det.get("confidence", det.get("score", 0.0) or 0.0))

            mapped = []
            try:
                # try mapping within this region using word_boxes
                mapped = map_text_spans_to_image_bboxes([{"text": text, "start": start, "end": end}], word_boxes, region_bbox, page)
            except Exception:
                mapped = []

            if mapped:
                for m in mapped:
                    canonical.append({
                        "entity_type": ent,
                        "text": m.get("text", text),
                        "start_char": int(m.get("start") or m.get("start_char") or 0),
                        "end_char": int(m.get("end") or m.get("end_char") or 0),
                        "page_number": int(m.get("page_number", page)),
                        "bbox": m.get("bbox"),
                        "confidence": float(conf),
                    })
            else:
                # fall back: if det contained a bbox already, use it; else use region bbox
                bbox = det.get("bbox") or region_bbox
                canonical.append({
                    "entity_type": ent,
                    "text": text or region_text,
                    "start_char": int(start or 0),
                    "end_char": int(end or 0),
                    "page_number": page,
                    "bbox": bbox,
                    "confidence": float(conf),
                })

    # Deduplicate by (entity_type, text, page_number, bbox)
    seen = set()
    out: List[Dict[str, Any]] = []
    for c in canonical:
        key = (str(c.get("entity_type")), str(c.get("text")), int(c.get("page_number", 1)), tuple(c.get("bbox") or ()))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)

    return out

    def merge_overlapping_regions(self, regions: List[Dict[str, Any]], overlap_thresh: float = 0.3) -> List[Dict[str, Any]]:
        """Merge overlapping bounding boxes preserving the highest-confidence text.

        This implementation uses intersection-over-minimum-area (IoM) as the
        overlap metric: inter_area / min(areaA, areaB). Regions with IoM >=
        overlap_thresh are merged together. The merged bbox is the union of
        merged boxes and the canonical text is taken from the highest
        confidence member.
        """
        if not regions:
            return []

        def area(b):
            return max(0, (b[2] - b[0])) * max(0, (b[3] - b[1]))

        def iom(boxA, boxB):
            # intersection over min area
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[2], boxB[2])
            yB = min(boxA[3], boxB[3])
            interW = max(0, xB - xA)
            interH = max(0, yB - yA)
            interArea = interW * interH
            minArea = min(area(boxA), area(boxB))
            if minArea <= 0:
                return 0.0
            return float(interArea) / float(minArea)

        boxes = [tuple(r["bbox"]) for r in regions]
        scores = [float(r.get("confidence", 0.0)) for r in regions]

        used = [False] * len(boxes)
        groups: List[List[int]] = []

        for i in range(len(boxes)):
            if used[i]:
                continue
            group = [i]
            used[i] = True
            for j in range(i + 1, len(boxes)):
                if used[j]:
                    continue
                try:
                    if iom(boxes[i], boxes[j]) >= overlap_thresh:
                        group.append(j)
                        used[j] = True
                except Exception:
                    continue
            groups.append(group)

        merged_results: List[Dict[str, Any]] = []
        for grp in groups:
            if not grp:
                continue
            if len(grp) == 1:
                merged_results.append(dict(regions[grp[0]]))
                continue
            # choose canonical as highest confidence
            best_idx = max(grp, key=lambda ii: scores[ii])
            # union bbox
            xs1 = [boxes[ii][0] for ii in grp]
            ys1 = [boxes[ii][1] for ii in grp]
            xs2 = [boxes[ii][2] for ii in grp]
            ys2 = [boxes[ii][3] for ii in grp]
            union_bbox = (min(xs1), min(ys1), max(xs2), max(ys2))
            merged = dict(regions[best_idx])
            merged["bbox"] = union_bbox
            merged["merged_from"] = [regions[ii].get("bbox") for ii in grp]
            merged_results.append(merged)

        return merged_results

    def validate_detection_quality(self, image: np.ndarray, detections: List[Dict[str, Any]]) -> Dict[str, float]:
        """Return a quality report for detections on the image.

        Metrics include density (regions per area), average confidence, and
        uniformity (std of confidences).
        """
        img_area = float(image.shape[0] * image.shape[1])
        region_count = len(detections)
        density = region_count / (img_area / (1000 * 1000))  # regions per megapixel
        confs = [float(d.get("confidence", 0.0)) for d in detections] or [0.0]
        avg_conf = float(sum(confs) / len(confs))
        std_conf = float(np.std(confs))

        report = {
            "region_count": float(region_count),
            "density_per_megapixel": density,
            "avg_confidence": avg_conf,
            "std_confidence": std_conf,
        }

        # heuristics: flag if too few or too many regions compared to expected
        self.audit_logger.info(json.dumps({"event": "validate_detection_quality", "report": report}))

        return report

    def batch_process_images(self, images: List[np.ndarray], progress_callback: Optional[Callable] = None) -> List[List[Dict[str, Any]]]:
        """Process multiple images concurrently and return OCR results per image.

        Uses ThreadPoolExecutor to bound memory usage; for GPU-heavy workloads a
        different batching strategy would be required.
        """
        start_all = time.time()
        results: List[List[Dict[str, Any]]] = []
        max_workers = min(8, max(1, int(self.config.batch_size)))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._process_single_image, idx, img): idx for idx, img in enumerate(images)}
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    res = fut.result()
                except MemoryError:
                    logger.exception("MemoryError during batch processing")
                    res = []
                except Exception:
                    logger.exception("Error during image processing")
                    res = []
                results.append(res)
                if progress_callback:
                    try:
                        progress_callback(len(results), len(images))
                    except Exception:
                        logger.debug("progress_callback failed")

        total_time = time.time() - start_all
        self.audit_logger.info(json.dumps({"event": "batch_process_images", "count": len(images), "time_s": total_time}))
        return results

    # ------------------------- helpers & internals -------------------------
    def _process_single_image(self, image_id: int, image: np.ndarray) -> List[Dict[str, Any]]:
        t0 = time.time()
        try:
            regs = self.detect_text_regions(image)
            ocred = self.extract_text_content(image, regs)
            processed = self.process_clinical_text(ocred)
            perf = time.time() - t0
            self.audit_logger.info(json.dumps({"event": "process_image", "image_id": image_id, "time_s": perf, "regions": len(processed)}))
            return processed
        except Exception:
            logger.exception("Failed to fully process image %s", image_id)
            return []

    def filter_by_confidence(self, detections: List[Dict[str, Any]], threshold: float, slack: Optional[float] = None) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split detections into accepted and rejected using thresholding.

        Uses adaptive thresholding heuristics: short text receives higher min
        confidence requirements because of increased ambiguity.
        """
        accepted = []
        rejected = []
        # Determine slack: prefer explicit arg, otherwise use config default
        if slack is None:
            try:
                slack = float(getattr(self.config, "confidence_slack", 0.1))
            except Exception:
                slack = 0.1
        for d in detections:
            txt = d.get("text", "")
            conf = float(d.get("confidence", 0.0))
            # adaptive threshold: shorter texts need higher conf
            length = len(txt.strip())
            adapt = 0.0
            if length <= 3:
                adapt = 0.15
            elif length <= 6:
                adapt = 0.08
            adj_threshold = min(0.99, threshold + adapt)
            # allow slack below adjusted threshold
            effective_thresh = max(0.0, adj_threshold - float(slack))
            if conf >= effective_thresh:
                accepted.append(d)
            else:
                d["rejection_reason"] = f"confidence {conf:.3f} < {effective_thresh:.3f} (adj {adj_threshold:.3f}, slack {slack:.3f})"
                rejected.append(d)

        # Audit filter decisions
        try:
            self.audit_logger.info(json.dumps({"event": "filter_by_confidence", "accepted": len(accepted), "rejected": len(rejected)}))
        except Exception:
            logger.debug("Failed to write audit log for filter_by_confidence")

        # If no regions accepted, keep the highest-confidence region as a fallback
        if not accepted and detections:
            best = max(detections, key=lambda x: float(x.get("confidence", 0.0)))
            # remove best from rejected if present
            rejected = [r for r in rejected if r is not best]
            if best not in accepted:
                accepted.append(best)

        return accepted, rejected


# ------------------------- Clinical-specific utilities -------------------------
def detect_medical_displays(image: np.ndarray) -> Dict[str, Any]:
    """Heuristic detection of medical displays (ultrasound, monitor, workstation).

    Returns a dict with type and simple characteristics.
    """
    h, w = image.shape[:2]
    # simple heuristics: ultrasound often has black borders and curved display
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if cv2 is not None else np.mean(image, axis=2).astype(np.uint8)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 40:
        display = "ultrasound_like"
    elif w > h and mean_brightness > 100:
        display = "monitor_like"
    else:
        display = "workstation_like"

    return {"display_type": display, "mean_brightness": mean_brightness, "resolution": (w, h)}


def optimize_for_clinical_text(image: np.ndarray, display_type: str) -> np.ndarray:
    """Apply display-specific preprocessing to enhance text for OCR.

    This is intentionally conservative and reversible.
    """
    img = image.copy()
    if display_type == "ultrasound_like":
        # increase local contrast
        if cv2 is not None:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l2 = clahe.apply(l)
            lab2 = cv2.merge((l2, a, b))
            img = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
    elif display_type == "monitor_like":
        # sharpen and boost contrast
        if cv2 is not None:
            kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
            img = cv2.filter2D(img, -1, kernel)

    # universal: convert to 3-channel BGR if needed
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.stack([img] * 3, axis=-1)

    return img


def validate_medical_terminology(text: str) -> Tuple[bool, float]:
    """Best-effort medical vocabulary check.

    Returns (is_medical, score) where score is 0..1 indicating how
    strongly the text matches medical vocabulary. This is intentionally
    lightweight; integrate a real terminology server for production.
    """
    if not text:
        return False, 0.0
    medical_tokens = {"blood", "pressure", "heart", "rate", "dose", "mg", "ml", "dob", "bp", "hr"}
    toks = {t.lower().strip(".,;:\'\"()") for t in text.split()}
    match = toks & medical_tokens
    score = float(len(match)) / max(1, len(toks))
    return (len(match) > 0), score


__all__ = [
    "TextDetector",
    "TextRegion",
    "OCRResult",
    "detect_medical_displays",
    "optimize_for_clinical_text",
    "validate_medical_terminology",
]
