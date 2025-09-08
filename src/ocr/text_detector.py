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

try:
    import cv2
except Exception:
    cv2 = None  # optional; some environments may not have OpenCV

from src.core.config import OCRConfig, SETTINGS

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

        # Merge overlapping detections
        merged = self.merge_overlapping_regions([r for r in regions if not r.get("low_confidence")])

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
                ocr_out = self.reader.readtext(crop, detail=1)
                # ocr_out: list of (bbox, text, confidence)
                texts = []
                confs = []
                for item in ocr_out:
                    _, txt, conf = item
                    # normalize whitespace
                    txt = " ".join(str(txt).split())
                    texts.append(txt)
                    confs.append(float(conf))

                combined_text = "\n".join(texts)
                avg_conf = float(sum(confs) / len(confs)) if confs else 0.0

                r_out = dict(r)
                r_out.update({"text": combined_text, "confidence": avg_conf, "ocr_lines": texts})
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

    def filter_by_confidence(self, detections: List[Dict[str, Any]], threshold: float) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split detections into accepted and rejected using thresholding.

        Uses adaptive thresholding heuristics: short text receives higher min
        confidence requirements because of increased ambiguity.
        """
        accepted = []
        rejected = []
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
            if conf >= adj_threshold:
                accepted.append(d)
            else:
                d["rejection_reason"] = f"confidence {conf:.3f} < {adj_threshold:.3f}"
                rejected.append(d)

        # Audit filter decisions
        try:
            self.audit_logger.info(json.dumps({"event": "filter_by_confidence", "accepted": len(accepted), "rejected": len(rejected)}))
        except Exception:
            logger.debug("Failed to write audit log for filter_by_confidence")

        return accepted, rejected

    def merge_overlapping_regions(self, regions: List[Dict[str, Any]], iou_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Merge overlapping bounding boxes preserving the highest-confidence text."""
        if not regions:
            return []

        boxes = [tuple(r["bbox"]) for r in regions]
        scores = [float(r.get("confidence", 0.0)) for r in regions]

        idxs = list(range(len(boxes)))
        keep = []
        while idxs:
            current = idxs.pop(0)
            cur_box = boxes[current]
            cur_score = scores[current]
            to_merge = [current]
            remove = []
            for other in idxs:
                if _iou(cur_box, boxes[other]) >= iou_threshold:
                    to_merge.append(other)
                    remove.append(other)
            # remove merged indices from idxs
            idxs = [i for i in idxs if i not in remove]

            # pick highest confidence region as canonical
            best = max(to_merge, key=lambda i: scores[i])
            merged = dict(regions[best])
            # if more than one region, expand bbox to cover union
            if len(to_merge) > 1:
                xs = [b[0] for b in boxes] + [b[2] for b in boxes]
                ys = [b[1] for b in boxes] + [b[3] for b in boxes]
                merged_bbox = (min(xs), min(ys), max(xs), max(ys))
                merged["bbox"] = merged_bbox
                merged["merged_from"] = [regions[i].get("bbox") for i in to_merge]
            keep.append(merged)

        return keep

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
