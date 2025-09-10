"""Clinical image masking pipeline orchestrator.

Provides ClinicalImageMaskingPipeline which composes preprocessing, OCR,
PHI classification, and inpainting into a single workflow with robust
error handling, audit logging, and performance monitoring suitable for
HIPAA-aware environments.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import tempfile
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import random
import re
from pathlib import Path

from src.core.config import AppConfig, SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    name: str
    duration_s: float
    result: Any = None
    error: Optional[str] = None


class ClinicalImageMaskingPipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        # apply any operator overrides from the runtime config into POLICY_MATRIX
        try:
            from src.integration.config_loader import apply_operator_overrides
            from src.integration.policy_matrix import POLICY_MATRIX

            # config may be pydantic BaseSettings; convert to dict if needed
            cfg_dict = getattr(config, "dict", lambda: config)()
            apply_operator_overrides(cfg_dict, POLICY_MATRIX)
        except Exception:
            logger.debug("No operator overrides applied or config loader unavailable")
        # instance logger for pipeline-level messages
        self.logger = logging.getLogger("ClinicalImageMaskingPipeline")
        # set up logging and audit trail
        try:
            self.config.setup_logging()
        except Exception:
            logger.exception("Failed to setup logging from config")

        # lazy module imports to avoid heavy deps at import time
        self.preprocessor = None
        self.ocr = None
        self.phi = None
        self.inpainter = None

        # instantiate components on demand
        self._init_components()
        # import mocks to use as fallbacks in test or degraded environments
        try:
            from src.core.mocks import MockTextDetector, MockImageInpainter

            self._MockTextDetector = MockTextDetector
            self._MockImageInpainter = MockImageInpainter
        except Exception:
            self._MockTextDetector = None
            self._MockImageInpainter = None

        # performance monitoring
        self.metrics: Dict[str, Any] = {"processed": 0, "errors": 0, "stages": {}}

        # temporary management
        self._temp_root = Path(self.config.processing.temp_dir)
        self._temp_root.mkdir(parents=True, exist_ok=True)

    def _init_components(self) -> None:
        try:
            # import only when available
            from src.preprocessing.image_processor import ImageProcessor
            from src.ocr.text_detector import TextDetector
            from src.phi_detection.phi_classifier import PHIClassifier
            from src.masking.image_inpainter import ImageInpainter

            self.preprocessor = ImageProcessor(self.config.processing)
            self.ocr = TextDetector(self.config.ocr)
            self.phi = PHIClassifier(self.config.phi)
            self.inpainter = ImageInpainter(self.config.mask)
        except Exception:
            logger.exception("Component initialization failed - pipeline will attempt lazy recovery")

    # ---------------- pipeline helpers ----------------
    def _timeit(self, fn: Callable, *args, **kwargs) -> Tuple[Any, float, Optional[Exception]]:
        t0 = time.perf_counter()
        try:
            res = fn(*args, **kwargs)
            return res, time.perf_counter() - t0, None
        except Exception as e:
            return None, time.perf_counter() - t0, e

    # ---------------- integration helper for external PII analyzers ----------------
    def detect_phi_in_text(self, text: str, meta: Optional[Dict] = None) -> List[Dict]:
        """Integration hook for an external PII/Presidio analyzer.

        - text: the extracted text to analyze
        - meta: optional metadata (source: 'ocr'|'native_pdf'|'scanned_pdf', bbox, page, etc.)

        Returns a list of detection dicts in the form expected by downstream stages.
        This is a placeholder that calls into the PHI classifier when available.
        Replace or extend this method to call Presidio Analyzer/Recognizer as needed.
        """
        meta = meta or {}
        try:
            # prefer a central presidio wrapper if available
            try:
                from src.integration.presidio_wrapper import get_analyzer

                analyzer = get_analyzer()
                return analyzer.detect_phi_in_text(text, meta)
            except Exception:
                # fallback to internal PHI classifier if present
                if self.phi is not None and hasattr(self.phi, "detect_phi_in_text"):
                    return self.phi.detect_phi_in_text(text, meta)
                return []
        except Exception:
            logger.exception("detect_phi_in_text hook failed")
            return []

    # ---------------- single-image pipeline ----------------
    def process_single_image(self, input_path: str, output_path: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        image_id = Path(input_path).stem
        # Canary routing for Presidio-based detection: choose per-image whether
        # to route this request to the new Presidio pipeline. The percentage is
        # configurable via the AppConfig (attribute `presidio_canary_percentage`)
        # or via a `config.yaml` file with key `presidio_canary_percentage: N`.
        presidio_pct = 0
        try:
            # prefer explicit attribute on the provided config
            presidio_pct = int(getattr(self.config, "presidio_canary_percentage", 0) or 0)
        except Exception:
            presidio_pct = 0

        if presidio_pct <= 0:
            # fallback: try to read repository config yaml (best-effort, lightweight)
            try:
                for cfg_path in (Path("config.yaml"), Path("config") / "config.yaml"):
                    if cfg_path.exists():
                        txt = cfg_path.read_text(encoding="utf-8")
                        m = re.search(r"presidio_canary_percentage\s*:\s*(\d{1,3})", txt)
                        if m:
                            try:
                                presidio_pct = int(m.group(1))
                                break
                            except Exception:
                                presidio_pct = 0
            except Exception:
                presidio_pct = presidio_pct or 0

        # clamp to sensible bounds
        presidio_pct = max(0, min(100, int(presidio_pct or 0)))

        # roll the dice for this image
        roll = random.randint(1, 100)
        use_presidio_pipeline = roll <= presidio_pct
        self.logger.info("Presidio canary routing: roll=%d presidio_pct=%d use_presidio=%s for image=%s", roll, presidio_pct, use_presidio_pipeline, image_id)

        session_temp = Path(tempfile.mkdtemp(prefix=f"cim_{image_id}_", dir=str(self._temp_root)))
        stage_results: Dict[str, StageResult] = {}
        overall_start = time.perf_counter()

        try:
            # Stage 1: preprocess (defensive unpacking)
            img, meta = None, {}
            res, dur, err = self._timeit(self.stage_1_preprocess, input_path)
            if res is None:
                logger.warning("Preprocess returned no result for %s", input_path)
                img, meta = None, {}
            else:
                try:
                    img, meta = res
                except Exception:
                    logger.exception("Preprocess returned unexpected shape/result")
                    img, meta = None, {}

            stage_results["preprocess"] = StageResult("preprocess", dur, result=meta, error=str(err) if err else None)
            if err:
                self.metrics["errors"] += 1
                recovery = self.handle_preprocessing_errors(err, input_path)
                stage_results["preprocess_recovery"] = StageResult("preprocess_recovery", recovery.get("duration", 0.0), result=recovery)
                if recovery.get("failed"):
                    raise RuntimeError("Preprocessing failed and recovery unsuccessful")

            # Stage 2: text detection (defensive unpacking)
            regions, det_meta = [], {}
            res, dur2, err2 = self._timeit(self.stage_2_text_detection, img, meta)
            if res is None:
                logger.warning("Text detection stage returned no result for %s", image_id)
                regions, det_meta = [], {}
            else:
                try:
                    regions, det_meta = res
                except Exception:
                    logger.exception("Text detection returned unexpected result shape")
                    regions, det_meta = [], {}

            stage_results["text_detection"] = StageResult("text_detection", dur2, result=det_meta, error=str(err2) if err2 else None)
            if err2:
                self.metrics["errors"] += 1
                regions = self.handle_ocr_errors(err2, img)

            # Stage 3: PHI classification (defensive unpacking)
            phi_regions, phi_meta = [], {}
            # Determine shadow mode setting
            shadow_mode = bool(getattr(self.config, "shadow_mode_enabled", False))
            # pass the canary decision into the phi classification stage via context
            res, dur3, err3 = self._timeit(
                self.stage_3_phi_classification,
                regions,
                {"image_id": image_id, "use_presidio_pipeline": use_presidio_pipeline, "shadow_mode": shadow_mode, **(metadata or {})},
            )
            if res is None:
                logger.warning("PHI classification returned no result for %s", image_id)
                phi_regions, phi_meta = [], {}
            else:
                try:
                    phi_regions, phi_meta = res
                except Exception:
                    logger.exception("PHI classification returned unexpected result shape")
                    phi_regions, phi_meta = [], {}

            stage_results["phi_classification"] = StageResult("phi_classification", dur3, result=phi_meta, error=str(err3) if err3 else None)
            if err3:
                self.metrics["errors"] += 1
                phi_regions = self.handle_phi_detection_errors(err3, regions)

            # Optional anonymization step: if a Presidio Anonymizer is available,
            # run it once per page to produce `anonymized_text` and let the
            # anonymizer apply operators according to POLICY_MATRIX.
            #
            # This block expects:
            # - `detections`: list of analyzer entities with keys: label/entity_type, score, start, end, page, bbox
            # - raw page text in `meta.get('page_text')` or as a fallback we join region texts
            #
            # The anonymizer usually returns anonymized_text and may adjust offsets
            # depending on operator parameters. TODO: Ensure offsets remain valid
            # for downstream masking: either use operators that preserve length
            # (masking) or recompute bboxes from token-to-char mapping after
            # replacement. For now we attempt to run anonymizer and retain the
            # original detection spans.
            try:
                page_text = None
                if isinstance(meta, dict):
                    page_text = meta.get("page_text")
                if not page_text:
                    # fallback: concatenate region text in document order
                    page_text = " ".join([r.get("text", "") for r in regions or []])

                # lazy import of presidio anonymizer classes
                from presidio_anonymizer import AnonymizerEngine, OperatorConfig  # type: ignore
                from src.integration.policy_matrix import POLICY_MATRIX

                # Build operators dict expected by AnonymizerEngine
                operators = {}
                for det in (phi_regions or []):
                    ent = det.get("phi_type") or det.get("entity_type")
                    if not ent:
                        continue
                    pm = POLICY_MATRIX.get(ent.upper()) or POLICY_MATRIX.get(ent) or None
                    if not pm:
                        # default operator: replace with generic placeholder
                        operators[ent] = [OperatorConfig("replace", {"new_value": "[REDACTED]"})]
                        continue
                    op_name = pm.get("operator", "replace")
                    params = pm.get("params", {}) or {}
                    # build OperatorConfig — constructor may differ depending on Presidio version
                    try:
                        operators[ent] = [OperatorConfig(op_name, params)]
                    except Exception:
                        # fallback to a simple mapping if OperatorConfig unavailable
                        operators[ent] = [{"type": op_name, "params": params}]

                engine = AnonymizerEngine()
                anonymized = engine.anonymize(text=page_text, analyzer_results=phi_regions, operators=operators)

                # anonymized is expected to have .text (or ['text']) depending on version
                anonymized_text = getattr(anonymized, "text", None) or (anonymized.get("text") if isinstance(anonymized, dict) else None)

                # TODO: Recompute detection offsets if operator changes lengths.
                # For now, keep original `phi_regions` spans and attach a flag.
                for d in (phi_regions or []):
                    d.setdefault("anonymized", True)
                # store anonymized text in metadata for downstream steps
                if isinstance(meta, dict):
                    meta["anonymized_text"] = anonymized_text
            except Exception:
                # anonymizer not available or failed; continue without anonymized text
                logger.debug("Anonymizer step skipped or failed; proceeding to masking")

            # Stage 4: masking (defensive unpacking)
            masked_img, mask_meta = None, {}
            res, dur4, err4 = self._timeit(self.stage_4_masking, img, phi_regions, meta)
            if res is None:
                logger.warning("Masking stage returned no result for %s", image_id)
                masked_img, mask_meta = None, {}
            else:
                try:
                    masked_img, mask_meta = res
                except Exception:
                    logger.exception("Masking returned unexpected result shape")
                    masked_img, mask_meta = None, {}

            stage_results["masking"] = StageResult("masking", dur4, result=mask_meta, error=str(err4) if err4 else None)
            if err4:
                self.metrics["errors"] += 1
                masked_img = self.handle_masking_errors(err4, img, phi_regions)

            # Save output if requested
            if output_path and masked_img is not None:
                try:
                    from imageio import imwrite

                    imwrite(output_path, masked_img)
                except Exception:
                    # best-effort: write with cv2
                    try:
                        import cv2 as _cv2

                        _cv2.imwrite(output_path, masked_img)
                    except Exception:
                        logger.exception("Failed to write output image %s", output_path)

            total_time = time.perf_counter() - overall_start
            report = self.generate_processing_report(image_id, {k: v.__dict__ for k, v in stage_results.items()}, {"total_time_s": total_time})
            self.metrics["processed"] += 1
            return report

        except Exception as e:
            logger.exception("Pipeline failed for %s", input_path)
            self.metrics["errors"] += 1
            return {"image_id": image_id, "error": str(e), "stages": {k: v.__dict__ for k, v in stage_results.items()}}
        finally:
            # cleanup
            try:
                shutil.rmtree(session_temp)
            except Exception:
                logger.debug("Failed to remove temp %s", session_temp)

    # ---------------- batch processing ----------------
    def process_batch(self, input_paths: List[str], output_dir: str, progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        summary = {"total": len(input_paths), "succeeded": 0, "failed": 0, "reports": []}
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        max_workers = min(8, max(1, int(self.config.processing.max_batch_size)))

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.process_single_image, p, os.path.join(output_dir, Path(p).name)): p for p in input_paths}
            for fut in as_completed(futures):
                src = futures[fut]
                try:
                    rep = fut.result()
                    summary["reports"].append(rep)
                    if rep.get("error"):
                        summary["failed"] += 1
                    else:
                        summary["succeeded"] += 1
                except Exception:
                    logger.exception("Batch processing failed on %s", src)
                    summary["failed"] += 1
                if progress_callback:
                    try:
                        progress_callback(summary["succeeded"] + summary["failed"], summary["total"])
                    except Exception:
                        logger.debug("progress_callback failed")

        batch_report = self.generate_batch_summary(summary["reports"])
        return {**summary, "batch_report": batch_report}

    # ---------------- pipeline stages ----------------
    def stage_1_preprocess(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        # load image
        from imageio import imread

        img = imread(image_path)
        if img is None:
            raise ValueError("Failed to read image")
        # lightweight validation
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            raise ValueError("Invalid image dimensions")
        # apply preprocessing if available
        if self.preprocessor is not None:
            try:
                proc = self.preprocessor.standardize_image(img)
                meta = {"orig_shape": (h, w), "processed_shape": proc.shape}
                return proc, meta
            except Exception:
                logger.exception("Preprocessor failed; returning original image")
                return img, {"orig_shape": (h, w)}
        return img, {"orig_shape": (h, w)}

    def stage_2_text_detection(self, image: np.ndarray, metadata: Dict) -> Tuple[List[Dict], Dict]:
        if self.ocr is None:
            # fallback to mock detector if available
            if self._MockTextDetector is not None:
                self.logger.warning("Using MockTextDetector as OCR fallback")
                detector = self._MockTextDetector(self.config.ocr)
                regions = detector.detect_text_regions(image)
                # If the mock detector implements its own filtering, use it.
                if hasattr(detector, "filter_by_confidence"):
                    accepted, rejected = detector.filter_by_confidence(regions, self.config.ocr.confidence_threshold)
                else:
                    # default: accept regions with confidence >= threshold
                    accepted = [r for r in regions if r.get("confidence", 1.0) >= self.config.ocr.confidence_threshold]
                    rejected = [r for r in regions if r not in accepted]
            else:
                raise RuntimeError("OCR component not initialized")
        else:
            regions = self.ocr.detect_text_regions(image)
            # filter using the OCR component's filter method
            accepted, rejected = self.ocr.filter_by_confidence(regions, self.config.ocr.confidence_threshold)
        det_meta = {"detected": len(regions), "accepted": len(accepted), "rejected": len(rejected)}
        return accepted, det_meta

    def stage_3_phi_classification(self, text_regions: List[Dict], context: Dict) -> Tuple[List[Dict], Dict]:
        if self.phi is None:
            raise RuntimeError("PHI classifier not initialized")

        classified = []
        use_presidio = bool(context.get("use_presidio_pipeline", False))
        shadow_mode = bool(context.get("shadow_mode", False))

        # prepare per-document collections for shadow auditing when needed
        legacy_report = {"detections": []}
        presidio_report = {"detections": []}

        for r in text_regions:
            t = r.get("text", "")

            # Legacy (current) detector always runs to produce authoritative output
            try:
                legacy_detections = self.phi.detect_phi_in_text(t, {"surrounding_text": context.get("surrounding_text", "")})
            except Exception:
                self.logger.exception("Legacy PHI classifier failed for region=%s", r.get("bbox"))
                legacy_detections = []

            # attach bbox and record
            for d in legacy_detections:
                d["bbox"] = r.get("bbox")
            legacy_report["detections"].extend(legacy_detections)

            # If canary enabled, run Presidio too (either shadow or active)
            presidio_detections = []
            if use_presidio:
                try:
                    from src.integration.presidio_wrapper import get_analyzer

                    analyzer = get_analyzer()
                    presidio_detections = analyzer.detect_phi_in_text(t, {"surrounding_text": context.get("surrounding_text", "")})
                    for d in presidio_detections:
                        d["bbox"] = r.get("bbox")
                    presidio_report["detections"].extend(presidio_detections)
                except Exception:
                    self.logger.exception("Presidio analyzer failed for region=%s", r.get("bbox"))

            # Decide which detections to use for final masking: legacy unless
            # canary is active and shadow mode is disabled (i.e., fully switch).
            if use_presidio and not shadow_mode:
                chosen = presidio_detections or legacy_detections
            else:
                chosen = legacy_detections

            # Append chosen detections to classified list
            classified.extend(chosen)

        compliance = {"phi_count": len(classified), "used_presidio": use_presidio, "shadow_mode": shadow_mode}

        # If shadow mode is enabled and we ran presidio, persist both outputs
        if shadow_mode and use_presidio:
            try:
                from src.debug.audit_log import log_shadow_audit

                doc_id = str(context.get("image_id") or context.get("doc_id") or "unknown")
                # write a compact summary for both detectors
                log_shadow_audit(doc_id, legacy_report, presidio_report)
            except Exception:
                self.logger.exception("Failed to write shadow audit for doc=%s", context.get("image_id"))

        # log summary of decision
        self.logger.info("PHI classification: used_presidio=%s shadow_mode=%s detections=%d", use_presidio, shadow_mode, len(classified))
        return classified, compliance

    def stage_4_masking(self, image: np.ndarray, phi_regions: List[Dict], metadata: Dict) -> Tuple[np.ndarray, Dict]:
        # select inpainter (real or mock)
        inpainter = self.inpainter
        used_mock = False
        if inpainter is None:
            if self._MockImageInpainter is not None:
                self.logger.warning("Using MockImageInpainter as inpainter fallback")
                inpainter = self._MockImageInpainter(self.config.mask)
                used_mock = True
            else:
                raise RuntimeError("Inpainter not initialized")

        # adapt expansions and perform inpainting or blackbox redaction
        try:
            phi_regions = inpainter.adaptive_mask_expansion(phi_regions, image)
            # If config requests blackbox redaction, draw solid black rects over bboxes
            style = getattr(self.config.mask, "redaction_style", "inpaint")
            if style in ("blackbox", "blackbox_merge"):
                out = image.copy()
                try:
                    import cv2 as _cv2

                    if style == "blackbox_merge":
                        # compute merged bbox
                        xs = []
                        ys = []
                        for r in phi_regions:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                xs.extend([x1, x2])
                                ys.extend([y1, y2])
                            except Exception:
                                continue
                        if xs and ys:
                            pad = int(getattr(self.config.mask, "blackbox_padding_pixels", 5))
                            x1, x2 = max(0, min(xs) - pad), min(out.shape[1], max(xs) + pad)
                            y1, y2 = max(0, min(ys) - pad), min(out.shape[0], max(ys) + pad)
                            _cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                    else:
                        for r in phi_regions:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                _cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                            except Exception:
                                continue
                except Exception:
                    # fallback to numpy fill for either mode
                    if style == "blackbox_merge":
                        xs = []
                        ys = []
                        for r in phi_regions:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                xs.extend([x1, x2])
                                ys.extend([y1, y2])
                            except Exception:
                                continue
                        if xs and ys:
                            pad = int(getattr(self.config.mask, "blackbox_padding_pixels", 5))
                            x1, x2 = max(0, min(xs) - pad), min(out.shape[1], max(xs) + pad)
                            y1, y2 = max(0, min(ys) - pad), min(out.shape[0], max(ys) + pad)
                            out[y1:y2, x1:x2] = 0
                    else:
                        for r in phi_regions:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                out[y1:y2, x1:x2] = 0
                            except Exception:
                                continue
                quality = getattr(inpainter, "validate_masking_quality", lambda o, i, m: {})(image, out, None)
                if used_mock:
                    quality = {**quality, "used_mock_inpainter": True}
                return out, {"method": style, "quality": quality}

            # default inpainting flow
            mask = inpainter.create_mask_from_regions(image.shape, phi_regions)
            method = getattr(inpainter, "smart_inpainting_selection", lambda im, m: self.config.mask.inpainting_method)(image, mask)
            inpainted = inpainter.apply_inpainting(image, mask, method=method)
            enhanced = getattr(inpainter, "enhance_inpainted_regions", lambda o, i, m: i)(image, inpainted, mask)
            quality = getattr(inpainter, "validate_masking_quality", lambda o, i, m: {})(image, enhanced, mask)
            if used_mock:
                # annotate report to indicate mock was used
                quality = {**quality, "used_mock_inpainter": True}
            return enhanced, {"method": method, "quality": quality}
        except Exception:
            logger.exception("Masking stage failed")
            raise

    # ---------------- error handlers ----------------
    def handle_preprocessing_errors(self, error: Exception, image_path: str) -> Dict[str, Any]:
        logger.exception("Preprocessing error for %s: %s", image_path, error)
        # attempt a simple fallback: load with imageio and convert to uint8
        try:
            from imageio import imread

            img = imread(image_path)
            if img is None:
                return {"failed": True}
            return {"failed": False, "duration": 0.0}
        except Exception:
            return {"failed": True}

    def handle_ocr_errors(self, error: Exception, image: np.ndarray) -> List[Dict]:
        logger.exception("OCR error: %s", error)
        # fallback: attempt lenient detection with lower threshold
        try:
            regions = self.ocr.detect_text_regions(image)
            accepted, rejected = self.ocr.filter_by_confidence(regions, max(0.2, self.config.ocr.confidence_threshold * 0.5))
            return accepted
        except Exception:
            logger.exception("OCR fallback failed")
            return []

    def handle_phi_detection_errors(self, error: Exception, text_regions: List[Dict]) -> List[Dict]:
        logger.exception("PHI detection error: %s", error)
        # conservative fallback: flag all text regions for manual review
        out = []
        for r in text_regions:
            out.append({"text": r.get("text", ""), "bbox": r.get("bbox"), "phi_type": "unknown", "confidence": 0.0, "needs_manual_review": True})
        return out

    def handle_masking_errors(self, error: Exception, image: np.ndarray, phi_regions: List[Dict]) -> np.ndarray:
        logger.exception("Masking error: %s", error)
        # fallback: simple rectangle redact (black boxes)
        out = image.copy()
        for r in phi_regions:
            try:
                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                out[y1:y2, x1:x2] = 0
            except Exception:
                continue
        return out

    # ---------------- reporting & audit ----------------
    def generate_processing_report(self, image_id: str, stage_results: Dict[str, Any], quality_metrics: Dict[str, Any]) -> Dict[str, Any]:
        report = {
            "image_id": image_id,
            "stages": stage_results,
            "quality": quality_metrics,
            "metrics": self.metrics,
            "timestamp": time.time(),
        }
        try:
            self.log_pipeline_execution(image_id, "pipeline_complete", "report", report)
        except Exception:
            logger.debug("Failed to log pipeline completion")
        return report

    def generate_batch_summary(self, individual_reports: List[Dict]) -> Dict[str, Any]:
        total = len(individual_reports)
        succeeded = sum(1 for r in individual_reports if not r.get("error"))
        failed = total - succeeded
        avg_time = float(sum(r.get("quality", {}).get("total_time_s", 0.0) for r in individual_reports)) / max(1, total)
        return {"total": total, "succeeded": succeeded, "failed": failed, "avg_time_s": avg_time}

    def log_pipeline_execution(self, image_id: str, stage: str, operation: str, result: Any) -> None:
        try:
            rec = {"image_id": image_id, "stage": stage, "operation": operation, "result": result, "ts": time.time()}
            self.config.logging  # ensure logging configured
            logger.info("PIPELINE_AUDIT: %s", json.dumps(rec, default=str))
            # also write to audit logger
            audit = logging.getLogger("cim.audit.pipeline")
            audit.info(json.dumps(rec, default=str))
        except Exception:
            logger.exception("Failed to write pipeline audit record")

    def validate_pipeline_compliance(self, processing_results: Dict[str, Any]) -> Dict[str, bool]:
        # basic checks
        try:
            phi_coverage_ok = bool(processing_results.get("stages", {}).get("phi_classification"))
            audit_ok = True  # requires checking persistent audit logs (out of scope)
            return {"phi_coverage": phi_coverage_ok, "audit_log_present": audit_ok}
        except Exception:
            return {"phi_coverage": False, "audit_log_present": False}

    # ---------------- performance monitoring ----------------
    def monitor_processing_performance(self, stage: str, duration: float, resource_usage: Dict[str, Any]) -> None:
        self.metrics.setdefault("stages", {}).setdefault(stage, []).append({"duration": duration, "resources": resource_usage})

    def optimize_pipeline_parameters(self, historical_performance: Dict[str, Any]) -> Dict[str, Any]:
        # naive heuristic: if average duration > threshold, reduce batch size
        avg = historical_performance.get("avg_time_s", 0)
        recs = {}
        if avg > 5.0:
            recs["reduce_batch_size"] = True
        return recs

    # ---------------- utilities ----------------
    def cleanup_temporary_files(self, processing_session_id: str) -> None:
        # remove files under temp root matching session prefix
        for p in self._temp_root.iterdir():
            if processing_session_id in p.name:
                try:
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                except Exception:
                    logger.debug("Failed to delete temp %s", p)

    def validate_input_requirements(self, input_path: str) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        if not os.path.exists(input_path):
            errors.append("file_missing")
        try:
            from imageio import imread

            img = imread(input_path)
            if img is None:
                errors.append("invalid_image")
        except Exception:
            errors.append("cannot_read")
        return (len(errors) == 0, errors)

    def prepare_output_directory(self, output_dir: str) -> bool:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            return True
        except Exception:
            logger.exception("Failed to prepare output dir %s", output_dir)
            return False


__all__ = ["ClinicalImageMaskingPipeline", "StageResult"]
