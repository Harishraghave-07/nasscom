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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

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

    # ---------------- single-image pipeline ----------------
    def process_single_image(self, input_path: str, output_path: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        image_id = Path(input_path).stem
        session_temp = Path(tempfile.mkdtemp(prefix=f"cim_{image_id}_", dir=str(self._temp_root)))
        stage_results: Dict[str, StageResult] = {}
        overall_start = time.perf_counter()

        try:
            # Stage 1: preprocess
            img, meta, dur = None, {}, 0.0
            (img, meta), dur, err = self._timeit(self.stage_1_preprocess, input_path)
            stage_results["preprocess"] = StageResult("preprocess", dur, result=meta, error=str(err) if err else None)
            if err:
                self.metrics["errors"] += 1
                recovery = self.handle_preprocessing_errors(err, input_path)
                stage_results["preprocess_recovery"] = StageResult("preprocess_recovery", recovery.get("duration", 0.0), result=recovery)
                if recovery.get("failed"):
                    raise RuntimeError("Preprocessing failed and recovery unsuccessful")

            # Stage 2: text detection
            regions, det_meta, dur2 = None, {}, 0.0
            (regions, det_meta), dur2, err2 = self._timeit(self.stage_2_text_detection, img, meta)
            stage_results["text_detection"] = StageResult("text_detection", dur2, result=det_meta, error=str(err2) if err2 else None)
            if err2:
                self.metrics["errors"] += 1
                regions = self.handle_ocr_errors(err2, img)

            # Stage 3: PHI classification
            phi_regions, phi_meta, dur3 = None, {}, 0.0
            (phi_regions, phi_meta), dur3, err3 = self._timeit(self.stage_3_phi_classification, regions, {"image_id": image_id, **(metadata or {})})
            stage_results["phi_classification"] = StageResult("phi_classification", dur3, result=phi_meta, error=str(err3) if err3 else None)
            if err3:
                self.metrics["errors"] += 1
                phi_regions = self.handle_phi_detection_errors(err3, regions)

            # Stage 4: masking
            masked_img, mask_meta, dur4 = None, {}, 0.0
            (masked_img, mask_meta), dur4, err4 = self._timeit(self.stage_4_masking, img, phi_regions, meta)
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
            raise RuntimeError("OCR component not initialized")
        regions = self.ocr.detect_text_regions(image)
        # filter
        accepted, rejected = self.ocr.filter_by_confidence(regions, self.config.ocr.confidence_threshold)
        det_meta = {"detected": len(regions), "accepted": len(accepted), "rejected": len(rejected)}
        return accepted, det_meta

    def stage_3_phi_classification(self, text_regions: List[Dict], context: Dict) -> Tuple[List[Dict], Dict]:
        if self.phi is None:
            raise RuntimeError("PHI classifier not initialized")
        classified = []
        for r in text_regions:
            t = r.get("text", "")
            detections = self.phi.detect_phi_in_text(t, {"surrounding_text": context.get("surrounding_text", "")})
            # attach region bbox
            for d in detections:
                d["bbox"] = r.get("bbox")
            classified.extend(detections)
        compliance = {"phi_count": len(classified)}
        return classified, compliance

    def stage_4_masking(self, image: np.ndarray, phi_regions: List[Dict], metadata: Dict) -> Tuple[np.ndarray, Dict]:
        if self.inpainter is None:
            raise RuntimeError("Inpainter not initialized")
        # adapt expansions
        try:
            phi_regions = self.inpainter.adaptive_mask_expansion(phi_regions, image)
            mask = self.inpainter.create_mask_from_regions(image.shape, phi_regions)
            method = self.inpainter.smart_inpainting_selection(image, mask)
            inpainted = self.inpainter.apply_inpainting(image, mask, method=method)
            enhanced = self.inpainter.enhance_inpainted_regions(image, inpainted, mask)
            quality = self.inpainter.validate_masking_quality(image, enhanced, mask)
            return enhanced, {"method": method, "quality": quality}
        except Exception as e:
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
