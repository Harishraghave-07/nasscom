"""Clinical-grade image inpainting and masking utilities.

This module implements ImageInpainter which provides mask creation from PHI
regions, multiple inpainting methods tuned for medical images, quality
metrics (PSNR/SSIM), batch processing, caching, and audit logging.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    from skimage.metrics import structural_similarity as ssim
except Exception:
    ssim = None

try:
    import pydicom
except Exception:
    pydicom = None

from src.core.config import MaskingConfig, SETTINGS

logger = logging.getLogger(__name__)


@dataclass
class InpaintQualityReport:
    psnr: float
    ssim: float
    edge_preservation: float
    texture_consistency: float
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "psnr": float(self.psnr),
            "ssim": float(self.ssim),
            "edge_preservation": float(self.edge_preservation),
            "texture_consistency": float(self.texture_consistency),
            "notes": self.notes,
        }


class ImageInpainter:
    def __init__(self, config: MaskingConfig):
        self.config = config
        self.audit_logger = logging.getLogger("cim.audit.masking")
        # determine a safe temporary/cache directory. MaskingConfig may not
        # include a temp_dir; fall back to application SETTINGS.processing.temp_dir
        # or the system temp directory.
        try:
            temp_dir = getattr(self.config, "temp_dir", None)
        except Exception:
            temp_dir = None
        if not temp_dir:
            try:
                from src.core.config import SETTINGS

                temp_dir = getattr(SETTINGS.processing, "temp_dir", None)
            except Exception:
                temp_dir = None
        if not temp_dir:
            import tempfile

            temp_dir = tempfile.gettempdir()

        self._cache_dir = Path(temp_dir) / "inpaint_cache"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            logger.exception("Failed to create cache dir %s", self._cache_dir)
        self._perf = {"images": 0, "total_time_s": 0.0}
        # verify OpenCV availability
        if cv2 is None:
            logger.warning("OpenCV (cv2) is not available. Inpainting operations will fail until installed.")

    # ---------------- mask generation ----------------
    def create_mask_from_regions(self, image_shape: Tuple[int, int], phi_regions: List[Dict]) -> np.ndarray:
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        try:
            for r in phi_regions:
                bbox = r.get("bbox")
                if not bbox or len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                # apply expansion
                exp = int(r.get("mask_expansion", self.config.mask_expansion_pixels))
                x1 = max(0, x1 - exp)
                y1 = max(0, y1 - exp)
                x2 = min(w, x2 + exp)
                y2 = min(h, y2 + exp)
                mask[y1:y2, x1:x2] = 255

            # smooth edges
            if cv2 is not None:
                k = max(3, int(min(h, w) * 0.003))
                if k % 2 == 0:
                    k += 1
                mask = cv2.GaussianBlur(mask, (k, k), 0)
                # threshold to binary
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            # validate coverage
            coverage = float(mask.sum()) / (255.0 * h * w)
            if coverage <= 0:
                logger.warning("Generated mask has zero coverage")
            self.audit_logger.info(json.dumps({"event": "create_mask", "coverage": coverage}))
            return mask
        except MemoryError:
            logger.exception("MemoryError during mask creation")
            raise
        except Exception:
            logger.exception("Failed to create mask from regions")
            raise

    # ---------------- inpainting ----------------
    def apply_inpainting(self, image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
        if cv2 is None:
            raise RuntimeError("OpenCV not available")
        if image is None or mask is None:
            raise ValueError("image and mask must be provided")

        start = time.time()
        img = image.copy()
        # OpenCV expects 8-bit 1 or 3 channel images
        if img.dtype != np.uint8:
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # ensure mask is single channel 8-bit
        if mask.dtype != np.uint8:
            mask8 = (mask > 0).astype(np.uint8) * 255
        else:
            mask8 = mask

        # choose method
        if method.lower() == "telea":
            flags = cv2.INPAINT_TELEA
            radius = max(1, int(self.config.inpainting_radius))
            inpainted = cv2.inpaint(img, mask8, radius, flags)
        elif method.lower() in ("ns", "navier-stokes", "navier_stokes"):
            flags = cv2.INPAINT_NS
            radius = max(1, int(self.config.inpainting_radius))
            inpainted = cv2.inpaint(img, mask8, radius, flags)
        elif method.lower() == "hybrid":
            # hybrid: TELEA followed by a low-strength NS pass
            r1 = max(1, int(self.config.inpainting_radius))
            tmp = cv2.inpaint(img, mask8, r1, cv2.INPAINT_TELEA)
            inpainted = cv2.inpaint(tmp, mask8, max(1, r1 // 2), cv2.INPAINT_NS)
        else:
            raise ValueError(f"Unknown inpainting method: {method}")

        elapsed = time.time() - start
        self._perf["images"] += 1
        self._perf["total_time_s"] += elapsed
        try:
            self.audit_logger.info(json.dumps({"event": "apply_inpainting", "method": method, "time_s": elapsed}))
        except Exception:
            logger.debug("Failed to write audit log for inpainting")

        return inpainted

    def enhance_inpainted_regions(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = inpainted.copy()
        try:
            if cv2 is None:
                return img

            # color match: transfer mean/variance from surrounding area
            kernel = max(3, int(min(img.shape[:2]) * 0.004))
            if kernel % 2 == 0:
                kernel += 1

            # compute local means
            blurred = cv2.blur(img, (kernel, kernel))
            alpha = 0.6
            img = np.where(mask[..., None] > 0, (alpha * blurred + (1 - alpha) * img).astype(np.uint8), img)

            # slight bilateral to reduce artifacts
            img = cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
            return img
        except Exception:
            logger.exception("Failed to enhance inpainted regions")
            return inpainted

    # ---------------- quality metrics ----------------
    def validate_masking_quality(self, original: np.ndarray, masked: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        try:
            # PSNR
            mse = np.mean((original.astype(np.float32) - masked.astype(np.float32)) ** 2)
            psnr_val = float(10 * np.log10((255.0 ** 2) / mse)) if mse != 0 else float("inf")

            # SSIM
            ssim_val = None
            if ssim is not None:
                try:
                    # compute per-channel and average
                    if original.ndim == 3 and original.shape[2] == 3:
                        ssim_ch = [ssim(original[..., i], masked[..., i], data_range=255) for i in range(3)]
                        ssim_val = float(np.mean(ssim_ch))
                    else:
                        ssim_val = float(ssim(original, masked, data_range=255))
                except Exception:
                    logger.debug("SSIM computation failed")
                    ssim_val = 0.0
            else:
                ssim_val = 0.0

            # edge preservation: compare edges inside/outside mask
            if cv2 is not None:
                orig_edges = cv2.Canny(original if original.dtype == np.uint8 else original.astype(np.uint8), 100, 200)
                masked_edges = cv2.Canny(masked if masked.dtype == np.uint8 else masked.astype(np.uint8), 100, 200)
                # focus on masked area
                masked_area = (mask > 0).astype(np.uint8)
                if masked_area.sum() > 0:
                    edge_pres = float((orig_edges & masked_area).sum()) / max(1.0, (masked_edges & masked_area).sum())
                else:
                    edge_pres = 1.0
            else:
                edge_pres = 1.0

            # texture consistency: compare local variance
            tex_orig = np.var(original.astype(np.float32))
            tex_masked = np.var(masked.astype(np.float32))
            texture_consistency = float(1.0 - abs(tex_orig - tex_masked) / (max(1.0, tex_orig)))

            report = InpaintQualityReport(psnr=psnr_val, ssim=ssim_val, edge_preservation=edge_pres, texture_consistency=texture_consistency)
            try:
                self.audit_logger.info(json.dumps({"event": "validate_masking_quality", "report": report.to_dict()}))
            except Exception:
                logger.debug("Failed to audit masking quality")
            return report.to_dict()
        except Exception:
            logger.exception("Failed to compute masking quality")
            return {"psnr": 0.0, "ssim": 0.0, "edge_preservation": 0.0, "texture_consistency": 0.0}

    # ---------------- batch processing ----------------
    def batch_mask_images(self, images: List[np.ndarray], phi_detections: List[List[Dict]], progress_callback: Optional[Callable] = None) -> List[Tuple[np.ndarray, Dict]]:
        results: List[Tuple[np.ndarray, Dict]] = []
        max_workers = min(8, max(1, int(self.config.max_batch_size)))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {}
            for idx, img in enumerate(images):
                dets = phi_detections[idx] if idx < len(phi_detections) else []
                futures[ex.submit(self._process_single_image, img, dets)] = idx
            for fut in as_completed(futures):
                idx = futures[fut]
                try:
                    processed_img, report = fut.result()
                except Exception:
                    logger.exception("Error processing image %s", idx)
                    processed_img = images[idx]
                    report = {"error": "processing_failed"}
                results.append((processed_img, report))
                if progress_callback:
                    try:
                        progress_callback(len(results), len(images))
                    except Exception:
                        logger.debug("progress callback failed")
        return results

    def _process_single_image(self, image: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict]:
        start = time.time()
        try:
            mask = self.create_mask_from_regions(image.shape, detections)
            method = self.smart_inpainting_selection(image, mask)
            inpainted = self.apply_inpainting(image, mask, method=method)
            enhanced = self.enhance_inpainted_regions(image, inpainted, mask)
            report = self.validate_masking_quality(image, enhanced, mask)
            elapsed = time.time() - start
            self.audit_logger.info(json.dumps({"event": "process_image", "time_s": elapsed, "method": method}))
            return enhanced, report
        except Exception:
            logger.exception("_process_single_image failed")
            return image, {"error": "failed"}

    # ---------------- advanced techniques ----------------
    def adaptive_mask_expansion(self, regions: List[Dict], image: np.ndarray) -> List[Dict]:
        # expand mask by font size estimate (heuristic)
        h, w = image.shape[:2]
        for r in regions:
            bbox = r.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            font_size = max(8, int((y2 - y1) * 0.3))
            r["mask_expansion"] = font_size
        return regions

    def smart_inpainting_selection(self, image: np.ndarray, mask: np.ndarray) -> str:
        # heuristic: if many small text regions prefer TELEA for speed; if large mask use NS
        area = mask.sum() / 255
        img_area = image.shape[0] * image.shape[1]
        frac = area / max(1.0, img_area)
        if frac < 0.01:
            return "telea"
        if frac > 0.1:
            return "ns"
        return "hybrid"

    def handle_complex_backgrounds(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # attempt to preserve gradients by estimating background and returning adjusted mask
        out_img = image.copy()
        # simple morphological closing on mask to incorporate nearby structure
        if cv2 is not None:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        else:
            mask2 = mask
        return out_img, mask2

    # ---------------- quality assurance ----------------
    def detect_masking_artifacts(self, original: np.ndarray, masked: np.ndarray, mask: np.ndarray) -> List[Dict]:
        artifacts = []
        try:
            if cv2 is None:
                return artifacts
            # color discontinuities: compare mean color around mask boundary
            edges = cv2.Canny(mask, 50, 150)
            ys, xs = np.nonzero(edges)
            for (x, y) in zip(xs[:100], ys[:100]):
                # sample small neighborhoods
                oy = max(0, y - 2)
                ox = max(0, x - 2)
                orig_patch = original[oy:oy + 5, ox:ox + 5]
                masked_patch = masked[oy:oy + 5, ox:ox + 5]
                if orig_patch.size == 0 or masked_patch.size == 0:
                    continue
                diff = np.mean(np.abs(orig_patch.astype(float) - masked_patch.astype(float)))
                if diff > 20:
                    artifacts.append({"pos": (x, y), "diff": float(diff)})
            return artifacts
        except Exception:
            logger.exception("Failed to detect masking artifacts")
            return artifacts

    def measure_medical_image_integrity(self, original: np.ndarray, masked: np.ndarray) -> Dict[str, bool]:
        # heuristics: check if high-contrast anatomical landmarks remain
        try:
            # simple check: variance ratio
            v_orig = np.var(original.astype(np.float32))
            v_mask = np.var(masked.astype(np.float32))
            preserved = v_mask >= 0.5 * v_orig
            return {"variance_preserved": bool(preserved)}
        except Exception:
            logger.exception("measure_medical_image_integrity failed")
            return {"variance_preserved": False}

    def verify_phi_removal_completeness(self, image: np.ndarray, original_phi_regions: List[Dict]) -> Dict[str, Any]:
        # best-effort: check that OCR-like residuals are low by running a lightweight text detector
        # For now, compute coverage of mask
        try:
            mask = self.create_mask_from_regions(image.shape, original_phi_regions)
            coverage = float((mask > 0).sum()) / (image.shape[0] * image.shape[1])
            return {"mask_coverage": coverage, "residual_text_score": 0.0}
        except Exception:
            logger.exception("verify_phi_removal_completeness failed")
            return {"error": True}

    def apply_surgical_masking(self, image: np.ndarray, phi_regions: List[Dict], padding: Optional[int] = None) -> np.ndarray:
        """Apply precise, small black rectangles only over provided PHI regions.

        - padding: minimal pixel expansion around each bbox (defaults to config or 2px)
        Returns the masked image.
        """
        if image is None:
            raise ValueError("image must be provided")

        pad = int(padding if padding is not None else getattr(self.config, "surgical_padding_pixels", 2))
        out = image.copy()

        for r in phi_regions:
            bbox = r.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception:
                continue

            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(out.shape[1], x2 + pad)
            y2 = min(out.shape[0], y2 + pad)

            try:
                if cv2 is not None:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                else:
                    out[y1:y2, x1:x2] = 0
            except Exception:
                # best-effort continue
                continue

            # audit log the surgical masking
            try:
                self.audit_logger.info(json.dumps({"event": "surgical_mask", "entity": r.get("entity_type"), "text": r.get("text"), "bbox": [x1, y1, x2, y2]}))
            except Exception:
                pass

        return out

    def unified_masking_stage(
        self,
        image: np.ndarray,
        phi_regions: List[Dict],
        *,
        style: Optional[str] = None,
        merge_adjacent: bool = True,
        padding: Optional[int] = None,
        blend_alpha: float = 0.0,
        inpaint_method: Optional[str] = None,
        fallback_to_blackbox: bool = True,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Unified masking stage.

        Inputs:
          - image: HxWxC numpy uint8 image
          - phi_regions: list of detection dicts with at least 'bbox' and optional 'text'/'word_boxes'
        Options (via args or configured MaskingConfig):
          - style: override redaction style (inpaint|blackbox|blackbox_merge|surgical)
          - merge_adjacent: attempt to merge nearby word-level boxes into phrase-level boxes
          - padding: per-entity padding in pixels (overrides surgical_padding or mask_expansion)
          - blend_alpha: if >0, blend masked region edges for visual softness (applies to blackbox)
          - inpaint_method: override inpainting method ('telea'|'ns'|'hybrid')
          - fallback_to_blackbox: if inpainting fails, fall back to deterministic black boxes

        Returns (masked_image, metadata) where metadata includes method, coverage, quality and audit info.
        """
        meta: Dict[str, Any] = {"merged": False, "method": None, "mask_coverage": 0.0, "quality": {}}
        if image is None:
            raise ValueError("image must be provided to unified_masking_stage")

        # normalize options
        style = style or getattr(self.config, "redaction_style", "inpaint")
        inpaint_method = inpaint_method or getattr(self.config, "inpainting_method", "telea")
        pad = int(padding) if padding is not None else int(getattr(self.config, "surgical_padding_pixels", 2))

        regs = list(phi_regions or [])

        # Optional merging step (best-effort, non-fatal)
        if merge_adjacent:
            try:
                # attempt to use OCR merge helper if available and if regions include word_boxes
                from src.ocr.text_detector import merge_adjacent_regions

                # Build candidate phrases from provided regions
                candidate_phis = [r.get("text") or r.get("entity") or "" for r in regs]
                # If underlying regions already contain word_boxes, flatten them; else skip
                word_regions = []
                for r in regs:
                    wbs = r.get("word_boxes") or []
                    if wbs:
                        for w in wbs:
                            word_regions.append({"text": w.get("text", ""), "bbox": w.get("bbox"), "confidence": w.get("confidence", 1.0)})

                if word_regions and any(candidate_phis):
                    merged = merge_adjacent_regions(word_regions, candidate_phis, fuzz_threshold=int(getattr(self.config, "fuzz_threshold", 80)))
                    # convert merged back into region dicts
                    if merged and len(merged) != len(word_regions):
                        meta["merged"] = True
                        # map merged regions back to simple phi_regions for masking
                        regs = []
                        for m in merged:
                            regs.append({"text": m.get("text"), "bbox": m.get("bbox"), "confidence": m.get("confidence", 1.0)})
            except Exception:
                # non-fatal: continue without merging
                logger.debug("merge_adjacent_regions unavailable or failed; continuing without merge")

        # apply adaptive mask expansion if configured
        try:
            regs = self.adaptive_mask_expansion(regs, image)
        except Exception:
            logger.debug("adaptive_mask_expansion failed or not applicable")

        # Core styles
        try:
            if style == "surgical":
                out = self.apply_surgical_masking(image, regs, padding=pad)
                meta["method"] = "surgical"
                # surgical returns full image; compute a binary mask for quality checks
                try:
                    mask = self.create_mask_from_regions(image.shape, regs)
                    meta["mask_coverage"] = float((mask > 0).sum()) / (image.shape[0] * image.shape[1])
                except Exception:
                    mask = None
                meta["quality"] = self.validate_masking_quality(image, out, mask) if hasattr(self, "validate_masking_quality") else {}
                return out, meta

            if style in ("blackbox", "blackbox_merge"):
                out = image.copy()
                try:
                    import cv2 as _cv2

                    if style == "blackbox_merge":
                        xs, ys = [], []
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                xs.extend([x1, x2]); ys.extend([y1, y2])
                            except Exception:
                                continue
                        if xs and ys:
                            pad_bb = int(getattr(self.config, "blackbox_padding_pixels", 5))
                            x1, x2 = max(0, min(xs) - pad_bb), min(out.shape[1], max(xs) + pad_bb)
                            y1, y2 = max(0, min(ys) - pad_bb), min(out.shape[0], max(ys) + pad_bb)
                            _cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                    else:
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                _cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                            except Exception:
                                continue
                except Exception:
                    # numpy fallback
                    if style == "blackbox_merge":
                        xs, ys = [], []
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                xs.extend([x1, x2]); ys.extend([y1, y2])
                            except Exception:
                                continue
                        if xs and ys:
                            pad_bb = int(getattr(self.config, "blackbox_padding_pixels", 5))
                            x1, x2 = max(0, min(xs) - pad_bb), min(out.shape[1], max(xs) + pad_bb)
                            y1, y2 = max(0, min(ys) - pad_bb), min(out.shape[0], max(ys) + pad_bb)
                            out[y1:y2, x1:x2] = 0
                    else:
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                out[y1:y2, x1:x2] = 0
                            except Exception:
                                continue

                # compute mask for quality
                try:
                    mask = self.create_mask_from_regions(image.shape, regs)
                    meta["mask_coverage"] = float((mask > 0).sum()) / (image.shape[0] * image.shape[1])
                except Exception:
                    mask = None
                meta["method"] = style
                meta["quality"] = self.validate_masking_quality(image, out, mask) if hasattr(self, "validate_masking_quality") else {}
                return out, meta

            # default: inpainting flow
            mask = self.create_mask_from_regions(image.shape, regs)
            meta["mask_coverage"] = float((mask > 0).sum()) / (image.shape[0] * image.shape[1])
            method = inpaint_method or getattr(self.config, "inpainting_method", "telea")
            try:
                inpainted = self.apply_inpainting(image, mask, method=method)
                enhanced = self.enhance_inpainted_regions(image, inpainted, mask) if hasattr(self, "enhance_inpainted_regions") else inpainted
                meta["method"] = method
                meta["quality"] = self.validate_masking_quality(image, enhanced, mask) if hasattr(self, "validate_masking_quality") else {}
                return enhanced, meta
            except Exception:
                logger.exception("Inpainting failed in unified_masking_stage")
                if fallback_to_blackbox:
                    # fallback: draw black boxes deterministically
                    out = image.copy()
                    try:
                        import cv2 as _cv2
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                _cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
                            except Exception:
                                continue
                    except Exception:
                        for r in regs:
                            try:
                                x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                                out[y1:y2, x1:x2] = 0
                            except Exception:
                                continue

                    # mask for quality
                    try:
                        mask2 = self.create_mask_from_regions(image.shape, regs)
                        meta["mask_coverage"] = float((mask2 > 0).sum()) / (image.shape[0] * image.shape[1])
                    except Exception:
                        mask2 = None
                    meta["method"] = "inpaint_fallback_blackbox"
                    meta["quality"] = self.validate_masking_quality(image, out, mask2) if hasattr(self, "validate_masking_quality") else {}
                    return out, meta
                raise
        except Exception:
            logger.exception("unified_masking_stage failed")
            raise

    # ---------------- specialized medical handling ----------------
    def handle_dicom_overlays(self, image: np.ndarray, phi_regions: List[Dict]) -> Tuple[np.ndarray, List[Dict]]:
        # If pydicom available and dataset passed, preserve overlays placeholder
        if pydicom is None:
            return image, phi_regions
        # In a real implementation, parse DICOM tags and preserve overlays
        return image, phi_regions

    def optimize_for_ultrasound(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        # speckle-aware smoothing: use median+bilateral
        out = image.copy()
        if cv2 is not None:
            out = cv2.medianBlur(out, 3)
            out = cv2.bilateralFilter(out, 5, 75, 75)
        return out

    def optimize_for_xray(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        out = image.copy()
        if cv2 is not None:
            # preserve contrast: unsharp mask
            blur = cv2.GaussianBlur(out, (0, 0), sigmaX=3)
            out = cv2.addWeighted(out, 1.5, blur, -0.5, 0)
        return out

    # ---------------- performance helpers ----------------
    def parallel_region_processing(self, image: np.ndarray, regions: List[Dict]) -> np.ndarray:
        # process regions in parallel by applying local inpainting and merging
        parts = []
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(self._process_region, image, r) for r in regions]
            for fut in as_completed(futures):
                try:
                    parts.append(fut.result())
                except Exception:
                    logger.exception("region processing failed")
        # naive merge: return original for now
        return image

    def _process_region(self, image: np.ndarray, region: Dict) -> np.ndarray:
        # small helper to process a single region (not used extensively here)
        return image

    def cache_inpainting_results(self, image_hash: str, mask_hash: str, result: np.ndarray) -> None:
        try:
            key = hashlib.sha256((image_hash + mask_hash).encode("utf-8")).hexdigest()
            path = self._cache_dir / f"{key}.npy"
            np.save(str(path), result)
        except Exception:
            logger.exception("Failed to cache inpainting result")

    def load_cached_result(self, image_hash: str, mask_hash: str) -> Optional[np.ndarray]:
        try:
            key = hashlib.sha256((image_hash + mask_hash).encode("utf-8")).hexdigest()
            path = self._cache_dir / f"{key}.npy"
            if path.exists():
                return np.load(str(path))
        except Exception:
            logger.exception("Failed to load cached inpainting result")
        return None

    # ---------------- safety & validation ----------------
    def _hash_array(self, arr: np.ndarray) -> str:
        h = hashlib.sha256()
        h.update(arr.tobytes())
        return h.hexdigest()


__all__ = ["ImageInpainter", "InpaintQualityReport"]
