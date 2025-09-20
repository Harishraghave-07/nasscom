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
            # respect global in-memory temp setting to avoid persistent cache
            use_in_memory = False
            try:
                use_in_memory = bool(getattr(SETTINGS.processing, "use_in_memory_temp", False))
            except Exception:
                use_in_memory = False
            if not use_in_memory:
                self._cache_dir.mkdir(parents=True, exist_ok=True)
            else:
                # when in-memory, keep cache_dir as Path but do not create it
                logger.debug("In-memory temp enabled; skipping creation of inpaint cache dir %s", self._cache_dir)
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
        for r in phi_regions:
            # prefer masking at the word level when available to avoid over-masking labels
            word_boxes = r.get("word_boxes") or []
            if word_boxes:
                for wb in word_boxes:
                    bbox = wb.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    try:
                        x1, y1, x2, y2 = map(int, bbox)
                    except Exception:
                        continue
                    try:
                        exp = int(wb.get("mask_expansion", r.get("mask_expansion", self.config.mask_expansion_pixels)))
                    except Exception:
                        exp = int(self.config.mask_expansion_pixels)
                    x1 = max(0, x1 - exp)
                    y1 = max(0, y1 - exp)
                    x2 = min(w, x2 + exp)
                    y2 = min(h, y2 + exp)
                    if x1 >= x2 or y1 >= y2:
                        continue
                    mask[y1:y2, x1:x2] = 255
                continue

            # fallback to region bbox
            bbox = r.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            try:
                x1, y1, x2, y2 = map(int, bbox)
            except Exception as exc:
                logger.error("Invalid bbox for region %s: %s", r, exc)
                raise ValueError(f"Invalid bbox in region: {r}") from exc
            # apply expansion
            try:
                exp = int(r.get("mask_expansion", self.config.mask_expansion_pixels))
            except Exception:
                exp = int(self.config.mask_expansion_pixels)
            x1 = max(0, x1 - exp)
            y1 = max(0, y1 - exp)
            x2 = min(w, x2 + exp)
            y2 = min(h, y2 + exp)
            # validate coordinates
            if x1 >= x2 or y1 >= y2:
                logger.error("Computed invalid expanded bbox for region %s -> (%d,%d,%d,%d)", r, x1, y1, x2, y2)
                raise ValueError(f"Invalid expanded bbox for region: {r}")
            mask[y1:y2, x1:x2] = 255

        # smooth edges
        try:
            if cv2 is not None:
                k = max(3, int(min(h, w) * 0.003))
                if k % 2 == 0:
                    k += 1
                mask = cv2.GaussianBlur(mask, (k, k), 0)
                # threshold to binary
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        except (cv2.error, ValueError) as exc:
            logger.exception("Mask smoothing step failed with cv2 error: %s", exc)
            # surface as ValueError for callers to handle deterministically
            raise ValueError("Mask smoothing failed") from exc
        except Exception as exc:
            # unexpected error: log and re-raise so callers see failure
            logger.exception("Unexpected error during mask smoothing: %s", exc)
            raise

        # validate coverage
        coverage = float(mask.sum()) / (255.0 * h * w)
        if coverage <= 0:
            logger.warning("Generated mask has zero coverage")
        try:
            self.audit_logger.info(json.dumps({"event": "create_mask", "coverage": coverage}))
        except Exception:
            logger.debug("Failed to audit mask creation")
        return mask

    def draw_debug_overlays(self, image: np.ndarray, word_boxes: List[Dict], phi_bboxes: List[Dict], applied_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Return a copy of image with overlays for debug visualization.

        - word_boxes: list of word box dicts ({'bbox':[x1,y1,x2,y2], 'text':...}) drawn in blue
        - phi_bboxes: list of detection dicts (with 'bbox') drawn in red
        - applied_mask: binary mask drawn as semi-transparent green
        """
        out = image.copy()
        try:
            import cv2 as _cv2

            # draw word boxes (blue)
            for wb in word_boxes or []:
                try:
                    bx = list(map(int, wb.get("bbox", [])))
                    if len(bx) != 4:
                        continue
                    _cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), (255, 0, 0), 1)
                except Exception:
                    continue

            # draw phi bboxes (red)
            for p in phi_bboxes or []:
                try:
                    bx = list(map(int, p.get("bbox", [])))
                    if len(bx) != 4:
                        continue
                    _cv2.rectangle(out, (bx[0], bx[1]), (bx[2], bx[3]), (0, 0, 255), 2)
                except Exception:
                    continue

            # overlay mask (green) with alpha
            if applied_mask is not None:
                try:
                    colored = _cv2.cvtColor(out, _cv2.COLOR_RGB2RGBA) if out.shape[2] == 3 else _cv2.cvtColor(out, _cv2.COLOR_BGR2BGRA)
                    alpha = 0.35
                    mask_rgb = _cv2.merge([_cv2.threshold(applied_mask, 127, 255, _cv2.THRESH_BINARY)[1]] * 3 + [(_cv2.threshold(applied_mask, 127, 255, _cv2.THRESH_BINARY)[1] * int(255 * alpha)).astype('uint8')])
                    # blend where mask present
                    mask_bool = applied_mask > 127
                    colored[mask_bool] = (_cv2.addWeighted(colored, 1.0, mask_rgb, alpha, 0))[mask_bool]
                    out = _cv2.cvtColor(colored, _cv2.COLOR_RGBA2RGB)
                except Exception:
                    logger.exception("Failed to overlay mask visualization")
        except Exception:
            # cv2 not available or overlay failed; return original copy
            logger.debug("Debug overlay generation skipped (cv2 missing or error)")
        return out

    # ---------------- language heuristics ----------------
    def detect_language_group(self, phi_regions: List[Dict]) -> str:
        """Coarse language group detection.

        This deployment supports only English. Forcing 'latin' group so masking
        heuristics remain English-centric and avoid CJK-specific padding.
        """
        return "latin"

    def get_language_heuristics(self, lang_group: str) -> Dict[str, int]:
        """Return heuristic parameters for a coarse language group."""
        heur = {
            "padding": int(getattr(self.config, "lang_padding_default", getattr(self.config, "surgical_padding_pixels", 2))),
            "inpainting_radius": int(getattr(self.config, "inpainting_radius", 3)),
        }
        if lang_group == "cjk":
            heur["padding"] = int(getattr(self.config, "lang_padding_cjk", heur["padding"]))
            heur["inpainting_radius"] = int(getattr(self.config, "inpainting_radius_cjk", heur["inpainting_radius"]))
        return heur

    # ---------------- inpainting ----------------
    def apply_inpainting(self, image: np.ndarray, mask: np.ndarray, method: str = "telea", radius: Optional[int] = None) -> np.ndarray:
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

        # choose method and radius (allow override)
        if radius is None:
            radius = max(1, int(getattr(self.config, "inpainting_radius", 3)))

        if method.lower() == "telea":
            flags = cv2.INPAINT_TELEA
            radius = max(1, int(radius))
            inpainted = cv2.inpaint(img, mask8, radius, flags)
        elif method.lower() in ("ns", "navier-stokes", "navier_stokes"):
            flags = cv2.INPAINT_NS
            radius = max(1, int(self.config.inpainting_radius))
            inpainted = cv2.inpaint(img, mask8, radius, flags)
        elif method.lower() == "hybrid":
            # hybrid: TELEA followed by a low-strength NS pass
            r1 = max(1, int(radius))
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
        # allow specific exceptions to propagate; callers may choose to fall
        # back if desired. Log detailed context on failure to aid debugging.
        mask = self.create_mask_from_regions(image.shape, detections)
        method = self.smart_inpainting_selection(image, mask)
        inpainted = self.apply_inpainting(image, mask, method=method)
        enhanced = self.enhance_inpainted_regions(image, inpainted, mask)
        report = self.validate_masking_quality(image, enhanced, mask)
        elapsed = time.time() - start
        try:
            self.audit_logger.info(json.dumps({"event": "process_image", "time_s": elapsed, "method": method}))
        except Exception:
            logger.debug("Failed to write audit event for process_image")
        return enhanced, report

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

    # ---------------- surgical masking ----------------
    def apply_surgical_masking(self, image: np.ndarray, phi_regions: List[Dict], padding: Optional[int] = None, lang_group: Optional[str] = None, merge_entity: bool = True) -> np.ndarray:
        """Apply precise, small black rectangles only over provided PHI regions.

        - padding: minimal pixel expansion around each bbox (defaults to config or 2px)
        - merge_entity: if True, merge all word-level boxes of a region into a single bbox
          before masking; otherwise mask each word box separately.
        Returns the masked image.
        """
        if image is None:
            raise ValueError("image must be provided")
        if lang_group is None:
            lang_group = self.detect_language_group(phi_regions)
        heur = self.get_language_heuristics(lang_group)
        pad = int(padding if padding is not None else heur.get("padding", getattr(self.config, "surgical_padding_pixels", 2)))
        out = image.copy()

        for r in phi_regions:
            word_boxes = r.get("word_boxes") or []
            if word_boxes:
                if merge_entity:
                    xs1 = [int(wb["bbox"][0]) for wb in word_boxes if wb.get("bbox")]
                    ys1 = [int(wb["bbox"][1]) for wb in word_boxes if wb.get("bbox")]
                    xs2 = [int(wb["bbox"][2]) for wb in word_boxes if wb.get("bbox")]
                    ys2 = [int(wb["bbox"][3]) for wb in word_boxes if wb.get("bbox")]
                    if xs1 and ys1 and xs2 and ys2:
                        x1, y1, x2, y2 = min(xs1), min(ys1), max(xs2), max(ys2)
                        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                        x2, y2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
                        # mask merged entity precisely
                        out[y1:y2, x1:x2] = 0
                        try:
                            self.audit_logger.info(json.dumps({"event": "surgical_mask_entity", "entity": r.get("entity_type"), "text": r.get("text"), "word_count": len(word_boxes)}))
                        except Exception:
                            pass
                    continue
                for wb in word_boxes:
                    bbox = wb.get("bbox")
                    if not bbox or len(bbox) != 4:
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                    x2, y2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
                    # mask individual word precisely
                    out[y1:y2, x1:x2] = 0
                try:
                    self.audit_logger.info(json.dumps({"event": "surgical_mask_words", "entity": r.get("entity_type"), "text": r.get("text"), "word_count": len(word_boxes)}))
                except Exception:
                    pass
                continue

            # fallback to masking value part for labels
            text = (r.get("text") or "").strip()
            if ":" in text:
                x1, y1, x2, y2 = map(int, r.get("bbox", [0,0,0,0]))
                split_x = int(x1 + (x2 - x1) * 0.4)
                sx1, sy1 = max(0, split_x - pad), max(0, y1 - pad)
                sx2, sy2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
                # mask value part precisely
                out[sy1:sy2, sx1:sx2] = 0
                try:
                    self.audit_logger.info(json.dumps({"event": "surgical_mask_value_only", "text": text, "bbox": [sx1, sy1, sx2, sy2]}))
                except Exception:
                    pass
                continue

            # final fallback whole region
            bbox = r.get("bbox")
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
                x2, y2 = min(out.shape[1], x2 + pad), min(out.shape[0], y2 + pad)
                # mask full region precisely
                out[y1:y2, x1:x2] = 0
                try:
                    self.audit_logger.info(json.dumps({"event": "surgical_mask", "entity": r.get("entity_type"), "text": r.get("text"), "bbox": [x1, y1, x2, y2]}))
                except Exception:
                    pass

        return out

    # ---------------- caching ----------------
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

    # add unified_masking_stage back so masking script works
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
        """Unified masking stage supporting surgical and inpaint styles."""
        # choose style
        style = style or getattr(self.config, "redaction_style", "inpaint")
        if style == "surgical":
            masked = self.apply_surgical_masking(image, phi_regions, padding=padding)
            meta = {"method": style}
            return masked, meta
        if style == "inpaint":
            mask = self.create_mask_from_regions(image.shape, phi_regions)
            method = inpaint_method or getattr(self.config, "inpainting_method", "telea")
            inpainted = self.apply_inpainting(image, mask, method)
            meta = {"method": style}
            return inpainted, meta
        # fallback: surgical blackbox
        masked = self.apply_surgical_masking(image, phi_regions, padding=padding)
        meta = {"method": "surgical"}
        return masked, meta


__all__ = ["ImageInpainter", "InpaintQualityReport"]
