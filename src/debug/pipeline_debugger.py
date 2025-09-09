"""Pipeline debugger utilities for the Clinical Image Masking pipeline.

Provides PipelineDebugger which runs the full pipeline with detailed
debugging, visualizations, per-stage outputs, and simple automated tests.

This module uses best-effort imports for heavy libraries (cv2, matplotlib,
EasyOCR, spaCy) and falls back to safe operations when they're not
available so the debugger can run in CI or dev machines without GPU.
"""
from __future__ import annotations

import json
import logging
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
except Exception:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from src.core.config import AppConfig
from src.core.pipeline import ClinicalImageMaskingPipeline

LOGGER = logging.getLogger("PipelineDebugger")


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class PipelineDebugger:
    """Debugging harness for the masking pipeline.

    The class produces per-stage images and a small HTML report while
    collecting timing and simple quality metrics. Designed to be used by
    developers investigating failures in OCR -> PHI -> masking flows.
    """

    def __init__(self, pipeline_config: AppConfig):
        # setup verbose logging
        logging.basicConfig(level=logging.DEBUG)
        LOGGER.setLevel(logging.DEBUG)

        self.config = pipeline_config
        # directories
        self.debug_root = Path("debug_output")
        _safe_mkdir(self.debug_root)
        self.stage_dirs = {f"stage{i}": self.debug_root / f"stage{i}" for i in range(1, 5)}
        for d in self.stage_dirs.values():
            _safe_mkdir(d)

        # visualization control
        self.save_images = True

        # performance counters
        self.timings: Dict[str, float] = {}

        # error tracking
        self.errors: List[str] = []

        # instantiate pipeline (will lazy-load heavy components)
        self.pipeline = ClinicalImageMaskingPipeline(self.config)

    # ---------------- utilities ----------------
    def _save_array_image(self, arr: np.ndarray, path: Path) -> None:
        try:
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(str(path))
        except Exception:
            LOGGER.exception("Failed to save image %s", path)

    def _annotate_and_save(self, image: np.ndarray, boxes: List[Dict], path: Path, title: Optional[str] = None) -> None:
        try:
            if image.dtype != np.uint8:
                img = np.clip(image, 0, 255).astype(np.uint8)
            else:
                img = image
            pil = Image.fromarray(img.copy())
            draw = ImageDraw.Draw(pil)
            font = ImageFont.load_default()
            for i, b in enumerate(boxes):
                try:
                    x1, y1, x2, y2 = map(int, b.get("bbox", (0, 0, 0, 0)))
                    conf = b.get("confidence", None)
                    color = (255, 0, 0)
                    if conf is not None:
                        if conf >= 0.75:
                            color = (0, 255, 0)
                        elif conf >= 0.4:
                            color = (255, 200, 0)
                        else:
                            color = (255, 0, 0)
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                    txt = f"{b.get('text','')[:30]}\n{conf}" if conf is not None else b.get("text", "")
                    draw.text((x1 + 4, y1 + 4), txt, fill=color, font=font)
                except Exception:
                    continue
            if title:
                draw.text((4, 4), title, fill=(255, 255, 255), font=font)
            pil.save(str(path))
        except Exception:
            LOGGER.exception("Failed to annotate/save %s", path)

    def _write_json(self, obj: Any, path: Path) -> None:
        try:
            path.write_text(json.dumps(obj, default=str, indent=2))
        except Exception:
            LOGGER.exception("Failed to write json %s", path)

    def _time(self, key: str, start: float) -> None:
        self.timings[key] = time.perf_counter() - start

    # ---------------- top-level debug runner ----------------
    def debug_full_pipeline(self, input_image_path: str) -> Dict[str, Any]:
        report: Dict[str, Any] = {"stages": {}}
        try:
            t0 = time.perf_counter()
            # Stage 1: preprocessing
            pre = self.debug_preprocessing(input_image_path)
            report["stages"]["preprocessing"] = pre

            proc_img = pre.get("standardized_image_arr")
            if proc_img is None:
                LOGGER.error("Preprocessing failed to produce an image")
                self.errors.append("preprocessing_failed")
                return {"error": "preprocessing_failed", "report": report}

            # Stage 2: text detection
            td = self.debug_text_detection(proc_img)
            report["stages"]["text_detection"] = td

            # Stage 3: PHI detection
            phi = self.debug_phi_detection(td.get("accepted_regions", []) )
            report["stages"]["phi_detection"] = phi

            # Stage 4: masking
            maskr = self.debug_masking(proc_img, phi.get("phi_regions", []))
            report["stages"]["masking"] = maskr

            # final report
            report["summary"] = {"total_time_s": time.perf_counter() - t0, "timings": self.timings}
            # create HTML report
            html = self.create_debug_report(report)
            report["html_report"] = html
            # sanitize report to be JSON serializable (convert numpy arrays, Paths, etc.)
            return self._sanitize_for_json(report)
        except Exception as e:
            LOGGER.exception("Full pipeline debug failed")
            self.errors.append(str(e))
            return {"error": str(e), "stages": report}

    # ---------------- preprocessing debug ----------------
    def debug_preprocessing(self, image_path: str) -> Dict[str, Any]:
        start = time.perf_counter()
        outdir = self.stage_dirs["stage1"]
        try:
            img = Image.open(image_path).convert("RGB")
            arr = np.array(img)
            self._save_array_image(arr, outdir / "original.jpg")
            # simple denoise: median blur (via cv2) or gaussian via numpy
            if cv2 is not None:
                den = cv2.medianBlur(arr.copy(), 3)
            else:
                den = arr
            self._save_array_image(den, outdir / "denoised.jpg")

            # contrast enhancement - simple CLAHE (cv2) or stretch
            if cv2 is not None:
                lab = cv2.cvtColor(den, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                cl = clahe.apply(l)
                limg = cv2.merge((cl, a, b))
                enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            else:
                # linear stretch
                p2, p98 = np.percentile(den, (2, 98))
                enhanced = np.clip((den - p2) * 255.0 / (p98 - p2 + 1e-6), 0, 255).astype(np.uint8)
            self._save_array_image(enhanced, outdir / "enhanced.jpg")

            # standardization: resize / pad to pipeline.processing.max_image_size
            target = min(self.config.processing.max_image_size, max(arr.shape[:2]))
            # scale so max dimension == target
            h, w = enhanced.shape[:2]
            scale = target / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            if cv2 is not None:
                standardized = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                standardized = np.array(Image.fromarray(enhanced).resize((new_w, new_h)))
            self._save_array_image(standardized, outdir / "standardized.jpg")

            # stats and histograms
            def stats(a: np.ndarray) -> Dict[str, float]:
                return {"mean": float(a.mean()), "std": float(a.std()), "min": int(a.min()), "max": int(a.max())}

            analysis = {
                "original_stats": stats(arr),
                "denoised_stats": stats(den),
                "enhanced_stats": stats(enhanced),
                "standardized_stats": stats(standardized),
            }
            self._write_json(analysis, outdir / "preprocess_stats.json")
            self._time("preprocessing", start)
            # return standardized image as array for next stage
            return {"standardized_image_arr": standardized, "analysis": analysis}
        except Exception as e:
            LOGGER.exception("Preprocessing debug failed")
            self.errors.append(str(e))
            return {"error": str(e)}

    # ---------------- text detection debug ----------------
    def debug_text_detection(self, processed_image: np.ndarray) -> Dict[str, Any]:
        start = time.perf_counter()
        outdir = self.stage_dirs["stage2"]
        try:
            # try to use pipeline's OCR if available, otherwise note fallback
            regions = []
            ocr_reader = getattr(self.pipeline, "ocr", None)
            if ocr_reader is None:
                # use mock if present
                try:
                    from src.core.mocks import MockTextDetector

                    detector = MockTextDetector(self.config.ocr)
                    regions = detector.detect_text_regions(processed_image)
                except Exception:
                    LOGGER.warning("No OCR available; returning empty regions")
                    regions = []
            else:
                regions = ocr_reader.detect_text_regions(processed_image)

            # save raw results
            self._write_json(regions, outdir / "ocr_raw_results.json")

            # create visualization of all boxes
            self._annotate_and_save(processed_image, regions, outdir / "all_detections.jpg", title="All detections")

            # create confidence heatmap: draw confidence as alpha overlay
            heat = np.zeros(processed_image.shape[:2], dtype=np.float32)
            for r in regions:
                try:
                    x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                    conf = float(r.get("confidence", 1.0))
                    heat[y1:y2, x1:x2] = np.maximum(heat[y1:y2, x1:x2], conf)
                except Exception:
                    continue
            if plt is not None:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(processed_image)
                ax.imshow(heat, cmap="jet", alpha=0.5)
                ax.axis("off")
                fig.savefig(outdir / "detection_heatmap.png", bbox_inches="tight")
                plt.close(fig)

            # test threshold buckets
            thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
            thresholded = {}
            for t in thresholds:
                accepted = [r for r in regions if float(r.get("confidence", 1.0)) >= t]
                thresholded[t] = {"accepted_count": len(accepted), "accepted": accepted}
                # save visualization per threshold
                self._annotate_and_save(processed_image, accepted, outdir / f"accepted_thr_{int(t*100)}.jpg", title=f"thr={t}")

            self._time("text_detection", start)
            return {"accepted_regions": thresholded[0.5]["accepted"], "raw_regions": regions, "threshold_analysis": thresholded}
        except Exception as e:
            LOGGER.exception("Text detection debug failed")
            self.errors.append(str(e))
            return {"error": str(e)}

    # ---------------- PHI detection debug ----------------
    def debug_phi_detection(self, text_regions: List[Dict]) -> Dict[str, Any]:
        start = time.perf_counter()
        outdir = self.stage_dirs["stage3"]
        try:
            # load phi classifier if available
            try:
                from src.phi_detection.phi_classifier import PHIClassifier

                phi = PHIClassifier(self.config.phi)
            except Exception:
                phi = None

            results = []
            for r in text_regions:
                text = r.get("text", "")
                row = {"text": text, "bbox": r.get("bbox")}
                # regex
                regex_hits = []
                try:
                    patterns = self.config.get_phi_patterns()
                    for name, pat in patterns.items():
                        try:
                            import re

                            if re.search(pat, text):
                                regex_hits.append(name)
                        except Exception:
                            continue
                except Exception:
                    patterns = {}
                row["regex_hits"] = regex_hits

                # spaCy NER
                ner_hits = []
                try:
                    if phi is not None and getattr(phi, "nlp", None) is not None:
                        ents = phi.detect_phi_in_text(text)
                        ner_hits = ents
                except Exception:
                    LOGGER.debug("spaCy detection not available or failed")
                row["ner_hits"] = ner_hits
                results.append(row)

            self._write_json(results, outdir / "phi_analysis.json")
            self._time("phi_detection", start)
            # flatten phi regions (simple mapping)
            phi_regions = []
            for r, row in zip(text_regions, results):
                if row.get("regex_hits") or row.get("ner_hits"):
                    entry = {"bbox": r.get("bbox"), "text": r.get("text"), "phi_types": row.get("regex_hits") + [e.get("label") for e in row.get("ner_hits", []) if isinstance(e, dict)]}
                    phi_regions.append(entry)

            return {"phi_regions": phi_regions, "raw": results}
        except Exception as e:
            LOGGER.exception("PHI debug failed")
            self.errors.append(str(e))
            return {"error": str(e)}

    # ---------------- masking debug ----------------
    def debug_masking(self, image: np.ndarray, phi_regions: List[Dict]) -> Dict[str, Any]:
        start = time.perf_counter()
        outdir = self.stage_dirs["stage4"]
        try:
            # save original boxes
            self._annotate_and_save(image, phi_regions, outdir / "original_boxes.jpg", title="Original boxes")

            # test mask expansion values
            expansions = [0, 3, 5, 10]
            expansion_results = {}
            for e in expansions:
                # create mask with expansion
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                for r in phi_regions:
                    try:
                        x1, y1, x2, y2 = map(int, r.get("bbox", (0, 0, 0, 0)))
                        x1 = max(0, x1 - e)
                        y1 = max(0, y1 - e)
                        x2 = min(image.shape[1], x2 + e)
                        y2 = min(image.shape[0], y2 + e)
                        mask[y1:y2, x1:x2] = 255
                    except Exception:
                        continue
                # save expanded mask visualization
                cmap = np.stack([mask, mask, mask], axis=2)
                self._save_array_image(cmap, outdir / f"expanded_mask_{e}.jpg")

                # attempt inpainting with TELEA/NS if cv2 available
                inpaint_results = {}
                if cv2 is not None:
                    try:
                        telea = cv2.inpaint(image.copy(), mask, 3, cv2.INPAINT_TELEA)
                        ns = cv2.inpaint(image.copy(), mask, 3, cv2.INPAINT_NS)
                        self._save_array_image(telea, outdir / f"telea_{e}.jpg")
                        self._save_array_image(ns, outdir / f"ns_{e}.jpg")
                        inpaint_results = {"telea": str(outdir / f"telea_{e}.jpg"), "ns": str(outdir / f"ns_{e}.jpg")}
                    except Exception:
                        LOGGER.exception("OpenCV inpainting failed for expansion %s", e)
                else:
                    # fallback: simple blur of masked areas
                    blurred = image.copy()
                    try:
                        if plt is not None:
                            # gaussian via scipy not guaranteed; use PIL blur via Image.filter if available
                            pil = Image.fromarray(image.copy())
                            pil.save(outdir / f"telea_{e}.jpg")
                            inpaint_results = {"telea": str(outdir / f"telea_{e}.jpg")}
                    except Exception:
                        LOGGER.exception("Fallback inpainting failed")

                expansion_results[e] = {"mask_path": str(outdir / f"expanded_mask_{e}.jpg"), "inpaint_results": inpaint_results}

            self._time("masking", start)
            # final mask (use expansion 5 as default)
            final_mask = expansion_results.get(5, {}).get("mask_path")
            return {"expansion_results": expansion_results, "final_mask_path": final_mask}
        except Exception as e:
            LOGGER.exception("Masking debug failed")
            self.errors.append(str(e))
            return {"error": str(e)}

    # ---------------- visual report builder ----------------
    def visualize_detection_pipeline(self, image: np.ndarray, detections: List[Dict], output_path: str) -> None:
        try:
            self._annotate_and_save(image, detections, Path(output_path), title="Detections Overview")
        except Exception:
            LOGGER.exception("visualize_detection_pipeline failed")

    def create_debug_report(self, debug_results: Dict) -> str:
        try:
            out = self.debug_root / "report.html"
            parts = ["<html><head><meta charset='utf-8'><title>Pipeline Debug Report</title></head><body>"]
            parts.append(f"<h1>Pipeline Debug Report</h1>")
            parts.append(f"<h2>Timings</h2><pre>{json.dumps(self.timings, indent=2)}</pre>")
            parts.append("<h2>Stage outputs</h2>")
            for i in range(1, 5):
                d = self.stage_dirs[f"stage{i}"]
                parts.append(f"<h3>Stage {i}</h3>")
                # list images
                if d.exists():
                    for p in sorted(d.glob("*.jpg")):
                        parts.append(f"<div><img src='{p.as_posix()}' style='max-width:800px'/></div>")
            parts.append("</body></html>")
            out.write_text("\n".join(parts))
            return str(out)
        except Exception:
            LOGGER.exception("Failed to create debug report")
            return ""

    # ---------------- diagnosis helpers ----------------
    def diagnose_no_text_detected(self, image: np.ndarray) -> Dict[str, Any]:
        issues = []
        try:
            # try extreme threshold
            checks = {}
            for thr in [0.01, 0.05, 0.1, 0.2]:
                try:
                    ocr = getattr(self.pipeline, "ocr", None)
                    if ocr is None:
                        from src.core.mocks import MockTextDetector
                        det = MockTextDetector(self.config.ocr).detect_text_regions(image)
                    else:
                        det = ocr.detect_text_regions(image)
                    checks[thr] = len([r for r in det if float(r.get("confidence", 1.0)) >= thr])
                except Exception:
                    checks[thr] = 0
            # analyze basic image stats
            stats = {"mean": float(image.mean()), "std": float(image.std()), "shape": image.shape}
            issues.append({"checks": checks, "stats": stats})
            return {"diagnosis": issues}
        except Exception as e:
            LOGGER.exception("diagnose_no_text_detected failed")
            return {"error": str(e)}

    def diagnose_text_detected_but_no_phi(self, text_regions: List[Dict]) -> Dict[str, Any]:
        try:
            out = {"detected_texts": [r.get("text") for r in text_regions], "pattern_matches": []}
            patterns = self.config.get_phi_patterns()
            import re
            for t in text_regions:
                hits = []
                for name, pat in patterns.items():
                    try:
                        if re.search(pat, t.get("text", "")):
                            hits.append(name)
                    except Exception:
                        continue
                out["pattern_matches"].append({"text": t.get("text"), "hits": hits})
            return out
        except Exception as e:
            LOGGER.exception("diagnose_text_detected_but_no_phi failed")
            return {"error": str(e)}

    def diagnose_phi_detected_but_no_masking(self, image: np.ndarray, phi_regions: List[Dict]) -> Dict[str, Any]:
        try:
            res = {"masks_valid": True, "details": []}
            for r in phi_regions:
                try:
                    bbox = r.get("bbox")
                    if not bbox or len(bbox) != 4:
                        res["masks_valid"] = False
                        res["details"].append({"region": r, "reason": "invalid_bbox"})
                        continue
                    x1, y1, x2, y2 = map(int, bbox)
                    if x2 <= x1 or y2 <= y1:
                        res["masks_valid"] = False
                        res["details"].append({"region": r, "reason": "zero_area"})
                except Exception:
                    res["masks_valid"] = False
                    res["details"].append({"region": r, "reason": "exception"})
            return res
        except Exception as e:
            LOGGER.exception("diagnose_phi_detected_but_no_masking failed")
            return {"error": str(e)}

    # ---------------- automated tests ----------------
    def test_with_synthetic_image(self) -> Dict[str, Any]:
        # generate synthetic with known PHI and run full debug
        tmp = tempfile.mkdtemp(prefix="dbg_synth_")
        path = Path(tmp) / "synth.png"
        try:
            w, h = 800, 600
            img = Image.new("RGB", (w, h), (255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((20, 20), "Johnathan A. DOE", fill=(0, 0, 0))
            d.text((20, 80), "MRN: 12345678", fill=(0, 0, 0))
            img.save(path)
            report = self.debug_full_pipeline(str(path))
            shutil.rmtree(tmp)
            return report
        except Exception as e:
            LOGGER.exception("test_with_synthetic_image failed")
            return {"error": str(e)}

    def test_with_minimal_example(self) -> Dict[str, Any]:
        tmp = tempfile.mkdtemp(prefix="dbg_min_")
        path = Path(tmp) / "mini.png"
        try:
            img = Image.new("RGB", (200, 200), (255, 255, 255))
            d = ImageDraw.Draw(img)
            d.text((10, 80), "John Doe MRN-123456", fill=(0, 0, 0))
            img.save(path)
            report = self.debug_full_pipeline(str(path))
            shutil.rmtree(tmp)
            return report
        except Exception as e:
            LOGGER.exception("test_with_minimal_example failed")
            return {"error": str(e)}

    def _sanitize_for_json(self, obj: Any) -> Any:
        """Recursively convert numpy types, arrays, Path objects, and other
        non-serializable values into JSON-serializable Python types.
        """
        try:
            import numpy as _np
        except Exception:
            _np = None

        def _fix(x):
            # numpy arrays
            if _np is not None and isinstance(x, _np.ndarray):
                try:
                    return x.tolist()
                except Exception:
                    return None
            # numpy scalar
            if _np is not None and isinstance(x, (_np.integer, _np.floating, _np.bool_)):
                return x.item()
            # Path objects
            if isinstance(x, Path):
                return str(x)
            # bytes
            if isinstance(x, (bytes, bytearray)):
                try:
                    return x.decode('utf-8', errors='replace')
                except Exception:
                    return str(x)
            # dict
            if isinstance(x, dict):
                return {str(k): _fix(v) for k, v in x.items()}
            # list/tuple/set
            if isinstance(x, (list, tuple, set)):
                return [_fix(v) for v in list(x)]
            # simple types
            if isinstance(x, (str, int, float, bool)) or x is None:
                return x
            # fallback: try to stringify
            try:
                return str(x)
            except Exception:
                return None

        return _fix(obj)


__all__ = ["PipelineDebugger"]
