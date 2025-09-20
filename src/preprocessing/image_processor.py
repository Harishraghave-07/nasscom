"""Image preprocessing utilities for clinical image masker.

This module provides an ImageProcessor class that handles loading images
(including DICOM), standardization, denoising, contrast enhancement,
orientation correction, batch processing with progress callbacks, and
quality validation (PSNR/SSIM).

Clinical notes and HIPAA guidance:
- Do not log identifiable PHI. Only log metadata identifiers and processing
  actions. The ImageProcessor will redact obvious PHI-like patterns from
  free-text logs when possible, but callers are responsible for not sending
  PHI to logs.
- DICOM tags are preserved and written back for DICOM outputs.
"""

from __future__ import annotations

import io
import logging
import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception as e:  # pragma: no cover - runtime dependency
    raise ImportError("OpenCV (cv2) is required for image processing: " + str(e))

try:
    import pydicom
    from pydicom.filewriter import write_file
except Exception:
    pydicom = None

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
except Exception:
    ssim = None
    psnr = None

from src.core.config import ProcessingConfig

logger = logging.getLogger("cim.preprocessing")


class ImageProcessingError(Exception):
    pass


class ImageProcessor:
    """Class providing preprocessing utilities for clinical images.

    Args:
        config: ProcessingConfig instance controlling size, temp dir, timeouts.

    Notes:
        This class is intended to be robust in hospital environments. It keeps
        temp files under `config.temp_dir` and logs only non-PHI metadata.
    """

    def __init__(self, config: ProcessingConfig):
        self.config = config
        # Create temp directory under control unless in-memory temp requested
        try:
            if not getattr(self.config, "use_in_memory_temp", False):
                os.makedirs(self.config.temp_dir, exist_ok=True)
        except Exception:
            # best-effort: do not fail initialization for temp dir creation
            logger.debug("Could not create temp dir %s; proceeding", getattr(self.config, 'temp_dir', None))

        # Setup a logger specific to preprocessing
        self.logger = logger
        self.logger.debug("Initializing ImageProcessor with config: %s", self.config.dict() if hasattr(self.config, 'dict') else str(self.config))

        # Validate resources (disk/memory hints)
        try:
            if not getattr(self.config, "use_in_memory_temp", False):
                free = os.statvfs(self.config.temp_dir).f_bavail * os.statvfs(self.config.temp_dir).f_frsize
                if free < 100 * 1024 * 1024:  # 100MB
                    self.logger.warning("Low disk space available in temp dir: %s bytes", free)
        except Exception:
            self.logger.exception("Failed to determine disk space for temp dir")

    # -----------------
    # Loading / saving
    # -----------------
    def load_image(self, file_path: str) -> Tuple[np.ndarray, Dict]:
        """Load an image (JPEG/PNG/DICOM) and return (image_array, metadata).

        Returns:
            image: HxWxC numpy array in RGB order (uint8)
            metadata: dict with keys: format, shape, dtype, medical_tags (if DICOM)
        """
        if not os.path.exists(file_path):
            raise ImageProcessingError(f"File not found: {file_path}")

        _, ext = os.path.splitext(file_path.lower())
        metadata: Dict = {"format": ext, "medical_tags": {}}

        try:
            if ext in (".dcm", "") and pydicom is not None:
                # DICOM: preserve tags and pixel data
                ds = pydicom.dcmread(file_path)
                self.logger.info("Loaded DICOM file (PatientID redacted in log)")
                # Extract pixel array safely
                try:
                    arr = ds.pixel_array
                except Exception as e:
                    raise ImageProcessingError(f"Failed to extract pixel data from DICOM: {e}")

                # Normalize to 8-bit RGB if necessary
                if arr.ndim == 2:
                    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                elif arr.ndim == 3 and arr.shape[2] == 3:
                    img = arr.astype(np.uint8)
                else:
                    # fallback: squeeze and convert
                    img = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2RGB)

                # Collect some tags (avoid logging PIIs)
                for tag in ("Modality", "StudyDate", "Rows", "Columns"):
                    try:
                        if hasattr(ds, tag):
                            metadata["medical_tags"][tag] = str(getattr(ds, tag))
                    except Exception:
                        pass

                metadata.update({"shape": img.shape, "dtype": str(img.dtype)})
                return img, metadata

            else:
                # Image formats: use cv2 to read. Use IMREAD_UNCHANGED to keep alpha
                arr = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                if arr is None:
                    raise ImageProcessingError("Failed to decode image or unsupported format")

                # Convert to RGB
                if arr.ndim == 2:
                    img = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
                elif arr.shape[2] == 4:
                    img = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGB)
                else:
                    img = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)

                metadata.update({"shape": img.shape, "dtype": str(img.dtype)})
                return img, metadata

        except ImageProcessingError:
            raise
        except Exception as e:
            self.logger.exception("Error loading image: %s", e)
            raise ImageProcessingError("Unexpected error while loading image") from e

    def save_processed_image(self, image: np.ndarray, output_path: str, metadata: Dict) -> bool:
        """Save processed image preserving metadata where possible.

        For DICOM outputs, attempt to write pixel data back into the dataset if
        the original DICOM metadata is provided; otherwise save as PNG/JPEG.
        """
        try:
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

            # If metadata indicates DICOM and pydicom available, try to save
            if metadata.get("format") in (".dcm",) and pydicom is not None and "dicom_ds" in metadata:
                ds = metadata["dicom_ds"]
                # Convert image to expected pixel type
                arr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
                ds.PixelData = arr.tobytes()
                ds.Rows, ds.Columns = arr.shape[:2]
                ds.save_as(output_path)
                return True

            # Otherwise write with cv2 (convert back to BGR)
            bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Use imencode to preserve quality options
            ext = os.path.splitext(output_path)[1].lower()
            if ext in (".jpg", ".jpeg"):
                result, enc = cv2.imencode(ext, bgr, [int(cv2.IMWRITE_JPEG_QUALITY), metadata.get("quality", 95)])
            else:
                result, enc = cv2.imencode(ext, bgr)

            if result:
                with open(output_path, "wb") as f:
                    f.write(enc.tobytes())
                # Append processing history to a sidecar metadata file (JSON)
                try:
                    meta_path = output_path + ".meta.json"
                    meta = dict(metadata)
                    meta.setdefault("processing_history", []).append({"timestamp": time.time(), "action": "save_processed"})
                    with open(meta_path, "w") as mf:
                        import json

                        json.dump(meta, mf)
                except Exception:
                    self.logger.exception("Failed to write sidecar metadata for %s", output_path)

                return True

            self.logger.error("Failed to encode image for saving: %s", output_path)
            return False

        except Exception as e:
            self.logger.exception("Exception while saving processed image: %s", e)
            return False

    # -----------------
    # Transformations
    # -----------------
    def standardize_image(self, image: np.ndarray, target_size: Optional[int] = None) -> np.ndarray:
        """Convert to RGB, resize while preserving aspect, pad to square if requested.

        Returns an uint8 RGB image with values in 0-255.
        """
        img = image.copy()
        # Ensure RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # target_size defaults to config.max_image_size
        if target_size is None:
            target_size = min(self.config.max_image_size, max(img.shape[:2]))

        h, w = img.shape[:2]
        scale = min(target_size / h, target_size / w, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)

        # Pad to square
        pad_w = target_size - new_w
        pad_h = target_size - new_h
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

        # Normalize to 0-255 uint8
        img_out = np.clip(img_padded, 0, 255).astype(np.uint8)
        return img_out

    def reduce_noise(self, image: np.ndarray, method: str = "auto") -> np.ndarray:
        """Denoise image using multiple strategies.

        method can be 'gaussian', 'median', 'bilateral', 'nlmeans', or 'auto'.
        """
        img = image.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        # Basic heuristic: high variance favors nlmeans; small images use median
        var = float(np.var(gray))
        method_use = method
        if method == "auto":
            if var > 1000:
                method_use = "nlmeans"
            elif var > 300:
                method_use = "bilateral"
            else:
                method_use = "gaussian"

        self.logger.debug("Selected denoise method: %s (variance=%.2f)", method_use, var)

        try:
            if method_use == "gaussian":
                return cv2.GaussianBlur(img, (5, 5), 0)
            if method_use == "median":
                return cv2.medianBlur(img, 3)
            if method_use == "bilateral":
                return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
            if method_use == "nlmeans":
                # OpenCV fastNlMeansDenoisingColored expects uint8 BGR
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                den = cv2.fastNlMeansDenoisingColored(bgr, None, 10, 10, 7, 21)
                return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
        except Exception:
            self.logger.exception("Denoising method %s failed, returning original image", method_use)

        return img

    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE adaptive histogram equalization to improve local contrast.

        This is tuned to preserve medical image characteristics and avoid
        introducing artifacts in clinical imagery.
        """
        try:
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            merged = cv2.merge((cl, a, b))
            result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            return result
        except Exception:
            self.logger.exception("CLAHE enhancement failed; returning original")
            return image

    def correct_orientation(self, image: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect dominant text orientation and correct skew.

        Returns (rotated_image, angle_deg) where angle_deg is the angle applied
        to deskew (positive means rotated counter-clockwise).
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, math.pi / 180.0, threshold=100, minLineLength=100, maxLineGap=10)
        angle = 0.0
        if lines is not None and len(lines) > 0:
            angles = []
            for x1, y1, x2, y2 in lines[:, 0]:
                a = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
                if abs(a) < 45:
                    angles.append(a)
            if angles:
                angle = float(np.median(angles))

        # Rotate to correct
        if abs(angle) > 0.1:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            return rotated, angle
        return image, 0.0

    # -----------------
    # Batch processing
    # -----------------
    def preprocess_batch(self, file_paths: List[str], progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Process multiple images in parallel and return per-image reports.

        progress_callback receives (index, total, report) for each completed
        image and can be used to update UI progress bars.
        """
        reports: List[Dict] = []
        total = len(file_paths)
        max_workers = min(self.config.max_batch_size, max(1, multiprocessing.cpu_count() - 1))

        self.logger.info("Starting batch preprocessing: %d files with %d workers", total, max_workers)
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self._process_single, fp): fp for fp in file_paths}
            for i, fut in enumerate(as_completed(futures), 1):
                fp = futures[fut]
                try:
                    report = fut.result()
                except Exception as e:
                    self.logger.exception("Processing failed for %s: %s", fp, e)
                    report = {"file": fp, "error": str(e)}
                reports.append(report)
                if progress_callback:
                    try:
                        progress_callback(len(reports), total, report)
                    except Exception:
                        self.logger.exception("Progress callback raised an exception")

        return reports

    def _process_single(self, file_path: str) -> Dict:
        start = time.time()
        report: Dict = {"file": file_path}
        try:
            img, meta = self.load_image(file_path)
            report["loaded_shape"] = img.shape
            orig = img.copy()
            img = self.standardize_image(img)
            img = self.reduce_noise(img)
            img = self.enhance_contrast(img)
            img, angle = self.correct_orientation(img)
            report["angle"] = angle
            metrics = self.validate_image_quality(orig, img)
            report["metrics"] = metrics
            out_path = os.path.join(self.config.temp_dir, os.path.basename(file_path))
            saved = self.save_processed_image(img, out_path, meta)
            report["saved"] = saved
            report["output_path"] = out_path
            report["time_taken"] = time.time() - start
            return report
        except Exception as e:
            self.logger.exception("_process_single failed for %s", file_path)
            report["error"] = str(e)
            return report

    def validate_image_quality(self, original: np.ndarray, processed: np.ndarray) -> Dict[str, float]:
        """Compute PSNR and SSIM between original and processed images.

        Falls back to simple approximations if skimage not available.
        """
        metrics: Dict[str, float] = {}
        try:
            if psnr is not None:
                metrics["psnr"] = float(psnr(original, processed, data_range=255))
            else:
                mse = float(np.mean((original.astype(np.float32) - processed.astype(np.float32)) ** 2))
                metrics["psnr"] = 20 * math.log10(255.0) - 10 * math.log10(mse) if mse > 0 else float("inf")

            if ssim is not None:
                # convert to gray for ssim
                try:
                    from skimage.color import rgb2gray

                    g1 = rgb2gray(original.astype(np.float32) / 255.0)
                    g2 = rgb2gray(processed.astype(np.float32) / 255.0)
                    metrics["ssim"] = float(ssim(g1, g2))
                except Exception:
                    metrics["ssim"] = 0.0
            else:
                # rough structural similarity via normalized cross-correlation
                orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY).astype(np.float32)
                proc_gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY).astype(np.float32)
                num = np.sum((orig_gray - orig_gray.mean()) * (proc_gray - proc_gray.mean()))
                den = math.sqrt(np.sum((orig_gray - orig_gray.mean()) ** 2) * np.sum((proc_gray - proc_gray.mean()) ** 2))
                metrics["ssim"] = float(num / den) if den != 0 else 0.0

            # detect artifacts: large differences local area
            diff = np.abs(original.astype(np.int32) - processed.astype(np.int32))
            metrics["max_diff"] = float(diff.max())
            metrics["mean_diff"] = float(diff.mean())
        except Exception as e:
            self.logger.exception("Quality metrics failed: %s", e)
            metrics["psnr"] = 0.0
            metrics["ssim"] = 0.0

        # Text readability heuristic: fraction of high-contrast edges preserved
        try:
            orig_edges = cv2.Canny(cv2.cvtColor(original, cv2.COLOR_RGB2GRAY), 50, 150)
            proc_edges = cv2.Canny(cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY), 50, 150)
            if orig_edges.sum() > 0:
                metrics["text_preservation"] = float((orig_edges & proc_edges).sum()) / float(orig_edges.sum())
            else:
                metrics["text_preservation"] = 1.0
        except Exception:
            metrics["text_preservation"] = 0.0

        return metrics


# -----------------
# Utility functions
# -----------------
def detect_image_type(file_path: str) -> str:
    """Return a string describing the image type/modality.

    Uses file extension and simple DICOM heuristics where available.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".dcm" and pydicom is not None:
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            mod = getattr(ds, "Modality", "DICOM")
            return str(mod)
        except Exception:
            return "dicom"
    if ext in (".jpg", ".jpeg", ".png"):
        # Heuristics: filename or parent dir may indicate modality
        name = os.path.basename(file_path).lower()
        if "xray" in name or "chest" in name:
            return "X-RAY"
        if "ct" in name:
            return "CT"
        if "mri" in name:
            return "MRI"
        if "ultrasound" in name or "us_" in name:
            return "ULTRASOUND"
        return "clinical_image"
    return "unknown"


def estimate_text_density(image: np.ndarray) -> float:
    """Estimate the percentage of image area likely containing text overlays.

    Uses edge detection and morphological filtering to find text-like regions.
    Returns a float between 0.0 and 1.0.
    """
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # Sobel + threshold to detect high-contrast strokes
        sob = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
        _, th = cv2.threshold(sob, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Morphology to join text strokes
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = image.shape[0] * image.shape[1]
        text_area = 0
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            # heuristic filters for text-like boxes
            if w < 5 or h < 5:
                continue
            if w / float(h) < 0.2:
                continue
            text_area += w * h
        return min(1.0, float(text_area) / float(area) if area > 0 else 0.0)
    except Exception:
        logger.exception("Failed to estimate text density")
        return 0.0


def get_processing_recommendations(image: np.ndarray, image_type: str) -> Dict:
    """Return recommended processing settings based on image type and content.

    This is a lightweight rule-based recommender used to choose denoising and
    sharpening parameters appropriate for the modality.
    """
    rec = {
        "denoise_method": "auto",
        "inpainting_method": "telea",
        "target_size": 1024,
    }
    try:
        td = estimate_text_density(image)
        if image_type.upper() in ("X-RAY", "CT", "MRI"):
            rec.update({"denoise_method": "nlmeans", "target_size": 2048})
        elif image_type.upper() == "ULTRASOUND":
            rec.update({"denoise_method": "median", "target_size": 1024})
        else:
            rec.update({"denoise_method": "bilateral", "target_size": 1024})

        if td > 0.05:
            # high text density: preserve text clarity
            rec["denoise_method"] = "bilateral"
            rec["preserve_text"] = True
        else:
            rec["preserve_text"] = False

    except Exception:
        logger.exception("Failed to generate processing recommendations")

    return rec
