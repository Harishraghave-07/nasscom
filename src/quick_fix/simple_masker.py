"""A minimal, bulletproof masker for quick fixes.

Designed to use only OpenCV and EasyOCR (best-effort) and provide
guaranteed rectangular black-box masking for all detected text.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Any

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - environment dependent
    cv2 = None

# Note: EasyOCR is a heavy dependency; import lazily inside functions that
# need it to avoid increasing module import time or requiring torch at
# module import time in environments (tests, lightweight containers).


class SimpleMasker:
    """Extremely conservative masker that draws black rectangles over all
    OCR-detected text. Focus is on reliability and clear visual feedback.
    """

    def __init__(self) -> None:
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("SimpleMasker")
        self.logger.setLevel(logging.DEBUG)

        # conservative defaults
        self.expand_pixels = 5
        self.ocr_gpu = False
        self.temp_dir = Path("debug_output")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def validate_environment(self) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        # OpenCV
        try:
            if cv2 is None:
                raise ImportError("cv2 not available")
            results["opencv"] = {"installed": True, "version": cv2.__version__}
        except Exception as e:
            results["opencv"] = {"installed": False, "error": str(e)}

        # EasyOCR (lazy import)
        try:
            import easyocr as _easyocr  # type: ignore
            try:
                _reader = _easyocr.Reader(["en"], gpu=self.ocr_gpu)
                results["easyocr"] = {"installed": True, "initialized": True}
            except Exception as ie:
                results["easyocr"] = {"installed": True, "initialized": False, "error": str(ie)}
        except Exception as e:
            results["easyocr"] = {"installed": False, "error": str(e)}

        # numpy
        try:
            results["numpy"] = {"installed": True, "version": np.__version__}
        except Exception as e:
            results["numpy"] = {"installed": False, "error": str(e)}

        return results

    def mask_image_simple(self, image_path: str, output_path: str) -> Dict[str, Any]:
        """Load image, run OCR with permissive settings, and draw black
        rectangles over each detected text region. Returns details dict.
        """
        try:
            self.logger.debug("Loading image: %s", image_path)
            if cv2 is None:
                return {"error": "OpenCV (cv2) is required", "success": False}

            image = cv2.imread(str(image_path))
            if image is None:
                self.logger.error("Failed to load image: %s", image_path)
                return {"error": "Failed to load image", "success": False}

            self.logger.debug("Image loaded shape=%s", getattr(image, "shape", None))

            # Attempt to import EasyOCR lazily and initialize reader (force CPU)
            try:
                import easyocr  # type: ignore
            except Exception:
                self.logger.error("EasyOCR not installed; cannot detect text")
                return {"error": "EasyOCR not installed", "success": False}

            try:
                reader = easyocr.Reader(["en"], gpu=self.ocr_gpu)
            except Exception as e:
                self.logger.exception("Failed to initialize EasyOCR: %s", e)
                return {"error": f"EasyOCR init failed: {e}", "success": False}

            # run detection with permissive thresholds
            self.logger.debug("Running EasyOCR readtext (low thresholds)")
            try:
                # detail=1 returns (bbox, text, confidence)
                results = reader.readtext(image, detail=1, width_ths=0.1, height_ths=0.1)
            except Exception as e:
                self.logger.exception("EasyOCR readtext failed: %s", e)
                return {"error": f"EasyOCR read failed: {e}", "success": False}

            masked = image.copy()
            mask_count = 0

            for item in results:
                try:
                    bbox, text, confidence = item
                    self.logger.debug("Detected text: %s (conf=%.3f)", text, float(confidence))
                    pts = np.array(bbox, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)

                    # expand and clip
                    ex = int(self.expand_pixels)
                    x0 = max(0, x - ex)
                    y0 = max(0, y - ex)
                    x1 = min(masked.shape[1], x + w + ex)
                    y1 = min(masked.shape[0], y + h + ex)

                    cv2.rectangle(masked, (x0, y0), (x1, y1), (0, 0, 0), thickness=-1)
                    mask_count += 1
                    self.logger.debug("Applied mask #%d at (%d,%d)->(%d,%d)", mask_count, x0, y0, x1, y1)
                except Exception:
                    self.logger.exception("Failed to apply mask for detection: %s", item)
                    continue

            # ensure output dir exists
            outp = Path(output_path)
            outp.parent.mkdir(parents=True, exist_ok=True)

            saved = cv2.imwrite(str(outp), masked)
            if not saved:
                self.logger.error("Failed to save masked image to %s", output_path)
                return {"error": "Failed to save output image", "success": False}

            return {
                "success": True,
                "masks_applied": mask_count,
                "original_shape": image.shape,
                "text_regions_found": len(results),
                "output_path": str(outp),
            }

        except Exception as e:
            self.logger.exception("Unexpected error in mask_image_simple: %s", e)
            return {"error": str(e), "success": False}

    def test_basic_functionality(self) -> Dict[str, Any]:
        # create minimal test input
        if cv2 is None:
            return {"success": False, "error": "cv2 not available"}

        test_dir = self.temp_dir
        test_dir.mkdir(parents=True, exist_ok=True)
        test_path = test_dir / "test_input.jpg"
        out_path = test_dir / "test_output.jpg"

        test_image = np.ones((300, 400, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(test_image, "John Doe", (50, 100), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "MRN-123456", (50, 150), font, 1, (0, 0, 0), 2)
        cv2.putText(test_image, "555-123-4567", (50, 200), font, 1, (0, 0, 0), 2)

        cv2.imwrite(str(test_path), test_image)

        return self.mask_image_simple(str(test_path), str(out_path))


def main() -> None:
    print("=" * 60)
    print("SIMPLE MASKER - QUICK TEST")
    print("=" * 60)

    masker = SimpleMasker()
    print("Validating environment...")
    env = masker.validate_environment()
    for k, v in env.items():
        print(k, ":", v)

    print("Running basic functionality test...")
    res = masker.test_basic_functionality()
    print("Test result:", res)

    # Optional: accept CLI args for a real image
    import sys
    if len(sys.argv) > 1:
        inp = sys.argv[1]
        out = sys.argv[2] if len(sys.argv) > 2 else str(Path("debug_output/real_output.jpg"))
        print("Masking real image...", inp, "->", out)
        print(masker.mask_image_simple(inp, out))


if __name__ == "__main__":
    main()
