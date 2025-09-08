"""Lightweight mocks for OCR and inpainting used in local/testing runs.

These provide predictable behavior without heavy dependencies so the
pipeline flow can be validated in CI or on developer machines.
"""
from typing import List, Dict, Any
import numpy as np
import logging

try:
    import cv2
except Exception:
    cv2 = None

logger = logging.getLogger(__name__)


class MockTextDetector:
    def __init__(self, config=None):
        self.config = config

    def detect_text_regions(self, image: np.ndarray) -> List[Dict[str, Any]]:
        h, w = image.shape[:2]
        # very small heuristic: split image in bands and pretend each band
        # contains a text row. This is only for pipeline validation.
        regions = []
        rows = min(5, max(1, h // 100))
        for i in range(rows):
            y1 = int(i * h / rows + 10)
            y2 = int((i + 1) * h / rows - 10)
            x1 = int(w * 0.05)
            x2 = int(w * 0.95)
            bbox = (x1, y1, x2, y2)
            # fabricate a plausible text snippet for tests
            text = f"SYNTH_TEXT_ROW_{i+1}"
            regions.append({"bbox": bbox, "text": text, "confidence": 0.95})
        return regions


class MockImageInpainter:
    def __init__(self, config=None):
        self.config = config

    def adaptive_mask_expansion(self, regions: List[Dict[str, Any]], image: np.ndarray) -> List[Dict[str, Any]]:
        # ensure regions have a mask_expansion field
        for r in regions:
            r.setdefault("mask_expansion", getattr(self.config, "mask_expansion_pixels", 5) if self.config is not None else 5)
        return regions

    def create_mask_from_regions(self, image_shape, phi_regions: List[Dict[str, Any]]):
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for r in phi_regions:
            bbox = r.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = map(int, bbox)
            exp = int(r.get("mask_expansion", 5))
            x1 = max(0, x1 - exp)
            y1 = max(0, y1 - exp)
            x2 = min(w, x2 + exp)
            y2 = min(h, y2 + exp)
            mask[y1:y2, x1:x2] = 255
        # simple blur to soften edges if cv2 available
        if cv2 is not None:
            k = 5
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        return mask

    def smart_inpainting_selection(self, image: np.ndarray, mask: np.ndarray) -> str:
        return getattr(self.config, "inpainting_method", "telea") if self.config is not None else "telea"

    def apply_inpainting(self, image: np.ndarray, mask: np.ndarray, method: str = "telea") -> np.ndarray:
        # Simple fallback: blur masked regions to simulate inpainting
        out = image.copy()
        try:
            if cv2 is not None:
                # create a blurred version and copy into masked areas
                blurred = cv2.GaussianBlur(out, (21, 21), 0)
                mask3 = mask if mask.ndim == 2 else mask[:, :, 0]
                mask_bool = mask3 > 0
                out[mask_bool] = blurred[mask_bool]
            else:
                # if cv2 not present, fill masked regions with mean color
                mean = out.mean(axis=(0, 1), dtype=int)
                mask3 = mask if mask.ndim == 2 else mask[:, :, 0]
                out[mask3 > 0] = mean
        except Exception:
            logger.exception("Mock inpainting failed; returning original image")
        return out

    def enhance_inpainted_regions(self, original: np.ndarray, inpainted: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return inpainted

    def validate_masking_quality(self, original: np.ndarray, masked: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        # crude metrics for pipeline tests
        return {"psnr": 99.0, "ssim": 1.0, "edge_preservation": 1.0, "texture_consistency": 1.0, "notes": []}
