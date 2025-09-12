"""Detect highlighted regions (yellow highlighter) on PDF pages and mask them.

Saves a side-by-side comparison image under tmp/ similar to test_pdf_masking.py.
"""
import sys
from pathlib import Path
import numpy as np

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path("tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH:
    print("No PDF path provided")
    sys.exit(2)

try:
    import fitz
except Exception as e:
    print("PyMuPDF required:", e)
    sys.exit(3)

try:
    import cv2
except Exception:
    cv2 = None

from src.masking.image_inpainter import ImageInpainter
from src.core.config import MaskingConfig


def render_page_image(page, scale=2.0):
    mat = fitz.Matrix(scale, scale)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    return img


def detect_highlight_bboxes(img: np.ndarray) -> list:
    """Return list of bboxes [x1,y1,x2,y2] for yellow-ish highlighted regions."""
    h, w = img.shape[:2]
    if cv2 is None:
        # fallback: no color detection available
        return []
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    # yellow-ish thresholds (H: 15-40 in OpenCV 0-179)
    lower = np.array([15, 60, 100], dtype=np.uint8)
    upper = np.array([40, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    # morphological ops to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    # find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, ww, hh = cv2.boundingRect(cnt)
        # ignore tiny specks
        if ww * hh < 50:
            continue
        x1, y1, x2, y2 = x, y, x + ww, y + hh
        # clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        boxes.append([int(x1), int(y1), int(x2), int(y2)])
    return boxes


def main(pdf_path):
    doc = fitz.open(pdf_path)
    cfg = MaskingConfig()
    inp = ImageInpainter(cfg)
    all_masked_imgs = []
    for pno in range(min(1, doc.page_count)):
        page = doc.load_page(pno)
        img = render_page_image(page, scale=2.0)
        boxes = detect_highlight_bboxes(img)
        regions = []
        for b in boxes:
            # scale back? we rendered at scale, but bboxes are in rendered pixel space and ImageInpainter uses image pixels
            regions.append({"bbox": b, "text": "[HIGHLIGHT]", "word_boxes": []})
        # apply surgical masking tightly
        static = inp.apply_surgical_masking(img, regions, padding=0, merge_entity=True)
        dynamic = inp.apply_surgical_masking(img, regions, padding=None, merge_entity=True)
        # inpaint comparison
        mask = inp.create_mask_from_regions(img.shape, regions)
        inpainted = inp.apply_inpainting(img, mask, method=getattr(cfg, "inpainting_method", "telea")) if cv2 is not None else img

        # save comparisons
        from PIL import Image

        si = Image.fromarray(static)
        di = Image.fromarray(dynamic)
        W = si.width + di.width
        H = max(si.height, di.height)
        comp = Image.new("RGB", (W, H))
        comp.paste(si, (0, 0))
        comp.paste(di, (si.width, 0))
        out_path = OUT_DIR / (Path(pdf_path).stem + f"_highlight_mask_p{pno+1}_compare.png")
        comp.save(str(out_path))

        out_inpaint = OUT_DIR / (Path(pdf_path).stem + f"_highlight_mask_p{pno+1}_inpaint.png")
        ii = Image.fromarray(inpainted)
        ii.save(str(out_inpaint))

        print(f"Page {pno+1}: detected {len(boxes)} highlight regions; outputs: {out_path}, {out_inpaint}")


if __name__ == "__main__":
    main(PDF_PATH)
