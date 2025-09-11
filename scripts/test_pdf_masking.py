"""Render a PDF page, extract text blocks, and compare static vs language-aware surgical masking.

Saves a side-by-side PNG under tmp/ and prints metadata.
"""
import os
import sys
from pathlib import Path
import numpy as np

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None
OUT_DIR = Path("tmp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

if not PDF_PATH:
    print("No PDF path provided")
    sys.exit(2)

# try fitz (PyMuPDF) first
pil_img = None
blocks = []
try:
    import fitz
    doc = fitz.open(PDF_PATH)
    page = doc.load_page(0)
    # render at 150 dpi
    mat = fitz.Matrix(2, 2)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB"
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    pil_img = img
    # get text blocks and words
    blocks = page.get_text("blocks")  # list of (x0, y0, x1, y1, text, block_no)
    words = page.get_text("words")  # list of (x0, y0, x1, y1, word, block_no)
    regions = []
    for b in blocks:
        x0, y0, x1, y1, text, *_ = b
        text = (text or "").strip()
        if not text:
            continue
        # gather words inside this block
        wbs = []
        for w in words:
            wx0, wy0, wx1, wy1, wtext, *_ = w
            # simple containment test
            if wx0 >= x0 - 1 and wy0 >= y0 - 1 and wx1 <= x1 + 1 and wy1 <= y1 + 1:
                wbs.append({"text": wtext, "bbox": [int(wx0), int(wy0), int(wx1), int(wy1)]})
        regions.append({"bbox": [int(x0), int(y0), int(x1), int(y1)], "text": text, "word_boxes": wbs})
except Exception as e:
    print("PyMuPDF not available or failed:", e)
    # try pdf2image + pytesseract fallback
    try:
        from pdf2image import convert_from_path
        from PIL import Image
        pages = convert_from_path(PDF_PATH, dpi=150)
        page = pages[0]
        pil_img = np.array(page.convert("RGB"))
        # try pytesseract to get boxes
        try:
            import pytesseract
            data = pytesseract.image_to_data(page, output_type=pytesseract.Output.DICT)
            n = len(data['text'])
            words = []
            for i in range(n):
                txt = (data['text'][i] or "").strip()
                if not txt:
                    continue
                x = data['left'][i]; y = data['top'][i]; w = data['width'][i]; h = data['height'][i]
                words.append({"text": txt, "bbox": [int(x), int(y), int(x + w), int(y + h)]})
            # naive grouping: each word becomes its own region with a single word_box
            regions = []
            for w in words:
                regions.append({"bbox": w["bbox"], "text": w["text"], "word_boxes": [w]})
        except Exception as e2:
            print("pytesseract not available or failed:", e2)
            regions = []
    except Exception as e3:
        print("Failed to render PDF:", e3)
        sys.exit(3)

if pil_img is None:
    print("Failed to produce an image from the PDF")
    sys.exit(4)

# import ImageInpainter and config
from src.core.config import MaskingConfig
from src.masking.image_inpainter import ImageInpainter

cfg = MaskingConfig()
# record defaults
static_padding = int(getattr(cfg, "surgical_padding_pixels", 2))

inp = ImageInpainter(cfg)

# prepare regions: use extracted blocks if any, else create a single full-page region
if not regions:
    h, w = pil_img.shape[:2]
    regions = [{"bbox": [10, 10, w - 10, h - 10], "text": "[UNKNOWN]"}]

# Apply static surgical masking (explicit padding)
static_img = inp.apply_surgical_masking(pil_img, regions, padding=static_padding, lang_group=None)
# Apply dynamic (language-aware) surgical masking
dynamic_img = inp.apply_surgical_masking(pil_img, regions, padding=None, lang_group=None)

# Save side-by-side comparison
from PIL import Image
si = Image.fromarray(static_img)
di = Image.fromarray(dynamic_img)
W = si.width + di.width
H = max(si.height, di.height)
comp = Image.new("RGB", (W, H))
comp.paste(si, (0, 0))
comp.paste(di, (si.width, 0))
out_path = OUT_DIR / (Path(PDF_PATH).stem + "_mask_compare.png")
comp.save(str(out_path))

# Also run unified_masking_stage for inpaint flows (exposes meta)
masked_static, meta_s = inp.unified_masking_stage(pil_img, regions, style="surgical", padding=static_padding)
masked_dynamic, meta_d = inp.unified_masking_stage(pil_img, regions, style="surgical")

print("Output saved:", out_path)
print("Regions count:", len(regions))
print("Static meta:", meta_s)
print("Dynamic meta:", meta_d)
print("Done")

# --- Inpaint comparisons ---
try:
    mask = inp.create_mask_from_regions(pil_img.shape, regions)
    # static inpaint (use config radius)
    static_radius = int(getattr(cfg, "inpainting_radius", 3))
    inpaint_static = inp.apply_inpainting(pil_img, mask, method=getattr(cfg, "inpainting_method", "telea"), radius=static_radius)
    # dynamic inpaint via unified stage (this will compute heuristics internally)
    inpaint_dynamic, meta_inpaint_dyn = inp.unified_masking_stage(pil_img, regions, style="inpaint")

    si = Image.fromarray(inpaint_static)
    di = Image.fromarray(inpaint_dynamic)
    W = si.width + di.width
    H = max(si.height, di.height)
    comp2 = Image.new("RGB", (W, H))
    comp2.paste(si, (0, 0))
    comp2.paste(di, (si.width, 0))
    out_inpaint = OUT_DIR / (Path(PDF_PATH).stem + "_inpaint_compare.png")
    comp2.save(str(out_inpaint))
    print("Inpaint comparison saved:", out_inpaint)
    print("Inpaint dynamic meta:", meta_inpaint_dyn)
except Exception as e:
    print("Inpaint comparison failed:", e)
