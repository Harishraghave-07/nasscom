"""Generate per-region before/after crops and an HTML index for quick review.

Usage:
  PYTHONPATH=$(pwd) python scripts/generate_region_crops.py /path/to/file.pdf

Outputs saved to tmp/regions/<pdf_stem>/
"""
import sys
from pathlib import Path
import json
import os

PDF_PATH = sys.argv[1] if len(sys.argv) > 1 else None
OUT_ROOT = Path("tmp/regions").absolute()
OUT_ROOT.mkdir(parents=True, exist_ok=True)

if not PDF_PATH:
    print("Usage: python scripts/generate_region_crops.py /path/to/file.pdf")
    sys.exit(2)

try:
    import fitz
except Exception as e:
    print("PyMuPDF (fitz) is required but not available:", e)
    sys.exit(3)

import numpy as np
from PIL import Image

# import project
from src.core.config import MaskingConfig
from src.masking.image_inpainter import ImageInpainter

pdf = fitz.open(PDF_PATH)
page = pdf.load_page(0)
mat = fitz.Matrix(2, 2)
pix = page.get_pixmap(matrix=mat, alpha=False)
img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

# extract blocks and words
blocks = page.get_text("blocks")
words = page.get_text("words")
regions = []
for b in blocks:
    x0, y0, x1, y1, text, *_ = b
    text = (text or "").strip()
    if not text:
        continue
    wbs = []
    for w in words:
        wx0, wy0, wx1, wy1, wtext, *_ = w
        if wx0 >= x0 - 1 and wy0 >= y0 - 1 and wx1 <= x1 + 1 and wy1 <= y1 + 1:
            wbs.append({"text": wtext, "bbox": [int(wx0), int(wy0), int(wx1), int(wy1)]})
    regions.append({"bbox": [int(x0), int(y0), int(x1), int(y1)], "text": text, "word_boxes": wbs})

cfg = MaskingConfig()
inp = ImageInpainter(cfg)

# produce surgical and inpaint masked images (dynamic) and static surgical with explicit small padding
surgical_static = inp.apply_surgical_masking(img, regions, padding=cfg.surgical_padding_pixels)
surgical_dynamic = inp.apply_surgical_masking(img, regions, padding=None)

# inpaint dynamic
mask = inp.create_mask_from_regions(img.shape, regions)
inpaint_dynamic, meta = inp.unified_masking_stage(img, regions, style="inpaint")

out_dir = OUT_ROOT / Path(PDF_PATH).stem
out_dir.mkdir(parents=True, exist_ok=True)

# ensure numpy arrays
if not isinstance(surgical_static, np.ndarray):
    surgical_static = np.array(surgical_static)
if not isinstance(surgical_dynamic, np.ndarray):
    surgical_dynamic = np.array(surgical_dynamic)
if not isinstance(inpaint_dynamic, np.ndarray):
    inpaint_dynamic = np.array(inpaint_dynamic)

# save full images
Image.fromarray(surgical_static).save(out_dir / "surgical_static.png")
Image.fromarray(surgical_dynamic).save(out_dir / "surgical_dynamic.png")
Image.fromarray(inpaint_dynamic).save(out_dir / "inpaint_dynamic.png")

# per-region crops
index = []
for i, r in enumerate(regions):
    bx = r.get("bbox")
    if not bx or len(bx) != 4:
        continue
    x1, y1, x2, y2 = bx
    # pad a small margin for context
    pad = 6
    sx1 = max(0, x1 - pad)
    sy1 = max(0, y1 - pad)
    sx2 = min(img.shape[1], x2 + pad)
    sy2 = min(img.shape[0], y2 + pad)
    crop_orig = img[sy1:sy2, sx1:sx2].copy()
    crop_surg_dyn = surgical_dynamic[sy1:sy2, sx1:sx2].copy()
    crop_inpaint = inpaint_dynamic[sy1:sy2, sx1:sx2].copy()

    pil_orig = Image.fromarray(crop_orig)
    pil_surg_dyn = Image.fromarray(crop_surg_dyn)
    pil_inpaint = Image.fromarray(crop_inpaint)

    # save side-by-side composite for this region
    w = pil_orig.width
    h = pil_orig.height
    comp = Image.new("RGB", (w * 3 + 8 * 2, h))
    comp.paste(pil_orig, (0, 0))
    comp.paste(pil_surg_dyn, (w + 8, 0))
    comp.paste(pil_inpaint, (2 * (w + 8), 0))
    fname = out_dir / f"region_{i:03d}.png"
    comp.save(fname)

    index.append({
        "index": i,
        "text": r.get("text"),
        "bbox": bx,
        "file": str(fname.name),
    })

# write index.json and simple HTML
with open(out_dir / "index.json", "w") as fh:
    json.dump(index, fh, indent=2)

html_lines = ["<html><body><h1>Region crops</h1><table>"]
for it in index:
    html_lines.append("<tr><td><img src=\"%s\" style=\"max-width:300px\"></td><td>%s</td></tr>" % (it['file'], it['text']))
html_lines.append("</table></body></html>")
with open(out_dir / "index.html", "w") as fh:
    fh.write("\n".join(html_lines))

print("Saved per-region crops to:", out_dir)
print("Open index:", out_dir / "index.html")
print("Done")
