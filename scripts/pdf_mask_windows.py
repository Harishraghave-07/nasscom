# scripts/pdf_mask_windows.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fitz  # pymupdf
from src.quick_fix.simple_masker import SimpleMasker

if len(sys.argv) < 2:
    print("Usage: python scripts/pdf_mask_windows.py <path-to-pdf>")
    sys.exit(1)

pdf_path = sys.argv[1]
out_dir = os.path.join("debug_output", "pdf_pages")
os.makedirs(out_dir, exist_ok=True)

doc = fitz.open(pdf_path)
masker = SimpleMasker()

for i, page in enumerate(doc, start=1):
    pix = page.get_pixmap(dpi=200)  # increase dpi for better OCR
    page_path = os.path.join(out_dir, f"page_{i}.jpg")
    pix.save(page_path)
    masked_out = os.path.join(out_dir, f"page_{i}_masked.jpg")
    print(f"Processing page {i}: saving {page_path} -> masking -> {masked_out}")
    result = masker.mask_image_simple(page_path, masked_out)
    print("  result:", result)
print("Done. Masked pages in:", out_dir)
