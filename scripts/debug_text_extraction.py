#!/usr/bin/env python3
"""Diagnostics: native PDF text extraction + OCR + PHI pattern verification.

Outputs per page:
 - Native text (PyMuPDF)
 - OCR text with confidence (EasyOCR)
 - Character counts and comparisons
 - PHI pattern matches (names, SSN, DOB, phone, email)

Usage:
  python3 scripts/debug_text_extraction.py --input "Highlighted PHI Details Emily_Dawson.pdf"

The script writes temporary PNGs for OCR in a temp dir and cleans up after.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import textwrap
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple


PHI_PATTERNS: Dict[str, List[str]] = {
    "names": [r"Emily R\. Dawson", r"Dr\. Marissa Greene"],
    "ssn": [r"527-84-1937"],
    "dob": [r"March 12, 1975", r"03/12/1975"],
    "phone": [r"\(303\)\s*555[- ]?7294", r"303-555-7294"],
    "email": [r"emily\.dawson75@example\.com"],
}


def extract_native_text(pdf_path: Path) -> List[str]:
    try:
        import fitz  # PyMuPDF
    except Exception as e:
        print("Missing dependency PyMuPDF (fitz). Install with: pip install pymupdf", file=sys.stderr)
        raise

    doc = fitz.open(str(pdf_path))
    pages_text = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text")
        pages_text.append(txt)
    doc.close()
    return pages_text


def pdf_pages_to_pngs(pdf_path: Path, out_dir: Path, dpi: int = 150) -> List[Path]:
    try:
        import fitz
    except Exception:
        print("Missing dependency PyMuPDF (fitz). Install with: pip install pymupdf", file=sys.stderr)
        raise

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pngs: List[Path] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        outp = out_dir / f"page_{i+1:03d}.png"
        pix.save(str(outp))
        pngs.append(outp)
    doc.close()
    return pngs


def run_easyocr_on_images(images: List[Path]) -> List[List[Tuple[str, float]]]:
    try:
        import easyocr
    except Exception:
        print("Missing dependency easyocr. Install with: pip install easyocr", file=sys.stderr)
        raise

    # initialize reader (English); GPU disabled here to be safe
    reader = easyocr.Reader(["en"], gpu=False)
    all_results: List[List[Tuple[str, float]]] = []
    for img in images:
        raw = reader.readtext(str(img))
        # raw items: [ (bbox, text, confidence), ... ]
        results = [(item[1], float(item[2])) for item in raw]
        all_results.append(results)
    return all_results


def find_phi_in_text(text: str) -> List[Tuple[str, str, int]]:
    matches = []
    for label, pats in PHI_PATTERNS.items():
        for pat in pats:
            for m in re.finditer(pat, text, flags=re.IGNORECASE):
                matches.append((label, m.group(0), int(m.start())))
    return matches


def concat_ocr_text(ocr_page: List[Tuple[str, float]]) -> str:
    return "\n".join(t for t, _ in ocr_page)


def print_separator():
    print("\n" + "-" * 80 + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description="PDF text extraction + OCR diagnostics for PHI content")
    p.add_argument("--input", required=True, help="Path to PDF to analyze")
    p.add_argument("--dpi", type=int, default=150, help="PNG render DPI for OCR (default 150)")
    p.add_argument("--no-ocr", action="store_true", help="Skip OCR step")
    args = p.parse_args()

    pdf_path = Path(args.input)
    if not pdf_path.exists():
        print(f"File not found: {pdf_path}", file=sys.stderr)
        return 2

    print(f"Analyzing: {pdf_path}\n")

    # Native extraction
    try:
        native_pages = extract_native_text(pdf_path)
    except Exception as e:
        print(f"Native text extraction failed: {e}", file=sys.stderr)
        native_pages = []

    # Render pages to images + OCR
    ocr_results: List[List[Tuple[str, float]]] = []
    image_paths: List[Path] = []

    if not args.no_ocr:
        with tempfile.TemporaryDirectory(prefix="cim_pdf_ocr_") as td:
            outdir = Path(td)
            try:
                image_paths = pdf_pages_to_pngs(pdf_path, outdir, dpi=args.dpi)
            except Exception as e:
                print(f"Failed to render PDF pages to PNG: {e}", file=sys.stderr)
                image_paths = []

            if image_paths:
                try:
                    ocr_results = run_easyocr_on_images(image_paths)
                except Exception as e:
                    print(f"EasyOCR failed: {e}", file=sys.stderr)
                    ocr_results = []

            # keep temp dir alive until after OCR; it will be cleaned up when leaving

            # Present results per page
            page_count = max(len(native_pages), len(image_paths), len(ocr_results))
            for i in range(page_count):
                print_separator()
                print(f"PAGE {i+1}\n")

                native = native_pages[i] if i < len(native_pages) else ""
                print("Native text (first 800 chars):\n")
                print(textwrap.fill(native[:800] or "(no native text)", width=120))
                print(f"\nNative text length: {len(native)} chars\n")

                if args.no_ocr:
                    continue

                ocr_page = ocr_results[i] if i < len(ocr_results) else []
                print("OCR results (text + confidence):\n")
                if not ocr_page:
                    print("(no OCR text)")
                else:
                    for text, conf in ocr_page:
                        # show short preview
                        preview = text if len(text) < 200 else text[:197] + "..."
                        print(f"  [{conf:.3f}] {preview}")

                ocr_concat = concat_ocr_text(ocr_page)
                print(f"\nOCR concatenated length: {len(ocr_concat)} chars")
                if native is not None:
                    try:
                        ratio = (len(ocr_concat) / max(1, len(native)))
                        print(f"OCR/native length ratio: {ratio:.3f}")
                    except Exception:
                        pass

                # PHI pattern search in native and OCR
                print("\nPHI pattern matches in native text:")
                native_matches = find_phi_in_text(native)
                if not native_matches:
                    print("  (none found)")
                else:
                    for lab, matched, pos in native_matches:
                        print(f"  [{lab}] '{matched}' at char pos {pos}")

                print("\nPHI pattern matches in OCR text:")
                ocr_matches = find_phi_in_text(ocr_concat)
                if not ocr_matches:
                    print("  (none found)")
                else:
                    for lab, matched, pos in ocr_matches:
                        print(f"  [{lab}] '{matched}' at char pos {pos}")

            print_separator()
    else:
        # no OCR requested; just report native pages
        for i, native in enumerate(native_pages):
            print_separator()
            print(f"PAGE {i+1}\n")
            print(native[:1200] or "(no native text)")
            print(f"\nNative text length: {len(native)} chars\n")
            nm = find_phi_in_text(native)
            print("PHI matches:")
            for lab, matched, pos in nm:
                print(f"  [{lab}] '{matched}' at char pos {pos}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
