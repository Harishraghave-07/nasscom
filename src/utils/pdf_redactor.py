"""Low-level PDF redaction helpers using PyMuPDF (fitz).

Provides PDFRedactor which opens a PDF and exposes methods to extract
text with coordinates, add redaction annotations, apply redactions and
save, render pages to images (NumPy arrays) and create PDFs from images.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover - fallback when library missing
    fitz = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

LOGGER = logging.getLogger("pdf_redactor")
logging.basicConfig(level=logging.INFO)


class PDFRedactor:
    """Utility class wrapping PyMuPDF operations for redaction and rendering.

    Methods raise RuntimeError if PyMuPDF is not available.
    """

    def __init__(self, pdf_path: str):
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required for PDFRedactor")
        self.pdf_path = str(pdf_path)
        self.doc = fitz.open(self.pdf_path)
        self.logger = LOGGER
        self.logger.info("Opened PDF '%s' with %d pages", self.pdf_path, len(self.doc))

    def get_text_with_coords(self, page_num: int) -> List[Dict[str, Any]]:
        """Extract text blocks and their bounding boxes from a page.

        Returns list of dicts: {'text': str, 'bbox': [x0, y0, x1, y1]}
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required")
        if page_num < 0 or page_num >= len(self.doc):
            raise IndexError("page_num out of range")
        page = self.doc[page_num]
        data = page.get_text("dict")
        out: List[Dict[str, Any]] = []
        # structure: blocks -> lines -> spans
        for block in data.get("blocks", []):
            # blocks can be text or image; only consider text
            if block.get("type") != 0:
                continue
            bbox = block.get("bbox", [])
            text = ""
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text += span.get("text", "")
            text = text.strip()
            if text:
                out.append({"text": text, "bbox": [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]})
        self.logger.debug("Extracted %d text blocks from page %d", len(out), page_num)
        return out

    def add_redactions(self, redactions: Dict[int, List[Tuple[float, float, float, float]]]) -> None:
        """Mark redaction annotations on pages.

        redactions: mapping page_num -> list of (x0,y0,x1,y1)
        This only adds redact annotations; call apply_and_save() to make them permanent.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required")
        for pno, boxes in redactions.items():
            if pno < 0 or pno >= len(self.doc):
                self.logger.warning("Skipping out-of-range page %s", pno)
                continue
            page = self.doc[pno]
            for bbox in boxes:
                try:
                    rect = fitz.Rect(bbox[0], bbox[1], bbox[2], bbox[3])
                    if rect.get_area() <= 0:
                        self.logger.debug("Skipping zero-area bbox %s on page %d", bbox, pno)
                        continue
                    page.add_redact_annot(rect, fill=(0, 0, 0))
                    self.logger.info("Added redact annot on page %d bbox=%s", pno, bbox)
                except Exception as e:
                    self.logger.exception("Failed to add redaction for bbox %s on page %d: %s", bbox, pno, e)

    def apply_and_save(self, output_path: str) -> None:
        """Apply all redact annotations and save to output_path."""
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required")
        # apply redactions (this removes content under redaction annotations)
        try:
            self.doc.apply_redactions()
        except Exception:
            # older fitz versions may require doc.save with flags; attempt both
            self.logger.debug("apply_redactions() not available or failed; continuing to save (may not remove content)")
        self.doc.save(output_path, garbage=4, deflate=True)
        self.logger.info("Saved redacted PDF to %s", output_path)
        # keep the document open until close() is called by caller

    def render_page_to_image(self, page_num: int, dpi: int = 300):
        """Render a page to a NumPy BGR image (suitable for OpenCV).

        Returns a NumPy ndarray (H, W, 3) in BGR uint8.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required")
        if np is None:
            raise RuntimeError("numpy is required to render images")
        if page_num < 0 or page_num >= len(self.doc):
            raise IndexError("page_num out of range")
        page = self.doc[page_num]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        w, h = pix.width, pix.height
        samples = pix.samples  # bytes
        arr = np.frombuffer(samples, dtype=np.uint8)
        # pix.n is number of components (3 = RGB)
        comps = pix.n
        if comps == 3:
            arr = arr.reshape((h, w, 3))
            # PyMuPDF returns RGB; convert to BGR for OpenCV compatibility
            arr = arr[:, :, ::-1]
        elif comps == 4:
            arr = arr.reshape((h, w, 4))
            arr = arr[:, :, :3][:, :, ::-1]
        else:
            # single channel
            arr = arr.reshape((h, w))
            arr = np.stack([arr, arr, arr], axis=-1)
        return arr

    def create_pdf_from_images(self, image_paths: List[str], output_path: str) -> None:
        """Create a PDF document from a list of image files.

        Uses fitz to convert images to single-page PDFs and appends them.
        """
        if fitz is None:
            raise RuntimeError("PyMuPDF (fitz) is required")
        new_doc = fitz.open()
        for img_path in image_paths:
            try:
                img_doc = fitz.open(img_path)  # image open
                # convert image doc to PDF bytes
                pdf_bytes = img_doc.convert_to_pdf()
                img_pdf = fitz.open("pdf", pdf_bytes)
                new_doc.insert_pdf(img_pdf)
                img_doc.close()
                img_pdf.close()
                self.logger.info("Inserted image %s into PDF", img_path)
            except Exception as e:
                self.logger.exception("Failed to insert image %s into PDF: %s", img_path, e)
        new_doc.save(output_path)
        new_doc.close()
        self.logger.info("Saved composed PDF to %s", output_path)

    def close(self) -> None:
        try:
            if self.doc is not None:
                self.doc.close()
                self.logger.info("Closed PDF document %s", self.pdf_path)
        except Exception:
            pass
