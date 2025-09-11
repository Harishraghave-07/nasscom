"""PDF processing and redaction utilities.

This module provides PDFRedactor and PDFProcessor. PDFProcessor chooses
between native (text) redaction and scanned (image) redaction using an
existing image masking pipeline. All actions are logged for audit purposes.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.utils.pdf_redactor import PDFRedactor
from src.core.config import SETTINGS
import cv2

# lazy import for PyMuPDF (fitz)
try:
    import fitz
except Exception:
    fitz = None

LOGGER = logging.getLogger("PDFProcessor")
logging.basicConfig(level=logging.INFO)


class PDFProcessor:
    """Main PDF processor which chooses native vs scanned processing."""

    def __init__(self, image_pipeline: Any, config: Any):
        self.image_pipeline = image_pipeline
        self.config = config
        self.logger = logging.getLogger("PDFProcessor")
        self.redactor = PDFRedactor(self.logger)
        # Attempt to load PHIClassifier if available in project
        try:
            # prefer the centralized PHI classifier implementation
            from src.phi_detection.phi_classifier import PHIClassifier

            # pass top-level config when available so classifier uses centralized flags
            try:
                self.phi_clf = PHIClassifier(self.config)
            except Exception:
                self.phi_clf = PHIClassifier(SETTINGS.phi)
        except Exception:
            # Fallback simple regex-based PHI detector
            self.phi_clf = None
            self._phi_patterns = {
                "name": re.compile(r"[A-Z][a-z]+ [A-Z][a-z]+"),
                "mrn": re.compile(r"MRN[-\s]?\d{4,}"),
                "ssn": re.compile(r"\d{3}[-\s]?\d{2}[-\s]?\d{4}"),
                "phone": re.compile(r"\d{3}[-\s]?\d{3}[-\s]?\d{4}"),
            }

    def _is_native_pdf(self, pdf_path: str, sample_pages: int = 2, text_threshold: int = 100) -> bool:
        """Heuristic: extract text from first pages; if text length > threshold, consider native."""
        if fitz is None:
            self.logger.warning("PyMuPDF not available; assuming scanned PDF")
            return False
        try:
            doc = fitz.open(pdf_path)
            total_chars = 0
            for pno in range(min(sample_pages, len(doc))):
                page = doc[pno]
                txt = page.get_text().strip()
                total_chars += len(txt)
            doc.close()
            self.logger.debug("_is_native_pdf: sampled chars=%d", total_chars)
            return total_chars >= text_threshold
        except Exception as e:
            self.logger.exception("Error testing pdf type: %s", e)
            return False

    def process_pdf(self, input_path: str, output_path: str) -> Dict[str, Any]:
        report: Dict[str, Any] = {"input": input_path, "output": output_path, "timestamp": datetime.utcnow().isoformat(), "native": None, "pages": 0, "redactions": []}
        try:
            is_native = self._is_native_pdf(input_path)
            report["native"] = is_native
            if is_native:
                r = self._process_native_pdf(input_path, output_path)
            else:
                r = self._process_scanned_pdf(input_path, output_path)
            report.update(r)
        except Exception as e:
            self.logger.exception("process_pdf failed: %s", e)
            report["error"] = str(e)
        # write audit log
        audit_dir = Path("debug_output") / "pdf_audit"
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / f"pdf_audit_{Path(input_path).stem}_{int(datetime.utcnow().timestamp())}.json"
        audit_file.write_text(json.dumps(report, indent=2, default=str))
        return report

    def _classify_text_phi(self, text: str) -> bool:
        """Return True if text is likely PHI using classifier or simple regexes."""
        try:
            if self.phi_clf is not None:
                return bool(self.phi_clf.is_phi(text))
        except Exception:
            self.logger.debug("phi classifier failed, falling back to regex")
        for name, pat in getattr(self, "_phi_patterns", {}).items():
            if pat.search(text):
                return True
        return False

    def _process_native_pdf(self, input_path: str, output_path: str) -> Dict[str, Any]:
        self.logger.info("Processing native PDF: %s", input_path)
        # use PDFRedactor.get_text_with_coords to extract blocks per page
        # PDFRedactor in utils returns per-page blocks via get_text_with_coords
        blocks_per_page: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
        try:
            # iterate pages and extract blocks
            for pno in range(len(self.redactor.doc)):
                page_blocks = self.redactor.get_text_with_coords(pno)
                # convert structure to (bbox, text) tuples
                converted: List[Tuple[Tuple[float, float, float, float], str]] = []
                for item in page_blocks:
                    bbox = tuple(item.get("bbox", []))
                    text = item.get("text", "")
                    converted.append((bbox, text))
                blocks_per_page[pno] = converted
        except Exception as e:
            self.logger.exception("Failed to extract text blocks via PDFRedactor: %s", e)
            raise
        bboxes_per_page: Dict[int, List[Tuple[Tuple[float, float, float, float], str]]] = {}
        redaction_log: List[Dict[str, Any]] = []
        pages_processed = 0
        for pno, page_blocks in blocks_per_page.items():
            pages_processed += 1
            for bbox, text in page_blocks:
                if self._classify_text_phi(text):
                    bboxes_per_page.setdefault(pno, []).append((bbox, text))
                    redaction_log.append({"page": pno, "bbox": bbox, "text": text})
                    self.logger.info("Redact found on page %d bbox=%s text=%s", pno, bbox, text)

        # apply redactions
        if bboxes_per_page:
            # use utils.PDFRedactor.add_redactions + apply_and_save
            simple_map: Dict[int, List[Tuple[float, float, float, float]]] = {}
            for p, items in bboxes_per_page.items():
                simple_map[p] = [it[0] for it in items]
            self.redactor.add_redactions(simple_map)
            self.redactor.apply_and_save(output_path)
            self.logger.info("Redacted PDF saved to %s", output_path)
        else:
            # no redactions; copy original to output
            shutil.copyfile(input_path, output_path)
            self.logger.info("No redactions found; copied original to %s", output_path)

        return {"pages": pages_processed, "redactions": redaction_log}

    def _process_scanned_pdf(self, input_path: str, output_path: str) -> Dict[str, Any]:
        self.logger.info("Processing scanned PDF (render pages to images): %s", input_path)
        if self.redactor is None or not hasattr(self.redactor, "doc"):
            raise RuntimeError("PDFRedactor is required for scanned PDF processing")
        tmpdir = Path(tempfile.mkdtemp(prefix="pdf_pages_"))
        masked_images: List[str] = []
        # render pages via PDFRedactor.render_page_to_image which returns a numpy array
        doc = self.redactor.doc
        try:
            for pno in range(len(doc)):
                img_path = str(tmpdir / f"page_{pno+1}.png")
                # render to numpy array
                img_arr = self.redactor.render_page_to_image(pno, dpi=300)
                # write image via cv2
                cv2.imwrite(img_path, img_arr)
                self.logger.debug("Rendered page %d -> %s", pno + 1, img_path)
                # call image pipeline
                try:
                    # process_single_image may return dict or path
                    res = self.image_pipeline.process_single_image(img_path)
                    out_img = None
                    if isinstance(res, dict):
                        out_img = res.get("output_path") or res.get("masked_path") or res.get("output")
                    elif isinstance(res, str):
                        out_img = res
                    if not out_img or not Path(out_img).exists():
                        # try default masked filename alongside input
                        candidate = str(Path(img_path).with_name(Path(img_path).stem + "_masked.png"))
                        if Path(candidate).exists():
                            out_img = candidate
                    if not out_img or not Path(out_img).exists():
                        raise RuntimeError(f"Image pipeline did not produce masked image for page {pno+1}")
                    masked_images.append(out_img)
                except Exception as e:
                    self.logger.exception("Image pipeline failed on page %d: %s", pno + 1, e)
                    raise
        finally:
            doc.close()

        # compose masked images into a PDF
        self.redactor.create_pdf_from_images(masked_images, output_path)
        # cleanup temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
        self.logger.info("Scanned PDF processed and saved to %s", output_path)
        return {"pages": len(masked_images), "masked_images": masked_images}
