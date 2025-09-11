#!/usr/bin/env python3
"""Test Presidio Analyzer + Anonymizer integration.

Usage:
  python3 scripts/test_presidio_integration.py /path/to/input.pdf --output-dir debug_output/presidio_test

The script will:
 - extract text from PDF (via PyMuPDF if available, otherwise try OCR via pdf2image+pytesseract)
 - run presidio_analyzer.AnalyzerEngine.analyze on each page
 - run presidio_anonymizer.AnonymizerEngine.anonymize for detected entities
 - print a per-page human-readable summary and save original+anonymized text files

The script is defensive: it prints clear errors when dependencies or models are missing.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger("presidio_test")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def import_optional(name: str):
    try:
        module = __import__(name)
        return module
    except Exception:
        return None


fitz = import_optional("fitz")  # PyMuPDF
pytesseract = import_optional("pytesseract")
pdf2image = import_optional("pdf2image")
try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
except Exception as e:
    AnalyzerEngine = None  # type: ignore
    AnonymizerEngine = None  # type: ignore
    presidio_import_err = e
else:
    presidio_import_err = None


def extract_text_from_pdf_with_fitz(path: Path) -> List[str]:
    doc = fitz.open(str(path))
    pages = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        txt = page.get_text("text")
        pages.append(txt)
    return pages


def ocr_pdf_to_text(path: Path) -> List[str]:
    # require pdf2image and pytesseract
    if pdf2image is None or pytesseract is None:
        raise RuntimeError("OCR fallback requires pdf2image and pytesseract; install them or install PyMuPDF (fitz)")
    from pdf2image import convert_from_path
    from PIL import Image

    images = convert_from_path(str(path))
    pages = []
    for img in images:
        txt = pytesseract.image_to_string(img)
        pages.append(txt)
    return pages


def load_text_file(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    # treat whole file as single page
    return [txt]


def highlight_text(text: str, entities: List[dict]) -> str:
    """Return text with detected spans highlighted inline using <<TYPE:...>> markers."""
    if not entities:
        return text
    # sort entities by start
    ents = sorted(entities, key=lambda e: getattr(e, "start", 0))
    out = []
    last = 0
    for e in ents:
        try:
            s = int(e.start)
            t = int(e.end)
            label = getattr(e, "entity_type", getattr(e, "label", getattr(e, "recognizer", "PII")))
        except Exception:
            continue
        if s < last:
            # overlapping/invalid; skip
            continue
        out.append(text[last:s])
        snippet = text[s:t]
        out.append(f"<<{label}:{snippet}>>")
        last = t
    out.append(text[last:])
    return "".join(out)


def extract_words_and_text_from_pdf_with_fitz(path: Path) -> Tuple[List[str], List[List[dict]]]:
    """Return tuple (pages_text, pages_words) where pages_words is a list per page of
    dicts: {text, bbox=(x0,y0,x1,y1)}. Uses page.get_text("words").
    """
    if fitz is None:
        raise RuntimeError("PyMuPDF (fitz) not available")
    doc = fitz.open(str(path))
    pages_text: List[str] = []
    pages_words: List[List[dict]] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        words = page.get_text("words")  # list of tuples (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        # sort by y then x for reading order
        words = sorted(words, key=lambda w: (int(w[1]), int(w[0])))
        word_dicts = []
        text_parts = []
        for w in words:
            x0, y0, x1, y1, word_text = w[0], w[1], w[2], w[3], w[4]
            bbox = (int(x0), int(y0), int(x1), int(y1))
            word_dicts.append({"text": str(word_text), "bbox": bbox})
            text_parts.append(str(word_text))
        pages_text.append(" ".join(text_parts))
        pages_words.append(word_dicts)
    return pages_text, pages_words


def summarize_entities(entities: List, limit: int = 20) -> str:
    if not entities:
        return "(no entities)"
    lines = []
    for i, e in enumerate(entities[:limit]):
        try:
            etype = e.entity_type
        except Exception:
            etype = getattr(e, "label", getattr(e, "recognizer", "PII"))
        score = getattr(e, "score", getattr(e, "confidence", 0.0))
        txt = getattr(e, "text", None)
        # presidio RecognizerResult doesn't always include text property; try slicing
        if txt is None:
            try:
                txt = e.raw_text
            except Exception:
                txt = "<text unavailable>"
        lines.append(f" - {etype}: {txt} (score={score:.3f})")
    if len(entities) > limit:
        lines.append(f"  ... and {len(entities)-limit} more")
    return "\n".join(lines)


def run_presidio_on_pages(pages: List[str], lang: str = "en") -> List[Tuple[List, str]]:
    if AnalyzerEngine is None or AnonymizerEngine is None:
        raise RuntimeError(f"Presidio modules not available: {presidio_import_err}")
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()
    results = []
    for page_idx, txt in enumerate(pages, start=1):
        try:
            entities = analyzer.analyze(text=txt, language=lang)
        except Exception as e:
            raise RuntimeError(f"Analyzer failed on page {page_idx}: {e}")
        # anonymize
        try:
            anon = anonymizer.anonymize(text=txt, analyzer_results=entities)
            anon_text = getattr(anon, "text", str(anon))
        except Exception as e:
            raise RuntimeError(f"Anonymizer failed on page {page_idx}: {e}")
        results.append((entities, anon_text))
    return results


def write_outputs(outdir: Path, input_name: str, pages: List[str], presidio_results: List[Tuple[List, str]]):
    outdir.mkdir(parents=True, exist_ok=True)
    combined_orig = []
    combined_anon = []
    for i, (txt, (entities, anon_text)) in enumerate(zip(pages, presidio_results), start=1):
        page_base = outdir / f"{input_name}_page_{i:03d}"
        # highlighted original
        try:
            highlighted = highlight_text(txt, entities)
        except Exception:
            highlighted = txt
        (page_base.with_suffix('.orig.txt')).write_text(highlighted, encoding='utf-8')
        (page_base.with_suffix('.anon.txt')).write_text(anon_text, encoding='utf-8')
        combined_orig.append(highlighted)
        combined_anon.append(anon_text)
    # combined files
    (outdir / f"{input_name}_combined.orig.txt").write_text("\n\n--- PAGE BREAK ---\n\n".join(combined_orig), encoding='utf-8')
    (outdir / f"{input_name}_combined.anon.txt").write_text("\n\n--- PAGE BREAK ---\n\n".join(combined_anon), encoding='utf-8')


def try_make_pdf_from_texts(outdir: Path, input_name: str, pages_anon: List[str]):
    # optional: if PyMuPDF is available, create a simple PDF with anonymized text
    if fitz is None:
        logger.debug("PyMuPDF not available; skipping anonymized PDF generation")
        return
    doc = fitz.open()
    for txt in pages_anon:
        # create a page with default white background and print wrapped text
        page = doc.new_page()
        rect = page.rect
        text = txt or ""
        # simple text insertion
        text_writer = fitz.TextWriter(rect)
        text_writer.append(rect.tl, text)
        text_writer.write_text(page)
    outpdf = outdir / f"{input_name}_anonymized.pdf"
    doc.save(str(outpdf))
    logger.info("Wrote anonymized PDF: %s", outpdf)


    def build_char_index_for_words(words: List[dict], sep: str = " ") -> List[Tuple[int, int, tuple]]:
        """Given a list of word dicts {'text','bbox'}, return list of (start_idx,end_idx,bbox)
        representing the character span for each word when joined by `sep`.
        """
        spans: List[Tuple[int, int, tuple]] = []
        idx = 0
        for i, w in enumerate(words):
            txt = w.get("text", "")
            start = idx
            end = idx + len(txt)
            spans.append((start, end, tuple(w.get("bbox", (0, 0, 0, 0)))))
            idx = end
            # add separator char except after last
            if i != len(words) - 1:
                idx += len(sep)
        return spans

    def find_bboxes_for_entity(entity, char_spans: List[Tuple[int, int, tuple]]) -> List[tuple]:
        """Return list of bboxes for words that overlap the entity's character span."""
        try:
            s = int(entity.start)
            e = int(entity.end)
        except Exception:
            return []
        boxes: List[tuple] = []
        for ws, we, bbox in char_spans:
            # overlap
            if we <= s:
                continue
            if ws >= e:
                continue
            boxes.append(bbox)
        return boxes

    def union_bboxes(boxes: List[tuple]) -> List[tuple]:
        if not boxes:
            return []
        # merge to single bbox (for simplicity we union all into one per entity)
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        return [(min(xs1), min(ys1), max(xs2), max(ys2))]

    def mask_image_with_bboxes(image, bboxes: List[tuple], fill=(0, 0, 0)):
        """Mask the provided PIL image by filling given bboxes with fill color."""
        try:
            from PIL import ImageDraw
        except Exception:
            raise RuntimeError("Pillow required for image masking; install pillow")
        draw = ImageDraw.Draw(image)
        for bbox in bboxes:
            draw.rectangle(bbox, fill=fill)
        return image

def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Test Presidio Analyzer/Anonymizer on a file (PDF or text)")
    p.add_argument("input", help="Input file path (PDF or text)")
    p.add_argument("--output-dir", default="debug_output/presidio_test", help="Directory to write reports")
    p.add_argument("--ocr-fallback", action='store_true', help="Allow OCR fallback if PDF text extraction fails")
    p.add_argument("--mask-images", action='store_true', help="Produce image-based masked pages (requires PyMuPDF and Pillow)")
    p.add_argument("--lang", default="en", help="Language code for Presidio analyzer (default: en)")
    args = p.parse_args(argv)

    infile = Path(args.input)
    outdir = Path(args.output_dir)
    input_name = infile.stem

    if presidio_import_err is not None:
        logger.error("Presidio import failed: %s", presidio_import_err)
        logger.error("Install presidio-analyzer and presidio-anonymizer and ensure recognizers are available (e.g., Spacy models)")
        return 2

    if not infile.exists():
        logger.error("Input file not found: %s", infile)
        return 3

    # extract pages
    pages: List[str]
    try:
        if infile.suffix.lower() == ".pdf":
            if args.mask_images:
                if piftz is None:
                    logger.error("Image masking requires PyMuPDF (fitz). Install it or disable --mask-images")
                    return 8
                logger.info("Extracting words+text from PDF using PyMuPDF for image masking")
                pages, pages_words = extract_words_and_text_from_pdf_with_fitz(infile)
                # pages is plain text built from words; pages_words are word dicts
            else:
                if piftz is not None:
                    logger.info("Extracting text from PDF using PyMuPDF (fitz)")
                    pages = extract_text_from_pdf_with_fitz(infile)
                else:
                    if args.ocr_fallback:
                        logger.info("PyMuPDF not available; attempting OCR fallback with pdf2image+pytesseract")
                        pages = ocr_pdf_to_text(infile)
                    else:
                        logger.error("PyMuPDF (fitz) not available; pass --ocr-fallback to enable OCR fallback (requires pdf2image+pytesseract)")
                        return 4
        else:
            pages = load_text_file(infile)
    except Exception as e:
        logger.exception("Failed to extract text: %s", e)
        return 5

    logger.info("Extracted %d page(s)", len(pages))

    # run presidio
    try:
        presidio_results = run_presidio_on_pages(pages, lang=args.lang)
    except Exception as e:
        logger.exception("Presidio processing failed: %s", e)
        return 6

    # print detailed summary and save outputs
    out_pairs = []
    total_entities = 0
    for idx, (entities, anon_text) in enumerate(presidio_results, start=1):
        logger.info("--- Page %d ---", idx)
        logger.info("Detected %d entities", len(entities))
        total_entities += len(entities)
        if entities:
            print(summarize_entities(entities))
        else:
            print("(no entities detected)")
        out_pairs.append((pages[idx-1], entities, anon_text))

    logger.info("Total entities detected across document: %d", total_entities)

    # write outputs
    try:
        write_outputs(outdir, input_name, pages, presidio_results)
        # optional PDF
        anon_pages = [anon for (_ents, anon) in presidio_results]
        try_make_pdf_from_texts(outdir, input_name, anon_pages)
    except Exception:
        logger.exception("Failed to write outputs")
        return 7

    # image masking step (optional)
    if args.mask_images:
        try:
            from PIL import Image
        except Exception:
            logger.error("Pillow required for image masking; install pillow")
            return 9
        try:
            doc = piftz.open(str(infile))
            for page_idx in range(doc.page_count):
                page = doc.load_page(page_idx)
                # render page to PIL Image
                pix = page.get_pixmap()
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                # build char spans for this page from words
                words = pages_words[page_idx]
                char_spans = build_char_index_for_words(words)
                entities = presidio_results[page_idx][0]
                bboxes = []
                for ent in entities:
                    boxes = find_bboxes_for_entity(ent, char_spans)
                    unions = union_bboxes(boxes)
                    bboxes.extend(unions)
                if bboxes:
                    masked = mask_image_with_bboxes(img, bboxes, fill=(0, 0, 0))
                    outimg = outdir / f"{input_name}_page_{page_idx+1:03d}_masked.png"
                    masked.save(outimg)
                    logger.info("Wrote masked image: %s", outimg)
        except Exception:
            logger.exception("Image masking failed")
            return 10

    logger.info("Wrote reports to %s", outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
